"""CpuLab: the kitting workstation in MuJoCo with the same low-level API as the GPU workshop's ChallengeLab.

60 Hz control (10 physics steps each), damped-least-squares IK on the fingertip centre (TCP), Franka position servos.
Per-episode randomization, progression events and perturbations follow vinci_lab/challenge/world.py.
"""
from __future__ import annotations

import collections
import math
import time

import mujoco
import numpy as np

from . import scene as S
from .objects import KEEP, KITTING_TRAY, PIECES, TOP_DRAWER, WORK_ORDER

SIM_HZ = 60.0
SUBSTEPS = 10
MAX_DQ = 0.25
HOME_Q = [0.0, 0.0, 0.0, -1.571, 0.0, 1.571, -0.785]   # Menagerie "home": hand down, fingers along world x
GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0
_R_DOWN = np.diag([1.0, -1.0, -1.0])                    # hand z pointing down (same as the GPU workshop's DOWN_QUAT)


def result(ok: bool, reason: str, t0: float, **evidence) -> dict:
    return {"ok": bool(ok), "reason": reason, "duration_s": round(time.time() - t0, 3), "evidence": evidence}


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def down_quat(yaw: float) -> tuple:
    """(w, x, y, z) of the hand pointing down, turned by `yaw` about the vertical (fingers close along its y)."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (rot_z(yaw) @ _R_DOWN).ravel())
    return tuple(float(v) for v in q)


DOWN_QUAT = down_quat(0.0)


def quat_to_mat(q) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float) / np.linalg.norm(q))
    return m.reshape(3, 3)


def _slerp(q0, q1, a):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)


class CpuLab:
    def __init__(self, level: dict, cameras: bool = False, viewer: bool = False, realtime: float = 0.0):
        self.level = level
        self.model = S.build(tuple(level["pieces"]))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.arm_qadr = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 8)]
        self.arm_dadr = [m.jnt_dofadr[m.joint(f"joint{i}").id] for i in range(1, 8)]
        self.finger_qadr = [m.jnt_qposadr[m.joint(n).id] for n in ("finger_joint1", "finger_joint2")]
        dj = m.joint("drawer_slide").id
        self.drawer_qadr, self.drawer_dadr = m.jnt_qposadr[dj], m.jnt_dofadr[dj]
        self.drawer_body = m.body("drawer").id
        self.tcp_site = m.site("tcp").id
        self.handle_geom = m.geom("handle").id
        self.piece_ids = {pid: m.body(pid).id for pid in level["pieces"]}
        self.piece_qadr = {pid: m.jnt_qposadr[m.joint(f"{pid}_free").id] for pid in level["pieces"]}
        self.piece_geom = {pid: m.geom(f"{pid}_geom").id for pid in level["pieces"]}
        self._base_mass = {pid: float(m.body_mass[b]) for pid, b in self.piece_ids.items()}
        self._base_inertia = {pid: m.body_inertia[b].copy() for pid, b in self.piece_ids.items()}
        self.pieces = dict(self.piece_ids)
        self.rules = {pid: level.get("rules", {}).get(pid, PIECES[pid].rule) for pid in level["pieces"]}
        self.steps, self.stopped = 0, False
        self.q_target = np.array(HOME_Q)
        self.grip_cmd = GRIP_OPEN
        self._perturb, self.initial, self.perturb_log = {}, {}, []
        self.events = self._fresh_events([])
        self.drawer_vel_hist = collections.deque([0.0] * 8, maxlen=8)
        self.cameras = cameras
        self._renderer = None
        self.frames, self.recording, self.record_every = [], False, 2
        self.viewer, self.realtime, self._wall0 = None, realtime, None
        if viewer:
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(m, self.data, show_left_ui=False, show_right_ui=False)
            self.viewer.cam.lookat[:] = S.FRONT_TARGET
            self.viewer.cam.distance, self.viewer.cam.azimuth, self.viewer.cam.elevation = 2.3, 180.0, -38.0
        self.reset()

    # ------------------------------------------------------------------ stepping
    def step(self, n: int = 1):
        m, d = self.model, self.data
        for _ in range(n):
            grip = self.grip_cmd
            pert = self._perturb.get("transport_slip")
            if pert and not pert["armed"] and pert["steps_left"] > 0:
                grip = GRIP_OPEN
                pert["steps_left"] -= 1
            d.ctrl[:7] = self.q_target
            d.ctrl[7] = grip
            mujoco.mj_step(m, d, nstep=SUBSTEPS)
            self.steps += 1
            self.drawer_vel_hist.append(float(d.qvel[self.drawer_dadr]))
            if self.steps % 6 == 0:
                self._track_events()
            if self.recording and self.steps % self.record_every == 0:
                self.frames.append(self.render("front", 320, 240))
            if self.viewer is not None:
                self._sync_viewer()

    def _sync_viewer(self):
        if self.steps % 2:
            return
        if not self.viewer.is_running():
            self.viewer = None
            return
        self.viewer.sync()
        if self.realtime:
            if self._wall0 is None:
                self._wall0 = (time.time(), self.steps)
            ahead = (self.steps - self._wall0[1]) / SIM_HZ / self.realtime - (time.time() - self._wall0[0])
            if ahead > 0:
                time.sleep(ahead)

    def hold(self, n: int = 30, gripper: float | None = None):
        if gripper is not None:
            self.grip_cmd = GRIP_OPEN if gripper > 0 else GRIP_CLOSED
        self.step(n)

    # ------------------------------------------------------------------ episode
    @staticmethod
    def _fresh_events(pieces):
        return {"inspected": False, "verified_after_last_action": False, "drawer_opened": False,
                "grasped": {p: False for p in pieces}, "wrong_place": {p: False for p in pieces}}

    def reset(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_qadr] = HOME_Q
        d.qpos[self.finger_qadr] = 0.04
        self.q_target, self.grip_cmd, self.stopped = np.array(HOME_Q), GRIP_OPEN, False
        mujoco.mj_forward(m, d)

    def reset_episode(self, seed: int) -> dict:
        """Randomize the workstation for this level (same parameters as the GPU workshop). Returns the initial state."""
        rng = np.random.default_rng(seed)
        L, m, d = self.level, self.model, self.data
        self.reset()
        placed, poses = [], {}
        reg = L["table_region"]
        for pid in L["pieces"]:
            p = PIECES[pid]
            for _ in range(500):
                x, y = rng.uniform(*reg["x"]), rng.uniform(*reg["y"])
                if all(math.hypot(x - qx, y - qy) > p.radius + qr + L["min_gap"] for qx, qy, qr in placed):
                    break
            placed.append((x, y, p.radius))
            yaw = math.radians(rng.uniform(-L["yaw_deg"], L["yaw_deg"]))
            poses[pid] = (float(x), float(y), float(yaw))
            fr = float(rng.uniform(*L["friction"]))
            m.geom_friction[self.piece_geom[pid], 0] = fr
            scale = float(rng.uniform(*L["mass_scale"]))
            b = self.piece_ids[pid]
            m.body_mass[b] = self._base_mass[pid] * scale
            m.body_inertia[b] = self._base_inertia[pid] * scale
        damping = float(rng.uniform(*L["drawer_damping"]))
        sticky = rng.random() < L.get("sticky_drawer", {}).get("p", 0.0)
        if sticky:
            damping = L["sticky_drawer"]["damping"]
        m.dof_damping[self.drawer_dadr] = damping
        ajar = 0.0
        if rng.random() < L["drawer_ajar"]["p"]:
            ajar = float(rng.uniform(*L["drawer_ajar"]["range"]))
        pert = L.get("perturbations", {})
        self._perturb = {}
        if "transport_slip" in pert and rng.random() < pert["transport_slip"]["p"]:
            self._perturb["transport_slip"] = {"steps_left": pert["transport_slip"]["open_steps"], "armed": True}
        # masses changed: refresh the solver's inverse weights (stale ones make contacts explode). mj_setConst
        # rewrites qpos with qpos0, so the episode's poses are written after it.
        mujoco.mj_setConst(m, d)
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0.0
        d.qpos[self.arm_qadr] = HOME_Q
        d.qpos[self.finger_qadr] = 0.04
        for pid, (x, y, yaw) in poses.items():
            a = self.piece_qadr[pid]
            d.qpos[a:a + 3] = [x, y, S.TABLE_TOP + PIECES[pid].height / 2 + 0.002]
            d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        d.qpos[self.drawer_qadr] = ajar
        mujoco.mj_forward(m, d)
        self.perturb_log = []
        self.events = self._fresh_events(L["pieces"])
        self.steps = 0
        self.hold(40)
        self.steps = 0
        self.initial = {"seed": seed, "poses": poses, "drawer_damping": round(damping, 2), "sticky": bool(sticky),
                        "drawer_ajar_m": round(ajar, 3), "perturbations": sorted(self._perturb),
                        "piece_pos": {pid: self.piece_pos(pid).tolist() for pid in self.pieces}}
        return self.initial

    def note_action(self):
        self.events["verified_after_last_action"] = False

    def note_inspection(self):
        self.events["inspected"] = True
        self.events["verified_after_last_action"] = True

    def _track_events(self):
        ev = self.events
        if not ev["drawer_opened"] and self.drawer_opening() >= 0.18:
            ev["drawer_opened"] = True
        w = self.gripper_width()
        if 0.005 < w < 0.075:
            tcp = self.ee_pos()
            for pid in self.pieces:
                q = self.piece_pos(pid)
                held = np.linalg.norm(tcp - q) < 0.06 and q[2] > S.TABLE_TOP + 0.06
                if held and not ev["grasped"].get(pid):
                    ev["grasped"][pid] = True
                s = self._perturb.get("transport_slip")
                if s and s["armed"] and held and q[2] > S.TABLE_TOP + 0.10:
                    s["armed"] = False
                    self.perturb_log.append({"type": "transport_slip", "piece": pid, "sim_t": round(self.steps / SIM_HZ, 2)})

    # ------------------------------------------------------------------ state
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_site].copy()

    def ee_pose(self):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.data.site_xmat[self.tcp_site])
        return self.ee_pos(), q

    def gripper_width(self) -> float:
        return float(self.data.qpos[self.finger_qadr].sum())

    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.arm_qadr].copy()

    def drawer_opening(self) -> float:
        return float(self.data.qpos[self.drawer_qadr])

    def drawer_velocity(self) -> float:
        return float(self.data.qvel[self.drawer_dadr])

    def drawer_pos(self) -> np.ndarray:
        return self.data.xpos[self.drawer_body].copy()

    def handle_pos(self) -> np.ndarray:
        return self.data.geom_xpos[self.handle_geom].copy()

    def piece_pos(self, pid) -> np.ndarray:
        return self.data.xpos[self.piece_ids[pid]].copy()

    def piece_quat(self, pid) -> np.ndarray:
        return self.data.xquat[self.piece_ids[pid]].copy()

    def piece_yaw(self, pid) -> float:
        w, x, y, z = self.piece_quat(pid)
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def piece_tilt_deg(self, pid) -> float:
        w, x, y, z = self.piece_quat(pid)
        return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))

    def in_drawer(self, pos) -> bool:
        rel = np.asarray(pos) - self.drawer_pos()
        return (abs(rel[0]) <= S.DRAWER_DEPTH / 2 and abs(rel[1]) <= S.DRAWER_HALF_W
                and -0.01 <= rel[2] <= S.DRAWER_WALL_H + 0.03)

    def in_kitting_tray(self, pos) -> bool:
        return (abs(pos[0] - S.TRAY_CENTER[0]) <= S.TRAY_INNER[0] / 2 and abs(pos[1] - S.TRAY_CENTER[1]) <= S.TRAY_INNER[1] / 2
                and S.TRAY_FLOOR - 0.01 <= pos[2] <= S.TRAY_FLOOR + S.TRAY_WALL_H + 0.03)

    def on_table(self, pos) -> bool:
        return (abs(pos[0] - S.TABLE_CENTER[0]) <= S.TABLE_SIZE[0] / 2 and abs(pos[1] - S.TABLE_CENTER[1]) <= S.TABLE_SIZE[1] / 2
                and S.TABLE_TOP - 0.01 <= pos[2] <= S.TABLE_TOP + 0.08)

    def where(self, pid) -> str:
        p = self.piece_pos(pid)
        if self.in_drawer(p):
            return TOP_DRAWER
        if self.in_kitting_tray(p):
            return KITTING_TRAY
        if p[2] < 0.25:
            return "floor"
        w = self.gripper_width()
        if np.linalg.norm(self.ee_pos() - p) < 0.06 and 0.005 < w < 0.075:
            return "gripper"
        if self.on_table(p):
            return "table"
        return "elsewhere"

    def observe_workstation(self) -> dict:
        names = {(0.85, 0.10, 0.10): "red", (0.10, 0.25, 0.85): "blue", (0.95, 0.80, 0.10): "yellow"}
        items = {}
        for pid in self.pieces:
            piece = PIECES[pid]
            items[pid] = {"shape": piece.shape, "size_m": list(piece.size), "color": names[piece.color],
                          "pos": [round(float(v), 4) for v in self.piece_pos(pid)], "yaw_rad": round(self.piece_yaw(pid), 3),
                          "where": self.where(pid)}
        return {"pieces": items, "drawer_opening_m": round(self.drawer_opening(), 4),
                "work_order": WORK_ORDER + (" | " + self.level["work_order_note"] if self.level.get("work_order_note") else ""),
                "rules": dict(self.rules), "tray_center": list(S.TRAY_CENTER)}

    # ------------------------------------------------------------------ cameras
    def render(self, camera: str = "front", width: int = 128, height: int = 128) -> np.ndarray:
        if self._renderer is None or self._renderer.width != width or self._renderer.height != height:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height, width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def observe(self) -> dict:
        p, q = self.ee_pose()
        obs = {"joint_pos": self.joint_pos().round(4).tolist(), "gripper_width": round(self.gripper_width(), 4),
               "ee_pos": p.round(4).tolist(), "ee_quat": q.round(4).tolist(), "handle_pos": self.handle_pos().round(4).tolist(),
               "drawer_opening_m": round(self.drawer_opening(), 4), "sim_time_s": round(self.steps / SIM_HZ, 2)}
        if self.cameras:
            obs["rgb_front"], obs["rgb_wrist"] = self.render("front"), self.render("wrist")
        return obs

    def start_recording(self):
        self.frames, self.recording = [], True

    def save_video(self, path: str, fps: int = 30) -> str | None:
        self.recording = False
        if not self.frames:
            return None
        try:
            import imageio.v2 as imageio
            imageio.mimwrite(path, self.frames, fps=fps)
        except Exception:  # noqa: BLE001  no video encoder: keep a GIF
            path = path.rsplit(".", 1)[0] + ".gif"
            from PIL import Image
            ims = [Image.fromarray(f) for f in self.frames[::2]]
            ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(2000 / fps), loop=0)
        self.frames = []
        return path

    # ------------------------------------------------------------------ actions
    def _ik_dq(self, goal_p, goal_R, lam: float = 0.05, k_null: float = 0.1) -> np.ndarray:
        m, d = self.model, self.data
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp_site)
        J = np.vstack([jacp[:, self.arm_dadr], jacr[:, self.arm_dadr]])
        R = d.site_xmat[self.tcp_site].reshape(3, 3)
        e_pos = goal_p - d.site_xpos[self.tcp_site]
        e_rot = 0.5 * sum(np.cross(R[:, i], goal_R[:, i]) for i in range(3))
        e = np.concatenate([e_pos, e_rot])
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(6))
        q = self.joint_pos()
        null = (np.eye(7) - J_pinv @ J) @ (k_null * (np.array(HOME_Q) - q))
        return J_pinv @ e + null

    def move_to(self, pos, quat=DOWN_QUAT, speed: float = 0.25, tol: float = 0.006, settle: int = 20,
                max_extra_steps: int = 150) -> dict:
        """Move the fingertip centre (TCP) to `pos` (m) with orientation `quat` (w, x, y, z), along a straight line."""
        t0 = time.time()
        if self.stopped:
            return result(False, "stopped", t0)
        goal_p = np.asarray(pos, float)
        start_p, start_q = self.ee_pose()
        goal_q = np.asarray(quat, float)
        dist = float(np.linalg.norm(goal_p - start_p))
        n = max(int(dist / (speed / SIM_HZ)), 10)
        err = float("inf")
        for i in range(n + max_extra_steps):
            a = min(1.0, (i + 1) / n)
            tcp_p = start_p + a * (goal_p - start_p)
            tcp_R = quat_to_mat(_slerp(start_q, goal_q, a))
            dq = self._ik_dq(tcp_p, tcp_R)
            dq *= min(1.0, MAX_DQ / (np.abs(dq).max() + 1e-9))
            lo, hi = self.model.jnt_range[[self.model.joint(f"joint{k}").id for k in range(1, 8)]].T
            self.q_target = np.clip(self.joint_pos() + dq, lo, hi)
            self.step(1)
            err = float(np.linalg.norm(self.ee_pos() - goal_p))
            if a >= 1.0 and err < tol:
                break
        self.hold(settle)
        err = float(np.linalg.norm(self.ee_pos() - goal_p))
        ok = err < max(tol * 2, 0.012)
        return result(ok, "reached" if ok else f"residual error {err * 1000:.1f} mm (unreachable or blocked)", t0,
                      target=[float(v) for v in pos], final_error_m=round(err, 4), steps=self.steps)

    def move_joints(self, q_target=None, speed: float = 0.8, settle: int = 20) -> dict:
        t0 = time.time()
        q0 = self.q_target.copy()
        q1 = np.array(HOME_Q if q_target is None else q_target, float)
        n = max(int(float(np.abs(q1 - q0).max()) / (speed / SIM_HZ)), 5)
        for i in range(n):
            self.q_target = q0 + (q1 - q0) * (i + 1) / n
            self.step(1)
        self.hold(settle)
        err = float(np.abs(self.joint_pos() - q1).max())
        return result(err < 0.03, "reached" if err < 0.03 else f"joint error {err:.3f} rad", t0, max_joint_err=round(err, 4))

    def gripper(self, open: bool, steps: int = 40) -> dict:
        t0 = time.time()
        if self.stopped:
            return result(False, "stopped", t0)
        self.hold(steps, gripper=1.0 if open else -1.0)
        w = self.gripper_width()
        if open:
            ok = w > 0.07
            reason = "opened" if ok else f"could not open (width {w * 1000:.0f} mm)"
        else:
            ok = w > 0.004
            reason = "closed on object" if ok else "closed on nothing (empty grasp)"
        return result(ok, reason, t0, width_m=round(w, 4))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        if self._renderer is not None:
            self._renderer.close()
