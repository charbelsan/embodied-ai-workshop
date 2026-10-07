"""ChallengeLab: the lab API on the kitting workstation, with per-level randomization and perturbations."""
from __future__ import annotations

import collections
import math

import numpy as np
import torch

from ..api import LabEnv
from .objects import KEEP, KITTING_TRAY, PIECES, TOP_DRAWER, WORK_ORDER
from .scene import TABLE_CENTER, TABLE_SIZE, TABLE_TOP, TRAY_CENTER, TRAY_FLOOR, TRAY_INNER, TRAY_WALL_H, ChallengeEnvCfg


class ChallengeLab(LabEnv):
    def __init__(self, level: dict, cameras: bool = False, device: str = "cuda:0"):
        cfg = ChallengeEnvCfg(pieces=tuple(level["pieces"]))
        self.level = level
        self._perturb, self.initial, self.perturb_log = {}, {}, []  # used by step_action during base-class reset
        self.drawer_vel_hist = collections.deque([0.0] * 8, maxlen=8)  # for predict_outcome (world model input)
        self.events = self._fresh_events([])  # progression tracking (grader), filled during the episode
        super().__init__(cfg=cfg, device=device, cameras=cameras)
        self.pieces = {pid: self.env.scene[pid] for pid in level["pieces"]}
        self.rules = {pid: level.get("rules", {}).get(pid, PIECES[pid].rule) for pid in level["pieces"]}

    # ------------------------------------------------------------------ episode
    def reset_episode(self, seed: int) -> dict:
        """Randomize the workstation for this level. Returns the initial-state record (for the grader/report)."""
        rng = np.random.default_rng(seed)
        L = self.level
        self.reset()
        # pieces: non-overlapping poses on the table region
        placed = []
        reg = L["table_region"]
        poses = {}
        for pid in L["pieces"]:
            p = PIECES[pid]
            for _ in range(500):
                x, y = rng.uniform(*reg["x"]), rng.uniform(*reg["y"])
                if all(math.hypot(x - qx, y - qy) > p.radius + qr + L["min_gap"] for qx, qy, qr in placed):
                    break
            placed.append((x, y, p.radius))
            yaw = math.radians(rng.uniform(-L["yaw_deg"], L["yaw_deg"]))
            poses[pid] = (float(x), float(y), float(yaw))
            obj = self.pieces[pid]
            pose = torch.tensor([[x, y, TABLE_TOP + p.height / 2 + 0.002, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]],
                                dtype=torch.float32, device=self.device)
            pose[:, :3] += self.env.scene.env_origins[0]
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.device))
            # physics: friction (all shapes of the body) and mass
            mats = obj.root_physx_view.get_material_properties()
            fr = float(rng.uniform(*L["friction"]))
            mats[..., 0], mats[..., 1] = fr, 0.85 * fr
            obj.root_physx_view.set_material_properties(mats, torch.arange(1))
            masses = obj.root_physx_view.get_masses()
            masses[:] = PIECES[pid].mass * float(rng.uniform(*L["mass_scale"]))
            obj.root_physx_view.set_masses(masses, torch.arange(1))
        # drawer: damping (sticky), initial opening (ajar)
        damping = float(rng.uniform(*L["drawer_damping"]))
        sticky = rng.random() < L.get("sticky_drawer", {}).get("p", 0.0)
        if sticky:
            damping = L["sticky_drawer"]["damping"]
        self.cabinet.write_joint_damping_to_sim(torch.tensor([[damping]], device=self.device), joint_ids=[self.drawer_jid])
        ajar = 0.0
        if rng.random() < L["drawer_ajar"]["p"]:
            ajar = float(rng.uniform(*L["drawer_ajar"]["range"]))
        jp = self.cabinet.data.joint_pos.clone()
        jp[0, self.drawer_jid] = ajar
        self.cabinet.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        # perturbations (decided now, triggered by state during the episode)
        pert = L.get("perturbations", {})
        self._perturb = {}
        if "transport_slip" in pert and rng.random() < pert["transport_slip"]["p"]:
            self._perturb["transport_slip"] = {"steps_left": pert["transport_slip"]["open_steps"], "armed": True}
        if "arm_push" in pert and rng.random() < pert["arm_push"]["p"]:
            d = rng.normal(size=7)
            d = d / np.linalg.norm(d) * pert["arm_push"]["joint_offset_rad"]
            self._perturb["arm_push"] = {"steps_left": pert["arm_push"]["steps"], "armed": True, "delta": d.tolist()}
        self.perturb_log = []
        self.events = self._fresh_events(L["pieces"])
        self.hold(40)
        self.initial = {"seed": seed, "poses": poses, "drawer_damping": round(damping, 2), "sticky": bool(sticky),
                        "drawer_ajar_m": round(ajar, 3), "perturbations": sorted(self._perturb),
                        "piece_pos": {pid: self.piece_pos(pid).tolist() for pid in self.pieces}}
        return self.initial

    @staticmethod
    def _fresh_events(pieces):
        return {"inspected": False, "verified_after_last_action": False, "drawer_opened": False,
                "grasped": {p: False for p in pieces}, "wrong_place": {p: False for p in pieces}}

    def note_action(self):
        """Called by every manipulation skill: the world may have changed, the Brain must look again to 'verify'."""
        self.events["verified_after_last_action"] = False

    def note_inspection(self):
        self.events["inspected"] = True
        self.events["verified_after_last_action"] = True

    # ------------------------------------------------------------------ perception helpers
    def piece_pos(self, pid) -> torch.Tensor:
        return self.pieces[pid].data.root_pos_w[0] - self.env.scene.env_origins[0]

    def piece_tilt_deg(self, pid) -> float:
        w, x, y, z = self.pieces[pid].data.root_quat_w[0].tolist()
        return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))

    def piece_yaw(self, pid) -> float:
        w, x, y, z = self.pieces[pid].data.root_quat_w[0].tolist()
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def in_kitting_tray(self, pos) -> bool:
        return (abs(pos[0] - TRAY_CENTER[0]) <= TRAY_INNER[0] / 2 and abs(pos[1] - TRAY_CENTER[1]) <= TRAY_INNER[1] / 2
                and TRAY_FLOOR - 0.01 <= pos[2] <= TRAY_FLOOR + TRAY_WALL_H + 0.03)

    def on_table(self, pos) -> bool:
        return (abs(pos[0] - TABLE_CENTER[0]) <= TABLE_SIZE[0] / 2 and abs(pos[1] - TABLE_CENTER[1]) <= TABLE_SIZE[1] / 2
                and TABLE_TOP - 0.01 <= pos[2] <= TABLE_TOP + 0.08)

    def where(self, pid) -> str:
        pos = self.piece_pos(pid)
        p = pos.tolist()
        if self.in_tray(self.in_drawer_frame(self.pieces[pid].data.root_pos_w[0])):
            return TOP_DRAWER
        if self.in_kitting_tray(p):
            return KITTING_TRAY
        if p[2] < 0.25:
            return "floor"
        tcp = self.ee_pose()[0].tolist()
        w = float(self.robot.data.joint_pos[0, -2:].sum())
        if math.dist(tcp, p) < 0.06 and 0.005 < w < 0.075:
            return "gripper"
        if self.on_table(p):
            return "table"
        return "elsewhere"

    def observe_workstation(self) -> dict:
        items = {}
        for pid, piece in ((k, PIECES[k]) for k in self.pieces):
            items[pid] = {"shape": piece.shape, "size_m": list(piece.size), "color": {(0.85, 0.10, 0.10): "red", (0.10, 0.25, 0.85): "blue",
                          (0.95, 0.80, 0.10): "yellow"}[piece.color], "pos": [round(v, 4) for v in self.piece_pos(pid).tolist()],
                          "yaw_rad": round(self.piece_yaw(pid), 3), "where": self.where(pid)}
        return {"pieces": items, "drawer_opening_m": round(self.drawer_opening(), 4),
                "work_order": WORK_ORDER + (" | " + self.level["work_order_note"] if self.level.get("work_order_note") else ""),
                "rules": dict(self.rules), "tray_center": list(TRAY_CENTER)}

    # ------------------------------------------------------------------ perturbations (inside every control step)
    def step_action(self, action):
        pert = self._perturb
        if pert:
            action = action.clone()
            w = float(self.robot.data.joint_pos[0, -2:].sum())
            tcp = self.ee_pose()[0] - 0  # world == env frame for env 0 at origin offset handled below
            tcp_l = (tcp - self.env.scene.env_origins[0]).tolist()
            s = pert.get("transport_slip")
            if s and s["steps_left"] > 0:
                held = [pid for pid in self.pieces if math.dist(tcp_l, self.piece_pos(pid).tolist()) < 0.05]
                lifted = held and float(self.piece_pos(held[0])[2]) > TABLE_TOP + 0.10 and 0.005 < w < 0.075
                if s["armed"] and lifted:
                    s["armed"] = False
                    self.perturb_log.append({"type": "transport_slip", "piece": held[0], "sim_t": round(self.steps / 60, 2)})
                if not s["armed"]:
                    action[:, 7] = 1.0
                    s["steps_left"] -= 1
            a = pert.get("arm_push")
            if a and a["steps_left"] > 0:
                near = [pid for pid in self.pieces if math.dist(tcp_l[:2], self.piece_pos(pid).tolist()[:2]) < 0.06
                        and tcp_l[2] - float(self.piece_pos(pid)[2]) < 0.12]
                if a["armed"] and near and w > 0.07:
                    a["armed"] = False
                    self.perturb_log.append({"type": "arm_push", "piece": near[0], "sim_t": round(self.steps / 60, 2)})
                if not a["armed"]:
                    action[:, :7] += torch.tensor(a["delta"], device=self.device)
                    a["steps_left"] -= 1
        out = super().step_action(action)
        self.drawer_vel_hist.append(float(self.cabinet.data.joint_vel[0, self.drawer_jid]))
        if self.steps % 6 == 0:  # progression events (10 Hz): drawer opened, pieces grasped (lifted in the gripper)
            ev = self.events
            if not ev["drawer_opened"] and self.drawer_opening() >= 0.18:
                ev["drawer_opened"] = True
            w = float(self.robot.data.joint_pos[0, -2:].sum())
            if 0.005 < w < 0.075:
                tcp = (self.ee_pose()[0] - self.env.scene.env_origins[0]).tolist()
                for pid in self.pieces:
                    if not ev["grasped"].get(pid):
                        q = self.piece_pos(pid).tolist()
                        if math.dist(tcp, q) < 0.06 and q[2] > TABLE_TOP + 0.06:
                            ev["grasped"][pid] = True
        return out
