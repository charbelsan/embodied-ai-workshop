"""High-level skills for the Brain. Same contract for every skill, whatever builds it:

    {"ok": bool, "reason": str, "duration_s": float, "evidence": {...}}

A skill can have several providers (analytic IK, RL policy, VLA policy). The Brain picks a provider
by name; it never sees how the skill works inside.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from .api import DOWN_QUAT, LabEnv, down_quat, result
from .scene import CUBE_REST_Z, STAND_POS, STAND_SIZE

# gripper pointing +x into the cabinet front and pitched 30 deg down, fingers above/below the handle bar.
# (a horizontal gripper at handle height is out of reach for the pedestal-mounted arm: reach sweep, 5-10 cm error)
HANDLE_PITCH = math.radians(30)
HANDLE_QUAT = (0.35355, 0.61237, 0.61237, 0.35355)  # qy(30 deg) * (0.5, 0.5, 0.5, 0.5)
OPEN_TARGET = 0.25  # m of drawer travel pulled by the analytic skill (max 0.4)
SAFE_Z = 0.86  # travel height: above the drawer rim (0.78) yet inside reach near the base (0.92 was not)
PLACE_LOCAL = (0.12, -0.12)
PICK_START = (0.30, 0.30, 0.90)  # clear of the open drawer; start pose of pick_and_place demos / VLA episodes  # drawer-frame xy of the drop point (front half of the tray, toward the stand side)
POLICY_DIR = Path(__file__).resolve().parent.parent / "policies"


class Skills:
    open_target = OPEN_TARGET  # m of drawer travel pulled by the analytic skill

    def __init__(self, lab: LabEnv):
        self.lab = lab
        self._rl_policy = None
        self.vla_client = None  # set by the VLA track (pick_and_place provider="vla")

    # ------------------------------------------------------------------ helpers
    def _named_target(self, target):
        lab = self.lab
        if isinstance(target, (list, tuple)):
            return list(map(float, target)), DOWN_QUAT
        o = lab.observe()
        if target == "home":
            return [0.45, 0.0, 0.75], DOWN_QUAT
        if target == "above_cube":
            c = o["cube_pos"]
            return [c[0], c[1], c[2] + 0.15], down_quat(math.atan2(c[1], c[0]))
        if target == "handle":
            return o["handle_pos"], HANDLE_QUAT
        if target in ("above_drawer", "drawer"):
            d = self.drawer_drop_point(z=SAFE_Z)
            return d, down_quat(math.atan2(d[1], d[0]))
        raise ValueError(f"unknown target {target!r}; known: home, above_cube, handle, above_drawer, or [x,y,z]")

    def drawer_drop_point(self, z: float) -> list:
        from isaaclab.utils.math import quat_apply
        dp, dq = self.lab.drawer_pose()
        local = torch.tensor([[PLACE_LOCAL[0], PLACE_LOCAL[1], 0.0]], device=self.lab.device)
        w = dp + quat_apply(dq[None], local)[0]
        return [float(w[0]), float(w[1]), z]

    def grasp_yaw(self) -> float:
        """Cube face direction closest to the radial yaw (radial = least wrist twist)."""
        q = self.lab.cube.data.root_quat_w[0].tolist()
        w, x, y, z = q
        cube_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c = self.lab.cube.data.root_pos_w[0]
        radial = math.atan2(float(c[1]), float(c[0]))
        return cube_yaw + round((radial - cube_yaw) / (math.pi / 2)) * (math.pi / 2)

    def go_pick_start(self) -> dict:
        return self.lab.move_to(list(PICK_START), down_quat(math.atan2(PICK_START[1], PICK_START[0])))

    def _where_is_cube(self, o) -> str:
        c = o["cube_pos"]
        rel = torch.tensor(o["cube_in_drawer_frame"], device=self.lab.device)
        if self.lab.in_tray(rel):
            return "in_top_drawer"
        ee = o["ee_pos"]
        if o["gripper_width"] > 0.004 and o["gripper_width"] < 0.06 and sum((a - b) ** 2 for a, b in zip(ee, c)) ** 0.5 < 0.05:
            return "in_gripper"
        if abs(c[2] - CUBE_REST_Z) < 0.02 and abs(c[0] - STAND_POS[0]) < 0.07 and abs(c[1] - STAND_POS[1]) < 0.07:
            return "on_stand"
        return "elsewhere(z=%.2f)" % c[2]

    # ------------------------------------------------------------------ skills
    def inspect_scene(self) -> dict:
        t0 = time.time()
        o = self.lab.observe()
        ev = {
            "objects": {
                "cube": {"pos": o["cube_pos"], "where": self._where_is_cube(o), "size_m": 0.045, "color": "red"},
                "top_drawer": {"opening_m": o["drawer_opening_m"], "state": "open" if o["drawer_opening_m"] > 0.2
                               else ("closed" if o["drawer_opening_m"] < 0.02 else "partially_open"),
                               "handle_pos": o["handle_pos"], "max_opening_m": 0.4},
                "stand": {"pos": [STAND_POS[0], STAND_POS[1], STAND_SIZE[2]]},
            },
            "gripper": {"width_m": round(o["gripper_width"], 4), "tcp_pos": o["ee_pos"]},
            "task_success": self.lab.success(),
        }
        return result(True, "scene inspected", t0, **ev)

    def move_to(self, target) -> dict:
        pos, quat = self._named_target(target)
        r = self.lab.move_to(pos, quat)
        r["evidence"]["target_name"] = target if isinstance(target, str) else "xyz"
        return r

    def open_drawer(self, provider: str = "analytic") -> dict:
        t0 = time.time()
        if provider == "rl":
            r = self._open_drawer_rl()
        elif provider == "analytic":
            r = self._open_drawer_analytic()
        elif provider == "visual":  # DAgger student: RGB-D cameras + joints only, no privileged state
            from .dagger.skill import open_drawer_visual  # opens the gripper and goes home itself
            r = open_drawer_visual(self.lab)
        else:
            return result(False, f"unknown provider {provider!r} (analytic|rl|visual)", t0)
        r["evidence"]["provider"] = provider
        r["duration_s"] = round(time.time() - t0, 3)
        return r

    def _open_drawer_analytic(self) -> dict:
        t0, lab = time.time(), self.lab
        h, _ = lab.handle_pose()
        h = h.tolist()
        c, s_ = math.cos(HANDLE_PITCH), math.sin(HANDLE_PITCH)
        steps = [
            ("pregrasp", lambda: lab.move_to([h[0] - 0.10 * c, h[1], h[2] + 0.10 * s_], HANDLE_QUAT)),
            ("open", lambda: lab.gripper(True)),
            ("approach", lambda: lab.move_to([h[0] + 0.005, h[1], h[2]], HANDLE_QUAT, speed=0.1, tol=0.015)),
            ("grasp", lambda: lab.gripper(False, steps=50)),
            ("pull", lambda: lab.move_to([h[0] - self.open_target, h[1], h[2]], HANDLE_QUAT, speed=0.12, tol=0.02)),
            ("release", lambda: lab.gripper(True)),
            ("retreat", lambda: lab.move_to([h[0] - self.open_target - 0.08 * c, h[1], h[2] + 0.10], HANDLE_QUAT, tol=0.02)),
        ]
        trace = []
        for name, fn in steps:
            r = fn()
            trace.append({"step": name, "ok": r["ok"], "reason": r["reason"]})
        opening = lab.drawer_opening()
        ok = opening > 0.18
        return result(ok, f"drawer open {opening:.2f} m" if ok else f"drawer only at {opening:.2f} m (grasp slipped?)",
                      t0, opening_m=round(opening, 3), substeps=trace)

    def _load_rl(self):
        """The selected RL checkpoint (checkpoints.py: team selection or workshop reference), reloaded when its sha256 changes."""
        from vinci_lab import checkpoints

        pol, info = checkpoints.load("open_drawer_rl", lambda p: torch.jit.load(p, map_location=self.lab.device).eval())
        self._rl_policy, self.rl_checkpoint = pol, info
        return pol

    def _open_drawer_rl(self, max_steps: int = 480, target_open: float = 0.30) -> dict:
        t0, lab = time.time(), self.lab
        try:
            pol = self._load_rl()
        except Exception as e:  # noqa: BLE001  missing / mismatching checkpoint: say which
            return result(False, f"RL policy not loaded: {e}", t0)
        # the policy was trained from the stock home configuration: go back there in joint space first
        lab.gripper(True, steps=10)
        lab.move_joints()
        best, n = 0.0, 0
        with torch.inference_mode():
            for n in range(max_steps):
                act = pol(lab.last_obs["policy"])
                lab.step_action(act)
                best = max(best, lab.drawer_opening())
                if lab.drawer_opening() >= target_open:
                    break
        lab.gripper(True)
        h, _ = lab.handle_pose()
        lab.move_to([h[0] - 0.12, h[1], h[2] + 0.05], HANDLE_QUAT, tol=0.02)
        opening = lab.drawer_opening()
        ok = opening > 0.2
        return result(ok, f"drawer open {opening:.2f} m" if ok else f"policy stalled at {best:.2f} m",
                      t0, opening_m=round(opening, 3), policy_steps=n + 1, best_opening_m=round(best, 3),
                      checkpoint_sha256=self.rl_checkpoint["sha256"], checkpoint_source=self.rl_checkpoint["source"])

    def pick_and_place(self, obj: str = "cube", target: str = "top_drawer", provider: str = "scripted") -> dict:
        t0 = time.time()
        if obj != "cube" or target != "top_drawer":
            return result(False, "only obj='cube', target='top_drawer' exist in this scene", t0)
        if provider == "vla":
            if self.vla_client is None:
                return result(False, "no VLA policy connected (vla_client is None)", t0)
            if self.lab.drawer_opening() < 0.18:
                return result(False, f"top drawer not open enough ({self.lab.drawer_opening():.2f} m < 0.18)", t0)
            self.lab.gripper(True, steps=10)
            self.go_pick_start()  # same start state as the demonstrations the VLA was trained on
            r = self.vla_client(self.lab)
            self.lab.hold(30)
            where = self._where_is_cube(self.lab.observe())
            r.setdefault("evidence", {})["cube_where"] = where
            if r.get("ok") and where != "in_top_drawer":
                r["ok"], r["reason"] = False, f"VLA finished but cube ended {where}"
        elif provider == "scripted":
            r = self._pick_and_place_scripted()
        else:
            return result(False, f"unknown provider {provider!r} (scripted|vla)", t0)
        r["evidence"]["provider"] = provider
        r["duration_s"] = round(time.time() - t0, 3)
        return r

    def _pick_and_place_scripted(self) -> dict:
        t0, lab = time.time(), self.lab
        if lab.drawer_opening() < 0.18:
            return result(False, f"top drawer not open enough ({lab.drawer_opening():.2f} m < 0.18)", t0)
        c = lab.observe()["cube_pos"]
        drop = self.drawer_drop_point(z=0.83)
        qc = down_quat(self.grasp_yaw())  # fingers square to the cube faces (a diagonal pinch slips)
        qd = down_quat(math.atan2(drop[1], drop[0]))
        steps = [
            ("above_cube", lambda: lab.move_to([c[0], c[1], c[2] + 0.15], qc)),
            ("open", lambda: lab.gripper(True)),
            ("descend", lambda: lab.move_to([c[0], c[1], c[2] + 0.005], qc, speed=0.12)),
            ("grasp", lambda: lab.gripper(False, steps=40)),
            ("lift", lambda: lab.move_to([c[0], c[1], SAFE_Z], qc, speed=0.2)),
            ("above_drawer", lambda: lab.move_to([drop[0], drop[1], SAFE_Z], qd)),
            ("lower", lambda: lab.move_to(drop, qd, speed=0.12)),
            ("release", lambda: lab.gripper(True, steps=30)),
            ("retreat", lambda: lab.move_to([drop[0], drop[1], SAFE_Z], qd)),
        ]
        trace = []
        for name, fn in steps:
            r = fn()
            trace.append({"step": name, "ok": r["ok"], "reason": r["reason"]})
            if name == "grasp" and not r["ok"]:
                return result(False, "grasp failed: " + r["reason"], t0, substeps=trace)
        lab.hold(40)
        where = self._where_is_cube(lab.observe())
        ok = where == "in_top_drawer"
        return result(ok, "cube placed in top drawer" if ok else f"cube ended {where}", t0, cube_where=where, substeps=trace)

    def close_drawer(self) -> dict:
        t0, lab = time.time(), self.lab
        h, _ = lab.handle_pose()
        h = h.tolist()
        closed_x = h[0] + lab.drawer_opening()  # handle x when the drawer is shut (drawer slides along -x)
        # Brain run 1 lesson: starting right after a place (arm above the drawer) the path to the drawer front
        # crosses the drawer and jams the arm -> first leave through a free via-pose
        if lab.ee_pose()[0][0].item() > h[0] - 0.05 and lab.ee_pose()[0][2].item() > h[2] + 0.05:
            self.go_pick_start()
        z = h[2] - 0.035  # push the drawer face below the handle bar (above it, the fingers slide over the rim)
        trace = []
        plan = [
            ("close_fingers", lambda: lab.gripper(False, steps=25)),
            ("in_front", lambda: lab.move_to([h[0] - 0.08, h[1], z], HANDLE_QUAT, tol=0.02)),
            ("push", lambda: lab.move_to([closed_x + 0.02, h[1], z], HANDLE_QUAT, speed=0.12, tol=0.03)),
        ]
        for name, fn in plan:
            r = fn()
            trace.append({"step": name, "ok": r["ok"], "reason": r["reason"]})
        if lab.drawer_opening() > 0.02:  # one corrective push from wherever the drawer stopped
            h2 = lab.handle_pose()[0].tolist()
            for name, fn in [("back_off", lambda: lab.move_to([h2[0] - 0.08, h2[1], z], HANDLE_QUAT, tol=0.02)),
                             ("push_again", lambda: lab.move_to([closed_x + 0.03, h2[1], z], HANDLE_QUAT, speed=0.1, tol=0.03))]:
                r = fn()
                trace.append({"step": name, "ok": r["ok"], "reason": r["reason"]})
        r = lab.move_to([closed_x - 0.15, h[1], z], HANDLE_QUAT, tol=0.03)
        trace.append({"step": "retreat", "ok": r["ok"], "reason": r["reason"]})
        opening = lab.drawer_opening()
        ok = opening < 0.02
        return result(ok, "drawer closed" if ok else f"drawer still open {opening:.2f} m", t0,
                      opening_m=round(opening, 3), substeps=trace)

    def reset(self) -> dict:
        return self.lab.reset()

    def stop(self) -> dict:
        t0 = time.time()
        self.lab.hold(5)
        self.lab.stopped = True
        return result(True, "robot holding position; further motion refused until reset()", t0)
