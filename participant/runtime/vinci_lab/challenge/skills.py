"""Skills on the kitting workstation (same contract {ok, reason, duration_s, evidence}), multi-object."""
from __future__ import annotations

import math
import time

from pathlib import Path

import torch

from ..api import down_quat, result
from ..skills import SAFE_Z, Skills
from .objects import KITTING_TRAY, PIECES, TOP_DRAWER
from .scene import TRAY_CENTER, TRAY_FLOOR

DRAWER_SLOTS = [(0.14, -0.15), (0.14, 0.0), (0.14, 0.15)]  # drawer-local xy (front half of the tray, reachable)
TRAY_SLOTS = [(-0.05, -0.05), (0.05, 0.05), (-0.05, 0.05), (0.05, -0.05)]  # world offsets from the tray center
DRAWER_DROP_Z = 0.83
PICK_Q = [0.3439, -0.2988, 0.3270, -1.5399, 0.0979, 1.2639, 0.6911]  # arm config of PICK_START (tcp 0.30, 0.30, 0.90)
TRANSPORT_Z = 0.96  # above the drawer's front panel: at 0.86 the carried piece clipped it and pushed the drawer shut


class ChallengeSkills(Skills):
    open_target = 0.33  # wider than the lab (0.25): margin for a drawer that gets nudged back while working
    # ------------------------------------------------------------------ perception
    def inspect_scene(self) -> dict:
        t0, lab = time.time(), self.lab
        lab.note_inspection()
        ws = lab.observe_workstation()
        ws["gripper"] = {"width_m": round(float(lab.robot.data.joint_pos[0, -2:].sum()), 4),
                         "tcp_pos": [round(v, 4) for v in lab.ee_pose()[0].tolist()]}
        ws["top_drawer"] = {"opening_m": ws.pop("drawer_opening_m"), "max_opening_m": 0.4}
        return result(True, "workstation inspected", t0, **ws)

    def grasp_yaw(self, obj: str) -> float:
        """Top-down grasp yaw: aligned with the piece faces (cube: every 90 deg, bar: across its short side, cylinder:
        any), choosing the one closest to the radial direction (least wrist twist). FROZEN as measured."""
        lab, p = self.lab, PIECES[obj]
        c = lab.piece_pos(obj).tolist()
        radial = math.atan2(c[1], c[0])
        if p.shape == "cylinder":
            return radial
        period = math.pi / 2 if p.shape == "cube" else math.pi
        yaw = lab.piece_yaw(obj)
        return yaw + round((radial - yaw) / period) * period

    def _slot(self, target: str, obj: str):
        lab = self.lab
        others = [lab.piece_pos(k).tolist() for k in lab.pieces if k != obj]
        if target == TOP_DRAWER:
            from isaaclab.utils.math import quat_apply
            import torch
            dp, dq = lab.drawer_pose()
            for sx, sy in DRAWER_SLOTS:
                w = (dp + quat_apply(dq[None], torch.tensor([[sx, sy, 0.0]], device=lab.device))[0]).tolist()
                if all(math.dist(w[:2], o[:2]) > 0.06 for o in others):
                    return [w[0], w[1], DRAWER_DROP_Z]
            return None
        for ox, oy in TRAY_SLOTS:
            w = [TRAY_CENTER[0] + ox, TRAY_CENTER[1] + oy]
            if all(math.dist(w, o[:2]) > 0.05 for o in others):
                return [w[0], w[1], TRAY_FLOOR + PIECES[obj].height + 0.03]
        return None



    # ------------------------------------------------------------------ drawer-safe motions
    def clear_of_drawer(self) -> dict:
        """Back straight away from the drawer (keep orientation), rise above the cabinet, THEN reorient.
        Reorienting next to an open, low-damping drawer sweeps the fingers into it and pushes it closed."""
        t0, lab = time.time(), self.lab
        p, q = lab.ee_pose()
        p, q = p.tolist(), tuple(q.tolist())
        steps = []
        if p[0] > 0.18 and abs(p[1]) < 0.45 and p[2] > 0.55:
            steps.append(lab.move_to([max(0.18, p[0] - 0.14), p[1], p[2] + 0.02], q, tol=0.02))
            steps.append(lab.move_to([0.18, p[1], 0.96], q, tol=0.02))
        steps.append(self.lab.move_to([0.30, 0.30, TRANSPORT_Z], down_quat(math.atan2(0.30, 0.30)), tol=0.02))
        return result(all(r["ok"] for r in steps), "clear of the drawer", t0, drawer_m=round(lab.drawer_opening(), 3))

    def drawer_settle(self, max_steps: int = 180) -> float:
        """Wait until the drawer stops sliding (a pushed low-damping drawer keeps coasting for seconds)."""
        for _ in range(max_steps // 10):
            if abs(float(self.lab.cabinet.data.joint_vel[0, self.lab.drawer_jid])) < 0.004:
                break
            self.lab.hold(10)
        return self.lab.drawer_opening()

    def move_to(self, target) -> dict:
        self.lab.note_action()
        return super().move_to(target)

    def close_drawer(self) -> dict:
        self.lab.note_action()
        return super().close_drawer()

    def open_drawer(self, provider: str = "rl") -> dict:
        self.lab.note_action()
        r = super().open_drawer(provider)
        self.lab.hold(20)
        c = self.clear_of_drawer()
        self.drawer_settle()
        r["evidence"]["opening_after_clear_m"] = c["evidence"]["drawer_m"]
        return r

    # ------------------------------------------------------------------ manipulation
    def pick_and_place(self, obj: str = "bolt_box", target: str = TOP_DRAWER, provider: str = "scripted") -> dict:
        t0, lab = time.time(), self.lab
        if obj not in lab.pieces:
            return result(False, f"unknown piece {obj!r}; pieces: {sorted(lab.pieces)}", t0)
        if target not in (TOP_DRAWER, KITTING_TRAY):
            return result(False, f"unknown target {target!r}: top_drawer | kitting_tray", t0)
        if provider != "scripted":
            return result(False, f"provider {provider!r} not available on the workstation (scripted)", t0)
        if target == TOP_DRAWER and lab.drawer_opening() < 0.18:
            return result(False, f"top drawer not open enough ({lab.drawer_opening():.2f} m < 0.18)", t0)
        lab.note_action()
        p = PIECES[obj]
        c = lab.piece_pos(obj).tolist()
        qc = down_quat(self.grasp_yaw(obj))
        grasp_z = c[2] + (0.005 if p.height >= 0.04 else 0.0)
        trace = []

        def do(name, fn):
            r = fn()
            trace.append({"step": name, "ok": r["ok"], "reason": r["reason"], "drawer_m": round(lab.drawer_opening(), 3),
                          "piece_z": round(float(lab.piece_pos(obj)[2]), 3)})
            return r

        def put_back(reason):  # recovery inside the skill: never carry a piece towards a target we cannot use
            do("put_back", lambda: lab.move_to([c[0], c[1], grasp_z + 0.01], qc, speed=0.12))
            do("release", lambda: lab.gripper(True, steps=30))
            do("retreat", lambda: lab.move_to([c[0], c[1], TRANSPORT_Z], qc))
            return result(False, reason, t0, piece_where=lab.where(obj), substeps=trace,
                          piece_pos=[round(v, 3) for v in lab.piece_pos(obj).tolist()], piece_tilt_deg=round(lab.piece_tilt_deg(obj), 1))

        # reset the arm posture above the table first: coming straight from a drawer slot, the (path-dependent) IK keeps
        # the elbow on the drawer side and the arm catches the open drawer while reaching down to the table
        tcp = lab.ee_pose()[0].tolist()
        if tcp[0] > 0.38 or tcp[1] < 0.30:
            do("via_clear", lambda: lab.move_to([0.30, 0.30, TRANSPORT_Z], down_quat(math.atan2(0.30, 0.30)), tol=0.02))
        do("posture", lambda: lab.move_joints(PICK_Q))
        do("above_piece", lambda: lab.move_to([c[0], c[1], c[2] + 0.15], qc))
        do("open", lambda: lab.gripper(True))
        do("descend", lambda: lab.move_to([c[0], c[1], grasp_z], qc, speed=0.12))
        g = do("grasp", lambda: lab.gripper(False, steps=40))
        if not g["ok"]:
            lab.gripper(True, steps=15)
            lab.move_to([c[0], c[1], c[2] + 0.15], qc)
            return result(False, "grasp failed: " + g["reason"], t0, piece_where=lab.where(obj), substeps=trace,
                          piece_pos=[round(v, 3) for v in lab.piece_pos(obj).tolist()], piece_tilt_deg=round(lab.piece_tilt_deg(obj), 1))
        do("lift", lambda: lab.move_to([c[0], c[1], TRANSPORT_Z], qc, speed=0.2))
        # the target is read NOW (after the grasp), not at the start: the drawer may have moved meanwhile
        if target == TOP_DRAWER and lab.drawer_opening() < 0.18:
            return put_back(f"drawer closed to {lab.drawer_opening():.2f} m during the pick: {obj} put back on the table")
        drop = self._slot(target, obj)
        if drop is None:
            return put_back(f"no free slot in {target}: {obj} put back on the table")
        qd = down_quat(math.atan2(drop[1], drop[0]))
        do("above_target", lambda: lab.move_to([drop[0], drop[1], TRANSPORT_Z], qd))
        do("lower", lambda: lab.move_to(drop, qd, speed=0.12))
        do("release", lambda: lab.gripper(True, steps=30))
        do("retreat", lambda: lab.move_to([drop[0], drop[1], TRANSPORT_Z], qd))
        lab.hold(30)
        where = lab.where(obj)
        if where in (TOP_DRAWER, KITTING_TRAY) and where != lab.rules.get(obj):
            lab.events["wrong_place"][obj] = True  # placed in a container that the work order forbids
        ok = where == target
        return result(ok, f"{obj} placed in {target}" if ok else f"{obj} ended {where}", t0, piece_where=where, substeps=trace,
                      piece_pos=[round(v, 3) for v in lab.piece_pos(obj).tolist()], piece_tilt_deg=round(lab.piece_tilt_deg(obj), 1))
