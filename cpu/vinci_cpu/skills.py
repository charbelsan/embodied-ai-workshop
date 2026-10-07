"""Skills of the kitting workstation (CPU edition): same names, arguments and {ok, reason, duration_s, evidence}
contract as the GPU workshop (vinci_lab/challenge/skills.py). Ported gestures; heights adapted to the MuJoCo arm."""
from __future__ import annotations

import math
import time

import numpy as np

from . import scene as S
from .lab import DOWN_QUAT, down_quat, result
from .objects import KITTING_TRAY, PIECES, TOP_DRAWER

DRAWER_SLOTS = [(-0.14, -0.15), (-0.14, 0.0), (-0.14, 0.15)]   # from the drawer centre, front half (towards the robot)
TRAY_SLOTS = [(-0.05, -0.05), (0.05, 0.05), (-0.05, 0.05), (0.05, -0.05)]
DRAWER_DROP_Z = 0.83
TRANSPORT_Z = 0.86          # above the drawer rim / front panel (0.78) with a piece in hand
LIFT_Z = 0.66               # first lift, 20 cm above the table: reachable over the whole table
VIA_DRAWER = (0.30, 0.20)   # outside the open drawer: rise there before crossing its front
PICK_START = (0.30, 0.30, 0.90)
OPEN_TARGET = 0.28          # m of drawer travel pulled by the analytic skill (>= 0.18 needed to place)
HANDLE_PITCH = math.radians(20)
PUSH_Y, PUSH_Z = -0.22, 0.67  # close_drawer pushes the face here: beside the handle bar (|y| < 0.11), below the rim


def handle_quat(pitch: float = HANDLE_PITCH) -> tuple:
    """Tool pointing at the cabinet (+x), tilted down by `pitch`; fingers close vertically on the horizontal bar."""
    import mujoco
    c, s = math.cos(pitch), math.sin(pitch)
    z = np.array([c, 0.0, -s]); y = np.array([s, 0.0, c]); x = np.cross(y, z)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).ravel())
    return tuple(float(v) for v in q)


HANDLE_QUAT = handle_quat()


class Skills:
    open_target = OPEN_TARGET

    def __init__(self, lab):
        self.lab = lab
        self._pick_q = None

    # ------------------------------------------------------------------ perception
    def inspect_scene(self) -> dict:
        t0, lab = time.time(), self.lab
        lab.note_inspection()
        ws = lab.observe_workstation()
        ws["gripper"] = {"width_m": round(lab.gripper_width(), 4), "tcp_pos": [round(float(v), 4) for v in lab.ee_pos()]}
        ws["top_drawer"] = {"opening_m": ws.pop("drawer_opening_m"), "max_opening_m": S.DRAWER_MAX_OPEN}
        return result(True, "workstation inspected", t0, **ws)

    def grasp_yaw(self, obj: str) -> float:
        lab, p = self.lab, PIECES[obj]
        c = lab.piece_pos(obj)
        radial = math.atan2(c[1], c[0])
        if p.shape == "cylinder":
            return radial
        period = math.pi / 2 if p.shape == "cube" else math.pi
        yaw = lab.piece_yaw(obj)
        return yaw + round((radial - yaw) / period) * period

    def _slot(self, target: str, obj: str):
        lab = self.lab
        others = [lab.piece_pos(k) for k in lab.pieces if k != obj]
        if target == TOP_DRAWER:
            dp = lab.drawer_pos()
            for sx, sy in DRAWER_SLOTS:
                w = [dp[0] + sx, dp[1] + sy]
                if all(math.dist(w, o[:2]) > 0.06 for o in others):
                    return [w[0], w[1], DRAWER_DROP_Z]
            return None
        for ox, oy in TRAY_SLOTS:
            w = [S.TRAY_CENTER[0] + ox, S.TRAY_CENTER[1] + oy]
            if all(math.dist(w, o[:2]) > 0.05 for o in others):
                return [w[0], w[1], S.TRAY_FLOOR + PIECES[obj].height + 0.03]
        return None

    # ------------------------------------------------------------------ motions
    def _named_target(self, target):
        if isinstance(target, (list, tuple)):
            return [float(v) for v in target], DOWN_QUAT
        if target == "home":
            return [0.45, 0.0, 0.75], DOWN_QUAT
        if target == "handle":
            return self.lab.handle_pos().tolist(), HANDLE_QUAT
        if target in ("above_drawer", "drawer"):
            d = self.lab.drawer_pos()
            return [d[0] - 0.14, d[1], TRANSPORT_Z], down_quat(math.atan2(d[1], d[0]))
        raise ValueError(f"unknown target {target!r}; known: home, handle, above_drawer, or [x,y,z]")

    def move_to(self, target) -> dict:
        self.lab.note_action()
        pos, quat = self._named_target(target)
        r = self.lab.move_to(pos, quat)
        r["evidence"]["target_name"] = target if isinstance(target, str) else "xyz"
        return r

    def go_pick_start(self) -> dict:
        return self.lab.move_to(list(PICK_START), down_quat(math.atan2(PICK_START[1], PICK_START[0])))

    def clear_of_drawer(self) -> dict:
        """Back straight away from the drawer, rise, then go to the pick start (reorienting next to an open drawer
        sweeps the fingers into it)."""
        t0, lab = time.time(), self.lab
        p, q = lab.ee_pose()
        steps = []
        if p[0] > 0.18 and abs(p[1]) < 0.45 and p[2] > 0.55:
            steps.append(lab.move_to([max(0.18, p[0] - 0.14), p[1], p[2] + 0.02], q, tol=0.02))
            steps.append(lab.move_to([0.18, p[1], TRANSPORT_Z], q, tol=0.02))
        steps.append(lab.move_to([0.30, 0.30, TRANSPORT_Z], down_quat(math.atan2(0.30, 0.30)), tol=0.02))
        return result(all(r["ok"] for r in steps), "clear of the drawer", t0, drawer_m=round(lab.drawer_opening(), 3))

    def drawer_settle(self, max_steps: int = 180) -> float:
        for _ in range(max_steps // 10):
            if abs(self.lab.drawer_velocity()) < 0.004:
                break
            self.lab.hold(10)
        return self.lab.drawer_opening()

    def stop(self) -> dict:
        t0 = time.time()
        self.lab.hold(5)
        self.lab.stopped = True
        return result(True, "robot holding position; further motion refused until the next episode", t0)

    # ------------------------------------------------------------------ drawer
    def open_drawer(self, provider: str = "analytic") -> dict:
        t0, lab = time.time(), self.lab
        if provider != "analytic":
            return result(False, f"provider {provider!r} is not available in the CPU edition (analytic only; "
                                 "set providers.open_drawer: analytic in team/config.yaml)", t0, provider=provider)
        lab.note_action()
        h = lab.handle_pos()
        c, s_ = math.cos(HANDLE_PITCH), math.sin(HANDLE_PITCH)
        trace = []

        def do(name, fn):
            r = fn()
            trace.append({"step": name, "ok": r["ok"], "reason": r["reason"]})
            return r

        do("open_fingers", lambda: lab.gripper(True, steps=20))
        do("pregrasp", lambda: lab.move_to([h[0] - 0.10 * c, h[1], h[2] + 0.10 * s_], HANDLE_QUAT, tol=0.01))
        do("approach", lambda: lab.move_to([h[0] + 0.004, h[1], h[2]], HANDLE_QUAT, speed=0.1, tol=0.012))
        do("grasp", lambda: lab.gripper(False, steps=50))
        do("pull", lambda: lab.move_to([h[0] - self.open_target, h[1], h[2]], HANDLE_QUAT, speed=0.12, tol=0.02))
        do("release", lambda: lab.gripper(True, steps=30))
        hp = lab.handle_pos()
        do("retreat", lambda: lab.move_to([hp[0] - 0.10 * c, hp[1], hp[2] + 0.10], HANDLE_QUAT, tol=0.02))
        lab.hold(20)
        cl = self.clear_of_drawer()
        opening = self.drawer_settle()
        ok = opening > 0.18
        return result(ok, f"drawer open {opening:.2f} m" if ok else f"drawer only at {opening:.2f} m (grasp slipped?)", t0,
                      provider=provider, opening_m=round(opening, 3), opening_after_clear_m=cl["evidence"]["drawer_m"],
                      substeps=trace)

    def close_drawer(self) -> dict:
        """Push the drawer face with the closed fingers pointing down, beside the handle (y = PUSH_Y): clear of the
        handle bar and within reach all the way to the closed position."""
        t0, lab = time.time(), self.lab
        lab.note_action()
        trace = []

        def do(name, fn):
            r = fn()
            trace.append({"step": name, "ok": r["ok"], "reason": r["reason"]})
            return r

        face_closed_x = S.DRAWER_FRONT_X - 0.02          # front face of the shut drawer
        q = down_quat(0.0)
        y, z = PUSH_Y, PUSH_Z

        def face():
            return face_closed_x - lab.drawer_opening()

        p = lab.ee_pos()
        if p[0] > face() - 0.03:                          # above the open drawer: come back over its front at height
            do("leave_drawer", lambda: lab.move_to([face() - 0.06, p[1], TRANSPORT_Z], lab.ee_pose()[1], tol=0.02))
        do("close_fingers", lambda: lab.gripper(False, steps=25))
        do("above_face", lambda: lab.move_to([face() - 0.04, y, TRANSPORT_Z], q, tol=0.02))
        do("down", lambda: lab.move_to([face() - 0.04, y, z], q, tol=0.015))
        do("push", lambda: lab.move_to([face_closed_x + 0.015, y, z], q, speed=0.12, tol=0.03))
        if lab.drawer_opening() > 0.02:  # one corrective push from wherever the drawer stopped
            do("back_off", lambda: lab.move_to([face() - 0.04, y, z], q, tol=0.02))
            do("push_again", lambda: lab.move_to([face_closed_x + 0.025, y, z], q, speed=0.1, tol=0.03))
        do("retreat", lambda: lab.move_to([lab.ee_pos()[0] - 0.08, y, TRANSPORT_Z], q, tol=0.03))
        opening = lab.drawer_opening()
        ok = opening < 0.02
        return result(ok, "drawer closed" if ok else f"drawer still open {opening:.2f} m", t0,
                      opening_m=round(opening, 3), substeps=trace)

    # ------------------------------------------------------------------ manipulation
    def _posture(self):
        """Arm posture above the table (computed once by IK): resets the elbow before reaching down."""
        lab = self.lab
        if self._pick_q is None:
            lab.move_to(list(PICK_START), down_quat(math.atan2(PICK_START[1], PICK_START[0])))
            self._pick_q = lab.joint_pos().tolist()
            return result(True, "posture computed", time.time())
        return lab.move_joints(self._pick_q)

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
        c = lab.piece_pos(obj)
        qc = down_quat(self.grasp_yaw(obj))
        grasp_z = c[2] + (0.005 if p.height >= 0.04 else 0.0)
        trace = []

        def do(name, fn):
            r = fn()
            trace.append({"step": name, "ok": r["ok"], "reason": r["reason"], "drawer_m": round(lab.drawer_opening(), 3),
                          "piece_z": round(float(lab.piece_pos(obj)[2]), 3)})
            return r

        def evidence():
            return dict(piece_where=lab.where(obj), substeps=trace, piece_pos=[round(float(v), 3) for v in lab.piece_pos(obj)],
                        piece_tilt_deg=round(lab.piece_tilt_deg(obj), 1))

        def put_back(reason):
            do("put_back", lambda: lab.move_to([c[0], c[1], grasp_z + 0.01], qc, speed=0.12))
            do("release", lambda: lab.gripper(True, steps=30))
            do("retreat", lambda: lab.move_to([c[0], c[1], TRANSPORT_Z], qc))
            return result(False, reason, t0, **evidence())

        tcp = lab.ee_pos()
        if tcp[0] > 0.38 or tcp[1] < 0.30:
            do("via_clear", lambda: lab.move_to([0.30, 0.30, TRANSPORT_Z], down_quat(math.atan2(0.30, 0.30)), tol=0.02))
        do("posture", self._posture)
        do("above_piece", lambda: lab.move_to([c[0], c[1], c[2] + 0.15], qc))
        do("open", lambda: lab.gripper(True))
        do("descend", lambda: lab.move_to([c[0], c[1], grasp_z], qc, speed=0.12))
        g = do("grasp", lambda: lab.gripper(False, steps=40))
        if not g["ok"]:
            lab.gripper(True, steps=15)
            lab.move_to([c[0], c[1], c[2] + 0.15], qc)
            return result(False, "grasp failed: " + g["reason"], t0, **evidence())
        do("lift", lambda: lab.move_to([c[0], c[1], LIFT_Z], qc, speed=0.2))
        if target == TOP_DRAWER and lab.drawer_opening() < 0.18:
            return put_back(f"drawer closed to {lab.drawer_opening():.2f} m during the pick: {obj} put back on the table")
        drop = self._slot(target, obj)
        if drop is None:
            return put_back(f"no free slot in {target}: {obj} put back on the table")
        qd = down_quat(math.atan2(drop[1], drop[0]))
        if target == TOP_DRAWER:
            do("via", lambda: lab.move_to([VIA_DRAWER[0], VIA_DRAWER[1], TRANSPORT_Z], down_quat(math.atan2(VIA_DRAWER[1], VIA_DRAWER[0])), tol=0.02))
        do("above_target", lambda: lab.move_to([drop[0], drop[1], TRANSPORT_Z], qd))
        do("lower", lambda: lab.move_to(drop, qd, speed=0.12))
        do("release", lambda: lab.gripper(True, steps=30))
        do("retreat", lambda: lab.move_to([drop[0], drop[1], TRANSPORT_Z], qd))
        lab.hold(30)
        where = lab.where(obj)
        if where in (TOP_DRAWER, KITTING_TRAY) and where != lab.rules.get(obj):
            lab.events["wrong_place"][obj] = True
        ok = where == target
        return result(ok, f"{obj} placed in {target}" if ok else f"{obj} ended {where}", t0, **evidence())
