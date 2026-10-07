"""The kitting workstation in MuJoCo (CPU): same layout as the GPU workshop, in metres, robot base at the origin.

Franka Emika Panda from MuJoCo Menagerie (Apache-2.0, fetched by setup.sh) on a 0.25 m pedestal, a work table,
a kitting tray, a cabinet whose top drawer slides towards the robot, and the work-order pieces.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import mujoco
import numpy as np

from .objects import PIECES

PEDESTAL_H = 0.25
TABLE_TOP = 0.45
TABLE_CENTER = (0.25, 0.56)
TABLE_SIZE = (0.36, 0.34)
TRAY_CENTER = (0.35, -0.45)
TRAY_INNER = (0.22, 0.22)
TRAY_WALL_H = 0.06
TRAY_FLOOR = TABLE_TOP
_T = 0.01
# top drawer, closed: front face at x = DRAWER_FRONT_X, it slides towards the robot (-x) by `opening`
DRAWER_FRONT_X = 0.70
DRAWER_DEPTH = 0.50
DRAWER_HALF_W = 0.36
DRAWER_FLOOR_Z = 0.66
DRAWER_WALL_H = 0.12
HANDLE_X_OFF = 0.045       # handle bar axis, in front of the front panel
HANDLE_Z = 0.73
HANDLE_R = 0.008
DRAWER_MAX_OPEN = 0.42
FRONT_EYE, FRONT_TARGET = (-0.70, 0.05, 1.75), (0.45, 0.05, 0.45)

MENAGERIE = Path(os.environ.get("VINCI_CPU_MENAGERIE", Path(__file__).resolve().parents[1] / ".cache" / "mujoco_menagerie"))


def _rgba(c, a=1.0):
    return [c[0], c[1], c[2], a]


def _box(body, name, half, pos, rgba, contype=1):
    g = body.add_geom()
    g.name, g.type, g.size, g.pos, g.rgba = name, mujoco.mjtGeom.mjGEOM_BOX, half, pos, rgba
    g.contype = g.conaffinity = contype
    return g


def _look_at_xyaxes(eye, target):
    f = np.array(target, float) - np.array(eye, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, f)
    return list(right) + list(up)


def build(pieces=("bolt_box", "sensor", "bracket", "gauge"), cam_res: int = 128) -> mujoco.MjModel:
    panda = MENAGERIE / "franka_emika_panda" / "panda.xml"
    if not panda.exists():
        raise FileNotFoundError(f"Franka model missing: {panda} (run cpu/setup.sh)")
    spec = mujoco.MjSpec.from_file(str(panda))
    spec.option.timestep = 1.0 / 600.0           # 10 physics steps per 60 Hz control step
    # grasping small parts: elliptic friction cones + high impratio (MuJoCo's advice); noslip made them explode
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
    spec.body("link0").pos = [0.0, 0.0, PEDESTAL_H]
    for name in ["link%d" % i for i in range(8)] + ["hand", "left_finger", "right_finger"]:
        spec.body(name).gravcomp = 1.0             # no gravity sag of the position servos (millimetre-accurate IK)
    for name in ("left_finger", "right_finger"):   # fingertip pads: high friction, soft contact
        for g in spec.body(name).geoms:
            if g.contype or g.conaffinity:
                g.friction = [1.5, 0.01, 0.0005]
                g.condim = 4
    # gripper: Menagerie's position servo squeezes ~1-2 N; a real Franka hand grips up to 70 N. Stiffer servo, same
    # 0-255 command (255 = open 8 cm), force capped at 70 N
    g8 = spec.actuator("actuator8")
    kp = 2000.0
    g8.gainprm[0], g8.biasprm[1], g8.biasprm[2] = kp * 0.04 / 255.0, -kp, -100.0
    g8.forcerange = [-70.0, 70.0]
    wb = spec.worldbody
    light = wb.add_light(); light.pos, light.dir = [0.3, 0.0, 2.5], [0.0, 0.0, -1.0]
    floor = wb.add_geom(); floor.type, floor.size, floor.rgba = mujoco.mjtGeom.mjGEOM_PLANE, [3, 3, 0.05], [0.82, 0.82, 0.80, 1]
    floor.name = "floor"
    _box(wb, "pedestal", [0.15, 0.15, PEDESTAL_H / 2], [0, 0, PEDESTAL_H / 2], [0.25, 0.25, 0.28, 1])
    tx, ty = TABLE_CENTER
    _box(wb, "worktable", [TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2, TABLE_TOP / 2], [tx, ty, TABLE_TOP / 2], [0.55, 0.55, 0.60, 1])
    cx, cy = TRAY_CENTER; ix, iy = TRAY_INNER; grey = [0.35, 0.38, 0.42, 1]
    _box(wb, "tray_column", [ix / 2, iy / 2, (TRAY_FLOOR - _T) / 2], [cx, cy, (TRAY_FLOOR - _T) / 2], grey)
    _box(wb, "tray_floor", [ix / 2 + _T, iy / 2 + _T, _T / 2], [cx, cy, TRAY_FLOOR - _T / 2], [0.2, 0.5, 0.25, 1])
    zc = TRAY_FLOOR + TRAY_WALL_H / 2
    _box(wb, "tray_wall_n", [ix / 2 + _T, _T / 2, TRAY_WALL_H / 2], [cx, cy + iy / 2 + _T / 2, zc], grey)
    _box(wb, "tray_wall_s", [ix / 2 + _T, _T / 2, TRAY_WALL_H / 2], [cx, cy - iy / 2 - _T / 2, zc], grey)
    _box(wb, "tray_wall_e", [_T / 2, iy / 2, TRAY_WALL_H / 2], [cx + ix / 2 + _T / 2, cy, zc], grey)
    _box(wb, "tray_wall_w", [_T / 2, iy / 2, TRAY_WALL_H / 2], [cx - ix / 2 - _T / 2, cy, zc], grey)
    # cabinet carcass (static): a shelf the drawer slides on, side panels, top, back, base
    wood = [0.78, 0.74, 0.68, 1]
    x0, x1 = DRAWER_FRONT_X + 0.02, DRAWER_FRONT_X + DRAWER_DEPTH + 0.06
    xm, xh = (x0 + x1) / 2, (x1 - x0) / 2
    _box(wb, "cab_shelf", [xh, DRAWER_HALF_W + 0.04, 0.01], [xm, 0, DRAWER_FLOOR_Z - 0.015], wood)
    _box(wb, "cab_base", [xh, DRAWER_HALF_W + 0.04, (DRAWER_FLOOR_Z - 0.035) / 2], [xm, 0, (DRAWER_FLOOR_Z - 0.035) / 2], wood)
    for s in (-1, 1):
        _box(wb, f"cab_side_{'l' if s > 0 else 'r'}", [xh, 0.015, 0.44], [xm, s * (DRAWER_HALF_W + 0.055), 0.44], wood)
    _box(wb, "cab_top", [xh, DRAWER_HALF_W + 0.07, 0.015], [xm, 0, 0.895], wood)
    _box(wb, "cab_back", [0.015, DRAWER_HALF_W + 0.04, 0.44], [x1 + 0.015, 0, 0.44], wood)
    # the top drawer: an open box on a slide joint; positive joint value = opened towards the robot
    dr = wb.add_body(); dr.name = "drawer"; dr.pos = [DRAWER_FRONT_X + DRAWER_DEPTH / 2, 0.0, DRAWER_FLOOR_Z]
    j = dr.add_joint(); j.name, j.type, j.axis = "drawer_slide", mujoco.mjtJoint.mjJNT_SLIDE, [-1.0, 0.0, 0.0]
    j.range, j.limited, j.damping, j.frictionloss = [0.0, DRAWER_MAX_OPEN], mujoco.mjtLimited.mjLIMITED_TRUE, [2.5, 0.0, 0.0], 0.2
    j.armature = 0.05
    hw, hd, inner = DRAWER_HALF_W, DRAWER_DEPTH / 2, [0.70, 0.66, 0.60, 1]
    _box(dr, "drawer_floor", [hd, hw, 0.005], [0, 0, 0.005], inner).mass = 1.5
    _box(dr, "drawer_back", [0.008, hw, DRAWER_WALL_H / 2], [hd - 0.008, 0, DRAWER_WALL_H / 2], inner).mass = 0.3
    for s in (-1, 1):
        _box(dr, f"drawer_side_{'l' if s > 0 else 'r'}", [hd, 0.008, DRAWER_WALL_H / 2], [0, s * (hw - 0.008), DRAWER_WALL_H / 2], inner).mass = 0.3
    # front panel: from below the shelf up to the drawer rim (0.78 m), so a carried piece clears it at transport height
    front = _box(dr, "drawer_front", [0.01, hw + 0.03, 0.085], [-hd - 0.01, 0, DRAWER_WALL_H - 0.085], [0.62, 0.58, 0.52, 1])
    front.mass = 0.5
    for s in (-1, 1):  # handle standoffs + bar (along y)
        _box(dr, f"handle_post_{'l' if s > 0 else 'r'}", [HANDLE_X_OFF / 2, 0.006, 0.006],
             [-hd - 0.02 - HANDLE_X_OFF / 2, s * 0.09, HANDLE_Z - DRAWER_FLOOR_Z], [0.2, 0.2, 0.22, 1]).mass = 0.05
    bar = dr.add_geom(); bar.name, bar.type = "handle", mujoco.mjtGeom.mjGEOM_CAPSULE
    bar.size = [HANDLE_R, 0.11, 0]; bar.pos = [-hd - 0.02 - HANDLE_X_OFF, 0, HANDLE_Z - DRAWER_FLOOR_Z]
    bar.quat = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]   # capsule axis along y
    bar.rgba, bar.friction, bar.condim, bar.mass = [0.2, 0.2, 0.22, 1], [1.5, 0.01, 0.0005], 4, 0.05
    # pieces: free bodies
    for pid in pieces:
        p = PIECES[pid]
        b = wb.add_body(); b.name = pid; b.pos = [0.2, 0.55, TABLE_TOP + p.height / 2 + 0.002]
        fj = b.add_freejoint(); fj.name = f"{pid}_free"
        g = b.add_geom(); g.name = f"{pid}_geom"
        if p.shape == "cylinder":
            g.type, g.size = mujoco.mjtGeom.mjGEOM_CYLINDER, [p.size[0] / 2, p.size[2] / 2, 0]
        else:
            g.type, g.size = mujoco.mjtGeom.mjGEOM_BOX, [s / 2 for s in p.size]
        g.rgba, g.mass, g.condim, g.friction = _rgba(p.color), p.mass, 4, [1.2, 0.01, 0.0005]
    # cameras: front (sees table, drawer, tray) and wrist (on the hand, looking along the fingers)
    cam = wb.add_camera(); cam.name = "front"; cam.pos = list(FRONT_EYE); cam.fovy = 55
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES; cam.alt.xyaxes = _look_at_xyaxes(FRONT_EYE, FRONT_TARGET)
    wc = spec.body("hand").add_camera(); wc.name = "wrist"; wc.pos = [0.06, 0.0, 0.02]; wc.fovy = 70
    wc.quat = [0.0, 1.0, 0.0, 0.0]  # looks along +z of the hand (towards the fingers)
    tcp = spec.body("hand").add_site(); tcp.name, tcp.pos, tcp.size = "tcp", [0, 0, 0.1034], [0.005, 0, 0]
    tcp.rgba = [1, 0, 0, 0]
    return spec.compile()
