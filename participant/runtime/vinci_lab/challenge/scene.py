"""Kitting workstation scene: the lab scene (robot on pedestal, cabinet, cameras) + work table + kitting tray + pieces."""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from ..scene import VinciCabinetEnvCfg
from .objects import PIECES

TABLE_TOP = 0.45
# the OPEN drawer (|y| <= 0.38, z 0.66-0.78, x >= 0.34) must never overhang the pieces: reaching under it pushes it shut
TABLE_CENTER = (0.25, 0.56)
TABLE_SIZE = (0.36, 0.34)  # x in [0.07, 0.43], y in [0.39, 0.73]
TRAY_CENTER = (0.35, -0.45)
TRAY_INNER = (0.22, 0.22)
TRAY_WALL_H = 0.06
TRAY_FLOOR = TABLE_TOP  # tray floor at table height (on its own column)
_T = 0.01  # wall / floor thickness


def _static_box(name, size, pos, color):
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos))


def _piece_cfg(pid: str, default_pos):
    p = PIECES[pid]
    common = dict(rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
                  mass_props=sim_utils.MassPropertiesCfg(mass=p.mass),
                  collision_props=sim_utils.CollisionPropertiesCfg(),
                  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=p.color))
    if p.shape == "cylinder":
        spawn = sim_utils.CylinderCfg(radius=p.size[0] / 2, height=p.size[2], axis="Z", **common)
    else:
        spawn = sim_utils.CuboidCfg(size=p.size, **common)
    return RigidObjectCfg(prim_path="{ENV_REGEX_NS}/" + pid, spawn=spawn,
                          init_state=RigidObjectCfg.InitialStateCfg(pos=(default_pos[0], default_pos[1], TABLE_TOP + p.height / 2 + 0.002)))


@configclass
class ChallengeEnvCfg(VinciCabinetEnvCfg):
    """Pieces are chosen by the level; poses/physics are randomized per episode by ChallengeLab."""

    pieces: tuple = ("bolt_box", "sensor", "bracket", "gauge")

    def __post_init__(self):
        super().__post_init__()
        self.scene.stand = None
        self.scene.cube = None
        tx, ty = TABLE_CENTER
        self.scene.worktable = _static_box("WorkTable", (TABLE_SIZE[0], TABLE_SIZE[1], TABLE_TOP), (tx, ty, TABLE_TOP / 2), (0.55, 0.55, 0.60))
        cx, cy = TRAY_CENTER
        ix, iy = TRAY_INNER
        grey = (0.35, 0.38, 0.42)
        self.scene.tray_column = _static_box("TrayColumn", (ix, iy, TRAY_FLOOR - _T), (cx, cy, (TRAY_FLOOR - _T) / 2), grey)
        self.scene.tray_floor = _static_box("TrayFloor", (ix + 2 * _T, iy + 2 * _T, _T), (cx, cy, TRAY_FLOOR - _T / 2), (0.2, 0.5, 0.25))
        zc = TRAY_FLOOR + TRAY_WALL_H / 2
        self.scene.tray_wall_n = _static_box("TrayWallN", (ix + 2 * _T, _T, TRAY_WALL_H), (cx, cy + iy / 2 + _T / 2, zc), grey)
        self.scene.tray_wall_s = _static_box("TrayWallS", (ix + 2 * _T, _T, TRAY_WALL_H), (cx, cy - iy / 2 - _T / 2, zc), grey)
        self.scene.tray_wall_e = _static_box("TrayWallE", (_T, iy, TRAY_WALL_H), (cx + ix / 2 + _T / 2, cy, zc), grey)
        self.scene.tray_wall_w = _static_box("TrayWallW", (_T, iy, TRAY_WALL_H), (cx - ix / 2 - _T / 2, cy, zc), grey)
        # spawn poses spread along the table (overwritten at every episode reset)
        for k, pid in enumerate(self.pieces):
            setattr(self.scene, pid, _piece_cfg(pid, (0.14 + 0.07 * (k % 4), 0.48 + 0.12 * (k // 4))))
        # front camera: see table, drawer and tray
        self.viewer.eye = (-0.70, 0.05, 1.75)
        self.viewer.lookat = (0.45, 0.05, 0.45)
