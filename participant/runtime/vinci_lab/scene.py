"""VINCI lab scene: Isaac Lab's Franka + Sektion cabinet task, extended with a stand, a cube and two cameras.

The `policy` observation group and the action space are left IDENTICAL to Isaac-Open-Drawer-Franka-v0,
so an RL policy trained on the stock task runs unchanged here, and every skill (IK, RL, VLA) speaks
the same 8-D action: 7 arm joint-position offsets from the default pose + 1 binary gripper command.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.agents.rsl_rl_ppo_cfg import CabinetPPORunnerCfg
from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.joint_pos_env_cfg import FrankaCabinetEnvCfg

STAND_SIZE = (0.12, 0.12, 0.45)
STAND_POS = (0.25, 0.50)  # in reach, clear of the open drawer (open 0.3: x in [0.38, 0.68], |y| <= 0.38)
CUBE_REST_Z = STAND_SIZE[2] + 0.045 / 2
CUBE_SIZE = 0.045
IMG_H = IMG_W = 256
FRONT_EYE = (-0.55, 1.10, 1.55)
FRONT_TARGET = (0.55, 0.12, 0.60)
TASK_TEXT = "put the red cube in the top drawer and close it"
TASK_PICK = "put the red cube in the top drawer"  # language instruction of the VLA pick_and_place demos
# a floor-mounted Franka pointing down cannot lift its fingertips above ~0.76 m near its base, while the
# top drawer rim is at 0.78 m: the robot stands on a pedestal, like a counter-top arm
PEDESTAL_H = 0.25
# stock cabinet sits at x=0.8; with the raised base its handle would be too close to the shoulder when pulled open
CABINET_X = 1.0


def _vinci_dynamics(cfg: FrankaCabinetEnvCfg):
    """Shared by the RL training task and the lab scene, so a trained policy transfers as-is."""
    # stock FRANKA_PANDA_CFG (low PD + gravity) sags ~0.2 rad when holding a pose: IK cannot track it
    cfg.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.robot.init_state.pos = (0.0, 0.0, PEDESTAL_H)
    cfg.scene.cabinet.init_state.pos = (CABINET_X, 0.0, 0.4)
    cfg.scene.pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        spawn=sim_utils.CuboidCfg(size=(0.30, 0.30, PEDESTAL_H), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.28))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, PEDESTAL_H / 2)),
    )
    # stock task springs the drawer shut (stiffness 10); here a released drawer must stay where it was left
    cfg.scene.cabinet.actuators["drawers"].stiffness = 0.0
    cfg.scene.cabinet.actuators["drawers"].damping = 2.5
    cfg.scene.cabinet_frame.debug_vis = False  # frame markers would leak into the VLA camera images


@configclass
class VinciOpenDrawerRLEnvCfg(FrankaCabinetEnvCfg):
    """RL training task for the open_drawer primitive: stock Isaac Lab MDP, VINCI robot/drawer dynamics."""

    def __post_init__(self):
        super().__post_init__()
        _vinci_dynamics(self)


@configclass
class VinciOpenDrawerRLEnvCfg_PLAY(VinciOpenDrawerRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False


@configclass
class VinciOpenDrawerPPORunnerCfg(CabinetPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "vinci_open_drawer"


@configclass
class VinciCabinetEnvCfg(FrankaCabinetEnvCfg):
    """1-env interactive lab scene (cameras on). Use `with_cameras=False` style overrides for fast loops."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0
        self.episode_length_s = 3600.0  # no timeout-driven auto-reset during an interactive session
        self.observations.policy.enable_corruption = False
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        _vinci_dynamics(self)

        self.scene.stand = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Stand",
            spawn=sim_utils.CuboidCfg(
                size=STAND_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(STAND_POS[0], STAND_POS[1], STAND_SIZE[2] / 2)),
        )
        self.scene.cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.1, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(STAND_POS[0], STAND_POS[1], STAND_SIZE[2] + CUBE_SIZE / 2 + 0.002)
            ),
        )
        self.scene.front_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/FrontCam",
            update_period=0.0,
            height=IMG_H,
            width=IMG_W,
            data_types=["rgb", "distance_to_image_plane"],  # depth: input of the DAgger vision student
            spawn=sim_utils.PinholeCameraCfg(focal_length=16.0, clipping_range=(0.05, 20.0)),
        )
        # wrist camera pose borrowed from Isaac Lab's stack visuomotor task (panda_hand mount)
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=IMG_H,
            width=IMG_W,
            data_types=["rgb", "distance_to_image_plane"],  # depth: input of the DAgger vision student
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.02, 5.0)),
            offset=CameraCfg.OffsetCfg(pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"),
        )
        self.viewer.eye = FRONT_EYE
        self.viewer.lookat = FRONT_TARGET
