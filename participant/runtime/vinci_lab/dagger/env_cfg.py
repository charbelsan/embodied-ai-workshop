"""DAgger track: one MDP, two observers.

- TEACHER (privileged): sees the simulator's truth (handle pose, drawer joint, drawer damping/friction...).
  Trained with PPO, no cameras, thousands of parallel envs.
- STUDENT (deployable): sees only what a real robot would have: 2 RGB-D cameras (same poses as the lab
  scene) + joint positions/velocities. Trained by supervised learning (BC, then DAgger) from teacher labels.

Both share the robot, the randomized cabinet and the 8-D action (7 arm joint offsets + binary gripper),
so a teacher action is a valid label for any state the student visits.
"""
from __future__ import annotations

import math

import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.agents.rsl_rl_ppo_cfg import CabinetPPORunnerCfg
from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.joint_pos_env_cfg import FrankaCabinetEnvCfg

from vinci_lab.scene import CUBE_SIZE, FRONT_EYE, FRONT_TARGET, STAND_POS, STAND_SIZE, _vinci_dynamics

# ---------------------------------------------------------------- randomization (one row per episode)
# columns: cabinet dx, dy [m], dyaw [rad], drawer damping scale, drawer joint friction, 7 arm-joint offsets [rad]
PARAM_NAMES = ["cab_dx", "cab_dy", "cab_dyaw", "drawer_damping_scale", "drawer_friction"] + [f"dq{i}" for i in range(7)]
N_PARAMS = len(PARAM_NAMES)
LO = [-0.06, -0.12, -0.20, 0.5, 0.0] + [-0.15] * 7
HI = [0.06, 0.12, 0.20, 4.0, 0.3] + [0.15] * 7
BASE_DRAWER_DAMPING = 2.5  # set by vinci_lab.scene._vinci_dynamics

CONTROL_HZ = 30  # decimation 2 at 60 Hz physics: halves the rendering cost of the camera student
EPISODE_S = 5.0
SUCCESS_OPENING = 0.20  # m, same success definition as the rest of the lab

CAM_RES = 128  # rendered; the student sees it 2x2-average-pooled to 64x64 (like the lab's 256 -> 64 downsample)
STUDENT_RES = 64
DEPTH_MAX = 3.0


def sample_params(n: int, device, generator: torch.Generator | None = None) -> torch.Tensor:
    lo = torch.tensor(LO, device=device)
    hi = torch.tensor(HI, device=device)
    u = torch.rand(n, N_PARAMS, generator=generator, device="cpu" if generator is not None else device).to(device)
    return lo + u * (hi - lo)


def eval_params(n: int, seed: int, device) -> torch.Tensor:
    """Fixed evaluation set: episode k always has the same cabinet pose, drawer dynamics and start pose."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return sample_params(n, device, g)


def _params(env) -> torch.Tensor:
    if not hasattr(env, "dagger_params"):
        env.dagger_params = torch.zeros(env.num_envs, N_PARAMS, device=env.device)
        env.dagger_params[:, 3] = 1.0
    return env.dagger_params


def dagger_randomize(env, env_ids: torch.Tensor | None):
    """Reset event: randomize cabinet pose, drawer dynamics and robot start. `env.dagger_forced` pins the values."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    forced = getattr(env, "dagger_forced", None)
    p = forced[env_ids].clone() if forced is not None else sample_params(len(env_ids), env.device)
    _params(env)[env_ids] = p

    cab = env.scene["cabinet"]
    root = cab.data.default_root_state[env_ids, :7].clone()
    root[:, :3] += env.scene.env_origins[env_ids]
    root[:, 0] += p[:, 0]
    root[:, 1] += p[:, 1]
    z = torch.zeros_like(p[:, 0])
    root[:, 3:7] = quat_mul(quat_from_euler_xyz(z, z, p[:, 2]), root[:, 3:7])
    cab.write_root_pose_to_sim(root, env_ids=env_ids)
    cab.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)
    jid = cab.find_joints("drawer_top_joint")[0]
    cab.write_joint_damping_to_sim((BASE_DRAWER_DAMPING * p[:, 3]).unsqueeze(-1), joint_ids=jid, env_ids=env_ids)
    cab.write_joint_friction_coefficient_to_sim(p[:, 4:5], joint_ids=jid, env_ids=env_ids)

    robot = env.scene["robot"]
    q = robot.data.default_joint_pos[env_ids].clone()
    q[:, :7] += p[:, 5:12]
    lim = robot.data.soft_joint_pos_limits[env_ids]
    q = torch.maximum(torch.minimum(q, lim[..., 1]), lim[..., 0])
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)


# ---------------------------------------------------------------- privileged observation terms (teacher only)
def _unique(q: torch.Tensor) -> torch.Tensor:
    return torch.where(q[:, :1] < 0, -q, q)


def handle_pos_b(env) -> torch.Tensor:
    return env.scene["cabinet_frame"].data.target_pos_w[:, 0] - env.scene["robot"].data.root_pos_w


def handle_quat_w(env) -> torch.Tensor:
    return _unique(env.scene["cabinet_frame"].data.target_quat_w[:, 0])


def ee_pos_b(env) -> torch.Tensor:
    return env.scene["ee_frame"].data.target_pos_source[:, 0]


def ee_quat_b(env) -> torch.Tensor:
    return _unique(env.scene["ee_frame"].data.target_quat_source[:, 0])


def drawer_dynamics(env) -> torch.Tensor:
    p = _params(env)
    return torch.stack([p[:, 3] / HI[3], p[:, 4] / HI[4]], dim=-1)


@configclass
class StudentProprioCfg(ObsGroup):
    """What the student may use besides its cameras: its own joint positions and velocities."""

    joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


def _look_at_quat_world(eye, target) -> tuple:
    """Quaternion (w,x,y,z) of a camera at `eye` looking at `target`, Isaac Lab 'world' convention (+X fwd, +Z up)."""
    f = [t - e for t, e in zip(target, eye)]
    n = math.sqrt(sum(c * c for c in f))
    f = [c / n for c in f]
    left = [-f[1], f[0], 0.0]  # up(0,0,1) x f
    n = math.sqrt(sum(c * c for c in left))
    left = [c / n for c in left]
    up = [f[1] * left[2] - f[2] * left[1], f[2] * left[0] - f[0] * left[2], f[0] * left[1] - f[1] * left[0]]
    m = [[f[0], left[0], up[0]], [f[1], left[1], up[1]], [f[2], left[2], up[2]]]
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)


# ---------------------------------------------------------------- env configs
@configclass
class VinciDaggerTeacherEnvCfg(FrankaCabinetEnvCfg):
    """Randomized open-drawer task with the privileged observation set. No cameras: fast PPO."""

    def __post_init__(self):
        super().__post_init__()
        _vinci_dynamics(self)
        self.decimation = 60 // CONTROL_HZ
        self.sim.render_interval = self.decimation
        self.episode_length_s = EPISODE_S
        self.scene.num_envs = 1024
        self.scene.env_spacing = 2.5
        # same props as the lab scene, so the student's images match what it will see at deployment
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
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.1, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(STAND_POS[0], STAND_POS[1], STAND_SIZE[2] + CUBE_SIZE / 2 + 0.002)
            ),
        )
        # randomization replaces the stock +-0.1 rad joint reset
        self.events.reset_robot_joints = None
        self.events.dagger_randomize = EventTerm(func=dagger_randomize, mode="reset")
        # privileged extras on top of the stock observation (joints, drawer joint, handle-ee vector, last action)
        pol = self.observations.policy
        pol.enable_corruption = False
        pol.handle_pos = ObsTerm(func=handle_pos_b)
        pol.handle_quat = ObsTerm(func=handle_quat_w)
        pol.ee_pos = ObsTerm(func=ee_pos_b)
        pol.ee_quat = ObsTerm(func=ee_quat_b)
        pol.drawer_dynamics = ObsTerm(func=drawer_dynamics)


@configclass
class VinciDaggerStudentEnvCfg(VinciDaggerTeacherEnvCfg):
    """Same MDP + the student's sensors: 2 RGB-D cameras at the lab scene's poses, and a proprio group."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 128
        self.scene.env_spacing = 8.0  # keep neighbouring envs out of the camera views (the lab has a single env)
        self.num_rerenders_on_reset = 1
        self.observations.student = StudentProprioCfg()
        self.scene.front_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/FrontCam",
            height=CAM_RES,
            width=CAM_RES,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=16.0, clipping_range=(0.05, 20.0)),
            offset=TiledCameraCfg.OffsetCfg(pos=FRONT_EYE, rot=_look_at_quat_world(FRONT_EYE, FRONT_TARGET), convention="world"),
        )
        self.scene.wrist_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            height=CAM_RES,
            width=CAM_RES,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.02, 5.0)),
            offset=TiledCameraCfg.OffsetCfg(pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"),
        )


@configclass
class VinciDaggerTeacherPPORunnerCfg(CabinetPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "vinci_dagger_teacher"
        self.max_iterations = 400
        self.save_interval = 25
        self.num_steps_per_env = 48  # 1.6 s of 30 Hz control per rollout
