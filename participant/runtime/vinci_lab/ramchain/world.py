"""RamLab: the `pcgr` scene (VINCI variant of EmbodiedSWE `pc_gpu_ram`) set up for ONE narrow skill.

    ram1 already in the gripper -> insert it into dimm1 of a case at a random pose, from the cameras.

What changes relative to `vinci_lab.pcgr.world.PcgrLab` (reused as is, not modified):
  - two RGB-D cameras (TiledCamera): `front` fixed in the world over the case, `wrist` on the panda hand;
  - case pose randomized for the vision to matter: +-XY m and +-YAW deg about its origin (the case is ONE kinematic
    body: board, DIMM/PCIe channels and walls move together); the card and ram0 stay in their holders on the table;
  - start state written before the clock: hand at a FIXED world pose above the nominal slot, ram1 written between the
    open fingers, gripper closed until the weld-on-closure grasp engages (the grasp is a declared controller);
  - the weld pool (upstream: 8 engages per part and process) is raised so one process can play many episodes.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from vinci_lab.pcgr.world import PcgrLab

PART, SLOT_I = "ram1", 1
PINCH_DZ = 0.006  # ram1 pinched 6 mm high on its grip band (the reference stick grasp)
START_HAND_Z = 0.45  # hand start height (world), stick bottom ~0.30: clear of the 195 mm walls
CAM_RES = 128  # rendered; the student sees 64x64 (area-downsampled, as in the cabinet chain)
FRONT_EYE, FRONT_TARGET, FRONT_FOCAL = (0.12, -0.42, 0.95), (0.43, -0.03, 0.03), 22.0
# wrist camera, hand frame (x along the held stick, y finger closing, z approach; the stick bottom is ~0.15 m down z)
WRIST_VIEWS = {
    "upstream": ((0.0465, -0.02, 0.036), (0.0465, -0.02, 0.3), 18.0),  # upstream Franka wrist view (narrower lens): looks down
}
GRASP_POOL = 400
# showcase camera: for humans only (notebook pictures, run video), never an input of the student or of a Brain.
# Re-aimed at each episode on the slot (the case moves): from above, on the side away from the arm, so neither the case walls
# nor the arm hide the slot. Chosen on t10 among 6 framings rendered during a real student insertion (start, 3 moments, end).
SHOW_RES = (1280, 720)
SHOW_FROM_SEAT, SHOW_LOOK_DZ, SHOW_FOCAL = (0.0, 0.25, 0.50), 0.12, 18.0
# after insert_ram: the empty hand climbs straight up to this world height (fingers down), above where the insertion ended.
# Probed (look_probe, look_probe2): a fixed world pose toward the base clears the front view but leaves the slot a few pixels
# wide; straight above it, the wrist camera looks down on dimm1 and tells a seated stick from an empty slot.
LOOK_Z = 0.50
START_SEED_Q = (0.0, 0.35, 0.0, -2.0, 0.0, 2.4, 0.8)  # arm joints the start-pose IK starts from (7-DOF: picks the branch)


def look_at_quat(eye, target) -> tuple:
    """(w,x,y,z) of a camera at `eye` looking at `target`, Isaac Lab 'world' convention (+X fwd, +Z up)."""
    from isaaclab.utils.math import quat_from_matrix

    f = torch.tensor([t - e for t, e in zip(target, eye)], dtype=torch.float32)
    f = f / f.norm().clamp_min(1e-9)
    left = torch.linalg.cross(torch.tensor([0.0, 0.0, 1.0]), f)
    if left.norm() < 1e-6:
        left = torch.tensor([0.0, 1.0, 0.0])
    left = left / left.norm()
    up = torch.linalg.cross(f, left)
    return tuple(quat_from_matrix(torch.stack([f, left, up], dim=1)).tolist())


def _inject_contact_check():
    """Diagnostics only (wall check): net contact force on every robot body (the fingers, always on the stick, are the witness)."""
    from isaaclab.sensors import ContactSensorCfg
    from robobench.core.registries import ROBOTS

    robot_cls = ROBOTS.get("franka")
    r_assets = robot_cls.assets

    def assets(self):
        out = r_assets(self)
        out[self.name].spawn.activate_contact_sensors = True
        out["robot_contacts"] = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*")
        return out

    robot_cls.assets = assets


def _inject_cameras(res: int, wrist_view: str = "upstream", showcase: tuple | None = None):
    """Add the two TiledCameras to the scene/robot asset dicts before the env is built (as data_engine replay does);
    `showcase` = (w, h) also adds the human-only ShowCam (aimed later by RamLab.aim_showcase)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg
    from robobench.core.registries import ROBOTS, SCENES

    scene_cls, robot_cls = SCENES.get("pc_gpu_ram"), ROBOTS.get("franka")
    scene_cls.GRASP_POOL = GRASP_POOL
    if getattr(scene_cls, "_ramchain_cams", False):
        return
    WRIST_EYE, WRIST_TARGET, WRIST_FOCAL = WRIST_VIEWS[wrist_view]
    dtypes = ["rgb", "distance_to_image_plane"]
    front = TiledCameraCfg(prim_path="{ENV_REGEX_NS}/FrontCam", width=res, height=res, data_types=dtypes, update_period=0.0,
                           offset=TiledCameraCfg.OffsetCfg(pos=FRONT_EYE, rot=look_at_quat(FRONT_EYE, FRONT_TARGET), convention="world"),
                           spawn=sim_utils.PinholeCameraCfg(focal_length=FRONT_FOCAL, clipping_range=(0.05, 5.0)))
    wrist = TiledCameraCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam", width=res, height=res, data_types=dtypes,
                           update_period=0.0,
                           offset=TiledCameraCfg.OffsetCfg(pos=WRIST_EYE, rot=look_at_quat(WRIST_EYE, WRIST_TARGET), convention="world"),
                           spawn=sim_utils.PinholeCameraCfg(focal_length=WRIST_FOCAL, clipping_range=(0.01, 5.0)))
    s_assets, r_assets = scene_cls.assets, robot_cls.assets
    extra = {"front_cam": front}
    if showcase:
        extra["show_cam"] = TiledCameraCfg(prim_path="{ENV_REGEX_NS}/ShowCam", width=showcase[0], height=showcase[1], data_types=["rgb"],
                                           update_period=0.0,
                                           offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.6, 0.9), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
                                           spawn=sim_utils.PinholeCameraCfg(focal_length=SHOW_FOCAL, clipping_range=(0.02, 10.0)))
    scene_cls.assets = lambda self: {**s_assets(self), **extra}
    robot_cls.assets = lambda self: {**r_assets(self), "wrist_cam": wrist}
    scene_cls._ramchain_cams = True


class RamLab(PcgrLab):
    def __init__(self, device: str = "cpu", cameras: bool = True, xy: float = 0.05, yaw_deg: float = 12.0, cam_res: int = CAM_RES,
                 wrist_view: str = "upstream",
                 contact_check: bool = False, showcase: tuple | None = None):
        import robobench
        from robobench.core.registries import SCENES

        robobench.discover()
        if contact_check:
            _inject_contact_check()
        if cameras:
            _inject_cameras(cam_res, wrist_view, showcase)
        else:
            SCENES.get("pc_gpu_ram").GRASP_POOL = GRASP_POOL
        super().__init__(device=device, randomize=True)
        self.cameras = cameras
        self.wrist_view = wrist_view
        self.xy, self.yaw = xy, math.radians(yaw_deg)
        iscene = self.env.iscene
        self.front = iscene["front_cam"] if cameras else None
        self.wrist = iscene["wrist_cam"] if cameras else None
        self.show_cam = iscene["show_cam"] if cameras and showcase else None
        self.contact_sensor = iscene["robot_contacts"] if contact_check else None
        self.q_default = self.F.data.default_joint_pos.clone()
        self.start_p, self.start_q = self._start_pose()
        self.params = {}
        # once per process: drive the empty hand to the start pose and remember the joints (also runs the first sim steps:
        # a case pose written before them does not stick)
        q = self.F.data.joint_pos.clone()
        q[:, :7] = torch.tensor(START_SEED_Q, device=self.dev)  # IK seed: less folded elbow, farther from the q2=0 singularity
        self._write_arm(q)
        self.hold(5, grip=0.04)
        self.goto(self.start_p, self.start_q, 0.04, tol=0.001, max_steps=600, max_u=1.0)
        self.hold(20, grip=0.04)
        self.q_start = self.F.data.joint_pos.clone()

    # ------------------------------------------------------------------ geometry
    def _stick_quat(self, yaw: float):
        from isaaclab.utils.math import quat_from_angle_axis

        return quat_from_angle_axis(torch.tensor([yaw], device=self.dev), torch.tensor([[0.0, 0.0, 1.0]], device=self.dev))

    def hand_quat_for_stick(self, sq):
        """Hand orientation that pinches a stick of orientation `sq` across its blade (fingers down) — grasp()'s formula."""
        from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul

        _obj, p0, p1 = self.sites[PART]
        bdir = quat_apply(sq, torch.tensor([[p1[i] - p0[i] for i in range(3)]], device=self.dev))
        across = torch.linalg.cross(bdir, torch.tensor([[0.0, 0.0, 1.0]], device=self.dev))
        yaw = math.atan2(across[0, 1].item(), across[0, 0].item()) - math.pi / 2
        return quat_mul(quat_from_angle_axis(torch.tensor([yaw], device=self.dev), torch.tensor([[0.0, 0.0, 1.0]], device=self.dev)),
                        torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=self.dev))

    def stick_pose_for_hand(self, hp, hq, sq):
        """Stick origin such that the hand's pinch point sits on its grip band, PINCH_DZ high (inverse of grasp())."""
        from isaaclab.utils.math import quat_apply

        _obj, p0, p1 = self.sites[PART]
        bc = torch.tensor([[(p0[i] + p1[i]) / 2 for i in range(3)]], device=self.dev)
        ez = torch.tensor([[0.0, 0.0, 1.0]], device=self.dev)
        return hp - ez * (self.s.GRASP_PINCH_OFFSET + PINCH_DZ) - quat_apply(sq, bc)

    def _start_pose(self):
        """Fixed world hand pose: ram1 upright in the nominal slot heading, its origin above the NOMINAL dimm1 seat."""
        from isaaclab.utils.math import quat_apply

        cp, cq = self.case_home
        seat = cp + quat_apply(cq, torch.tensor([self.c.ram_seat_pos[SLOT_I]], device=self.dev))
        sq = cq.clone()
        hq = self.hand_quat_for_stick(sq)
        hp = seat.clone()
        hp[0, 2] = START_HAND_Z
        off = self.stick_pose_for_hand(hp, hq, sq) - hp  # stick origin relative to the hand
        hp[0, :2] -= off[0, :2]  # put the stick origin (not the hand) over the seat
        return hp, hq

    # ------------------------------------------------------------------ episode
    def _write_arm(self, q):
        F = self.F
        q = q.clone()
        q[:, -2:] = 0.04  # fingers open
        F.write_joint_state_to_sim(q, torch.zeros_like(q))
        F.set_joint_position_target(q)
        ctrl = getattr(self.env.robot, "controller", None)
        if ctrl is not None and hasattr(ctrl, "reset"):
            ctrl.reset()

    def reset_episode(self, seed: int, case: tuple | None = None) -> None:
        """New episode: case at a random pose (or `case` = (dx, dy, dyaw_deg)), ram1 welded in the hand at the start pose."""
        from isaaclab.utils.math import quat_from_angle_axis, quat_mul

        rng = np.random.default_rng(seed)
        self.env.reset()
        s, dev = self.s, self.dev
        cp, cq = self.case_home
        if case is None:
            dxy, dyaw = rng.uniform(-self.xy, self.xy, 2), rng.uniform(-self.yaw, self.yaw)
        else:
            dxy, dyaw = np.array(case[:2], dtype=float), math.radians(case[2])
        st = torch.zeros(1, 7, device=dev)
        st[:, 0:3] = cp + torch.tensor([[dxy[0], dxy[1], 0.0]], device=dev, dtype=torch.float32)
        st[:, 3:7] = quat_mul(quat_from_angle_axis(torch.tensor([dyaw], device=dev, dtype=torch.float32),
                                                   torch.tensor([[0.0, 0.0, 1.0]], device=dev)), cq)
        s.case.write_root_pose_to_sim(st)
        fr = float(rng.uniform(0.22, 0.38))
        s._set_friction(self.part(PART), fr)
        self._write_arm(self.q_start)
        hp, hq = self.hand()
        self.act(hp, hq, 0.04, max_u=0.3)
        sq = self._stick_quat(0.0)
        sq = quat_mul(sq, cq)
        obj = self.part(PART)
        closed = False
        for _ in range(60):  # hold the stick in place between the fingers while they close, until the weld engages
            pose = torch.zeros(1, 13, device=dev)
            pose[:, 0:3] = self.stick_pose_for_hand(hp, hq, sq)
            pose[:, 3:7] = sq
            obj.write_root_state_to_sim(pose)
            self.act(hp, hq, 0.0, max_u=0.3)
            if self.held() == PART:
                closed = True
                break
        self.hold(10, grip=0.0)
        self.steps = 0
        self.ever_held = {p: False for p in self.ever_held}
        self.initial = {"seed": seed, "case_dxy_mm": [round(float(v) * 1000, 2) for v in dxy], "case_dyaw_deg": round(math.degrees(dyaw), 2),
                        "friction_ram1": round(fr, 3), "grasped_at_start": closed}
        self.params = {"dx": float(dxy[0]), "dy": float(dxy[1]), "dyaw": float(dyaw)}

    # ------------------------------------------------------------------ student-side observations
    def render(self):
        self.env.sim.render()

    def images(self):
        """(front, wrist) raw camera outputs: dicts with rgb (1,H,W,3) uint8 and depth (1,H,W,1) float."""
        fo, wo = self.front.data.output, self.wrist.data.output
        return (fo["rgb"], fo["distance_to_image_plane"]), (wo["rgb"], wo["distance_to_image_plane"])

    def aim_showcase(self) -> None:
        """Point the human-only ShowCam at this episode's slot (call after reset_episode)."""
        if self.show_cam is None:
            return
        seat = self.seat_w(PART)
        eye = seat + torch.tensor([SHOW_FROM_SEAT], device=seat.device, dtype=seat.dtype)
        target = seat + torch.tensor([[0.0, 0.0, SHOW_LOOK_DZ]], device=seat.device, dtype=seat.dtype)
        self.show_cam.set_world_poses_from_view(eye, target)

    def showcase(self):
        """Last ShowCam frame, (H, W, 3) uint8 numpy (humans only)."""
        return self.show_cam.data.output["rgb"][0, ..., :3].cpu().numpy()

    def proprio(self):
        """18-D: 9 joint positions and 9 joint velocities, relative to the default pose (what the robot knows about itself)."""
        F = self.F
        return torch.cat([F.data.joint_pos - self.q_default, F.data.joint_vel], dim=-1)

    def case_contacts(self, thresh: float = 0.5) -> set:
        """Diagnostics: robot bodies with a net contact force > `thresh` N this step. Only the case, the table or the parts
        can touch the arm: any body other than the fingers (which pinch the stick: the positive control) is a collision."""
        f = self.contact_sensor.data.net_forces_w[0].norm(dim=-1)
        return {self.contact_sensor.body_names[i] for i in (f > thresh).nonzero().flatten().tolist()}

    def ram_seated(self) -> bool:
        return self.seated(PART)

    # ------------------------------------------------------------------ privileged state (teacher / evaluator only)
    def priv(self) -> dict:
        obj = self.part(PART)
        hp, hq = self.hand()
        return {"hand_p": hp, "hand_q": hq, "part_p": obj.data.root_pos_w.clone(), "part_q": obj.data.root_quat_w.clone(),
                "seat_p": self.seat_w(PART), "case_q": self.case_pose()[1]}

    # ------------------------------------------------------------------ the student's action interface
    def step_action(self, u6: torch.Tensor, hold: int = 3) -> None:
        """Apply one 6-D world-frame EE delta (diff_ik units, clamped to [-1, 1]) for `hold` control steps, gripper closed."""
        u = torch.zeros(1, self.env.robot.action_dim, device=self.dev)
        u[:, 0:6] = u6.reshape(1, 6).to(self.dev).clamp(-1.0, 1.0)
        for _ in range(hold):
            self.env.step(u)
            self.steps += 1
            h = self.held()
            if h:
                self.ever_held[h] = True

    def go_look_pose(self, z: float = LOOK_Z) -> None:
        """Declared controller, proprio only: the (empty) hand climbs vertically to world height `z`, fingers down, and holds."""
        hp, _hq = self.hand()
        up = hp.clone()
        up[0, 2] = max(float(hp[0, 2]), z)
        self.goto(up, self.start_q, 0.04, tol=0.01, max_steps=90)
        self.hold(10, grip=0.04)

    def release_and_retreat(self, retreat: float = 0.10) -> None:
        """Declared controller, proprio only: open the gripper, back the hand off vertically from where it is, settle."""
        hp, hq = self.hand()
        self.act(hp, hq, 0.04, max_u=0.3, n=int(10 * self.SM))
        self.goto(hp + torch.tensor([[0.0, 0.0, retreat]], device=self.dev), hq, 0.04, tol=0.01, max_steps=25)
        self.hold(15, grip=0.04)
