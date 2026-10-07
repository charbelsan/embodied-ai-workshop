"""Participant-facing robot API: observe(), move_to(), gripper(), reset(), success().

Every call that moves the robot returns the common skill contract:
    {"ok": bool, "reason": str, "duration_s": float, "evidence": {...}}
All actions go through the env's single 8-D action space (7 arm joint offsets + binary gripper).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

TCP_OFFSET = 0.1034  # panda_hand -> fingertip centre (same as Isaac Lab ee_tcp frame)
DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)  # (w, x, y, z): gripper pointing straight down


def down_quat(yaw: float) -> tuple:
    """Gripper pointing straight down, rotated by `yaw` (rad) about world z: qz(yaw) * qx(pi)."""
    import math
    return (0.0, math.cos(yaw / 2), math.sin(yaw / 2), 0.0)
SIM_HZ = 60.0
MAX_DQ = 0.25  # rad lead of the PD target over the measured joints (0.05 capped speed at ~0.25 rad/s)


def result(ok: bool, reason: str, t0: float, **evidence) -> dict:
    return {"ok": bool(ok), "reason": reason, "duration_s": round(time.time() - t0, 3), "evidence": evidence}


class LabEnv:
    """Thin, readable wrapper around the Isaac Lab ManagerBasedRLEnv for the VINCI cabinet scene."""

    def __init__(self, cfg=None, device: str = "cuda:0", record_every: int = 2, cameras: bool = True):
        from isaaclab.envs import ManagerBasedRLEnv

        from .scene import FRONT_EYE, FRONT_TARGET, VinciCabinetEnvCfg

        cfg = cfg or VinciCabinetEnvCfg()
        cfg.sim.device = device
        if not cameras:
            cfg.scene.front_cam = None
            cfg.scene.wrist_cam = None
        self.cfg = cfg
        self.env = ManagerBasedRLEnv(cfg=cfg)
        self.device = device
        s = self.env.scene
        self.robot, self.cabinet = s["robot"], s["cabinet"]
        self.cube = s.rigid_objects.get("cube")  # None in multi-object scenes (challenge)
        self.ee_frame, self.handle_frame = s["ee_frame"], s["cabinet_frame"]
        self.front_cam = s["front_cam"] if cameras else None
        self.wrist_cam = s["wrist_cam"] if cameras else None
        self.arm_ids, _ = self.robot.find_joints("panda_joint.*")
        self.hand_idx = self.robot.find_bodies("panda_hand")[0][0]
        self.jacobi_idx = self.hand_idx - 1  # fixed base: jacobians skip the root body
        self.drawer_jid = self.cabinet.find_joints("drawer_top_joint")[0][0]
        self.drawer_body = self.cabinet.find_bodies("drawer_top")[0][0]
        self._front_view = (FRONT_EYE, FRONT_TARGET)
        self.record_every = record_every
        self.frames: list[np.ndarray] = []
        self.recording = False
        self.demo_recorder = None  # callable(lab, action) set by scripts/gen_demos.py
        self.gripper_cmd = 1.0
        self.stopped = False
        self.steps = 0
        self.last_obs = None
        self.reset()

    # ------------------------------------------------------------------ low level
    def _arm_target_now(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self.arm_ids].clone()

    def step_joint_targets(self, arm_joint_targets: torch.Tensor, gripper: float | None = None):
        """One 60 Hz control step with absolute arm joint targets (1,7) and gripper (+1 open / -1 close)."""
        if gripper is not None:
            self.gripper_cmd = float(gripper)
        default = self.robot.data.default_joint_pos[:, self.arm_ids]
        action = torch.cat([arm_joint_targets - default,
                            torch.full((1, 1), self.gripper_cmd, device=self.device)], dim=-1)
        return self.step_action(action)

    def step_action(self, action: torch.Tensor):
        """Raw env action (1,8): used by learned policies (RL / VLA) and by the IK helpers alike."""
        if self.demo_recorder is not None and self.steps % 2 == 0:  # 30 fps (obs_t, a_t) pairs for imitation
            self.demo_recorder(self, action)
        self.last_obs, _, _, _, _ = self.env.step(action)
        self.steps += 1
        if self.recording and self.front_cam is not None and self.steps % self.record_every == 0:
            self.frames.append(self.front_cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
        return self.last_obs

    def hold(self, n: int = 30, gripper: float | None = None):
        target = self._arm_target_now()
        for _ in range(n):
            self.step_joint_targets(target, gripper)

    # ------------------------------------------------------------------ perception
    def ee_pose(self):
        return self.ee_frame.data.target_pos_w[0, 0].clone(), self.ee_frame.data.target_quat_w[0, 0].clone()

    def handle_pose(self):
        return self.handle_frame.data.target_pos_w[0, 0].clone(), self.handle_frame.data.target_quat_w[0, 0].clone()

    def drawer_opening(self) -> float:
        return float(self.cabinet.data.joint_pos[0, self.drawer_jid])

    def drawer_pose(self):
        return (self.cabinet.data.body_pos_w[0, self.drawer_body].clone(),
                self.cabinet.data.body_quat_w[0, self.drawer_body].clone())

    def in_drawer_frame(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import subtract_frame_transforms
        dp, dq = self.drawer_pose()
        rel, _ = subtract_frame_transforms(dp[None], dq[None], pos_w[None])
        return rel[0]

    def cube_in_drawer_frame(self) -> torch.Tensor:
        from isaaclab.utils.math import subtract_frame_transforms
        dp, dq = self.drawer_pose()
        cp = self.cube.data.root_pos_w[0]
        rel, _ = subtract_frame_transforms(dp[None], dq[None], cp[None])
        return rel[0]

    def observe(self) -> dict:
        """Everything a participant (or a Brain) may look at. Images are uint8 HxWx3."""
        ee_p, ee_q = self.ee_pose()
        h_p, _ = self.handle_pose()
        jp = self.robot.data.joint_pos[0]
        obs = {
            "joint_pos": jp.cpu().numpy().round(4).tolist(),
            "gripper_width": float(jp[-2:].sum()),
            "ee_pos": ee_p.cpu().numpy().round(4).tolist(),
            "ee_quat": ee_q.cpu().numpy().round(4).tolist(),
            "cube_pos": self.cube.data.root_pos_w[0].cpu().numpy().round(4).tolist() if self.cube is not None else None,
            "handle_pos": h_p.cpu().numpy().round(4).tolist(),
            "drawer_opening_m": round(self.drawer_opening(), 4),
            "cube_in_drawer_frame": self.cube_in_drawer_frame().cpu().numpy().round(4).tolist() if self.cube is not None else None,
            "sim_time_s": round(self.steps / SIM_HZ, 2),
        }
        if self.front_cam is not None:
            obs["rgb_front"] = self.front_cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            obs["rgb_wrist"] = self.wrist_cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        return obs

    # ------------------------------------------------------------------ actions
    def _ik_dq(self, hand_goal_p, hand_goal_q, lam: float = 0.05, k_null: float = 0.1) -> torch.Tensor:
        """Damped least-squares IK step on the hand frame + null-space pull toward the home configuration
        (keeps the elbow up and avoids the twisted postures plain DLS drifts into)."""
        from isaaclab.utils.math import compute_pose_error
        hp = self.robot.data.body_pos_w[:, self.hand_idx]
        hq = self.robot.data.body_quat_w[:, self.hand_idx]
        pos_err, rot_err = compute_pose_error(hp, hq, hand_goal_p, hand_goal_q, rot_error_type="axis_angle")
        e = torch.cat([pos_err, rot_err], dim=-1)[0]  # world == base frame (robot root at identity)
        J = self.robot.root_physx_view.get_jacobians()[0, self.jacobi_idx][:, self.arm_ids]  # (6,7)
        JJt = J @ J.T + (lam ** 2) * torch.eye(6, device=self.device)
        J_pinv = J.T @ torch.linalg.inv(JJt)
        q = self.robot.data.joint_pos[0, self.arm_ids]
        q_rest = self.robot.data.default_joint_pos[0, self.arm_ids]
        null = (torch.eye(7, device=self.device) - J_pinv @ J) @ (k_null * (q_rest - q))
        return (J_pinv @ e + null)[None]

    def move_to(self, pos, quat=DOWN_QUAT, speed: float = 0.25, tol: float = 0.006, settle: int = 20,
                max_extra_steps: int = 150) -> dict:
        """Move the fingertip centre (TCP) to `pos` (world, m) with orientation `quat` (w,x,y,z), straight line."""
        from isaaclab.utils.math import quat_apply, quat_slerp

        t0 = time.time()
        if self.stopped:
            return result(False, "stopped", t0)
        goal_p = torch.tensor(pos, dtype=torch.float32, device=self.device)
        goal_q = torch.tensor(quat, dtype=torch.float32, device=self.device)
        start_p, start_q = self.ee_pose()
        if torch.dot(start_q, goal_q) < 0:
            goal_q = -goal_q  # same rotation, shortest slerp
        dist = float(torch.linalg.norm(goal_p - start_p))
        n = max(int(dist / (speed / SIM_HZ)), 10)
        offset = torch.tensor([[0.0, 0.0, TCP_OFFSET]], device=self.device)
        err = float("inf")
        for i in range(n + max_extra_steps):
            a = min(1.0, (i + 1) / n)
            tcp_p = start_p + a * (goal_p - start_p)
            tcp_q = quat_slerp(start_q, goal_q, a)
            hand_p = tcp_p[None] - quat_apply(tcp_q[None], offset)
            dq = self._ik_dq(hand_p, tcp_q[None])
            dq = dq * torch.clamp(MAX_DQ / (dq.abs().max() + 1e-9), max=1.0)  # bounded step, direction preserved
            self.step_joint_targets(self.robot.data.joint_pos[:, self.arm_ids] + dq)
            err = float(torch.linalg.norm(self.ee_pose()[0] - goal_p))
            if a >= 1.0 and err < tol:
                break
        self.hold(settle)
        err = float(torch.linalg.norm(self.ee_pose()[0] - goal_p))
        ok = err < max(tol * 2, 0.012)
        return result(ok, "reached" if ok else f"residual error {err*1000:.1f} mm (unreachable or blocked)", t0,
                      target=list(map(float, pos)), final_error_m=round(err, 4), steps=self.steps)

    def move_joints(self, q_target=None, speed: float = 0.8, settle: int = 20) -> dict:
        """Joint-space move (rad/s on the largest joint). Default target: the task's home configuration."""
        t0 = time.time()
        q0 = self._arm_target_now()
        q1 = self.robot.data.default_joint_pos[:, self.arm_ids].clone() if q_target is None else \
            torch.tensor([q_target], dtype=torch.float32, device=self.device)
        n = max(int(float((q1 - q0).abs().max()) / (speed / SIM_HZ)), 5)
        for i in range(n):
            self.step_joint_targets(q0 + (q1 - q0) * (i + 1) / n)
        self.hold(settle)
        err = float((self.robot.data.joint_pos[:, self.arm_ids] - q1).abs().max())
        return result(err < 0.03, "reached" if err < 0.03 else f"joint error {err:.3f} rad", t0, max_joint_err=round(err, 4))

    def gripper(self, open: bool, steps: int = 40) -> dict:
        t0 = time.time()
        if self.stopped:
            return result(False, "stopped", t0)
        self.hold(steps, gripper=1.0 if open else -1.0)
        w = float(self.robot.data.joint_pos[0, -2:].sum())
        if open:
            ok = w > 0.07
            reason = "opened" if ok else f"could not open (width {w*1000:.0f} mm)"
        else:
            ok = w > 0.004  # something between the fingers
            reason = "closed on object" if ok else "closed on nothing (empty grasp)"
        return result(ok, reason, t0, width_m=round(w, 4))

    def reset(self) -> dict:
        t0 = time.time()
        self.last_obs, _ = self.env.reset()
        self.stopped = False
        self.gripper_cmd = 1.0
        if self.front_cam is not None:
            eye, tgt = self._front_view
            self.front_cam.set_world_poses_from_view(torch.tensor([eye], device=self.device),
                                                     torch.tensor([tgt], device=self.device))
        self.hold(30, gripper=1.0)
        return result(True, "scene reset", t0, cube_pos=self.cube.data.root_pos_w[0].tolist() if self.cube is not None else None)

    def success(self) -> dict:
        """Task success: cube inside the top drawer tray AND drawer closed."""
        rel = self.cube_in_drawer_frame()
        inside = bool(self.in_tray(rel))
        closed = self.drawer_opening() < 0.02
        return {"success": inside and closed, "cube_in_drawer": inside, "drawer_closed": closed,
                "drawer_opening_m": round(self.drawer_opening(), 4),
                "cube_in_drawer_frame": rel.cpu().numpy().round(3).tolist()}

    # tray box in the drawer_top body frame, calibrated by scripts/probe_scene.py
    # drawer_top local bbox: x [-0.258, 0.273] (front/handle at +x), y [-0.379, 0.379], z [-0.062, 0.062]
    TRAY_CENTER = (0.0075, 0.0, 0.0)
    TRAY_HALF = (0.255, 0.365, 0.07)

    def in_tray(self, rel) -> bool:
        c = torch.tensor(self.TRAY_CENTER, device=self.device)
        h = torch.tensor(self.TRAY_HALF, device=self.device)
        return bool(torch.all(torch.abs(rel - c) <= h))

    # ------------------------------------------------------------------ recording
    def start_recording(self):
        self.frames, self.recording = [], True

    def save_video(self, path: str, fps: int = 30) -> str | None:
        self.recording = False
        if not self.frames:
            return None
        import imageio.v2 as imageio
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, self.frames, fps=fps, macro_block_size=1)
        return path
