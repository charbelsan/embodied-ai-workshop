"""Learned skills that EXECUTE without any simulator truth. They report what they did, never whether the task succeeded.

    from vinci_lab.student_skills import open_drawer_student
    r = open_drawer_student(lab)   # lab: vinci_lab.api.LabEnv with front + wrist RGB-D cameras

Contract of these skills (it differs from the expert skills of vinci_lab.skills):
  - inputs of the policy: front + wrist RGB-D (64 px) + the robot's joint positions/velocities, nothing else;
  - fixed horizon, the one the policy was trained with (no stop on the drawer's true opening);
  - the retreat is relative to the end effector (no handle pose);
  - `ok` means "the skill ran to its end without a technical error". It does NOT say that the drawer is open:
    checking the result is the caller's job (look at the images of observe()). The grader alone reads the drawer.
  - the checkpoint comes from vinci_lab.checkpoints (team selection or reference) and its sha256 is returned.
"""
from __future__ import annotations

import time

import torch

from vinci_lab import checkpoints
from vinci_lab.api import result
from vinci_lab.dagger.student import Student, to_student_image

HORIZON = 149          # policy steps at 30 Hz = the 5 s training episodes of the DAgger pipeline
RETREAT_M = 0.10       # after releasing: back off along the hand's approach axis, then up
OBSERVATIONS = "front+wrist RGB-D 64px + joint pos/vel"


def _load_student(device):
    def build(path):
        m = Student().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        return m.eval()
    return checkpoints.load("open_drawer_student", build)


def _rgbd(cam):
    out = cam.data.output
    depth = out.get("distance_to_image_plane")
    return None if depth is None else to_student_image(out["rgb"][..., :3], depth)


def open_drawer_student(lab, horizon: int = HORIZON, mask_images: bool = False) -> dict:
    """Run the DAgger vision student for `horizon` steps from the home pose, release, back off. Execution report only.
    mask_images=True replaces both images by zeros (ablation: what the policy does without vision)."""
    t0 = time.time()
    if lab.front_cam is None or lab.wrist_cam is None:
        return result(False, "technical error: the simulator runs without cameras", t0)
    try:
        model, ck = _load_student(lab.device)
    except Exception as e:  # noqa: BLE001  missing file / sha256 mismatch: say which
        return result(False, f"technical error: student model not loaded: {e}", t0)
    lab.gripper(True, steps=10)
    lab.move_joints()  # the student was trained from (randomized) starts around the home pose
    robot = lab.robot
    with torch.no_grad():
        for n in range(horizon):
            f, w = _rgbd(lab.front_cam), _rgbd(lab.wrist_cam)
            if f is None or w is None:
                return result(False, "technical error: the cameras do not output depth", t0)
            if mask_images:
                f, w = torch.zeros_like(f), torch.zeros_like(w)
            prop = torch.cat([robot.data.joint_pos - robot.data.default_joint_pos,
                              robot.data.joint_vel - robot.data.default_joint_vel], dim=-1)[:, :18]
            a = model(f, w, prop)
            lab.step_action(a)
            lab.step_action(a)  # 30 Hz policy on the 60 Hz lab
    lab.gripper(True)
    from isaaclab.utils.math import quat_apply

    p, q = lab.ee_pose()
    approach = quat_apply(q.unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]], device=q.device))[0]
    back = p - RETREAT_M * approach
    back[2] += 0.05
    lab.move_to(back.tolist(), q.tolist(), tol=0.02)
    return result(True, f"student executed {horizon} steps, released and backed off; check the result with observe()",
                  t0, policy_steps=horizon, checkpoint_sha256=ck["sha256"], checkpoint_source=ck["source"],
                  checkpoint_path=ck["path"], checkpoint_reloaded=ck["reloaded"], observations=OBSERVATIONS,
                  images_masked=mask_images, provider="dagger_student")
