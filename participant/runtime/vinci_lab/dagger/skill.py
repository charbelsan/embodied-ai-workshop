"""`open_drawer_visual()`: the DAgger student as a Brain-callable skill (common contract {ok, reason, duration_s, evidence}).

Unlike `open_drawer(provider="rl")`, it does NOT read the simulator's truth: only the lab's front + wrist cameras
(RGB-D, downsampled to 64 px) and the robot's own joint positions/velocities, exactly what it was trained on.

    from vinci_lab.dagger.skill import open_drawer_visual
    r = open_drawer_visual(lab)      # lab: vinci_lab.api.LabEnv (its cameras must also output depth, see below)
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from vinci_lab.api import result
from vinci_lab.dagger.student import Student, to_student_image

POLICY = Path(__file__).resolve().parents[2] / "policies" / "open_drawer_visual.pt"
_cache: dict = {}


def _load(device):
    if "m" not in _cache:
        m = Student().to(device)
        m.load_state_dict(torch.load(POLICY, map_location=device))
        _cache["m"] = m.eval()
    return _cache["m"]


def _rgbd(cam):
    out = cam.data.output
    depth = out.get("distance_to_image_plane")
    if depth is None:
        return None
    return to_student_image(out["rgb"][..., :3], depth)


def open_drawer_visual(lab, max_policy_steps: int = 149, target_open: float = 0.30) -> dict:
    """Run the vision student at 30 Hz (each action held 2 lab steps at 60 Hz), from the home pose."""
    t0 = time.time()
    if not POLICY.exists():
        return result(False, f"no DAgger student exported yet ({POLICY.name})", t0)
    if lab.front_cam is None or lab.wrist_cam is None:
        return result(False, "lab started without cameras: the visual skill needs front + wrist cameras", t0)
    model = _load(lab.device)
    lab.gripper(True, steps=10)
    lab.move_joints()  # the student was trained from (randomized) starts around the home pose
    robot = lab.robot
    best, n = 0.0, 0
    with torch.no_grad():
        for n in range(max_policy_steps):
            f, w = _rgbd(lab.front_cam), _rgbd(lab.wrist_cam)
            if f is None or w is None:
                return result(False, "lab cameras do not output depth: add 'distance_to_image_plane' to front_cam/wrist_cam "
                                     "data_types in vinci_lab/scene.py", t0)
            prop = torch.cat([robot.data.joint_pos - robot.data.default_joint_pos,
                              robot.data.joint_vel - robot.data.default_joint_vel], dim=-1)[:, :18]
            a = model(f, w, prop)
            lab.step_action(a)
            lab.step_action(a)
            best = max(best, lab.drawer_opening())
            if lab.drawer_opening() >= target_open:
                break
    lab.gripper(True)
    try:
        from vinci_lab.skills import HANDLE_QUAT

        h, _ = lab.handle_pose()
        lab.move_to([h[0] - 0.12, h[1], h[2] + 0.05], HANDLE_QUAT, tol=0.02)
    except Exception:  # noqa: BLE001  retreat is cosmetic; the drawer state is what matters
        pass
    opening = lab.drawer_opening()
    ok = opening > 0.2
    return result(ok, f"drawer open {opening:.2f} m (vision student)" if ok else f"vision student stalled at {best:.2f} m",
                  t0, opening_m=round(opening, 3), best_opening_m=round(best, 3), policy_steps=n + 1,
                  provider="dagger_student", observations="front+wrist RGB-D 64px + joint pos/vel (no simulator state)")
