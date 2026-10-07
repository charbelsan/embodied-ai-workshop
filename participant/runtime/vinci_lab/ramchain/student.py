"""The RAM-insertion student: `vinci_lab.dagger.student.Student` reused as is (2 RGB-D 64 px encoders + 18-D proprio),
only the action head is 6-D (world-frame EE delta, diff_ik units) instead of 8-D. The frozen DAgger module is not modified.
"""
from __future__ import annotations

import hashlib

import torch
import torch.nn as nn

from vinci_lab.dagger.student import Student, to_student_image

ACT_DIM = 6
FRONT_DEPTH_MAX, WRIST_DEPTH_MAX = 1.5, 0.6  # m: depth normalisation per camera (the wrist sees 0.1-0.45 m)


class RamStudent(Student):
    def __init__(self, act_dim: int = ACT_DIM):
        super().__init__()
        self.head[-1] = nn.Linear(self.head[-1].in_features, act_dim)
        self.register_buffer("act_mean", torch.zeros(act_dim))
        self.register_buffer("act_std", torch.ones(act_dim))


def student_images(front, wrist):
    """(rgb, depth) pairs of the two cameras -> the student's (N,4,64,64) uint8 images."""
    return (to_student_image(front[0], front[1], depth_max=FRONT_DEPTH_MAX),
            to_student_image(wrist[0], wrist[1], depth_max=WRIST_DEPTH_MAX))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(model: RamStudent, path: str, meta: dict) -> str:
    """Immutable checkpoint (state dict + metadata); returns its sha256."""
    torch.save({"state_dict": model.state_dict(), "act_dim": ACT_DIM, "meta": meta}, path)
    return sha256_file(path)


def load_checkpoint(path: str, device: str = "cpu") -> tuple[RamStudent, str, dict]:
    ck = torch.load(path, map_location=device)
    m = RamStudent(ck.get("act_dim", ACT_DIM)).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, sha256_file(path), ck.get("meta", {})
