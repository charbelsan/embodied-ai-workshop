"""The deployable student: 2 RGB-D images (64x64) + joint pos/vel -> the same 8-D action as the teacher.

Pure PyTorch (no Isaac import), so the notebook and the skill can load it anywhere.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PROP_DIM = 18  # 9 joint positions + 9 joint velocities (relative to default), what the robot knows about itself
ACT_DIM = 8


class Encoder(nn.Module):
    def __init__(self, in_ch: int = 4, out: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 8 * 8, out), nn.LayerNorm(out), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_front = Encoder()
        self.enc_wrist = Encoder()
        self.prop = nn.Sequential(nn.Linear(PROP_DIM, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(256 + 256 + 64, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, ACT_DIM))
        for name, dim in (("act_mean", ACT_DIM), ("act_std", ACT_DIM), ("prop_mean", PROP_DIM), ("prop_std", PROP_DIM)):
            self.register_buffer(name, torch.zeros(dim) if "mean" in name else torch.ones(dim))

    def set_stats(self, act: torch.Tensor, prop: torch.Tensor):
        self.act_mean.copy_(act.mean(0))
        self.act_std.copy_(act.std(0).clamp_min(1e-3))
        self.prop_mean.copy_(prop.mean(0))
        self.prop_std.copy_(prop.std(0).clamp_min(1e-3))

    def forward_norm(self, front_u8, wrist_u8, prop):
        f = self.enc_front(front_u8.float() / 255.0 - 0.5)
        w = self.enc_wrist(wrist_u8.float() / 255.0 - 0.5)
        p = self.prop((prop - self.prop_mean) / self.prop_std)
        return self.head(torch.cat([f, w, p], dim=-1))

    def forward(self, front_u8, wrist_u8, prop):
        return self.forward_norm(front_u8, wrist_u8, prop) * self.act_std + self.act_mean


def to_student_image(rgb: torch.Tensor, depth: torch.Tensor, depth_max: float = 3.0, out: int = 64) -> torch.Tensor:
    """(N,H,W,3) uint8 + (N,H,W,1) float [m] -> (N,4,out,out) uint8, area-downsampled (works for 128 or 256 input)."""
    d = torch.nan_to_num(depth.float(), nan=depth_max, posinf=depth_max, neginf=0.0).clamp(0, depth_max) / depth_max * 255.0
    x = torch.cat([rgb[..., :3].float(), d], dim=-1).permute(0, 3, 1, 2)
    if x.shape[-1] != out:
        x = F.adaptive_avg_pool2d(x, out)
    return x.round().clamp(0, 255).to(torch.uint8)


def random_shift(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ-style +-pad pixel random translation (same for all channels of a sample)."""
    n, c, h, w = x.shape
    x = F.pad(x.float(), (pad, pad, pad, pad), mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    ar = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device)[:h]
    ar = ar.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base = torch.cat([ar, ar.transpose(1, 0)], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=torch.float32) * 2.0 / (h + 2 * pad)
    return F.grid_sample(x, base + shift, padding_mode="zeros", align_corners=False)


class Dataset:
    """Aggregated (image, proprio, TEACHER action) samples, kept in CPU RAM; `weight` drives failure mining."""

    KEYS = ("front", "wrist", "prop", "act", "weight", "episode", "round", "failed")

    def __init__(self):
        self.parts: dict[str, list[torch.Tensor]] = {k: [] for k in self.KEYS}

    def add(self, front, wrist, prop, act, episode, rnd: int, failed, weight):
        n = front.shape[0]
        for k, v in (("front", front), ("wrist", wrist), ("prop", prop.float()), ("act", act.float()), ("episode", episode.long()),
                     ("round", torch.full((n,), rnd, dtype=torch.long)), ("failed", failed.bool()), ("weight", weight.float())):
            self.parts[k].append(v.detach().cpu())

    def extend(self, other: "Dataset"):
        for k in self.KEYS:
            self.parts[k].extend(other.parts[k])

    def __len__(self):
        return sum(t.shape[0] for t in self.parts["act"])

    def cat(self, key):
        return torch.cat(self.parts[key], dim=0)

    def save(self, path):
        torch.save({k: self.cat(k) for k in self.KEYS if self.parts[k]}, path)

    @classmethod
    def load(cls, path):
        d = cls()
        for k, v in torch.load(path).items():
            d.parts[k] = [v]
        return d


def train_student(model: Student, data: Dataset, steps: int, lr: float = 3e-4, bs: int = 256, use_weights: bool = False,
                  log=print, log_every: int = 1000) -> list[float]:
    dev = next(model.parameters()).device
    front, wrist, prop, act = data.cat("front"), data.cat("wrist"), data.cat("prop"), data.cat("act")
    w = data.cat("weight") if use_weights else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.1)
    model.train()
    losses = []
    n = act.shape[0]
    for i in range(steps):
        idx = torch.multinomial(w, bs, replacement=True) if w is not None else torch.randint(0, n, (bs,))
        f = random_shift(front[idx].to(dev, non_blocking=True)).to(torch.uint8)
        wr = random_shift(wrist[idx].to(dev, non_blocking=True)).to(torch.uint8)
        pred = model.forward_norm(f, wr, prop[idx].to(dev))
        target = (act[idx].to(dev) - model.act_mean) / model.act_std
        loss = F.smooth_l1_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item()) if i % 50 == 0 else None
        if log_every and (i + 1) % log_every == 0:
            log(f"  step {i + 1}/{steps} loss {sum(losses[-20:]) / len(losses[-20:]):.4f}")
    model.eval()
    return losses
