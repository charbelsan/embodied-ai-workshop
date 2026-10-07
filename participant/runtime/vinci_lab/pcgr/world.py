"""PcgrLab: the VINCI variant of EmbodiedSWE `pc_gpu_ram` (single Franka, graphics card + 2 RAM sticks into a PC).

Upstream physics and assets are kept as they are (PCIe/DIMM channels, end stops, rear I/O cutout, weld-on-closure grasp,
the scene's own `gpu_seated` / `ram_seated` gates). What the VINCI variant changes:
  - layout: ram0 is staged 6 cm further west, off the card's foam holder (upstream spot: the open Franka finger lands on
    that holder and the stick can never be pinched);
  - per-episode variability (upstream Franka env is fully deterministic): the case pose (+-4 mm, +-1.5 deg), the parts
    inside their holders' clearance (+-0.8 mm), part friction U(0.22, 0.38);
  - the arm runs in diff_ik (joint PD): OSC leaves 8-23 mm steady errors near the base.
"""
from __future__ import annotations

import math

import numpy as np
import torch

ENV_NAME = "assembly.pc_gpu_ram.franka.diff_ik"
PARTS = ("card", "ram0", "ram1")
SLOT_OF = {"card": "pcie", "ram0": "dimm0", "ram1": "dimm1"}
RAM0_XY = (-0.35, -0.32)  # table-relative (upstream: -0.26, -0.32)
SAFE_Z = 0.45  # hand height that clears the 195 mm case walls


class PcgrLab:
    def __init__(self, device: str = "cpu", randomize: bool = True):
        import robobench
        from robobench.core import ENVS

        robobench.discover()
        cfg = ENVS.get(ENV_NAME)()
        cfg.scene_cfg.ram_init_xy = (RAM0_XY, cfg.scene_cfg.ram_init_xy[1])
        cfg.scene_cfg.reset_pos_jitter = 0.0
        self.env = cfg.build(num_envs=1, device=device, seed=0)
        self.env.reset()
        self.s, self.c, self.dev = self.env.scene, self.env.scene.cfg, self.env.device
        self.F = self.env.robot.articulation
        self.i_hand = self.F.find_bodies("panda_hand")[0][0]
        self.ctrl_hz = 1.0 / (self.env.robot.control_period * self.env.dt)
        self.SM = max(1.0, self.ctrl_hz / 15.0)
        self.randomize = randomize
        self.case_home = (self.s.case.data.root_pos_w.clone(), self.s.case.data.root_quat_w.clone())
        self.site_idx = {name: i for i, (name, *_r) in enumerate(self.s.grasp_sites())}
        self.sites = {name: (obj, p0, p1) for name, obj, p0, p1, _w in self.s.grasp_sites()}
        self.steps = 0
        self.ever_held = {p: False for p in PARTS}
        self.initial = {}

    # ------------------------------------------------------------------ episode
    def reset_episode(self, seed: int) -> None:
        from isaaclab.utils.math import quat_from_angle_axis, quat_mul

        rng = np.random.default_rng(seed)
        self.env.reset()
        s, dev = self.s, self.dev
        cp, cq = self.case_home
        dxy, dyaw = np.zeros(2), 0.0
        if self.randomize:
            dxy, dyaw = rng.uniform(-0.004, 0.004, 2), math.radians(rng.uniform(-1.5, 1.5))
        st = torch.zeros(1, 7, device=dev)
        st[:, 0:3] = cp + torch.tensor([[dxy[0], dxy[1], 0.0]], device=dev, dtype=torch.float32)
        st[:, 3:7] = quat_mul(quat_from_angle_axis(torch.tensor([dyaw], device=dev, dtype=torch.float32),
                                                   torch.tensor([[0.0, 0.0, 1.0]], device=dev)), cq)
        s.case.write_root_pose_to_sim(st)  # kinematic body: pose only
        fr = {}
        for name in PARTS:
            obj = self.part(name)
            if self.randomize:
                p = obj.data.root_state_w.clone()
                p[:, 0:2] += torch.tensor(rng.uniform(-0.0008, 0.0008, 2), device=dev, dtype=torch.float32)
                obj.write_root_state_to_sim(p)
                fr[name] = float(rng.uniform(0.22, 0.38))
                s._set_friction(obj, fr[name])
        self.steps = 0
        self.ever_held = {p: False for p in PARTS}
        self.hold(10)
        self.steps = 0
        self.initial = {"seed": seed, "case_dxy_mm": [round(v * 1000, 2) for v in dxy], "case_dyaw_deg": round(math.degrees(dyaw), 2),
                        "friction": fr}

    # ------------------------------------------------------------------ low level
    def sim_time(self) -> float:
        return self.steps / self.ctrl_hz

    def part(self, name):
        return self.s.card if name == "card" else self.s.rams[int(name[3:])]

    def hand(self):
        return self.F.data.body_pos_w[:, self.i_hand].clone(), self.F.data.body_quat_w[:, self.i_hand].clone()

    def gripper_width(self) -> float:
        return float(self.F.data.joint_pos[0, -2:].sum())

    def held(self) -> str | None:
        h = self.s.grasp_held[0]
        for name, i in self.site_idx.items():
            if bool(h[i]):
                return name
        return None

    def act(self, pos_t, q_t, grip: float, max_u: float = 1.5, n: int = 1) -> None:
        from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

        for _ in range(n):
            hp, hq = self.hand()
            u = torch.zeros(1, self.env.robot.action_dim, device=self.dev)
            u[:, 0:3] = ((pos_t - hp) / 0.02).clamp(-max_u, max_u)
            qe = quat_mul(q_t, quat_conjugate(hq))
            qe = torch.where(qe[:, :1] < 0, -qe, qe)
            u[:, 3:6] = (axis_angle_from_quat(qe) / 0.097).clamp(-3 * max_u, 3 * max_u)
            u[:, 6:] = grip
            self.env.step(u)
            self.steps += 1
            h = self.held()
            if h:
                self.ever_held[h] = True

    def hold(self, n: int = 1, grip: float | None = None) -> None:
        hp, hq = self.hand()
        g = grip if grip is not None else (0.0 if self.gripper_width() < 0.07 else 0.04)
        self.act(hp, hq, g, max_u=0.3, n=int(n * self.SM))

    def goto(self, pos_t, q_t, grip: float, tol: float = 0.003, max_steps: int = 60, max_u: float = 1.5) -> bool:
        from isaaclab.utils.math import quat_error_magnitude

        for _ in range(int(max_steps * self.SM)):
            self.act(pos_t, q_t, grip, max_u)
            hp, hq = self.hand()
            if (hp - pos_t).norm() < tol and quat_error_magnitude(hq, q_t).item() < math.radians(2):
                return True
        return False

    # ------------------------------------------------------------------ geometry / state
    def case_pose(self):
        return self.s.case.data.root_pos_w.clone(), self.s.case.data.root_quat_w.clone()

    def seat_w(self, part: str):
        from isaaclab.utils.math import quat_apply

        cp, cq = self.case_pose()
        off = self.c.gpu_seat_pos if part == "card" else self.c.ram_seat_pos[int(part[3:])]
        return cp + quat_apply(cq, torch.tensor([off], device=self.dev))

    def seated(self, part: str) -> bool:
        return bool(self.s.gpu_seated()[0]) if part == "card" else bool(self.s.ram_seated()[0, int(part[3:])])

    def depth(self, part: str) -> float:
        return float(self.s.gpu_engaged()[0]) if part == "card" else float(self.s.ram_engaged()[0, int(part[3:])])

    def where(self, part: str) -> str:
        obj = self.part(part)
        p = obj.data.root_pos_w[0]
        if self.held() == part:
            return "gripper"
        if self.seated(part):
            return "seated"
        rel = (obj.data.root_pos_w - self.seat_w(part))[0]
        if rel[:2].norm() < 0.006 and self.depth(part) > -0.002:
            return "slot_partial"
        cp, _ = self.case_pose()
        if p[2] < self.c.surface_z - 0.05:
            return "floor"
        if abs(float(p[0] - cp[0, 0])) < 0.22 and abs(float(p[1] - cp[0, 1])) < 0.30 and p[2] > cp[0, 2] - 0.01:
            return "in_case"
        return "table"

    def tilt_deg(self, part: str) -> float:
        from isaaclab.utils.math import quat_error_magnitude

        return math.degrees(quat_error_magnitude(self.part(part).data.root_quat_w, self.case_pose()[1]).item())

    def observe_workstation(self) -> dict:
        parts = {}
        for name in PARTS:
            obj = self.part(name)
            parts[name] = {"where": self.where(name), "slot": SLOT_OF[name], "pos": [round(v, 4) for v in obj.data.root_pos_w[0].tolist()],
                           "quat": [round(v, 4) for v in obj.data.root_quat_w[0].tolist()], "tilt_deg": round(self.tilt_deg(name), 2),
                           "depth_mm": round(self.depth(name) * 1000, 2)}
        slots = {SLOT_OF[n]: {"seat_pos": [round(v, 4) for v in self.seat_w(n)[0].tolist()], "occupied": self.seated(n)} for n in PARTS}
        cp, cq = self.case_pose()
        hp, _ = self.hand()
        return {"parts": parts, "slots": slots, "case": {"pos": [round(v, 4) for v in cp[0].tolist()], "quat": [round(v, 4) for v in cq[0].tolist()],
                                                          "wall_height_m": 0.195},
                "gripper": {"width_m": round(self.gripper_width(), 4), "holding": self.held(), "tcp_pos": [round(v, 4) for v in hp[0].tolist()]},
                "safe_transit_z": SAFE_Z}
