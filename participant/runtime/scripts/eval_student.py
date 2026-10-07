"""Evaluate a DAgger vision student ALONE on the randomized drawer task, with its real images or with masked images.

    python scripts/eval_student.py --headless --checkpoint policies/open_drawer_visual.pt --num_envs 128 [--perturb] [--mask_images]

Same environment, fixed evaluation episodes (eval_params, --eval_seed), disturbances and metric as the DAgger pipeline:
the cabinet pose (+-6/+-12 cm, +-0.2 rad), drawer dynamics and arm start vary per episode; the student acts at 30 Hz from
front + wrist RGB-D (64 px) + joint pos/vel; nothing else drives the arm (no teacher takeover). The simulator's drawer
opening is read by this evaluator only. --mask_images replaces both images by zeros: if the success rate does not drop,
the policy does not use vision (it replays a memorized trajectory).
Prints one JSON line: success (> 0.20 m), reached_0.30m, mean max opening, checkpoint sha256.
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--eval_seed", type=int, default=2026)
p.add_argument("--perturb", action="store_true")
p.add_argument("--push_rad", type=float, default=0.35)
p.add_argument("--slip_prob", type=float, default=0.7)
p.add_argument("--mask_images", action="store_true")
p.add_argument("--swap_images", action="store_true", help="each episode sees the images of ANOTHER episode (other cabinet pose)")
p.add_argument("--out", default="")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import vinci_lab.dagger  # noqa: F401,E402
from vinci_lab.dagger.env_cfg import SUCCESS_OPENING, VinciDaggerStudentEnvCfg, eval_params  # noqa: E402
from vinci_lab.dagger.student import Student, to_student_image  # noqa: E402

DEV = "cuda:0"
try:
    N = args.num_envs
    cfg = VinciDaggerStudentEnvCfg()
    cfg.scene.num_envs = N
    cfg.sim.device = DEV
    env = ManagerBasedRLEnv(cfg=cfg)
    sc = env.scene
    cab, front, wrist = sc["cabinet"], sc["front_cam"], sc["wrist_cam"]
    JID = cab.find_joints("drawer_top_joint")[0][0]
    T = int(env.max_episode_length) - 1
    sha = hashlib.sha256(open(args.checkpoint, "rb").read()).hexdigest()
    model = Student().to(DEV)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEV))
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(args.eval_seed + 7)
    pert = {"t_push": torch.randint(8, 36, (N,), generator=g).to(DEV),
            "delta": ((torch.rand(N, 7, generator=g) * 2 - 1) * args.push_rad).to(DEV),
            "do_slip": (torch.rand(N, generator=g) < args.slip_prob).to(DEV),
            "slip_start": torch.full((N,), -1, device=DEV, dtype=torch.long)}
    env.dagger_forced = eval_params(N, args.eval_seed, DEV)
    obs, _ = env.reset()
    env.dagger_forced = None
    max_open = torch.zeros(N, device=DEV)
    t0 = time.time()
    with torch.no_grad():
        for t in range(T):
            opening = cab.data.joint_pos[:, JID]
            fo, wo = front.data.output, wrist.data.output
            f = to_student_image(fo["rgb"], fo["distance_to_image_plane"])
            w = to_student_image(wo["rgb"], wo["distance_to_image_plane"])
            if args.mask_images:
                f, w = torch.zeros_like(f), torch.zeros_like(w)
            if args.swap_images:  # realistic images, wrong scene: vision content must matter, not just its presence
                f, w = torch.roll(f, N // 2, dims=0), torch.roll(w, N // 2, dims=0)
            a = model(f, w, obs["student"]).clone()
            if args.perturb:
                push = (t >= pert["t_push"]) & (t < pert["t_push"] + 10)
                a[push, :7] += pert["delta"][push]
                newly = pert["do_slip"] & (pert["slip_start"] < 0) & (opening > 0.06)
                pert["slip_start"][newly] = t
                slip = (pert["slip_start"] >= 0) & (t < pert["slip_start"] + 15)
                a[slip, 7] = 1.0
            max_open = torch.maximum(max_open, opening)
            obs, _, _, _, _ = env.step(a)
    max_open = torch.maximum(max_open, cab.data.joint_pos[:, JID])
    res = {"checkpoint": args.checkpoint, "sha256": sha, "episodes": N, "eval_seed": args.eval_seed,
           "perturbed": args.perturb, "images_masked": args.mask_images, "images_swapped": args.swap_images,
           "success_0.20m": round((max_open > SUCCESS_OPENING).float().mean().item(), 3),
           "reached_0.30m": round((max_open >= 0.30).float().mean().item(), 3),
           "mean_max_opening_m": round(max_open.mean().item(), 3), "rollout_s": round(time.time() - t0, 1)}
    print("EVAL_STUDENT " + json.dumps(res), flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
except BaseException:
    import traceback
    traceback.print_exc()
sys.stdout.flush()
os._exit(0)
