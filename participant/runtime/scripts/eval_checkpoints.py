"""Evaluate every saved PPO checkpoint of an open_drawer run: success rate over N parallel envs + one video each.

    python scripts/eval_checkpoints.py --headless --run_dir logs/rsl_rl/vinci_open_drawer/<run> --num_envs 32

For each model_<it>.pt: success = top drawer opened > 0.20 m before the episode ends.
Writes <run>/eval/eval.jsonl and <run>/eval/it_<it>.mp4 (env 0 viewport) and exports the best
checkpoint to policies/open_drawer_rl.pt (TorchScript) for the `open_drawer(provider="rl")` skill.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--run_dir", required=True)
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--every", type=int, default=1, help="evaluate every k-th checkpoint")
p.add_argument("--export_to", default=os.path.join(os.path.dirname(__file__), "..", "policies", "open_drawer_rl.pt"))
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit
from rsl_rl.runners import OnPolicyRunner

import vinci_lab  # noqa: F401
from vinci_lab.scene import VinciOpenDrawerPPORunnerCfg, VinciOpenDrawerRLEnvCfg_PLAY

env_cfg = VinciOpenDrawerRLEnvCfg_PLAY()
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = "cuda:0"
env_cfg.viewer.eye = (-0.8, 1.4, 1.6)
env_cfg.viewer.lookat = (0.6, 0.0, 0.6)
env = gym.make("Vinci-Open-Drawer-Franka-Play-v0", cfg=env_cfg, render_mode="rgb_array")
venv = RslRlVecEnvWrapper(env)
agent_cfg = VinciOpenDrawerPPORunnerCfg()
unwrapped = env.unwrapped
cabinet = unwrapped.scene["cabinet"]
jid = cabinet.find_joints("drawer_top_joint")[0][0]
horizon = int(unwrapped.max_episode_length)

ckpts = sorted(glob.glob(os.path.join(args.run_dir, "model_*.pt")), key=lambda f: int(re.findall(r"model_(\d+)", f)[0]))
ckpts = ckpts[:: args.every] + ([ckpts[-1]] if ckpts and ckpts[-1] not in ckpts[:: args.every] else [])
out = os.path.join(args.run_dir, "eval")
os.makedirs(out, exist_ok=True)
best = (-1.0, None)
runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
for ck in ckpts:
    it = int(re.findall(r"model_(\d+)", ck)[0])
    runner.load(ck)
    policy = runner.get_inference_policy(device="cuda:0")
    obs, _ = venv.reset()
    max_open = torch.zeros(args.num_envs, device="cuda:0")
    first_success_step = torch.full((args.num_envs,), -1, device="cuda:0")
    frames = []
    t0 = time.time()
    with torch.no_grad():  # not inference_mode: its tensors cannot be updated in place by the next reset/load
        for step in range(horizon - 1):
            obs, _, _, _ = venv.step(policy(obs))
            o = cabinet.data.joint_pos[:, jid]
            newly = (o > 0.20) & (first_success_step < 0)
            first_success_step[newly] = step
            max_open = torch.maximum(max_open, o)
            if step % 2 == 0:
                frames.append(env.render())
    succ = (max_open > 0.20).float()
    rec = {"iteration": it, "success_rate": round(succ.mean().item(), 3), "n": args.num_envs,
           "mean_max_opening_m": round(max_open.mean().item(), 3),
           "failures": {"never_moved(<2cm)": int((max_open < 0.02).sum()), "partial(2-20cm)": int(((max_open >= 0.02) & (max_open <= 0.2)).sum())},
           "median_steps_to_success": int(first_success_step[first_success_step >= 0].median().item()) if (first_success_step >= 0).any() else None,
           "env0_success": bool(succ[0].item()), "eval_s": round(time.time() - t0, 1),
           "video": os.path.join(out, f"it_{it:04d}.mp4")}
    imageio.mimsave(rec["video"], [np.asarray(f) for f in frames], fps=30, macro_block_size=1)
    print("EVAL " + json.dumps(rec), flush=True)
    with open(os.path.join(out, "eval.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    if rec["success_rate"] >= best[0]:
        best = (rec["success_rate"], ck)
if best[1] is not None:
    r = runner
    r.load(best[1])
    best = (best[0], int(re.findall(r"model_(\d+)", best[1])[0]))
    os.makedirs(os.path.dirname(os.path.abspath(args.export_to)), exist_ok=True)
    try:
        normalizer = r.obs_normalizer
    except AttributeError:
        normalizer = None
    policy_nn = r.alg.policy if hasattr(r.alg, "policy") else r.alg.actor_critic
    export_policy_as_jit(policy_nn, normalizer, os.path.dirname(os.path.abspath(args.export_to)), os.path.basename(args.export_to))
    print(f"EXPORTED best it={best[1]} success={best[0]} -> {args.export_to}", flush=True)
os._exit(0)
