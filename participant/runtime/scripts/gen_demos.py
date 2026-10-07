"""Generate pick-and-place demonstrations in OUR Isaac scene with the scripted skill (the "teacher").

    python scripts/gen_demos.py --headless --episodes 60 --out demos/raw

Each episode starts with the top drawer already open (0.20-0.28 m) and the cube jittered +/-3 cm on the stand.
Records (obs_t, a_t) at 30 fps: front + wrist RGB 256x256, 9-D joint state, 8-D action. Keeps successes only.
Raw episodes -> <out>/ep_XXXX.npz ; convert with scripts/to_lerobot.py (lerobot venv).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=60)
p.add_argument("--out", default="demos/raw")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--start_index", type=int, default=0, help="first file index (to append to an existing raw dir)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
T0 = time.time()
app = AppLauncher(args).app

import numpy as np
import torch

from vinci_lab.api import LabEnv
from vinci_lab.scene import TASK_PICK
from vinci_lab.skills import Skills

os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(args.seed)
lab = LabEnv()
sk = Skills(lab)
buf = {}


def recorder(lab_, action):
    o = lab_.observe()
    buf["front"].append(o["rgb_front"])
    buf["wrist"].append(o["rgb_wrist"])
    buf["state"].append(np.asarray(o["joint_pos"], dtype=np.float32))
    buf["action"].append(action[0].detach().cpu().numpy().astype(np.float32))


stats = {"ok": 0, "fail": 0, "reasons": {}}
for ep in range(args.episodes):
    lab.reset()
    sk.go_pick_start()  # the home pose would collide with the opened drawer (raised base)
    # precondition: drawer open, cube jittered in xy and yaw (joint/root writes, then settle)
    jp = lab.cabinet.data.joint_pos.clone()
    jp[0, lab.drawer_jid] = float(rng.uniform(0.20, 0.28))
    lab.cabinet.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    pose = lab.cube.data.default_root_state[:, :7].clone()
    pose[0, :2] += torch.tensor(rng.uniform(-0.03, 0.03, 2), dtype=torch.float32, device=lab.device)
    pose[0, :3] += lab.env.scene.env_origins[0]
    yaw = float(rng.uniform(-0.35, 0.35))
    pose[0, 3:7] = torch.tensor([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=torch.float32, device=lab.device)
    lab.cube.write_root_pose_to_sim(pose)
    lab.cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=lab.device))
    lab.hold(30)
    buf = {"front": [], "wrist": [], "state": [], "action": []}
    t_ep = time.time()
    lab.demo_recorder = recorder
    r = sk.pick_and_place("cube", "top_drawer")
    lab.hold(10)
    lab.demo_recorder = None
    if r["ok"]:
        stats["ok"] += 1
        np.savez_compressed(os.path.join(args.out, f"ep_{args.start_index + stats['ok'] - 1:04d}.npz"),
                            front=np.stack(buf["front"]), wrist=np.stack(buf["wrist"]),
                            state=np.stack(buf["state"]), action=np.stack(buf["action"]), task=TASK_PICK)
    else:
        stats["fail"] += 1
        stats["reasons"][r["reason"]] = stats["reasons"].get(r["reason"], 0) + 1
    print(f"EP {ep} ok={r['ok']} frames={len(buf['action'])} wall={time.time() - t_ep:.1f}s reason={r['reason']}", flush=True)
stats["total_s"] = round(time.time() - T0, 1)
print("DEMOS " + json.dumps(stats), flush=True)
os._exit(0)
