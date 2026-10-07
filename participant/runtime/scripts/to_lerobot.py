"""Convert raw demo episodes (gen_demos.py .npz) into a LeRobot dataset (lerobot 0.6.1 format). Run in the lerobot venv.

    <lerobot venv>/bin/python scripts/to_lerobot.py --raw demos/raw \\
        --root demos/vinci_pick_place_drawer --repo_id vinci/pick_place_drawer
"""
import argparse
import glob
import json
import time

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

p = argparse.ArgumentParser()
p.add_argument("--raw", required=True)
p.add_argument("--root", required=True)
p.add_argument("--repo_id", default="vinci/pick_place_drawer")
p.add_argument("--vcodec", default="h264", help="h264 (CPU), h264_nvenc (GPU) or libsvtav1 (lerobot default, slow on 8 vCPU)")
p.add_argument("--episodes", type=int, default=0, help="0 = all")
p.add_argument("--action_mode", choices=["abs", "rel"], default="abs",
               help="abs: arm = joint targets as offsets from home (SPEC). rel: arm = target - CURRENT joint pos (visual-servo style)")
p.add_argument("--streaming", action="store_true", help="lerobot streaming encoder (dropped 1 wrist frame of ep 11 in 2 of 3 runs: off by default)")
args = p.parse_args()
t0 = time.time()
HOME = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741], dtype=np.float32)  # Franka default arm pose (SPEC)
JOINTS = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
features = {
    "observation.images.front": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
    "observation.images.wrist": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (9,), "names": JOINTS},
    "action": {"dtype": "float32", "shape": (8,), "names": [f"{j}_{args.action_mode}" for j in JOINTS[:7]] + ["gripper_binary"]},
}
from lerobot.configs.video import RGBEncoderConfig

enc = RGBEncoderConfig(vcodec=args.vcodec, crf=23 if args.vcodec.startswith("h264") else 30,
                       preset="veryfast" if args.vcodec == "h264" else None)
ds = LeRobotDataset.create(repo_id=args.repo_id, fps=30, root=args.root, robot_type="franka_panda",
                           features=features, use_videos=True, rgb_encoder=enc, streaming_encoding=args.streaming,
                           image_writer_threads=0 if args.streaming else 8)
n_frames = 0
files = sorted(glob.glob(f"{args.raw}/ep_*.npz"))
files = files[: args.episodes] if args.episodes else files
for f in files:
    with np.load(f) as z:  # NpzFile decompresses the whole array on EVERY z[key] access: load each key once
        d = {k: z[k] for k in ("front", "wrist", "state", "action", "task")}
    task = str(d["task"])
    if args.action_mode == "rel":  # target_abs = HOME + a[:7]; rel = target_abs - q_t
        d["action"] = d["action"].copy()
        d["action"][:, :7] = d["action"][:, :7] + HOME[None] - d["state"][:, :7]
    for t in range(len(d["action"])):
        ds.add_frame({"observation.images.front": d["front"][t], "observation.images.wrist": d["wrist"][t],
                      "observation.state": d["state"][t], "action": d["action"][t], "task": task})
    ds.save_episode()
    n_frames += len(d["action"])
if hasattr(ds, "finalize"):
    ds.finalize()
# integrity check: every episode's video segment must contain exactly `length` frames, for every camera
import pandas as pd

eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{args.root}/meta/episodes/*/*.parquet"))])
bad = []
for cam in ("front", "wrist"):
    k0, k1 = f"videos/observation.images.{cam}/from_timestamp", f"videos/observation.images.{cam}/to_timestamp"
    n = ((eps[k1] - eps[k0]) * 30).round().astype(int)
    bad += [(cam, int(e), int(l), int(m)) for e, l, m in zip(eps.episode_index, eps.length, n) if l != m]
if bad:
    raise SystemExit(f"VIDEO INTEGRITY FAILED (camera, episode, length, encoded): {bad}")
print("video integrity OK:", len(eps), "episodes x 2 cameras")
print("LEROBOT " + json.dumps({"episodes": len(files), "frames": n_frames, "root": args.root, "vcodec": args.vcodec,
                              "s": round(time.time() - t0, 1), "s_per_episode": round((time.time() - t0) / max(len(files), 1), 1)}))
