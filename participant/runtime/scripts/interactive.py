"""Interactive mode: 1 env, Isaac Sim UI (run inside the DCV desktop), the robot shows a first move, then idles.

    python scripts/interactive.py            # from the DCV terminal (DISPLAY=:0)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--demo", action="store_true", help="play a first move_to so the participant sees the robot move")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
T0 = time.time()
app = AppLauncher(args).app

from vinci_lab.api import LabEnv
from vinci_lab.skills import Skills

lab = LabEnv()
sk = Skills(lab)
print(f"INTERACTIVE_READY {time.time() - T0:.1f}s", flush=True)
if args.demo:
    r = sk.move_to("above_cube")
    print(f"FIRST_MOTION {time.time() - T0:.1f}s {r['ok']} {r['reason']}", flush=True)
while app.is_running():
    lab.hold(1)
