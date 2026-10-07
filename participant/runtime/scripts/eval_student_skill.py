"""Integration check of the learned skill in the workshop scene: open_drawer_student() alone, N episodes.

    python scripts/eval_student_skill.py --headless --episodes 20 [--mask_images] [--out file.json]

The skill runs without simulator truth (fixed horizon, retreat relative to the end effector) and reports only its
execution + checkpoint sha256; THIS evaluator reads the drawer opening afterwards (success = opening >= 0.30 m, the
skill's target, and > 0.20 m). The workshop scene's cabinet does not move: vision relevance is measured in the
randomized environment (scripts/eval_student.py), this script checks the skill end to end in the participants' scene.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=20)
p.add_argument("--mask_images", action="store_true")
p.add_argument("--out", default="")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

from vinci_lab.api import LabEnv  # noqa: E402
from vinci_lab.student_skills import open_drawer_student  # noqa: E402

try:
    lab = LabEnv()
    eps = []
    for i in range(args.episodes):
        lab.reset()
        t0 = time.time()
        r = open_drawer_student(lab, mask_images=args.mask_images)
        opening = lab.drawer_opening()  # evaluator only
        eps.append({"episode": i, "ran": r["ok"], "reason": r["reason"], "opening_m": round(opening, 3),
                    "success_0.30m": opening >= 0.30, "success_0.20m": opening > 0.20,
                    "sha256": r["evidence"].get("checkpoint_sha256"), "wall_s": round(time.time() - t0, 1),
                    "sim_s": lab.observe()["sim_time_s"]})
        print("EP " + json.dumps(eps[-1]), flush=True)
    n = len(eps)
    res = {"episodes": n, "images_masked": args.mask_images, "sha256": eps[0]["sha256"] if eps else None,
           "ran": sum(e["ran"] for e in eps), "success_0.30m": sum(e["success_0.30m"] for e in eps),
           "success_0.20m": sum(e["success_0.20m"] for e in eps),
           "mean_opening_m": round(sum(e["opening_m"] for e in eps) / max(n, 1), 3),
           "mean_wall_s": round(sum(e["wall_s"] for e in eps) / max(n, 1), 1)}
    print("EVAL_SKILL " + json.dumps(res), flush=True)
    if args.out:
        json.dump({"summary": res, "episodes": eps}, open(args.out, "w"), indent=1)
except BaseException:
    import traceback
    traceback.print_exc()
sys.stdout.flush()
os._exit(0)
