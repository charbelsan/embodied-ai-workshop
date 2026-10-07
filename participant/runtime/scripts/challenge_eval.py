"""Run a Brain (baseline | yours) on N episodes of a challenge level and grade each final state.

    python scripts/challenge_eval.py --headless --level dev --solution baseline --episodes 5 --seed0 0 \\
        --out results/baseline_dev

Eval modes: --cameras 0 (default, fast: valid only if no skill uses vision, i.e. open_drawer rl|analytic and
pick_and_place scripted) or --cameras 1 (needed for open_drawer visual / pick_and_place vla; slower).
Writes episodes.jsonl (one grader report per episode) and summary.json (success, Wilson 95% CI, failure modes, times).
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--level", default="dev")
p.add_argument("--solution", default="baseline",
               help="baseline | path/to/my_solution.py:function  (function(sk, lab, budget_s, open_provider) -> (trace, summary))")
p.add_argument("--episodes", type=int, default=10)
p.add_argument("--seed0", type=int, default=0)
p.add_argument("--level_seeds", action="store_true", help="use the seeds listed in the level file, ignoring --seed0")
p.add_argument("--cameras", type=int, default=0)
p.add_argument("--open_provider", default="rl")
p.add_argument("--sim_device", default="cuda:0", help="cuda:0 or cpu (PhysX CPU: 1 env, no per-step GPU syncs)")
p.add_argument("--video_every", type=int, default=0, help="record the front camera every k-th episode (needs --cameras 1)")
p.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = bool(args.cameras)
T0 = time.time()
app = AppLauncher(args).app

from vinci_lab.challenge.grader import grade  # noqa: E402
from vinci_lab.challenge.levels import load_level  # noqa: E402
from vinci_lab.challenge.skills import ChallengeSkills  # noqa: E402
from vinci_lab.challenge import solutions as reference_solutions  # noqa: E402
from vinci_lab.challenge.world import ChallengeLab  # noqa: E402

os.makedirs(args.out, exist_ok=True)
level = load_level(args.level)
SEEDS = level["seeds"][: args.episodes] if args.level_seeds and level.get("seeds") else [args.seed0 + i for i in range(args.episodes)]
lab = ChallengeLab(level, cameras=bool(args.cameras), device=args.sim_device)
sk = ChallengeSkills(lab)
REFERENCE = {"baseline": "baseline_solution"}
if args.solution in REFERENCE:
    if not hasattr(reference_solutions, REFERENCE[args.solution]):
        sys.exit(f"solution {args.solution!r} is not available here (use 'baseline' or path.py:function)")
    solve = getattr(reference_solutions, REFERENCE[args.solution])
else:  # participant solution: "file.py:function"
    import importlib.util
    path, fn = args.solution.rsplit(":", 1)
    spec = importlib.util.spec_from_file_location("participant_solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    solve = getattr(mod, fn)
t_ready = time.time() - T0
reports = []
for i, seed in enumerate(SEEDS):
    t_ep = time.time()
    lab.steps = 0
    lab.reset_episode(seed)
    rec = bool(args.video_every) and args.cameras and i % args.video_every == 0
    if rec:
        lab.start_recording()
    trace, summ = solve(sk, lab, budget_s=level.get("sim_time_budget_s", 240), open_provider=args.open_provider)
    rep = grade(lab, summ)
    rep["solution"], rep["episode_wall_s"] = os.path.basename(args.solution), round(time.time() - t_ep, 1)
    if rec:
        rep["video"] = lab.save_video(os.path.join(args.out, f"ep_{seed:04d}_{'ok' if rep['success'] else 'fail'}.mp4"))
    reports.append(rep)
    with open(os.path.join(args.out, "episodes.jsonl"), "a") as f:
        f.write(json.dumps({**rep, "trace": trace}, default=str) + "\n")
    print(f"EP seed={seed} success={rep['success']} score={rep['score']}/{rep['max_score']} progression={rep['progression']} failure={rep['failure']} "
          f"calls={summ['skill_calls']} sim={summ['sim_time_s']}s wall={rep['episode_wall_s']}s", flush=True)


def wilson(k, n, z=1.96):
    if not n:
        return [0.0, 0.0]
    ph = k / n; d = 1 + z * z / n; c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


k = sum(r["success"] for r in reports)
summary = {"level": args.level, "solution": args.solution, "episodes": len(reports), "success": k,
           "success_rate": round(k / max(len(reports), 1), 3), "wilson95": wilson(k, len(reports)),
           "mean_score_pct": round(100 * sum(r["score"] / r["max_score"] for r in reports) / max(len(reports), 1), 1),
           "mean_progression": round(sum(r["progression"] for r in reports) / max(len(reports), 1), 1),
           "progression_components_mean": {k: round(sum(r["progression_components"][k] for r in reports) / max(len(reports), 1), 1)
                                           for k in reports[0]["progression_components"]} if reports else {},
           "failure_modes": dict(Counter(r["failure"] for r in reports if not r["success"])),
           "mean_sim_s": round(sum(r["sim_time_s"] for r in reports) / max(len(reports), 1), 1),
           "mean_episode_wall_s": round(sum(r["episode_wall_s"] for r in reports) / max(len(reports), 1), 1),
           "sim_ready_s": round(t_ready, 1), "cameras": args.cameras, "seeds": [SEEDS[0], SEEDS[-1]] if not args.level_seeds else "level_seeds"}
json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
print("CHALLENGE " + json.dumps(summary), flush=True)
os._exit(0)
