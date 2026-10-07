"""./evaluate.sh (CPU edition): play DEV episodes with team/brain.py, grade them, print the same report as the GPU workshop.

    ./evaluate.sh                     # settings from team/config.yaml (evaluate: episodes, procs, seed0)
    ./evaluate.sh --episodes 20       # more episodes = a more reliable score
    ./evaluate.sh --view              # watch the robot (one process, real time)
    ./evaluate.sh --videos 2          # also film 2 episodes (GIF or MP4 in results/)
    ./evaluate.sh --level training    # the 1-piece warm-up level
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]           # cpu/
TEAM_DIR = Path(os.environ.get("VINCI_TEAM_DIR", HERE / "team"))
RESULTS = Path(os.environ.get("VINCI_RESULTS", HERE / "results"))
HINTS = {
    "grasp_failed": "a piece was never picked up (missed grasp) and stayed on the table",
    "dropped_in_transport": "a piece slipped out of the gripper on the way",
    "wrong_destination": "a piece ended in the wrong container",
    "drawer_not_closed": "the drawer was left open at the end",
    "distractor_moved": "the gauge (don't touch!) was moved",
    "object_on_floor": "a piece fell on the floor",
    "timeout": "the episode ran out of simulated time",
}
MAXPTS = {"identified": 10, "grasped": 15, "placed_correctly": 25, "wrong_placement_avoided": 10,
          "drawer_handled": 15, "final_state_verified": 10, "task_complete": 15}


def _worker(args):
    level_name, seeds, team_dir, out_dir, view, videos = args
    from .grader import grade
    from .lab import CpuLab
    from .levels import load_level
    from .skills import Skills
    from .tools import load_brain, load_config, solve
    level = load_level(level_name)
    config = load_config(Path(team_dir))
    brain = load_brain(Path(team_dir))
    lab = CpuLab(level, cameras=bool(videos), viewer=view, realtime=1.0 if view else 0.0)
    sk = Skills(lab)
    reps = []
    for i, seed in enumerate(seeds):
        t = time.time()
        lab.reset_episode(seed)
        rec = videos and i < videos
        if rec:
            lab.start_recording()
        trace, summ = solve(brain, sk, lab, level.get("sim_time_budget_s", 240), config)
        rep = grade(lab, summ)
        rep["episode_wall_s"] = round(time.time() - t, 1)
        if rec:
            rep["video"] = lab.save_video(str(Path(out_dir) / f"ep_{seed:04d}_{'ok' if rep['success'] else 'fail'}.mp4"))
        rep["trace"] = trace
        reps.append(json.loads(json.dumps(rep, default=str)))
        print(f"  episode seed {seed}: {'OK  ' if rep['success'] else 'FAIL'} {rep['progression']:5.1f}/100 "
              f"({rep['sim_time_s']:.0f} s simulated, {rep['episode_wall_s']:.0f} s)", flush=True)
    lab.close()
    return reps


def team_hash(team_dir: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(team_dir.glob("*")):
        if f.is_file() and f.suffix in (".py", ".yaml", ".yml", ".md"):
            h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


def main():
    from .tools import load_config
    cfg = load_config(TEAM_DIR)
    ev = cfg.get("evaluate") or {}
    p = argparse.ArgumentParser(description="Evaluate team/brain.py on DEV episodes (CPU edition)")
    p.add_argument("--episodes", type=int, default=int(ev.get("episodes", 8)))
    p.add_argument("--procs", type=int, default=int(ev.get("procs", 4)))
    p.add_argument("--seed0", type=int, default=int(ev.get("seed0", 0)))
    p.add_argument("--videos", type=int, default=int(ev.get("videos", 0)))
    p.add_argument("--level", default="dev", choices=["training", "dev"])
    p.add_argument("--view", action="store_true", help="watch the robot in a window (one process, real time)")
    a = p.parse_args()
    if a.episodes < 1 or a.procs < 1 or a.videos < 0:
        p.error("episodes/procs must be positive; videos cannot be negative")
    run = RESULTS / (time.strftime("run_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    (run / "episodes").mkdir(parents=True)
    snap = run / "team_snapshot"
    shutil.copytree(TEAM_DIR, snap, ignore=shutil.ignore_patterns("__pycache__"))
    th = team_hash(TEAM_DIR)
    seeds = list(range(a.seed0, a.seed0 + a.episodes))
    procs = 1 if (a.view or a.videos) else max(1, min(a.procs, len(seeds), os.cpu_count() or 1))
    print(f"VINCI Robotics Challenge (CPU edition) — playing {a.episodes} {a.level.upper()} episodes with team/brain.py "
          f"(team {th}) on {procs} process(es) ... results in {run.relative_to(HERE)}/", flush=True)
    t0 = time.time()
    chunks = [seeds[i::procs] for i in range(procs)]
    jobs = [(a.level, c, str(snap), str(run / "episodes"), a.view, a.videos if k == 0 else 0) for k, c in enumerate(chunks) if c]
    if procs == 1:
        parts = [_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(procs) as pool:
            parts = pool.map(_worker, jobs)
    eps = sorted((e for part in parts for e in part), key=lambda e: e["initial"]["seed"])
    with open(run / "episodes" / "episodes.jsonl", "w") as f:
        for e in eps:
            f.write(json.dumps(e) + "\n")
    crashed = sum(any(t.get("type") == "decision" and str(t.get("text", "")).startswith("BRAIN CRASHED") for t in e["trace"])
                  for e in eps)
    complete = [e["initial"]["seed"] for e in eps] == seeds and not crashed
    (run / "report.json").write_text(json.dumps({"complete": complete, "level": a.level, "seeds": seeds,
                                                 "episode_scores": [e["progression"] for e in eps], "team_hash": th,
                                                 "edition": "cpu"}))
    lines = [f"VINCI Robotics Challenge (CPU edition) — {a.level.upper()} — team/brain.py ({th})", ""]
    for i, e in enumerate(eps, 1):
        lines.append(f"Episode {i:2d} (seed {e['initial']['seed']}): {'OK  ' if e['success'] else 'FAIL'} "
                     f"{e['progression']:5.1f}/100" + (f"   {e['failure']}" if e.get("failure") else ""))
    n, k = len(eps), sum(e["success"] for e in eps)
    lines += ["", f"Complete successes: {k}/{n}", f"Average score:      {sum(e['progression'] for e in eps) / n:.1f}/100",
              f"Average sim time:   {sum(e['sim_time_s'] for e in eps) / n:.0f} s", "",
              "Score detail (average points per episode / max):"]
    comps = [e.get("progression_components") or {} for e in eps]
    parts_txt = [f"{c} {sum(x.get(c, 0) for x in comps) / n:.1f}/{m}" for c, m in MAXPTS.items()]
    lines += ["  " + " | ".join(parts_txt[:4]), "  " + " | ".join(parts_txt[4:]),
              "A complete success = the final state is right; the score also counts the steps (e.g. verifying the final state)."]
    fails = {}
    for e in eps:
        if e.get("failure"):
            fails[e["failure"]] = fails.get(e["failure"], 0) + 1
    if fails:
        lines += ["", "What went wrong:"] + [f"  {c} x{m}: {HINTS.get(c, c)}" for c, m in sorted(fails.items(), key=lambda x: -x[1])]
    if crashed:
        lines += ["", f"WARNING: brain.py crashed in {crashed} episode(s): see the Python error above / in the traces."]
    vids = [e["video"] for e in eps if e.get("video")]
    if vids:
        lines += ["", "Videos: " + ", ".join(str(Path(v).relative_to(HERE)) for v in vids)]
    lines += ["", f"Details (trace of every decision and skill call): {(run / 'episodes').relative_to(HERE)}/episodes.jsonl",
              f"Took {(time.time() - t0) / 60:.1f} min"]
    text = "\n".join(lines)
    (run / "report.txt").write_text(text + "\n")
    latest = RESULTS / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run.name)
    print("\n" + text, flush=True)
    if a.view:  # the viewer's OpenGL teardown at interpreter exit can segfault: everything is written, leave now
        os._exit(0)


if __name__ == "__main__":
    main()
