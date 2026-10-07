"""./evaluate.sh: play DEV episodes with team/brain.py, grade them, print a readable report.

    ./evaluate.sh                     # settings from team/config.yaml (evaluate: ...)
    ./evaluate.sh --episodes 20       # more episodes = a more reliable score
    ./evaluate.sh --videos 2          # also film 2 episodes (cameras on, slower)
    ./evaluate.sh --level training    # the 1-piece warm-up level
"""
import argparse
import glob
import fcntl
import math
import uuid
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timeline  # noqa: E402
from team_entry import RUNTIME, TEAM_DIR, WORKSPACE, load_config, load_task

ROOT = WORKSPACE  # results/ live in the team workspace  # noqa: E402

HINTS = {
    "grasp_failed": "a piece was never picked up (missed grasp) and stayed on the table",
    "dropped_in_transport": "a piece slipped out of the gripper on the way",
    "wrong_destination": "a piece ended in the wrong container",
    "drawer_not_open": "the drawer was not open enough to put a piece in",
    "drawer_not_closed": "the drawer was left open at the end",
    "distractor_moved": "the gauge (don't touch!) was moved",
    "object_on_floor": "a piece fell on the floor",
    "timeout": "the episode ran out of simulated time",
}


def run_logged(cmd, log_path, env, timeout_s):
    """Run one harness command with a hard timeout; on timeout kill its whole process group (Isaac children too)."""
    import signal
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, cwd=ROOT,
                             start_new_session=True)
        try:
            p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            log.write(f"\nEVALUATE: stopped after the {timeout_s:.0f} s hard timeout\n")
            print(f"stopped after {timeout_s / 60:.0f} min (hard timeout): see {log_path}", flush=True)
        return p.returncode


def team_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(TEAM_DIR.glob("*")):
        if f.is_file() and f.suffix in (".py", ".yaml", ".yml", ".md"):
            h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


def main():
    cfg = load_config()
    ev = cfg.get("evaluate") or {}
    task = load_task(cfg.get("task", "kitting_cabinet"))
    p = argparse.ArgumentParser(description="Evaluate team/brain.py on DEV episodes")
    p.add_argument("--episodes", type=int, default=int(ev.get("episodes", 8)))
    p.add_argument("--procs", type=int, default=int(ev.get("procs", 4)))
    p.add_argument("--seed0", type=int, default=int(ev.get("seed0", 0)))
    p.add_argument("--videos", type=int, default=int(ev.get("videos", 0)))
    p.add_argument("--level", default=task.get("default_level", "dev"), choices=task["levels"])
    a = p.parse_args()

    if a.episodes < 1 or a.procs < 1 or a.videos < 0:
        p.error("episodes/procs must be positive; videos cannot be negative")
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = open(os.environ.get("VINCI_EVAL_LOCK", str(ROOT / ".evaluation.lock")), "a")
    print("Waiting for any previous evaluation to finish...", flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    py = sys.executable
    run = ROOT / "results" / (time.strftime("run_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    shutil.copytree(TEAM_DIR, run / "team_snapshot", ignore=shutil.ignore_patterns("__pycache__"))
    th = team_hash()
    template = RUNTIME / "templates" / "team"
    if template.exists() and not timeline.seen("team_modified"):
        cur = {f.name: f.read_bytes() for f in TEAM_DIR.glob("*") if f.is_file()}
        ref = {f.name: f.read_bytes() for f in template.glob("*") if f.is_file()}
        if cur != ref:
            timeline.log("team_modified", team_hash=th)
    timeline.log("evaluate_start", run=run.name, team_hash=th, episodes=a.episodes, level=a.level)

    solution = f"{RUNTIME / 'lib' / 'team_entry.py'}:solve"
    harness = str(RUNTIME / task["harness"])
    vision = [f"{k}={v}" for k, v in (cfg.get("providers") or {}).items() if v in (task.get("vision_providers") or {}).get(k, [])]
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}  # Jupyter's inline backend breaks Isaac
    env.update(OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1", VINCI_TEAM_DIR=str(run / "team_snapshot"))
    # hard timeout: boot + 3x the real-time cost of a 240 s-simulated episode, per sequential batch
    per_proc = -(-a.episodes // max(a.procs, 1))
    hard_s = 300 + 240 * per_proc
    t0 = time.time()
    print(f"VINCI Robotics Challenge — {task['title']}\nplaying {a.episodes} {a.level.upper()} episodes with team/brain.py"
          f" (team {th}) ... results in {run.relative_to(ROOT)}/", flush=True)
    if vision:
        print(f"note: {', '.join(vision)} uses the cameras -> one process with rendering, slower", flush=True)
        cmd = [py, harness, "--headless", "--cameras", "1", "--level", a.level, "--solution", solution,
               "--episodes", str(a.episodes), "--seed0", str(a.seed0), "--out", str(run / "episodes")]
        harness_rc = run_logged(cmd, run / "eval.log", env, 300 + 480 * a.episodes)
    else:
        cmd = [py, str(RUNTIME / task["dev_runner"]), "--solution", solution, "--episodes", str(a.episodes),
               "--procs", str(a.procs), "--seed0", str(a.seed0), "--level", a.level, "--out", str(run / "episodes")]
        harness_rc = run_logged(cmd, run / "eval.log", env, hard_s)
    videos = []
    if a.videos > 0:
        print(f"filming {a.videos} episode(s) ...", flush=True)
        cmd = [py, harness, "--headless", "--cameras", "1", "--video_every", "1", "--level", a.level, "--solution", solution,
               "--episodes", str(a.videos), "--seed0", str(a.seed0), "--out", str(run / "videos")]
        run_logged(cmd, run / "videos.log", env, 300 + 480 * a.videos)
        videos = sorted(glob.glob(str(run / "videos" / "*.mp4")))

    eps = [json.loads(l) for f in sorted(glob.glob(str(run / "episodes" / "**" / "episodes.jsonl"), recursive=True))
           for l in open(f) if l.strip()]
    expected = list(range(a.seed0, a.seed0 + a.episodes))
    seeds = sorted(e["initial"]["seed"] for e in eps)
    scores_valid = all(type(e.get("progression")) in (int, float) and math.isfinite(e["progression"]) and 0 <= e["progression"] <= 100 for e in eps)
    brain_errors = any(any(t.get("type") == "decision" and str(t.get("text", "")).startswith("BRAIN CRASHED") for t in e.get("trace", [])) for e in eps)
    complete = harness_rc == 0 and seeds == expected and scores_valid and not brain_errors
    ordered = sorted(eps, key=lambda e: e["initial"]["seed"])
    (run / "report.json").write_text(json.dumps({"complete": complete, "level": a.level,
        "seeds": seeds, "episode_scores": [e["progression"] for e in ordered] if scores_valid else [],
        "episodes_expected": a.episodes, "harness_exit_code": harness_rc, "team_hash": th}, allow_nan=False))
    lines = [f"VINCI Robotics Challenge — {a.level.upper()} — team/brain.py ({th})"]
    models = {}
    try:  # the exact models the skills loaded (same registry, same team/config.yaml)
        sys.path.insert(0, str(RUNTIME))
        from vinci_lab import checkpoints
        for name in ("open_drawer_rl", "open_drawer_student"):
            sel = checkpoints.selected(name)
            models[name] = sel["sha256"]
            lines.append(f"model {name}: sha256 {sel['sha256'][:12]}... ({sel['source']})")
    except Exception as e:  # noqa: BLE001
        lines.append(f"models: not resolved ({e})")
    lines.append("")
    if not eps:
        lines.append(f"No episode finished. See {run.relative_to(ROOT)}/eval.log and episodes_part*.log (a Python error in brain.py?)")
    eps.sort(key=lambda e: e["initial"]["seed"])
    for i, e in enumerate(eps, 1):
        mark = "OK  " if e["success"] else "FAIL"
        lines.append(f"Episode {i:2d} (seed {e['initial']['seed']}): {mark} {e['progression']:5.1f}/100"
                     + (f"   {e['failure']}" if e.get("failure") else ""))
    crashed = sum(any(t.get("type") == "decision" and str(t.get("text", "")).startswith("BRAIN CRASHED") for t in e.get("trace", []))
                  for e in eps)
    if eps:
        n, k = len(eps), sum(e["success"] for e in eps)
        lines += ["", f"Complete successes: {k}/{n}", f"Average score:      {sum(e['progression'] for e in eps) / n:.1f}/100",
                  f"Average sim time:   {sum(e['sim_time_s'] for e in eps) / n:.0f} s"]
        # where the points come from: a complete success can still miss points (e.g. never verifying the final state)
        MAXPTS = {"identified": 10, "grasped": 15, "placed_correctly": 25, "wrong_placement_avoided": 10,
                  "drawer_handled": 15, "final_state_verified": 10, "task_complete": 15}
        comps = [e.get("progression_components") or {} for e in eps]
        if any(comps):
            parts = [f"{k} {sum(c.get(k, 0) for c in comps) / n:.1f}/{m}" for k, m in MAXPTS.items()]
            lines += ["", "Score detail (average points per episode / max):", "  " + " | ".join(parts[:4]), "  " + " | ".join(parts[4:]),
                      "A complete success = the final state is right; the score also counts the steps (e.g. verifying the final state)."]
        fails = Counter(e["failure"] for e in eps if not e["success"])
        if fails:
            lines += ["", "Failures:"] + [f"- {f}: {c}   ({HINTS.get(f, '')})" for f, c in fails.most_common()]
        if crashed:
            lines += ["", f"WARNING: brain.py raised an exception in {crashed} episode(s): the Python error is in {run.relative_to(ROOT)}/episodes_part*.log"]
    if videos:
        lines += ["", "Videos:"] + [f"  {Path(v).relative_to(ROOT)}" for v in videos]
    lines += ["", f"Details (trace of every decision and skill call): {run.relative_to(ROOT)}/episodes/",
              f"Took {(time.time() - t0) / 60:.1f} min"]
    text = "\n".join(lines)
    (run / "report.txt").write_text(text + "\n")
    latest = ROOT / "results" / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run.name)
    print("\n" + text, flush=True)
    if not complete:
        print("EVALUATION INCOMPLETE: no valid leaderboard score; see eval.log", flush=True)
        raise SystemExit(1)
    timeline.log("evaluate_end", run=run.name, team_hash=th, models=models, episodes=len(eps),
                 success=sum(e["success"] for e in eps), mean_score=round(sum(e["progression"] for e in eps) / max(len(eps), 1), 1))


if __name__ == "__main__":
    main()
