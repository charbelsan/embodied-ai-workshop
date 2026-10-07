"""score-dev: run N DEV episodes with your solution and show success + the 0-100 progression score (per component).

    python scripts/score_dev.py --solution my_brain.py:solve --episodes 20 [--procs 4] [--seed0 0]
Runs challenge_eval.py in parallel processes (CPU physics), then prints one table. The final evaluation runs on the organizers' side.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter

p = argparse.ArgumentParser()
p.add_argument("--solution", default="baseline")
p.add_argument("--episodes", type=int, default=20)
p.add_argument("--procs", type=int, default=4)
p.add_argument("--seed0", type=int, default=0)
p.add_argument("--level", default="dev")
p.add_argument("--out", default=None)
a = p.parse_args()
if a.level not in ("training", "dev"):
    sys.exit("score-dev runs on training/dev only; the final evaluation is played by the organizers")
out = a.out or f"runs/score_dev_{os.path.basename(a.solution).replace(':', '_')}_{int(time.time())}"
here = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
per = [a.episodes // a.procs + (1 if i < a.episodes % a.procs else 0) for i in range(a.procs)]
procs, s0 = [], a.seed0
for k, n in enumerate(per):
    if n == 0:
        continue
    cmd = [sys.executable, os.path.join(here, "challenge_eval.py"), "--headless", "--device", "cpu", "--sim_device", "cpu",
           "--level", a.level, "--solution", a.solution, "--episodes", str(n), "--seed0", str(s0), "--out", f"{out}/part{k}"]
    procs.append(subprocess.Popen(cmd, stdout=open(f"{out}_part{k}.log", "w") if os.makedirs(out, exist_ok=True) is None else None,
                                  stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}))
    s0 += n
for pr in procs:
    pr.wait()
eps = [json.loads(l) for f in sorted(glob.glob(f"{out}/part*/episodes.jsonl")) for l in open(f)]
if not eps:
    sys.exit(f"no episodes produced, see {out}_part*.log")
n = len(eps)
succ = sum(e["success"] for e in eps)
comp = Counter()
for e in eps:
    comp.update(e["progression_components"])
print(f"\n{a.solution} on {a.level}: {succ}/{n} complete successes, mean progression {sum(e['progression'] for e in eps)/n:.1f}/100  ({(time.time()-t0)/60:.1f} min)")
print(f"{'component':26s} mean/max")
for k, m in (("identified", 10), ("grasped", 15), ("placed_correctly", 25), ("wrong_placement_avoided", 10), ("drawer_handled", 15), ("final_state_verified", 10), ("task_complete", 15)):
    print(f"  {k:24s} {comp[k]/n:5.1f} / {m}")
print("failure modes:", dict(Counter(e["failure"] for e in eps if not e["success"])))
print(f"{'seed':>8s} {'ok':>3s} {'prog':>5s}  failure")
for e in eps:
    print(f"{e['initial']['seed']:8d} {'yes' if e['success'] else 'no':>3s} {e['progression']:5.1f}  {e['failure'] or ''}")
print("details:", out)
