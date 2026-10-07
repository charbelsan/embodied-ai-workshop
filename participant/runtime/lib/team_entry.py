"""Bridge between the team workspace (team/brain.py + team/config.yaml) and the challenge harness.

The harness (scripts/challenge_eval.py of the workshop runtime) calls `solve(sk, lab, budget_s, open_provider)` once per episode.
This module loads team/config.yaml and team/brain.py, hands the brain a `Tools` object, and returns
(trace, summary) as the harness expects. Participants never need to edit this file.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import yaml

RUNTIME = Path(os.environ.get("VINCI_RUNTIME", Path(__file__).resolve().parents[1]))  # <runtime>/lib/team_entry.py
WORKSPACE = Path(os.environ.get("VINCI_WORKSPACE", Path.home() / "vinci-workshop"))
TEAM_DIR = Path(os.environ.get("VINCI_TEAM_DIR", WORKSPACE / "team"))
TASKS_DIR = RUNTIME / "tasks"


def load_config(team_dir: Path = TEAM_DIR) -> dict:
    return yaml.safe_load(open(team_dir / "config.yaml")) or {}


def load_task(name: str) -> dict:
    path = TASKS_DIR / f"{name}.yaml"
    if not path.exists():
        known = sorted(p.stem for p in TASKS_DIR.glob("*.yaml"))
        raise ValueError(f"unknown task {name!r} in team/config.yaml (known: {known})")
    return {"name": name, **yaml.safe_load(open(path))}


class Tools:
    """What the brain can use. Every skill returns {ok, reason, duration_s, evidence}.

    tools.call(name, **args)   run a skill (providers from config.yaml are filled in if you don't pass one)
    tools.inspect()            = tools.call("inspect_scene")
    tools.rules()              the work order: {piece: destination}
    tools.decide(text)         write a line in the episode trace (shows up in results/)
    tools.time_left()          simulated seconds left in this episode
    tools.lab                  low-level API (observe, move_to, gripper, ...) to build your own skills
    tools.config               team/config.yaml as a dict
    """

    def __init__(self, sk, lab, budget_s: float, config: dict, task: dict):
        self.sk, self.lab, self.budget_s, self.config, self.task = sk, lab, budget_s, config, task
        self.trace, self.calls, self.fails, self.t0 = [], 0, 0, time.time()
        self.providers = dict(config.get("providers") or {})

    # --- time
    def sim_time(self) -> float:
        return self.lab.steps / 60.0

    def time_left(self) -> float:
        return self.budget_s - self.sim_time()

    def over_budget(self) -> bool:
        return self.time_left() < 0

    # --- trace
    def decide(self, text: str) -> None:
        self.trace.append({"type": "decision", "sim_t": round(self.sim_time(), 1), "text": str(text)})

    # --- skills
    def call(self, name: str, **args) -> dict:
        if name.startswith("_") or name not in self.task["primitives"]:
            raise ValueError(f"{name!r} is not a skill of task {self.task['name']} (available: {self.task['primitives']})")
        if "provider" not in args and name in self.providers:
            args["provider"] = self.providers[name]
        r = getattr(self.sk, name)(**args)
        self.calls += 1
        self.fails += 0 if r["ok"] else 1
        keep = ("piece_where", "opening_m", "provider", "piece_pos", "piece_tilt_deg")
        ev = {k: v for k, v in (r.get("evidence") or {}).items() if k in keep}
        self.trace.append({"type": "skill", "sim_t": round(self.sim_time(), 1), "skill": name, "args": args,
                           "ok": r["ok"], "reason": r["reason"], "evidence": ev})
        return r

    def inspect(self) -> dict:
        return self.call("inspect_scene")

    def rules(self) -> dict:
        return dict(self.lab.rules)

    def summary(self) -> dict:
        return {"skill_calls": self.calls, "failed_calls": self.fails, "sim_time_s": round(self.sim_time(), 1),
                "wall_s": round(time.time() - self.t0, 1), "timed_out": self.over_budget()}


_BRAIN = None


def _load_brain():
    global _BRAIN
    if _BRAIN is None:
        sys.path.insert(0, str(TEAM_DIR))  # so brain.py can `import skills` (team/skills.py)
        spec = importlib.util.spec_from_file_location("team_brain", TEAM_DIR / "brain.py")
        _BRAIN = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_BRAIN)
    return _BRAIN


def solve(sk, lab, budget_s=240, open_provider="rl"):
    """Harness entry point (signature fixed by challenge_eval.py)."""
    config = load_config()
    task = load_task(config.get("task", "kitting_cabinet"))
    tools = Tools(sk, lab, budget_s, config, task)
    try:
        _load_brain().solve(tools)
    except Exception as e:  # a crashing brain still gets its final state graded
        tools.decide(f"BRAIN CRASHED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    return tools.trace, tools.summary()
