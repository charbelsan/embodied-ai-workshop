"""The Brain's `tools` object (same API as the GPU workshop's team_entry.py) and the team loader."""
from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from pathlib import Path

import yaml

PRIMITIVES = ["inspect_scene", "open_drawer", "pick_and_place", "close_drawer", "move_to", "go_pick_start",
              "clear_of_drawer", "stop"]
TASK = {"name": "kitting_cabinet", "title": "VINCI kitting workstation (CPU edition)", "primitives": PRIMITIVES}


def load_config(team_dir: Path) -> dict:
    return yaml.safe_load(open(team_dir / "config.yaml")) or {}


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

    def __init__(self, sk, lab, budget_s: float, config: dict):
        self.sk, self.lab, self.budget_s, self.config, self.task = sk, lab, budget_s, config, TASK
        self.trace, self.calls, self.fails, self.t0 = [], 0, 0, time.time()
        self.providers = dict(config.get("providers") or {})

    def sim_time(self) -> float:
        return self.lab.steps / 60.0

    def time_left(self) -> float:
        return self.budget_s - self.sim_time()

    def over_budget(self) -> bool:
        return self.time_left() < 0

    def decide(self, text: str) -> None:
        self.trace.append({"type": "decision", "sim_t": round(self.sim_time(), 1), "text": str(text)})

    def call(self, name: str, **args) -> dict:
        if name.startswith("_") or name not in PRIMITIVES:
            raise ValueError(f"{name!r} is not a skill of task {TASK['name']} (available: {PRIMITIVES})")
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


def load_brain(team_dir: Path):
    sys.path.insert(0, str(team_dir))  # so brain.py can `import skills` (team/skills.py)
    sys.modules.pop("skills", None)
    spec = importlib.util.spec_from_file_location("team_brain", team_dir / "brain.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def solve(brain, sk, lab, budget_s: float, config: dict):
    tools = Tools(sk, lab, budget_s, config)
    try:
        brain.solve(tools)
    except Exception as e:  # a crashing brain still gets its final state graded
        tools.decide(f"BRAIN CRASHED: {type(e).__name__}: {e}")
        traceback.print_exc()
    return tools.trace, tools.summary()
