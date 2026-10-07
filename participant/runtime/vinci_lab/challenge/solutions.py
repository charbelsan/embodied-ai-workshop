"""Baseline Brain for the kitting challenge: reads the work order, runs the plan once, blindly.

Returns a trace (list of dicts) and a summary; the grader judges the final state independently.
Your team brain lives in team/brain.py (see ./evaluate.sh).
"""
from __future__ import annotations

import time

from .objects import KEEP, KITTING_TRAY, TOP_DRAWER  # noqa: F401


class _Run:
    def __init__(self, sk, lab, budget_s):
        self.sk, self.lab, self.budget_s = sk, lab, budget_s
        self.trace, self.calls, self.fails, self.t0 = [], 0, 0, time.time()

    def over_budget(self) -> bool:
        return self.lab.steps / 60.0 > self.budget_s

    def decide(self, text):
        self.trace.append({"type": "decision", "sim_t": round(self.lab.steps / 60, 1), "text": text})

    def call(self, name, **kw):
        r = getattr(self.sk, name)(**kw)
        self.calls += 1
        self.fails += 0 if r["ok"] else 1
        ev = {k: v for k, v in r.get("evidence", {}).items() if k in ("piece_where", "opening_m", "provider", "piece_pos", "piece_tilt_deg")}
        if "substeps" in r.get("evidence", {}):
            ev["substeps"] = [(s["step"], s["ok"], s.get("drawer_m"), s.get("piece_z")) for s in r["evidence"]["substeps"]]
        self.trace.append({"type": "skill", "sim_t": round(self.lab.steps / 60, 1), "skill": name, "args": kw,
                           "ok": r["ok"], "reason": r["reason"], "evidence": ev})
        return r

    def summary(self):
        return {"skill_calls": self.calls, "failed_calls": self.fails, "sim_time_s": round(self.lab.steps / 60, 1),
                "wall_s": round(time.time() - self.t0, 1), "timed_out": self.over_budget()}


def _plan(lab):
    tray = [p for p, r in lab.rules.items() if r == KITTING_TRAY]
    drawer = [p for p, r in lab.rules.items() if r == TOP_DRAWER]
    return tray, drawer


def baseline_solution(sk, lab, budget_s=240, open_provider="rl"):
    R = _Run(sk, lab, budget_s)
    tray, drawer = _plan(lab)
    R.decide(f"plan (blind): tray {tray}, open drawer ({open_provider}), drawer {drawer}, close")
    for p in tray:
        R.call("pick_and_place", obj=p, target=KITTING_TRAY)
    if drawer:
        R.call("open_drawer", provider=open_provider)
        for p in drawer:
            R.call("pick_and_place", obj=p, target=TOP_DRAWER)
    R.call("close_drawer")
    return R.trace, R.summary()
