"""Single grader (same as the GPU workshop): judges the FINAL workstation state and the steps of the task."""
from __future__ import annotations

import math

from .objects import KEEP, PIECES

DRAWER_CLOSED_M = 0.02
DISTRACTOR_TOL_M = 0.02
FAILURE_ORDER = ["object_on_floor", "distractor_moved", "grasp_failed", "dropped_in_transport", "wrong_destination",
                 "drawer_not_closed", "timeout"]


def progression(lab, pieces: dict, closed: bool, distractor_ok: bool, success: bool) -> dict:
    """0-100 progression score from the steps of the task (tracked during the episode), not only the final state."""
    ev = lab.events
    targets = [p for p, r in lab.rules.items() if r != KEEP]
    need_drawer = any(lab.rules[p] == "top_drawer" for p in targets)
    n = max(len(targets), 1)
    comp = {
        "identified": 10 * ev["inspected"],
        "grasped": round(15 * sum(ev["grasped"].get(p, False) for p in targets) / n, 1),
        "placed_correctly": round(25 * sum(pieces[p]["ok"] for p in targets) / n, 1),
        "wrong_placement_avoided": 10 * (distractor_ok and not any(ev["wrong_place"].get(p, False) for p in targets)),
        "drawer_handled": 7.5 * ((not need_drawer) or ev["drawer_opened"]) + 7.5 * closed,
        "final_state_verified": 10 * ev["verified_after_last_action"],
        "task_complete": 15 * success,
    }
    return {"components": comp, "score": round(sum(comp.values()), 1)}


def grade(lab, trace_summary: dict | None = None) -> dict:
    """lab: CpuLab after the solution finished (+1 s settle). trace_summary: optional {timed_out, skill_calls, ...}."""
    lab.hold(60)  # 1 s settle before judging
    pieces, n_ok, floor = {}, 0, 0
    distractor_ok = True
    for pid, rule in lab.rules.items():
        where = lab.where(pid)
        pos = [float(v) for v in lab.piece_pos(pid)]
        if rule == KEEP:
            moved = math.dist(pos[:2], lab.initial["piece_pos"][pid][:2])
            ok = moved <= DISTRACTOR_TOL_M and where == "table"
            distractor_ok &= ok
            pieces[pid] = {"expected": "untouched", "where": where, "moved_cm": round(100 * moved, 1), "ok": ok}
        else:
            ok = where == rule
            n_ok += ok
            pieces[pid] = {"expected": rule, "where": where, "ok": ok}
        floor += where == "floor"
    opening = lab.drawer_opening()
    closed = opening < DRAWER_CLOSED_M
    n_targets = sum(1 for r in lab.rules.values() if r != KEEP)
    success = n_ok == n_targets and closed and distractor_ok and floor == 0
    score = 20 * n_ok + 20 * closed + 10 * distractor_ok + 10 * (floor == 0) - 10 * floor
    # failure category (first match in priority order)
    cats = []
    if floor:
        cats.append("object_on_floor")
    if not distractor_ok:
        cats.append("distractor_moved")
    for pid, st in pieces.items():
        if st["expected"] == "untouched" or st["ok"]:
            continue
        if st["where"] == "table":
            cats.append("grasp_failed")
        elif st["where"] in ("elsewhere", "gripper"):
            cats.append("dropped_in_transport")
        elif st["where"] != "floor":
            cats.append("wrong_destination")
    if not closed:
        cats.append("drawer_not_closed")
    if trace_summary and trace_summary.get("timed_out"):
        cats.append("timeout")
    cats = sorted(set(cats), key=FAILURE_ORDER.index)
    prog = progression(lab, pieces, closed, distractor_ok, success)
    return {"success": bool(success), "score": int(score), "max_score": 20 * n_targets + 40,
            "progression": prog["score"], "progression_components": prog["components"],
            "pieces": pieces, "drawer_opening_m": round(opening, 3), "drawer_closed": bool(closed),
            "distractor_intact": bool(distractor_ok), "objects_on_floor": int(floor),
            "failure": cats[0] if cats else None, "failure_all": cats,
            "level": lab.level["name"], "initial": lab.initial, "perturbations_triggered": list(lab.perturb_log),
            **(trace_summary or {})}
