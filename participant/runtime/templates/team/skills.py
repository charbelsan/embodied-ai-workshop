"""YOUR SKILLS: reusable building blocks for brain.py. Add as many as you want.

Every skill of the robot is called with tools.call(name, **args) and returns
    {"ok": bool, "reason": str, "duration_s": float, "evidence": {...}}
Available skills: inspect_scene, open_drawer, pick_and_place, close_drawer, move_to, go_pick_start,
clear_of_drawer, stop. Low-level API for new gestures: tools.lab (observe, move_to, gripper, ...).
"""


def where_is(tools, piece):
    """Where the piece is now: table | top_drawer | kitting_tray | gripper | floor | elsewhere."""
    scene = tools.inspect()["evidence"]
    return scene["pieces"][piece]["where"]


def place(tools, piece, target, retries=0, verify=False):
    """Pick `piece` and put it in `target`. Optionally check the result and repeat the same call."""
    for attempt in range(1 + retries):
        r = tools.call("pick_and_place", obj=piece, target=target)
        ok = r["ok"]
        if verify:
            ok = where_is(tools, piece) == target
        if ok or tools.time_left() < 30:
            return ok
        tools.decide(f"{piece}: attempt {attempt + 1} failed ({r['reason']})")
    return False
