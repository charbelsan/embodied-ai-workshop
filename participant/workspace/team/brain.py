"""YOUR BRAIN: the logic that decides what the robot does, and when.

./evaluate.sh plays DEV episodes with this file and prints your score.
./submit.sh sends it (with skills.py and config.yaml) to the organizers for the final evaluation.

This starter brain is the baseline: it reads the work order, then runs the plan once, blindly.
It already succeeds often, but not always. Your job: make the system as robust as you can.
Run ./evaluate.sh, look at the failures in results/, change something, run again.
"""
import skills  # your own reusable skills: team/skills.py


def solve(tools):
    cfg = tools.config
    retries = int(cfg.get("max_retries", 0))
    verify = bool((cfg.get("verification") or {}).get("enabled", False))

    # 1. Read the work order: which piece goes where ("kitting_tray", "top_drawer" or "keep" = don't touch).
    rules = tools.rules()
    tray = [p for p, dest in rules.items() if dest == "kitting_tray"]
    drawer = [p for p, dest in rules.items() if dest == "top_drawer"]
    tools.decide(f"plan: tray {tray}, then open the drawer, drawer {drawer}, then close it")

    # 2. Pieces for the kitting tray.
    for piece in tray:
        skills.place(tools, piece, "kitting_tray", retries=retries, verify=verify)

    # 3. Pieces for the top drawer: open it first (it must be open >= 18 cm).
    if drawer:
        tools.call("open_drawer")  # provider comes from config.yaml
        for piece in drawer:
            skills.place(tools, piece, "top_drawer", retries=retries, verify=verify)

    # 4. Leave the workstation in its final state.
    tools.call("close_drawer")
