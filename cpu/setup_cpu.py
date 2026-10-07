"""Prepare the CPU edition (Linux, macOS, Windows): robot model + team workspace. Run it once, after installing the requirements:

    python -m pip install -r cpu/requirements.txt
    python cpu/setup_cpu.py

Downloads the Franka Emika Panda model of MuJoCo Menagerie (Apache-2.0, pinned commit, 80 files, ~34 MB) into cpu/.cache/,
and creates your team workspace cpu/team/ from the workshop's starter team (drawer opening set to the CPU provider).
Safe to run again: existing files are kept.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MENAGERIE_REPO = "google-deepmind/mujoco_menagerie"
MENAGERIE_COMMIT = "f054586a8e90465d49ee5be15335c4a0c7f57caf"
DEST = HERE / ".cache" / "mujoco_menagerie"


def fetch_robot() -> None:
    panda = DEST / "franka_emika_panda" / "panda.xml"
    if panda.exists():
        print(f"robot model: already in {panda.parent}")
        return
    url = f"https://api.github.com/repos/{MENAGERIE_REPO}/git/trees/{MENAGERIE_COMMIT}?recursive=1"
    tree = json.load(urllib.request.urlopen(url, timeout=60))
    files = [t["path"] for t in tree["tree"] if t["type"] == "blob" and t["path"].startswith("franka_emika_panda/")]
    print(f"robot model: downloading {len(files)} files of MuJoCo Menagerie ({MENAGERIE_COMMIT[:7]}) ...", flush=True)
    for i, path in enumerate(files, 1):
        out = DEST / path
        out.parent.mkdir(parents=True, exist_ok=True)
        raw = f"https://raw.githubusercontent.com/{MENAGERIE_REPO}/{MENAGERIE_COMMIT}/{path}"
        with urllib.request.urlopen(raw, timeout=120) as r, open(out, "wb") as f:
            shutil.copyfileobj(r, f)
        if i % 20 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", flush=True)


def make_team() -> None:
    team = HERE / "team"
    if team.exists():
        print(f"team workspace: kept {team} (delete it to start again from the starter team)")
        return
    starter = HERE.parent / "participant" / "workspace" / "team"
    shutil.copytree(starter, team, ignore=shutil.ignore_patterns("__pycache__"))
    cfg = team / "config.yaml"
    text = re.sub(r"(?m)^(\s*open_drawer:\s*)\w+", r"\1analytic", cfg.read_text())
    cfg.write_text(text.replace("# rl: trained RL policy", "# CPU edition: analytic (rl and visual need the GPU edition). rl: trained RL policy", 1))
    print(f"team workspace: {team} (starter brain, drawer opening = analytic)")


def check() -> None:
    try:
        import mujoco  # noqa: F401
        import numpy  # noqa: F401
        import yaml  # noqa: F401
    except ImportError as e:
        sys.exit(f"missing package ({e.name}): first run  python -m pip install -r {HERE / 'requirements.txt'}")


if __name__ == "__main__":
    check()
    fetch_robot()
    make_team()
    print("\nReady. Next:\n  python cpu/evaluate.py            # 8 DEV episodes, ~1-3 min on a laptop\n"
          "  python cpu/evaluate.py --view     # watch the robot in a window")
