"""Append-only timeline of what a team did (results/timeline.jsonl): used to measure the participant experience.

    python <runtime>/lib/timeline.py <event> [key=value ...]
"""
import json
import os
import sys
import time
from pathlib import Path

WORKSPACE = Path(os.environ.get("VINCI_WORKSPACE", Path.home() / "vinci-workshop"))
FILE = WORKSPACE / "results" / "timeline.jsonl"


def log(event: str, **kw) -> None:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    rec = {"event": event, "t": round(time.time(), 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}
    with open(FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")


def seen(event: str) -> bool:
    return FILE.exists() and any(json.loads(l).get("event") == event for l in open(FILE) if l.strip())


if __name__ == "__main__":
    log(sys.argv[1], **dict(a.split("=", 1) for a in sys.argv[2:]))
