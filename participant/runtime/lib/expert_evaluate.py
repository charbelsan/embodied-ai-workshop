"""Expert mission measurement. No agent claims, no points for inspection.
The operator runs this trusted harness; arbitrary participant Python is not imported.
Not a security sandbox against the administrator of the VM.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import signal
import time
import uuid


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode(lab, provider, expected_sha):
    """One learned opening, then fixed composition. Never rescue with a different policy."""
    lab.reset()
    initial = lab.success()
    r = lab.skill("open_drawer_student") if provider == "visual" else lab.skill("open_drawer", provider="rl")
    actual = r.get("evidence", {}).get("checkpoint_sha256")
    if actual != expected_sha:
        raise RuntimeError("Executed checkpoint differs from the frozen selection; no valid result")
    opened = lab.success()
    opening = float(opened["drawer_opening_m"])
    calls = [{"skill": "open_drawer", "provider": provider, "result": r}]
    # This gate belongs to the evaluator, not to a participant perception model.
    # A failed opening is counted as such and never hidden by another controller.
    if opening > 0.20:
        ready = lab.skill("go_pick_start")
        calls.append({"skill": "go_pick_start", "result": ready})
        if ready.get("ok"):
            placed = lab.skill("pick_and_place", obj="cube", target="top_drawer")
            calls.append({"skill": "pick_and_place", "result": placed})
            if lab.success().get("cube_in_drawer"):
                calls.append({"skill": "close_drawer", "result": lab.skill("close_drawer")})
    final = lab.success()
    return {"initial": initial, "opening_m": opening, "opening_success": opening > 0.20,
            "mission_success": bool(opening > 0.20 and final.get("cube_in_drawer") and final.get("drawer_closed")),
            "final": final, "checkpoint_sha256": actual, "calls": calls}


@contextlib.contextmanager
def selection(config_path, model_name, reference):
    import yaml
    original = config_path.read_bytes()
    cfg = yaml.safe_load(original) or {}
    if reference:
        (cfg.get("checkpoints") or {}).pop(model_name, None)
        changed = yaml.safe_dump(cfg, sort_keys=False).encode()
    else:
        if not (cfg.get("checkpoints") or {}).get(model_name):
            raise RuntimeError("No personal checkpoint selected. Use ./train.sh select results/<run>, or --reference")
        changed = original
    if changed != original:
        config_path.write_bytes(changed)
    try:
        yield changed
    finally:
        # Never overwrite a concurrent participant edit; save the old selection for recovery.
        if config_path.read_bytes() == changed:
            if changed != original:
                config_path.write_bytes(original)
        else:
            backup = config_path.with_name("config.before_expert_" + uuid.uuid4().hex[:8] + ".yaml")
            backup.write_bytes(original)
            raise RuntimeError(f"Configuration changed during evaluation; result invalid. Previous config: {backup}")


def main():
    p = argparse.ArgumentParser(description="Measure learned opening + cube placement + closure. Watch Isaac Sim.")
    p.add_argument("command", choices=["evaluate"])
    p.add_argument("--provider", choices=["visual", "rl"], default="visual")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--reference", action="store_true", help="temporarily measure shipped reference; restore team selection")
    p.add_argument("--label", default="experiment")
    a = p.parse_args()
    def interrupted(signum, frame):
        raise SystemExit(f"Interrupted by signal {signum}; measurement invalid")
    signal.signal(signal.SIGTERM, interrupted)
    if not 1 <= a.episodes <= 30:
        p.error("episodes must be between 1 and 30")
    ws = Path(os.environ.get("VINCI_WORKSPACE", Path.home() / "vinci-workshop"))
    rt = Path(os.environ.get("VINCI_RUNTIME", "/opt/vinci-workshop"))
    sys.path.insert(0, str(rt))
    from vinci_lab.client import Lab
    from vinci_lab import checkpoints
    name = "open_drawer_student" if a.provider == "visual" else "open_drawer_rl"
    run = ws / "results" / "expert" / (time.strftime("measure_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
    run.mkdir(parents=True)
    lock = open(ws / ".evaluation.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another evaluation is running. Wait before starting this one.")
    report = {"status": "incomplete", "label": a.label, "provider": a.provider, "reference": a.reference,
              "requested_episodes": a.episodes, "episodes": [], "protocol": "expert-learning-02",
              "note": "Live scene resets; not paired seeded trials. No hidden benchmark, no leaderboard."}
    (run / "report.json").write_text(json.dumps(report, indent=2))
    try:
        with selection(ws / "team/config.yaml", name, a.reference) as cfg:
            info = checkpoints.selected(name)
            frozen_sha = sha(info["path"])
            report["model"] = info
            (run / "config.yaml").write_bytes(cfg)
            print(f"MODEL {info['source']} | {info['sha256']} | {info['path']}", flush=True)
            print("Watch Isaac Sim. Do not run another notebook, reset, training or agent on this scene during measurement.", flush=True)
            lab = Lab(timeout=300)
            for i in range(a.episodes):
                if (ws / "team/config.yaml").read_bytes() != cfg or sha(info["path"]) != frozen_sha:
                    raise RuntimeError("Model or configuration changed during measurement")
                t0 = time.monotonic()
                row = episode(lab, a.provider, frozen_sha)
                row.update(index=i, wall_s=round(time.monotonic()-t0, 2))
                report["episodes"].append(row)
                (run / "report.json").write_text(json.dumps(report, indent=2))
                print(f"EPISODE {i+1}/{a.episodes}: opening={row['opening_m']:.3f}m | learned opening={row['opening_success']} | mission={row['mission_success']}", flush=True)
            if sha(info["path"]) != frozen_sha:
                raise RuntimeError("Model changed during measurement")
        report.update(status="complete", opening_successes=sum(r["opening_success"] for r in report["episodes"]),
                      mission_successes=sum(r["mission_success"] for r in report["episodes"]))
        print(f"COMPLETE: learned opening {report['opening_successes']}/{a.episodes}; mission {report['mission_successes']}/{a.episodes}", flush=True)
    except BaseException as e:
        report.update(status="invalid", error=str(e))
        raise
    finally:
        (run / "report.json").write_text(json.dumps(report, indent=2))
        print(f"REPORT {run / 'report.json'}", flush=True)
        lock.close()


if __name__ == "__main__":
    main()
