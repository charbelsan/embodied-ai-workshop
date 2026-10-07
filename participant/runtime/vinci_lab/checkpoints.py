"""Checkpoint registry: which model file a skill runs, identified by its path AND its sha256.

The team selects a model in team/config.yaml (written by `./train.sh select <run>`):

    checkpoints:
      open_drawer_student:
        path: results/train_dagger_20260929_210000/policy/open_drawer_student.pt
        sha256: 3f5c...

Without a selection, the reference model shipped with the workshop is used. The selection is re-read at every call:
when the selected sha256 changes, the model is reloaded (never a silent cache of an older model), and a file whose
content no longer matches its declared sha256 is refused. Every skill reports the sha256 it actually ran.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parent.parent / "policies"
REFERENCE = {
    "open_drawer_rl": POLICY_DIR / "open_drawer_rl.pt",            # RL demonstration of the guided lab (privileged inputs)
    "open_drawer_student": POLICY_DIR / "open_drawer_visual.pt",   # DAgger vision student (RGB-D + proprio)
    "dagger_teacher": POLICY_DIR / "dagger_teacher.pt",            # privileged teacher that labels the DAgger data
}
_hash_cache: dict = {}
_models: dict = {}


class CheckpointError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _hash_cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _hash_cache[key] = h.hexdigest()
    return _hash_cache[key]


def workspace() -> Path:
    """The team workspace: $VINCI_WORKSPACE, else WS_DIR of /etc/vinci/workshop.env (the simulator service), else ~/vinci-workshop."""
    if os.environ.get("VINCI_WORKSPACE"):
        return Path(os.environ["VINCI_WORKSPACE"])
    try:
        for line in open("/etc/vinci/workshop.env"):
            if line.startswith("WS_DIR="):
                return Path(line.split("=", 1)[1].strip().strip('"'))
    except OSError:
        pass
    return Path.home() / "vinci-workshop"


def team_config() -> dict:
    import yaml

    p = Path(os.environ.get("VINCI_TEAM_DIR", workspace() / "team")) / "config.yaml"
    try:
        return yaml.safe_load(open(p)) or {}
    except FileNotFoundError:
        return {}


def selected(name: str) -> dict:
    """{name, path, sha256, source}: the team's selection if any, else the shipped reference."""
    sel = (team_config().get("checkpoints") or {}).get(name)
    if sel:
        path = Path(sel["path"])
        if not path.is_absolute():
            path = workspace() / path
        if not path.exists():
            raise CheckpointError(f"{name}: selected file {path} does not exist (./train.sh select ...)")
        actual = sha256(path)
        want = str(sel.get("sha256", "")).strip()
        if want and actual != want:
            raise CheckpointError(f"{name}: {path} has sha256 {actual[:12]}..., config.yaml expects {want[:12]}...: refused")
        return {"name": name, "path": str(path), "sha256": actual, "source": "team selection"}
    ref = REFERENCE[name]
    if not ref.exists():
        raise CheckpointError(f"{name}: reference model {ref} missing")
    return {"name": name, "path": str(ref), "sha256": sha256(ref), "source": "workshop reference"}


def load(name: str, loader):
    """(model, info). `loader(path)` builds the model; it is called again whenever the selected sha256 changes."""
    info = selected(name)
    cached = _models.get(name)
    if cached is None or cached[0] != info["sha256"]:
        _models[name] = (info["sha256"], loader(info["path"]))
        info["reloaded"] = True
    else:
        info["reloaded"] = False
    return _models[name][1], info
