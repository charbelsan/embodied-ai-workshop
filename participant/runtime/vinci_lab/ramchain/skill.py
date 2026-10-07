"""`insert_ram_student`: the learned RAM insertion, EXECUTION ONLY, without any simulator truth.

    from vinci_lab.ramchain.skill import insert_ram_student
    r = insert_ram_student(lab)          # lab: RamLab with its two cameras, ram1 already in the gripper

Contract (same as the drawer's student skill):
  - the policy sees front + wrist RGB-D (64 px) + its 18-D joint state, nothing else;
  - fixed horizon (the training one), then release + vertical retreat from where the hand is, then the hand climbs straight
    up to a fixed height (LOOK_Z) so the wrist camera looks down on the slot (declared controllers, proprio only);
  - it reports that it RAN (or the technical error that stopped it) and which checkpoint (sha256) ran, never whether the
    stick is seated: the grader alone reads `ram_seated`, the caller verifies with the images;
  - checkpoint: argument > $VINCI_RAM_CHECKPOINT > policies/insert_ram_student.json ({path, sha256}); reloaded whenever
    the file's sha256 changes (no silent cache of an old model); a registry sha256 that does not match is an error.
"""
from __future__ import annotations

import json
import os
import time

import torch

from .student import load_checkpoint, sha256_file, student_images

HORIZON = 120  # student steps at 16 Hz (7.5 s), the training/eval horizon of the chain
OBSERVATIONS = "front+wrist RGB-D 64px + joint pos/vel (18-D)"
REGISTRY = os.path.join(os.path.dirname(__file__), "..", "..", "policies", "insert_ram_student.json")
_cache = {"sha256": None, "model": None, "meta": {}}


def resolve_checkpoint(checkpoint: str | None = None) -> tuple[str, str | None, str]:
    """(path, expected sha256 or None, source)."""
    if checkpoint:
        return checkpoint, None, "argument"
    if os.environ.get("VINCI_RAM_CHECKPOINT"):
        return os.environ["VINCI_RAM_CHECKPOINT"], None, "env VINCI_RAM_CHECKPOINT"
    reg = json.load(open(REGISTRY))
    path = os.path.expanduser(reg["path"])
    if not os.path.isabs(path):  # relative to the root that holds policies/ (the workshop runtime)
        path = os.path.normpath(os.path.join(os.path.dirname(REGISTRY), "..", path))
    return path, reg.get("sha256"), "reference (policies/insert_ram_student.json)"


def load_student(checkpoint: str | None = None, device: str | None = None):
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    path, expected, source = resolve_checkpoint(checkpoint)
    sha = sha256_file(path)
    if expected and sha != expected:
        raise ValueError(f"{path}: sha256 {sha[:12]} does not match the registry's {expected[:12]}")
    reloaded = sha != _cache["sha256"]
    if reloaded:
        model, _sha, meta = load_checkpoint(path, device=device)
        _cache.update(model=model, sha256=sha, meta=meta)
    return _cache["model"], {"sha256": sha, "path": path, "source": source, "reloaded": reloaded, "device": device,
                             "meta": _cache.get("meta", {})}


def insert_ram_student(lab, checkpoint: str | None = None, horizon: int = HORIZON, mask_images: bool = False,
                       on_step=None, look_pose: bool = True) -> dict:
    """Run the vision student for `horizon` steps, release, back off. Returns {executed, reason, steps, checkpoint_sha256, ...}.
    `on_step(k, front64, wrist64, prop, action)` is an optional observer (recording), it cannot change the action."""
    t0 = time.time()
    out = {"executed": False, "steps": 0, "checkpoint_sha256": None, "observations": OBSERVATIONS, "images_masked": mask_images}
    if not getattr(lab, "cameras", False):
        return {**out, "reason": "technical error: the simulator runs without cameras", "duration_s": round(time.time() - t0, 2)}
    width = lab.gripper_width()  # finger joints (proprio): a pinched stick holds them ~7.6 mm apart
    if not 0.004 < width < 0.015:
        return {**out, "reason": f"technical error: no stick pinched between the fingers (gripper width {width * 1000:.1f} mm)",
                "duration_s": round(time.time() - t0, 2)}
    try:
        model, ck = load_student(checkpoint)
    except Exception as e:  # noqa: BLE001  missing file / sha256 mismatch: say which
        return {**out, "reason": f"technical error: student model not loaded: {e}", "duration_s": round(time.time() - t0, 2)}
    dev = ck["device"]
    want = ck["meta"].get("wrist_view", "upstream")
    if want != getattr(lab, "wrist_view", "upstream"):
        return {**out, "reason": f"technical error: checkpoint trained with wrist camera '{want}', the simulator has "
                                 f"'{getattr(lab, 'wrist_view', 'upstream')}'", "duration_s": round(time.time() - t0, 2)}
    for _ in range(3):  # fresh frames (the renderer accumulates over frames)
        lab.render()
    with torch.no_grad():
        for k in range(horizon):
            f64, w64 = student_images(*lab.images())
            prop = lab.proprio().float()
            fi, wi = f64.to(dev), w64.to(dev)
            if mask_images:
                fi, wi = torch.zeros_like(fi), torch.zeros_like(wi)
            u = model(fi, wi, prop.to(dev))[0].clamp(-1.0, 1.0).cpu()
            if on_step is not None:
                on_step(k, f64, w64, prop, u)
            lab.step_action(u)
            lab.render()
    lab.release_and_retreat()
    if look_pose:  # declared controller: the wrist camera ends above the slot, the caller can verify on the images
        lab.go_look_pose()
    return {**out, "executed": True, "steps": horizon, "checkpoint_sha256": ck["sha256"], "checkpoint_path": ck["path"],
            "checkpoint_source": ck["source"], "checkpoint_reloaded": ck["reloaded"],
            "reason": f"student executed {horizon} steps, released, backed off" + (" and lifted the hand to look down on the slot"
                                                                              if look_pose else "") + "; check the result with observe()",
            "duration_s": round(time.time() - t0, 2)}
