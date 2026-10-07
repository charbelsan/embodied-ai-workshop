"""The robot, as tools for your coding agent (Claude Code, Codex): an MCP server over the workshop simulators.

    claude mcp list          # .mcp.json of the workspace declares it as "vinci-robot"
    codex mcp list           # ~/.codex/config.toml declares it too

Tools (and nothing else):
  observe(scene)     the two camera images (front, wrist) + the robot's own joint state. No object pose, no measurement
                     of the drawer or of the parts: judging the scene from the images is your job.
  open_drawer()      learned skill: the DAgger vision student (RGB-D + joints) opens the top drawer. It reports that it
                     ran and which checkpoint (sha256) it ran, not whether the drawer is open.
  close_drawer()     declared controller (scripted, it uses the handle pose): pushes the top drawer closed.
  insert_ram()       learned skill of the PC scene: RAM stick already in the gripper -> visual insertion in its slot,
                     then release and a look pose above the slot (no re-grasp).
Every call is appended to results/mcp_calls.jsonl (tool, arguments, result, time) so a run can be replayed and audited.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Image

SCENES = {"cabinet": os.environ.get("VINCI_CABINET_URL", "http://127.0.0.1:8765"),
          "pc": os.environ.get("VINCI_PC_URL", "http://127.0.0.1:8766")}
WS = Path(os.environ.get("VINCI_WORKSPACE", Path.home() / "vinci-workshop"))
LOG = Path(os.environ.get("VINCI_MCP_LOG", WS / "results" / "mcp_calls.jsonl"))

mcp = FastMCP("vinci-robot", instructions=(
    "Tools of a simulated Franka robot arm. observe() returns camera images and the robot's joint state only; "
    "skills report that they ran, not whether they succeeded: look at the images to verify, and decide what to do next."))


def _log(tool: str, args: dict, out) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as f:
            f.write(json.dumps({"t": round(time.time(), 2), "tool": tool, "args": args, "result": out}, default=str) + "\n")
    except OSError:
        pass


def _post(scene: str, path: str, body: dict | None = None, timeout: float = 900) -> dict:
    if scene not in SCENES:
        raise ValueError(f"unknown scene {scene!r}: {list(SCENES)}")
    req = urllib.request.Request(SCENES[scene] + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        raise RuntimeError(f"the {scene} simulator is not answering ({SCENES[scene]}): run ./start.sh") from e


def _skill(scene: str, name: str, **args) -> dict:
    return _post(scene, "/skill", {"name": name, "args": args})


@mcp.tool()
def observe(scene: str = "cabinet") -> list:
    """Front and wrist camera images of the scene, then the robot's joint state (JSON). scene: 'cabinet' or 'pc'."""
    o = _post(scene, "/api/observe_allowed")
    state = {k: o[k] for k in ("joint_pos", "joint_vel", "gripper_width_m", "ee_pos", "ee_quat", "sim_time_s") if k in o}
    _log("observe", {"scene": scene}, {"state": state, "images": ["front", "wrist"]})
    return ["front camera:", Image(data=base64.b64decode(o["rgb_front"]), format="png"),
            "wrist camera:", Image(data=base64.b64decode(o["rgb_wrist"]), format="png"),
            "robot state: " + json.dumps(state)]


@mcp.tool()
def open_drawer() -> str:
    """Learned skill (DAgger vision student): open the top drawer of the cabinet. Returns whether it RAN and the
    checkpoint sha256 it ran; it does not tell whether the drawer is open (use observe)."""
    r = _skill("cabinet", "open_drawer_student")
    ev = r.get("evidence", {})
    out = {"ran": bool(r.get("ok")), "message": r.get("reason"),
           **{k: ev[k] for k in ("checkpoint_sha256", "checkpoint_source", "policy_steps", "observations") if k in ev}}
    _log("open_drawer", {}, out)
    return json.dumps(out)


@mcp.tool()
def close_drawer() -> str:
    """Declared controller (scripted, uses the handle pose): push the top drawer closed. Returns whether it RAN."""
    r = _skill("cabinet", "close_drawer")
    ran = "reason" in r  # the controller answered; its own verdict reads the drawer's true state, so it is not forwarded
    out = {"ran": ran, "message": "scripted push done; check the result with observe()",
           "controller": "declared, uses the handle pose"}
    _log("close_drawer", {}, out)
    return json.dumps(out)


@mcp.tool()
def insert_ram() -> str:
    """Learned skill of the PC scene: the RAM stick already held by the gripper is inserted into its slot from the
    cameras. At the end the gripper releases the stick and the hand climbs straight up so the wrist camera looks down on
    the slot: verify with observe(scene='pc'). Returns whether it RAN and the checkpoint sha256. There is no re-grasp: a
    second call with an empty gripper returns a technical error."""
    r = _skill("pc", "insert_ram_student")
    ev = r.get("evidence", {})
    out = {"ran": bool(r.get("ok")), "message": r.get("reason"),
           **{k: ev[k] for k in ("checkpoint_sha256", "checkpoint_source", "policy_steps", "observations") if k in ev}}
    _log("insert_ram", {}, out)
    return json.dumps(out)


if __name__ == "__main__":
    mcp.run()
