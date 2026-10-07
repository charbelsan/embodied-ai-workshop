"""Skill server: keeps ONE live Isaac scene and lets a Brain (LLM agent, notebook, or human) call skills over HTTP.

    python scripts/skill_server.py --headless --port 8765        # or without --headless inside DCV to watch

POST /run          {"run_id": "...", "goal": "...", "reset": true}   start a run (trace + video)
POST /decision     {"text": "..."}                                   log the Brain's reasoning / choice
POST /skill        {"name": "open_drawer", "args": {"provider": "rl"}}  execute a skill -> contract dict
POST /intervention {"text": "..."}                                   log a human intervention
POST /end          {"notes": "..."}                                  stop: save video, final success, timings
GET  /observe                                                        inspect_scene() without logging a skill
POST /api/<fn>     low-level participant API: observe (images as base64 PNG), move_to, move_joints, gripper,
                   reset, success   e.g. {"pos": [0.3, 0.3, 0.9]} / {"open": false}

Trace: <runs_dir>/<run_id>/trace.jsonl (+ video.mp4)
"""
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=8765)
p.add_argument("--runs_dir", default="brain_runs")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

from vinci_lab.api import LabEnv
from vinci_lab.skills import Skills
from vinci_lab.student_skills import open_drawer_student

lab = LabEnv()
sk = Skills(lab)
SKILLS = {"inspect_scene": sk.inspect_scene, "move_to": sk.move_to, "open_drawer": sk.open_drawer,
          "pick_and_place": sk.pick_and_place, "close_drawer": sk.close_drawer, "reset": sk.reset,
          "stop": sk.stop, "go_pick_start": sk.go_pick_start,
          # learned skill that executes without simulator truth (reports execution + checkpoint sha256, not success)
          "open_drawer_student": lambda **kw: open_drawer_student(lab, **kw)}
state = {"run": None}


def log(entry):
    run = state["run"]
    entry = {"t_wall": round(time.time() - run["t0"], 2), "sim_time_s": round(lab.steps / 60.0, 2), **entry}
    run["n"] += 1
    with open(os.path.join(run["dir"], "trace.jsonl"), "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def _png_b64(img):
    import base64
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def api_call(fn, body):
    """Participant API over HTTP. Logged into the current run trace when a run is open."""
    if fn == "observe_allowed":  # what a learned skill / a Brain may see: images + the robot's own state, nothing else
        o = lab.observe()
        r = lab.robot
        return {"rgb_front": _png_b64(o["rgb_front"]), "rgb_wrist": _png_b64(o["rgb_wrist"]),
                "joint_pos": o["joint_pos"], "joint_vel": r.data.joint_vel[0].cpu().numpy().round(4).tolist(),
                "gripper_width_m": round(o["gripper_width"], 4), "ee_pos": o["ee_pos"], "ee_quat": o["ee_quat"],
                "sim_time_s": o["sim_time_s"]}
    if fn == "observe":
        o = lab.observe()
        o["rgb_front"], o["rgb_wrist"] = _png_b64(o["rgb_front"]), _png_b64(o["rgb_wrist"])
        return o
    if fn == "success":
        return lab.success()
    calls = {"move_to": lambda: lab.move_to(body["pos"], tuple(body.get("quat", (0.0, 1.0, 0.0, 0.0)))),
             "move_joints": lambda: lab.move_joints(body.get("q")),
             "gripper": lambda: lab.gripper(bool(body.get("open", True))),
             "reset": lambda: lab.reset()}
    if fn not in calls:
        return {"ok": False, "reason": f"unknown api {fn}; known: observe, success, {list(calls)}"}
    if state["run"]:
        log({"type": "api_call", "api": fn, "args": body})
    r = calls[fn]()
    if state["run"]:
        log({"type": "api_result", "api": fn, **r})
    return r


class H(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        b = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/observe":
            return self._send(sk.inspect_scene())
        self._send({"error": "unknown"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path.startswith("/api/"):
            return self._send(api_call(self.path[5:], body))
        if self.path == "/attach_vla":  # {"port": 6200}: connect pick_and_place(provider="vla") to a policy server
            sys.path.insert(0, body.get("vla_track", "vla-track"))
            from vla_pipeline.isaac_vla_client import make_vla_client
            # budget must exceed the demo length (537-596 actions): 450 (client default) truncates successful episodes
            sk.vla_client = make_vla_client(port=int(body.get("port", 6200)), max_actions=int(body.get("max_actions", 750)))
            info = {"ok": True, "reason": f"vla_client attached on port {body.get('port', 6200)}",
                    "checkpoint": body.get("checkpoint", "")}
            if state["run"]:
                log({"type": "provider_attached", **info})
            return self._send(info)
        if self.path == "/run":
            rid = body.get("run_id") or time.strftime("run_%Y%m%d_%H%M%S")
            d = os.path.join(args.runs_dir, rid)
            os.makedirs(d, exist_ok=True)
            state["run"] = {"id": rid, "dir": d, "t0": time.time(), "n": 0, "calls": 0, "fails": 0}
            if body.get("reset", True):
                sk.reset()
            lab.start_recording()
            return self._send(log({"type": "goal", "goal": body.get("goal", ""), "brain": body.get("brain", "participant")}))
        if self.path == "/skill" and state["run"] is None:  # outside a Brain run: execute, no trace
            name, kw = body.get("name"), body.get("args", {}) or {}
            if name not in SKILLS:
                return self._send({"ok": False, "reason": f"unknown skill {name}; known {list(SKILLS)}"}, 400)
            try:
                return self._send(SKILLS[name](**kw))
            except Exception as e:
                return self._send({"ok": False, "reason": f"exception: {e!r}", "duration_s": 0.0, "evidence": {}})
        if state["run"] is None:
            return self._send({"error": "POST /run first"}, 400)
        if self.path == "/decision":
            return self._send(log({"type": "decision", "text": body.get("text", "")}))
        if self.path == "/intervention":
            return self._send(log({"type": "human_intervention", "text": body.get("text", "")}))
        if self.path == "/skill":
            name, kw = body.get("name"), body.get("args", {}) or {}
            if name not in SKILLS:
                return self._send({"ok": False, "reason": f"unknown skill {name}; known {list(SKILLS)}"}, 400)
            log({"type": "skill_call", "skill": name, "args": kw})
            try:
                r = SKILLS[name](**kw)
            except Exception as e:  # a skill crash is a failed skill for the Brain, not a server crash
                r = {"ok": False, "reason": f"exception: {e!r}", "duration_s": 0.0, "evidence": {}}
            state["run"]["calls"] += 1
            state["run"]["fails"] += 0 if r.get("ok") else 1
            log({"type": "skill_result", "skill": name, **r})
            return self._send(r)
        if self.path == "/end":
            run = state["run"]
            video = lab.save_video(os.path.join(run["dir"], "video.mp4"))
            s = lab.success()
            summ = {"type": "end", "success": s, "skill_calls": run["calls"], "failed_calls": run["fails"],
                    "time_to_brain_task_success_s": round(time.time() - run["t0"], 1) if s["success"] else None,
                    "sim_time_s": round(lab.steps / 60.0, 1), "video": video, "notes": body.get("notes", "")}
            out = log(summ)
            state["run"] = None
            return self._send(out)
        self._send({"error": "unknown"}, 404)


srv = HTTPServer(("127.0.0.1", args.port), H)
srv.timeout = 0.05
print(f"SKILL_SERVER_READY port={args.port}", flush=True)
while app.is_running():
    srv.handle_request()
    if not args.headless:  # keep the GUI alive in DCV; headless: no idle frames in the run video
        lab.hold(1)
