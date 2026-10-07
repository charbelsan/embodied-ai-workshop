"""Skill server of the PC scene: ONE live RamLab (ram1 in the gripper, case at a random pose) over HTTP, port 8766.
Same protocol as the cabinet server (scripts/skill_server.py, 8765); the MCP routes `insert_ram` and `observe(scene='pc')` here.

    <Isaac python> scripts/pcgr_skill_server.py --port 8766 --runs_dir <workspace>/results/pc_runs

POST /run                {"run_id": "...", "goal": "...", "reset": true}     start a run (new episode, trace + video)
POST /skill              {"name": "insert_ram_student", "args": {}}         learned skill -> {ok, reason, duration_s, evidence}
                         ok = "ran to its end without a technical error"; it says NOTHING about the stick being seated
POST /decision           {"text": "..."}                                    log the Brain's reasoning
POST /end                {"notes": "..."}                                   close the run (video, calls); no grade in the answer
GET  /showcase           {"rgb_showcase": PNG base64}: 1280x720 view of the slot, for humans (notebook); the run video is
                         this view with the student's two 128 px cameras inset. Never routed to a Brain (MCP: /api/observe).
POST /api/observe_allowed (alias /api/observe, GET /observe)                 front + wrist images (PNG base64) + the robot's
                         joint state, gripper width, hand pose (its own kinematics). No part, slot or case pose.
GET  /health             {"ok", "pid", "uptime_s", "scene", "skills"}: the process is up and its HTTP loop turns
Evaluator only, never routed to a Brain: POST /eval/reset {"seed": n} (a given episode), GET /eval/grade (ram_seated...).

Exit codes: 0 stopped (SIGTERM / app closed), 1 exception (boot or loop), 3 boot longer than --boot_timeout, 4 HTTP loop
stalled longer than --stall_timeout (in-process watchdog thread, best effort: a native hang holding the GIL can starve it).
A GPU crash at boot (`VkResult: ERROR_DEVICE_LOST`, ~1 boot in 5 on the L40S) can leave the process hung: run the server
under scripts/pcgr_server_supervisor.py, which watches the log and /health and restarts it (bounded).
"""
import argparse
import base64
import io
import json
import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=8766)
p.add_argument("--runs_dir", default="pc_runs")
p.add_argument("--checkpoint", default="", help="default checkpoint of insert_ram_student (else env / registry)")
p.add_argument("--wrist_view", default="upstream", help="wrist camera mount; must match the checkpoint's")
p.add_argument("--showcase", default="1280x720", help="human-only camera WxH (notebook pictures, run video); 0 = none")
p.add_argument("--boot_timeout", type=float, default=300.0, help="s: exit 3 if the server is not ready by then")
p.add_argument("--stall_timeout", type=float, default=180.0, help="s: exit 4 if the HTTP loop does not turn for that long")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
T_START = time.time()
BEAT = {"t": time.time(), "ready": False}


def _watchdog():
    """Best effort: turn a stuck boot or a stalled HTTP loop into an exit the supervisor sees."""
    while True:
        time.sleep(2.0)
        now = time.time()
        if not BEAT["ready"] and now - T_START > args.boot_timeout:
            print(f"PCGR_WATCHDOG boot not finished after {args.boot_timeout:.0f} s: exit 3", flush=True)
            os._exit(3)
        if BEAT["ready"] and now - BEAT["t"] > args.stall_timeout:
            print(f"PCGR_WATCHDOG HTTP loop stalled for {now - BEAT['t']:.0f} s: exit 4", flush=True)
            os._exit(4)


threading.Thread(target=_watchdog, daemon=True).start()
args.headless = True
args.enable_cameras = True
if not getattr(args, "device", None) or args.device == "cuda:0":
    args.device = "cpu"  # physics on CPU (1 env), rendering on the GPU
if not getattr(args, "rendering_mode", None):
    args.rendering_mode = "balanced"
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vinci_lab.pcgr.world import PARTS  # noqa: E402
from vinci_lab.ramchain.skill import insert_ram_student  # noqa: E402
from vinci_lab.ramchain.world import PART, RamLab  # noqa: E402

SHOW = tuple(int(v) for v in args.showcase.lower().split("x")) if args.showcase not in ("", "0") else None
try:
    lab = RamLab(device=args.device, cameras=True, wrist_view=args.wrist_view, showcase=SHOW)
except BaseException:
    import traceback

    traceback.print_exc()
    sys.stdout.flush()
    os._exit(1)
state = {"run": None, "writer": None, "video": None, "seed": None}


def new_episode(seed=None):
    seed = int(seed) if seed is not None else int(time.time() * 1000) % 1_000_000
    lab.reset_episode(seed)
    lab.aim_showcase()
    for _ in range(3):
        lab.render()
    state["seed"] = seed


def result(ok, reason, t0, **evidence):
    return {"ok": bool(ok), "reason": reason, "duration_s": round(time.time() - t0, 2), "evidence": evidence}


def _png(img):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _frame():
    """One run-video frame: the showcase view with the student's two views inset (what the policy sees), or the two views."""
    (fr, _), (wr, _) = lab.images()
    fr, wr = fr[0, ..., :3].cpu().numpy(), wr[0, ..., :3].cpu().numpy()
    if SHOW is None:
        return np.concatenate([fr, wr], 1)
    from PIL import Image, ImageDraw, ImageFont

    im = Image.fromarray(np.ascontiguousarray(lab.showcase()))
    k, pad = 2, 16
    y = im.height - fr.shape[0] * k - pad
    for i, v in enumerate((fr, wr)):
        tile = Image.fromarray(np.ascontiguousarray(v)).resize((v.shape[1] * k, v.shape[0] * k), Image.NEAREST)
        im.paste(tile, (pad + i * (tile.width + 8), y))
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    ImageDraw.Draw(im).text((pad, y - 26), f"student cameras, {fr.shape[1]} px: what the policy sees", font=font, fill=(255, 255, 255),
                            stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(im)


def _record():
    if state["writer"] is None:
        import imageio.v2 as imageio

        state["video"] = os.path.join(state["run"]["dir"], "video.mp4")
        state["writer"] = imageio.get_writer(state["video"], fps=16, quality=7, macro_block_size=8)
    state["writer"].append_data(_frame())


def _close_video():
    w, v = state["writer"], state["video"]
    state["writer"], state["video"] = None, None
    if w is not None:
        w.close()
    return v


def observe_allowed():
    lab.render()
    (fr, _fd), (wr, _wd) = lab.images()
    F = lab.F
    hp, hq = lab.hand()
    return {"rgb_front": _png(fr[0, ..., :3].cpu().numpy()), "rgb_wrist": _png(wr[0, ..., :3].cpu().numpy()),
            "joint_names": F.joint_names, "joint_pos": [round(v, 4) for v in F.data.joint_pos[0].tolist()],
            "joint_vel": [round(v, 4) for v in F.data.joint_vel[0].tolist()], "gripper_width_m": round(lab.gripper_width(), 4),
            "ee_pos": [round(v, 4) for v in hp[0].tolist()], "ee_quat": [round(v, 4) for v in hq[0].tolist()],
            "sim_time_s": round(lab.sim_time(), 2), "scene": "pc"}


def grade():
    """Evaluator: the stick's true state (never forwarded to a Brain)."""
    return {"seed": state["seed"], "ram_seated": lab.ram_seated(), "where": lab.where(PART), "depth_mm": round(lab.depth(PART) * 1000, 2),
            "tilt_deg": round(lab.tilt_deg(PART), 2), "initial": lab.initial, "parts": {n: lab.where(n) for n in PARTS}}


def skill_insert(**kw):
    t0 = time.time()
    rec = state["run"] is not None

    def on_step(k, f64, w64, prop, u):
        BEAT["t"] = time.time()  # a running skill is not a stalled loop
        if rec:
            _record()

    r = insert_ram_student(lab, checkpoint=kw.get("checkpoint") or args.checkpoint or None, on_step=on_step)
    ev = {k: r[k] for k in r if k not in ("executed", "reason", "duration_s")}
    ev["policy_steps"] = r["steps"]
    return result(r["executed"], r["reason"], t0, provider="dagger_student", **ev)


SKILLS = {"insert_ram_student": skill_insert}


def log(entry):
    run = state["run"]
    entry = {"t_wall": round(time.time() - run["t0"], 2), "sim_time_s": round(lab.sim_time(), 2), **entry}
    with open(os.path.join(run["dir"], "trace.jsonl"), "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


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
            return self._send(observe_allowed())
        if self.path == "/health":
            return self._send({"ok": True, "pid": os.getpid(), "uptime_s": round(time.time() - T_START, 1), "scene": "pc",
                               "skills": list(SKILLS)})
        if self.path == "/eval/grade":
            return self._send(grade())
        if self.path == "/showcase" and SHOW is not None:
            lab.render()
            return self._send({"rgb_showcase": _png(lab.showcase())})
        self._send({"error": "unknown"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path in ("/api/observe_allowed", "/api/observe"):
            return self._send(observe_allowed())
        if self.path == "/eval/reset":
            new_episode(body.get("seed"))
            return self._send({"ok": True, "seed": state["seed"]})
        if self.path == "/run":
            rid = body.get("run_id") or time.strftime("pc_run_%Y%m%d_%H%M%S")
            d = os.path.join(args.runs_dir, rid)
            os.makedirs(d, exist_ok=True)
            _close_video()
            state["run"] = {"id": rid, "dir": d, "t0": time.time(), "calls": 0, "fails": 0}
            if body.get("reset", True):
                new_episode(body.get("seed"))
            return self._send(log({"type": "goal", "goal": body.get("goal", ""), "brain": body.get("brain", "")}))
        if self.path == "/skill":
            name, kw = body.get("name"), body.get("args", {}) or {}
            if name not in SKILLS:
                return self._send({"ok": False, "reason": f"unknown skill {name}; known {list(SKILLS)}"}, 400)
            if state["run"]:
                log({"type": "skill_call", "skill": name, "args": kw})
            try:
                r = SKILLS[name](**kw)
            except Exception as e:  # a skill crash is a failed call for the Brain, not a server crash
                r = {"ok": False, "reason": f"technical error: {e!r}", "duration_s": 0.0, "evidence": {}}
            if state["run"]:
                state["run"]["calls"] += 1
                state["run"]["fails"] += 0 if r.get("ok") else 1
                log({"type": "skill_result", "skill": name, **r})
            return self._send(r)
        if state["run"] is None:
            return self._send({"error": "POST /run first"}, 400)
        if self.path == "/decision":
            return self._send(log({"type": "decision", "text": body.get("text", "")}))
        if self.path == "/end":
            run = state["run"]
            video = _close_video()
            out = log({"type": "end", "skill_calls": run["calls"], "failed_calls": run["fails"], "video": video, "notes": body.get("notes", "")})
            state["run"] = None
            return self._send(out)
        self._send({"error": "unknown"}, 404)


import signal  # noqa: E402

signal.signal(signal.SIGTERM, lambda *_: os._exit(0))  # the Kit loop ignores SIGTERM otherwise: `kill PID` must stop the server
code = 0
try:
    new_episode(0)
    srv = HTTPServer(("127.0.0.1", args.port), H)
    srv.timeout = 0.05
    BEAT.update(t=time.time(), ready=True)
    print(f"PCGR_SKILL_SERVER_READY port={args.port} pid={os.getpid()} boot_s={time.time() - T_START:.1f}", flush=True)
    while app.is_running():
        srv.handle_request()
        BEAT["t"] = time.time()
except BaseException:
    import traceback

    traceback.print_exc()
    code = 1
sys.stdout.flush()
os._exit(code)
