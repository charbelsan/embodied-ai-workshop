"""Supervisor of the PC skill server (8766): start it, watch it, restart it on a crash or a hang, a bounded number of times.

    python3 scripts/pcgr_server_supervisor.py --port 8766 [--max_starts 5] [-- <extra pcgr_skill_server.py args>]

Standard library only (any python3). Why: a GPU crash at boot (`VkResult: ERROR_DEVICE_LOST`, ~1 boot in 5 on the L40S)
leaves the Isaac process hung instead of exiting, and a hung process holds the port.
  - start `pcgr_skill_server.py` in its own process group, log to <log_dir>/server_<n>.log;
  - ready = GET /health answers within --boot_timeout. Then writes <run_dir>/pcgr_server.ready (JSON: pid, port, starts)
    and prints `PCGR_SERVER_UP port=... pid=...`;
  - restart (SIGKILL the group, start again) when: the log shows a GPU-crash signature, the process exits, the boot does
    not finish in --boot_timeout, or /health fails --health_failures times in a row (each try waits --health_timeout:
    a running skill makes /health wait ~10 s, it is not a failure);
  - at most --max_starts starts in total: then exit 2 (the port is free, the reason is printed and in the log);
  - SIGTERM / SIGINT: stop the server (SIGTERM, then SIGKILL after 10 s), remove the ready/pid files, exit 0.
Files: <run_dir>/pcgr_server.pid (the server's pid), <run_dir>/pcgr_server.ready (only while healthy).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CRASH_SIGNATURES = ("ERROR_DEVICE_LOST", "A GPU crash occurred", "GPU crash is detected")
KIT = "--/physics/numThreads=1 --/plugins/carb.tasking.plugin/threadCount=3"

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8766)
ap.add_argument("--python", default=sys.executable, help="the Isaac venv python")
ap.add_argument("--cwd", default=".")
ap.add_argument("--run_dir", default=os.path.expanduser("~/pcgr_server"))
ap.add_argument("--max_starts", type=int, default=5)
ap.add_argument("--boot_timeout", type=float, default=300.0)
ap.add_argument("--health_period", type=float, default=10.0)
ap.add_argument("--health_timeout", type=float, default=60.0)
ap.add_argument("--health_failures", type=int, default=3)
ap.add_argument("server_args", nargs=argparse.REMAINDER, help="after --: extra arguments of pcgr_skill_server.py")
a = ap.parse_args()
extra = [x for x in a.server_args if x != "--"]
os.makedirs(a.run_dir, exist_ok=True)
PID_FILE, READY_FILE = os.path.join(a.run_dir, "pcgr_server.pid"), os.path.join(a.run_dir, "pcgr_server.ready")
state = {"proc": None, "stopping": False}


def say(msg):
    print(f"[pcgr-supervisor {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def health(timeout):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=timeout) as r:
            return json.loads(r.read()).get("ok") is True
    except Exception:  # noqa: BLE001
        return False


def clear_files():
    for f in (PID_FILE, READY_FILE):
        if os.path.exists(f):
            os.remove(f)


def kill_group(proc, grace=0.0):
    if proc is None or proc.poll() is not None:
        return
    try:
        if grace:
            os.killpg(proc.pid, signal.SIGTERM)
            t = time.time()
            while proc.poll() is None and time.time() - t < grace:
                time.sleep(0.5)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass


def stop(*_):
    state["stopping"] = True
    say("stop requested: stopping the server")
    kill_group(state["proc"], grace=10.0)
    clear_files()
    sys.exit(0)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


class Log:
    """New lines of the server log since the last read."""

    def __init__(self, path):
        self.path, self.pos = path, 0

    def crash(self):
        try:
            with open(self.path, errors="replace") as f:
                f.seek(self.pos)
                chunk = f.read()
                self.pos = f.tell()
        except OSError:
            return None
        return next((sig for sig in CRASH_SIGNATURES if sig in chunk), None)


def start(n):
    log_path = os.path.join(a.run_dir, f"server_{n}.log")
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1"}
    cmd = [a.python, os.path.join(REPO, "scripts", "pcgr_skill_server.py"), "--port", str(a.port), f"--kit_args={KIT}",
           "--boot_timeout", str(a.boot_timeout)] + extra
    proc = subprocess.Popen(cmd, cwd=a.cwd, env=env, stdin=subprocess.DEVNULL, stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))
    say(f"start {n}/{a.max_starts}: pid {proc.pid}, log {log_path}")
    return proc, Log(log_path)


def supervise_once(n):
    """One server life. Returns the reason it ended (to restart), or never returns while it stays healthy."""
    proc, log = start(n)
    state["proc"] = proc
    t0 = time.time()
    while True:  # boot
        time.sleep(2.0)
        sig = log.crash()
        if sig:
            return proc, f"GPU crash signature at boot ({sig})"
        if proc.poll() is not None:
            return proc, f"exited at boot with code {proc.returncode}"
        if health(5.0):
            break
        if time.time() - t0 > a.boot_timeout:
            return proc, f"not ready after {a.boot_timeout:.0f} s"
    with open(READY_FILE, "w") as f:
        json.dump({"pid": proc.pid, "port": a.port, "start": n, "boot_s": round(time.time() - t0, 1), "t": time.time()}, f)
    say(f"PCGR_SERVER_UP port={a.port} pid={proc.pid} boot_s={time.time() - t0:.1f}")
    fails = 0
    while True:  # serve
        time.sleep(a.health_period)
        sig = log.crash()
        if sig:
            return proc, f"GPU crash signature ({sig})"
        if proc.poll() is not None:
            return proc, f"exited with code {proc.returncode}"
        fails = 0 if health(a.health_timeout) else fails + 1
        if fails >= a.health_failures:
            return proc, f"/health failed {fails} times in a row"


for n in range(1, a.max_starts + 1):
    proc, why = supervise_once(n)
    if os.path.exists(READY_FILE):
        os.remove(READY_FILE)
    say(f"server pid {proc.pid}: {why}; killing its process group")
    kill_group(proc)
say(f"PCGR_SERVER_GAVE_UP after {a.max_starts} starts")
clear_files()
sys.exit(2)
