"""Reference agentic system: a real coding agent (Claude Code, headless) as the Brain, through the MCP robot tools only.

    python3 scripts/brain_reference_run.py --runs 10 --out ~/brain_ref      # needs the cabinet skill server on :8765

Per run: the evaluator resets the scene, starts `claude -p` with ONLY the vinci-robot MCP tools (observe, open_drawer,
close_drawer; no shell, no file access, no oracle), then measures the truth ITSELF by polling the simulator between tool
calls (max drawer opening, final opening). Logged: model, tools called in order, checkpoint sha256 executed,
observations available to the Brain, the Brain's own final claim, and the evaluator's verdict.
Task success = the drawer reached >= 0.30 m during the run AND ends closed (< 0.02 m).
"""
import argparse
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

MISSION = """You control a simulated Franka robot arm in front of a cabinet, through the tools of the MCP server
"vinci-robot" (and nothing else). Mission: open the TOP drawer of the cabinet, check on the camera images that it is
really open, try again if it is not (at most 3 attempts), then close it and check on the images that it is closed.
Tools report whether they ran, not whether they succeeded: judge from the images of observe().
End your answer with one line of JSON: {"opened_seen": true/false, "closed_seen": true/false, "attempts": N}"""
TOOLS = ["mcp__vinci-robot__observe", "mcp__vinci-robot__open_drawer", "mcp__vinci-robot__close_drawer"]


def post(url, path, body=None, timeout=900):
    req = urllib.request.Request(url + path, data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--out", default=str(Path.home() / "brain_ref"))
    ap.add_argument("--server", default="http://127.0.0.1:8765")
    ap.add_argument("--workspace", default=str(Path.home() / "vinci-workshop"))
    ap.add_argument("--runtime", default="/opt/vinci-workshop")
    ap.add_argument("--model", default="", help="Claude model (empty = the CLI default)")
    ap.add_argument("--timeout_s", type=int, default=1500)
    ap.add_argument("--env_file", default="", help="file of `export KEY=VALUE` lines (e.g. the agent's login token), "
                                                   "passed to the agent process only, never logged")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # a clean agent session: drop the variables of any calling agent session, add the login from --env_file
    agent_env = {k: v for k, v in os.environ.items() if not (k.startswith("CLAUDE_CODE_") or k in ("CLAUDECODE", "CLAUDE_PID"))}
    if a.env_file:
        for line in open(a.env_file):
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                agent_env[key.strip()] = val.strip().strip('"').strip("'")
    runs = []
    for k in range(a.runs):
        d = out / f"run_{k:02d}"
        d.mkdir(exist_ok=True)
        post(a.server, "/api/reset")
        truth = {"max_opening_m": 0.0, "samples": 0}
        stop = threading.Event()

        def poll():  # evaluator only: served by the simulator between the Brain's tool calls
            while not stop.is_set():
                try:
                    s = post(a.server, "/api/success", timeout=900)
                    truth["max_opening_m"] = max(truth["max_opening_m"], s["drawer_opening_m"])
                    truth["samples"] += 1
                except Exception:  # noqa: BLE001
                    pass
                stop.wait(1.5)

        cfg = {"mcpServers": {"vinci-robot": {"command": f"{a.runtime}/nb-venv/bin/python", "args": [f"{a.runtime}/lib/mcp_server.py"],
                                              "env": {"VINCI_WORKSPACE": a.workspace, "VINCI_MCP_LOG": str(d / "mcp_calls.jsonl")}}}}
        (d / "mcp.json").write_text(json.dumps(cfg))
        cmd = ["claude", "-p", MISSION, "--mcp-config", str(d / "mcp.json"), "--strict-mcp-config",
               "--allowedTools", ",".join(TOOLS), "--output-format", "json"]
        if a.model:
            cmd += ["--model", a.model]
        th = threading.Thread(target=poll, daemon=True)
        th.start()
        t0 = time.time()
        try:
            p = subprocess.run(cmd, cwd=a.workspace, capture_output=True, text=True, timeout=a.timeout_s, stdin=subprocess.DEVNULL,
                               env=agent_env)
            raw, err = p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            raw, err = (e.stdout or ""), "timeout"
        wall = time.time() - t0
        stop.set()
        th.join(timeout=900)
        final = post(a.server, "/api/success")["drawer_opening_m"]
        truth["max_opening_m"] = max(truth["max_opening_m"], final)
        (d / "claude_output.json").write_text(raw or "")
        (d / "claude_stderr.txt").write_text(err or "")
        try:
            res = json.loads(raw)
        except Exception:  # noqa: BLE001
            res = {}
        text = res.get("result", "") if isinstance(res, dict) else ""
        claim = {}
        for line in reversed(text.strip().splitlines()):
            if line.strip().startswith("{"):
                try:
                    claim = json.loads(line.strip().strip("`"))
                    break
                except Exception:  # noqa: BLE001
                    pass
        calls = [json.loads(l) for l in open(d / "mcp_calls.jsonl")] if (d / "mcp_calls.jsonl").exists() else []
        opened = truth["max_opening_m"] >= 0.30
        closed = final < 0.02
        rec = {"run": k, "models": sorted((res.get("modelUsage") or {}).keys()) if isinstance(res, dict) else [],
               "tools_called": [c["tool"] for c in calls],
               "checkpoints_executed": sorted({c["result"].get("checkpoint_sha256") for c in calls
                                              if c["tool"] == "open_drawer" and isinstance(c.get("result"), dict)} - {None}),
               "observations_available": "observe(): front + wrist camera images + joint state; skills report execution only",
               "brain_claim": claim, "evaluator": {"max_opening_m": round(truth["max_opening_m"], 3), "final_opening_m": round(final, 3),
                                                   "opened_0.30m": opened, "closed": closed, "task_success": opened and closed},
               "claim_matches_truth": (claim.get("opened_seen") == opened and claim.get("closed_seen") == closed) if claim else None,
               "num_turns": res.get("num_turns") if isinstance(res, dict) else None,
               "cost_usd": res.get("total_cost_usd") if isinstance(res, dict) else None, "wall_s": round(wall, 1)}
        runs.append(rec)
        with open(out / "runs.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("RUN " + json.dumps(rec), flush=True)
    n = len(runs)
    summ = {"runs": n, "task_success": sum(r["evaluator"]["task_success"] for r in runs),
            "opened": sum(r["evaluator"]["opened_0.30m"] for r in runs), "closed": sum(r["evaluator"]["closed"] for r in runs),
            "claims_match_truth": sum(bool(r["claim_matches_truth"]) for r in runs),
            "open_drawer_calls": sum(r["tools_called"].count("open_drawer") for r in runs),
            "models": sorted({m for r in runs for m in r["models"]}),
            "checkpoints": sorted({c for r in runs for c in r["checkpoints_executed"]}),
            "median_wall_s": sorted(r["wall_s"] for r in runs)[n // 2] if n else None,
            "total_cost_usd": round(sum(r["cost_usd"] or 0 for r in runs), 2)}
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print("SUMMARY " + json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
