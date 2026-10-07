"""Integration check: N episodes through the PC skill server's HTTP protocol, graded by the evaluator endpoint.

    python3 scripts/ram_skill_eval.py --url http://127.0.0.1:8766 --seeds 20000:20020 --out /tmp/skill_eval.jsonl
Per episode: POST /eval/reset {seed} (evaluator) -> POST /skill insert_ram_student (what a Brain calls: execution report
only) -> GET /eval/grade (evaluator: ram_seated). Prints the success rate with its Wilson 95 % interval.
"""
import argparse
import json
import math
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8766")
p.add_argument("--seeds", default="20000:20020")
p.add_argument("--out", required=True)
a = p.parse_args()


def call(path, body=None, method="POST"):
    data = None if method == "GET" else json.dumps(body or {}).encode()
    req = urllib.request.Request(a.url + path, data=data, headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


# the MCP's view (lib/mcp_server.py): observe(scene='pc') and insert_ram() read these keys only
o = call("/api/observe_allowed")
import base64  # noqa: E402

png_ok = all(base64.b64decode(o[k])[:4] == b"\x89PNG" for k in ("rgb_front", "rgb_wrist"))
state_keys = [k for k in ("joint_pos", "joint_vel", "gripper_width_m", "ee_pos", "ee_quat", "sim_time_s") if k in o]
leaks = [k for k in o if any(w in k.lower() for w in ("seat", "slot", "case", "part", "ram", "grade", "seed", "where", "depth"))]
print(f"OBSERVE_ALLOWED png={png_ok} state_keys={state_keys} other_keys={sorted(set(o) - set(state_keys) - {'rgb_front', 'rgb_wrist'})} "
      f"leaks={leaks}", flush=True)
run = call("/run", {"run_id": f"skill_eval_{int(time.time())}", "goal": "insert the RAM stick", "reset": True, "seed": 424242})
r = call("/skill", {"name": "insert_ram_student", "args": {}})
again = call("/skill", {"name": "insert_ram_student", "args": {}})  # gripper now empty: must be a technical error, not a crash
end = call("/end", {"notes": "integration check"})
print(f"RUN_FLOW skill_ok={r['ok']} evidence_keys={sorted(r['evidence'])} second_call={again['ok']}:{again['reason'][:60]} "
      f"end_keys={sorted(end)} video={end.get('video')}", flush=True)
seeds = list(range(*[int(v) for v in a.seeds.split(":")])) if ":" in a.seeds else [int(v) for v in a.seeds.split(",")]
k = 0
for s in seeds:
    t0 = time.time()
    call("/eval/reset", {"seed": s})
    r = call("/skill", {"name": "insert_ram_student", "args": {}})
    g = call("/eval/grade", method="GET")
    k += bool(g["ram_seated"])
    rec = {"seed": s, "skill": r, "grade": g, "wall_s": round(time.time() - t0, 1)}
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"seed={s} ran={r['ok']} sha={str(r['evidence'].get('checkpoint_sha256'))[:12]} seated={g['ram_seated']} "
          f"where={g['where']} wall={rec['wall_s']}s", flush=True)
n = len(seeds)
ph = k / n
z = 1.96
d = 1 + z * z / n
c = ph + z * z / (2 * n)
h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
print(f"SKILL_SUCCESS {k}/{n} = {100 * ph:.0f}% IC95 [{100 * (c - h) / d:.1f}, {100 * (c + h) / d:.1f}]", flush=True)
