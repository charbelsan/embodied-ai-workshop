#!/bin/bash
# End-to-end check of a workshop machine, run as the desktop user (e.g. over SSH). Exit code != 0 on any failure.
# It works on a TEMPORARY copy of the workspace, so the team's workspace, results and timeline stay untouched.
#
#   ./smoke_test.sh [--full-notebook] [--workspace DIR] [--runtime DIR] [--keep] [--extra-leaks FILE]
#   FILE: extra leak rules, one per line: "path <file or folder under each home>" or "pattern <extended regex>"
#
# Checks: GPU + driver 580, DCV service + console session, desktop icons, read-only runtime, start.sh READY,
# guided notebook executed (quick mode unless --full-notebook), PC scene through the MCP (insert_ram), evaluate.sh on 2 episodes, a team/config.yaml change
# moves the score, submit.sh --dry-run, and a leak scan of the whole disk.
set -uo pipefail
RT=/opt/vinci-workshop; WS="$HOME/vinci-workshop"; FULL=0; KEEP=0; EXTRA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --full-notebook) FULL=1 ;;
    --workspace) WS="$2"; shift ;;
    --runtime) RT="$2"; shift ;;
    --keep) KEEP=1 ;;
    --extra-leaks) EXTRA="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac; shift
done
PASS=0; FAIL=0; REPORT=()
t_now() { date +%s; }
check() {  # name, status(0 ok), detail
  if [ "$2" = 0 ]; then PASS=$((PASS + 1)); REPORT+=("PASS  $1  $3"); else FAIL=$((FAIL + 1)); REPORT+=("FAIL  $1  $3"); fi
  printf '%s  %-34s %s\n' "$([ "$2" = 0 ] && echo PASS || echo FAIL)" "$1" "$3"
}
SUDO=""; sudo -n true 2>/dev/null && SUDO="sudo -n"
port_pid() { ss -ltnpH "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2; }

echo "== VINCI workshop smoke test ($(date -u +%FT%TZ), $(hostname))"
# ---------------------------------------------------------------- machine
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
[[ "$drv" == 580.* ]]; check "GPU + driver 580" $? "driver=${drv:-none} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
if command -v dcv >/dev/null || systemctl cat dcvserver >/dev/null 2>&1; then   # cloud team machine
  systemctl is-active --quiet dcvserver; check "DCV service" $? "$(systemctl is-active dcvserver 2>/dev/null)"
  sess=$($SUDO dcv list-sessions 2>/dev/null)
  echo "$sess" | grep -q "console"; check "DCV console session" $? "$(echo "$sess" | head -1 | cut -c1-80)"
else echo "info  DCV not installed: single-machine install, remote desktop checks skipped"; fi
DESK="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"; [ "$DESK" = "$HOME" ] && DESK="$HOME/Desktop"
missing=""
for i in 01-start-workshop 02-isaac-sim 03-jupyter 04-vscode 05-terminal 06-results 07-challenge-sheet 08-restart-workshop; do
  [ -x "$DESK/$i.desktop" ] || missing="$missing $i"
done
[ -z "$missing" ]; check "desktop icons" $? "${missing:+missing:$missing}${missing:-8 icons in $DESK}"
command -v code >/dev/null; check "VS Code installed" $? "$(code --version 2>/dev/null | head -1)"
owner=$(stat -c %U "$RT" 2>/dev/null); ( touch "$RT/.w" 2>/dev/null && rm -f "$RT/.w" ) && rw=1 || rw=0
[ "$owner" = root ] && [ "$rw" = 0 ]; check "runtime read-only" $? "$RT owner=$owner writable_by_user=$rw"
[ -d "$WS/team" ] && [ -x "$WS/evaluate.sh" ]; check "team workspace" $? "$WS"
nb_ws=$(md5sum < "$WS/notebooks/01_guided_lab.ipynb" 2>/dev/null); nb_rt=$(md5sum < "$RT/templates/notebooks/01_guided_lab.ipynb" 2>/dev/null)
[ -n "$nb_rt" ] && [ "$nb_ws" = "$nb_rt" ]; check "workspace = runtime version" $? \
  "$([ "$nb_ws" = "$nb_rt" ] && echo "guided notebook as shipped" || echo "the workspace notebook is not the runtime's: stale workspace or pristine copy?")"

# ---------------------------------------------------------------- temporary workspace
T=$(mktemp -d /tmp/vinci_smoke_XXXX); cp -r "$WS" "$T/ws"; rm -rf "$T/ws/results"/*; export VINCI_WORKSPACE="$T/ws"
pre_jup=$(port_pid 8888); pre_sim=$(port_pid 8765); pre_pc=$(port_pid 8766)
cd "$T/ws"
t0=$(t_now); out=$(timeout 400 ./start.sh 2>&1); rc=$?
echo "$out" | grep -q "^READY"; check "start.sh READY" $? "$(( $(t_now) - t0 )) s, rc=$rc"
[ "$rc" = 0 ] || echo "$out" | tail -8

# Check actual image motion before trusting visual policy results. This resets only the live scene.
t0=$(t_now)
camera_out=$(timeout 180 "$RT/nb-venv/bin/python" "$RT/scripts/check_live_cameras.py" 2>&1); camera_rc=$?
check "live camera motion" "$camera_rc" "$(( $(t_now) - t0 )) s, $camera_out"

# guided notebook
NB="$T/ws/notebooks/01_guided_lab.ipynb"
[ "$FULL" = 1 ] && unset VINCI_NB_FAST || export VINCI_NB_FAST=1
[ "${NB_LANG:-fr}" = en ] && sed -i 's/LANG = \\"fr\\"/LANG = \\"en\\"/' "$NB"   # NB_LANG=en: run it in English
t0=$(t_now)
( cd "$T/ws/notebooks" && env -u MPLBACKEND timeout 2400 "$RT/nb-venv/bin/jupyter" nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=1500 --ExecutePreprocessor.kernel_name=vinci-nb --output "$T/nb_out.ipynb" "$NB" ) \
    > "$T/nb.log" 2>&1
rc=$?
nbinfo=$("$RT/nb-venv/bin/python" - "$T/nb_out.ipynb" <<'PY' 2>/dev/null
import json, sys
from datetime import datetime
nb = json.load(open(sys.argv[1]))
errs, slow = 0, []
mission_ok = False
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    errs += any(o.get("output_type") == "error" for o in c.get("outputs", []))
    if 'tl("first_agent")' in ''.join(c.get("source", [])):
        output = ''.join(''.join(o.get("text", [])) for o in c.get("outputs", []) if o.get("output_type") == "stream")
        mission_ok = "TÂCHE RÉUSSIE" in output or "TASK SUCCEEDED" in output
    ex = c.get("metadata", {}).get("execution", {})
    try:
        d = (datetime.fromisoformat(ex["shell.execute_reply"].replace("Z", "+00:00")) -
             datetime.fromisoformat(ex["iopub.execute_input"].replace("Z", "+00:00"))).total_seconds()
        if d > 20:
            slow.append(f"cell{i}={d:.0f}s")
    except Exception:
        pass
errs += int(not mission_ok)
print(f"errors={errs} mission_success={mission_ok} {' '.join(slow)}")
PY
)
[ "$rc" = 0 ] && [[ "$nbinfo" == errors=0* ]]; check "guided notebook ($([ "$FULL" = 1 ] && echo full || echo quick), ${NB_LANG:-fr})" $? \
  "$(( $(t_now) - t0 )) s, $nbinfo"
[ "$rc" = 0 ] || tail -5 "$T/nb.log"

# PC scene through the coding agents' MCP server: new episode (reset.sh), insert_ram, observe(scene='pc')
t0=$(t_now); ./reset.sh > "$T/reset.log" 2>&1
pcinfo=$(VINCI_WORKSPACE="$T/ws" timeout 300 "$RT/nb-venv/bin/python" - "$RT" <<'PY' 2>/dev/null
import asyncio, json, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
rt = sys.argv[1]
want = json.load(open(f"{rt}/policies/insert_ram_student.json"))["sha256"]
async def main():
    srv = StdioServerParameters(command=f"{rt}/nb-venv/bin/python", args=[f"{rt}/lib/mcp_server.py"])
    async with stdio_client(srv) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            ins = json.loads((await s.call_tool("insert_ram", {})).content[0].text)
            obs = await s.call_tool("observe", {"scene": "pc"})
            imgs = sum(c.type == "image" for c in obs.content)
            ok = ins.get("ran") and ins.get("checkpoint_sha256") == want and imgs == 2
            print(f"{'OK' if ok else 'BAD'} ran={ins.get('ran')} sha={str(ins.get('checkpoint_sha256'))[:12]} images={imgs}")
asyncio.run(main())
PY
)
grep -q "PC scene reset" "$T/reset.log" && [[ "$pcinfo" == OK* ]]; check "PC scene via MCP (insert_ram)" $? "${pcinfo:-no answer}, $(( $(t_now) - t0 )) s"

# evaluate, change the team config, evaluate again
score() { grep -o "Average score: *[0-9.]*" "$1" | grep -o "[0-9.]*$"; }
t0=$(t_now); timeout 900 ./evaluate.sh --episodes 2 --procs 2 > "$T/eval1.log" 2>&1; rc=$?
s1=$(score "$T/eval1.log")
[ "$rc" = 0 ] && [ -n "$s1" ]; check "evaluate.sh (2 episodes)" $? "score=${s1:-none}/100, $(( $(t_now) - t0 )) s"
sed -i 's/^  enabled: false/  enabled: true/' team/config.yaml
t0=$(t_now); timeout 900 ./evaluate.sh --episodes 2 --procs 2 > "$T/eval2.log" 2>&1; rc=$?
s2=$(score "$T/eval2.log")
[ "$rc" = 0 ] && [ -n "$s2" ] && [ "$s1" != "$s2" ]; check "config change moves the score" $? \
  "verification.enabled=true: ${s1:-?} -> ${s2:-?}, $(( $(t_now) - t0 )) s"
out=$(./submit.sh --dry-run 2>&1); rc=$?
[ "$rc" = 0 ] && echo "$out" | grep -qi "dry run OK"; check "submit.sh --dry-run" $? "$(echo "$out" | grep -i -m1 "dry run")"

# stop what this test started, so the team's START WORKSHOP starts clean
SUPF="$HOME/.vinci/pc_scene/supervisor.pid"  # the PC scene's supervisor would restart a killed server: stop it first
if [ -z "$pre_pc" ] && [ -f "$SUPF" ] && kill "$(cat "$SUPF")" 2>/dev/null; then
  for _ in $(seq 20); do [ -z "$(port_pid 8766)" ] && break; sleep 1; done
fi
for port in 8888 8765 8766; do
  pid=$(port_pid $port); pre=$(case $port in 8888) echo "$pre_jup" ;; 8765) echo "$pre_sim" ;; *) echo "$pre_pc" ;; esac)
  if [ -n "$pid" ] && [ -z "$pre" ]; then  # the simulator ignores SIGTERM: escalate
    kill "$pid" 2>/dev/null; for _ in 1 2 3 4 5; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null
  fi
done
cd /; [ "$KEEP" = 1 ] && echo "kept: $T" || rm -rf "$T"

# ---------------------------------------------------------------- leak scan (whole disk)
# Credentials only: a fresh ~/.claude or ~/.claude.json (no login) is normal on a team machine.
hits=""
LPATHS=".config/gh .git-credentials .aws/credentials .netrc .bash_history .claude/.credentials.json .codex/auth.json"
LPAT='AK''IA[0-9A-Z]{16}|AS''IA[0-9A-Z]{16}|gh''p_[A-Za-z0-9]{20,}|github''_pat_|hf_[A-Za-z0-9]{30,}|sk-an''t-[A-Za-z0-9_-]{20,}|-----BEGIN ([A-Z]+ )?PRIV''ATE KEY-----'
XPAT=""
if [ -n "$EXTRA" ]; then
  LPATHS="$LPATHS $(awk '$1=="path"{print $2}' "$EXTRA" | tr '\n' ' ')"
  XPAT=$(awk '$1=="pattern"{sub(/^pattern +/,""); print}' "$EXTRA" | paste -sd'|')
fi
for h in /root /home/*; do
  for p in $LPATHS; do $SUDO test -e "$h/$p" && hits="$hits $h/$p"; done
  $SUDO grep -qE '"(oauthAccount|primaryApiKey)"' "$h/.claude.json" 2>/dev/null && hits="$hits $h/.claude.json(logged-in)"
done
EXCL=(--exclude-dir={site-packages,dist-packages,.venv,venv,nb-venv,node_modules,.cache,.uv-cache,.nv,.git,embodiedswe,aws-cli})
SKIP='/(kit/cache|ov/data)/|^/etc/(ssh|dcv|ssl|pki)/|^/usr/local/lib/python'
# one pass over the disk; the --extra-leaks name patterns only count where team data and workshop files live, not in /usr or /etc
content=$($SUDO grep -rIlE "${EXCL[@]}" "$LPAT${XPAT:+|$XPAT}" /home /root /opt /srv /etc /tmp /var/tmp /var/lib/cloud /usr/local 2>/dev/null \
  | grep -vE "$SKIP" | while read -r f; do case "$f" in /etc/*|/usr/*) $SUDO grep -qE "$LPAT" "$f" && echo "$f" ;; *) echo "$f" ;; esac; done)
# A trusted HTTPS endpoint needs a private server key. Exempt only the active
# Nginx key when root owns it with mode 0600; readable/stale/unexpected keys fail.
filtered=""
while IFS= read -r f; do
  [ -n "$f" ] || continue
  if [[ "$f" == /etc/letsencrypt/archive/*/privkey*.pem ]] &&
     $SUDO python3 "$RT/lib/check_tls_key.py" "$f"; then
    echo "info  protected active TLS server key: $f"
  else
    filtered+="$f"$'\n'
  fi
done <<< "$content"
content=$filtered
content=$(echo "$content" | grep -v '^$' | grep -vxF "${EXTRA:-/nonexistent}" | sort -u | head -20)
[ -z "$hits" ] && [ -z "$content" ]; check "leak scan (disk)" $? "${hits:+paths:$hits }${content:+files: $(echo $content | tr '\n' ' ')}"
for c in claude codex; do printf 'info  %s: %s\n' "$c" "$(command -v $c || echo 'not in PATH')"; done
grep -q "^PERSIST_BUCKET=." /etc/vinci/workshop.env 2>/dev/null && echo "info  team outputs persistence configured (/etc/vinci/workshop.env)"

echo "== $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
