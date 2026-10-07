#!/bin/bash
# START WORKSHOP: check the machine, start what is missing, open the guided notebook. Safe to run twice.
# The boot warm-up owns the simulators until it finishes. A visible DCV desktop
# can precede that point; do not launch competing scene servers in that interval.
case "$(systemctl is-active vinci-warmup.service 2>/dev/null || true)" in
  activating|active)
    echo "NOT READY: the machine is finishing its automatic warm-up."
    echo "Please wait 3 minutes, then run START WORKSHOP again."
    exit 1
    ;;
esac
source "${VINCI_RUNTIME:-/opt/vinci-workshop}/lib/env.sh"
cd "$WS"; mkdir -p results/logs
ok() { printf '  %-28s OK\n' "$1"; }
bad() { printf '  %-28s PROBLEM: %s\n' "$1" "$2"; FAILED=1; }
echo "VINCI Embodied AI Workshop — starting"
tl start
nvidia-smi -L >/dev/null 2>&1 && ok "GPU" || bad "GPU" "no GPU visible: call an organizer"
[ -x "$PY" ] && ok "Isaac Sim / Isaac Lab" || bad "Isaac Sim / Isaac Lab" "not found: call an organizer"
if service_up; then ok "robot simulator"; else
  echo "  starting the robot simulator (about 30 s)..."
  HL=""; [ -z "$DISPLAY" ] && HL="--headless"
  (cd "$WS" && exec nohup "$PY" "$RT/scripts/skill_server.py" --port 8765 $HL --runs_dir "$WS/results/brain_runs") \
      </dev/null > "$WS/results/logs/simulator.log" 2>&1 &
  for _ in $(seq 180); do service_up && break; sleep 1; done
  service_up && ok "robot simulator" || bad "robot simulator" "not answering, see results/logs/simulator.log"
fi
service_up && tl simulator_ready
# PC scene: headless, under a supervisor that restarts it on a GPU crash or a hang (5 starts at most)
PCDIR="$HOME/.vinci/pc_scene"; SUP="$PCDIR/supervisor.pid"; PCLOG="$WS/results/logs/pc_scene.log"
sup_alive() { [ -f "$SUP" ] && kill -0 "$(cat "$SUP")" 2>/dev/null; }
if pc_up; then ok "PC scene (RAM insertion)"
elif ss -ltn 2>/dev/null | grep -q ":${PC_URL##*:} "; then ok "PC scene (RAM insertion, busy)"  # single-threaded: a skill is running
else
  if ! sup_alive; then
    echo "  starting the PC scene (about 60 s)..."
    mkdir -p "$PCDIR"
    (cd "$WS" && exec nohup python3 "$RT/scripts/pcgr_server_supervisor.py" --port "${PC_URL##*:}" --run_dir "$PCDIR" \
        --python "$PY" --cwd "$WS" -- --runs_dir "$WS/results/pc_runs") </dev/null > "$PCLOG" 2>&1 &
    echo $! > "$SUP"
  fi
  for _ in $(seq 240); do pc_up && break; sup_alive || break; sleep 1; done
  if pc_up; then ok "PC scene (RAM insertion)"
  elif grep -q PCGR_SERVER_GAVE_UP "$PCLOG" 2>/dev/null; then
    bad "PC scene (RAM insertion)" "could not start after 5 tries: call an organizer (details: results/logs/pc_scene.log)"
  elif sup_alive; then bad "PC scene (RAM insertion)" "still starting: run ./start.sh again in a minute"
  else bad "PC scene (RAM insertion)" "stopped unexpectedly: run ./start.sh again (details: results/logs/pc_scene.log)"
  fi
fi
if ! curl -s -o /dev/null -m 3 http://127.0.0.1:8888/lab; then
  (exec nohup "$NB_BIN/jupyter" lab --no-browser --ip=127.0.0.1 --port=8888 --ServerApp.token='' --ServerApp.password='' \
      --notebook-dir="$WS") </dev/null > "$WS/results/logs/jupyter.log" 2>&1 &
  for _ in $(seq 30); do curl -s -o /dev/null -m 2 http://127.0.0.1:8888/lab && break; sleep 1; done
fi
curl -s -o /dev/null -m 3 http://127.0.0.1:8888/lab && ok "guided notebook (Jupyter)" || bad "guided notebook (Jupyter)" "see results/logs/jupyter.log"
NB_URL="http://127.0.0.1:8888/lab/tree/notebooks/01_guided_lab.ipynb"
BROWSER_OK=""
if [ -n "$DISPLAY" ]; then   # Detach from XFCE terminal's process group: it ends with this script.
  setsid -f xdg-open "$NB_URL" </dev/null >>"$WS/results/logs/browser.log" 2>&1
  for _ in 1 2 3 4 5 6; do pgrep -u "$(id -u)" -x chrome >/dev/null || pgrep -u "$(id -u)" -x firefox >/dev/null && { BROWSER_OK=1; break; }; sleep 1; done
fi
echo
if [ -z "$FAILED" ]; then
  tl ready
  echo "READY"
  if [ -n "$BROWSER_OK" ]; then echo "  Part 1 (guided): follow the notebook in the browser that just opened ($NB_URL)"
  else echo "  Part 1 (guided): double-click the Jupyter Lab icon on the desktop, or open $NB_URL in the browser"; fi
  echo "  Part 2 (team challenge): VS Code on team/  +  this Terminal:"
  echo "      ./evaluate.sh    ./train.sh    ./reset.sh    ./submit.sh"
else
  tl start_failed
  echo "NOT READY: see the PROBLEM lines above, then run ./start.sh again or call an organizer."
  exit 1
fi
