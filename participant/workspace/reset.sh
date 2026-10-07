#!/bin/bash
# ./reset.sh          put the robot and the scene back to their start state
# ./reset.sh --team   also restore team/ to the starter files (your current team/ is kept in results/)
# ./reset.sh --all    start the whole workshop again (desktop icon RESTART WORKSHOP): stop your trainings and evaluations,
#                     restore the guided notebook and team/, reset both scenes, reopen the notebook. Nothing is deleted:
#                     your work moves to results/archive_<time>/. Add --yes to skip the question.
source "${VINCI_RUNTIME:-/opt/vinci-workshop}/lib/env.sh"
cd "$WS"

stop_jobs() {  # our trainings / evaluations / submissions and their children; never the two scene servers, never ourselves
  local me=$$ pids="" p
  local keep='skill_server\.py|pcgr_server_supervisor\.py|pcgr_skill_server\.py|mcp_server\.py|reset\.sh|/share/code/|(^|/)code( |$)'
  local jobs="$RT/scripts/|$RT/lib/(expert_)?evaluate\.py|$RT/lib/team_entry\.py|$WS/expert_verifier\.py|$WS/team/|(^|[ /])(train|evaluate|submit|expert)\.sh( |$)|ipykernel_launcher"
  while read -r p args; do
    [ "$p" = "$me" ] && continue
    echo "$args" | grep -qE "$keep" && continue
    echo "$args" | grep -qE "$jobs" && pids="$pids $p"
  done < <(ps -u "$(id -u)" -o pid=,args=)
  [ -z "$pids" ] && { echo "  no training or evaluation running"; return; }
  kill -TERM $pids 2>/dev/null
  for _ in $(seq 10); do sleep 1; p=""; for q in $pids; do kill -0 "$q" 2>/dev/null && p="$p $q"; done; [ -z "$p" ] && break; done
  [ -n "$p" ] && kill -KILL $p 2>/dev/null
  echo "  stopped:$(echo $pids | wc -w) process(es)"
}

stop_notebook() {  # Jupyter (its kernels die with it) and the browser, so no open tab writes the old notebook back
  local p
  p=$(ps -u "$(id -u)" -o pid=,args= | awk '/jupyter(-lab| lab)/ && /--port=8888/ {print $1}')
  [ -n "$p" ] && { kill -TERM $p 2>/dev/null; for _ in $(seq 10); do kill -0 $p 2>/dev/null || break; sleep 1; done; kill -KILL $p 2>/dev/null; }
  for b in chrome firefox; do  # main browser process only (clean exit, no "restore pages" bubble)
    p=$(ps -u "$(id -u)" -o pid=,args= | awk -v b="$b" '$2 ~ "/"b"$|^"b"$" && !/--type=/ {print $1}')
    [ -n "$p" ] && { kill -TERM $p 2>/dev/null; for _ in $(seq 8); do kill -0 $p 2>/dev/null || break; sleep 1; done; }
  done
  rm -f "$HOME"/.jupyter/lab/workspaces/*.jupyterlab-workspace 2>/dev/null
  echo "  notebook and browser closed"
}

if [ "$1" = "--all" ]; then
  if [ "$2" != "--yes" ]; then
    echo "RESTART WORKSHOP: stop your trainings, restore the guided notebook and team/ as at the start, reset both scenes."
    echo "Nothing is deleted: your work goes to results/archive_<time>/."
    read -r -p "Start again from zero? [y/N] " a
    case "$a" in y|Y|o|O|yes|oui) ;; *) echo "nothing changed"; exit 0 ;; esac
  fi
  trap '' HUP PIPE   # keep going if this runs in a Jupyter terminal that closes with Jupyter
  A="results/archive_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$A"
  echo "1/4 stopping trainings and evaluations"; stop_jobs
  echo "2/4 closing the notebook";               stop_notebook
  echo "3/4 restoring the starting files (your work: $A)"
  for f in results/*; do  # keep the running servers' folders, the logs, the timeline and the MCP call log in place
    case "$(basename "$f")" in logs|brain_runs|pc_runs|timeline.jsonl|mcp_calls.jsonl|.keep|archive_*) ;; *) mv "$f" "$A/" ;; esac
  done
  mv team "$A/team"; cp -r "$RT/templates/team" team; chmod -R u+w team
  mv notebooks "$A/notebooks"; mkdir -p notebooks; cp "$RT"/templates/notebooks/*.ipynb notebooks/; chmod -R u+w notebooks
  tl reset_all archive="$A"
  echo "4/4 resetting the scenes and reopening the notebook"
fi

if [ "$1" = "--team" ]; then
  B="results/team_backup_$(date +%Y%m%d_%H%M%S)"; cp -r team "$B"
  rm -rf team; cp -r "$RT/templates/team" team; chmod -R u+w team
  echo "team/ restored to the starter files (your previous version: $B)"; tl reset_team backup="$B"
fi
if service_up; then
  curl -s -m 120 -X POST "$SERVICE_URL/api/reset" >/dev/null && echo "scene reset" || echo "the simulator did not answer the reset"
else
  echo "the robot simulator is not running: ./start.sh"
fi
if pc_up; then  # PC scene: a new episode (RAM stick back in the gripper, case at a new pose)
  curl -s -m 120 -X POST -H 'Content-Type: application/json' -d "{\"run_id\": \"session_$(date +%H%M%S)\", \"goal\": \"reset\", \"reset\": true}" \
      "$PC_URL/run" >/dev/null && echo "PC scene reset" || echo "the PC scene did not answer the reset"
fi
tl reset
if [ "$1" = "--all" ]; then
  echo; exec "$WS/start.sh"
fi
