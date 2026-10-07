#!/bin/bash
# Desktop icon "Isaac Sim": bring the simulator window to the front, or start it if it is not running.
WS="${1:-$HOME/vinci-workshop}"
if command -v wmctrl >/dev/null && wmctrl -l | grep -q "Isaac Sim"; then
  wmctrl -a "Isaac Sim"
else
  T="xfce4-terminal"; command -v $T >/dev/null || T="x-terminal-emulator"
  exec $T --hold --title="START WORKSHOP" --working-directory="$WS" -x "$WS/start.sh"
fi
