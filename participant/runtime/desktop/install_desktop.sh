#!/bin/bash
# Put the workshop icons on the desktop of the current user.
#   install_desktop.sh [WORKSPACE]      (default: ~/vinci-workshop)
RT="$(cd "$(dirname "$0")/.." && pwd)"
WS="${1:-$HOME/vinci-workshop}"
DESK="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"; [ "$DESK" = "$HOME" ] && DESK="$HOME/Desktop"
mkdir -p "$DESK"
T="xfce4-terminal"; command -v $T >/dev/null || T="x-terminal-emulator"
icon() {  # file, title, command, icon
  f="$DESK/$1.desktop"
  printf '[Desktop Entry]\nVersion=1.0\nType=Application\nName=%s\nExec=%s\nIcon=%s\nTerminal=false\n' "$2" "$3" "$4" > "$f"
  chmod +x "$f"
  command -v gio >/dev/null && gio set "$f" metadata::trusted true 2>/dev/null
  command -v gio >/dev/null && gio set "$f" metadata::xfce-exe-checksum "$(sha256sum "$f" | cut -d' ' -f1)" 2>/dev/null
}
icon 01-start-workshop "START WORKSHOP" "$T --hold --title=\"START WORKSHOP\" --working-directory=$WS -x $WS/start.sh" media-playback-start
icon 02-isaac-sim "Isaac Sim" "$RT/desktop/isaac_window.sh $WS" applications-games
icon 03-jupyter "Jupyter Lab (guided notebook)" "xdg-open http://127.0.0.1:8888/lab/tree/notebooks/01_guided_lab.ipynb" accessories-text-editor
icon 04-vscode "VS Code (team)" "code --new-window $WS/team $WS/team/brain.py" com.visualstudio.code
icon 05-terminal "Terminal" "$T --working-directory=$WS" utilities-terminal
icon 06-results "Results" "xdg-open $WS/results" folder
icon 07-challenge-sheet "Challenge sheet" "xdg-open $WS/CHALLENGE_SHEET.pdf" x-office-document
icon 08-restart-workshop "RESTART WORKSHOP (from zero)" "$T --hold --title=\"RESTART WORKSHOP\" --working-directory=$WS -x $WS/reset.sh --all" view-refresh
echo "icons installed in $DESK"
