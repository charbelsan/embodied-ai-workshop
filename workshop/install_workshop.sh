#!/bin/bash
# Install the VINCI Embodied AI Workshop on a machine that already has the base stack
# (NVIDIA driver 580, Amazon DCV with a desktop session, Isaac Sim 5.1 + Isaac Lab 2.3.2 in a python venv).
#
#   sudo ./install_workshop.sh --isaac-python /path/to/isaac/venv/bin/python [options]
#
#   --user USER            desktop user of the DCV session (default: ubuntu)
#   --runtime DIR          read-only runtime (default: /opt/vinci-workshop)
#   --workspace DIR        the team's workspace (default: /home/USER/vinci-workshop)
#   --isaac-python PATH    python of the Isaac Lab venv (default: WS_PY of /etc/vinci/workshop.env)
#   --reset-workspace      replace an existing workspace (default: keep it)
#   --no-vscode            do not install VS Code
#   --no-warmup            skip the camera/shader cache warm-up (~3-5 min)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -d "$HERE/runtime" ] || HERE="$(cd "$HERE/../participant" && pwd)"   # repository layout: workshop/ next to participant/
USER_NAME=ubuntu; RT=/opt/vinci-workshop; WS=""; ISAAC_PY=""; RESET_WS=0; VSCODE=1; WARMUP=1
while [ $# -gt 0 ]; do
  case "$1" in
    --user) USER_NAME="$2"; shift ;;
    --runtime) RT="$2"; shift ;;
    --workspace) WS="$2"; shift ;;
    --isaac-python) ISAAC_PY="$2"; shift ;;
    --reset-workspace) RESET_WS=1 ;;
    --no-vscode) VSCODE=0 ;;
    --no-warmup) WARMUP=0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac; shift
done
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
HOME_DIR="$(getent passwd "$USER_NAME" | cut -d: -f6)"
WS="${WS:-$HOME_DIR/vinci-workshop}"
ENV_FILE=/etc/vinci/workshop.env
if [ -z "$ISAAC_PY" ] && [ -f "$ENV_FILE" ]; then ISAAC_PY="$(bash -c "source $ENV_FILE; echo \${WS_PY:-}")"; fi
[ -x "$ISAAC_PY" ] || { echo "Isaac Lab python not found: pass --isaac-python"; exit 1; }
say() { echo "[install] $*"; }
export OMNI_KIT_ACCEPT_EULA=YES
as_user() { sudo -u "$USER_NAME" -H env HOME="$HOME_DIR" "$@"; }

say "checks"
nvidia-smi -L >/dev/null || { echo "no GPU / driver"; exit 1; }
"$ISAAC_PY" -c "import isaaclab" </dev/null >/dev/null 2>&1 || say "warning: 'import isaaclab' failed outside the app launcher (usually fine)"
export DEBIAN_FRONTEND=noninteractive
apt-get install -y -q wmctrl xfce4-terminal xdg-utils curl python3-venv >/dev/null

for m in open_drawer_rl open_drawer_visual dagger_teacher insert_ram_student; do [ -f "$HERE/runtime/policies/$m.pt" ] || { echo "models missing in $HERE/runtime/policies: run scripts/fetch_assets.sh first"; exit 1; }; done
say "runtime -> $RT (read-only)"
rm -rf "$RT"; cp -r "$HERE/runtime" "$RT"
python3 -m venv "$RT/nb-venv"
"$RT/nb-venv/bin/pip" install -q --upgrade pip
"$RT/nb-venv/bin/pip" install -q "jupyterlab>=4,<5" ipykernel nbconvert numpy pillow matplotlib pyyaml "mcp>=1.10,<2"
mkdir -p "$RT/nb-venv/etc/jupyter/labconfig"
printf '{"disabledExtensions": {"@jupyterlab/apputils-extension:announcements": true}}\n' > "$RT/nb-venv/etc/jupyter/labconfig/page_config.json"
chown -R root:root "$RT"; chmod -R a+rX,go-w "$RT"; chmod 755 "$RT"/lib/*.sh "$RT"/desktop/*.sh

say "notebook kernel"
as_user "$RT/nb-venv/bin/python" -m ipykernel install --user --name vinci-nb --display-name "VINCI lab" >/dev/null

say "workspace -> $WS"
if [ -d "$WS" ] && [ "$RESET_WS" = 0 ]; then
  say "workspace exists: kept (use --reset-workspace to replace it)"
else
  rm -rf "$WS"; cp -r "$HERE/workspace" "$WS"; rm -f "$WS/results/.keep"
fi
chown -R "$USER_NAME:$USER_NAME" "$WS"; chmod +x "$WS"/*.sh

if [ "$VSCODE" = 1 ]; then
  say "VS Code"
  if ! command -v code >/dev/null; then
    curl -fsSL -o /tmp/vscode.deb "https://update.code.visualstudio.com/latest/linux-deb-x64/stable"
    apt-get install -y -q /tmp/vscode.deb >/dev/null; rm -f /tmp/vscode.deb
  fi
  as_user mkdir -p "$HOME_DIR/.config/Code/User"
  cat > "$HOME_DIR/.config/Code/User/settings.json" <<'JSON'
{
  "security.workspace.trust.enabled": false,
  "terminal.integrated.cwd": "__WS__",
  "workbench.startupEditor": "none",
  "workbench.welcomePage.walkthroughs.openOnInstall": false,
  "workbench.colorTheme": "Default Dark Modern",
  "workbench.tips.enabled": false,
  "chat.disableAIFeatures": true,
  "workbench.secondarySideBar.defaultVisibility": "hidden",
  "git.openRepositoryInParentFolders": "never",
  "telemetry.telemetryLevel": "off",
  "update.mode": "none",
  "extensions.ignoreRecommendations": true,
  "files.exclude": {"**/__pycache__": true}
}
JSON
  sed -i "s#__WS__#$WS#" "$HOME_DIR/.config/Code/User/settings.json"   # VS Code terminal opens where ./evaluate.sh lives, not in team/
  chown -R "$USER_NAME:$USER_NAME" "$HOME_DIR/.config/Code"
  as_user code --install-extension ms-python.python >/dev/null 2>&1 || say "python extension not installed (offline?): editing still works"
fi

say "robot tools for coding agents (MCP): $WS/.mcp.json (Claude Code) + ~/.codex/config.toml (Codex)"
sed -i "s#/opt/vinci-workshop#$RT#g; s#/home/ubuntu/vinci-workshop#$WS#g" "$WS/.mcp.json"
as_user mkdir -p "$HOME_DIR/.codex"
CODEX_CFG="$HOME_DIR/.codex/config.toml"
if ! grep -q '^\[mcp_servers.vinci-robot\]' "$CODEX_CFG" 2>/dev/null; then
  printf '\n[mcp_servers.vinci-robot]\ncommand = "%s"\nargs = ["%s"]\nenv = { VINCI_WORKSPACE = "%s" }\n' \
    "$RT/nb-venv/bin/python" "$RT/lib/mcp_server.py" "$WS" >> "$CODEX_CFG"
fi
chown -R "$USER_NAME:$USER_NAME" "$HOME_DIR/.codex"
if [ "${AGENT_CLIS:-1}" = 1 ]; then  # optional: the participants' own coding agents (they log in with their account)
  CLAUDE_BIN="$HOME_DIR/.local/bin/claude"  # the native installer puts it here, outside the PATH of an already open desktop session
  [ -x "$CLAUDE_BIN" ] || as_user bash -c 'curl -fsSL https://claude.ai/install.sh | bash' >/dev/null 2>&1 || say "Claude Code not installed (offline?)"
  if [ -x "$CLAUDE_BIN" ] && { [ -L /usr/local/bin/claude ] || [ ! -e /usr/local/bin/claude ]; }; then ln -sfn "$CLAUDE_BIN" /usr/local/bin/claude; fi
  if ! command -v codex >/dev/null; then
    command -v npm >/dev/null || { curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null 2>&1 && apt-get install -y -q nodejs >/dev/null 2>&1; }
    npm install -g @openai/codex >/dev/null 2>&1 || say "Codex CLI not installed (offline?)"
  fi
fi

say "desktop icons"
as_user bash "$RT/desktop/install_desktop.sh" "$WS"

say "$ENV_FILE"
mkdir -p /etc/vinci; touch "$ENV_FILE"
set_kv() {  # key value: replace or append, keep every other line (e.g. the persistence settings)
  local k="$1" v="$2"
  if grep -q "^$k=" "$ENV_FILE"; then sed -i "s#^$k=.*#$k=$v#" "$ENV_FILE"; else echo "$k=$v" >> "$ENV_FILE"; fi
}
# the organizers' reset kills the team's processes by name: include the PC scene's supervisor (it would restart the server)
if grep -q '^RESET_KILL_RE="' "$ENV_FILE" && ! grep -q 'pcgr_server_supervisor' "$ENV_FILE"; then
  sed -i 's/^RESET_KILL_RE="/RESET_KILL_RE="pcgr_server_supervisor|/' "$ENV_FILE"
fi
set_kv WS_DIR "$WS"
set_kv WS_PY "$ISAAC_PY"
set_kv FIRST_ENV_CMD "\"$RT/scripts/interactive.py --demo\""
set_kv SERVICE_CMD "\"$RT/scripts/skill_server.py --port 8765 --runs_dir $WS/results/brain_runs\""
set_kv SMOKE_NOTEBOOK "$WS/notebooks/01_guided_lab.ipynb"
set_kv SMOKE_CELLS 3
set_kv NB_BIN "$RT/nb-venv/bin"
set_kv GPU_JOB_CMD "\"$RT/scripts/challenge_eval.py --headless --cameras 1 --level training --solution baseline --episodes 1 --out /tmp/vinci_gpu_job\""
set_kv PERSIST_PATHS "\"$WS/results $WS/team /var/log/vinci-timings.log /var/log/vinci-warmup\""
chmod 644 "$ENV_FILE"

if [ "$WARMUP" = 1 ]; then
  say "warm-up: one camera episode (shader/camera caches), ~3-5 min"
  as_user env OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 VINCI_WORKSPACE="$WS" \
    timeout -k 20 900 "$ISAAC_PY" "$RT/scripts/challenge_eval.py" --headless --cameras 1 --level training \
    --solution baseline --episodes 1 --out /tmp/vinci_warmup </dev/null > /tmp/vinci_warmup.log 2>&1 \
    && say "warm-up done" || say "warm-up failed: see /tmp/vinci_warmup.log"
  rm -rf /tmp/vinci_warmup
  say "warm-up: PC scene (its own shaders), ~1-3 min"
  as_user env OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 bash -c '
    "$0" "$1/scripts/pcgr_skill_server.py" --port 8799 --runs_dir /tmp/vinci_warmup_pc </dev/null > /tmp/vinci_warmup_pc.log 2>&1 &
    p=$!
    for _ in $(seq 600); do curl -s -m 3 http://127.0.0.1:8799/health | grep -q "\"ok\": true" && break; kill -0 $p 2>/dev/null || break; sleep 1; done
    curl -s -m 3 http://127.0.0.1:8799/health | grep -q "\"ok\": true"; rc=$?
    kill $p 2>/dev/null; sleep 3; kill -9 $p 2>/dev/null; exit $rc' "$ISAAC_PY" "$RT" \
    && say "PC scene warm-up done" || say "PC scene warm-up failed: see /tmp/vinci_warmup_pc.log"
  rm -rf /tmp/vinci_warmup_pc
fi
say "done: runtime $RT, workspace $WS, icons on the desktop of $USER_NAME. Next, as $USER_NAME: smoke_test.sh"
