#!/bin/bash
# Install the VINCI Embodied AI Workshop on your own Ubuntu machine with an NVIDIA RTX-class GPU.
# Run it as your normal user: it asks for sudo once. ~30-60 min the first time (~30 GB downloaded). Safe to re-run.
#
#   ./install/check_machine.sh          # first: is this machine supported?
#   ./install/install.sh                # then: install (Isaac Sim 5.1 + Isaac Lab 2.3.2 + the workshop)
#   cd ~/vinci-workshop && ./start.sh   # finally: start the simulators and open the guided notebook
#
# Options: --force (install despite a NOT SUPPORTED verdict)  --with-agents (also install the Claude Code and
# Codex command-line tools; you log in with your own account)  --no-vscode  --no-warmup
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
FORCE=0; AGENTS=0; EXTRA=()
for a in "$@"; do
  case "$a" in
    --force) FORCE=1 ;;
    --with-agents) AGENTS=1 ;;
    --no-vscode|--no-warmup) EXTRA+=("$a") ;;
    -h|--help) sed -n 2,10p "$0"; exit 0 ;;
    *) echo "unknown option $a (see --help)"; exit 2 ;;
  esac
done
[ "$(id -u)" != 0 ] || { echo "Run as your normal user, not with sudo: the script asks for sudo when needed."; exit 1; }
ME="$(id -un)"
ISAAC_VENV=/opt/isaac-venv
say() { echo; echo "==> $*"; }

say "1/5 machine check"
set +e; "$REPO/install/check_machine.sh"; verdict=$?; set -e
if [ "$verdict" = 2 ] && [ "$FORCE" = 0 ]; then echo "Stopping: this machine is not supported (use --force to try anyway)."; exit 2; fi

say "2/5 system packages (sudo)"
sudo true || { echo "This installation needs sudo rights."; exit 1; }
sudo DEBIAN_FRONTEND=noninteractive apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q build-essential curl wget git jq unzip rsync python3-venv python3-pip \
  libglu1-mesa libxrandr2 libxinerama1 libxcursor1 libvulkan1 xdg-utils wmctrl >/dev/null
grep -q '^OMNI_KIT_ACCEPT_EULA=' /etc/environment || echo 'OMNI_KIT_ACCEPT_EULA=YES' | sudo tee -a /etc/environment >/dev/null

say "3/5 workshop models and videos (~22 MB, pinned by RELEASE.json)"
if [ -f "$REPO/participant/runtime/policies/open_drawer_rl.pt" ] && [ -f "$REPO/participant/runtime/policies/insert_ram_student.pt" ]; then
  echo "already present"
else
  "$REPO/scripts/fetch_assets.sh"
fi

say "4/5 NVIDIA Isaac Sim 5.1 + Isaac Lab 2.3.2 in $ISAAC_VENV (~20-30 min the first time)"
echo "Using Isaac Sim means accepting the NVIDIA Omniverse License Agreement (see https://docs.isaacsim.omniverse.nvidia.com)."
sudo OWNER="$ME" ISAAC_VENV="$ISAAC_VENV" bash "$REPO/workshop/install_isaac.sh"

say "5/5 the workshop (runtime in /opt/vinci-workshop, your workspace in ~/vinci-workshop)"
sudo mkdir -p /etc/vinci
if [ ! -f /etc/vinci/workshop.env ]; then
  sudo cp "$REPO/workshop/machine/workshop.env.defaults" /etc/vinci/workshop.env
  echo "WS_PY=$ISAAC_VENV/bin/python" | sudo tee -a /etc/vinci/workshop.env >/dev/null
fi
sudo AGENT_CLIS="$AGENTS" bash "$REPO/workshop/install_workshop.sh" --user "$ME" --isaac-python "$ISAAC_VENV/bin/python" "${EXTRA[@]}"

echo
echo "Installed. To start the workshop:"
echo "    cd ~/vinci-workshop && ./start.sh"
echo "The guided notebook opens in your browser. The challenge instructions are in ~/vinci-workshop/CHALLENGE_SHEET.md."
