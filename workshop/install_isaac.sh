#!/bin/bash
# Isaac Sim 5.1 + Isaac Lab 2.3.2 in a python 3.11 venv, with the installer of the EmbodiedSWE project (pinned commit).
# Shared by the cloud team machines (bootstrap_machine.sh) and the single-machine install (install/install.sh).
#
#   sudo OWNER=<user> ./install_isaac.sh        # ~20-30 min, ~25 GB; skipped when the venv already imports isaaclab
set -euo pipefail
OWNER="${OWNER:-ubuntu}"
ISAAC_VENV="${ISAAC_VENV:-/opt/isaac-venv}"
EMBODIEDSWE_REPO="${EMBODIEDSWE_REPO:-https://github.com/EmbodiedSWE/EmbodiedSWE.git}"
EMBODIEDSWE_REF="${EMBODIEDSWE_REF:-d34837e}"
EMBODIEDSWE_DIR="${EMBODIEDSWE_DIR:-/opt/embodiedswe}"
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
say() { echo "[isaac $(date -u +%H:%M:%S)] $*"; }
if [ ! -x "$ISAAC_VENV/bin/python" ] || ! "$ISAAC_VENV/bin/python" -c "import isaaclab" </dev/null >/dev/null 2>&1; then
  say "Isaac Sim 5.1 + Isaac Lab 2.3.2 into $ISAAC_VENV (~20-30 min, ~25 GB)"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh >/dev/null
  [ -d "$EMBODIEDSWE_DIR/.git" ] || git clone -q "$EMBODIEDSWE_REPO" "$EMBODIEDSWE_DIR"
  git -C "$EMBODIEDSWE_DIR" checkout -q "$EMBODIEDSWE_REF"
  mkdir -p "$ISAAC_VENV"; chown -R "$OWNER:$OWNER" "$ISAAC_VENV" "$EMBODIEDSWE_DIR"
  # the simulator writes its logs and user config inside the venv: it belongs to the user who runs the workshop
  sudo -u "$OWNER" -H env COSIGEN_VENV_DIR="$ISAAC_VENV" OMNI_KIT_ACCEPT_EULA=YES \
    bash "$EMBODIEDSWE_DIR/scripts/bootstrap_isaaclab_5_1.sh" </dev/null
fi
sudo -u "$OWNER" -H env OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_VENV/bin/python" -c "import isaacsim, isaaclab; print('isaac ok')" </dev/null
