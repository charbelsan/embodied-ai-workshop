#!/bin/bash
# Base stack of a workshop machine, from a bare Ubuntu 22.04 x86_64 GPU instance (tested family: g6e, NVIDIA L40S).
#
#   sudo ./bootstrap_machine.sh            # ~40-60 min; reboots are handled by running it twice (it resumes)
#
# Installs, in order (each step is skipped when already done, so the script can be re-run safely):
#   1. system packages, then the NVIDIA data-center driver 580 (runfile, open kernel module, DKMS), pinned:
#      the kernel packages are held and unattended upgrades disabled so nothing replaces it
#   2. XFCE desktop + LightDM auto-login of the desktop user on the GPU's Xorg (screen blanking/DPMS disabled)
#   3. Amazon DCV server + web viewer, one console session owned by the desktop user, system authentication
#   4. Isaac Sim 5.1 + Isaac Lab 2.3.2 in a python 3.11 venv (installer of the EmbodiedSWE project, pinned commit)
#   5. the machine services of the workshop (Xorg GPU fix, warm-up, simulator service, health, reset, persistence)
# Then run ./install_workshop.sh (the workshop itself) and ../scripts/smoke_test.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DESKTOP_USER="${DESKTOP_USER:-ubuntu}"
DRIVER_VERSION="${DRIVER_VERSION:-580.178.04}"         # any 580.x data-center release (580.178.04 validated)
DRIVER_URL="${DRIVER_URL:-https://us.download.nvidia.com/tesla/$DRIVER_VERSION/NVIDIA-Linux-x86_64-$DRIVER_VERSION.run}"
DCV_URL="${DCV_URL:-https://d1uj6qtbmh3dt5.cloudfront.net/nice-dcv-ubuntu2204-x86_64.tgz}"
ISAAC_VENV="${ISAAC_VENV:-/opt/isaac-venv}"
EMBODIEDSWE_REPO="${EMBODIEDSWE_REPO:-https://github.com/EmbodiedSWE/EmbodiedSWE.git}"
EMBODIEDSWE_REF="${EMBODIEDSWE_REF:-d34837e}"
EMBODIEDSWE_DIR="${EMBODIEDSWE_DIR:-/opt/embodiedswe}"
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
say() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }
export DEBIAN_FRONTEND=noninteractive
UHOME="$(getent passwd "$DESKTOP_USER" | cut -d: -f6)"

# ---------------------------------------------------------------- 1. packages + NVIDIA driver 580
say "system packages"
apt-get update -q
apt-get install -y -q build-essential dkms "linux-headers-$(uname -r)" pkg-config curl wget git jq unzip rsync \
  python3-venv python3-pip xfce4 xfce4-terminal lightdm lightdm-gtk-greeter xserver-xorg x11-utils x11-xserver-utils \
  wmctrl xdg-utils mesa-utils imagemagick libglu1-mesa libxrandr2 libxinerama1 libxcursor1 libvulkan1 >/dev/null
apt-get purge -y -q xfce4-screensaver light-locker unattended-upgrades >/dev/null 2>&1 || true
apt-mark hold linux-aws linux-image-aws linux-headers-aws "linux-image-$(uname -r)" "linux-headers-$(uname -r)" >/dev/null 2>&1 || true
if ! command -v aws >/dev/null; then
  curl -fsSL -o /tmp/awscli.zip https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip && unzip -q -o /tmp/awscli.zip -d /tmp && /tmp/aws/install >/dev/null
fi
cur=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
if [[ "$cur" != 580.* ]]; then
  say "NVIDIA driver $DRIVER_VERSION (current: ${cur:-none})"
  printf 'blacklist nouveau\noptions nouveau modeset=0\n' > /etc/modprobe.d/blacklist-nouveau.conf
  if lsmod | grep -q '^nouveau'; then update-initramfs -u; say "nouveau is loaded: REBOOT, then run this script again"; exit 3; fi
  curl -fsSL -o /tmp/nvidia.run "$DRIVER_URL"
  sh /tmp/nvidia.run --silent --dkms -m=kernel-open --no-questions
  rm -f /tmp/nvidia.run
fi
nvidia-smi -L
nvidia-smi --query-gpu=driver_version --format=csv,noheader | grep -q '^580\.' || { echo "driver is not 580"; exit 1; }

# ---------------------------------------------------------------- 2. desktop on the GPU
say "Xorg on the GPU, desktop, auto-login"
[ -f /etc/X11/xorg.conf ] || nvidia-xconfig --preserve-busid --enable-all-gpus --connected-monitor=DFP-0 >/dev/null
grep -q '"DPMS" "false"' /etc/X11/xorg.conf || sed -i '/Section "Monitor"/a\    Option         "DPMS" "false"' /etc/X11/xorg.conf
mkdir -p /etc/X11/xorg.conf.d
printf 'Section "ServerFlags"\n    Option "BlankTime" "0"\n    Option "StandbyTime" "0"\n    Option "SuspendTime" "0"\n    Option "OffTime" "0"\nEndSection\n' \
  > /etc/X11/xorg.conf.d/10-no-blanking.conf
mkdir -p /etc/lightdm/lightdm.conf.d
printf '[Seat:*]\nautologin-user=%s\nautologin-user-timeout=0\nautologin-session=xfce\nuser-session=xfce\n' "$DESKTOP_USER" \
  > /etc/lightdm/lightdm.conf.d/50-workshop-autologin.conf
mkdir -p "$UHOME/.config/autostart"
printf '[Desktop Entry]\nType=Application\nName=No screen blanking\nExec=sh -c "xset s off -dpms s noblank"\n' > "$UHOME/.config/autostart/no-blanking.desktop"
chown -R "$DESKTOP_USER:$DESKTOP_USER" "$UHOME/.config"
systemctl set-default graphical.target >/dev/null
grep -q '^OMNI_KIT_ACCEPT_EULA=' /etc/environment || echo 'OMNI_KIT_ACCEPT_EULA=YES' >> /etc/environment
grep -q '^PYTHONUNBUFFERED=' /etc/environment || echo 'PYTHONUNBUFFERED=1' >> /etc/environment

# ---------------------------------------------------------------- 3. Amazon DCV
if ! dpkg -l nice-dcv-server >/dev/null 2>&1; then
  say "Amazon DCV"
  rm -rf /tmp/dcv && mkdir -p /tmp/dcv && curl -fsSL "$DCV_URL" | tar xz -C /tmp/dcv --strip-components=1
  wget -q -O /tmp/NICE-GPG-KEY https://d1uj6qtbmh3dt5.cloudfront.net/NICE-GPG-KEY && gpg --import /tmp/NICE-GPG-KEY >/dev/null 2>&1 || true
  apt-get install -y -q /tmp/dcv/nice-dcv-server_*.deb /tmp/dcv/nice-dcv-web-viewer_*.deb >/dev/null
  usermod -aG video dcv
fi
cat > /etc/dcv/dcv.conf <<EOF
[license]
[log]
[session-management]
create-session = true
[session-management/defaults]
[session-management/automatic-console-session]
owner = "$DESKTOP_USER"
[display]
target-fps = 30
[connectivity]
enable-quic-frontend = true
[security]
authentication = "system"
EOF
systemctl enable dcvserver >/dev/null

# ---------------------------------------------------------------- 4. Isaac Sim 5.1 + Isaac Lab 2.3.2
OWNER="$DESKTOP_USER" ISAAC_VENV="$ISAAC_VENV" EMBODIEDSWE_REPO="$EMBODIEDSWE_REPO" EMBODIEDSWE_REF="$EMBODIEDSWE_REF" \
  EMBODIEDSWE_DIR="$EMBODIEDSWE_DIR" bash "$HERE/install_isaac.sh"

# ---------------------------------------------------------------- 5. machine services of the workshop
say "workshop machine services"
install -m 755 "$HERE"/machine/vinci-* /usr/local/bin/
install -m 644 "$HERE"/machine/*.service /etc/systemd/system/
mkdir -p /etc/vinci /var/log/vinci-warmup
[ -f /etc/vinci/workshop.env ] || echo "WS_PY=$ISAAC_VENV/bin/python" > /etc/vinci/workshop.env
grep -q '^WS_PY=' /etc/vinci/workshop.env || echo "WS_PY=$ISAAC_VENV/bin/python" >> /etc/vinci/workshop.env
cat "$HERE/machine/workshop.env.defaults" | while IFS= read -r l; do
  k="${l%%=*}"; [ -z "$k" ] || [[ "$k" == \#* ]] || grep -q "^$k=" /etc/vinci/workshop.env || echo "$l" >> /etc/vinci/workshop.env
done
systemctl daemon-reload
systemctl enable vinci-xorg-fix.service vinci-warmup.service >/dev/null
say "web browser (the guided notebook opens in it, inside the DCV desktop)"
/usr/local/bin/vinci-install-browser
say "done. Reboot, then: sudo ./install_workshop.sh --isaac-python $ISAAC_VENV/bin/python"
