#!/bin/bash
# Can this machine run the VINCI Embodied AI Workshop (NVIDIA Isaac Sim 5.1)? Read-only, ~5 s, no sudo.
#
#   ./install/check_machine.sh            # prints one line per check and a verdict
#   exit code: 0 = ready, 1 = ready with limits (warnings), 2 = not supported
set -uo pipefail
WARN=0; FAIL=0
ok()   { printf '  OK    %-14s %s\n' "$1" "$2"; }
warn() { printf '  WARN  %-14s %s\n' "$1" "$2"; WARN=1; }
fail() { printf '  NO    %-14s %s\n' "$1" "$2"; FAIL=1; }
echo "VINCI Embodied AI Workshop — machine check"

# Operating system: the installer is tested on Ubuntu 22.04 x86_64 (Isaac Sim 5.1 also lists 24.04).
arch=$(uname -m)
. /etc/os-release 2>/dev/null || true
os="${PRETTY_NAME:-unknown}"
if [ "$arch" != x86_64 ]; then fail "system" "$os on $arch: x86_64 required"
elif [ "${ID:-}" = ubuntu ] && [ "${VERSION_ID:-}" = 22.04 ]; then ok "system" "$os (tested)"
elif [ "${ID:-}" = ubuntu ] && [ "${VERSION_ID:-}" = 24.04 ]; then warn "system" "$os: supported by Isaac Sim, not tested with this workshop"
elif grep -qi microsoft /proc/version 2>/dev/null; then fail "system" "Windows WSL is not supported by Isaac Sim: use native Ubuntu 22.04"
else fail "system" "$os: Ubuntu 22.04 required"; fi

# GPU: NVIDIA with RT cores (RTX / L4 / L40 / A10...). A100, H100, V100 have no RT cores: Isaac Sim cannot render.
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
  fail "GPU" "no NVIDIA GPU or driver found (nvidia-smi): install the NVIDIA driver first"
  gpu=""; vram=0; drv=""
else
  IFS=',' read -r gpu vram drv < <(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -1)
  # cards sold as 24 GB report ~22.5-24 GiB: thresholds are in MiB, the display is rounded
  gpu=$(echo "$gpu" | xargs); vmib=${vram// /}; vram=$(( (vmib + 512) / 1024 )); drv=$(echo "$drv" | xargs)
  case "$gpu" in
    *A100*|*H100*|*H200*|*H800*|*A800*|*A30*|*V100*|*P100*|*B200*|*GB200*)
      fail "GPU" "$gpu: no RT cores, Isaac Sim cannot render on it (use RTX, L4, L40S, A10...)";;
    *RTX*|*" L4"*|*L40*|*A10*|*A40*|*A16*|*" A2"*|*Quadro*|*" T4"*)
      if [ "$vmib" -ge 22000 ]; then ok "GPU" "$gpu, $vram GB"
      elif [ "$vmib" -ge 11000 ]; then warn "GPU" "$gpu, $vram GB: works for the guided notebook; 24 GB recommended for training and both scenes"
      else fail "GPU" "$gpu, $vram GB: at least 12 GB of GPU memory needed"; fi
      case "$gpu" in *" T4"*) warn "GPU" "T4 is below Isaac Sim's minimum: expect slow rendering";; esac;;
    *) warn "GPU" "$gpu, $vram GB: not a known RTX-class GPU, Isaac Sim may refuse it";;
  esac
  # Isaac Sim 5.1 + this workshop: validated with driver 580; 595 crashed its RTX renderer at start-up in our test.
  major=${drv%%.*}
  if [ "${major:-0}" = 580 ]; then ok "driver" "NVIDIA $drv (tested)"
  elif [ "${major:-0}" -ge 590 ] 2>/dev/null; then fail "driver" "NVIDIA $drv: Isaac Sim 5.1 crashed with driver 595 in our test; install driver 580 (see README)"
  elif [ "${major:-0}" -ge 550 ] 2>/dev/null; then warn "driver" "NVIDIA $drv: the workshop is tested with 580; install 580 if Isaac Sim fails to start"
  else fail "driver" "NVIDIA $drv: too old for Isaac Sim 5.1, install driver 580 (see README)"; fi
fi

# Memory, disk, CPU
ram=$(awk '/MemTotal/{print int($2/1048576+0.5)}' /proc/meminfo)   # a "32 GB" machine reports ~30-31 GiB
if [ "$ram" -ge 30 ]; then ok "memory" "$ram GB"
elif [ "$ram" -ge 22 ]; then warn "memory" "$ram GB: 32 GB recommended (training may be slow)"
else fail "memory" "$ram GB: at least 24 GB needed, 32 GB recommended"; fi
free_gb() { df -Pk "$1" 2>/dev/null | awk 'NR==2{print int($4/1048576)}'; }
d_opt=$(free_gb /opt); d_home=$(free_gb "$HOME")
if [ "${d_opt:-0}" -ge 60 ] && [ "${d_home:-0}" -ge 20 ]; then ok "disk" "${d_opt} GB free for /opt, ${d_home} GB for your home"
elif [ "${d_opt:-0}" -ge 40 ]; then warn "disk" "${d_opt} GB free for /opt: 60 GB recommended (Isaac Sim ~25 GB + caches)"
else fail "disk" "${d_opt:-?} GB free for /opt: at least 40 GB needed"; fi
cpu=$(nproc)
if [ "$cpu" -ge 8 ]; then ok "CPU" "$cpu threads"; else warn "CPU" "$cpu threads: 8+ recommended (evaluations run in parallel)"; fi

# Display: the scenes open windows; without a desktop they run headless (fine for evaluations and the notebook).
if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then ok "display" "desktop session found"
else warn "display" "no desktop session: the simulator will run headless (no 3D window)"; fi
command -v sudo >/dev/null && ok "sudo" "available (needed once, for the installation)" || fail "sudo" "the installation needs sudo"

echo
if [ "$FAIL" = 1 ]; then echo "VERDICT: NOT SUPPORTED — fix the NO lines above, or use a cloud GPU machine (see docs/OTHER_CLOUD.md)"; exit 2
elif [ "$WARN" = 1 ]; then echo "VERDICT: READY WITH LIMITS — you can install (./install/install.sh); read the WARN lines"; exit 1
else echo "VERDICT: READY — next: ./install/install.sh"; exit 0; fi
