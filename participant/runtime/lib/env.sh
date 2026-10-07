# Common settings for the workshop commands (sourced by start/reset/evaluate/train/submit).
RT="${VINCI_RUNTIME:-/opt/vinci-workshop}"
WS="${VINCI_WORKSPACE:-$(cd "$(dirname "$0")" && pwd)}"
[ -f /etc/vinci/workshop.env ] && source /etc/vinci/workshop.env
PY="${VINCI_PY:-${WS_PY:-python3}}"   # the Isaac Lab python, set in /etc/vinci/workshop.env
NB_BIN="${VINCI_NB_BIN:-$RT/nb-venv/bin}"
SERVICE_URL="${VINCI_SERVICE_URL:-http://127.0.0.1:8765}"
export VINCI_RUNTIME="$RT" VINCI_WORKSPACE="$WS" OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
unset MPLBACKEND   # Jupyter exports its inline backend: the simulator processes cannot load it
tl() { "$PY" "$RT/lib/timeline.py" "$@" >/dev/null 2>&1 || true; }
service_up() { [ "$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$SERVICE_URL/observe")" = "200" ]; }
PC_URL="${VINCI_PC_URL:-http://127.0.0.1:8766}"   # PC scene (learned RAM insertion), headless
pc_up() { curl -s -m 5 "$PC_URL/health" 2>/dev/null | grep -q '"ok": true'; }
