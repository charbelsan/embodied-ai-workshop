#!/bin/bash
# ./evaluate.sh [--episodes N] [--videos K] [--level training|dev]   score team/brain.py on practice episodes
source "${VINCI_RUNTIME:-/opt/vinci-workshop}/lib/env.sh"
cd "$WS" && exec "$PY" "$RT/lib/evaluate.py" "$@"
