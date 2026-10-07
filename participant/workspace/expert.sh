#!/bin/bash
# Optional learning challenge: trusted measurement, no agent claims, no leaderboard.
set -e
source "${VINCI_RUNTIME:-/opt/vinci-workshop}/lib/env.sh"
cd "$WS"
exec "$NB_BIN/python" "$RT/lib/expert_evaluate.py" "$@"
