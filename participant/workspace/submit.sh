#!/bin/bash
# Freeze code/models, evaluate DEV in the background, publish the best valid submitted score.
set -e
source "${VINCI_RUNTIME:-/opt/vinci-workshop}/lib/env.sh"
cd "$WS"
exec "$PY" "$RT/lib/submission.py" "$@"
