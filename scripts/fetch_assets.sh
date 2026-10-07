#!/bin/bash
# Download exactly the models/videos pinned by this checkout's RELEASE.json.
# Offline: ASSETS_FILE=/path/archive.tar.gz scripts/fetch_assets.sh
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mapfile -t META < <(python3 "$REPO_DIR/scripts/verify_assets.py" metadata "$REPO_DIR/RELEASE.json")
[ "${#META[@]}" = 2 ] || { echo "Invalid or missing RELEASE.json" >&2; exit 1; }
TAG=${META[0]}; NAME=${META[1]}
[ -z "${ASSETS_TAG:-}" ] || [ "$ASSETS_TAG" = "$TAG" ] || { echo "ASSETS_TAG conflicts with this checkout ($TAG)" >&2; exit 1; }
if [ -z "${ASSETS_FILE:-}" ]; then
  T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
  REPOSITORY=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("repository", ""))' "$REPO_DIR/RELEASE.json")
  # public release: plain HTTPS download, no GitHub account needed; gh is the fallback (private copies)
  if [ -n "$REPOSITORY" ] && curl -fsSL -o "$T/$NAME" "https://github.com/$REPOSITORY/releases/download/$TAG/$NAME"; then :
  elif command -v gh >/dev/null; then (cd "$REPO_DIR" && gh release download "$TAG" --pattern "$NAME" --dir "$T")
  else echo "Download failed: get $NAME from the $TAG release page, then ASSETS_FILE=/path/$NAME $0" >&2; exit 1; fi
  ASSETS_FILE="$T/$NAME"
fi
python3 "$REPO_DIR/scripts/verify_assets.py" install "$REPO_DIR/RELEASE.json" "$ASSETS_FILE" "$REPO_DIR/participant/runtime"
