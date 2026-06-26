#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# provision_lectures.sh — runs ON THE POD. Clones the official CS336 repos
# (the spec + test oracle the repo's CLAUDE.md calls "../lectures/").
# These are PUBLIC, so no token needed. Idempotent.
#
# Usage:  provision_lectures.sh [DEST_DIR]   (default /root/cs336/lectures)
# ---------------------------------------------------------------------------
set -euo pipefail

DEST="${1:-/root/cs336/lectures}"
BASE="https://github.com/stanford-cs336"
ASSIGNMENTS=(assignment1-basics assignment2-systems assignment3-scaling
             assignment4-data assignment5-alignment)

clone_or_pull() { # url dir
  if [[ -d "$2/.git" ]]; then git -C "$2" pull --ff-only --quiet || true
  else git clone --depth 1 --quiet "$1" "$2"; fi
  echo "    ✓ $(basename "$2")"
}

mkdir -p "$DEST"
echo "==> CS336 lecture code -> ${DEST}"
clone_or_pull "${BASE}/lectures.git" "$DEST"
echo "==> Assignment scaffolds (the tests/adapters.py oracle)"
for a in "${ASSIGNMENTS[@]}"; do
  clone_or_pull "${BASE}/${a}.git" "${DEST}/${a}"
done
echo "==> Oracle ready at ${DEST}"
