#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# sync_checkpoints.sh — pull big artifacts OFF the pod onto your laptop.
# Run on YOUR laptop. rsync is resumable and only sends diffs, so re-running
# mid-training is cheap.
#
# WHY this exists: checkpoints/datasets do not belong in git (the .gitignore
# excludes *.pt, *.safetensors, checkpoints/). The pod is ephemeral — if the
# host is preempted, anything not pulled (or pushed to a bucket) is gone. Run
# this periodically during long runs.
#
# Usage:
#   INSTANCE=12345 ./deploy/sync_checkpoints.sh
#   INSTANCE=12345 REMOTE_DIR=/root/cs336/scratch_llm/runs ./deploy/sync_checkpoints.sh
# ---------------------------------------------------------------------------
set -euo pipefail

INSTANCE="${INSTANCE:?set INSTANCE=<instance_id>}"
REMOTE_DIR="${REMOTE_DIR:-/root/cs336/scratch_llm/checkpoints}"
LOCAL_DIR="${LOCAL_DIR:-./_pod_artifacts/${INSTANCE}}"

URL=$(vastai ssh-url "$INSTANCE")          # ssh://root@host:port
HOSTPORT="${URL#ssh://}"
PORT="${HOSTPORT##*:}"
HOST="${HOSTPORT%:*}"

mkdir -p "$LOCAL_DIR"
echo "⬇️  rsync ${HOST}:${REMOTE_DIR}  ->  ${LOCAL_DIR}"
rsync -avzP -e "ssh -p ${PORT} -o StrictHostKeyChecking=accept-new" \
  "${HOST}:${REMOTE_DIR}/" "${LOCAL_DIR}/"
echo "✅ Pulled to ${LOCAL_DIR}"
