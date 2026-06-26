#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# provision.sh — runs ON THE POD. Reconstitutes the repo on fresh metal.
# Invoked by 01_launch.sh, but safe to re-run by hand (idempotent).
#
# Expects in env:
#   REPO=andreidhoang/scratch_llm
#   GITHUB_TOKEN=...   (used only to clone; not persisted to disk)
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="${REPO:?set REPO=owner/name}"
GITHUB_TOKEN="${GITHUB_TOKEN:?set GITHUB_TOKEN}"
DEST="${DEST:-/root/scratch_llm}"

echo "==> System deps (git, build tools)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git build-essential >/dev/null

echo "==> uv (fast Python package manager)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Clone / update ${REPO}"
if [[ -d "$DEST/.git" ]]; then
  git -C "$DEST" pull --ff-only
else
  # Token used inline for the clone, then the remote is rewritten to a
  # token-free URL so the secret isn't left sitting in .git/config.
  git clone "https://${GITHUB_TOKEN}@github.com/${REPO}.git" "$DEST"
  git -C "$DEST" remote set-url origin "https://github.com/${REPO}.git"
fi

echo "==> Create venv + install (core + gpu + dev)"
cd "$DEST"
uv venv --python 3.11
# torch wheel matches the CUDA in the base image; [gpu] pulls torch/transformers/datasets.
uv pip install -e ".[gpu,dev]"

echo "==> Smoke check"
uv run python - <<'PY'
import torch, scratch_llm
print("torch", torch.__version__, "cuda?", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

echo "==> Done. Repo at ${DEST}. Activate with:  cd ${DEST} && source .venv/bin/activate"
