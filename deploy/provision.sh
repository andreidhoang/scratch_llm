#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# provision.sh — runs ON THE POD. Reconstitutes the FULL cs336 workspace on
# fresh metal so the pod's Claude Code agents have the same plan + context
# engineering as your laptop. Invoked by 01_launch.sh; safe to re-run.
#
# Reproduces this exact topology (so all the repo's ../ refs resolve unchanged):
#
#   /root/cs336/
#   ├── STRATEGY.md  DELTA.md  README.md   (vendored from the repo's context bundle)
#   ├── lectures/                          (the CS336 oracle — optional)
#   └── scratch_llm/                       (this repo; Claude auto-loads its CLAUDE.md)
#
# …and installs the global SuperClaude config to the pod's ~/.claude/ so
# PRINCIPLES.md / RULES.md / the learning methodology load every turn too.
#
# Expects in env:
#   REPO=andreidhoang/scratch_llm
#   GITHUB_TOKEN=...        (clone only; not persisted to disk)
#   SKIP_LECTURES=1         (optional — skip the 452M-ish oracle clone)
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="${REPO:?set REPO=owner/name}"
GITHUB_TOKEN="${GITHUB_TOKEN:?set GITHUB_TOKEN}"
WORKSPACE="${WORKSPACE:-/root/cs336}"
DEST="$WORKSPACE/scratch_llm"

echo "==> System deps (git, build tools)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git build-essential >/dev/null

echo "==> uv (fast Python package manager)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Clone / update ${REPO} into ${DEST}"
mkdir -p "$WORKSPACE"
if [[ -d "$DEST/.git" ]]; then
  git -C "$DEST" pull --ff-only
else
  git clone "https://${GITHUB_TOKEN}@github.com/${REPO}.git" "$DEST"
  git -C "$DEST" remote set-url origin "https://github.com/${REPO}.git"  # strip token from config
fi

echo "==> Lay out the vendored context bundle into the workspace topology"
# Parent-dir docs the repo's CLAUDE.md points to via ../STRATEGY.md etc.
cp -f "$DEST"/deploy/context/workspace/*.md "$WORKSPACE"/
# Global SuperClaude config: ~/.claude/CLAUDE.md @-imports PRINCIPLES.md & RULES.md
# (siblings), so all three land together.
mkdir -p "$HOME/.claude"
cp -f "$DEST"/deploy/context/claude_global/*.md "$HOME/.claude/"
echo "    workspace docs -> ${WORKSPACE}/ ; global config -> ${HOME}/.claude/"

echo "==> CS336 oracle (official Stanford repos — the test adapters)"
if [[ "${SKIP_LECTURES:-0}" == "1" ]]; then
  echo "    skipped (SKIP_LECTURES=1)"
else
  bash "$DEST/deploy/provision_lectures.sh" "$WORKSPACE/lectures"
fi

echo "==> Create venv + install (core + gpu + dev)"
cd "$DEST"
uv venv --python 3.11
uv pip install -e ".[gpu,dev]"

echo "==> Smoke check"
uv run python - <<'PY'
import torch, scratch_llm
print("torch", torch.__version__, "cuda?", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

cat <<EOF
==> Done. Parity established.
    Workspace : ${WORKSPACE}  (STRATEGY/DELTA/README + lectures + scratch_llm)
    Repo      : ${DEST}
    Global cfg: ${HOME}/.claude/{CLAUDE,PRINCIPLES,RULES}.md
    Start work:  cd ${DEST} && source .venv/bin/activate
    Launch Claude Code from ${DEST} so it auto-loads CLAUDE.md + the plan.
EOF
