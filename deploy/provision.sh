#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# provision.sh — runs ON THE POD. Reconstitutes the cs336 PROJECT workspace on
# fresh metal so the pod's Claude Code agents load this project's plan, context
# engineering, and roadmap. Invoked by 01_launch.sh; safe to re-run.
#
# Reproduces this exact topology (so all the repo's ../ refs resolve unchanged):
#
#   /root/cs336/
#   ├── STRATEGY.md  DELTA.md  README.md   (vendored from the repo's context bundle)
#   ├── lectures/                          (the CS336 oracle — optional)
#   └── scratch_llm/                       (this repo; Claude auto-loads its CLAUDE.md)
#
# Scope: PROJECT context only (repo CLAUDE.md + docs/ + .claude/ travel with git;
# the parent ../ docs are vendored). This does NOT touch the machine's global
# ~/.claude setup — that's yours to configure per machine.
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

# NOTE (31/08): deploy/context/workspace/ was deleted — it vendored duplicates of a June-era
# research bundle that was itself retired the same day. The plan a pod needs now travels in the
# repo itself: PLAN.md (the only plan) + CLAUDE.md + docs/. Nothing to lay out.

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
==> Done. Project context in place.
    Workspace : ${WORKSPACE}  (lectures + scratch_llm; the plan travels IN the repo: PLAN.md)
    Repo      : ${DEST}
    Start work:  cd ${DEST} && source .venv/bin/activate
    Launch Claude Code from ${DEST} so it auto-loads CLAUDE.md + docs/ (the plan).
EOF
