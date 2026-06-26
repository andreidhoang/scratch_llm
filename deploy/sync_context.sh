#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# sync_context.sh — runs on YOUR laptop. Refreshes the vendored PROJECT docs
# (deploy/context/workspace/) from the live parent-dir sources, so what the pod
# pulls matches what you actually edit. Run this before launching a pod after
# you've changed ../STRATEGY.md (the roadmap) / ../DELTA.md / the workspace README.
#
# Scope: this project's roadmap/context docs only — NOT your global ~/.claude
# setup (that's per-machine, not carried by this repo).
#
# It only copies; you review the diff and commit. Nothing is pushed automatically.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."                       # repo root: .../cs336/scratch_llm
WS="$(cd .. && pwd)"                           # the cs336 workspace (parent)

echo "==> Refresh parent workspace docs from ${WS}"
for f in STRATEGY.md DELTA.md README.md; do
  [[ -f "$WS/$f" ]] && cp -f "$WS/$f" "deploy/context/workspace/$f" && echo "    ✓ $f"
done

echo
echo "==> Diff vs last commit:"
git --no-pager diff --stat deploy/context/ || true
echo
echo "Review, then:  git add deploy/context && git commit -m 'context: refresh bundle' && git push"
