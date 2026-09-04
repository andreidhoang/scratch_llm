#!/usr/bin/env bash
# Fresh Vast.ai pod → working scratch_llm + Claude Code, in one command.
# Run AFTER: cloning this repo and `cd scratch_llm`. Idempotent — safe to re-run.
#
#   bash scripts/bootstrap-pod.sh
#
# What it does (each step explained in docs/VASTAI_BOOTSTRAP.md):
#   1. Python env from the pinned lockfile + the sm120/cu130 torch stack this repo was measured on.
#   2. Re-link the green-CI pre-commit hook (git hooks are NOT cloned — .git/ is per-checkout).
#   3. Restore Claude Code auto-memory from the in-repo snapshot to the live ~/.claude location.
#   4. Verify: torch+CUDA, the green gate, and print the current curriculum node to orient from.
# It does NOT: install Claude Code, or authenticate git/gh (those need your input — see the runbook).
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
say() { printf '\n\033[1m▶ %s\033[0m\n' "$1"; }

say "1/4 · Python env (uv venv + dev deps + torch cu130 for sm120 Blackwell)"
uv venv .venv --python "$(command -v python3.12 || command -v python3)"
uv pip install --python .venv/bin/python -e ".[dev]"
# sm120 (RTX PRO 4000 Blackwell / RTX 50xx) needs a CUDA >= 12.8 wheel; this repo measured
# torch 2.12.1+cu130. On a DIFFERENT GPU, drop the --index-url and let uv pick the default build.
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu130
# K2 KDA parity-gate oracle (docs/k3/K2_PROPOSAL_KDA.md §5): fla-core >= 0.4.0 ships chunk_kda /
# fused_recurrent_kda. Installed at bootstrap so the GPU parity day is execution, not setup.
uv pip install --python .venv/bin/python "fla-core>=0.4.0"

say "2/4 · Re-link green-CI pre-commit hook (.git/hooks is not version-controlled)"
ln -sf ../../.claude/hooks/green-ci-gate.sh .git/hooks/pre-commit
echo "linked .git/hooks/pre-commit -> ../../.claude/hooks/green-ci-gate.sh"

say "3/4 · Restore Claude Code auto-memory from the in-repo snapshot"
mem_dir="$(ls -d "$HOME"/.claude/projects/*/memory 2>/dev/null | head -1)"
if [ -z "$mem_dir" ]; then
  # derive the slug from the workspace root the way Claude Code does (path with / -> -)
  mem_dir="$HOME/.claude/projects/-workspace/memory"
fi
mkdir -p "$mem_dir"
# NOTE (31/08): the in-repo .claude/memory-snapshot/ mirror was deleted (stale duplicate of
# ~/.claude/projects/*/memory/). Nothing to restore — a fresh session self-orients from
# PLAN.md § TODAY -> bench/RESULTS.md -> MASTERY_LEDGER.md, which is the design intent anyway.
echo "memory: no in-repo snapshot (removed 31/08) — orient from PLAN.md § TODAY"

say "4/4 · Verify (torch+CUDA · green gate · current node)"
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-only'))"
.venv/bin/ruff check src tests >/dev/null 2>&1 && echo "ruff: clean" || echo "ruff: ISSUES (run: .venv/bin/ruff check src tests)"
.venv/bin/pytest -m "not gpu" -q >/dev/null 2>&1 && echo "pytest -m 'not gpu': GREEN" || echo "pytest: FAILURES (run: .venv/bin/pytest -m 'not gpu')"
say "Current curriculum node (orient before building):"
grep -m1 '▶ NOW' docs/KERNEL_MASTERY_SPEC.md 2>/dev/null || echo "(see PLAN.md + docs/KERNEL_MASTERY_SPEC.md §12.5)"

cat <<'EOF'

DONE. Still to do BY HAND (need your credentials — see docs/VASTAI_BOOTSTRAP.md):
  · git identity :  git config user.name "Huy Hoang Dang"; git config user.email "danghuy19990804@gmail.com"
  · GitHub auth  :  gh auth login   (then: git config credential.helper '!gh auth git-credential')
  · Claude Code  :  installed?  run `claude`  in this dir — CLAUDE.md + hooks auto-load.
EOF
