#!/usr/bin/env bash
# SessionStart — CONTEXT-INJECTION hook (Lever 4 feeding Lever 1).
# stdout on exit 0 is added to the session context. Surface only DYNAMIC state the static
# CLAUDE.md cannot know — branch, uncommitted count, venv presence. Keep it to ~2 lines:
# every line here is paid once per session, so it must out-earn its tokens.
set -uo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '(no git yet — run: git init)')"
dirty="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
venv=$([ -d .venv ] && echo 'ready' || echo 'missing → uv venv && source .venv/bin/activate')

echo "scratch_llm · CS336 from-scratch | branch: ${branch} | uncommitted: ${dirty} files | venv: ${venv}"
echo "green-CI is hook-enforced on git commit."
exit 0
