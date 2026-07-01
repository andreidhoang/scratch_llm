#!/usr/bin/env bash
# SessionStart — CONTEXT-INJECTION hook (Lever 4 feeding Lever 1).
# stdout on exit 0 is added to the session context. Surface only DYNAMIC state the static CLAUDE.md
# cannot know — branch, uncommitted count, venv, the latest commits, and the current perf-curriculum
# node — so every session can "orient before you build" (CLAUDE.md) from real state, not assumption.
# Every line is paid once per session; each must out-earn its tokens (don't echo static CLAUDE.md).
set -uo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '(no git yet — run: git init)')"
dirty="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
venv=$([ -d .venv ] && echo 'ready' || echo 'missing → uv venv && source .venv/bin/activate')

echo "scratch_llm · CS336 from-scratch | branch: ${branch} | uncommitted: ${dirty} files | venv: ${venv}"
echo "green-CI is hook-enforced on git commit. | open the day with /standup, close with /eod (docs/OPERATING_RHYTHM.md)."

# Orient-before-you-build (CLAUDE.md): the DYNAMIC 'what just shipped / where we are' the static docs
# can't hold. This is the STARTING slice — read deeper (git log -15, PERF_PLAN, bench/RESULTS.md) before building.
recent="$(git log --oneline -5 2>/dev/null || true)"
if [ -n "$recent" ]; then
  echo "latest commits — analyze before proceeding:"
  echo "$recent" | sed 's/^/  /'
fi
node="$(grep -m1 '^Phase:' performance/PERF_PLAN.md 2>/dev/null || true)"
[ -n "$node" ] && echo "current node → ${node#Phase: } (source: performance/PERF_PLAN.md)"
exit 0
