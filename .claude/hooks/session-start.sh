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
[ -n "$node" ] && echo "perf node → ${node#Phase: } (source: performance/PERF_PLAN.md)"
# Close-the-loop / frontier-ablation front (ADR-0018): the buildable next rung for a fresh session.
fnode="$(grep -m1 'Next-node:' docs/FRONTIER_2026_TASKSPEC.md 2>/dev/null | sed -E 's/.*Next-node: *//; s/ *-->.*//' || true)"
[ -n "$fnode" ] && echo "frontier node → ${fnode} (source: docs/FRONTIER_2026_TASKSPEC.md — START HERE block + §0)"
# Learning track (teach-back mastery): the next Bài to master so a /master session resumes seamlessly.
lnode="$(grep -m1 '^Learning-node:' docs/learning/PROGRESS.md 2>/dev/null | sed -E 's/^Learning-node: *//' || true)"
[ -n "$lnode" ] && echo "learning node → ${lnode} (source: docs/learning/PROGRESS.md — 89-Bài teach-back ledger; ✅=owned, don't re-derive)"
[ -n "$lnode" ] && echo "  teach via PRR loop (default): Navigator predicts COLD → run real code/number → reconcile only the gap → re-derive+draw → teach-back gate. NO monologue; open with a spaced recall Q. (CLAUDE.md §How we build)"
[ -n "$lnode" ] && echo "  hiring linkage (PRR step 6): each Bài earns a named interview gate + FOP trait + build-vs-know-it + scarce bucket → docs/learning/FRONTIER_HIRING_MAP.md. Hired on EXECUTION/shipped artifacts, not plans (FOP-1)."
exit 0
