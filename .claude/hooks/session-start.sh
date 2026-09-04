#!/usr/bin/env bash
# SessionStart — dynamic state only, ≤ 10 lines. Rules live in CLAUDE.md and ~/Desktop/ladders/CLAUDE.md.
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
echo "scratch_llm · $(git rev-parse --abbrev-ref HEAD 2>/dev/null) · uncommitted $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') · unpushed $(git log origin/main..HEAD --oneline 2>/dev/null | wc -l | tr -d ' ')"
locks="$(find .git -maxdepth 1 -name '*.lock' 2>/dev/null)"
[ -n "$locks" ] && echo "⚠ stale git lock: $locks — confirm no git process is live (ps ax | grep git), then rm -f it"
grep -q NotImplementedError src/scratch_llm/mastery/reference.py 2>/dev/null && echo "oracle: STUB" || echo "oracle: written · E001 predictions unfilled: $(awk '/^PREDICTED/,/^}/' tests/test_e001_regression.py 2>/dev/null | grep -c '): None')/3 · results: $(ls results/*.json 2>/dev/null | wc -l | tr -d ' ')"
[ -f "$HOME/Desktop/ladders/experiments/CURRENT" ] && echo "current rung ▶ $(cat "$HOME/Desktop/ladders/experiments/CURRENT")  (workspace: ~/Desktop/ladders)"
command -v nvidia-smi >/dev/null 2>&1 && echo "GPU ▶ $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" || echo "GPU ▶ none here — measure on a rented box"
git log --oneline -3 2>/dev/null | sed 's/^/  /'
exit 0
