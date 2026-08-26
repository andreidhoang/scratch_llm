#!/usr/bin/env bash
# SessionStart — CONTEXT-INJECTION hook (Lever 4 feeding Lever 1).
# stdout on exit 0 is added to the session context. Surface only DYNAMIC state the static CLAUDE.md
# cannot know. REWRITTEN 2026-08-14: state is now COMPUTED from the filesystem, never asserted, and
# the hook names THE ONE NEXT ACTION rather than describing the plan. Lines that stop mattering
# (the pre-registration seal) delete themselves once the state says they are moot.
set -uo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '(no git)')"
dirty="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"

# ---- THE ONE PROCESS METRIC: is the remote PUBLIC? A private commit scores zero. -----------------
vis="$(timeout 4 gh repo view --json isPrivate -q .isPrivate 2>/dev/null || echo unknown)"
case "$vis" in
  false) pub="PUBLIC ✔" ;;
  true)  pub="PRIVATE ✘ — the process metric is 0 until this changes: gh repo edit --visibility public --accept-visibility-change-consequences" ;;
  *)     pub="unknown (gh unavailable) — verify before claiming a public-change day" ;;
esac

# ---- E001 instrument state, computed ------------------------------------------------------------
oracle="WRITTEN"
grep -q "NotImplementedError" src/scratch_llm/mastery/reference.py 2>/dev/null && oracle="STUB"
blanks="$(grep -c '________' MASTERY_LEDGER.md 2>/dev/null || echo 0)"
preds="$(grep -c '1e-__' MASTERY_LEDGER.md 2>/dev/null || echo 0)"
runs="$(ls results/*.json 2>/dev/null | wc -l | tr -d ' ')"

echo "scratch_llm — ONE repo, ONE vehicle | branch ${branch} | uncommitted ${dirty} | remote ${pub}"

# Stale-lock detector. An interrupted git write leaves .git/*.lock behind and every later git
# call dies with "Unable to create '.git/index.lock'". Two were found on 2026-08-26 (19/08 21:24
# and 21/08 11:27) and had been silently blocking git since. Report, never auto-delete: deleting
# a lock a LIVE git process holds corrupts the index. Confirm no git is running, then remove.
_locks="$(find .git -maxdepth 2 -name '*.lock' 2>/dev/null)"
if [ -n "${_locks}" ]; then
  echo "⚠ STALE GIT LOCK ▶ $(echo "${_locks}" | tr '\n' ' ')"
  echo "   git writes are BLOCKED. Check nothing is live (ps ax | grep git), then: rm -f ${_locks}"
fi
echo "execution-mode=$(cat .claude/execution-mode 2>/dev/null || echo '?') · L2 is the DEFAULT posture (propose N variants, ranking hidden; human predicts; then measure)."
echo "SEALED — agents never write these four: mastery/reference.py · the measurement harness · RL loss math · verifier logic + tolerances. Everything else is delegable and SHOULD be delegated."
echo "E001 state — oracle: ${oracle} · ledger blanks: ${blanks} · predictions unwritten: ${preds} · result files: ${runs}"

# ---- name the ONE next action from that state ---------------------------------------------------
if [ "$vis" = "true" ]; then
  echo "NEXT ▶ make the remote public. Nothing else today outranks it: legibility multiplies evidence, and it is currently 0."
elif [ "$blanks" -gt 0 ] && [ "$oracle" = "STUB" ]; then
  echo "NEXT ▶ Row 000 derivations on paper (D1 intensity · D2 dependency chain · D3 what chunking buys/costs), THEN the ~5 lines of recurrent_reference, then --self-test."
elif [ "$oracle" = "STUB" ]; then
  echo "NEXT ▶ write recurrent_reference (~5 lines; equation is in its docstring), then: python -m experiments.e001_gate_sweep --self-test"
elif [ "$preds" -gt 0 ]; then
  echo "NEXT ▶ write the three predictions + falsifier in MASTERY_LEDGER.md Row 001. NO PREDICTION, NO RUN."
elif [ "$runs" -eq 0 ]; then
  echo "NEXT ▶ python -m experiments.e001_gate_sweep --run   (gate axis; separates H2 from H3)"
else
  echo "NEXT ▶ conditioning axis (--chunks / --beta-scale), then the cost axis (--solve-dtype), then figures + write-up + the #42960 comment."
fi

# ---- the pre-registration seal: printed ONLY while it can still be violated ----------------------
if [ "$preds" -gt 0 ]; then
  echo "SEAL ▶ do NOT run 'pytest tests/test_wy_identity.py' yet — it asserts fp64 agreement at log_gate=0.0, which IS prediction #1. Predict first, then it is free."
fi

# ---- verified operating envelope: cheaper to print than to rediscover at 23:00 -------------------
echo "ENVELOPE ▶ |log_gate| x chunk < 88, else fp32 underflows (a=exp(log_gate*C) -> 0, beta/a -> inf). C=256 needs |gate| <= 0.25. fp64 survives every cell: a NaN fp64 survives is UNDERFLOW, not a hypothesis."
echo "FINDING ▶ cond(T) is EXACTLY gate-invariant (measured spread 0.000e+00) — H1 cannot be tested on the gate axis; its knobs are --beta-scale and --chunks."
echo "DAY ▶ one binary externally-checkable outcome · predict before you measure · ship before 21:00 or the day scores 0 · close with one ledger row (predicted/measured/bound/root cause). Run /op."

# ---- one-repo tripwire (ERRATA-C C7) ------------------------------------------------------------
if [ -d "$HOME/Desktop/mastery_llm" ]; then
  echo "!! WARNING: ~/Desktop/mastery_llm exists again — a RETIRED duplicate oracle (ERRATA-C C7). Do not work in it."
fi

recent="$(git log --oneline -3 2>/dev/null || true)"
[ -n "$recent" ] && { echo "latest commits:"; echo "$recent" | sed 's/^/  /'; }
exit 0
