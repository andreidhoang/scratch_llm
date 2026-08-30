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
# 29/08: predictions moved OUT of MASTERY_LEDGER.md (markdown blanks are not pre-registration)
# and INTO tests/test_e001_regression.py::PREDICTED, where the commit timestamp proves ordering.
# Count the dicts, not the prose.
_reg="tests/test_e001_regression.py"
oracle="WRITTEN"
grep -q "NotImplementedError" src/scratch_llm/mastery/reference.py 2>/dev/null && oracle="STUB"
# match only unfilled dict ENTRIES ("): None"), never the "float | None" type annotation
preds="$(awk '/^PREDICTED/,/^}/' "$_reg" 2>/dev/null | grep -c '): None' || echo 0)"
base="$(awk '/^BASELINE/,/^}/' "$_reg" 2>/dev/null | grep -c '): None' || echo 0)"
grep -q '^TOL_RATIO.*None' "$_reg" 2>/dev/null && base=$((base + 1))
runs="$(ls results/*.json 2>/dev/null | wc -l | tr -d ' ')"

echo "scratch_llm — ONE repo, ONE vehicle | branch ${branch} | uncommitted ${dirty} | remote ${pub}"
echo "PLAN ▶ PLAN.md (repo root) is the ONLY plan — production-first, fork resolved (X) 26/08. Read it before building; deep detail: docs/KERNEL_MASTERY_SPEC.md (§9 = production operating system)."
echo "VISION ▶ 5 dòng → E1 map → review #45819 + claim #48613 → PR vLLM → E2/K2 → seat. Every task names its link (rule 7). Why-depth: CLAUDE.md North Star · wall-map artifact (5 min, VN)."

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
echo "E001 state — oracle: ${oracle} · PREDICTED unset: ${preds}/3 · BASELINE+TOL unset: ${base}/4 · result files: ${runs}"

# ---- hardware, COMPUTED — the repo asserted a GPU it does not have until 29/08 -------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU ▶ $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1) · ncu: $(command -v ncu >/dev/null 2>&1 && echo yes || echo NO)"
else
  echo "GPU ▶ NONE on this host (no nvidia-smi; nvcc/ncu/nsys absent too — no local compile gate). Every kernel/profile rung is RENTAL-gated — not '\$0 on the standing card'. 5090 KVM vms_enabled ≈\$0.33/hr is the ncu path; RunPod/Modal/Vast-docker are ncu-BLOCKED. Law: every rental names its measurement + consumer (PLAN.md)."
  echo "ARCH ▶ Route rungs by the KERNEL'S TARGET ARCH, never by the biggest card (PLAN.md § arch-routing, 30/08). sm_120 does NOT load on sm_90; a Triton recompile is a different kernel instance. 4 of 5 ncu-debt metrics in bench/ sat 57 days misfiled to 'the H100 day' — they are sm_120, \$0.33/hr. Same CC != same card: ledger rows are 70 SMs/0.551 TB/s, a 5090 is ~170 SMs/~1.8 TB/s — re-run R0 to re-anchor before comparing."
fi
[ -d "$HOME/Desktop/oss/fla" ] && echo "OSS ▶ ~/Desktop/oss/{fla,vllm} cloned — read upstream FIRST-PARTY, never from memory. Dispatch findings F1-F4: spec §12.2."

# ---- name the ONE next action from that state ---------------------------------------------------
if [ "$vis" = "true" ]; then
  echo "NEXT ▶ make the remote public. Nothing else today outranks it: legibility multiplies evidence, and it is currently 0."
elif [ "$oracle" = "STUB" ] || [ "$preds" -gt 0 ]; then
  echo "NEXT ▶ ONE commit: the ~5 lines of recurrent_reference (equation is in its docstring) + the 3 PREDICTED values in tests/test_e001_regression.py. Same commit = the timestamp proves prediction preceded measurement. Then: python -m experiments.e001_gate_sweep --self-test"
elif [ "$runs" -eq 0 ]; then
  echo "NEXT ▶ python -m experiments.e001_gate_sweep --run   (gate axis; separates H2 from H3)"
elif [ "$base" -gt 0 ]; then
  echo "NEXT ▶ fill BASELINE ×3 + TOL_RATIO from the run, plus the one-line tolerance argument. That turns a script into a regression gate — which is the artifact class the JDs ask for."
else
  echo "NEXT ▶ conditioning axis (--chunks / --beta-scale), then the cost axis (--solve-dtype, incl. tf32 emulation per F2), then the adjudicating review on vLLM #45819 + the harness claim-comment on #48613 (PLAN.md lane 1 — the clock is external)."
fi

# ---- the pre-registration seal: printed ONLY while it can still be violated ----------------------
# NARROWED 29/08: test_wy_identity.py imports chunked_wy ONLY -- it never touches reference.py, and
# its assertion is a BOUND (rel < 1e-10), printed only on failure. It leaks a bound, not a value.
if [ "$preds" -gt 0 ]; then
  echo "SEAL ▶ PREDICTED still has blanks. 'pytest tests/test_wy_identity.py' asserts rel < 1e-10 in fp64 — a bound overlapping prediction #1's range, so predict first. It does NOT import reference.py; a full 'pytest -m \"not gpu\"' sweep collects it and leaks only that bound."
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
