"""E001 as a REGRESSION GATE, not a one-off script.

Why this file exists, in hiring terms: 0 of 45 sampled postings say "batch
invariance", but 10 of 45 say numerics / correctness / reproducibility /
regression detection. A script that produced a number once is a blog post. A
pipeline that fails CI when the number moves is the thing those 10 postings are
actually asking for. Same code, different artifact class.

Why it exists in scientific terms: every claim E1 publishes is "the divergence
is X at gate g in dtype d." That claim has a shelf life of exactly one torch
release unless something re-checks it. This is that something.

------------------------------------------------------------------------------
FILL BASELINE YOURSELF, FROM YOUR OWN FIRST --run.

The tolerances are yours -- an agent must not choose them, because choosing them
IS the verifier-design skill DeltaProof is built on, and because a tolerance you
did not pick is a tolerance you cannot defend in an interview. Two questions to
answer before you type a number:

  1. What is the SMALLEST drift you would want CI to shout about? (Not the
     largest you can tolerate -- the smallest you would want to know about.)
  2. Is the right comparison an absolute band, or a RATIO to baseline? Argue it
     in one line in MASTERY_LEDGER.md, then encode that argument here.

Until BASELINE is filled these tests SKIP, so CI stays green today and becomes a
gate the moment you have your first run.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest
import torch

from scratch_llm.mastery.divergence import make_inputs, _rel
from scratch_llm.mastery.paths import chunked_wy

# (log_gate, dtype, solve_dtype) -> measured rel_err_o from YOUR first --run.
BASELINE: dict[tuple[float, str, str], float | None] = {
    (0.0,  "float64",  "auto"): None,
    (0.0,  "bfloat16", "auto"): None,
    (-1.0, "bfloat16", "auto"): None,
}

# Multiplicative band. YOURS. 2.0 means "fail if it moves by more than 2x".
TOL_RATIO: float | None = None

_FIXED = dict(T=512, dh=64, C=64, seed=0)
_DT = {"float64": torch.float64, "float32": torch.float32, "bfloat16": torch.bfloat16}


def _measure(log_gate: float, dtype: str, solve_dtype: str) -> float:
    """One point, exactly as the sweep computes it: oracle is fp64 chunked-vs-itself
    is NOT valid here, so we use the fp64 path as ground truth the same way sweep()
    does -- via the reference. Imported lazily so a missing reference.py skips
    rather than errors at collection time."""
    from scratch_llm.mastery.reference import recurrent_reference

    q, k, v, la, b = make_inputs(_FIXED["T"], _FIXED["dh"], _FIXED["dh"],
                                 log_gate, _FIXED["seed"])
    O_ref, _ = recurrent_reference(q, k, v, la, b)
    dt = _DT[dtype]
    sd = None if solve_dtype == "auto" else _DT[solve_dtype]
    O_c, _ = chunked_wy(q.to(dt), k.to(dt), v.to(dt), la.to(dt), b.to(dt),
                        chunk_size=_FIXED["C"], solve_dtype=sd)
    return _rel(O_c, O_ref)


@pytest.mark.parametrize("key", list(BASELINE.keys()))
def test_divergence_has_not_drifted(key):
    expected = BASELINE[key]
    if expected is None or TOL_RATIO is None:
        pytest.skip("BASELINE / TOL_RATIO not filled -- run --run, then fill them yourself")
    try:
        from scratch_llm.mastery.reference import recurrent_reference  # noqa: F401
    except NotImplementedError:
        pytest.skip("reference.py not implemented yet")
    got = _measure(*key)
    lo, hi = expected / TOL_RATIO, expected * TOL_RATIO
    assert lo <= got <= hi, (
        f"E001 REGRESSION at {key}: baseline {expected:.3e}, now {got:.3e} "
        f"({got / expected:.2f}x). Either torch changed under you, or the paths did. "
        f"Find out which BEFORE you update the baseline."
    )


def test_chunk_size_does_not_change_fp64_answer():
    """Free, no reference.py needed, and it is the invariant the whole result rests on:
    chunk size is an implementation detail, so in fp64 it must not move the answer.
    If this ever fails, every divergence number in the repo is measuring a bug."""
    q, k, v, la, b = make_inputs(256, 32, 32, -0.02, seed=1)
    ref, _ = chunked_wy(q, k, v, la, b, chunk_size=256)
    for C in (8, 32, 64, 128):
        O, _ = chunked_wy(q, k, v, la, b, chunk_size=C)
        rel = _rel(O, ref)
        assert rel < 1e-10, f"C={C} moved the fp64 answer, rel={rel:.3e}"
