"""KDA parity against upstream flash-linear-attention (plan v5, artifact A1, unit K3/kda-v0).

Four tests, in the order they become runnable:

1. ``test_fla_chunk_matches_fla_recurrent`` -- FLA's own ``naive_chunk_kda`` against its own
   ``naive_recurrent_kda``. No code of ours is involved. It runs today, on a CPU, and it proves
   two things before anyone writes a kernel: the pinned ``oss/fla`` checkout is the reference we
   think it is, and the chunked (training) algorithm reproduces the recurrent (decode) one for
   the per-channel-gated delta rule that Kimi Linear / K3 use.

2. ``test_huy_kda_recurrent_matches_fla`` -- Huy's from-blank ``kda_recurrent`` against FLA's
   ``naive_recurrent_kda``. Skips, with the contract in the message, until the v0 exists.

3. ``test_huy_kda_chunk_matches_huy_recurrent_fp64`` -- Huy's ``kda_chunk`` against his own
   ``kda_recurrent``, both float64. Algebraically identical algorithms, so the bar is the
   structural one E001's self-test already uses (1e-10). Partial tail chunk included.

4. ``test_huy_kda_chunk_matches_fla_chunk`` -- Huy's ``kda_chunk`` against FLA's
   ``naive_chunk_kda`` (float32 inside; ``T % chunk_size == 0`` only, an upstream assert).

The unit is done when this file reports 9 passed, 0 skipped (nine parametrized cases across the four
tests; today: 3 passed, 6 skipped).

Both upstream functions cast their inputs to float32 internally (``naive.py``: ``.to(torch.float)``)
and cast the output back. So agreement with them is float32-level even with float64 inputs; the
float64 oracle for this family is Huy's own recurrence (test 3), and FLA is the *independent*
cross-check (tests 2, 4), which is the property that matters for a reference (AGENTS.md:
independence is a separate derivation, not a copy). ``oss/fla`` is loaded by file path so
``import fla`` (which pulls Triton) is never needed.

Contract for the v0 (pure functions, shapes exactly as FLA's naive, so the same inputs feed both):

    kda_recurrent(q, k, v, g, beta, scale=None, initial_state=None) -> (o, S)
    kda_chunk(q, k, v, g, beta, scale=None, initial_state=None, chunk_size=64) -> (o, S)
        q, k : [B, T, H, K]      v : [B, T, HV, V]      HV % H == 0 (GVA: q/k heads repeat)
        g    : [B, T, HV, K]     per-channel log-space decay, <= 0
        beta : [B, T, HV]        write strength in (0, 1)
        o    : [B, T, HV, V]     S : [B, HV, K, V]      scale defaults to K ** -0.5
        S_t = Diag(exp(g_t)) S_{t-1}
        S_t = S_t + beta_t * k_t (v_t - k_t^T S_t)^T          # delta rule: erase, then write
        o_t = (scale * q_t)^T S_t                              # read AFTER absorbing token t
    kda_chunk handles T % chunk_size != 0 (a partial tail chunk), as
    scratch_llm.mastery.paths.chunked_wy -- its scalar-gate diff partner -- does.

The equation is the whole spec. Write it from a blank file in ``src/scratch_llm/k3/core/kda.py``
after reading ``oss/fla/fla/ops/kda/naive.py`` once and closing it; then diff. This file is the
tests half of the contract -- whoever writes ``kda.py`` (the file-ownership boundary was removed
2026-09-22; see AGENTS.md) diffs against it.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

# Upstream's own bar for exactly this comparison (chunk vs recurrent, float inputs):
# oss/fla/tests/ops/test_kda.py:84  `assert_close("o", ref, tri, 0.005)` -- its get_err_ratio is
# RMS(err)/RMS(ref), the same norm ratio `_rel` computes. Reused, not chosen here.
FLA_UPSTREAM_RATIO = 0.005

# fp64 vs fp64 for two algebraically identical algorithms: the bar
# experiments/e001_gate_sweep.py::self_test already uses for chunked-vs-recurrent. Reused.
FP64_STRUCTURAL_REL = 1e-10

# Huy's. The v0 is float64, FLA's naive is float32 inside, so this is a float32-level bar and the
# one-line argument for it belongs in the K3/kda-v0 spec. Until it is set, tests 2 and 4 report
# the measured error and skip, so a run of this file is informative before it is a gate.
HUY_V0_REL_TOL: float | None = None

_KDA = Callable[..., tuple[torch.Tensor, torch.Tensor]]


def _fla_kda_naive() -> ModuleType:
    """Load oss/fla/fla/ops/kda/naive.py by path; skip cleanly if the checkout or einops is absent."""
    root = os.environ.get("LADDERS_OSS")
    oss = Path(root) if root else Path(__file__).resolve().parents[1].parent / "ladders" / "oss"
    path = oss / "fla" / "fla" / "ops" / "kda" / "naive.py"
    if not path.exists():
        pytest.skip(f"no fla checkout at {path} (set LADDERS_OSS or run infra/oss.sh)")
    if importlib.util.find_spec("einops") is None:
        pytest.skip("einops missing: uv pip install --python .venv/bin/python einops")
    spec = importlib.util.spec_from_file_location("_fla_kda_naive", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _huy(name: str) -> _KDA:
    """Huy's v0, resolved at runtime: the module may be the deleted template (ImportError) or
    exist without the function yet (no attribute). Either way: skip and name the contract."""
    try:
        fn = getattr(importlib.import_module("scratch_llm.k3.core.kda"), name, None)
    except ImportError:
        fn = None
    if fn is None:
        pytest.skip(
            f"no scratch_llm.k3.core.kda.{name} yet -- contract is in this file's docstring"
        )
    return fn


def _inputs(
    B: int, T: int, H: int, HV: int, K: int, V: int, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    """Float64 inputs in FLA's layout. k is L2-normalised so the delta-rule eraser is a projection
    (the same argument as scratch_llm.mastery.divergence.make_inputs); g is per-channel and <= 0."""
    gen = torch.Generator().manual_seed(seed)
    d = torch.float64
    q = torch.randn(B, T, H, K, generator=gen, dtype=d)
    k = torch.randn(B, T, H, K, generator=gen, dtype=d)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, T, HV, V, generator=gen, dtype=d)
    g = F.logsigmoid(torch.randn(B, T, HV, K, generator=gen, dtype=d))
    beta = torch.sigmoid(torch.randn(B, T, HV, generator=gen, dtype=d))
    h0 = torch.randn(B, HV, K, V, generator=gen, dtype=d)
    return q, k, v, g, beta, h0


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    return ((a64 - b64).norm() / b64.norm()).item()


def _skip_or_assert_vs_fla(eo: float, es: float, what: str) -> None:
    if HUY_V0_REL_TOL is None:
        pytest.skip(
            f"measured: {what} o {eo:.3e}, S {es:.3e} vs FLA (float32 inside). "
            "Set HUY_V0_REL_TOL with its one-line argument in the K3/kda-v0 spec."
        )
    assert eo < HUY_V0_REL_TOL and es < HUY_V0_REL_TOL, (
        f"{what} vs FLA: o {eo:.3e}, S {es:.3e} (bar {HUY_V0_REL_TOL})"
    )


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "K", "V", "chunk"),
    [
        (1, 64, 1, 1, 16, 16, 64),  # one chunk, no GVA: the algebra alone
        (2, 64, 2, 4, 16, 16, 16),  # four chunks + GVA (G=2): inter-chunk carry and head repeat
        (1, 128, 1, 2, 32, 16, 32),  # K != V, four chunks
    ],
)
def test_fla_chunk_matches_fla_recurrent(
    B: int, T: int, H: int, HV: int, K: int, V: int, chunk: int
) -> None:
    fla = _fla_kda_naive()
    q, k, v, g, beta, h0 = _inputs(B, T, H, HV, K, V)
    o_r, S_r = fla.naive_recurrent_kda(q, k, v, g, beta, initial_state=h0, output_final_state=True)
    o_c, S_c = fla.naive_chunk_kda(
        q, k, v, g, beta, initial_state=h0, output_final_state=True, chunk_size=chunk
    )
    eo, es = _rel(o_c, o_r), _rel(S_c, S_r)
    assert eo < FLA_UPSTREAM_RATIO and es < FLA_UPSTREAM_RATIO, (
        f"fla chunk vs recurrent disagree: o {eo:.3e}, S {es:.3e} (bar {FLA_UPSTREAM_RATIO})"
    )


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "K", "V"),
    [(1, 37, 1, 1, 8, 8), (2, 64, 2, 4, 16, 16)],  # odd T: no chunk assumption in a recurrence
)
def test_huy_kda_recurrent_matches_fla(B: int, T: int, H: int, HV: int, K: int, V: int) -> None:
    kda_recurrent = _huy("kda_recurrent")
    fla = _fla_kda_naive()
    q, k, v, g, beta, h0 = _inputs(B, T, H, HV, K, V, seed=1)
    o_f, S_f = fla.naive_recurrent_kda(q, k, v, g, beta, initial_state=h0, output_final_state=True)
    o_h, S_h = kda_recurrent(q, k, v, g, beta, initial_state=h0)
    _skip_or_assert_vs_fla(_rel(o_h, o_f), _rel(S_h, S_f), "v0 recurrent")


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "K", "V", "chunk"),
    [(1, 37, 1, 1, 8, 8, 16), (2, 64, 2, 4, 16, 16, 16)],  # 37 = 2 full chunks + a 5-token tail
)
def test_huy_kda_chunk_matches_huy_recurrent_fp64(
    B: int, T: int, H: int, HV: int, K: int, V: int, chunk: int
) -> None:
    """The fp64 self-check: chunked and recurrent are the same algebra, so they must agree to
    machine precision times O(T). If this fails and test 2 passes, the chunk derivation is wrong
    (E001's self-test names the two usual suspects: o_t read from S_{t-1}; erase after write)."""
    kda_recurrent, kda_chunk = _huy("kda_recurrent"), _huy("kda_chunk")
    q, k, v, g, beta, h0 = _inputs(B, T, H, HV, K, V, seed=2)
    o_r, S_r = kda_recurrent(q, k, v, g, beta, initial_state=h0)
    o_c, S_c = kda_chunk(q, k, v, g, beta, initial_state=h0, chunk_size=chunk)
    eo, es = _rel(o_c, o_r), _rel(S_c, S_r)
    assert eo < FP64_STRUCTURAL_REL and es < FP64_STRUCTURAL_REL, (
        f"v0 chunk vs v0 recurrent (fp64): o {eo:.3e}, S {es:.3e} (bar {FP64_STRUCTURAL_REL})"
    )


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "K", "V", "chunk"),
    [(1, 64, 1, 1, 16, 16, 64), (1, 128, 1, 2, 32, 16, 32)],  # T % chunk == 0: FLA's assert
)
def test_huy_kda_chunk_matches_fla_chunk(
    B: int, T: int, H: int, HV: int, K: int, V: int, chunk: int
) -> None:
    kda_chunk = _huy("kda_chunk")
    fla = _fla_kda_naive()
    q, k, v, g, beta, h0 = _inputs(B, T, H, HV, K, V, seed=3)
    o_f, S_f = fla.naive_chunk_kda(
        q, k, v, g, beta, initial_state=h0, output_final_state=True, chunk_size=chunk
    )
    o_h, S_h = kda_chunk(q, k, v, g, beta, initial_state=h0, chunk_size=chunk)
    _skip_or_assert_vs_fla(_rel(o_h, o_f), _rel(S_h, S_f), "v0 chunk")
