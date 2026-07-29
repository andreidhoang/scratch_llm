"""A2.1 roofline (backward) — Triton FA2 fwd+bwd vs SDPA-flash fwd+bwd, on the machine's peaks.

The FA2 *backward* (``kernels/attention/prefill/fa2.py`` ``TritonFlashAttention``) shipped
implemented-not-measured; FOP-3 says a kernel's DoD is a **profile**, not a green test. This is that
profile. Same hardened methodology as ``bench/flash_roofline.py`` (its forward twin): gate on
correctness first (a fast wrong kernel scores zero), time with CUDA events + L2 flush, report the
median with p20–p80 spread, anchor to the box's **measured** roofs, and print provenance + SM clock.

Predict-before-run (docs/design/L2_flash_attention_SPEC.md §4, extended to bwd): the Triton fwd+bwd
should reach ≥ 45% of SDPA-flash fwd+bwd at seq 4096, d=64, causal, bf16 on sm120 (predicted ~50%,
below the forward's ~55–65% because our bwd accumulates dQ via atomics — the known cost, see the
kernel docstring). Below 35% → ship the honest gap and name the bottleneck (atomic contention on dQ).

Work model (per (B,H), documented so the number is auditable):
  - forward FLOPs ≈ 4·N²·d (QKᵀ + PV, 2 FLOP/MAC), ×0.5 causal.
  - backward FLOPs ≈ 2.5× forward (recompute S, then dV, dP, dQ, dK — five N²·d matmuls vs the
    forward's two); this is the FA2-paper ratio. So fwd+bwd ≈ 3.5× forward FLOPs.
  - fwd+bwd HBM traffic ≈ 3× the forward's ideal 4·B·H·N·d (bwd also streams O, dO, L and writes
    dQ, dK, dV) — a lower bound; atomics on dQ add real traffic this model does not credit.

Run on the GPU box:  PYTHONPATH=../../src python -m bench.kernels.attention.fa2_bwd_roofline   (--help for shape/dtype overrides)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# bench/ on sys.path for the shared _harness (resolve upward to the dir holding _harness.py).
_BENCH_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "_harness.py").exists())
sys.path.insert(0, str(_BENCH_ROOT))
from _harness import Roofs, bench_ms, provenance_line, smi, spread_pct  # noqa: E402

sys.path.insert(0, str(_BENCH_ROOT.parent / "src"))
from scratch_llm.kernels.attention.prefill.fa2 import TritonFlashAttention  # noqa: E402

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _FLASH_CTX = lambda: sdpa_kernel(SDPBackend.FLASH_ATTENTION)  # noqa: E731
except Exception:  # pragma: no cover - very old torch
    import contextlib

    _FLASH_CTX = contextlib.nullcontext  # type: ignore

_FWD_FLOP_PER_BH = 4.0  # QKᵀ + PV, 2 FLOP/MAC
_BWD_MULT = 2.5  # FA2-paper backward:forward FLOP ratio
_BYTES_MULT = 3.0  # fwd+bwd HBM traffic vs the forward's 4·B·H·N·d ideal (lower bound)


def _fwdbwd_flops(b: int, h: int, n: int, d: int, causal: bool) -> float:
    fwd = _FWD_FLOP_PER_BH * b * h * n * n * d * (0.5 if causal else 1.0)
    return fwd * (1.0 + _BWD_MULT)


def _fwdbwd_bytes(b: int, h: int, n: int, d: int, elem: int) -> float:
    return _BYTES_MULT * 4.0 * b * h * n * d * elem


def _leaves(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fresh grad-tracking leaves each rep (so ``.backward()`` starts from a clean graph)."""
    return (
        q.detach().requires_grad_(True),
        k.detach().requires_grad_(True),
        v.detach().requires_grad_(True),
    )


def _correctness_gate(causal: bool) -> None:
    """Refuse to time a wrong backward: grads must match SDPA's autograd (fp32 then bf16)."""
    for dtype, tol in ((torch.float32, 2e-3), (torch.bfloat16, 2e-2)):
        base = [torch.randn(2, 4, 256, 64, device="cuda", dtype=dtype) for _ in range(3)]
        do = torch.randn(2, 4, 256, 64, device="cuda", dtype=dtype)
        qs, ks, vs = _leaves(*base)
        TritonFlashAttention.apply(qs, ks, vs, causal, dtype == torch.float32).backward(do)
        qr, kr, vr = _leaves(*base)
        with _FLASH_CTX():
            F.scaled_dot_product_attention(qr, kr, vr, is_causal=causal).backward(do)
        for g_tri, g_ref, name in ((qs, qr, "dq"), (ks, kr, "dk"), (vs, vr, "dv")):
            assert g_tri.grad is not None and g_ref.grad is not None
            torch.testing.assert_close(
                g_tri.grad.float(), g_ref.grad.float(), atol=tol, rtol=tol, msg=f"{name}@{dtype}"
            )


def run(
    seqs: tuple[int, ...] = (512, 1024, 2048, 4096, 8192),
    b: int = 2,
    h: int = 8,
    d: int = 64,
    causal: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    gate: bool = True,
) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("flash_bwd_roofline requires CUDA.")
    if gate:
        _correctness_gate(causal)  # raises before a single millisecond is reported

    elem = torch.tensor([], dtype=dtype).element_size()
    roofs = Roofs.measure(dtype)

    print(provenance_line("FA2-Triton fwd+bwd vs SDPA-flash fwd+bwd"))
    print(
        f"# shape B={b} H={h} d={d} causal={causal} dtype={dtype}  (gate={'pass' if gate else 'SKIPPED'})"
    )
    print(
        f"# measured roofs: compute {roofs.compute_flops_s / 1e12:6.1f} TFLOP/s | "
        f"HBM {roofs.bw_bytes_s / 1e9:6.1f} GB/s | ridge {roofs.ridge:5.1f} FLOP/byte  "
        f"(bwd FLOPs modeled as {_BWD_MULT}× fwd)"
    )
    print(
        f"{'seq':>6} {'tri_ms':>8} {'±%':>5} {'sdpa_ms':>8} {'triTF':>6} {'sdpaTF':>6} "
        f"{'%SDPA':>6} {'%roof':>6} {'bind':>4} {'smclk':>6}"
    )

    flagged: list[int] = []
    for n in seqs:
        q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=dtype) for _ in range(3))
        do = torch.randn(b, h, n, d, device="cuda", dtype=dtype)

        def tri_step(q=q, k=k, v=v, do=do) -> None:
            qs, ks, vs = _leaves(q, k, v)
            TritonFlashAttention.apply(qs, ks, vs, causal, False).backward(do)

        def sdpa_step(q=q, k=k, v=v, do=do) -> None:
            qs, ks, vs = _leaves(q, k, v)
            with _FLASH_CTX():
                F.scaled_dot_product_attention(qs, ks, vs, is_causal=causal).backward(do)

        med, lo, hi = bench_ms(tri_step)
        sdpa_ms = bench_ms(sdpa_step)[0]

        flops = _fwdbwd_flops(b, h, n, d, causal)
        tri_tf = flops / (med * 1e-3) / 1e12
        sdpa_tf = flops / (sdpa_ms * 1e-3) / 1e12
        attain, bind = roofs.attainable(flops, _fwdbwd_bytes(b, h, n, d, elem))
        pct_roof = 100.0 * (flops / (med * 1e-3)) / attain
        pct_sdpa = 100.0 * tri_tf / sdpa_tf

        spread = spread_pct(med, lo, hi)
        spread_s = f"{spread:.1f}{'!' if spread > 5.0 else ''}"
        if spread > 5.0:
            flagged.append(n)
        sm_clk = smi("clocks.sm")[0]

        print(
            f"{n:>6} {med:>8.3f} {spread_s:>5} {sdpa_ms:>8.3f} {tri_tf:>6.1f} {sdpa_tf:>6.1f} "
            f"{pct_sdpa:>5.1f}% {pct_roof:>5.1f}% {bind:>4} {sm_clk:>6.0f}"
        )

    if flagged:
        print(f"# WARNING: spread > 5% at seq {flagged} — clocks unstable; re-run warmed/pinned.")
    print(
        "# legend: fwd+bwd timed as one step (fresh leaves/rep); %SDPA relative to SDPA-flash "
        "fwd+bwd; %roof absolute (min(compute, intensity·BW)); bind = limiting roof. Our bwd "
        "accumulates dQ via atomics — expect it to trail the forward's %SDPA."
    )


def _parse_dtype(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seqs", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    p.add_argument("--b", type=int, default=2)
    p.add_argument("--h", type=int, default=8)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--no-causal", dest="causal", action="store_false")
    p.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16, help="bf16|fp16|fp32")
    p.add_argument(
        "--skip-gate", dest="gate", action="store_false", help="skip the correctness gate"
    )
    a = p.parse_args()
    run(seqs=tuple(a.seqs), b=a.b, h=a.h, d=a.d, causal=a.causal, dtype=a.dtype, gate=a.gate)
