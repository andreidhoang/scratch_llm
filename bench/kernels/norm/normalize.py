"""A2 Rung 3 roofline — Triton RMSNorm / LayerNorm on the last dim of ``(M, N)`` bf16.

These are **memory-bound** kernels: one HBM read of x, one write of y, an O(N) weight/bias read; no
reuse. The DoD (FOP-3, "the number is the deliverable") is the achieved fraction of the card's
measured HBM peak (~0.55 TB/s), and the answer to the rung's question — *what does LayerNorm's second
reduction cost over RMSNorm's one?*

The LADDER (naive → optimized), measured per shape as GB/s of the *ideal* traffic
``2·M·N·2B (+ weight/bias)`` — the useful bytes a perfect kernel would move, so redundant reads show
up as GB/s **below** the arms that avoid them:

  1. **torch-unfused** — the norm written as separate torch ops (``x*rsqrt(mean(x²)+eps)*w`` etc.).
     Inductor is *not* used, so x is streamed through HBM several times across the launches → the
     naive floor.
  2. **torch-fused** — ``F.rms_norm`` / ``F.layer_norm``, the vendor CUDA kernel (single fused pass):
     the "what good looks like" anchor.
  3. **triton-fused** — the owned Rung-3 kernel (whole row in one block, fp32 reduction, autotuned
     warps): one HBM read → should track the vendor kernel and approach HBM peak at large N.

The RMSNorm-vs-LayerNorm win is arm 3 rmsnorm vs arm 3 layernorm at each N: identical ideal traffic
(both read x once), so at large N — memory-bound — they measure nearly equal GB/s and the
one-reduction edge is a small compute/latency effect at small N. The table reports the measured %.

Honesty (raw-CUDA ladder vs Triton): the classic A2 memory-kernel ladder step —
**float4 vectorized loads** so each thread moves 128 bits and the row is read in coalesced 16-byte
transactions — is **compiler-managed** here: Triton's ``tl.load`` over a contiguous ``tl.arange``
block auto-vectorizes and coalesces (the block *is* the row, contiguous). We do not hand-write the
float4 path; a raw-CUDA variant would add explicit ``float4``/``__ldg`` and a warp-shuffle tree
reduction, and would be where an ncu ``sectors/request`` (ideal 4) check lives. See ncu_debt below.

Run on the GPU box:  PYTHONPATH=../../src python -m bench.kernels.norm.normalize   (--help for shape overrides)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

# bench/ on sys.path for the shared _harness (resolve upward to the dir holding _harness.py).
_BENCH_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "_harness.py").exists())
sys.path.insert(0, str(_BENCH_ROOT))
from _harness import Roofs, bench_ms, provenance_line, spread_pct  # noqa: E402

sys.path.insert(0, str(_BENCH_ROOT.parent / "src"))
from scratch_llm.kernels.norm.normalize import layernorm_triton, rmsnorm_triton  # noqa: E402

# The ncu metric the raw-CUDA float4 variant would inspect on the H100 day (counters blocked here).
NCU_DEBT = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / requests (sectors/request, ideal 4 = fully coalesced 128-bit loads)"


def _ideal_bytes(m: int, n: int, elem: int, *, bias: bool) -> float:
    """A perfect norm reads x once + writes y once (2·M·N) + the O(N) weight (+bias)."""
    extra = (2 if bias else 1) * n  # weight (+bias)
    return (2.0 * m * n + extra) * elem


def _correctness_gate(n: int = 1024) -> None:
    """Refuse to publish timings for a wrong kernel (a fast wrong kernel scores zero)."""
    x = torch.randn(8, n, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        rmsnorm_triton(x, w, eps=1e-6).float(),
        F.rms_norm(x, (n,), w, eps=1e-6).float(),
        rtol=1e-3,
        atol=1e-2,
    )
    torch.testing.assert_close(
        layernorm_triton(x, w, b, eps=1e-5).float(),
        F.layer_norm(x, (n,), w, b, eps=1e-5).float(),
        rtol=1e-3,
        atol=1e-2,
    )


def _rms_unfused(x: Tensor, w: Tensor, eps: float) -> Tensor:
    ms = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(ms + eps) * w.float()).to(x.dtype)


def _ln_unfused(x: Tensor, w: Tensor, b: Tensor, eps: float) -> Tensor:
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = (xf - mean).pow(2).mean(-1, keepdim=True)
    return ((xf - mean) * torch.rsqrt(var + eps) * w.float() + b.float()).to(x.dtype)


def _row(name: str, fn, m: int, n: int, elem: int, roofs: Roofs, *, bias: bool) -> None:
    med, lo, hi = bench_ms(fn)
    gbps = _ideal_bytes(m, n, elem, bias=bias) / (med * 1e-3) / 1e9
    pct = 100.0 * gbps * 1e9 / roofs.bw_bytes_s
    print(f"  {name:<16}{med:>9.4f}{spread_pct(med, lo, hi):>8.1f}{gbps:>10.1f}{pct:>8.1f}")


def _bench_norm(kind: str, m: int, n: int, roofs: Roofs) -> tuple[float, float]:
    """Print the 3-arm ladder for one norm at (m, n). Return (triton GB/s, triton ms)."""
    elem = 2  # bf16
    x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    bias = kind == "layernorm"

    print(f" {kind}  M={m} N={n}")
    print(f"  {'arm':<16}{'ms':>9}{'spread%':>8}{'GB/s':>10}{'%HBM':>8}")
    if kind == "rmsnorm":
        arms = [
            ("torch-unfused", lambda: _rms_unfused(x, w, 1e-6)),
            ("torch-fused", lambda: F.rms_norm(x, (n,), w, eps=1e-6)),
            ("triton-fused", lambda: rmsnorm_triton(x, w, eps=1e-6)),
        ]
        tri = lambda: rmsnorm_triton(x, w, eps=1e-6)  # noqa: E731
    else:
        arms = [
            ("torch-unfused", lambda: _ln_unfused(x, w, b, 1e-5)),
            ("torch-fused", lambda: F.layer_norm(x, (n,), w, b, eps=1e-5)),
            ("triton-fused", lambda: layernorm_triton(x, w, b, eps=1e-5)),
        ]
        tri = lambda: layernorm_triton(x, w, b, eps=1e-5)  # noqa: E731
    for name, fn in arms:
        _row(name, fn, m, n, elem, roofs, bias=bias)
    tri_ms = bench_ms(tri)[0]
    tri_gbps = _ideal_bytes(m, n, elem, bias=bias) / (tri_ms * 1e-3) / 1e9
    return tri_gbps, tri_ms


def run(ns: tuple[int, ...], m: int) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("norm bench requires CUDA.")
    _correctness_gate()  # raises before a single millisecond is reported

    roofs = Roofs.measure(torch.bfloat16)
    print(provenance_line("A2 R3 — RMSNorm / LayerNorm roofline"))
    print(f"# HBM peak {roofs.bw_bytes_s / 1e12:.3f} TB/s (measured) · M={m} · dtype bf16")
    print(
        "# ideal traffic = 2·M·N·2B (+weight/bias); GB/s below arms that avoid redundant reads = "
        "extra HBM passes\n"
    )

    print(
        f"{'N':>7} {'rms GB/s':>9} {'rms %HBM':>9} {'ln GB/s':>9} {'ln %HBM':>9} {'rms faster':>11}"
    )
    summary: list[tuple[int, float, float, float, float]] = []
    for n in ns:
        rms_gbps, rms_ms = _bench_norm("rmsnorm", m, n, roofs)
        ln_gbps, ln_ms = _bench_norm("layernorm", m, n, roofs)
        summary.append((n, rms_gbps, ln_gbps, rms_ms, ln_ms))
        torch.cuda.empty_cache()  # shared 24 GB GPU — release this shape before the next
        print()

    peak = roofs.bw_bytes_s / 1e9
    print(f"\n# SUMMARY (triton-fused arm) — HBM peak {peak:.0f} GB/s")
    print(
        f"{'N':>7} {'rms GB/s':>9} {'rms %HBM':>9} {'ln GB/s':>9} {'ln %HBM':>9} {'rms faster':>11}"
    )
    for n, rms_gbps, ln_gbps, rms_ms, ln_ms in summary:
        faster = 100.0 * (ln_ms - rms_ms) / ln_ms  # % less time RMSNorm takes vs LayerNorm
        print(
            f"{n:>7} {rms_gbps:>9.1f} {100.0 * rms_gbps * 1e9 / roofs.bw_bytes_s:>8.1f}% "
            f"{ln_gbps:>9.1f} {100.0 * ln_gbps * 1e9 / roofs.bw_bytes_s:>8.1f}% {faster:>10.1f}%"
        )
    print(f"\n# ncu_debt (blocked here; discharged on the H100 day): {NCU_DEBT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ns", type=int, nargs="+", default=[1024, 4096, 8192, 16384, 32768])
    p.add_argument("--m", type=int, default=4096, help="rows (kept modest — shared 24 GB GPU)")
    a = p.parse_args()
    run(ns=tuple(a.ns), m=a.m)
