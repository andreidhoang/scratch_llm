"""A4 Rungs 2+3 — MEASURE the existing Triton FA2 forward on THIS sm120 Blackwell card.

This does NOT rebuild FA2. It benchmarks the shipped `flash_attention_triton_forward`
(`kernels/attention/prefill/fa2.py`, the autotuned FA2 Algorithm-1 forward) against
`F.scaled_dot_product_attention` (SDPA, flash backend) on the standing RTX PRO 4000 Blackwell
(sm120, 70 SMs), and reports the four things the A4 rung asks for:

  1. CORRECTNESS — the Triton output equals SDPA to <1e-2 (bf16) / <1e-3 (fp32), causal and
     non-causal. A wrong kernel scores zero; we gate before timing (same discipline as
     flash_roofline).
  2. ROOFLINE placement — FA FLOPs = 4·B·H·N²·d (halved for causal, since flash never visits the
     upper triangle). We report the achieved TFLOP/s, its **% of the 72 TF/s measured bf16 peak**
     (the compute roof of this card), and **% of SDPA time** — the repo's prior 4090 number was
     53% of SDPA; this gets the sm120 number.
  3. NO OOM at large N — the whole point of flash vs the R0 naive baseline. Naive materializes the
     B·H·N×N score matrix (O(N²) HBM); FA keeps the working set in SMEM (O(1)) and streams O(N)
     HBM. We measure peak memory for both across the sweep: naive grows quadratically and OOMs;
     FA stays linear and runs 8K fine.
  4. ONE variant — we demonstrate BOTH consequences the rung offers:
       * CAUSAL: causal halves the (i,j) pairs → ~2× less compute; we measure the causal/non-causal
         time ratio at each N.
       * GQA (n_kv_heads < n_heads, repeat_interleave): the KV cache shrinks by h/h_kv while FA
         compute is unchanged (K,V are broadcast up to h query heads before the matmul). We report
         the KV-memory saving and confirm the runtime is unaffected.

Measurement methodology is the shared `_harness` (CUDA-event timing, per-rep L2 flush, median with
p20–p80 spread, peaks measured on THIS GPU). ncu is blocked in this unprivileged container, so the
absolute anchor is %-of-measured-peak, not an SM-level profile (ncu-debt).

Run on the GPU box:  PYTHONPATH=../../src python -m bench.kernels.attention.fa2_sm120   (--help for shape/dtype overrides)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# bench/ on sys.path for the shared _harness (resolve upward to the dir holding _harness.py).
_BENCH_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "_harness.py").exists())
sys.path.insert(0, str(_BENCH_ROOT))
from _harness import Roofs, bench_ms, provenance_line, smi, spread_pct  # noqa: E402

sys.path.insert(0, str(_BENCH_ROOT.parent / "src"))
from scratch_llm.kernels.attention.prefill.fa2 import flash_attention_triton_forward  # noqa: E402

# The card's advertised/stat-sheet bf16 tensor peak, used as the fixed roofline anchor the rung
# names ("% of the 72 TF/s bf16 peak"). Roofs.measure() confirms it live (~72.3 TF/s cuBLAS GEMM).
PEAK_BF16_TFLOPS = 72.0

# Pin SDPA to the flash backend so "%SDPA" compares against ONE kernel across the whole sweep.
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _FLASH_CTX = lambda: sdpa_kernel(SDPBackend.FLASH_ATTENTION)  # noqa: E731
except Exception:  # pragma: no cover - very old torch
    import contextlib

    _FLASH_CTX = contextlib.nullcontext  # type: ignore


def _attn_flops(b: int, h: int, n: int, d: int, causal: bool) -> float:
    # fwd ≈ 4·B·H·N²·d (QKᵀ then PV, 2 FLOP/MAC); causal attends ~half the (i,j) pairs.
    return 4.0 * b * h * n * n * d * (0.5 if causal else 1.0)


def _naive_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
) -> torch.Tensor:
    """The R0 baseline flash replaces: materialize the full B·H·N×N score matrix in HBM. This is the
    O(N²)-memory kernel whose OOM at large N is the reason flash exists."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,N,N) — the quadratic tensor
    if causal:
        n = q.shape[-2]
        mask = torch.triu(torch.ones(n, n, device=q.device, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.matmul(s.softmax(dim=-1), v)


# --------------------------------------------------------------------------------------------------
# Correctness gate — refuse to publish timings for a wrong kernel.
# --------------------------------------------------------------------------------------------------
def _correctness_gate() -> None:
    for causal in (False, True):
        # fp32 vs SDPA (default/math backend — flash SDPA is bf16/fp16 only on sm120), bit-faithful
        # matmuls (allow_tf32=False) at a tractable shape — <1e-3.
        q, k, v = (torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float32) for _ in range(3))
        o_tri, _ = flash_attention_triton_forward(q, k, v, is_causal=causal, allow_tf32=False)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        torch.testing.assert_close(o_tri, ref, atol=1e-3, rtol=1e-3)
        # bf16 vs SDPA-flash at the benchmark precision — <1e-2.
        q, k, v = (
            torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        )
        o_tri, _ = flash_attention_triton_forward(q, k, v, is_causal=causal)
        with _FLASH_CTX():
            ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        torch.testing.assert_close(o_tri.float(), ref.float(), atol=1e-2, rtol=1e-2)
    print("# correctness gate: PASS  (Triton == SDPA  <1e-3 fp32, <1e-2 bf16, causal + non-causal)")


def _peak_mem_mb(fn) -> float:
    """Forward peak allocated MB for `fn` (or math.inf if it OOMs)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        fn()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1 << 20)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return math.inf


# --------------------------------------------------------------------------------------------------
def _roofline_sweep(seqs, b, h, d, dtype, roofs) -> None:
    print("\n## Roofline placement — FA2-Triton vs SDPA-flash (causal + non-causal)")
    print(
        f"# FLOPs=4·B·H·N²·d (causal halved) | peak anchor {PEAK_BF16_TFLOPS:.0f} TF/s bf16 "
        f"(measured {roofs.compute_flops_s / 1e12:.1f}) | HBM {roofs.bw_bytes_s / 1e9:.0f} GB/s"
    )
    print(
        f"{'seq':>6} {'causal':>7} {'tri_ms':>8} {'±%':>5} {'sdpa_ms':>8} "
        f"{'triTF':>6} {'sdpaTF':>7} {'%peak':>6} {'%SDPA':>6} {'smclk':>6}"
    )
    for causal in (False, True):
        for n in seqs:
            q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=dtype) for _ in range(3))
            med, lo, hi = bench_ms(
                lambda q=q, k=k, v=v, c=causal: flash_attention_triton_forward(q, k, v, is_causal=c)
            )
            with _FLASH_CTX():
                sdpa_ms = bench_ms(
                    lambda q=q, k=k, v=v, c=causal: F.scaled_dot_product_attention(
                        q, k, v, is_causal=c
                    )
                )[0]
            flops = _attn_flops(b, h, n, d, causal)
            tri_tf = flops / (med * 1e-3) / 1e12
            sdpa_tf = flops / (sdpa_ms * 1e-3) / 1e12
            pct_peak = 100.0 * tri_tf / PEAK_BF16_TFLOPS
            pct_sdpa = 100.0 * sdpa_ms / med  # tri time as %-of-SDPA-throughput = sdpa_ms/tri_ms
            spread = spread_pct(med, lo, hi)
            spread_s = f"{spread:.1f}{'!' if spread > 5.0 else ''}"
            print(
                f"{n:>6} {str(causal):>7} {med:>8.3f} {spread_s:>5} {sdpa_ms:>8.3f} "
                f"{tri_tf:>6.1f} {sdpa_tf:>7.1f} {pct_peak:>5.1f}% {pct_sdpa:>5.1f}% "
                f"{smi('clocks.sm')[0]:>6.0f}"
            )


def _oom_sweep(seqs, b, h, d, dtype) -> None:
    print("\n## No-OOM at large N — FA (O(1) SMEM working set) vs R0 naive (O(N²) score matrix)")
    print(
        f"# forward peak memory, B={b} H={h} d={d}, causal; naive materializes the B·H·N×N scores"
    )
    print(
        f"{'seq':>6} {'naive_scoreMB':>13} {'naive_peakMB':>13} {'FA_peakMB':>11} {'verdict':>18}"
    )
    for n in seqs:
        q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=dtype) for _ in range(3))
        score_mb = b * h * n * n * torch.tensor([], dtype=dtype).element_size() / (1 << 20)
        fa_mb = _peak_mem_mb(
            lambda q=q, k=k, v=v: flash_attention_triton_forward(q, k, v, is_causal=True)
        )
        naive_mb = _peak_mem_mb(lambda q=q, k=k, v=v: _naive_attention(q, k, v, True))
        if math.isinf(naive_mb):
            verdict = "naive OOM / FA ok"
            naive_s = "OOM"
        else:
            verdict = f"FA {naive_mb / fa_mb:4.1f}× leaner"
            naive_s = f"{naive_mb:.0f}"
        print(f"{n:>6} {score_mb:>13.0f} {naive_s:>13} {fa_mb:>11.0f} {verdict:>18}")


def _variant_gqa(n, b, h, d, dtype) -> None:
    print("\n## Variant — GQA (n_kv_heads < n_heads via repeat_interleave)")
    print(f"# B={b} H={h} d={d} N={n} causal; KV stored at h_kv, broadcast to h for the matmul")
    elem = torch.tensor([], dtype=dtype).element_size()
    q = torch.randn(b, h, n, d, device="cuda", dtype=dtype)
    print(f"{'h_kv':>5} {'ratio':>6} {'kv_MB':>8} {'tri_ms':>8} {'note':>26}")
    for h_kv in (h, h // 2, h // 8):
        kc = torch.randn(b, h_kv, n, d, device="cuda", dtype=dtype)  # cached at h_kv heads
        vc = torch.randn(b, h_kv, n, d, device="cuda", dtype=dtype)
        rep = h // h_kv
        k = kc.repeat_interleave(rep, dim=1)  # broadcast up to h query heads
        v = vc.repeat_interleave(rep, dim=1)
        med = bench_ms(
            lambda q=q, k=k, v=v: flash_attention_triton_forward(q, k, v, is_causal=True)
        )[0]
        kv_mb = 2 * b * h_kv * n * d * elem / (1 << 20)  # K+V cache at h_kv
        note = "MHA baseline" if h_kv == h else f"KV {h // h_kv}× smaller, same FLOPs"
        print(f"{h_kv:>5} {f'{h}:{h_kv}':>6} {kv_mb:>8.1f} {med:>8.3f} {note:>26}")
    kv_full = 2 * b * h * n * d * elem / (1 << 20)
    kv_8 = 2 * b * (h // 8) * n * d * elem / (1 << 20)
    print(
        f"# consequence: KV cache {kv_full:.0f} MB (MHA) → {kv_8:.0f} MB at h_kv={h // 8} "
        f"(8× less KV memory); compute unchanged (K,V broadcast, FLOPs identical)."
    )


def _causal_consequence(seqs, b, h, d, dtype) -> None:
    print(
        "\n## Variant — causal compute consequence (causal skips the upper triangle → ~2× less work)"
    )
    print(f"{'seq':>6} {'noncausal_ms':>13} {'causal_ms':>11} {'speedup':>9}")
    for n in seqs:
        q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=dtype) for _ in range(3))
        nc = bench_ms(
            lambda q=q, k=k, v=v: flash_attention_triton_forward(q, k, v, is_causal=False)
        )[0]
        ca = bench_ms(
            lambda q=q, k=k, v=v: flash_attention_triton_forward(q, k, v, is_causal=True)
        )[0]
        print(f"{n:>6} {nc:>13.3f} {ca:>11.3f} {nc / ca:>8.2f}×")


def run(
    seqs: tuple[int, ...] = (1024, 2048, 4096, 8192),
    b: int = 2,
    h: int = 8,
    d: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    gate: bool = True,
) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("flash_sm120 requires CUDA.")
    print(provenance_line("A4 FA2 sm120 measurement"))
    roofs = Roofs.measure(dtype)
    if gate:
        _correctness_gate()  # raises before a single millisecond is reported
    _roofline_sweep(seqs, b, h, d, dtype, roofs)
    _oom_sweep(seqs, b, h, d, dtype)
    _causal_consequence(seqs, b, h, d, dtype)
    _variant_gqa(seqs[-1], b, h, d, dtype)
    print(
        "\n# legend: %peak = triTF / 72 (absolute, bf16 tensor roof) · %SDPA = sdpa_ms/tri_ms "
        "(>100% = faster than SDPA) · ncu-debt: no SM-level profile (ncu blocked in container)."
    )


def _parse_dtype(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seqs", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    p.add_argument("--b", type=int, default=2)
    p.add_argument("--h", type=int, default=8)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16, help="bf16|fp16|fp32")
    p.add_argument(
        "--skip-gate", dest="gate", action="store_false", help="skip the correctness gate"
    )
    a = p.parse_args()
    run(seqs=tuple(a.seqs), b=a.b, h=a.h, d=a.d, dtype=a.dtype, gate=a.gate)
