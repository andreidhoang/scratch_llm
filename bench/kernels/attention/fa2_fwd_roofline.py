"""A2.1 roofline — Triton FA2 forward vs SDPA-flash, anchored to the machine's measured peaks.

The deliverable of the FA2 work is THIS number, not the kernel (A2 guide §6/§8). Predict-before-run
(docs/design/L2_flash_attention_SPEC.md §4): the Triton forward should reach ≥ 60% of SDPA at seq
4096, d=64, causal, bf16 on the 4090 (predicted ~65%); below 60% → ship the honest gap.

This is the hardened harness. A roofline that only reports "% of SDPA" answers *relative* position
but hides *absolute* headroom, and a single mean-of-mostly-throttled-runs number is not measurement.
So we additionally:

  * gate on correctness first — a fast wrong kernel scores zero; we refuse to time a kernel that does
    not equal the pure-PyTorch oracle (fp32) and SDPA (bf16) at a tractable shape;
  * time with CUDA events + per-rep L2 flush (triton.testing.do_bench) and report the **median with
    the p20–p80 spread**, flagging any row whose spread > 5% as untrustworthy (clock noise);
  * pin the SDPA backend to FLASH so "%SDPA" compares against one fixed kernel across the whole seq
    sweep, not whichever backend the dispatcher happened to pick per shape;
  * anchor to **measured** roofs on this box — HBM bandwidth (copy) and bf16 tensor TFLOP/s (cuBLAS
    GEMM) — and report %roof + which roof binds (mem vs compute) per shape;
  * expose wave quantization (CTAs vs SM count) so a flattering TFLOP/s on an under-filled grid is
    visible;
  * log provenance + the achieved SM clock per row, so a throttled run is self-evidently invalid.

Clocks: lock them if the host allows it (`sudo nvidia-smi -pm 1 && nvidia-smi -lgc <clk>`); inside an
unprivileged container that is usually blocked, so we instead warm to steady state and print the SM
clock per row — if it drifts down across the sweep, distrust the tail.

Run on the GPU box:  PYTHONPATH=../../src python -m bench.kernels.attention.fa2_fwd_roofline   (--help for shape/dtype overrides)
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
from _harness import Roofs, bench_ms, provenance_line, smi, spread_pct, waves  # noqa: E402

sys.path.insert(0, str(_BENCH_ROOT.parent / "src"))
from scratch_llm.kernels import (
    flash_attention_forward,  # pure-PyTorch oracle (the ground truth)  # noqa: E402
)
from scratch_llm.kernels.attention.prefill.fa2 import flash_attention_triton_forward  # noqa: E402

# Optional like-for-like baseline: the upstream FA2 CUDA kernel. Absent on this box; column shows n/a.
try:
    from flash_attn import flash_attn_func  # type: ignore
except Exception:  # pragma: no cover - depends on the environment
    flash_attn_func = None

# Force SDPA to the flash backend so the baseline is ONE kernel across the sweep, not a moving target.
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _FLASH_CTX = lambda: sdpa_kernel(SDPBackend.FLASH_ATTENTION)  # noqa: E731
except Exception:  # pragma: no cover - very old torch
    import contextlib

    _FLASH_CTX = contextlib.nullcontext  # type: ignore


# --------------------------------------------------------------------------------------------------
# Attention work + traffic models (flash never materializes the N×N scores — that is the whole point).
# --------------------------------------------------------------------------------------------------
def _attn_flops(b: int, h: int, n: int, d: int, causal: bool) -> float:
    # fwd ≈ 4·B·H·N²·d (QKᵀ then PV, 2 FLOP/MAC); causal attends ~half the (i,j) pairs.
    return 4.0 * b * h * n * n * d * (0.5 if causal else 1.0)


def _attn_bytes(b: int, h: int, n: int, d: int, elem: int) -> float:
    # Ideal flash HBM traffic: read Q,K,V + write O = 4·B·H·N·d elements (L is O(N), negligible).
    return 4.0 * b * h * n * d * elem


def _selected_block_q(n: int, d: int) -> int:
    """Best-effort read of the autotuner's chosen BLOCK_Q for this shape (drives the CTA count).
    Falls back to 128 (the larger candidate → fewer CTAs → conservative under-fill warning)."""
    try:
        from scratch_llm.kernels.attention.prefill.fa2 import _fa2_fwd_kernel

        for key, cfg in _fa2_fwd_kernel.cache.items():
            if n in key and d in key:
                return int(cfg.kwargs.get("BLOCK_Q", 128))
    except Exception:
        pass
    return 128


# --------------------------------------------------------------------------------------------------
# Correctness gate — refuse to publish timings for a kernel that is wrong.
# --------------------------------------------------------------------------------------------------
def _correctness_gate(causal: bool) -> None:
    # fp32 vs the pure oracle at a tractable shape (the N² oracle can't run at bench seq lengths).
    q, k, v = (torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float32) for _ in range(3))
    o_tri, _ = flash_attention_triton_forward(q, k, v, is_causal=causal, allow_tf32=False)
    o_ref, _ = flash_attention_forward(q, k, v, is_causal=causal)
    torch.testing.assert_close(o_tri, o_ref, atol=2e-4, rtol=2e-4)
    # bf16 vs SDPA-flash at the precision we actually benchmark.
    q, k, v = (torch.randn(2, 4, 512, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    o_tri, _ = flash_attention_triton_forward(q, k, v, is_causal=causal)
    with _FLASH_CTX():
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(o_tri.float(), ref.float(), atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------------------------------
def run(
    seqs: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384),
    b: int = 2,
    h: int = 8,
    d: int = 64,
    causal: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    gate: bool = True,
) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("flash_roofline requires CUDA.")

    if gate:
        _correctness_gate(causal)  # raises (AssertionError) before a single millisecond is reported

    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    elem = torch.tensor([], dtype=dtype).element_size()
    roofs = Roofs.measure(dtype)  # compute + HBM ceilings, measured on THIS GPU

    print(provenance_line("FA2-Triton vs SDPA-flash"))
    print(
        f"# shape B={b} H={h} d={d} causal={causal} dtype={dtype}  (gate={'pass' if gate else 'SKIPPED'})"
    )
    print(
        f"# measured roofs: compute {roofs.compute_flops_s / 1e12:6.1f} TFLOP/s (bf16 GEMM) | "
        f"HBM {roofs.bw_bytes_s / 1e9:6.1f} GB/s | ridge {roofs.ridge:5.1f} FLOP/byte"
    )
    if flash_attn_func is None:
        print("# flash_attn lib not installed → fa2_ms/%FA2 columns = n/a")
    print(
        f"{'seq':>6} {'tri_ms':>8} {'±%':>5} {'sdpa_ms':>8} {'fa2_ms':>7} "
        f"{'triTF':>6} {'sdpaTF':>6} {'%SDPA':>6} {'%FA2':>5} {'%roof':>6} {'bind':>4} "
        f"{'wv':>5} {'smclk':>6}"
    )

    flagged: list[int] = []
    for n in seqs:
        q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=dtype) for _ in range(3))

        med, lo, hi = bench_ms(
            lambda q=q, k=k, v=v: flash_attention_triton_forward(q, k, v, is_causal=causal)
        )
        with _FLASH_CTX():
            sdpa_ms = bench_ms(
                lambda q=q, k=k, v=v: F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            )[0]

        fa2_ms = float("nan")
        if flash_attn_func is not None:
            fa2 = flash_attn_func  # narrowed non-None binding (pyright can't see the guard in a lambda)
            # FA2 lib wants (B, N, H, d); ours is (B, H, N, d).
            qf, kf, vf = (t.transpose(1, 2).contiguous() for t in (q, k, v))
            try:
                fa2_ms = bench_ms(
                    lambda qf=qf, kf=kf, vf=vf, fa2=fa2: fa2(qf, kf, vf, causal=causal)
                )[0]
            except Exception:
                fa2_ms = float("nan")

        flops = _attn_flops(b, h, n, d, causal)
        tri_tf = flops / (med * 1e-3) / 1e12
        sdpa_tf = flops / (sdpa_ms * 1e-3) / 1e12

        # Roofline: attainable = min(compute peak, intensity · bandwidth). %roof = absolute headroom.
        attain, bind = roofs.attainable(flops, _attn_bytes(b, h, n, d, elem))
        pct_roof = 100.0 * (flops / (med * 1e-3)) / attain

        # Wave quantization: how many waves of CTAs the grid launches over the SMs.
        ctas = b * h * math.ceil(n / _selected_block_q(n, d))
        n_waves, underfills = waves(ctas, sm_count)
        wv = f"{n_waves:.1f}{'*' if underfills else ''}"  # '*' = grid does not even fill the GPU once

        spread = spread_pct(med, lo, hi)
        spread_s = (
            f"{spread:.1f}{'!' if spread > 5.0 else ''}"  # '!' = clock-noisy, distrust this row
        )
        if spread > 5.0:
            flagged.append(n)

        sm_clk = smi("clocks.sm")[0]
        pct_sdpa = 100.0 * tri_tf / sdpa_tf
        has_fa2 = fa2_ms == fa2_ms  # False when NaN (lib absent / kernel rejected the shape)
        fa2_cell = f"{fa2_ms:7.3f}" if has_fa2 else "    n/a"
        fa2_pct_cell = (
            f"{100.0 * tri_tf / (flops / (fa2_ms * 1e-3) / 1e12):4.0f}" if has_fa2 else " n/a"
        )

        print(
            f"{n:>6} {med:>8.3f} {spread_s:>5} {sdpa_ms:>8.3f} {fa2_cell:>7} "
            f"{tri_tf:>6.1f} {sdpa_tf:>6.1f} {pct_sdpa:>5.1f}% {fa2_pct_cell:>4}% "
            f"{pct_roof:>5.1f}% {bind:>4} {wv:>5} {sm_clk:>6.0f}"
        )

    if flagged:
        print(
            f"# WARNING: spread > 5% at seq {flagged} — clocks unstable; re-run pinned or warm longer."
        )
    print(
        "# legend: ±% = (p80−p20)/median spread (! >5%); %SDPA/%FA2 relative; %roof absolute "
        "(min(compute, intensity·BW)); bind = limiting roof; wv = CTA waves over SMs (* under-fills)."
    )


def _parse_dtype(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seqs", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    p.add_argument("--b", type=int, default=2)
    p.add_argument("--h", type=int, default=8)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--no-causal", dest="causal", action="store_false")
    p.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16, help="bf16|fp16|fp32")
    p.add_argument(
        "--skip-gate",
        dest="gate",
        action="store_false",
        help="skip the correctness gate (not advised)",
    )
    a = p.parse_args()
    run(seqs=tuple(a.seqs), b=a.b, h=a.h, d=a.d, causal=a.causal, dtype=a.dtype, gate=a.gate)
