"""A2 Rung 2 — softmax roofline: the memory-bound reduction ladder (twopass → online → fused).

The deliverable of this rung is THIS number, not the kernel. Softmax's arithmetic intensity is a few
FLOP/byte — far left of this card's ~130 FLOP/byte ridge — so it is pinned to the HBM roof and the
only lever is *bytes moved*. The ladder is the Milakov–Gimelshein pass count:

    twopass  3 read-passes + write ≈ 4N   (safe softmax: max, denom, normalize)
    online   1 fused pass  + write ≈ 3N   (running max+denom folded → 4/3 = 1.33× fewer bytes)
    fused    row-resident load  + write ≈ 2N   (ideal traffic, the peak-HBM stage)

We report, per stage:
  * ``ms`` + spread (do_bench median with per-rep L2 flush; >5% ⇒ distrust the clock);
  * ``GB/s`` and ``%peak`` against the *algorithmic* HBM byte model (are we saturating the roof?);
  * ``effGB/s`` / ``%peak_eff`` against the *ideal* 2N traffic (useful throughput — the number that
    actually improves down the ladder, since less byte-moving = less time for the same useful work).

Honesty — the L2 caveat (load-bearing on THIS box). The ~1.33× online-vs-twopass win is a **DRAM**
bound: it only manifests when a row's re-reads miss cache. This card has a **48 MB L2**, so at the
M=N=4096 target the whole 32 MB tensor is L2-resident and the streamed re-reads are L2 hits, not
DRAM — the *measured* twopass/online latency ratio therefore understates the 1.33× analytic DRAM
ratio (the extra passes are paid at L2 bandwidth, not HBM). The 1.33× is the out-of-cache bound — the
same insight that motivates flash-attention's single-pass online softmax over KV that does not fit in
SRAM. We print both the analytic byte ledger (exact) and the measured latencies (what the L2 permits)
and never conflate them.

Triton owns the CUDA-level detail (coalescing, float4 vectorization, SMEM bank-conflict swizzle) — see
the kernel docstring. ncu is blocked here (ERR_NVGPUCTRPERM); the metric to discharge is
``l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / requests`` (sectors/request → coalescing) plus
``sm__throughput`` Memory% (Speed-of-Light: confirm the kernel is DRAM-bound, not launch-bound).
Discharge on **sm_120 + counters** (KVM 5090, ~$0.33/hr) — NOT the H100 day this was misfiled to:
Hopper would recompile to different SASS and different autotune configs, i.e. a different kernel
instance, which does not discharge a claim made about this one.

Run on the GPU box:  PYTHONPATH=../../src python -m bench.kernels.reduce.softmax   (--help for shape overrides)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# bench/ on sys.path for the shared _harness + kernel_roofline (resolve upward to the dir holding
# _harness.py — move-proof across bench/kernels/<family>/).
_BENCH_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "_harness.py").exists())
sys.path.insert(0, str(_BENCH_ROOT))
from _harness import Roofs, provenance_line  # noqa: E402
from kernel_roofline import KernelProfile, profile  # noqa: E402

sys.path.insert(0, str(_BENCH_ROOT.parent / "src"))
from scratch_llm.kernels.reduce.softmax import softmax_triton  # noqa: E402

# stage → (read-passes over the row); +1 for the write pass = algorithmic HBM passes.
_STAGES: dict[str, int] = {"twopass": 3, "online": 2, "fused": 1}


def _gate(x: torch.Tensor) -> None:
    """Refuse to publish timings for a wrong kernel: every stage must equal F.softmax (rtol 1e-3)."""
    ref = F.softmax(x, dim=-1).float()
    for mode in _STAGES:
        out = softmax_triton(x, mode).float()
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def _profile_stage(x: torch.Tensor, mode: str, roofs: Roofs) -> tuple[KernelProfile, float]:
    m_rows, n_cols = x.shape
    elem = x.element_size()
    passes = _STAGES[mode] + 1  # + write
    algo_bytes = float(passes * m_rows * n_cols * elem)
    # exp dominates the (tiny) FLOP; count ~2 exp-passes for streamed, 1 for fused, ~5 FLOP/exp.
    flops = 5.0 * m_rows * n_cols * (1 if mode == "fused" else 2)
    prof = profile(mode, lambda: softmax_triton(x, mode), flops, algo_bytes, roofs)
    ideal_bytes = float(2 * m_rows * n_cols * elem)  # 1 read + 1 write
    eff_gbps = ideal_bytes / (prof.ms * 1e-3) / 1e9
    return prof, eff_gbps


def run(shapes: tuple[tuple[int, int], ...], gate: bool = True) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("bench/softmax.py requires CUDA.")

    roofs = Roofs.measure(torch.bfloat16)
    peak_gbps = roofs.bw_bytes_s / 1e9
    l2_mb = torch.cuda.get_device_properties(0).L2_cache_size / (1 << 20)

    print(provenance_line("A2 R2 — softmax roofline (twopass → online → fused)"))
    print(
        f"# measured HBM peak {peak_gbps:.1f} GB/s · L2 {l2_mb:.0f} MB · "
        f"ridge {roofs.ridge:.0f} FLOP/byte · dtype bf16"
    )
    print(
        f"{'shape':>12} {'stage':>8} {'algoN':>6} {'ms':>8} {'±%':>5} "
        f"{'GB/s':>8} {'%peak':>6} {'effGB/s':>8} {'%peak_eff':>9} {'bound':>6}"
    )

    for m_rows, n_cols in shapes:
        tensor_mb = m_rows * n_cols * 2 / (1 << 20)
        x = torch.randn(m_rows, n_cols, device="cuda", dtype=torch.bfloat16)
        if gate:
            _gate(x)

        ms_by_stage: dict[str, float] = {}
        for mode in _STAGES:
            prof, eff_gbps = _profile_stage(x, mode, roofs)
            ms_by_stage[mode] = prof.ms
            spread = f"{prof.spread_pct:.1f}{'!' if prof.spread_pct > 5.0 else ''}"
            print(
                f"{f'{m_rows}x{n_cols}':>12} {mode:>8} {f'{_STAGES[mode] + 1}N':>6} "
                f"{prof.ms:>8.4f} {spread:>5} {prof.gbps:>8.1f} {prof.pct_roof:>5.1f}% "
                f"{eff_gbps:>8.1f} {100.0 * eff_gbps / peak_gbps:>8.1f}% {prof.bound:>6}"
            )
        # the win: analytic 4N/3N = 1.33× fewer DRAM bytes; measured latency ratio (L2-limited here).
        analytic = (_STAGES["twopass"] + 1) / (_STAGES["online"] + 1)
        measured = ms_by_stage["twopass"] / ms_by_stage["online"]
        note = " (< analytic: 32MB<L2, re-reads hit L2)" if tensor_mb < l2_mb else " (>L2)"
        print(
            f"# {m_rows}x{n_cols} ({tensor_mb:.0f} MB): online vs twopass — analytic DRAM "
            f"{analytic:.2f}× fewer bytes · measured {measured:.2f}× faster{note}"
        )
        del x
        torch.cuda.empty_cache()

    print(
        "# legend: algoN = algorithmic HBM passes (read+write); GB/s/%peak vs that byte model — "
        "%peak > 100% ⇒ re-reads served by L2 (fewer real DRAM bytes than the model); effGB/s vs "
        "ideal 2N (useful throughput, the honest per-stage number). ncu-debt: "
        "l1tex sectors/request (coalescing) + sm__throughput Mem% (Speed-of-Light DRAM-bound check)."
    )


def _parse_shape(s: str) -> tuple[int, int]:
    a, b = s.lower().split("x")
    return int(a), int(b)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--shapes",
        type=_parse_shape,
        nargs="+",
        default=[(4096, 4096), (8192, 8192), (2048, 16384)],
        help="MxN shapes, e.g. 4096x4096",
    )
    p.add_argument("--skip-gate", dest="gate", action="store_false")
    a = p.parse_args()
    run(shapes=tuple(a.shapes), gate=a.gate)
