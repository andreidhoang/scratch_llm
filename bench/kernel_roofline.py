"""A2 Rung 0 — the profiler + roofline harness every A2 kernel is measured against.

One measurement methodology (imported from `_harness`: CUDA-event timing with per-rep L2 flush,
machine peaks measured on THIS GPU, the roofline arithmetic) plus the A2-specific rig: profile a
kernel → place it on the roofline (% of the binding roof, mem/cmp) → append to a CSV → plot. The
R0 gate is reproduction of the `bench/RESULTS.md` R0 baseline (~0.55 TB/s HBM, ~72 TF/s bf16).

**ncu is BLOCKED on this box** (`ERR_NVGPUCTRPERM`, unprivileged container) — so the Speed-of-Light
/ MemoryWorkloadAnalysis / bank-conflict counters the A2 spec's ncu command would give are recorded
as **ncu-debt** (see `A2_ncu_debt` below): each A2 kernel names the ncu metric it WOULD inspect, and
the H100 rental day (where counters are enabled) discharges the list. In the meantime a kernel's
bound is established by achieved-vs-measured-peak % (this harness) + nsys traces for launch/timeline.

Run:  python bench/kernel_roofline.py --device cuda   # reproduces peaks, writes CSV + roofline.png
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass

import torch
from _harness import Roofs, bench_ms, provenance_line, spread_pct

# The A2 spec's ncu sections, mapped to the metric each kernel will inspect on the H100 day.
A2_ncu_debt: dict[str, str] = {
    "SpeedOfLight": "sm__throughput.avg.pct_of_peak_sustained_elapsed (Memory% vs Compute%)",
    "MemoryWorkload": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / requests (sectors/request, ideal 4)",
    "BankConflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum (≈0 after XOR swizzle)",
    "WarpStalls": "warp state: 'Stall MIO Throttle' (SMEM excess) / 'No Eligible' (data-dep serialization)",
    "TensorPipe": "sm__pipe_tensor_op_hmma.avg.pct_of_peak_sustained (WGMMA util, ≥85% target — H100)",
}


@dataclass(frozen=True)
class KernelProfile:
    name: str
    ms: float
    spread_pct: float
    gflops: float
    gbps: float
    ai: float  # arithmetic intensity, FLOP/byte
    pct_roof: float  # achieved / roofline-attainable
    bound: str  # 'mem' | 'cmp'


def profile(name: str, fn, flops: float, bytes_: float, roofs: Roofs) -> KernelProfile:
    """Measure `fn` and place it on the roofline: %-of-attainable + which roof binds."""
    med, lo, hi = bench_ms(fn)
    sec = med * 1e-3
    achieved = flops / sec
    attainable, bound = roofs.attainable(flops, bytes_)
    return KernelProfile(
        name=name,
        ms=med,
        spread_pct=spread_pct(med, lo, hi),
        gflops=achieved / 1e9,
        gbps=bytes_ / sec / 1e9,
        ai=flops / bytes_,
        pct_roof=100.0 * achieved / attainable,
        bound=bound,
    )


def write_csv(profiles: list[KernelProfile], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "ms", "spread_%", "GFLOP/s", "GB/s", "AI", "%roof", "bound"])
        for p in profiles:
            w.writerow(
                [
                    p.name,
                    f"{p.ms:.4f}",
                    f"{p.spread_pct:.1f}",
                    f"{p.gflops:.1f}",
                    f"{p.gbps:.1f}",
                    f"{p.ai:.3f}",
                    f"{p.pct_roof:.1f}",
                    p.bound,
                ]
            )


def plot_roofline(profiles: list[KernelProfile], roofs: Roofs, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ai = [10.0**x for x in [-2 + 0.05 * i for i in range(121)]]  # 1e-2 .. 1e4
    mem = [a * roofs.bw_bytes_s / 1e12 for a in ai]
    cmp_roof = roofs.compute_flops_s / 1e12
    roof = [min(m, cmp_roof) for m in mem]
    ax.loglog(ai, roof, "k-", lw=2, label="roofline (measured peaks)")
    ax.axvline(roofs.ridge, ls="--", c="gray", lw=1, label=f"ridge {roofs.ridge:.0f} FLOP/B")
    for p in profiles:
        ax.loglog(p.ai, p.gflops / 1e3, "o", ms=8)
        ax.annotate(
            p.name, (p.ai, p.gflops / 1e3), textcoords="offset points", xytext=(6, 4), fontsize=8
        )
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("TFLOP/s")
    ax.set_title(
        f"A2 roofline — {torch.cuda.get_device_name(0)}\n{roofs.bw_bytes_s / 1e12:.2f} TB/s · {cmp_roof:.0f} TF/s bf16"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser(description="A2 R0 — profiler + roofline harness")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--csv", default="a2_roofline.csv")  # relative to the bench/ run dir
    ap.add_argument("--png", default="a2_roofline.png")
    args = ap.parse_args()

    print(provenance_line("A2 R0 — profiler + roofline harness"))
    roofs = Roofs.measure(torch.bfloat16)
    print(
        f"# measured peaks: {roofs.bw_bytes_s / 1e12:.3f} TB/s HBM · {roofs.compute_flops_s / 1e12:.1f} TF/s bf16 "
        f"· ridge {roofs.ridge:.0f} FLOP/byte"
    )
    # R0 gate: reproduce the bench/RESULTS.md baseline (0.55 TB/s, 72 TF/s) within noise
    bw_ok = 0.45e12 <= roofs.bw_bytes_s <= 0.62e12
    tf_ok = 60e12 <= roofs.compute_flops_s <= 85e12
    print(
        f"# R0 baseline reproduction: HBM {'PASS' if bw_ok else 'CHECK'} · compute {'PASS' if tf_ok else 'CHECK'}"
    )

    # two reference points anchoring the two roofs: a pure-copy (memory) and a big GEMM (compute)
    n = 8192
    a = torch.randn(n, n, device=args.device, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=args.device, dtype=torch.bfloat16)
    big = torch.empty(1 << 26, device=args.device, dtype=torch.float32)
    big2 = torch.empty_like(big)
    profiles = [
        profile(
            "copy(256MB)",
            lambda: big2.copy_(big),
            flops=big.numel(),
            bytes_=2 * big.numel() * 4,
            roofs=roofs,
        ),
        profile(
            "gemm(8192³)",
            lambda: torch.matmul(a, b),
            flops=2.0 * n**3,
            bytes_=3 * n * n * 2,
            roofs=roofs,
        ),
    ]
    print(
        f"\n{'kernel':<14}{'ms':>9}{'spread%':>9}{'GFLOP/s':>10}{'GB/s':>8}{'AI':>9}{'%roof':>8}{'bound':>7}"
    )
    for p in profiles:
        print(
            f"{p.name:<14}{p.ms:>9.4f}{p.spread_pct:>9.1f}{p.gflops:>10.1f}{p.gbps:>8.1f}{p.ai:>9.2f}{p.pct_roof:>8.1f}{p.bound:>7}"
        )
    write_csv(profiles, args.csv)
    plot_roofline(profiles, roofs, args.png)
    print(f"\n# wrote {args.csv} + {args.png}")
    print(
        f"# ncu-debt (blocked on this box; discharged on the H100 day): {len(A2_ncu_debt)} metrics registered"
    )


if __name__ == "__main__":
    main()
