"""A2 R1 roofline — the GEMV ladder vs ``torch.mv``, anchored to this GPU's measured HBM peak.

GEMV (``y = A @ x``) reads A once for one MAC/element ⇒ ``AI ≈ 1 FLOP/byte`` ≪ ridge: it is
**memory-bound**, so the deliverable is **GB/s and %-of-HBM-peak**, not FLOP/s. Traffic model:
``bytes = M*N*2 (A, bf16) + N*2 (x) + M*2 (y)`` — A dominates. FLOPs = ``2*M*N``.

The ladder (see ``kernels/gemm/triton/gemv.py``): naive one-warp-per-row → coalesced block-per-row
(autotuned) → split-N two-stage (atomics). Correctness is gated first (a fast wrong kernel scores
zero) against ``torch.mv``. Target: the best stage > 80% of the measured HBM peak (>440 GB/s of
0.55 TB/s) at M=N=8192. ``torch.mv`` is reported as the vendor reference.

Run on the GPU box:  PYTHONPATH=../../src python -m bench.kernels.gemm.triton_gemv   (--help for shape overrides)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# bench/ on sys.path for the shared _harness (resolve upward to the dir holding _harness.py).
_BENCH_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "_harness.py").exists())
sys.path.insert(0, str(_BENCH_ROOT))
from _harness import Roofs, bench_ms, provenance_line, spread_pct, waves  # noqa: E402

sys.path.insert(0, str(_BENCH_ROOT.parent / "src"))
from scratch_llm.kernels.gemm.triton.gemv import gemv_blockrow, gemv_naive, gemv_split  # noqa: E402

# The ncu section this kernel would inspect (counters BLOCKED on this box, ERR_NVGPUCTRPERM): a GEMV
# lives or dies on load efficiency, so the sector/request ratio is the one. Discharge on sm_120 +
# counters (KVM 5090) — NOT the H100 day this was misfiled to; see kernel_roofline.py routing note.
NCU_DEBT = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / requests (sectors/request, ideal 4 = fully coalesced)"


def _bytes(m: int, n: int, elem: int) -> int:
    return m * n * elem + n * elem + m * elem  # A (dominant) + x + y


def _gate(a: torch.Tensor, x: torch.Tensor, dtype: torch.dtype) -> None:
    """Refuse to time a wrong kernel. rtol 1e-2 bf16 / 1e-5 fp32, as in tests/test_gemv.py."""
    ref = torch.mv(a, x).float()
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    atol = 1e-2 if dtype == torch.bfloat16 else 1e-4
    for name, fn in (("naive", gemv_naive), ("blockrow", gemv_blockrow), ("split", gemv_split)):
        out = fn(a, x).float()
        if not torch.allclose(out, ref, rtol=rtol, atol=atol):
            raise SystemExit(
                f"correctness gate FAILED for {name} @ {dtype}: "
                f"max|Δ|={(out - ref).abs().max().item():.3e}"
            )


def run(shapes: list[tuple[int, int]], dtype: torch.dtype) -> None:
    print(provenance_line("A2 R1 — GEMV ladder roofline"))
    roofs = Roofs.measure(dtype)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    peak_gbps = roofs.bw_bytes_s / 1e9
    print(
        f"# measured HBM peak {peak_gbps:.1f} GB/s ({roofs.bw_bytes_s / 1e12:.3f} TB/s) · "
        f"bf16 compute {roofs.compute_flops_s / 1e12:.1f} TF/s · ridge {roofs.ridge:.0f} FLOP/byte"
    )
    print(f"# target: best stage > 80% of HBM peak = {0.8 * peak_gbps:.0f} GB/s · dtype {dtype}")
    print(
        f"\n{'shape':>13} {'stage':>10} {'ms':>8} {'±%':>5} {'GB/s':>8} "
        f"{'%HBM':>6} {'%mv':>6} {'wv':>7} {'bound':>6}"
    )

    elem = torch.tensor([], dtype=dtype).element_size()
    for m, n in shapes:
        a = torch.randn(m, n, device="cuda", dtype=dtype)
        x = torch.randn(n, device="cuda", dtype=dtype)
        _gate(a, x, dtype)
        byts = _bytes(m, n, elem)
        flops = 2.0 * m * n

        # torch.mv reference (vendor GEMV) + the three ladder stages.
        stages: list[tuple[str, object, int]] = [
            ("torch.mv", lambda a=a, x=x: torch.mv(a, x), m),
            ("naive", lambda a=a, x=x: gemv_naive(a, x), m),
            ("blockrow", lambda a=a, x=x: gemv_blockrow(a, x), m),
            ("split", lambda a=a, x=x: gemv_split(a, x), m * 8),
        ]
        _, bound = roofs.attainable(flops, byts)
        mv_gbps = float("nan")
        for name, fn, ctas in stages:
            med, lo, hi = bench_ms(fn)
            gbps = byts / (med * 1e-3) / 1e9
            if name == "torch.mv":
                mv_gbps = gbps
            n_waves, under = waves(ctas, sm_count)
            wv = f"{n_waves:.1f}{'*' if under else ''}"
            sp = spread_pct(med, lo, hi)
            sp_s = f"{sp:.1f}{'!' if sp > 5.0 else ''}"
            pct_mv = "—" if name == "torch.mv" else f"{100.0 * gbps / mv_gbps:.0f}%"
            shape_col = f"{m:>6}x{n:<6}" if name == "torch.mv" else " " * 13
            print(
                f"{shape_col} {name:>10} {med:>8.4f} {sp_s:>5} {gbps:>8.1f} "
                f"{100.0 * gbps / peak_gbps:>5.1f}% {pct_mv:>6} {wv:>7} {bound:>6}"
            )
        del a, x
        torch.cuda.empty_cache()  # shared 24 GB box — free between shapes

    print(
        "\n# legend: %HBM = GB/s / measured-HBM-peak (absolute headroom); %mv vs torch.mv; "
        "wv = CTA waves over SMs (* under-fills once); ±% = p20–p80 spread (! >5% clock-noisy)."
    )
    print(f"# ncu-debt (blocked on this box; discharge on sm_120 + counters): {NCU_DEBT}")


def _parse_dtype(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


def _parse_shape(s: str) -> tuple[int, int]:
    m, n = s.lower().split("x")
    return int(m), int(n)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--shapes",
        type=_parse_shape,
        nargs="+",
        default=[(8192, 8192), (16384, 4096), (4096, 16384), (2048, 8192)],
        help="MxN pairs, e.g. 8192x8192",
    )
    p.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16, help="bf16|fp16|fp32")
    a = p.parse_args()
    run(shapes=list(a.shapes), dtype=a.dtype)
