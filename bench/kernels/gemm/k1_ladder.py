"""The K1 GEMM ladder's measurement entry point — one methodology for six rungs and their floors.

Every K1 rung is measured by this script, and so is the floor it is judged against. That is the
whole point: H-R1 through B-R6 differ only in which kernel is called, so if they were six scripts
they would drift into six methodologies and "H-R3 reached 71%" would stop being comparable to
"H-R1 reached 28%". One file, one timing loop, one FLOP model, one correctness gate.

    python bench/kernels/gemm/k1_ladder.py --rung H-R1                 measure the rung, vs its floor
    python bench/kernels/gemm/k1_ladder.py --floor cublas              measure the floor alone
    python bench/kernels/gemm/k1_ladder.py --rung H-R1 --shape npot    a different spec shape
    python bench/kernels/gemm/k1_ladder.py --rung H-R1 --dry-run       print the plan, run nothing
    python bench/kernels/gemm/k1_ladder.py --list                      the rung and shape registries

The last line of stdout is always a single bare number — the rung's metric, or the floor's TFLOP/s.
``experiments/K1/<rung>/{run,floor}.sh`` read that line, so nothing else may be printed after it.

This is never invoked directly for a recorded measurement. ``infra/bench.sh`` wraps it: it locks
clocks, records provenance, captures ncu, and refuses to run at all unless a prediction for the rung
already exists in the ledger. A number produced by running this script by hand is a number with no
prediction in front of it, which is the one thing the ledger exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "bench"))

# ---------------------------------------------------------------------------------------------
# Registries. Hand-curated, like bench/kernels/run.py's _BENCHES — a rung that is not here cannot
# be measured, which is the correct failure mode for a rung nobody wired up.
# ---------------------------------------------------------------------------------------------

#: name -> (M, N, K). Kept identical to SHAPES in tests/kernels/gemm/test_k1_h_r1.py; the test
#: suite asserts the correctness gate covers every shape the bench can measure.
SHAPES: dict[str, tuple[int, int, int]] = {
    "sq4096": (4096, 4096, 4096),
    "rect8192": (8192, 8192, 4096),
    "skinny16": (16, 4096, 4096),
    "npot": (257, 1023, 512),
    "untuned": (1536, 6144, 2560),
}


@dataclass(frozen=True)
class Rung:
    """One rung: where its kernel lives, which arch it needs, and what it is measured against."""

    name: str  # "H-R1"
    module: str  # import path of the wrapper
    fn: str  # callable in that module taking (a, b) -> C
    arch: tuple[int, int]  # the compute capability whose ISA the kernel uses
    floor: str  # key into FLOORS
    metric: str  # the ledger metric name — must match `make predict M=...`
    note: str = ""
    default_shapes: tuple[str, ...] = field(default=("sq4096",))


RUNGS: dict[str, Rung] = {
    "H-R1": Rung(
        "H-R1",
        "scratch_llm.kernels.gemm.cuda.h_r1",
        "h_r1_gemm",
        (9, 0),
        "cublas",
        "pct_of_cublas",
        "wgmma from smem, single stage, 128x128 tile, 1 warpgroup",
    ),
}

#: The production implementations a rung's number only means something against. Matched dtype and
#: shape is not a nicety — a bf16 kernel compared to an fp16 cuBLAS call is comparing two problems.
FLOORS: dict[str, str] = {
    "cublas": "torch.matmul on bf16 operands — cuBLAS's own bf16 kernel selection",
    "cublaslt": "torch._scaled_mm / cuBLASLt epilogue path (B200 rungs; see B-R5's spec)",
}


def _list() -> int:
    print(f"{'rung':<8} {'arch':<8} {'floor':<10} {'metric':<18} kernel")
    print("-" * 100)
    for r in RUNGS.values():
        print(
            f"{r.name:<8} sm_{r.arch[0] * 10 + r.arch[1]:<5} {r.floor:<10} {r.metric:<18} {r.module}.{r.fn}"
        )
    print("-" * 100)
    print(f"{'shape':<10} {'M':>7} {'N':>7} {'K':>7}   GFLOP")
    for name, (m, n, k) in SHAPES.items():
        print(f"{name:<10} {m:>7} {n:>7} {k:>7}   {2.0 * m * n * k / 1e9:8.1f}")
    missing = sorted({"H-R1", "H-R2", "H-R3", "H-R4", "B-R5", "B-R6"} - set(RUNGS))
    if missing:
        print(f"\n# not yet wired: {', '.join(missing)}  (add a Rung row when its wrapper exists)")
    return 0


def _resolve(rung: Rung):  # noqa: ANN202 - returns an opaque callable
    import importlib

    return getattr(importlib.import_module(rung.module), rung.fn)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Recorded measurements go through infra/bench.sh, which enforces the prediction gate.",
    )
    ap.add_argument("--rung", choices=sorted(RUNGS) or None, help="which K1 rung to measure")
    ap.add_argument("--floor", choices=sorted(FLOORS), help="measure a floor instead of a rung")
    ap.add_argument("--shape", choices=sorted(SHAPES), default="sq4096")
    ap.add_argument("--dtype", default="bf16", choices=["bf16"], help="K1's floor is cuBLAS bf16")
    ap.add_argument("--warmup", type=int, default=20, help="workspace invariant 4: >= 20")
    ap.add_argument("--iters", type=int, default=50, help="workspace invariant 4: >= 50")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None, help="also write the row as JSON here")
    ap.add_argument(
        "--dry-run", action="store_true", help="print the plan and exit; measure nothing"
    )
    ap.add_argument("--list", action="store_true", help="print the rung and shape registries")
    args = ap.parse_args()

    if args.list:
        return _list()
    if not args.rung and not args.floor:
        ap.error("one of --rung or --floor is required (or --list)")

    m, n, k = SHAPES[args.shape]
    flops = 2.0 * m * n * k
    what = f"rung {args.rung}" if args.rung else f"floor {args.floor}"

    if args.dry_run:
        print(f"# k1_ladder [dry-run] · {what} · shape {args.shape} = {m}x{n}x{k} · {args.dtype}")
        print(
            f"#   {flops / 1e9:.1f} GFLOP · warmup {args.warmup} · iters {args.iters} · seed {args.seed}"
        )
        if args.rung:
            r = RUNGS[args.rung]
            print(
                f"#   kernel {r.module}.{r.fn} · needs sm_{r.arch[0] * 10 + r.arch[1]}a · floor {r.floor}"
            )
            print(
                f"#   metric {r.metric} — `make predict L=K1 R={r.name} M={r.metric} V=<yours>` first"
            )
        return 0

    import torch

    # _harness imports triton, which only exists on a GPU box. Check for a device FIRST so this
    # script stays runnable on the Mac (--list and --dry-run are the CPU-side of the contract, and
    # experiments/K1/*/run.sh must be able to explain itself without one).
    if not torch.cuda.is_available():
        print(
            "# no CUDA device — a GEMM number is a statement about silicon, and there is none here."
        )
        print(f"# on the box:  infra/bench.sh -l K1 -r {args.rung or 'H-R1'} -n -- \\")
        print(f"#                python bench/kernels/gemm/k1_ladder.py {' '.join(sys.argv[1:])}")
        return 0

    from _harness import bench_ms, provenance_line, spread_pct  # type: ignore[import-not-found]

    torch.manual_seed(args.seed)
    # TF32 must be off for the floor: a bf16 GEMM compared against a silently-TF32 baseline is the
    # oldest way to manufacture a flattering number, and the maintainer agent checks for it first.
    torch.backends.cuda.matmul.allow_tf32 = False
    print(provenance_line(f"K1 {what} · {args.shape} {m}x{n}x{k} · {args.dtype}"))

    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

    def _tflops(fn) -> tuple[float, float, float]:
        med, lo, hi = bench_ms(fn, warmup=args.warmup, rep=args.iters)
        return flops / (med * 1e-3) / 1e12, med, spread_pct(med, lo, hi)

    floor_key = args.floor or RUNGS[args.rung].floor
    floor_tf, floor_ms, floor_spread = _tflops(lambda a=a, b=b: torch.matmul(a, b))
    print(
        f"# floor {floor_key:<9} {floor_tf:7.1f} TF/s  ({floor_ms:8.3f} ms · IQR {floor_spread:4.1f}%)  {FLOORS[floor_key]}"
    )

    row: dict[str, object] = {
        "ladder": "K1",
        "shape": args.shape,
        "mnk": [m, n, k],
        "dtype": args.dtype,
        "floor_name": floor_key,
        "floor_tflops": floor_tf,
        "floor_ms": floor_ms,
        "floor_iqr_pct": floor_spread,
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
    }

    result: float = floor_tf
    if args.rung:
        r = RUNGS[args.rung]
        kernel = _resolve(r)
        # Correctness before performance (workspace invariant 3). A kernel that is fast and wrong
        # is measured here as what it is: nothing. The tolerance itself lives in the rung's test.
        out = kernel(a, b)
        ref = torch.matmul(a.float(), b.float())
        rel = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-30)
        print(
            f"# correctness  normalized rel err vs fp32 oracle = {rel:.3e}  (gate: tests/kernels/gemm/test_k1_{r.name.lower().replace('-', '_')}.py)"
        )
        del out, ref

        tf, ms, spread = _tflops(lambda a=a, b=b: kernel(a, b))
        pct = 100.0 * tf / floor_tf
        print(f"# {r.name:<12} {tf:7.1f} TF/s  ({ms:8.3f} ms · IQR {spread:4.1f}%)  {r.note}")
        print(f"# {r.metric:<12} {pct:7.1f}")
        if spread > 5.0:
            print(
                "# WARNING IQR > 5% — clocks are not stable; this row should not enter the ledger."
            )
        row |= {
            "rung": r.name,
            "metric": r.metric,
            "tflops": tf,
            "ms": ms,
            "iqr_pct": spread,
            "rel_err": rel,
        }
        result = pct
        print(
            f'#\n# record it:  make measure L=K1 R={r.name} M={r.metric} V={pct:.1f} DEV="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" FLOOR={floor_key} FLOORV={floor_tf:.1f}'
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(row, indent=1))

    del a, b
    torch.cuda.empty_cache()
    # The last line is the number, bare — run.sh and floor.sh read exactly this.
    print(f"{result:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
