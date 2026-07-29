"""The kernel-ladder runner — one command to list, filter, and run every kernel bench.

This is the operability layer the kernel tree was missing: ``bench/kernels/`` mirrors
``src/scratch_llm/kernels/<family>/<backend>/``, but without an index each bench was a standalone
``python bench/kernels/<family>/<x>.py`` invocation with no discovery. This module is that index —
a hardcoded registry (no magic) mapping a short name to (script path, rung label, kernel exercised,
GPU-required) so the ladder is auditable and runnable from one entry point.

Usage (from the repo root, with the GPU venv active)::

    python -m bench.kernels.run                         # list every kernel bench
    python -m bench.kernels.run gemm                    # run every GEMM bench
    python -m bench.kernels.run gemm/triton_tiled       # run ONE bench (tab-completable path)
    python -m bench.kernels.run --gpu-check             # print arch + measured peaks, run nothing

Each bench is launched as a subprocess (so a Triton/nvcc import failure in one bench does not poison
the runner, and each runs in its own clean process as it was designed to). GPU benches are auto-
skipped on a CPU box with a clear message — the same discipline as the ``gpu`` pytest marker.

Adding a bench: drop the script under ``bench/kernels/<family>/``, then add a row to ``_BENCHES``
below AND to the table in ``bench/kernels/README.md``. The registry is the source of truth; the
README is its human-readable mirror.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # bench/kernels/run.py → repo root


# Each bench: (name, script path relative to repo root, rung label, kernel module exercised, GPU?).
# `name` is the short handle passed on the CLI (<family>/<slug>); families are filterable alone.
# Kept hand-curated so the ladder is explicit and reviewable — not auto-discovered.
@dataclass(frozen=True)
class Bench:
    name: str  # "gemm/triton_tiled"  — the CLI handle (also tab-completable as a path)
    script: str  # "bench/kernels/gemm/triton_tiled.py"
    rung: str  # "A2 R5+6"            — the assignment rung this bench satisfies
    kernel: str  # "kernels.gemm.triton.tiled (naive/tiled/autotuned)"
    gpu: bool = True  # every kernel bench is GPU-only (CPU path is the framework fallback)


_BENCHES: list[Bench] = [
    # --- GEMM ladder (5 backends: Triton tiled/GEMV, CUDA-core, mma.sync, WMMA) ---
    Bench(
        "gemm/triton_tiled",
        "bench/kernels/gemm/triton_tiled.py",
        "A2 R5+6",
        "kernels.gemm.triton.tiled (gemm_naive/tiled/autotuned)",
    ),
    Bench(
        "gemm/triton_gemv",
        "bench/kernels/gemm/triton_gemv.py",
        "A2 R1",
        "kernels.gemm.triton.gemv (naive/blockrow/split)",
    ),
    Bench(
        "gemm/cuda_smem",
        "bench/kernels/gemm/cuda_smem.py",
        "A3 R0",
        "kernels.gemm.cuda.smem_tiled (CUDA cores, no tensor cores)",
    ),
    Bench(
        "gemm/cuda_mma_sync",
        "bench/kernels/gemm/cuda_mma_sync.py",
        "A3 R2",
        "kernels.gemm.cuda.mma_sync (PTX mma.sync m16n8k16 + ldmatrix)",
    ),
    Bench(
        "gemm/wmma",
        "bench/kernels/gemm/wmma.py",
        "A3 R1",
        "kernels.gemm.wmma.gemm (nvcuda::wmma fragments, fp32 acc)",
    ),
    # --- Frontier GEMM rungs (arch-gated; sm_90a/sm_100a rental box only) ---
    Bench(
        "gemm/cuda_wgmma",
        "bench/kernels/gemm/cuda_wgmma.py",
        "A3 R3.1",
        "kernels.gemm.cuda.wgmma (Hopper WGMMA, sm_90a; compile-gated, rental runtime)",
    ),
    Bench(
        "gemm/cuda_tcgen05",
        "bench/kernels/gemm/cuda_tcgen05.py",
        "A3 R4.1",
        "kernels.gemm.cuda.tcgen05 (Blackwell tcgen05, sm_100a; compile-gated, rental runtime)",
    ),
    Bench(
        "gemm/cuda_fp8",
        "bench/kernels/gemm/cuda_fp8.py",
        "A3 R3.2",
        "kernels.gemm.cuda.fp8 (Hopper FP8, sm_90a; STUB — learning rep)",
    ),
    Bench(
        "gemm/cuda_stream_k",
        "bench/kernels/gemm/cuda_stream_k.py",
        "A3 R3.3",
        "kernels.gemm.cuda.stream_k (Hopper stream-K, sm_90a; STUB — learning rep)",
    ),
    Bench(
        "gemm/cuda_persistent",
        "bench/kernels/gemm/cuda_persistent.py",
        "A3 R3.4",
        "kernels.gemm.cuda.persistent (Hopper persistent GEMV, sm_90a; STUB — learning rep)",
    ),
    # --- Attention (FA2 Triton fwd/bwd + the sm120 measurement sweep) ---
    Bench(
        "attention/fa2_fwd_roofline",
        "bench/kernels/attention/fa2_fwd_roofline.py",
        "A2.1 fwd",
        "kernels.attention.prefill.fa2 (flash_attention_triton_forward) vs SDPA-flash",
    ),
    Bench(
        "attention/fa2_bwd_roofline",
        "bench/kernels/attention/fa2_bwd_roofline.py",
        "A2.1 bwd",
        "kernels.attention.prefill.fa2 (TritonFlashAttention fwd+bwd) vs SDPA",
    ),
    Bench(
        "attention/fa2_sm120",
        "bench/kernels/attention/fa2_sm120.py",
        "A4 R2+3",
        "kernels.attention.prefill.fa2 on sm120: correctness + roofline + OOM + causal/GQA",
    ),
    Bench(
        "attention/fa3_hopper",
        "bench/kernels/attention/fa3_hopper.py",
        "A3 R3.5",
        "kernels.attention.prefill.fa3 (Hopper FA3, sm_90a; compile-gated, rental runtime)",
    ),
    # --- Norm (RMSNorm + LayerNorm, the memory-bound reduction family) ---
    Bench(
        "norm/normalize",
        "bench/kernels/norm/normalize.py",
        "A2 R3",
        "kernels.norm.normalize (rmsnorm_triton/layernorm_triton)",
    ),
    # --- Reduce (softmax ladder + top-k, the other memory-bound reductions) ---
    Bench(
        "reduce/softmax",
        "bench/kernels/reduce/softmax.py",
        "A2 R2",
        "kernels.reduce.softmax (twopass/online/fused)",
    ),
    Bench(
        "reduce/topk",
        "bench/kernels/reduce/topk.py",
        "A2 R4",
        "kernels.reduce.topk (topk_last_dim + fused_softmax_topk)",
    ),
]


def _have_cuda() -> bool:
    """True iff a CUDA device is visible. Cheap probe; never imports triton/nvcc."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _gpu_check() -> int:
    """Print the arch + measured peaks (via common.arch + bench/_harness), run no bench."""
    try:
        from scratch_llm.kernels.common.arch import (
            arch_name,
            compute_capability,
            is_blackwell,
            is_hopper,
        )

        cc = compute_capability()
        print(f"# arch_name()      = {arch_name()!r}")
        print(f"# compute_cap      = {cc!r}")
        print(f"# is_hopper()      = {is_hopper()}")
        print(f"# is_blackwell()   = {is_blackwell()}")
    except Exception as e:
        print(f"# common.arch probe failed: {e}")
    if not _have_cuda():
        print("# CUDA: not available (CPU box — GPU benches will skip).")
        return 0
    # Measure the roofs via the shared harness (the same peaks every bench anchors to).
    bench_root = REPO_ROOT / "bench"
    sys.path.insert(0, str(bench_root))
    try:
        import torch

        from _harness import Roofs  # type: ignore[import-not-found]

        for dt in (torch.bfloat16, torch.float16):
            roofs = Roofs.measure(dt)
            print(
                f"# measured roofs {dt}: "
                f"compute {roofs.compute_flops_s / 1e12:6.1f} TF/s | "
                f"HBM {roofs.bw_bytes_s / 1e9:6.1f} GB/s | "
                f"ridge {roofs.ridge:5.1f} FLOP/byte"
            )
    except Exception as e:
        print(f"# roof measurement skipped: {e}")
    return 0


def _list() -> int:
    """Print the bench index — the same table as README.md, generated from the registry."""
    have_cuda = _have_cuda()
    print(f"{'name':<32} {'rung':<12} {'kernel':<58} {'gpu':>3}")
    print("-" * 108)
    for b in _BENCHES:
        runnable = "yes" if (have_cuda or not b.gpu) else "—"
        print(f"{b.name:<32} {b.rung:<12} {b.kernel:<58} {runnable:>3}")
    print("-" * 108)
    print(f"# {len(_BENCHES)} benches. CUDA available: {have_cuda}.")
    print("# run one:    python -m bench.kernels.run <name>   (e.g. gemm/triton_tiled)")
    print("# run family: python -m bench.kernels.run <family> (e.g. gemm)")
    print("# run all:    python -m bench.kernels.run --all")
    return 0


def _run_one(b: Bench, extra_args: list[str]) -> int:
    """Launch a single bench as a subprocess (clean process; import failures stay isolated)."""
    script = REPO_ROOT / b.script
    if not script.exists():
        print(f"ERROR: {b.script} not found (registry out of sync?)", file=sys.stderr)
        return 2
    if b.gpu and not _have_cuda():
        print(f"SKIP {b.name}: GPU bench, no CUDA device available (CPU box).")
        return 0
    print(f"\n=== {b.name}  [{b.rung}]  {b.kernel} ===")
    print(f"--- {b.script} {' '.join(extra_args)}".rstrip())
    cmd = [sys.executable, str(script), *extra_args]
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="bench.kernels.run",
        description="List and run the kernel ladder. With no args: list every bench.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Bench names tab-complete as paths (gemm/triton_tiled). Families filter (gemm).",
    )
    ap.add_argument(
        "target",
        nargs="?",
        default=None,
        help="bench name (gemm/triton_tiled) or family (gemm) to run; omit to list",
    )
    ap.add_argument("--all", action="store_true", help="run every bench in the registry")
    ap.add_argument(
        "--gpu-check", action="store_true", help="print arch + measured peaks, run no bench"
    )
    ap.add_argument(
        "bench_args",
        nargs=argparse.REMAINDER,
        help="extra args forwarded to the bench (after -- ), e.g. --n 8192",
    )
    ap.add_argument("--list", action="store_true", help="list every bench (same as no args)")
    args = ap.parse_args()

    # Normalize the forwarded args: argparse REMAINDER keeps a leading "--" as a token.
    extra = [a for a in args.bench_args if a != "--"]

    if args.gpu_check:
        return _gpu_check()
    if args.all:
        rc = 0
        for b in _BENCHES:
            rc |= _run_one(b, extra)
        return rc
    if args.list or args.target is None:
        return _list()

    # Match: exact name, exact family (run all in family), or helpful error.
    matches = [b for b in _BENCHES if b.name == args.target]
    if not matches:
        matches = [b for b in _BENCHES if b.name.split("/")[0] == args.target]
    if not matches:
        print(f"ERROR: no bench or family named {args.target!r}.", file=sys.stderr)
        print("Available: " + ", ".join(sorted({b.name for b in _BENCHES})), file=sys.stderr)
        print(
            "Families:  " + ", ".join(sorted({b.name.split("/")[0] for b in _BENCHES})),
            file=sys.stderr,
        )
        return 2
    rc = 0
    for b in matches:
        rc |= _run_one(b, extra)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
