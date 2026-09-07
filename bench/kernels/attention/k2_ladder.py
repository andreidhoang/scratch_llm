"""The K2 attention ladder's measurement entry point — one methodology for three rungs and their floors.

The sibling of ``bench/kernels/gemm/k1_ladder.py``, and for the same reason: A-R1 through A-R3 are
compared to each other as much as to their floors, so they must be timed the same way or "A-R2
reached 61% of FA3" stops being commensurable with "A-R1 reached 78% of SDPA".

Attention differs from GEMM in two ways that this file has to get right, and that a GEMM harness
would get silently wrong:

  1. **The FLOP count depends on the mask.** A causal kernel does roughly half the work of a dense
     one, so quoting the dense 4·B·H·S²·D against a causal kernel doubles its apparent TFLOP/s.
     :func:`attention_flops` counts what is actually computed, and the causal factor is applied to
     BOTH the kernel and its floor or the comparison means nothing.
  2. **Decode is memory-bound, prefill is compute-bound.** A-R3's exit is stated as a fraction of
     HBM bandwidth, not of peak FLOP/s, because at batch 64 and one query token there is no
     arithmetic intensity to speak of — the kernel is a KV-cache streaming problem. So the decode
     rung reports achieved GB/s and % of measured HBM alongside its latency, and an L2 flush is
     mandatory there (a cache-resident KV read is not a decode).

    python bench/kernels/attention/k2_ladder.py --rung A-R1                measure the rung vs its floor
    python bench/kernels/attention/k2_ladder.py --floor sdpa               measure a floor alone
    python bench/kernels/attention/k2_ladder.py --rung A-R1 --shape s8192  a different spec shape
    python bench/kernels/attention/k2_ladder.py --list                     the registries
    python bench/kernels/attention/k2_ladder.py --rung A-R1 --dry-run      print the plan, run nothing

The last line of stdout is always a single bare number — the rung's metric, or the floor's value.
``experiments/K2/<rung>/{run,floor}.sh`` read exactly that line.

Recorded measurements go through ``infra/bench.sh``, which locks clocks, records provenance, and
refuses to run without a prediction already in the ledger.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "bench"))


@dataclass(frozen=True)
class Shape:
    """One attention problem. ``q_len == 1`` marks a decode shape, which changes the FLOP model."""

    b: int  # batch
    h: int  # query heads
    kv_h: int  # key/value heads (kv_h < h is GQA)
    s: int  # KV sequence length (context)
    d: int  # head dim
    q_len: int | None = None  # queries per sequence; None means prefill (q_len == s)
    causal: bool = True
    page: int | None = None  # paged-KV page size, decode rungs only

    @property
    def is_decode(self) -> bool:
        return self.q_len == 1

    @property
    def queries(self) -> int:
        return self.q_len if self.q_len is not None else self.s


#: The shapes K2 measures. A-R3's is fixed by the plan (§05: B=64, ctx 8k, page 16, hdim 128,
#: GQA 8:1) because it is the shape the FlashInfer comparison is stated at — changing it would
#: mean the exit number no longer refers to the thing the plan committed to.
SHAPES: dict[str, Shape] = {
    "s2048": Shape(b=8, h=32, kv_h=32, s=2048, d=128),
    "s4096": Shape(b=4, h=32, kv_h=32, s=4096, d=128),
    "s8192": Shape(b=2, h=32, kv_h=32, s=8192, d=128),  # where causal block-skip pays most
    "gqa8k": Shape(b=4, h=32, kv_h=4, s=8192, d=128),  # GQA 8:1, the serving-shaped prefill
    "noncausal": Shape(
        b=4, h=32, kv_h=32, s=4096, d=128, causal=False
    ),  # the mask factor, isolated
    "decode64": Shape(
        b=64, h=32, kv_h=4, s=8192, d=128, q_len=1, page=16
    ),  # plan §05 A-R3, verbatim
}


@dataclass(frozen=True)
class Rung:
    name: str
    module: str
    fn: str
    arch: tuple[int, int] | None  # None = any CUDA device (the Triton rung runs anywhere)
    floor: str
    metric: str
    note: str = ""
    default_shapes: tuple[str, ...] = field(default=("s4096",))
    l2_flush: bool = False  # decode reads the KV cache from HBM; a warm L2 is not a decode


RUNGS: dict[str, Rung] = {}


#: What a rung's number only means something against, at matched dtype, shape and mask.
FLOORS: dict[str, str] = {
    "sdpa": "torch.nn.functional.scaled_dot_product_attention, FLASH backend only "
    "(the MATH backend is a different algorithm and would flatter a flash kernel)",
    "fa3": "flash-attn 3's Hopper kernel (flash_attn_interface). Absent on non-Hopper, and the "
    "rung says so rather than silently falling back to FA2",
    "flashinfer": "flashinfer's paged decode (BatchDecodeWithPagedKVCacheWrapper) at the same "
    "page size and GQA ratio",
    "hbm": "measured HBM bandwidth on this device — the roof a decode kernel is judged against, "
    "since at one query token there is no arithmetic intensity to speak of",
}


def attention_flops(sh: Shape) -> float:
    """FLOPs actually computed, mask included.

    Two GEMMs per (batch, head): ``S = QK^T`` is ``2·q·s·d`` and ``O = PV`` is another ``2·q·s·d``,
    so ``4·B·H·q·S·D`` dense. Causal masking removes the strictly-upper triangle, which for a
    full-length prefill is very nearly half — and applying that factor to the kernel but not to the
    floor (or the reverse) is the single easiest way to manufacture a wrong percentage here.
    """
    dense = 4.0 * sh.b * sh.h * sh.queries * sh.s * sh.d
    if not sh.causal or sh.is_decode:
        return dense
    # (q·s + q)/2 of the q·s score entries survive, exactly, for a bottom-right aligned mask.
    return dense * (sh.queries * sh.s + sh.queries) / (2.0 * sh.queries * sh.s)


def attention_bytes(sh: Shape, elem_bytes: int = 2) -> float:
    """HBM traffic a perfect implementation would move — the denominator of the decode roofline.

    Decode reads the whole KV cache once and writes one output row per query head; Q is negligible.
    Prefill also streams Q and O, but is compute-bound, so this is reported and not gated on.
    """
    kv = 2.0 * sh.b * sh.kv_h * sh.s * sh.d * elem_bytes
    qo = 2.0 * sh.b * sh.h * sh.queries * sh.d * elem_bytes
    return kv + qo


def _list() -> int:
    print(f"{'rung':<8} {'arch':<9} {'floor':<12} {'metric':<22} kernel")
    print("-" * 108)
    for r in RUNGS.values():
        arch = "any" if r.arch is None else f"sm_{r.arch[0] * 10 + r.arch[1]}"
        print(f"{r.name:<8} {arch:<9} {r.floor:<12} {r.metric:<22} {r.module}.{r.fn}")
        blocked = [x for x in SHAPES if x not in r.default_shapes]
        print(
            f"{'':<8} runs: {' '.join(r.default_shapes)}"
            + (f"   | not applicable: {' '.join(blocked)}" if blocked else "")
        )
    if not RUNGS:
        print("(no rungs wired yet — add a Rung row when a wrapper exists)")
    print("-" * 108)
    print(
        f"{'shape':<11} {'B':>4} {'H':>4} {'KVH':>4} {'S':>6} {'D':>4} {'q':>5} {'causal':>7}   GFLOP    KV MiB"
    )
    for name, sh in SHAPES.items():
        print(
            f"{name:<11} {sh.b:>4} {sh.h:>4} {sh.kv_h:>4} {sh.s:>6} {sh.d:>4} {sh.queries:>5} "
            f"{str(sh.causal):>7}   {attention_flops(sh) / 1e9:7.1f}  {attention_bytes(sh) / 2**20:8.1f}"
        )
    missing = sorted({"A-R1", "A-R2", "A-R3"} - set(RUNGS))
    if missing:
        print(f"\n# not yet wired: {', '.join(missing)}")
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
    ap.add_argument("--rung", choices=sorted(RUNGS) or None, help="which K2 rung to measure")
    ap.add_argument("--floor", choices=sorted(FLOORS), help="measure a floor instead of a rung")
    ap.add_argument("--shape", choices=sorted(SHAPES), default="s4096")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--backward", action="store_true", help="measure the backward pass (A-R1 only)")
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
    if args.rung and args.shape not in RUNGS[args.rung].default_shapes:
        ap.error(
            f"{args.rung} does not run shape {args.shape}. It runs: "
            f"{', '.join(RUNGS[args.rung].default_shapes)}"
        )

    sh = SHAPES[args.shape]
    flops = attention_flops(sh)
    what = f"rung {args.rung}" if args.rung else f"floor {args.floor}"

    if args.dry_run:
        print(f"# k2_ladder [dry-run] · {what} · shape {args.shape} · {args.dtype}")
        print(
            f"#   B={sh.b} H={sh.h} KVH={sh.kv_h} S={sh.s} D={sh.d} q={sh.queries} "
            f"causal={sh.causal}{f' page={sh.page}' if sh.page else ''}"
        )
        print(
            f"#   {flops / 1e9:.1f} GFLOP (mask-adjusted) · {attention_bytes(sh) / 2**20:.1f} MiB KV+QO"
            f" · warmup {args.warmup} · iters {args.iters} · seed {args.seed}"
        )
        if args.rung:
            r = RUNGS[args.rung]
            print(f"#   kernel {r.module}.{r.fn} · floor {r.floor} · L2 flush {r.l2_flush}")
            print(
                f"#   metric {r.metric} — `make predict L=K2 R={r.name} M={r.metric} V=<yours>` first"
            )
        return 0

    import torch

    if not torch.cuda.is_available():
        print(
            "# no CUDA device — an attention number is a statement about silicon, and there is none here."
        )
        print(f"# on the box:  infra/bench.sh -l K2 -r {args.rung or 'A-R1'} -n -- \\")
        print(
            f"#                python bench/kernels/attention/k2_ladder.py {' '.join(sys.argv[1:])}"
        )
        return 0

    print("# K2 rungs are not wired into this harness yet — add a Rung row when a wrapper exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
