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
import json
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


RUNGS: dict[str, Rung] = {
    "A-R1": Rung(
        name="A-R1",
        module="scratch_llm.kernels.attention.prefill.fa2_tuned",
        fn="fa2_tuned_forward",
        arch=None,  # Triton; runs on any CUDA device hopper_contracts.ARCH has a row for
        floor="sdpa",
        metric="pct_of_sdpa",
        note="arch-keyed configs, causal block skip, exp2, split bwd. --backward measures "
        "fa2_tuned_backward against SDPA's, as the second ledger row pct_of_sdpa_bwd.",
        # gqa8k is absent by construction: the wrapper flattens every leading dim into one batch
        # axis, so a KV head cannot be shared by several query heads, and unsupported_reason
        # rejects the shape by name rather than quietly attending to the wrong K.
        default_shapes=("s2048", "s4096", "s8192", "noncausal"),
    ),
    "A-R2": Rung(
        name="A-R2",
        module="scratch_llm.kernels.attention.prefill.fa3_v2",
        fn="fa3_hopper_v2_fwd",
        arch=(9, 0),
        floor="fa3",
        metric="pct_of_fa3",
        note="TMA K/V mbarrier ring, 1 producer warp + 2 consumer warpgroups, wgmma RS for O+=PV. "
        "bf16, D=128, 128x128, 2 stages. Exit >= 60% of FA3 fwd; kill < 40%.",
        default_shapes=("s2048", "s4096", "s8192", "gqa8k", "noncausal"),
    ),
    "A-R3": Rung(
        name="A-R3",
        module="scratch_llm.kernels.attention.decode.paged_split_kv",
        fn="a_r3_paged_decode",
        arch=(9, 0),
        floor="flashinfer",
        metric="pct_of_hbm",
        # Rung carries one metric and this rung's exit has two clauses, so the harness prints
        # pct_of_flashinfer beside pct_of_hbm and puts both in the JSON row. The bare last line is
        # pct_of_hbm, which is what `make predict M=pct_of_hbm` is written against.
        note="split-KV paged decode + log-sum-exp merge. Exit is BOTH >= 85% of measured HBM and "
        "<= 15% behind FlashInfer; the harness prints both, the ledger row is pct_of_hbm.",
        default_shapes=("decode64",),
        l2_flush=True,
    ),
}


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


def _alloc_prefill(sh: Shape, layout: str, device: str = "cuda"):
    """``(q, k, v)`` bf16 for a prefill shape, in the layout the rung and its floor both read.

    Allocated per layout rather than transposed into place. A ``.transpose(1, 2)`` view of a
    contiguous ``[B,S,H,D]`` has exactly the strides SDPA's flash backend wants, so that side is
    free — but ``fa2_tuned`` reshapes its leading dims, and a reshape of a non-contiguous view
    copies. That copy would land inside the region a later reader takes for pure kernel. Two
    allocations of the same bytes cost nothing outside the timed window and remove the question.
    """
    import torch

    q_shape = (sh.b, sh.s, sh.h, sh.d) if layout == "bshd" else (sh.b, sh.h, sh.s, sh.d)
    kv_shape = (sh.b, sh.s, sh.kv_h, sh.d) if layout == "bshd" else (sh.b, sh.kv_h, sh.s, sh.d)

    def mk(shape: tuple[int, ...]):  # noqa: ANN202 - torch.Tensor, without importing torch at module scope
        return torch.randn(shape, device=device, dtype=torch.bfloat16)

    return mk(q_shape), mk(kv_shape), mk(kv_shape)


def _alloc_decode(sh: Shape, seed: int, device: str = "cuda"):
    """``(q, k_cache, v_cache, table)`` for a decode shape — one query token, a paged KV pool.

    ``shuffle_seed`` is passed on purpose: a sequential page table is the easy case, and a kernel
    that computes its page address from the logical index instead of reading ``indices`` is exactly
    right on it and wrong on the fragmented table a real server has. The FlashInfer floor reads
    ``indices`` too, so both sides see the same fragmentation.
    """
    import torch

    from scratch_llm.kernels.attention.decode.page_table import PageTable, empty_kv_pool

    assert sh.page is not None, "a decode shape must declare a page size"
    table = PageTable.from_seq_lens(
        [sh.s] * sh.b, page_size=sh.page, shuffle_seed=seed, device=device
    )
    k_cache, v_cache = empty_kv_pool(
        num_pages=table.num_pages,
        page_size=sh.page,
        num_kv_heads=sh.kv_h,
        head_dim=sh.d,
        device=device,
    )
    k_cache.normal_()
    v_cache.normal_()
    q = torch.randn(sh.b, sh.h, sh.d, device=device, dtype=torch.bfloat16)
    return q, k_cache, v_cache, table


def _floor_prefill(key: str, sh: Shape, q, k, v):  # noqa: ANN001, ANN202
    """``(timed_fn, ctx)`` for a prefill floor. Everything that is not the kernel is done here.

    ``ctx`` is entered around the whole timing loop, not inside the timed callable: forcing SDPA's
    backend costs a dispatcher push/pop per call, and at these shapes that is small but it is not
    zero, and it is not part of what the floor is.
    """
    import contextlib

    import torch.nn.functional as F  # noqa: N812

    if key == "sdpa":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        gqa = sh.kv_h != sh.h
        # FLASH forced, and no fallback: the MATH backend is a different algorithm with different
        # numerics and a different memory profile, and a flash kernel measured against it looks
        # good for a reason that has nothing to do with this rung.
        ctx = sdpa_kernel(SDPBackend.FLASH_ATTENTION)

        def run():  # noqa: ANN202
            return F.scaled_dot_product_attention(q, k, v, is_causal=sh.causal, enable_gqa=gqa)

        return run, ctx

    if key == "fa3":
        from scratch_llm.kernels.attention.prefill import fa3_v2

        why = fa3_v2.floor_unavailable_reason()
        if why is not None:
            raise RuntimeError(f"FA3 floor unavailable: {why}")
        from flash_attn_interface import flash_attn_func  # noqa: PLC0415

        def run():  # noqa: ANN202
            return flash_attn_func(q, k, v, causal=sh.causal)

        return run, contextlib.nullcontext()

    raise ValueError(f"{key} is not a prefill floor")


def _oracle_maxdiff(out, ref) -> float:  # noqa: ANN001
    """max |out − ref| / max |ref| — a scale-free agreement number, not a gate.

    The gate is the rung's test file with a tolerance Huy derives. This is the smoke check that
    stops a benchmark from reporting throughput for a kernel that is computing something else.
    """
    d = (out.float() - ref.float()).abs().max().item()
    return d / (ref.float().abs().max().item() + 1e-30)


def _resolve_named(module: str, fn: str):  # noqa: ANN202 - returns an opaque callable
    import importlib

    return getattr(importlib.import_module(module), fn)


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
    if args.backward and args.rung != "A-R1":
        ap.error("--backward exists for A-R1 only; A-R2 and A-R3 are forward-only rungs")
    if args.floor == "flashinfer" and not SHAPES[args.shape].is_decode:
        ap.error("the flashinfer floor is a paged DECODE kernel; use --shape decode64")
    if args.rung == "A-R3" and args.floor:
        ap.error("A-R3 measures against its own floor; drop --floor (or use it alone)")

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
            # The backward is a second ledger row against the same floor, not a variant of the
            # first: it has its own prediction, its own metric, and 5 GEMMs to the forward's 2.
            metric = "pct_of_sdpa_bwd" if args.backward else r.metric
            if args.backward:
                print(f"#   BACKWARD — FLOPs x 2.5 = {flops * 2.5 / 1e9:.1f} GFLOP (5 GEMMs vs 2)")
            print(f"#   kernel {r.module}.{r.fn} · floor {r.floor} · L2 flush {r.l2_flush}")
            print(
                f"#   metric {metric} — `make predict L=K2 R={r.name} M={metric} V=<yours>` first"
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

    from _harness import (  # type: ignore[import-not-found]
        bench_ms,
        measure_mem_bw_bytes_s,
        provenance_line,
        spread_pct,
    )

    torch.manual_seed(args.seed)
    print(provenance_line(f"K2 {what} · {args.shape} · {args.dtype}"))
    # do_bench flushes the L2 between reps unconditionally, so EVERY row here — kernel and floor
    # alike — is cold-L2 steady state. Rung.l2_flush is therefore a declaration of which rung the
    # flush is load-bearing for (A-R3: a cache-resident KV read is not a decode), not a switch.
    print("# timing      CUDA events, L2 flushed per rep (do_bench); median of p20/p50/p80")

    floor_key = args.floor or RUNGS[args.rung].floor
    row: dict[str, object] = {
        "ladder": "K2",
        "shape": args.shape,
        "dims": {
            "b": sh.b,
            "h": sh.h,
            "kv_h": sh.kv_h,
            "s": sh.s,
            "d": sh.d,
            "q_len": sh.queries,
            "causal": sh.causal,
            "page": sh.page,
        },
        "dtype": args.dtype,
        "flops": flops,
        "bytes": attention_bytes(sh),
        "floor_name": floor_key,
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
        "backward": bool(args.backward),
    }

    def timed(fn, ctx=None) -> tuple[float, float]:  # noqa: ANN001
        """``(median ms, p20-p80 spread %)``, with any backend-forcing context entered around it."""
        import contextlib

        with ctx or contextlib.nullcontext():
            med, lo, hi = bench_ms(fn, warmup=args.warmup, rep=args.iters)
        return med, spread_pct(med, lo, hi)

    # =========================================================================================
    # The HBM roof. Decode is judged against it, so it is measured on this card rather than read
    # off a datasheet — and it is measured whenever a decode shape is in play, including for the
    # floor alone, because `pct_of_hbm` without the denominator measured here is not a number.
    # =========================================================================================
    if floor_key == "hbm" or sh.is_decode:
        hbm_bytes_s = measure_mem_bw_bytes_s()
        row["hbm_bytes_s"] = hbm_bytes_s
        print(
            f"# floor hbm       {hbm_bytes_s / 1e9:7.1f} GB/s  measured (d2d copy)  {FLOORS['hbm']}"
        )
        if floor_key == "hbm":
            if args.json:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(json.dumps(row, indent=1))
            print(f"{hbm_bytes_s / 1e9:.4f}")
            return 0

    # =========================================================================================
    # Prefill: A-R1, A-R2, and the sdpa / fa3 floors. TFLOP/s with the mask factor applied to both
    # sides — attention_flops() already carries it, and it is the same function for kernel and
    # floor, which is the only way the ratio means anything.
    # =========================================================================================
    if not sh.is_decode:
        layout = "bshd" if floor_key == "fa3" else "bhsd"
        q, k, v = _alloc_prefill(sh, layout)
        try:
            floor_fn, floor_ctx = _floor_prefill(floor_key, sh, q, k, v)
        except RuntimeError as e:
            # No bare number on the last line: bench.sh must record nothing rather than record a
            # number against a floor that was not the floor.
            print(f"# REFUSED — {e}")
            return 3

        bwd_scale = 2.5  # bwd is 5 GEMMs to the forward's 2 (dQ, dK, dV, plus recompute of S, P)
        if args.backward:
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)
            with floor_ctx:
                floor_out = floor_fn()
            dout = torch.randn_like(floor_out)

            def floor_bwd(out=floor_out, dout=dout):  # noqa: ANN202
                return torch.autograd.grad(out, (q, k, v), dout, retain_graph=True)

            floor_ms, floor_spread = timed(floor_bwd, floor_ctx)
            floor_tf = flops * bwd_scale / (floor_ms * 1e-3) / 1e12
        else:
            floor_ms, floor_spread = timed(floor_fn, floor_ctx)
            floor_tf = flops / (floor_ms * 1e-3) / 1e12
        print(
            f"# floor {floor_key:<9} {floor_tf:7.1f} TF/s  ({floor_ms:8.3f} ms · IQR "
            f"{floor_spread:4.1f}%)  {FLOORS[floor_key]}"
        )
        row |= {"floor_tflops": floor_tf, "floor_ms": floor_ms, "floor_iqr_pct": floor_spread}
        result = floor_tf

        if args.rung:
            r = RUNGS[args.rung]
            kernel = _resolve(r)
            # fa2_tuned says `is_causal`, fa3_v2 says `causal`. Named here rather than normalised
            # in the wrappers: each wrapper matches the upstream call it stands next to.
            kw = {"causal": sh.causal} if r.name == "A-R2" else {"is_causal": sh.causal}
            with torch.no_grad():
                out = kernel(q, k, v, **kw)
                o_kernel = out[0] if isinstance(out, tuple) else out
                # The oracle is the tiled fp32 recurrence over the SAME bf16 operands. Cheap
                # enough to run at the bench shape because it never materialises S x S.
                from scratch_llm.kernels.attention.prefill.fa3_v2 import reference_attention

                o_ref, _lse_ref = reference_attention(q, k, v, causal=sh.causal, layout=layout)
                rel = _oracle_maxdiff(o_kernel, o_ref)
            print(
                f"# correctness  max|dO|/max|O| vs fp32 oracle = {rel:.3e}   "
                f"(the GATE, with the tolerance and the LSE check, is "
                f"tests/kernels/attention/test_k2_{r.name.lower().replace('-', '_')}.py)"
            )
            del out, o_kernel, o_ref, _lse_ref

            if args.backward:
                bwd = _resolve_named(r.module, "fa2_tuned_backward")
                with torch.no_grad():
                    o_f, lse_f = kernel(q, k, v, **kw)
                do = torch.randn_like(o_f)

                def rung_fn(o_f=o_f, lse_f=lse_f, do=do):  # noqa: ANN202
                    return bwd(q, k, v, o_f, lse_f, do, is_causal=sh.causal)
            else:

                def rung_fn():  # noqa: ANN202
                    return kernel(q, k, v, **kw)

            ms, spread = timed(rung_fn)
            tf = flops * (bwd_scale if args.backward else 1.0) / (ms * 1e-3) / 1e12
            pct = 100.0 * tf / floor_tf
            metric = "pct_of_sdpa_bwd" if args.backward else r.metric
            print(f"# {r.name:<12} {tf:7.1f} TF/s  ({ms:8.3f} ms · IQR {spread:4.1f}%)  {r.note}")
            print(f"# {metric:<14} {pct:7.1f}")
            if spread > 5.0:
                print(
                    "# WARNING IQR > 5% — clocks are not stable; this row must not enter the ledger."
                )
            row |= {
                "rung": r.name,
                "metric": metric,
                "tflops": tf,
                "ms": ms,
                "iqr_pct": spread,
                "rel_err": rel,
            }
            result = pct
            print(
                f"#\n# record it:  make measure L=K2 R={r.name} M={metric} V={pct:.1f} "
                f'DEV="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" '
                f"FLOOR={floor_key} FLOORV={floor_tf:.1f}"
            )

    # =========================================================================================
    # Decode: A-R3 and the flashinfer floor. The metric is % of measured HBM, because at one query
    # token there is no arithmetic intensity — 512 MiB of KV read against 512 KiB written.
    # =========================================================================================
    else:
        q, k_cache, v_cache, table = _alloc_decode(sh, args.seed)
        kv_bytes = attention_bytes(sh)
        from scratch_llm.kernels.attention.decode import paged_split_kv as psk

        wrapper, kv = psk.flashinfer_reference(q, k_cache, v_cache, table, use_tensor_cores=False)
        # plan() syncs to the host, so it is here and not in the timed callable. Read back which
        # kernel the floor actually planned: FlashInfer's split-KV only fires when
        # batch*gdy >= max_grid_size (scheduler.cuh:183), and the floor is a different kernel in
        # the two regimes — a fact that cannot be assumed from the batch size alone.
        plan_info = getattr(wrapper, "_plan_info", None)
        if plan_info is not None:
            info = [int(x) for x in plan_info]
            print(
                f"# floor plan   padded_batch_size={info[0]} split_kv={info[9]}  "
                f"(read back, not assumed — scheduler.cuh:183)"
            )
            row |= {"floor_padded_batch_size": info[0], "floor_split_kv": info[9]}

        floor_ms, floor_spread = timed(lambda: wrapper.run(q, kv))
        floor_gbs = kv_bytes / (floor_ms * 1e-3) / 1e9
        print(
            f"# floor {floor_key:<9} {floor_ms * 1e3:7.1f} us  ({floor_gbs:7.1f} GB/s · IQR "
            f"{floor_spread:4.1f}%)  {FLOORS[floor_key]}"
        )
        row |= {
            "floor_us": floor_ms * 1e3,
            "floor_gbs": floor_gbs,
            "floor_iqr_pct": floor_spread,
            "floor_use_tensor_cores": False,
        }
        result = floor_ms * 1e3  # the floor's own ledger metric is flashinfer_decode64_us

        if args.rung:
            r = RUNGS[args.rung]
            kernel = _resolve(r)
            o_ref, _lse_ref = psk.reference_paged_decode(q, k_cache, v_cache, table)
            o_k, _lse_k = kernel(q, k_cache, v_cache, table)
            rel = _oracle_maxdiff(o_k, o_ref)
            print(
                f"# correctness  max|dO|/max|O| vs fp32 de-paged oracle = {rel:.3e}   "
                f"(the GATE is tests/kernels/attention/test_k2_a_r3.py)"
            )
            del o_ref, _lse_ref, o_k, _lse_k

            ms, spread = timed(lambda: kernel(q, k_cache, v_cache, table))
            gbs = kv_bytes / (ms * 1e-3) / 1e9
            pct_hbm = 100.0 * (kv_bytes / (ms * 1e-3)) / hbm_bytes_s
            pct_fi = 100.0 * floor_ms / ms  # >100 == faster than FlashInfer
            print(
                f"# {r.name:<12} {ms * 1e3:7.1f} us  ({gbs:7.1f} GB/s · IQR {spread:4.1f}%)  {r.note}"
            )
            print(f"# pct_of_hbm     {pct_hbm:7.1f}      <- the ledger row")
            print(
                f"# pct_of_flashinfer {pct_fi:7.1f}   <- the second exit clause (>= 85 means within 15%)"
            )
            if spread > 5.0:
                print(
                    "# WARNING IQR > 5% — clocks are not stable; this row must not enter the ledger."
                )
            row |= {
                "rung": r.name,
                "metric": r.metric,
                "us": ms * 1e3,
                "gbs": gbs,
                "iqr_pct": spread,
                "rel_err": rel,
                "pct_of_hbm": pct_hbm,
                "pct_of_flashinfer": pct_fi,
            }
            result = pct_hbm
            print(
                f"#\n# record it:  make measure L=K2 R={r.name} M={r.metric} V={pct_hbm:.1f} "
                f'DEV="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" '
                f"FLOOR={floor_key} FLOORV={floor_ms * 1e3:.1f}"
            )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(row, indent=1))
    torch.cuda.empty_cache()
    # The last line is the number, bare — run.sh and floor.sh read exactly this.
    print(f"{result:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
