"""S1/S-R4 — the fused-MoE measurement entry point: two regimes, two floors, one methodology.

Qwen3-30B-A3B's MoE layer, measured at the two points the plan row names, against vLLM's fused
MoE at matched model geometry, dtype, shape AND routing. They are two different problems wearing
one name, which is why they are two ledger rows and never one:

  * **prefill 4k** — 4096 tokens x top_k 8 = 32768 rows over 128 experts, ~256 rows each. The
    expert GEMMs are big enough to be arithmetic; the question is what the permutation costs
    (128 MiB of activations gathered and scattered that vLLM's index-gather never moves) and
    whether the ragged tiles keep the tensor pipe fed.
  * **decode B=64** — 512 rows over 128 experts, ~4 each. There is no arithmetic here at all: the
    layer reads ~1.2 GB of expert weights to multiply 512 rows by them, so it is a weight-streaming
    problem with a tile-quantisation tax on top (a group of 4 rows still costs a whole BLOCK_M
    tile). Empty experts are normal at this size and the kernel has to survive them.
  * **EP=2** — the same decode point split over two H100s: half the experts and half the tokens
    each, so every slot whose expert lives on the peer travels twice (dispatch, combine). The
    number is the comm fraction, and it needs ``torchrun --nproc_per_node=2``.

The floor is vLLM's own Triton path, fed the IDENTICAL ``topk_ids``/``topk_weights`` this rung
routes with, so a difference in output is arithmetic and never a difference in which experts were
picked. Today that path is ``fused_topk`` + ``fused_experts`` (the monolithic ``fused_moe`` entry
point is gone from ``oss/vllm``); the expert compute is the floor row, and the router's own time is
printed beside it because this rung's instrument takes the routing as an input.

    python bench/kernels/moe/s1_moe.py --point prefill4k             the rung vs its floor
    python bench/kernels/moe/s1_moe.py --floor vllm --point decode64 the floor alone
    python bench/kernels/moe/s1_moe.py --point decode64 --dry-run    the plan, measuring nothing
    torchrun --nproc_per_node=2 bench/kernels/moe/s1_moe.py --point ep2_decode64

The last line of stdout is always a single bare number — the point's metric, or the floor's value.
``experiments/S1/S-R4/{run,floor}.sh`` read exactly that line. Recorded measurements go through
``infra/bench.sh``, which locks clocks, records provenance, and refuses to run without a prediction
already in the ledger.

Spec: experiments/S1/S-R4/spec.md   ·   Map: experiments/S1/S-R4/map.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "bench"))


@dataclass(frozen=True)
class ModelShape:
    """One MoE layer's geometry. Qwen3-30B-A3B, from the plan row."""

    n_experts: int
    top_k: int
    hidden: int
    intermediate: int

    @property
    def weight_bytes(self) -> int:
        """bf16 expert weights for the whole layer: w1 (E, 2I, H) + w2 (E, H, I)."""
        return self.n_experts * 3 * self.intermediate * self.hidden * 2


#: E=128 and I=768 are corroborated by vLLM's own tuned-config filenames
#: (``oss/vllm/vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=*.json``).
#: hidden=2048, top_k=8 and ``norm_topk_prob`` come from the HF ``config.json``, which is not in
#: this tree — VERIFY THEM ON THE BOX before the first recorded run; a wrong hidden size changes
#: every byte count in the spec.
QWEN3_30B_A3B = ModelShape(n_experts=128, top_k=8, hidden=2048, intermediate=768)


@dataclass(frozen=True)
class Point:
    """One regime. ``ep_size > 1`` makes it a distributed point and changes the metric."""

    name: str
    n_tokens: int  # tokens the LAYER sees (summed over ranks when ep_size > 1)
    metric: str
    ep_size: int = 1
    note: str = ""

    @property
    def is_distributed(self) -> bool:
        return self.ep_size > 1


POINTS: dict[str, Point] = {
    "prefill4k": Point(
        name="prefill4k",
        n_tokens=4096,
        metric="pct_of_vllm_prefill4k",
        note="plan §05: prefill 4k. 32768 routed rows; the permutation's 128 MiB is the question.",
    ),
    "decode64": Point(
        name="decode64",
        n_tokens=64,
        metric="pct_of_vllm_decode64",
        note="plan §05: decode B=64. 512 routed rows against ~1.2 GB of weights; empty experts are normal.",
    ),
    "ep2_decode64": Point(
        name="ep2_decode64",
        n_tokens=64,
        metric="comm_fraction_ep2_decode64",
        ep_size=2,
        note="plan §05: EP=2 on 2xH100, NCCL all-to-all. The number is comm time / layer time.",
    ),
}

FLOORS: dict[str, str] = {
    "vllm": "vllm.model_executor.layers.fused_moe.fused_experts — the Triton fused MoE, at the "
    "same weights, dtype and routing tensors (fused_moe.py:1593)",
}


def _floor_metric(point: Point) -> str:
    return f"vllm_fused_experts_{point.name}_ms"


def _alloc(point: Point, model: ModelShape, seed: int, device: str = "cuda"):  # noqa: ANN202
    """``(x, w1, w2, router_logits)`` in bf16 at vLLM's weight layout: w1 (E, 2I, H), w2 (E, H, I).

    Scaled by 1/sqrt(fan_in) so the activations stay in a range where bf16 rounding is the only
    error in play — an unscaled ``randn`` weight makes the second GEMM overflow the exponent at
    K=2048 and turns a correctness check into a comparison of two infinities.
    """
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    e, i, h = model.n_experts, model.intermediate, model.hidden
    x = torch.randn(point.n_tokens, h, generator=g, device=device, dtype=torch.bfloat16)
    w1 = (
        torch.randn(e, 2 * i, h, generator=g, device=device, dtype=torch.float32) * h**-0.5
    ).bfloat16()
    w2 = (
        torch.randn(e, h, i, generator=g, device=device, dtype=torch.float32) * i**-0.5
    ).bfloat16()
    logits = torch.randn(point.n_tokens, e, generator=g, device=device, dtype=torch.float32)
    return x, w1, w2, logits


def _list() -> int:
    print(f"{'point':<14} {'tokens':>7} {'rows':>7} {'EP':>3}  {'metric':<28} note")
    print("-" * 132)
    for p in POINTS.values():
        rows = p.n_tokens * QWEN3_30B_A3B.top_k
        print(f"{p.name:<14} {p.n_tokens:>7} {rows:>7} {p.ep_size:>3}  {p.metric:<28} {p.note}")
    print("-" * 132)
    m = QWEN3_30B_A3B
    print(
        f"# Qwen3-30B-A3B MoE layer: E={m.n_experts} top_k={m.top_k} hidden={m.hidden} "
        f"intermediate={m.intermediate} · bf16 weights {m.weight_bytes / 2**30:.2f} GiB/layer"
    )
    print(f"# floors: {', '.join(f'{k} = {v}' for k, v in FLOORS.items())}")
    return 0


def _dry_run(point: Point, model: ModelShape, args) -> int:  # noqa: ANN001
    rows = point.n_tokens * model.top_k
    from scratch_llm.kernels.moe.grouped_gemm import default_config, max_m_tiles

    cfg = default_config(
        n_tokens=point.n_tokens,
        n_experts=model.n_experts,
        n_out=2 * model.intermediate,
        k_dim=model.hidden,
    )
    print(f"# s1_moe [dry-run] · point {point.name} · bf16 · {point.note}")
    print(
        f"#   tokens={point.n_tokens} top_k={model.top_k} rows={rows} experts={model.n_experts} "
        f"hidden={model.hidden} intermediate={model.intermediate} ep_size={point.ep_size}"
    )
    print(
        f"#   {6.0 * rows * model.hidden * model.intermediate / 1e9:.1f} GFLOP · weights "
        f"{model.weight_bytes / 2**30:.2f} GiB · tiles <= {max_m_tiles(rows, model.n_experts, cfg.block_m)} "
        f"at BLOCK_M={cfg.block_m} · config {cfg.as_row()}"
    )
    print(f"#   warmup {args.warmup} · iters {args.iters} · seed {args.seed} · L2 flushed per rep")
    print(
        f"#   metric {point.metric} — `make predict L=S1 R=S-R4 M={point.metric} V=<yours>` first"
    )
    if not point.is_distributed:
        print(f"#   floor  {_floor_metric(point)} — {FLOORS['vllm']}")
    else:
        print("#   needs: torchrun --nproc_per_node=2 (NCCL all-to-all; one process is not EP=2)")
    return 0


def _vllm_floor(x, w1, w2, topk_weights, topk_ids):  # noqa: ANN001, ANN202
    """The floor call, with the identical routing tensors this rung used. Raises if vLLM is absent."""
    from vllm.model_executor.layers.fused_moe import fused_experts  # noqa: PLC0415

    def run():  # noqa: ANN202
        return fused_experts(x, w1, w2, topk_weights, topk_ids)

    return run


def _relerr(out, ref) -> float:  # noqa: ANN001
    """max|out − ref| / max|ref| — the smoke check, not the gate. The gate is
    tests/kernels/moe/test_s1_s_r4.py with a tolerance Huy sets."""
    d = (out.float() - ref.float()).abs().max().item()
    return d / (ref.float().abs().max().item() + 1e-30)


def _run_single(point: Point, model: ModelShape, args) -> int:  # noqa: ANN001
    """The non-distributed points: the rung against the vLLM floor, or the floor alone."""
    import torch

    from _harness import bench_ms, provenance_line, spread_pct  # type: ignore[import-not-found]
    from scratch_llm.kernels.moe.fused import reference_moe, s_r4_fused_moe, trace_moe
    from scratch_llm.kernels.moe.grouped_gemm import exact_m_tiles, max_m_tiles
    from scratch_llm.kernels.moe.routing import route_topk

    torch.manual_seed(args.seed)
    print(provenance_line(f"S1 S-R4 · {point.name} · bf16"))
    print("# timing      CUDA events, L2 flushed per rep (do_bench); median of p20/p50/p80")

    x, w1, w2, logits = _alloc(point, model, args.seed)
    routing = route_topk(logits, model.top_k)
    trace = trace_moe(routing, hidden=model.hidden, intermediate=model.intermediate, elem_bytes=2)
    row: dict[str, object] = {
        "ladder": "S1",
        "rung": "S-R4",
        "point": point.name,
        "dtype": "bf16",
        "model": {
            "n_experts": model.n_experts,
            "top_k": model.top_k,
            "hidden": model.hidden,
            "intermediate": model.intermediate,
        },
        "n_tokens": point.n_tokens,
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
        "trace": trace.as_row(),
    }
    load = trace.load
    print(
        f"# load        {load.n_empty} empty experts · max/mean {load.max_over_mean:.2f} · "
        f"{load.total_slots} slots (= {point.n_tokens} tokens x top_k {model.top_k})"
    )
    exact = exact_m_tiles(trace.permutation.group_offsets, trace.config.block_m)
    bound = max_m_tiles(trace.n_rows, model.n_experts, trace.config.block_m)
    print(
        f"# tiles       {exact} carry rows, {bound} launched · row padding "
        f"{trace.row_padding_ratio:.2f}x at BLOCK_M={trace.config.block_m}"
    )

    # ---- the floor -------------------------------------------------------------------------
    floor_ms = float("nan")
    try:
        floor_fn = _vllm_floor(x, w1, w2, routing.topk_weights, routing.topk_ids)
        floor_out = floor_fn()
        floor_ms, lo, hi = bench_ms(floor_fn, warmup=args.warmup, rep=args.iters)
        print(
            f"# floor vllm  {floor_ms:8.3f} ms  (IQR {spread_pct(floor_ms, lo, hi):4.1f}%)  {FLOORS['vllm']}"
        )
        row |= {
            "floor_name": "vllm",
            "floor_ms": floor_ms,
            "floor_iqr_pct": spread_pct(floor_ms, lo, hi),
        }
    except Exception as e:  # noqa: BLE001 - any vLLM import/dispatch failure is the same refusal
        if args.floor:
            print(f"# REFUSED — the floor did not run: {e}")
            return 3
        print(f"# floor vllm  UNAVAILABLE ({e}) — no percentage can be quoted; rung timing only")
        floor_out = None

    if args.floor:
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(row, indent=1))
        print(f"{floor_ms:.6f}")
        return 0

    # ---- the rung --------------------------------------------------------------------------
    out = s_r4_fused_moe(x, w1, w2, routing)
    ref = reference_moe(x, w1, w2, routing)
    print(
        f"# correctness max|dy|/max|y| vs fp32 per-expert oracle = {_relerr(out, ref):.3e}   "
        f"(the GATE is tests/kernels/moe/test_s1_s_r4.py)"
    )
    if floor_out is not None:
        print(f"# vs floor    max|dy|/max|y| vs vLLM fused_experts = {_relerr(out, floor_out):.3e}")
    row["rel_err_oracle"] = _relerr(out, ref)
    del ref, floor_out

    ms, lo, hi = bench_ms(
        lambda: s_r4_fused_moe(x, w1, w2, routing), warmup=args.warmup, rep=args.iters
    )
    iqr = spread_pct(ms, lo, hi)
    tflops = trace.flops / (ms * 1e-3) / 1e12
    pct = 100.0 * floor_ms / ms
    print(f"# S-R4        {ms:8.3f} ms  ({tflops:7.2f} TF/s · IQR {iqr:4.1f}%)  {point.note}")
    print(
        f"# {point.metric:<24} {pct:7.1f}   <- the ledger row (100 = parity; >= 66.7 is '<= 1.5x behind')"
    )
    if iqr > 5.0:
        print("# WARNING IQR > 5% — clocks are not stable; this row must not enter the ledger.")
    row |= {"metric": point.metric, "ms": ms, "iqr_pct": iqr, "tflops": tflops, point.metric: pct}
    print(
        f"#\n# record it:  make measure L=S1 R=S-R4 M={point.metric} V={pct:.1f} "
        f'DEV="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" '
        f"FLOOR={_floor_metric(point)} FLOORV={floor_ms:.4f}"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(row, indent=1))
    print(f"{pct:.4f}")
    return 0


def _run_ep(point: Point, model: ModelShape, args) -> int:  # noqa: ANN001
    """The EP=2 point: comm fraction of one MoE layer, under a real NCCL all-to-all.

    Structure, and it is the same on both ranks: route this rank's own tokens → permute into
    expert-major order (which is also rank-major, because placement is linear) → all-to-all the
    counts → all-to-all the rows → regroup the arrivals into local expert groups → the two expert
    GEMMs → all-to-all back → un-permute locally. The two all-to-alls are timed separately from the
    rest, and the fraction is their median over the layer's median.
    """
    import torch
    import torch.distributed as dist

    from _harness import provenance_line, wallclock_ms  # type: ignore[import-not-found]
    from scratch_llm.kernels.moe.ep import (
        DispatchPlan,
        ExpertParallel,
        all_to_all_rows,
        comm_fraction,
        counts_by_expert,
        plan_dispatch,
        plan_receive,
        traffic,
    )
    from scratch_llm.kernels.moe.fused import resolve_gemm, silu_and_mul
    from scratch_llm.kernels.moe.routing import combine, gather_tokens, route_topk

    if "RANK" not in os.environ:
        print("# REFUSED — the EP point needs a process group. On the box:")
        print(
            f"#   torchrun --nproc_per_node={point.ep_size} bench/kernels/moe/s1_moe.py --point {point.name}"
        )
        return 3
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != point.ep_size:
        if rank == 0:
            print(
                f"# REFUSED — {point.name} is EP={point.ep_size}, launched with world size {world}"
            )
        dist.destroy_process_group()
        return 3
    torch.cuda.set_device(rank)

    ep = ExpertParallel(n_experts=model.n_experts, ep_size=world)
    start, stop = ep.local_range(rank)
    local_tokens = point.n_tokens // world
    torch.manual_seed(args.seed + rank)
    x = torch.randn(local_tokens, model.hidden, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn(local_tokens, model.n_experts, device="cuda", dtype=torch.float32)
    n_local = stop - start
    w1 = (
        torch.randn(n_local, 2 * model.intermediate, model.hidden, device="cuda")
        * model.hidden**-0.5
    ).bfloat16()
    w2 = (
        torch.randn(n_local, model.hidden, model.intermediate, device="cuda")
        * model.intermediate**-0.5
    ).bfloat16()

    routing = route_topk(logits, model.top_k)
    plan = plan_dispatch(routing.topk_ids, ep, rank)
    # The metadata exchange: a receiver cannot size a buffer or build a group offset without it.
    # Done ONCE here, outside both timed callables, and that is a deliberate understatement — a real
    # serving step pays this exchange on every MoE layer, so the fraction below is a LOWER bound on
    # a real EP layer's. Say that when quoting it.
    mine = counts_by_expert(routing.topk_ids, ep).cuda()
    all_counts = torch.zeros(world * model.n_experts, dtype=mine.dtype, device="cuda")
    dist.all_gather_into_tensor(all_counts, mine)
    recv_counts = all_counts.view(world, model.n_experts)[:, start:stop].cpu()
    plan = DispatchPlan(rank, ep, plan.permutation, plan.send_counts, recv_counts.sum(dim=1))
    # Hoisted out of the timed callables: the regroup indices' host->device copy and the split
    # sizes' `.tolist()` are once-per-shape setup, not part of a layer, and timing them would put a
    # host sync inside the window that the number is a statement about.
    rplan = plan_receive(recv_counts).to("cuda")
    gemm = resolve_gemm("cuda")
    local_offsets = rplan.group_offsets
    send_splits, recv_splits = plan.send_counts.tolist(), plan.recv_counts.tolist()
    # The dispatch payload, built once: gathering it is the permutation's cost, not the wire's. It
    # belongs inside `layer` (where it is rebuilt every call) and not inside `comm_only`, which has
    # to time the two collectives and nothing else.
    sent_payload = gather_tokens(x, plan.permutation)

    def comm_only() -> None:
        back = all_to_all_rows(sent_payload, plan)
        home = torch.empty_like(sent_payload)
        dist.all_to_all_single(
            home, back, output_split_sizes=send_splits, input_split_sizes=recv_splits
        )

    def layer() -> None:
        sent = gather_tokens(x, plan.permutation)
        arrived = all_to_all_rows(sent, plan)
        grouped = rplan.apply(arrived)
        h = silu_and_mul(gemm(grouped, w1, local_offsets).to(x.dtype))
        y_local = gemm(h, w2, local_offsets).to(x.dtype)
        home = torch.empty_like(sent)
        dist.all_to_all_single(
            home, rplan.undo(y_local), output_split_sizes=send_splits, input_split_sizes=recv_splits
        )
        combine(home, plan.permutation, routing.topk_weights)

    comm_ms = wallclock_ms(comm_only, warmup=args.warmup, iters=args.iters)[0]
    layer_ms = wallclock_ms(layer, warmup=args.warmup, iters=args.iters)[0]
    frac = comm_fraction(comm_ms, layer_ms)
    tr = traffic([plan], hidden=model.hidden)

    if rank == 0:
        print(provenance_line(f"S1 S-R4 · {point.name} · bf16 · EP={world}"))
        print("# timing      host clock, device synced inside the window (a layer is not a kernel)")
        print(f"# comm        {comm_ms:8.3f} ms  · layer {layer_ms:8.3f} ms")
        print(f"# on the wire {tr.dispatch_bytes_on_wire / 2**20:7.2f} MiB dispatch, the same back")
        print(f"# {point.metric:<24} {frac:7.4f}   <- the ledger row")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps(
                    {
                        "ladder": "S1",
                        "rung": "S-R4",
                        "point": point.name,
                        "metric": point.metric,
                        point.metric: frac,
                        "comm_ms": comm_ms,
                        "layer_ms": layer_ms,
                        "traffic": tr.as_row(),
                    },
                    indent=1,
                )
            )
        print(
            f"#\n# record it:  make measure L=S1 R=S-R4 M={point.metric} V={frac:.4f} "
            f'DEV="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"'
        )
        print(f"{frac:.6f}")
    dist.destroy_process_group()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Recorded measurements go through infra/bench.sh, which enforces the prediction gate.",
    )
    ap.add_argument("--point", choices=sorted(POINTS), default="prefill4k")
    ap.add_argument("--floor", choices=sorted(FLOORS), default=None, help="measure the floor alone")
    ap.add_argument("--warmup", type=int, default=20, help="workspace invariant 4: >= 20")
    ap.add_argument("--iters", type=int, default=50, help="workspace invariant 4: >= 50")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None, help="also write the row as JSON here")
    ap.add_argument(
        "--dry-run", action="store_true", help="print the plan and exit; measure nothing"
    )
    ap.add_argument("--list", action="store_true", help="print the point registry")
    args = ap.parse_args(argv)

    if args.list:
        return _list()
    point, model = POINTS[args.point], QWEN3_30B_A3B
    if args.floor and point.is_distributed:
        ap.error(
            "the EP point has no separate floor: its number is a fraction of its own layer time"
        )
    if args.dry_run:
        return _dry_run(point, model, args)

    import torch

    if not torch.cuda.is_available():
        print(
            "# no CUDA device — a fused-MoE number is a statement about silicon, and there is none here."
        )
        print(f"# on the box:  bash experiments/S1/S-R4/run.sh   (POINT={args.point})")
        return 0
    return _run_ep(point, model, args) if point.is_distributed else _run_single(point, model, args)


if __name__ == "__main__":
    raise SystemExit(main())
