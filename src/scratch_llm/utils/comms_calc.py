"""Communication algebra — closed-form cost models for distributed-training collectives.

A2 systems §8. Intent: answer "when does adding devices stop helping?" analytically, before
renting a single GPU. The model: N devices, each with egress bandwidth ``link_bw`` bytes/s and
``gpu_flops`` FLOP/s; a training step is compute-bound iff its per-device wire time fits under its
per-device compute time (the two overlap). Every function is a pure closed form whose derivation
lives in its docstring; `docs/design/A2_COMMS_ALGEBRA.md` is generated from these numbers and
`tests/test_comms_calc.py` locks them.

Invariant: ring all-reduce == reduce-scatter + all-gather, each moving (W−1)/W·S bytes per device
— so all-reduce bytes → 2S as W → ∞, independent of world size. That single identity yields the
DP bound N ≤ B·W/C, the TP bound N ≤ (3/2)·D_ff·W/C, and their 2D product.

Interview question: "You have B tokens/step, a D_ff-wide model, C FLOP/s per chip and W bytes/s
egress — how many chips can you scale to before the network is the bottleneck, and which knob do
you turn when you get there?" (Answer: DP dies at B·W/C, TP at (3/2)·D_ff·W/C, together
(3/2)·B·D_ff·(W/C)²; past that only faster interconnect or a bigger critical batch helps.)
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def _check_world(world: int) -> None:
    if world < 1:
        raise ValueError(f"world must be >= 1, got {world}")


def _check_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def all_gather_bytes(size_bytes: float, world: int) -> float:
    """Per-device egress bytes for a ring all-gather of a tensor of `size_bytes` total.

    Derivation: each device starts with a chunk of S/W bytes; in each of the W−1 ring steps it
    forwards one chunk of S/W to its right neighbour, so it sends (W−1)·S/W = (W−1)/W·S bytes.
    """
    _check_world(world)
    return (world - 1) / world * size_bytes


def reduce_scatter_bytes(size_bytes: float, world: int) -> float:
    """Per-device egress bytes for a ring reduce-scatter of a tensor of `size_bytes` total.

    Derivation: identical traffic pattern to the ring all-gather — W−1 steps, one S/W-byte chunk
    (a running partial sum) sent per step — the adds are free on the wire, so (W−1)/W·S bytes.
    """
    _check_world(world)
    return (world - 1) / world * size_bytes


def ring_allreduce_bytes(size_bytes: float, world: int) -> float:
    """Per-device egress bytes for a ring all-reduce: 2·(W−1)/W·S.

    Derivation: all-reduce = reduce-scatter (each device ends with its 1/W of the sum) followed by
    all-gather (everyone collects all the summed chunks); each phase moves (W−1)/W·S per device.
    Limit W→∞: 2S — wire volume per device is bounded no matter the world size, which is *why*
    data parallelism scales at all.
    """
    return reduce_scatter_bytes(size_bytes, world) + all_gather_bytes(size_bytes, world)


def naive_allreduce_bytes(size_bytes: float, world: int) -> float:
    """Per-device egress bytes if every device sends its full tensor to every other: (W−1)·S.

    Derivation: W−1 peers × S bytes each. Ring/naive ratio = 2/W — the ring wins by chunking so
    each byte of the reduction crosses each link once instead of the whole tensor crossing W−1.
    """
    _check_world(world)
    return (world - 1) * size_bytes


def alternate_ring_allreduce_bytes(size_bytes: float, world: int) -> float:
    """Per-device egress bytes for the A2 §8.1 alternate ring all-reduce: (W−1)·S.

    Derivation: the alternate algorithm passes *whole tensors* (size S) around the ring for W−1
    steps, each device accumulating as it forwards — no chunking, so (W−1)·S per device, a factor
    W/2 worse than the standard ring's 2·(W−1)/W·S.
    """
    _check_world(world)
    return (world - 1) * size_bytes


def ring_allreduce_time(
    size_bytes: float,
    world: int,
    link_bw: float,
    latency: float = 0.0,
    hops: int = 1,
) -> float:
    """Seconds for a ring all-reduce under the α–β model: 2(W−1)·hops·α + 2(W−1)/W·S/β.

    Derivation: 2(W−1) ring steps (reduce-scatter + all-gather), each paying the per-message
    latency `latency` across `hops` physical links (the α term), plus the bandwidth term = total
    per-device bytes / egress bandwidth. The latency floor is why DDP flattens many small grads
    into one bucket (see utils/ddp.py "flat"): 2(W−1)·α per *tensor* dwarfs S/β when S is tiny.
    """
    _check_world(world)
    _check_positive("link_bw", link_bw)
    return 2 * (world - 1) * hops * latency + ring_allreduce_bytes(size_bytes, world) / link_bw


def alternate_ring_allreduce_time(
    size_bytes: float,
    world: int,
    link_bw: float,
    latency: float = 0.0,
    hops: int = 1,
) -> float:
    """Seconds for the alternate ring all-reduce: (W−1)·(hops·α + S/β).

    Derivation: W−1 steps, each sending a full S-byte tensor — the official
    `alternate_ring_all_reduce` answer, (N−1)·S/W seconds at zero latency.
    """
    _check_world(world)
    _check_positive("link_bw", link_bw)
    return (world - 1) * (hops * latency + size_bytes / link_bw)


def ddp_step_bytes(n_params: int, dtype: torch.dtype, world: int) -> float:
    """Per-device wire bytes per DDP step: one ring all-reduce of the gradients.

    Derivation: gradients are n_params·itemsize bytes; DDP all-reduces them once per step, so
    2·(W−1)/W·P·b per device — ~2 gradient-copies on the wire regardless of world size.
    """
    return ring_allreduce_bytes(n_params * dtype.itemsize, world)


def zero1_step_bytes(n_params: int, dtype: torch.dtype, world: int) -> float:
    """Per-device wire bytes per ZeRO-1 step: reduce-scatter grads + all-gather updated params.

    Derivation: each rank only needs the summed grads for its own optimizer shard
    (reduce-scatter, (W−1)/W·P·b) and then republishes its updated param shard (all-gather,
    (W−1)/W·P·b) — exactly the two halves of a ring all-reduce. ZeRO-1's W× optimizer-state
    memory saving is therefore *comms-free*: same wire volume as plain DDP.
    """
    size = n_params * dtype.itemsize
    return reduce_scatter_bytes(size, world) + all_gather_bytes(size, world)


def fsdp_step_bytes(n_params: int, dtype: torch.dtype, world: int) -> float:
    """Per-device wire bytes per FSDP (ZeRO-3) step: 3·(W−1)/W·P·b.

    Derivation: params are sharded, so each step must all-gather the full weights for the forward
    ((W−1)/W·P·b), all-gather them again for the backward (they were freed after use), and
    reduce-scatter the gradients to their shard owners ((W−1)/W·P·b). Three one-way passes vs
    DDP's two — FSDP/DDP wire ratio is exactly 3/2 at any world size, the price of holding only
    1/W of the model.
    """
    size = n_params * dtype.itemsize
    return 2 * all_gather_bytes(size, world) + reduce_scatter_bytes(size, world)


def tp_step_bytes(
    batch: int,
    seq: int,
    d_model: int,
    n_layers: int,
    dtype: torch.dtype,
    world: int,
) -> float:
    """Per-device wire bytes per tensor-parallel step: 4 activation all-reduces per block.

    Derivation: with column-parallel in / row-parallel out (Megatron), each transformer block
    needs one all-reduce of the (batch, seq, d_model) activation after the attention output
    projection and one after the FFN down-projection — 2 in forward, mirrored 2 on the input
    gradients in backward. Total: 4·L ring all-reduces of batch·seq·d_model·b bytes. Unlike
    DP/FSDP this scales with *activations*, not weights, and sits on the critical path — which is
    why TP stays inside the NVLink island.
    """
    activation = batch * seq * d_model * dtype.itemsize
    return 4 * n_layers * ring_allreduce_bytes(activation, world)


def compute_step_time(flops: float, gpu_flops: float) -> float:
    """Seconds to execute `flops` on one device at `gpu_flops` FLOP/s (roofline compute leg)."""
    _check_positive("gpu_flops", gpu_flops)
    return flops / gpu_flops


def ffn_fwd_flops(tokens: int, d_model: int, d_ff: int) -> float:
    """Forward FLOPs of the A2 §8.2 gated FFN on `tokens` rows: 6·B·D·D_ff.

    Derivation: three matmuls — x·W1 and x·W2 at (B,D)(D,D_ff) and z·W3 at (B,D_ff)(D_ff,D) —
    each 2·B·D·D_ff FLOPs; elementwise ops ignored.
    """
    return 6.0 * tokens * d_model * d_ff


def ffn_bwd_flops(tokens: int, d_model: int, d_ff: int) -> float:
    """Backward FLOPs of the gated FFN: 12·B·D·D_ff (2× forward).

    Derivation: six matmuls of 2·B·D·D_ff each — dz=dy·W3ᵀ, the two dx contributions
    dx1·W1ᵀ + dx2·W2ᵀ, and the three weight grads zᵀdy, xᵀdx2, xᵀdx1.
    """
    return 12.0 * tokens * d_model * d_ff


def ffn_weight_bytes(d_model: int, d_ff: int, dtype: torch.dtype) -> float:
    """Bytes of the gated FFN's weights (= its grads): 3·D·D_ff params × itemsize."""
    return 3 * d_model * d_ff * dtype.itemsize


def transformer_nonembed_params(n_layers: int, d_model: int, d_ff: int) -> int:
    """Non-embedding params of the A1 architecture: L·(4·D² attention + 3·D·D_ff SwiGLU FFN)."""
    return n_layers * (4 * d_model**2 + 3 * d_model * d_ff)


def dp_max_world(tokens: int, link_bw: float, gpu_flops: float) -> float:
    """Largest N_DP before the DP backward is comms-bound: N ≤ B·W/C.

    Derivation (large-W limit (N−1)/N → 1): comm = all-reduce of the 2·3·D·D_ff-byte fp16 grads
    ≈ 12·D·D_ff/W; compute = 12·B·D·D_ff/(N·C). comm ≤ compute ⇔ N ≤ B·W/C. D and D_ff cancel:
    the DP ceiling is set purely by tokens-per-step × interconnect-to-compute ratio.
    """
    _check_positive("link_bw", link_bw)
    _check_positive("gpu_flops", gpu_flops)
    return tokens * link_bw / gpu_flops


def fsdp_max_world(tokens: int, link_bw: float, gpu_flops: float) -> float:
    """Largest N_FSDP before either pass is comms-bound: N ≤ B·W/C — identical to DP.

    Derivation: backward comm = all-gather + reduce-scatter ≈ 12·D·D_ff/W vs compute
    12·B·D·D_ff/(N·C) ⇒ N ≤ B·W/C; forward comm = all-gather ≈ 6·D·D_ff/W vs 6·B·D·D_ff/(N·C) ⇒
    the same bound. FSDP buys W× memory for 1.5× wire bytes but *the scaling ceiling is unmoved*.
    """
    return dp_max_world(tokens, link_bw, gpu_flops)


def tp_max_world_fwd(d_ff: int, link_bw: float, gpu_flops: float) -> float:
    """Largest N_TP before the TP forward is comms-bound: N ≤ (3/2)·D_ff·W/C.

    Derivation: forward comm = one all-reduce of the fp16 (B,D) output ≈ 4·B·D/W; compute =
    6·B·D·D_ff/(N·C). comm ≤ compute ⇔ N ≤ (3/2)·D_ff·W/C. B and D cancel — the TP ceiling is
    set by model *width*, not batch, which is why TP ~ 8 inside a node no matter the job size.
    """
    _check_positive("link_bw", link_bw)
    _check_positive("gpu_flops", gpu_flops)
    return 1.5 * d_ff * link_bw / gpu_flops


def tp_max_world_bwd(d_ff: int, link_bw: float, gpu_flops: float) -> float:
    """Largest N_TP before the TP backward is comms-bound: N ≤ 3·D_ff·W/C.

    Derivation: backward comm = one all-reduce of the (B,D) input gradient ≈ 4·B·D/W; compute =
    12·B·D·D_ff/(N·C) — twice the forward FLOPs against the same wire bytes, so twice the bound.
    The forward's (3/2)·D_ff·W/C is therefore the binding constraint.
    """
    return 2.0 * tp_max_world_fwd(d_ff, link_bw, gpu_flops)


def fsdp_tp_max_world(
    tokens: int,
    d_ff: int,
    link_bw: float,
    gpu_flops: float,
    overlapped: bool = True,
) -> float:
    """Largest N = N_TP·N_FSDP before the 2D forward is comms-bound.

    Derivation (forward): FSDP-axis comm = all-gather of this TP rank's weight slice ≈
    6·D·D_ff/(N_TP·W); TP-axis comm = all-reduce of the batch-sharded output ≈ 4·B·D/(N_FSDP·W);
    compute = 6·B·D·D_ff/(N·C). Overlapped axes ⇒ comm = max of the two: each branch gives an
    independent bound (N_FSDP ≤ B·W/C and N_TP ≤ (3/2)·D_ff·W/C) and they multiply:
    N ≤ (3/2)·B·D_ff·(W/C)². Sequential axes ⇒ comm = sum; minimizing over the split
    (N_TP* = √(3·D_ff·N/(2B)), where the two terms equalize) gives N ≤ (3/8)·B·D_ff·(W/C)² —
    exactly 1/4 of the overlapped ceiling.
    """
    _check_positive("link_bw", link_bw)
    _check_positive("gpu_flops", gpu_flops)
    coeff = 1.5 if overlapped else 3.0 / 8.0
    return coeff * tokens * d_ff * (link_bw / gpu_flops) ** 2


def comms_bound_world_size(
    step_flops: float,
    per_device_step_bytes: Callable[[int], float],
    gpu_flops: float,
    link_bw: float,
    max_world: int = 1 << 24,
) -> int | None:
    """Smallest world size W ≥ 2 at which comm time exceeds compute time — "scaling stops here".

    Model: per-device compute = step_flops/W/gpu_flops (perfect FLOP split), per-device comm =
    per_device_step_bytes(W)/link_bw, fully overlapped, bound when comm > compute. Assumes wire
    bytes are nondecreasing in W (true for every ring collective here) so the comm−compute gap is
    monotone: exponential bracketing + bisection. Returns None if compute-bound through
    `max_world`.
    """
    _check_positive("gpu_flops", gpu_flops)
    _check_positive("link_bw", link_bw)

    def bound(world: int) -> bool:
        comm = per_device_step_bytes(world) / link_bw
        return comm > compute_step_time(step_flops / world, gpu_flops)

    if bound(2):
        return 2
    lo, hi = 2, 4
    while hi < max_world and not bound(hi):
        lo, hi = hi, hi * 2
    hi = min(hi, max_world)
    if not bound(hi):
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if bound(mid):
            hi = mid
        else:
            lo = mid
    return hi


def crossover_link_bandwidth(
    step_flops: float,
    per_device_step_bytes: float,
    gpu_flops: float,
    world: int,
) -> float:
    """Minimum egress bandwidth (bytes/s) keeping `world` devices compute-bound.

    Derivation: comm ≤ compute ⇔ bytes/bw ≤ step_flops/(W·C) ⇔ bw ≥ bytes·W·C/step_flops — the
    "what interconnect do I need to buy" inverse of `comms_bound_world_size`.
    """
    _check_world(world)
    _check_positive("gpu_flops", gpu_flops)
    _check_positive("step_flops", step_flops)
    return per_device_step_bytes * world * gpu_flops / step_flops
