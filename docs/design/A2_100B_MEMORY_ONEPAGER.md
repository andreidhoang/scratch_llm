# A2 one-pager — the 100B-parameter training-memory answer

> **Provenance rule (DoD):** every number below is produced by a tested function in
> [`src/scratch_llm/utils/memory_math.py`](../../src/scratch_llm/utils/memory_math.py) and pinned
> byte-exact by [`tests/test_memory_math.py`](../../tests/test_memory_math.py) — none typed by
> hand. Units are decimal (GB = 10⁹ B, TB = 10¹² B); the "80 GB" H100 budget is `80 * GB` bytes.
> Concrete 100B shape used for activations: L=80, d_model=10240, n_heads=80, seq 4096
> (12·L·d² ≈ 100.7B non-embedding params — `test_one_pager_config_is_really_100b`).

## The verbatim interview answer — "how would you train a 100B model?"

> "Start from the state math: with Adam, fp32 training costs **16 bytes per parameter** — 4 B
> weight + 4 B grad + 4 B momentum + 4 B variance — and bf16 mixed precision makes it *bigger*,
> **18–20 B/param**, because the fp32 master, fp32 grad, m and v all stay and you add 2 B bf16
> weight (and usually grad) compute copies. At 100B params that is **1.6–2.0 TB of state before a
> single activation** — 20–25× an 80 GB HBM, so one GPU is out by more than an order of magnitude,
> and plain DDP never helps because it replicates all of it on every rank. So I shard the state
> with the ZeRO ladder: ZeRO-1 shards optimizer state (the 16 B/param bucket → /W), ZeRO-2 also
> shards grads, ZeRO-3/FSDP also shards the weights — at W=64 that's 425 GB → 228 GB → **31 GB per
> rank**, and only ZeRO-3 gets under 80 GB; ~25 H100s is the floor for state alone. But ZeRO-3
> all-gathers every weight every fwd and bwd, so at scale I compose **3D parallelism**: tensor
> parallel within the NVLink node (8-way), pipeline parallel across nodes with micro-batches to
> shrink the bubble, and data parallel + ZeRO-1 outermost. Activations are the other TB-scale term:
> at seq 4k this model stores ~8.1 GB/layer with vanilla attention — **651 GB** for the stack —
> FlashAttention removes the quadratic term (→ 114 GB) and gradient checkpointing keeps only the
> ~84 MB residual per block boundary (~6.7 GB) and recomputes the rest, trading ~⅓ more FLOPs for
> the memory to raise micro-batch size."

Everything below is that answer with the receipts.

## 1. State accounting — bytes per parameter, and the 100B totals

`training_state_breakdown` / `optimizer_state_bytes` / `param_bytes` (`TestPerParamStories`):

| Term | fp32 story | bf16 mixed story | 100B bytes/term |
|---|---|---|---|
| Weights (fp32 / fp32 master) | 4 | 4 | 400 GB |
| Grads (fp32) | 4 | 4 | 400 GB |
| Adam m (fp32) | 4 | 4 | 400 GB |
| Adam v (fp32) | 4 | 4 | 400 GB |
| bf16 weight compute copy | — | 2 | 200 GB |
| bf16 grad buffer (optional) | — | 0–2 | 0–200 GB |
| **Total B/param** | **16** | **18–20** | |
| **Total @ 100B** | **1.6 TB** | **1.8–2.0 TB** | |

Mixed precision speeds up the matmuls and halves the *activation* dtype — it does **not** shrink
training state; it grows it. The fp32 grad is grouped with the optimizer terms because it exists
to feed the fp32 Adam step (under ZeRO it is materialized only for a rank's owned shard).

## 2. Why one GPU can't

State alone is `optimizer_state_bytes(100e9, mixed) / 80 GB` = **20×** (fp32) to **25×** (mixed,
20 B/param) an H100's HBM (`test_zero3_state_only_gpu_counts` inverts this: 20 and 25 GPUs are the
ZeRO-3 state-only floors). Plain DDP (stage 0) replicates the full 2.0 TB on *every* rank — adding
GPUs adds throughput, never capacity. And that is before activations (§4) or any CUDA/comms
buffers.

## 3. The ZeRO ladder — per-rank state at W=64, mixed precision

`zero_shard_bytes(stage, 100e9, 64, mixed_precision=True)` (`TestZeroLadder`):

| Stage | What shards | What stays replicated | Bytes/rank @ W=64 | Fits 80 GB? |
|---|---|---|---|---|
| 0 (DDP) | nothing | weights + grads + optimizer | 2000 GB | no |
| 1 | optimizer bucket (master + fp32 grad + m + v, 16 B/param) | bf16 weights + bf16 grads (400 GB) | 425 GB | no |
| 2 | + grads (reduce-scatter, not all-reduce) | bf16 weights (200 GB) | 228.125 GB | no |
| 3 (FSDP) | + weights (all-gather on demand) | nothing | **31.25 GB** | **yes** |

Two facts the tests pin: ZeRO-1 cuts the optimizer bucket **exactly 64×** while params+grads stay
replicated (`test_zero1_at_w64_...`), and ZeRO-3 divides the whole 2.0 TB exactly by W
(`test_zero3_divides_everything...`). Corollary (`test_zero1_and_zero2_cannot_fit...`): for 100B on
80 GB, ZeRO-1 and ZeRO-2 fail at **every** world size — their replicated buckets alone exceed HBM.
`gpus_needed` puts the ZeRO-3 state-only floor at **25 GPUs** (mixed) / **20** (fp32).

## 4. Activations at seq 4096 — the other TB, and the checkpointing trade

`activation_bytes_per_layer` implements the Korthikanti et al. 2022 residual-stream accounting
`s·b·h·34 + 5·a·s²·b` bytes/layer (bf16 activations, 4h MLP; assumptions in the docstring). Per
micro-batch of 1 at seq 4096 on the 100B shape (`TestActivations`):

| Quantity | Vanilla attention | FlashAttention |
|---|---|---|
| Per layer | 8,136,949,760 B ≈ 8.14 GB | 1,426,063,360 B ≈ 1.43 GB |
| × 80 layers | ≈ **651 GB** | ≈ **114 GB** |

Flash removes exactly the `5·a·s²·b` quadratic term — the two materialized (seq × seq) attention
matrices — by recomputing them tile-wise in backward; at seq 4k that term is ~82% of the layer's
activation memory, and it grows as s². Even with flash, 114 GB per unit micro-batch dwarfs the
31 GB/rank ZeRO-3 leaves free, which is where **gradient checkpointing** completes the story: keep
only each block's boundary residual (`residual_stream_bytes` = 2·s·b·d_model ≈ 84 MB/layer,
≈ 6.7 GB for the stack) and recompute the block interior during backward — O(√N)-style peak memory
for ~⅓ extra forward FLOPs. Checkpointing and flash compose: flash shrinks what a recomputed block
touches, checkpointing decides how many blocks are live at once, and the freed memory is spent on
larger micro-batches (which is also what keeps the pipeline bubble small in §5).

## 5. Beyond ZeRO — DP + TP + PP (why sharding alone isn't the end state)

ZeRO-3 fits, but pays for it in communication: every parameter is all-gathered once per forward
and once per backward, so weight bytes-on-wire scale with model size per *step*, not per replica.
The production composition is 3D: **TP** splits individual matmuls across the 8 GPUs inside an
NVLink node (activation-sized all-reduces on the fastest fabric, and it also divides per-GPU
activation memory); **PP** splits the layer stack across nodes (point-to-point boundary tensors —
the cheapest inter-node pattern — with micro-batching to amortize the bubble); **DP (+ ZeRO-1)**
replicates that TP×PP grid outermost for throughput, sharding optimizer state across replicas
almost for free since it adds no per-step weight traffic. The memory math above decides the grid:
TP×PP must bring per-GPU weights+activations under HBM; ZeRO-1 on the DP axis handles the
optimizer bucket; the comms algebra for *when* each axis saturates is the companion doc
(`A2_COMMS_ALGEBRA.md`, node W4).
