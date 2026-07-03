# A1 Rung 4.1 — PagedAttention · engineering spec (pre-registration)

> **Spec-with-falsifiers (FOP-2).** Written BEFORE building (2026-07-03). Parent:
> `performance/A1_transformer_inference.md` §4.1 (oracle = contiguous-KV attention, bit-for-bit;
> gate = <4% waste vs a slab allocator + adversarial block cases + CoW). Predecessor: R3b shipped
> the dense `BatchedKVCache` slot buffer and **measured its two costs** — (a) capacity: every slot
> reserves `max_ctx` up front; (b) traffic/compute: the mixed-age batch keeps `L_view` high, a
> ~1.27× per-step tax (9.6 vs 6.3 ms) vs the uniform-march control. R4.1 attacks (a) with paged
> storage and (b) with a paged decode-attention kernel.

## 1. Mechanism (first principles) — KV memory as virtual memory

A slot buffer is `malloc(max_ctx)` per request: contiguous, simple, and **reserved-capacity
waste** — a request at length ℓ holds `max_ctx − ℓ` tokens of dead HBM. Paged KV is the OS page
table transplanted: the pool is `N` fixed **blocks** of 16 tokens; each slot maps logical block
`i` → physical block via a **block table**; allocation happens on demand (a new block only when ℓ
crosses a 16 boundary). Waste collapses from *reservation* (∝ `max_ctx − ℓ`) to *internal
fragmentation* (≤ 15 tokens, E[·] ≈ 8, in the last block). Sharing becomes possible for free:
two requests with a common full-block prefix point at the same physical blocks (refcount), and a
write into a shared block first copies it (**CoW**).

```
capacity math (per row, time-averaged over a decode of final length L, prompt P):
  slab reserved   = max_ctx                    = 2048 tok               (RUNG1_CONFIG)
  paged allocated = ⌈ℓ/16⌉·16 ≈ ℓ + 8          (ℓ grows P → L)
  heavy-tail trace: E[ℓ] ≈ 32 + E[out]/2 ≈ 96  → slab utilization ≈ 4.7%  (≈95% waste)
                                               → paged frag ≈ 8/104 ≈ 7.7%
  capacity ratio (rows per GB) ≈ 2048/104 ≈ 20× on-demand; ≥10× even reserving final-length
```

**Why the R3b tax needs a *kernel*, not just paged storage:** the dense decode reads/computes over
the padded `[B, :L_view]` view. The bytes model explains only ~0.5 ms of the measured 3.3 ms tax
(KV delta between mixed-age L_view≈544 and uniform-march mean ≈288 is ~260 MB ≈ 0.48 ms at 0.55
TB/s); the remainder is **padded-SDPA compute/materialization** (fp32 scores `B×H×L_view`,
`repeat_interleave` GQA copies, softmax over padding). A gather-then-SDPA paged path *keeps* all
of that and *adds* a gather copy — paged storage alone is a capacity win and a **speed loss**. The
fused paged decode kernel (one query row/token, online softmax over that row's blocks, GQA groups
in-kernel, no materialized scores/repeats) removes both the padding traffic and the padded compute.

## 2. Design

- **`SlotKVCache` base** (extract from R3b's `BatchedKVCache`): slots, `lengths`/`active` device
  tensors, python mirrors, `view_len`, `advance`/`mirror_*`, `free_slot` — the ownership contract
  (device state graph-owned; python mirror + **all allocation decisions** scheduler-owned) is
  unchanged and is what keeps compile guards stable (the R3b lesson, measured).
- **`BatchedKVCache(SlotKVCache)`** — dense storage, exactly today's behavior (R3b tests must stay
  green unchanged).
- **`PagedKVCache(SlotKVCache)`** — per layer, a pool `(n_blocks, H_kv, 16, d)`; block table
  `(n_slots, max_blocks)` int32 device tensor; free-list allocator + per-block **refcounts**
  (python, scheduler-owned). **Block 0 is the reserved trash block**: freed/inactive rows' table
  slot 0 points at it, so the static-shape decode write (every row writes, R3b design) lands
  harmlessly and the write-then-mask NaN guarantee carries over verbatim.
- **Allocation points (all outside the graph):** `reserve_prefill(slots, lens)` inside the eager
  admission path; `pre_decode_reserve()` called by the engine each step — allocates a block for
  active rows at `ℓ % 16 == 0`, and performs **CoW there** (a row about to write into a
  refcount>1 block gets a private copy first — allocation decisions are scheduler turf).
- **Attention paths:** (a) *gather* — reconstruct the dense `[B, H_kv, :L_view, d]` view from the
  table (one advanced-index per layer), then the existing SDPA branch: this is the **bit-exact
  oracle path** (same math, different storage); (b) *kernel* — `kernels/paged_decode_triton.py`
  reads the block table directly. CoW at cache level only; scheduler prefix *detection* is out of
  scope (noted, not claimed).
- **Engine:** `serve(..., cache_kind="dense"|"paged")` — one engine, the measured Δ isolates
  storage, same method as R3b's policy comparison.

## 3. Falsifiable predictions (pre-registered; sm120, GQA-4 0.84B, R3b bench config)

| # | experiment | predicted | note |
|---|---|---|---|
| P4.1.1 | fragmentation (paged) vs reservation waste (slab) | frag **4–8%** heavy-tail (short-ℓ trace), **≤4%** on 16/16 (E[ℓ]≈352 → ~2%); slab reserved-waste **≈95%**; capacity **10–20×** rows/GB | the headline; <4% at production lengths is the vLLM number — our short trace honestly reads higher |
| P4.1.2 | paged-gather vs contiguous, ragged churn + deliberately scattered + poisoned free blocks | **bit-exact** (`torch.equal`) | same SDPA math, different storage; block-boundary (ℓ≡0 mod 16) adversarial cases included |
| P4.1.3 | paged-gather step time vs dense (negative control) | **+10–25%/step** (gather adds ~0.9–1.1 GB/step r+w across 16 layers ≈ +1.6–2 ms) → wall ratio 2.30× → **~1.9–2.1×** | paged storage alone LOSES throughput; capacity is its win — register it so nobody ships the gather path as "PagedAttention" |
| P4.1.4 | Triton paged decode kernel vs dense SDPA path | per-step **9.6 → ≤8.3 ms** (≥50% of the 1.27× tax reclaimed) → heavy-tail wall ratio **≥2.6×**; correctness: allclose (bf16 ≤1e-2 vs fp32 oracle) + **greedy token-exact e2e** | the tax decomposition (bytes ~0.5 ms vs padded-compute ~2.8 ms) is itself measured by which part the kernel recovers |

**Kill criteria:**
- Gather path ≠ bit-exact ⇒ table/indexing bug — fix before any capacity or speed number.
- Fragmentation > 8% on heavy-tail ⇒ allocator bug (leak or over-allocation), not a trace property.
- Kernel slower than gather+SDPA at B=32, L≤544 ⇒ ship R4.1a (capacity win) alone; kernel reclaim
  moves to the A4 tie-in with the failure profiled (small-L launch/occupancy suspected first).
- Refcount accounting drifts (pool leak after churn) ⇒ CoW bug; the churn test must end with
  `free_blocks == n_blocks − 1` (trash) after all slots freed.

## 4. Test-first plan (executable spec)

1. `tests/test_paged_cache.py` (CPU): allocator alloc/free/reuse + OOM raise + trash-block never
   allocated; **scattered-block oracle** (force a shuffled free-list → paged ragged churn decode ==
   `BatchedKVCache` decode token-exact, and == single-stream greedy); block-boundary crossing
   (ℓ = 15, 16, 17, 32); poison free blocks → active rows bit-identical; waste accounting == hand
   analytic on a constructed scenario; CoW: shared full-block prefix → one physical copy, diverge →
   copy-on-write fires once, outputs still oracle-exact; churn ends with zero leaked blocks.
2. `serve(cache_kind="paged")` through `tests/test_continuous.py`-style oracle (each request ==
   standalone greedy) — one parametrized run.
3. `tests/test_paged_kernel.py` (gpu-marked): kernel vs gather-path allclose + greedy-exact.
4. `bench/continuous.py --cache paged [--paged-kernel]`: waste report + step time + wall ratio.
