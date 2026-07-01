# A1 Rung 3 — Continuous batching · engineering spec (pre-registration)

> **Spec-with-falsifiers (FOP-2).** Written BEFORE building. Parent: `PERF_ENGINEERING_SPEC.md` §A1
> Rung 3 (oracle = Rung-1 per-request greedy; gate = each request token-exact + ≥2× aggregate
> throughput vs static batch). Predecessors: R1 (decode is memory-bound, overhead-limited eager),
> R2 (GQA is the capacity lever that makes a big batch fit).

## 1. Mechanism (first principles) — batching is how you beat the memory wall

At B=1, decode reads **all** weights (2·P bytes) to make **one** token → AI≈1, memory-bound (R1). The
fix that beats the wall is not a faster kernel — it is **batching**: read each weight once, apply it to
**B** sequences in the same forward → **B tokens per weight-read** → AI≈B. The weight traffic amortizes.

Roofline for a batched decode step (short ctx, weights dominate):
```
bytes/step = 2P  +  B·KV_seq         FLOPs/step = 2P·B
AI = 2PB / (2P + B·KV_seq) ≈ B       (memory-bound while B < ridge≈130 on sm120)
aggregate tok/s = B / step_time = B·BW / (2P + B·KV_seq) ≈ B·(BW/2P) = B × single-stream ceiling
```
So aggregate throughput rises ~**linearly in B** until either **B·KV_seq** overtakes **2P** (KV-bound,
where R2's GQA/MQA matters) or **B≈130** crosses the compute ridge. Batching also amortizes the *launch
overhead* R1 found (one forward's ~955 launches now serve B tokens) — so per-stream tok/s should *rise*
with B, not just aggregate.

**Static vs continuous (Orca).** *Static* (request-level) batching runs a fixed batch until the
**longest** sequence finishes; short requests finish early and their slots idle → the effective batch
decays, weights get re-read for a shrinking batch, and padding tokens waste compute. *Continuous*
(iteration-level) batching reschedules **every decode step**: evict finished requests (return them
immediately), admit queued ones into freed slots → the batch stays **saturated**. The ≥2× win is from
killing head-of-line blocking + padding waste.

## 2. Worked example (GQA-4 config, 0.84B, ctx=512, bf16)

- weights 1.68 GB; B=32 KV = `32·512·32 KB` = 536 MB → bytes/step ≈ 2.2 GB, AI = `2P·32 / 2.2e9` ≈ **24**
  (still memory-bound). aggregate ceiling = `32 / (2.2e9/0.55e12)` = 32/4.0 ms ≈ **8,000 tok/s** vs the
  ~327 single-stream ceiling → **~24×**.
- Mixed trace (16 requests len-128, 16 len-512), static B=32: runs 512 steps, the 16 short requests idle
  for 384 steps → ~**37% slot utilization**. Continuous refills the freed slots → **~2.7× throughput**.

## 3. Design (where the linchpin lands)

- `serving/batched_cache.py` — **`BatchedKVCache`**: a **static** `[B, H_kv, max_ctx, head_dim]` buffer
  per layer + per-slot `length[b]` + per-slot `active`. Write new K,V at `length[b]`, bump. Slot
  alloc/free. **This contiguous static buffer IS the R4.1/R4.4 prerequisite** — cudagraph-capturable,
  and the thing PagedAttention (R4.1) later replaces with 16-token blocks + a block table.
- **Batched-decode attention**: one forward over B rows (query len 1), each row attends keys
  `[0, length[b])`. Needs a **per-row key-padding mask** `(B, 1, max_len)`. Recommended (single source of
  attention truth): extend the model forward to accept the batched cache + per-row lengths; the existing
  single-request path is unchanged when lengths are uniform. The alternative (reimplement attention in
  the serving layer) duplicates RoPE/GQA/SDPA and is rejected.
- `serving/continuous.py` — the **scheduler**: a request queue; loop = *admit* (prefill new requests into
  free slots) → *decode one step* over active slots → *sample per row* → *append* to caches → *evict*
  EOS / max_tokens → repeat until queue and batch are empty. Reuses `serving/metrics.py` for TTFT/ITL.

## 4. Phasing (tractable, each independently falsifiable)

- **R3a — static batched decode, uniform lengths.** `BatchedKVCache` + batched decode step. **Oracle:**
  row `b` == single-stream `decode(prompt_b)` token-exact. **Measure:** aggregate tok/s vs
  B∈{1,2,4,8,16,32} (compiled) → AI≈B, ~linear scaling = the weight-amortization roofline. *This is R3's
  headline number.*
- **R3b — continuous scheduler, variable lengths + join/leave.** The Orca loop on a mixed-length trace.
  **Oracle:** each request's output == its standalone greedy decode. **Measure:** ≥2× aggregate
  throughput vs static batching (+ slot-utilization to explain the gap).

## 5. Falsifiable predictions (pre-registered; sm120, GQA-4, compiled)

| # | experiment | predicted | bound |
|---|---|---|---|
| R3.1 | decode AI at batch B (short ctx) | **≈ B** (memory-bound while B<130) | memory |
| R3.2 | aggregate tok/s, B=32 vs B=1 | **≥ 10×** (weights amortized; per-stream tok/s also rises as overhead amortizes); ~linear in the memory regime | memory |
| R3.3 | batched row `b` output vs single-stream greedy | **token-exact** | — |
| R3.4 | continuous vs static, mixed 128/512 trace, B=32 | **≥ 2×** aggregate throughput | — |
| R3.5 | per-token latency (ITL) at B=32 vs B=1 | **higher** (the throughput↔latency tradeoff — the honest cost of batching) | — |

**R3.2 is the load-bearing falsifier:** if aggregate tok/s stays flat as B grows, the batched forward is
**not** amortizing weights (bug: B separate forwards, or weights re-read per row) — stop and fix.

## 6. Oracle · DoD · kill

**Oracle (D1).** `BatchedKVCache` batched decode == B independent per-request `KVCache` decodes,
token-exact, fixed seed. The per-row length mask is the load-bearing correctness surface: a short row
must **not** attend another row's padding.

**DoD (all `[FACT]`-logged):**
- [ ] Batched row token-exact vs single-stream (R3.3).
- [ ] Aggregate tok/s vs B measured; AI≈B shown three ways (hand roofline, %HBM, tok/s) (R3.1/R3.2).
- [ ] Continuous ≥2× static on a mixed trace, with slot-utilization explaining it (R3.4).
- [ ] TTFT/ITL/throughput/goodput reported via `serving/metrics.py` (R3.5).

**Kill criteria:**
- Batched row ≠ single-stream → per-row length-mask / positions bug; fix before any throughput number.
- Aggregate flat in B → not amortizing weights (measure bytes/step: it must be ~2P+B·KV, not B·2P).
- Continuous <2× static → the scheduler isn't refilling freed slots; measure utilization before blaming the trace.

## 7. Test-first plan (executable spec)

1. `tests/test_batched_cache.py` (CPU, green): `BatchedKVCache` slot alloc/free/write; **batched decode
   row `b` == single-stream `decode(prompt_b)` token-exact** (small model, uniform then ragged lengths);
   per-row length-mask correctness (a shorter row's output is invariant to another row's padding).
2. `bench/batched_decode.py` (gpu): aggregate tok/s vs B sweep (compiled) + AI + %HBM → the
   weight-amortization roofline. Log rows.
3. (R3b) `tests/test_continuous.py`: mixed-length trace, each request == standalone greedy;
   `bench/continuous.py`: continuous vs static ≥2× + slot utilization.

## 8. Linchpin realized (the payoff for R4.x)

The `BatchedKVCache` static buffer built here is the exact primitive R4.1 (PagedAttention: replace the
contiguous per-slot region with 16-token blocks + a block table) and R4.4 (wrap the batched step in a
CUDA graph — R1's cudagraph failure was the cat-cache; a static buffer is capturable) need. **One
refactor, three rungs** — R3 pays the design cost that R4.1/R4.4 then cash in.
