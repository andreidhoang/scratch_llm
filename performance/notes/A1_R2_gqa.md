# A1 Rung 2 — GQA/MQA reduction · engineering spec (pre-registration)

> **Spec-with-falsifiers (FOP-2).** Written BEFORE building, per the rung workflow. Registers the
> mechanism, the falsifiable predictions + bound, the oracle/DoD/kill criteria, and the test-first
> plan. Parent spec: `performance/PERF_ENGINEERING_SPEC.md` §A1 (Rung 2 row + predictions). Ledger:
> `bench/RESULTS.md` (predict-before-run rows registered alongside this note).
>
> **Honest framing carried from R1 (`da64bfb`).** Decode at B=1 is **overhead-bound** on this box
> (eager 15% of HBM; torch.compile-fused 53%). So GQA's *decode-tok/s* payoff is **not** visible at
> B=1 short-context — it is a **KV-capacity** lever (fit bigger batch / longer context) and a
> **long-context traffic** lever (once KV read ≈ weight read). R2 measures those two facts honestly,
> and demonstrates the tok/s effect only where it is real: **long context, under `torch.compile`**
> (at the wall). This is why R2 sits *before* R3 (batching) — GQA is what makes a big batch fit.

## 1. Mechanism (first principles)

Each decode step reads **every weight once** (2·P bytes) **plus the entire KV cache once**
(2·L·H_kv·d_head·ctx·dtype bytes). In full MHA every query head carries its own K,V head. **GQA**
shares one K,V head across a group of `G = n_heads / n_kv_heads` query heads; **MQA** is the extreme
`n_kv_heads = 1`. The query projection is unchanged (still `n_heads`); only K,V shrink by `H/H_kv`.
The grouping is realized at attention time by `repeat_interleave(K, G, dim=heads)` — this repo already
does it (`model.py:284–287`), so R2 is a **measurement + capacity** rep on existing correct code, not
a new kernel (like R1; no Mode-3 body).

Two independent consequences, both `∝ H_kv`:
- **Capacity** — KV bytes *stored* per token = `2·L·H_kv·d_head·dtype`. Smaller H_kv ⇒ more tokens /
  bigger batch fit in 25 GB. This is the binding constraint on throughput (you can't batch what won't
  fit).
- **Traffic** — KV bytes *read* per decode step = capacity/token × ctx. Below the "crossover" context
  weight traffic dominates (GQA invisible); above it KV traffic dominates and GQA/MQA directly cut the
  step time.

## 2. Worked numeric example (Rung-1 config: d=2048, L=16, H=32, d_head=64, bf16)

KV **stored per token** = `2·L·H_kv·d_head·2 = 4096·H_kv` bytes:

| config | H_kv | KV/token | vs MHA | max ctx in ~23 GB (B=1) | crossover ctx (KV read = 1.68 GB weight read) |
|---|---|---|---|---|---|
| MHA | 32 | **128 KB** | 1× | ~175 K tok | **~12.8 K** |
| GQA-4 | 8 | **32 KB** | 4× less | ~702 K tok | **~51 K** |
| MQA | 1 | **4 KB** | 32× less | ~5.6 M tok | **~410 K** |

The B=1 capacity numbers are absurdly large *because weights dominate at B=1* — the honest reading is
the **batch** view: at ctx=2048, KV = `B·2048·(KV/token)`. A **B=256** serving batch needs **68 GB**
(MHA, OOM) vs **17 GB** (GQA-4, fits) vs **2 GB** (MQA). **GQA/MQA is what lets you batch — and
batching (R3) is what beats the memory wall.** That sentence is the whole point of R2→R3.

## 3. Falsifiable predictions (pre-registered; sm120, 0.55 TB/s)

| # | experiment | predicted | bound |
|---|---|---|---|
| 1 | KV bytes/token (measured `KVCache` tensor nbytes) vs `4096·H_kv` formula | **exact match**, H_kv∈{32,8,1} | — |
| 2 | KV/token ratio MHA:GQA-4:MQA | **32:8:1** (i.e. 4× and 32×) | — |
| 3 | crossover ctx (KV read = weight read) | MHA **~13 K**, GQA-4 **~51 K**, MQA **~410 K** | — |
| 4 | decode tok/s @ B=1, ctx=2 K, **compiled** — MQA vs MHA | **~equal** (both weight-bound; KV≪weights) | memory (weights) |
| 5 | decode tok/s @ B=1, ctx=16 K, **compiled** — MQA vs MHA | **MQA >20% faster** (MHA KV read ≈ 2.1 GB > 1.68 GB weights; MQA KV ≈ 67 MB) | memory (KV for MHA) |

Prediction 5 is the load-bearing falsifier: it is the *only* place GQA/MQA moves decode tok/s on this
box, and only because R1 gave us the compiled path that actually reaches the wall.

## 4. Oracle · DoD · kill

**Oracle (D1).** Grouped attention must equal the explicit-repeat reference: compute GQA attention two
ways — (a) the model's `repeat_interleave` path, (b) an explicit per-group loop mapping query head `h`
→ kv head `h // G` — assert equal to 1e-5 on a fixed seed. (Existing `test_model` already pins
cached==recompute for GQA; this adds the grouping-map assertion + the memory formula.)

**DoD (all `[FACT]`-logged):**
- [ ] KV bytes/token measured == analytic `4096·H_kv` for H_kv∈{32,8,1} (capacity + max-ctx + crossover table).
- [ ] Grouping oracle-correct (query-head→kv-head map), fixed seed.
- [ ] Decode tok/s (compiled) at short vs long ctx for MHA/GQA-4/MQA — prediction 5 confirmed or the gap explained.

**Kill criteria:**
- If grouped math ≠ explicit-repeat reference → the `repeat_interleave` group map is wrong; fix before any memory number is trusted.
- If MQA shows **no** tok/s edge over MHA at ctx=16 K **under compile** → either the run isn't at the wall (re-check compiled achieved-BW ≈ 53%+) or the KV-traffic model is wrong; diagnose before advancing.

## 5. Test-first plan (executable spec)

1. `tests/test_kv_memory.py` (CPU, green): build tiny MHA/GQA/MQA `KVCache`s, prefill N tokens, assert
   each layer's K,V tensor `nbytes` == `H_kv·d_head·N·dtype` (and the 32:8:1 ratio). Assert the
   grouping-map oracle. **This is R2's correctness rep — the memory math is what R2 is about.**
2. `bench/kv_memory.py` (gpu): reuse `RUNG1_CONFIG`; sweep `n_kv_heads ∈ {32,8,1}`; print the capacity
   table (KV/token, max-ctx, crossover); then, under `torch.compile` (reuse `decode_overhead_strip`'s
   fused path), measure decode tok/s + achieved BW at ctx ∈ {2 K, 16 K}. Log rows to `bench/RESULTS.md`.
3. Run `pytest -m "not gpu"` (test 1 green) then `pytest -m gpu` (bench correctness), then the bench;
   log measured vs the §3 predictions; diagnose any miss with `roofline-analyst`.

## 6. Near-sequence (engineered runway — the 4.x linchpin)

The plan sequence after R2 is **R3 → R4.1 → R4.2 → R4.3 → R4.4**. R1 surfaced the linchpin:

- **R3 continuous batching** — raise B so weight traffic amortizes across sequences: AI climbs from
  ~1 toward ~B, the decode step crosses from memory-bound toward compute-bound, aggregate tok/s ≥2×.
  *GQA (R2) is the enabler* — it's what makes a large B fit in 25 GB.
- **R4.1 PagedAttention** and **R4.4 CUDA-graph decode** — **both require the same refactor**: replace
  the `torch.cat`-grown `KVCache` (`model.py:220–228`) with a **static pre-allocated buffer**
  (`[B, H_kv, max_ctx, d_head]`, write at `length`, slice `[:length]`). R1 proved this: `reduce-overhead`
  cudagraph capture *failed* on the cat-cache. **One refactor unblocks batching (R3), paging (R4.1),
  and graphs (R4.4)** — so schedule the static-buffer `KVCache` rewrite as the shared prerequisite the
  moment R3 needs efficient batched storage. It is the highest-leverage single change in the A1 arc.

> **Sequencing note (honest).** R2 is correct-but-low-signal for *decode tok/s* at B=1 (predictions 1–4
> are footprint facts, only 5 moves tok/s). Its real value is (a) the capacity math that makes R3
> possible and (b) the grouping correctness rep. If EV-ranking over strict order, the static-buffer
> `KVCache` rewrite (R4.4/R4.1 closer) is the higher-signal node — but per the plan's numeric sequence,
> R2 → R3 first, then the static-buffer refactor lands as R3's batched-storage need + the 4.x prereq.
