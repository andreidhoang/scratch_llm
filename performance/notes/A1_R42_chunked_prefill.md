# A1 Rung 4.2 — Chunked prefill · engineering spec (pre-registration)

> **Spec-with-falsifiers (FOP-2).** Written BEFORE building (2026-07-04, delegate mode ADR-0013).
> Parent: `performance/A1_transformer_inference.md` §4.2. Oracle = R3b/R4.1 greedy (token-exact).
> Gate = TTFT/ITL curve vs chunk size measurable; **no output divergence**; kills the measured
> ITL p99 ~29 ms admission spikes (R3b/R4.1 finding). Predecessor substrate: the R3b `serve()`
> engine + `BatchedKVCache` slot buffer; the KVCache attention branch already computes
> offset-prefill correctly (RoPE at absolute positions ⇒ chunked KV is bit-identical to one-shot).

## 1. Mechanism (first principles) — why a long prefill spikes ITL, and why chunking fixes it

The R3b/R4.1 engine admits a request with **one** prefill forward over its **entire** prompt,
inserted into the decode stream. A 512-token prefill is ~512× the compute of a 1-token decode step,
so the decode step that shares that iteration stalls for the whole prefill — every in-flight
request sees one enormous inter-token gap. Measured: **ITL p99 ~29 ms** admission spikes vs **~6 ms**
p50 (`bench/RESULTS.md` R3b/R4.1). This is head-of-line blocking of decode by prefill.

**Chunked prefill** (Sarathi-Serve / vLLM-V1) bounds the prefill work done between two consecutive
decode steps: split a prompt of length L into ⌈L/C⌉ chunks of ≤ C tokens; process **one chunk per
scheduler iteration**, interleaved with a decode step over the already-active slots. The per-gap
prefill work is capped at C tokens instead of L, so the p99 spike flattens toward p50.

Correctness is free from RoPE: token at absolute position t is rotated at t regardless of chunk
boundaries, and attention is causal over absolute positions. So chunk-then-decode writes KV
**bit-identical** to one-shot prefill ⇒ first token and all subsequent tokens are token-exact.
The engine routes a prefilling slot through a `ChunkPrefillView` that duck-types the single-request
`KVCache` interface (`length`=offset, `append` writes the chunk at [offset, offset+s), `get`
returns the slot's full [0, offset+s) K,V) — reusing the existing `past_len`/causal-mask branch of
`MultiHeadSelfAttention.forward` with **no new attention math**.

Scheduling model (this build): a prefilling request holds a **reserved but inactive** slot (masked
out of the decode batch by write-then-mask — the same inactive-row handling R3b already has); each
iteration advances the oldest prefilling slot by one chunk, then runs a decode step over active
slots; the last chunk emits the request's first token and flips the slot active (TTFT). Production
engines *piggyback* the chunk into the same batched forward as the decodes (stall-free batching);
sequential interleave is the token-exact equivalent on this dense-SDPA substrate and measures the
same TTFT/ITL curve — the piggyback fusion is a documented further step (design note).

## 2. Predictions (pre-registered — the curve IS the deliverable)

Standing GPU sm_120, `RUNG1_CONFIG` (~1B bf16), heavy-tail trace (prompts 32, outputs 64/256/512),
N_SLOTS=32, compiled decode. Baseline (chunk=∞, = R4.1 paged-kernel arm): ITL p50 ~6 ms, p99 ~29 ms.

| # | experiment | predicted | bound / rationale |
|---|---|---|---|
| P4.2.1 | token-exactness, every chunk size ∈ {∞,128,64,16,1} | **bit-identical** output to R4.1 (and to `generate`) | RoPE absolute-position ⇒ chunked KV bit-identical; kill line: any divergence ⇒ offset/mask bug, fix before measuring |
| P4.2.2 | ITL p99 vs chunk size, C: ∞→16 | **falls ≥2× (29 ms → ≤14 ms)**; monotone down as C↓ | per-gap prefill work capped at C tokens; the spike is prefill-in-the-gap, sliced |
| P4.2.3 | ITL p50 vs chunk size, C: ∞→16 | **rises modestly (~6 → 8–11 ms)** | every gap now carries a chunk forward; the honest cost (mirror of R3b's ITL cost of batching) |
| P4.2.4 | TTFT p50 vs chunk size, C: ∞→16 | **rises** (more iterations to first token; ⌈L/C⌉ chunk-gaps of queue+prefill) | the throughput/latency-of-first-token tradeoff |
| P4.2.5 | goodput under an ITL-SLO (p99 ≤ 15 ms) with vs without chunking, bursty trace | **chunking ≥1.3× goodput** at the SLO (or explicitly killed) | the spike violates the SLO for a burst of decodes; chunking keeps them under it |

**Kill criteria:** (a) any token divergence ⇒ stop, fix the offset/causal-mask math (never measure a
broken oracle). (b) if ITL p99 does NOT fall as C↓ after the mechanism is confirmed correct ⇒ the
chunk forward is not actually bounded (profile the chunk forward's cost). (c) if p50 blows up
super-linearly as C→1 ⇒ per-iteration fixed overhead dominates (expected floor; document, don't
gold-plate).

## 3. DoD

- [ ] Token-exact across all chunk sizes (P4.2.1) — oracle green on CPU + GPU.
- [ ] TTFT/ITL curve vs chunk size measured + logged (`bench/RESULTS.md`), the p99-spike reduction shown.
- [ ] Goodput-under-SLO with vs without chunking on a bursty trace (P4.2.5).
- [ ] Adversarial cases: prompt < C (one chunk); prompt = C exactly; prompt not a multiple of C;
      C=1 (degenerate token-by-token prefill); prompt spanning many chunks.
- [ ] Design-note paragraph: the sequential-interleave vs piggyback distinction + the measured curve.

## Measured outcome (2026-07-04) — mechanism ✓, spike-reduction FALSIFIED, R4.2b scoped

Full numbers + verdict in `bench/RESULTS.md` §"Measured — A1 R4.2". Headline:
- **Correctness MET (P4.2.1):** token-exact / bit-identical KV — 44 tests green.
- **Latency win FALSIFIED (P4.2.2–5):** the sequential-interleave scheduler *regresses* every metric
  (ITL p50 ×6.6–14.4, p99 did not fall, throughput 547→145 tok/s). Diagnosed cause: (1) serialized
  admission (1 prefill slot/iteration → 160–550 *unbatched* prefills vs one-shot's 14 batched) starves
  decode; (2) the chunk is a separate sequential forward, so its full cost lands in every decode gap
  (ITL p50 6.6× worse even at C=512 = single chunk).
- **Lesson:** chunked prefill's win is a kernel/batching property (Sarathi "stall-free batching"
  piggybacks the chunk INTO the fused decode forward), not a scheduling-only one. A separate sequential
  forward per chunk costs more than the spike it removes.
- **R4.2b (deferred, not an R4.3 prerequisite):** fused mixed-query-length prefill+decode kernel
  (extend the R4.1 paged Triton kernel to query-len-C rows alongside query-len-1) + batched chunk
  admission + a Poisson/shallow trace to isolate the spike from queue saturation. The token-exact
  `ChunkPrefillView` mechanism built here is the substrate R4.2b reuses.
