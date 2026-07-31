# A1 — Transformer Inference as a System

> **Book chapter:** 5 (Transformer Inference in CUDA) · **Frontier thesis:** Inference is not "the forward pass run repeatedly." It is a *memory-bandwidth-bound scheduling problem* whose economics are set by goodput-under-SLO. You will build a decoder from scratch, prove where its time goes, and turn it into a serving engine using the techniques that run in vLLM, SGLang, and TensorRT-LLM today.
>
> **Primary hardware:** 1× H100 (A100 or even a 4090 is fine for rungs 0–3). FP8-KV and large-model stretch goals want an H100/H200. · **Est. time:** 1.5–2 weeks · **Prereqs:** book ch1–4; the discipline in `00_foundations.md`.

---

## §0 Why this matters

A frontier lab spends far more compute *serving* a model than training it. The difference between a serving stack at 40% and 80% goodput is, directly, half the fleet. The book's chapter 5 ends on a deliberately humbling result: a hand-written *naive* CUDA decoder is **slower** than PyTorch (TTFT 16 ms vs 12 ms; 85 vs 95 tok/s). That is the correct lesson — raw CUDA is not automatically fast, and inference speed comes from *system-level* decisions (caching, batching, scheduling, precision), not from rewriting matmuls in C.

This assignment is where you stop thinking "model" and start thinking "engine." Every optimization here — KV cache, paged memory, continuous batching, speculative decoding, prefill/decode disaggregation — exists because someone measured where the time and memory actually went and attacked *that*. By the end you will be able to look at a serving workload and say, with numbers, "this is decode-bound, your KV cache is fragmenting 30% of HBM, and you're leaving 2× on the table by not batching the decode phase."

---

## §1 Learning objectives

You can:

1. **Derive** the arithmetic intensity of prefill vs decode and place each on the roofline, and prove from first principles why decode is memory-bandwidth-bound while prefill is compute-bound.
2. **Implement** a KV-cache autoregressive decoder from scratch and demonstrate the GEMM→GEMV transition the cache induces.
3. **Compute** the exact KV-cache memory footprint for any model/seq/batch, and explain how GQA, MQA, and **MLA** shrink it — including MLA's weight-absorption trick and decoupled RoPE.
4. **Build** a serving layer: PagedAttention block management, continuous (iteration-level) batching, and chunked prefill, and reason about the TTFT/ITL/throughput/**goodput** trade-offs they govern.
5. **Implement** lossless speculative decoding and measure its acceptance rate and speedup; explain EAGLE/Medusa/MTP at the mechanism level.
6. **Eliminate** launch overhead with CUDA graphs and quantify the win at batch 1.
7. **Articulate** prefill/decode disaggregation: why splitting the phases across GPU pools raises goodput, and what the KV-transfer cost is.

---

## §2 First-principles theory

### 2.1 Two phases, opposite hardware profiles

**Prefill** ingests the prompt's `T` tokens in parallel. The per-layer projections are matrix–matrix (GEMM): `[T, C] @ [C, C] → [T, C]`. Arithmetic intensity is high → **compute-bound**, tensor cores matter (this is the book's figure 5.11 "GEMM (prefill)").

**Decode** generates one token at a time. With a KV cache, each step's projections are matrix–vector (GEMV): `[1, C] @ [C, C] → [1, C]`, and attention is one query against the cached K/V (`[1, C] @ [C, S]`). Arithmetic intensity ≈ 1 op/byte → **memory-bandwidth-bound**. This is the book's central figure 5.6 / 5.11 ("GEMV (decode)"), and it is the single most important fact in inference.

**Do the roofline by hand.** A decode step reads every weight once to produce one token's worth of output. For a model with `P` parameters in BF16, that's `2P` bytes moved for `~2P` FLOPs → AI ≈ 1. On an H100 (3.35 TB/s, ridge ≈ 295 FLOP/byte for FP16), AI=1 is *far* left of the ridge: your ceiling is `bandwidth / 2 bytes-per-param` tokens/sec, full stop. A 7B model in BF16 reads ~14 GB/token → at 3.35 TB/s the hard ceiling is ~240 tok/s/request regardless of how fast the tensor cores are. **Memorize this calculation; it predicts decode throughput before you write a line of code.**

The corollary that drives everything downstream: to use the idle tensor cores during decode, you must **raise arithmetic intensity** — by *batching* (many requests share one weight read, GEMV→GEMM) or by *speculation* (verify many candidate tokens in one forward pass). Both are in §4.

### 2.2 The KV cache and its cost

Without a cache, generating token `t` recomputes attention over all prior tokens — generating 100 tokens costs `1+2+…+100 = 5050` position-computations. The cache stores each token's K and V once; subsequent steps compute only the new token's K/V and *read* the rest (the book's ~50× compute reduction). The cache is what turns the decode attention from GEMM into GEMV.

**Exact sizing** (the formula you must be able to write from memory):
```
KV_bytes = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes × seq_len × batch
```
The leading `2` is K and V. The book's listing 5.19 is the special case `2 × batch × seq × head_dim × 4` for a single-KV-head FP32 toy. Internalize the general form and its consequences: KV memory grows **linearly in both sequence length and batch**, and at long context it dwarfs the weights. This is *the* capacity constraint on how many concurrent requests you can serve, which is *the* lever on throughput. (Book exercise 5.3: at what sequence length does the cache exceed 1 GB? Answer it for a real 7B-class config, not the toy.)

**Shrinking the cache** (each is a `n_kv_heads` or precision reduction):
- **MQA** (Multi-Query Attention): all query heads share **one** K/V head. Cache shrinks H×; quality can drop.
- **GQA** (Grouped-Query): query heads in `g` groups share K/V heads. Interpolates MHA↔MQA; near-MHA quality at a fraction of the cache. The de-facto standard (Llama-3, Mistral, Qwen). The win is a memory-**bandwidth** win — decode reads less KV per step.
- **MLA** (Multi-head Latent Attention, DeepSeek-V2/V3): cache a single **low-rank latent** `c_KV` per token (dim `d_c`≈512) instead of full K/V, and reconstruct per-head K/V via learned up-projections. The inference magic is **weight absorption**: because scores are `(W^UQ c^Q)·(W^UK c^KV)`, you fold `W^UK` into the query projection and `W^UV` into the output projection, so you never materialize full K/V at decode — you attend *in the latent space*. Reported KV cache ≈ **4–14% of MHA** while matching or beating quality. The catch: RoPE doesn't commute with the absorbed matrices, so MLA carries a small **decoupled-RoPE** key/query pathway (dim `d_h/2`) alongside the compressed path.
- **FP8 / INT4 KV**: halve or quarter the bytes (covered in A5; production default is FP8 E4M3 KV, near-lossless).

### 2.3 The metrics that actually matter

You will measure four numbers for every experiment, from day one:
- **TTFT** (time to first token) — gated by prefill; the responsiveness metric.
- **ITL / TPOT** (inter-token latency / time-per-output-token) — gated by decode.
- **Throughput** — aggregate tokens/sec across all concurrent requests.
- **Goodput** — the throughput that *meets an SLO* (e.g., TTFT < 200 ms AND ITL < 50 ms). This is the metric serving systems optimize, formalized by DistServe. A system with huge raw throughput but missed SLOs has low goodput and loses customers.

Request latency ≈ `TTFT + ITL × num_output_tokens`. Hold this identity in your head — it tells you which optimization (prefill-side vs decode-side) moves which number.

---

## §3 The from-scratch build ladder

Validate every rung against the oracle (§5) before promoting. Keep the journal.

**Rung 0 — Reference decoder + metrics harness.**
Take a small decoder (the book's char-level GPT, or nanoGPT, or a HF Llama-3.2-1B for realism) and run greedy autoregressive generation in PyTorch eager. Build the **metrics harness** now — TTFT, ITL, throughput, percentile latencies — because you reuse it in every later rung and in A4/A5. Reproduce the book's honest result: a naive token-by-token loop is *not* faster than batched PyTorch. *Outcome:* a trustworthy oracle and a measurement rig. *Target:* none — this is your baseline.

**Rung 1 — KV cache from scratch.**
Implement a contiguous KV cache; at each decode step compute only the new token's Q/K/V, append K/V to the cache, attend against the cache. Demonstrate the **GEMM→GEMV** shift (book fig 5.6/5.11) by logging the attention op shapes. Implement the **sizing formula** and sweep seq_len/batch until you hit HBM limits. *Correctness:* token-exact vs Rung 0 (greedy). *Profiling target:* prove decode is memory-bound — Nsight SoL shows Memory% ≫ Compute%, and measured tok/s approaches the `bandwidth/2P` ceiling you computed by hand. This is the most important measurement in the assignment.

**Rung 2 — GQA / MQA.**
Reduce `n_kv_heads` (group the query heads). Measure the KV-cache memory reduction and the decode bandwidth reduction. *Correctness:* match a GQA reference (or, for a from-scratch model, verify the grouped attention math against a full-MHA equivalent on a fixed seed). *Outcome:* you can state the memory/quality trade-off with numbers.

**Rung 3 — Static → continuous batching.**
Run `B` requests at once. Show that as `B` grows, the decode projections go GEMV→GEMM and **arithmetic intensity rises** — measure tok/s climbing toward the compute roofline. Then replace the static batch (all requests start/stop together) with an **iteration-level scheduler**: finished requests exit and new ones join after each decode step (Orca-style continuous batching). *Correctness:* each request's output matches its single-request greedy result. *Target:* demonstrate ≥2× aggregate throughput vs static batching on a mixed-length request trace, and quantify the latency cost to in-flight requests.

> The MoE path from the book (listing 5.20/5.21 — gate → softmax → top-k → dispatch) is a worthwhile optional rung here. Build the router; reproduce the book's **numerical-stability lesson**: a 1e-6 difference in the softmax/top-k can flip which expert a token routes to, and those divergences compound. This is your first encounter with "correct in isolation, wrong in composition," and it's exactly why production systems validate routing against a reference. Carry that paranoia into every kernel you write.

---

## §4 Frontier-2026 core (required)

This is the point of the assignment. The book stops at a single-request KV-cache decoder; a frontier serving engine is built from the following. Implement at least **PagedAttention, continuous batching + chunked prefill, speculative decoding, and CUDA graphs** to completion; **MLA** and **disaggregation** may be done at toy scale but must be demonstrated.

**4.1 PagedAttention — KV memory as virtual memory.**
Store the KV cache in fixed-size **blocks** (e.g., 16 tokens) placed non-contiguously in HBM, with a per-request **block table** mapping logical→physical blocks (vLLM's design). Implement a paged-KV attention kernel (Triton is the right tool for a first version) that gathers blocks via the table. *Demonstrate:* memory waste drops below ~4% vs a slab allocator that reserves max-length per request (which wastes 60–80%), and the freed memory lets you raise the batch size — directly raising throughput. Add **copy-on-write** so requests sharing a system prompt point at the same physical blocks until one diverges. *This is the single highest-leverage serving optimization; vLLM's 2–4× throughput came from here.*

**4.2 Continuous batching + chunked prefill.**
You have continuous batching from Rung 3. Now add **chunked prefill**: split a long prompt's prefill into N-token chunks and interleave decode steps between them, so a big incoming prefill doesn't stall (head-of-line block) all in-flight decodes and spike their ITL. *Demonstrate:* the TTFT/ITL trade-off curve as you vary chunk size — this is the knob real schedulers (Sarathi-Serve, vLLM V1) expose. Report goodput under a fixed SLO with and without chunking on a bursty trace.

**4.3 Speculative decoding (lossless).**
A small **draft** model proposes K tokens; the target model **verifies all K in one forward pass**; you accept the longest correct prefix via the rejection-sampling rule that makes the output distribution *identical* to the target's (this losslessness is the whole point — prove it). *Demonstrate:* acceptance rate and end-to-end speedup on code/math vs open-ended prompts (acceptance is higher on structured text). *Mechanism literacy (explain in your writeup, implement one as stretch):* Medusa (extra heads on the target predict multiple future tokens, tree-verified), EAGLE-2/3 (autoregress at the *feature* level with dynamic draft trees — current SOTA among lossless methods), MTP (DeepSeek-V3's lightweight next-2-token heads, ~85–90% 2nd-token acceptance → ~1.8× TPS), and **DSpark** (arXiv:2607.05147, 2026-06 — reports +57–85% over a single MTP head; the current edge of the MTP lineage). Speculation raises arithmetic intensity, attacking the §2.1 decode bottleneck from the other side.

**4.4 CUDA graphs for decode.**
A decode step launches hundreds of tiny kernels; at batch 1 the ~5–10 µs/launch CPU overhead can be 20–40% of step time (the GPU sits idle waiting for the CPU). Capture the decode step as a CUDA graph and replay it as one op. *Demonstrate:* the launch-overhead reduction (~28% per-decode-step is the reported vLLM-V1 figure) via an `nsys` timeline showing the CPU gaps before/after. This is *the* reason decode is CPU-bound without graphs, and why it's a default-on optimization in every serious engine.

**4.5 MLA latent cache (toy scale ok).**
Implement MLA's compressed KV (`c_KV` latent + decoupled RoPE) on a small model and verify the **weight-absorption** identity numerically (attending in latent space gives the same scores as reconstructing full K/V). *Demonstrate:* the cache-size reduction vs MHA/GQA. You don't need to train it — you need to *show the mechanism works and why it's a serving win.*

**4.6 Prefill/decode disaggregation (demonstrate).**
Because prefill is compute-bound and decode bandwidth-bound, co-locating them forces a bad compromise. Stand up **two processes** — a prefill worker and a decode worker — and transfer the KV cache between them (start with local IPC/NVLink; the KV transfer is the real cost and the active research frontier — Mooncake, DistServe, Splitwise). *Demonstrate:* goodput vs a co-located baseline on an SLO'd trace, and characterize the KV-transfer overhead. (Full multi-GPU disaggregation connects to A6.)

> **Production stack (2026, mechanism literacy):** NVIDIA **Dynamo** (1.0 GA, GTC-2026) is the reference disaggregated-serving stack — KV-aware routing, an autoscaling planner, and **NIXL** for KV transfer over NVLink/RDMA; vLLM (≥0.25) and SGLang both ship PD-disagg behind KV-connector APIs, and **KV offloading/tiering** (host-RAM/SSD tiers — Mooncake's KVCache pool, SGLang HiCache, vLLM's KV-offloading connector) is the long-context complement. One step beyond PD-disagg is **AFD (Attention–FFN disaggregation)**: Step3's MFA+AFD co-design (stepfun-ai/Step3; vLLM RFC #22799, shipped as the vLLM AFD plugin 2026-07) separates the attention layers (KV-heavy, TP-friendly) from the MoE FFN (wide-EP-friendly) onto different device pools — the same "different compute/memory profiles want different placement" logic as PD-disagg, applied *within* a layer. Know it exists and why it wins on high-expert-count MoE; PD-disagg remains the required build here.

---

## §5 Correctness & numerics

- **Greedy dense decode must be token-exact** vs the Rung-0 oracle. Any divergence is a bug (off-by-one in positions, wrong cache slice, RoPE applied at the wrong index). The book's listing 5.23 does exactly this assertion — adopt it.
- **MoE / routing**: tolerance-based, not exact — and *understand why* (the 1e-6→expert-flip lesson). Compare against a CPU reference; track the token-divergence rate; a high rate means a real bug, a low rate is expected sparse-routing noise.
- **Speculative decoding must be distribution-equivalent** to plain decoding. Verify: with greedy targets, spec-decode output must be *identical* to non-spec greedy. With sampling, verify the accepted-token distribution matches the target's (run many samples, compare histograms). If spec-decode changes outputs, your rejection rule is wrong — this is the classic bug.
- **Paged attention** must match contiguous-KV attention bit-for-bit (same math, different memory layout). Test with non-contiguous block allocations and a request whose blocks are deliberately scattered.
- **Adversarial cases:** empty prompt, prompt longer than one block, a request that finishes mid-batch, two requests sharing a prefix (CoW), sequence length crossing a block boundary.
- **Batch-invariance / determinism (the 2026 serving-correctness frontier).** A request's output should not depend on *which other requests share its batch* — but batch-size-dependent reduction orders (split-KV choices, chunked prefill boundaries, variable tile splits) make it so. This is not pedantry: RL post-training compares rollout (inference-engine) logprobs against trainer logprobs, and batch-variance becomes a real train↔infer mismatch. The fix family is **batch-invariant kernels** (Thinking Machines, "Defeating Nondeterminism in LLM Inference," 2025-09; SGLang's deterministic mode, LMSYS blog 2025-09-22; vLLM batch-invariant ops) — fixed reduction orders, at a measurable throughput cost. *Test:* the same greedy request at batch 1 and batch 64 must produce bitwise-identical tokens (when determinism is enabled); if not, name the kernel whose reduction order is the culprit.

---

## §6 Profiling & performance

- **The headline measurement:** decode is memory-bound. Show it three ways — the hand roofline (AI≈1), the Nsight SoL section (Memory% ≫ Compute%), and measured tok/s approaching `HBM_BW / 2P`. If your decode is *not* near the bandwidth ceiling, find the overhead (almost always launch overhead → CUDA graphs, or an unfused op).
- **Batching sweep:** plot tok/s and AI vs batch size; annotate where you cross from bandwidth-bound toward compute-bound. This plot *is* the justification for continuous batching.
- **Launch overhead:** `nsys` timeline before/after CUDA graphs, with the CPU-side gaps measured.
- **The metrics dashboard:** TTFT/ITL/throughput/goodput at p50/p95/p99 for every serving experiment. Goodput-under-SLO is the number that decides whether an optimization shipped.
- **Benchmarking hygiene** per `00_foundations.md`: locked clocks, warmup discarded, CUDA-event timing.

---

## §7 Stretch goals (competition-grade)

- **Real model end-to-end:** serve Llama-3.1-8B or Qwen3-class weights with your paged-KV + continuous-batching engine; compare goodput against vLLM on the same trace (you will lose — by how much, and where, is the interesting writeup).
- **FP8 KV cache** (bridges to A5): E4M3 KV with per-channel-K/per-token-V scales; show near-lossless quality and doubled KV capacity.
- **Integrate FlashInfer** as the attention backend and compare against your hand-rolled paged kernel.
- **RadixAttention prefix cache** (SGLang-style): a radix tree over KV for automatic cross-request prefix reuse; benchmark on a multi-turn/few-shot trace.
- **EAGLE-3 draft head:** train a feature-level draft head and beat your draft-model acceptance rate.

---

## §8 Deliverables & definition of done

1. **Code:** the decoder + KV cache, paged attention, continuous-batching + chunked-prefill scheduler, speculative decoder, CUDA-graph decode, MLA toy, and the disaggregation demo. Each with its oracle test.
2. **The metrics harness** (reusable) and the experiment plots: roofline placement of decode, the batching AI/throughput sweep, the chunked-prefill TTFT/ITL curve, the spec-decode acceptance/speedup, the CUDA-graph timeline.
3. **Design note (2–3 pages): "Where does inference time and memory go, and what did I do about it?"** Lead with the roofline argument; walk each optimization with its before/after number and the Nsight/nsys evidence; end with the honest gap to vLLM and *why* it exists. Write it for a frontier-lab peer.

**Done when:** every rung matches its oracle; decode is demonstrably at the bandwidth ceiling; PagedAttention, continuous+chunked batching, lossless spec decoding, and CUDA graphs all work and are measured; MLA and disaggregation are demonstrated; and the design note would survive review by someone who builds serving systems.

---

## §9 Principal-level rubric

- **Pass:** KV-cache decoder works, token-exact, with a correct sizing formula.
- **Strong:** + PagedAttention and continuous batching working and measured; decode proven memory-bound on the roofline.
- **Principal-grade:** all of §4 demonstrated; goodput-under-SLO is your reporting metric; you can predict decode tok/s from the hand roofline *before* measuring and the prediction holds; speculative decoding is provably lossless; and your design note correctly diagnoses the remaining gap to a production engine (e.g., "we're 1.7× off vLLM because our paged kernel doesn't fuse the dequant and our scheduler lacks prefix caching — here's the evidence"). The differentiator is **systems judgment**: knowing that the win is in scheduling and memory, not in the matmul.

---

## §10 References (see `references.md` for full list)

- Kwon et al., **PagedAttention / vLLM**, arXiv:2309.06180 + vLLM V1 blog (2025-01-27).
- Zheng et al., **SGLang / RadixAttention**, arXiv:2312.07104.
- Ye et al., **FlashInfer**, arXiv:2501.01005 (MLSys'25 best paper).
- Zhong et al., **DistServe** (goodput, disaggregation), arXiv:2401.09670; **Splitwise**, ISCA'24; **Mooncake**, arXiv:2407.00079.
- DeepSeek-AI, **DeepSeek-V2 (MLA)** arXiv:2405.04434; **V3 (MLA+MoE+MTP)** arXiv:2412.19437.
- Li et al., **EAGLE-2** arXiv:2406.16858 / **EAGLE-3** arXiv:2503.01840; **Medusa**.
- Yuan et al., **LLM Inference Unveiled** (survey + roofline), arXiv:2402.16363.
- **DeepEP** (MoE all-to-all), github.com/deepseek-ai/DeepEP.
- **NVIDIA Dynamo** (disaggregated serving: KV-aware routing, planner, NIXL KV transfer), 1.0 GA GTC-2026, github.com/ai-dynamo/dynamo.
- **Step3 / AFD** (Attention-FFN disaggregation), github.com/stepfun-ai/Step3 + vLLM RFC #22799 + vLLM AFD-plugin blog (2026-07-23).
- **Deterministic inference:** Thinking Machines, "Defeating Nondeterminism in LLM Inference" (2025-09); LMSYS, "Towards Deterministic Inference in SGLang and Reproducible RL Training" (2025-09-22).
- **DSpark** (speculative decoding), arXiv:2607.05147.
