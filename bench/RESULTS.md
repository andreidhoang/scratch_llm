# Measurement ledger — predicted vs measured (the "DoD is a profile" record)

> The durable record of every **measured** performance number. Git + [`../docs/STATUS.md`](../docs/STATUS.md)
> track *what is built*; this file tracks *what is measured* — because a rented GPU is released and the
> number must persist. The discipline (FOP-3 / FOP-4, and every rung's *Profile (DoD)* in
> [`../docs/GPU_FROM_ZERO.md`](../docs/GPU_FROM_ZERO.md)): **predict the number and the bound first, then
> measure, then log the gap and the root cause.** The spine that says *which* numbers matter is
> [`../docs/PERFORMANCE_TRACK.md`](../docs/PERFORMANCE_TRACK.md).
>
> A row is a result only if it is `[FACT]` — measured under `cuda.synchronize`, fixed seed, warm-ups.
> An unmeasured expectation is `[INFERENCE]` and does not belong here until measured.

## Format

`| date | rung / artifact | hardware | metric | predicted | measured | bound | root cause (1 line) | next experiment |`

- **bound** = the roofline verdict: `compute` / `memory` / `comms` / `overhead` / `latency`.
- Use the honesty constants from `PERFORMANCE_TRACK.md §5` (dense ~295 FLOP/byte H100 ridge, FP4 = 2× FP8, …).

## Ledger

| date | rung / artifact | hardware | metric | predicted | measured | bound | root cause | next experiment |
|---|---|---|---|---|---|---|---|---|
| 2026-06 (prior) | Rung 5 · FA2 fwd roofline | RTX 4090 | % of SDPA @ seq 4k | ~65% | **53%** | memory | Triton fwd leaves HBM traffic on the table vs SDPA's fused schedule (below the 60% kill line — shipped as a documented negative) | re-measure post-bwd; larger tiles / fewer reloads; compare vs `torch.compile` fused attn |
| 2026-06-29 | hardware baseline | RTX PRO 4000 Blackwell (sm120) | bf16 GEMM 8192³ · HBM copy · ridge | — | **72 TF/s · 0.55 TB/s · ridge≈130 FLOP/B** | — | the standing card; bf16 modest (Blackwell headroom is FP4/FP8) — peaks now in `bench.py _PEAKS` | re-measure FA2 % of SDPA on *this* card; measure FP8/FP4 GEMM |
| 2026-06-29 | Triton stack check | RTX PRO 4000 Blackwell | gpu-test pass/fail | all pass | **FA2-Triton + gemv ✅ ; CUDA rmsnorm.cu = NaN** | — | Triton-primary (ADR-0011) validated on sm120; CUDA-C++ rmsnorm regresses (arch/build) — 2nd-tier, superseded by the R3 Triton norm rebuild | triage `rmsnorm.cu` sm120 build flags or quarantine `@pytest.mark.gpu` |
| 2026-07-01 | A1 R1 · decode tok/s (0.84B bf16) | RTX PRO 4000 Blackwell (sm120) | decode tok/s @B=1 | 315 (mem ceiling) | **51** (16% of roof) | overhead | memory-bound *workload* (AI≈0.96, 136× below ridge) but the eager run is launch-overhead-bound, NOT memory-saturated: achieves only 89 GB/s = 16% of the card's 550 GB/s; nsys = ~955 kernel launches/token (dominant kernel = GEMV, the decode shape) + a `.item()` host-sync per token; memory traffic is 16% of the 19.6 ms/token wall, 84% is overhead | CUDA graphs (R4.4) collapse the 955 launches → predict climb toward 315 tok/s & achieved-BW→550 (only then memory-bound); keep token-id on GPU to kill the per-token sync |

<!-- append new measurements below; never edit a logged row (it is a dated record) -->

### Pre-registration — A1 R3 continuous batching (predict-before-run, D5)

Registered 2026-07-01 BEFORE building. Spec: `performance/notes/A1_R3_continuous_batching.md`.
Thesis: **batching is how you beat the memory wall** — read weights once, apply to B sequences → AI≈B,
weight traffic amortized. Phased: R3a static batched decode (weight-amortization roofline) → R3b
continuous scheduler (≥2× vs static). Config: GQA-4 0.84B, compiled (the R1 path that reaches the wall).

| # | experiment | predicted | measured | bound | note |
|---|---|---|---|---|---|
| R3.1 | decode AI at batch B (short ctx) | ≈ B (memory while B<130) | — (pending) | memory | AI = 2PB/(2P+B·KV) |
| R3.2 | aggregate tok/s B=32 vs B=1 | **≥10×** (weights + overhead amortize); ~linear | — (pending) | memory | THE falsifier: flat ⇒ not amortizing |
| R3.3 | batched row b vs single-stream greedy | token-exact | — (pending) | — | per-row length mask correctness |
| R3.4 | continuous vs static, mixed 128/512 @B=32 | **≥2×** aggregate | — (pending) | — | kills head-of-line + padding waste |
| R3.5 | ITL @B=32 vs B=1 | higher (throughput↔latency) | — (pending) | — | the honest cost of batching |

**Kill line:** batched row ≠ single-stream ⇒ per-row length-mask/positions bug (fix first). Aggregate
flat in B ⇒ not amortizing weights (bytes/step must be ~2P+B·KV, not B·2P). Continuous <2× static ⇒
scheduler not refilling freed slots (measure slot utilization before blaming the trace).

### Measured — A1 R3a static batched decode (2026-07-01, `bench/batched_decode.py` + `tests/test_batched_decode.py` green)

| date | rung / artifact | hardware | metric | predicted | measured | bound | root cause | next experiment |
|---|---|---|---|---|---|---|---|---|
| 2026-07-01 | A1 R3a · batched decode agg tok/s vs B, **compiled** | RTX PRO 4000 Blackwell (sm120) | agg tok/s scaling | ≥10× @B=32; ~linear; AI≈B | **R3.1/R3.2 CONFIRMED** — AI≈B exact; agg(B=32)/agg(B=1)=**24.9×**; 185→9,255 tok/s B=1→64 (50× for 64×); **peak 12,220 @B=256 = 66×** | memory→compute | Weights read once/step amortize over B rows (per-stream ~145–185 tok/s, %HBM ~50–56% in the memory regime). Roofline **crosses memory→compute at B≈128** (AI≈ridge 131); plateau ~12K tok/s = the naive `@` matmul's ~28% of the 72 TF/s compute peak | R3b continuous scheduler (≥2× vs static on mixed trace); A2/A3 tensor-core GEMM to raise the compute ceiling |
| 2026-07-01 | A1 R3a · batched decode (eager control) | RTX PRO 4000 Blackwell (sm120) | agg tok/s scaling | (pred: overhead masks amortization) | **PREDICTION CORRECTED** — eager ALSO scales: agg(32)/agg(1)=30.7×, peak 13,462 @B=256 | overhead | Batching amortizes the **fixed ~20 ms/step launch overhead** across B tokens too (step time ~constant in B → per-stream ~48, aggregate linear). Compiled wins at low-mid B (185 vs 48 tok/s @B=1, strips overhead → 56% HBM) but both converge at high B (~20 ms/step, compute/overhead-bound) | — |

**Verdict (A1 R3a DONE — R3.1/R3.2/R3.3 [FACT]):** batching is the lever that beats the B=1 memory wall —
aggregate tok/s scales ~linearly with B (weights amortized) until the roofline crosses to compute-bound
at **B≈128 (AI≈ridge)**, peaking ~66× the single-stream rate. The oracle (test_batched_decode) pins
batched row == single-stream. Remaining for Rung-3 close: **R3b** (continuous scheduler: variable lengths
+ join/leave, ≥2× vs static — R3.4/R3.5), which needs the static `BatchedKVCache` buffer (the R4.1/R4.4
linchpin). Correction logged: eager does not mask amortization — it amortizes the fixed launch overhead.

### Pre-registration CORRECTION + R3b registration (2026-07-03, before build)

The R3.4 row above registered **≥2× on a 16×128 + 16×512 trace** — that gate is **analytically
unreachable**: the spec note inverted idle-fraction and utilization (16·384 idle slot-steps of 32·512 =
37.5% *idle* → **utilization 62.5%**, and perfect continuous batching on a saturated queue is bounded by
1/util = **1.6×**, ~1.4–1.5× after the admission-prefill tax). Built as registered, the kill criterion
("<2× ⇒ scheduler not refilling") would have misfired on a *correct* scheduler. Model + corrected gates
(spec: `performance/notes/A1_R3_continuous_batching.md` §2, corrected in place with provenance):

`speedup ≈ max_len / (mean_len + B·τ_p/τ)` — static-wave util = mean_len/max_len; admit tax ≈ B·τ_p/τ.

| # | experiment (B=32, prompt 32, GQA-4 0.84B compiled) | predicted | measured | bound | note |
|---|---|---|---|---|---|
| R3.4 | continuous vs static-wave, **heavy-tail 24×64+6×256+2×512** | **≥2×** (point ~2.7×; ceiling 4.0×) | — (pending) | — | the corrected primary gate |
| R3.4s | continuous vs static-wave, 16×128+16×512 (as first registered) | ~1.4–1.5× (ceiling 1.6×) | — (pending) | — | sensitivity: win is a fn of length dispersion |
| R3.6 | TTFT p95 continuous vs static-wave (queued trace) | **≥4× lower** | — (pending) | — | admit-on-slot-free vs wait-for-wave-end |

R3.3 (ragged oracle) and R3.5 (ITL cost) stand as registered. Kill lines updated: continuous < 0.8× its
*analytic ceiling* ⇒ measure slot utilization first (util≈100% ⇒ prefill stalls; <90% ⇒ refill bug);
compile recompiling every step ⇒ a dynamic shape leaked (fix before benching — it forfeits R4.4 capture).

### Measured — A1 R3b continuous batching (2026-07-03, `serving/continuous.py` + `bench/continuous.py`; 21 CPU oracle tests green)

| date | rung / artifact | hardware | metric | predicted | measured | bound | root cause | next experiment |
|---|---|---|---|---|---|---|---|---|
| 2026-07-03 | A1 R3b · continuous vs static-wave, **heavy-tail** (24×64+6×256+2×512)×8 waves, B=32, compiled decode + eager prefill | RTX PRO 4000 Blackwell (sm120) | agg tok/s ratio | **≥2×** (point ~2.7×; ceiling 4.0×) | **R3.4 PASS: 2.30× wall** (2,283 vs 995 tok/s); **2.93× by step count** (1,396 vs 4,088) | memory | Iteration-level refill keeps util at 72.8% (drain-tail-limited) vs wave's 24.9% (== analytic exactly). Wall < step-ratio because the mixed-age batch holds `L_view` high most steps → **dense-buffer padding traffic: 9.6 ms/step vs wave 6.3** (the 2.93→2.30 gap ≈ 1.27×) | **R4.1 PagedAttention** — reclaim the measured padding tax; R4.4 cudagraphs on the same buffer |
| 2026-07-03 | A1 R3b · sensitivity trace 16×128+16×512 (×8 waves) | RTX PRO 4000 Blackwell (sm120) | agg tok/s ratio | ~1.4–1.5× (ceiling 1.6×) | **1.11× wall · 1.46× by steps** (2,809 vs 4,088; util 90.9%) | memory | Step-ratio lands exactly in the predicted band (scheduling math confirmed); the same padding tax (10.4 vs 8.0 ms/step) compresses the wall ratio below it. Win is a fn of length dispersion, as registered | same |
| 2026-07-03 | A1 R3b · TTFT p95, shallow queue (2 waves), heavy trace | RTX PRO 4000 Blackwell (sm120) | wave/continuous | **≥4×** | **R3.6 PASS: 4.9×** (852 ms vs 4,146 ms; p50 9.4×) | — | Admit-on-slot-free vs wait-for-wave-end. Sensitivity: on 16/16 the p95 ratio is 1.0× (a tail request waits on 512-len rows holding half the slots either way) while p50 is still 4.9× — the TTFT win also scales with length dispersion. At 8-wave saturation TTFT is queue-wait-dominated for both (not the R3.6 surface) | — |
| 2026-07-03 | A1 R3b · ITL cost of batching (R3.5) | RTX PRO 4000 Blackwell (sm120) | ITL p50 | higher than B=1 | **B=1 5.4 ms → B=32 continuous 9.6–10.8 ms (~1.8–2×)** for ~12–15× aggregate; ITL p99 ~30 ms spikes at admission events (predicted — prefill in the decode stream) | memory | The honest throughput↔latency trade. Wave ITL p50 6.3 ms < continuous 9.6: the padding-traffic effect again (uniform-age batch reads a smaller mean `L_view`) | R4.2 chunked prefill would smooth the p99 admission spikes |

**Engineering findings (the two dynamo leaks + the trace-design lesson, all `[FACT]`):**
1. **Saturation is part of the spec:** at 2 wave-mixes the post-queue **drain tail** dominated (continuous
   util 44%, ratio 1.70×) — the analytic model assumes a saturated queue, so the measured trace must be
   deep enough (8 waves) that steady state dominates. Latency (R3.6) is measured at *shallow* queue, where
   admission policy — not queue wait — sets TTFT; throughput (R3.4) at saturation. One trace can't do both.
2. **Compiling the prefill path is a shape-churn trap:** steady-state admissions arrive as n=1,2,3,… and
   every new width compiled a fresh graph (47 graphs, ~35 s of in-run compile, plus a per-call linear guard
   scan taxing *every* step). Fix: **prefill runs eager** (it is ~1% of wall); decode owns the compile budget.
3. **Python list state read in-graph bakes ordering guards:** `view_len = max(py_lengths)` traced into the
   forward → Dynamo guarded on *which slot holds the max* (`py_lengths[26] > py_lengths[16]`); ragged churn
   permutes it → recompile-limit hit → **eager fallback in the continuous arm only** (wave's uniform lengths
   kept one stable ordering — a bias that *favored the baseline*). Fix: `view_len` is a plain int attribute
   recomputed only by the scheduler-owned mirror ops. After both fixes: **unique_graphs = 2**.

**Verdict (A1 Rung 3 CLOSED — R3.3/R3.4/R3.5/R3.6 `[FACT]`, R3.4s reported):** iteration-level scheduling
delivers **2.93× fewer lockstep steps** on the heavy-tail trace (util 24.9%→72.8%) and **2.30× wall
throughput** (gate ≥2× PASS); the 1.27× gap between the two is the **dense slot buffer's padding traffic,
measured** — which is exactly the itch R4.1 (PagedAttention) exists to scratch, and the static buffer built
here is its prerequisite (and R4.4's). Oracle: every request token-exact vs single-stream greedy across
churn/ragged/poisoned-slot tests (21 CPU tests). Prediction quality: step-count ratios landed inside the
corrected pre-registration (2.93 vs ceiling 4.0 with drain; 1.46 vs 1.4–1.5 band); the wall-ratio shortfall
exposed a real unmodeled term (padding traffic) — logged, quantified, and now the next rung's target.

### Pre-registration — A1 R4.1 PagedAttention (predict-before-run, 2026-07-03)

Registered BEFORE building. Spec: `performance/notes/A1_R41_paged_attention.md`. Thesis: paged KV
converts the slab's **reservation waste** (~95% on this trace: slots reserve max_ctx=2048, E[ℓ]≈96)
into **internal fragmentation** (E≈8 tok/row) → 10–20× capacity; but paged *storage alone* is a
throughput LOSS (gather adds traffic; the padded-SDPA compute stays) — only the fused paged decode
kernel reclaims the R3b-measured 1.27× mixed-age tax, whose decomposition (bytes ≈0.5 ms of 3.3)
the kernel result itself will measure.

| # | experiment | predicted | measured | bound | note |
|---|---|---|---|---|---|
| P4.1.1 | frag (paged) vs reservation waste (slab); capacity ratio | frag 4–8% heavy-tail, ≤4% 16/16; slab ≈95%; **10–20× rows/GB** | — (pending) | — | the headline |
| P4.1.2 | paged-gather vs contiguous, scattered blocks + churn + poison | **bit-exact** | — (pending) | — | oracle before any number |
| P4.1.3 | paged-gather step vs dense (negative control) | **+10–25%** (wall ratio 2.30→~1.9–2.1×) | — (pending) | memory | gather ≈ +1.6–2 ms/step |
| P4.1.4 | Triton paged decode kernel | step 9.6→**≤8.3 ms**; wall ratio **≥2.6×**; allclose + greedy-exact | — (pending) | memory/compute | kill: kernel > gather+SDPA at B=32, L≤544 |

Kill lines: gather ≠ bit-exact ⇒ table bug (fix first). frag >8% ⇒ allocator bug. Kernel slower ⇒
ship capacity win alone, kernel → A4 tie-in with profile. Refcount leak after churn ⇒ CoW bug.

### Measured — A1 R4.1 PagedAttention (2026-07-03, `bench/continuous.py --r41`; 13 new tests green)

Four arms, heavy-tail ×8 waves, B=32, compiled decode + eager prefill, scheduling identical
(1,396 continuous steps in every arm — storage is the only variable). dynamo unique_graphs=4.

| # | rung / artifact | metric | predicted | measured | bound | root cause / note |
|---|---|---|---|---|---|---|
| P4.1.1 | frag + capacity | frag %, rows/GB | 4–8% heavy-tail; 10–20× | **frag 5.0%** (dead-center); **capacity ×9.3** (peak-alloc 7,040 tok vs 65,536 reserved) | — | frag exactly as derived (E[tail]≈8/⟨ℓ⟩≈160). Capacity ×9.3 on the STRICTEST accounting (peak concurrent allocation — the provisioning number) vs the band floor 10 (mean-based would read higher); band edge missed by 7%, mechanism confirmed |
| P4.1.2 | contiguous-match oracle | bit-exact | torch.equal | **CONFIRMED** — paged == dense bit-exact on deliberately scattered blocks; boundary 15/16/17/32; poisoned free blocks; churn leak-free; CoW refcounts balance | — | 10 CPU tests; the kernel path additionally fp32 ≤1e-5 / bf16 ≤2e-2 + greedy token-exact e2e (3 GPU tests) |
| P4.1.3 | paged-gather step (negative control) | +10–25% | **+6.3%** (10.26 vs 9.65 ms/step; ×2.11 vs wave-dense) | memory | direction CONFIRMED (paged storage alone loses throughput); magnitude below band — inductor fuses the gather into the attention read better than the hand model. Registering the miss: the gather-copy term was over-modeled ~2× |
| P4.1.4 | fused Triton paged decode | ≤8.3 ms/step; ≥2.6× vs wave | **5.90 ms/step · ×3.52 vs wave-dense · 3,528 tok/s agg (+55% over dense-continuous)** | memory | **Reclaimed the ENTIRE 1.27× mixed-age tax and beat every dense arm**: ITL p50 6.2 ms ≈ wave-dense's 6.1 (uniform-march) — reading only real tokens via the table removes padding bytes AND the padded-SDPA compute. Tax decomposition (the open question from R3b): ~3.2 ms was fp32-score materialization + repeat_interleave + softmax-over-padding; only ~0.5 ms was bytes — as the spec hypothesized |

**Engineering findings `[FACT]`:** two triton-under-`torch.compile` type-inference traps (both compile
fine in eager triton): (1) a python-float loop carry (`m = -inf`) promotes to f64 → "loop-carried
variable re-assigned to fp64"; carry explicitly-typed fp32 tensors. (2) a runtime python-float
kernel arg (`scale`) is passed as f64 by the inductor wrapper and contaminates the whole
online-softmax carry chain; cast the product to fp32 in-kernel.

**Verdict (A1 R4.1 CLOSED — P4.1.1–P4.1.4 `[FACT]`):** PagedAttention delivers both halves on this
card: **~20× less KV memory held** (5% frag vs 95% reservation waste; ×9.3 peak-provisioned
capacity) *and* — only with the fused kernel — **+55% throughput** over the dense slab at identical
scheduling (×3.52 vs static-wave overall). The pre-registered warning stands confirmed: paged
storage WITHOUT a paged kernel is a throughput loss (+6.3%/step) — the win is storage+kernel as a
unit, which is exactly what vLLM shipped. ITL p99 ~29 ms admission spikes persist in all arms —
prefill-in-the-decode-stream, the freshly-measured motivation for **R4.2 (chunked prefill)**.
The fixed-address block pool remains CUDA-graph-capturable (R4.4's substrate).

### Pre-registration — A1 R2 GQA/MQA reduction (predict-before-run, D5)

Registered 2026-07-01 BEFORE running `bench/kv_memory.py`. Spec: `performance/notes/A1_R2_gqa.md`.
KV stored/token = `2·L·H_kv·d_head·dtype` = `4096·H_kv` bytes (Rung-1 config, bf16). Honest frame:
decode is overhead-bound at B=1, so GQA moves tok/s **only** at long ctx under `torch.compile` (pred 5).

| # | experiment | predicted | measured | bound | note |
|---|---|---|---|---|---|
| R2.1 | KV/token vs `4096·H_kv`, H_kv∈{32,8,1} | 128 / 32 / 4 KB (32:8:1) | — (pending) | — | footprint fact |
| R2.3 | crossover ctx (KV read = 1.68 GB weights) | MHA ~13 K · GQA-4 ~51 K · MQA ~410 K | — (pending) | — | GQA pushes KV-bound point out |
| R2.5 | decode tok/s @ ctx=16 K, compiled — MQA vs MHA | **MQA >20% faster** (MHA KV≈2.1 GB > weights) | — (pending) | memory | the only tok/s-moving falsifier |

**Kill line:** grouped math ≠ explicit-repeat reference ⇒ group-map bug (fix first). MQA no edge over
MHA at ctx=16 K under compile ⇒ not at the wall (re-check compiled BW ≈53%) or KV-traffic model wrong.

### Measured — A1 R2 GQA/MQA (2026-07-01, `bench/kv_memory.py` + `tests/test_kv_memory.py` green)

| date | rung / artifact | hardware | metric | predicted | measured | bound | root cause | next experiment |
|---|---|---|---|---|---|---|---|---|
| 2026-07-01 | A1 R2 · KV capacity MHA/GQA-4/MQA | RTX PRO 4000 Blackwell (sm120) | KV/token · crossover ctx | 128/32/4 KB · 13K/51K/410K | **128/32/4 KB · 14.4K/51.3K/396K** | — | **R2.1/R2.3 CONFIRMED.** Exact formula match `2·L·H_kv·d_head·dtype`; GQA-4 pushes the KV-read=weight-read crossover 14K→51K, MQA→396K (test_kv_memory green) | R3: GQA is what makes a big batch fit — B=256 ctx=2K is 68 GB (MHA, OOM) vs 17 GB (GQA-4) |
| 2026-07-01 | A1 R2 · MQA vs MHA decode @ ctx, **compiled** | RTX PRO 4000 Blackwell (sm120) | tok/s ratio (MQA/MHA) | >1.2× @ 16K | **1.03× @2K → 1.81× @8K → 1.92× @16K** (MQA 102 vs MHA 53 tok/s @16K) | memory | **R2.5 CONFIRMED.** Under torch.compile the step is memory-bound, so shrinking KV (MHA 4.03 → MQA 1.69 GB/step) directly cuts step time; the ctx-trend traces MHA's 14K crossover | R3 continuous batching (raise AI by batch); the static-KV-buffer refactor (R4.1/R4.4) |
| 2026-07-01 | A1 R2 · eager decode (control) | RTX PRO 4000 Blackwell (sm120) | tok/s ratio (MQA/MHA) | ≈1× (overhead hides KV) | **0.91–0.95×** (all ~50–55 tok/s regardless of ctx/H_kv) | overhead | **Control CONFIRMED.** Eager step time is set by ~955 launches, not bytes — MHA@16K even shows *higher* achieved BW (220 GB/s) than MQA (84) at the *same* tok/s: extra KV bytes ride free under launch overhead. This is *why* R2.5 needs the compiled path | (same as above) |
| 2026-07-01 | A1 R2 · compiled decode %HBM (MHA @2K) | RTX PRO 4000 Blackwell (sm120) | achieved BW | — | **388 GB/s = 71% HBM** (vs R1's 53%); falls to ~30–40% at 16K | memory | bigger per-step byte volume (weights+KV) amortizes residual launch overhead better than R1's tiny short-ctx step → closer to the wall. %HBM drops at long ctx: naive SDPA over the cache is less BW-efficient than the GEMV weight reads | R4.4 cudagraphs for the residual launches; a fused decode-attention for the long-ctx KV read |

**Verdict (A1 R2 DONE — all 3 DoD boxes [FACT]):** KV footprint == analytic formula (test green +
capacity table); grouping oracle green; the compiled long-ctx sweep **confirms R2.5 (MQA 1.92× MHA @
16K)** while eager confirms the control (overhead hides it). The GQA/MQA lever is now measured on both
axes — **capacity** (32× for MQA) and **long-context traffic** (1.92× decode). Next per plan sequence:
**R3 continuous batching** — raise arithmetic intensity by batching (which GQA makes fit), target ≥2×
aggregate throughput; that motivates the static-buffer `KVCache` rewrite (the R4.1/R4.4 linchpin).

### Pre-registration — A1 R1 overhead-strip (predict-before-run, D5)

Registered 2026-07-01 BEFORE running `bench/decode_overhead_strip.py`. Baseline is the row above
(51 tok/s = 16% of the 315 ceiling; 89 GB/s = 16% of 550; ~955 launches/token + a `.item()`/token
host-sync). Hypothesis: the eager decode is **overhead-bound, not memory-bound** — strip the overhead
and achieved BW climbs toward the 550 GB/s wall (only then is the memory-bound thesis *demonstrated*).

| # | strip (one variable) | predicted tok/s | predicted BW | predicted bound after strip |
|---|---|---|---|---|
| A | kill per-token `.item()` host-sync (argmax stays on GPU) | 51 → **~60–70** | ~89 → ~110 GB/s | still overhead (launches dominate) |
| B | + `torch.compile` (fusion / reduce-overhead) collapse the ~955 launches | → **~150–300** | ~89 → **250–500 GB/s** | **memory** (if the wall appears) |

**Kill line (falsifier):** if achieved BW stays **< 200 GB/s** after both strips, launch overhead is
NOT the dominant confound — re-profile for the real one (unfused RMSNorm/RoPE eager ops? the growing
`torch.cat` KV realloc? Python-side dispatch?). A surprise up is as suspicious as a surprise down (D5).
**Known risk:** the KV cache grows via `torch.cat` (dynamic shape) — CUDA-graphs need static addresses,
so `reduce-overhead` may not capture the step; if so, that scopes R4.4 (static pre-allocated KV buffer).

### Measured — A1 R1 overhead-strip (2026-07-01, `bench/decode_overhead_strip.py`)

| date | rung / artifact | hardware | metric | predicted | measured | bound | root cause | next experiment |
|---|---|---|---|---|---|---|---|---|
| 2026-07-01 | A1 R1 · strip A: nosync (argmax on GPU) | RTX PRO 4000 Blackwell (sm120) | decode tok/s @B=1 | ~60–70 (pred A) | **50** (15% roof · 83 GB/s · 15% HBM) | overhead | **Pred A FALSIFIED.** Killing the per-token `.item()` host-sync is ~free (48→50): against ~955 sequential kernel launches/token the single sync is negligible — the confound is launch *dispatch*, not the sync | fuse the launches (`torch.compile`) — that's the real lever, not the sync |
| 2026-07-01 | A1 R1 · strip B: compiled[default] (inductor fusion) | RTX PRO 4000 Blackwell (sm120) | decode tok/s @B=1 | 150–300 · 250–500 GB/s (pred B) | **173** (53% roof · **291 GB/s · 53% HBM**) | overhead→memory | **Pred B CONFIRMED.** Inductor fuses the ~955 pointwise launches → achieved BW climbs 81→291 GB/s (15%→53% of the wall). **Memory-bound thesis demonstrated in trend** (3 pts on a line: strip launches → BW rises). Residual gap = the ~180 GEMV/matmul launches inductor still issues/token | **R4.4 CUDA-graph decode** to collapse the residual launches → predict climb toward the 327 tok/s ceiling / 550 GB/s |
| 2026-07-01 | A1 R1 · compiled[reduce-overhead] (cudagraphs) | RTX PRO 4000 Blackwell (sm120) | graph capture | expected to help | **capture FAILED** | — | `RuntimeError: accessing tensor output of CUDAGraphs overwritten by a subsequent run` — the `torch.cat`-grown KV cache (dynamic addresses) violates cudagraph static-memory capture. **Scopes R4.4 precisely:** need a static pre-allocated KV buffer with in-place writes (the same primitive PagedAttention R4.1 needs) | R4.4: rewrite `KVCache` as a fixed `[B, kv_heads, max_ctx, head_dim]` buffer, write at `length`, slice `[:length]` — then re-attempt cudagraph capture |

**Verdict (A1 R1 close-out):** the founding thesis — *decode is memory-bound; eager launch overhead hides
the wall* — is now **demonstrated in trend** on the standing GPU: stripping launches moves achieved HBM
15% → 53%, on a straight line. It is **not yet closed to the DoD bar** ("within 15% of ceiling / Nsight
SoL Memory%≫Compute%") — the last ~47% to the wall is the residual per-token launch overhead that only
static-graph capture removes, and that is blocked by the `cat`-cache. Next node **R4.4** (static KV
buffer → CUDA graphs) is now the pre-registered continuation, with the cudagraph failure as its spec.
