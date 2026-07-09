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

---

## Main track (CS336 A2→A5) — analysis results

> Additive section for the main-track front ([ADR-0014](../docs/adr/ADR-0014-cs336-main-track-delivery-sprint.md);
> spec `../docs/EXECUTION_SPEC_CS336_FINISH.md`). CPU-analysis fits go here — same predict-before-run
> discipline, no GPU profile required.

### W5 · A3 — IsoFLOP / Chinchilla fit (2026-07-03, `scripts/a3_isoflop.py` → `bench/a3_isoflop.png`)

Fit of the course dataset `lectures/assignment3-scaling/data/isoflops_curves.json` (9 budgets, 72 runs;
per-budget argmin loss → log-log linfit, both exponents free). Deterministic: two runs give identical
stdout and identical PNG. Tests: `tests/test_scaling.py` 8/8 green.

| date | node / artifact | metric | predicted (pre-registered) | measured | note |
|---|---|---|---|---|---|
| 2026-07-03 | W5 · A3 IsoFLOP fit | scaling exponents a, b | a ≈ b ≈ 0.5; a+b ∈ [0.95, 1.05] | **N_opt = 1.1634·C^0.4687 · D_opt = 0.1433·C^0.5313 · a+b = 1.0000 → gate PASS** | a+b = 1 holds exactly (1e-9): D is derived from the same (C, N) rows via C = 6ND, so the identity is structural under a linear log-log fit |
| 2026-07-03 | W5 · A3 IsoFLOP predictions | N_opt, D_opt @ 1e23 / 1e24 FLOPs | — (extrapolation output) | **N_opt ≈ 7.01e10 @1e23 · 2.06e11 @1e24; D_opt ≈ 2.38e11 @1e23 · 8.09e11 @1e24** | extrapolation factor **×333** past the largest fitted budget (3e21) — stated on the plot's shaded region |

**The two one-sentence answers (script output, `[FACT]` as fits of the course data):**
(a) The compute-optimal model size is **N_opt ≈ 7.01e10 params (~70B)** at 1e23 FLOPs and
**≈ 2.06e11 (~206B)** at 1e24 (shape via `propose_shape`: n_layer=71/d_model=9068 and
n_layer=102/d_model=12977 at aspect ratio 128).
(b) The compute-optimal data budget is **D_opt = C/(6·N_opt) ≈ 2.38e11 tokens (~238B)** at 1e23 FLOPs
and **≈ 8.09e11 (~809B)** at 1e24.

**Honesty note `[INFERENCE]`:** both 1e23/1e24 predictions extrapolate **×33–×333** beyond the largest
fitted budget (3e21) — the fit is exact on the data, but the answers inherit the power-law-holds
assumption. Artifacts: `bench/a3_isoflop.png` (data + both fit lines + shaded extrapolation region with
the factor) · `scripts/a3_isoflop.py` (deterministic reproduction) · `src/scratch_llm/scaling/isoflop.py`.

### W8c · A5 — GRPO / Dr.GRPO engine, toy end-to-end validation (2026-07-04, `tests/test_grpo_algos.py`)

Engine-validation run (guide §5 framing): `grpo_train_loop` with the Dr.GRPO defaults
(`normalize_by_std=False`, `length_normalization="constant"`) on a learnable single-token
Countdown-style env (`_N_TASKS=3`, answer = one token) + a tiny A1 `TransformerLM`
(`d_model=32, n_layers=2, vocab=11`) and a temperature=1.2 sampler, 30 CPU steps, `group_size=12`,
AdamW lr=0.05, seed 0. Deterministic (identical on re-run). Proves the loop composes
(rollout → grade → group-normalize advantage → microbatch update → mandatory logging) and *learns* —
nothing about a real model (the real-Countdown "aha" is the GPU capstone, `A5_countdown_aha.md`).

| date | node / artifact | metric | predicted (pre-registered) | measured | note |
|---|---|---|---|---|---|
| 2026-07-04 | W8c · GRPO toy engine | mean reward over the run | reward **strictly rises** (loop learns) | **E[r] 0.186 → 0.666 · sampled mean_reward 0.194 → 0.667** (learns 2 of 3 tasks) | endpoints; deterministic seed 0; ~8 s CPU |
| 2026-07-04 | W8c · GRPO toy engine | response-token entropy | falls (policy sharpens as it learns) | **0.543 → 0.005** | the entropy-collapse monitor confirms real learning, not noise |
| 2026-07-04 | W8c · GRPO plateau | why it stops at 2/3 | group-variance plateau (all-same-reward group → 0 advantage) | **3rd task's groups go all-wrong → A=0 → no gradient** | the honest GRPO limitation the run exhibits; `[INFERENCE]` from the trace |

**Predicted-before-run holds `[FACT]` (measured on the toy):** the Dr.GRPO advantage on the correct
rollout `(1 − group_mean) > 0` drives `p(answer)` up; the exact (noise-free) expected reward rises
0.186 → 0.666 while its Monte-Carlo estimate (the sampled `mean_reward`) tracks it 0.194 → 0.667. The
loop plateaus at 2/3 because the third task's groups become all-wrong (uniform reward → zero
advantage → zero gradient) — the group-variance requirement GRPO can't escape without exploration.
Also ledgered: the real byte-level `CountdownEnv` integration smoke runs the loop end-to-end in
~0.6 s with response-token entropy ≈ `log 256` at init (loss-at-init sanity), reward ~0 (random tiny
model, expected). Artifacts: `src/scratch_llm/algos/grpo.py` · `tests/test_grpo_algos.py` (10/10
green, 12.5 s) · `docs/adr/ADR-0017-a5-drgrpo-default-and-aggregation.md`.

---

## Perf track (A1 R4.2 — chunked prefill)

### Pre-registration — A1 R4.2 (predict-before-run, D5)

Spec: `performance/notes/A1_R42_chunked_prefill.md`. Standing GPU sm_120, `RUNG1_CONFIG` (~1B bf16),
heavy-tail trace, N_SLOTS=32, compiled decode; baseline chunk=∞ = the R4.1 paged-kernel arm
(ITL p50 ~6 ms, p99 ~29 ms). Oracle = R4.1/R3b greedy (token-exact).

| # | experiment | predicted | measured | bound | note |
|---|---|---|---|---|---|
| P4.2.1 | token-exactness, chunk ∈ {∞,128,64,16,1} | bit-identical to R4.1 & `generate` | — (pending) | — | RoPE absolute-pos ⇒ chunked KV bit-identical |
| P4.2.2 | ITL p99, C: ∞→16 | falls ≥2× (29→≤14 ms), monotone | — (pending) | scheduling | per-gap prefill capped at C tokens |
| P4.2.3 | ITL p50, C: ∞→16 | rises modestly (~6→8–11 ms) | — (pending) | scheduling | every gap carries a chunk |
| P4.2.4 | TTFT p50, C: ∞→16 | rises (⌈L/C⌉ chunk-gaps) | — (pending) | scheduling | latency-of-first-token tradeoff |
| P4.2.5 | goodput @ ITL-SLO p99≤15 ms, chunk vs no-chunk | chunking ≥1.3× (or killed) | — (pending) | scheduling | spike violates SLO for a decode burst |

**Kill line(s):** any token divergence ⇒ offset/causal-mask bug, fix before measuring · ITL p99 flat
as C↓ (mechanism correct) ⇒ chunk forward not bounded, profile it · p50 super-linear as C→1 ⇒
per-iteration fixed-overhead floor (document, don't gold-plate).

### Measured — A1 R4.2 chunked prefill (2026-07-04, `bench/chunked_prefill.py` + `tests/test_chunked_prefill.py` 44/44 green)

Standing GPU RTX PRO 4000 Blackwell (sm120), `RUNG1_CONFIG` 0.84B bf16, compiled decode + eager
prefill, B=32, 160-req trace (26 × 512-tok prompts interspersed among 32-tok prompts). Clocks NOT
locked (`nvidia-smi -lgc` blocked in this unprivileged container — same as `ncu` counters); all arms
measured back-to-back so the RELATIVE curve is robust to clock drift. unique_graphs=1.

| date | rung | hardware | metric | predicted | measured | bound | root cause (1 line) | next |
|---|---|---|---|---|---|---|---|---|
| 2026-07-04 | R4.2 · token-exactness (P4.2.1) | sm120 | greedy divergence | bit-identical | **token-exact** (single-chunk KV bit-identical `torch.equal`; float64 exact all chunk sizes; float32 divergences are argmax tie-flips, not logic) | — | RoPE absolute-pos ⇒ chunked KV algebraically identical to one-shot | ✓ correctness gate MET |
| 2026-07-04 | R4.2 · ITL p99 vs C (P4.2.2) | sm120 | ITL p99 ratio, C ∞→32 | **falls ≥2×** | **ROSE ×1.03→×1.91** (198.8→204/284/294/380/304 ms at C=512/256/128/64/32) — **FALSIFIED** | scheduling→overhead | chunking did NOT reduce the spike | R4.2b (piggyback) |
| 2026-07-04 | R4.2 · ITL p50 vs C (P4.2.3) | sm120 | ITL p50 ratio | rises modestly (~6→8-11 ms) | **ROSE ×6.56→×14.44** (12.6→82.6…181.9 ms) — **FALSIFIED (magnitude)** | overhead | un-piggybacked chunk-forward cost lands in every decode gap (up even at C=512=single-chunk) | R4.2b |
| 2026-07-04 | R4.2 · throughput vs C | sm120 | agg tok/s | (not pre-reg) | **547→366→319→251→201→145** monotone DOWN as C↓ | overhead | serialized admission starves decode: **14 batched prefills (∞) → 160–550 unbatched (chunked)** | R4.2b |
| 2026-07-04 | R4.2 · TTFT p50 (P4.2.4) | sm120 | TTFT ratio | rises | **ROSE ×1.46→×4.05** (dir. met, magnitude large; saturated trace ⇒ queue-dominated) | latency | reserved-but-prefilling slots + serialized admission inflate queue wait | R4.2b + shallow trace |
| 2026-07-04 | R4.2 · goodput@SLO (P4.2.5) | sm120 | goodput @ ITL≤25.2 ms | ≥1.3× vs one-shot | **0 tok/s ALL arms (incl. one-shot)** — not measurable on the saturated B=32 trace | — | saturated trace's ITL exceeds the SLO for every arm; needs a Poisson/shallow trace | R4.2b |

**Verdict (A1 R4.2 — mechanism SHIPPED & token-exact; spike-reduction FALSIFIED for the
sequential-interleave design, diagnosed).** The chunked-prefill *mechanism* is correct and lossless:
`ChunkPrefillView` writes KV bit-identical to a one-shot prefill (RoPE rotates each token at its
absolute position), proven by 44 tests (single-chunk `torch.equal`; all chunk sizes token-exact in
float64; the float32 divergences are greedy-argmax tie-flips from batched-inference reduction-order
non-associativity — the same class the R3b path already has vs `generate`, removed entirely by
float64). `[FACT]` P4.2.1.

But the pre-registered latency win is **FALSIFIED**: this sequential-interleave scheduler
*regresses* every serving metric (ITL p50 ×6.6–14.4, ITL p99 did NOT fall, throughput 547→145
tok/s). Root cause, diagnosed from the ledgered `prefills`/`steps` columns `[FACT]`: (1) **serialized
admission** — the scheduler advances ONE prefilling slot per iteration, so 160 requests become
160–550 *unbatched* prefill forwards vs one-shot's **14 batched** (11–13 admits/forward), starving
the decode batch (util↓ → 475→644 decode steps for the same tokens); (2) **the chunk is a separate
sequential forward**, so its full cost lands in every decode gap — ITL p50 is 6.6× worse even at
C=512 (single chunk == one-shot prefill), isolating this as pure per-forward overhead, not slicing.

**The lesson (principal-level systems judgment):** chunked prefill's benefit is a KERNEL/BATCHING
property, not a scheduling-only one. The production technique (Sarathi-Serve / vLLM-V1 "stall-free
batching") **piggybacks the prefill chunk INTO the batched decode forward** — one fused ragged
prefill+decode kernel — so the chunk adds compute but no extra launch and no idle slots. A scheduler
that pays a separate sequential forward per chunk loses more than the spike it removes. This is
exactly WHY real engines implement it as a fused batched kernel. **Scoped as R4.2b** (deferred): a
fused mixed-query-length prefill+decode kernel (natural extension of the R4.1 paged Triton kernel to
rows with query-len C alongside query-len 1) + batched (not serialized) chunk admission + a
Poisson/shallow-arrival trace to isolate the spike from queue saturation. R4.2b is NOT a prerequisite
for R4.3–R4.6 (spec decode, CUDA graphs, MLA, disagg are independent), so the node advances to R4.3;
R4.2b returns with the paged-kernel work.

---

## Perf track (A1 R4.3 — speculative decoding, lossless)

### Pre-registration — A1 R4.3 (predict-before-run, D5)

Spec: `performance/notes/A1_R43_speculative_decoding.md`. Standing GPU sm_120, `RUNG1_CONFIG`
(~1B bf16), single-stream, drafter = prompt-lookup (n-gram, training-free), K=4. Oracle =
`sampling.generate` greedy (temperature 0).

| # | experiment | predicted | measured | bound | note |
|---|---|---|---|---|---|
| P4.3.1 | greedy losslessness | spec == generate, token-exact, every prompt | — (pending) | — | correctness gate; kill: any mismatch = rejection/rollback bug |
| P4.3.2 | acceptance, repetitive prompt | 40–75% mean accept fraction | — (pending) | — | n-gram hits on structured/repeated text |
| P4.3.3 | acceptance, random prompt | near 0 | — (pending) | — | no n-gram structure ⇒ drafts miss |
| P4.3.4 | speedup, repetitive prompt | 1.3–2.0× tok/s vs greedy (or killed) | — (pending) | latency | E[accept+1]/(1+K·c_draft/c_target); n-gram draft ~free |
| P4.3.5 | speedup, random prompt | <1× net LOSS | — (pending) | overhead | zero acceptance + K+1-wide verify > 1-wide decode |

**Kill line(s):** losslessness fails once ⇒ fix rejection/rollback before any speedup number ·
acceptance >0 but speedup <1 on repetitive ⇒ verify forward not amortizing (profile) · prompt-lookup
acceptance ~0 on hand-crafted repetition ⇒ n-gram match/propose bug.

### Measured — A1 R4.3 speculative decoding (2026-07-04, `bench/speculative.py` + `tests/test_speculative.py` 27/27 green)

Standing GPU sm120, `RUNG1_CONFIG` 0.84B bf16, eager target (apples-to-apples: the ratio isolates
speculation, not compilation), drafter = n-gram(n=3), max_new=128, median of 7 (clocks unlocked —
eager wall is noisy; median + the exact count-based tok/target-forward are the robust metrics).

| date | rung | hardware | metric | predicted | measured | bound | root cause (1 line) | next |
|---|---|---|---|---|---|---|---|---|
| 2026-07-04 | R4.3 · losslessness (P4.3.1) | sm120 | greedy divergence | token-exact | **token-exact** (27 tests: float64 exact all drafters/K; wrong-drafter still exact ⇒ KV rollback clean; float32 ≥99% agree) | — | pending-token invariant + KVCache.truncate roll back rejected drafts | ✓ correctness gate MET |
| 2026-07-04 | R4.3 · acceptance repetitive (P4.3.2) | sm120 | mean accept frac | 40–75% | **43–63%** (K=2/4/8: 62.7/57.8/43.0%) | — | n-gram hits on the (degenerate) greedy tail | ✓ MET |
| 2026-07-04 | R4.3 · acceptance random (P4.3.3) | sm120 | mean accept frac | near 0 | **54–72%** (K=2/4/8: 72.1/58.9/54.2%) — **FALSIFIED** | — | acceptance tracks the MODEL's low output entropy (untrained ⇒ degenerate-repetitive greedy), not the prompt | note: trained model would show the predicted prompt/domain dependence |
| 2026-07-04 | R4.3 · speedup repetitive (P4.3.4) | sm120 | tok/s vs greedy | 1.3–2.0× | **×1.21–1.29 wall** (61.6→77.4/79.4/74.4); **1.33–1.39 tok/target-forward** | latency | zero-cost n-gram draft ⇒ speedup ≈ E[accept+1]; K+1-wide verify ~1-wide cost | lower band met |
| 2026-07-04 | R4.3 · speedup random (P4.3.5) | sm120 | tok/s vs greedy | <1× LOSS | **×1.35–1.39 WIN** (57.2→77.0/77.9/79.4); 1.32–1.54 tok/fwd — **FALSIFIED** | latency | same degenerate-model cause: high acceptance ⇒ net win, not the predicted loss | — |

**Verdict (A1 R4.3 — SHIPPED, lossless, modest real speedup; two predictions falsified for one honest
reason).** Speculative decoding is **provably lossless** on this substrate `[FACT]` P4.3.1: with a
greedy target the output is token-identical to `sampling.generate` (float64-exact across every drafter
and K; a *deliberately-wrong* drafter still yields the exact greedy sequence, proving the rejection +
`KVCache.truncate` rollback discard rejected drafts cleanly; float32 ≥99% agreement, the residual
being batched-vs-sequential argmax tie-flips). The engine is drafter-agnostic (`Drafter` protocol:
n-gram prompt-lookup measured, model-drafter tested).

Measured speedup `[FACT]`: **~1.2–1.4× wall / 1.3–1.5 tokens per target forward** via *zero-cost*
n-gram drafting — a real but modest win (each K+1-wide verify forward commits 1.3–1.5 tokens instead
of 1). **Two predictions FALSIFIED, both from one cause:** the untrained 0.84B model's greedy output
is **degenerate-repetitive (low entropy)**, so the n-gram drafter hits ~54–72% even on a RANDOM
prompt — acceptance here reflects the MODEL's output structure, not the prompt's (P4.3.3), and the
predicted random-prompt *loss* (P4.3.5) is instead a ×1.35–1.39 *win*. On a trained model the
acceptance would be prompt/domain-dependent as originally predicted (structured text > open-ended),
and the drafter-family research (Medusa / EAGLE-2/3 feature-level trees / MTP ~85–90% 2nd-token) is
entirely about raising E[accept] — the term that dominates the speedup formula (note §mechanism
literacy). Node advances to **R4.4 CUDA-graph decode** (the paged Triton kernel's fixed `(B,H)` grid +
fixed-address pool is the capturable substrate; closes the R1 eager→wall launch-overhead gap).

---

## Perf track (A1 R4.4 — CUDA-graph decode)

### Measured — A1 R4.4 (2026-07-04, `bench/cudagraph_decode.py` + `tests/test_cudagraph_decode.py` gpu 4/4)

Pre-registration (from the R1 close-out forward-pointer): B=1 step-time reduction **~20–28%** (vLLM-V1),
bound = launch overhead; closes the R1 gap (eager 51 → compiled 173 tok/s = 53% of the 327 tok/s wall).
Substrate = the R4.1 paged Triton kernel (fixed `(B,H)` grid + fixed-address pool). Both arms use the
SAME paged kernel — the delta isolates the CUDA graph. sm120 0.84B bf16, prompt=32, decode=48, median.

| date | rung | hardware | metric | predicted | measured | bound | root cause (1 line) | next |
|---|---|---|---|---|---|---|---|---|
| 2026-07-04 | R4.4 · B=1 step-time | sm120 | ms/step reduction | −20–28% | **−74.3%** (15.38→3.96 ms) — pred exceeded | overhead→memory | eager ≈54–68 host launches/step (nsys) → 1 cudaGraphLaunch | ✓ closes R1 gap |
| 2026-07-04 | R4.4 · B=1 tok/s | sm120 | decode tok/s | climb toward 327 ceiling | **253 tok/s = 77% of wall** (eager 65 = 20%) | memory | launches removed ⇒ the memory wall is finally the bound (vs 53% compiled) | — |
| 2026-07-04 | R4.4 · B=8 / B=32 | sm120 | ms/step reduction | — | **−71.3% / −68.4%** (agg 1749 / 6307 tok/s) | overhead→memory | graph win shrinks slightly as B raises compute:launch ratio | — |
| 2026-07-04 | R4.4 · token-exactness | sm120 | graph vs eager | identical | **token-exact** (gpu test B∈{1,4,8}, across 16-block boundaries) | — | capture changes launch, not math | ✓ oracle MET |

**Verdict (A1 R4.4 — SHIPPED; prediction FALSIFIED in the good direction, R1 wall gap CLOSED).** CUDA-graph
decode over the fixed-address paged pool is token-exact `[FACT]` and cuts B=1 step time **−74.3%**
(15.38→3.96 ms), far past the pre-registered −20–28% — because our eager baseline is far more
launch-overhead-bound than vLLM-V1's optimized H100 path: nsys shows ~54–68 `cudaLaunchKernel`/step
collapsing to **1 `cudaGraphLaunch`/step** `[FACT]`, and step time fell ~4× at B=1 where compute is
negligible ⇒ the eager time was almost all CPU launch dispatch. **B=1 reaches 253 tok/s = 77% of the
327 tok/s memory wall** — the R1 thesis (decode is memory-bound; launch overhead hides the wall) is now
closed three ways: eager 20% → compiled 53% → **cudagraph 77%**. `torch.compile` reduce-overhead
REFUSES this path (the in-place `lengths += active` is a "mutated input"); manual capture owns the
mutation. Clocks unlocked but the ~4× ratio dwarfs ±15% drift. Node advances to **R4.5 (MLA toy)** →
R4.6 (PD-disagg). Deferred extension: continuous-batching + cudagraph (graph pool per batch size).

---

## Perf track (A1 R4.5 — MLA latent cache, toy)

### Measured — A1 R4.5 (2026-07-04, `tests/test_mla.py` 5/5, `src/scratch_llm/mla.py`)

Toy MLA (no training); oracle = full K/V reconstruction. Gate = weight-absorption identity + KV
reduction. Spec/note: `performance/notes/A1_R45_mla_latent_cache.md`.

| date | rung | hardware | metric | predicted | measured | bound | root cause | next |
|---|---|---|---|---|---|---|---|---|
| 2026-07-04 | R4.5 · absorption identity | CPU/sm120 | max\|naive−absorbed\| | numerically identical | **1.4e-15** (float64 = eps; float32 <1e-4) | — | (W_UK^T q_c)·c_KV == q_c·(W_UK c_KV) is pure algebra; decoupled RoPE keeps content-K position-free | ✓ gate MET |
| 2026-07-04 | R4.5 · KV/token (R1-like) | — | MLA vs MHA/GQA bytes | MLA ≈ 4–14% of MHA | **MLA 1152 B = 1.8% of MHA (65536 B); 3.56× < GQA-8 (4096 B)** | — | caches (d_latent+d_rope) not 2·n_heads·d_head | ✓ (cross-model 4.7× vs Llama-70B = differing L/dims) |

**Verdict (A1 R4.5 — SHIPPED).** MLA's weight-absorption identity holds to machine precision `[FACT]`:
attending in latent space (W_UK folded into the query, W_UV into the output) equals reconstruct-then-
attend, so an engine caches the low-rank latent `c_KV`+`k_R` instead of per-head K,V at **zero quality
change** — a 3.6× (vs GQA-8) / ~55× (vs MHA-128) per-layer cache reduction. The decoupled-RoPE
pathway is load-bearing: it keeps the content-K position-independent so W_UK is a *static* fold.
Bridges to A5 (FP8 latent) + the Phase-4 serving day (P5 MLA-KV audit).

---

## Perf track (A1 R4.6 — prefill/decode disaggregation, demonstrate)

### Measured — A1 R4.6 (2026-07-04, `bench/disagg.py`)

Single-GPU demonstration (the two measurable halves; no two real workers). sm120 0.84B bf16,
N_SLOTS=32. Spec/note: `performance/notes/A1_R46_disaggregation.md`.

| date | rung | hardware | metric | predicted | measured | bound | root cause | next |
|---|---|---|---|---|---|---|---|---|
| 2026-07-04 | R4.6 · KV-transfer tax | sm120 | D2D copy of 16.8 MB KV (512-tok) | ~bytes/HBM_BW | **0.180 ms (187 GB/s; analytic 0.061)** one-time | overhead | 32 small per-layer copies ⇒ launch-bound below HBM peak | fused copy / NVLink-RDMA at node (P6) |
| 2026-07-04 | R4.6 · ITL p99 disagg vs co-located | sm120 | decode-worker ITL p99 | disagg ≥2× better | **20.2 vs 61.5 ms = 3.0× better** | scheduling | co-located prefill bursts steal decode cycles; disagg keeps the stream clean | — |
| 2026-07-04 | R4.6 · goodput @ ITL SLO p99≤30ms | sm120 | meets SLO? | disagg wins | **disagg MEETS (20.2), co-located VIOLATES (61.5); agg 1707 vs 1038 tok/s** | scheduling | the spike violates the SLO for a burst of decodes | — |

**Verdict (A1 R4.6 — SHIPPED, demonstrated; A1 serving rungs R0–R4.6 COMPLETE).** Disaggregation
trades a recurring ITL p99 spike (3×: 61.5→20.2 ms) for a negligible one-time KV transfer
(0.18 ms ≪ ~40 ms spike removed), and the decode worker runs at higher throughput (1707 vs 1038
tok/s). Goodput under an ITL SLO strictly favours disagg. `[FACT]` Honest scope: single-GPU
simulation — the disagg arm is a clean decode-only stream; a real 2-GPU disagg pays the cross-device
KV-transfer bandwidth (D2D here as the proxy; NVLink/RDMA at node scale = Phase-4 P6). **A1 closed** →
A1 design note next, then A2 (CUDA-core kernel ladder).

---

## Perf track (A2 — CUDA-core kernel ladder)

### Measured — A2 R0 profiler + roofline harness (2026-07-04, `bench/kernel_roofline.py`)

The reusable A2 measurement spine. Gate = reproduce the R0 baseline. ncu BLOCKED
(ERR_NVGPUCTRPERM) → re-based on achieved-vs-peak % + nsys; 5 ncu metrics registered as debt for the
H100 day. Spec/note: `performance/notes/A2_R0_roofline_harness.md`.

| date | rung | hardware | metric | predicted | measured | bound | root cause | next |
|---|---|---|---|---|---|---|---|---|
| 2026-07-04 | A2 R0 · peaks reproduction | sm120 | HBM BW / bf16 TF/s / ridge | ~0.55 TB/s / ~72 TF/s / ~130 | **0.551 TB/s · 72.1 TF/s · ridge 131** — gate PASS | — | reproduces 2026-06-29 baseline within noise | R1 GEMV ladder |
| 2026-07-04 | A2 R0 · harness anchors | sm120 | %roof of copy / gemm | 100% each | **copy 551 GB/s = 100.1% mem · gemm 72.1 TF/s = 100.0% cmp** (spread <1.2%) | mem / cmp | validates the profiler places kernels correctly on both roofs | R1 |

**Verdict (A2 R0 — SHIPPED).** The profiler+roofline harness reproduces the baseline `[FACT]` and
places kernels on the roofline (%-of-binding-roof + mem/cmp) with CSV + plot; ncu counters (blocked
on this unprivileged box) are re-based on achieved-vs-peak % + nsys, with 5 metrics registered as
"ncu debt" for the H100 rental day. Next: A2 R1 GEMV ladder (naive→coalesced→two-stage→float4, target
>80% of 0.55 TB/s) → R2 softmax → R3 RMSNorm → R4 TopK → R5/R6 GEMM (sm120); §4.1–4.6 WGMMA/TMA/FP8
are H100-code-only.

### Measured — A2 R1–R6 CUDA-core kernel ladder (2026-07-04, Triton, workflow-built + adversarially verified + main-thread gpu-tested)

sm120, measured HBM peak 0.551 TB/s / bf16 compute 72 TF/s (R0 harness). Every rung: oracle vs torch
at spec tolerance (79 gpu tests green under my own run) + an independent adversarial verifier
(re-ran oracle, added own adversarial inputs, checked tolerance not gamed, measurement honest). ncu
blocked → %-of-measured-peak + nsys; ncu-debt registered per kernel.

| date | rung | metric | predicted | measured | bound | note |
|---|---|---|---|---|---|---|
| 2026-07-04 | A2 R1 GEMV (naive→blockrow→split) | %HBM at 8192² | >80% of 0.55 TB/s | **blockrow 528.8 GB/s = 95.9% HBM** (107% of torch.mv); naive 78% | mem | coalesced block-per-row saturates HBM; float4/swizzle are Triton-compiler-managed (honest) |
| 2026-07-04 | A2 R2 softmax (twopass→online→fused) | ~peak HBM + online<twopass bytes | **fused ~100% HBM (551 GB/s)**; online 3N vs twopass 4N = **1.33× fewer bytes, 1.17× faster >L2** | mem | Milakov online recurrence; adversarial +1e4/all-eq/all-−inf pass (no NaN) |
| 2026-07-04 | A2 R3 RMSNorm/LayerNorm | ~peak HBM; RMS<LN | **both ~100–101% HBM @N≥4096**; RMS vs LN within ±1% (small-N edge = noise, honest) | mem | one reduction (RMS) vs two (LN); zero-row ε path NaN-free |
| 2026-07-04 | A2 R4 TopK + fused softmax+topk | measure the failure | **iter-max 258 GB/s = 46.9% peak** (the honest poor-GPU-fit); **fusion 3.39× faster / 3.00× less traffic** | mem/occupancy | serial dependent reductions, ~0 AI; bit-exact vs torch.topk, tie-break lowest-index |
| 2026-07-04 | A2 R5+R6 GEMM (naive→tiled→autotuned) | siboehm shape 1%→~78% | **0.2% → 128.5% → 134.3% of cuBLAS-proxy** (naive 0.2 → autotuned 101.9 TF/s) | compute | tl.dot tiled+autotuned beats torch.matmul default at this shape (%roof>100 disclosed); AI 1365, ridge 134 |

**Verdict (A2 R1–R6 — SHIPPED).** Five CUDA-core kernels, each oracle-correct (79 gpu tests, adversarial
inputs incl. non-contiguous/odd-prime/outlier) and roofline-placed: the four memory-bound kernels
(GEMV, softmax, both norms) reach **~96–100% of measured HBM peak** — the memory wall IS the bound and
they hit it; TopK is honestly **46.9%** (a poor GPU fit: serial reductions, ~0 AI — the spec's "measure
the failure"), redeemed by a **3.4× fused-softmax+topk** traffic win; GEMM crosses the ridge
(compute-bound, AI 1365) with the naive→tiled→autotuned ladder reproducing the siboehm SHAPE
(0.2%→134% of the torch.matmul/cuBLAS proxy — Triton's autotuned `tl.dot` beats the default cuBLAS
heuristic at 4096³, honestly disclosed). Honesty (ADR-0011 Triton-primary): float4 vectorization,
coalescing, and bank-conflict avoidance are Triton-compiler-managed — the raw-CUDA ladder step is noted
per kernel as what a hand kernel would add, not fabricated. ncu counters blocked → each kernel names its
ncu-debt metric for the H100 day. Node → A2 §4 (H100 code-only) + A3.

---

## Perf track (A5 — quantization numerics ladder)

### Measured — A5 R0–R4 (2026-07-04, CPU numerics, workflow-built + adversarially verified, 43 CPU tests)

Fake-quant numerics; oracle = SQNR/MSE/bit-exact (NOT allclose for FP4/FP8). Each rung independently
adversarially verified (re-ran oracle, recomputed the headline from scratch, checked no scale/clamp/
rounding was target-fitted). Runs in the `not gpu` CI gate. Module: `src/scratch_llm/quant/`.

| date | rung | metric | predicted | measured | note |
|---|---|---|---|---|---|
| 2026-07-04 | A5 R0 INT8 sym per-tensor | round-trip SQNR | ~44 dB floor | **40.50 dB** Gaussian; book example [1.2,−0.8,2.5,−1.7]→codes [61,−41,127,−86] reproduced; law verified (2× scale → −6 dB) | 6.75 dB/bit measured (ideal 6.02) |
| 2026-07-04 | A5 R1 INT8 asym + per-channel | asym>sym, per-ch>per-tensor | asym > sym; per-ch > per-tensor | **asym 43.50 > sym 37.44 (+6.07 dB)** on skewed data; **per-ch 42.34 > per-tensor 33.25 (+9.09)**; W8A8 int-GEMM rel 1e-2 | scale hoisted out of the GEMM (dequant accumulator once) |
| 2026-07-04 | A5 R2 group INT4 (g=128) | bit-exact pack/unpack | bit-exact | **pack/unpack bit-exact** (500-case fuzz + 0x78/0xFF adversarial); **SQNR 18.64 vs analytic 18.60 floor**; group 18.69 > per-tensor 17.12 | 2 nibbles/byte, sign-extend unpack |
| 2026-07-04 | A5 R3 NVFP4 vs MXFP4 | NVFP4 MSE < MXFP4 | NVFP4 < MXFP4 | **NVFP4 MSE 9.05e-3 (20.43 dB) < MXFP4 1.34e-2 (18.74 dB) = 1.48×**; block-scaled GEMM 0.28% vs bf16-same-W; **verifier confirmed the MXFP4 baseline is the STRONGER ceil-E8M0 variant** (not rigged) | E4M3 non-pow2 block scale + k=16 vs k=32 |
| 2026-07-04 | A5 R4 FP8-E4M3 KV | within noise of BF16 KV | FP8 near-lossless, INT4 degrades | **FP8 SQNR 31.81 dB, E2E 24.45 dB** vs BF16-KV; per-channel-K **2.49×** better than per-token (drops to 0.60× on outlier-free data — tracks structure); **INT4-KV visibly degrades (19.86 dB)**; bytes **0.552×** | real torch.float8_e4m3fn, clamp-before-cast (E4M3 NaN>448) |

**Verdict (A5 R0–R4 — SHIPPED).** The quantization ladder is numerically sound and honest: INT8
reproduces the ~44 dB-class floor (40.5 dB) and the 6 dB/bit law, asym beats sym / per-channel beats
per-tensor on skewed data by the predicted margins; group-INT4 pack/unpack is bit-exact under a
500-case fuzz and hits its analytic SQNR floor; **NVFP4 beats MXFP4 by MSE 1.48×** for the two named
mechanisms (E4M3 non-power-of-two block scale + finer k=16 blocks) — and the adversarial verifier
confirmed the MXFP4 baseline is the *stronger* variant, so the win is real not rigged; FP8 E4M3 KV is
near-lossless (E2E 24.45 dB) at 0.552× bytes with per-channel-K 2.49× better than per-token, while
INT4-KV visibly degrades (the stress test bites). §7 NVFP4-native-MMA throughput → B200 code-only.

---

## A4 flash-attention ladder (R0–R1 — the pedagogical baselines that motivate FA)

### Measured — A4 R0 naive 3-kernel attention + R1 online softmax (2026-07-04, workflow-built, 43 tests)

The *why* of flash attention, made falsifiable. R0 = naive attention as three separate ops
(S=QKᵀ/√d, P=softmax(S) causal, O=P@V) that materializes the full N×N score matrix; R1 = the
single-row online-softmax recurrence in isolation. Modules: `src/scratch_llm/kernels/attention_naive.py`,
`online_softmax.py`. Oracle-first: R0 vs `F.scaled_dot_product_attention` <1e-3 fp32; R1 vs a literal
3-pass reference softmax <1e-6 fp64. 42 CPU tests (`not gpu` gate) + 1 gpu-marked peak-memory test.

| date | rung | hardware | metric | predicted | measured | bound | root cause |
|---|---|---|---|---|---|---|---|
| 2026-07-04 | A4 R0 naive attention · O(N²) blowup | RTX PRO 4000 Blackwell (sm120) | peak CUDA mem vs N (d=16, 1 head) | ∝N² (score matrix dominates) | naive **8.1→32.1→128.2 MB** at N=1024→2048→4096 (exactly **4× per doubling** = N²); fused oracle **0.1→0.3 MB** (flat) → **>400× smaller** at N=4096 | memory | naive writes/re-reads the N×N score+prob matrices to HBM; the fused kernel never materializes N×N (O(N·d) output + O(tile²) SRAM scratch) |
| 2026-07-04 | A4 R0 · analytic 16K footprint | — (arithmetic) | score-matrix bytes vs on-chip scratch | 1 GiB, ≫ SRAM | N=16K fp32 1-head score matrix = **1.00 GiB**; = **65536×** the fused kernel's constant 16 KiB tile² on-chip scratch (247× its total working set incl. output) | memory | 1 GiB cannot live in the ~KBs of SRAM and never needs to — tiling holds one tile² block at a time |
| 2026-07-04 | A4 R1 online softmax · late-outlier rescale | CPU (fp64) | online == 3-pass softmax | bit-exact <1e-6 | matches 3-pass ref <1e-6 for all tile sizes; **ADVERSARIAL +50 outlier in the last tile passes** (the exp(m_old−m_new) denominator rescale is what makes this correct) | — | the numerical core FA fuses into the tile loop: one streaming (m,d) pass replaces the 3-pass max+sum |

**Verdict (A4 R0–R1 — SHIPPED).** The motivation is now falsifiable and measured: naive attention's
peak memory tracks the N×N score matrix (clean 4×-per-doubling quadratic on this sm120 card) while the
fused FA oracle stays flat (>400× smaller at N=4096); analytically a 16K score matrix is 1 GiB = 65536×
the fused kernel's constant on-chip scratch, so it *cannot* live on-chip and never needs to. The online
softmax recurrence reproduces a 3-pass reference to fp64 including the adversarial late-+50-outlier that
exercises the running-max denominator rescale — the exact `corr = exp(m − m_new)` line FA2 fuses into
its tile loop. ncu-debt: peak-memory via `torch.cuda.max_memory_allocated` (allocator-level, not ncu
DRAM counters — ncu blocked on this box); the quadratic signature is read from the increment ratios
(constant workspace offset cancels in Δ).

### Measured — A5 §4.3 AWQ PTQ (2026-07-04, `src/scratch_llm/quant/awq.py`, 7 CPU tests)

Real PTQ on an actual `Linear[256,512]` (weights ~N(0,0.02²), 8/512 activation-salient channels ×12),
group_size=128, IDENTICAL INT4 grid for both arms — only the scale placement differs.

| date | rung | metric | predicted | measured | note |
|---|---|---|---|---|---|
| 2026-07-04 | A5 §4.3 AWQ INT4 | AWQ output MSE < naive @ same 4-bit/g=128 | AWQ beats naive | **AWQ 5.47e-3 (20.90 dB) vs naive 9.03e-3 (18.72 dB) = 1.71× mean recovery** (min 1.65×, 5 seeds); **held-out 1.69×** (not calib-overfit); α=0≡naive bit-exact; interior optimum α≈0.25; salient-col err ↓2.87× | activation-aware per-channel scale `s=act_scale^α` grid-searched to min calib MSE; salient-by-activation weight cols scaled up before INT4 round, 1/s folded to activations |

**Verdict (A5 §4.3 — SHIPPED).** AWQ recovers **1.71×** the output MSE of naive round-to-nearest at the
same bit-width by protecting the ~1.5% of weight channels that carry the largest activations (scale them
up before rounding, fold 1/s into the activations) — the recovery holds on held-out tokens (1.69×, not
calibration-overfit) and the α grid has a genuine interior optimum (protect-vs-inflate tradeoff), α=0
reproducing naive bit-exactly. Honest scope: a layer-level MSE demonstration (a full-model perplexity
run is the rental-gated SKIP). Completes the A5 numerics track; native NVFP4-MMA throughput (§7) →
B200 (`B200_day_runbook.md`).

---

## Frontier ablations (close-the-loop front — ADR-0018, spec `docs/FRONTIER_2026_ABLATIONS.md`)

The EV-ranked, pre-registered, iso-FLOP ablation study on a **real trained** nanochat-grade base.
Predict-before-run (FOP-2/3): each rung's falsifiable number + kill criterion is registered here
*before* the run; `[FACT]` only once the measured column is filled. Node pointer: **F1**.

### Pre-registration — F1 MuonAdamW (2026-07-04, PENDING)

**Hypothesis.** Orthogonalizing the 2D-matrix momentum update (5-step Newton–Schulz, coeffs
3.4445/−4.7750/2.0315 in bf16) with Moonlight RMS-matching (`0.2·√max(A,B)`, WD 0.1) beats AdamW in
loss-per-FLOP while reusing AdamW's LR band. Muon on 2D block matrices only; the **weight-tied
embed/head tensor** (`model.py:917`) + all 1-D params stay on AdamW.

| rung | metric | predicted (pre-reg) | KILL if | measured |
|---|---|---|---|---|
| F1 NS orthogonality | σ spectrum of the update after 5 steps | all ∈ [0.7, 1.3] | any σ ∉ [0.5, 1.5] | **⚠ over-claim FALSIFIED → corrected.** NS *compresses* σ into a band ~[0.68,1.14], never inflates (σ_max<1.35), spread collapses (q90/q10<2 vs κ≫1 input). "all ∈[0.7,1.3]" + "median≈1" are false at 5 steps: a worst-case **square Gaussian** has near-0 σ (min≈0.08) the iteration can't lift, and median≈0.77. Inherent to few-step NS, **immaterial to Muon** (needs only the update *direction*). `tests/test_optim.py` asserts the corrected invariants. |
| F1 hybrid wiring | overfit-one-batch (Muon-matrices + AdamW-rest) | loss < 1e-2 in AdamW's step budget | fails to overfit | **PASS** — hybrid drives 1 batch to loss <0.05 in 300 steps (same budget/threshold as the AdamW-only oracle). |
| F1 param partition | tied embed/head + every 1-D param routed to AdamW; no overlap | exact partition, Σnumel matches | any 2-D tied tensor in Muon group | **PASS** (tied **and** untied): disjoint cover, Σnumel over unique params matches, the shared 2-D embed/head tensor + every RMSNorm(1-D) land in AdamW, block projections in Muon. |
| F1 iso-FLOP (30–50M) | val loss vs AdamW at fixed C=6ND | Muon reaches AdamW loss with ≥15% fewer tokens (or ≥0.02 nats lower @ iso-FLOP) | token saving <5% or divergence at reused AdamW LR | — pending (needs the training run — task 7 wiring + GPU) |
| F1 NS overhead | wall-clock of Newton–Schulz vs step | <1% (analytic bound T·m/B) | >3% | — pending (GPU) |

**Verdict (F1 unit level — 2026-07-04, `optim.py` + 10 tests green).** `Muon` (NS5 + Nesterov +
Moonlight RMS-match) and `split_muon_adamw_params` are built and green. Two pre-registered
over-claims on the NS spectrum were honestly falsified and the invariants corrected (band
compression + no inflation, not "all σ→1"); the corrected-math RMS-match (`0.2·√max(A,B)` ⇒ RMS≈0.2,
shape-independent) is asserted directly — the test *is* the check on the `1/√max` vs `1/max`
correction. The iso-FLOP loss-per-FLOP claim (the headline) remains pending the real run.

Sources: Keller Jordan (Muon writeup); Moonlight `2502.16982` (Lemma 1 — RMS = 1/√max(A,B),
corrected from the draft); Kimi-K2 `2507.20534` (MuonClip at trillion scale). Reuses
`src/scratch_llm/scaling/isoflop.py` to hold compute constant.

### Measured — F1/F4 train wiring (2026-07-04, `train.py` + 8 tests green, GPU-verified sm120)

`build_optimizer` + `CombinedOptimizer` wire the MuonAdamW hybrid + bf16 autocast + `torch.compile`
into `train()` (defaults = the A1 AdamW/fp32 path, unchanged). Verified END-TO-END on the standing
**RTX PRO 4000 Blackwell (sm120)** — a structured-corpus run (d_model 128 · 4 layers · 60 steps):

| path | result | note |
|---|---|---|
| muon_adamw · fp32 · eager | **loss 4.79 → 8.3e-4 (LEARNED)** | the hybrid learns; Muon verified correct on GPU |
| muon_adamw · **bf16** · eager | **loss → 9e-4 (LEARNED)** | F4 bf16 autocast works (no GradScaler needed) |
| muon_adamw · fp32 · **compile** | **loss → 8.3e-4 (LEARNED)** | `torch.compile` works alone |
| any · **bf16 · compile** | **NaN@~step5–10 → guard RAISES** | **`[FACT]`** box-specific: bf16-autocast **+** `torch.compile` NaNs on **sm120 / torch-2.12 inductor** — **reproduces with plain AdamW** (not Muon, not our logic); `matmul_precision="high"` doesn't help. Each works ALONE. bf16+compile is the **H100-rental** path. |

**F4 finding (honest, predict-vs-measure):** predicted bf16+compile = the cheap-MFU combo; measured
= it diverges on *this* inductor. The two features are individually correct + valuable (bf16 ✓,
compile ✓); their product is deferred to the H100 tier. A **loud NaN guard** (`train.py`, fires at
log cadence — no extra host sync) now fails any divergent run with a clear message instead of burning
compute on a silent NaN — the FRONTIER "silent-divergence triage" discipline, and a real hygiene win
independent of this bug. The F1 iso-FLOP Muon-vs-AdamW loss-per-FLOP measurement is the next rung.

### Measured — Phase 0: THE LOOP CLOSES (2026-07-04, `speedrun.py` + `eval/`, GPU-verified)

The end-to-end spine (`scripts/speedrun.sh` → `scratch_llm.speedrun`) ran **tokenizer → pretrain →
eval → sample** end-to-end and produced a model you can sample from — the artifact the whole front
exists to build (ADR-0018 §5, Phase 0). Nano run on the sm120 Blackwell (depth 4, byte-BPE vocab
384, ctx 64, 150 MuonAdamW steps, built-in corpus):

| stage | result |
|---|---|
| params / tokens / wall | 3.41M · 1,682 tok · **8.3 s** |
| eval report card | **val_bpb 0.0206** · 0.0655 nats/tok (in-sample repeated corpus — the metric computes) |
| sample (prompt "the quick brown…") | **coherent continuation** — *"fox jumps over the lazy dog. a language model learns to predict the next token… attention is all you need; the transformer reads the whole context at once. we own every layer from the byte"* |

**Verdict.** The component museum now runs the loop: BPE → MuonAdamW pretrain → `val_bpb` report card
→ a talking sample, one `--depth` knob (depth 20 ⇒ d_model 1280 / 10 heads / ~561M = the nanochat
d20 headline). The nano pre-flight (`--nano`, CPU, ~1 min) is the cheap gate before the $100 d20
rental. Report-card harness (`eval/`): `val_bpb` + MC (ARC/MMLU) + generative (GSM8K/HumanEval) +
CORE-style aggregate, 8 tests. Next: F1 iso-FLOP on the real loop, then F2 MTP.

### Pre-registration — A1 real-corpus shards (2026-07-09, MEASURED same day)

**Hypothesis.** A headerless memmap token-shard format (nanoGPT/nanochat convention: raw
little-endian uint16, one `<|eot|>` id after every document, `.meta.json` sidecar) feeds
`train.py::get_batch` unchanged and replaces the in-RAM toy corpus as the pretrain data path —
the piece every downstream rung (A2 chaining, F1-run, midtrain/SFT, the d20) consumes. Base spec
is nano-only; the ~11B-token shuffled multi-shard streamer is the §D d20 extension (deferred).

| rung | falsifier | predicted (pre-reg) | KILL if | measured |
|---|---|---|---|---|
| A1 round-trip | shard bytes → ids == `encode(d0)+[eot]+encode(d1)+[eot]` | byte-exact | any drift | **PASS** (`tests/test_shards.py`, byte-exact incl. sidecar meta) |
| A1 size | file bytes == `2·n_tokens`, zero header/padding | exact | format needs a header | **PASS** — headerless `.bin` == 2·n_tokens exactly |
| A1 alignment | `get_batch` on the loaded memmap: `targets` == stream shifted by one | exact, dtype int64 | misalignment or dtype break | **PASS** — every sampled window located uniquely in the stream, targets = shift-by-one, int64 via `.long()` (torch 2.12 consumes the uint16 memmap directly) |
| A1 dtype kill-switch | any token id ≥ 2¹⁶ | auto-switch to uint32, `4·n_tokens` bytes, meta records it | silent overflow/wraparound | **PASS** — ids straddling 2¹⁶ produce a uint32 shard, 4·n bytes, values exact |
| A1 FineWeb compression | bytes/token of a FineWeb-EDU slice under our slice-trained BPE | ∈ [3.0, 5.0] B/tok | < 2.5 (trainer under-merging) or > 6 | **3.804 B/tok** — 64 real docs (246,033 B) → 64,742-token uint16 shard, vocab 4096, BPE train 6.4 s |
| A1 real-corpus nano | trained-nano `val_bpb` vs random-init `val_bpb` on the shard tail | trained ≤ 0.7 × random-init | trained ≥ random-init (nothing learned from real text) | **ratio 0.467** — trained 1.393 vs random-init 2.979 bpb (5.31M params, 200 steps, CPU-Mac 594 s; sample = web-English-shaped babble, as it should be at this scale) |

**Verdict (A1 — 2026-07-09, `data/shards.py` + 10 tests green).** The shard substrate holds all four
format falsifiers and both real-corpus numbers, first try. Honesty caveats: (a) the nano `val_bpb`
tail is **in-sample** (the model trained on the whole shard — a held-out split arrives with the A2
stage spine; the trained-vs-random margin is still meaningful because both see the same tail); (b)
run on the **CPU Mac**, not the sm120 box — these are correctness/learning signals, not perf numbers.
`speedrun --data-dir <dir>` is now the real-corpus path; the ~11B-token shuffled streamer stays the
§D d20 extension. Next: **A2 checkpoint chaining**, then **F1-run**.

---

## Perf track (A4 — flash attention)

### Measured — A4 R0–R3 + backward + variant (2026-07-04, workflow-built + adversarially verified + gpu-tested)

sm120. R0/R1 new pedagogical rungs; R2/R3 MEASURE the existing `flash_attention_triton_forward` +
`FlashAttentionPyTorch` (built in the CS336 A2 work). Oracle = `F.scaled_dot_product_attention`.

| date | rung | metric | predicted | measured | note |
|---|---|---|---|---|---|
| 2026-07-04 | A4 R0 naive 3-kernel attention | O(N²) memory blowup | matches SDPA <1e-3; blows up | **matches SDPA 6.4e-7**; score matrix 1.0 GiB @16K = **65536× on-chip blowup**; peak-mem GROWTH is clean N² (increments ×3.96/×3.98 per doubling) — the motivation for FA | growth-ratio is the robust claim (absolute-peak ratio is allocator-state-dependent) |
| 2026-07-04 | A4 R1 single-row online softmax | match 3-pass, +50 outlier | <1e-6 | **0–7e-18 vs fp64 3-pass**; running-max = true row-max; +50-late-in-stream + ×1000-overflow-bait all match | the Milakov recurrence in isolation |
| 2026-07-04 | A4 R2/R3 FA2-Triton on sm120 | ~50–73% peak; no OOM | ~50% of SDPA | **50.0% (causal 48.3%) of SDPA @ seq4096**; FA peak 40–96 MB vs naive 105–4256 MB = **44× leaner @8K, no OOM**; causal speedup 1.11→1.74× | constant SMEM (the FA win vs R0) |
| 2026-07-04 | A4 backward + GQA variant | gradcheck; KV consequence | grads match | **backward gradcheck 5/5 vs SDPA autograd**; GQA KV 32→4 MB (n_kv 4 vs 32), runtime flat | the §4.3 variant + recomputation bwd |

**Verdict (A4 R0–R3 — SHIPPED).** The flash-attention ladder is complete and honest: the naive 3-kernel
baseline materializes the full N×N score matrix (1 GiB at 16K, 65536× the on-chip working set — the O(N²)
memory wall that motivates FA, with clean N² peak-memory GROWTH the robust claim vs the allocator-state-
dependent absolute ratio); the single-row online softmax matches a 3-pass reference to machine-fp64
precision through late outliers; the fused FA2 Triton kernel runs at **~50% of SDPA** on sm120 (the repo's
prior 53%-on-4090 number, now pinned on this card) with constant SMEM — **44× leaner memory than naive at
8K and no OOM** — and its recomputation backward passes a gradcheck vs SDPA autograd. GQA shrinks the KV
8× (32→4 MB). Honest gap to the ceiling: the ~50%-of-SDPA is the FA2-on-consumer-Blackwell number; the
FA3-class Hopper kernel (warp-spec + TMA + FP8, ~75% util / ~740 TF/s) is the H100 day (§4, R4).

---

## Perf track (A3 — tensor cores, sm120)

### Measured — A3 R0–R2 (2026-07-04, CUDA C++ via torch cpp_extension, -arch=sm_120, workflow-built + gpu-tested)

The tensor-core ladder that Triton's tl.dot abstracts — built explicitly in CUDA to show the fragment
machinery. sm120, 4096³ f16→fp32, cuBLAS = torch.matmul proxy (~72–84 TF/s). 34 gpu tests, element-exact
vs torch.matmul at fp32-accum tolerance. Bank-conflict counts are ncu-debt (blocked). R0 verifier-confirmed.

| date | rung | metric | predicted | measured | note |
|---|---|---|---|---|---|
| 2026-07-04 | A3 R0 naive SMEM GEMM (CUDA cores) | the floor WMMA climbs from | ~4–5% dense | **3.5 TF/s = 4.1% of cuBLAS** (no tensor cores) | 32×32 SMEM tile, fp32 accumulate; element-exact |
| 2026-07-04 | A3 R1 WMMA GEMM (nvcuda::wmma) | ≥40% of peak | 40–60% tuned band | **28.3 TF/s = 38.9% of cuBLAS** (39.4% peak) — **~8× over the CUDA-core floor** | 128×128 block, 8 warps × 4×2 16×16×16 frags; FP16-accum error grows with K (FP32-accum justified); sync (no cp.async yet) |
| 2026-07-04 | A3 R2 mma.sync + ldmatrix + XOR-swizzle | ≥60% of peak; conflicts≈0 | ≥60% | **59.0 TF/s = 81.9% of cuBLAS** (82.0% peak); rel err 6.6e-6 | mma.sync.aligned.m16n8k16 + ldmatrix + padded/swizzled SMEM; bank-conflict≈0 is ncu-debt |

**Verdict (A3 R0–R2 — SHIPPED).** The tensor-core ladder is built explicitly in CUDA (where Triton's
`tl.dot` hides the fragment machinery — the A3 lesson) and climbs correctly: **CUDA-core floor 4.1% →
WMMA 38.9% → mma.sync+swizzle 81.9% of cuBLAS**, all element-exact vs `torch.matmul` (6.6e-6) on sm120,
34 gpu tests. WMMA's ~8× jump over the scalar SMEM baseline IS the tensor-core win; the FP16-accumulate
variant's error growing with K justifies FP32 accumulation (tested). mma.sync + `ldmatrix` +
XOR-swizzled SMEM reaches **82% of the cuBLAS proxy** — exceeding the ≥60% DoD — with the warp-collective
fragment load and a conflict-free SMEM layout (the bank-conflict=0 gate is ncu-debt, blocked here,
H100-day). Honest gap: the next rung is `cp.async` double-buffering (hide the SMEM load behind the MMA,
the book's climb into the higher band) and then WGMMA/TMA — **H100-gated** (`H100_day_runbook.md` §3),
where the PTX artifact (`performance/artifacts/wgmma_descriptor_manual.md`) is the decode reference.

---

## Perf track (ISA-gated kernels — compile-verified, runtime deferred to rental)

### Compile-verified — A2§4.2 / A3 R3+§4.2 / A4 R4 (2026-07-04, `performance/rental/kernels/`)

These target sm_90a (Hopper) / sm_100a (datacenter Blackwell) — they CANNOT run on the sm120 box, so
**no runtime-correctness or measured-TF/s claim is made** (that is the rental day). The gate here is:
compiles cleanly with the exact nvcc command + the target ISA is present in the emitted PTX + structural
review vs CUTLASS/PTX-ISA/the WGMMA descriptor artifact. Workflow-built + adversarially verified +
main-thread re-compiled.

| date | kernel | arch | gate | verified |
|---|---|---|---|---|
| 2026-07-04 | WGMMA GEMM mainloop | sm_90a | nvcc -ptx exit 0; PTX has **4× wgmma.mma_async.m64n64k16.f32.f16.f16** + fence/commit/wait | descriptor encode matches CUTLASS GmmaDescriptor (start>>4, LBO 16, SBO 1024, 128B swizzle) — the PTX artifact |
| 2026-07-04 | FA3-class attention fwd | sm_90a | nvcc -ptx AND **-cubin exit 0** (full ptxas); PTX has **8× wgmma + 3× cp.async.bulk.tensor (TMA) + 14 mbarrier + setmaxnreg** (warp-spec) | producer/consumer warpgroups + TMA + online-softmax tile loop; fragment map/causal mask flagged DEFER |
| 2026-07-04 | tcgen05/UMMA GEMM (TMEM accum) | sm_100a | nvcc -ptx AND **-cubin exit 0**; PTX has **21× tcgen05 incl. tcgen05.mma.cta_group::1.kind::f16** + TMA; SASS (nvdisasm) = **UTCHMMA + LDTM.x4** | alloc→TMA+mbarrier→single-thread mma→commit→full-warpgroup 32-lane TMEM drain; idesc/descriptor consts flagged [INFERENCE] |

**Verdict — COMPILE-VERIFIED (SHIPPED as rental-day source).** The three ISA-gated kernels the sm120 box
cannot run now COMPILE for their target arches with the intended tensor-core ISA emitted in the PTX (2 of
3 pass full `-cubin` ptxas; tcgen05 SASS-cross-checked). This upgrades the rental prep from runbook-only
to compiled source — the H100/B200 days start from working-compiling kernels, not a blank editor. Honestly
framed: runtime correctness + TF/s are DEFERRED to the hardware (labeled throughout), so nothing here is a
`[FACT]` performance result — it is a `[compiles + structurally-correct]` artifact. Discharged on the
rental days (`H100_day_runbook.md`, `B200_day_runbook.md`).

---

## Perf track (A6 — distributed primitives, gloo/CPU code-only)

### Verified — A6 code-only (2026-07-04, gloo/CPU, workflow-built + adversarially mutation-tested, 112 CPU tests)

The "buildable NOW without the multi-GPU node" A6 primitives: CODE + gloo/CPU-testable CORRECTNESS.
Measured busbw / MFU-at-scale / the NVLink cliff are RENTAL-GATED (8×H200 day, `serving_day_8xH200_runbook.md`).
Reuses utils/comms_calc.py + memory_math.py (comms algebra + ZeRO calc, already shipped by the CS336 front).

| date | primitive | oracle | verified |
|---|---|---|---|
| 2026-07-04 | Megatron TP MLP (`utils/tp_mlp.py`) | gloo TP output + all grads == single-process ReferenceMLP (rtol 1e-5); **exactly 1 all-reduce/fwd, 2 fwd+bwd** | mutation test: bias-double-count + missing-all-reduce both FAIL (oracle has teeth); world_size 2 + 4 |
| 2026-07-04 | 1F1B / GPipe pipeline (`utils/pipeline_schedule.py`) | schedule makespan == independent Kahn longest-path DAG; **bubble == (p−1)/m** (p4m8→0.375, p8m16→0.4375); peak-act **1F1B min(p,m) vs GPipe m** | 77 tests; verifier's own event-sim reproduced all 9 (p,m); mutations (warmup off-by-one, dropped B-edge) caught |
| 2026-07-04 | MoE expert-parallel all-to-all (`utils/ep_moe.py`) | gloo dispatch→expert→combine == single-process dense-gather reference (allclose); zero token-divergence | real cross-rank traffic (rank0 sends 3/12, 5/10 off-rank); mutation (remove inverse-permute) caught |
| 2026-07-04 | MFU/HFU 6ND calculator (`utils/mfu.py`) | reproduces **PaLM 540B 46.2% MFU** (computed 45.70%, Δ0.005) / **57.8% HFU** (57.17%) from 6ND; six-killer decomposition sums to 1 | 29 tests; N·D invariance, HFU≥MFU monotone in recompute, 4/3 full-recompute ratio |

**Verdict (A6 code-only — SHIPPED, gloo/CPU-verified).** The four A6 primitives the standing single GPU
can build+test are correct: the Megatron TP MLP is numerically identical to single-GPU with exactly the 2
all-reduces/layer the theory predicts; the 1F1B scheduler's bubble matches (p−1)/m and its peak-activation
advantage (min(p,m) vs m) is pinned by an independent DAG simulator; the EP-MoE all-to-all matches a
dense-gather reference with real cross-rank traffic; the MFU calculator reproduces PaLM's published
46.2%/57.8% from C=6ND. **Measured numbers are rental-gated** (busbw at line rate, MFU at 16/32/64 GPUs,
the ~18× NVLink→IB cliff — the 8×H200 serving day + optional Phase-5 multi-node). A6 R0/R1 (topology +
TP micro) execute inside that day; these primitives + comms_calc/memory_math are the arrive-prepared code.
