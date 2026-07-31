# A4 — Flash Attention: The Kernel That Never Writes S to HBM

> **Book chapter:** 8 (Flash Attention) · **Frontier thesis:** Attention is not a compute problem — it is a *data-movement* problem disguised as a matmul. The naive algorithm is correct and unusable: it materializes the N×N score matrix in HBM, so its cost is `Ω(N²)` memory traffic, not the `O(N²d)` of arithmetic. Flash Attention's one move — *fuse the two GEMMs and the softmax into a single kernel whose intermediate scores never leave SRAM* — turns a memory-bound algorithm into a compute-bound one. The engine that makes the fusion legal is **online softmax**: a one-pass recurrence that computes the exact, globally-normalized softmax while only ever seeing one tile of scores at a time. You will build this from a 3-kernel oracle up to an FA3-class Hopper kernel, and acquire mechanism literacy on FA4/Blackwell.
>
> **Primary hardware:** 1× H100 for the required frontier core (FA3 needs WGMMA+TMA → sm_90a). Rungs 0–3 run on a 4090/A100. The FA4 stretch rung needs a B200 (tcgen05/TMEM). · **Est. time:** 2–2.5 weeks · **Prereqs:** book ch1–7; `00_foundations.md` (roofline reflex, locked-clock benchmarking, the sm_ gates). **Builds directly on A3** (WMMA → WGMMA → tcgen05/UMMA) and **A2** (warp/block reductions — the softmax max/sum *is* a reduction). This assignment *fuses A3's tensor-core matmul with A2's reductions* under one kernel.

---

## §0 Why this matters

The book's chapter 7 spent itself making matmul fast. Chapter 8 opens by admitting that for attention, *that was the wrong fight*. Standard self-attention computes `S = QKᵀ/√d` (shape N×N), `P = softmax(S)` row-wise, then `O = PV` — two GEMMs with a row-wise softmax wedged between them. The matmuls are exactly what tensor cores are built for. The catastrophe is the `S` matrix in the middle: at N=8192 in FP32 it is 268 MB *per head*; at 128K context it is over 65 GB. The naive kernel writes `S` to HBM, reads it back for softmax, writes `P`, reads `P` back for `O@V` (the book's figure 8.1). The GPU sits idle waiting on memory while the tensor cores starve.

This is the single most important shape-shift in modern ML systems. Attention went from compute-bound to **memory-bound**, and the fix — Flash Attention (Dao et al. 2022) — is now the load-bearing kernel of every transformer on earth. It is also a perfect principal-level object lesson: the win came not from a faster matmul but from an **I/O-optimal algorithm** and a numerical trick (online softmax) that most GPU programmers initially believe is impossible because softmax "needs the whole row." By the end of this assignment you will have built the kernel that never writes `S` to HBM, proven it bit-exact against a 3-kernel oracle, pushed it to ~75% of H100 peak with the FA3 techniques (warp specialization, TMA, WGMMA, ping-pong, FP8), and be able to explain — at the feeds-and-speeds level — why FA4 had to move work *off* the tensor cores on Blackwell.

The mantra for the whole assignment: **never write S to HBM. Online softmax is the engine that makes that legal.**

---

## §1 Learning objectives

You can:

1. **Derive** attention's arithmetic intensity and place naive vs fused attention on the roofline, and prove from first principles that naive attention is `Ω(N²)` HBM-bound while Flash Attention is I/O-optimal at `O(N²d²·M⁻¹)` HBM accesses (M = SRAM size).
2. **State and derive** the online-softmax recurrence (running max `m`, running denom `d`, the rebasing factor `e^{m_{i-1}−m_i}`) and prove it produces *exactly* the same result as a 3-pass safe softmax — it is not an approximation.
3. **Implement** a naive 3-kernel attention (QKᵀ kernel, row-softmax kernel, PV kernel + host logic) as a **correctness oracle**, and explain precisely why it is memory-bound.
4. **Implement** a single fused tiled Flash-Attention-1 kernel: SMEM layout, outer loop over Q-tiles, inner loop over K/V-tiles, QKᵀ via WMMA, online-softmax update of `(m, d, O)` in SRAM, deferred `÷d`, causal masking, write-back; demonstrate exact match to the oracle and linear (constant-per-step) memory.
5. **Apply** the FA2 work-partitioning improvements — defer the rescale division, parallelize over the sequence dimension, split-Q warp partitioning (no inter-warp reduction) — and measure the speedup toward 50–73% of A100 peak.
6. **Build** an FA3-class Hopper kernel — producer/consumer warp specialization, TMA tile loads, WGMMA async matmul, **ping-pong** scheduling that overlaps softmax `exp()` (SFU) with WGMMA (tensor cores), and **FP8 attention** with block quantization + **incoherent processing** — and benchmark it against FlashAttention-3 and cuDNN.
7. **Explain** the FA4/Blackwell thesis at feeds-and-speeds granularity (asymmetric scaling, exp-bound forward / SMEM-bound backward, tcgen05/UMMA, conditional rescaling, software-emulated exp2) and integrate **one** variant (MLA, GQA, or paged/sink attention) into your kernel.
8. **Implement** the Flash-Attention backward pass (store `O`+`logsumexp`, recompute `S`,`P`) and explain why recomputation beats storage when the operation is memory-bound.

---

## §2 First-principles theory

> *Setup, fixed for the whole assignment.* One head: `Q, K, V ∈ ℝ^{N×d}`. `S = QKᵀ/√d` (N×N). `P = softmax(S)` row-wise. `O = PV` (N×d). Two GEMMs with a row-wise softmax between them. Causal attention masks `S[i,j]` for `j > i`.

### 2.1 Why attention is memory-bound (the roofline, by hand)

Do this on paper first, per `00_foundations.md §3`. An H100 delivers ~989 BF16 TFLOP/s but only ~3.35 TB/s of HBM bandwidth → the roofline **ridge point** is `989e12 / 3.35e12 ≈ 295 FLOP/byte`. Anything below 295 arithmetic intensity is bandwidth-bound and the tensor cores idle.

Now count the naive algorithm's HBM traffic. `QKᵀ` reads Q,K (`2Nd` elements) and **writes `S` (N² elements) to HBM**. Safe softmax is 3 passes over each row (max, sum-of-exp, normalize) → reads `S` and writes `P`, i.e. `Θ(N²)` more traffic. `PV` reads `P` (N²) and V (Nd), writes O (Nd). Total HBM ≈ `Θ(Nd + N²)`, and **the `N²` term dominates** for any realistic N. The compute is `Θ(N²d)`. So arithmetic intensity is `≈ Θ(N²d)/Θ(N²) = Θ(d)` — for d=128 that's far left of the 295 ridge, and worse, the constant factors (the round-trips) crush it. The book's figure 8.1 is exactly this: write S, read S, write P, read P, all across the slow bus. **The matmuls are not the problem; the N×N matrix in HBM is.**

### 2.2 Flash Attention: tile + never materialize S → I/O-optimal

Flash Attention's insight is to **fuse** the whole pipeline into one kernel and process attention in *tiles* small enough that `S`'s tile lives in SRAM and is consumed immediately — it is never written to HBM. Dao et al. (arXiv:2205.14135) prove this kernel performs `Θ(N²d²·M⁻¹)` HBM accesses (M = SRAM bytes), and prove a matching lower bound: it is **I/O-optimal** for exact attention. Crucially it is **exact**, not approximate — bit-for-bit equal to standard attention (within floating-point reassociation). Memory is **linear** in N (you store O and per-row statistics, never S). The original paper reported ~3× wall-clock on GPT-2 training and *enabled* 64K-context models that previously OOM'd. This is gradient checkpointing's logic — trade recompute for memory — but specialized to attention and turned into an *I/O* win because attention is memory-bound.

The book frames the three misconceptions this breaks (8.2.1): (1) tiling attention is *not* like tiling GEMM, because softmax is a *global* row operation — token 0 attending to token 1 needs the scores of token 0 against *all* N keys; (2) you do **not** compute independent local softmaxes per tile and combine them — that is mathematically wrong; (3) the GEMM "inching-inwards" loop is still there, just nested one level down. The resolution to all three is online softmax.

### 2.3 Online softmax — the recurrence that is the engine

The enabling algorithm is **online softmax** (Milakov & Gimelshein, arXiv:1805.02867): compute a numerically-stable, globally-normalized softmax in a *single pass* over the logits, maintaining two running scalars per row — the running max `m` and the running denominator `d` (the book calls them `m` and `l`).

Initialize `m₀ = −∞`, `d₀ = 0`. For each new logit `xᵢ` (processed left to right):

```
mᵢ = max(m_{i-1}, xᵢ)                                  # running max
dᵢ = d_{i-1} · e^{m_{i-1} − mᵢ}  +  e^{xᵢ − mᵢ}         # rescale prior, add new
```

The factor **`e^{m_{i-1} − mᵢ}` re-bases every previously-accumulated term to the new maximum**. When `xᵢ` is a new record, `mᵢ > m_{i-1}`, the factor is `< 1`, and it shrinks the old denominator to the new scale; when `xᵢ` is not a record, `mᵢ = m_{i-1}` and the factor is `1` (no-op). After the final logit, `softmax(x)ⱼ = e^{xⱼ − m_N} / d_N` — identical to the 3-pass safe softmax, but computed in one pass without ever storing the row. **Prove this by induction** (do it in your journal): the invariant is `dᵢ = Σ_{k≤i} e^{x_k − mᵢ}`.

Fuse the **output** update into the same pass. For query row `q`, iterating key `i`, with running output `o`:

```
oᵢ = o_{i-1} · (d_{i-1}/dᵢ) · e^{m_{i-1} − mᵢ}  +  (e^{xᵢ − mᵢ}/dᵢ) · V[i]
```

The `o_{i-1} · (d_{i-1}/dᵢ) · e^{m_{i-1} − mᵢ}` term rescales the accumulated output to the new max *and* new denominator; the second term adds the new value's contribution. This is the book's "bank-account balance" mental model (8.3.4): `O` in SRAM is one running total, updated in place, never a growing list of partials — which is *why* memory stays constant regardless of sequence length.

**In the tiled kernel** (the real FA), `x` is not one logit but a `Q-tile × K-tile` *block* of scores; `m` and `d` are per-row **vectors** (one entry per query row in the tile); the rescale `e^{m_old − m_new}` is applied per-tile to the running `O` and `d` after each K/V tile is processed. The **FA2 trick** is to *defer the `÷dᵢ`* — carry the un-normalized `Õ` and the running `d` through the whole inner loop, and divide exactly once at the very end (`O = Õ / d_N`). This removes a per-tile division from the hot loop. This online-softmax recurrence is the **conceptual spine of this entire assignment**; everything else is making it fast on a given GPU.

### 2.4 The three-level loop hierarchy (book 8.3.3)

Flash Attention nests three loops, which dissolves the "GEMM intuition doesn't fit" confusion:

- **Grid level — parallel over output-row tiles.** Each thread block owns a complete *brick* of output rows (e.g., rows 0–63) and produces them independently. No cross-block communication; running stats `(m, d)` live in that block's registers/SMEM. (Book figure 8.3.)
- **Block level — sequential march across the sequence.** The block loops over all K/V tiles (`j = 0…Tc`), loading `Kⱼ, Vⱼ` into SMEM, applying the online-softmax update each step. Memory footprint per step is *fixed* — a length-128K sequence uses the same SRAM as length-8K, just more iterations.
- **Warp/thread level — the GEMM "inching-inwards" loop over `d`.** Each score tile is a standard mini-GEMM accumulating over the head dimension via WMMA/WGMMA. This is the loop you already know from A3.

The middle (sequential) loop produces a complete score tile, the online-softmax immediately consumes it into `(m, d, O)`, and **the score tile is discarded** before the next iteration. That discard is the "never write S to HBM."

### 2.5 The evolution, as a thesis (book 8.4)

Each FA generation chased the same target — *get closer to the hardware's peak* — by removing whatever was now the bottleneck:

- **FA1** established tiling + online softmax + recompute-in-backward. Forward hit only **30–50%** of peak, backward **25–35%** — bottleneck shifted from HBM to *suboptimal parallelism and work distribution*.
- **FA2** fixed the partitioning: fewer non-matmul FLOPs (defer rescale; on A100 a non-matmul op is ~16× costlier than a tensor-core matmul op), parallelize over the **sequence** dimension, split-Q warp partitioning. ~2× FA1 → **50–73%** of A100 peak. But only ~35% on H100 → motivated FA3.
- **FA3** exploited Hopper's *asynchronous* hardware (TMA, WGMMA) with warp specialization and ping-pong, plus FP8. Up to **75%** of H100 peak.
- **FA4** confronts Blackwell's *asymmetric* scaling: tensor-core throughput jumped but the SFU/exp unit and SMEM bandwidth did not, so the bottleneck moved **off** the tensor cores. The fix is mostly software (emulated exp, conditional rescaling, a dedicated correction warpgroup).

The meta-lesson (book 8.4.4): **algorithmic** innovation delivered the big early win (tiling + online softmax); later gains are **hardware-software co-design** — inseparable from the specific GPU, and requiring constant rewriting as architectures evolve. That is the principal-level reason CUTLASS/CuTe-DSL exist.

---

## §3 The from-scratch build ladder

Climb in order. Each rung is gated on **bit-exact (within fp tolerance) agreement with the R0 oracle** before promotion. Keep the engineering journal (`00_foundations.md`): hypothesis → measurement → roofline placement → Nsight evidence. Targets assume H100-class, head dim **d=128**, **FP16** unless noted.

**Rung 0 — Naive 3-kernel attention = the correctness oracle.**
Build exactly the book's baseline (listings 8.1–8.4): a `naive_qk_matmul_kernel` (each thread one element of `S = QKᵀ·scale`), a `naive_softmax_kernel` (each thread one full row: max → exp-sum → normalize), a `naive_pv_matmul_kernel` (each thread one element of `O = P@V`), and host logic that `cudaMalloc`s the full N×N `S` in HBM and launches the three in sequence. Also keep a **PyTorch SDPA reference** (`torch.nn.functional.scaled_dot_product_attention` with a math/no-flash backend) as a second oracle.
*Mechanism:* it is correct precisely because it does the textbook thing — and it materializes `S` and `P` in HBM, so it is the memory-bound disaster of §2.1.
*Correctness check:* matches PyTorch SDPA to `max|Δ| < 1e-3` (the book's tolerance).
*Performance target:* **none — this is the baseline.** Expect ~5–15% of peak and **OOM past ~16K** seqlen. Profile it once with Nsight to *see* the HBM-bound SoL (Memory% ≫ Compute%) and the three kernels serialized on the timeline. That picture is your motivation.

**Rung 1 — Single-pass online softmax, one query row.**
Before any tiling, internalize the recurrence. In Numba or a trivial one-thread CUDA kernel, compute softmax-weighted output for a *single* query against all N keys using only the running `(m, d, o)` recurrence of §2.3 — never storing the score row. Compare against a naive 3-pass softmax on the same row.
*Mechanism:* the rebasing factor `e^{m_old−m_new}` and the in-place output rescale, in isolation, with no parallelism to hide bugs.
*Correctness check:* `max|Δ| < 1e-6` vs 3-pass safe softmax on random rows *and* on an adversarial row with one huge logit (tests the rescale).
*Performance target:* none — this rung exists to make the rescale reflexive.

**Rung 2 — Fused tiled Flash Attention (the real FA1).**
Build the book's fused kernel (listings 8.5–8.11): templated tile sizes `<Br, Bc>`; the SMEM layout partitioning one raw buffer into `Qᵢ, Kⱼ, Vⱼ, Sᵢⱼ, Oᵢ, mᵢ, lᵢ`; the **outer loop over Q-tiles**; the **inner loop loading K/V tiles** cooperatively into SMEM; `QKᵀ` via WMMA (`mma_sync` over the head dim, `store_matrix_sync` the tile to SMEM); the **online-softmax update** of `(m, l)` and the in-place rescale of `Oᵢ`; `P@V` via WMMA accumulated into `Oᵢ`; and the final **normalize-by-`l` + write-back** to HBM. Add **causal masking** (skip or `-INF` the `j>i` tiles — and skip entire K/V tiles above the diagonal for a free speedup). Defer the `÷l` to the end (the FA2 trick) even in this rung.
*Mechanism:* this is the whole chapter — fusion + online softmax + WMMA, `S`/`P` never touch HBM (book figure 8.2).
*Correctness check:* `max|Δ| < 1e-3` vs R0 across N ∈ {256, 512, 1024, 4096}, causal and non-causal; **constant SMEM** regardless of N (prove it — the footprint per step is fixed); no OOM at 64K.
*Performance target:* **~40–60% of A100 FA1/early-FA2** (the book's pedagogical kernel hits only ~5–15% because 7/8 warps do redundant work and only 16/256 threads do the softmax — your job here is to fix the obvious waste, not yet to specialize). Beat the R0 baseline by the book's 5–15× and explain the caveat (the speedup conflates fusion + tensor-cores + launch-count, per book 8.3.5).

**Rung 3 — FA2 work-partitioning.**
Apply the three FA2 improvements (arXiv:2307.08691): **(1)** drive non-matmul FLOPs down — keep the rescale division out of the inner loop (already deferred), minimize `exp` re-bases. **(2)** Parallelize over the **sequence dimension**: make the outer Q-block loop a grid dimension so small batch×heads still fills the GPU (forward parallelizes over Q-blocks; backward over K-blocks). **(3)** **Split-Q warp partitioning**: each warp owns a slice of the Q-tile's rows and the *whole* K/V tile, so warps never need an inter-warp reduction to combine partial `QKᵀ` — contrast FA1's "split-K" which forced SMEM round-trips and `__syncthreads`. Fix the book kernel's redundant-warp waste here.
*Correctness check:* still bit-exact vs R0; verify occupancy rose (Nsight `sm__warps_active`).
*Performance target:* **50–73% of A100 peak (~200–225 TFLOP/s FP16)**. On H100 you'll see only ~35% — that gap is the entire reason FA3 exists, and the bridge to §4.

**Rung 4 — FA3-class Hopper kernel (this is §4; see there for the full spec).**
WGMMA + TMA + warp specialization + ping-pong + FP8. Start from **CUTLASS/CuTe, not raw PTX**. Target ~75% of H100 peak. Detailed in §4.

**Rung 5 (stretch) — FA4-class Blackwell kernel.**
tcgen05/UMMA + TMEM-resident accumulators, conditional online-softmax rescaling, software-emulated exp2, 2-CTA backward with DSMEM. CuTe-DSL. Target ~71% of B200 peak. Detailed in §7.

---

## §4 Frontier-2026 core (required)

**The required deliverable is an FA3-class Hopper forward kernel** (Rung 4) that uses warp specialization, TMA, WGMMA, and ping-pong, supports FP8 with incoherent processing, and is **benchmarked against FlashAttention-3 and cuDNN** on an H100. Plus **required mechanism literacy** on FA4/Blackwell (you must be able to explain it cold), and **one variant** (MLA, GQA, or paged/sink) integrated into your kernel. Build on A3 — you already have WGMMA and tcgen05 from there; here you wrap them in the attention pipeline and fuse A2's reductions (the softmax max/sum) into the same kernel.

### 4.1 FA3 forward kernel on Hopper (arXiv:2407.08608) — required

Five mechanisms, each removing a bottleneck FA2 left on the H100:

1. **Producer/consumer warp specialization.** Split the block's warps into roles: *producer* warps issue async loads; *consumer* warpgroups run the matmuls and softmax. They communicate through a circular SMEM buffer guarded by `mbarrier` async barriers. This is the "write a state machine, not a function" shift the book names in 8.4.4 — a register-reallocation (`setmaxnreg`) gives the math warpgroups more registers and the producer fewer.
2. **TMA tile loads.** Use the Tensor Memory Accelerator (`cp.async.bulk.tensor`) to stream `K`/`V` tiles HBM→SMEM with a single thread issuing a whole-tile copy, overlapped with compute. Replaces the book kernel's cooperative element-by-element load loop. (Hopper-only — `00_foundations.md` gate.)
3. **WGMMA async matmul.** `wgmma.mma_async` does both GEMMs warpgroup-wide and asynchronously, so the math warpgroup issues the matmul and proceeds while tensor cores work. This is your A3 WGMMA rung, now inside the attention loop.
4. **Ping-pong scheduling (the key FA3 idea).** Softmax's `exp()` runs on the **SFU/MUFU** units; WGMMA runs on the **tensor cores** — *different* hardware units. FA3 schedules **two** Q-tiles (or two warpgroups) so that while warpgroup A runs softmax `exp()` on tile *i*, warpgroup B runs WGMMA on tile *i−1*, and they swap. The slow special-function `exp` (book: ~3.9 TFLOP/s on the SFU) is **hidden under** the 989 TFLOP/s of WGMMA. This is the principal-level move: overlap by *unit*, not by warp.
5. **FP8 attention.** Run the GEMMs in FP8 (E4M3) for ~2× tensor-core throughput, with two numerics tricks: **(a) block quantization** — per-block scale factors instead of a single per-tensor scale, so a few outliers don't blow up the whole tile's error; **(b) incoherent processing** — multiply Q and K by a *shared* random orthogonal/Hadamard matrix M before quantizing. Because `(QM)(KM)ᵀ = QMMᵀKᵀ = QKᵀ` for orthogonal M, the scores are **mathematically unchanged**, but M *spreads outliers* across channels, shrinking the per-element quantization error. Accumulate in FP16/FP32; do the softmax `exp` in FP32 always.

**FA3 numbers (from the paper):** 1.5–2× FA2 on H100; FP16 up to **740 TFLOP/s = 75% of H100 peak**; FP8 up to **~1.2 PFLOP/s**; FP8 numerical error **2.6× lower** than a baseline (per-tensor) FP8 attention. These are the FP16/FP8 *paper* numbers — cite them as such.

*Correctness:* FP16 path bit-exact (`<1e-3`) vs R0; FP8 path within a documented error budget vs FP16 (report `max|Δ|` and RMS; show incoherent processing reduces it — reproduce the "2.6× lower" qualitatively).
*Performance target:* **≥70% of H100 FP16 peak (~700+ TFLOP/s)**; FP8 path measurably faster than FP16. **Benchmark head-to-head vs `flash-attn` v3 and cuDNN's fused attention** (cuDNN 9.x `cudnnFusedAttention` / the SDPA cuDNN backend) on identical shapes; you should land within a small factor of FA3 and be honest about the gap (almost always: less perfect ping-pong overlap, suboptimal tile size, or epilogue inefficiency).

### 4.2 Required mechanism literacy: FA4 / Blackwell

You must be able to explain FA4 at feeds-and-speeds granularity (implementation is the §7 stretch). **Honesty flags up front (re-verified 2026-07-30):** FA4 is **public** — paper arXiv:2603.05451 (Zadouri et al., 2026-03-05), Tri Dao's blog (tridao.me/blog/2026/flash4), and code shipping as `pip install flash-attn-4` (CuTe-DSL, JIT — no nvcc; `flash_attn/cute/`, a `cu13` extra for CUDA 13), optimized for **Hopper and Blackwell**. The **BF16 forward** numbers below are public. Three caveats that post-date the launch coverage: **(a)** cuDNN **adopted the same techniques** (not a code merge) and current cuDNN (9.24) **matches FA4** — the launch claim "1.3× cuDNN 9.13" is dated; benchmark against the cuDNN of the day. **(b)** **FA4 decode on Hopper regresses vs FA3 at long sequence (−49% @16K — no SplitKV):** FA4 is not a strict upgrade; FA3 remains the Hopper decode answer. **(c)** **low-precision (FP8/FP4) FA4** accuracy/throughput numbers are **still not public** — do **not** invent them; the exact rescale threshold τ and the precise MUFU/FMA split for emulated exp are also not public.

**The thesis — asymmetric hardware scaling.** H100→B200 roughly **doubled** BF16 tensor-core throughput (1 → ~2.25 PFLOP/s dense) but the **SFU/exp unit and SMEM bandwidth did not scale proportionally**. So the bottleneck moves *off* the tensor cores: the **forward becomes exp-bound** and the **backward (1-CTA) becomes SMEM-bound**. Per-SM feeds & speeds at M=N=D=128 (from the blog): tensor cores ~**8192 ops/cycle**, the exp unit only ~**16 ops/cycle**, SMEM ~**128 B/cycle**. When the matmul is that cheap relative to `exp`, *softmax* is the critical path.

**Hardware it rides:** **tcgen05 / UMMA** — accumulate in **TMEM** (tensor memory, not registers), a 128×256×16 MMA atom (2× Hopper's), issued by a *single thread*; plus **2-CTA MMA** (a 256×256×16 op spanning two CTAs). This is your A3 tcgen05 rung.

**Forward pipeline.** Ping-pong **two Q-tiles** against **two softmax warpgroups**, *synchronized so the two softmax groups never hit the exp unit simultaneously* (since the exp unit is the bottleneck, serialize access to it deliberately). **Conditional online-softmax rescaling:** only apply the `e^{m_old−m_new}` rescale when the running max **jumps by more than a threshold τ** (at warp granularity) — most tiles don't move the max much, so skipping the rescale ~10× of the time relieves the bottleneck while staying within numerical tolerance. A dedicated **"correction" warpgroup** runs the deferred rescales **off the critical path**. And **software-emulated exp2**: instead of the scarce `MUFU.EX2`, compute `exp2` as **Cody-Waite range reduction + a degree-3 Horner polynomial**, splitting the work across `MUFU.EX2` *and* the FMA pipes so the SFU isn't the sole bottleneck.

**Backward pipeline.** The backward does ~**2.5× the tensor-core work** of the forward (five MMAs: `dV`, `dP`, `dS`, `dQ`, `dK`) and is **SMEM-bound** (1-CTA). Fixes: **TMEM-resident accumulators**; **transposed S/P recompute**; **2-CTA MMA + DSMEM exchange** to halve operand-B SMEM traffic (one CTA's SMEM serves both). A **deterministic** mode runs at ~**85–90%** of the nondeterministic throughput.

**FA4 numbers (public, BF16):** B200 BF16 up to **≈1.6 PFLOP/s = 71% utilization** (blog 1605 TF/s, abstract 1613 — cite ≈1.6 PF); launch comparison **1.3× cuDNN 9.13** and **2.7× Triton** (both dated — see caveat (a) above); entirely in **CuTe-DSL**. Warp-role anatomy: **1 load + 1 MMA + 8 softmax + 4 correction warps** per block. (FP8/FP4 FA4: still not public as of 2026-07-30 — flag it.)

### 4.3 One variant, integrated (required — pick one)

Wire a real attention variant into your kernel, not just describe it:

- **GQA** (Ainslie et al., arXiv:2305.13245): query heads share K/V in `g` groups (uptrain from MHA at ~5% of pretraining compute; the Llama-3/Mistral default). In the kernel, multiple Q-row-tiles index the *same* K/V tile — a load-amortization win. Easiest integration; verify grouped math against full-MHA on a seed.
- **MQA** (Shazeer, arXiv:1911.02150): the `g=1` extreme — one K/V head for all query heads.
- **MLA** (DeepSeek-V2, arXiv:2405.04434): cache a low-rank **latent** `c_KV` (d_c≈512) and reconstruct per-head K/V via `W^UK`/`W^UV`. **Weight absorption** folds `W^UK` into `W^Q` and `W^UV` into the output projection so you **attend in latent space** — KV cache ≈ **4–14% of MHA**. Carries a **decoupled-RoPE** pathway (a separate small `d_h/2` head dim) because RoPE doesn't commute with the absorbed matrices. (Connects to A1.) Hardest, most impressive.
- **Paged / sink attention:** **PagedAttention** (arXiv:2309.06180) — block KV with a logical→physical block table, kernel gathers blocks. **Attention sinks / StreamingLLM** (Xiao et al., arXiv:2309.17453): the empirical finding that models dump attention mass onto the **first few tokens**; *keep those sink tokens' KV + a sliding window* → stream to **4M+ tokens**, up to **22.2×** vs sliding-window-with-recompute. This is the origin of the learnable "sink token." (Sliding window: Longformer arXiv:2004.05150, Mistral.)
- **Sparse attention — NSA / DSA (the 2026 production edge).** Long-context frontier models no longer attend densely. **NSA** (DeepSeek, arXiv:2502.11089): native, hardware-aligned *trainable* sparsity — three gated branches (compressed blocks, selected top-n blocks, sliding window) so the FLOPs shrink with the pattern. **DSA** (DeepSeek-V3.2, arXiv:2512.02556): a cheap **lightning indexer** (few heads, low-rank, FP8, ReLU-scored) predicts the top-k (k≈2048) tokens per query, and full MLA attention runs only over the selected set — O(L·k) not O(L²); in production in V3.2/V4 with kernels shipped in TileLang + CUDA. The kernel shape is **"gather-then-attend"** — a selection/index pass feeding a block-sparse FA. (This repo's `src/scratch_llm/dsa.py` is the from-scratch reference.) If you integrate this variant, the oracle is dense attention restricted to the selected subset.
- **Hybrid linear attention (context — the DELTA bridge).** Frontier 2026 models (Qwen3.5, Kimi Linear, Nemotron, GLM-5-class hybrids) mix a few full-attention layers with **linear-attention layers** (Gated DeltaNet / Mamba-class): chunked-scan kernels for prefill, **fused-recurrent state kernels** for decode (the FLA library). At batch-1 decode the recurrent state round-trips HBM every token — memory-bound, exactly the GEMV regime of A1 — and a low-precision (FP8/NVFP4) recurrent-*state* decode path is the open kernel niche (the DELTA project). Mechanism literacy only here.

*For any choice:* prove the variant's attention output matches a reference implementation, and report the memory/throughput consequence with numbers.

> **Ecosystem context to cite (not required to build):** **FlexAttention** (PyTorch) compiles a user `score_mod`/`mask_mod` *into* the FA kernel — its **FA4 backend** landed 2026-03-05; Blackwell min sparse block is 256×128. **FlashInfer** (arXiv:2501.01005, MLSys'25 best paper) is the serving attention engine (JIT, paged/MLA/spec backends) backing vLLM/SGLang/TRT-LLM. These are where your kernel would *live* in production.

---

## §5 Correctness & numerics

- **The oracle is non-negotiable.** Every rung matches **R0 (naive 3-kernel)** AND PyTorch SDPA (math backend) to `max|Δ| < 1e-3` in FP16, `< 1e-5` in FP32. Online softmax is *exact* — any larger divergence is a bug (off-by-one in the rescale, wrong `m_old`, a tile not masked), not "numerical noise."
- **Online softmax is bit-exact, prove it.** At Rung 1, match a 3-pass safe softmax to `< 1e-6`. The classic bug: forgetting to rescale the running `O` (not just `d`) when the max updates — it passes on benign inputs and fails when a late tile contains the row max. **Test the adversarial row** (one logit = +50, rest near 0) on purpose.
- **Numerical-stability discipline.** Always subtract the running max before `exp` (no naked `exp(score)` — overflow). Keep `m, d` and the `exp`/softmax accumulation in **FP32** even when the matmuls are FP16/FP8. This mirrors A1's MoE lesson: a tiny softmax perturbation can flip a discrete decision and compound — carry that paranoia.
- **FP8 numerics (the frontier risk).** E4M3 has ~2 decimal digits of precision. Validate the FP8 path against the FP16 path: report `max|Δ|`, RMS error, and the **worst affected row**. Show **block quantization** beats per-tensor, and that **incoherent processing** (orthogonal `M`) further shrinks error — confirm the `(QM)(KM)ᵀ = QKᵀ` identity numerically *first* (it should be exact up to fp roundoff), then measure the error reduction. Always accumulate FP8 matmuls in FP16/FP32; always do softmax `exp` in FP32.
- **Causal mask correctness.** Verify masked positions contribute *exactly zero* (their `P` entries are 0, not tiny). Test the diagonal tile (partial mask) specifically, and seqlen crossing a tile boundary.
- **Backward (if built).** Recompute `S = QKᵀ`, `P = exp(S − L)` from stored `O`+`logsumexp L`. Gradient-check `dQ, dK, dV` against `torch.autograd` finite differences and against PyTorch's analytic grads to `< 1e-2` (FP16). Why recompute beats storing `P`: the op is memory-bound, so re-doing the FLOPs is cheaper than the HBM traffic of saving/loading `P` — attention-specialized gradient checkpointing.
- **Adversarial cases:** N not a multiple of tile size (ragged last tile); N=1; all-equal logits (uniform softmax); a row that is fully masked except the diagonal; very long N (64K) for the linear-memory claim.

---

## §6 Profiling & performance

- **The headline measurement: prove the shift from memory-bound to compute-bound.** Show R0 (naive) is HBM-bound (Nsight SoL Memory% ≫ Compute%; the three kernels serialized on the `nsys` timeline; OOM at large N) and that your fused kernel is **compute-bound** (Compute% dominant, tensor-core pipe `sm__pipe_tensor_*` busy). This before/after *is* the chapter's thesis, measured.
- **Roofline placement** (`00_foundations.md §3`): annotate naive attention (far left, bandwidth-bound) and fused FA (near the tensor-core ceiling) on the H100 roofline. Predict your FLOP/s from tile sizes and occupancy *before* measuring; explain any gap.
- **Tile-size sweep.** `Br × Bc` ∈ {64, 128, 256}² — plot TFLOP/s and SMEM occupancy. The book's pedagogical 16×16 tile is far from optimal; 128×128 is typical. Find your kernel's sweet spot and explain the SMEM/occupancy trade-off.
- **Ping-pong evidence (FA3).** Show on the timeline that `exp` (SFU/MUFU pipe) and WGMMA (tensor pipe) are **concurrent** — `sm__pipe_fma_*`/`mufu` busy *simultaneously* with `sm__pipe_tensor_*`. If they're serialized, your ping-pong isn't overlapping and that's your missing throughput.
- **FP8 vs FP16.** Throughput ratio (should approach ~2×) and the numerics cost (§5). Plot error vs quantization scheme (per-tensor → block → block+incoherent).
- **Head-to-head.** TFLOP/s and % -of-peak for *your* kernel vs **FlashAttention-3** vs **cuDNN** vs **PyTorch SDPA (flash backend)**, across seqlens, causal and non-causal. You will likely trail FA3 — *where and why* is the principal-grade writeup.
- **Benchmarking hygiene** (`00_foundations.md`): locked clocks (`nvidia-smi -lgc`), warmup discarded, CUDA-event timing, report median + p95, fixed shapes. Use **achieved** FLOP/s (count the actual `N²d`·2·2 attention FLOPs, halve for causal) — not the hardware peak.

---

## §7 Stretch goals (competition-grade)

- **FA4 Blackwell kernel (Rung 5).** On a B200, implement the FA4 forward in **CuTe-DSL**: tcgen05/UMMA with **TMEM-resident** accumulators, the **2-Q-tile / 2-softmax-warpgroup ping-pong** synced off the exp unit, **conditional rescaling** (skip the `e^{m_old−m_new}` rescale unless the max jumps > τ, warp granularity — tune τ yourself, it's not public), a dedicated **correction warpgroup** off the critical path, and **software-emulated exp2** (Cody-Waite range reduction + degree-3 Horner, split across `MUFU.EX2`+FMA). *Target:* approach the public **≈1.6 PFLOP/s = 71%** of B200 BF16 peak; beat Triton; parity with **current cuDNN (9.24 — it adopted the FA4 techniques)** is the real bar. Honesty: low-precision FA4 is uncharted (not public) — if you try FP8/FP4, you're doing original work, report it as such.
- **FA4 backward.** The hard one: ~2.5× the TC work, SMEM-bound, needs **2-CTA MMA + DSMEM** operand sharing and TMEM accumulators; implement the **deterministic** mode and measure the ~85–90% throughput cost vs nondeterministic.
- **Variable-length / packed batching.** Real serving packs ragged sequences (no padding) with a cumulative-seqlen index (`cu_seqlens`), the `flash_attn_varlen` interface. Implement it and show the throughput win vs padding to max length.
- **A second variant.** Having done one of §4.3, add another (e.g., MLA *and* paged) and integrate with the A1 serving engine — paged-KV FlashInfer-style block gather inside your kernel.
- **Persistent-kernel / tile-scheduler design.** Launch one block per SM that pulls tiles from a global work queue (à la FA4's tile dispatcher) instead of one block per output tile — better load balance under causal masking's triangular work.

---

## §8 Deliverables & definition of done

1. **Code:** R0 naive 3-kernel oracle; R1 single-row online softmax; R2 fused tiled FA1 (causal); R3 FA2 work-partitioning; **R4 FA3-class Hopper kernel** (warp-spec + TMA + WGMMA + ping-pong + FP8/incoherent); one integrated variant from §4.3; the backward pass. Each with its oracle test.
2. **The benchmark harness** (reusable, achieved-FLOP/s, locked clocks) and the plots: the **memory-bound→compute-bound** Nsight SoL before/after; the roofline placement; the tile-size sweep; the **ping-pong concurrency** timeline; FP8-vs-FP16 throughput+error; and the **head-to-head vs FA3 + cuDNN + PyTorch SDPA**.
3. **Design note (3–4 pages): "Why attention is a memory problem, and how the fusion + online softmax fix it — up to the frontier."** Lead with the §2.1 roofline and the `Ω(N²)` HBM argument; derive the online-softmax recurrence and *prove* exactness; walk each rung's before/after number with Nsight/nsys evidence; explain FA3's ping-pong (overlap by hardware unit) and FA4's asymmetric-scaling thesis (with the honest public/not-public split); end with your measured gap to FA3/cuDNN and *why*. Written for a frontier-lab peer.

**Done when:** every rung is bit-exact (within fp tolerance) vs the R0 oracle; you have *demonstrated* the memory-bound→compute-bound shift on the roofline and in Nsight; the FA3-class kernel hits ≥70% of H100 FP16 peak and is benchmarked against FA3 and cuDNN; FP8 works with block-quant + incoherent processing within a documented error budget; one variant is integrated and verified; the backward pass gradient-checks; and the design note correctly diagnoses your remaining gap to a production kernel and explains the FA4/Blackwell frontier without inventing the non-public numbers.

---

## §9 Principal-level rubric

- **Pass:** Fused tiled FA kernel (R2) works, bit-exact vs the oracle, constant SMEM, online softmax correct and *proven* exact. Can derive the recurrence and explain "never write S to HBM."
- **Strong:** + FA2 partitioning (R3) at 50–73% of A100 peak; the memory-bound→compute-bound shift demonstrated three ways (hand roofline, Nsight SoL, achieved TFLOP/s); causal masking and the backward pass correct.
- **Principal-grade:** the **FA3-class Hopper kernel** is real — warp specialization, TMA, WGMMA, and **ping-pong that demonstrably overlaps `exp` (SFU) with WGMMA (tensor cores) on the timeline** — at ≥70% of H100 peak, benchmarked head-to-head against FA3 and cuDNN with the gap diagnosed; **FP8 attention works** with block quantization + incoherent processing (you verified the orthogonal-`M` identity and measured the error reduction); a real variant (MLA/GQA/paged/sink) is integrated and verified; and you can **explain FA4/Blackwell cold** — the asymmetric-scaling thesis, exp-bound forward vs SMEM-bound backward, tcgen05/TMEM, conditional rescaling and emulated exp2 — *while correctly flagging which FA4 numbers are public (BF16: ≈1.6 PF, 71% — launch comparisons vs cuDNN 9.13/Triton now dated) and which (FP8/FP4) are not.* The differentiator is **knowing the bottleneck moves**: HBM (FA1) → partitioning/occupancy (FA2) → async overlap (FA3) → off the tensor cores entirely, onto exp/SMEM (FA4) — and engineering each kernel against *its* binding constraint, with the roofline and the profiler as proof, not the matmul as the goal.

---

## §10 References (see `references.md` for the full list)

- Dao, Fu, Ermon, Rudra, Ré. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.** arXiv:2205.14135 (2022). *(I/O-optimality, tiling, recompute-backward.)*
- Dao. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.** arXiv:2307.08691 (2023). *(Defer rescale, seq-dim parallelism, split-Q.)*
- Shah, Bikshandi, Zhang, Thakkar, Dao et al. **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision.** arXiv:2407.08608 (2024). *(Warp-spec, TMA, WGMMA, ping-pong, FP8 + incoherent processing; 740 TFLOP/s FP16, ~1.2 PFLOP/s FP8.)*
- Zadouri, Hoehnerbach, Shah, Thakkar, Dao et al. **FlashAttention-4** — arXiv:2603.05451 (2026-03-05) + blog tridao.me/blog/2026/flash4 + `pip install flash-attn-4` (CuTe-DSL, JIT). *(Asymmetric scaling; tcgen05/TMEM; conditional rescale; emulated exp2; ≈1.6 PF BF16 = 71%. Caveats re-verified 2026-07-30: current cuDNN (9.24) matches it; Hopper decode regresses vs FA3 (no SplitKV); FP8/FP4 numbers still not public.)*
- Milakov, Gimelshein. **Online normalizer calculation for softmax.** arXiv:1805.02867 (2018). *(The online-softmax recurrence — the engine.)*
- Ainslie et al. **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.** arXiv:2305.13245 (2023). · Shazeer. **Fast Transformer Decoding (MQA).** arXiv:1911.02150 (2019).
- DeepSeek-AI. **DeepSeek-V2 (MLA).** arXiv:2405.04434 (2024). *(Latent KV, weight absorption, decoupled RoPE.)*
- Yuan et al. **NSA (Native Sparse Attention).** arXiv:2502.11089 (2025). · DeepSeek-AI. **DeepSeek-V3.2 (DSA — lightning indexer + top-k selection under MLA).** arXiv:2512.02556 (2025-12).
- Xiao et al. **Efficient Streaming Language Models with Attention Sinks (StreamingLLM).** arXiv:2309.17453 (2023). · Beltagy et al. **Longformer.** arXiv:2004.05150 (2020).
- Kwon et al. **PagedAttention / vLLM.** arXiv:2309.06180 (2023). · Ye et al. **FlashInfer.** arXiv:2501.01005 (MLSys'25 best paper). · **FlexAttention** (PyTorch).
- NVIDIA. **CUTLASS / CuTe / CuTe-DSL** docs; **cuDNN 9.x** fused attention; Hopper & Blackwell architecture whitepapers (WGMMA, TMA, tcgen05, TMEM). See `00_foundations.md`.
- Book: **CUDA for Deep Learning** (MEAP), **Ch 8 — Flash Attention** (listings 8.1–8.11; the naive 3-kernel baseline and the fused WMMA kernel this assignment builds on).
