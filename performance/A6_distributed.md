# A6 — Distributed Training & Inference as One Communication Problem

> **Book chapter:** 10 (Distributed computing) · **Frontier thesis:** Scaling is not "more GPUs run the same code faster." It is a *communication-placement problem* on a hierarchy that spans two orders of magnitude in bandwidth (NVLink 900 GB/s–1.8 TB/s intra-node vs InfiniBand ~50–100 GB/s/GPU inter-node). Every modern parallelism scheme — DP/ZeRO, TP, PP, EP, CP — is a different answer to "which tensor do I split, and which wire pays for it?" You will build each from NCCL primitives, compose them into one DeviceMesh, and prove that the only scoreboard that matters is **Model FLOPs Utilization (MFU)** at scale. The spine: **map the chattiest collective onto the fattest link, then hide what's left under compute.**
>
> **Primary hardware:** 1× 8×H100 SXM node (NVLink/NVSwitch, 3.6 TB/s bisection) for rungs R0–R2; a **2-node, 16-GPU cluster** over InfiniBand NDR for R3 + §4. This assignment *requires* multi-GPU rental — see the `00_foundations.md` rental plan (spot 8×H100 for single-node; reserved 2-node for the multi-node core; budget the cross-node hours, they are the expensive part). · **Est. time:** 2–2.5 weeks · **Prereqs:** book ch1–9; A1 (inference-as-a-system) and A5 (quantization) help; the benchmarking discipline in `00_foundations.md` (locked clocks, warmup discard, CUDA-event timing) is mandatory here because noisy multi-GPU timings will lie to you.

---

## §0 Why this matters

A frontier training run is a distributed system that happens to do linear algebra. Llama-3 405B trained on **16,384 H100s for 54 days** and ate **466 interruptions** — roughly one every three hours — yet held **>90% effective training time** through automation [U: per-category failure integers below predate re-verification]. At that scale the difference between 38% and 50% MFU is *months of wall-clock and tens of millions of dollars*. The lever is almost never the matmul — cuBLAS/CUTLASS already hit 70–80% of peak on a single GPU (A3). The lever is whether your communication overlaps compute, whether your pipeline bubble is small, and whether you mapped each collective onto a wire that can carry it.

The book's chapter 10 builds the muscle honestly. Its tensor-parallel benchmark hits **~100% scaling efficiency on one 8×H100 node** (NVLink), then loses a **~28% per-GPU tax the instant the workload crosses a node boundary** (NVLink 600 GB/s vs InfiniBand 25 GB/s = ~24× slower) — even though aggregate efficiency still reads 99.8% because the AllReduce is tiny relative to the GEMM. Its pipeline goes from **27% efficiency (naive, blocking sync) to 4.17× / "104%" (CUDA streams + events overlap)**. And it ends on the load-bearing admission: **scaling efficiency is ~90%+ to 8 GPUs, 70–80% at 16, and 50–70% at 32+ GPUs depending on the compute-to-communication ratio.** That decay *is* the subject. This assignment takes you from the book's MPI+cuBLAS+NCCL foundation to the 2025-era frontier: multi-node expert-parallel all-to-all with compute/comm overlap, measured against the bandwidth cliff, with the fault tolerance that makes a 54-day run survivable.

---

## §1 Learning objectives

You can:

1. **Map** a multi-GPU machine (`nvidia-smi topo -m`: `NV*`/`PIX`/`SYS`), validate every link against line rate with `nccl-tests` (busbw vs algbw), and state the intra-node↔inter-node **bandwidth cliff** in numbers for your hardware.
2. **Derive** the memory and communication cost of each parallelism axis from first principles: ZeRO's `16Ψ`-byte optimizer-state accounting and its 1/N sharding; Megatron TP's 2-AllReduce-per-layer; GPipe/1F1B's `(p−1)/m` bubble; EP's all-to-all; CP's ring/all-to-all.
3. **Implement** the build ladder from NCCL primitives: single-node AllReduce → TP GEMM with AllReduce → 1F1B pipeline with stream/event overlap → multi-node EP all-to-all.
4. **Compose** DP×TP×PP(×EP×CP) as a single `DeviceMesh`, and **justify the placement** — TP/EP on intra-node NVLink (cap TP≤8), PP/DP across nodes on IB — by the cliff.
5. **Compute and measure MFU** (`C ≈ 6ND`, no-recompute numerator) and **HFU** (with recompute), diagnose the four MFU killers (unoverlapped comm, pipeline bubble, memory-bound kernels, MoE imbalance/stragglers), and reproduce the book's honest scaling-efficiency curve.
6. **Overlap** communication with compute: TP comm decomposed into p2p ring steps pipelined against GEMM tiles (Async-TP / TransformerEngine userbuffers), and EP all-to-all hidden under the DualPipe schedule.
7. **Make a run survivable:** async distributed checkpointing (DCP) and elastic restart (`torchrun --max-restarts`), and reason about the failure-rate arithmetic that forces them.

---

## §2 First-principles theory

Everything below is one idea applied five times: **a parallelism axis trades a memory/compute split for a collective, and the collective's cost is `data ÷ link bandwidth`.** Pick the split so the biggest collective lands on the fastest link.

### 2.1 The interconnect hierarchy — the ground truth (book §10.1, §10.4)

The book's first lesson is *map the hardware before writing a line of distributed code* (Lab 1: `nvidia-smi topo -m`). The hierarchy, with 2026 numbers:

| Link | Bandwidth (bidirectional) | Scope |
|---|---|---|
| Registers / shared mem | ~20 / ~15 TB/s | per-thread / per-block |
| L2 cache | ~10 TB/s | per-GPU |
| HBM3/3e (global) | 3.35–4.8 TB/s | per-GPU |
| **NVLink** gen3 (A100) / gen4 (H100/H200) / gen5 (B200) / gen6 (Rubin) | **600 / 900 / 1800 / 3600 GB/s** | intra-node, GPU↔GPU |
| NVSwitch bisection (HGX H100 8-GPU) | 3.6 TB/s | intra-node, non-blocking all-to-all |
| **InfiniBand** NDR 400 Gb/s / XDR 800 Gb/s | **~50 / ~100 GB/s per GPU** | inter-node |
| Commodity Ethernet | ~12.5 GB/s | inter-node (avoid for training) |

**The cliff:** intra-node 900 GB/s–1.8 TB/s vs inter-node ~50–100 GB/s/GPU is a **9–18× drop** (up to ~36× for B200 NVLink vs NDR). The book frames it as NVLink 600 GB/s vs IB 25 GB/s ≈ 24× for the A100/HDR generation, and as NVLink being **~48× faster** than the ~12.5 GB/s naive cross-node path. GB200 NVL72 collapses the cliff *within* a 72-GPU NVLink domain (130 TB/s aggregate all-to-all, "acts as one GPU"), which is exactly why the frontier is buying it. **Rule of the assignment: move data over NVLink frequently, over IB sparingly, over Ethernet never.**

### 2.2 NCCL: the collectives and their bandwidth model (book §10.1.3–4)

Six core collectives (book) plus the one MoE needs: **AllReduce, Broadcast, Reduce, Gather, ReduceScatter, AllGather, and AllToAll.** The identity you must hold: **AllReduce = ReduceScatter + AllGather** (NCCL implements it exactly this way for large messages). FSDP/ZeRO live on the right-hand side of that identity.

**Two bandwidths, never confuse them:** `algbw = size / time` (what you'd naively divide); `busbw` normalizes by the algorithm's unavoidable data movement so you can compare against line rate. For ring AllReduce, `busbw = algbw × 2(n−1)/n`. Ring AllReduce is **bandwidth-optimal** — time `t = (S/B)·2(n−1)/n`, every link active simultaneously — but its latency is **O(n) steps** (`2(n−1)` for AllReduce), which caps it at hundreds of GPUs. **Tree** (double-binary, NCCL 2.4+) gives **log-latency** — NCCL reports a "180× latency improvement at 24,576 GPUs on Summit" — and wins for small/medium messages at scale. **NVLS** (NVLink SHARP, `NCCL_ALGO=NVLS`) does the reduction *inside* the 3rd-gen+ NVSwitch; **SHARP** (IB/NVSwitch in-network reduction) roughly **2× internode AllReduce BW** by aggregating up the switch tree once and multicasting down. NCCL picks ring/tree automatically by message size (book: tree <256 KB, ring >1 MB, hybrid between) — your job is to *validate* the choice with `nccl-tests` and tune only when busbw sits below line rate.

> Tuning knobs you will actually touch: `NCCL_ALGO` (Ring/Tree/NVLS), `NCCL_PROTO` (LL/LL128/Simple — LL128 needs NVLink), `NCCL_BUFFSIZE`, `NCCL_IB_HCA`, `NCCL_NET_GDR_LEVEL` (GPUDirect RDMA), `NCCL_MIN/MAX_NCHANNELS`. Always end with: does measured busbw approach line rate? If not, the algorithm/protocol/topology is wrong, not the model.

> **2026 addition — the NCCL device API (2.28+) and symmetric memory.** Communication has moved *on-device*: NCCL's device-side API (three modes: **LSA** load/store access within a node, **Multimem** NVLink-multicast ops, **GIN** GPU-initiated networking for internode, over **symmetric memory**) lets a CUDA kernel issue and complete collectives *from inside the kernel* — no host round-trip, no launch gap on the critical path. This is the substrate under DeepEP-style fused comm/compute kernels and their "~0-SM overlap" claims; NVSHMEM 3.x is the lower-level PGAS form. Know the three modes and *why* comm moved on-device: at decode-size messages, launch/sync overhead dominates — the same wall A1's CUDA-graph rung attacks.

### 2.3 Data parallel + ZeRO — the memory accounting you must write from memory (arXiv:1910.02054)

Mixed-precision Adam keeps, **per device**, model state of **`16Ψ` bytes** for `Ψ` parameters: `2Ψ` (fp16 params) + `2Ψ` (fp16 grads) + `12Ψ` optimizer (fp32 master weights `4Ψ` + fp32 momentum `4Ψ` + fp32 variance `4Ψ`). Plain DDP replicates all `16Ψ` on every GPU and does **one AllReduce of gradients per step** (book's data-parallel section: 7 GB/iter for a 1B model, ~7 ms on 8×H100 NVLink). ZeRO shards this across `N` data-parallel ranks:

| Stage | What it shards | Per-device model-state bytes | Comm vs DDP |
|---|---|---|---|
| **ZeRO-1** | optimizer states | `4Ψ + 12Ψ/N` | same (AllReduce) |
| **ZeRO-2** | + gradients | `2Ψ + (2Ψ+12Ψ)/N` | same volume; AllReduce → **ReduceScatter + AllGather** |
| **ZeRO-3** | + parameters | `16Ψ/N` | **~1.5×** (params AllGather'd on demand fwd & bwd) |

**FSDP2/DTensor** (PyTorch's modern realization of ZeRO-3): per-parameter **dim-0 sharding** — each parameter is a `DTensor` with placement `Shard(0)` on a `DeviceMesh`. Pre-forward and pre-backward hooks **AllGather** the shard into the full parameter; post-hooks **free** it; gradients are **ReduceScatter**'d back to shards. Because every parameter is independently a DTensor, sharded state dicts are *communication-free* to save/load. The book reaches FSDP from exactly this angle (shard `56 GB → 7 GB` per GPU for a 7B model on 8×H100, at ~3× the communication of plain DP). FSDP **prefetch** (`forward_prefetch`, `backward_prefetch`, `limit_all_gathers`, a separate CUDA stream for the *next* AllGather) is what keeps the extra comm hidden.

### 2.4 Tensor parallel — Megatron's conjugate trick (arXiv:1909.08053; book §10.2)

Split a single layer's weights across GPUs, compute locally, combine. The book demonstrates the pattern (column-shard the weight, local cuBLAS GEMM, AllReduce to sum partial outputs) and is explicit that its benchmark allocates the full matrix for simplicity — a production TP shards `K×(N/8)` per GPU and uses `ncclAllReduce(ncclSum)`. The real Megatron MLP:

- **GEMM-1 column-parallel:** split the weight's *output columns* across GPUs → each computes a slice of the hidden activation with **no communication** (the GeLU is element-wise, stays local).
- **GEMM-2 row-parallel:** split the weight's *input rows* to match → each produces a partial output; **one AllReduce** sums them.

Formalized as two **conjugate operators**: `f` (identity in forward, AllReduce in backward) and `g` (AllReduce in forward, identity in backward). Net cost: **2 AllReduce in forward + 2 in backward per transformer layer** (one for the MLP block, one for attention). Because that AllReduce is on the critical path of *every* layer, **TP must live intra-node on NVLink, and you cap TP ≤ 8** (a single node). Push TP across the cliff and the AllReduce tax destroys you — this is the book's 28% per-GPU drop, made structural.

### 2.5 Pipeline parallel — bubbles and the schedules that shrink them (arXiv:1811.06965; book §10.3)

Assign *different layers* to different GPUs; a microbatch flows stage→stage. The book's naive version (blocking `cudaDeviceSynchronize` after each stage) runs **one batch through the whole pipeline at a time → 27% efficiency** (only 1 of 4 GPUs active per timestep). The fix is **concurrency via CUDA streams + events**: give each microbatch its own stream, signal the next stage with `cudaEventRecord`/`cudaStreamWaitEvent` instead of a global sync, and only synchronize once at the very end → a **staircase** where all stages stay busy → **4.17× / "104%"** (the >100% is overlap of H2D transfer with compute, not magic). That is the from-scratch seed of every production pipeline schedule:

- **GPipe:** `m` microbatches, bubble fraction **`(p−1)/m`** (`p` stages). Stores **all `m`** activation sets → memory blows up with `m`.
- **1F1B:** same `(p−1)/m` bubble, but interleaves one-forward-one-backward so in-flight activations are bounded to **~p** (not `m`) — the standard.
- **Interleaved 1F1B:** assign `v` *virtual* stages per device → bubble shrinks by factor **`v`**, at **`v×` the p2p communication**. The throughput-vs-comm knob.

The boundary activations a pipeline ships are *small* (one microbatch's layer output), so **PP rides inter-node IB comfortably** — it is the axis you spend the cliff on.

### 2.6 Expert parallel — the all-to-all axis (arXiv:2412.19437)

MoE routes each token to a few of many experts. With experts sharded across GPUs, a layer is: **all-to-all dispatch** (send each token to its expert's GPU) → expert GEMM → **all-to-all combine** (return results). The **capacity factor** (tokens-per-expert budget; drop or pad the overflow) governs the memory/quality trade-off [U: typical ranges ~1.0–2.0 / 1.25 for train, ~2.0+ for eval cited variously — re-verify before quoting a number]. DeepSeek-V3 uses **256 routed experts + 1 shared, 8 routed active per token**, with **auxiliary-loss-free** load balancing (a learned per-expert bias nudges routing instead of a balancing loss term). All-to-all is the chattiest collective in the stack, so **EP belongs intra-node on NVLink** wherever possible; when it must cross nodes, you *hide* it under compute (§2.8).

### 2.7 Sequence / context parallel — splitting the sequence (arXiv:2310.01889, arXiv:2309.14509)

For long context, the activations and KV themselves don't fit. Two designs:

- **Ring Attention:** shard the sequence across GPUs; each holds a query block and **rotates K/V blocks ring-style**, overlapping the block transfer with the local attention compute. Memory per GPU drops with the shard; communication hides under the flash-attention math.
- **DeepSpeed-Ulysses:** an **all-to-all transpose** that swaps the partition from sequence to head dimension before attention and back after — cheap, but **capped by the number of KV heads** (you can't split into more shards than heads, a hard ceiling with GQA/MQA).

### 2.8 Composition (3D/4D/5D) and overlap — where principal judgment lives

Real runs compose axes as **one DeviceMesh**: `DP × TP × PP (× EP × CP)`. Placement is dictated by the cliff:

- **TP** → innermost, **intra-node NVLink** (2 AllReduce/layer, TP≤8).
- **EP** → MoE layers, **intra-node NVLink** (all-to-all).
- **PP** → **inter-node IB** (small boundary p2p).
- **DP/ZeRO** → outermost (one overlappable gradient sync per step).

Two real topologies anchor it: **Llama-3 = 4D TP×CP×PP×DP** (arXiv:2407.21783); **DeepSeek-V3 = PP×EP×ZeRO-1, with NO TP at all** (it leans on MLA to shrink KV and on EP+DualPipe instead). There is no single right answer — the mapping is a function of the model (dense vs MoE), the context length, and the interconnect.

**Overlap is the second half of the spine.** Even correctly placed, comm on the critical path caps MFU. The frontier techniques:

- **TP overlap (Async-TP / userbuffers):** decompose the TP AllGather/ReduceScatter into **p2p ring steps pipelined against GEMM tiles** (TransformerEngine `tp_comm_overlap`); **FLUX** fuses collective + GEMM into a single kernel (arXiv:2406.06858).
- **DualPipe (DeepSeek-V3):** a **bidirectional** pipeline fed from both ends; each chunk is split into **4 components — attention, all-to-all dispatch, MLP, all-to-all combine** — and forward/backward compute is overlapped with comm so the **cross-node EP all-to-all is fully hidden**. Bubble `(PP/2 − 1)(F&B + B − 3W)`; cost is **2× params/device** (two pipeline copies). (github.com/deepseek-ai/DualPipe)
- **DeepEP (DeepSeek MoE all-to-all kernels):** *normal* kernels do **NVLink→RDMA forwarding** (intranode EP8 ~153 GB/s NVLink; internode EP64 ~51 GB/s RDMA — **[U: these predate a documented +30% optimization, re-verify]**); *low-latency* pure-RDMA decode kernels (dispatch ~77→194 µs from EP8→EP256) use **hook-based overlap that consumes ZERO SMs**, so comm steals no compute. Built on **NVSHMEM** (PGAS one-sided put/get *from inside* CUDA kernels — removes kernel launch/sync from the critical path). (github.com/deepseek-ai/DeepEP)
- **TorchTitan MXFP8 + DeepEP (training-side proof point, 2026):** PyTorch's TorchTitan ran a DeepSeek-V3-style pretraining recipe at **+41% throughput on B200** by pairing MXFP8 GEMMs with DeepEP expert-parallel comms and async-TP — this section's overlap techniques composed with A5's low-precision, on a vendor-neutral stack. (pytorch.org blog; [FACT] 2026-07.)

### 2.9 MFU — the scoreboard (arXiv:2204.02311; arXiv:2205.05198)

**MFU = observed throughput ÷ theoretical-peak throughput**, where the numerator counts only the FLOPs the model *mathematically needs* (no activation recompute), via `C ≈ 6ND` (6 FLOPs per parameter per token: 2 fwd + 4 bwd). **HFU** (Hardware FLOPs Utilization) *includes* recompute, so **HFU ≥ MFU** always; the gap is your recomputation overhead. Realistic anchors to calibrate against:

| Model / scale | MFU | HFU | Source |
|---|---|---|---|
| PaLM 540B | 46.2% | 57.8% | arXiv:2204.02311 |
| Megatron 1T | 56.3% | — | arXiv:2104.04473 |
| Llama-3 405B (BF16, 8K ctx, 8K GPUs) | **43%** | — | arXiv:2407.21783 |
| Llama-3 405B (16K GPUs) | ~41% | — | " |
| Llama-3 405B (131K context) | ~38% | — | " |

**Reality:** well-tuned dense at moderate scale hits **50–60% MFU**; **35–45% is common**; MoE typically lower (imbalance + all-to-all). **What kills MFU:** (1) **unoverlapped communication**, (2) **pipeline bubbles** `(p−1)/m`, (3) **memory-bound kernels** (norms, element-wise — A2), (4) **small per-GPU batch**, (5) **MoE imbalance / token-drop**, (6) **stragglers / failures**. The book's honest scaling-efficiency curve (**~90%+ to 8 GPUs, 70–80% at 16, 50–70% at 32+, depending on compute-to-comm ratio**) is the macro view of the same six forces. MFU is the single number your design note leads with.

### 2.10 Fault tolerance — the arithmetic that forces checkpointing

At 16,384 GPUs the **mean time between failures is hours, not days.** Llama-3's run logged **466 interruptions over 54 days** (~1 per 3 h), **GPU-related ~58.7%** of root causes [U: per-category integers], and still delivered **>90% effective training time** via automation. The tools:

- **PyTorch DCP (Distributed Checkpoint):** each rank writes its **local shard in parallel**, tagged with a `ShardedTensor`/`DTensor` descriptor so the checkpoint can be **re-sharded on load** (resume on a different parallelism layout).
- **Async checkpoint:** stage to CPU, flush on a **background thread** → ~**6× faster** wall-clock than synchronous.
- **In-memory / peer checkpoint** at scale: Gemini (SOSP'23) reports **>13× recovery** speedup; CheckFreq (FAST'21) auto-tunes frequency.
- **Elastic restart:** `torchrun --max-restarts` + **hot spares** so a single failed rank doesn't kill the job.

### 2.11 Orchestration — how the job launches (book §10.5 SLURM)

- **torchrun:** sets `RANK` / `LOCAL_RANK` / `WORLD_SIZE`; `--nnodes`, `--nproc-per-node`, `--max-restarts`, `--rdzv-backend c10d` (rendezvous). The modern default.
- **SLURM** (book's cluster manager): `sbatch --nodes=2 --ntasks-per-node=8 --gres=gpu:8`; `srun --mpi=pmix`. Production pattern = **one torchrun per node**, `SLURM_PROCID` → `node_rank`, `MASTER_ADDR` from the first node.
- **MPI** (book's launcher): bootstraps the processes (`MPI_Init`, `cudaSetDevice(rank % 8)`); **NCCL still does the GPU collectives**. Fine for ≤4-node experiments; SLURM owns shared clusters.

> **RL trainer↔inference bridge** (where this connects to A1's serving world): veRL/HybridFlow (arXiv:2409.19256) — colocated vs disaggregated actors, with **weight resharding every step** because the *training* sharding ≠ the *inference* sharding (~**140 GB/step** for a 70B model, up to **36.4% of iteration time**); rollout can be **>90% of RL runtime** with long-tail response stragglers. NeMo-Aligner (arXiv:2405.01481) refits TRT-LLM weights in place. Resharding is the distributed-systems crux of modern RLHF. 2026 practice has largely moved past disk round-trips: serving engines expose **sleep/wake + in-place weight sync** (NCCL broadcast or CUDA-IPC from the trainer straight into the engine's weights), and async-RL frameworks (AReaL) trade bounded staleness for rollout saturation.

---

## §3 The from-scratch build ladder

Each rung adds one parallelism axis, built from NCCL primitives, validated for **correctness** (numerical match to a single-GPU reference) and a **performance metric** before you promote it. Keep the engineering journal (`00_foundations.md`). **Cross-cutting, before you scale past one node:** wire in **async DCP checkpoint + `torchrun --max-restarts`** so R3 failures are recoverable — at multi-node scale you *will* lose a rank.

**Rung 0 — Single-node NCCL AllReduce + the topology map (book §10.1).**
Run Lab 1 first: `nvidia-smi topo -m`, read off the `NV*`/`PIX`/`SYS` matrix, write down your node's NVLink bisection. Launch 8 ranks (`torchrun --nproc-per-node=8`), init the `nccl` backend, AllReduce a large tensor. *Mechanism:* one process per GPU, NCCL routes over NVLink automatically. *Metric:* **busbw from `nccl-tests` approaches NVLink line rate** (the book's "4–5 TB/s aggregate AllReduce" on 8×H100 is your target); sweep message size and **compare `NCCL_ALGO=Ring` vs `Tree` vs `NVLS`** — reproduce the crossover (tree wins small, ring wins large). *Done when* you can point at a plot and say "this collective is at X% of line rate, and here's why the algorithm switches at ~256 KB."

**Rung 1 — Tensor-parallel GEMM with AllReduce (book §10.2; Megatron arXiv:1909.08053).**
Build the Megatron MLP: **GEMM-1 column-parallel** (no comm) → element-wise GeLU → **GEMM-2 row-parallel** → **one AllReduce**. Use cuBLAS for the local GEMM (the book's `cublasHgemm`, ~700–800 TFLOPS/GPU on H100), `ncclAllReduce(ncclSum)` for the combine. Implement the `f`/`g` conjugate operators so the backward pass AllReduces correctly. *Correctness:* output is **numerically identical** (within fp tolerance) to the same MLP on a single GPU; assert **exactly 2 AllReduce in forward**. *Metric:* the book's **~100% single-node scaling efficiency** to 8 GPUs, and **communication < ~20% of step time** (if it's higher, your GEMM is too small to amortize the AllReduce — the TP-efficiency lesson). *Done when* TP-8 matches single-GPU bit-compatibly and comm is a thin slice of the step.

**Rung 2 — 1F1B pipeline with stream/event overlap (book §10.3; arXiv:1811.06965).**
Split a multi-layer MLP across GPUs (book: 4-layer MLP, one layer/GPU). First build the **naive blocking** version and reproduce **~27% efficiency** — *feel* the bubble. Then convert to **streams + events**: per-microbatch stream, `cudaStreamWaitEvent` for stage dependency, single final sync — reproduce the **~4.17× / staircase**. Then implement **1F1B** scheduling and, as the capstone of the rung, **interleaved 1F1B** (`v` virtual stages). *Correctness:* pipeline output matches the single-GPU forward. *Metric:* **measured bubble fraction vs the predicted `(p−1)/m`**, and the **factor-`v` bubble shrink** from interleaving; **watch activation memory** (GPipe-style storage of all `m` microbatches is the trap — show 1F1B bounding it to ~`p`). *Done when* your measured bubble tracks the formula and you can trade bubble for comm by turning the `v` knob.

**Rung 3 — Multi-node + expert-parallel all-to-all (book §10.4 extended to MoE).**
Move to the **2-node, 16-GPU IB cluster**. First reproduce the book's **16-GPU TP** result (`mpirun -np 16` / SLURM across two hostfile nodes): **99.8% aggregate efficiency but ~28% per-GPU TFLOPS drop** crossing the boundary — *measure the cliff yourself*. Then build the **MoE all-to-all** layer (dispatch → expert GEMM → combine), or wire in **DeepEP** kernels. **Compose** `DP × TP × PP × EP` as one DeviceMesh, keeping **TP and EP intra-node** and **PP/DP inter-node**. *Correctness:* MoE output matches a single-GPU dense-gather reference within routing tolerance (carry A1's "1e-6 → expert-flip" paranoia — validate routing against a CPU reference). *Metric:* **end-to-end MFU at 16 → 64 GPUs** and **all-to-all busbw vs RDMA line rate**; confirm the **placement survives the cliff** and that **overlap (DualPipe-style) hides the all-to-all** (compute timeline shows no all-to-all stall). *Done when* you can show MFU at scale and prove the chattiest collective is on the fattest available link with the rest hidden.

---

## §4 Frontier-2026 core (required)

This is the assignment's point. The book stops at single-node TP + a streamed pipeline + a 16-GPU TP demo. A 2025-era frontier system is the following, and you must **demonstrate it on the 2-node cluster**: a **multi-node expert-parallel all-to-all (DeepEP-style) with compute/comm overlap (DualPipe idea), measured against the NVLink→IB bandwidth cliff, reported as MFU at scale, made survivable with async checkpointing + elastic restart.**

**4.1 Multi-node EP all-to-all with overlap (the core build).**
Stand up MoE expert parallelism across the two nodes. Implement the **all-to-all dispatch/combine** yourself first (NCCL `AllToAll`), then integrate **DeepEP** (normal NVLink→RDMA-forwarding kernels for prefill-style throughput; low-latency pure-RDMA kernels for decode-style latency). Overlap the cross-node all-to-all with compute using the **DualPipe decomposition** (split each chunk into attention / dispatch / MLP / combine and interleave fwd/bwd compute with comm). *Demonstrate:* an `nsys`/`nsight-systems` timeline showing the all-to-all **fully hidden under compute** (the EP comm SMs are idle-or-zero, per DeepEP's hook-based design), and **all-to-all busbw vs the RDMA line rate** of your IB fabric. *This is the single highest-leverage MoE-scaling result; it is why DeepSeek-V3 trains economically without TP.*

**4.2 Map the cliff, quantitatively.**
Produce **the bandwidth-cliff plot**: AllReduce/all-to-all busbw **intra-node (NVLink)** vs **inter-node (IB)** across message sizes, annotated with line rates and the **9–18× (book's ~24×) ratio**. Then prove your *placement* respects it: show that moving TP or EP across the node boundary (deliberately mis-mapped) **collapses MFU**, and that the correct mapping (TP/EP intra-node, PP/DP inter-node) recovers it. *This plot is the evidence behind every placement decision in your design note.*

**4.3 MFU at scale — the scoreboard.**
Instrument `C ≈ 6ND` and report **MFU and HFU at 16, 32, and 64 GPUs** for your composed 4D run. Decompose the gap from peak into the **six killers** (§2.9): measure the pipeline bubble, the unoverlapped-comm fraction, the memory-bound-kernel time, the MoE imbalance/token-drop rate. *Demonstrate:* your measured curve lands inside the book's honest band (**~90%+ to 8, 70–80% at 16, 50–70% at 32+**) and you can attribute the decay to specific forces with numbers — not hand-waving. Calibrate your absolute MFU against the §2.9 anchors (you should be in the 35–50% range for a well-mapped run; if you're at 20%, find the unhidden collective).

**4.4 Compose the DeviceMesh.**
Express the whole run as a single `DeviceMesh` with named dims (`dp`, `tp`, `pp`, `ep`) and the parallelism plans hung off it (FSDP2 `Shard(0)` on `dp`, Megatron column/row plans on `tp`, pipeline schedule on `pp`, expert sharding on `ep`). *Demonstrate:* one config change re-maps the topology, and you can articulate — for *your* model and interconnect — whether you'd choose Llama-3's TP×CP×PP×DP or DeepSeek-V3's PP×EP×ZeRO-1-no-TP, and **why** (dense vs MoE, context length, the cliff).

**4.5 Make it survivable (cross-cutting, required).**
Add **async DCP checkpointing** (parallel sharded writes, CPU-staged background flush — show the ~6× speedup vs synchronous) and **elastic `torchrun --max-restarts`** with a hot spare. *Demonstrate:* kill a rank mid-run and recover from the last async checkpoint with **re-sharding on load** (resume under a *different* parallelism layout to prove the DTensor descriptors work). Tie it to the arithmetic: at your GPU count, what's the implied MTBF, and how much wall-clock does async checkpointing buy back? *This is the difference between a demo and a system that could survive 54 days.*

---

## §5 Correctness & numerics

- **TP must be bit-compatible** with the single-GPU reference (same math, different layout). The conjugate `f`/`g` operators are the classic bug site — a missing backward AllReduce silently corrupts gradients; assert the forward AllReduce count and diff gradients against single-GPU.
- **Pipeline output must match** the single-GPU forward exactly (greedy/fixed-seed). Off-by-one in microbatch indexing or a missed event dependency produces *plausible-but-wrong* outputs — the most dangerous failure.
- **MoE routing is tolerance-based, and you must understand why** (carry A1's lesson): a `1e-6` difference in the gate softmax/top-k can **flip which expert a token routes to**, and divergences compound. Validate routing against a CPU reference; track the **token-divergence rate** — a *low* rate is expected sparse-routing noise, a *high* rate is a real bug (and at scale, a load-balance failure).
- **Collective numerics:** AllReduce sum order is non-deterministic across runs (floating-point non-associativity) — your reference must tolerate it, and your loss curves must be robust to it. Don't chase a `1e-7` "regression" that's just reduction order.
- **Checkpoint round-trip must be exact:** save → load (ideally re-sharded onto a different layout) → assert parameters and optimizer state are identical. A silent re-shard bug surfaces as a loss spike *after* a resume — instrument for it.
- **Adversarial cases:** a rank that dies mid-step (does the collective hang or the watchdog fire?); a pipeline stage slower than the rest (straggler → bubble); an MoE batch that overflows capacity (token-drop path); a message size that straddles NCCL's tree↔ring crossover; resume on a different GPU count.

---

## §6 Profiling & performance

- **The headline plot — the bandwidth cliff:** `nccl-tests` busbw intra-node (NVLink) vs inter-node (IB) across message sizes, with line rates marked and the ratio annotated. Everything downstream references it.
- **MFU/HFU dashboard:** `C ≈ 6ND`-based MFU at 16/32/64 GPUs, the HFU gap (your recompute cost), and the decomposition into the six killers. This is what your design note leads with.
- **Overlap evidence:** `nsys`/Nsight Systems timeline showing communication **overlapped with compute** — the TP comm pipelined against GEMM tiles, the EP all-to-all hidden under the DualPipe schedule (DeepEP's comm consuming ~0 SMs). A timeline with a visible all-to-all *stall* is a failing grade.
- **Pipeline bubble:** measured bubble fraction vs `(p−1)/m`, and the factor-`v` shrink from interleaving, on the per-device occupancy timeline.
- **Scaling-efficiency curve:** reproduce the book's strong-scaling (fix problem, add GPUs) and weak-scaling (grow problem with GPUs) sweeps; land inside the **90%+ / 70–80% / 50–70%** band and attribute the decay.
- **busbw, not algbw, always.** Validate every collective against line rate before optimizing the model. Per `00_foundations.md`: locked clocks, discarded warmup, CUDA-event timing — multi-GPU noise will otherwise hand you a fictional speedup.

---

## §7 Stretch goals (competition-grade)

- **Full DualPipe:** implement the bidirectional schedule (4-component chunk overlap, `(PP/2−1)(F&B+B−3W)` bubble) and beat 1F1B's bubble at the same `p`, paying the 2× param/device cost — measure the trade.
- **NVSHMEM kernel:** write a one-sided put/get collective *from inside* a CUDA kernel (remove launch/sync from the critical path) and compare against the NCCL equivalent — the substrate DeepEP is built on.
- **Async-TP / FLUX fusion:** fuse a TP collective with its GEMM into a single kernel (arXiv:2406.06858) and show the comm cost vanish into the matmul.
- **SHARP / NVLS in-network reduction:** enable `NCCL_ALGO=NVLS` (or IB SHARP) and measure the **~2× internode AllReduce BW** from in-switch reduction.
- **Context parallel for long sequences:** add Ring Attention or Ulysses, push context to 131K, and reproduce the MFU drop the book/Llama-3 see at long context (43% → 38%).
- **RL resharding bridge:** stand up the veRL-style train→infer weight reshard (train sharding ≠ infer sharding) and measure the per-step reshard cost (~140 GB/step class) — the modern RLHF crux.
- **GB200/NVL72 thought experiment** (if you can rent one): show how a 72-GPU NVLink domain *collapses the cliff* and changes every placement decision.

---

## §8 Deliverables & definition of done

1. **Code:** the four rungs — single-node NCCL AllReduce harness; Megatron TP MLP (cuBLAS + AllReduce, `f`/`g`); 1F1B + interleaved pipeline (streams/events); multi-node EP all-to-all (hand-rolled + DeepEP) — composed as one **DeviceMesh** (`DP×TP×PP×EP`). Plus async DCP checkpoint + elastic `torchrun --max-restarts`. Each with its single-GPU oracle test.
2. **The measurement set:** the bandwidth-cliff plot; the MFU/HFU-at-scale dashboard with the six-killer decomposition; the overlap timelines (TP+GEMM, EP all-to-all hidden); the pipeline-bubble-vs-`(p−1)/m` plot; the strong/weak scaling-efficiency curves landing in the 90/70-80/50-70 band; the `nccl-tests` busbw-vs-line-rate validations.
3. **Design note (3–4 pages): "Which tensor did I split, which wire paid for it, and what's my MFU?"** Lead with the cliff and the MFU scoreboard. For each axis: the collective it costs, the link you mapped it to, the overlap that hides it, the before/after number. End with the honest scaling-efficiency decay and *which of the six killers* dominates at 64 GPUs — with evidence. Written for a frontier-lab peer who has debugged a stalled all-to-all at 2 a.m.

**Done when:** every rung matches its single-GPU oracle; the bandwidth cliff is mapped in numbers for your hardware; multi-node EP all-to-all works with comm provably overlapped (zero-SM, no timeline stall); MFU is reported at 16/32/64 GPUs and lands in a defensible range with the gap attributed to specific killers; the placement (TP/EP intra-node, PP/DP inter-node) is justified by the cliff and shown to collapse when violated; a killed rank recovers from async checkpoint with re-sharding on load; and the design note would survive review by someone who has shipped a multi-node training run.

---

## §9 Principal-level rubric

- **Pass:** single-node TP and a streamed pipeline work and match the single-GPU reference; you can map the hardware (`nvidia-smi topo -m`) and validate AllReduce busbw against NVLink line rate; you can write the `16Ψ` ZeRO accounting and the `(p−1)/m` bubble from memory.
- **Strong:** + multi-node EP all-to-all working and measured; the DeviceMesh composes `DP×TP×PP×EP` with placement justified by the cliff; MFU reported at scale; pipeline bubble tracks the formula; async checkpoint + elastic restart recover a killed rank.
- **Principal-grade:** all of §4 demonstrated; **the cross-node EP all-to-all is provably hidden under compute** (DualPipe-style, zero-SM, no `nsys` stall) and busbw sits near RDMA line rate; **MFU-at-scale is your reporting metric** and you can *predict* the scaling-efficiency decay from the compute-to-comm ratio *before* measuring and have the prediction hold; you can defend why DeepSeek-V3 runs PP×EP×ZeRO-1 with **no TP** while Llama-3 runs 4D TP×CP×PP×DP, in terms of *your* model and interconnect; and your design note correctly attributes the remaining gap to peak (e.g., "we're at 41% MFU because the interleaved-1F1B bubble is 6% and the cross-node all-to-all isn't fully hidden on the combine phase — here's the timeline"). The differentiator is **systems judgment**: knowing the win is in *placement and overlap*, not the matmul — that you scale by mapping the chattiest collective onto the fattest link and hiding the rest, and that MFU is the only honest scoreboard.

---

## §10 References (see `references.md` for the full list)

- **ZeRO** (16Ψ accounting, stages 1–3): Rajbhandari et al., arXiv:1910.02054.
- **Megatron-LM** (TP, f/g conjugate operators): Shoeybi et al., arXiv:1909.08053; **Megatron 3D** (PTD-P, 1T MFU): Narayanan et al., arXiv:2104.04473.
- **GPipe** (pipeline, `(p−1)/m` bubble): Huang et al., arXiv:1811.06965; **activation recompute / HFU**: Korthikanti et al., arXiv:2205.05198.
- **PaLM** (MFU vs HFU): Chowdhery et al., arXiv:2204.02311.
- **Llama-3** (4D parallelism, MFU at scale, 466 interruptions): Grattafiori et al., arXiv:2407.21783.
- **DeepSeek-V3** (MoE, EP, aux-loss-free balancing, no-TP): DeepSeek-AI, arXiv:2412.19437; **DualPipe**: github.com/deepseek-ai/DualPipe; **DeepEP**: github.com/deepseek-ai/DeepEP.
- **Async-TP / FLUX** (collective+GEMM fusion): Chang et al., arXiv:2406.06858.
- **Ring Attention**: Liu et al., arXiv:2310.01889; **DeepSpeed-Ulysses**: Jacobs et al., arXiv:2309.14509.
- **veRL / HybridFlow** (RL resharding): Sheng et al., arXiv:2409.19256; **NeMo-Aligner**: arXiv:2405.01481.
- **NCCL 2.28+ device API** (LSA / Multimem / GIN + symmetric memory; NVIDIA developer blog) · **NVSHMEM 3.x** docs.
- PyTorch, **TorchTitan MXFP8 + DeepEP** (+41% DeepSeek-V3-style pretraining on B200, 2026) — cross-ref A5.
- **NCCL User Guide** (collectives, ring/tree, NVLS, tuning env vars); **nccl-tests** (busbw vs algbw); **HuggingFace Ultra-Scale Playbook** (the practitioner's companion to all of the above).
- Hardware: NVLink/NVSwitch & InfiniBand NDR/XDR data sheets; **GB200 NVL72** architecture brief. Cross-reference `00_foundations.md` for the interconnect-hierarchy table and the multi-GPU rental plan.
