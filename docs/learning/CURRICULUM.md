# Master Curriculum — mastery TOÀN BỘ repo, đúng thứ tự, cho senior AI research engineer

> **Đây là gì.** Một **con đường học DUY NHẤT, xếp đúng thứ tự**, hợp nhất *hai* lộ trình mastery của
> repo thành một chương trình liền mạch — để master mọi thứ theo trình tự một **senior AI research
> engineer ở frontier lab** thực sự dựng kiến thức: **make it work → make it fast → scale → feed →
> align → frontier → serve → prove.**
>
> - Nửa **MÔ HÌNH** — model HỌC thế nào: [`roadmap_model/`](roadmap_model/README.md) — 10 série (M1–M10), 48 Bài.
> - Nửa **TỐC ĐỘ** — model CHẠY NHANH thế nào: [`roadmap/`](roadmap/README.md) — 6 série (S1–S6), 41 Bài.
> - **Tổng: 16 série · 89 Bài.** File này KHÔNG lặp lại nội dung — nó **xếp thứ tự** 16 série đó và giải
>   thích *vì sao đúng thứ tự này*, rồi trỏ vào từng série. Học theo đây; mỗi série đọc theo thứ tự Bài
>   nội bộ của nó.

---

## Hai định luật tổ chức — đọc trước, giữ suốt hành trình

Cả chương trình treo trên hai câu, một cho mỗi nửa:

> **① Nửa mô hình (M):** *Một language model là một bản NÉN của phân phối dữ liệu, học bằng gradient
> descent trên next-token prediction.* Mọi lựa chọn — arch, optimizer, systems, data, RL — là **một đòn
> bẩy** lên *loss-per-FLOP* hoặc lên *việc gợi ra một năng lực*. Frontier là các đòn bẩy **hội tụ**;
> ablation (M10) **chứng minh** đòn bẩy nào gánh việc.
>
> **② Nửa tốc độ (S):** *Decode là memory-bandwidth-bound* (AI ≈ 1, cách ridge ~130×). Mọi kỹ thuật
> inference là **một đòn bẩy** kéo tok/s đo được về bức tường băng thông — hoặc nâng AI để dùng tensor
> core đang ngồi không. B=1 đi **16% → 53% → 77%** của bức tường khi bóc overhead.

Hai định luật gặp nhau ở đúng một chỗ: **các op bạn *dựng* ở nửa mô hình (matmul, softmax, norm,
attention) chính là các op bạn *tăng tốc* ở nửa tốc độ.** Đó là lý do thứ tự dưới đây đan hai nửa vào
nhau tại các mối nối tự nhiên, thay vì "học hết model rồi mới học perf".

---

## Nguyên tắc xếp thứ tự (vì sao đan xen, không nối khối)

| Mối nối | Quy tắc phụ thuộc |
|---|---|
| **Kernels (S3–S4) ngay sau model+training (M1–M4)** | "Make it work, then make it fast." Bạn vừa *dẫn xuất* matmul/softmax/norm/attention (M2–M3) — giờ xem chúng chạy ở tốc độ ánh sáng trên GPU. Op còn nóng hổi ⇒ khắc sâu nhất. |
| **Distributed serving (S6) kề distributed training (M5)** | Song sinh: M5 = data-parallel phía huấn luyện (DDP→ZeRO→FSDP); S6 = TP/PP/EP + ISA phía phục vụ. Cùng "trải ra nhiều GPU". |
| **Serving+quant (S1–S2, S5) SAU frontier arch (M9)** | MLA (S2 Bài 2.8) và MTP (S2 spec-decode) *là* thứ làm serving thú vị. Bạn phục vụ một model **frontier đã huấn luyện**, không phải toy random-weight. |
| **Ablation (M10) cuối cùng** | Capstone: cần MỌI thứ trước đó — model biết nói (M1–M9) + serving thật (S1–S2) để de-confound, rồi *chứng minh* F1–F9. |

Roofline xuất hiện **hai lần, đúng lúc**: mức-kernel ở **S3 Bài 3.0** (Stage 2, khi tối ưu compute) và
mức-decode ở **S1 Bài 1.0** (Stage 7, khi phục vụ). Không phải trùng lặp — hai mặt của cùng định luật ②.

---

## 🎓 Con đường — 8 chặng, 16 série, 89 Bài (học đúng thứ tự này)

### Chặng 0 · Định hướng
Đọc hai README lộ trình + hai định luật ở trên. Nắm giao thức teach-back (mục cuối). *(meta, không Bài)*

### Chặng 1 · MODEL — dựng model chạy được trên 1 GPU ("make it work")
| # | Série | Master điều gì | Neo (aha) |
|---|---|---|---|
| 1 | [M1 · Tokenizer](roadmap_model/M1_tokenizer.md) | byte→token: vì sao subword, thuật toán BPE merge, pretok regex, round-trip | round-trip = identity; loss-at-init ≈ log V |
| 2 | [M2 · Transformer forward](roadmap_model/M2_transformer_architecture.md) | attention as KV-retrieval, GQA, RoPE, RMSNorm, SwiGLU, QK-norm, tied-emb | causal-no-leak; loss-at-init ≈ log V |
| 3 | [M3 · Objective + optimization](roadmap_model/M3_objective_and_optimization.md) | cross-entropy/MLE, AdamW, cosine, **Muon** (Newton-Schulz), param-split | NS band [0.68,1.14]; RMS-match ≈0.2 |
| 4 | [M4 · Training loop](roadmap_model/M4_training_loop.md) | step wiring, **overfit-one-batch**, bf16 autocast, compile (F4), repro/monitors | overfit → <1e-2; val_bpb 0.02 (Phase-0) |
> **Qua Chặng 1 ⇒** bạn train được một model nhỏ end-to-end. Đây là "make it work".

### Chặng 2 · PERF — đưa compute về tốc độ ánh sáng ("make it fast")
| # | Série | Master điều gì | Neo (aha) |
|---|---|---|---|
| 5 | [S3 · CUDA-core kernels](roadmap/S3_cuda_core_kernels.md) | roofline harness (Bài 3.0) → GEMV/softmax/RMSNorm/TopK/GEMM = chính các op của M2/M3 trên GPU | 96–100% HBM; GEMM 134% cuBLAS-proxy |
| 6 | [S4 · Tensor cores + Flash](roadmap/S4_tensor_cores_and_flash.md) | SMEM→WMMA→mma.sync (matmul nhanh), online softmax, FA2 fwd+bwd = attention của M2 hợp nhất | 4.1%→81.9% cuBLAS; FA2 ~50% SDPA |
> **Qua Chặng 2 ⇒** bạn biết forward/backward chạy ở metal thế nào — và nghĩ bằng roofline. Nối tới M4 (MFU).

### Chặng 3 · Trải ra nhiều GPU
| # | Série | Master điều gì | Neo (aha) |
|---|---|---|---|
| 7 | [M5 · Distributed training](roadmap_model/M5_distributed_training.md) | DDP + ring all-reduce → ZeRO-1/2/3 → FSDP → activation-ckpt → comms algebra | DDP == single-proc; ZeRO memory 16Ψ |
| 8 | [S6 · Distributed + ISA](roadmap/S6_distributed_and_isa.md) | TP MLP (2 all-reduce/layer), 1F1B bubble, EP-MoE, MFU; WGMMA/FA3/tcgen05 | bubble (p-1)/m; MFU tái tạo PaLM 46.2% |
> **Song sinh:** M5 = phía huấn luyện, S6 = phía phục vụ/primitives. Cùng một mối: *ánh xạ collective chattiest lên link béo nhất.*

### Chặng 4 · Khoa học & nhiên liệu
| # | Série | Master điều gì | Neo (aha) |
|---|---|---|---|
| 9 | [M6 · Scaling laws](roadmap_model/M6_scaling_laws.md) | vì sao power-law, IsoFLOP fit, Chinchilla ~20 tok/param, budget planner | fitter a≈0.469/b≈0.531; C=6ND |
| 10 | [M7 · Data pipeline](roadmap_model/M7_data_pipeline.md) | vì sao data-quality thắng, filter → quality classifier → MinHash/LSH dedup | LSH collision-prob; discard accounting |
> **Qua Chặng 4 ⇒** bạn lên kế hoạch được một run compute-optimal trên data chất lượng.

### Chặng 5 · Gợi năng lực — post-training
| # | Série | Master điều gì | Neo (aha) |
|---|---|---|---|
| 11 | [M8 · Post-training / RL](roadmap_model/M8_post_training_rl.md) | SFT → policy-gradient → **GRPO/Dr.GRPO** → RLVR grader → DPO → rollout/KL seam | toy reward↑; R1 "aha" (F7 PREDICTION) |
> **Qua Chặng 5 ⇒** bạn biến base model thành model aligned/reasoning. (rollout seam trỏ tới serving ở Chặng 7.)

### Chặng 6 · Kiến trúc frontier
| # | Série | Master điều gì | Neo (aha) |
|---|---|---|---|
| 12 | [M9 · MoE · MLA · MTP](roadmap_model/M9_moe_mla_mtp.md) | vì sao sparse, router + aux-loss-free bias, MLA low-rank KV, MTP draft-head | router entropy >0.9·logN; KV 1152 vs 4096 B |
> **Qua Chặng 6 ⇒** bạn nắm spine DeepSeek/GLM. Chính MLA/MTP mở Chặng 7.

### Chặng 7 · PERF — phục vụ model frontier (nhanh & rẻ)
| # | Série | Master điều gì | Neo (aha) |
|---|---|---|---|
| 13 | [S1 · Serving substrate](roadmap/S1_serving_substrate.md) | decode-roofline (Bài 1.0), KVCache→Slot/Batched/**Paged**, Prefill/ChunkPrefill, paged kernel | paged 5.90 ms/step ×3.52; frag 5% |
| 14 | [S2 · Serving engines](roadmap/S2_serving_engines.md) | continuous batching, chunked-prefill (âm tính thật), **spec-decode ← MTP**, CUDA-graph, MLA, disagg | CUDA-graph 77% wall; spec ×1.2–1.4 |
| 15 | [S5 · Quantization](roadmap/S5_quantization.md) | INT8 sym/asym, group-INT4, NVFP4 vs MXFP4, FP8-KV, AWQ | INT8 40.5 dB; NVFP4 1.48×<MXFP4 |
> **Qua Chặng 7 ⇒** bạn phục vụ một model frontier ở hiệu suất production. (MTP của M9 = drafter của S2.)

### Chặng 8 · Đóng vòng & chứng minh frontier (CAPSTONE)
| # | Série | Master điều gì | Neo (aha) |
|---|---|---|---|
| 16 | [M10 · Close-the-loop + F1–F9 ablations](roadmap_model/M10_close_the_loop_and_ablations.md) | museum→model gap, nanochat speedrun, report-card oracle, **F1–F9 ablation AS derivation** | Phase-0 val_bpb 0.02; CORE≈0.256 (target) |
> **Qua Chặng 8 ⇒** bạn đã train một model biết nói VÀ chứng minh (iso-FLOP, pre-registered) đòn bẩy frontier nào gánh việc. Đây là artifact hireable.

---

## Đồ thị phụ thuộc (vì sao đan xen)

```
        M1 → M2 ───────────────► (op nóng hổi) ───► [PPPM Ch4/5/6/10/11] → S3 → S4   ┐ compute
              │  \                                              ↑ raw CUDA C++      │ kernels
              ▼   \                                             │ fundamentals       │
        M3 → M4    \                                     (roofline mindset)          │
              │     \                                                                   │
              ▼      ▼ (twin)                                                           ▼
        M5 ──────── S6 ← [5D Ch04 Ring Attn]                                     ┌──► M4 MFU
         │              ↑ insert between 6.2 and 6.3                              │
         ▼              │                                                         │
        M6 → M7 → M8 ──(rollout seam)───────────┐                                 │
                        │                       │ dùng                              │
                        ▼                       ▼ serving                            │
                       M9 (MLA,MTP) ──────► S1 → S2 → S5                            │ serving
                        │  MLA→S2.2.8  MTP→S2 spec-decode                          │
                        └──────────────► M10 (capstone) ◄──┘                        │
```
Mũi tên = "phải hiểu trước". Ba mối đan then chốt: **[PPPM]→S3** (raw CUDA C++ fundamentals BEFORE
Triton abstractions), **M2→S3/S4** (op→kernel), **M9→S1/S2** (arch→serving). Xem [`BOOK_CHAPTER_MAP.md`](BOOK_CHAPTER_MAP.md)
cho chi tiết từng chương sách chèn ở đâu.

---

## Cách dùng — giao thức mastery (một Bài một lần)

1. **Theo thứ tự chặng 1→8.** Trong mỗi série, đọc theo thứ tự Bài nội bộ của nó (M2: 2.1→2.6; S1: 1.0→1.7…).
2. **Mỗi micro-concept = PRR loop** (mặc định — "Master understanding — the PRR loop" của [`../../CLAUDE.md`](../../CLAUDE.md)): **Bắt đầu bằng PROBLEM/GOAL (không bắt đầu bằng lý thuyết) → Predict COLD** (Navigator viết dự đoán TRƯỚC mọi giải thích) → **Run** code thật (in shape/số/token, test red→green) → **Reconcile** chỉ chỗ lệch (Driver 3 dòng, không monologue) → **Re-derive + tự VẼ** ví dụ mới → **cổng teach-back + modify-and-predict** → frontier. KHÔNG đọc wall-of-text; internalize = Navigator *generate/run/draw*, không phải đọc. **Sách chỉ mở khi kẹt** — đọc đúng section sửa lỗi, không đọc cả chương trước khi build. Chunk nhỏ hơn một Bài.
3. **Cổng là thật:** chưa dạy lại được (trên ví dụ MỚI) thì chưa sang Bài sau. Qua cổng ⇒ tick ở study-queue [`INDEX.md`](INDEX.md). **Spaced callback:** mở session kế bằng 1 câu recall từ Bài ✅ trước.
4. **Trung thực số (FOP-4):** "Neo" đã gắn nhãn — invariant đã đo / số thật trong `bench/RESULTS.md` / PREDICTION pre-registered. Nửa mô hình phần lớn built+toy-tested (run thật rental-gated); nửa tốc độ sm120 đã đo. Đừng tin một con số chưa gắn nhãn measured — và trong PRR, *chính con số đo được* là cái đối chiếu với prediction của bạn.

---

## 📚 Sách & chương — đọc gì, ở đâu, vì sao → [`BOOK_CHAPTER_MAP.md`](BOOK_CHAPTER_MAP.md)

Trước mỗi chặng, consult `BOOK_CHAPTER_MAP.md` để biết chương sách nào **PHẢI đọc** (gap thật),
chương nào **SKIP** (đã owned sâu hơn trong code). Tóm tắt:

| Chặng | Book/Chapter | Verdict | Khi nào |
|---|---|---|---|
| **2 (S3)** | **PPPM Ch4/5/6** — SM arch, SMEM tiling, coalescing | **READ** (raw CUDA C++ fundamentals that Triton abstracted away) | **Before S3 3.1** (Triton kernels) |
| **2 (S3)** | **PPPM Ch10** — Reduction (`__shfl_down_sync`) | **READ + BUILD** `csrc/fundamentals/reduction_warp.cu` | **URGENT — k_live Level 2** |
| **2 (S3)** | **PPPM Ch11** — Prefix Sum (Kogge-Stone scan) | **READ + BUILD** `csrc/fundamentals/prefix_scan.cu` | After Ch10 |
| **2 (S3)** | PPPM Ch7 (halo pattern) · Ch9 (atomics) · Ch14 (stream compaction) · Ch15 (sparse/CSR) | **SKIM** — transfers to sliding-window attn, FA2 backward, KV eviction, MoE grouped GEMM | When relevant |
| **2 (S3)** | CUDA for DL ch6 | 🔒 Answer key — Phase 2 `/rebuild` only | During P0.5 rebuild rungs |
| **2 (S4)** | CUDA for DL ch7.3–7.5 (WGMMA/TMA) | ✅ Free reading (only book chapter still ahead of ledger) | Before P1 H100 day |
| **2 (S4)** | CUDA for DL ch8 (FA) | 🔒 Answer key | During P0.5 rebuild |
| **3 (S6)** | **5D Parallelism Ch04** — Sequence/Context Parallelism + Ring Attention | **READ ALL (70 min)** — the one genuine distributed gap | Between S6 6.2 and 6.3 |
| **3 (S6)** | 5D Parallelism Ch03 "TP vs ZeRO for Inference" | **SKIM (13 min)** | After S6 6.1 |
| **3 (S6)** | 5D Parallelism Ch05 "Megatron 3D" | **SKIM (14 min)** | After S6 6.2 |
| **7 (S5)** | CUDA for DL ch9 (Quant) | 🔒 Answer key | During P0.5 rebuild |

## 🔮 2026 Frontier Extensions → [`FRONTIER_2026_EXTENSION.md`](FRONTIER_2026_EXTENSION.md)

89-Bài là **nền text-LLM**. Frontier 2026 mở rộng 4 hướng (Stage 9+, additive không thay thế):

| Frontier | Là gì | Loại | Tier |
|---|---|---|---|
| **A: Multimodal** (M11/S7) | VLM (ViT+LLaVA), DiT diffusion, video, audio — kết nối trực tiếp M2 Transformer + S5 quant | 🔨 Build ViT; ⚪ know DiT | ★ 2026 differentiator |
| **B: Reasoning AI** (S2.10) | Test-time compute, parallel sampling, search-at-inference, token budget | ⚪ know-it-discuss | ★ 2026 differentiator |
| **C: Agentic Systems** (mock) | Multi-turn tool use, sandboxing, safety stack, agent eval | Mock round `a-agentic` | ★ Anthropic core |
| **D: Physical AI** (mock) | GR00T VLA, Cosmos world models, sim-to-real, Jetson deployment | Mock round `k-physical` | ★ NVIDIA robotics |

## Neo tuyển dụng — mỗi chặng "ăn" gate phỏng vấn nào (join với [`FRONTIER_HIRING_MAP.md`](FRONTIER_HIRING_MAP.md))

Master từ first principles CHÍNH LÀ dựng artifact được tuyển. Mỗi chặng đóng một *universal interview
gate* + phô một *FOP trait*. Bản đồ đầy đủ (7 FOP · trait scorecard · gate · build-vs-know-it · scarce
bucket, đúng tới từng bit, nguồn = `CLAUDE.md §FOP` + `STRATEGY.md` + `README.md` + `FRONTIER_PRACTICE_2026.md`):
[`FRONTIER_HIRING_MAP.md`](FRONTIER_HIRING_MAP.md). Tóm tắt per-chặng:

| Chặng · Série | Gate phỏng vấn ăn được | FOP trait | Tier |
|---|---|---|---|
| 1 · M1–M4 | *MHA/Transformer from scratch trong 45'* | overfit/sanity gates (loss-at-init≈log V), spec-with-falsifiers | table-stakes |
| 2 · S3–S4 | *vì sao batch-1 decode memory-bound; roofline* | roofline-first/predict-the-number; measured≠theoretical | ★ kernels |
| 3 · M5+S6 | *train model 100B: DP+TP+PP + memory math* | comms discipline; lock evals before arch | table-stakes |
| 4 · M6+M7 | *derive scaling laws; compute-optimal N,D* | extrapolation honesty (a+b=1); dedup-scope judgment | table-stakes |
| 5 · M8 | *RLHF vs DPO; RL stability; GRPO advantage* | **mandatory RL logging**; own the KL toggle | **★ differentiator** |
| 6 · M9 | *MLA vs GQA tradeoff; aux-loss-free* | convergent-defaults reasoning | table-stakes |
| 7 · S1–S2+S5 | *KV-cache; quant; train/inference drift* | inference co-design; TCO≈electricity | **★ differentiator** |
| 8 · M10 | *what would falsify your result?* (capstone) | **spec-with-falsifiers (A+)** · ablate-one-var · **epistemic honesty (A+)** | ★ artifact |

**Trục bị tuyển = EXECUTION** (`STRATEGY.md`: planning ~90% vs execution ~25% — "committee hires on
execution"). Vì vậy PRR bắt mỗi micro-concept **chạm một con số chạy được**, không phải một đoạn văn — đó
cũng là vì sao FOP-1 = *Execution > analysis*. Portfolio ăn tiền = **repo green-CI + tests + design docs
giải thích được từ first principles** (`README.md`) — chính là `scratch_llm/`.

---

## Vì sao thứ tự này = "senior AI research engineer"

Một senior RE ở frontier lab không master "model" hay "systems" rời rạc — họ master **cả vòng, đúng
thứ tự nhân-quả**, và whiteboard được từng mắt xích lạnh:

- **Chặng 1–2** cho phản xạ *make-it-work-then-fast*: dựng transformer, rồi thấy chính op đó ở metal (roofline).
- **Chặng 3** cho tư duy *systems*: bộ nhớ/comms là ràng buộc thật, không phải FLOP.
- **Chặng 4–5** cho *research taste*: compute-optimal (M6), data quyết định (M7), RLVR gợi reasoning (M8).
- **Chặng 6–7** cho *frontier engineering*: MoE/MLA/MTP (M9) và cách phục vụ chúng (S1–S2, S5) — đúng thứ nan-labs làm 2026.
- **Chặng 8** là *artifact*: đóng vòng thành model biết nói và **chứng minh** đòn bẩy nào thật — kỷ luật pre-registered iso-FLOP chính là kỹ năng được tuyển.

Master hết 89 Bài theo thứ tự này ⇒ bạn defend được **byte → BPE → Transformer → optimizer → systems →
scaling → data → RL → MoE/MLA/MTP → serving → kernels → ablation** cold, từ first principles — mục tiêu
của [`../../CLAUDE.md`](../../CLAUDE.md) "The organizing principle".
