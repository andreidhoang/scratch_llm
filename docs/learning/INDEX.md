# Learning track — chuỗi bài mastery (tiếng Việt), trace code thật

> **Đây là gì.** Bài giảng chạy theo **PRR loop** — giao thức "Master understanding — the PRR loop"
> của repo (`CLAUDE.md` §How we build): mỗi micro-concept = **Predict COLD → Run code/số thật →
> Reconcile chỉ chỗ lệch → Re-derive + tự vẽ → teach-back gate + modify-and-predict**. KHÔNG đọc
> monologue rồi gật (fluency illusion — thấy hiểu nhưng không encode); Navigator *generate/run/draw*,
> Driver chỉ hỏi/chạy/lấp chỗ lệch. Mỗi bài mổ MỘT component đã build và đã đo — mọi trích dẫn là
> file/hàm/dòng thật, mọi con số là số đã log trong `bench/RESULTS.md` (và là số bạn đối chiếu với
> prediction của mình).
>
> **Luật dùng:** học tuần tự, mỗi micro-concept **predict COLD trước** (đừng mở đáp án gấp trong
> `<details>` cho tới khi đã viết dự đoán), chạy code thật, rồi **cổng teach-back** trên ví dụ MỚI —
> chưa dạy lại được thì chưa sang bài sau. Mở session kế bằng 1 câu recall từ bài ✅ trước (spaced).
> Đúng nhịp "one concept at a time".

## Quy ước

- **Số dòng pin theo commit** ghi ở đầu mỗi bài (code trôi thì số dòng trôi — tên
  file · class · hàm là mỏ neo chính, số dòng là tiện ích tra nhanh tại commit đó).
- Code giữ nguyên tiếng Anh (nó là sự thật trên đĩa); giải thích + chú thích tiếng Việt;
  thuật ngữ kỹ thuật (tensor, mask, slot, cache, kernel…) giữ tiếng Anh khi tự nhiên hơn.
- Mỗi bất biến nêu ra phải chỉ được **test nào găm nó** trong `tests/`.
- **Code-anchored bắt buộc (2026-07-05).** Mọi giải thích phải **mở đúng file `src/` và bám sát code THẬT
  y như đã viết** (không phiên bản lý tưởng hoá) — trace theo `file · hàm · dòng` ở HEAD hiện tại (số dòng
  trôi thì verify lại). Vì vậy mỗi lần giảng đồng thời là một lượt **review code**: vừa dạy vừa chỉ ra bug,
  edge-case thiếu, và chỗ tối ưu được trong chính dòng đang trace (nêu ra, đừng làm mượt bỏ qua) — học code
  VÀ cải thiện nó. Giọng: **senior AI research engineer ở frontier lab 2026** (first-principles, trade-off,
  measured>implied, đối chiếu frontier), không phải giọng tutorial.
- **Feynman deep-dive bắt buộc, tiếng Việt (2026-07-05).** Mỗi concept phải có phần **giải thích SÂU hơn,
  mở rộng, bằng tiếng Việt** theo đúng **kỹ thuật Feynman**: (a) phát biểu lại bằng lời thật đơn giản, (b)
  **dẫn xuất từ first principles VÌ SAO thiết kế/engineering lại như vậy** — vì sao cấu trúc NÀY chứ không
  phải phương án khác, (c) tự soi chỗ hổng trong lời giải thích rồi bịt lại. Đi **càng low-level, chi tiết
  càng tốt** — visualization sâu nhất có thể (bố cục byte/bit, shape+stride tensor chính xác, memory/
  data-flow, ví dụ số hand-trace), **trace qua code TỪNG bước** (`file · hàm · dòng`, theo đúng control
  flow đã viết). Chốt mỗi concept bằng **how/why/what một frontier AI lab (DeepSeek/GLM/Kimi/Qwen/OpenAI/
  Anthropic-tier) sẽ implement và giải thích** — biến thể production, trade-off họ tối ưu, khung interview.
  Chiều sâu + Feynman tiếng Việt là MẶC ĐỊNH, không phải phần thêm; ngắn gọn chỉ khi user yêu cầu.
- **Neo tuyển dụng (PRR bước 6).** Chốt mỗi Bài bằng linkage tuyển dụng theo
  [`FRONTIER_HIRING_MAP.md`](FRONTIER_HIRING_MAP.md) §8: (a) gate phỏng vấn ăn được, (b) FOP trait phô ra,
  (c) build-from-scratch hay know-it-discuss (nếu know-it: nêu *khung trade-off*, đừng code), (d) scarce
  bucket (RL/inference/kernels = differentiator, còn lại table-stakes). Master từ first principles = dựng
  artifact được tuyển; bị tuyển trên **EXECUTION/shipped**, không phải plan (FOP-1).

## ▶️ ĐANG Ở ĐÂU / HỌC GÌ TIẾP → [`PROGRESS.md`](PROGRESS.md) (sổ cái teach-back 89 Bài)

> **Con trỏ resume của nửa học** (song sinh với `performance/PERF_PLAN.md` của nửa build). Đầu file có
> dòng `Learning-node:` = Bài kế tiếp cần master — `session-start.sh` tiêm vào MỌI session mới, nên bất kỳ
> session nào (kể cả "não trắng") **pick up ngay** đúng chỗ, không re-derive Bài đã ✅. Qua cổng teach-back
> ⇒ tick ✅ + ngày + neo ở đó, advance `Learning-node:`. Đây là cái làm trải nghiệm học **liền mạch xuyên
> session**.

## 📚 BOOK CHAPTER MAP → [`BOOK_CHAPTER_MAP.md`](BOOK_CHAPTER_MAP.md)

> **Khi đọc sách nào, chương nào, vì sao.** Bản đồ duy nhất gắn 3 cuốn sách (CUDA for DL · 5D
> Parallelism · PPPM) vào đúng điểm chèn của curriculum. Mỗi chương: **READ** (gap thật) · **SKIM**
> (delta nhỏ) · **SKIP** (đã owned sâu hơn trong code). Load file này trước mỗi drill/mock để biết
> chương nào phải đọc TRƯỚC round đó.
>
> **Tóm tắt ưu tiên build (raw CUDA C++, PPPM):**
> 1. `csrc/fundamentals/reduction_warp.cu` ← PPPM Ch10 — k_live Level 2 (URGENT)
> 2. `csrc/fundamentals/tiled_transpose.cu` ← PPPM Ch5/6 — k_live Level 3
> 3. `csrc/fandamentals/prefix_scan.cu` ← PPPM Ch11 — scan classic
>
> **Gap duy nhất 5D Parallelism:** Ch04 (Sequence/Context Parallelism + Ring Attention) — chèn
> giữa S6 6.2 và 6.3.

## 🧭 PRE-FLIGHT before train GPU thật → [`CODEBASE_READING_ORDER.md`](CODEBASE_READING_ORDER.md)

> Cách một senior frontier-RE (hay Karpathy) **đọc cả codebase trước khi tiêu một GPU-hour**: đúng thứ
> tự file (theo dòng chảy của tensor), mỗi file kèm oracle cần verify + anchor `file · symbol · line`.
> Không phải derivation — là **orientation pass** trước run thật; ghép với `deploy/runbooks/frontier_gpu_day.md`.

## 🎓 BẮT ĐẦU Ở ĐÂY — Master Curriculum (thứ tự học đúng) → [`CURRICULUM.md`](CURRICULUM.md)

> **Một con đường DUY NHẤT, xếp đúng thứ tự**, hợp nhất cả hai lộ trình (model + performance) thành 8
> chặng · 16 série · **89 Bài**, theo trình tự senior AI research engineer dựng kiến thức: *make it work
> → make it fast → scale → feed → align → frontier → serve → prove.* Đan hai nửa tại các mối nối tự
> nhiên (kernels ngay sau model forward; serving ngay sau MoE/MLA/MTP; ablation là capstone). **Nếu chỉ
> đọc một file, đọc file này** — nó trỏ vào 16 série bên dưới theo đúng thứ tự.

## 🗺️ Hai lộ trình mastery — bản đồ học TOÀN BỘ repo, từ first principles

Repo có **hai nửa**, mỗi nửa một lộ trình song sinh (mỗi Bài: câu hỏi first-principles → Feynman → dẫn
xuất → **trace code thật** file·hàm·dòng → cổng teach-back → frontier). [`CURRICULUM.md`](CURRICULUM.md)
xếp thứ tự học; hai lộ trình dưới đây là nội dung. Đây là hàng đợi học của chế độ delegate
([ADR-0013](../adr/ADR-0013-execution-mode-full-delegation.md)) — code ship trước, học sau.

### 🧠 Lộ trình MODEL — LLM from scratch (CS336 + nanochat + frontier) — [`roadmap_model/README.md`](roadmap_model/README.md)
> **Nửa MÔ HÌNH**: model HỌC thế nào. 10 série (pin `4ad0ac5`): M1 tokenizer → M2 transformer → M3
> objective+optimization → M4 training loop → M5 distributed → M6 scaling laws → M7 data → M8 post-training/RL
> → M9 MoE·MLA·MTP → M10 close-the-loop + F1–F9 ablations. **DERIVATION-first + đối chiếu frontier nặng**
> (DeepSeek/GLM/Kimi/Qwen/nanochat — bản tham chiếu trong `.venv/.../transformers/models/`). Định luật:
> *model là bản nén học bằng gradient descent; frontier là các đòn bẩy hội tụ; ablation chứng minh đòn bẩy
> nào gánh việc.*

### ⚡ Lộ trình PERFORMANCE — serving + kernels — [`roadmap/README.md`](roadmap/README.md)
> **Nửa TỐC ĐỘ**: model CHẠY NHANH thế nào. 6 série (pin `9e61d7a`): S1 serving substrate → S2 engines →
> S3 CUDA-core kernels → S4 tensor cores+flash → S5 quantization → S6 distributed+ISA. Định luật: *decode
> là memory-bound; B=1 đi 16%→53%→77% của bức tường bộ nhớ.*

Study-queue bên dưới map vào các série này. Đọc MODEL trước (dựng model), PERFORMANCE sau (làm nó nhanh).

## Series 0 — Meta (cách hệ thống vận hành)

| # | Bài | Nội dung | Trạng thái |
|---|---|---|---|
| 00 | [Claude Code vận hành thế nào trên repo này](00-how-claude-code-works.md) | first principles: LLM không trạng thái → git+docs là bộ nhớ, hook là kỷ luật; chuỗi file một session mới đi qua để biết "làm gì tiếp" | ✅ |

## Series 1 — Serving substrate (A1 R3b + R4.1: continuous batching → PagedAttention)

Nền tảng đo lường: `bench/RESULTS.md` §2026-07-03 (R3.x, P4.1.x). Node curriculum:
`performance/PERF_PLAN.md`. Spec gốc: `performance/notes/A1_R3_continuous_batching.md`,
`performance/notes/A1_R41_paged_attention.md`.

| # | Bài | Component (file chính) | Trạng thái |
|---|---|---|---|
| 01 | [BatchedKVCache — bộ đệm slot + write-then-mask](serving/01-batched-kv-cache.md) | `model.py` · `SlotKVCache`/`BatchedKVCache` | ✅ |
| 02 | RoPE per-row positions + per-row mask trong attention | `model.py` · `RotaryPositionalEmbedding.forward`, `MultiHeadSelfAttention.forward` | ⬜ (viết khi dạy) |
| 03 | PrefillView — duck-typing: prefill = forward thường + hook ghi | `model.py` · `PrefillView` | ⬜ |
| 04 | serve() — evict→admit→decode + hợp đồng sở hữu graph/scheduler (3 trận dynamo-storm) | `serving/continuous.py` · `serve` | ⬜ |
| 05 | PagedKVCache — block table, allocator, trash block, CoW | `model.py` · `PagedKVCache` | ⬜ |
| 06 | Kernel Triton — online softmax trên từng block qua block table | `kernels/paged_decode_triton.py` | ⬜ |
| 07 | Phương pháp đo — trace 2 tải, 4 arms, 9.65→5.90 ms/step | `bench/continuous.py` | ⬜ |

Series sau (dự kiến, mở khi tới node): kernel ladder A2 (GEMV→GEMM), tensor cores A3, FA A4,
quantization A5 — theo `performance/PERF_PLAN.md`.

## Study queue — mastery debt của chế độ `delegate` (ADR-0013)

> **Đây là gì.** Từ 2026-07-03 repo chạy chế độ **`delegate`** (xem
> [ADR-0013](../adr/ADR-0013-execution-mode-full-delegation.md)): agent build toàn bộ, mastery học
> SAU qua code đã ship. Mỗi rung ship xong được **append một dòng vào bảng này** (rung → commit →
> file chính → con số headline). Đây là *hàng đợi học* cho các session `/master` sau — nợ mastery
> được TRACK, không bị bỏ rơi. Khi học xong một mục thì viết bài giảng tương ứng và tick ✅.
>
> Cột "Đã học": ⬜ chưa học · ✅ đã qua teach-back.

| Rung | Commit | File chính | Con số headline | Đã học |
|---|---|---|---|---|
| A1 R4.2 chunked prefill (mechanism + measured negative) | _(2026-07-04)_ | `model.py` ChunkPrefillView · `serving/continuous.py` prefill_chunk_size · `bench/chunked_prefill.py` | token-exact ✓; sequential-interleave REGRESSES (ITL p50 ×6.6–14.4, agg 547→145 tok/s) → cần Sarathi piggyback (R4.2b). Bài học: win của chunked prefill là kernel/batching property, không phải scheduling-only | ⬜ |
| A1 R4.3 speculative decoding (lossless) | _(2026-07-04)_ | `serving/speculative.py` (Drafter · NGramDrafter · speculative_generate) · `model.py` KVCache.truncate · `bench/speculative.py` | lossless token-exact ✓ (pending-token invariant + KV rollback); ×1.2–1.4 wall / 1.3–1.5 tok/forward (n-gram, zero-cost draft). Bài học: acceptance bám output-entropy của MODEL (untrained ⇒ degenerate ⇒ n-gram hit cả prompt random), không bám prompt; drafter-family (Medusa/EAGLE/MTP) đều tối ưu E[accept] | ⬜ |
| A1 R4.4 CUDA-graph decode | _(2026-07-04)_ | `serving/cudagraph.py` CudaGraphDecoder · `bench/cudagraph_decode.py` · `tests/test_cudagraph_decode.py` | token-exact ✓; B=1 −74.3% step (15.4→3.96 ms), 253 tok/s = 77% of wall (eager 20%→compiled 53%→graph 77%). Bài học: decode là launch-bound trên GPU consumer (54–68 launch/step → 1 graph launch); paged kernel (fixed grid) là substrate DUY NHẤT capture được; torch.compile reduce-overhead từ chối vì in-place advance | ⬜ |
| A1 R4.5 MLA latent cache (toy) | _(2026-07-04)_ | `mla.py` MultiHeadLatentAttention (forward_naive vs forward_absorbed) · `tests/test_mla.py` | weight-absorption identity đúng tới machine eps (1.4e-15); MLA KV 3.56× < GQA-8. Bài học: `q·(W_UK c) = (W_UK^T q)·c` cho phép attend trong latent space, cache latent thay vì K,V; decoupled RoPE là điều kiện (rotation trên content-K sẽ phá fold vì W_UK phải static) | ⬜ |
| A1 R4.6 PD-disaggregation (demo) | _(2026-07-04)_ | `bench/disagg.py` · note `A1_R46_disaggregation.md` | decode-worker ITL p99 3× tốt hơn (20.2 vs 61.5 ms) đổi lấy KV transfer một lần 0.18 ms (16.8 MB). Bài học: prefill (compute-bound) và decode (memory-bound) trái profile; tách ra để prefill burst không cướp cycle của decode; giá là bandwidth chuyển KV (NVLink/RDMA ở node scale) |
| **A1 design note (đóng A1)** | _(2026-07-04)_ | `performance/notes/A1_design_note.md` | arc 16%→53%→77% của memory wall; mỗi rung before/after; gap thật tới vLLM ~1.3× latency / ~1.5–2× throughput = chunked-prefill piggyback + prefix cache | ⬜ |
| A2 R0 roofline harness | _(2026-07-04)_ | `bench/kernel_roofline.py` | peaks 0.551 TB/s / 72.1 TF/s reproduced; ncu blocked → %-of-peak + nsys, ncu-debt cho H100 day. Bài học: đo peak trên CHÍNH card, roofline = mem/cmp bound | ⬜ |
| A2 R1–R6 kernel ladder (GEMV/softmax/RMSNorm/TopK/GEMM) | _(2026-07-04)_ | `kernels/{gemv,softmax,norm,topk,gemm}_triton.py` · `bench/*.py` · 79 gpu tests | 4 kernel memory-bound đạt 96–100% HBM peak; TopK 46.9% (poor GPU fit) + fusion 3.4×; GEMM 0.2%→134% cuBLAS-proxy. Bài học: Triton lo coalescing/float4/swizzle; craft = chạm tường HBM / tensor-core; A2 design note "Climbing to the Hopper ceiling" | ⬜ |
| A5 R0–R4 quantization (INT8/INT4/NVFP4-MXFP4/FP8-KV) | _(2026-07-04)_ | `quant/{int8,int4_group,nvfp4_mxfp4,fp8_kv}.py` · 43 CPU tests | SQNR/MSE oracle (không allclose); NVFP4 MSE 1.48× < MXFP4; FP8-KV E2E 24.45 dB; INT4-KV degrade. Bài học: SQNR ~6.02·bits+c; per-channel > per-tensor; two-level scale placement; E4M3 clamp-before-cast | ⬜ |
| A4 R3b Triton FA2 backward (recomputation + D-vector + model integration) | _(2026-07-05)_ | `kernels/flash_attention_triton.py` · `model.py` · `tests/test_flash_attention_triton.py` · `tests/test_model.py` | gradcheck 5/5 matching SDPA autograd, model use_triton_attention=True forward/backward verified on GPU. Bài học: D-vector recenters softmax Jacobian to avoid storing N² probability matrix; dQ needs global atomic_add because parallel Key thread blocks accumulate gradients on overlapping Query rows. | ⬜ |
