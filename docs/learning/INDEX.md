# Learning track — chuỗi bài mastery (tiếng Việt), trace code thật

> **Đây là gì.** Bài giảng theo đúng giao thức "Master understanding (forced)" của repo
> (`CLAUDE.md` §How we build): **first principles → visualize 3 lăng kính (tensor shapes + đúng
> dòng `src/`, data-flow ASCII, ví dụ số tay) → predict-before-run → teach-back gate**. Mỗi bài mổ
> MỘT component đã build và đã đo — không lý thuyết suông, mọi trích dẫn là file/hàm/dòng thật,
> mọi con số là số đã log trong `bench/RESULTS.md`.
>
> **Luật dùng:** học tuần tự, mỗi bài kết thúc bằng **cổng teach-back** — tự trả lời (nói to /
> viết ra) trước khi mở đáp án gấp trong `<details>`; chưa dạy lại được thì chưa sang bài sau.
> Bài mới chỉ được viết khi bài trước đã qua cổng (đúng nhịp "one concept at a time").

## Quy ước

- **Số dòng pin theo commit** ghi ở đầu mỗi bài (code trôi thì số dòng trôi — tên
  file · class · hàm là mỏ neo chính, số dòng là tiện ích tra nhanh tại commit đó).
- Code giữ nguyên tiếng Anh (nó là sự thật trên đĩa); giải thích + chú thích tiếng Việt;
  thuật ngữ kỹ thuật (tensor, mask, slot, cache, kernel…) giữ tiếng Anh khi tự nhiên hơn.
- Mỗi bất biến nêu ra phải chỉ được **test nào găm nó** trong `tests/`.

## 🗺️ Lộ trình mastery Performance Engineering (A1→A6 + ISA) — [`roadmap/README.md`](roadmap/README.md)

> **Bản đồ học TOÀN BỘ phần perf đã build**, từ first principles xuống tận từng file·hàm·dòng (pin
> commit `9e61d7a`), 6 série theo thứ tự [S1 serving substrate → S2 engines → S3 CUDA-core kernels →
> S4 tensor cores+flash → S5 quantization → S6 distributed+ISA]. Định luật xuyên suốt: *decode là
> memory-bound; B=1 đi 16%→53%→77% của bức tường bộ nhớ*. Mỗi Bài: câu hỏi first-principles → Feynman
> → cơ chế derive → **trace code thật** → cổng teach-back → frontier. Đây là hàng đợi học của chế độ
> delegate ([ADR-0013](../adr/ADR-0013-execution-mode-full-delegation.md)) — code ship trước, học sau.
> Study-queue bên dưới map 1-1 vào các série này.

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
