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
