# Lộ trình mastery — Performance Engineering từ đầu đến gốc (A1→A6 + ISA)

> **Đây là gì.** Bản đồ học *toàn bộ* phần performance-engineering đã build (perf-curriculum A1–A6 +
> các kernel ISA-gated), đi **từ first principles xuống tận từng file · từng hàm · từng dòng**. Không
> lý thuyết suông: mỗi Bài mổ MỘT component đã ship + đã đo, mọi con số là số thật trong
> `bench/RESULTS.md`, mọi mỏ neo code là file·hàm·dòng thật (pin theo commit `9e61d7a`). Đây là *hàng
> đợi học* của chế độ delegate ([ADR-0013](../../adr/ADR-0013-execution-mode-full-delegation.md)):
> code đã ship xong, giờ học SAU qua chính code đó.
>
> **Ngôn ngữ:** tiếng Việt; thuật ngữ kỹ thuật giữ tiếng Anh (roofline, kernel, warp, WGMMA, SQNR…).
> Toán/số viết plain-text (không LaTeX). Xem [`../INDEX.md`](../INDEX.md) cho quy ước chung.
>
> **Anh em song sinh:** lộ trình này là nửa *TỐC ĐỘ* (model chạy nhanh thế nào). Nửa *MÔ HÌNH* — model
> HỌC thế nào (tokenizer → transformer → optimizer → training → scaling → data → RL → MoE/MLA/MTP →
> close-the-loop + F1–F9 ablations) — ở [`../roadmap_model/README.md`](../roadmap_model/README.md). Học
> nửa mô hình trước, nửa tốc độ (đây) sau.

---

## Định luật tổ chức — sợi chỉ xuyên suốt cả lộ trình

Chỉ có **một ý** mà mọi Bài đều xoay quanh, phát biểu hai nửa:

> **Nửa serving (A1):** *decode là memory-bandwidth-bound* — mỗi token đọc trọn bộ trọng số một lần,
> arithmetic intensity AI ≈ 1 FLOP/byte, cách ridge ~130×. Trần = `HBM_BW / weight_bytes` (≈327 tok/s
> cho model 0.84B trên card này). MỌI kỹ thuật inference 2026 là **một đòn bẩy** kéo tok/s đo được về
> phía bức tường băng thông đó — hoặc nâng AI lên để dùng tensor core đang ngồi không lúc decode.
>
> **Nửa kernel (A2–A4):** với op memory-bound (GEMV/softmax/norm/attention-bytes) *nghề là chạm cho
> tới bức tường HBM*; chỉ GEMM vượt qua ridge và thành compute-bound — đó là chỗ tensor core (A3) mới
> là câu chuyện. A4 hợp nhất A2+A3 thành MỘT kernel không bao giờ ghi ma trận score N×N.

Cái "aha" xâu chuỗi cả A1: **B=1 decode đi 16% → 53% → 77% của bức tường bộ nhớ** khi ta bóc dần
overhead (eager → `torch.compile` fusion → CUDA-graph). Đọc hết lộ trình mà giữ được một câu này là
đủ: *inference nhanh không phải chuyện matmul, mà chuyện bộ nhớ và lịch (scheduling).*

---

## Sáu série — theo thứ tự first-principles (mỗi série tựa lên série trước)

| # | Série | Đi từ… đến | File lộ trình |
|---|---|---|---|
| 1 | **Serving substrate** — bộ máy KV-cache | roofline decode → KVCache → Slot/Batched/Paged → Prefill/ChunkPrefill views → paged Triton kernel | [`S1_serving_substrate.md`](S1_serving_substrate.md) |
| 2 | **Serving engines** — lịch + tối ưu R4.x | generate oracle → metrics → static→continuous batch → chunked-prefill (âm tính thật) → spec-decode → CUDA-graph → MLA → disagg | [`S2_serving_engines.md`](S2_serving_engines.md) |
| 3 | **CUDA-core kernels** (Triton) | roofline harness → GEMV → softmax → RMSNorm → TopK → GEMM ladder | [`S3_cuda_core_kernels.md`](S3_cuda_core_kernels.md) |
| 4 | **Tensor cores + Flash** | SMEM GEMM → WMMA → mma.sync/swizzle → attention naive → online softmax → FA2 + backward | [`S4_tensor_cores_and_flash.md`](S4_tensor_cores_and_flash.md) |
| 5 | **Quantization** — numerics | INT8 sym/asym → group-INT4 → NVFP4 vs MXFP4 → FP8-KV → AWQ | [`S5_quantization.md`](S5_quantization.md) |
| 6 | **Distributed + ISA kernels** | TP MLP → 1F1B → EP-MoE → MFU → WGMMA/FA3/tcgen05 (compile-only) | [`S6_distributed_and_isa.md`](S6_distributed_and_isa.md) |

**Đọc theo thứ tự 1→6.** Série 1–2 dựng cái *why* (thời gian/bộ nhớ inference đi đâu) + bộ máy
serving; 3 cho phản xạ profiling + craft trên CUDA core; 4 mở khoá tensor core rồi hợp nhất thành
kernel attention; 5 làm tất cả low-precision; 6 trải ra nhiều GPU. Mỗi Bài trong một série tựa lên
Bài trước — đừng nhảy cóc.

> **Lưu ý thứ tự (2026-07-05):** "1→6" ở đây là thứ tự phụ thuộc *nội bộ* nửa perf (đọc perf standalone).
> Khi **đan xen với nửa mô hình**, nguồn-sự-thật là [`../CURRICULUM.md`](../CURRICULUM.md): nó xếp
> **S3/S4 ngay sau M1–M4** (op còn nóng) và **S1/S2 sau M9** (MLA/MTP mới làm serving thú vị) — tức
> KHÔNG phải S1-first. Nếu hai doc lệch, CURRICULUM thắng. (Ghi chú: MLA nằm ở **S2 Bài 2.8**, không phải S1.)

---

## Cách dùng lộ trình (giao thức mastery, one component per lesson)

Với **mỗi micro-concept**, chạy **PRR loop** (mặc định — "Master understanding — the PRR loop" của
[`../../../CLAUDE.md`](../../../CLAUDE.md) §How we build). KHÔNG đọc monologue rồi gật: đọc derivation hay
= *fluency illusion* (thấy hiểu nhưng không encode). Internalize = bạn *generate/run/draw*:

1. **Predict — COLD.** Che "Số đo (aha)"; viết dự đoán con số/shape/bound của bạn TRƯỚC mọi giải thích (đoán sai là tốt — cam kết một con số là lúc encode xảy ra). Ở perf đây đúng là roofline-first: đoán bound (mem/flop/comm) + con số trước khi chạy.
2. **Run.** Chạy đúng file·hàm·dòng / script đo trên GPU; in shape/số thật. Prediction đụng số ĐO ĐƯỢC — hiểu = số bạn đoán khớp số đo; lệch ⇒ *khoảng lệch chính là ràng buộc bạn bỏ sót* (launch/latch/occupancy/bank-conflict).
3. **Reconcile.** Chỉ giải thích ĐÚNG chỗ lệch (3 dòng, không phải tường) + trace code đúng chỗ đó.
4. **Re-derive + tự VẼ** trên ví dụ/shape MỚI (tự vẽ data-flow/roofline — người vẽ mới nhớ).
5. **Teach-back — CỔNG + modify-and-predict** ("đổi X → số đổi sao?"); **chưa dạy lại được thì chưa sang Bài sau.** Rồi nối **Frontier** (thực hành 2026 / câu hỏi interview). Mở session kế bằng 1 câu recall từ Bài ✅ trước (spaced).

Khi qua cổng một Bài, tick nó ở bảng study-queue trong [`../INDEX.md`](../INDEX.md) (và viết bài giảng
đầy đủ nếu muốn — mẫu: [`../serving/01-batched-kv-cache.md`](../serving/01-batched-kv-cache.md), Bài
1.1 đã viết sẵn).

---

## Bản đồ code — 34 module, ở đâu

```
src/scratch_llm/
  model.py           KVCache · SlotKVCache · BatchedKVCache · PagedKVCache · PrefillView · ChunkPrefillView   (S1)
  sampling.py        generate / _decode — token-exact oracle                                                  (S2)
  mla.py             MultiHeadLatentAttention — weight-absorption                                              (S2)
  serving/           metrics · baseline · batched · continuous · speculative · cudagraph                      (S2)
  kernels/
    paged_decode_triton.py   fused paged decode                                                               (S1)
    {gemv,softmax,norm,topk,gemm}_triton.py   A2 ladder                                                        (S3)
    gemm_smem_cuda.cu · wmma_gemm.py · gemm_mma_sync.{py,cu}   A3 tensor cores                                 (S4)
    attention_naive.py · online_softmax.py · flash_attention{,_triton}.py   A4                                (S4)
  quant/             int8 · int4_group · nvfp4_mxfp4 · fp8_kv · awq                                            (S5)
  utils/             tp_mlp · pipeline_schedule · ep_moe · mfu (+ comms_calc · memory_math)                   (S6)
bench/               kernel_roofline · _harness · + mỗi rung một script đo                                    (S3)
performance/rental/kernels/   wgmma · fa3 · tcgen05 (compile-only, sm_90a/sm_100a)                            (S6)
performance/notes/   design notes A1–A7 (đọc kèm mỗi série)                                                    (all)
```

Số đo gốc cho MỌI Bài: [`../../../bench/RESULTS.md`](../../../bench/RESULTS.md) (append-only, có ngày).
