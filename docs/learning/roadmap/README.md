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

---

## Cách dùng lộ trình (giao thức mastery, one component per lesson)

Với **mỗi Bài**, theo đúng "Master understanding (forced)" của [`../../../CLAUDE.md`](../../../CLAUDE.md)
§How we build:

1. **First principles** — đọc "Câu hỏi first-principles" + "Feynman", tự trả lời bằng lời TRƯỚC khi mở code.
2. **Predict-before-run** — nhìn "Số đo (aha)" bị che, tự đoán con số/shape trước.
3. **Trace code** — mở đúng file·hàm·dòng trong "Trace code", đi theo thứ tự chạy; đối chiếu với test được nêu.
4. **Teach-back — CỔNG** — trả lời câu dạy-lại + biến thể "sửa-và-đoán"; **chưa dạy lại được thì chưa sang Bài sau.**
5. **Frontier** — nối tới thực hành 2026 / câu hỏi interview.

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
