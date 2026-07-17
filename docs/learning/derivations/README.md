# Derivations — first-principles derivation labs (with measured numbers)

> **Đây là gì.** Các doc dẫn-xuất **từ nguyên lý gốc** cho từng pillar của mastery roadmap: mỗi concept có
> (1) sự thật nền tảng, (2) dẫn xuất có math, (3) neo code `file·func·line` (verify tại HEAD), (4) hình ảnh
> ASCII + shape/stride/dtype + numeric trace tay, (5) **số đo THẬT chạy lại được** (GPU-gated thì cite
> `bench/RESULTS.md` hoặc gắn nhãn `[PREDICTED]` — không bịa), (6) frontier framing + cổng phỏng vấn. Dùng để
> **re-learn cold bất kỳ lúc nào** (PRR: Predict → Run → Reconcile) — mỗi doc kết bằng checklist recall + bảng số đo.
>
> Khác `roadmap_model/` + `roadmap/` (reference chung, derivation-first của toàn roadmap): thư mục này là
> **derivation lab có số đo** — bạn đồng hành với `PROGRESS.md` (sổ cái teach-back) và `CURRICULUM.md` (con đường).
>
> **Trạng thái viết:** 📚 **16/16 pillar COMPLETE** (2026-07-14) · 9,782 dòng · mọi neo code verified, math
> re-derived, số đo chạy lại được. *Viết-doc ≠ teach-back:* cột teach-back trong `PROGRESS.md` track **cold
> defense của Navigator** riêng — đó mới là tín hiệu mastery thật (FOP-4). Doc là *substrate để re-learn*.

## Chặng 1 · MODEL M1–M4 — model chạy được trên 1 GPU
| Doc | Pillar | Concepts | Dòng |
|---|---|---|---|
| [`M1_tokenizer.md`](M1_tokenizer.md) | M1 · tokenizer | subword/V/loss=logV · BPE merge · pretok-regex+special · round-trip/byte-fallback | 483 |
| [`M2_transformer_forward.md`](M2_transformer_forward.md) | M2 · forward pass | attention·GQA·RoPE·RMSNorm/pre-norm·SwiGLU·QK-norm·tied-emb·CE | 418 |
| [`M3_optimization.md`](M3_optimization.md) | M3 · objective+optim | CE-from-MLE·AdamW·cosine/warmup/clip·Muon(NS-orthogonalize)·param-split/TIED-2D | 505 |
| [`M4_training_loop.md`](M4_training_loop.md) | M4 · training loop | train-step+memmap+overfit·bf16 autocast·activation-ckpt·compile+NaN-guard·seed+monitors | 584 |

## Chặng 2 · PERF S3–S4 — compute về tốc độ ánh sáng
| Doc | Pillar | Concepts | Dòng |
|---|---|---|---|
| [`S3_cuda_core_kernels.md`](S3_cuda_core_kernels.md) | S3 · CUDA-core kernels | roofline·GEMV·softmax online·norms·TopK·GEMM cross-ridge | 654 |
| [`S4_tensor_cores_and_flash.md`](S4_tensor_cores_and_flash.md) | S4 · tensor cores + Flash | SMEM GEMM·WMMA·mma.sync+swizzle·naive-attn O(N²)·online-softmax·FA2 | 635 |

## Chặng 3 · Trải ra nhiều GPU — M5, S6
| Doc | Pillar | Concepts | Dòng |
|---|---|---|---|
| [`M5_distributed_training.md`](M5_distributed_training.md) | M5 · distributed | DP+ring all-reduce·ZeRO-1/2/3·FSDP·100B memory·comms algebra | 609 |
| [`S6_distributed_and_isa.md`](S6_distributed_and_isa.md) | S6 · distributed+ISA | TP MLP·GPipe/1F1B bubble·EP-MoE all-to-all·MFU 6ND·WGMMA/FA3/tcgen05 | 768 |

## Chặng 4 · Khoa học & nhiên liệu — M6, M7
| Doc | Pillar | Concepts | Dòng |
|---|---|---|---|
| [`M6_scaling_laws.md`](M6_scaling_laws.md) | M6 · scaling laws | vì sao scaling·IsoFLOP (N*,D*)·Chinchilla ~20·budget planner | 578 |
| [`M7_data_pipeline.md`](M7_data_pipeline.md) | M7 · data pipeline | data thắng fixed-compute·Gopher/C4 filter·DCLM classifier·MinHash/LSH dedup | 585 |

## Chặng 5–6 · Align & frontier arch — M8, M9
| Doc | Pillar | Concepts | Dòng |
|---|---|---|---|
| [`M8_post_training_rl.md`](M8_post_training_rl.md) | M8 · post-training/RL | lifecycle·SFT masking·REINFORCE→GRPO/Dr.GRPO·RLVR grader·ExpertIter·DPO·rollout-KL logging | 762 |
| [`M9_moe_mla_mtp.md`](M9_moe_mla_mtp.md) | M9 · MoE·MLA·MTP | sparse decouple·router+aux-loss-free·MLA weight-absorption·MTP draft head | 533 |

## Chặng 7 · PERF phục vụ frontier — S1, S2, S5
| Doc | Pillar | Concepts | Dòng |
|---|---|---|---|
| [`S1_serving_substrate.md`](S1_serving_substrate.md) | S1 · serving substrate | decode memory-bound·BatchedKVCache·SlotKVCache·PagedKVCache+CoW·Prefill/Chunk views·paged-decode kernel | 772 |
| [`S2_serving_engines.md`](S2_serving_engines.md) | S2 · serving engines | decode oracle·TTFT/ITL metrics·static/continuous batch·spec-decode·CUDA-graph·MLA·PD-disagg | 753 |
| [`S5_quantization.md`](S5_quantization.md) | S5 · quantization | INT8 SQNR·group-INT4 packing·NVFP4/MXFP4·FP8 KV·AWQ | 594 |

## Chặng 8 · Đóng vòng — M10
| Doc | Pillar | Concepts | Dòng |
|---|---|---|---|
| [`M10_close_the_loop.md`](M10_close_the_loop.md) | M10 · close-the-loop | museum-vs-model+speedrun·report-card oracle·model tiers C=6ND·F1–F9 ablations | 533 |

---
*16 pillar · 89 Bài · 9,782 dòng derivation. Con đường + vì sao thứ tự: [`../CURRICULUM.md`](../CURRICULUM.md).
Sổ cái teach-back (cold defense): [`../PROGRESS.md`](../PROGRESS.md).*
