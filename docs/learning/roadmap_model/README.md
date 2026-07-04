# Lộ trình mastery — LLM from scratch: model · training · post-training · frontier

> **Đây là gì.** Bản đồ học *toàn bộ* nửa **mô hình** của repo — CS336 A1–A5 (byte → BPE → Transformer →
> optimizer → systems → scaling → data → RL) + spine **close-the-loop nanochat** + **nghiên cứu ablation
> frontier F1–F9**. Đây là **anh em song sinh** của lộ trình performance ([`../roadmap/`](../roadmap/README.md)):
> lộ trình kia mổ *serving + kernels* (model chạy NHANH thế nào); lộ trình này mổ *chính model* — nó
> **HỌC** thế nào, và vì sao mỗi lựa chọn thiết kế là như vậy. Đi **từ first principles xuống tận từng
> file · hàm · dòng** (pin commit `4ad0ac5`), và **đối chiếu từng lớp với frontier 2026** (DeepSeek V2/V3/
> R1/V3.2-DSA, GLM-4.5/4.6/5.2, Kimi-K2, Qwen3, nanochat — bản tham chiếu nằm sẵn trong
> `.venv/.../transformers/models/` để đọc trực tiếp).
>
> **Khác lộ trình perf ở hai điểm** (chủ ý): (1) **DERIVATION-first** — mỗi Bài *dẫn xuất* component từ số
> 0 (bài toán → toán → vì sao thiết kế này), không chỉ mô tả code; (2) **frontier cross-reference nặng** —
> phần "Frontier" mỗi Bài chỉ đích danh một implementation/paper frontier và so ta khác/giống ở đâu.
>
> **Ngôn ngữ:** tiếng Việt; thuật ngữ giữ tiếng Anh. Toán plain-text. Quy ước: [`../INDEX.md`](../INDEX.md).

---

## Định luật tổ chức — sợi chỉ xuyên suốt

Một câu cho cả nửa mô hình:

> **Một language model là một bản NÉN của phân phối dữ liệu, học bằng gradient descent trên next-token
> prediction.** Mọi lựa chọn thiết kế — kiến trúc (M2, M9), optimizer (M3), systems (M5), data (M7),
> post-training (M8) — là **một đòn bẩy** lên *loss-per-FLOP* (pretrain hiệu quả hơn) hoặc lên *việc gợi
> ra một năng lực* (post-training). Frontier (DeepSeek/GLM/Kimi/Qwen) là một tập **câu trả lời hội tụ,
> được nhiều lab độc lập chọn** cho các đòn bẩy đó — và **ablation study (M10) là cách CHỨNG MINH** đòn
> bẩy nào thực sự gánh việc, thay vì chạy theo mốt.

Ba "aha" xâu chuỗi cả lộ trình:
1. **Loss-at-init ≈ log(V).** Oracle rẻ nhất trong stack: model mới toanh phải cho cross-entropy ≈ đều
   (M1, M3). Lệch ⇒ bug head/embedding/mask. Đây là sợi chỉ đúng-đắn xuyên M1→M3.
2. **Overfit-one-batch → 0.** Trước mọi run thật, ép train loss một batch về ~0 (M4). Không được ⇒ hỏng
   wiring optimizer/data/loss, *không phải* thiếu data. Sợi chỉ *wiring* xuyên M3→M4→M8.
3. **Convergent defaults là bằng chứng mạnh nhất.** MTP (V3 *và* GLM-4.5), Muon (nanochat + Moonlight +
   Kimi-K2), aux-loss-free MoE (V3 + GLM + Qwen3), QK-norm (Qwen3/Gemma3/OLMo2) — được các lab *độc lập*
   chọn ⇒ load-bearing, không phải fashion. Đây là kim chỉ nam của M9–M10.

Đọc hết mà giữ được: *pretrain là nén hiệu quả (loss-per-FLOP); post-training là gợi năng lực; frontier
là các đòn bẩy đã hội tụ; và bạn chỉ TIN một đòn bẩy khi một ablation iso-FLOP, một biến, pre-registered
đã đo nó.*

---

## Mười série — theo thứ tự first-principles (byte → model biết nói)

| # | Série | Dẫn xuất từ… đến | Frontier đối chiếu | File |
|---|---|---|---|---|
| 1 | **Tokenizer** (byte→token) | vì sao subword → thuật toán BPE merge → pretok regex → round-trip | tiktoken · nanochat Rust BPE (vocab 2¹⁶) | [`M1_tokenizer.md`](M1_tokenizer.md) |
| 2 | **Transformer forward** | attention as KV-retrieval → GQA → RoPE → RMSNorm → SwiGLU → QK-norm/tied-emb | Qwen3 · nanochat · DeepSeek-V3 arch | [`M2_transformer_architecture.md`](M2_transformer_architecture.md) |
| 3 | **Objective + optimization** | cross-entropy/loss-at-init → AdamW → cosine → **Muon** (Newton-Schulz) → param-split | Muon (Moonlight/Kimi-K2) · muP | [`M3_objective_and_optimization.md`](M3_objective_and_optimization.md) |
| 4 | **Training loop + efficiency** | step wiring/overfit-one-batch → bf16 autocast → torch.compile (F4) → repro/monitors | F4 MFU · Llama-3 MFU | [`M4_training_loop.md`](M4_training_loop.md) |
| 5 | **Distributed training** (A2) | DDP+ring-allreduce → ZeRO-1/2/3 → FSDP → activation-ckpt → comms algebra | DeepSeek DualPipe · ZeRO · NVLink cliff | [`M5_distributed_training.md`](M5_distributed_training.md) |
| 6 | **Scaling laws** (A3) | vì sao power-law → IsoFLOP fit → Chinchilla ~20 tok/param → budget planner | Chinchilla · EV-ranked ablation method | [`M6_scaling_laws.md`](M6_scaling_laws.md) |
| 7 | **Data pipeline** (A4) | vì sao data-quality thắng → filter → quality classifier → MinHash/LSH dedup | FineWeb · DCLM classifier | [`M7_data_pipeline.md`](M7_data_pipeline.md) |
| 8 | **Post-training / RL** (A5) | SFT → policy-gradient → **GRPO/Dr.GRPO** → RLVR grader → DPO → rollout/KL seam | DeepSeek-R1 "aha" (GRPO) · F7 | [`M8_post_training_rl.md`](M8_post_training_rl.md) |
| 9 | **Frontier arch: MoE·MLA·MTP** | vì sao sparse → router+aux-loss-free → MLA low-rank KV → MTP draft-head | DeepSeek-V2/V3 · GLM4-MoE · V3.2-DSA | [`M9_moe_mla_mtp.md`](M9_moe_mla_mtp.md) |
| 10 | **Close the loop + ablations** | museum→model gap → nanochat speedrun → report-card oracle → **F1–F9 as derivation** | nanochat · toàn cảnh 2026 §6 | [`M10_close_the_loop_and_ablations.md`](M10_close_the_loop_and_ablations.md) |

**Đọc 1→10.** M1–M4 dựng model + cách nó học (một GPU); M5 trải training ra nhiều GPU; M6 nói *lớn tới đâu*;
M7 *nuôi bằng gì*; M8 *hậu-huấn-luyện* để gợi năng lực; M9 các viên gạch frontier (MoE/MLA/MTP); M10 **đóng
vòng** thành model biết nói rồi *chứng minh* đòn bẩy nào gánh việc. Mỗi Bài tựa lên Bài trước.

---

## Cách dùng — giao thức DERIVATION mastery

Với **mỗi Bài**, theo "Master understanding (forced)" của [`../../../CLAUDE.md`](../../../CLAUDE.md):

1. **First principles / derive** — đọc "Câu hỏi first-principles" + "Feynman" + "Dẫn xuất", rồi **tự dựng
   lại toán/thiết kế từ blank** trước khi mở code. Đây là điểm khác cốt lõi: mục tiêu là *re-derive được
   lạnh*, không chỉ đọc hiểu.
2. **Predict-before-run** — che "Neo (invariant/số đo/prediction)", tự đoán invariant/con số trước.
3. **Trace code** — mở đúng file·hàm·dòng, đi theo thứ tự chạy; đối chiếu test được nêu.
4. **Teach-back — CỔNG** — dạy lại + biến thể "sửa-và-đoán"; chưa dạy lại được thì chưa sang Bài sau.
5. **Frontier** — đọc implementation frontier được trỏ (trong `.venv/.../transformers/models/`), trả lời
   "ta khác họ ở đâu, vì sao" + câu hỏi interview.

Qua cổng ⇒ tick ở study-queue [`../INDEX.md`](../INDEX.md).

---

## Trung thực (FOP-4) — đọc trước khi tin bất kỳ con số nào

**Nửa mô hình này phần lớn mới ở mức BUILT + TOY-TESTED; các run huấn luyện/eval THẬT đều rental-gated.**
Vì vậy "Neo" mỗi Bài được **gắn nhãn trung thực**, theo thứ tự ưu tiên: (a) một **invariant đã đo** (loss-
at-init ≈ log V; overfit-one-batch <1e-2; NS singular values trong dải đo [0.68,1.14]; Muon RMS ~0.2; TP
đúng 1 all-reduce/fwd; 1F1B bubble (p-1)/m; MFU tái tạo PaLM 46.2%; MoE router entropy >0.9·logN); (b) một
**số đo thật** từ `bench/RESULTS.md` — hiện chỉ có **§Frontier ablations F1/F4**, **§Phase 0 speedrun
val_bpb 0.02 + sample mạch lạc**, và bộ test; (c) một **prediction pre-registered** từ
[`../../FRONTIER_2026_ABLATIONS.md`](../../FRONTIER_2026_ABLATIONS.md) — **gắn nhãn PREDICTION, không bao
giờ [FACT]**. Bảng "Real vs toy" (spec §1) là bản đồ cái gì thật/cái gì toy: MLA là *toy* (1 layer, chưa
nối KV cache); MoE *chưa* train data thật; RL *chưa* có "aha" trên model thật. M10 = danh sách những gì
rental days sẽ đo.

---

## Bản đồ code — nửa mô hình ở đâu

```
src/scratch_llm/
  tokenizer.py        byte-level BPE (train/encode/decode)                                       (M1)
  model.py            Linear·Embedding·RMSNorm·SDPA·RoPE·SwiGLU·MHSA(GQA,qk_norm)·Block·TransformerLM·cross_entropy·MTP-seam  (M2,M3,M9)
  optim.py            AdamW·gradient_clipping·cosine_lr·Newton-Schulz·Muon·split_muon_adamw·build_optimizer  (M3)
  train.py            training loop · memmap · autocast · compile · NaN-guard                    (M4)
  utils/              ddp·zero1·fsdp·comms_calc·memory_math·checkpointing·mixed_precision         (M5)
  scaling/            isoflop · planner                                                          (M6)
  data/               pipeline · filters · quality · dedup                                       (M7)
  algos/              sft · expert_iteration · grpo · dpo                                         (M8)
  rewards/·envs/·rollout/   r1_zero · countdown·gsm_math · local·types                           (M8)
  moe.py · mla.py     DeepSeekMoE (aux-loss-free) · MLA (weight-absorption toy)                  (M9)
  speedrun.py · eval/ close-the-loop spine · report_card·generative·multiple_choice·metrics       (M10)
scripts/speedrun.sh   tokenizer→pretrain→midtrain→SFT→RL→eval→chat, một --depth                  (M10)
docs/FRONTIER_2026_ABLATIONS.md   F1–F9 rung cards · frontier landscape · honesty ledger          (M10)
```

Frontier oracle (đọc trực tiếp, đừng copy): `.venv/.../transformers/models/{deepseek_v2,deepseek_v3,
deepseek_v32,glm4_moe,nanochat,qwen3}/modeling_*.py`. Số đo gốc: [`../../../bench/RESULTS.md`](../../../bench/RESULTS.md).
Frontier defaults per-pillar: [`../../FRONTIER_PRACTICE_2026.md`](../../FRONTIER_PRACTICE_2026.md).
