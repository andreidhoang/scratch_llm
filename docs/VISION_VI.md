# scratch_llm — CS336 From-Scratch · Bản hợp nhất Tầm nhìn & Kế hoạch (tiếng Việt)

> **Bản này là gì.** Bản tổng hợp tiếng Việt của tầm nhìn, đặc tả (specs) và kế hoạch kỹ thuật của
> dự án — chính xác những gì một Claude agent tái dựng từ `CLAUDE.md` (auto-load) + `docs/`
> (đọc on-demand) + `../STRATEGY.md §8`. Nó **không thay thế** các tài liệu nguồn; nó là bản dịch/
> tổng hợp để đọc nhanh. Khi có mâu thuẫn, **nguồn chân lý là**: `CLAUDE.md` (hiến pháp) ·
> `docs/IMPLEMENTATION_PLAN.md` (xương sống) · `docs/STATUS.md` (trạng thái) · `../STRATEGY.md §8`
> (thứ tự ship).

---

## 1. Tầm nhìn (Vision) — dự án này là gì

**Một bản hiện thực từ con số 0, đạt chuẩn production, của toàn bộ stack CS336** (Stanford "Language
Modeling from Scratch"): tự tay xây mọi tầng — từ **byte → BPE → Transformer → systems → scaling →
data → RL post-training** — và có thể **đứng bảng trắng giải thích + bảo vệ từng mảnh một cách lạnh
lùng (cold)** trong phỏng vấn frontier lab.

**Nguyên lý tổ chức (organizing principle), thay cho mọi metric đơn lẻ:**
> *Sở hữu mọi tầng của một language model, đạt chuẩn kỹ thuật production, và whiteboard được từng
> mảnh cold.*

Điểm cốt lõi về **triết lý nghề nghiệp**: chất lượng kỹ thuật (engineering quality) *chính là* tín
hiệu tuyển dụng mà các lab frontier âm thầm sàng lọc — **kỹ thuật yếu là lý do bị loại thầm lặng phổ
biến nhất** của các ứng viên vốn mạnh. Vì vậy mastery là *phương tiện*; **mục tiêu cuối** là một repo
public, CI xanh, sạch, mà bạn giải thích được từ first principles.

**Hai vai trò:**
- **Bạn = Navigator** — định hướng chiến lược, ra quyết định, *sở hữu sự hiểu biết*.
- **AI = Driver** — hiện thực, giải thích khái niệm, đề xuất phương án.

## 2. Quan hệ repo ↔ giáo trình chính thức (nguồn chân lý)

| Bạn tự xây trong `scratch_llm` | Bạn *chạy* trong scaffold chính thức `../lectures/assignment*` |
|---|---|
| A1 substrate · A2 kernels/distributed · A3 IsoFLOP fitter · A4 curation+dedup · A5 SFT/GRPO/DPO | các phần khoá-mạng / leaderboard không thể "sở hữu": A3 Stanford **training-API leaderboard**, A2 **8B leaderboard**, A4 **Paloma** full run |

`../lectures/` = **đặc tả (spec) + oracle kiểm thử**. File `tests/adapters.py` của mỗi assignment là
**acceptance test khách quan**: cắm code from-scratch của bạn vào sau adapter, chạy test suite → có
oracle đúng/sai. **Quy tắc bất biến: tự viết từng dòng** — scaffold là spec, KHÔNG phải code để chép.

## 3. Triết lý xây dựng — "forced first-principles mastery × relentless execution"

Mỗi phiên có **hai mệnh lệnh song song**:

**(A) Master understanding (bắt buộc).** Với mỗi khái niệm load-bearing, theo 6 bước:
1. **First principles** — dẫn xuất cơ chế (vấn đề, toán học, vì sao thiết kế thế này), không khẳng định suông.
2. **Visualize 3 lăng kính** — (a) **tensor shapes** xuyên qua op + đúng dòng `src/scratch_llm/…`;
   (b) **data-flow hệ thống** (ASCII); (c) **ví dụ số nhỏ tự tính tay**.
3. **Predict-before-run** — viết con số/shape *có thể bị bác bỏ* TRƯỚC khi chạy (mỏ neo cho cả debug lẫn học).
4. **Build test-first** — viết bất biến thành test, rồi làm cho nó pass, CI xanh.
5. **Teach-back — CỔNG (the gate)** — bạn giải thích lại bằng lời của mình + sửa-một-biến-rồi-dự-đoán.
   **Không sang khái niệm mới cho tới khi dạy lại được.** Đây là *concept gate*, **KHÔNG phải commit
   gate** (không thủ tục, không F-ID).
6. **Connect to frontier** — nối với `docs/FRONTIER_PRACTICE_2026.md` (lab 2026 làm gì / câu hỏi phỏng vấn).

**(B) Execute relentlessly.** Luôn biết pillar hiện tại + bước load-bearing kế tiếp (`STATUS.md` +
`IMPLEMENTATION_PLAN.md`); không để build idle; bước cần GPU thì **thuê (vast.ai)**, không bao giờ âm
thầm bỏ. Dùng `/master <concept>` để đào sâu, `/next` để định hướng + bắt đầu bước kế tiếp test-first.

## 4. Năm kỷ luật kỹ thuật (cách lab sàng lọc — nhúng vào test)

1. **loss-at-init ≈ log(vocab_size)** — oracle đúng/sai rẻ nhất; lệch ⇒ bug ở head/embedding/masking.
2. **overfit-one-batch** — trước mọi run thật, ép train loss → ~0 trên 1 batch; không được ⇒ hỏng
   optimizer/data/loss wiring.
3. **fixed-seed reproducibility** — seed py/numpy/torch; chạy lại tái lập metric.
4. **mandatory RL logging** — log **entropy** + các **KL riêng biệt** (`KL(current‖ref)`,
   `KL(current‖old)`, train↔infer drift) + IS-ratio histogram + reward + **length stats** (bắt
   reward-hacking độ dài). Thiếu ⇒ run RL vô nghĩa.
5. **predict-before-you-run** — viết con số falsifiable trước; đó là mỏ neo debug.

**Green-CI (ép buộc, không phải đề nghị):** một commit chỉ ship khi xanh: `ruff check` +
`ruff format --check` + `pyright` + `pytest -m "not gpu"`. Hook `.claude/hooks/green-ci-gate.sh` chặn
`git commit` đỏ (exit 2). Commit message: `<area>: <imperative>`.

## 5. Đặc tả 5 assignment → tầng → file nguồn (Specs)

| Assignment | Tầng | Bạn xây (lõi load-bearing) | Nằm ở |
|---|---|---|---|
| **A1** Basics | Substrate | byte-level BPE · Transformer (RMSNorm·RoPE·SwiGLU·GQA-MHA) · cross-entropy · AdamW · cosine schedule · grad clip · data loading · checkpoint · decoding | `tokenizer.py`, `model.py`, `moe.py`, `optim.py`, `train.py`, `sampling.py` |
| **A2** Systems | Systems | Triton FlashAttention-2 (fwd+bwd) + roofline · DDP (naive→overlap) · ZeRO-1 · FSDP · gradient checkpointing · mixed precision · "100B memory math" | `kernels/`, `rollout/`, `utils/monitors.py`, `utils/` |
| **A3** Scaling | Scaling | IsoFLOP/Chinchilla fit (N,D tối ưu theo compute) + budget-constrained training-API leaderboard | `scaling/` |
| **A4** Data | Data | CommonCrawl: extract → filter → **quality classifier** → exact + **MinHash/LSH dedup**; thứ tự pipeline + discard accounting | `data/` |
| **A5** Alignment | Post-training | SFT → Expert Iteration → **GRPO/Dr.GRPO** + grader phần thưởng kiểm chứng được; supplement: **DPO**, reward modeling, safety | `algos/`, `rewards/`, `envs/` |

**Lõi load-bearing 20% (phải master cold), ví dụ:**
- **A1:** quy tắc merge BPE (tie-break tất định), RoPE, SDPA + causal mask, SwiGLU, decoupled-AdamW,
  đẳng thức loss-at-init → cổng "viết Transformer from scratch trong ~45 phút".
- **A2:** FA2 tiling + online softmax (+ backward tái tính với D-vector); roofline (arithmetic
  intensity, % peak, BW- vs compute-bound); toán ~16–20 B/param optimizer-state → vì sao 100B cần
  DP+TP+PP; DDP overlap; ZeRO-1.
- **A3:** power-law fitter log-log + cầu `C=6ND` + check `a+b≈1` + extrapolation trung thực.
- **A4:** máy dedup (MinHash `P[match]=Jaccard`, đường cong S của LSH), thiết kế *tín hiệu*
  quality-classifier, thứ tự pipeline (rẻ/phá-huỷ trước; dedup cuối).
- **A5:** SFT masked cross-entropy (đúng `response_mask` — mọi thứ phụ thuộc nó); advantage nhóm
  GRPO; trust-region GRPO-clip + IS ratio; toggle de-bias Dr.GRPO; mục tiêu DPO.

## 6. Trạng thái hiện tại (tính tới 2026-06-20)

| Assignment | Đã xanh ✅ | Còn phải xây ⬜ |
|---|---|---|
| **A1** Basics | tokenizer · model (RMSNorm/RoPE/SwiGLU/GQA + QK-norm opt-in) · MoE opt-in · AdamW+clip+cosine · train · sampling | — (hoàn tất) |
| **A2** Systems | FA2 (oracle+Triton+roofline, **53% SDPA @ seq 4k**) · KV-cache · `monitors.py` · rollout seam (`LocalBackend`) | **DDP · ZeRO-1 · FSDP** · 100B memory one-pager · SGLang serving thật (cần Hopper) |
| **A3 / A4 / A5** | A5: protocol env/grader (`envs/protocol.py`) | A3 `scaling/` · A4 `data/` · A5 `algos/`+`rewards/`+`envs/` (đều là stub sạch) |
| **Capstone DELTA** | design doc + kế hoạch 4 tuần (fact-checked 2026-06-14) | **chưa có code kernel** — chặn sau Step-0 gate |

**92 test xanh, ruff sạch, pyright 0 lỗi, CPU suite ≈16s; commit code cuối: Jun 8.**

## 7. Kế hoạch kỹ thuật — **build-order ≠ ship-order**

Đây là điểm tinh tế nhất của kế hoạch:
- **Build-order = số học (A1→A5):** mỗi tầng dựng trên tầng trước → bạn *master* theo thứ tự số.
- **Ship-order = xếp theo EV (kỳ vọng giá trị):** bạn *ưu tiên ship* theo độ khan hiếm/giá trị tuyển
  dụng 2026.

Vì **A1 đã xong**, artifact tiếp theo cần *ship* KHÔNG phải A2→A3 tuyến tính, mà là **"aha" RL của
A5** (cụm khan hiếm nhất 2026, EV cao nhất).

### Danh sách hành động chính tắc (`../STRATEGY.md §8`) — **base-first**
1. **Đóng băng docs. Hôm nay.** Không thêm doc chiến lược nào cho tới khi có thứ *biên dịch được* mà
   trước đó chưa.
2. **Đóng DELTA Step-0 tuần này** — nhà cung cấp H100 + giá + test khoá-clock; cam kết architecture-only
   + e2e *dự phóng*. Nửa ngày, blocking.
3. **Ship "aha" RL của A5 TRƯỚC** — GRPO/Dr.GRPO trên **Countdown** với **Qwen2.5-1.5B** (~$30–100;
   "aha" *nổi lên ở 1.5B, thất bại ở 0.5B*). `monitors.py` đã có sẵn nên guardrails miễn phí.
   **Nước đi EV cao nhất trên bàn.**
4. **OSS Rung-1 song song, từ Tuần 1** (~2h/tuần) — một PR `recipes`/`[Doc]`/quant-config vào **vLLM
   hoặc SGLang** (review lâu → bắt đầu ngay).
5. **Rồi mới DELTA, P3-first** — match `fla` recurrent trên GDN *thường* trước khi viết một dòng GDN-2.
6. **Thu các "free wins"** (mỗi cái vài ngày, đều ⬜): **100B-memory one-pager**; **vượt SDPA hoặc ship
   roofline writeup** giải thích con số 53%; **video signaling GDM**.

**Định nghĩa "landed" (cả portfolio):** ≥3 PR nhỏ merged + 1 PR thực chất, trên nền một RL run hội
tụ-giải-thích-được + kết quả decode-kernel trung thực (tốc độ *hoặc* characterization) + 100B
one-pager + video GDM. Gói đó vượt bar NVIDIA/Baseten/Fireworks và là "reach" khả tín ở
OpenAI/Anthropic/DeepMind.

### Capstone DELTA — chiến lược "barbell" (tạ đôi)
Với **ngân sách GPU thuê** + **vai trò mục tiêu chưa chốt**, portfolio là **tạ đôi**, không phải một
cược đơn:
- **Spike (đầu nặng phương sai cao) = DELTA** — một **fused decode-step kernel** cho **GatedDeltaNet-2**
  (NVIDIA, arXiv 2605.22791): oracle đúng-sai + roofline bộ nhớ H100 *đo thật* + luận điểm "tách
  erase/write là *miễn phí* lúc decode" + điểm cắt batch `C(B)`. Khả thi trên H100 thuê bursty
  (~20–40 H100-hrs). Khác biệt hoá đúng tầng (GDN/linear-attention) mà làn sóng production hiện đang
  *decode-bound*.
- **Base (đầu phương sai thấp) = A5 (GRPO/Dr.GRPO "aha") + hoàn tất A2** — cụm khan hiếm 2026 còn lại
  (RL ổn định với reward kiểm chứng được) + xương sống hệ thống. **Harness Phase-1 của DELTA *chính
  là* phần A2 inference-systems finish** — spike và base *dùng chung* hạ tầng, không cạnh tranh.

**Scope guards đã verify:** NVlabs chỉ thả *training code, không checkpoint* → correctness/roofline
chạy architecture-only, e2e *dự phóng* qua Amdahl; config 1.3B (16 heads, `d_k=d_v=128`) = 32 KB/head
state; lớp hybrid phi-tuyến là 2K sliding-window attention; license NVIDIA Source-Code-NC
(portfolio-only). **Sequencing 4 tuần:** Step-0 gate → Tuần 1 harness (= A2 finish) → Tuần 2 Triton
decode kernel (correctness-first) → Tuần 3 roofline + free-decoupling ablation + B* → Tuần 4 e2e *dự
phóng* + postmortem + writeup song ngữ. A5 đan xen CPU-side xuyên suốt (gần như zero chi phí thuê).

### Sổ rủi ro (điều phá kế hoạch — `../STRATEGY.md §4`)
1. **P3 là deliverable rủi ro nhất** — "match/beat `fla` recurrent ≥90%" gate toàn bộ kết quả GDN-2.
   **Derisk P3 trên GDN thường trước; không đụng GDN-2 cho tới khi qua.**
2. **3 nhánh chưa xây, 1 kỹ sư, 4 tuần, GPU bursty** → mặc định thất bại = 3 thứ làm dở. Tôn trọng
   kill-criterion bằng cách sequencing base-first.
3. **FA2 ở 53% SDPA** — headline hai lưỡi: ổn *kèm* roofline writeup giải thích *vì sao*, là điểm yếu
   nếu để trần.
4. **Doc-drift đang diễn ra** (đó là những ngày từ Jun 8).
5. **Step-0 quá hạn** — mọi thứ phía sau chặn ở đây.

## 8. Lớp frontier (GDM/Feinberg) cần bổ sung — `../STRATEGY.md §5`

Bổ sung scope thật sự duy nhất là **distillation**; phần còn lại là framing/elevation:
- **5.1 Model knowledge-distillation (build-lab — gap lớn nhất):** đòn bẩy hạ tầng Feinberg nhấn mạnh
  nhất ("một bản viết lại distillation 4 tháng đã hé lộ scaling law mới, trực tiếp tạo ra Gemini
  Flash"). Hiện **zero** model-KD. → `algos/distill.py`: logit KD (forward KL, hệ số `T²`),
  reverse-KL/on-policy GKD (mode-seeking, dùng lại rollout seam), sequence-level KD (dùng lại SFT
  step), + mini-fit distillation-scaling-law. Bất biến test-first: `T=1, α=1` ⇒ KD loss
  *byte-identical* với `cross_entropy`.
- **5.2 Quantization (build-lab):** "~99% TCO là điện" → giảm bit toán hạng. "Phép màu thật" là **lượng
  tử hoá activation runtime**, không chỉ weight. Mở rộng lab FP8 fake-quant của A2 sang INT8/INT4
  weight **+ activation**; báo cáo perplexity-delta theo bit-width.
- **5.3 Tín hiệu GDM rẻ nhất:** bài tập tay Scaling Book + transformer-from-scratch (`model.py` *đã là*
  bài này) + video walkthrough.

## 9. Định nghĩa "Done" (mỗi assignment)

Một assignment đạt "mastered to production" khi:
- module load-bearing tự xây trong `src/scratch_llm/`, xanh dưới ruff/pyright/pytest;
- pass `../lectures/assignment*/tests/adapters.py` nơi có adapter;
- các cổng kỷ luật được assert thành test (loss-at-init, overfit-one-batch, seed-repro, RL logging);
- quyết định không-hiển-nhiên ghi thành ADR; docstring nêu intent + bất biến chính;
- **whiteboard được lõi 20% và trả lời câu hỏi phỏng vấn cold.**

---

## Con trỏ nguồn chân lý (đọc on-demand)

| Cần gì | Đọc ở |
|---|---|
| Hiến pháp + kỷ luật + green-CI | `CLAUDE.md` |
| Xương sống build (A1→A5 + DELTA §7) | `docs/IMPLEMENTATION_PLAN.md` |
| Hướng dẫn từng assignment (lõi 20%) | `docs/assignment_guides/` (bắt đầu `INDEX.md`) |
| Trạng thái build (nguồn chân lý đơn) | `docs/STATUS.md` |
| Thực hành frontier 2026 | `docs/FRONTIER_PRACTICE_2026.md` |
| Thứ tự ship chính tắc + sổ rủi ro | `../STRATEGY.md` (§8, §4, §5) |
| Capstone DELTA (RFC + kế hoạch 4 tuần) | `../DELTA.md` |
| Giáo trình chính thức (spec + oracle) | `../lectures/` |
