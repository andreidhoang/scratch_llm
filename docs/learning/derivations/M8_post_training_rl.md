# M8 — Post-training / RL · First-Principles Derivations

> **Đây là gì.** Bản dẫn-xuất **từ nguyên lý gốc** (first principles) cho TOÀN BỘ tầng post-training của
> một LLM — 7 micro-concept M8 (Bài 8.1→8.7): cú nhảy hàm mục tiêu → substrate SFT+mask → cầu thang
> policy-gradient (REINFORCE→baseline→PPO-clip→GRPO→Dr.GRPO) → grader RLVR → Expert Iteration →
> DPO → rollout seam + KL + logging bắt buộc. Mỗi mục: (1) **câu hỏi** falsifiable, (2) **sự thật nền
> tảng** (áp lực toán/vật lý ép ra thiết kế), (3) **dẫn xuất** kèm toán, (4) **neo code** `file·func·line`,
> (5) **hình ảnh** (ASCII + shape/stride/dtype + numeric trace), (6) **số đo THẬT** (chạy trên chính repo
> này), (7) **frontier framing** + cổng phỏng vấn.
>
> **Cách dùng để re-learn.** Đọc phần *Câu hỏi* của mỗi mục → **tự trả lời cold** (che phần dưới) → mở
> code đối chiếu → đọc *Dẫn xuất* để reconcile cái gap. Cuối doc có **checklist recall cold** + bảng số
> đo. Đây là *derivation lab* có số đo, bạn đồng hành của `roadmap_model/M8_post_training_rl.md` (bản đồ
> dẫn đường) — doc này OWN phần dẫn xuất cuối.
>
> **Cảnh báo neo.** Line number **trôi** theo commit. Cite ở đây pin theo HEAD ngày 2026-07-14
> (`algos/{sft,grpo,dpo,expert_iteration,chat_sft}.py`, `rewards/r1_zero.py`, `envs/{protocol,countdown,
> gsm_math}.py`, `rollout/{local,types}.py`, `utils/monitors.py`). Nếu lệch, `grep` tên hàm — đừng tin số
> dòng cứng (luật repo: verify, don't trust).
>
> **Nguồn số đo.** Mọi số gắn nhãn **MEASURED** chạy lại được bằng một `PYTHONPATH=src python -c` ngắn gọi
> **thẳng các hàm được neo** trong mục đó (`_group_normalize`, `compute_entropy`, `r1_zero_reward`,
> `per_instance_dpo_loss`, `monitors.*`, …) — không có script bọc riêng; số ở đây đã verify lại theo cách đó.
> Đây là "DoD là một profile, không phải test xanh" (FOP-3) áp cho
> việc học. **Honesty rail (FOP-4):** phía RL này đã build + toy-test trên CPU; **KHÔNG có "aha" model-thật
> nào được ghi nhận** (rental-gated, F7). Số toy (env 3-task, model tí hon) KHÔNG nói gì về model thật —
> nhãn **[PREDICTED]** cho mọi phát biểu về scale thật.

---

## Bức tranh lớn — post-training là đổi *hàm mục tiêu*, không phải đổi kiến trúc

Pretrain xong, ta có một model *dự đoán token kế tiếp* — mọi thứ là **MLE trên corpus cho sẵn**. Post-training
KHÔNG đổi kiến trúc (vẫn cái Transformer M2); nó đổi **thứ ta tối ưu**: từ "khả năng của token có sẵn" sang
"phần thưởng kỳ vọng của token model TỰ SINH". Đó là cú nhảy trung tâm — và là *cluster kỹ năng khan hiếm
nhất, EV cao nhất* của 2026 (FOP-5, F7).

```
                       CÚ NHẢY HÀM MỤC TIÊU  (8.1)
   MLE:  argmax_θ  E_{data}[ log p_θ(y|x) ]        y CHO SẴN  (corpus/demo)
   RL:   argmax_θ  E_{x, y~π_θ}[ R(x,y) ]          y TỰ SINH  (π_θ)  ⇒ data dist phụ thuộc θ
                        │
                        ▼  cầu thang khó dần về CREDIT-ASSIGNMENT
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │  SFT (8.2)          bắt chước demo · 0 exploration · gradient CHỈ kéo-lên       │  ← MLE + mask
   │    │  khiếm khuyết: bị chặn trên bởi chất lượng demo                            │
   │    ▼                                                                            │
   │  Expert Iter (8.5)  lọc-đúng-rồi-SFT · exploration · tín hiệu keep/drop MỨC-CHUỖI │  ← REINFORCE, A=1[R≥τ]
   │    │  khiếm khuyết: 0 credit-assignment per-token · 0 gradient âm · plateau     │
   │    ▼                                                                            │
   │  GRPO/Dr.GRPO (8.3) advantage PER-TOKEN · có gradient ÂM · group baseline       │  ← policy gradient
   │    │  cần: reward (8.4) · rollout tươi mỗi bước (8.7) · rào ổn định (clip/KL)    │
   │    └── nhánh offline: DPO (8.6)  contrastive · reward-model-free · Z(x) cancel   │
   └──────────────────────────────────────────────────────────────────────────────┘
                        │
     SUBSTRATE CHUNG (8.2):  get_response_log_probs (scoring) + masked_mean/masked_normalize (aggregate)
     REWARD (8.4):           r1_zero_reward = format × answer  (verifiable, no human, no RM)
     GUARDRAILS (8.7):       entropy · KL(cur‖ref) · KL(cur‖old) · kl_train_infer · IS/ESS · length-by-correct
```

**Một sự thật xuyên suốt: mỗi tầng *giải một khiếm khuyết ĐO ĐƯỢC* của tầng trước.** SFT rẻ+ổn nhưng chặn
trên bởi demo → EI thêm exploration nhưng chỉ keep/drop mức-chuỗi → GRPO thêm advantage per-token + gradient
âm → DPO là nhánh offline bỏ cả reward model. Học M8 = học *khiếm khuyết → lời giải*. Và tất cả tựa lên MỘT
substrate: cỗ máy log-prob-per-token + masking mà Bài 8.2 dựng — bốn algo import **chung** từ `sft.py`.

**Áp lực nào ép ra cái gì:** *bản chất bài toán* (RL vì MLE không dạy được "tự giải đúng") → 8.1; *ổn định
số + off-by-one* (mask, entropy logsumexp) → 8.2; *variance/bias của ước lượng gradient* → 8.3; *reward
hacking + an toàn eval* → 8.4; *rẻ+ổn định trước khi cần advantage* → 8.5; *đắt của RLHF online* → 8.6;
*hai engine train↔infer lệch nhau* → 8.7.

---

## 8.1 · Cú nhảy hàm mục tiêu — MLE→expected-reward, và sự xuất hiện của gradient ÂM

**Câu hỏi.** Base model đã học next-token trên hàng nghìn tỉ token — CÒN THIẾU gì mà cần cả một tầng
post-training? Cái "thiếu" đó đổi hàm mục tiêu tối ưu thế nào, ở mức *một dòng gradient*?

**Sự thật nền tảng.** MLE dạy "nói giống người viết". Nhưng với bài *có đáp án kiểm-chứng-được* (toán/code),
bắt chước lời-giải-người KHÔNG dạy "tự giải ĐÚNG bằng chuỗi suy luận của CHÍNH MÌNH". Và MLE **không có khái
niệm sai** — gradient của `−log p_θ(y)` chỉ *kéo lên* xác suất token cho-sẵn, không bao giờ *đẩy xuống* một
lời-giải-sai. Muốn "học bằng thưởng-phạt", ta phải đổi mục tiêu để gradient **có dấu**.

**Dẫn xuất.** Ba mục tiêu, cùng một model `π_θ`:

```
Pretrain/SFT (MLE):  max_θ  Σ_t log p_θ(y_t | y_<t)          y cho sẵn
RL (policy grad):    max_θ  J(θ) = E_{y~π_θ}[ R(y) ]         y ~ π_θ (TỰ SINH)
```

Gradient của mục tiêu RL — **log-derivative trick** (nền của toàn bộ Bài 8.3):
```
∇_θ E_{y~π_θ}[R(y)] = ∇_θ Σ_y π_θ(y) R(y) = Σ_y R(y) ∇_θ π_θ(y)
                    = Σ_y R(y) π_θ(y) ∇_θ log π_θ(y)          (vì ∇π = π·∇log π)
                    = E_{y~π_θ}[ R(y) · ∇_θ log π_θ(y) ]
```
So MLE `∇ = E[∇log p_θ(y)]`: hệ số trước `∇logπ` **luôn +1** (kéo lên). RL: hệ số là `R(y)` (hay advantage
`A` sau baseline) → khi `A<0`, gradient **đẩy xuống**. *Sự xuất hiện của gradient âm* là toàn bộ khác biệt.

Điểm sống-còn thứ hai: `y~π_θ` nghĩa là **phân phối dữ liệu THAY ĐỔI theo θ** (non-stationary) — khác hẳn MLE
(corpus cố định). Hệ quả: cần **rollout mới mỗi bước** (8.7) và **rào chắn ổn định** (baseline/clip/KL, 8.3)
vì "đất dưới chân dịch chuyển" khi ta cập nhật.

**Neo code** (docstring bản đồ mục tiêu từng tầng — đọc cuối mỗi file):
```python
# algos/sft.py · docstring "Interview question this module answers" :37  → MLE + mask
# algos/grpo.py · docstring :38  → "batch of graded rollouts → policy gradient step"
# algos/grpo.py · compute_naive_policy_gradient_loss :161
    return -_as_column(raw_rewards_or_advantages) * policy_log_probs   # −A·logπ  ← dấu nằm ở A
```
Cả bốn algo import CHUNG từ `sft.py` (`grpo.py:53-58`, `dpo.py:54`, `expert_iteration.py:40`): cùng một
`get_response_log_probs` (scoring) + `masked_mean/masked_normalize` (aggregate). Cùng một mũi tên gradient,
chỉ đổi **trọng số per-token**.

**Hình ảnh — cùng `logπ`, hai mục tiêu:**
```
token có logπ_θ = −2.0  (p_θ = e^{−2} ≈ 0.135)
                              d(loss)/d(logπ)     optimizer đẩy logπ
  SFT/MLE   loss = −logπ = +2.0        −1.0            LÊN   (luôn luôn, mọi token)
  PG (A=+1) loss = −A·logπ = +2.0      −1.0            LÊN
  PG (A=−1) loss = −A·logπ = −2.0      +1.0            XUỐNG ◄── cái MLE KHÔNG có
```

**Số đo THẬT** (`measure_m8.py`, MEASURED):
```
A=+1: pg_loss=+2.000  d(loss)/d(logp)=−1.0  → optimizer moves logp UP
A=−1: pg_loss=−2.000  d(loss)/d(logp)=+1.0  → optimizer moves logp DOWN
SFT/MLE: sft_loss=+2.000  d(loss)/d(logp)=−1.0  → ALWAYS moves logp UP (no down branch)
```
Không có single số cho "lifecycle"; anchor là **cấu trúc pipeline** + số gradient-sign trên. [PREDICTED, F7]:
trên base thật, reward↑ đơn điệu + KL(cur‖ref) bounded — chưa [FACT], rental-gated.

**Frontier / cổng.** DeepSeek-R1 (2501.12948) là minh chứng: **R1-Zero bỏ hẳn SFT**, RL thẳng từ base bằng
GRPO trên reward verifiable → quan sát "aha moment" (model *tự* kéo dài CoT). Vì sao dám bỏ SFT? Base đủ mạnh
+ reward verifiable → RL tự bootstrap cả format lẫn correctness. Gate = "lifecycle pretrain→SFT→RLVR, mỗi
tầng thêm gì". Trait = first-principles (dẫn được log-derivative trick cold). Scarce bucket = **RL
(differentiator 2026)**.

---

## 8.2 · SFT substrate — MLE + loss masking per-token (nền của cả bốn algo)

**Câu hỏi.** Khi SFT trên cặp (prompt, response): token NÀO vào loss? Mask căn thế nào **sau causal shift**?
Và vì sao lựa chọn length-normalization đổi *thứ gradient tối ưu*?

**Sự thật nền tảng.** SFT ép model sinh đúng response mẫu — nhưng **KHÔNG được tính loss trên token của
prompt** (prompt là *điều kiện*, không phải thứ ta muốn model học *sinh ra*). Nên cần `response_mask` chọn
đúng vùng response. Sau causal shift (input=token trước, label=token sau), mask **lệch một-ô** — sai đúng
một ô là mọi số RL hạ nguồn "âm thầm hỏng".

**Dẫn xuất.**

*Loss:* SFT NLL per-example = `Σ_t −log p_θ(o_t | prefix) · mask_t / C`. Toàn bộ "nghề" là làm `mask_t` đúng.

*Shift-AFTER-pad (pin snapshot):* ghép `prompt+output` ids mỗi hàng → **pad chuỗi ĐÃ ghép** tới max batch →
`input_ids = padded[:, :-1]`, `labels = padded[:, 1:]`. Vì shift SAU pad, một hàng ngắn giữ token cuối của
mình trong `input_ids`. "Shift-rồi-pad" là đáp án sai kinh điển (pad chèn vào giữa input/label).

*Căn mask trong tọa độ label:* label index `j` chấm token ở **full position `j+1`**. Nên response ở full
positions `[len(prompt), len(concat))` ánh xạ thành mask indices `[len(prompt)−1, len(concat)−1)` — tức
`response_mask_full` dựng ở tọa độ full rồi **dịch trái một** (`[:, 1:]`).

*Entropy ổn định (monitor entropy-collapse cho RL):* `H = logsumexp(logits) − Σ p·logit`. KHÔNG bao giờ mũ
hóa raw logit → **bất biến dịch cộng**: `H(logits + c) = H(logits)`. Logits đều → đúng `log V`.

*Đòn bẩy length-norm (cốt lõi, nối 8.3):*
```
masked_mean(x,m)       = Σ(x·m) / Σm         ← chia theo ĐỘ DÀI HÀNG  → up-weight token trong response NGẮN
masked_normalize(x,m,C)= Σ(x·m) / C          ← chia HẰNG cố định     → mọi token cùng trọng số, bất kể độ dài
```
Đây CHÍNH là cần gạt GRPO-vs-Dr.GRPO (8.3). Chọn `masked_mean` → response dài, token nhẹ gradient hơn.

**Neo code** (`algos/sft.py`):
```python
def tokenize_prompt_and_output(...):                       # :82
    padded[i, : len(concat)] = torch.tensor(concat, ...)   # :111  pad chuỗi ĐÃ ghép
    response_mask_full[i, len(prompt) : len(concat)] = True# :112  mask ở tọa độ FULL
    return {"input_ids": padded[:, :-1],                   # :117  shift SAU pad
            "labels":    padded[:, 1:],                    # :118
            "response_mask": response_mask_full[:, 1:]}    # :119  dịch trái 1 → tọa độ label

def compute_entropy(logits):                               # :123
    log_z = torch.logsumexp(logits, dim=-1)                # :130  không mũ hóa raw logit
    probs = torch.softmax(logits, dim=-1)                  # :131
    return log_z - (probs * logits).sum(dim=-1)            # :132  H = logZ − Σp·logit

def get_response_log_probs(model, input_ids, labels, ...): # :135  SEAM scoring duy nhất
    logits = logits.float()                                # :151  fp32 trước log_softmax
    log_probs = F.log_softmax(logits, -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # :152 CÓ graph

def sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, ...):  # :209
    nll = -policy_log_probs                                # :231
    per_example = masked_normalize(nll, response_mask, normalize_constant, dim=-1)  # :232
    microbatch_loss = per_example.mean()                   # :233  MEAN-over-sequences (grad-accum invariance)
    loss = microbatch_loss / gradient_accumulation_steps   # :234
    loss.backward()                                        # :235
```

**Hình ảnh — mask alignment, ví dụ tay** (prompt=`[10,11]`, response=`[20,21,22]`):
```
concat (full)   :  [10, 11, 20, 21, 22]          full positions:  0   1   2   3   4
response_mask_full: [ F,  F,  T,  T,  T]          (response = full pos [2,5) = [len(p),len(concat)))
                          │ shift SAU pad, label = full[1:], mask = full[1:]
input_ids  (:,:-1):  [10, 11, 20, 21]     shape (1,4) long
labels     (:, 1:):  [11, 20, 21, 22]     shape (1,4) long   ← label idx j chấm full pos j+1
resp_mask  (:, 1:):  [ F,  T,  T,  T]     shape (1,4) bool    ← 20 (label idx1, full pos2) = response ✓
                          └─ token 11 (label idx0) là PROMPT → mask F → không vào loss
```

**Số đo THẬT** (MEASURED):
```
input_ids: [[10, 11, 20, 21]]   labels: [[11, 20, 21, 22]]   resp_mask: [[F, T, T, T]] (shape (1,4) bool)
entropy(uniform V=1000)      = 6.907755   log V = 6.907755      ⇒ đúng log V
entropy(uniform + 12.34)     = 6.907755                          ⇒ SHIFT-INVARIANT
loss@init (masked-mean NLL)  = 6.9190     log V = 6.9078  |diff| = 0.0113  ⇒ oracle discipline #1 pass
grad-accum: max|g_full − g_accum| = 1.49e−08  (2 microbatch ÷2 == 1 full batch)  ⇒ bit-close
```

**Frontier / cổng.** SFT hiện đại (Tülu-3/OLMo-2): **loss masking + packing** (nhiều mẫu một chuỗi, mask
chéo-mẫu) là chuẩn công nghiệp; ta build đúng nửa masking. `chat_sft.py` mở rộng: mask từ
`render_conversation` — True trên **mọi assistant turn + closing `<|eot|>`**, False trên user/role-marker
(model học *nói VÀ dừng*). Gate = "token nào vào loss, mask căn sau causal shift, length-norm đổi gì". Trait
= cheapest-oracle discipline (loss@init≈log V). Table-stakes (nhưng off-by-one là bẫy loại candidate).

---

## 8.3 · Policy-gradient — REINFORCE → baseline → PPO-clip → GRPO → Dr.GRPO

**Câu hỏi.** Từ `∇E[R]` trần trụi, dựng lại VÌ SAO cần group baseline (bỏ critic), clip mua được gì, và HAI
bias nào Dr.GRPO gỡ — mỗi bước là một biến-thể-giảm-variance/bias tái dựng được cold.

**Sự thật nền tảng.** Grader là hộp đen (không vi phân được môi trường). Policy gradient là mẹo: *thưởng cao
→ làm chuỗi đó xác suất cao hơn*, và ta CÓ `∇log π`. Nhưng REINFORCE trần có **variance khủng khiếp** (reward
tuyệt đối vào gradient). Cả chuỗi biến-thể là **cùng một gradient, chỉ đổi TRỌNG SỐ per-token** để giảm
variance/bias.

**Dẫn xuất — năm bước, mỗi bước gỡ một khiếm khuyết đo được.**

**(i) REINFORCE.** `∇J = E[R(y)·∇logπ(y)]` (log-derivative trick, 8.1). Ước lượng MC per-token:
`loss = −R·logπ` (R broadcast qua token). *Khiếm khuyết:* variance cao, và `R>0` luôn → **mọi** chuỗi bị kéo
lên, chỉ khác tốc độ (không có "sai để đẩy xuống").

**(ii) Baseline.** Trừ một hằng `b` không phụ thuộc action:
```
E[(R−b)·∇logπ] = E[R·∇logπ] − b·E[∇logπ] = E[R·∇logπ] − b·∇Σπ = E[R·∇logπ] − b·∇1 = E[R·∇logπ]
```
vì `E[∇logπ] = Σπ·∇logπ = Σ∇π = ∇Σπ = ∇1 = 0`. → **unbiased**, nhưng variance giảm mạnh nếu `b ≈ E[R]`.
Baseline tốt nhất "miễn phí": **trung bình reward của G rollout cùng prompt** (không cần critic).

**(iii) GRPO advantage (Eq. 28).** `A^(i) = (r^(i) − mean(r_group)) / (std(r_group) + ε)` — center bằng mean
nhóm, chia std nhóm (`unbiased=True`, ddof=1, snapshot-pinned). Giờ **có DẤU**: chuỗi dưới-trung-bình → `A<0`
→ gradient ĐẨY XUỐNG. Đây là thứ EI (8.5) thiếu. *Điểm bán của GRPO:* nhóm G rollout cùng prompt CHO KHÔNG
một baseline → **bỏ critic** (PPO cần value net = thêm nửa model).

**(iv) PPO/GRPO-clip (Eq. 33).** Khi tái dùng một batch rollout cho `>1` epoch, `π_θ` trôi khỏi `π_old` đã
sinh nó → **off-policy**. Sửa bằng importance ratio + **clip trust-region**:
```
ρ_t = exp(logπ_θ(o_t) − logπ_old(o_t))
loss = −min( ρ_t·A , clip(ρ_t, 1−ε, 1+ε)·A )
```
Vì sao min-of-clip? Khi `A>0`, clip chặn `ρ` ở `1+ε` (đừng tăng xác suất quá xa); khi `A<0`, floor ở `1−ε`.
`min` chọn nhánh *bảo thủ hơn* → gỡ động cơ đuổi `ρ` ra khỏi vùng tin cậy → cho phép `>1` epoch an toàn.

**(v) Dr.GRPO (Eq. 31) — gỡ HAI bias:**

- *÷std là question-difficulty bias.* std đo độ *spread* của reward trong nhóm. Nhóm variance THẤP (bài dễ,
  reward chụm như `[0.6,0.4]`) bị chia std nhỏ → advantage **phình lên parity** với nhóm variance cao
  (`[1.0,0.0]`). Kết quả: GRPO cho bài dễ cùng gradient magnitude như bài khó → up-weight bài dễ. Dr.GRPO
  **bỏ ÷std**, giữ `A = r − mean` → **giữ biên reward thật**.
- *masked_mean là length bias.* chia theo độ-dài-hàng → response DÀI, mỗi token nhẹ gradient hơn → response
  SAI dài bị *phạt nhẹ hơn* → policy học **viết dài ra** (length inflation, verbosity reward-hacking).
  Dr.GRPO dùng `masked_normalize` (hằng C) → mọi token cùng trọng số.

**Neo code** (`algos/grpo.py`):
```python
def _group_normalize(raw_rewards, group_size, advantage_eps, normalize_by_std):   # :81
    groups = raw_rewards.view(-1, group_size)                                     # :99
    centered = groups - groups.mean(dim=1, keepdim=True)                          # :100  Eq.31 = dừng ở đây
    if normalize_by_std:                                                          # :101
        std = groups.std(dim=1, keepdim=True, unbiased=True)                      # :102  ddof=1 pin snapshot
        centered = centered / (std + advantage_eps)                              # :103  Eq.28 = GRPO
    return centered.reshape(-1)

def compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange):  # :174
    ratio = torch.exp(policy_log_probs - old_log_probs)                          # :188  ρ = exp(Δlogπ)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)         # :189
    loss = -torch.minimum(ratio * adv, clipped_ratio * adv)                      # :190  −min(ρA, clip·A)
    was_clipped = (ratio < 1.0 - cliprange) | (ratio > 1.0 + cliprange)          # :191

def _aggregate(per_token_loss, response_mask, length_normalization, normalize_constant):  # :233
    if length_normalization == "mean":                                           # :246  GRPO
        per_sequence = masked_mean(per_token_loss, response_mask, dim=-1)         # :247
    elif length_normalization == "constant":                                     # :248  Dr.GRPO
        per_sequence = masked_normalize(per_token_loss, response_mask, normalize_constant, dim=-1)  # :249
    return per_sequence.mean()                                                   # :252  LUÔN mean-over-seq cuối

# grpo_train_loop :422  — defaults Dr.GRPO: normalize_by_std=False :433, length_normalization="constant" :435
if epochs_per_rollout_batch > 1 and loss_type != "grpo_clip":                    # :456
    loss_type = "grpo_clip"                                                       # :457  >1 epoch ⇒ bắt buộc clip
```

**Hình ảnh — cầu thang biến-thể, mỗi bước gỡ một khiếm khuyết:**
```
REINFORCE    loss=−R·logπ                variance cao · R>0 → chỉ kéo-lên          (khiếm khuyết)
   +baseline loss=−(R−b)·logπ            unbiased (E[b∇logπ]=0) · variance↓        (nhưng b từ đâu?)
   +group    A=(r−mean)/(std+ε)          b = mean nhóm G  ⇒ BỎ CRITIC · có DẤU     (÷std thiên vị)
   +clip     −min(ρA, clip(ρ)A)          ρ=exp(Δlogπ) · trust region ⇒ >1 epoch    (off-policy an toàn)
   Dr.GRPO   A=r−mean (bỏ ÷std) + masked_normalize (bỏ length bias)               (gỡ 2 bias)
```

**Số đo THẬT** (MEASURED, `raw=[1,0,0,1]`, `group_size=2`):
```
GRPO    A = [+0.7071, −0.7071, −0.7071, +0.7071]   (÷ unbiased std √0.5 = 0.7071)
Dr.GRPO A = [+0.5,    −0.5,    −0.5,    +0.5]        (chỉ trừ mean)
all-correct group [1,1] → A = [0.0, 0.0]            ⇒ ZERO gradient (chính là plateau)

STD-BIAS (nhóm dễ vs khó):        GRPO                 Dr.GRPO
  group=[0.6, 0.4]  →  A = [+0.7071, −0.7071]     |   A = [+0.1, −0.1]
  group=[1.0, 0.0]  →  A = [+0.7071, −0.7071]     |   A = [+0.5, −0.5]
  ⇒ GRPO: |A_easy|/|A_hard| = 1.00 (BÌNH ĐẲNG SAI)  Dr.GRPO: 0.1/0.5 = 0.20 (GIỮ BIÊN THẬT)

LENGTH-BIAS (cùng A=+1, độ dài 1 vs 3):     masked_mean        masked_normalize(C=6)
  L=1  →  loss = 1.0000                       1/L = 1.0          1/6 = 0.1667
  L=3  →  loss = 1.0000  (SUM giống ⇒ token nhẹ hơn)             3/6 = 0.5000 (token cùng weight)

CLIP: ρ=[1.5, 0.5], ε=0.2 → was_clipped=[True, True], clip_fraction=1.00
```
Toy loop end-to-end (W8c, `bench/RESULTS.md:270`, MEASURED toy · env 3-task, model tí hon): **E[r] 0.186→0.666,
sampled mean_reward 0.194→0.667, entropy 0.543→0.005** (learns 2/3 task; task 3 groups toàn-sai → A=0 →
plateau). **SỐ TOY — không nói gì model thật.** [PREDICTED, F7]: base ~0.5B thật, reward↑ + *độ-dài-đúng
tăng* (aha) + KL(cur‖ref) bounded — rental-gated.

**Frontier / cổng.** GRPO = DeepSeekMath (2402.03300) + DeepSeek-R1 (2501.12948). R1 **GIỮ** `−β·D_KL` trong
objective; DAPO (2503.14476) + Dr.GRPO (2503.20783) **BỎ** KL cho pure-reasoning RLVR — biết BẬT/TẮT KL khi
nào là senior signal. Cùng trục: RLOO (baseline leave-one-out), GSPO (Qwen3 — IS mức-CHUỖI, thắng ở MoE
scale), CISPO/MiniMax-M1. Gate (docstring `:38`) = "từ batch rollout đã chấm tới policy step — baseline nhóm
từ đâu, clip mua gì, Dr.GRPO gỡ hai bias nào". Trait = predict-the-number (±0.7071 vs ±0.5). Scarce = **RL**.

---

## 8.4 · RLVR grader r1_zero — reward kiểm-chứng-được, phân rã format × correctness

**Câu hỏi.** RL cần một hàm reward. Với bài có đáp án, làm sao chấm KHÔNG cần người/không cần reward model, và
vì sao phải tách *format* khỏi *answer*, phạt **0 tổng** cho "đúng-format-sai-đáp-án"?

**Sự thật nền tảng.** Reward model học được (RLHF) đắt và bị **hack** (model tìm lỗ hổng của judge). Với
toán/code/Countdown, ta có thứ tốt hơn: **quy tắc kiểm-chứng-được**. "42 có đúng không?" là câu hỏi *quyết
định được*, không cần con người. Nhưng model phải xuất theo *hợp đồng định dạng* để trích được đáp án — nên
reward tách hai trục.

**Dẫn xuất.** `reward = format_reward × answer_reward`, cả hai ∈ {0, 1}:

- *format gate:* 1.0 iff response chứa CHÍNH XÁC separator `"</think> <answer>"` (một dấu cách, strict như
  official grader) VÀ có `"</answer>"` đóng. Prompt r1-zero đã kết thúc bằng `"<think>"` → *response* mang
  phần còn lại.
- *trích đáp án:* format fail → `None` (đáp án **KHÔNG BAO GIỜ được chấm**, dù chuỗi đúng xuất hiện đâu đó).
  Pass → `split("<answer>")[-1].replace("</answer>","")` (mirror official verbatim: `<answer>` cuối thắng).
- *khớp đáp án:* strip; cả hai parse được số → `math.isclose(rel_tol=1e-6)`; ngược lại casefold equality.
  Dung sai 1e-6 nhận `"42.000"=="42"` nhưng bác `"3.1416"` vs `"3.14159"` (lệch ở 1e-5).
- *composition:* format=0 → all-zeros; else `reward = fmt × ans`.

**Vì sao 0 tổng, không partial credit?** Cho `format_reward=1` (để *log* biết vì sao) nhưng `reward=0` cho
"đúng-format-sai-đáp-án". Nếu cho **0.5 điểm format**, model sẽ *hack*: phun đúng tag mà bỏ trống suy luận để
ăn điểm rẻ → reward-hacking surface. Tách hai trục để *log* diagnose (format ổn chưa? answer đúng chưa?) mà
*không* thưởng cấu phần rỗng.

**Bất biến sống-còn của grader-eval-code** (Countdown, `evaluate_countdown_expression`): **an toàn với text
đối kháng**. Model có thể phun text tùy ý → nếu `eval()` thẳng, `__import__('os').system(...)` là RCE, biểu
thức lồng sâu là RecursionError/MemoryError. Giải: cap `MAX_EXPRESSION_CHARS=256` trước `ast.parse` → walk
**allowlist** (chỉ int literal + nhị phân `+−*/`, không name/call/attribute/power/unary) → enforce ràng buộc
**multiset** (dùng lại số quá số lần cho phép → reject) → tính bằng `Fraction` exact → trả `None` (không
raise) trên mọi input hỏng kể cả chia 0.

**Neo code** (`rewards/r1_zero.py` + `envs/countdown.py`):
```python
_FORMAT_SEPARATOR = "</think> <answer>"                                        # r1_zero.py:48  strict 1 space
def response_format_reward(response_text):                                     # r1_zero.py:60
    ok = _FORMAT_SEPARATOR in response_text and _ANSWER_CLOSE in response_text # r1_zero.py:66
def r1_zero_reward(*, response_text, ground_truth):                            # r1_zero.py:107
    fmt = response_format_reward(response_text)                                # r1_zero.py:113
    if fmt == 0.0: return {"reward":0.0,"format_reward":0.0,"answer_reward":0.0}# r1_zero.py:115  answer NEVER graded
    ans = 1.0 if answers_match(span, ground_truth) else 0.0                    # r1_zero.py:118
    return {"reward": fmt*ans, "format_reward": fmt, "answer_reward": ans}     # r1_zero.py:119

# envs/countdown.py — safe evaluator
if len(expression) > MAX_EXPRESSION_CHARS: return None                          # :65   cap TRƯỚC ast.parse
def ev(node):                                                                  # :74   walk allowlist
    if isinstance(node, ast.Constant): ... used[value] += 1; return Fraction(value)  # :75  int literal
    if isinstance(node, ast.BinOp): ... ast.Div: None if right==0 else left/right     # :92  chia 0 → None
    return None  # names, calls, attributes, unary ops, **, //, %                # :94   allowlist reject
if any(count > budget[number] for ...): return None                            # :103  multiset constraint

# envs/protocol.py — seam mọi env cắm vào (structural Protocol, không inheritance)
class VerifiableEnv(Protocol):                                                 # protocol.py:63
    def decode(self) -> DecodeFn: ...                                          # protocol.py:67
    def tasks(self) -> list[Task]: ...                                         # protocol.py:69
    def grade(self, task, rollout) -> Graded: ...                             # protocol.py:71  → Graded :38, RewardDict :19
# envs/gsm_math.py — env SONG SINH của countdown: chấm bằng KHỚP số thay vì EVAL biểu thức
class GSMMathEnv:                                                              # gsm_math.py:57
    def grade(self, task, rollout):                                           # gsm_math.py:87
        rd = r1_zero_reward(response_text=text, ground_truth=task.ground_truth)# gsm_math.py:89  cùng grader r1_zero
```
Hai env là **hai nửa của verifiable rewarding**: Countdown *evaluate* một biểu thức model-viết (an toàn với
text đối kháng); `GSMMathEnv` chỉ *match* một đáp án số cuối (`r1_zero_reward` thẳng) — không thêm grading
logic nào của riêng nó, nên official-grader semantics (malformed→0, formatted-wrong→0) đúng by construction.

**Hình ảnh — hai gate nối tiếp:**
```
response ──► [ format gate ]──fail──► reward=0, answer NEVER inspected  (dù "42" có trong text)
                 │ pass (fmt=1)
                 ▼
             extract <answer>…</answer> (last wins)
                 │
                 ▼
             [ answer match ]──fail──► fmt=1, ans=0, reward = 1×0 = 0   ◄── "đúng-format-sai-đáp-án"
                 │ pass                                                       (log fmt=1 để diagnose, reward=0)
                 ▼
             reward = 1 × 1 = 1
```

**Số đo THẬT** (MEASURED):
```
correct              → reward=1.0  fmt=1.0  ans=1.0
formatted-but-wrong  → reward=0.0  fmt=1.0  ans=0.0     ⇒ 0 tổng, KHÔNG partial
malformed            → reward=0.0  fmt=0.0  ans=0.0     ⇒ answer never graded
'42.000' vs '42'      → answer_reward=1.0   (numeric tolerance 1e-6)
'3.1416' vs '3.14159' → answer_reward=0.0   (lệch ở 1e-5 → bác)
eval("__import__('os').system('ls')", [1,2,3,8]) = None   ⇒ RCE bị chặn (allowlist)
eval("2**999999",  [1,2,3,8]) = None   (** reject)     eval("1/0", …) = None (chia 0)
eval("8 / 3 * 3", [8,3,3]) = 8  (Fraction exact, không float round-trip)   float 8/3*3 = 8.0
eval("8 / 3 * 3", [8,3])   = None  (reused 3 beyond budget ⇒ multiset reject)
witness solutions (8 tasks, seed 0) grading correct = 8/8   ⇒ solvable-by-construction
  ví dụ task0: numbers=7,7,1  target=49  solution=((7 / 1) * 7)
```

**Frontier / cổng.** RLVR = trục "verifiable reward" của DeepSeek-R1 + Tülu-3. Grader r1-zero mirror
`cs336_alignment/drgrpo_grader.r1_zero_reward_fn` official. Frontier thêm: maj@k/self-consistency +
best-of-N-with-verifier để eval; cẩn thận reward hacking (model phun bare delimiter lừa LLM-judge). Gate
(docstring `:23`) = "vì sao chấm format tách answer, vì sao đúng-format-sai-đáp-án phải 0 tổng". Trait =
adversarial-robustness (safe eval). Scarce = **RL / verifiable environments**.

---

## 8.5 · Expert Iteration = STaR/RFT — RL bậc-thấp-nhất, ca đặc biệt của REINFORCE

**Câu hỏi.** RL không-policy-gradient rẻ nhất trông thế nào — lọc-rồi-SFT tại sao *cải thiện* model, và khiếm
khuyết cấu trúc nào khiến nó plateau, buộc phải lên GRPO?

**Sự thật nền tảng.** EI (= STaR/RFT) là "học từ chính thành công của mình": sample G rollout mỗi task từ
policy hiện tại, GIỮ LẠI chỉ những cái ĐÚNG (verifiable grader), rồi SFT trên tập lọc — lặp. Nó *khuếch đại*
xác suất mà model ĐÃ đặt lên hành vi đúng. **Không cần advantage/IS/critic** — chỉ keep/drop nhị phân + SFT
(rẻ, ổn định). Nhưng đó cũng chính là **trần** của nó.

**Dẫn xuất.** EI = reward-weighted regression với reward nhị phân + hard threshold:
```
gradient = Σ_{y : R(y) ≥ τ}  ∇log π(y)        (mọi token của rollout-được-giữ reinforce ĐỀU NHAU = SFT)
```
So Bài 8.3: đây là REINFORCE với `A = 1[R ≥ τ]` và **KHÔNG có nhánh âm** (`A ∈ {0,1}`, không bao giờ <0).
Ba khiếm khuyết cấu trúc suy ra trực tiếp:

- **Không credit-assignment per-token:** một lời giải gần-đúng-chỉ-một-bước-sai bị **DROP hoàn toàn** → token
  tốt trong rollout xấu mất trắng (dạy được = 0, y hệt noise ngẫu nhiên).
- **Không gradient âm:** không có lực đẩy xác suất RA KHỎI đáp án sai (chỉ kéo lên đáp án đúng).
- **Không áp lực exploration** ngoài nhiệt độ sample. Khi policy ngừng sinh rollout ĐÚNG MỚI trên task
  chưa-giải, tập lọc **ngừng đổi → fixed point** → task khó ở lại chưa giải. Đó CHÍNH là động lực GRPO
  (advantage per-token + gradient âm giải đúng cả ba khiếm khuyết).

**Neo code** (`algos/expert_iteration.py`):
```python
def expert_iteration(model, optimizer, env, sample_fn, *, n_ei_steps, group_size, keep_threshold=1.0, ...):  # :101
    for task in tasks:
        rollouts = sample_fn(task, group_size)                          # :140  sample từ policy HIỆN TẠI
        graded = env.grade(task, rollout)                               # :146  verifiable grader
        if graded.reward >= keep_threshold:                             # :151  giữ (1.0 = format AND answer)
            key = (task.task_id, rollout.response_ids)                  # :153
            if deduplicate and key in seen: continue                    # :154  G bản giống hệt tính MỘT lần
            kept.append((task.prompt_ids, rollout.response_ids))        # :157
    if kept:                                                            # :161  zero-kept → skip → model BẤT ĐỘNG
        batch = collate_prompt_response_ids(...)                        # :166  cùng masking như sft.py (từ IDS)
        out = get_response_log_probs(model, ...)                        # :169  seam chung
        sft_microbatch_train_step(out["log_probs"], batch["response_mask"], gradient_accumulation_steps=1, ...)  # :170
```

**Hình ảnh — vòng lặp lọc-rồi-SFT (mũi tên REINFORCE bậc thấp nhất):**
```
     policy π_t
        │  sample G/task
        ▼
   ┌─ G rollout ─┐  grade (verifiable)
   │ ✔ ✘ ✔ ✘ ✘ ✔ │ ─────────────────►  keep {R ≥ τ}  ──► dedup ──► SFT ──► π_{t+1}
   └─────────────┘                          │
   task chưa-giải: G toàn ✘ → keep {} → 0 đóng góp → task đứng yên (plateau khi π ngừng sinh ✔ MỚI)
   rollout 9/10 bước đúng, 1 sai → ✘ → DROP CẢ → 0 credit cho 9 bước tốt  (no per-token assignment)
```

**Số đo THẬT** (MEASURED, scripted deterministic sampler + tiny model):
```
step 0: kept_fraction=0.667  n_kept=2  sft_loss=253.50
step 1: kept_fraction=0.667  n_kept=2  sft_loss=216.82     ⇒ SFT loss trên tập kept GIẢM (đang học)
step 2: kept_fraction=0.667  n_kept=2  sft_loss=182.34
zero-kept step: n_kept=0  →  params identical before/after = True   ⇒ 0 kept ⇒ 0 update (invariant)
```
(Invariant "kept_fraction TĂNG đơn điệu `[0]<0.25 → [-1]≥0.9`" pin ở `test_ei_kept_fraction_rises_over_steps`
với sampler *học*; ở đây sampler cố định nên kept_fraction phẳng — nhưng zero-kept-untouched là clean.)

**Frontier / cổng.** EI/STaR (Zelikman 2022) + RFT (Rejection-sampling Fine-Tuning) là baseline RL rẻ; nanochat
midtrain/SFT-then-RL dùng đúng spine này. Frontier: iterative-DPO và RAFT là họ hàng offline của EI. Gate
(docstring `:25`) = "vì sao lọc-rồi-SFT cải thiện reasoning, giới hạn cấu trúc nào khiến GRPO cần thiết". Trait
= subtract-before-add (RL rẻ nhất trước khi cần máy phức tạp). Scarce = **RL**.

---

## 8.6 · DPO — contrastive, reward-model-free, Z(x) cancel

**Câu hỏi.** RLHF = reward model + PPO online, đắt và phức tạp. Làm sao *thu gọn* reward model VÀO chính
policy, biến RL online thành một loss contrastive **offline** chỉ cần bốn log-prob?

**Sự thật nền tảng.** RLHF cổ điển: (1) học reward model `r(x,y)` từ so-sánh-cặp của người, (2) PPO tối ưu
policy chống `r` — online, phải sample, hai model, dễ sập. DPO hỏi: nếu ĐÃ biết *nghiệm tối ưu dạng đóng* của
bước (2) theo `r`, có thể **đảo ngược** để biểu diễn `r` theo chính policy, rồi cắm vào loss so-sánh-cặp — bỏ
hẳn reward model VÀ RL online?

**Dẫn xuất — bốn bước.**

**(i) Bradley-Terry.** Một preference cặp là so-sánh logistic của reward vô hướng:
```
P(y_w ≻ y_l | x) = σ( r(x,y_w) − r(x,y_l) )        ℓ_RM = −log σ( r(x,y_w) − r(x,y_l) )
```
Reward bằng nhau → `σ(0)=0.5` → `−log 0.5 = log 2` (coin-flip).

**(ii) Nghiệm KL-regularized RLHF.** Bước (2) cực đại `E_{y~π}[r(x,y)] − β·KL(π‖π_ref)`. Dẫn nghiệm đóng bằng
biến đổi về một KL:
```
E_π[r] − β·KL(π‖π_ref) = −β·Σ_y π(y)·log( π(y) / (π_ref(y)·exp(r/β)) )
                        = −β·KL( π ‖ (1/Z)·π_ref·exp(r/β) ) + β·log Z(x),   Z(x)=Σ_y π_ref(y)·exp(r(x,y)/β)
```
KL ≥ 0, đạt 0 khi hai phân phối bằng nhau ⇒ **maximizer**:
```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y) / β )       (Gibbs/Boltzmann tilt của π_ref)
```

**(iii) Đảo ngược.** Giải ra reward ẩn theo policy tối ưu:
```
r(x,y) = β·log( π*(y|x) / π_ref(y|x) ) + β·log Z(x)
```

**(iv) Cắm vào BT — Z(x) triệt tiêu.** Trong *hiệu* cặp cùng `x`, `β·log Z(x)` (intractable) **biến mất** vì
nó phụ thuộc `x` KHÔNG phụ thuộc `y` → cùng một hằng cho cả `y_w` và `y_l`:
```
r(x,y_w) − r(x,y_l) = β·log(π_θ(y_w)/π_ref(y_w)) − β·log(π_θ(y_l)/π_ref(y_l)) + β·logZ − β·logZ
ℓ_DPO = −log σ( β·[logπ_θ(y_w|x) − logπ_ref(y_w|x)] − β·[logπ_θ(y_l|x) − logπ_ref(y_l|x)] )     (Eq. 3)
```
Đúng bốn log-prob điều kiện mỗi cặp, `π_ref` đóng băng. `β` điều khiển độ mạnh dây-KL neo về `π_ref`. Tại
`π_θ == π_ref`: bốn số triệt tiêu → margin 0 → `σ(0)` → **log 2** (mọi β). *Đánh đổi:* mất on-policy
exploration — DPO chỉ *đổi trọng số* hành vi ĐÃ có trong dữ liệu preference, không khám phá hành vi mới.

**Neo code** (`algos/dpo.py`):
```python
def bradley_terry_rm_loss(reward_chosen, reward_rejected):                     # :66
    return -F.logsigmoid(reward_chosen - reward_rejected)                      # :75   −log σ(Δr)

def _response_log_prob(model, prompt_ids, response_ids):                       # :83   Σ_t logπ(response_t)
    per_token = get_response_log_probs(model, full[:, :-1], full[:, 1:])["log_probs"]  # :93  seam chung
    return per_token[0, len(prompt_ids) - 1 :].sum()                          # :94   sum mức chuỗi

def per_instance_dpo_loss(policy_model, ref_model, tokenizer, beta, prompt, response_chosen, response_rejected):  # :97
    logp_theta_w = _response_log_prob(policy_model, prompt_ids, chosen_ids)    # :129  CÓ graph
    logp_theta_l = _response_log_prob(policy_model, prompt_ids, rejected_ids)  # :130
    with torch.no_grad():                                                      # :131  π_ref ĐÓNG BĂNG (KL anchor)
        logp_ref_w = _response_log_prob(ref_model, prompt_ids, chosen_ids)     # :132
        logp_ref_l = _response_log_prob(ref_model, prompt_ids, rejected_ids)   # :133
    policy_margin = logp_theta_w - logp_theta_l                                # :135
    ref_margin = (logp_ref_w - logp_ref_l).to(policy_margin.device)           # :136
    return -F.logsigmoid(beta * (policy_margin - ref_margin))                 # :137  Eq. 3
```

**Hình ảnh — reward model TAN vào policy:**
```
RLHF (online):   preference ──► train r(x,y) ──► PPO(π vs r)   [2 model · sample · dễ sập]
                                     │
                       DPO nhận ra:  r(x,y) = β·log(π_θ/π_ref) + β·logZ(x)   ⇒ CHÍNH π_θ ĐÃ LÀ thước đo
                                     │  cắm vào BT, cùng x:
   ℓ_DPO = −log σ( β·[ (logπ_θ^w − logπ_ref^w)  −  (logπ_θ^l − logπ_ref^l) ] )
                        └──── logZ(x) ────┘  triệt tiêu (phụ thuộc x, không y)
   4 log-prob · π_ref frozen · offline · no RM · no sampling
```

**Số đo THẬT** (MEASURED):
```
π_θ == π_ref:  DPO loss = 0.6931472 cho MỌI β ∈ {0.1, 0.5, 1.0, 5.0}   log 2 = 0.6931472   ⇒ margin 0
BT(1.5, 1.5) = 0.693147 = log 2   (reward bằng nhau ⇒ coin-flip)
BT(margin=2) = 0.126928 = −log σ(2)   (hand formula khớp)
tiny-gpt2 fixture (β=0.5): loss ≈ 0.9104   (test_dpo_algos.py :386, official scaffold verbatim)
DPO ≡ Bradley-Terry(implicit reward) như IDENTITY   (test_dpo_algos.py :282)
```

**Frontier / cổng.** DPO (Rafailov 2023). Frontier thêm: online/iterative-DPO (2401.10020), IPO/KTO/SimPO (họ
contrastive biến-thể). Nuance senior: DPO offline chỉ reweight hành vi CÓ trong data — với reasoning cần *khám
phá* lời giải mới thì RLVR (GRPO) mới là công cụ (DPO không có exploration). Gate (docstring `:39`) = "RLHF vs
DPO — derive reward model biến mất thế nào, β điều khiển gì, đi offline mất gì". Trait = first-principles
(dẫn Z(x) cancel cold). Scarce = alignment (table-stakes, nhưng derive là senior signal).

---

## 8.7 · Rollout seam + train↔infer KL + logging RL bắt buộc

**Câu hỏi.** Rollout đến từ một *engine phục vụ* (nhanh, fused, quantized) nhưng gradient tính trên một *engine
train* (eager, fp32) — nếu hai engine bất đồng về `logπ` thì mọi gradient tính trên policy **KHÔNG PHẢI** cái
đang được serve. Đo và chặn drift đó thế nào? Và tối thiểu phải log gì để một RL run *đọc-hiểu-được*?

**Sự thật nền tảng.** RL production có HAI process tách rời: rollout do inference engine (vLLM/SGLang — kernel
fused, KV quantized) sinh; gradient do trainer (FSDP/Megatron — eager, precision cao) tính. Chúng là hai *cài
đặt khác nhau của cùng trọng số* → có thể cho `logπ` khác nhau cho cùng token. Nếu khác nhiều, ta tối ưu một
policy *không phải* cái phục vụ user — **gradient thiên lệch âm thầm**. Và vì phân phối dữ liệu RL phụ thuộc θ
(8.1), một run *không log* = một run **mù**: không biết nó chết vì entropy collapse, length explosion, hay
engine drift.

**Dẫn xuất.**

*Rollout lưu logπ NÀO (ADR-0006):* raw policy `log π(a|s) = log_softmax(logits)[a]` tại **temperature 1** của
token ĐÃ lấy — KHÔNG phải phân phối sau temperature/top-p. Vì nếu lưu phân phối-lấy-mẫu, advantage/IS-ratio/KL
sẽ *dính* vào knob exploration (tăng temp → rescale gradient âm thầm). Temp/top-p chỉ định *token nào* được
rút, không định *xác suất được log*.

*Ba KL tách riêng (`utils/monitors.py`):*
```
KL(cur‖ref)      = trôi khỏi model gốc (dây neo — có nên xa base không?)
KL(cur‖old)      = trôi trong off-policy epoch (clip có đủ chặt không?)
kl_train_infer   = KL(train ‖ infer) — DRIFT ENGINE, load-bearing, HALT@0.10
```
`KL(p‖q) = Σ_x p(x)·(log p(x) − log q(x)) ≥ 0`, với `0·log 0 ≡ 0`. Bất đối xứng p,q.

*IS ratio + ESS:* `r = exp(logπ_cur − logπ_old)` (per-token, taken action); `ESS = (Σw)²/Σ(w²)` → =N khi đều,
→1 khi một trọng số áp đảo (off-policy correction nổ → gradient estimate không tin được).

*Length-by-correctness:* độ dài response tách theo đúng/sai — tell của verbosity-reward-hacking (chính length
bias Dr.GRPO gỡ, 8.3). Grad-bearing `logπ` luôn **recompute** trong train step; `logπ` lưu trên rollout là
`π_old` đóng băng, chỉ cho IS-ratio + clip.

**Neo code** (`rollout/local.py` · `utils/monitors.py` · `grpo.py`):
```python
# rollout/types.py:18  Rollout frozen: prompt_ids, response_ids, logprobs (temp 1), stop_reason
class LocalBackend:                                                            # rollout/local.py:22  DOUBLE role
    def generate(self, prompt_ids, params):                                   # :29  train-engine + infer stand-in
        ids, logprobs = generate_with_logprobs(...)                           # :31  inline logπ temp 1
    def score(self, prompt_ids, response_ids):                                # :47  teacher-force cross-check
        rows = self._response_logprob_rows(...)                               # :56  == generate (KV-cache invariant)

KL_TRAIN_INFER_HALT = 0.10                                                     # monitors.py:32
def kl_divergence(log_p, log_q, axis=-1):                                      # monitors.py:42
    terms = np.where(p > 0, p * (log_p - log_q), 0.0)                          # monitors.py:50   0·log0 = 0
def importance_ratios(logp_current, logp_old):                                 # monitors.py:59
    return np.exp(logp_current - logp_old)                                     # monitors.py:64
def effective_sample_size(weights):                                           # monitors.py:67
    return float(w.sum()**2 / s2)                                             # monitors.py:76   (Σw)²/Σw²
def build_snapshot(*, kl_current_ref, kl_current_old, kl_train_infer, ...):   # monitors.py:131  keyword-only ALL-required

# grpo.py:_log_step :549 — mỗi step wire MỌI mandatory log
snapshot = monitors.build_snapshot(
    kl_current_ref=monitors.mean_kl(cur_np, ref_np),                          # :581
    kl_current_old=monitors.mean_kl(cur_np, old_np),                          # :582
    kl_train_infer=0.0,                                                        # :583  ==0 vì LocalBackend cả hai vai
    ...)                                                                       #       kênh thành load-bearing khi SGLang cắm vào
```

**Hình ảnh — hai engine, một policy, ba KL:**
```
        π_θ (một bộ trọng số)
       ╱                     ╲
  INFER engine            TRAIN engine
  (SGLang, fused,         (FSDP, eager,
   KV-quant, bf16)         fp32 accum)
      │  rollout (logπ_infer)   │  recompute logπ_train (grad-bearing)
      └──────────┬──────────────┘
                 ▼
     kl_train_infer = KL(train‖infer)  ── HALT nếu > 0.10  (đang tối ưu policy KHÔNG được serve)

  ba KL:  cur‖ref (neo về base) · cur‖old (off-policy epoch) · train‖infer (engine drift)
  IS/ESS: r=exp(Δlogπ) · ESS=(Σw)²/Σw² → collapse = correction nổ
```

**Số đo THẬT** (MEASURED, LocalBackend cả hai vai):
```
kl_train_infer (LocalBackend vs LocalBackend) = 0.000e+00   (≈0 by construction; HALT@0.10)
IS ratios (cur==old) = [1.0, 1.0, 1.0, 1.0]   mean=1.0000   (step 0: cur==old ⇒ ratio 1)
ESS uniform N=8 → 8.0000  normalized=1.0000    ESS [1,1,1,100] → 1.0606  (→1 khi một weight áp đảo)
score() vs inline generate logprobs: max|Δ| = 4.77e−07   ⇒ cached-decode logit-identical với recompute
```
Cặp HF thật (Qwen2.5-0.5B, `L2_kl_train_infer_SPEC.md` §5, MEASURED-ledger): **eager/bf16 = 0.00982,
eager/fp32 = 0.00173** (KL(train‖infer) exact, HALT ok) — **prediction FALSIFIED**: fp32 GẦN bf16-serve hơn
~5.7× (SDPA cộng-dồn softmax/PV bằng fp32 dù tensor bf16 → serve-bf16 thực chất là attention fp32-precision →
eager/bf16 mới là outlier).

**Frontier / cổng.** Đây là *nghề* RE post-training 2026: "sáng nào cũng nhìn dashboard — entropy
(collapse=run chết), KL(policy‖ref)/KL(policy‖old), IS-ratio histogram, reward mean/std tách format-vs-answer,
length-by-correctness". Train-infer mismatch: gần đây phát hiện BF16 rounding là gốc, fix = **UNIFORM FP16 cả
train+infer** (2510.26788) — rollout-only FP16 chống BF16-trainer KHÔNG hội tụ logprob. Gate = "RL run của bạn
chết — ba số đầu tiên bạn nhìn là gì, và kl_train_infer đo cái gì mà hai KL kia không". Trait = claims-honesty
(measured>implied; prediction bị falsify được ghi trung thực). Scarce = **RL + inference** (giao điểm).

---

## Bảng số đo THẬT (chạy lại được — DoD-là-profile)

| Concept | Đo | Kết quả | Ý nghĩa |
|---|---|---|---|
| 8.1 | d(loss)/d(logπ): SFT vs PG(A<0) | −1.0 (UP) vs **+1.0 (DOWN)** | sự xuất hiện của gradient âm |
| 8.2 | mask alignment (p=2,r=3) | resp_mask `[F,T,T,T]` | dịch-trái-một sau shift |
| 8.2 | entropy(uniform V=1000) | **6.907755 = log V** · shift-invariant | monitor entropy-collapse |
| 8.2 | loss@init masked-mean NLL | **6.9190 ≈ log V 6.9078** (|Δ|0.0113) | oracle discipline #1 |
| 8.2 | grad-accum 2 microbatch /2 | max\|Δg\| = **1.49e−08** | mean-over-seq ⇒ bit-close |
| 8.3 | GRPO vs Dr.GRPO adv `[1,0,0,1]` | **±0.7071 vs ±0.5** | ÷std vs trừ-mean |
| 8.3 | all-correct group `[1,1]` | **A=0 → 0 gradient** | plateau |
| 8.3 | std-bias easy/hard ratio | GRPO **1.00** vs Dr.GRPO **0.20** | question-difficulty bias |
| 8.3 | length-bias L=1 vs 3 | mean 1.0/1.0 · const 0.167/0.5 | length inflation |
| 8.4 | reward compose | correct=1 · fmt-wrong=0 · malformed=0 | format×answer, no partial |
| 8.4 | safe eval adversarial | `__import__`→None · `**`→None · 1/0→None | allowlist + no-raise |
| 8.4 | Fraction exact `8/3*3` [8,3,3] | **= 8** (exact rational) | no float round-trip |
| 8.4 | witness solvable | **8/8** grade correct | solvable-by-construction |
| 8.5 | zero-kept step | params **identical=True** | 0 kept ⇒ 0 update |
| 8.5 | SFT loss trên kept | 253→217→182 (giảm) | học trên tập lọc |
| 8.6 | DPO loss tại π_θ=π_ref | **0.6931472 = log 2** (mọi β) | margin 0 · Z cancel |
| 8.6 | BT(margin=2) | 0.126928 = −log σ(2) | hand formula |
| 8.6 | tiny-gpt2 fixture β=0.5 | **≈ 0.9104** | official scaffold verbatim |
| 8.7 | kl_train_infer LocalBackend² | **0.0** (HALT@0.10) | engine-drift channel |
| 8.7 | ESS uniform / dominated | 8.0 / **1.06** | off-policy correction health |
| 8.7 | score() vs generate logπ | max\|Δ\| = **4.77e−07** | KV-cache invariant |
| 8.7 | HF Qwen2.5-0.5B (L2 ledger) | bf16 0.00982 · fp32 0.00173 | prediction FALSIFIED |

---

## Checklist recall COLD (che phần trên, tự trả lời — đây là cách re-learn)

1. **8.1** Dẫn log-derivative trick `∇E_{y~π_θ}[R] = E[R·∇logπ]`. Cái gì MLE không có mà RL có? (gradient âm)
   Vì sao `y~π_θ` khiến RL khó? (data dist phụ thuộc θ, non-stationary).
2. **8.2** Vẽ mask cho prompt 2 tok, response 3 tok — index nào True, vì sao dịch-trái-một sau shift? Vì sao
   entropy dùng logsumexp (shift-invariant)? Loss@init ≈ mấy?
3. **8.2** `masked_mean` vs `masked_normalize` khác gì? Cái nào up-weight token trong response ngắn?
4. **8.3** Dựng từ `∇E[R]` tới GRPO advantage. Vì sao trừ baseline giữ unbiased (`E[b·∇logπ]=0`)? Baseline
   nhóm từ đâu, và nó thay thế cái gì của PPO? (critic).
5. **8.3** `[1,0,0,1]` gs=2: advantage GRPO vs Dr.GRPO? Nhóm `[1,1]` → advantage? gradient? (plateau).
6. **8.3** HAI bias Dr.GRPO gỡ — dẫn *tại sao* ÷std là question-difficulty bias và masked_mean là length
   bias. Clip mua được gì (off-policy >1 epoch)?
7. **8.4** Vì sao "đúng-format-sai-đáp-án" PHẢI = 0 tổng? Điều gì hỏng nếu cho 0.5 điểm format? Ba lớp bảo vệ
   của safe evaluator là gì?
8. **8.5** Giải thích "không credit-assignment": rollout 9/10 bước đúng → EI học được gì? (0). Ba khiếm khuyết
   cấu trúc → cái nào GRPO advantage giải?
9. **8.6** Dẫn `Z(x)` biến mất — nó phụ thuộc gì, vì sao hiệu cặp cùng x triệt tiêu? Loss tại `π_θ=π_ref`? β
   điều khiển gì? Đi offline mất gì? (on-policy exploration).
10. **8.7** Rollout lưu logπ ở temperature mấy, vì sao không phải sau top-p? Ba KL đo gì khác nhau? Vì sao
    `kl_train_infer` load-bearing? ESS →1 nghĩa là gì?

> Trả lời cold được cả 10 = **M8 thật sự OWNED** (interview-grade RL/alignment). Vấp câu nào → mở đúng mục đó,
> hoặc blank-slate hàm tương ứng (`_group_normalize` / `per_instance_dpo_loss` / `r1_zero_reward` → `raise
> NotImplementedError` → test đỏ → tự dẫn → xanh) để re-own bằng tay.

---

*Cross-ref: `roadmap_model/M8_post_training_rl.md` (bản đồ dẫn đường) · `PROGRESS.md` (ledger 89 Bài) ·
`CURRICULUM.md` (con đường) · sibling `M2_transformer_forward.md` (forward pass — base mà RL post-train tựa
lên) · `bench/RESULTS.md` W8c (toy GRPO ledger) · ADR-0006 (grad-bearing recompute) / ADR-0017 (Dr.GRPO hai
toggle) · `docs/design/L2_{rollout_seam,kl_train_infer}_SPEC.md`. Honesty: built + toy-tested CPU; "aha"
model-thật (F7) là PREDICTION rental-gated, chưa [FACT]. Concept kế: F-rung ablations (F7 real-corpus RL run).*
