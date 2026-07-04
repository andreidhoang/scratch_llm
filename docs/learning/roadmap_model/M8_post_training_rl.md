# SÉRIE 8 — POST-TRAINING: SFT → EI → RLVR, dẫn xuất từ đầu (CS336 A5)

> **Số dòng pin theo commit `4ad0ac5` (HEAD).** Roadmap = bản đồ dẫn-xuất + trace để dẫn đường cho
> teach-back sâu về sau, KHÔNG phải bài giảng đầy đủ mỗi mục. Mỗi Bài ~một màn hình, tự chứa, và
> **derive component từ blank** (bài toán → toán → vì sao thiết kế này) rồi nối tới một **frontier
> reference cụ thể**.
> Code: `algos/{sft,expert_iteration,grpo,dpo}.py` · `rewards/r1_zero.py` · `envs/{countdown,gsm_math,
> protocol}.py` · `rollout/{local,types}.py` · `utils/monitors.py`. Số đo: `bench/RESULTS.md` (W8c),
> ADR-0006/0017, `docs/design/L2_*`.
> Đây là TWIN model-side của roadmap perf (`docs/learning/roadmap/` — serving+kernel). Nơi hai bên
> chạm nhau (MLA weight-absorption, TP/PP/EP) ta **cross-link, không giảng lại**.

**Vì sao série này.** Série trước (model + pretrain) dạy một model *dự đoán token kế tiếp* — mọi thứ
là MLE trên một corpus cho sẵn. Post-training đổi *hàm mục tiêu*: từ "khả năng của token có sẵn" sang
"phần thưởng kỳ vọng của token model TỰ SINH ra". Đó là bước nhảy trung tâm của cả série — và là kỹ
năng khan hiếm nhất, EV cao nhất của cluster 2026 (FOP-5, F7). Ta đi theo một *cầu thang khó dần về
credit-assignment*: SFT (bắt chước demo, 0 exploration) → Expert Iteration (lọc-rồi-SFT, exploration
nhưng chỉ tín hiệu keep/drop mức-chuỗi) → GRPO/Dr.GRPO (advantage per-token, có gradient âm) → DPO
(nhánh offline, contrastive không cần reward model). Mỗi tầng *giải một khiếm khuyết đo được* của tầng
trước. Tất cả chia CHUNG một substrate: cỗ máy log-prob-per-token + masking mà Bài 8.2 dựng.

**Honesty rail (FOP-4).** Phía này **đã build + test toy trên CPU**; **KHÔNG có "aha" model-thật nào
được ghi nhận** (rental-gated, F7). Mỗi Neo được gắn nhãn trung thực: MEASURED (test-invariant hoặc số
toy trong RESULTS.md) hoặc PREDICTION (từ spec ablation, không bao giờ [FACT]). "Reward tăng 0.19→0.67"
là số TOY (env 3-task, model tí hon), không nói gì về model thật.

Thứ tự đọc: **8.1 (vì sao/objective shift) → 8.2 (substrate SFT + masking) → 8.3 (policy-gradient
derive: REINFORCE→baseline→clip→GRPO→Dr.GRPO) → 8.4 (RLVR grader — nguồn reward) → 8.5 (EI, ca đặc biệt
STaR) → 8.6 (DPO, nhánh offline) → 8.7 (rollout seam + KL consistency + logging bắt buộc).**

---

## Bài 8.1 — Vì sao post-training: lifecycle SFT→EI→RL + cú nhảy hàm mục tiêu (`algos/` toàn série · đối chiếu `sft.py:39` / `grpo.py:40` interview docstrings)
> **Câu hỏi first-principles:** một base model đã học next-token trên hàng nghìn tỉ token rồi — CÒN
> THIẾU gì mà cần cả một tầng post-training? Và cái "thiếu" đó đổi hàm mục tiêu tối ưu thế nào?
> **Neo (invariant):** cú nhảy `argmax_θ E_{data}[log p_θ(y|x)]` (MLE, cho token có sẵn) → `argmax_θ
> E_{x, y~π_θ}[R(x,y)]` (expected reward, cho token TỰ SINH). Không có single số đo cho Bài này; anchor
> là **cấu trúc pipeline** + PREDICTION F7 (reward↑ đơn điệu, KL(cur‖ref) bounded) — nhãn PREDICTION.

**1. Feynman — bài toán bằng lời.** Base model là "kẻ đọc-vẹt siêu phàm": cho một tiền tố, nó đoán token
kế theo phân phối của internet. Nhưng (a) nó không biết *dừng đúng lúc*, không theo *định dạng* ta muốn
(SFT sửa); (b) với bài toán *có đáp án kiểm-chứng-được* (toán, code), MLE trên lời giải người viết chỉ
dạy "nói giống người", không dạy "giải ĐÚNG bằng chuỗi suy luận của CHÍNH MÌNH" — model phải *tự khám
phá* lời giải rồi được thưởng (RL). Analogy: SFT = học sinh chép lại bài mẫu; RL = học sinh tự làm đề,
chấm đúng/sai, rồi điều chỉnh. Đánh đổi cốt lõi của cả série: **SFT rẻ, ổn định, nhưng bị chặn trên bởi
chất lượng demo; RL đắt, dễ sập (entropy collapse, reward hack), nhưng vượt được demo vì tín hiệu là
*reward* chứ không phải *bắt chước*.**

**2. Dẫn xuất từ đầu (derive).** Ba mục tiêu, ba công thức, cùng một model π_θ:
- **Pretrain/SFT (MLE):** cực đại `Σ_t log p_θ(y_t | y_<t)` — token `y` là *cho sẵn* (corpus / demo).
  Gradient chỉ *kéo lên* xác suất token đúng; không có khái niệm "sai để đẩy xuống".
- **RL (policy gradient):** cực đại `J(θ) = E_{y~π_θ}[R(y)]`. Token `y` do CHÍNH π_θ sinh → gradient
  `∇J = E[R(y)·∇log π_θ(y)]` (Bài 8.3 derive). Giờ *có* dấu: R thấp → đẩy xác suất chuỗi đó xuống.
- **Cầu nối EI (Bài 8.5):** ca đặc biệt — thay `R·∇logπ` bằng "lọc R=1 rồi MLE trên tập lọc" =
  reward-weighted regression với reward nhị phân. Cùng một mũi tên, bậc thấp nhất.

Điểm sống-còn: cú nhảy từ "y cho sẵn" sang "y~π_θ" là toàn bộ lý do RL khó — phân phối dữ liệu *thay đổi
theo θ* (non-stationary), nên cần rollout mới mỗi bước (Bài 8.7) và cần rào chắn ổn định (baseline, clip,
KL — Bài 8.3, monitor 8.7).

**3. Trace code — pipeline như một chuỗi seam chung.** Cả bốn algo import CHUNG từ `sft.py`:
`get_response_log_probs` (`sft.py:135`) là seam scoring duy nhất; `masked_mean`/`masked_normalize`
(`sft.py:177`/`:160`) là seam aggregation duy nhất. `grpo.py:53-58` import đúng bốn hàm đó;
`dpo.py:54` import `get_response_log_probs`; `expert_iteration.py:40` import cả `sft_microbatch_train_step`.
Reward đến từ `rewards/r1_zero.py` (Bài 8.4) qua contract `envs/protocol.py:63` (`VerifiableEnv`). Rollout
đến từ `rollout/` (Bài 8.7). Đọc docstring "Interview question this module answers" ở cuối mỗi file — chúng
là bản đồ mục tiêu của từng tầng.

**4. Cổng teach-back.** (a) Giải thích cú nhảy hàm mục tiêu bằng MỘT câu, chạm được chữ "y cho sẵn vs
y~π_θ" và hệ quả "phân phối dữ liệu phụ thuộc θ". (b) *Modify-and-predict:* nếu ta CHỈ làm SFT mãi mãi
(bỏ RL) trên demo người-viết cho bài toán Countdown — chặn trên là gì, và vì sao model không tự tìm được
lời giải mới ngoài demo? (Gợi: MLE = bắt chước; không có gradient đẩy-xuống lời-giải-sai, không exploration.)

**5. Frontier.** DeepSeek-R1 (2501.12948) là minh chứng frontier của chính cú nhảy này: R1-Zero **bỏ hẳn
SFT**, RL thẳng từ base bằng GRPO trên reward kiểm-chứng-được, và quan sát "aha moment" — model *tự* dài
CoT ra để giải khó hơn. `docs/FRONTIER_PRACTICE_2026.md` A5 §700 mô tả một ngày của RE post-training 2026:
"phần lớn là giữ RL run interpretable và không diverge, không phải phát minh loss". Câu interview: "Lifecycle
pretrain→SFT→RLVR — mỗi tầng thêm gì, và vì sao R1-Zero dám bỏ SFT?" (base đủ mạnh + reward verifiable →
RL tự bootstrap format lẫn correctness).

---

## Bài 8.2 — SFT substrate: MLE trên demo + loss masking chuẩn-từng-token (`algos/sft.py` · `tokenize_prompt_and_output` :82 · `sft_microbatch_train_step` :209)
> **Câu hỏi first-principles:** khi SFT trên cặp (prompt, response), token NÀO vào loss, mask căn thế nào
> sau causal shift, và vì sao lựa chọn length-normalization đổi cả thứ gradient tối ưu?
> **Neo (invariant · MEASURED):** entropy của logits đều = `log V` đúng (`sft.py:123`, test `:154`);
> **loss-at-init: masked-mean NLL của TransformerLM tươi ≈ log V** (discipline #1, test `:216`, sai số
> `|nll − log V| < 0.3`); **overfit-one-batch: 200 SFT step → per-token NLL < 0.1** (discipline #2);
> tổng grad qua k microbatch bằng `gradient_accumulation_steps=k` khớp full-batch bit-gần-đúng.

**1. Feynman — bài toán bằng lời.** SFT = "dạy bằng ví dụ hoàn chỉnh": đưa prompt + response mẫu, ép model
sinh đúng response đó. Nhưng KHÔNG được tính loss trên token của *prompt* — prompt là điều kiện, không phải
thứ ta muốn model học *sinh ra*. Nên phải có một `response_mask` chọn đúng vùng response. Đánh đổi tinh vi:
sau causal shift (input=token trước, label=token sau), mask lệch một-ô — sai đúng một-ô là mọi số RL hạ
nguồn "âm thầm hỏng" (docstring `:6`). Analogy: chấm một bài luận nhưng chỉ tính điểm phần *trả lời*, không
tính phần *đề bài đã in sẵn* — và ranh giới đề/trả-lời phải cắt CHÍNH XÁC.

**2. Dẫn xuất từ đầu (derive).** *Loss:* SFT NLL per-example = `Σ_t −log p_θ(o_t | prefix)·mask_t`. Toàn bộ
"nghề" là làm `mask_t` đúng.
- *Shift-after-pad (pin snapshot):* ghép `prompt+output` ids mỗi hàng → **pad chuỗi ĐÃ ghép** tới max
  batch → `input_ids = padded[:, :-1]`, `labels = padded[:, 1:]` (`:117-119`). Vì shift SAU pad, một hàng
  ngắn giữ token cuối của mình trong input; "shift-rồi-pad" là đáp án sai kinh điển.
- *Căn mask trong tọa độ label:* label index `j` chấm token ở vị trí full `j+1`. Nên response ở vị trí full
  `[len(prompt), len(concat))` ánh xạ thành mask index `[len(prompt)−1, len(concat)−1)`. Code: dựng
  `response_mask_full` ở tọa độ full (`:112`) rồi trả `response_mask_full[:, 1:]` (`:119`) — dịch trái một.
- *Entropy ổn định (`:123`):* `H = logsumexp(logits) − Σ p·logit` — KHÔNG bao giờ mũ hóa raw logit → bất
  biến với dịch cộng, `H(logits+c)=H(logits)`; logits đều → đúng `log V`. Đây là monitor entropy-collapse
  cho RL (Bài 8.7).
- *Đòn bẩy length-norm (cốt lõi, nối 8.3):* `masked_mean` (`:177`, `Σ(x·m)/Σm`) chia theo *độ dài hàng
  đó* → up-weight token trong response ngắn; `masked_normalize` (`:160`, `Σ(x·m)/C` hằng cố định) → mọi
  token cùng trọng số bất kể độ dài. Đây CHÍNH là cần gạt GRPO-vs-Dr.GRPO (Bài 8.3).

**3. Trace code.** Một bước SFT: `tokenize_prompt_and_output` (`:82`) → dict `input_ids/labels/
response_mask` → `get_response_log_probs(model, input_ids, labels)` (`:135`): forward, `float()`,
`log_softmax(...).gather(-1, labels)` (`:152`) → per-token logπ **có graph** (grad-bearing, ADR-0006);
`token_entropy` (nếu bật) detach — monitor, không phải loss. Rồi `sft_microbatch_train_step` (`:209`):
`nll = −logπ` → `per_example = masked_normalize(nll, mask, C, dim=-1)` (`:232`) → `microbatch_loss =
per_example.mean()` → `loss = microbatch_loss / grad_accum` → `loss.backward()` (`:235`). Bất biến
grad-accum: chỉ vì mean-over-sequences ở tầng trên cùng nên k microbatch chia `1/k` tái tạo full-batch
grad (test pinned). Test loại-oracle: `test_loss_at_init_is_log_vocab` (`:216`).

**4. Cổng teach-back.** (a) Vẽ căn mask cho một ví dụ tay: prompt 2 token, response 3 token — chỉ ra mask
index nào True, và vì sao dịch-trái-một. (b) *Modify-and-predict:* nếu đổi `masked_normalize` (hằng C)
thành `masked_mean` (chia độ dài hàng) trong `sft_microbatch_train_step` — gradient trên token của một
response DÀI đổi thế nào so với response NGẮN, và điều đó ảnh hưởng gì nếu tất cả demo đều dài bằng nhau?
(Gợi: khi độ dài đều, hai cái *tỉ lệ*; khác nhau chỉ lộ ra khi độ dài lệch.)

**5. Frontier.** `docs/FRONTIER_PRACTICE_2026.md` A5 xếp SFT→preference→RLVR theo khuôn Tülu-3/OLMo-2. Điểm
frontier của SFT hiện đại: **loss masking + packing** (nhiều mẫu một chuỗi, mask chéo-mẫu) là chuẩn công
nghiệp; ta build đúng nửa masking, packing là mở rộng. Câu interview (chính là docstring `:37`): "SFT trên
(prompt, response) — token nào vào loss, mask căn thế nào sau causal shift, vì sao length-norm đổi thứ
gradient tối ưu?"

---

## Bài 8.3 — Policy-gradient dẫn xuất: REINFORCE → baseline → PPO-clip → GRPO → Dr.GRPO (`algos/grpo.py` · `_group_normalize` :81 · `compute_grpo_clip_loss` :174 · ADR-0017)
> **Câu hỏi first-principles:** từ `∇E[R]` trần trụi, dựng lại VÌ SAO cần group baseline (bỏ critic),
> clip mua được gì, và HAI bias nào Dr.GRPO gỡ — mỗi bước là một biến-thể-giảm-variance/bias có thể tái
> dựng cold.
> **Neo (số đo · MEASURED):** advantage GRPO trên `[1,0,0,1]`, `group_size=2` → **±0.7071** (÷ std
> unbiased ddof=1); Dr.GRPO → **±0.5** (chỉ trừ mean) — pin snapshot, test `test_grpo_algos.py`. Loop
> toy: **E[r] 0.186→0.666, sampled mean_reward 0.194→0.667, entropy 0.543→0.005** (W8c, RESULTS `:270`),
> plateau 2/3 task (group toàn-sai → A=0 → 0 gradient). SỐ TOY, không nói gì model thật.

**1. Feynman — bài toán bằng lời.** Muốn "dạy bằng thưởng-phạt" mà không biết đạo hàm của môi trường (grader
là hộp đen, không vi phân được). Policy gradient là mẹo: *thưởng cao thì làm chuỗi đó xác suất cao hơn,
thưởng thấp thì thấp hơn* — và ta CÓ đạo hàm của log π. Nhưng REINFORCE trần variance khủng khiếp (reward
tuyệt đối vào gradient). Chuỗi biến-thể trong Bài này đều là **cùng một gradient, chỉ đổi TRỌNG SỐ per-token**
để giảm variance/bias. Đánh đổi lớn: PPO dùng critic (value net) làm baseline — thêm nửa model; GRPO nhận ra
*nhóm G rollout cùng prompt* CHO KHÔNG một baseline → bỏ critic. Đó là cả điểm bán của GRPO.

**2. Dẫn xuất từ đầu (derive) — năm bước, mỗi bước gỡ một khiếm khuyết đo được.**
- **(i) REINFORCE:** `∇J = E[R(y)·∇logπ(y)]`. Ước lượng Monte-Carlo: `−R·logπ` per-token (`compute_naive_
  policy_gradient_loss` :161, R broadcast qua token). Khiếm khuyết: variance cao, và R>0 luôn → *mọi* chuỗi
  bị kéo lên, chỉ khác tốc độ.
- **(ii) Baseline:** trừ một hằng b không phụ thuộc action: `E[(R−b)∇logπ] = E[R∇logπ]` (vì `E[b·∇logπ]=0`)
  → *unbiased* nhưng giảm variance nếu b≈E[R]. Baseline tốt nhất "miễn phí": trung bình reward của G rollout
  cùng prompt.
- **(iii) GRPO advantage (Eq. 28):** `A^(i) = (r^(i) − mean(r_group)) / (std(r_group)+ε)` — center bằng
  mean nhóm, chia std nhóm (`_group_normalize` :99-104, `unbiased=True` ddof=1 pin snapshot). Giờ có DẤU:
  chuỗi dưới-trung-bình → A<0 → gradient ĐẨY XUỐNG. Đây là thứ EI (Bài 8.5) thiếu.
- **(iv) PPO/GRPO-clip (Eq. 33):** khi tái dùng một batch rollout cho >1 epoch, π_θ trôi khỏi π_old đã
  sinh nó → off-policy. Sửa bằng importance ratio `ρ = exp(logπ_θ − logπ_old)` và **clip trust-region**:
  `−min(ρ·A, clip(ρ,1−ε,1+ε)·A)` (`compute_grpo_clip_loss` :174, `ratio` :188, `was_clipped` :191). Clip
  chặn một update đuổi quá xa rollout nó không được sinh từ đó → cho phép >1 epoch an toàn.
- **(v) Dr.GRPO (Eq. 31) — gỡ HAI bias (ADR-0017):**
  · *÷std là question-difficulty bias:* nhóm variance thấp (bài dễ / bài khó thất bại đều) bị chia std nhỏ
  → up-weight lên parity với nhóm variance cao. Dr.GRPO **bỏ ÷std**, giữ `A = r − mean` (`normalize_by_std=
  False`). Đo: hai nhóm L=[0.6,0.4], H=[1.0,0.0] → Dr.GRPO |A_L|/|A_H| = 0.2 (giữ biên thật) vs GRPO ≈1.0.
  · *masked_mean là length bias:* chia theo độ-dài-hàng → response dài, token nhẹ gradient hơn → response
  SAI dài bị *phạt nhẹ hơn* → policy học viết dài ra (length inflation). Dr.GRPO dùng `masked_normalize`
  (hằng C) → mọi token cùng trọng số. Đo (ADR-0017): lengths 1 và 3, A=+1 → masked_mean cho grad −0.5 vs
  −1/6; masked_normalize cho cả hai −1/6.

**3. Trace code.** `_group_normalize` (:81) view `(-1, group_size)`, center (:100), tùy chọn ÷(std+ε)
(:101-103). `compute_policy_gradient_loss` (:199) dispatch 3 nhánh: `no_baseline` (raw R), `reinforce_
with_baseline` (advantages), `grpo_clip`. `_aggregate` (:233): `"mean"`=masked_mean per-seq rồi `.mean()`;
`"constant"`=masked_normalize per-seq rồi `.mean()` — LUÔN mean-over-seq ở tầng cuối (grad-accum invariance,
ADR-0017 §3). `grpo_microbatch_train_step` (:255) = per-token loss → aggregate → `/grad_accum` → backward.
Loop `grpo_train_loop` (:422): mỗi step rollout G/prompt → grade → `_group_normalize` (:493) → nếu
`epochs>1` tự chuyển `grpo_clip` + cache π_old (:503) → microbatch update → `_log_step` (:549). Defaults
Dr.GRPO: `normalize_by_std=False`, `length_normalization="constant"` (:433-435). `expected_reward` (:605)
= oracle reward noise-free (P(answer token)) mà test khẳng định tăng đơn điệu.

**4. Cổng teach-back.** (a) Dựng lại từ `∇E[R]` tới GRPO advantage, chỉ rõ tại sao trừ baseline giữ
unbiased (`E[b·∇logπ]=0`). (b) *Modify-and-predict:* trên `[1,0,0,1]` group_size=2, tính tay advantage
GRPO và Dr.GRPO (kết quả ±0.7071 vs ±0.5) — rồi đoán: nếu một group toàn `[1,1]` (mọi rollout đúng),
advantage bằng bao nhiêu và gradient step làm gì? (Gợi: mean=1, center=0, A=0 → 0 gradient — chính là
plateau 2/3 task của run toy.)

**5. Frontier.** Đây đúng là GRPO của DeepSeekMath (2402.03300) + DeepSeek-R1 (2501.12948, F7). R1 **GIỮ**
term −β·D_KL trong objective (Eq. 1); DAPO (2503.14476) và Dr.GRPO (2503.20783) **BỎ** KL cho pure-reasoning
RLVR — biết BẬT/TẮT KL khi nào là senior signal (`FRONTIER_PRACTICE_2026.md` :712-718). Các biến-thể cùng
trục: RLOO (2402.14740, baseline leave-one-out unbiased), GSPO (Qwen3, IS mức-CHUỖI thay per-token, thắng ở
MoE scale), CISPO/MiniMax-M1. **PREDICTION F7** (nhãn PREDICTION, spec §84/150): trên base ~0.5B thật, reward
tăng đơn điệu, *độ dài câu-trả-lời-đúng tăng* (aha), KL(cur‖ref) bounded; một neural-RM control sẽ reward-hack
để làm nổi điểm rule-based. Chưa chạy — rental-gated. Câu interview (docstring :38): "Từ batch rollout đã
chấm tới một policy step — baseline nhóm đến từ đâu, clip mua gì, Dr.GRPO gỡ hai bias nào?"

---

## Bài 8.4 — RLVR + grader r1_zero: reward kiểm-chứng-được, phân rã format×correctness (`rewards/r1_zero.py` · `r1_zero_reward` :107 · `envs/countdown.py` · `evaluate_countdown_expression` :55)
> **Câu hỏi first-principles:** RL cần một hàm reward — với bài toán có đáp án, làm sao chấm KHÔNG cần
> người/không cần reward model, và vì sao phải tách *format* khỏi *answer*, phạt 0 tổng cho "đúng-format-sai-đáp-án"?
> **Neo (invariant · MEASURED):** với MỌI input, `reward == format_reward × answer_reward`, mỗi thành phần
> đúng 0.0 hoặc 1.0 (unit test `test_r1_zero`, docstring `:19`); Countdown: mọi `metadata["solution"]` →
> `answer_reward == 1.0`; `evaluate_countdown_expression` **không bao giờ raise** trên input đối kháng.

**1. Feynman — bài toán bằng lời.** Reward model học được (RLHF) đắt và bị *hack* (model tìm lỗ hổng của
judge). Với toán/code/Countdown, ta có thứ tốt hơn: **quy tắc kiểm-chứng-được**. "42 có đúng không?" là câu
hỏi *quyết định được*, không cần con người. Nhưng model phải xuất theo một *hợp đồng định dạng* để ta trích
được đáp án — nên reward tách hai trục: (1) có đúng khuôn `<think>…</think> <answer>…</answer>` không, (2)
đáp án trong tag có khớp không. Đánh đổi/quyết định cốt lõi: **không cho partial credit** — "đúng-format,
sai-đáp-án" được ghi `format_reward=1` (để *log* biết vì sao) nhưng `reward=0`. Nếu cho điểm-format bộ phận,
model sẽ *hack*: phun đúng tag mà bỏ trống suy luận để ăn điểm rẻ.

**2. Dẫn xuất từ đầu (derive).** Reward = `format_reward × answer_reward`, cả hai ∈ {0,1}:
- *format gate (`response_format_reward` :60):* 1.0 iff response chứa CHÍNH XÁC separator `"</think>
  <answer>"` (một dấu cách, strict như official grader) VÀ có `"</answer>"` đóng. Prompt r1-zero (`:37`)
  đã kết thúc bằng `"<think>"`, nên *response* phải mang phần còn lại.
- *trích đáp án (`extract_answer_span` :70):* nếu format fail → None (đáp án KHÔNG BAO GIỜ được chấm, dù
  chuỗi đúng có xuất hiện đâu đó). Nếu pass → `split("<answer>")[-1].replace("</answer>","")` (mirror
  official verbatim: `<answer>` cuối thắng).
- *khớp đáp án (`answers_match` :93):* strip; nếu cả hai parse được số → `math.isclose(rel_tol=1e-6)`
  ("42.000"=="42", bỏ dấu phẩy nghìn); ngược lại casefold string equality. Dung sai 1e-6 nhận drift biểu
  diễn nhưng bác "3.1416" vs "3.14159" (lệch ở 1e-5).
- *composition (`r1_zero_reward` :107):* format=0 → all-zeros; else `reward = fmt × ans`.

Bất biến sống-còn của grader-eval-code (Countdown, `evaluate_countdown_expression` :55): **an toàn với text
đối kháng** — cap `MAX_EXPRESSION_CHARS=256` (:52) trước `ast.parse` (chặn RecursionError/MemoryError),
walk allowlist (chỉ int literal + nhị phân `+−*/`, không name/call/attr/power/unary), enforce ràng buộc
multiset (dùng lại số quá số lần cho phép → reject), tính bằng `Fraction` exact (`(8/3)*3==8` chính xác),
trả `None` — **không raise** — trên mọi input hỏng kể cả chia 0.

**3. Trace code.** `render_r1_zero_prompt` (`r1_zero.py:55`) nhồi câu hỏi vào template. Env
(`countdown.py:CountdownEnv` :228) `grade` (:251): decode ids→text → `grade_countdown_response` (:114) =
format gate (`response_format_reward`) × `grade_countdown_answer` (:108, gọi safe evaluator :55) →
`Graded` (`envs/protocol.py:38`). `generate_countdown_tasks` (:188) sinh task *solvable-by-construction*:
`_build_solution` (:140) fold số thành biểu thức left-assoc giữ giá trị nguyên-dương mỗi bước → value CHÍNH
LÀ target → luôn tồn tại witness (lưu `metadata["solution"]`, test khẳng định grade=1). `gsm_math.py` = env
song sinh: grade bằng *khớp số* thay vì *eval biểu thức* — hai nửa của verifiable rewarding.

**4. Cổng teach-back.** (a) Vì sao "đúng-format-sai-đáp-án" PHẢI = 0 tổng reward, và điều gì hỏng nếu cho
0.5 điểm format? (reward-hacking surface: phun tag rỗng ăn điểm rẻ). (b) *Modify-and-predict:* nếu bỏ cap
256-char + allowlist trong `evaluate_countdown_expression` và cho `eval()` thẳng — model đối kháng phun
`__import__(...)` hay biểu thức lồng sâu sẽ làm gì grader, và vì sao đó là lỗ hổng RL-loop chứ không chỉ
lỗi parsing?

**5. Frontier.** Đây là RLVR — trục "verifiable reward" của DeepSeek-R1 (2501.12948) và Tülu-3. Grader
r1-zero mirror `cs336_alignment/drgrpo_grader.r1_zero_reward_fn` official. Frontier hiện đại thêm:
maj@k/self-consistency + best-of-N-with-verifier để eval (`FRONTIER_PRACTICE_2026.md` :700), và cẩn thận
*reward hacking* (model phun bare delimiter lừa LLM-judge). Câu interview (docstring :23): "Vì sao RLVR chấm
format tách answer, và vì sao đúng-format-sai-đáp-án phải 0 tổng thay vì partial?"

---

## Bài 8.5 — Expert Iteration = STaR/RFT, ca đặc biệt bậc-thấp-nhất (`algos/expert_iteration.py` · `expert_iteration` :101)
> **Câu hỏi first-principles:** RL không-policy-gradient rẻ nhất trông thế nào — lọc-rồi-SFT tại sao *cải
> thiện* model, và khiếm khuyết cấu trúc nào khiến nó plateau, buộc phải lên GRPO?
> **Neo (invariant · MEASURED):** `kept_fraction` tăng đơn điệu mỗi step, `[0] < 0.25` → `[-1] ≥ 0.9`
> (test `test_ei_kept_fraction_rises_over_steps` :128); zero-kept step → tham số model **bất động**
> (`test_ei_zero_kept_leaves_model_untouched` :160, `torch.equal` before/after); loop deterministic theo seed.

**1. Feynman — bài toán bằng lời.** EI (= STaR/RFT) là "học từ chính thành công của mình": sample G rollout
mỗi task từ policy hiện tại, GIỮ LẠI chỉ những cái ĐÚNG (verifiable grader), rồi SFT trên tập lọc — lặp.
Nó *khuếch đại* xác suất mà model ĐÃ đặt lên hành vi đúng. Analogy: học sinh làm 8 bản nháp, giữ 2 bản đúng,
chép lại 2 bản đó cho nhuần, rồi làm đề khó hơn. Đánh đổi: **không cần advantage/IS/critic** — chỉ keep/drop
nhị phân + SFT (rẻ, ổn định). Nhưng đó cũng chính là trần của nó.

**2. Dẫn xuất từ đầu (derive).** EI = reward-weighted regression với reward nhị phân + hard threshold:
gradient = `Σ_{y: R(y)≥τ} ∇log π(y)` (mọi token của rollout-được-giữ được reinforce ĐỀU NHAU, SFT). So Bài
8.3: đây là REINFORCE với `A = 1[R≥τ]` và *không có nhánh âm*. Ba khiếm khuyết cấu trúc suy ra trực tiếp
(docstring :10-18, chính là DoD):
- **Không credit-assignment per-token:** một lời giải gần-đúng-chỉ-một-bước-sai bị DROP hoàn toàn → dạy
  đúng bằng noise ngẫu nhiên (= 0). Token tốt trong rollout xấu mất trắng.
- **Không gradient âm:** không có lực đẩy xác suất RA KHỎI đáp án sai (chỉ kéo lên đáp án đúng).
- **Không áp lực exploration** ngoài nhiệt độ sample. Khi policy ngừng sinh rollout ĐÚNG MỚI trên task
  chưa-giải-được, tập lọc ngừng đổi → fixed point → task khó ở lại chưa giải. Đó CHÍNH là động lực GRPO.

**3. Trace code.** `expert_iteration` (:101) mỗi step: với mỗi task `sample_fn(task, group_size)` → grade
(`env.grade` :146) → nếu `reward >= keep_threshold` (default 1.0 = format AND answer, :151) thì giữ;
`deduplicate` (:154) bỏ trùng exact `(task_id, response_ids)` (G bản greedy giống hệt tính MỘT lần — bội số
không phải tín hiệu học). Rồi SFT trên tập kept (:161-178): `collate_prompt_response_ids` (:72, cùng masking
pinned như `sft.py` nhưng bắt đầu từ IDS vì rollout mang ids) → `get_response_log_probs` →
`sft_microbatch_train_step` (grad_accum=1) → `optimizer.step/zero_grad`. Metric đo policy TRƯỚC update của
step (`EIStepMetrics` :56). Zero-kept → khối SFT bị skip → model bất động (invariant test).

**4. Cổng teach-back.** (a) Giải thích "không credit-assignment" bằng ví dụ: rollout 10 bước, 9 bước đúng
1 bước sai → EI dạy được gì từ nó? (Zero — bị drop cả). (b) *Modify-and-predict:* nếu hạ `keep_threshold`
xuống 0.5 (giữ cả đúng-format-sai-đáp-án) — `kept_fraction` đổi thế nào và model học nhầm gì? Rồi: điều gì
ở GRPO advantage giải quyết đúng khiếm khuyết mà EI plateau vì nó?

**5. Frontier.** EI/STaR (Zelikman 2022) + RFT (Rejection-sampling Fine-Tuning) là baseline RL rẻ; nanochat
midtrain/SFT-then-RL dùng đúng spine này (`FRONTIER_2026_ABLATIONS.md` §4). Frontier hiện đại: iterative-DPO
và RAFT là họ hàng offline của EI. Câu interview (docstring :24): "Vì sao lọc-rồi-SFT (STaR/EI) cải thiện
reasoning model, và giới hạn cấu trúc nào khiến policy-gradient (GRPO) rốt cuộc là cần thiết?"

---

## Bài 8.6 — DPO: dẫn xuất contrastive không-reward-model từ objective RLHF-KL (`algos/dpo.py` · `per_instance_dpo_loss` :97 · `bradley_terry_rm_loss` :66)
> **Câu hỏi first-principles:** RLHF = reward model + PPO online, đắt và phức tạp. Làm sao *thu gọn* reward
> model VÀO chính policy, biến RL online thành một loss contrastive offline chỉ cần bốn log-prob?
> **Neo (invariant · MEASURED):** tại `π_θ == π_ref` margin = 0 và loss = **log 2 chính xác** với MỌI β
> (test `test_policy_equals_ref_gives_log2_exactly` :129, `abs_tol=1e-7`); hand-computed = `−log σ(β(δ−ρ))`
> (:150-174); **DPO ≡ Bradley-Terry(implicit reward)** như identity (fixture tiny-gpt2 0.9104,
> `test_dpo_algos.py` :11).

**1. Feynman — bài toán bằng lời.** RLHF cổ điển: (1) học một reward model `r(x,y)` từ so-sánh-cặp của
người, (2) PPO tối ưu policy chống lại `r` — online, phải sample, hai model, dễ sập. DPO hỏi: nếu ta ĐÃ biết
*nghiệm tối ưu dạng đóng* của bước (2) là gì theo `r`, có thể *đảo ngược* để biểu diễn `r` theo chính policy,
rồi cắm vào loss so-sánh-cặp — bỏ hẳn reward model VÀ RL online? Câu trả lời là có. Analogy: thay vì "học
thước đo rồi tối ưu theo thước", ta nhận ra "chính tỉ số xác suất policy/ref ĐÃ LÀ thước đo" và tối ưu trực
tiếp. Đánh đổi: mất on-policy exploration — DPO chỉ *đổi trọng số* hành vi ĐÃ có trong dữ liệu preference,
không khám phá hành vi mới.

**2. Dẫn xuất từ đầu (derive) — bốn bước (docstring :4-16).**
- **(i) Bradley-Terry:** một preference cặp là so-sánh logistic của reward vô hướng: `ℓ_RM = −log σ(r(x,y_w)
  − r(x,y_l))` (supp Eq. 1, `bradley_terry_rm_loss` :66). Reward bằng nhau → `log 2` (coin-flip).
- **(ii) Nghiệm KL-regularized RLHF:** cực đại `E[r] − β·KL(π_θ‖π_ref)` có nghiệm đóng `π*(y|x) ∝
  π_ref(y|x)·exp(r(x,y)/β)`.
- **(iii) Đảo ngược:** giải ra reward ẩn `r(x,y) = β·log(π_θ(y|x)/π_ref(y|x)) + β·log Z(x)`.
- **(iv) Cắm vào BT:** trong *hiệu* cặp, `β·log Z(x)` (intractable) **triệt tiêu** vì cùng x → Eq. 3:
  `ℓ_DPO = −log σ(β·[logπ_θ(y_w|x) − logπ_ref(y_w|x)] − β·[logπ_θ(y_l|x) − logπ_ref(y_l|x)])`.
  Đúng bốn log-prob điều kiện mỗi cặp, π_ref đóng băng. `test`: `ℓ_DPO == ℓ_BT(β·logratio_w, β·logratio_l)`
  — đúng phép rút gọn, như identity. β điều khiển độ mạnh dây-KL neo về π_ref.

**3. Trace code.** `per_instance_dpo_loss` (:97): wrap prompt bằng `ALPACA_PROMPT_TEMPLATE` (:59) → append
EOS id (mức token, không phải string — string EOS có thể retokenize) → `_response_log_prob` (:83) cho bốn
số: `logπ_θ(w)`, `logπ_θ(l)` **có graph**; `logπ_ref(w)`, `logπ_ref(l)` dưới `torch.no_grad` (:131, ref là
KL anchor đóng băng — không bao giờ gradient path) → `policy_margin − ref_margin` → `−F.logsigmoid(β·...)`
(:137), landing trên device của policy. `_response_log_prob` (:83) tổng logπ trên token response(+EOS) mức
chuỗi; logπ của prompt triệt tiêu cặp nên dùng dạng điều kiện tái dùng `get_response_log_probs` (seam chung
với SFT/GRPO). Tại `π_θ==π_ref`: bốn số triệt tiêu → σ(0) → log 2.

**4. Cổng teach-back.** (a) Dựng lại vì sao `Z(x)` biến mất — chỉ rõ nó phụ thuộc x không phụ thuộc y, và
hiệu cặp cùng x. (b) *Modify-and-predict:* nếu bỏ term `ref_margin` (coi π_ref đều) — loss thành gì, và tại
sao mất dây-KL khiến policy dễ trôi khỏi phân phối tự nhiên (reward-over-optimization)? Test `test_hand_
computed_case_ref_margin_subtracts` (:164) cho thấy loss có thể > log 2 dù policy tự-nó thích chosen — giải
thích tại sao.

**5. Frontier.** DPO (Rafailov 2023). `FRONTIER_PRACTICE_2026.md` A5 xếp preference-optimization là OPTIONAL
vì primitive DPO đã có; frontier thêm: online/iterative-DPO (2401.10020), IPO/KTO/SimPO (họ contrastive
biến-thể). Nuance senior: DPO offline nên chỉ reweight hành vi trong dữ liệu — với reasoning cần *khám phá*
lời giải mới thì RLVR (GRPO, Bài 8.3) mới là công cụ. Câu interview (docstring :39): "RLHF (RM+PPO) vs DPO —
derive reward model biến mất thế nào, β điều khiển gì, đi offline mất gì?" (mất on-policy exploration).

---

## Bài 8.7 — Rollout seam + train↔infer KL consistency + logging RL bắt buộc (`rollout/types.py` :19 · `utils/monitors.py` :131 · `algos/grpo.py:_log_step` :549 · ADR-0006 · L2 specs)
> **Câu hỏi first-principles:** rollout đến từ một *engine phục vụ* (nhanh, fused, quantized) nhưng gradient
> tính trên một *engine train* (eager, fp32) — nếu hai engine bất đồng về logπ thì mọi gradient tính trên
> policy KHÔNG PHẢI cái đang được serve. Đo và chặn drift đó thế nào, và tối thiểu phải log gì để một RL run
> *đọc-hiểu-được*?
> **Neo (số đo · MEASURED):** LocalBackend vs LocalBackend → **kl_train_infer ≈ 0 by construction** (scaffold
> CPU); cặp HF thật (Qwen2.5-0.5B, L2 spec §5): **eager/bf16 = 0.00982, eager/fp32 = 0.00173** (KL(train‖infer)
> exact, HALT@0.10 ok) — **prediction FALSIFIED** (fp32 GẦN bf16-serve hơn ~5.7×, vì SDPA cộng-dồn fp32 dù
> tensor bf16). Toàn bộ mandatory-log wired vào loop, không bolt-on.

**1. Feynman — bài toán bằng lời.** RL production có HAI process tách rời: rollout do inference engine
(vLLM/SGLang — nhanh, kernel fused, KV quantized) sinh; gradient do trainer (FSDP/Megatron — eager, precision
cao) tính. Chúng là hai *cài đặt khác nhau của cùng trọng số* → có thể cho logπ khác nhau cho cùng token.
Nếu khác nhiều, ta đang tối ưu một policy *không phải* cái phục vụ user — gradient thiên lệch âm thầm. Đánh
đổi/quyết định: đo bằng `kl_train_infer` (KL train‖infer) và HALT nếu vượt 0.10. Và vì phân phối dữ liệu RL
phụ thuộc θ (Bài 8.1), một run *không log* = một run mù: không biết nó chết vì entropy collapse, length
explosion, hay engine drift. Analogy: hai đồng hồ (bếp và phòng khách) phải khớp, không thì lịch nấu sai giờ.

**2. Dẫn xuất từ đầu (derive).** *Rollout lưu logπ nào (ADR-0006):* raw policy `log π(a|s) = log_softmax
(logits)[a]` tại **temperature 1** của token ĐÃ lấy — KHÔNG phải phân phối sau temperature/top-p. Vì nếu lưu
phân phối-lấy-mẫu, advantage/IS-ratio/KL sẽ *dính* vào knob exploration (tăng temp → rescale gradient âm
thầm). Temp/top-p chỉ định *token nào* được rút, không định *xác suất được log*. *Ba KL tách riêng
(monitors.py):* `KL(cur‖ref)` (trôi khỏi model gốc — dây neo), `KL(cur‖old)` (trôi trong off-policy epoch),
`kl_train_infer` (drift engine, load-bearing, HALT@0.10 :32). *IS ratio + ESS:* `r = exp(logπ_cur −
logπ_old)` (`importance_ratios` :59); `ESS = (Σw)²/Σ(w²)` (:67) → =N khi đều, →1 khi một trọng số áp đảo
(off-policy correction nổ). *Length-by-correctness:* độ dài response tách theo đúng/sai — tell của
verbosity-reward-hacking (chính length bias Dr.GRPO gỡ, Bài 8.3). Grad-bearing logπ luôn **recompute** trong
train step; logπ lưu trên rollout là π_old đóng băng, chỉ cho IS-ratio + clip (ADR-0006).

**3. Trace code.** `Rollout` (`rollout/types.py:19`) frozen, mang `prompt_ids/response_ids/logprobs/
stop_reason`. `LocalBackend` (`rollout/local.py:22`) chơi HAI vai (train + infer stand-in): `generate` (:29)
bắt inline logπ; `score` (:47) teacher-force độc lập cross-check → khớp vì cached-decode logit-identical với
recompute. `monitors.build_snapshot` (:131) bó MỌI trường bắt buộc vào `MonitorSnapshot` (:107) — keyword-only
all-required, nên không guardrail nào bị bỏ sót; `.halt` (:126) True khi `kl_train_infer > 0.10`. Loop wiring:
`grpo.py:_log_step` (:549) mỗi step gọi `_response_rows` (:402) cho cur/ref/old → `monitors.mean_kl` (:54)
hai KL, `importance_ratios` → `build_snapshot` (:580) với `kl_train_infer=0.0` (:582, vì LocalBackend cả hai
vai — kênh thành load-bearing NGAY khi SGLang thật cắm vào). `GRPOStepMetrics` (:305) carry entropy, hai KL,
IS-mean+ESS, reward stats, length_correct/incorrect_mean. Cross-link perf: engine-drift đo thật ở
`docs/design/L2_kl_train_infer_SPEC.md` (backends SGLang/HF mất trong incident 2026-06-29, rebuild-gated).

**4. Cổng teach-back.** (a) Vì sao rollout phải lưu logπ ở temperature 1 chứ không phải phân phối sau top-p
— nêu ĐỦ hậu quả nếu lưu sai (advantage/IS/KL dính exploration knob). (b) *Modify-and-predict:* prediction
"train fp32 drift khỏi serve bf16 NHIỀU hơn train bf16" bị FALSIFIED — giải thích cơ chế (SDPA cộng-dồn
softmax/PV bằng fp32 dù tensor bf16 → serve-bf16 thực chất là attention fp32-precision → eager/bf16 mới là
outlier). Rồi đoán: nếu serve chuyển INT4-KV thật, kl_train_infer đi hướng nào so với floor 0.00982?

**5. Frontier.** Đây là *nghề* RE post-training 2026 (`FRONTIER_PRACTICE_2026.md` :700): "sáng nào cũng nhìn
dashboard — entropy (collapse=run chết), KL(policy‖ref)/KL(policy‖old), IS-ratio histogram, reward mean/std
tách format-vs-answer, length-by-correctness". Train-infer mismatch: gần đây phát hiện BF16 rounding là gốc,
fix = UNIFORM FP16 cả train+infer (2510.26788) — rollout-only FP16 chống BF16-trainer KHÔNG hội tụ logprob.
Cross-link perf roadmap: MLA weight-absorption (serving) ở `roadmap/S2` Bài 2.8; primitive distributed TP/PP/EP
(nơi train/infer engine tách vật lý) ở `roadmap/S6`. Câu interview: "RL run của bạn chết — ba số đầu tiên bạn
nhìn là gì, và kl_train_infer đo cái gì mà hai KL kia không?"

---

*Đóng série:* cú nhảy hàm mục tiêu (8.1: MLE→expected-reward) → substrate log-prob+mask (8.2) → chuỗi
policy-gradient giảm-variance/bias REINFORCE→baseline→clip→GRPO→Dr.GRPO (8.3, ±0.7071 vs ±0.5) → nguồn
reward verifiable (8.4, format×answer) → ca đặc biệt EI (8.5, plateau vì thiếu credit-assignment) → nhánh
offline DPO (8.6, log 2 tại π_θ=π_ref) → seam + KL consistency + logging bắt buộc (8.7, kl_train_infer). Tất
cả **built + toy-tested trên CPU**; "aha" model-thật (F7, reward↑ + độ-dài-đúng tăng + KL bounded) là
PREDICTION rental-gated, chưa [FACT]. Série này tựa lên série model+train (base phải đủ mạnh để RL bootstrap)
và là twin model-side của perf roadmap (nơi engine train/infer tách ra thật).
