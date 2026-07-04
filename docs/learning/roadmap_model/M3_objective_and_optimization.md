# SÉRIE M3 — THE OBJECTIVE + OPTIMIZATION (cách model HỌC), dẫn xuất từ đầu

> **Số dòng pin theo commit `4ad0ac5`.** Đây là roadmap *dẫn-xuất* (derive từ blank + trace) để dẫn
> đường cho teach-back sâu về sau — KHÔNG phải bài giảng đầy đủ mỗi mục. Mỗi Bài ~một màn hình, tự chứa.
> Twin của roadmap performance (`docs/learning/roadmap/`, phần serving+kernel); série này mổ *bản thân
> việc học*: hàm mất mát, optimizer, lịch LR, và Muon (frontier F1).
> Số đo thật: `bench/RESULTS.md` (§Frontier ablations F1/F4 + Phase-0). Code: `src/scratch_llm/{model,optim,train}.py`.

**Vì sao série này.** M1 dựng *kiến trúc* (embedding → RoPE-attention → SwiGLU → norm → head) và M2 dựng
*forward + data path*: cho một chuỗi token vào, ra một tensor logits `(B, S, V)`. Nhưng logits chỉ là những
con số vô nghĩa cho tới khi ta có (a) một **thước đo** nói "dự đoán này sai bao nhiêu" và (b) một **luật cập
nhật** biến sai số đó thành thay đổi trọng số. Série này là *toàn bộ vòng học*: từ MLE dẫn ra cross-entropy
(Bài 3.1), từ SGD dẫn lên AdamW (3.2), lịch LR + grad-clip giữ vòng ổn định (3.3), rồi bước lên frontier —
Muon orthogonalize gradient để "mỗi hướng học đều nhau" (3.4) và param-split hybrid ghép Muon+AdamW (3.5).
Mỗi Bài tựa lên Bài trước: 3.1 định nghĩa *cái* để tối thiểu hoá; 3.2–3.3 là *cách* tối thiểu hoá cổ điển;
3.4–3.5 là *cách 2026*. Sợi chỉ xuyên suốt: **một scalar loss → một gradient → một update có RMS đúng band**.

Thứ tự đọc: **3.1 (cái gì minimize) → 3.2 (update cổ điển) → 3.3 (lịch + clip giữ ổn định) → 3.4 (Muon:
orthogonalize) → 3.5 (hybrid split + wiring).**

---

## Bài 3.1 — cross_entropy: từ MLE tới causal-LM loss, log V ở init, logsumexp (`src/scratch_llm/model.py` · `cross_entropy` :969)
> **Câu hỏi first-principles:** vì sao "đoán token kế tiếp" lại đúng bằng *tối thiểu hoá cross-entropy*, và vì sao một model *chưa học gì* phải cho loss ≈ log(vocab_size) — không hơn, không kém?
> **Neo (MEASURED invariant):** loss-at-init ≈ log V. `tests/test_model.py:46` `test_loss_at_init_is_log_vocab` — model tươi trên dữ liệu ngẫu nhiên cho CE ≈ log(vocab_size) trong băng ±0.3. Lệch ⇒ bug ở head/embedding/mask. Đây là oracle đúng-đắn *rẻ nhất* cả stack (CLAUDE.md §Engineering disciplines #1).

**1. Feynman — bài toán bằng lời.** Một LM là một *máy nén*: nó gán xác suất cho câu tiếp theo. "Học tốt" =
"gán xác suất cao cho token thật sự xuất hiện". Cách đo tự nhiên: **maximum likelihood** — chọn tham số θ làm
cực đại xác suất của toàn corpus dưới model. Nhưng "nhân xác suất của triệu token" thì underflow về 0; ta lấy
**log** (biến tích thành tổng, đơn điệu nên cùng argmax) rồi **đổi dấu** (biến "cực đại likelihood" thành "cực
tiểu loss"). Kết quả đúng là cross-entropy trung bình. Analogy: loss = "số bit trung bình bạn phải trả để mã
hoá token thật khi cá cược theo phân phối model" (Shannon). Đánh đổi cốt lõi: ta *không* tối ưu accuracy (không
khả vi), mà tối ưu một surrogate khả vi — negative log-likelihood — smooth để backprop chạy được.

**2. Dẫn xuất từ đầu (derive).** Likelihood của corpus dưới chain-rule causal: `p(x) = Π_t p(x_t | x_<t)`.
Log-likelihood `= Σ_t log p(x_t | x_<t)`. Đặt `p(· | x_<t) = softmax(z_t)` với z_t = logits tại vị trí t. Khi
đó `−log p(x_t|x_<t) = −log softmax(z_t)[x_t] = log(Σ_j exp z_{t,j}) − z_{t,x_t}` = **logsumexp(z_t) − z_{t,target}**.
Đây CHÍNH là công thức code (:976–977). Hai điểm chốt:
- **Label shift (causal).** target tại vị trí t là token t+1: `inputs = ids[:, :-1]`, `targets = ids[:, 1:]`
  (test :74). Mỗi vị trí dự đoán *cái kế*, không phải chính nó — đó là "causal" trong causal-LM.
- **loss-at-init = log V.** Ở init logits ≈ đều (weights nhỏ, RMSNorm chuẩn hoá) ⇒ softmax ≈ uniform 1/V ⇒
  `−log(1/V) = log V`. Với V=vocab_size, đây là *entropy của phân phối đều* — chặn trên của mọi model có ích.
  Nếu đo được > log V nhiều: head bị lệch scale hoặc mask rò tương lai (thấy target ⇒ loss < log V một cách
  gian lận). Đây là lý do nó là oracle #1.
- **Ổn định số (logsumexp).** Không bao giờ tính `exp(z)` trực tiếp (z lớn → inf). `logsumexp` trừ max ra
  trước: `log Σ exp(z_j) = m + log Σ exp(z_j − m)` với m = max z. Code (:975) cast `.float()` trước để dù
  forward chạy bf16, việc trừ/luỹ thừa của loss vẫn fp32 — chống mất mát chính xác ở đuôi.

**3. Trace code.** `cross_entropy(logits, targets)` (:969): `logits.float()` (:975, ép fp32) → `log_z =
logsumexp(logits, dim=-1)` (:976, mẫu số ổn định) → `chosen = logits.gather(-1, targets…)` (:977, tử số =
logit của token thật) → `(log_z − chosen).mean()` (:978, trung bình mọi token). Gọi từ `train.py:166/168`:
`cross_entropy(logits, targets) + aux.total` (dense: aux=0; MoE: cộng aux/z-loss của Bài série MoE). Test pin:
`test_loss_at_init_is_log_vocab` (test_model.py:46) + `test_overfit_one_batch` (test_optim.py:67, đây cũng dùng
chính hàm này làm loss). Lưu ý wiring: `logits` chưa softmax — hàm nhận *raw logits*, không nhận log-prob.

**4. Cổng teach-back.** (a) Dẫn lại từ MLE tới `logsumexp(z) − z_target` — chạm được ba bước: chain-rule →
log → đổi dấu. Vì sao init cho đúng log V (không phải 0, không phải log 2)? (b) *Sửa-và-đoán:* nếu QUÊN label
shift và đặt `targets = ids` (dự đoán chính token đang nhìn) — loss-at-init còn ≈ log V không, và loss sẽ tụt
về đâu sau vài bước (gợi: model học identity qua leak nếu mask hở; nếu mask kín thì vị trí t phải đoán x_t từ
x_<t — bài toán khác hẳn)?

**5. Frontier.** DeepSeek-V3 (`modeling_deepseek_v3.py` / 2412.19437) và mọi LM 2026 vẫn dùng đúng token-level
CE này làm loss chính; các "cải tiến" chỉ *thêm* vào: **z-loss** (PaLM 2204.02311, phạt `logsumexp²` để giữ
logit bounded — ta có twin ở `moe.py` router z-loss) và **MTP aux-CE** (V3 dự đoán +2 token, F2 trong spec).
Ta giống hệt công thức lõi, khác ở chỗ chưa bật z-loss trên output head (c_z=0 ⇒ byte-identical, "free" khi
cần). Câu interview: "Vì sao loss-at-init phải ≈ log V, và nó bắt được lớp bug nào mà một unit test forward
thường bỏ lọt?" (head-scale, embedding-tie sai, causal-mask rò).

---

## Bài 3.2 — AdamW: từ SGD → momentum → Adam → decoupled weight decay (`src/scratch_llm/optim.py` · `AdamW` :24 · `step` :57)
> **Câu hỏi first-principles:** vì sao không dùng SGD thẳng — mỗi bước Adam thêm hai buffer bằng 2× số tham số, ta *mua* được gì bằng cái giá memory đó? Và vì sao "weight decay" phải TÁCH khỏi gradient?
> **Neo (MEASURED invariant):** (a) overfit-one-batch → loss < 0.05 trong 300 bước (`tests/test_optim.py:67`). (b) decoupled-WD chính xác: grad=0, lr=0.1, wd=0.5 ⇒ θ nhân đúng `1−lr·wd = 0.95` (`test_optim.py:32`, atol 1e-6). (c) trên quadratic hội tụ về target atol 1e-2 (`:19`).

**1. Feynman — bài toán bằng lời.** SGD đi xuống dốc bằng đúng gradient: `θ ← θ − lr·g`. Vấn đề: các chiều có
độ cong khác nhau — chiều dốc-thoải cần bước lớn, chiều dốc-đứng cần bước nhỏ; một `lr` chung không thể vừa cả
hai. Adam là "SGD có bộ nhớ + tự chuẩn hoá per-coordinate": nó giữ (i) *momentum* m (trung bình trượt của
gradient — làm mượt nhiễu, giữ đà) và (ii) *variance* v (trung bình trượt của g² — đo mỗi chiều dao động bao
nhiêu), rồi chia `m/√v` để **mỗi toạ độ tự có learning-rate riêng**. Giá phải trả: 2 buffer (m, v) = 2× param
— khoản memory *chi phối* ngân sách training (docstring :3–5). Analogy: SGD là người mù dò dốc bằng chân;
Adam là người dò dốc có la-bàn (m) và bản đồ độ-gồ-ghề mỗi hướng (v).

**2. Dẫn xuất từ đầu (derive).** Bốn lớp chồng lên nhau:
- **momentum:** `m ← β₁m + (1−β₁)g` — EMA của gradient, hằng thời gian ~1/(1−β₁).
- **RMSprop/variance:** `v ← β₂m + (1−β₂)g²` — EMA của bình phương, ước lượng scale mỗi chiều.
- **bias correction:** ở t nhỏ, m,v khởi 0 nên *chệch về 0*. Sửa: chia m cho `(1−β₁ᵗ)`, v cho `(1−β₂ᵗ)`. Code
  gộp cả hai vào step-size: `α_t = lr·√(1−β₂ᵗ)/(1−β₁ᵗ)` (:86) rồi `θ ← θ − α_t·m/(√v+ε)` (:87). Gộp thế này
  đúng đại số và rẻ hơn chia từng buffer.
- **decoupled weight decay (điểm AdamW ≠ Adam).** L2-reg cổ điển cộng `λθ` VÀO gradient → nó cũng bị `√v` chia
  → chiều nào v lớn thì bị decay *ít* đi, sai lệch. Loshchilov–Hutter (2019): tách decay ra khỏi bước Adam,
  áp thẳng lên θ: `θ ← θ − lr·λ·θ` (:90). Bất biến kiểm được: grad=0 ⇒ chỉ decay chạy ⇒ `θ·(1−lr·λ)` (test
  :32). Đây là "AdamW" — W = "decoupled Weight decay".
- **vì sao β₂=0.95 (không phải 0.999).** Default Adam 0.999 = cửa sổ variance ~1000 bước, quá ì với gradient
  LM *nhiễu*. LM (nhất là reasoning) gradient noisy hơn → v nên thích nghi nhanh hơn → β₂=0.95 (~cửa sổ 20
  bước). Docstring :34–35.

**3. Trace code.** `step` (:57) lặp `param_groups`→`params`: bỏ qua `grad is None` (:70); lazy-init `m,v =
zeros_like(p)` lần đầu (:74–77); `m.mul_(β₁).add_(g, α=1−β₁)` (:83) + `v.mul_(β₂).addcmul_(g,g, value=1−β₂)`
(:84) — in-place, không cấp tensor mới; `α_t` (:86); `p.addcdiv_(m, √v+ε, value=−α_t)` (:87); rồi decoupled WD
(:89–90). Wiring: `build_optimizer(kind="adamw")` (:355) bọc `model.parameters()`; loop `train.py:171`
`optimizer.step()` sau `gradient_clipping` (Bài 3.3). Test: `test_adamw_minimizes_quadratic` (:19),
`test_adamw_decoupled_weight_decay_shrinks_param` (:32).

**4. Cổng teach-back.** (a) Vẽ lại 4 lớp SGD→AdamW; giải thích *vì sao* coupled-L2 làm decay không đều còn
decoupled thì đều, bằng đúng chỗ `√v` chen vào. (b) *Sửa-và-đoán:* nếu đặt β₂=0.999 cho một run LM nhiễu —
loss-vs-step đổi thế nào ở giai đoạn đầu (gợi: v chậm cập nhật ⇒ step quá lớn ở chiều mới nổi ⇒ dễ spike)?

**5. Frontier.** DeepSeek-V3 (2412.19437) dùng đúng AdamW β=(0.9, 0.95), wd=0.1 — *chính* default của ta;
SmolLM3/nanochat cũng vậy. Đây là lý do AdamW vẫn là "THE default" 2026 (`FRONTIER_PRACTICE_2026.md`): rẻ,
robust, ai cũng hiểu. Frontier chỉ *thách thức* nó ở đúng phần "update trên ma trận 2-D" — đó là cửa ngõ vào
Muon (Bài 3.4). Câu interview: "Ba thứ Adam thêm so với SGD là gì, và vì sao decoupled WD lại quan trọng đúng
ở regularization?"

---

## Bài 3.3 — cosine schedule + warmup + gradient clipping: giữ vòng học ổn định (`optim.py` · `cosine_lr` :111 · `gradient_clipping` :95)
> **Câu hỏi first-principles:** learning-rate là *một* số — vì sao không để hằng, mà phải warmup rồi anneal theo hình cosine? Và một gradient khổng lồ lẻ tẻ có thể phá tan cả run — chặn nó thế nào mà KHÔNG méo hướng?
> **Neo (MEASURED invariant):** (a) 3 pha cosine đúng biên: `cosine_lr(0)=0`, `=max_lr` tại warmup, midpoint = `(max+min)/2`, phẳng `min_lr` sau (`test_optim.py:55`). (b) clip: grad norm 5 → đúng 1.0 khi cap=1; norm 0.5 < cap ⇒ **không đổi** (no-op), norm trả về = 5.0 pre-clip (`test_optim.py:42`, atol 1e-4).

**1. Feynman — bài toán bằng lời.** Hai cơ chế bảo hiểm cho vòng học, độc lập nhau:
- **Lịch LR (cosine).** Đầu run trọng số ngẫu nhiên, gradient hỗn loạn → bước lớn dễ nổ ⇒ **warmup**: bò từ 0
  lên max_lr tuyến tính, cho m,v của Adam "ấm" lên. Giữa run → **anneal** dần về min_lr theo cosine: sớm học
  nhanh (LR cao thoát vùng phẳng), cuối tinh chỉnh (LR thấp lắng vào đáy hẹp). Analogy: xe vào cua — vào chậm
  (warmup), giữa đường ga đều, gần đích rà phanh mượt (cosine tail) chứ không phanh gấp.
- **Grad clipping.** Thỉnh thoảng một batch "độc" cho gradient khổng lồ (norm 100×) → một bước Adam nhảy khỏi
  vùng tốt, loss spike/NaN. Clip = "van an toàn": nếu *tổng* ℓ₂-norm vượt ngưỡng thì co lại đúng ngưỡng, GIỮ
  NGUYÊN HƯỚNG (chỉ đổi độ dài). Đánh đổi: mất một chút signal của batch lớn, đổi lấy không bao giờ nổ.

**2. Dẫn xuất từ đầu (derive).** *cosine_lr* (:111), 3 pha:
- `step < warmup`: `max_lr · step/warmup` (:124) — dốc tuyến tính 0→max.
- `warmup ≤ step ≤ cosine`: `progress = (step−warmup)/(cosine−warmup)` ∈[0,1]; `min_lr + 0.5·(1+cos(π·
  progress))·(max−min)` (:126–127). Tại progress=0: cos0=1 ⇒ =max_lr; progress=1: cosπ=−1 ⇒ =min_lr;
  progress=0.5: cos(π/2)=0 ⇒ đúng `(max+min)/2`. Hình cosine (chứ tuyến tính) vì đạo hàm =0 ở hai đầu — vào
  và ra êm, không gãy khúc.
- `step > cosine`: phẳng `min_lr` (:128).
*gradient_clipping* (:95): gom mọi grad (:100), `total_norm = √Σ‖g‖²` — **GLOBAL** ℓ₂ trên toàn model, không
per-tensor (giữ *hướng* của vector gradient ghép) (:103). Nếu `> max` thì `scale = max/(norm+ε)` rồi
`g.mul_(scale)` in-place mọi grad (:104–107). Trả `total_norm` *pre-clip* để log (:108). Bất biến: dưới ngưỡng
⇒ no-op tuyệt đối (không đụng grad); trên ⇒ norm sau = max chính xác.

**3. Trace code.** Mỗi bước `train.py`: `lr = cosine_lr(step, max_lr, min_lr, warmup_steps, max_steps)` (:151)
→ ghi vào MỌI group: `group["lr"] = lr` (:153) — với hybrid Muon+AdamW thì `CombinedOptimizer.param_groups`
(Bài 3.5) trả *live* groups nên một dòng này mutate cả hai optimizer. Sau `loss.backward()`:
`gradient_clipping(model.parameters(), cfg.grad_clip)` (:170) → `optimizer.step()` (:171). Test:
`test_cosine_schedule_phases` (:55), `test_gradient_clipping_scales_when_over_and_noop_when_under` (:42).

**4. Cổng teach-back.** (a) Vì sao clip phải dùng norm GLOBAL, không phải clip từng tensor riêng? (gợi:
per-tensor đổi *hướng* của gradient ghép; global chỉ scale độ dài ⇒ vẫn là hướng steepest-descent.) (b)
*Sửa-và-đoán:* nếu bỏ warmup (LR = max_lr ngay từ step 0) trên một run bf16 lớn — điều gì hỏng trong ~vài chục
bước đầu, và vì sao nó tệ hơn ở β₂=0.95 (gợi: v chưa ấm ⇒ `m/√v` bùng ⇒ spike; nối về Bài 3.2)?

**5. Frontier.** 2026 nhiều lab chuyển sang **WSD** (warmup–stable–decay: MiniCPM 2404.06395, SmolLM3,
DeepSeek-V3) — thay đuôi cosine bằng "stable dài ở peak + decay ngắn cuối 10%", ưu điểm *không cần chốt horizon
trước* (branch-và-decay để đọc kết quả A3 scaling). Ta mới có cosine; WSD là một `wsd_lr()` nhỏ, cùng bất biến
phase-boundary (`FRONTIER_PRACTICE_2026.md` §WSD). Grad-clip 1.0 vẫn là default phổ quát. Câu interview: "Vì
sao cosine chứ không hằng LR, và WSD giải quyết ràng buộc nào mà cosine không?"

---

## Bài 3.4 — MUON: orthogonalize momentum bằng Newton–Schulz, RMS = 1/√max(A,B) (`optim.py` · `_zeropower_via_newtonschulz5` :136 · `Muon` :169)
> **Câu hỏi first-principles:** Adam chuẩn hoá gradient *per-element*; nhưng một ma trận trọng số có *cấu trúc* — vài singular direction nuốt hết update. Nếu ép mọi singular value của update ≈ 1 (orthogonalize), ta học được gì thêm? Và vì sao scale đúng là `0.2·√max(A,B)` chứ không phải `1/max`?
> **Neo (MEASURED, `bench/RESULTS.md` §Frontier F1):** sau 5 bước NS, singular values *nén vào băng ~[0.68, 1.14]* — σ_max < 1.35 (không bao giờ phồng), bulk q10>0.6 & q90<1.25, spread q90/q10 < 2 (input κ≫1 collapse) (`tests/test_optim.py:95`). Update scaled RMS ∈ (0.15, 0.28) ≈ **0.2**, độc lập shape (`test_optim.py:129`). GPU-verified: hybrid học, loss → 8.3e-4 (RESULTS §F1/F4). ⚠ Hai over-claim đã bị FALSIFIED và sửa (xem dưới).

**1. Feynman — bài toán bằng lời.** Gradient của một ma trận `[A,B]` = một ma trận. SVD của nó `g = UΣVᵀ`: Σ
là "mỗi hướng riêng học mạnh cỡ nào". Adam nhìn *từng ô* độc lập → bỏ lỡ cấu trúc này: nếu Σ rất lệch (một
direction áp đảo), update thực chất là rank thấp — chỉ đẩy model theo vài hướng. **Muon** nói: giữ nguyên
*hướng* (U, V) nhưng **san phẳng Σ về toàn 1** ⇒ update ≈ `UVᵀ`, semi-orthogonal — "học đều mọi hướng của ma
trận, không để hướng nào lấn". Analogy: gradient thô là ánh sáng lệch màu; orthogonalize = cân bằng trắng, giữ
hình nhưng phổ đều. Đánh đổi: mất thông tin *độ lớn* tương đối giữa các hướng, đổi lấy update giàu-rank hơn +
step-size dự đoán được (RMS cố định).

**2. Dẫn xuất từ đầu (derive).** Hai mảnh:
- **Newton–Schulz "zeroth power".** Muốn `UVᵀ` = `g·(gᵀg)^(−1/2)` — nhưng SVD/inverse-sqrt mỗi bước quá đắt.
  NS5 là *iteration đa thức* bậc-5 hội tụ về hàm dấu của Σ: `X ← a·X + (b·A + c·A²)·X` với `A = XXᵀ`, hệ số
  `(a,b,c)=(3.4445, −4.7750, 2.0315)` (:154) — Keller Jordan tune để *dốc cực đại tại 0* (kéo σ nhỏ lên nhanh)
  chứ không để hội tụ chính xác. Chuẩn bị: chia Frobenius norm `x/(‖x‖+ε)` (:159) — vì ‖·‖_F ≥ ‖·‖₂ nên MỌI σ
  ≤ 1 *trước khi* lặp (điều kiện hội tụ); transpose về hướng *rộng* (:156) để `XXᵀ` nhỏ FLOP; chạy bf16 (:155,
  iteration tự-sửa về fixed point nên đủ chính xác). Loop 5 lần (:160–163).
- **RMS-matching (Moonlight Lemma 1, corrected).** Sau orthogonalize, `O ≈ UVᵀ` full-rank có `‖O‖_F² =
  min(A,B)` trải trên `A·B` ô ⇒ **per-element RMS = √(min/(A·B)) = 1/√max(A,B)**. Muốn RMS của update ≈ 0.2
  (đúng band update của AdamW, Bài 3.2) ⇒ nhân `0.2·√max(A,B)` (:252). Đây là mấu chốt honesty: spec-draft ghi
  nhầm `1/max(A,B)` → verifier F1 sửa thành `1/√max` (`RESULTS.md` §8 [REFUTED→fixed]; `0.2·√max` và wd=0.1
  vốn đã đúng). Hệ quả: **một lịch LR/WD phục vụ cả Muon lẫn AdamW** — không cần sweep LR riêng cho Muon.
- **Update đầy đủ:** `buf ← momentum·buf + g` (:249); Nesterov `g̃ = g + momentum·buf` (:250); `O =
  NS5(g̃)` (:251); decoupled WD `θ ← (1−lr·wd)·θ` (:254) rồi `θ ← θ − lr·(0.2·√max)·O` (:255).
- **Honesty (đã đo, đừng over-claim):** NS5 *nén* σ vào [0.68, 1.14], KHÔNG đưa tất cả về đúng 1. Worst-case
  ma trận Gaussian *vuông* có σ_min ≈ 0.08 mà 5 bước không nâng nổi; median ≈ 0.77. Điều đó *cố hữu* với few-step
  NS và *không hại* Muon — Muon chỉ cần *hướng* (phổ xấp xỉ đều), không cần orthogonal chính xác. Test (:95)
  pin đúng bất biến đã-sửa: never-inflate + bulk-band + spread-collapse.

**3. Trace code.** `_zeropower_via_newtonschulz5(g, steps=5)` (:136): guard 2-D (:150) → bf16 (:155) →
transpose-nếu-tall (:156) → chia Frobenius (:159) → 5× quintic (:160–163: `aa=x@x.T`, `bb=b·aa+c·aa@aa`, `x=a·x
+bb@x`) → transpose-lại (:164). `Muon.step` (:222): lặp params, *raise nếu ndim≠2* (:239 — mọi non-2D phải sang
AdamW), buf momentum (:249), Nesterov (:250), ortho (:251), `scale=rms_scale·√max` (:252), decoupled WD (:254),
apply (:255). Test: `test_newton_schulz_orthogonalizes_singular_values` (:95),
`test_muon_update_rms_matches_adamw_band` (:129). GPU wiring verified: `RESULTS.md` §F1/F4 (fp32 eager & bf16
eager & fp32 compile đều LEARNED → ~9e-4).

**4. Cổng teach-back.** (a) Dẫn `1/√max(A,B)` từ `‖UVᵀ‖_F² = min(A,B)` — rồi giải thích vì sao `0.2·√max`
đưa RMS về đúng band AdamW *độc lập shape*. Vì sao đây là điều kiện để "một LR cho cả hai"? (b) *Sửa-và-đoán:*
nếu quên chia Frobenius norm ở :159 (vào NS với σ có thể > 1) — iteration làm gì với σ lớn (gợi: quintic phân kỳ
ngoài vùng hội tụ), và test :118 `σ_max < 1.35` bắn ở đâu?

**5. Frontier.** Đối chiếu impl vendored `torch/optim/_muon.py`: hàm `_adjust_lr` (:73) có ĐÚNG hai chế độ —
`"original"` = `√max(1, A/B)` (Keller Jordan gốc) và `"match_rms_adamw"` = `0.2·√max(A,B)` (:80–81) — ta
**hardcode đúng nhánh match_rms_adamw**, giống Moonlight/nanochat. Hệ số NS `(3.4445,−4.7750,2.0315)` khớp
`DEFAULT_A/B/C` của torch (:25–27) và Keller Jordan. Frontier chứng cứ: nanochat dùng Muon mặc định; Moonlight
(2502.16982, Lemma 1) — Muon-at-scale + đúng identity RMS=1/√max; **Kimi-K2 MuonClip** (2507.20534) — Muon ở
1T param / 15.5T token, zero loss-spike (thêm QK-Clip = F9). muP: framing đã *softened* — "Muon eases, not
replaces, muP" (Essential AI 2505.02222 transfer tới ~4B; `RESULTS §8`). Câu interview (chính F1): "Derive the
Muon update; vì sao orthogonalize momentum, và vì sao RMS-matching cho phép tái dùng LR của AdamW?"

---

## Bài 3.5 — Param-split + hybrid: tensor nào → Muon vs AdamW, và cái bẫy TIED-2D (`optim.py` · `split_muon_adamw_params` :260 · `CombinedOptimizer` :298 · `build_optimizer` :337)
> **Câu hỏi first-principles:** Muon chỉ chạy trên ma trận 2-D *hidden*; embedding/head/norm thì không. Làm sao partition tham số thành hai nhóm *đúng và disjoint* — và vì sao một tensor 2-D bị **tie** (embed=head chung storage) lại là cái bẫy phải route sang AdamW dù nó là 2-D?
> **Neo (MEASURED invariant):** partition là *disjoint cover*: `Σnumel(muon)+Σnumel(adamw) == Σnumel(unique params)`, tied tensor đếm một lần và nằm ở AdamW (`tests/test_optim.py:143`, cả tied lẫn untied `:167`). Hybrid overfit-one-batch → loss < 0.05 (`test_optim.py:179`). GPU-verified end-to-end: MuonAdamW loss 4.79 → 8.3e-4 (`RESULTS §F1/F4 :644`).

**1. Feynman — bài toán bằng lời.** Muon là dao mổ cho *khối biến đổi tuyến tính bên trong* (q/k/v/o proj,
MLP up/down) — nơi "học đều mọi hướng ma trận" có nghĩa. Nhưng:
- **Embedding & LM head** là *lookup / readout* theo vocab, không phải biến đổi hidden↔hidden; Muon-hoá chúng
  làm hỏng scale token-tần-suất. → AdamW (dù chúng 2-D). Đây là quy ước nanochat/Moonlight: *input/output layer
  luôn AdamW*.
- **Mọi tham số 1-D** (RMSNorm gain, bias, scalar) — không có "singular value" để san phẳng; NS5 raise trên
  ndim≠2. → AdamW.
- **Bẫy TIED-2D.** Khi `tie_embeddings=True`, `lm_head.weight = token_emb.weight` (`model.py:917`) — MỘT tensor
  2-D duy nhất *vừa là* embedding *vừa là* head. Nó 2-D nên "trông như" ứng viên Muon, nhưng nó CHÍNH là
  embed/head → *phải* AdamW. Nếu route nhầm sang Muon: (i) orthogonalize một bảng vocab là vô nghĩa, (ii) tệ
  hơn — hai optimizer cùng update một storage → double-step / xung đột. Đây là bẫy repo-specific mà spec F1 gọi
  tên riêng.
Đánh đổi cả bài: phức tạp hoá optimizer (hai luật update) để *đúng công cụ cho đúng loại tensor*.

**2. Dẫn xuất từ đầu (derive).** `split_muon_adamw_params` (:260): thu `special_ids` = id của
`token_emb.weight` và `lm_head.weight` (:277–282) — nếu tie, cả hai trỏ *cùng* id nên set có 1 phần tử. Duyệt
`model.parameters()` (:287) — PyTorch *đã yield tied tensor một lần* nên `seen` (:288) + kiểm `id` cho cover
disjoint tự nhiên. Luật route (:291): `id ∈ special_ids OR ndim≠2 → AdamW`; còn lại (2-D block proj) → Muon.
Bất biến chứng minh được: `len(muon)+len(adamw)` = số tensor *unique*, không overlap; tied-2D vào AdamW; mọi
Muon-param `ndim==2`.
*CombinedOptimizer* (:298): trình bày *một* interface optimizer mà `train.py`+checkpoint đã mong. `param_groups`
(:314) trả groups *live* nối chuỗi ⇒ lịch LR ghi `group["lr"]` mutate thật (Bài 3.3). `step` (:322) gọi tuần tự
mọi sub-opt; `state_dict`/`load_state_dict` (:326–334) round-trip từng sub-opt ⇒ checkpoint resume *khớp
chính xác*. *build_optimizer* (:337): `kind="adamw"` → một AdamW toàn bộ (:355); `kind="muon_adamw"` → split →
`Muon(muon_params)` + `AdamW(adamw_params)` → `CombinedOptimizer([muon, adamw])` (:357–360), **chia sẻ cùng
lr** (RMS-match cho phép, Bài 3.4).

**3. Trace code.** Khởi tạo: `train.py:134` `build_optimizer(kind=…, lr, betas, weight_decay)` → nếu hybrid,
`split_muon_adamw_params(model)` (:357). Loop: lịch ghi `group["lr"]` qua `CombinedOptimizer.param_groups`
(:153) → `gradient_clipping(model.parameters())` (một global-norm trên TẤT CẢ params, không phân biệt nhóm)
(:170) → `optimizer.step()` (:171) gọi cả hai. Test: `test_split_muon_adamw_params_routes_the_tied_tensor_to_adamw`
(:143), `…untied_keeps_both_embed_and_head_on_adamw` (:167), `test_overfit_one_batch_muon_hybrid` (:179),
`test_train.py:109` (Muon chỉ 2-D block; embed/head/norm trên AdamW).

**4. Cổng teach-back.** (a) Vì sao khi `tie_embeddings=True`, route tensor chung sang Muon gây *hai* lỗi (nêu
đủ cả hai: vô-nghĩa-orthogonalize + double-step xung đột storage)? Vì sao `model.parameters()` yield nó một lần
là điều kiện để partition disjoint "miễn phí"? (b) *Sửa-và-đoán:* nếu route rule dùng `ndim==2` LÀ ĐỦ (bỏ kiểm
`special_ids`) trên model *untied* — embed và head giờ vào nhóm nào, và overfit-one-batch có còn <0.05 không,
hay hỏng ở đâu (gợi: head 2-D untied bị Muon-hoá → readout scale sai)?

**5. Frontier.** Đây đúng split nanochat/Moonlight: `matrix_lr 0.02 / embedding_lr 0.2 / unembedding_lr 0.004,
momentum 0.95, ns_steps 5` (`FRONTIER_2026 §5 :167`) — ta hiện *chia sẻ một lr* cho cả hai (RMS-match làm được),
per-group LR à-la-nanochat là refinement sau (`build_optimizer` docstring :352). Impl vendored `torch/optim/
_muon.py` để người dùng tự gán param-group; nanochat/Moonlight *hardcode* luật "embed+head+1-D → AdamW, còn lại
→ Muon" — giống hệt ta. Câu interview: "Trong một MuonAdamW hybrid, tensor nào đi optimizer nào và vì sao — cụ
thể chuyện gì xảy ra với weight-tied embedding?"

---

*Đóng série.* Sợi chỉ: **logits → CE (log V ở init) → gradient → update**. AdamW là update per-element robust
(β₂=0.95, decoupled WD); cosine+warmup+clip giữ vòng khỏi nổ; Muon nâng update trên *ma trận* lên orthogonal
(RMS=1/√max → scale 0.2·√max để tái dùng LR AdamW); param-split ghép hai luật đúng-loại-tensor với bẫy tied-2D.
Trạng thái honesty (FOP-4): mọi *unit invariant* ở đây ĐÃ ĐO (NS band, RMS-match, overfit, disjoint cover) +
GPU-verified loop closes (Phase-0 `val_bpb 0.0206` in-sample; hybrid → 8.3e-4). Nhưng **headline F1** —
"Muon đạt loss của AdamW với **≥15% ít token hơn** ở iso-FLOP" — vẫn là **PREDICTION** (pre-registered
`RESULTS §F1 :622`, KILL nếu saving <5%), *chưa* chạy real-train (rental-gated). Série sau (M4+): objective
hậu-huấn (SFT → GRPO/Dr.GRPO) tựa lên đúng CE + optimizer này. Cross-link perf: distributed TP/PP/EP cho
optimizer state → `roadmap/S6`; MLA weight-absorption → `roadmap/S2 Bài 2.8`.
