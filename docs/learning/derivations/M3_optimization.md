# M3 — Objective + Optimization · First-Principles Derivations

> **Đây là gì.** Bản dẫn-xuất **từ nguyên lý gốc** cho chuỗi optimization M3 (Bài 3.1→3.5): từ "có gradient"
> (CE) → "dùng gradient sao cho hội tụ nhanh + ổn định" (AdamW → cosine → Muon → param-split). Cùng format
> M2: sự thật nền tảng → dẫn xuất → neo code `file·func·line` → hình ảnh → **số đo THẬT chạy lại được** →
> frontier + cổng phỏng vấn. Song sinh `M2_transformer_forward.md`; xem `README.md` cùng thư mục.
>
> **Cảnh báo neo.** Line number pin theo HEAD 2026-07-14 (`optim.py`, `model.py`). Trôi thì `grep` tên hàm.
>
> **Trạng thái viết:** 3.1 ✅ · 3.2 ✅ · 3.3 ✅ · 3.4 ✅ · 3.5 ✅ — **M3 COMPLETE** (2026-07-14, số đo CPU chạy lại được).

---

## Bức tranh lớn — optimization là gì?

Training = tìm `θ*` cực tiểu loss `L(θ)`. Ta chỉ có **gradient cục bộ** `g = ∇L` (hướng dốc nhất tại điểm
hiện tại). Một bước tối ưu = *dùng `g` (và lịch sử của nó) để chọn bước đi tiếp theo*. M3 dựng chuỗi:

```
 3.1  CE          : gradient ĐÚNG để đi theo (∂CE/∂z = softmax − onehot)
 3.2  AdamW       : SGD → momentum → Adam(÷√v) → decoupled WD    ← dùng gradient khôn ngoan
 3.3  schedule    : lr thay đổi theo thời gian (warmup + cosine) + grad clip
 3.4  Muon        : thay ÷√v bằng orthogonalization (Newton-Schulz)
 3.5  param-split : matrix params → Muon, còn lại → AdamW; bẫy TIED-2D
```

Sự thật xuyên suốt: **mỗi optimizer là một câu trả lời cho "loss landscape méo thế nào"** — ill-conditioning
(Adam), nhiễu (momentum), độ cong dị hướng (Muon).

---

## 3.1 · cross_entropy từ MLE — gradient = softmax − onehot

**Câu hỏi.** "Học" nghĩa là gì về mặt toán? Vì sao loss đúng là cross-entropy (không phải MSE)?

**Dẫn xuất MLE → NLL → CE.** Model gán `p_θ(token | ngữ cảnh) = softmax(z)`. "Học tốt" = chọn `θ` để dữ liệu
thật *ít bất ngờ nhất* = **Maximum Likelihood**:
```
maximize  Πₜ p_θ(xₜ|x_<t)   ──log──►   maximize Σₜ log p_θ   ──đổi dấu──►   minimize  −Σₜ log p_θ   (NLL)
```
Với `q` = one-hot tại token đúng `t`, cross-entropy `H(q,p) = −Σ_v q_v log p_v = −log p_t` = **đúng NLL**.
⇒ **CE *là* MLE viết lại**, không phải lựa chọn tuỳ tiện.

**Gradient — con số đẹp nhất.** `CE = logsumexp(z) − z_t` ⇒
```
∂CE/∂z_j = softmax(z)_j − [j==t]  =  (p − onehot)_j
```

**Neo code** (`model.py`):
```python
def cross_entropy(logits, targets):        # :490
    logits = logits.float()                # :496  fp32 chống overflow exp
    log_z  = torch.logsumexp(logits, -1)   # :497  log∘exp triệt tiêu (ổn định)
    chosen = logits.gather(-1, targets…)   # :498  = z_t
    return (log_z - chosen).mean()         # :499  = logsumexp − z_t
```

**Hình ảnh — gradient là "sai số dự đoán":**
```
target = 2
        v:   0     1     2     3     4
  p   :   0.61  0.10 [0.02] 0.23  0.04     ← model dự đoán (Σ=1)
onehot:    0     0    [1]    0     0       ← sự thật
  grad:   +.61  +.10 [−.98] +.23  +.04     ← p − onehot
         └push xuống┘ └KÉO LÊN┘ └push xuống┘
   token đúng bị kéo lên đúng (1−p_t)=.98 ; token sai bị đẩy xuống đúng p_j
```

**Số đo THẬT** (`V=5`, target=2):
```
grad (autograd) = [0.612, 0.098, -0.985, 0.231, 0.044]
p − onehot      = [0.612, 0.098, -0.985, 0.231, 0.044]   ⇒ allclose True
MLE: logit[t]+=0/2/5 → p_t = 0.015/0.100/0.691 → CE = 4.21/2.30/0.37 (tin hơn ⇒ loss thấp)
loss@init ≈ log V: đo 7.04 ≈ log1000 = 6.91
```

**Vì sao đẹp (không dùng MSE):** gradient = sai số dự đoán, **bounded [−1,1]**, `log∘exp` triệt tiêu ⇒
**no vanishing**. Softmax+MSE có thừa số `p(1−p)` → vanishing khi bão hoà. Gate = dẫn `∂CE/∂z` cold + vì sao
không MSE. Frontier: label smoothing, z-loss (phạt `logsumexp²` ổn định bf16).

---

## 3.2 · AdamW — SGD → momentum → Adam → decoupled WD

**Câu hỏi.** Có gradient rồi, đi thế nào để hội tụ nhanh khi loss landscape **méo** (ill-conditioned)?

### Vấn đề của SGD
`θ ← θ − lr·g`: **một `lr` cho mọi chiều**. Landscape có chiều dốc (`a=100`) + chiều thoải (`b=1`). `lr` phải
`≤ 2/a = 0.02` để chiều dốc không nổ → chiều thoải bò tốc độ `1−lr·b = 0.98`/step → **kẹt** (đo: sau 200 step
còn cách đáy 0.108).

### Momentum — EMA của gradient
```
m ← β₁·m + (1−β₁)·g       # optim.py:84
θ ← θ − lr·m
```
Chiều dốc: gradient đổi dấu liên tục → **triệt tiêu** trong trung bình (hết zigzag). Chiều thoải: gradient
cùng dấu → **tích luỹ** → tăng tốc (hòn bi có quán tính). Đo: `0.108 → 0.0001`.

### Adam — thêm EMA của grad², chia √v
```
v ← β₂·v + (1−β₂)·g²                # optim.py:85   "độ lớn điển hình² mỗi chiều"
θ ← θ − lr · m / (√v + eps)         # optim.py:88   CHIA √v = per-dim adaptive LR
```
Chia `√v` = **chuẩn hoá mỗi chiều về cùng scale**: chiều dốc (v lớn) bị ghì, chiều thoải (v nhỏ) được đẩy →
mọi chiều tiến đều → **đi gần thẳng**. Đo: hội tụ ở `lr=0.1` (5× max SGD).

### Bias correction
`m,v` init 0 → step đầu lệch về 0. Sửa: `α_t = lr·√(1−β₂ᵗ)/(1−β₁ᵗ)` (`optim.py:87`); `t→∞ ⇒ α_t→lr`. Gấp
cả 2 hệ số sửa vào **một** step-size.

### Decoupled Weight Decay (chữ "W")
```
θ ← θ − lr·λ·θ            # optim.py:91   TÁCH RIÊNG khỏi đường gradient
```
**L2 cũ (SAI với Adam):** cộng `λθ` vào grad → `g+λθ` **rồi bị chia √v** → param có v lớn thì decay `λθ/√v`
bị đè bẹp → decay **không đều**. AdamW tách decay ra → **shrink đều `(1−lr·λ)`** mọi param, độc lập grad.
Đo: grad=0, θ `10 → 9.9 = 10·(1−0.1·0.1)` — decay *dù không có gradient*.

**Neo code — 1 step đầy đủ** (`optim.py:84-91`):
```python
m.mul_(beta1).add_(grad, alpha=1-beta1)          # :84  m = β₁m + (1−β₁)g
v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)# :85  v = β₂v + (1−β₂)g²
alpha_t = lr * sqrt(1-beta2**t) / (1-beta1**t)   # :87  bias-corrected step size
p.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)# :88  θ −= α_t · m/(√v+eps)
if weight_decay != 0:
    p.add_(p, alpha=-lr*weight_decay)            # :91  θ −= lr·λ·θ  (decoupled)
```

**Hình ảnh — 3 quỹ đạo trên thung lũng hẹp `f=0.5(100x²+y²)`:**
```
 dốc │  SGD    : ●╱╲╱╲╱╲______   zigzag chiều dốc, chiều thoải bò chậm  → dist 0.108
 (x) │ momentum: ●╲__________    quán tính triệt tiêu zigzag, tăng tốc  → dist 0.0001
     │  Adam   : ●───────────    ÷√v: mỗi chiều tự chỉnh LR, đi thẳng   → dist 0.0003 (lr 5×)
 đáy └────────────────────► thoải (y)
```

**Số đo THẬT** (200 step, `f=0.5(100x²+y²)`):
```
SGD (lr=0.019)          dist = 0.1078   (chiều thoải bị lr ghì)
SGD+momentum(β=.9)      dist = 0.0001
Adam (lr=0.1, class thật) dist = 0.0003 (lr gấp 5× vẫn ổn)
decoupled WD grad=0:    θ 10.0 → 9.9000  = 10·(1−lr·λ)
```

**Frontier / cổng.** LLM chuẩn: `betas=(0.9, 0.95)` (β₂ thấp hơn CV vì grad LM nhiễu hơn), `wd=0.1`. Gate =
dẫn Adam từ SGD + vì sao **decoupled** WD (L2-in-Adam hỏng). Trait = first-principles. Muon (3.4) thay ÷√v
bằng orthogonalization.

---

## 3.3 · cosine schedule + warmup + gradient clipping — giữ vòng học khỏi nổ

**Câu hỏi.** LR là *một* số. (a) Vì sao không để **hằng**, mà phải **warmup** rồi **anneal theo cosine**?
(b) Một gradient khổng lồ lẻ tẻ có thể phá cả run — chặn nó thế nào mà **KHÔNG méo hướng**?

**Sự thật nền tảng.** Hai áp lực *độc lập* lên vòng học:
- **Thời gian (schedule).** Đầu run: weight ngẫu nhiên + buffer Adam `m,v` còn **lạnh** (=0) → `m/√v` chưa đáng
  tin → bước lớn dễ nổ. Cuối run: đang lắng vào đáy hẹp → bước lớn *nảy ra khỏi đáy*. ⇒ LR phải **thấp ở hai
  đầu, cao ở giữa**.
- **Không gian (clip).** Thỉnh thoảng một batch "độc" cho gradient norm 100× → một bước Adam nhảy khỏi vùng tốt
  → loss spike/NaN, **hỏng cả run** (bf16 thì NaN không hồi được). Cần van an toàn giới hạn *độ dài* bước mà giữ
  *hướng* steepest-descent.

**Dẫn xuất — cosine 3 pha.**

*Pha 1 (warmup, `step < warmup`)* — dốc tuyến tính `0 → max_lr`:
```
lr = max_lr · step / warmup
```
Vì sao warmup? Adam step = `α_t · m/(√v+ε)`. Ở `t` nhỏ, `v` là EMA của `g²` khởi từ 0 ⇒ `v` nhỏ + nhiễu ⇒
`1/√v` **phình** ⇒ step khổng lồ trên các chiều mới nổi ⇒ spike. Đúng chỗ `β₂=0.95` (3.2) làm tệ hơn: cửa sổ
variance ngắn ⇒ `v` càng dễ under-estimate ở đầu. Warmup cho `m,v` "ấm" lên trước khi thả LR đầy.

*Pha 2 (cosine anneal, `warmup ≤ step ≤ cosine`)* — đặt `progress = (step−warmup)/(cosine−warmup) ∈ [0,1]`:
```
lr = min_lr + ½·(1 + cos(π·progress))·(max_lr − min_lr)
```
Kiểm **3 biên**: `progress=0 → cos0=1 → lr=max_lr`; `progress=1 → cosπ=−1 → lr=min_lr`; `progress=½ →
cos(π/2)=0 → lr=(max+min)/2`. **Vì sao cosine chứ không tuyến tính?** Đạo hàm `d/dprogress cos(π·progress) =
−π·sin(π·progress)` **= 0 ở HAI đầu** (progress=0 và 1). ⇒ LR *vào* và *ra* êm (slope→0), dốc nhất ở giữa —
không gãy khúc. Analogy: xe vào cua — rà ga êm hai đầu, ga mạnh giữa đường. Đo được: |slope| ở đỉnh warmup =
0.000222, ở đuôi cosine = 0.000222, còn ở midpoint = 0.014135 (**gấp ~64×** — đúng "êm hai đầu, dốc giữa").

*Pha 3 (`step > cosine`)* — phẳng `min_lr` (train quá horizon vẫn có LR sàn để tinh chỉnh, không về 0).

**Dẫn xuất — grad clip GLOBAL-norm.** Gom *mọi* grad thành một vector ghép; `total_norm = √Σ‖gᵢ‖²` = ℓ₂
**toàn model** (không per-tensor). Nếu `> max`: `scale = max/(norm+ε)`, nhân *mọi* grad in-place cùng `scale`.
**Vì sao GLOBAL, không per-tensor?** Vector gradient ghép có MỘT *hướng* (steepest-descent trên toàn θ). Nhân
mọi thành phần cùng một scalar = **co độ dài, giữ nguyên hướng** (cos=1). Clip từng tensor bằng cap riêng = nhân
*các khối khác nhau bằng scalar khác nhau* → **xoay** vector ghép → không còn steepest-descent. Đo: per-tensor
clip → `cos(joint, clipped) = 0.786` (xoay ~38°!); global → `1.000000`. Trả `total_norm` *pre-clip* để log.

**Neo code** (`src/scratch_llm/optim.py`):
```python
def cosine_lr(step, max_lr, min_lr, warmup_steps, cosine_steps):     # :112
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)                  # :125  pha 1 tuyến tính
    if step <= cosine_steps:
        progress = (step - warmup_steps) / max(1, cosine_steps - warmup_steps)      # :127
        return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)# :128 cosine
    return min_lr                                                    # :129  pha 3 sàn

def gradient_clipping(parameters, max_l2_norm, eps=1e-6):            # :96
    grads = [p.grad for p in parameters if p.grad is not None]       # :101
    total_norm = torch.sqrt(sum((g.detach()**2).sum() for g in grads))  # :104  GLOBAL ℓ₂
    if total_norm > max_l2_norm:                                     # :105  chỉ scale khi vượt
        scale = max_l2_norm / (total_norm + eps)                     # :106
        for g in grads:
            g.mul_(scale)                                            # :108  cùng scalar ⇒ giữ hướng
    return total_norm                                               # :109  pre-clip, để log
```
Wiring (`train.py`): `lr = cosine_lr(step, cfg.max_lr, cfg.min_lr, cfg.warmup_steps, cfg.max_steps)` (:262) →
`group["lr"] = lr` (:264, mutate *mọi* group — kể cả `CombinedOptimizer` §3.5) → `loss.backward()` (:280) →
`gradient_clipping(model.parameters(), cfg.grad_clip)` (:281) → `optimizer.step()` (:282).

**Hình ảnh — cosine curve + clip preserve-direction:**
```
lr                                                     grad ghép (mọi param), ‖g‖=5 > cap=1:
max┤   ╭──╮___                slope→0 ở đỉnh (êm)               g
   │  ╱     ╰──╮                                               ╱ θ
   │ ╱warmup    ╰───╮___ ← cosine anneal            ─────●────╱───► (đáy)
min┤╱               (─────── min_lr sàn ──────)
 0 ┼───────┬──────────┬──────────► step             GLOBAL  ×(1/5) → ‖g'‖=1, HƯỚNG y nguyên  cos=1.000
   0    warmup     cosine                           PER-TENSOR ×khác-nhau mỗi khối → VECTOR XOAY cos=0.786
```
Data journey (clip): `p.grad` là `list[Tensor]` mọi shape → reduce `Σ(g²).sum()` thành **scalar** `total_norm`
`()` fp32 → so `> max` → nhân in-place mọi `g` (shape/dtype giữ nguyên) → hàm trả `total_norm ()`.
Hand-trace: `g=[3,4,0]`, `‖g‖=5`, cap=1 → `scale=1/5.000001` → `g'=[0.6,0.8,0]`, `‖g'‖=1.0000`, cùng hướng.

**Số đo THẬT** (`cosine_lr`, `max=1 min=.1 warmup=10 cosine=110`):
```
step 0→0.000000  5→0.500000  10→1.000000(max)  30→0.914058  60→0.550000(=½)  90→0.185942  110→0.100000  160→0.100000(sàn)
|slope| đỉnh-warmup 0.000222  ·  midpoint 0.014135  ·  đuôi-cosine 0.000222   ⇒ giữa dốc ~64× hai đầu (cosine êm)
realistic (max 3e-4, warmup 2000, total 100000): step2000→3.000e-4(peak)  51000→1.650e-4  100000→3.000e-5
clip over cap: pre-norm 5.000000 → post 1.000000, cos(pre,post)=1.000000  ·  under cap (‖g‖=0.5): unchanged True
GLOBAL vs PER-TENSOR (joint ‖·‖=5.0359): global cos=1.000000  ·  per-tensor cos=0.786318 (hướng bị xoay)
```

**Frontier / cổng.** 2026 nhiều lab chuyển **WSD** (warmup–stable–decay: MiniCPM 2404.06395, SmolLM3,
DeepSeek-V3) — thay đuôi cosine bằng "stable dài ở peak + decay ngắn 10% cuối", ưu điểm *không cần chốt horizon
trước* (branch-và-decay để đọc A3 scaling). Ta mới có cosine; WSD = một `wsd_lr()` nhỏ cùng bất biến
phase-boundary. Grad-clip 1.0 vẫn default phổ quát. Gate = "vì sao cosine không hằng, và clip phải GLOBAL không
per-tensor" (dẫn direction-preservation). Trait = predict-the-number (slope ratio, cos=0.786). Table-stakes.

---

## 3.4 · Muon — orthogonalize momentum bằng Newton–Schulz, RMS = 1/√max(A,B)

**Câu hỏi.** Adam chuẩn hoá gradient *per-element* (từng ô độc lập). Nhưng một ma trận trọng số có *cấu trúc*:
SVD `g = UΣVᵀ` — vài singular direction có thể **nuốt hết** update. Nếu ép mọi singular value của update ≈ 1
(orthogonalize), ta học được gì thêm? Và vì sao scale đúng là `0.2·√max(A,B)`, **không** phải `1/max`?

**Sự thật nền tảng.** Gradient của một ma trận `[A,B]` = một ma trận, `g = UΣVᵀ`. `Σ = diag(σ)` = "mỗi hướng
riêng học mạnh cỡ nào". Nếu `Σ` lệch (một direction áp đảo) → update thực chất **rank thấp** → chỉ đẩy model
theo vài hướng, phí năng lực ma trận. Adam nhìn *từng ô* → **mù** với cấu trúc SVD này.

**Dẫn xuất — 2 mảnh.**

*Mảnh 1: "zeroth power" `g → UVᵀ`.* Muốn giữ *hướng* (U,V) nhưng san `Σ` về toàn 1 ⇒ update lý tưởng
`= g·(gᵀg)^(−1/2) = UVᵀ` (semi-orthogonal). SVD/inverse-sqrt mỗi step **quá đắt** ⇒ dùng **Newton–Schulz** —
iteration đa thức bậc-5: `X ← a·X + (b·A + c·A²)·X` với `A = XXᵀ`, `(a,b,c) = (3.4445, −4.7750, 2.0315)`.
Trên spectrum (X đường chéo σ), map scalar là `p(σ) = a·σ + b·σ³ + c·σ⁵`. Keller Jordan tune để **slope cực đại
tại 0** (`p'(0)=a=3.4445` — kéo σ nhỏ lên *nhanh nhất*), CHỨ không để hội tụ chính xác về 1.

Hai mẹo chuẩn bị *trước* iteration (đọc thẳng trong code):
- **Chia Frobenius** `x/(‖x‖_F+ε)`: vì `‖·‖_F = √Σσ² ≥ ‖·‖₂ = σ_max`, chia Frobenius ⇒ **mọi σ ≤ 1** đi vào.
  Bắt buộc vì quintic chỉ *contraction* trong vùng `σ ≲ 1.3`; `σ > 1.35 → p(σ)` **phóng** (đo: `p(1.5)=4.48
  → p(4.48)≈3244 → inf`). Đây là điều kiện hội tụ, không phải trang trí.
- **Transpose về hướng rộng** nếu tall (`shape[0]>shape[1]`): `XXᵀ` ít FLOP hơn (min-dim × min-dim); transpose
  lại ở cuối ⇒ **shape output = shape input** (đo: `(384,128)` vào → `(384,128)` ra).
- **bf16**: iteration tự-sửa về fixed point ⇒ bf16 đủ chính xác (Jordan writeup).

Loop 5 lần. Kết quả THẬT: σ *nén vào băng ~[0.68, 1.14]* quanh 1 — **KHÔNG** delta tại đúng 1. (Fixed point
của `p` giải `p(x)=x`: `2.0315x⁴ − 4.775x² + 2.4445 = 0 ⇒ x ≈ 0.868` và `≈ 1.264`; 5 bước dao động vào băng
giữa hai điểm này. Đo orbit từ σ₀=0.5: `[0.5, 1.189, 0.896, 0.824, 0.938, 0.765]` — không hội tụ về 1, mà *dao
động trong băng*.)

*Mảnh 2: RMS-matching (Moonlight Lemma 1, đã sửa).* Sau orthogonalize, `O ≈ UVᵀ` full-rank có `min(A,B)`
singular value ≈ 1 ⇒
```
‖O‖_F² = Σ σ² ≈ min(A,B)   (trải trên A·B ô)
⇒ per-element RMS = √( min(A,B) / (A·B) ) = √( 1/max(A,B) ) = 1/√max(A,B)
```
Muốn RMS của update ≈ 0.2 (đúng band update của AdamW, §3.2) ⇒ nhân `0.2·√max(A,B)`:
```
RMS(scaled) = 0.2·√max · (1/√max) = 0.2      ← ĐỘC LẬP shape
```
Hệ quả honesty **load-bearing**: một lịch **LR/WD phục vụ CẢ Muon lẫn AdamW** — không cần sweep LR riêng cho
Muon. Đây là lý do `build_optimizer` (§3.5) cho hai optimizer **share cùng `lr`**.

⚠ **Honesty ledger (FOP-4).** (i) Spec-draft ghi nhầm `1/max(A,B)` → verifier F1 sửa thành `1/√max` (nếu
`1/max` đúng thì test RMS lệch một thừa số `√max`) — `RESULTS.md` §8 `[REFUTED→fixed]`. (ii) NS5 **KHÔNG** đưa
mọi σ về đúng 1: worst-case Gaussian *vuông* `256×256` có `σ_min ≈ 0.035`, `median ≈ 0.86` (κ~n, 5 bước không
nâng nổi đuôi ill-conditioned). Điều này *cố hữu* với few-step NS và *không hại* Muon — momentum-gradient thật
không worst-case, và Muon chỉ cần *hướng* (phổ **xấp xỉ** đều), không cần orthogonal chính xác.

**Neo code** (`src/scratch_llm/optim.py`):
```python
def _zeropower_via_newtonschulz5(g, steps=5):                # :137
    if g.ndim != 2: raise ValueError(...)                    # :151  Muon CHỈ 2-D
    a, b, c = 3.4445, -4.7750, 2.0315                        # :155  quintic coeffs (Keller Jordan)
    x = g.to(torch.bfloat16)                                 # :156
    transposed = x.shape[0] > x.shape[1]                     # :157  tall → làm việc ở hướng rộng
    if transposed: x = x.T                                   # :159
    x = x / (x.norm() + 1e-7)                                # :160  Frobenius ⇒ mọi σ ≤ 1 (điều kiện hội tụ)
    for _ in range(steps):                                   # :161
        aa = x @ x.T                                         # :162  A = XXᵀ
        bb = b * aa + c * (aa @ aa)                          # :163  bA + cA²
        x = a * x + bb @ x                                   # :164  X ← aX + (bA+cA²)X
    if transposed: x = x.T                                   # :166  shape ra = shape vào
    return x.to(g.dtype)                                     # :167

# Muon.step (:232) — 1 param 2-D:
buf.mul_(momentum).add_(grad)                                # :259  momentum buffer
g_eff = grad.add(buf, alpha=momentum) if nesterov else buf  # :260  Nesterov look-ahead
ortho = _zeropower_via_newtonschulz5(g_eff, ns_steps)       # :271  orthogonalize
scale = rms_scale * math.sqrt(max(p.shape[0], p.shape[1]))  # :272  0.2·√max(A,B)
p.mul_(1 - lr * weight_decay)                               # :274  decoupled WD (dùng θ hiện tại)
p.add_(ortho, alpha=-lr * scale)                            # :275  θ ← θ − lr·(0.2·√max)·O
```

**Hình ảnh — SVD flatten + quintic map:**
```
g = U Σ Vᵀ ,  Σ=diag(σ) LỆCH:  ▇ ▁ ▁ ▁   (1 hướng áp đảo → update rank thấp; Adam nhìn từng ô ⇒ mù)
                 ── NS5 (5×) ──►  ▆ ▆ ▆ ▆   (san ≈1: O≈UVᵀ semi-orthogonal ⇒ "học đều mọi hướng ma trận")

p(σ)=3.4445σ−4.775σ³+2.0315σ⁵          ┌ σ>1.35 → PHÓNG (p1.5=4.48→3244→inf) ⟸ VÌ SAO chia Frobenius trước
 4┤                              ╱     │
  │                             ╱      └ nếu KHÔNG chia: σ có thể >1 vào ⇒ iteration phân kỳ
 1┤····•────band [0.68,1.14]───•·······  σ=1 (không phải fixed point; dao động quanh 0.868 & 1.264)
  │  ╱  ↑ nén (đo: q10 0.69, q90 1.11)
 0┼─┴──────────────────────────────► σ    slope cực đại tại 0 (p(0.05)=0.17 = ×3.4, kéo σ nhỏ LÊN)
```
Data journey (`(384,128)` tall): `g (384,128) fp32` → `.bfloat16` → `transposed=True` → `x=(128,384)` →
`/‖x‖_F` (σ≤1) → 5× `[aa=(128,128); bb=(128,128); x=(128,384)]` → transpose lại → `(384,128) → .to(fp32)`.
Rồi `scale = 0.2·√max(384,128) = 0.2·√384 = 3.919`; `p.mul_(1−lr·wd)`; `p −= lr·3.919·O`.

**Số đo THẬT** (`_zeropower_via_newtonschulz5`, seed 0):
```
NS5 shape-preserve + nén phổ (input → output singular values):
  (128,384) wide : input spread q90/q10= 2.4 → output band [0.691,1.108] spread 1.60, σ_max 1.139 (<1.35)
  (256,256) square: input spread 10.0     → output min 0.114 med 0.867 max 1.202 band [0.696,1.126]
  (384,128) tall : out.shape=(384,128) giữ nguyên → band [0.688,1.115] (transpose path đúng)
RMS-match (scale=0.2·√max): (256,256) raw 0.05635 (≈1/√256=0.0625) → scaled 0.1803
                            (1024,256) raw 0.02979 (≈1/√1024=0.03125) → scaled 0.1907   [mọi shape ∈ (0.15,0.28)]
Honesty (256×256 worst-case): σ min 0.0354 · median 0.8581 · max 1.2029  (KHÔNG delta tại 1)
Quintic: p(0.05)=0.1716 (×3.4↑)  p(0.5)=1.1889  p(1.0)=0.7010  p(1.35)=2.0111  p(1.5)=4.4778  p(2.0)=33.70
   orbit σ0=0.5 → [0.5, 1.189, 0.896, 0.824, 0.938, 0.765]   ·   orbit σ0=1.5 → [1.5, 4.48, 3243.95, inf] (BLOW-UP)
ndim guard (1-D) → ValueError "needs a 2-D matrix"   ·   NS5 1024² CPU-bf16 7.2 s/call (bf16 emul CPU; GPU sub-ms)
```
> Neo GPU (không chạy được trên card này): hybrid học end-to-end, loss `4.79 → 8.3e-4`; NS-overhead <1% wall —
> ledger-cited `bench/RESULTS.md` §Frontier F1/F4 `[MEASURED-GPU]`. Headline F1 "Muon ≥15% ít token iso-FLOP"
> vẫn `[PREDICTED]` (rental-gated, KILL nếu saving <5%).

**Frontier / cổng.** Impl vendored `torch/optim/_muon.py` có đúng hai chế độ scale: `"original"=√max(1,A/B)`
(Keller Jordan gốc) và `"match_rms_adamw"=0.2·√max(A,B)` — ta **hardcode nhánh match_rms_adamw**, giống
Moonlight/nanochat. Hệ số NS `(3.4445,−4.7750,2.0315)` khớp `DEFAULT_A/B/C` của torch. Bằng chứng scale:
nanochat dùng Muon mặc định; **Moonlight** (2502.16982, Lemma 1) Muon-at-scale + identity RMS=1/√max;
**Kimi-K2 MuonClip** (2507.20534) Muon ở 1T param / 15.5T token, zero loss-spike (QK-Clip = F9,
`optim.py:392`). Gate (chính F1): "derive the Muon update; vì sao orthogonalize momentum, và vì sao RMS-matching
cho phép tái dùng LR của AdamW?" Trait = claims-honesty (band ≠ delta; `1/√max` fix). Bucket = **kernels/RL
scarce differentiator 2026**.

---

## 3.5 · param-split hybrid + bẫy TIED-2D — đúng công cụ cho đúng loại tensor

**Câu hỏi.** Muon chỉ chạy trên ma trận 2-D *hidden*; embedding/head/norm thì không. Partition params thành
hai nhóm *đúng + disjoint* thế nào? Và vì sao một tensor 2-D bị **tie** (embed=head chung storage) là cái bẫy
phải route sang AdamW **dù** nó là 2-D?

**Sự thật nền tảng.** Muon = dao mổ cho *biến đổi tuyến tính hidden↔hidden* (q/k/v/o proj, MLP up/down) — nơi
"học đều mọi hướng ma trận" (§3.4) có nghĩa. Ba loại tensor **KHÔNG** hợp:
- **Embedding & LM head** = lookup/readout theo *vocab* (mỗi hàng = 1 token), không phải biến đổi hidden.
  Orthogonalize một bảng vocab = **vô nghĩa** (phá scale token-tần-suất). → AdamW (dù 2-D). Quy ước
  nanochat/Moonlight: *input/output layer luôn AdamW*.
- **Mọi param 1-D** (RMSNorm gain, bias, scalar): không có "singular value" để san phẳng; `NS5 raise` trên
  `ndim≠2` (`optim.py:151/249`). → AdamW.
- **Bẫy TIED-2D.** `tie_embeddings=True` ⇒ `lm_head.weight = token_emb.weight` (`model.py:438`) ⇒ MỘT tensor
  2-D *vừa là* embed *vừa là* head. Nó 2-D nên "**trông như**" ứng viên Muon, nhưng CHÍNH là embed/head ⇒
  *phải* AdamW. Route nhầm sang Muon: (i) orthogonalize bảng vocab vô nghĩa; (ii) **tệ hơn** — hai optimizer
  cùng update một storage ⇒ **double-step / xung đột**. Bẫy repo-specific mà spec F1 gọi tên riêng.

Đánh đổi cả bài: phức tạp hoá optimizer (hai luật update) để **đúng công cụ cho đúng loại tensor**.

**Dẫn xuất.** `split_muon_adamw_params` (:280): thu `special_ids` = `id` của `token_emb.weight` +
`lm_head.weight` (:297–302) — nếu tie, **cả hai trỏ CÙNG id** ⇒ set 1 phần tử. Duyệt `model.parameters()`
(:307): PyTorch *yield tied tensor một lần* ⇒ `seen` (:306) + kiểm `id` cho **disjoint cover miễn phí**. Luật
route (:311): `id ∈ special_ids OR ndim≠2 → AdamW`; còn lại (2-D block proj) → Muon. Bất biến chứng minh được:
`Σnumel(muon)+Σnumel(adamw) == Σnumel(unique params)` (không overlap), tied-2D vào AdamW, mọi Muon-param
`ndim==2`.

`CombinedOptimizer` (:318): trình bày *một* interface optimizer mà `train.py` + checkpoint đã mong.
`param_groups` (:334) trả groups **live** nối chuỗi ⇒ lịch LR ghi `group["lr"]` (§3.3) mutate **cả hai** thật;
`step` (:342) gọi tuần tự mọi sub-opt; `state_dict`/`load_state_dict` (:346/349) round-trip từng sub-opt ⇒
checkpoint resume khớp *chính xác*.

`build_optimizer` (:357): `kind="adamw"` → một AdamW toàn bộ (:376); `kind="muon_adamw"` → split →
`Muon(muon_params)` + `AdamW(adamw_params)` → `CombinedOptimizer([muon, adamw])` (:378–387), **share cùng `lr`**
(RMS-match §3.4 cho phép — một schedule phục vụ cả hai).

**Neo code:**
```python
special_ids = set()                                   # :297
for attr in ("token_emb", "lm_head"):                 # :298
    weight = getattr(getattr(model, attr, None), "weight", None)
    if isinstance(weight, Tensor): special_ids.add(id(weight))   # :302  tie ⇒ 1 id chung
seen = set()                                          # :306
for p in model.parameters():                          # :307  yield tied tensor MỘT lần
    if not p.requires_grad or id(p) in seen: continue # :308
    seen.add(id(p))                                   # :310
    if id(p) in special_ids or p.ndim != 2:           # :311  ← LUẬT ROUTE
        adamw_params.append(p)                        # :312  embed/head/1-D → AdamW
    else:
        muon_params.append(p)                         # :314  2-D block → Muon
```

**Hình ảnh — disjoint cover + tied trap:**
```
model.parameters()  (PyTorch yield tied tensor ĐÚNG MỘT lần)
  ├─ token_emb.weight (V,d) ─┐ special_ids
  ├─ lm_head.weight   (d,V) ─┤  tie ⇒ CÙNG id ⇒ set 1 phần tử  ──────────► AdamW  ← BẪY: 2-D nhưng KHÔNG Muon
  ├─ final_norm.weight (d,)  1-D ─────────────────────────────────────────► AdamW
  ├─ blocks[i].attn.q_proj.weight (d,d) 2-D block ────────────────────────► Muon
  ├─ ... mọi proj q/k/v/o + MLP w1/w2/w3 (2-D) ...                        ─► Muon
  └─ route: (id∈special_ids) OR (ndim≠2) → AdamW ;  else → Muon
     ✔ Σnumel(muon)+Σnumel(adamw) == Σnumel(unique)   [disjoint cover, tied đếm 1 lần]
```
Data journey (config `V=256 d=128 n_layers=4 tie=True`): 38 unique tensors → 28 vào Muon (851,968 numel, mọi
`ndim==2`) + 10 vào AdamW (33,920 numel = embed/head-tied + các RMSNorm 1-D). Untie ⇒ head tách ra thành tensor
riêng → AdamW numel `33,920 → 66,688`, chênh **32,768 = V·d = 256·128** (đúng một bảng vocab-head thêm vào).

**Số đo THẬT** (`split_muon_adamw_params`, `V=256 d=128 n_layers=4 n_heads=8`):
```
tie=True : #muon=28  #adamw=10  #unique=38  ·  muon 851,968 + adamw 33,920 = 885,888 == total  ·  disjoint True overlap False
           token_emb IS lm_head True  →  in AdamW True / in Muon False  ·  all muon ndim==2 True  ·  final_norm(1-D) in AdamW True
tie=False: #muon=28  #adamw=11  #unique=39  ·  muon 851,968 + adamw 66,688 = 918,656 == total  ·  token_emb IS lm_head False
           chênh adamw numel 66,688 − 33,920 = 32,768 = V·d (head untied tách riêng, vẫn AdamW)
hybrid overfit-one-batch (Muon block + AdamW embed/head/norm, 300 step): CE 4.7826 (≈log64=4.1589) → 0.000005  (<0.05 ✓)
```
> Neo GPU: MuonAdamW end-to-end `4.79 → 8.3e-4` (`bench/RESULTS.md` §F1/F4 `[MEASURED-GPU]`).

**Frontier / cổng.** Đây đúng split nanochat/Moonlight: `matrix_lr 0.02 / embedding_lr 0.2 / unembedding_lr
0.004, momentum 0.95, ns_steps 5` — ta hiện *share một lr* (RMS-match làm được), per-group LR à-la-nanochat là
refinement sau (`build_optimizer` docstring :373). Impl vendored `torch/optim/_muon.py` để người dùng tự gán
param-group; nanochat/Moonlight **hardcode** luật "embed+head+1-D → AdamW, còn lại → Muon" — giống hệt ta. Gate
= "trong MuonAdamW hybrid, tensor nào đi optimizer nào và vì sao — cụ thể chuyện gì xảy ra với weight-tied
embedding?" Trait = systems-thinking (partition đúng + checkpoint round-trip). Bucket = training-infra.

---

## Bảng số đo THẬT (M3, chạy lại được)

| Concept | Đo | Kết quả | Ý nghĩa |
|---|---|---|---|
| 3.1 | ∂CE/∂z (autograd vs công thức) | khớp **allclose True** | gradient = softmax − onehot |
| 3.1 | MLE: đẩy logit[t] | p 0.015→0.69, CE 4.21→0.37 | min CE = max likelihood |
| 3.1 | loss@init | 7.04 ≈ log1000=6.91 | oracle |
| 3.2 | SGD ill-conditioned | dist 0.108 (kẹt) | 1 lr cho mọi chiều |
| 3.2 | +momentum | dist **0.0001** | EMA grad triệt zigzag |
| 3.2 | Adam ÷√v | dist 0.0003 @ lr 5× | per-dim adaptive LR |
| 3.2 | decoupled WD grad=0 | 10 → **9.9** | shrink đều (1−lrλ) |
| 3.3 | cosine phases | 0 → 1.0(max) → **0.55(½)** → 0.1(min) | 3 pha đúng biên |
| 3.3 | slope biên vs giữa | 0.000222 vs 0.014135 → **~64×** | cosine êm hai đầu, dốc giữa |
| 3.3 | clip over/under | 5.0 → **1.0** / 0.5 no-op | van an toàn, chỉ scale khi vượt |
| 3.3 | clip GLOBAL vs per-tensor | cos **1.000** vs 0.786 | global giữ hướng, per-tensor xoay |
| 3.4 | NS5 nén phổ (256²) | spread 10.0 → **1.62** | orthogonalize (σ về băng ~1) |
| 3.4 | NS5 shape (384,128) tall | out **(384,128)** giữ nguyên | transpose path đúng |
| 3.4 | RMS-match 0.2·√max | raw 0.056(≈1/√256) → **0.18** | 1 LR cho cả Muon+AdamW |
| 3.4 | honesty σ (256² worst) | min **0.035** med 0.86 (≠1) | band ≠ delta (few-step NS) |
| 3.4 | quintic no-Frobenius | orbit 1.5 → **inf** | vì sao chia ‖·‖_F trước |
| 3.5 | disjoint cover (tie) | 851,968+33,920 = **885,888** | Σmuon+Σadamw == unique |
| 3.5 | tied-2D route | token_emb **in AdamW** (not Muon) | bẫy tied → AdamW |
| 3.5 | untied−tied adamw | 66,688−33,920 = **32,768 = V·d** | head tách ra vẫn AdamW |
| 3.5 | hybrid overfit | CE 4.78 → **5e-6** (<0.05) | wiring Muon+AdamW đúng |

---

## Checklist recall COLD (che phần trên, tự trả lời)

1. **3.1** Dẫn `CE` từ MLE (3 mũi tên: likelihood→log→NLL). Vì sao NLL = cross-entropy?
2. **3.1** `∂CE/∂z_j = ?` Vì sao dạng đó "đẹp"? Vì sao KHÔNG dùng MSE cho phân loại?
3. **3.2** Vì sao SGD kẹt trên landscape ill-conditioned? `lr` bị chặn bởi cái gì?
4. **3.2** Momentum (EMA grad) giúp gì ở chiều dốc vs chiều thoải?
5. **3.2** Adam chia `√v` để làm gì? "Per-dim adaptive LR" nghĩa là gì?
6. **3.2** Vì sao **decoupled** WD? L2-in-Adam hỏng ở đâu? (gợi ý: `λθ` bị chia `√v`)
7. **3.2** Viết 1 step AdamW đầy đủ (m, v, α_t, update, decay) từ trí nhớ.
8. **3.3** Vì sao warmup? (dẫn từ `v` lạnh → `1/√v` phình). Vì sao cosine chứ không tuyến tính (slope ở biên)?
9. **3.3** Grad-clip phải dùng norm **GLOBAL** không per-tensor — vì sao? (dẫn direction-preservation, cos=1 vs 0.786).
10. **3.4** Dẫn `1/√max(A,B)` từ `‖UVᵀ‖_F² = min(A,B)`. Vì sao `0.2·√max` cho RMS≈0.2 *độc lập shape* ⇒ 1 LR cho cả hai?
11. **3.4** Vì sao PHẢI chia Frobenius trước NS? (quintic phân kỳ khi σ>1.35). NS5 đưa σ về đúng 1 hay về *băng*?
12. **3.5** Khi `tie_embeddings=True`, route tensor chung sang Muon gây **hai** lỗi nào? Vì sao `model.parameters()` yield-1-lần là điều kiện disjoint "miễn phí"?
13. **3.5** Trong hybrid, tensor nào → optimizer nào (embed/head/1-D/2-D-block)? Vì sao hai optimizer share được cùng `lr`?

> Cold cả 13 = **M3 thật sự OWNED** (interview-grade F1). Vấp câu nào → mở đúng mục, hoặc blank-slate hàm tương
> ứng (`_zeropower_via_newtonschulz5` / `split_muon_adamw_params` / `cosine_lr` → `raise NotImplementedError` →
> test đỏ → tự dẫn → xanh) để re-own bằng tay.

---

*Cross-ref: `M2_transformer_forward.md` · `roadmap_model/M3_objective_and_optimization.md` · `PROGRESS.md`
(ledger 89 Bài) · `bench/RESULTS.md` §Frontier F1/F4 (GPU-verified loop). Concept kế: M4 — training loop
(data loading · checkpoint · overfit-one-batch) rồi M8 — RL post-training (SFT → GRPO/Dr.GRPO) tựa lên đúng
CE + optimizer này.*
