# M2 — Transformer Forward Pass · First-Principles Derivations

> **Đây là gì.** Bản dẫn-xuất **từ nguyên lý gốc** (first principles) cho TOÀN BỘ forward pass của một
> LLM hiện đại — 6 micro-concept M2 (Bài 2.1→2.6). Mỗi mục: (1) **sự thật nền tảng** phải giải thích,
> (2) **dẫn xuất** từ đó, (3) **neo code** `file·func·line`, (4) **hình ảnh** (ASCII + shape/stride), (5)
> **số đo THẬT** (chạy trên chính repo này, không phán), (6) **frontier framing** + cổng phỏng vấn.
>
> **Cách dùng để re-learn.** Đọc phần *Câu hỏi* của mỗi mục → **tự trả lời cold** (che phần dưới) →
> mở ra đối chiếu. Cuối doc có **checklist recall cold** + bảng số đo. Đây là bạn đồng hành của
> `roadmap_model/M2_transformer_architecture.md` (reference chung) — doc này là *derivation lab* có số đo.
>
> **Cảnh báo neo.** Line number **trôi** theo commit. Cite ở đây pin theo HEAD ngày 2026-07-14
> (`model.py`). Nếu lệch, `grep` tên hàm — đừng tin số dòng cứng (luật repo: verify, don't trust).
>
> **Nguồn số đo.** Mọi số trong doc này chạy được lại bằng các `python -c` ghi trong từng mục; đó là
> "DoD là một profile, không phải test xanh" (FOP-3) áp cho việc học.

---

## Bức tranh lớn — forward pass là một hàm gì?

Một decoder LM là hàm `f: (chuỗi token id) → (phân phối xác suất token kế tiếp)`. Toàn bộ M2 dựng hàm đó:

```
token_ids (B,S) long
   │  token_emb  (V, d_model)                          [2.6 tied?]
   ▼
x (B,S,d_model)  ── "residual stream" = xa lộ thông tin chạy thẳng suốt N block
   │
   ├─►┌─ RMSNorm ─► GQA-Attention(+QK-norm, RoPE) ─┐   [2.1 attn · 2.2 GQA · 2.3 RoPE · 2.4 norm · 2.6 QK-norm]
   │  └──────────────── (nhánh) ───────────────────┘
   x = x + nhánh                                        ← pre-norm residual (identity path)
   │
   ├─►┌─ RMSNorm ─► SwiGLU MLP ─┐                       [2.4 norm · 2.5 SwiGLU]
   │  └──────── (nhánh) ────────┘
   x = x + nhánh
   │   (lặp ×N block)
   ▼
final_norm (RMSNorm) ─► lm_head (d_model, V) ─► logits (B,S,V)   [2.4 · 2.6]
   │
   ▼
cross_entropy(logits, targets) ─► loss   (≈ log V lúc init)      [2.6 · oracle Bài 1.1]
```

**Hai khối trộn/xử lý luân phiên:**
- **Attention** = *trộn thông tin GIỮA các token* (token này đọc token kia). ← 2.1–2.3
- **MLP (SwiGLU)** = *xử lý TỪNG token độc lập*, nơi chứa ~2/3 param, "kho kiến thức". ← 2.5
- **Norm + residual** = *chất keo* giữ tín hiệu + gradient ổn định qua độ sâu. ← 2.4
- **Embedding + CE** = *cửa vào/ra* nối không gian token ↔ không gian vector. ← 2.6

Một sự thật xuyên suốt: **mỗi thiết kế M2 là lời giải cho một "áp lực" cụ thể** — bộ nhớ (GQA), vị trí
(RoPE), độ sâu (pre-norm), biểu diễn (SwiGLU), ổn định số (QK-norm/CE). Học M2 = học *áp lực → lời giải*.

---

## 2.1 · Attention = differentiable KV-retrieval

**Câu hỏi.** Làm sao một token "đọc" thông tin từ các token khác, theo cách *khả vi* (học được bằng gradient)?

**Sự thật nền tảng.** Muốn học được, phép "tra cứu" phải **mềm** (soft): thay vì chọn cứng 1 token, ta lấy
**trung bình có trọng số** của mọi token, trọng số = độ "khớp" giữa câu hỏi và khoá.

**Dẫn xuất.** Mỗi token sinh 3 vector (phép chiếu tuyến tính học được):
- **Query** `q` = "tôi đang tìm gì",
- **Key** `k` = "tôi chứa đặc trưng gì" (để người khác tìm),
- **Value** `v` = "nếu bạn chọn tôi, đây là thứ tôi trả".

Độ khớp query↔key = **tích vô hướng** `q·k` (lớn khi cùng hướng). Chuẩn hoá thành xác suất bằng softmax:
```
attn(q, K, V) = softmax( q Kᵀ / √d_k ) V
```

**Vì sao chia `√d_k`?** `q·k = Σᵢ qᵢkᵢ` là tổng của `d_k` số hạng ~độc lập, phương sai ~`d_k` ⇒ độ lệch
chuẩn ~`√d_k`. Nếu không chia, logit **phình theo √d_k** → softmax **bão hoà** (một token ~1, còn lại ~0)
→ gradient qua softmax ~0 (đạo hàm softmax `p(1−p)→0`). Chia `√d_k` giữ logit ~O(1) ⇒ softmax không bão hoà
⇒ **gradient khoẻ ngay từ init**. Đây là "áp lực số học" đầu tiên.

**Causal mask.** LM dự đoán token kế ⇒ token `i` **không được nhìn** token `>i` (nếu không = gian lận, nhìn
đáp án). Mask = cộng `−∞` vào logit các vị trí tương lai trước softmax ⇒ `exp(−∞)=0` ⇒ trọng số 0.

**Neo code** (`src/scratch_llm/model.py`):
```python
def scaled_dot_product_attention(q, k, v, mask=None):        # :160
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)        # :166  chia √d_k
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))    # :168  causal = −∞
    attn = softmax(scores.float(), dim=-1).to(q.dtype)       # :169  softmax fp32 (ổn định)
    return attn @ v                                          # :170  trung bình có trọng số V
```

**Hình ảnh** (shape, 1 head, seq=4):
```
q:(4,d)  k:(4,d)              scores = qkᵀ/√d : (4,4)        sau causal mask (▓=−∞):
                              [ s00 s01 s02 s03 ]            [ s00  ▓   ▓   ▓  ]
q₂ hỏi ──► so khớp mọi k ──►  [ s10 s11 s12 s13 ]  ──mask──► [ s10 s11  ▓   ▓  ]
                              [ s20 s21 s22 s23 ]            [ s20 s21 s22  ▓  ]
                              [ s30 s31 s32 s33 ]            [ s30 s31 s32 s33 ]
                                                                    │ softmax mỗi hàng
                              out(i) = Σⱼ pᵢⱼ · vⱼ  ◄────────────────┘ (hàng i chỉ trộn token ≤ i)
```

**Frontier / cổng.** Đây là "differentiable dictionary". Interview gate: whiteboard attention + giải thích
`√d_k` từ phương sai (không phải "vì paper viết thế"). Trait: first-principles. Table-stakes.

---

## 2.2 · Multi-head + GQA/MQA — vì sao share KV

**Câu hỏi.** (a) Vì sao nhiều "head" thay vì một? (b) Vì sao các model 2026 cho query nhiều head nhưng
K/V ít head hơn?

**Sự thật nền tảng (a).** Một head = một *không gian con truy hồi* (một kiểu quan hệ: cú pháp / đồng tham
chiếu / vị trí…). Một head duy nhất phải nhồi mọi quan hệ vào một softmax → nghẽn. Chia `d_model` thành `H`
head song song ⇒ mỗi head học một kiểu quan hệ, rồi ghép lại. Head là **một chiều batch** (tính song song).

**Sự thật nền tảng (b) — áp lực bộ nhớ.** Khi *serving*, ta cache K,V của mọi token đã sinh (KV-cache) để
khỏi tính lại. Kích thước cache = `2 · n_kv · head_dim · n_layers · bytes / token`. **Chính cache này —
không phải weight — chặn batch size và độ dài context.** ⇒ giảm số K/V head = giảm cache tuyến tính.

**Dẫn xuất GQA.** Tách **hai núm độc lập**:
- **Query resolution** = số câu hỏi khác nhau = `n_heads` (rẻ, giữ cao),
- **KV resolution** = số bộ-nhớ/khoá khác nhau = `n_kv` (tốn cache).

Cho mỗi nhóm `n_heads/n_kv` query head **share chung 1 K/V head**. `n_kv=n_heads` = MHA đầy đủ; `n_kv=1` =
MQA (cực đoan); ở giữa = GQA.

**Vì sao GQA ~ giữ chất lượng, MQA hại?** MQA gộp mọi K/V về **1 không gian con** → 32 query head phải tra
cứu trên cùng một cơ sở → các head hết chuyên biệt hoá → mất chất lượng + bất ổn train. GQA `n_kv=8` giữ **8
không gian độc lập** — đã qua "điểm gãy" thực nghiệm (Ainslie 2023) ⇒ chất lượng ≈ MHA mà cache giảm `4×`.

**Neo code:**
```python
self.q_proj = Linear(d_model, n_heads * head_dim)   # :238  query: n_heads
self.k_proj = Linear(d_model, n_kv    * head_dim)   # :239  key:   n_kv  (ÍT hơn!) ← nguồn tiết kiệm
self.v_proj = Linear(d_model, n_kv    * head_dim)   # :240
...
q = ...view(b, s, n_heads, head_dim).transpose(1,2) # :262  (B, n_heads, S, head_dim)
if n_kv != n_heads:                                 # :318  GQA
    repeats = n_heads // n_kv                        # :319  GROUP SIZE (không phải n_kv!)
    k = k.repeat_interleave(repeats, dim=1)          # :320  8 head → 32 head (khối liền kề)
    v = v.repeat_interleave(repeats, dim=1)
```

**Hình ảnh — fan-in** (anchor `n_heads=32, n_kv=8, repeats=4`):
```
q0 q1 q2 q3 ─► kv0        repeat_interleave(kv, 4, dim=1):
q4 q5 q6 q7 ─► kv1          [kv0,kv0,kv0,kv0, kv1,kv1,kv1,kv1, ...]  ← khối LIỀN KỀ
...                         ⇒ mapping: query head i  →  kv head  i // repeats
q28..q31    ─► kv7
  \__ nhóm 4 share 1 K/V __/                (repeat/tile sẽ cho grouping SAI: [kv0..kv7, kv0..kv7])
```

**Vì sao phải repeat?** `QKᵀ` batch trên chiều head; `q` có 32, `k` có 8 → **không broadcast** (32 vs 8, không
cái nào =1) → `torch.matmul` **RAISES**. Nên "store 8, compute as 32": lưu ít (cache), nở ra lúc tính.

**Số đo THẬT** (bf16, `head_dim=128`):
```
KV bytes/tok/layer:  GQA(n_kv=8)=4096   MHA(n_kv=32)=16384   ratio = 4×
QKᵀ không repeat  →  RuntimeError: size of tensor a (32) must match b (8) at dim 1
n_heads=32, n_kv=3 → ValueError @model.py:81 (32 % 3 ≠ 0: repeats phải nguyên) — fail-fast ở biên
```

**Frontier / cổng.** Llama-3/Qwen: GQA `n_kv=8`. MQA gần như bị bỏ. **MLA** (DeepSeek) là bước kế: nén K/V
thành latent low-rank *chung* (cache < GQA) rồi up-project khôi phục biểu cảm per-head — cùng mẹo "store
small, compute big". Gate = giải thích KV-cache memory-wall (serving = **scarce differentiator 2026**).

---

## 2.3 · RoPE — từ yêu cầu relative-position

**Câu hỏi.** `softmax(QKᵀ)V` là **permutation-equivariant** — không biết thứ tự token. Làm sao nhét vị trí
vào, mà **chỉ khoảng cách tương đối `(m−n)`** mới ảnh hưởng attention (điều ngôn ngữ cần)?

**Sự thật nền tảng.** Nếu ta **quay** vector theo góc tỉ lệ vị trí, thì **tích vô hướng của hai vector đã
quay chỉ phụ thuộc hiệu góc** — vì phép quay *cộng góc*. Đó là toàn bộ phép màu.

**Dẫn xuất (m−n).** Chia `head_dim` thành cặp `(x₀,x₁),(x₂,x₃),…`; quay cặp `k` bằng góc `pos·θₖ`. Gọi `R_φ`
= ma trận quay 2×2 góc `φ`. Query ở `m`: `R_{mθ}q`; key ở `n`: `R_{nθ}k`. Logit:
```
(R_{mθ} q)ᵀ (R_{nθ} k) = qᵀ R_{mθ}ᵀ R_{nθ} k = qᵀ R_{(n−m)θ} k
```
Nhờ **2 tính chất của ma trận quay:**
1. `R_φᵀ = R_{−φ}` — vì quay là **orthogonal** (`RᵀR=I ⇒ Rᵀ=R⁻¹`), nghịch đảo "quay φ" = "quay −φ".
2. `R_a R_b = R_{a+b}` — quay **cộng góc** (30°+40°=70°).

`m, n` triệt tiêu, chỉ còn `(n−m)`. **Vì sao PHẢI quay, không cộng vector vị trí?** Cộng `p`: `(q+p_m)ᵀ(k+p_n)`
sinh số hạng chéo phụ thuộc riêng `m,n` — không gập về hiệu. Chỉ quay vừa **bảo toàn chuẩn** vừa **cộng-góc**.

**Phổ tần số.** `θₖ = 1/theta^(2k/d)`: cặp `k=0` quay **nhanh** (θ=1, phân giải vị trí *gần*), cặp cuối quay
**chậm** (phân giải vị trí *xa/thô*) — như **đồng hồ nhiều kim** (giây/phút/giờ) hay số nhị phân đa-hàng.

**Neo code:**
```python
inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2) / head_dim))   # :186  phổ tần số hình học
...
rot_even = x_even * cos - x_odd * sin       # :206  = R_φ · (x_even, x_odd)ᵀ
rot_odd  = x_even * sin + x_odd * cos       # :207
```

**Hình ảnh — một cặp quay:**
```
      x_odd
        │        (x_even, x_odd)
        │      ●
        │     ╱ ╲  quay góc φ = pos·θₖ
        │    ╱   ╲►● (rot_even, rot_odd)
        │   ╱  φ
        └──────────── x_even       ‖·‖ KHÔNG đổi (chỉ xoay) — đó là "bảo toàn chuẩn"
```

**Số đo THẬT** (`head_dim=8`):
```
(1) norm:  trước 3.395326 → sau 3.395327   ⇒ ROTATION (bảo toàn chuẩn)
(2) q_rot·k_rot chỉ f(m−n):  (5,3)=(7,5)=(12,10)=0.24372 (đều bằng nhau!);  (m,m)→raw q·k (R₀=I)
(3) inv_freq = [1.0, 0.1, 0.01, 0.001]  ⇒ cặp0 nhanh, cặp3 chậm
theta 1e4→1e6:  bước sóng cặp chậm nhất  6,283 → 198,692 token  ⇒ phân biệt vị trí XA hơn ⇒ long-context
```

**Frontier / cổng.** Llama-3 `theta=500000`; NTK/YaRN = chỉnh theta có nguyên lý cho long-context. Gate =
dẫn xuất `(m−n)` cold. Trait = first-principles derivation. Table-stakes arch (+ serving-adjacent).

---

## 2.4 · RMSNorm + pre-norm residual

**Câu hỏi.** (a) Vì sao residual stream cần *normalize*? (b) Vì sao norm đặt **trước** nhánh (pre-norm) chứ
không **bọc** tổng (post-norm)?

**Sự thật nền tảng (a).** Residual stream cộng dồn qua nhiều lớp → độ lớn **phình**. Đưa thẳng vào attention
→ logit nổ, softmax bão hoà, bf16 overflow. Cần **reset scale về ~1 trước mỗi nhánh**.

**Dẫn xuất RMSNorm.** Chuẩn hoá độ lớn = chia cho RMS: `out = x / RMS(x) · weight`, với `RMS(x)=√(mean(x²))`.
Khi đó `RMS(out) = RMS(x)/RMS(x) = 1` — **invariant**. `weight` (per-channel, init 1) cho phép học lại scale.
Bỏ **mean-subtraction** (centering) và **bias** của LayerNorm.

**Vì sao bỏ centering vẫn được?** Zhang & Sennrich 2019: lợi ích của LayerNorm là **re-scaling**, không phải
**re-centering**. Bỏ trừ mean ⇒ bớt 1 reduction + 1 pass đọc/ghi (norm là **memory-bound** ⇒ nhanh hơn *thật*).

**Sự thật nền tảng (b) — gradient qua độ sâu.** Pre-norm: `x = x + branch(norm(x))`. Đường `x` là **identity**
(không bị norm). Backprop: `∂L/∂x_in = ∂L/∂x_out · (I + ∂branch/∂x)` — số hạng **`I`** cho gradient chảy
thẳng, không suy giảm (gradient highway, ResNet). Post-norm: `x = norm(x + branch(x))` — gradient phải qua
**Jacobian của norm ở MỖI lớp**; Jacobian norm co lại → nhân dồn N lần → **tan biến**.

**Neo code:**
```python
rms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)   # :149  = 1/RMS(x)
return (x32 * rms * self.weight).to(dtype)                    # :150  = x/RMS · weight
...
x = x + self.attn(self.attn_norm(x), ...)    # :409  PRE-norm: norm trong nhánh, x là identity
x = x + delta                                # :414  delta = ffn(ffn_norm(x))
# ... sau N block:
self.final_norm = RMSNorm(d_model)           # :435  reset lần cuối (vì pre-norm làm ‖stream‖ phình)
```

**Hình ảnh — pre vs post:**
```
PRE-norm  (hiện đại):              POST-norm  (2017):
  x ──────────────┬──►  (+) ──► x    x ──┬──► (+) ──► norm ──► x
                  │      ▲            │    ▲
        norm→attn─┘──────┘           └─attn┘
  ‖ đường x KHÔNG norm ⇒ grad·I      ‖ norm BỌC tổng ⇒ grad qua Jᴺᵒʳᵐ mỗi lớp
```

**Số đo THẬT** (`d=64`, 50 block):
```
RMS invariant:  input RMS 6.53 → output RMS 1.000000
grad @ input:   PRE-norm 2.98e+02   POST-norm 7.45e-05   ⇒ tỉ lệ 4,005,028×  (post-norm tan biến)
‖residual stream‖: PRE 36.16 (phình → cần final_norm)   POST 8.00 (norm ghì lại mỗi lớp)
```

**Frontier / cổng.** Mọi LLM 2026 (Llama/Qwen/DeepSeek/Gemma) = pre-norm + RMSNorm. Post-norm 2017 cần LR
warmup, khó train sâu. Gate = pre vs post stability + norm ở đâu. Trait = predict-the-number (grad ratio).

---

## 2.5 · SwiGLU gated MLP (vs ReLU/GELU)

**Câu hỏi.** MLP cổ điển `w2(relu(w1 x))` — 2 ma trận, 1 phi tuyến. Vì sao model 2026 dùng **3 ma trận + cổng**?

**Sự thật nền tảng.** MLP xử lý *từng token*, là "kho kiến thức" (~2/3 param). Một **cổng nhân data-dependent**
cho FFN sức biểu diễn bậc-2 mà `relu` (bậc-1, piecewise-linear) không có — **miễn phí về param** nếu bóp `d_ff`.

**Dẫn xuất iso-param.** ReLU-MLP: `w1(d→4d)+w2(4d→d)` ⇒ `8d²` param (quy ước `d_ff=4d`). SwiGLU: `w1,w3(d→d_ff)
+ w2(d_ff→d)` ⇒ `3·d·d_ff`. Muốn **cùng ngân sách** `8d²`:
```
3·d·d_ff = 8d²   ⟹   d_ff = 8/3·d ≈ 2.667·d
```
Nếu để `d_ff=4d` thì SwiGLU = `12d²` (phình 50%) — so sánh KHÔNG công bằng. Luật frontier: so architecture
luôn **iso-param / iso-FLOP**.

**Dẫn xuất cổng.** Nhìn 1 chiều ẩn `j`: `h_j = silu((w1 x)_j) · (w3 x)_j` = **tích của 2 phép chiếu khác nhau
của cùng `x`** ⇒ tương tác **nhân** (bậc 2) ⇒ cổng `silu(w1 x)` *điều biến* giá trị `w3 x` (data-dependent
gating, họ **GLU** — Dauphin 2017; Shazeer 2020 đo SwiGLU perplexity tốt nhất).

**Vì sao SiLU không ReLU?** `silu(x)=x·sigmoid(x)`: mượt, **gradient≠0 khi x<0** (không "dead unit" như ReLU
zero-cứng), self-gate mềm (cho qua "70%" thay vì bật/tắt 0/100%). Cổng mềm khả vi → train tốt hơn.

**Neo code:**
```python
def silu(x): return x * torch.sigmoid(x)              # :211
class SwiGLU:
    self.w1 = Linear(d_model, d_ff)  # gate           # :220
    self.w3 = Linear(d_model, d_ff)  # up             # :221
    self.w2 = Linear(d_ff, d_model)  # down           # :222
    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x)) # :225  gate ⊙ up → down
```

**Hình ảnh — cổng:**
```
        x ──► w1 ──► SiLU ──┐
                            ⊙ (nhân element-wise) ──► w2 ──► out
        x ──► w3 ───────────┘
              gate  quyết định  bao nhiêu phần của "up" được cho qua, theo TỪNG input
```

**Số đo THẬT** (`d=4096`):
```
ReLU-MLP: d_ff=16384 (4d)   params 134,217,728  (2 ma trận)
SwiGLU:   d_ff=10923 (8/3d)  params 134,221,824  (3 ma trận)   ⇒ ratio 1.0000 (MATCH)
Llama-2-7B thật: d_ff=11008 = 2.688·d (8/3·4096=10923 làm tròn bội 256)
```

**Frontier / cổng.** SwiGLU = default FFN (Llama/Mistral/Qwen); Gemma=GeGLU; mỗi MoE expert là 1 SwiGLU
(`moe.py`). Gate = FFN design + iso-param comparison. Trait = iso-param honesty.

---

## 2.6 · QK-norm + tied embeddings + cross_entropy seam (đóng M2)

### 2.6a QK-norm — chặn logit khỏi nổ (bf16-instability #1)

**Sự thật.** Logit `q·kᵀ/√d`; nếu `q,k` lớn (stream phình / weight lớn) → logit nổ → softmax bão hoà +
bf16 mất chính xác/overflow → loss spike. **Giải:** RMSNorm mỗi head's `q,k` (**trước** RoPE) ⇒ `‖q‖=√head_dim`
cố định ⇒ logit ≤ `head_dim/√head_dim = √head_dim`, bounded.

**Vì sao trước RoPE?** RoPE là quay (bảo toàn chuẩn — 2.3) nên không phá norm; convention: chuẩn hoá per-head
rồi mới gắn vị trí.
```python
q = self.q_norm(q); k = self.k_norm(k)   # :267-268  RMSNorm(head_dim), no-op nếu cfg.qk_norm=False
q = self.rope(q, positions)              # :272  QK-norm TRƯỚC RoPE
```
Đo: `‖q‖ 364 → 5.6569 = √32` (bất kể input). Frontier: Qwen3/Gemma3/OLMo2; Kimi-K2 = QK-Clip (`:247-251`, F9).

### 2.6b Tied vs untied embeddings

**Sự thật.** `token_emb (V,d)` (vào) và `lm_head (d,V)` (ra) đều ánh xạ **vocab ↔ hidden** — về bản chất là
chuyển vị của cùng bản đồ. Tie = dùng chung 1 ma trận ⇒ tiết kiệm `V·d_model` param.
```python
self.lm_head = Linear(d_model, vocab_size)          # :436
if cfg.tie_embeddings:
    self.lm_head.weight = self.token_emb.weight     # :438  share 1 Parameter
```
Đo: untied − tied = **128,000 = V·d_model** (1000×128); scale thật `V=128k,d=4096` ⇒ ~512M param. Repo untie
mặc định (`:68` ADR-0004) để **giữ loss@init sạch**. Đánh đổi: tie = regularize/ít param; untie = linh hoạt
khi vai trò vào/ra phân kỳ (model lớn).

### 2.6c cross_entropy seam + loss-at-init = log V

**Dẫn xuất.** CE = `−log p(token đúng)`, `p=softmax(z)`:
```
−log( e^{z_t} / Σⱼ e^{zⱼ} ) = logsumexp(z) − z_t
```
```python
logits = logits.float()                            # :496  fp32: V lớn, exp/sum cần range
log_z  = torch.logsumexp(logits, dim=-1)           # :497
chosen = logits.gather(-1, targets…).squeeze(-1)   # :498  = z_t
return (log_z - chosen).mean()                     # :499
```
**Vì sao logsumexp?** `log∘exp` **triệt tiêu**, không vật chất hoá softmax, tự trừ max ⇒ không overflow.

**Loss@init ≈ log V.** Init: weight nhỏ ngẫu nhiên → logits ~đều → `p≈1/V` → `−log(1/V)=log V`. Đo:
**7.04 ≈ log 1000 = 6.91**. **Lệch xa log V ⇒ CÓ BUG** (init/mask/vocab). Đây là **oracle rẻ nhất** — full
circle về Bài 1.1. Gate = loss-at-init sanity + bf16 stability. Trait = cheapest-oracle discipline.

---

## Bảng số đo THẬT (chạy lại được — DoD-là-profile)

| Concept | Đo | Kết quả | Ý nghĩa |
|---|---|---|---|
| 2.1 | softmax stability | logit/√d_k ~O(1) | không bão hoà ⇒ grad khoẻ |
| 2.2 | KV cache GQA vs MHA | 4096 vs 16384 B/tok/lay → **4×** | serving memory-wall |
| 2.2 | QKᵀ không repeat | RuntimeError (32 vs 8) | vì sao cần repeat_interleave |
| 2.2 | n_kv=3, n_heads=32 | ValueError @:81 | fail-fast (32%3≠0) |
| 2.3 | RoPE norm | 3.395326 → 3.395327 | rotation (bảo toàn chuẩn) |
| 2.3 | q_rot·k_rot | (5,3)=(7,5)=(12,10)=**0.24372** | chỉ phụ thuộc (m−n) |
| 2.3 | theta 1e4→1e6 | wavelength 6,283 → 198,692 tok | long-context |
| 2.4 | RMS invariant | 6.53 → **1.000000** | reset scale |
| 2.4 | grad 50 block pre/post | 298 vs 7.4e-5 → **4.0e6×** | pre-norm gradient highway |
| 2.5 | SwiGLU iso-param | 134,221,824 ≈ 134,217,728 → **1.0000** | d_ff=8/3·d |
| 2.6 | ‖q‖ qua QK-norm | 364 → **5.6569 = √32** | chặn logit |
| 2.6 | tied tiết kiệm | **128,000 = V·d_model** | share vocab matrix |
| 2.6 | loss@init | **7.04 ≈ log1000=6.91** | oracle rẻ nhất |

---

## Checklist recall COLD (che phần trên, tự trả lời — đây là cách re-learn)

1. **2.1** Vì sao chia `√d_k`? (dẫn từ phương sai của `q·k`) Điều gì hỏng nếu bỏ?
2. **2.2** `n_heads=32, n_kv=8`: shape của `k` trước/sau `repeat_interleave`? `repeats`=? Vì sao `QKᵀ` cần nó?
3. **2.2** KV-cache ratio GQA vs MHA? Vì sao MQA (`n_kv=1`) hại còn GQA (`n_kv=8`) không?
4. **2.3** Dẫn `q_rot·k_rot = f(m−n)` — 2 tính chất nào của ma trận quay? Vì sao PHẢI quay?
5. **2.3** `theta` tăng → phân biệt vị trí xa hơn hay gần hơn? Vì sao?
6. **2.4** Vì sao `RMS(out)=1`? Vì sao pre-norm giữ grad khoẻ mà post-norm tan biến?
7. **2.4** RMSNorm bỏ gì của LayerNorm? Vì sao vẫn train tốt?
8. **2.5** Vì sao `d_ff=8/3·d`? Cổng cho FFN làm được gì mà ReLU-MLP không? Vì sao SiLU không ReLU?
9. **2.6** `‖q‖` sau QK-norm =? Vì sao chặn được logit? Vì sao trước RoPE?
10. **2.6** Tie embeddings tiết kiệm bao nhiêu param? Dẫn `CE = logsumexp(z) − z_t`. Vì sao loss@init≈log V?

> Trả lời cold được cả 10 = **M2 thật sự OWNED** (interview-grade). Vấp câu nào → mở đúng mục đó, hoặc
> blank-slate hàm tương ứng (`raise NotImplementedError` → test đỏ → tự dẫn → xanh) để re-own bằng tay.

---

*Cross-ref: `roadmap_model/M2_transformer_architecture.md` (reference chung) · `PROGRESS.md` (ledger 89 Bài) ·
`CURRICULUM.md` (con đường). Concept kế: M3 — objective + optimization (CE→AdamW→cosine→Muon).*
