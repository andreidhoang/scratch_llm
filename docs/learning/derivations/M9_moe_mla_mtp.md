# M9 — MoE · MLA · MTP · First-Principles Derivations (the frontier-architecture spine)

> **Đây là gì.** Bản dẫn-xuất **từ nguyên lý gốc** cho ba đòn bẩy kiến trúc frontier 2026 —
> 4 micro-concept M9 (Bài 9.1→9.4): **MoE** (tách *params* khỏi *FLOP*), **MLA** (tách *chất
> lượng attention* khỏi *byte KV/token*), **MTP** (tách *tín hiệu train* khỏi *một-nhãn-một-vị-trí*).
> Mỗi mục: (1) **câu hỏi** falsifiable, (2) **sự thật nền tảng** (áp lực vật lý/toán buộc ra thiết
> kế), (3) **dẫn xuất** có toán, (4) **neo code** `file·func·line`, (5) **hình ảnh** (ASCII +
> shape/stride/dtype + numeric hand-trace), (6) **số đo THẬT** (chạy trên chính repo này), (7)
> **frontier framing** + cổng phỏng vấn.
>
> **Cách dùng để re-learn.** Đọc phần *Câu hỏi* của mỗi mục → **tự trả lời COLD** (che phần dưới) →
> mở code thật → **đối chiếu cái GAP** (chỉ phần bạn đoán sai mới là bài học). Cuối doc có **checklist
> recall COLD** + bảng số đo. Đây là *derivation lab* có số đo, sinh đôi của roadmap chung
> `roadmap_model/M9_moe_mla_mtp.md`.
>
> **Cảnh báo neo.** Line number **trôi** theo commit. Cite pin theo HEAD ngày 2026-07-14 (`moe.py`,
> `mla.py`, `model.py`, `dsa.py`, `linear_attn.py`). Nếu lệch, `grep` tên hàm — đừng tin số dòng cứng
> (luật repo: verify, don't trust).
>
> **Nguồn số đo.** Mọi số trong doc chạy lại được bằng các `python -c` / snippet ghi trong từng mục;
> đó là "DoD là một profile, không phải test xanh" (FOP-3) áp cho việc học. **Honesty (FOP-4):** MoE đã
> integrate + unit-test *nhưng CHƯA train data thật*; MLA là **toy 1-layer** (chỉ verify identity
> float64, chưa nối KV-cache thật); MTP **chưa build** — mới có *cái seam* (`model.py:459`) + spec F2.
> Số nào chưa đo được đánh dấu **[PREDICTED]**, không bao giờ `[FACT]`.

---

## Bức tranh lớn — ba trục decouple mà dense buộc chặt vào nhau

Série trước (M2 dense decoder: RMSNorm·RoPE·SwiGLU·GQA) cho một model *đúng nhưng dày đặc*: mọi token
đi qua mọi tham số, KV-cache phình theo `n_heads·d_head·T`, mỗi vị trí học đúng một nhãn. Frontier
2026 (DeepSeek-V2/V3, GLM-4.5, Kimi-K2, Qwen3) là bước nhảy: **tách rời ba cặp đại lượng mà dense
architecture cột dính vào nhau.**

```
             DENSE (M2)                         FRONTIER (M9) — ba đòn bẩy độc lập
  ┌───────────────────────────┐      ┌──────────────────────────────────────────────────┐
  │ capacity  ≡  FLOP/token   │  9.1 │ MoE:  N experts (params) ⟂ k active (FLOP)         │  trục PARAMS
  │  (to hơn = chậm hơn)      │ ───► │       FLOP/tok = k·(3·d·d_ff)   params = N·(…)     │
  ├───────────────────────────┤      ├──────────────────────────────────────────────────┤
  │  balance = aux-loss       │  9.2 │ Router: sigmoid affinity + aux-loss-FREE bias      │  trục cân-bằng
  │  (đánh nhau với CE)       │ ───► │        b_i ngoài gradient, ±γ·sign(mean−load)      │  (không hy sinh quality)
  ├───────────────────────────┤      ├──────────────────────────────────────────────────┤
  │ attn quality ≡ KV bytes   │  9.3 │ MLA:  cache 1 latent c_KV (+k_R) ⟂ per-head K,V    │  trục BYTES
  │  (32 head × 128 × 2)      │ ───► │       weight-absorption identity (float64-exact)   │
  ├───────────────────────────┤      ├──────────────────────────────────────────────────┤
  │ signal = 1 nhãn / vị-trí  │  9.4 │ MTP:  dự đoán t+2 từ seam x-trước-lm_head          │  trục SIGNAL
  │  (gradient thưa)          │ ───► │       densify train + draft head speculative FREE  │
  └───────────────────────────┘      └──────────────────────────────────────────────────┘
```

**Throughline — mỗi trục là lời giải cho một áp lực cụ thể:**
- **9.1 MoE** ← áp lực **scale**: muốn nhiều kiến thức (capacity) mà không trả thêm compute/token. Đòn
  bẩy = `N/k` (V3: 256/8 = 32× capacity trên cùng FLOP).
- **9.2 Router** ← áp lực **stability**: routing rời rạc dễ collapse (vài expert nuốt hết, số còn lại
  chết đói); phải balance mà **không** kéo ngược loss dự đoán.
- **9.3 MLA** ← áp lực **memory** (serving): KV-cache — *không phải* weight — chặn batch-size và
  context-length. Đòn bẩy = cache một latent thấp chiều thay 2·H·d_head byte.
- **9.4 MTP** ← áp lực **data**: một hidden state đắt đỏ chỉ mang một nhãn "next token" ⇒ gradient
  thưa. Đòn bẩy = thêm nhãn t+2 (train dày hơn) + tặng kèm drafter học-được (serve nhanh hơn).

Ba trục — **params, bytes, signal** — độc lập; đây đúng spine của DeepSeek-V3. Đọc theo thứ tự
**9.1 (vì sao SPARSE) → 9.2 (router, phần khó nhất) → 9.3 (MLA) → 9.4 (MTP)**; 9.2 tựa lên 9.1,
9.3/9.4 độc lập.

---

## 9.1 · Sparse MoE — decouple *params* khỏi *FLOPs*

**Câu hỏi.** Một dense FFN muốn thêm *capacity* (kiến thức) phải trả thêm *FLOP/token* tuyến tính. Làm
sao thêm capacity mà **giữ nguyên FLOP/token** — và cái giá đánh đổi là gì?

**Sự thật nền tảng.** Trên GPU hiện đại có **hai tài nguyên khác giá**: (a) **HBM để LƯU** tham số —
rẻ (chỉ tốn dung lượng, đọc một lần theo batch), (b) **thời gian để TÍNH** mỗi token — đắt (mỗi token
trả trọn ma trận). Dense FFN buộc hai cái dính nhau: thêm param ⇒ thêm FLOP. MoE **mua cái rẻ (params
trong HBM) để né cái đắt (FLOP/token)** bằng cách để mỗi token chỉ chạm `k` trên `N` chuyên gia.

**Dẫn xuất.** Đặt `d = d_model`, một SwiGLU expert có `d_ff ≈ (8/3)·d` (quy ước M2.5, làm tròn bội 64).
FLOP một dense FFN/token ≈ `3·d·d_ff` (ba matmul w1, w3, w2 — xem M2.5). Thay bằng `N` experts, router
gửi mỗi token qua `k`:

```
FLOP/token  = k · (3·d·d_ff)          ← quyết định bởi k, KHÔNG bởi N
params      = N · (3·d·d_ff)          ← quyết định bởi N, KHÔNG bởi k
capacity / compute  = N / k           ← đòn bẩy sparse  (V3: 256/8 = 32×)
```

Đây là **phương trình decouple**: `N` và `k` là **hai núm độc lập**. Tăng `N` (nhiều kiến thức) không
đụng FLOP; tăng `k` (nhiều compute/token) không đụng số param. Đây là toàn bộ lý do MoE thắng ở scale.

**Vì sao *fine-grained* (chia expert nhỏ)?** Giữ FLOP budget `k·d_ff` cố định nhưng **tăng N, giảm
d_ff** cho router *nhiều tổ hợp expert hơn* để phối: số tổ hợp `C(N,k)` bùng nổ ⇒ biểu diễn mịn hơn
(DeepSeekMoE fine-grained segmentation). Trong repo: `MoEConfig.expert_d_ff` nhỏ hơn mặc định opt-in
vào fine-grained.

**Vì sao *shared expert* (always-on)?** Kiến thức *chung* mọi token đều cần (grammar, cú pháp). Nếu để
routed experts học lại phần chung ⇒ lãng phí capacity. Tách một expert **luôn-bật** gánh phần chung ⇒
routed experts chuyên hoá phần riêng (shared-expert isolation). Trong repo: `n_shared_experts` (default 1).

**Cái giá phải trả** (đánh đổi, không có bữa trưa miễn phí): (a) mỗi token chỉ dùng một mảnh model ⇒
**phải balance** để không expert nào chết đói → Bài 9.2; (b) routing rời rạc `topk` **không khả vi**;
(c) serving phải gather/scatter token theo expert ⇒ **comm** (expert-parallel all-to-all, `roadmap/S6`).

**Neo code** (`src/scratch_llm/moe.py`):
```python
def _default_expert_ffn(d_model: int) -> int:          # :46  quy ước SwiGLU: round (8/3)·d lên bội 64
    raw = int(8 / 3 * d_model)
    return ((raw + 63) // 64) * 64                       # :51

# MoEFeedForward.forward — đường sparse:
xf = x.reshape(-1, d)                                    # :161  flatten (B,S) → bag N token; route per-token
...
for e in range(self.n_routed):                           # :180  chỉ chạy expert e trên token đã chọn nó
    idx = (selected[:, e] > 0).nonzero(as_tuple=True)[0] # :181  (M,) token indices cho expert e
    out = self.routed_experts[e](xf[idx])                # :184  gather M token → 1 SwiGLU
    weighted = gates[idx, e].unsqueeze(-1) * out         # :185  nhân gate
    y = y.index_add(0, idx, weighted)                    # :186  scatter-add trở lại (autograd-safe)
for shared in self.shared_experts:                       # :190  shared expert: chạy MỌI token
    y = y + shared(xf)                                   # :191
return y.reshape(b, s, d), stats                         # :197  trả DELTA thôi (block owns residual)
```
Bất biến gốc: `forward` trả **delta FFN**, không phải `x+delta` — `TransformerBlock.forward` (`model.py:414`,
`x = x + delta`) mới cộng residual. Trả `x+delta` ở đây sẽ **double-add** residual (bug im lặng).

**Hình ảnh — sparse dispatch** (anchor `N=8, k=2, N_tokens=6`):
```
x (B=2,S=3,d)  ──reshape──►  xf (6, d) float32        ← "bag" 6 token, routing quên vị trí (per-token)
                                                          [decode 1-token route Y HỆT full-forward → cache parity]
   router(xf) → logits (6,8) ─sigmoid→ affinity (6,8)  ← mỗi ô = "expert i có hợp token t?"
   topk(affinity+bias, k=2) → mỗi token chọn 2/8 expert

   token t0 ─► e3, e5              expert loop (:180):
   token t1 ─► e0, e3               e0 chạy trên {t1,t4}      ← chỉ M token, KHÔNG cả 6
   token t2 ─► e1, e7               e1 chạy trên {t2}
   token t3 ─► e5, e6               ...
   token t4 ─► e0, e2               shared expert (:190) chạy TRÊN CẢ 6 token (always-on)
   token t5 ─► e1, e4
                                    y = Σ gate·routed(xf[idx])  +  Σ shared(xf)   → reshape (2,3,d)
```
Shape journey: `x (2,3,d) → xf (6,d) → logits (6,8) → affinity (6,8) → topk_idx (6,2) → y (6,d) → (2,3,d)`.
Active FLOP/token = `k/N` của dense-all-experts = `2/8 = 25%` — chạm 8× capacity, trả 25% compute.

**Hand-trace dense-equivalence** (chứng minh không có magic): đặt `N=1, k=1, shared=0`. Router chọn
đúng expert duy nhất; `gate_sel = affinity` (1 giá trị `s`); `gate_norm = s/s = 1` (`moe.py:171`). Nên
`y = 1·SwiGLU(x)` = **đúng một SwiGLU**. Đo: `max|MoE − single expert| = 0.0`. MoE là *strict
generalization* của dense — dense là case `N=1`.

**Số đo THẬT** (`d_model=256`, chạy từ repo root):
```
dense SwiGLU (d_ff=704):          540,672 params
MoE N=8,k=2:   routed params 4,325,376   = 8.0× dense  | active FLOP k/N = 2/8   = 25.0%
MoE N=256,k=8: routed params 138,412,032 = 256.0× dense | active FLOP k/N = 8/256 = 3.1%   (V3-scale)
dense-equivalence N=1,k=1:  max|MoE − single expert| = 0.000e+00   (gate = s/s = 1)
MoE model loss@init = 6.9247  vs  log V = log 1000 = 6.9078   Δ = 0.017   (< band ±0.3)
   aux_loss = 2.07e-4    z_loss = 1.76e-2    (rounding-error cạnh CE ⇒ balancer chính là bias, không phải aux)
```
`loss@init ≈ log V` là bằng chứng **không double-residual**: experts init gần-0 → delta ≈ 0 → head
chưa bị nhiễu → logits ~đều → `−log(1/V) = log V`. Đây là **oracle rẻ nhất** (full circle về M2.6c).

**Frontier / cổng.** DeepSeek-V3 (`modeling_deepseek_v3.py · DeepseekV3MoE`, arXiv:2412.19437):
`n_routed=256, n_shared=1, top_k=8, routed_scaling≈2.5`; GLM-4.5 (2508.06471) + Qwen3 cùng công thức
deep-narrow. **Convergent default** (3 lab độc lập) ⇒ load-bearing, không phải mốt. Ta *giống* spine
(shared + routed + fine-grained qua `expert_d_ff`); *khác*: default `routed_scaling_factor=1.0`
(`moe.py:69`, V3 dùng 2.5) cho loss-at-init sạch, và expert loop của ta là Python O(N) trên CPU
(`:180`) — V3 dùng grouped-GEMM + expert-parallel. **Gate phỏng vấn:** "MoE mua gì bằng gì? Viết
phương trình decouple." Trait = trade-off analysis (params rẻ / FLOP đắt). Bucket = **scarce 2026**
(MoE serving là differentiator).

---

## 9.2 · Router + top-k gating + aux-loss-free bias (phần khó nhất)

**Câu hỏi.** Top-k routing có thể **collapse** (mọi token đổ về vài expert, số còn lại chết đói). Cân
bằng tải bằng cách nào mà **KHÔNG đánh nhau** với loss dự đoán? Vì sao V3 **bỏ** *aux-loss*, chuyển
sang một *bias trên gate* — và vì sao bias phải `γ` nhỏ (1e-3), **sign-only**?

**Sự thật nền tảng.** Balance bằng *aux-loss* (Switch/GShard) là thêm một số hạng `L_aux` vào objective
→ `∇L_aux` là **một lực gradient** kéo router ra khỏi cấu hình mà `∇L_CE` muốn. Hai gradient tranh
chấp ⇒ đánh đổi *quality* lấy *balance*. Ý tưởng V3: cân bằng bằng một **cái núm KHÔNG nằm trên
gradient** — một bias `b_i` mỗi expert mà *trainer chỉnh tay* sau mỗi optimizer step. Vì nó ngoài
graph, `∂L/∂b_i` không tồn tại ⇒ **không tranh chấp** với loss dự đoán.

**Dẫn xuất — bốn quyết định thiết kế, dẫn từng cái:**

**(1) Sigmoid, KHÔNG softmax.** `s_{i,t} = sigmoid(u_t · e_i) ∈ (0,1)`, **độc lập** mỗi expert (mỗi
expert tự chấm "tôi có hợp token này không?"). Softmax (V2) đặt các expert lên một **simplex** `Σ_i
s_i = 1`: nâng score expert này *cơ học* hạ score expert khác. Vậy thì không thể đẩy một expert
lên/xuống mà không nhiễu mọi expert khác ⇒ **bias-trick không chạy** trên softmax. Sigmoid decouple là
**điều kiện tiên quyết** của bias-trick.

**(2) Bias vào SELECTION, KHÔNG vào VALUE.** Đây là nửa tinh tế nhất:
```
sel_scores = affinity + bias      →  topk(sel_scores)   quyết định AI ĐƯỢC VÀO   (route)
gate_sel   = affinity.gather(...) →  raw s, KHÔNG s+b    quyết định ĐÓNG GÓP BAO NHIÊU  (weight)
```
Vì sao tách: balancer cần **ép một expert đói vào hội đồng** mà **KHÔNG thổi phồng ảnh hưởng** của nó.
Nếu bias vào cả value, ép route sẽ *méo cả output* (expert đói bị ép nhận thêm token *và* trọng số bị
cộng thêm bias → sai phối). Tách selection/value cho phép "route ép, weight thật".

**(3) Chuẩn hoá gate.** `g = g'/Σ_selected g'` ⇒ tổng trọng số phối = 1 bất kể router tự tin cỡ nào.
Không cần `ε` (sigmoid > 0 nghiêm ngặt ⇒ mẫu không bao giờ 0) — đây cũng chính là neo dense-equivalence
9.1 (`k=1 ⇒ s/s = 1`).

**(4) Hand-rule update (sign-only).**
```
violation_i = mean(load) − load_i          # >0 ⇒ underloaded (đói) ⇒ nâng bias
b_i += γ · sign(violation_i)               # γ = 1e-3, MỘT nhích cố định ±γ mỗi step
```
**Vì sao sign-only, không magnitude?** Robust với outlier: một expert quá tải cực đoan không làm bias
nhảy vọt; mỗi step một nhích đều ⇒ ổn định. **Vì sao γ nhỏ vẫn đủ mạnh?** Affinity ∈ (0,1) ⇒ khoảng
cách affinity tối đa giữa hai expert < 1. Một **bias-gap > 1** đủ **lật BẤT KỲ** preference (đẩy
`s_j + b_j > s_i + b_i` bất kể `s`). Nên γ nhỏ chỉ cần *nhiều step* để tích luỹ gap > 1, không cần γ
lớn. Ở equilibrium: over/under-count cân nhau → các ±γ triệt tiêu → bias **đứng yên**.

**Hai regularizer phụ (backstop, tí xíu):** *seq-aux* `α·Σ f_i·P_i` (`moe.py:205`, α=1e-4) với `f_i` =
tải cứng (detached count), `P_i` = ưa-thích mềm (khả vi) → chỉ lớn khi expert *vừa* quá tải *vừa* được
ưa; *z-loss* `c_z·mean(logsumexp(logits))²` (`:209`, c_z=1e-3) phạt độ lớn logit raw (ST-MoE style,
*không* trong V3). Cả hai là **rounding-error cạnh CE** (đo aux=2.07e-4, z=1.76e-2 vs CE=6.92) — **bias
mới là balancer chính**, hai cái này chỉ chống lưng.

**Neo code** (`src/scratch_llm/moe.py`):
```python
logits   = self.router(xf)                              # :164  (N, n_routed) pre-sigmoid
affinity = torch.sigmoid(logits)                        # :165  s_{i,t} ∈ (0,1), độc lập  ← quyết định (1)
sel_scores = affinity + self.router.bias                # :168  s + b  ← bias vào SELECTION  ← quyết định (2)
topk_idx   = sel_scores.topk(self.k, dim=-1).indices    # :169  ai được vào
gate_sel   = affinity.gather(-1, topk_idx)              # :170  raw s (KHÔNG s+b)  ← gate VALUE  ← quyết định (2)
gate_norm  = gate_sel / gate_sel.sum(-1, keepdim=True)  # :171  Σ = 1  ← quyết định (3)
...
@torch.no_grad()
def update_bias(self, speed):                            # :130  gọi SAU optimizer.step()
    violation = self.load_count.mean() - self.load_count # :135  >0 ⇒ đói ⇒ nâng
    self.bias += speed * torch.sign(violation)           # :136  ±γ·sign  ← quyết định (4)
    self.load_count.zero_()                              # :137  reset accumulator
```
Trainer-hook: sau `optimizer.step()`, `model.moe_update_biases()` (`model.py:480`) gọi `update_bias`
mỗi MoE layer (`:487`) — một *seam trainer-hook* cùng kiểu với MTP sẽ dùng.

**Hình ảnh — sigmoid vs softmax + bias flow:**
```
SOFTMAX (V2, simplex):              SIGMOID (V3, độc lập):
  s0+s1+s2+s3 = 1                     s0,s1,s2,s3 ∈ (0,1) riêng rẽ
  ↑ nâng s0 ⇒ ↓ s1,s2,s3             ↑ b0 chỉ đẩy s0 vào top-k, KHÔNG đụng s1,s2,s3
  ⇒ bias-trick nhiễu mọi expert      ⇒ bias-trick chạy được

  SELECTION path:  affinity ─┬─(+bias)─► topk ─► "ai vào"     ← bias sống ở ĐÂY
  VALUE     path:  affinity ─┴─(gather)─► norm ─► "bao nhiêu" ← raw s, bias KHÔNG chạm
```
Tensor journey (anchor N_tokens=6, N=4, k=2): `logits (6,4) f32 → affinity (6,4) → sel_scores (6,4) →
topk_idx (6,2) int64 → gate_norm (6,2) → scatter → gates (6,4) dense → counts (4,)`.

**Hand-trace bias-steers-selection-not-value:** đặt mọi affinity = `sigmoid(0) = 0.5` (weight=0),
`bias[0] = 5`. Selection: `sel = 0.5 + [5,0,0,0]` ⇒ expert 0 luôn top-1. Value: `gate = affinity =
0.5` — **không** bị `+5`. Đo: expert 0 selected mọi token = True, gate value = **0.5000** (raw
sigmoid, không phình). Đây đúng cái test `test_bias_steers_selection_not_value` (`tests/test_moe.py:173`).

**Số đo THẬT:**
```
sigmoid row-sums (KHÔNG =1, độc lập):  [2.30, 0.97, 2.10]     ← quyết định (1) hiển thị
softmax row-sums (=1, simplex):        [1.0, 1.0, 1.0]
bias[0]=5 → expert 0 selected mọi token = True; gate VALUE = 0.5000 (raw sigmoid(0), KHÔNG +5)   ← (2)
entropy-at-init (32 expert, real token): H = 3.4581  vs  log32 = 3.4657 ; 0.9·log32 = 3.1192 → H>bar ✓
balancer 40 step (expert0 quá tải): bias = [-4.0, 4.0, 4.0, 4.0]  gap = 8.0  (>1 ⇒ lật mọi sigmoid gap)
```
Caveat (từ teach-back): với **identical token + top-k**, balancer chỉ *xoay* winner (bang-bang),
entropy kẹt ở `log k` — cần token đa dạng (real batch) mới về `log N`. Đo thấy: identical → H=0.693=log2
(collapse), diverse → H=3.458≈log32 (uniform). Balancer cần *diversity* để phát tán.

**Frontier / cổng.** DeepSeek-V3 (`DeepseekV3TopkRouter`): `sigmoid` → `router_logits +
e_score_correction_bias` (bias-vào-selection của ta) → `gather(raw sigmoid)` (gate-value tách của ta) →
`norm_topk_prob` (+1e-20, ta bỏ vì sigmoid>0) → `routed_scaling_factor`. GLM-4.5 y hệt. Ta **khác V3
một điểm lớn**: V3 có *group/node-limited routing* (`n_group`, `topk_group` — chọn score cao mỗi group,
mask để expert nằm ≤ M node, cắt comm expert-parallel); ta bỏ (single device). Bias update ±γ·sign là
train-time hand-rule (không có trong HF *inference* impl). **Gate phỏng vấn:** "Vì sao aux-loss-free
*không* đánh nhau với loss còn aux-loss thì có — và sigmoid liên quan gì?" Trait = first-principles
(dẫn sigmoid → bias-trick). Bucket = **scarce** (MoE training stability là differentiator).

---

## 9.3 · MLA — low-rank KV compression + weight-absorption identity

**Câu hỏi.** KV-cache của MHA phình theo `2·n_heads·d_head` byte/token — nút cổ chai long-context
serving. Làm sao cache **ít hơn ~3.5×** mà attention **KHÔNG đổi một bit chất lượng**? Và vì sao RoPE
**buộc** phải tách ra một nhánh riêng để trick này chạy?

**Sự thật nền tảng.** Lúc *serving*, cache K,V của mọi token đã sinh để khỏi tính lại. Kích thước cache
= `2 · n_kv · d_head · n_layers · bytes/token` — **chính cache này, không phải weight, chặn batch-size
và context-length** (M2.2). GQA đã cắt bằng *chia sẻ* KV qua nhóm head. MLA đi xa hơn: **đừng cache
K,V nữa** — cache **một latent `c_KV` thấp chiều** (`d_latent~512`) dùng chung mọi head, cộng một
**rotary key `k_R` nhỏ** (`d_rope~64`) cũng dùng chung. Cần K,V mỗi head thì *bung ra* bằng
up-projection `W_UK, W_UV`.

**Dẫn xuất — weight-absorption identity.** Content score mỗi head: `q_c · K^C`. Với `K^C = W_UK·c_KV`
(bung latent), kết hợp lại (associativity của phép nhân ma trận):
```
q_c · K^C = q_c · (W_UK c_KV) = (W_UKᵀ q_c) · c_KV          ← gộp W_UK vào query
```
Vế phải attend **thẳng `c_KV`** trong không gian `d_latent` — **không bao giờ dựng `K^C` mỗi head**.
Đây là *weight-absorption*: gấp `W_UK` vào query (`q_abs`), gấp `W_UV` vào output path (`Σ a_j V_j =
W_UV(Σ a_j c_KV_j)`). Decode chỉ đọc `c_KV` (+ `k_R`) từ cache → tiết kiệm bytes.

**Vì sao RoPE PHẢI decoupled?** RoPE là phép quay *phụ thuộc vị trí* `R(pos)` (M2.3). Trên content-K:
`K^C = R(pos)·W_UK·c_KV`. Muốn absorb thì cần `q · (R(pos)·W_UK·c_KV) = (W_UKᵀ R(pos)ᵀ q)·c_KV` — nhưng
`R(pos)` **phụ thuộc `pos`**, không phải hằng ⇒ không thể gấp vào `W_UK` *tĩnh* một lần (mỗi token một
`R(pos)` khác). Nên DeepSeek **tách**: content-key (không RoPE, gấp được) khỏi rotary-key `k_R` (có
RoPE, cache riêng, dùng chung mọi head). Score = **hai nhánh cộng**:
```
score = q_c·K^C  +  q_R·k_R            (content, gấp-được)  +  (rotary, cache riêng)
```
`KV bytes/token = (d_latent + d_rope)·2` thay `2·n_heads·d_head·2`. **Identity float64-exact là *lý do
được phép* cache latent thay K,V với ZERO quality change** — nếu không exact thì đây chỉ là xấp xỉ
lossy, không phải kiến trúc.

**Neo code** (`src/scratch_llm/mla.py`):
```python
def kv_bytes_per_token(self, dtype_bytes=2):            # :42  (d_latent + d_rope) values/token
    return (self.d_latent + self.d_rope) * dtype_bytes  # :44

# _project (:76): những gì được CACHE
c_kv   = self.down_kv(h)                                 # :84  (B,S,dc) — LATENT được cache
k_rope = self.k_r(h)....transpose(1, 2)                  # :85  (B,1,S,dr) — rotary key SHARED, được cache

# forward_absorbed (:111) — đường tiết kiệm (decode)
wk    = self.up_k.weight.view(H, dh, dc)                 # :118  (H, dh, dc)
q_abs = torch.einsum("bhsd,hdc->bhsc", q_c, wk)          # :119  gấp W_UK vào query  ← (W_UKᵀ q_c)
scores = matmul(q_abs, c_kv...) + matmul(q_rope, k_rope) # :121-124  attend c_kv trực tiếp + rotary
latent_out = torch.matmul(attn, c_kv.unsqueeze(1))       # :128  (B,H,S,dc) — trung bình latent
wv    = self.up_v.weight.view(H, dh, dc)                 # :130
out   = torch.einsum("bhsc,hdc->bhsd", latent_out, wv)   # :131  gấp W_UV ở output
```
`forward_naive` (`:92`, ORACLE) bung `k_c = up_k(c_kv)` (`:97`), `v = up_v(c_kv)` (`:98`) rồi attention
chuẩn — dùng để *kiểm định* identity. HF impl chỉ có đường naive; ta có SẴN cả hai để verify.

**Hình ảnh — cache MHA vs GQA vs MLA:**
```
per token, per layer, byte:
MHA-32h : [K: 32×128] [V: 32×128] × 2B  = 16384 B      ── phình theo n_heads
GQA-8   : [K:  8×128] [V:  8×128] × 2B  =  4096 B      ── chia sẻ 4 query/1 kv
MLA     : [c_KV: 512] [k_R: 64]   × 2B  =  1152 B      ── một latent + rotary key
          └──────────┬──────────┘
             KHÔNG per-head; bung ra bằng W_UK/W_UV LÚC TÍNH (store small, compute big)

DECODE (absorbed):  đọc c_KV (512) + k_R (64) từ cache
   q_c (B,H,S,dh) ──einsum W_UK──► q_abs (B,H,S,dc) ──attend──► c_KV (B,S,dc)
                                                       │
   q_R (B,H,S,dr) ──────────────────────────────► · k_R (B,1,S,dr)  (nhánh rotary, cộng)
                                                       ▼ softmax
   latent_out (B,H,S,dc) ──einsum W_UV──► out (B,H,S,dh) ──o_proj──► (B,S,d_model)
```
Tensor journey (anchor `H=8, dh=128, dc=512, dr=64, S=7`): `h (2,7,512) f64 → c_kv (2,7,512) [CACHED]
→ q_c (2,8,7,128) → q_abs (2,8,7,512) → scores (2,8,7,7) → latent_out (2,8,7,512) → out (2,8,7,128) →
(2,7,512)`. Điểm mấu chốt: `q_abs` sống ở `dc=512` (latent), attend thẳng `c_kv` — **không có tensor
`k_c (2,8,7,128)` per-head** nào được vật chất hoá ở đường absorbed.

**Hand-trace identity:** naive dựng `k_c, v` mỗi head rồi attend; absorbed gấp `W_UK/W_UV` và attend
latent. Hai đường **cùng một hàm toán** (associativity), khác nhau chỉ ở *thứ tự phép nhân*. Đo
float64: `|naive − absorbed| = 3.997e-15` = machine-epsilon ⇒ **identity, không phải xấp xỉ**.

**Số đo THẬT** (config R1-like `d_latent=512, d_rope=64, d_head=128`):
```
MLA cache/token   = 1152 B   = (512+64)·2
GQA-8 cache/token = 4096 B   = 2·8·128·2       → MLA = 3.56× nhỏ hơn GQA-8
MHA-32h cache/tok = 16384 B  = 2·32·128·2      → MLA = 7.0% của MHA-32h
|naive − absorbed| (float64) = 3.997e-15       → weight-absorption identity, float64-exact  [FACT]
```
(Ledger `bench/RESULTS.md:449` cite thêm "MLA 1152 B = 1.8% của MHA 65536 B" — 65536 là config
full-model 128-head khác; internal-consistent, chỉ khác baseline head-count.)

**Honest (FOP-4):** `mla.py` hiện là **toy 1-layer float64**, CHƯA nối `ModelConfig`/`TransformerBlock`/
KV-cache thật. **F5** = wire `attn='gqa'|'mla'` + train iso-param MLA-vs-GQA-8 ở 0.2–0.5B.
**[PREDICTED]** (F2/F5 spec): MLA trong **+0.02 val loss** của GQA-8 iso-param; decode uplift **≥1.2×
ở 16k ctx**; **KILL** nếu gap > 0.05 nats.

**Frontier / cổng.** DeepSeek-V2 (`DeepseekV2Attention`, arXiv:2405.04434 — origin MLA): `kv_a_proj_with_mqa`
chiếu `h → (kv_lora_rank + qk_rope_head_dim)` = latent + rotary-key gộp một Linear, `kv_a_layernorm`
norm latent, `kv_b_proj` bung `W_UK/W_UV`. HF impl *reconstruct* K,V mỗi forward (đường naive của ta) —
**không** làm weight-absorption inline (đó là tối ưu decode-time engine, không phải modeling). V3 kế
thừa MLA y nguyên. **Attention-axis extensions (cùng trục bytes/long-ctx, đã build mechanism):**
- **DSA** (DeepSeek-V3.2, `dsa.py`, F8.1): sparse attention O(L·k) trên top-k key mỗi query, chọn bằng
  *lightning indexer* train bằng KL(dense ‖ indexer). Đo (config test `d=128, d_index=32, n_index_heads=4`,
  `dsa_attention_flops`): `k≥L → |dense−sparse| = 0.0` (identity ở limit); FLOP@L=4096,k=512 = **1.27×**
  end-to-end (kể cả indexer O(L²)) = **4.27×** attention-only. Chồng lên MLA cho long-ctx.
- **Gated DeltaNet** (Qwen3-Next/Kimi-Linear, `linear_attn.py`, F10.1): linear-attention hybrid, state
  hằng-size thay KV growing. Đo (config test toy `d_head=16, n_heads=2, d_v=32`, `linear_attn_state_bytes`):
  state = **2 KB** (const in T) vs GQA-8 KV @ T=4096 = 2 MB → **1024× nhỏ hơn** (khớp `bench/RESULTS.md:677`
  + `test_linear_attn.py`; ở model thật `d_head=128 / 16k-ctx` tỉ lệ này lên ~4096×);
  `chunkwise == recurrent == float64 ref` (|Δ| = 4.16e-17). 3 trên 4 layer thay attention bằng recurrence.

**Gate phỏng vấn:** "Vì sao MLA cache 1 latent thay 2·H·d_head, và tại sao RoPE *bắt buộc* decouple để
weight-absorption chạy?" Trait = roofline-first / predict-the-number (KV bytes). Bucket = **scarce
2026** (long-context serving economics = top differentiator).

---

## 9.4 · MTP — tín hiệu train dày HƠN + draft head speculative miễn phí

**Câu hỏi.** LM chuẩn học **1 nhãn/vị-trí** (next token). Nếu bắt model dự đoán *thêm* token t+2,
gradient dày hơn — có cải thiện pretrain không, và vì sao **cùng module đó** *tái sử dụng* làm draft
head speculative? Vì sao V3 chọn *sequential* (D=1, nối tiếp) chứ không *parallel* (Gloeckle)?

**Sự thật nền tảng.** Mỗi vị trí `t`, LM thường có đúng **một** tín hiệu train: "token t+1 là gì?".
Đó là gradient *thưa* — một nhãn cho một hidden state đắt đỏ (đã đi qua N block). Nhưng hidden state
đó chứa đủ thông tin để đoán xa hơn t+1; ta đang *bỏ phí* tín hiệu. MTP: bắt cùng hidden state đó dự
đoán **thêm** token t+2 (và có thể t+3…) qua một module phụ.

**Dẫn xuất.** Điểm móc là **hidden state `x` ngay TRƯỚC `lm_head`** (`model.py:459`, `x = final_norm(x)`)
— biểu diễn đã "chín", chia sẻ được. Module MTP sequential (D=1, V3):
```
x_t (final hidden, :459) ─┐
                          ├─ eh_proj: Linear(2d→d) [ RMSNorm(x_t) ‖ RMSNorm(emb(tok_{t+1})) ]
emb(token_{t+1})         ─┘        → h' → 1 TransformerBlock → lm_head(shared) → dự đoán token_{t+2}

aux_loss = λ · CE(logits_MTP, token_{t+2})          (λ: 0.3 → 0.1 theo lịch)
```
**Hai lợi ích từ MỘT móc:**
1. **Densify signal** (train tốt hơn): mỗi vị trí giờ mang *nhiều* nhãn (t+1 từ head chính, t+2 từ MTP)
   → gradient dày hơn → model học "nhìn xa" → pretrain tốt hơn.
2. **Draft head miễn phí** (serve nhanh hơn): cái module dự-đoán-t+2 đó, lúc serve, chính là một
   *drafter* đoán token kế cho speculative decode. Nó **học được** nên đánh bại n-gram (prompt-lookup)
   trên văn xuôi mở.

**Vì sao sequential (V3) thắng parallel (Gloeckle)?** Sequential = MTP head *nhận vào* embedding của
token t+1 (nối tiếp thứ tự) ⇒ mô hình hoá phân phối **có điều kiện** `p(t+2 | t+1, context)` — đúng
nhân quả, khớp cách decode chạy ⇒ **draft chất lượng cao** (acceptance ~85–90%). Gloeckle (parallel,
2404.19737) đặt K head *độc lập* cùng đọc `x_t` dự đoán t+1..t+K song song — rẻ hơn nhưng head t+2
**không thấy** token t+1 ⇒ phân phối kém khớp lúc serve ⇒ acceptance thấp. V3 chọn sequential vì draft
chất lượng đáng giá hơn compute train.

**Vì sao lossless?** MTP chỉ *đề xuất* draft; target-model **xác minh** greedy → output **token-exact
bất kể draft đúng/sai** (giống truncate-rollback ở `roadmap/S1`). Speculative decode không đổi phân
phối, chỉ đổi *tốc độ*.

**Neo code** — cái seam ĐÃ tồn tại (`src/scratch_llm/model.py · TransformerLM.forward`):
```python
x = self.token_emb(token_ids)                            # :453  embed
for layer_idx, block in enumerate(self.blocks):          # :455  N blocks
    x, stats = block(x, positions, cache, layer_idx)     # :456
x = self.final_norm(x)                                    # :459  ← ĐIỂM MÓC MTP (x đã chín)
...
logits = self.lm_head(x)                                  # :462  head chính: dự đoán t+1
```
MTP module (**F2, CHƯA build**) sẽ *chèn giữa `:459` và `:462`*: đọc `x` + `token_emb(shifted)`, chạy
`eh_proj` + 1 block + lm_head-shared, phát aux-CE trên token +2. Drafter serve: `MTPDrafter` implement
protocol `Drafter` sẵn có (`serving/speculative.py`) — thay n-gram bằng head học-được. `model.py:480
moe_update_biases` (Bài 9.2) là ví dụ *seam trainer-hook* cùng kiểu.

**Hình ảnh — densify + draft từ một seam:**
```
TRAIN:                                          SERVE (speculative):
  ...N blocks... → x = final_norm(x) ─┬─► lm_head → t+1  (head chính)   drafter đoán t+2,t+3...
                                      │                                  ▼
                                      └─► MTP block → t+2  (aux, λ·CE)   target verify greedy
        gradient: hai nhãn/vị-trí (t+1 + t+2)  ← DÀY hơn   → nhận nếu khớp, rollback nếu sai (LOSSLESS)
```
Tensor journey (đo thật): seam `x = final_norm(x)` shape `(2, 16, 128)` dtype `float32` → feed
`lm_head.weight (1000, 128)` → `logits (2, 16, 1000)`. MTP module sẽ nhận `x (B,S,d_model)=(2,16,128)`
+ `emb(token_{t+1}) (2,16,128)`, ghép → `eh_proj (2d→d)` → 1 block → shared `lm_head` → `(2,16,1000)`
dự đoán token t+2.

**Số đo THẬT** (seam location, đo bằng forward-hook trên `final_norm`):
```
seam x = final_norm(x):  shape (2, 16, 128)  dtype float32   → feeds lm_head (1000, 128)
lm_head → logits (2, 16, 1000)   ✓  MTP chèn đúng giữa :459 và :462
```
**[PREDICTED]** (F2 spec, MTP chưa build): MTP module cho **≥1.5× tokens/target-forward** ở K=2 trên
open-text (n-gram ≈1.0× ở đó, lossless greedy token-exact); 2nd-token acceptance hướng vùng V3 85–90%.
**KILL** nếu MTP aux (λ=0.3→0.1) làm val loss xấu > 0.01 nats. Honest: **chưa có module** — mới có
*cái seam* + spec DoD.

**Frontier / cổng.** V3 (2412.19437) = MTP sequential D=1; GLM-4.5 (2508.06471) cũng ship MTP →
**convergent default** across labs độc lập (mạnh nhất trong FRONTIER §3). Lưu ý: HF `modeling_deepseek_
v3.py` **KHÔNG chứa** MTP module (grep rỗng) — MTP head bị lược khỏi impl *inference* (train-time +
optional draft), nên frontier-ref ở đây là *paper* (V3 §MTP; Gloeckle 2404.19737), không phải code
vendored. MTP head cũng là input cho perf-front (decode-kernel DELTA tối ưu, FRONTIER §9). **Gate phỏng
vấn:** "Vì sao MTP *đồng thời* cải thiện pretrain VÀ cho draft head miễn phí — và vì sao V3 sequential
thắng Gloeckle parallel?" Trait = spec-with-falsifiers (predict acceptance + KILL). Bucket = **scarce**
(speculative decode + train efficiency).

---

## Bảng số đo THẬT (chạy lại được — DoD-là-profile)

| Concept | Đo | Kết quả | Ý nghĩa |
|---|---|---|---|
| 9.1 | MoE params N=8,k=2 vs dense | 4,325,376 = **8.0×** dense · FLOP k/N = **25%** | decouple params↔FLOP |
| 9.1 | MoE params N=256,k=8 | 138,412,032 = **256×** dense · FLOP k/N = **3.1%** | V3-scale sparse |
| 9.1 | dense-equivalence N=1,k=1 | max\|MoE − expert\| = **0.0** (gate s/s=1) | MoE ⊃ dense |
| 9.1 | MoE loss@init | **6.9247 ≈ log1000 = 6.9078** (Δ0.017) | không double-residual (oracle rẻ nhất) |
| 9.2 | sigmoid vs softmax row-sum | sigmoid [2.30,0.97,2.10] ≠ 1 · softmax [1,1,1] | sigmoid độc lập ⇒ bias-trick chạy |
| 9.2 | bias steers selection | selected=True, gate value = **0.5000** (raw s) | bias→route, KHÔNG→weight |
| 9.2 | entropy-at-init (32 exp) | **H = 3.4581 ≈ log32 = 3.4657** (>0.9·log32) | không collapse |
| 9.2 | balancer 40 step | bias [-4,4,4,4], **gap = 8.0 > 1** | lật bất kỳ sigmoid affinity ∈(0,1) |
| 9.3 | KV bytes MLA/GQA/MHA | **1152 / 4096 / 16384 B** → MLA 3.56× < GQA-8 | trục bytes |
| 9.3 | weight-absorption identity | \|naive − absorbed\| = **3.997e-15** (float64) | cache latent = zero quality loss |
| 9.4 | MTP seam shape | x = final_norm → **(2,16,128) f32** → lm_head | densify + draft từ 1 móc |
| bonus | DSA topk == dense @k≥L | \|dense − sparse\| = **0.0** · FLOP **1.27×** e2e (4.27× attn-only) @L=4096,k=512 | sparse identity ở limit |
| bonus | GDN linear-attn state | **2 KB** const vs GQA-8 @T=4096 = 2 MB → **1024×** | hybrid memory-wall |

---

## Checklist recall COLD (che phần trên, tự trả lời — đây là cách re-learn)

1. **9.1** Viết **phương trình decouple** (FLOP/token vs params). Vì sao "params rẻ, FLOP đắt" là tiền
   đề khiến MoE thắng? Tăng k 8→16 (giữ N=256): FLOP đổi? capacity đổi? vì sao V3 KHÔNG làm?
2. **9.1** Vì sao có **shared expert** *và* **fine-grained** expert cùng lúc? Chúng giải áp lực gì?
3. **9.1** `N=1, k=1`: MoE bằng gì? (dẫn `gate = s/s = 1`) Vì sao `loss@init ≈ log V`?
4. **9.2** Vì sao **sigmoid** (không softmax) là *điều kiện tiên quyết* của bias-trick?
5. **9.2** Bias vào **selection** nhưng gate-value dùng **raw s** — vì sao phải tách? Điều gì hỏng nếu
   bias vào cả value?
6. **9.2** `update_bias`: dẫn `b_i += γ·sign(mean−load)`. Vì sao **sign-only**? Vì sao **γ nhỏ** vẫn
   đủ lật mọi preference? (gap > 1 vs affinity ∈ (0,1))
7. **9.2** Vì sao aux-loss-free *không* đánh nhau với CE còn aux-loss thì có?
8. **9.3** Dựng lại identity `q·(W c) = (Wᵀq)·c`. Vì sao RoPE trên content-K **phá** weight-absorption?
   (viết `R(pos)·W` không nhân-tử-hoá được)
9. **9.3** KV bytes MLA vs GQA-8 vs MHA (config R1)? Vì sao identity phải **float64-exact** mới cho
   phép cache latent?
10. **9.4** Hai lợi ích của MTP từ **một** móc `x`-trước-`lm_head`? Vì sao lossless **không** phụ thuộc
    draft đúng/sai? Vì sao V3 **sequential** thắng Gloeckle **parallel**?

> Trả lời COLD được cả 10 = **M9 thật sự OWNED** (interview-grade). Vấp câu nào → mở đúng mục đó, hoặc
> blank-slate hàm tương ứng (`raise NotImplementedError` → test đỏ → tự dẫn → xanh) để re-own bằng tay
> (`test_moe.py`, `test_mla.py` có răng: dense-equivalence, bias-steers, identity float64).

---

*Cross-ref: `roadmap_model/M9_moe_mla_mtp.md` (reference chung, số dòng pin `4ad0ac5`) ·
`derivations/M2_transformer_forward.md` (dense decoder, tiền đề) · `PROGRESS.md` (ledger 89 Bài) ·
`bench/RESULTS.md` §A1 R4.5 (MLA) + §Frontier ablations · `FRONTIER_2026_ABLATIONS.md` (F2 MTP · F5 MLA
wire · F6 MoE ablation · F8 DSA · F10 linear-hybrid). Trạng thái thật: MoE integrated+unit-test (chưa
train data thật), MLA toy 1-layer (F5 wire chờ), MTP chỉ có seam (F2 chờ), DSA/GDN mechanism-only.
Neo MEASURED mạnh nhất: MoE decouple + loss@init · router entropy→log N + balancer-overcomes · MLA 1152 B
+ identity float64. Phần loss/train là **[PREDICTED]** tới khi F5/F6/F2 chạy trên base thật. Série kế:
M10 close-the-loop (F1 Muon + F7 GRPO "aha") — khép loop nanochat → model biết nói.*
