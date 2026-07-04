# SÉRIE 9 — FRONTIER ARCHITECTURE: MoE · MLA · MTP, derived (the DeepSeek/GLM spine)

> **Số dòng pin theo commit `4ad0ac5`.** Đây là *learning-derivation roadmap* cho phần MODEL — sinh
> đôi của perf-roadmap `docs/learning/roadmap/` (phần serving+kernel). Khác perf-roadmap ở hai điểm
> **nặng ký**: mỗi Bài (1) **DẪN XUẤT** component từ số 0 — bài toán, toán, *vì sao thiết kế này chứ
> không phải cái khác* — và (2) **đối chiếu FRONTIER cụ thể**: đọc impl vendored
> (`.venv/.../transformers/models/{deepseek_v2,deepseek_v3,deepseek_v32,glm4_moe}/modeling_*.py`) +
> trích paper, nói rõ ta *giống/khác* chỗ nào và vì sao.
> **Ngôn ngữ:** tiếng Việt; thuật ngữ (MoE, MLA, MTP, aux-loss-free, sigmoid gate, weight-absorption,
> Newton-Schulz…) giữ tiếng Anh. Toán plain-text, không LaTeX.
> **Honesty (FOP-4):** MoE đã tích hợp + unit-test nhưng **CHƯA train trên data thật**; MLA là **toy
> 1-layer** (chỉ kiểm định identity float64, chưa nối vào KV-cache); MTP **chưa build** — mới có *cái
> seam* (`model.py:938`) + spec F2. Mỗi Neo được gắn nhãn: MEASURED test-invariant · MEASURED shape ·
> hoặc **PREDICTION** (không bao giờ [FACT] cho số chưa đo).

**Vì sao série này.** Série trước (dense decoder: RMSNorm·RoPE·SwiGLU·MHA) cho ta một model *đúng
nhưng dày đặc* — mọi token đi qua mọi tham số, KV-cache phình theo n_heads·d_head. Série 9 là bước
nhảy kiến trúc frontier 2026: **tách rời hai đại lượng mà dense buộc chặt vào nhau.** MoE tách
*số tham số* khỏi *FLOP/token* (nhiều capacity, ít compute). MLA tách *chất lượng attention* khỏi
*byte KV/token* (cache một latent thấp chiều thay vì K,V mỗi head). MTP tách *tín hiệu train* khỏi
*một-token-một-nhãn* (dự đoán nhiều token → gradient dày hơn) và tặng kèm một draft head speculative
miễn phí. Ba trục — params, bytes, signal — là ba đòn bẩy độc lập; đây đúng spine của DeepSeek-V2/V3,
GLM-4.5/5, Kimi-K2. Đọc theo thứ tự **9.1 (vì sao SPARSE) → 9.2 (router + balancing, phần khó nhất)
→ 9.3 (MLA, trục bytes) → 9.4 (MTP, trục signal)** — 9.2 tựa lên 9.1, 9.3/9.4 độc lập nhau.

Cross-link perf-roadmap: cơ chế *weight-absorption* ở góc serving đã có ở `roadmap/S2` (MLA decode) —
ở đây ta lo phần **dẫn xuất identity** + kế hoạch F5 wire-thật; primitives phân tán (TP/PP/EP-MoE) ở
`roadmap/S6` — ở đây chỉ lo *vì sao MoE cần EP* chứ không lặp comm.

---

## Bài 9.1 — Vì sao SPARSE: decouple params khỏi FLOPs (`src/scratch_llm/moe.py` · `MoEFeedForward.forward` :159 · `_default_expert_ffn` :46)
> **Câu hỏi first-principles:** một dense FFN muốn thêm *capacity* (kiến thức) phải trả thêm *FLOP/token*
> tuyến tính. Làm sao thêm capacity mà GIỮ NGUYÊN FLOP/token — và cái giá đánh đổi là gì?
> **Neo (invariant — MEASURED):** dense-equivalence — với `n_routed=1, k=1` thì MoE *bằng đúng* một
> SwiGLU (`gate_norm = s/s = 1`, `test_dense_equivalence_single_expert`); và loss-at-init của MoE ≈
> log(vocab): đo `ce = 3.63` vs `log 32 = 3.47`, Δ0.167 trong band ±0.3 (L1_moe_WALKTHROUGH §4a). FLOP
> hoạt-động/token = k/n_routed: ta top-2/4 = 50%, V3 top-8/256 ≈ 3%.

**1. Feynman — bài toán bằng lời.** Dense FFN = *một chuyên gia bách khoa* mà mọi câu hỏi đều phải hỏi
trọn: muốn nó biết nhiều hơn, phải làm nó to hơn, và mỗi token trả giá bằng toàn bộ ma trận w1/w2/w3.
MoE = *một hội đồng chuyên gia nhỏ* + một *lễ tân (router)*: mỗi token chỉ được gửi tới k chuyên gia
router chọn. Tổng *kho tham số* = N chuyên gia (capacity lớn), nhưng *chi phí mỗi token* = chỉ k
chuyên gia (compute nhỏ). Đánh đổi cốt lõi: **params rẻ (chỉ tốn HBM để lưu), FLOP đắt (tốn thời gian
mỗi token)** — MoE mua cái rẻ để né cái đắt. Cái giá: (a) mỗi token chỉ dùng một mảnh model → phải
*balance* để đừng có chuyên gia nào chết đói (Bài 9.2), (b) routing rời rạc (top-k) không khả vi, (c)
serving phải gather/scatter token theo expert (comm — EP, `roadmap/S6`).

**2. Dẫn xuất từ đầu (derive).** Đặt d = d_model, một SwiGLU expert có d_ff ≈ (8/3)·d (quy ước, làm
tròn bội 64 — `_default_expert_ffn` :46-51). FLOP dense FFN/token ≈ 3·d·d_ff (ba matmul w1,w3,w2).
Bây giờ thay bằng N experts, mỗi token đi qua k → FLOP/token ≈ k·(3·d·d_ff), *độc lập với N*. Còn số
tham số ≈ N·(3·d·d_ff) — *tăng tuyến tính theo N*. Đây là **phương trình decouple**:

```
FLOP/token  = k · (3·d·d_ff)          ← quyết định bởi k, KHÔNG bởi N
params      = N · (3·d·d_ff)          ← quyết định bởi N, KHÔNG bởi k
capacity/compute ratio = N/k          ← đòn bẩy sparse (V3: 256/8 = 32×)
```

Vì sao *fine-grained* (chia expert nhỏ, `expert_d_ff` < mặc định, `MoEConfig` :64) lại tốt: với cùng
FLOP budget k·d_ff cố định, tăng N và giảm d_ff cho router *nhiều tổ hợp expert hơn* để phối — số tổ
hợp C(N,k) bùng nổ, biểu diễn mịn hơn (DeepSeekMoE §fine-grained segmentation). Vì sao có **shared
expert** (always-on, `n_shared_experts` :63): kiến thức *chung* mọi token đều cần (grammar, cú pháp)
nếu để routed experts học lại thì lãng phí capacity — tách một expert luôn-bật gánh phần chung, routed
experts chuyên hoá phần riêng (shared-expert isolation). Loss-at-init ≈ log V là bằng chứng *không có
double-residual*: experts init gần-0 → delta ≈ 0 → head chưa bị nhiễu (Neo).

**3. Trace code.** `MoEFeedForward.forward` (:159): `xf = x.reshape(-1, d)` (:161) — *flatten (B,S)
thành bag N token*; routing per-token, position-independent (đây là bất biến gốc: một token route y hệt
dù ở full-forward hay decode 1-token → decode/cache parity). Đường compute sparse: expert loop (:180)
chỉ chạy expert e trên các token đã chọn nó (`xf[idx]`, :184), rồi `index_add` (:186) gộp lại có trọng
số gate. Shared experts (:190-191) chạy TRÊN MỌI token, cộng thẳng. `return y.reshape(b,s,d)` (:197)
trả **delta FFN thôi** — `TransformerBlock.forward` (`model.py:889-893`) làm `x = x + delta`; trả
`x+delta` ở đây sẽ double-add residual. Test ghim: `test_dense_equivalence_single_expert` (Neo).

**4. Cổng teach-back.** (a) Viết phương trình decouple và giải thích vì sao "params rẻ, FLOP đắt" là
tiền đề khiến MoE thắng — chạm được chữ *HBM-để-lưu* vs *thời-gian-mỗi-token*. (b) *Sửa-và-đoán:* nếu
tăng k từ 8→16 (giữ N=256), FLOP/token đổi thế nào, capacity đổi thế nào, và vì sao V3 KHÔNG làm vậy?
(Gợi: FLOP ×2, capacity y nguyên → mất tỷ số sparse; k lớn tiến về dense.)

**5. Frontier.** DeepSeek-V3 (`modeling_deepseek_v3.py` · `DeepseekV3MoE` :194, 2412.19437) ship
`n_routed=256, n_shared=1, top_k=8, moe_intermediate_size` nhỏ (fine-grained) + `routed_scaling≈2.5`;
GLM-4.5 (`glm4_moe` :375, 2508.06471) cùng công thức deep-narrow. Ta *giống* spine (shared + routed
+ fine-grained qua `expert_d_ff`) nhưng *khác*: default `routed_scaling_factor=1.0` (:69) cho
loss-at-init sạch (V3 dùng 2.5), và expert loop của ta là Python O(N) trên CPU (:180) — V3/GLM dùng
grouped-GEMM + expert-parallel. **Câu interview:** "MoE mua gì bằng gì? Vì sao fine-grained + shared
expert lại đồng thời xuất hiện ở V3, GLM và Qwen3?" (convergent default → load-bearing, không phải mốt).

---

## Bài 9.2 — Router + top-k gating + balancing: DERIVE aux-loss vs aux-loss-free bias (γ=1e-3) (`moe.py` · `Router` :108 · `update_bias` :129 · `MoEFeedForward.forward` :164-197 · `_stats` :199)
> **Câu hỏi first-principles:** top-k routing khả dĩ collapse (mọi token đổ về vài expert, số còn lại
> chết đói). Cân bằng tải bằng cách nào mà KHÔNG đánh nhau với loss dự đoán? Vì sao V3 bỏ *aux-loss*
> chuyển sang một *bias trên gate* — và vì sao bias phải γ nhỏ (1e-3), sign-only?
> **Neo (invariant — MEASURED):** router entropy giữ **> 0.9·log(N_routed)** suốt train; ở toy 4-expert
> đo H → 1.386 = log 4 (uniform) sau ~5 step balancing, `test_balancer_overcomes_any_routing_preference`
> chứng minh bias-gap > 1 lật được BẤT KỲ preference (L1_moe_WALKTHROUGH §5). Collapse ⇒ balancer hỏng
> (kill: >2 debug-day ⇒ dense).

**1. Feynman — bài toán bằng lời.** Ba việc mỗi token: **route** (chọn expert), **mix** (phối output
gate-weighted sum=1), **balance** (đừng để lễ tân thiên vị vài chuyên gia). Việc khó là *balance*. Cách
cũ (Switch/GShard): thêm một *aux-loss* phạt khi tải lệch — nhưng loss đó là một *lực gradient* kéo
ngược lại loss dự đoán, đánh đổi chất lượng lấy cân bằng. Ý tưởng DeepSeek-V3: cân bằng bằng một **cái
núm KHÔNG nằm trên gradient** — một bias b_i mỗi expert mà *trainer chỉnh tay* sau mỗi optimizer step
(expert quá tải → hạ bias, đói → nâng bias). Vì nó ngoài graph, nó *không tranh chấp* với loss. Đánh
đổi: mất tính "học được" của balancing (giờ là hand-rule ±γ), đổi lấy *không hy sinh quality*.

**2. Dẫn xuất từ đầu (derive).** Bốn quyết định thiết kế, dẫn từng cái:

- **Sigmoid, không softmax (`affinity` :165).** `s_{i,t} = sigmoid(u_t·e_i)` — mỗi expert tự chấm
  "tôi có hợp token này không?" ∈ (0,1), *độc lập*. Softmax (V2) đặt các expert lên một simplex (Σ=1):
  nâng score expert này *cơ học* hạ score expert khác → không thể đẩy một expert lên/xuống mà không
  nhiễu mọi expert khác. Sigmoid decouple → *là điều kiện cho bias trick hoạt động*.
- **Bias vào SELECTION, không vào VALUE (`sel_scores` :168 vs `gate_sel` :170).** Đây là nửa tinh tế:
  `topk(s + b)` quyết định *AI được vào* (route); `gate = s` (raw affinity, gather từ `affinity` :170,
  KHÔNG từ s+b) quyết định *đóng góp bao nhiêu* (weight). Vì sao tách: balancer cần *ép một expert vào
  hội đồng* mà KHÔNG thổi phồng ảnh hưởng của nó — nếu bias vào cả value, ép route sẽ méo cả output.
  `test_bias_steers_selection_not_value` ghim đúng chỗ này.
- **Chuẩn hoá gate (`gate_norm` :171):** `g = g'/Σ_selected g'` → tổng trọng số phối = 1 bất kể router
  tự tin cỡ nào. Không có ε (sigmoid > 0 nghiêm ngặt ⇒ mẫu không bao giờ 0); đây cũng là neo
  dense-equivalence Bài 9.1 (k=1 ⇒ s/s=1).
- **Hand-rule update (`update_bias` :129-137):** `violation = mean(load) − load_i`; `b_i += γ·sign(
  violation)` (:136). Overloaded (load > mean) → violation < 0 → bias xuống; underloaded → lên.
  **sign-only, không magnitude** → mỗi bước một nhích cố định ±γ, robust với outlier. γ=1e-3 (:68) nhỏ
  để không dao động; vì affinity ∈ (0,1), một bias-gap > 1 đủ lật *bất kỳ* preference (Neo), nên γ nhỏ
  vẫn đủ mạnh, chỉ cần nhiều step. Ở equilibrium, over/under-count cân nhau → các ±γ triệt tiêu → bias
  *đứng yên* (L1 §5: step 5 trở đi bias khựng, load = uniform).

Hai regularizer phụ (backstop, tí xíu): **seq-aux** `α·Σ f_i·P_i` (:202-205, α=1e-4) — f_i = tải cứng
(detached count), P_i = ưa-thích mềm (khả vi) → chỉ lớn khi expert *vừa* quá tải *vừa* được ưa; đẩy
router nhẹ ra khỏi expert đó. **z-loss** `c_z·mean(logsumexp(logits))²` (:209, c_z=1e-3) — phạt độ lớn
logit raw, giữ số ổn định (repo add-on kiểu ST-MoE, *không* trong V3). Cả hai là rounding-error cạnh CE
(L1 §4a: aux 1.1e-4, z 3.9e-3 vs ce 3.63) — **bias mới là balancer chính, hai cái này chỉ chống lưng.**

**3. Trace code.** `forward` (:164): `logits = router(xf)` (:164, pre-sigmoid) → `affinity` :165 →
`sel_scores = affinity + router.bias` :168 → `topk_idx` :169 (sparsity ra đời) → `gate_sel` :170 (raw
affinity, gather) → `gate_norm` :171 (Σ=1) → scatter thành dense `gates`/`selected` (:174-175) →
`counts = selected.sum(0)` :176. Loop expert (:180-186), scale (:187), shared (:190-191). Nếu training:
`router.load_count += counts` (:196) tích luỹ giữa các lần update. Sau `optimizer.step()`,
`model.moe_update_biases()` (`model.py:959`) gọi `Router.update_bias` (:129) mỗi MoE layer. `_stats`
(:199) tính aux/z/load_fraction/entropy (:217). Test: entropy > 0.9·logN + balancer-overcomes (Neo).

**4. Cổng teach-back.** (a) Nêu ĐỦ lý do vì sao sigmoid (không softmax) là *điều kiện tiên quyết* của
bias-trick, rồi vì sao bias phải vào selection nhưng gate-value phải dùng raw s. (b) *Sửa-và-đoán:* với
**top-1** routing + token gần giống nhau, load là one-hot mỗi bước — bias sẽ làm gì (spread được
không?), entropy ra sao? (Gợi: bang-bang, bias chỉ *xoay* winner e0→e1→e2, entropy kẹt 0; cần k≥2 hoặc
token đa dạng — L1 §5 caveat.)

**5. Frontier.** DeepSeek-V3 (`modeling_deepseek_v3.py` · `DeepseekV3TopkRouter` :139): `sigmoid` :215
→ `router_logits_for_choice = router_logits + e_score_correction_bias` :216 (đúng bias-vào-selection của
ta) → `topk_weights = router_logits.gather(...)` :232 (raw sigmoid = gate value, đúng tách của ta) →
`norm_topk_prob` :233 (+1e-20, ta bỏ vì sigmoid>0) → `routed_scaling_factor` :236. GLM-4.5 y hệt
(`glm4_moe` `Glm4MoeTopkRouter` :287, `e_score_correction_bias` :299, sigmoid :389). **Ta KHÁC V3 một
điểm lớn:** V3 có *group/node-limited routing* (`n_group`, `topk_group` :208-231 — chọn 2 score cao mỗi
group, mask group để giới hạn expert nằm rải ≤ M node, cắt comm expert-parallel). Ta bỏ (single device,
`moe.py` docstring "Out of scope"). Bias update (±γ·sign) *không* có trong HF inference impl (nó là
train-time hand-rule) — ta implement ở `update_bias` :136, đối chiếu V3 §Aux-loss-free. **Câu
interview:** "Vì sao aux-loss-free *không đánh nhau* với loss dự đoán còn aux-loss thì có — và sigmoid
liên quan gì?" (bias ngoài gradient + độc lập nhờ sigmoid).

---

## Bài 9.3 — MLA: low-rank KV compression + weight-absorption identity (`src/scratch_llm/mla.py` · `MultiHeadLatentAttention` :54 · `forward_naive` :92 · `forward_absorbed` :111 · `MLAConfig.kv_bytes_per_token` :42)
> **Câu hỏi first-principles:** KV-cache của MHA phình theo 2·n_heads·d_head byte/token — nút cổ chai
> long-context serving. Làm sao cache *ít hơn ~3.5×* mà attention KHÔNG đổi một bit chất lượng? Vì sao
> RoPE buộc phải *tách* ra một nhánh riêng để trick này chạy?
> **Neo (KV shape — MEASURED · identity — MEASURED):** MLA cache (d_latent+d_rope) = **1152 B/token**
> (R1-like config) vs GQA-8 **4096 B** = **3.56× nhỏ hơn** (vs MHA 65536 B = 1.8%), RESULTS.md:449.
> Path weight-absorbed **numerically identical** (float64) với naive-reconstruct — `tests/test_mla.py`
> 5/5. Honest: **toy 1-layer, chưa nối KV-cache/ModelConfig** — F5 mới wire thật.

**1. Feynman — bài toán bằng lời.** MHA cache K và V *mỗi head* mỗi token — với 32 head × 128 dim × 2
(K+V) = 8192 giá trị/token/layer. GQA đã cắt bằng cách *chia sẻ* KV qua nhóm head (n_kv_heads < n_heads).
MLA đi xa hơn: đừng cache K,V nữa — cache **một latent c_KV thấp chiều** (d_latent ~512) *dùng chung
cho mọi head*, cộng một **rotary key k_R nhỏ** (d_rope ~64) cũng dùng chung. Khi cần K,V mỗi head thì
*bung ra* bằng up-projection W_UK, W_UV. Phép màu: nhờ một *đồng nhất thức đại số* (weight-absorption),
decode KHÔNG cần bung — attend thẳng trong không gian latent. Đánh đổi: thêm hai matmul down/up-proj
lúc train (compute), đổi lấy KV-cache bé tí lúc serve (bytes) — long-context economics rơi thẳng ra từ
kiến trúc: một model 671B có cache/token *nhỏ hơn* một model 70B (ADR-0012).

**2. Dẫn xuất từ đầu (derive).** Content score mỗi head: q_c · K^C. Với K^C = W_UK·c_KV (bung latent),
viết ra:

```
q_c · K^C = q_c · (W_UK c_KV) = (W_UKᵀ q_c) · c_KV          ← kết hợp lại: gộp W_UK vào query
```

Vế phải attend **thẳng c_KV** trong không gian d_latent — không bao giờ dựng K^C mỗi head. Đây là
*weight-absorption*: gấp W_UK vào query (`q_abs`), gấp W_UV vào output path (`Σ a_j V_j = W_UV(Σ a_j
c_KV_j)`). Decode chỉ đọc c_KV (+ k_R) từ cache → tiết kiệm bytes. **Vì sao RoPE phải tách rời:** một
phép xoay RoPE *phụ thuộc vị trí* trên K không thể gấp vào W_UK *tĩnh* — R(pos)·W_UK·c_KV không nhân
tử hoá thành (const)·c_KV được. Nên DeepSeek *tách* content-key (không RoPE, gấp được) khỏi
rotary-key k_R (có RoPE, cache riêng, dùng chung mọi head). Score = q_c·K^C + q_R·k_R (hai nhánh cộng).
KV bytes/token = (d_latent + d_rope)·2 (`kv_bytes_per_token` :42-44) thay vì 2·n_heads·d_head·2 → chính
là 1152 vs 4096/65536 (Neo). Identity float64-exact là *lý do được phép* cache latent thay K,V với
*zero quality change* — nếu không exact thì đây chỉ là một xấp xỉ lossy, không phải kiến trúc.

**3. Trace code.** `_project` (:76-87): `q_c` (:81, content query), `q_rope` (:82-83, RoPE-rotated),
`c_kv = down_kv(h)` (:84, **latent được cache**), `k_rope = k_r(h)` (:85-86, shared rotary key được
cache). `forward_naive` (:92, ORACLE): bung `k_c = up_k(c_kv)` :97, `v = up_v(c_kv)` :98, score hai
nhánh (:101-104), causal, softmax, `out = attn·v` :108. `forward_absorbed` (:111, đường tiết kiệm):
`wk = up_k.weight.view(H,dh,dc)` :118 → `q_abs = einsum("bhsd,hdc->bhsc", q_c, wk)` :119 (gấp W_UK vào
query) → attend `c_kv` trực tiếp (:122) → `latent_out = attn·c_kv` :128 → `wv = up_v.weight` :130 →
`out = einsum("bhsc,hdc->bhsd", latent_out, wv)` :131 (gấp W_UV ở output). Test ghim: naive == absorbed
float64 (Neo). Cross-link cơ chế decode/absorption ở perf-roadmap **`roadmap/S2` Bài 2.8** (weight-
absorption ở góc serving/KV) — ở đây ta lo phần *dẫn xuất identity*.

**Kế hoạch F5 (wire-thật, PREDICTION).** `mla.py` hiện là toy 1-layer float64. F5 = nối `attn='gqa'|
'mla'` vào `ModelConfig`/`TransformerBlock` + KV-cache thật, train iso-param MLA-vs-GQA-8 ở 0.2–0.5B.
**PREDICTION (F2/F5 spec §3):** MLA trong **+0.02 val loss** của GQA-8 iso-param; KV/token ~3.5× nhỏ
(đo 1152 vs 4096 B — phần shape này ĐÃ đo); decode uplift **≥1.2× ở 16k ctx**. **KILL** nếu gap >0.05
nats. (Nhãn: PREDICTION, chưa train.)

**4. Cổng teach-back.** (a) Dựng lại identity `q·(W c) = (Wᵀq)·c` và giải thích *chính xác* vì sao RoPE
trên content-K sẽ phá weight-absorption (viết ra R(pos)·W không nhân tử hoá). (b) *Sửa-và-đoán:* nếu
gộp cả rotary-key vào latent (bỏ nhánh d_rope riêng, cho k_R đi qua W_UK) — identity còn giữ không, và
KV bytes/token đổi ra sao? (Gợi: mất tính gấp-được → decode buộc reconstruct → mất luôn cái tiết kiệm).

**5. Frontier.** DeepSeek-V2 (`modeling_deepseek_v2.py` · `DeepseekV2Attention` :287, 2405.04434 — origin
của MLA): `kv_a_proj_with_mqa` :317 chiếu h → (kv_lora_rank + qk_rope_head_dim) = *latent + rotary-key
gộp một Linear*, `kv_a_layernorm` :322 norm latent, `kv_b_proj` :323 bung ra n_heads·(nope+v_head_dim)
= up-proj W_UK/W_UV. Chú ý: HF impl *reconstruct* K,V mỗi forward (đường naive của ta, :92) — **không**
làm weight-absorption inline (đó là tối ưu decode-time engine, không phải modeling). Ta *giống* cấu trúc
down/up + decoupled-RoPE, *khác*: ta có SẴN cả hai path (naive + absorbed) để *kiểm định identity* trực
tiếp — HF chỉ có naive. V3 (2412.19437) kế thừa MLA y nguyên; V3.2-DSA (`deepseek_v32` · `DeepseekV32
Indexer` :166, 2512.02556) chồng thêm sparse-attention lên MLA cho long-ctx (F8 stretch). **Câu
interview:** "Vì sao MLA cache 1 latent thay 2·H·d_head, và tại sao RoPE *bắt buộc* phải decouple để
weight-absorption chạy?"

---

## Bài 9.4 — MTP: multi-token prediction — tín hiệu train dày HƠN + draft head speculative miễn phí (`src/scratch_llm/model.py` · `TransformerLM.forward` :919-944 · seam `x`-trước-`lm_head` :938)
> **Câu hỏi first-principles:** LM chuẩn học 1 nhãn/vị-trí (next token). Nếu bắt model dự đoán *thêm*
> token +2, gradient dày hơn — có cải thiện pretrain không, và vì sao cùng module đó *tái sử dụng* làm
> draft head speculative? Vì sao V3 chọn *sequential* (D=1, nối tiếp) chứ không *parallel* (Gloeckle,
> nhiều head độc lập)?
> **Neo (PREDICTION — chưa build):** F2 spec — MTP module cho **≥1.5× tokens/target-forward** ở K=2 trên
> open-text (n-gram ≈1.0× ở đó), greedy token-exact (lossless); 2nd-token acceptance hướng về vùng V3
> 85–90%. **KILL** nếu MTP aux (λ=0.3→0.1) làm val loss xấu >0.01 nats. Honest: **chưa có module** — mới
> có *cái seam* (`model.py:938`, x-trước-lm_head) + spec DoD; đây là PREDICTION, không phải số đo.

**1. Feynman — bài toán bằng lời.** Train LM thường: mỗi vị trí t, một tín hiệu — "token t+1 là gì?".
Đó là gradient *thưa*: một nhãn cho một hidden state đắt đỏ. MTP: bắt cùng hidden state đó dự đoán
*thêm* token t+2 (và có thể t+3…) qua một module phụ. Hai lợi ích trong một: (1) **densify** — mỗi vị
trí giờ mang nhiều nhãn → gradient dày hơn, model học "nhìn xa" → pretrain tốt hơn. (2) **draft head
miễn phí** — cái module dự-đoán-t+2 đó, lúc serve, chính là một *drafter* đoán token kế để speculative
decode xác minh; nó *học được* nên đánh bại n-gram (prompt-lookup) trên văn xuôi mở. Đánh đổi: thêm một
block + aux-loss (λ) lúc train; nếu λ quá nặng, nó kéo lệch loss chính (KILL >0.01 nats).

**2. Dẫn xuất từ đầu (derive).** Điểm móc là **hidden state x ngay TRƯỚC lm_head** (`model.py:938`,
`x = self.final_norm(x)`): đây là biểu diễn đã chín, chia sẻ được. Module MTP (D=1, sequential — V3):

```
x_t (final hidden, :938)  ─┐
                           ├─ eh_proj: Linear(2d→d) [RMSNorm(x_t) ‖ RMSNorm(emb(tok_{t+1}))]
emb(token_{t+1})          ─┘        → h'  → 1 TransformerBlock  → lm_head(shared)  → dự đoán token_{t+2}
aux_loss = λ · CE(logits_MTP, token_{t+2})     (λ: 0.3 → 0.1)
```

Sequential = MTP head *nhận vào* dự đoán/embedding của token t+1 (nối tiếp theo thứ tự), nên nó mô hình
hoá phân phối *có điều kiện* p(t+2 | t+1, context) — đúng nhân quả, khớp cách decode chạy. Gloeckle
(parallel, 2404.19737) đặt K head *độc lập* cùng đọc x_t dự đoán t+1,…,t+K song song — rẻ hơn nhưng mỗi
head không thấy token trước nó, phân phối kém khớp lúc serve. V3 chọn sequential vì **draft chất lượng
cao hơn** (acceptance cao) đáng giá hơn compute train. Chia sẻ `token_emb`+`lm_head` (DoD F2) giữ tham
số thêm tối thiểu. Lossless speculative: MTP chỉ *đề xuất* draft, target-model *xác minh* greedy —
output token-exact bất kể draft đúng/sai (giống `roadmap/S1` Bài 1.2 truncate-rollback).

**3. Trace code.** Cái seam ĐÃ tồn tại: `TransformerLM.forward` (:919): embed (:932) → N blocks
(:934-935) → `x = self.final_norm(x)` (**:938 — điểm móc MTP**) → `logits = self.lm_head(x)` (:941).
MTP module (F2, chưa build) sẽ *chèn giữa :938 và :941*: đọc `x` + `token_emb(shifted)`, chạy eh_proj
+ 1 block + lm_head-shared, phát aux-CE trên token +2. Drafter serve: `MTPDrafter` implement protocol
`Drafter` sẵn có (`serving/speculative.py:59`) — thay n-gram bằng head học-được. `model.py:959`
`moe_update_biases` (Bài 9.2) là ví dụ *seam trainer-hook* cùng kiểu (hàm gọi sau step). Honest: đây là
đường *sẽ đi*, code chưa có (Neo PREDICTION).

**4. Cổng teach-back.** (a) Giải thích hai lợi ích của MTP (densify signal + free drafter) từ *một* móc
x-trước-lm_head, và vì sao lossless không phụ thuộc draft đúng hay sai. (b) *Sửa-và-đoán:* nếu chọn
parallel (Gloeckle) thay sequential — acceptance-rate lúc serve đổi hướng nào và vì sao (head t+2 không
thấy t+1)? Nếu λ = 1.0 thay 0.3 → dự đoán gì về val loss chính (KILL nào bắn)?

**5. Frontier.** V3 (2412.19437) là MTP sequential D=1 (spine F2); GLM-4.5 (2508.06471) cũng ship MTP →
**convergent default** across labs độc lập (mạnh nhất trong FRONTIER §3). Lưu ý: HF `modeling_deepseek_
v3.py` **KHÔNG chứa MTP module** (grep rỗng) — MTP head bị lược khỏi impl *inference* (nó là train-time +
optional draft), nên frontier-ref ở đây là *paper* (V3 §MTP; Gloeckle 2404.19737) chứ không phải code
vendored. Đây cũng là input cho perf-front: MTP head trở thành decode-kernel mà DELTA tối ưu (FRONTIER
§9 synergy) — nhưng giờ trên model *thật*, không phải toy. **Câu interview:** "Vì sao MTP *đồng thời*
cải thiện pretrain VÀ cho draft head miễn phí — và vì sao V3 sequential thắng Gloeckle parallel?"
(FRONTIER F2 interview item, `FRONTIER_2026_ABLATIONS.md:125`).

---

*Đóng série.* Ba trục decouple: **params↔FLOP** (MoE, 9.1 — capacity/compute = N/k, balance bằng bias
ngoài-gradient 9.2) · **quality↔bytes** (MLA, 9.3 — cache latent 1152 B, weight-absorption identity
float64-exact) · **signal↔one-label** (MTP, 9.4 — densify train + draft head, seam :938). Trạng thái
thật: MoE integrated+unit-test (chưa train data thật, F6 ablation chờ), MLA toy (F5 wire-thật chờ), MTP
chỉ có seam (F2 chờ). Neo mạnh nhất đang *đo được*: router entropy → log N + balancer-overcomes-any
(MEASURED), MLA 1152 vs 4096 B + identity float64 (MEASURED); phần loss/train là **PREDICTION** cho tới
khi F5/F6/F2 chạy trên base thật (`bench/RESULTS.md` §Frontier ablations). Série kế: F1 Muon
(`roadmap` perf-front) + F7 GRPO "aha" (post-training), khép loop nanochat → model biết nói.
