# SÉRIE 2 — THE TRANSFORMER FORWARD PASS, derived (kiến trúc model, dẫn xuất từ đầu)

> **Số dòng pin theo commit `4ad0ac5` (HEAD).** Đây là ROADMAP mastery — bản đồ dẫn-xuất + trace để
> dẫn đường cho teach-back sâu về sau, KHÔNG phải bài giảng đầy đủ mỗi mục. Mỗi Bài ~một màn hình, tự
> chứa, DERIVE component từ blank (bài toán → toán → vì sao design NÀY) rồi trace xuống `file·hàm·dòng`.
> Đây là bản SINH ĐÔI của perf-roadmap (`docs/learning/roadmap/`, phủ serving+kernel): série này phủ
> chính MODEL — forward pass của một decoder LM. Số đo: test-invariant trong `tests/test_model.py` +
> spec `docs/FRONTIER_2026_ABLATIONS.md`. Frontier ref VENDORED, đã đọc:
> `.venv/.../transformers/models/{qwen3,nanochat,deepseek_v3}/modeling_*.py`.

**Vì sao série này.** Série 1 (perf) hỏi *decode một token TỐN gì* (memory-bound, AI~1). Série này lùi
một bước: *token đó được TÍNH thế nào* — forward pass của decoder. Toàn bộ model là một hàm thuần
`token_ids (B,S) → logits (B,S,V)`, gấp từ 6 primitive: nhúng token thành vector (Embedding), trộn thông
tin GIỮA các vị trí (attention + RoPE), trộn thông tin TRONG một vị trí (SwiGLU FFN), giữ activation
không nổ (RMSNorm), rồi chiếu ra vocab (lm_head). Mọi thứ khác — GQA, QK-norm, tie/untie — là các *đòn
bẩy* trên khung đó. Ta đi theo thứ tự first-principles: attention (2.1) → nhân bản thành nhiều head +
chia sẻ KV (2.2) → dạy nó biết vị trí (RoPE, 2.3) → ổn định activation (RMSNorm + pre-norm, 2.4) → phần
tính-trong-vị-trí (SwiGLU, 2.5) → hai đòn bẩy ổn-định/init cuối (QK-norm + tied/untied, 2.6). Neo xuyên
suốt: **causal no-leak** (không rò tương lai) và **loss-at-init ≈ log V** (softmax đều lúc khởi tạo) — hai
oracle rẻ nhất trong cả stack, mọi Bài phải giữ được chúng.

Thứ tự đọc: **2.1 (retrieval khả-vi) → 2.2 (multi-head + GQA) → 2.3 (RoPE) → 2.4 (RMSNorm + pre-norm) →
2.5 (SwiGLU) → 2.6 (QK-norm + tied/untied).** Mỗi Bài tựa lên Bài trước — đừng nhảy cóc.

---

## Bài 2.1 — Attention = truy hồi key-value khả-vi (`model.py` · `scaled_dot_product_attention` :134 · `softmax` :127)
> **Câu hỏi first-principles:** vì sao "trộn thông tin giữa các token" lại có dạng `softmax(QKᵀ/√d)·V`, và
> hai chi tiết `1/√d` + causal mask từ đâu ra (derive, đừng thuộc lòng)?
> **Neo (invariant):** `test_causal_attention_does_not_leak_future` (`tests/test_model.py:57`) — sửa mọi
> token SAU vị trí `cut`, logits tại `0..cut` phải **bit-identical** (`assert_close`, :73). + `test_sdpa_
> causal_mask_zeros_future` (:96): vị trí 0 chỉ attend chính nó ⇒ output = v[0] chính xác.

**1. Feynman — bài toán bằng lời.** Mỗi token cần "hỏi" các token trước: *token nào liên quan đến tôi?*
Đây là một **soft dictionary lookup**. Mỗi token phát ra ba vector: một **query** (tôi đang tìm gì), một
**key** (tôi chào hàng gì), một **value** (nếu bạn chọn tôi, tôi trao gì). Độ khớp query-key = tích vô
hướng `q·k`; softmax biến điểm khớp thành trọng số tổng bằng 1; output = trung bình có trọng số của các
value. Trade-off cốt lõi: một hash-table cứng (chọn ĐÚNG một key) không khả-vi — không backprop được;
softmax là *phiên bản mềm* của argmax, khả-vi hoàn toàn, nên gradient chảy qua "ai attend ai".

**2. Dẫn xuất từ đầu (derive).** Cho `q,k ∈ R^d` chuẩn hóa iid ~N(0,1). Điểm khớp `s = q·k = Σ qᵢkᵢ` là
tổng d biến độc lập mean 0 var 1 ⇒ `Var(s) = d`, độ lệch chuẩn `√d`. Nếu để nguyên, `s` lớn ~√d; đẩy vào
softmax thì với d lớn softmax bão hòa (một logit áp đảo, gradient ≈ 0 ở mọi chỗ khác — "softmax saturation").
Chia cho `√d` đưa `Var(s/√d) = 1` bất kể d ⇒ softmax giữ được entropy hợp lý lúc init, gradient khỏe. Đó
là gốc `1/√d_k` (`:139–140`), KHÔNG phải hằng số tùy tiện. **Causal mask:** LM tự-hồi-quy dự đoán token t+1
từ `≤t`; token tại vị trí i được attend key j **iff j ≤ i**. Cài bằng cách cộng `−∞` vào score của j>i
TRƯỚC softmax (`masked_fill(~mask, -inf)`, :142) ⇒ `exp(−∞)=0` ⇒ trọng số 0, không rò tương lai. **fp32
softmax:** `exp` của số lớn tràn bf16; `softmax` (:127) trừ max trước (`x - x.amax`, :129 — bất biến toán,
chống overflow) và ép fp32 (`.float()`, :143) rồi mới cast lại.

**3. Trace code.** `scaled_dot_product_attention(q,k,v,mask)` (:134): `scores = q @ kᵀ / √d_k` (:140) → nếu
có mask, `masked_fill(~mask, -inf)` (:142, mask bool True=attend) → `softmax(scores.float(), -1)` (:143) →
`attn @ v` (:144). `softmax` (:127) = trừ-max-rồi-exp-rồi-chia. Shapes: q,k,v là `(B,H,S,d)` (head là batch
dim — xem 2.2), scores `(B,H,S,S)`, out `(B,H,S,d)`. Test pin: `:96` cho mask cứng tril, `:57` cho no-leak
end-to-end qua cả model.

**4. Cổng teach-back.** (a) Derive lại vì sao chia `√d` chứ không phải `d` hay `√(2d)` — chạm được chữ
"Var(q·k)=d". (b) *Sửa-và-đoán:* nếu BỎ `√d_k` (chia 1), ở d_k=64 thì softmax lúc init trông thế nào, và
loss-at-init (Bài 2.6) lệch khỏi log V về hướng nào? (Gợi: logits nhọn hơn → không còn đều → nhưng lm_head
init nhỏ nên attention output vẫn nhỏ; hiệu ứng chính là gradient attention yếu, hội tụ chậm.)

**5. Frontier.** Mọi impl 2026 giữ nguyên công thức này. `modeling_nanochat.py:180` softmax fp32 rồi cast
(`softmax(..., dtype=torch.float32).to(query.dtype)`) — Y HỆT ta. `modeling_deepseek_v3.py:301` (hàm
`eager_attention_forward`, `scaling` là arg) — DeepSeek MLA đổi `scaling` thành `1/√d` có nhân `mscale`
(YaRN long-context, :359 `yarn_get_mscale`); ta dùng `1/√d` thuần vì không scale context. Câu interview:
"Vì sao attention score chia √d, và điều gì hỏng nếu bỏ?" — softmax saturation + gradient chết.

---

## Bài 2.2 — Multi-head + GQA/MQA: vì sao chia sẻ KV head (`model.py` · `MultiHeadSelfAttention` :753 · `ModelConfig.kv_heads` :63)
> **Câu hỏi first-principles:** một attention là đủ về mặt biểu diễn — vì sao chẻ thành H head song song,
> và vì sao 2026 lại CHIA SẺ key/value giữa các query head (GQA), đánh đổi cái gì lấy cái gì?
> **Neo (invariant / shapes):** `test_full_mha_when_kv_equals_heads` (`:106`) — `n_kv==n_heads` ⇒ đường MHA
> thuần, shape `(1,8,V)`. Shapes qua block: `x (B,S,d) → q (B,H,S,dh), k/v (B,H_kv,S,dh) → repeat_interleave
> KV lên H → out (B,H,S,dh) → reshape (B,S,H·dh) → o_proj → (B,S,d)`.

**1. Feynman — bài toán bằng lời.** Một head = một "kiểu quan hệ" (ví dụ: head này theo dõi subject-verb,
head kia theo dõi dấu ngoặc). Multi-head = chạy H phép attention SONG SONG trên các subspace `d/H` chiều
khác nhau rồi ghép lại — mô hình học nhiều loại quan hệ cùng lúc mà không tăng chi phí (tổng chiều vẫn d).
GQA (grouped-query attention) hỏi tiếp: KV-cache lúc decode là hàng đầu tốn bộ nhớ (Série 1 perf: KV chiếm
60–85% wall-clock ở context dài). Nếu 4 query head DÙNG CHUNG một KV head thì cache nhỏ đi `n_heads/n_kv`
lần. Trade-off: bớt "độ đa dạng" của key/value (một KV phải phục vụ cả nhóm query) đổi lấy cache nhỏ +
bandwidth decode thấp. MQA (n_kv=1) là cực đoan; GQA là điểm giữa.

**2. Dẫn xuất từ đầu (derive).** Head là *batch dimension*: thay vì d chiều một head, ta reshape thành
`(H, d/H)` rồi coi H như trục batch cho SDPA — không toán mới, chỉ bố cục. Chi phí KV-cache = `2 · n_layers
· n_kv · head_dim · ctx · B · dtype_bytes` (2 = K và V). Ở MHA thuần `n_kv=H`; GQA đặt `n_kv < H` với ràng
buộc `H % n_kv == 0` (mỗi KV head phục vụ đúng `H/n_kv` query head, chia đều — `__post_init__` :55 ép điều
này). Để attention chạy, KV phải được "phát" lên đủ H head: `repeat_interleave(K, H/n_kv)` (`:837`) — LẶP
chứ không tính lại, nên cache vẫn lưu bản nhỏ `n_kv`, chỉ nở ra lúc attend. Đây đúng là ADR-0002: *build
full MHA, expose GQA as flag*; MHA là ca đặc biệt `n_kv==n_heads`.

**3. Trace code.** `MultiHeadSelfAttention.__init__` (:756): `q_proj: d→H·dh`, nhưng `k_proj/v_proj: d→
n_kv·dh` (:762–764) — K/V hẹp hơn Q. `forward` (:772): project → `.view(b,s,n_heads,dh).transpose(1,2)`
đưa head thành batch dim (:781–783). Nếu `n_kv != n_heads`: `repeats = n_heads//n_kv; k = k.repeat_
interleave(repeats, dim=1)` (:835–838) NGAY TRƯỚC SDPA. Sau attention: `out.transpose(1,2).reshape(b,s,
H·dh)` gộp head lại → `o_proj` về d (:853–854). KV-cache (Série 1) LƯU bản `n_kv` để tận dụng GQA; repeat
xảy ra ở attention-time (docstring `KVCache` :206). **Cross-link perf:** vì sao GQA thắng lúc decode →
`roadmap/S1` (KV bytes là bức tường; GQA/MLA/paged đều giảm byte đọc).

**4. Cổng teach-back.** (a) Giải thích vì sao GQA giảm KV-cache đúng `n_heads/n_kv` lần mà KHÔNG giảm
FLOP attention (gợi: repeat_interleave nở lại đủ H trước SDPA). (b) *Sửa-và-đoán:* đặt `n_kv=1` (MQA) trên
model H=8 — cache nhỏ đi mấy lần, và chất lượng có xu hướng gì so với GQA-2 ở cùng param (gợi: một KV cho
cả 8 query = bottleneck biểu diễn; GQA-2..8 là sweet spot thực nghiệm)?

**5. Frontier.** `modeling_qwen3.py:237–243`: `q_proj` dùng `num_attention_heads`, `k/v_proj` dùng
`num_key_value_heads` — GQA y hệt ta, và `bias=config.attention_bias` mặc định False (Qwen3 *bỏ* QKV-bias
"for stable training", 2505.09388) — ta cũng không bias (`Linear` :87). GQA là default rộng 2026 (Llama4/
Qwen3/Gemma3/gpt-oss); MLA (DeepSeek/Kimi) là đường KV-compression khác — `FRONTIER_PRACTICE_2026` §MLA
(dòng 191) nói rõ "MLA = DeepSeek/Kimi bet; GQA = broad default". F5 trong ablation spec sẽ wire MLA thật
(`FRONTIER_2026_ABLATIONS.md` §3 F5: KV/token ~3.5× nhỏ hơn — PREDICTION). Câu interview: "GQA giải quyết
bài toán gì, và vì sao không phải mọi lab dùng MLA?"

---

## Bài 2.3 — RoPE: nhúng vị trí từ yêu cầu tương-đối (`model.py` · `RotaryPositionalEmbedding` :147)
> **Câu hỏi first-principles:** attention thuần là *hoán vị bất biến* (không biết thứ tự) — làm sao dạy nó
> vị trí sao cho score chỉ phụ thuộc **khoảng cách tương đối** i−j, không phụ thuộc vị trí tuyệt đối?
> **Neo (invariant):** `test_rope_score_depends_only_on_offset` (`:78`) — `score(5,0) ≈ score(20,15)` (cùng
> offset 5 ở vị trí tuyệt đối khác nhau, sai lệch <1e-4, :91); `score(5,0) ≠ score(6,0)` (offset khác →
> khác, guard chống RoPE no-op). Chính bất biến này là gốc lossless của KV-cache/speculative (Série 1).

**1. Feynman — bài toán bằng lời.** SDPA chỉ thấy `q·k` — nếu ta xáo trộn token, tập score y nguyên: model
"mù thứ tự". Cách cũ (absolute positional embedding cộng vào input) dạy được vị trí nhưng model phải TỰ suy
ra khoảng cách. RoPE làm thẳng: **XOAY** vector q,k một góc tỉ lệ với vị trí, sao cho tích vô hướng của hai
vector đã xoay chỉ phụ thuộc HIỆU góc = hiệu vị trí. Analogy: hai kim đồng hồ; góc lệch giữa chúng chỉ phụ
thuộc chênh lệch thời gian, không phụ thuộc giờ tuyệt đối. Trade-off: RoPE không thêm tham số (thuần hình
học), tổng quát hóa độ dài tốt hơn absolute, và — mấu chốt cho serving — vì xoay theo vị trí TUYỆT ĐỐI của
mỗi token, một token trong KV-cache đã xoay đúng dù chỉ đưa 1 token mới vào.

**2. Dẫn xuất từ đầu (derive).** Muốn `⟨R(i)q, R(j)k⟩` chỉ phụ thuộc `i−j`. Trong 2D, ma trận xoay `R(θ)`
thỏa `R(a)ᵀR(b) = R(b−a)` ⇒ `⟨R(θᵢ)q, R(θⱼ)k⟩ = qᵀR(θⱼ−θᵢ)k`, chỉ phụ thuộc `θⱼ−θᵢ`. Đặt `θ_pos = pos·ωₖ`
tuyến tính theo pos ⇒ hiệu góc = `(j−i)·ωₖ` ∝ khoảng cách. Chẻ head_dim thành d/2 CẶP tọa độ, cặp k dùng
tần số `ωₖ = θ^(−2k/d)` (`inv_freq`, :160) — dải tần từ nhanh (cặp đầu, bắt vị trí gần) tới chậm (cặp cuối,
bắt vị trí xa), giống positional encoding sin/cos gốc nhưng NHÂN (xoay) thay vì CỘNG. Mỗi cặp `(x_2k,
x_2k+1)` xoay: `rot_even = x_even·cos − x_odd·sin`, `rot_odd = x_even·sin + x_odd·cos` (:180–181) — đúng công
thức xoay 2D. `cos/sin` precompute cho mọi vị trí × mọi tần số, buffer non-persistent (:163–164, tính lại
lúc load, không checkpoint).

**3. Trace code.** `__init__` (:158): `inv_freq = 1/θ^(arange(0,dh,2)/dh)` (:160) → `freqs = outer(positions,
inv_freq)` (:162, `(max_len, dh/2)`) → buffer `cos, sin`. `forward(x, positions)` (:166): slice `cos[positions]`
theo vị trí THỰC (:169) — hỗ trợ `positions` 1D (shared, :174–177) HOẶC 2D `(B,seq)` per-row (ragged batched
decode: mỗi slot token ở vị trí tuyệt đối RIÊNG, :171–173) → tách even/odd (:178–179) → xoay (:180–181) →
`stack + flatten` xen kẽ lại (:182). Gọi tại `MultiHeadSelfAttention.forward` :791–792 — xoay Q,K SAU QK-norm,
TRƯỚC SDPA. Vị trí đến từ `TransformerLM.forward` :928/931 (`cache.lengths` per-row khi decode ragged, else
`arange(start, start+s)`). Test `:78` pin bất biến tương-đối.

**4. Cổng teach-back.** (a) Derive lại vì sao `R(a)ᵀR(b)=R(b−a)` cho ra tính chỉ-phụ-thuộc-offset (chạm
công thức xoay 2D). (b) *Sửa-và-đoán:* nếu RoPE dùng vị trí *tương đối trong khúc* thay vì tuyệt đối, thì
chunked-prefill (Série 1 Bài 1.6) còn bit-identical với prefill một-phát không? Ghi/attend gì lệch? (Gợi:
hỏng — token ở khúc 2 sẽ xoay như thể ở đầu chuỗi.)

**5. Frontier.** `modeling_nanochat.py:125 apply_rotary_pos_emb`, `:370 NanoChatRotaryEmbedding` — cùng RoPE.
`modeling_deepseek_v3.py:68–72` dùng `ROPE_INIT_FUNCTIONS` + `attention_scaling` (:117–118 nhân cos/sin) cho
YaRN long-context; và MLA phải TÁCH một phần "decoupled RoPE" vì không xoay được latent không-vị-trí
(`FRONTIER_PRACTICE_2026:192`). Long-context scaling (PI/NTK/YaRN) là chỉnh `inv_freq` — awareness-only
(`FRONTIER_PRACTICE_2026:205`). Câu interview: "Vì sao RoPE tổng quát hóa độ dài tốt hơn learned-absolute,
và vì sao MLA phải decouple RoPE?"

---

## Bài 2.4 — RMSNorm + pre-norm residual: giữ activation không nổ (`model.py` · `RMSNorm` :111 · `TransformerBlock` :857)
> **Câu hỏi first-principles:** derive RMSNorm TỪ LayerNorm bằng cách bỏ dần từng phần — phần nào thừa? Vì
> sao normalize trong fp32? Và vì sao norm đặt TRƯỚC sublayer (pre-norm) chứ không sau (post-norm)?
> **Neo (invariant):** `test_qk_norm_off_is_identity` gián tiếp + `test_loss_at_init_is_log_vocab` (`:46`) —
> RMSNorm init gain=1 nên forward lúc init là ánh xạ giữ scale, loss vẫn ≈ log V. Bất biến pre-norm: residual
> stream là đường thẳng identity từ embedding tới lm_head ⇒ gradient tới mọi layer không bị nén.

**1. Feynman — bài toán bằng lời.** Chồng 20+ block: nếu mỗi block khuếch đại activation một chút, tới đỉnh
stack activation nổ (hoặc teo) → training không ổn định. Normalization ép activation về scale cố định trước
mỗi sublayer. LayerNorm làm hai việc: (i) TRỪ mean (center), (ii) CHIA std (scale). RMSNorm hỏi: phần nào
thực sự cần? Trade-off: RMSNorm bỏ bước trừ-mean (rẻ hơn, 1 pass thay vì 2, không cần lưu mean) — thực
nghiệm cho thấy re-centering gần như vô hại với Transformer, chỉ re-scaling là load-bearing. Pre-norm vs
post-norm là câu hỏi *residual để làm gì*: pre-norm giữ một "đường cao tốc" identity xuyên suốt.

**2. Dẫn xuất từ đầu (derive).** LayerNorm: `y = (x − μ)/√(σ² + ε) · γ + β` với `μ = mean(x)`, `σ² =
var(x)`. Bỏ recentering (μ=0, β=0): `y = x/√(mean(x²) + ε) · γ` — đó CHÍNH LÀ RMSNorm, vì `RMS(x) =
√(mean(x²))`. `rsqrt(mean(x²)+ε)` (:123) chuẩn hóa mỗi token về RMS=1, rồi nhân gain per-channel `γ`
(`weight`, init 1, :117). **Vì sao fp32:** `mean(x²)` cộng d số bình phương — ở bf16 (7 bit mantissa) tổng
này mất chính xác nghiêm trọng ở d lớn / low-precision, và một RMS sai kéo lệch cả residual stream. Nên
`x32 = x.float()` (:122), tính norm fp32, rồi `.to(dtype)` cast lại (:124) — precision ở đúng chỗ đắt.
**Pre-norm:** block là `x + Sublayer(Norm(x))` (:888, :893) — residual `x` đi thẳng, KHÔNG qua norm; chỉ
INPUT của sublayer được norm. Vậy gradient từ lm_head chảy ngược qua đường `+x` identity tới mọi layer
không suy giảm ⇒ train sâu ổn định KHÔNG cần learning-rate warmup phức tạp. Post-norm (`Norm(x + Sublayer(x))`,
kiểu Transformer gốc) đặt norm TRÊN đường residual → gradient bị nén qua từng norm → cần warmup, dễ phân kỳ
sâu.

**3. Trace code.** `RMSNorm.forward` (:120): `x32 = x.float()` → `rms = rsqrt(x32.pow(2).mean(-1, keepdim=
True) + eps)` (:123) → `(x32 * rms * weight.float()).to(dtype)` (:124). `TransformerBlock` (:857): `attn_norm,
ffn_norm` là hai RMSNorm (:868, :870); forward `x = x + attn(attn_norm(x), ...)` (:888) rồi `x = x + ffn_
delta` với `ffn(ffn_norm(x))` (:890/893) — FFN trả DELTA, residual add giống hệt dù dense hay MoE. `Transformer
LM` (:897) đóng khung: `token_emb → N blocks → final_norm (:914) → lm_head (:915)` — một `final_norm` nữa
trước head (chuẩn pre-norm: norm cuối cùng vì residual stream chưa từng được norm ở đường thẳng).

**4. Cổng teach-back.** (a) Derive RMSNorm từ LayerNorm — nêu ĐÚNG hai thứ bị bỏ (mean-subtract + bias β) và
vì sao bỏ được. (b) *Sửa-và-đoán:* nếu tính RMS trong bf16 thay vì fp32, ở d_model=4096 context 8k thì
triệu chứng gì xuất hiện trước — loss-at-init lệch, hay spike ở step muộn? (Gợi: init vẫn ổn, hỏng là silent
drift/spike khi activation outlier lớn — đúng failure mode `FRONTIER_PRACTICE_2026:76`.)

**5. Frontier.** `modeling_qwen3.py:50 Qwen3RMSNorm` — `variance = hidden.to(float32).pow(2).mean(-1);
hidden * rsqrt(variance+eps)` rồi cast, Y HỆT ta (kể cả fp32-then-cast). `modeling_nanochat.py:50` `_norm(x)
= x * rsqrt(mean(x²)+eps)` nhưng `forward` gọi `_norm(x.float()).type_as(x)` (:53–54) — cùng ý; nanochat còn
thêm một norm TRƯỚC block đầu (`:414 "Additional norm before the layers"`) — biến thể ta không có.
Pre-norm là default 2026; peri-LN/sandwich (norm cả output sublayer) là ablation `FRONTIER_PRACTICE_2026:164`
(OLMo2 dùng reordered-norm). Câu interview: "Vì sao RMSNorm thay LayerNorm, và pre-norm giải quyết bài toán
train-sâu gì?"

---

## Bài 2.5 — SwiGLU: gated MLP, vì sao thắng ReLU/GELU (`model.py` · `SwiGLU` :189 · `silu` :185)
> **Câu hỏi first-principles:** phần "tính-trong-một-vị-trí" (FFN) vì sao lại là `W2(SiLU(W1x) ⊙ W3x)` —
> ba ma trận + một phép nhân element-wise — chứ không phải MLP hai lớp `W2·ReLU(W1x)` cổ điển?
> **Neo (invariant / shapes):** không có test riêng cho SwiGLU; neo là shape-contract + `ffn_dim` (:71) round
> `(8/3)·d` lên bội 64. Shapes: `x (·,d) → w1/w3: (·,d_ff) → SiLU(w1)⊙w3: (·,d_ff) → w2: (·,d)`. FFN trả DELTA
> để residual add đồng nhất (Bài 2.4). d_ff≈(8/3)d thay vì 4d để BÙ tham số cho ma trận thứ ba (iso-param).

**1. Feynman — bài toán bằng lời.** Attention trộn thông tin GIỮA các vị trí; FFN là phần model "suy nghĩ"
TRONG một vị trí — nâng lên chiều cao d_ff, phi tuyến, ép về d. MLP cổ điển: `W2·ReLU(W1x)` — một cổng
cứng (ReLU tắt/mở nhị phân). GLU (gated linear unit) thêm một *cổng học được*: nhánh `W3x` nhân element-wise
lên nhánh đã kích hoạt, cho model điều khiển MỀM luồng nào đi qua theo từng chiều, theo từng input. Analogy:
ReLU là công tắc bật/tắt; GLU là chiết áp (dimmer) — mỗi chiều một núm vặn do chính input quyết định. Trade-
off: thêm ma trận thứ ba (nhiều tham số hơn cho cùng d_ff) đổi lấy chất lượng-per-param tốt hơn; ta bù bằng
cách hạ d_ff từ 4d xuống ~(8/3)d để tổng param không đổi.

**2. Dẫn xuất từ đầu (derive).** GLU tổng quát: `(Wx) ⊙ σ(Vx)` — một nhánh giá trị, một nhánh cổng qua
hàm kích hoạt σ. Chọn σ = **SiLU** (Swish): `silu(x) = x·sigmoid(x)` (:186) — trơn, không-đơn-điệu, có
gradient khác 0 ở vùng âm (khác ReLU chết ở x<0), tự-gated. Ghép: `SwiGLU(x) = W2(SiLU(W1x) ⊙ W3x)` (:198)
— W1 là "gate" (qua SiLU), W3 là "up" (value tuyến tính), W2 là "down". Vì sao trơn hơn ReLU thắng: gradient
mượt giúp optimize; cổng nhân cho phép biểu diễn tương tác nhân tính (multiplicative) mà MLP cộng-tính không
có. Vì sao `d_ff≈(8/3)d`: MLP chuẩn 2-ma-trận dùng `d_ff=4d` (params `2·4d²=8d²`); SwiGLU 3-ma-trận với
`d_ff=(8/3)d` cho params `3·(8/3)d² = 8d²` — BẰNG NHAU, so sánh công bằng (iso-param). `ffn_dim` (:71) làm
đúng: `raw = int(8/3·d)`, round lên bội 64 (:75–76, cho GEMM alignment).

**3. Trace code.** `silu` (:185): `x * sigmoid(x)`. `SwiGLU.__init__` (:192): `w1: d→d_ff` (gate), `w3:
d→d_ff` (up), `w2: d_ff→d` (down) — tất cả `Linear` no-bias (:194–196). `forward` (:198): `w2(silu(w1(x)) *
w3(x))` — một dòng, đúng công thức. Được dùng làm FFN dense trong `TransformerBlock` (:879) khi layer không
phải MoE; MoE thay bằng `MoEFeedForward` (:877, Série sau) nhưng CẢ HAI trả delta để `x = x + delta` đồng
nhất (:893). `ffn_dim` từ `ModelConfig` (:71).

**4. Cổng teach-back.** (a) Vì sao `d_ff=4d` cho MLP nhưng `(8/3)d` cho SwiGLU là so-sánh-công-bằng — tính
params cả hai. (b) *Sửa-và-đoán:* thay SiLU bằng ReLU trong nhánh gate (`ReGLU`) hoặc bỏ hẳn nhánh gate về
`W2·ReLU²(W1x)` (nanochat) — cái nào mất tính "cổng mềm", và param đổi thế nào?

**5. Frontier.** ĐÂY là chỗ ta KHÁC nanochat rõ nhất. `modeling_nanochat.py:270 NanoChatMLP`: chỉ HAI ma
trận `fc1, fc2` + `activation_fn = ACT2FN["relu2"]` (config `hidden_act="relu2"`, :61) — tức `fc2(relu(fc1
x)²)`, **ReLU² KHÔNG cổng**, `intermediate_size=8192` cho d=768 (~10.7×d, rộng hơn vì không có nhánh gate
bù param). Karpathy chọn ReLU² (rẻ, đơn giản, một số bằng chứng ≈ SwiGLU ở scale nhỏ) — spec §8 liệt ReLU²-
vs-SwiGLU là "confirm at build time" (`FRONTIER_2026_ABLATIONS.md:224`). `modeling_qwen3.py:70 Qwen3MLP` thì
GIỐNG ta: `down_proj(act_fn(gate_proj(x)) * up_proj(x))` = SwiGLU. `modeling_deepseek_v3.py` dense MLP cũng
SwiGLU. Câu interview: "SwiGLU thắng ReLU-MLP nhờ đâu, và vì sao d_ff hạ xuống ~(2/3)·4d?"

---

## Bài 2.6 — QK-norm + tied/untied embeddings: hai đòn bẩy ổn-định & init (`model.py` · `q_norm/k_norm` :769 · `tie_embeddings` :916 · `cross_entropy` :969)
> **Câu hỏi first-principles:** (a) QK-norm — chặn attention-logit nổ ở đâu, và vì sao đặt TRƯỚC RoPE? (b)
> tied vs untied embedding — chia sẻ `token_emb` với `lm_head` được/mất gì, và vì sao *untied* làm derivation
> loss-at-init sạch hơn?
> **Neo (invariant):** `test_qk_norm_loss_at_init_holds` (`:150`) — QK-norm BẬT, loss vẫn ≈ log V (<0.3, :158)
> ⇒ norm đúng trục head_dim, không phải seq. `test_qk_norm_off_is_identity` (`:143`) — TẮT thì q_norm/k_norm
> là `nn.Identity` ⇒ path byte-identical model không-QK-norm. `test_loss_at_init_is_log_vocab` (`:46`).

**1. Feynman — bài toán bằng lời.** **QK-norm:** failure mode số 1 của model lớn ở bf16 là *attention logit
nổ* — `q·k` vọt lên vài trăm, softmax bão hòa, gradient chết, hoặc tràn bf16 → NaN ở step muộn (silent
divergence). QK-norm là fix rẻ nhất: RMSNorm mỗi head's q,k trên head_dim TRƯỚC khi vào softmax, ép `q,k`
về RMS đơn vị ⇒ `q·k` bị chặn ⇒ logit không chạy loạn. **Tied/untied:** `token_emb` (V×d, id→vector) và
`lm_head` (d→V, vector→logit) là hai ma trận HÌNH HỘP đối xứng — có thể DÙNG CHUNG một ma trận (tie). Tie
tiết kiệm V·d tham số (đáng kể khi V lớn, model nhỏ) và có lý thuyết "cùng không gian nghĩa"; untie cho hai
vai trò tự do riêng + init sạch. Trade-off này Bài 2.6 mổ cả hai.

**2. Dẫn xuất từ đầu (derive).** **QK-norm đặt trước RoPE:** RMSNorm chuẩn hóa ĐỘ DÀI vector; RoPE chỉ XOAY
(bảo toàn độ dài). Nếu norm SAU RoPE cũng cùng độ dài, nhưng convention Qwen3 là norm→rotate để norm thấy
q,k "thô" (trước khi trộn thông tin vị trí) — và quan trọng: norm trên **head_dim** (không phải seq/d_model),
nên loss-at-init không đổi (test :150 pin: nếu lỡ norm sai trục thì loss lệch khỏi log V). **loss-at-init ≈
log V:** derive — lúc init, mọi trọng số nhỏ/random, logits ≈ đều trên V lớp ⇒ softmax ≈ uniform ⇒ `p ≈
1/V` cho token đúng ⇒ cross-entropy `= −log(1/V) = log V`. `cross_entropy` (:969) tính qua logsumexp: `log_z
= logsumexp(logits)`, `loss = mean(log_z − chosen)` (:976–978) — ổn định số, khử log∘exp. **Untie làm sạch
derivation:** nếu `lm_head.weight = token_emb.weight` (tie, :917), thì init của head BỊ RÀNG BUỘC bởi init
embedding (`Embedding` init N(0,1) ±3σ, :104) — thang khác với `Linear` truncated-Xavier `std=√(2/(in+out))`
(:82). Untie (ADR-0004 §2, default `tie_embeddings=False` :46) để lm_head có init Xavier riêng ⇒ logits init
đều → `log V` sạch, không phải lý luận qua thang embedding.

**3. Trace code.** **QK-norm:** `MultiHeadSelfAttention.__init__` (:769): `q_norm = RMSNorm(head_dim) if
cfg.qk_norm else nn.Identity()` (cùng k_norm :770). `forward` (:786–787): `q = q_norm(q); k = k_norm(k)`
NGAY TRƯỚC `rope(q)/rope(k)` (:791–792). Off ⇒ Identity ⇒ byte-identical (test :143). **Tied/untied:**
`TransformerLM.__init__` (:910) `token_emb = Embedding(V,d)`; (:915) `lm_head = Linear(d,V)`; (:916) `if
cfg.tie_embeddings: lm_head.weight = token_emb.weight`. `forward` (:932) `token_emb(ids)` → blocks →
`final_norm` → `lm_head(x)` (:941). `cross_entropy` (:969) pin loss-at-init. `Embedding` init std=1 (:104)
vs `Linear` truncated-Xavier (:82) — chính chênh init này là lý do untie sạch hơn.

**4. Cổng teach-back.** (a) Derive loss-at-init = log V từ "logits đều lúc init" — và giải thích vì sao QK-
norm trên head_dim GIỮ được nó nhưng norm nhầm trục seq thì PHÁ. (b) *Sửa-và-đoán:* bật `tie_embeddings=
True` — loss-at-init còn ≈ log V không (gợi: test `FRONTIER_PRACTICE_2026:166` nói CÓ, vì head init = embed
init N(0,1) vẫn cho logits đủ nhỏ/đều)? Và tie tiết kiệm bao nhiêu param khi V=50k, d=768?

**5. Frontier.** **QK-norm:** `modeling_qwen3.py:248–249` `q_norm/k_norm = Qwen3RMSNorm(head_dim)` với
comment "only on the head dim!"; `:263–264` `q_norm(q_proj(x).view(...))` — Y HỆT ta (Qwen3 thêm QK-norm +
BỎ QKV-bias "for stable training", 2505.09388). `modeling_nanochat.py:222–223, 244–245` cũng q_norm/k_norm.
Gemma3 THAY soft-capping bằng QK-norm; Kimi-K2 dùng MuonClip/QK-Clip cho cùng failure (`FRONTIER_2026_
ABLATIONS.md` F9, :158 — chỉ cần nếu >1B logit vượt ~100). **Untied:** `modeling_nanochat.py:434 _tied_
weights_keys` + config `tie_word_embeddings=False` (:72) — nanochat *untie* (embedding_lr/unembedding_lr
riêng), và có `final_logit_softcapping=15.0` (:67, :500–502 tanh-softcap) mà TA KHÔNG có (spec §1: "no logit
softcap"). Câu interview: "QK-norm chặn instability gì, đặt trước hay sau RoPE, và tie-vs-untie embedding
đánh đổi ra sao?"

---

*Đóng série:* forward pass gấp từ 6 primitive — retrieval khả-vi (2.1) → multi-head+GQA (2.2) → RoPE
dạy vị trí (2.3) → RMSNorm+pre-norm giữ ổn định (2.4) → SwiGLU tính-trong-vị-trí (2.5) → QK-norm+untied
làm đòn bẩy cuối (2.6). Hai neo `causal-no-leak` + `loss-at-init ≈ log V` sống xuyên suốt. Série sau
(M3+): cross-entropy/AdamW/cosine schedule (training), rồi MoE (`moe.py`) + MLA (`mla.py`, F5) + Muon
(F1) + MTP (F2) — các đòn bẩy frontier trong `FRONTIER_2026_ABLATIONS.md` §3, mỗi cái pre-register một
PREDICTION iso-FLOP. Cross-link perf: MLA weight-absorption → `roadmap/S2 Bài 2.8`; TP/PP/EP → `roadmap/S6`.
