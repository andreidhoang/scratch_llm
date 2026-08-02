# Cẩm nang đọc Kimi K3 Tech Report — đọc để BUILD

> Paper: **Kimi K3: Open Frontier Intelligence** (Kimi Team, arXiv:2607.24653v1, 27/07/2026).
> Đây là *spec of record* của toàn K3 track (`docs/k3/ROADMAP.md`). Cẩm nang này KHÔNG dịch
> paper — nó chỉ cho bạn **đọc chỗ nào, theo thứ tự nào, trả lởi câu hỏi gì, và cẩn thận bẫy nào**,
> bản đồ thẳng vào các module bạn sẽ hand-build trong `src/scratch_llm/k3/core/`.
>
> **Quy tắc đọc (teach-back):** đọc xong một section → đóng PDF → viết lại phương trình chính +
> giải thích bằng lởi của bạn. Không viết lại được = chưa đọc xong, đọc lại. Đây cũng là chuẩn
> "delete test" của từng core module (`src/scratch_llm/k3/HANDCRAFTED.md`).
>
> Thuật ngữ giữ nguyên tiếng Anh (retention factor, chunkwise, router, gate…) — đó là từ bạn sẽ
> dùng khi viết code và khi đi phỏng vấn.

---

## 0. Bản đồ tổng — paper dài ~40 trang, bạn chỉ cần "sở hữu" ~12 trang

| Phần paper | Nội dung | Đọc thế nào | Gắn rung/module |
|---|---|---|---|
| Abstract + §1 + Fig 2 | Toàn cảnh: 3 trục scale (sequence/depth/width) | Kỹ, Pass 1 | định hướng |
| Table 1 (K2 vs K3) | Bảng hyperparameter chính thức | Kỹ, Pass 1 | `k3/config.py` (đã land, FACTS A13/A18) |
| §2.1.1 KDA | Trái tim paper: recurrence, chunkwise, decay g_min=−5, full-rank gate | **Kỹ nhất — đọc 3 lần** | `core/kda.py` (K2) |
| §2.1.2 Gated MLA | NoPE toàn cục + full-rank output gate + FP32 attention | Kỹ | `core/gated_mla.py` (K3) |
| §2.2 AttnRes | Pseudo-query học được, Block AttnRes | Kỹ | `core/attn_res.py` (K4) |
| §2.3 LatentMoE + SiTU + QB | Router sigmoid, RMSNorm, activation mới, Quantile Balancing | Kỹ | `core/latent_moe.py`, `core/situ.py` (K5) |
| §2.4 Vision | MoonViT-V2 from scratch | Lướt (K10 — reading-only) | — |
| §2.5 Per-Head Muon | Optimizer: orthogonalize theo từng head | Kỹ (ngắn) | `muon.py` (K6, paired) |
| §3 Pre-training | Data, scaling law, cosine vs WSD, long-context curriculum | Đọc khi tới K6/K8 | K6/K8 |
| §4 Post-training | SFT → RL → MOPD; §4.1.4 = QAT + draft model | §4.1.4 kỹ khi tới K7; còn lại lướt | `qat.py` (K7), K10.1 |
| §5 Infrastructure | FlashKDA, KDA Context Parallelism, MoonEP, serving kernels | §5.1 kỹ khi làm kernel; còn lại lướt | K10.2, K9 |
| §6 Evaluations | Con số benchmark | Lướt — biết model giỏi chỗ nào | K8 |
| §7 Case studies | RTL chip design, v.v. | Bỏ qua (tham khảo vui) | — |
| §B/§C/§D (appendix) | SiTU expansion + bound, QB derivation, histogram estimator | Đọc khi build module tương ứng | K5 |

---

## Pass 1 (~90 phút) — bức tranh tổng

**Đọc:** Abstract → §1 → §2 đoạn mở đầu (trước §2.1) → Table 1 → nhìn Fig 2.

**Câu hỏi phải trả lởi được trước khi sang Pass 2:**

1. Ba trục "scale information flow" là gì, và mỗi trục tương ứng cơ chế nào?
   *(sequence → Hybrid Attention 3 KDA : 1 Gated MLA; depth → AttnRes; width → Stable LatentMoE
   16-of-896.)*
2. Vì sao gọi là "2.5× scaling efficiency over K2" — con số đó đến từ đâu? *(§3.2, Fig 7: fit
   scaling-law trên held-out OOD validation; KHÔNG phải một benchmark.)*
3. Table 1: K3 khác K2 ở đâu? *(61→93 layers, 384→896 experts, 8→16 active, 1→2 shared,
   64→96 heads, MLA→Hybrid KDA–MLA, SwiGLU→SiTU-GLU, 128K→1M context, thêm Latent MoE dim
   3584 và ViT 401M. Hidden 7168 và vocab 160K GIỮ NGUYÊN.)*
4. Abstract thừa nhận điều gì? *(Vẫn thua Claude Fable 5 và GPT-5.6 Sol — đọc cả phần này để
   học cách một frontier lab viết claim trung thực.)*

**Teach-back Pass 1:** vẽ lại Fig 2 (một block = 3 KDA + 1 MLA + AttnRes + LatentMoE) từ trí nhớ.

---

## Pass 2 — đọc kỹ theo rung build (đây là phần chính)

### Rung K2/KDA → đọc §2.1.1 (spec của `core/kda.py`) — 3 lượt

**Lượt 1 — recurrence (Eq. 1–2).** Câu hỏi:

- State `S_t ∈ R^{dk×dv}` khác KV-cache ở điểm gì về mặt bộ nhớ? *(Kích thước CỐ ĐỊNH theo
  sequence length — đây là toàn bộ lý do tồn tại của linear attention.)*
- Eq. 1: `S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ`. Vai trò của từng thừa số:
  `Diag(α_t)` là gì? **Per-channel forget gate** — mỗi kênh của key quên với tốc độ RIÊNG
  (khác Gated DeltaNet: decay SCALAR per-head — đây là nâng cấp lớn nhất của KDA so với GDN
  mà `linear_attn.py` của bạn đang implement). `(I − β k kᵀ)` làm gì? *(Rank-1 ERASE trước khi
  WRITE — delta rule = online least-squares correction.)*
- Eq. 2: thứ tự `ShortConv → Swish → L2Norm(q,k)`. Vì sao L2Norm chỉ trên q,k? *(DeltaNet:
  ổn định eigenvalue; v thì không normalize.)*

**Lượt 2 — chunkwise form (Eq. 3–4).** Đây là chỗ khó nhất của cả paper — đọc chậm:

- `γ_{i→j} = Π_{r=i..j} α_r` (cumulative decay), `Γ` stack theo hàng. Vì sao cần `K/Γ`
  (reciprocal cumulative decay)? Và vì sao nó nguy hiểm? *(Tích các số ∈ (0,1) → reciprocal
  BÙNG NỔ theo độ dài chunk → overflow trong finite precision. Đây chính là bẫy dẫn tới Eq. 5.)*
- Eq. 4: `A[t] = Tril((Q⊙Γ)(K/Γ)ᵀ)`, `O[t] = (Γ⊙Q)S[t] + A[t]·Ṽ[t]` với `Ṽ = U − W·S`.
  Tách rõ hai số hạng: **inter-chunk** (đọc state cũ) vs **intra-chunk** (tương tác trong chunk).
- Bẫy nhỏ nhưng chí mạng: vì sao `Tril` GIỮ cả đường chéo (không phải strictly lower)?
  *(Vì output tại token i đọc state SAU update của chính token i — sửa chỗ này sai là
  chunkwise ≠ recurrent, lệnh test ba-đường của bạn bắt được ngay.)*
- UT transform: paper CHUYỂN TIẾP sang Kimi Linear [63] — đọc Kimi Linear §3.1 kèm chỗ này
  (đó là lý do rung K2 giao paper kèm theo).

**Lượt 3 — lower-bounded decay + full-rank gate (Eq. 5–6), Fig 3.** Đây là PHẦN MỚI của K3
so với Kimi Linear — nếu chỉ đọc Kimi Linear bạn sẽ thiếu hai thay đổi này:

- Eq. 5: `g = g_min · Sigmoid(e^A · z) ∈ (g_min, 0)`, `α = exp(g)`. So với Kimi Linear
  (`g = −e^A·Softplus(z) ∈ (−∞, 0)`): cận dưới `g_min = −5` ⇒ mọi retention factor
  `α > e⁻⁵ ≈ 6.7×10⁻³` ⇒ cumulative log-decay trên tile 16-token nằm trong (−80, 0) ⇒
  reciprocal < e⁸⁰, VỪA dải BF16 ⇒ **cả diagonal tile lẫn off-diagonal tile đều chạy được
  dense Tensor-Core matmul** (bỏ đường position-pair — bottleneck intra-chunk). Câu hỏi delete
  test: *nếu g_min = −20 thì sao?* (e⁸⁰×(20/5)=e³²⁰ → tràn BF16. Vậy −5 là con số engineering,
  không phải tuning.)
- `A_h` init = 0, learnable PER-HEAD log-scale. (Chú ý: HF reference code init `log U(1,16)`
  theo fla default — FACTS A13 ghi nhận mâu thuẫn này, theo report: init 0.)
- Eq. 6: output gate FULL-RANK `y = W_o[Sigmoid(W_g x) ⊙ RMSNorm(õ)]` — Kimi Linear dùng
  low-rank. Gate lấy từ **input x_t** (không phải từ output õ) — cite Gated Attention
  (arXiv:2505.06708).

**Teach-back KDA:** viết lại Eq. 1, Eq. 5, và giải thích bằng lởi vì sao bounded decay cho
phép all-Tensor-Core. Nói được "vì sao KDA vừa là attention vừa là positional encoding"
(→ liên hệ NoPE ở §2.1.2) là đạt.

### Rung K3/Gated MLA → đọc §2.1.2 (spec của `core/gated_mla.py`)

- Câu hỏi 1: K3 khác K2/K2.5 ở điểm gì trong MLA? **NoPE cho TẤT CẢ MLA layers** — không
  RoPE ở bất cứ đâu trong model. Ai gánh positional information? *(Các KDA layers xen kẽ —
  recurrent decay + ShortConv tạo recency bias; MLA chỉ lo "unrestricted global content
  interaction".)* Lợi ích hệ thống: extend context KHÔNG cần retune RoPE base / YaRN (§3.4).
- Câu hỏi 2 (bẫy sách giáo khoa): Vizuara capsule 15 nói "Decoupled RoPE in the MLA layers" —
  đúng hay sai với K3? **SAI** (FACTS A11/B8). Decoupled RoPE là di sản DeepSeek/K2 — học nó
  trong `mla.py` của repo (heritage path), build NoPE cho K3.
- Eq. 7: gate `y = W_o[Sigmoid(W_g x) ⊙ õ]` — full-rank, channel-wise, từ input.
- Chi tiết dễ bỏ sót: **FP32 attention output trong training** (sửa biased rounding của flash
  attention, ref [98]) — và họ phải redesign kernel (overlap output tile với KV staging buffers
  thay vì query tile) để đổi lấy shared memory. Đây là ví dụ mẫu của algorithm-system co-design.

### Rung K5/LatentMoE + SiTU + QB → đọc §2.3 (spec của `core/latent_moe.py` + `core/situ.py`)

Đọc theo thứ tự logic paper trình bày: vấn đề → 3 giải pháp.

- **Vấn đề (đoạn mở §2.3):** vì sao cần latent width ℓ? *(Routed expert nhận full d-dim →
  communication + weight traffic tỉ lệ với số expert active; LatentMoE tách model width khỏi
  expert width ⇒ 896 experts mới gánh nổi.)* Hai failure mode ở 2.8T scale: (1) chuỗi
  `W↓ → expert FFN → W↑` ≈ 4 matmul liên tiếp → **exploding internal activations**;
  (2) balance ~10³ experts vượt regime mà aux-loss-free bias update cổ điển hoạt động tốt.
- **Eq. 11:** `u = Σ_{i∈Tk} p_i E_i(W↓x)`, `y = Σ_j E_j_shared(x) + W↑ RMSNorm(u)`.
  Normalized LatentMoE (§2.3.1): RMSNorm nằm ở ĐÂU và vì sao? *(Giữa aggregation và W↑ —
  scale của u biến thiên theo expert được chọn; paper: "consistently improves validation loss".)*
- **Eq. 12 SiTU-GLU + Fig 4:** softcap `β·tanh(x/β)` đặt ở ĐÂU? *(Linear factor của Swish gate
  VÀ độc lập ở up branch; sigmoid factor KHÔNG cap.)* β1=4 (gate), β2=25 (up), bound
  |f| ≤ β1β2 = 100. Vì sao SwiGLU nguy hiểm ở low precision? *(Cả hai factor đều unbounded —
  coincident large coordinates → activation outliers → overflow.)* Đọc thêm **§B** (local
  expansion, limiting case, formal bound, so sánh hard clamp) khi hand-build `situ.py`.
- **Eq. 13–14 QB + Fig 5 + §C/§D:** đây là phần tinh vi nhất của MoE — đọc theo Fig 5
  (m=8, n=4, k=1):
  - Router: `s = Sigmoid(W_r x)`, `T = argtopk(s + b)`, nhưng mixture weight `p` tính từ s
    KHÔNG có b — vì sao tách? *(b chỉ điều phối dispatch, không chạy vào gradient của router.)*
  - Update cũ (DeepSeek): `b += γ·sign(ℓ̄ − ℓ_j)` — hạn chế gì? *(γ đánh đổi slow adaptation
    vs oscillation; 896 experts làm nó hỏng.)*
  - QB: dùng Top-(k+1) để lấy cutoff `α_i` (entry thứ k+1 = ngưỡng vào Top-k). Bias mới:
    `b̂_j ← −quantile_{1−k/n}(s_:,j − α)` rồi trừ mean. Chứng minh ngắn trong paper: count theo
    threshold monotone decreasing; đặt count = q = mk/n ⇒ đúng (1−k/n)-quantile.
  - **Causality (bẫy):** update chỉ có hiệu lực STEP SAU — "a batch is never routed with a bias
    derived from itself". Quên dòng này = bug silent.
  - **Histogram estimator (§D):** không gather hàng triệu margin; mỗi expert một histogram
    vài trăm bins, MỘT all-reduce cộng bin counts — counts additive nên đúng cho global batch
    bất kể sharding. Đây là chi tiết "production-grade" đáng học riêng.
  - Inference: bias ĐÓNG BĂNG.

**Teach-back MoE:** vẽ lại Fig 5 với m=8, n=4, k=1 và tự chạy một step QB trên giấy.

### Rung K4/AttnRes → đọc §2.2 (spec của `core/attn_res.py`)

- Ý tưởng gốc (đoạn mở — rất đáng ngâm): residual connection nén mọi thông tin trước đó vào
  MỘT state theo depth — "a bottleneck reminiscent of RNNs over time". Transformer thay
  recurrence bằng attention theo sequence; AttnRes làm điều tương tự theo **depth**.
- Eq. 8–9: pseudo-query `q_l = w_l` HỌC (một vector per layer, params = L·d), key=value=embedding
  + các block outputs, kernel `ϕ(q,k) = exp(qᵀ RMSNorm(k))`. Vì sao RMSNorm TRONG kernel?
  *(Tránh layer có magnitude lớn át weight.)*
- Full → Block (Eq. 10): vì sao phải block? *(Full rẻ về compute — O(L²d) với L<100 — nhưng
  tốn O(Ld) memory + PP-communication; Block giảm còn O(Nd).)* Cấu trúc: block rep = SUM các
  layer trong block; `b_0 = h_1` (embedding luôn là source); layer đầu block thấy các block rep
  trước, layer sau thấy thêm partial sum trong block; merge bằng **online softmax**.
- Con số K3: 8 blocks × 12 layers (93 → partial final block; 9 nguồn kể embedding).

### Rung K6 (phần paired) → đọc §2.5 Per-Head Muon (ngắn, 1 đoạn)

- Full-matrix Newton–Schulz trên Q,K,V coi mọi head là một khối coupled ⇒ head có momentum
  lớn át update direction. Per-Head: partition momentum theo head dim, orthogonalize từng block
  riêng ⇒ update scale cân bằng giữa các head + NS trên tall blocks RẺ HƠN. (Liên hệ: repo đã
  có Muon ở `optim.py` — bạn chỉ hand-write phần partition; xem HANDCRAFTED Class 2.)

### Rung K7 → đọc §4.1.4 (QAT + draft model) — đọc khi tới K7

- QAT: MXFP4 cho RỘING EXPERT WEIGHTS, activations MXFP8, mọi thứ khác giữ high precision;
  QAT xuyên suốt SFT + RL; **rollout và training dùng CÙNG quantization scheme** (xóa
  train–inference mismatch). Câu hỏi: vì sao chỉ quantize experts? *(97.9% params nằm ở đó —
  chính xác là con số `test_routed_expert_dominance` của bạn.)*
- Draft model (K10.1): MTP layer pre-train sẵn → fine-tune thành EAGLE-3-style draft, unroll
  7 step; features từ AttnRes block thứ 1/4/cuối; `W_E3` init `[0 0 I]` (lúc khởi tạo fusion ≡
  high-level feature — mẹo init rất đẹp); **LK loss = −log Σ_x min(p(x), q(x))** — optimize
  TRỰC TIẾP acceptance rate thay vì KL surrogate. Câu hỏi: vì sao KL không đủ? *(Với draft
  capacity-limited, min-KL không bảo đảm max acceptance.)*

---

## Pass 3 — đọc khi tới rung tương ứng

| Khi tới… | Đọc | Ghi chú build |
|---|---|---|
| K6 (train mini-K3) | §3.1–3.4 | Methodology đáng học: cosine vs WSD so sánh FAIR (mỗi schedule tự tune riêng — §3.2); 4-stage context curriculum 8K→64K→256K→1M (§3.4); long-doc cleaning + upsampling + synthetic "scattered-task" data |
| K8 (1M context) | §3.4 + §6.2 | NoPE ⇒ extrapolate 1M không cần sửa gì; cách họ synthesize data bắt model attend full-context |
| K9 (serving) | §5.4 | KDA-aware prefix cache; decode kernels §5.4.2 |
| K10.2 (kernel) | §5.1 | FlashKDA (CUTLASS, overlap intra-chunk với state propagation); **KCP Eq. 17**: state delta-rule KHÔNG cộng trực tiếp được như vanilla linear attention — phải tách `M_t←1` (cumulative transition) + `S̃` (state from zero), compose bằng prefix scan, chỉ MỘT all-gather kích thước cố định. Đây là đoạn math đẹp thứ hai của paper sau Eq. 4 |
| bất cứ lúc nào | §6 + Abstract | Đối chiếu claims; học cách trình bày eval trung thực |

---

## Checklist teach-back tổng (đậu = sẵn sàng build)

- [ ] Viết Eq. 1 (KDA recurrence) + nêu 2 khác biệt của K3 vs Kimi Linear (Eq. 5, Eq. 6).
- [ ] Giải thích vì sao `1/Γ` overflow và `g_min = −5` cứu nó thế nào (con số e⁸⁰ ở đâu ra).
- [ ] Vì sao K3 bỏ RoPE hoàn toàn mà vẫn "biết vị trí"? Ai gánh recency bias?
- [ ] Vẽ Fig 5 và chạy tay một QB update; nêu rõ tính causality (one-step delay).
- [ ] SiTU-GLU: softcap ở nhánh nào, β bao nhiêu, bound bao nhiêu, vì sao SwiGLU nguy hiểm
      ở MXFP8.
- [ ] AttnRes: vì sao nói "residual stream là RNN theo depth"? Block AttnRes tiết kiệm cái gì
      (không phải compute)?
- [ ] Per-Head Muon khác Muon thường ở đâu; lợi ích kép (stability + chi phí NS).
- [ ] QAT: quantize tensor nào, activation nào, ở stage nào, và vì sao rollout/training phải
      cùng scheme.

## Ghi chú trung thực (ledger)

- Những gì paper KHÔNG công bố: tổng số training tokens, peak LR, batch size, loại/số GPU,
  tổng FLOPs (FACTS §5). Đừng ai bịa ra những con số này.
- Khi sách Vizuara mở (3/8): cross-read capsule 7–12 với §2.1.1, capsule 15 với §2.1.2
  (kỳ vọng lệch: RoPE), capsule 19–23 với §2.3 — log khác biệt vào FACTS.md.
