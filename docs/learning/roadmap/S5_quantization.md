# SÉRIE 5 — A5 quantization numerics: precision là biến thiết kế per-tensor

> **Số dòng pin theo commit `9e61d7a`.** Code: `src/scratch_llm/quant/`. Số đo: `bench/RESULTS.md`
> §"Perf track (A5)" (dòng 532–601). Design note: `performance/notes/A5_design_note.md`.

**Vì sao série này.** Quantization không phải "nén cho nhỏ rồi chấp nhận sai" — nó là một **biến
thiết kế chọn RIÊNG cho từng lớp tensor** (weight, activation, KV). Xuyên suốt năm bài là ba câu
niệm chú: (1) **oracle luôn là SQNR/MSE, KHÔNG BAO GIỜ `allclose`** — số dB nói cho ta biết bao
nhiêu bit tín hiệu còn sống, `allclose` chỉ nói pass/fail mù; (2) một giá trị dB rơi **dưới sàn**
đạt được là **bug** (sai scale / thiếu clamp / sai rounding), không phải "mất mát chấp nhận được";
(3) **càng ít bit → càng cần granularity mịn hơn** (per-tensor → per-channel → group → two-level
block). Ta leo thang đúng theo trục đó: INT8 dựng luật 6 dB/bit → INT4 group + packing → FP4
two-level (NVFP4 vs MXFP4) → FP8-KV trong vòng serving → AWQ (activation-aware, cùng bit-width mà
recover accuracy).

---

## Bài 5.1 — INT8: sàn SQNR, luật 6 dB/bit, và scale factor-out khỏi GEMM (`quant/int8.py` · `quantize_symmetric`:101, `w8a8_linear`:188)

> **Câu hỏi first-principles:** Vì sao quantization là *cộng noise trắng biên độ s/2*, tại sao thêm
> 1 bit mua đúng 6.02 dB, và vì sao per-channel weight scale *biến mất* khỏi vòng trong của matmul?
> **Số đo (aha):** INT8 sym per-tensor **40.50 dB** Gaussian (đo 6.75 dB/bit, lý tưởng 6.02); asym
> **43.50 > sym 37.44** (+6.07 dB) trên data lệch; per-channel **42.34 > per-tensor 33.25** (+9.09).

**1. Feynman — bài toán bằng lời.** Quantize = làm tròn số thực về **lưới đều bước `s`**. Sai số
làm tròn bị chặn bởi `s/2` và hành xử như **noise cộng, phương sai `s²/12`** (uniform trên một
bước). Thêm 1 bit → `s` giảm một nửa → công suất noise giảm 4 lần → SNR tăng `10·log10(4) = 6.02
dB`. Đó là **luật 6 dB/bit** — sàn `SQNR ≈ 6.02·bits + c`. Đánh đổi cốt lõi nằm ở việc *chọn
full-scale*: một tensor Gaussian đặt range theo `amax` (outlier) phải trả **peak-to-average
penalty** → INT8 (~7.99 bit hiệu dụng) rơi ~40 dB chứ không phải ~44 dB lý tưởng. Ta báo số **đo
được**, và chỉ coi giá trị *dưới* sàn là bug.

**2. Cơ chế (derive từ đầu).** Sym: `s = amax/127`, `q = round(x/s).clamp(-127,127)` — pin 0.0 vào
code 0, phí toàn bộ nửa âm nếu data lệch (post-GELU toàn dương → mất ~6 dB). Asym affine sửa: fit
`[min,max]` lên `[0,255]` với zero-point nguyên `z = round(-min/s)` → giành lại range. Per-channel:
mỗi output row một scale → mỗi bước co về range thật của row đó, strictly beats per-tensor.
**Điểm load-bearing (factor-out):** GEMM co theo `k`: `Y[m,n] = Σ_k (q_X[m,k]·s_X[m])·(q_W[n,k]·s_W[n])
= s_X[m]·s_W[n]·Σ_k q_X[m,k]·q_W[n,k]`. Vì `s_X` (per-token) và `s_W` (per **output**-channel) đều
**KHÔNG phụ thuộc `k`**, cả hai kéo thẳng ra ngoài tổng → chạy INT8×INT8→INT32 accumulate, rồi
dequant **một lần** bằng outer-product `s_X[m]·s_W[n]`. Nếu weight scale biến thiên *dọc `k`* (per-
input-channel) → không hoist được → mất luôn integer GEMM. Đây là lý do weight quantize per-*output*-
channel, không phải per-input.

**3. Trace code.** `quantize_symmetric` (:101): `amax` reduce (:118/:121) → `s = (amax/127).clamp_min`
(:122) → `q = round(x/s).clamp(±127).to(int8)` (:123). `sqnr_db` (:81) tính float64, trả `+inf` nếu
noise=0. `quantize_affine` (:141): min/max (:159–164), ép 0 vào range (:166–167), `z` (:169), `q` unsigned
(:170). Payoff `w8a8_linear` (:188): per-token X `quantize_symmetric(x2d, axis=0)` (:204), per-channel W
(:206) → **INT32 matmul** `q_x.int32 @ q_w.int32.t()` (:209) → dequant một lần `acc * (s_x * s_w)` (:212).

**4. Cổng teach-back.** (a) Giải thích bằng lời vì sao asym thắng sym +6 dB trên tensor toàn dương —
chạm hai chữ *zero-point* và *nửa range bị phí*. (b) **Modify-and-predict:** nếu chuyển weight sang
per-*input*-channel scale (biến thiên dọc `k`), luật số ở §2 hỏng chỗ nào — vế phải còn tách được
`s_W` ra ngoài `Σ_k` không, và điều đó giết tính năng gì của phần cứng?

**5. Frontier.** W8A8 (SmoothQuant/TensorRT) là recipe INT8 chuẩn: per-token dynamic activation ×
per-channel static weight — chính vì scale hoist. Interview: "Vì sao weight quantize theo output
channel chứ không input channel?" → factor-out khỏi contraction axis.

---

## Bài 5.2 — Group-INT4: 2-nibble packing bit-exact + group scaling (`quant/int4_group.py` · `pack_int4`:62, `quantize_groupwise_int4`:108)

> **Câu hỏi first-principles:** Ở 4 bit, hai điều phải đúng: (1) *lưu* hai code 4-bit vào một byte
> mà round-trip **bit-exact** kể cả two's-complement âm và độ dài lẻ; (2) *chọn scale* mịn tới mức
> nào để 4 bit còn dùng được?
> **Số đo (aha):** pack/unpack **bit-exact** qua 500-case fuzz + adversarial (0x78/0xFF/odd);
> **SQNR 18.64 dB vs sàn giải tích 18.60**; group (g=128) **18.69 > per-tensor 17.12**.

**1. Feynman — bài toán bằng lời.** INT4 chỉ có 16 mức `{-8..7}`. Hai vấn đề tách rời. *Storage:*
một byte chứa hai nibble — nếu pack/unpack lệch 1 bit thì mọi số sau đó rác, nên đây là bài toán
**bit-exact tuyệt đối**, không phải "gần đúng". *Numeric:* 4 bit quá ít để một scale phủ cả tensor;
chia weight thành **group 128 phần tử liền nhau dọc contraction axis**, mỗi group một scale
`absmax/7` → lưới bám dynamic range cục bộ. Đánh đổi: thêm scale (overhead) đổi lấy độ mịn — đó là
điều làm 4-bit weight *dùng được* (GPTQ/AWQ/Marlin đều group-wise).

**2. Cơ chế (derive từ đầu).** Packing: mask low-4-bit `q & 0xF` biến `-8..-1 → 8..15` (two's-
complement), `byte = (hi<<4)|lo`. Unpack sign-extend: nibble `≥ 8 → trừ 16` (0xF→−1, 0x8→−8). Độ
lẻ: append một nibble 0, drop khi biết `n` gốc. Group quant: reshape `[out, n_groups, g]`, `absmax`
theo trục cuối, `s = absmax/7`, `round(w/s).clamp(-8,7)`. Sàn SQNR `≈ 6.02·4 = 24 dB` là bound mid-
tread cho uniform; trên Gaussian non-clip nó rơi thấp hơn (18.6) nhưng group *nâng* so per-tensor.

**3. Trace code.** `pack_int4` (:62): `nibbles = q & 0xF` (:76), pad lẻ (:77–78), `(hi<<4)|lo` (:81).
`unpack_int4` (:85): tách hi/lo (:96–97), interleave (:98), `where(out≥8, out−16)` sign-extend (:101).
`quantize_groupwise_int4` (:108): reshape group (:124), `absmax` (:125), `s = absmax/7` clamp tiny (:127),
`round/clamp/int8` (:128). `w4a16_linear` (:148): dequant về dtype activation (:162) rồi `F.linear` — mô
phỏng accumulate của kernel Marlin/GPTQ.

**4. Cổng teach-back.** (a) Vì sao unpack phải sign-extend, và điều gì hỏng nếu quên (0xF sẽ decode
thành gì)? (b) **Modify-and-predict:** giảm `group_size` từ 128 → 32 (mịn hơn). SQNR sẽ tăng hay
giảm, và cái *giá* nào đội lên (đếm số scale fp32 phải lưu cho weight `[out, in]`)?

**5. Frontier.** W4A16 group-128 là default suy luận LLM 2024–2026 (Marlin kernel unpack nibble
inline, dequant trong shared memory). Interview: "Vì sao pack hi-nibble-trước lại quan trọng?" — phải
khớp đúng thứ tự unpack, đây là hợp đồng bit-level test 500-case găm.

---

## Bài 5.3 — NVFP4 vs MXFP4: E2M1/E4M3/E8M0 codec, two-level scaling (`quant/nvfp4_mxfp4.py` · `nvfp4_quantize`:212, `mxfp4_quantize`:257)

> **Câu hỏi first-principles:** Khi element format giống hệt nhau (đều FP4 = E2M1), thì cái gì quyết
> định format nào chính xác hơn? Trả lời: **cách encode block scale** và **kích thước block**.
> **Số đo (aha):** **NVFP4 MSE 9.05e-3 (20.43 dB) < MXFP4 1.34e-2 (18.74 dB) = 1.48×**; block-scaled
> GEMM lệch **0.28%** vs bf16-same-W; verifier xác nhận MXFP4 baseline là bản E8M0-ceil *mạnh hơn*
> (không rigged).

**1. Feynman — bài toán bằng lời.** FP4 = E2M1 chỉ biểu diễn 8 độ lớn `{0,.5,1,1.5,2,3,4,6}` — quá
thô để phủ range. Cả hai format đắp thêm một **scale theo block nhỏ** rồi mới round element lên lưới
E2M1. Khác biệt nằm ở **scale codec**: MXFP4 lưu scale là **E8M0 = số mũ thuần (power-of-two, bias
127)** → làm tròn scale lý tưởng về octave gần nhất, lệch tới **2×**; NVFP4 dùng **hai tầng** — một
`s_global` fp32 per-tensor × một `s_block` **E4M3** (fp8 non-power-of-two) per-block-16 → bám scale
lý tưởng `block_amax/6` tới ~1 phần 16. Cộng thêm block NVFP4 mịn hơn (**k=16 vs k=32**). Hai cơ chế
đó là **toàn bộ** lý do NVFP4 thắng — nếu MXFP4 ever tie/thắng thì đó là bug đặt scale, không phải
"loss chấp nhận được".

**2. Cơ chế (derive từ đầu).** NVFP4 recipe (TensorRT-ModelOpt): `s_global = amax/(448·6)` (đảm bảo
`s_block ≤ 448` luôn E4M3-biểu-diễn-được), `s_block = quantize_E4M3(block_amax/6/s_global)`, `q =
round_E2M1(x/(s_global·s_block))`, dequant `x̂ = s_global·s_block·q`. MXFP4: `s_block = 2^ceil(log2(
block_amax/6))` — **no-overflow policy** (ceil để không element nào `|x|/s_b > 6`, không clip). Chọn
ceil-E8M0 (không round-nearest) chính là để MXFP4 là **baseline mạnh**, fair. `/6` (element-max) và
`·448` (fold global) là hai hằng số của recipe.

**3. Trace code.** Codec: `e2m1_round_to_code` (:85) argmin trên 8 anchor + sign bit3; `quantize_e4m3`
(:135) binade + 3-bit mantissa round, clamp 448; `quantize_e8m0` (:150) round-to-power-of-2;
`_e8m0_no_overflow_scale` (:162) `2^ceil(log2(amax/6))`. `nvfp4_quantize` (:212): `s_global` (:218),
`s_block = quantize_e4m3(block_amax/6/s_global)` (:220), `codes = e2m1_round_to_code(xb/scale)` (:222).
`mxfp4_quantize` (:257): `s_block = _e8m0_no_overflow_scale` (:263), codes (:264). `quant_linear_nvfp4`
(:296): quant weight per-row-per-K-block (:314–322), accumulate **block-by-block dọc K** với scale gấp
sẵn vào `w_hat` (:326–328) — mô phỏng block-scaled tensor-core GEMM.

**4. Cổng teach-back.** (a) Hai cơ chế khiến NVFP4 thắng MXFP4 là gì — nói rõ E4M3 vs E8M0 bám scale
lý tưởng khác nhau thế nào. (b) **Modify-and-predict:** đổi MXFP4 block từ 32 → 16 (bằng NVFP4)
nhưng vẫn giữ E8M0. Khoảng cách MSE 1.48× thu hẹp bao nhiêu — còn *cơ chế nào* của NVFP4 vẫn còn
thắng sau khi đã cân bằng block size?

**5. Frontier.** NVFP4 (Blackwell) vs MXFP4 (OCP microscaling, MI300/Blackwell) là mặt trận format
4-bit 2025–2026. Interview: "two-level scaling giải quyết vấn đề gì mà single-level E8M0 không?" →
non-power-of-two block scale không phí tới 2× dynamic range mỗi block.

---

## Bài 5.4 — FP8 E4M3 KV cache: per-channel-K vs per-token-V, clamp-before-cast (`quant/fp8_kv.py` · `quantize_fp8_e4m3`:65, `QuantizedKVCache`:131)

> **Câu hỏi first-principles:** KV cache là bộ nhớ runtime thống trị của decode (lớn tuyến tính theo
> seq len, đọc mỗi bước). Lưu FP8 thay BF16 halves nó — nhưng *trục scale nào* cho K, cho V, và vì
> sao E4M3 phải clamp **trước** khi cast?
> **Số đo (aha):** **FP8 SQNR 31.81 dB, E2E 24.45 dB** vs BF16-KV; per-channel-K **2.49×** tốt hơn
> per-token (rơi về 0.60× trên data không outlier — bám *cấu trúc* chứ không bám trục); INT4-KV
> **degrade rõ (19.86 dB)**; bytes **0.552×**.

**1. Feynman — bài toán bằng lời.** KV tensor `(B, H_kv, T, head_dim)`. K của Transformer có **outlier
channel** — vài `head_dim` channel biên độ lớn hơn hẳn. Nếu scale **per-token** (một scale mỗi row =
max theo channel), channel outlier nuốt gần hết code FP8, các channel nhỏ bị bước lượng tử khổng lồ.
**Per-channel** K (một scale mỗi `head_dim` column = max theo token) thích nghi từng channel → giành
lại precision. V không có cấu trúc outlier đó → **per-token** rẻ và đủ. Đánh đổi: per-channel-K chỉ
"free" nếu **calibrate scale MỘT LẦN từ prefill** rồi tái dùng — nếu tính lại mỗi 1-token-append thì
mỗi phần tử một scale (4 B/elem) → thua luôn cái lợi 1 B/elem của FP8.

**2. Cơ chế (derive từ đầu).** E4M3: 4 exp + 3 mantissa, bias 7, max ±448, **NaN-only ở đỉnh (không
±inf)** → giá trị > 448 **cast thành NaN chứ không saturate** → phải `clamp(±448)` **trước** `.to(
float8_e4m3fn)`. Scale `= amax/448` map đỉnh group lên đỉnh range. Byte accounting: FP8 = codes·1 +
scales·4; per-channel-K scale amortize ~0 B/elem qua chuỗi dài → FP8 (1 B) thật sự nửa BF16 (2 B).
Định luật granularity: `MSE(per-token-K)/MSE(per-channel-K) ≈ 2.49` khi K có outlier, ~0.60 khi
không — chứng minh nó bám **structure**, không bám axis.

**3. Trace code.** `quantize_fp8_e4m3` (:65): `amax` per-group (:72), `scale = amax/448` (:73),
**`clamp(±448).to(float8_e4m3fn)`** (:75 — clamp-before-cast). `QuantizedKVCache` (:131) duck-type
`KVCache` (`length`/`append`/`get`/`advance`) → dùng được trong `model.py` **không sửa model**.
`_quant_v` (:165) per-token axis=3; `_quant_k` (:175): nếu `k_per_channel`, **calibrate once** (:188–195)
`amax` reduce token axis=2, lưu `_k_scale[layer]`, tái dùng. `append` (:208) cộng `kv_bytes` từ storage
width thật. Driver `greedy_generate_with_cache` (:225) prefill + 1-token/step, trả logits để đo MSE.

**4. Cổng teach-back.** (a) Vì sao K per-channel còn V per-token — nói rõ K có gì mà V không? (b)
**Modify-and-predict:** bỏ dòng `clamp(±448)` (:75) trên data có vài giá trị `x/scale > 448`. Output
sẽ là gì (nhớ E4M3 NaN>448), và NaN đó lan tới đâu qua attention — kết quả decode ra sao?

**5. Frontier.** FP8-KV là cầu A5→A1: halve KV byte bill mà cả perf track (paged, MLA) đang đánh.
Production (vLLM/TensorRT-LLM) dùng đúng per-channel-K static-calibrated. Interview: "Vì sao E4M3
không saturate như INT8, và nó buộc bạn làm gì khác?" → NaN-only top → clamp-before-cast.

---

## Bài 5.5 — AWQ: activation-aware scaling, bảo vệ salient channel (`quant/awq.py` · `search_awq_scale`:122, `quantize_awq_int4`:154)

> **Câu hỏi first-principles:** Cùng bit-width (INT4) và cùng group size, làm sao *recover accuracy*
> mà không thêm một bit nào? Trả lời: chọn *chỗ tiêu* dynamic range của lưới — bảo vệ channel mà
> **activation** lớn, không phải channel mà **weight** lớn.
> **Số đo (aha):** **AWQ 5.47e-3 (20.90 dB) vs naive 9.03e-3 (18.72 dB) = 1.71× mean recovery** (min
> 1.65×, 5 seed); **held-out 1.69×** (không overfit calib); α=0 ≡ naive bit-exact; optimum α≈0.25;
> salient-col err ↓2.87×.

**1. Feynman — bài toán bằng lời.** Sai số weight-quant lan ra output `y_q − y = x @ (W_deq − W)ᵀ` —
đóng góp của input channel `j` bị **nhân với biên độ activation `x_j`**. Nên một nhúm channel bị
activation lớn (LLM "activation outlier") thống trị output error *dù weight của chúng bình thường*.
Round-to-nearest đối xử mọi channel như nhau → tiêu error budget sai chỗ. Đánh đổi cốt lõi: AWQ
không đổi bit, nó **dời độ mịn** tới đúng chỗ đau, bằng một đẳng thức đại số chính xác.

**2. Cơ chế (derive từ đầu).** Đẳng thức: với scale dương per-input-channel `s`, `(x/s) @ (W·s)ᵀ ==
x @ Wᵀ` — bất biến tuyệt đối. AWQ scale **cột weight salient LÊN** trước round (đẩy nó cao trên lưới
INT4 → error *tương đối* co lại, mà group step `absmax/7` gần như không đổi vì nhiều channel non-
salient vẫn giữ absmax), rồi fold `1/s` vào activation. Sau fold, noise cột đó chia cho `s` → share
output error (đã bị activation khuếch đại) giảm ~`s²`. Non-salient bị scale *xuống* chút bởi geo-mean
normalize, mang thêm chút error nhưng activation bé nên bỏ qua. Knob duy nhất: exponent `α` trong `s
= act_scale^α`, grid-search `α ∈ [0,1]` để min **calib output MSE** — no gradient, no retrain. **α=0
(s≡1) đúng bằng naive INT4** → search không bao giờ tệ hơn baseline.

**3. Trace code.** `act_scale` (:94) = `E[|x_j|]` per-input-channel (saliency signal). `_normalize_scale`
(:104) geo-mean-center `s/sqrt(max·min)` → pin geometric midpoint = 1. `search_awq_scale` (:122): ref
`y = x@Wᵀ` (:139), loop `α = i/grid` (:143–149): `s = normalize(a^α)` (:145) → `_awq_dequant_weight`
(:115: quant `W·s` group-INT4, dequant, fold `/s`) → đo MSE (:147), giữ min. `quantize_awq_int4` (:154)
quant `W·s` thành group-INT4 codes (:173–175). `awq_dequantize` (:178) `dequant(q)/chan_scale`.
Reuse `int4_group.quantize_groupwise_int4` — cùng lưới INT4 với naive, chỉ khác *đặt scale*.

**4. Cổng teach-back.** (a) Vì sao AWQ bảo vệ channel *activation*-lớn chứ không *weight*-lớn — dẫn
lại `y_q − y = x @ (W_deq − W)ᵀ`. (b) **Modify-and-predict:** đặt `α = 1` (scale mạnh hết cỡ). Salient
col được bảo vệ tối đa, nhưng *cả group step* bị kéo lên vì absmax của group giờ do salient col đặt —
MSE tổng tăng hay giảm so α≈0.25, và vì sao grid-search chọn optimum *bên trong* [0,1]?

**5. Frontier.** AWQ (arXiv:2306.00978) là PTQ W4A16 default 2024–2026, cặp với NVFP4/group-INT4 +
Marlin kernel. Interview: "Weight-quant, sao lại nhìn activation?" → error propagation nhân activation
magnitude; và "α làm gì?" → protect-vs-inflate tradeoff, α=0 là naive, có interior optimum.
