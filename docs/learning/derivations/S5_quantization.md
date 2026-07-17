# S5 — Quantization Numerics · First-Principles Derivations

> **Đây là gì.** Bản dẫn-xuất **từ nguyên lý gốc** (first principles) cho TOÀN BỘ pillar S5 —
> 5 micro-concept (Bài 5.1→5.5): INT8 → group-INT4 → NVFP4/MXFP4 → FP8-KV → AWQ. Mỗi mục có 6 phần:
> (1) **Câu hỏi** falsifiable, (2) **Sự thật nền tảng** (áp lực vật lý/toán học ép ra thiết kế),
> (3) **Dẫn xuất** kèm công thức, (4) **Neo code** `file·func·line`, (5) **Hình ảnh** (ASCII +
> shape/stride/dtype + numeric hand-trace), (6) **Số đo THẬT** (chạy trên chính repo này), rồi
> **Frontier / cổng phỏng vấn**.
>
> **Cách dùng để re-learn.** Đọc *Câu hỏi* → **tự trả lời cold** (che phần dưới) → mở code → đối
> chiếu **cái gap**. Đừng đọc trôi: quantization là môn mà "đọc hiểu" ≠ "dẫn được". Cuối doc có
> **checklist recall cold** + bảng số đo.
>
> **Cảnh báo neo.** Line number **trôi** theo commit. Cite ở đây pin theo HEAD `1e7dbc6`
> (2026-07-14), file `src/scratch_llm/quant/`. Nếu lệch, `grep` tên hàm — đừng tin số dòng cứng
> (luật repo: verify, don't trust). Roadmap-spine `roadmap/S5_quantization.md` pin commit `9e61d7a`
> nhưng line-set gần như trùng HEAD (code không drift nhiều).
>
> **Nguồn số đo.** Mọi số chạy lại được bằng: `bench/a5_fp8_kv.py`, `bench/a5_nvfp4_mxfp4.py`,
> `bench/a5_awq.py` (canonical, khớp `bench/RESULTS.md` §"Perf track (A5)"), cộng harness seed-pinned
> `scratchpad/measure_s5.py` cho 5.1/5.2. Đây là "DoD là một profile, không phải test xanh" (FOP-3).
>
> **Một cảnh báo trung thực đọc TRƯỚC (FOP-4).** Trong pillar này có một chỗ code **cố tình lệch**
> khỏi câu chuyện sách vở, và đó là bài học đắt nhất: **"per-channel-K law" (2.49×) được đo trên
> INT4, KHÔNG phải FP8** — vì E4M3 là float, mantissa của nó tự nuốt dynamic-range trong-token nên
> luật granularity yếu hẳn đi. Xem 5.4 §Số đo. Ta dạy *code thật*, và gọi tên cái gap.

---

## Bức tranh lớn — quantization là một hàm gì, và áp lực nào ép ra nó?

Quantization = hàm `Q: (tensor float) → (codes ít-bit + scale) → (tensor float xấp xỉ)`. Nó KHÔNG
phải "nén rồi chịu sai" — nó là một **biến thiết kế chọn RIÊNG cho từng lớp tensor** (weight,
activation, KV). Toàn bộ S5 leo một cái thang duy nhất — **granularity của scale** — bị ép bởi một
định luật: *càng ít bit → càng cần scale mịn hơn.*

```
        ÁP LỰC = MEMORY (fit weight/KV/activation vào ÍT byte hơn)  ⟂  NUMERICS (giữ SQNR trên sàn)
        ────────────────────────────────────────────────────────────────────────────────────────
  float x  ──►  chia scale s  ──►  round lên LƯỚI ít-bit  ──►  codes (int/fp4/fp8)  ──►  ·s  ──►  x̂
                    ▲                       ▲                        │
              [đặt scale Ở ĐÂU]      [lưới gì: uniform INT       [đóng gói bit: 2-nibble/byte,
               = granularity]         / non-uniform FP]           E2M1 packing → bit-exact]

  THANG GRANULARITY (mỗi nấc = phản ứng với "ít bit hơn"):
  per-tensor ─► per-channel ─► group(g=128) ─► two-level block(16) ─► activation-aware(AWQ)
    5.1 INT8      5.1 INT8         5.2 INT4        5.3 NVFP4              5.5 (cùng bit, dời scale)
    8 bit        8 bit            4 bit           4 bit                  4 bit — recover BẰNG scale
      │            │                │               │                      │
   1 scale/     1 scale/         1 scale/        2 scale (global fp32     scale theo ACTIVATION
   tensor       out-channel      128 elem        + E4M3 mỗi 16 elem)      không theo weight

  TRỤC THỨ HAI (song song, độc lập granularity) — CHỖ ĐẶT TENSOR trong hệ thống:
    weight (offline, one-shot)  →  INT8/INT4/NVFP4/AWQ           [5.1,5.2,5.3,5.5]
    KV     (runtime, decode)    →  FP8-E4M3, per-channel-K       [5.4]  ← cầu A5→A1 serving
```

**Ba câu niệm chú xuyên suốt (từ spine):**
1. **Oracle luôn là SQNR/MSE, KHÔNG BAO GIỜ `allclose`.** dB nói *bao nhiêu bit tín hiệu còn sống*;
   `allclose` chỉ nói pass/fail mù. FP4/FP8 KHÔNG BAO GIỜ pass `allclose` — nhưng vẫn đúng.
2. **Một giá trị dB rơi DƯỚI sàn đạt được là BUG** (sai scale / thiếu clamp / sai rounding), không
   phải "mất mát chấp nhận được". Đây là kỷ luật debug: viết số trước, rơi dưới = truy bug.
3. **Càng ít bit → granularity càng mịn.** Đó là toàn bộ cốt truyện của cái thang trên.

Một sự thật vật lý neo tất cả: **quantization = cộng noise trắng biên độ `s/2`, phương sai `s²/12`.**
Từ đó suy ra *mọi thứ* — luật 6 dB/bit, vì sao per-channel thắng, vì sao two-level thắng single-level.

---

## 5.1 · INT8 — sàn SQNR, luật 6 dB/bit, và scale factor-out khỏi GEMM

**Câu hỏi.** (a) Vì sao quantize = *cộng noise trắng phương sai `s²/12`*, và vì sao thêm **1 bit**
mua đúng **6.02 dB**? (b) Vì sao INT8 sym trên Gaussian rơi ~38 dB chứ KHÔNG phải ~44 dB "lý tưởng"?
(c) Vì sao per-channel **weight** scale *biến mất* khỏi vòng trong của matmul — nhưng chỉ khi quantize
theo **output** channel, không phải input channel?

**Sự thật nền tảng.** Round về lưới đều bước `s`: sai số `e = x − round(x/s)·s` phân bố **uniform trên
`[−s/2, s/2]`** (giả định x mịn so với s). Uniform trên khoảng rộng `s` có phương sai `s²/12` (tích
phân `∫_{−s/2}^{s/2} e²/s de = s²/12`). Đây là **noise cộng, độc lập tín hiệu** — mô hình additive
quantization noise. Mọi hệ quả chảy ra từ một dòng này.

**Dẫn xuất (a) — luật 6.02 dB/bit.** SQNR = công suất tín hiệu / công suất noise:
```
SQNR = P_signal / P_noise = P_signal / (s²/12)
```
Thêm 1 bit → số mức ×2 → bước `s` **giảm một nửa** → `s²/12` giảm **4 lần** → SQNR ×4. Trên dB:
```
ΔSQNR_dB = 10·log10(4) = 6.0206 dB   ⟹   SQNR_dB ≈ 6.02·b + c
```
Hằng `c` phụ thuộc *cách chọn full-scale so với tín hiệu*. Textbook `+1.76 dB` giả định full-scale
sinusoid. **Đây là luật falsifiable:** nhân đôi scale (= vứt 1 bit resolution) phải làm SQNR rớt đúng
−6.02 dB. Đo được (§Số đo): `38.39 → 32.37 dB`, delta **−6.02** khít.

**Dẫn xuất (b) — vì sao ~38 chứ không ~44 (crest-factor penalty).** INT8 sym dùng mức `±127`,
`s = amax/127`. Cho tensor Gaussian `N(0,σ²)`, `amax = k·σ` với `k` = **crest factor** (max/std). Khi
đó `P_signal = σ²`, `P_noise = s²/12 = k²σ²/(12·127²)`:
```
SQNR_dB = 10·log10( σ² · 12·127² / (k²σ²) )
        = 10·log10(12) + 20·log10(127) − 20·log10(k)
        ≈ 10.79 + 42.08 − 20·log10(k)
        ≈ 52.87 − 20·log10(crest)
```
Với 4096²=16.7M mẫu, `E[max] ≈ √(2 ln N)·σ ≈ 5.3σ` ⟹ `20·log10(5.3) ≈ 14.5 dB` phạt ⟹ **≈ 38.4 dB**.
Khớp đo 38.39 dB. **Cái mất 6 dB so với "44" chính là tiền trả cho việc một outlier kéo full-scale.**
Đây là hạt giống của toàn bộ pillar: *muốn dB cao hơn → co crest xuống → per-channel/group/two-level.*

**Dẫn xuất (c) — asym & per-channel & factor-out.**
- **Sym phí nửa range trên data lệch.** Sym map `[−amax, amax] → [−127, 127]`, zero-point ghim tại 0.
  Post-GELU/ReLU toàn dương → nửa mã âm **chết** → `s` gấp đôi cần thiết → mất ~6 dB. Asym affine fit
  `[min, max] → [0, 255]` với integer zero-point `z = round(−min/s)`, giành lại range: `+6.60 dB` đo.
- **Per-channel co crest cục bộ.** Mỗi output row một scale = mỗi bước bám range thật của row đó,
  strictly beats per-tensor: `+6.35 dB` đo.
- **Factor-out (điểm load-bearing).** GEMM co theo `k`: `Y[m,n] = Σ_k X[m,k]·W[n,k]`. Với per-token
  activation scale `s_X[m]` và per-**output**-channel weight scale `s_W[n]` — **cả hai KHÔNG phụ thuộc
  `k`**:
  ```
  Y[m,n] = Σ_k (q_X[m,k]·s_X[m])·(q_W[n,k]·s_W[n])
         = s_X[m]·s_W[n] · Σ_k q_X[m,k]·q_W[n,k]        ← INT8×INT8 → INT32 accumulate
                            └──────── vòng trong THUẦN INTEGER ────────┘
  ```
  Vì `s_X, s_W` hằng theo `k`, chúng **kéo thẳng ra ngoài Σ** → phần cứng chạy 1 integer matmul vào
  INT32 accumulator rồi dequant **một lần** bằng outer-product `s_X[m]·s_W[n]`. **Nếu weight scale
  biến thiên DỌC `k`** (per-*input*-channel) → *không hoist được* → mất luôn integer GEMM. Đó là lý do
  weight quantize per-**output**-channel, không per-input.

**Neo code** (`src/scratch_llm/quant/int8.py`):
```python
def quantize_symmetric(x, axis=None, eps=1e-12):            # :101
    amax = x.detach().abs().amax(dim=reduce_dims, keepdim=True)  # :121  per-channel nếu axis đặt
    s = (amax / _SYM_QMAX).clamp_min(eps)                   # :122  s = amax/127
    q = torch.round(x / s).clamp(-127, 127).to(torch.int8) # :123  round + clamp + int8

def quantize_affine(x, axis=None, eps=1e-12):              # :141  asym unsigned [0,255]
    xmin = torch.minimum(xmin, torch.zeros_like(xmin))     # :166  LUÔN gộp 0 vào range
    s = ((xmax - xmin) / _AFF_QMAX).clamp_min(eps)         # :168  s = (max−min)/255
    z = torch.round(-xmin / s).clamp(0, 255)               # :169  integer zero-point
    q = torch.round(x / s + z).clamp(0, 255).to(uint8)     # :170

def w8a8_linear(x, w):                                     # :188  PAYOFF
    q_x, s_x = quantize_symmetric(x2d, axis=0)             # :204  per-token  s_x:[M,1]
    q_w, s_w = quantize_symmetric(w, axis=0)               # :206  per-out-ch s_w:[N,1]
    acc = q_x.to(int32) @ q_w.to(int32).t()                # :209  INT32 accumulate (scale ĐÃ hoist)
    y2d = acc.to(float32) * (s_x * s_w.reshape(1, -1))     # :212  dequant MỘT LẦN (rank-1 outer)
```
Bug-review khi trace: `quantize_symmetric` cố ý bỏ mức `−128` (dùng `±127`) để `0.0 → code 0` chính
xác và `±` đối xứng — mất 1 mức nhưng đúng về semantics. `clamp_min(eps)` chặn chia-0 trên tensor
toàn-0. Cả hai đúng chuẩn production.

**Hình ảnh — factor-out (shape/dtype journey):**
```
x:(M,K) fp32 ─quantize_symmetric(axis=0)─► q_x:(M,K) int8 , s_x:(M,1) fp32
w:(N,K) fp32 ─quantize_symmetric(axis=0)─► q_w:(N,K) int8 , s_w:(N,1) fp32
                                              │  q_x @ q_w.t()   (int32 @ int32)
                                              ▼
                                           acc:(M,N) int32   ◄── KHÔNG có float ở vòng trong
                                              │  * (s_x * s_w.reshape(1,-1))   → (M,1)*(1,N) = (M,N)
                                              ▼
                                           y:(M,N) fp32       ◄── dequant 1 lần, rank-1 outer product

hand-trace book example [1.2,−0.8,2.5,−1.7]:  amax=2.5 → s=2.5/127=0.019685
  round(1.2/s)=round(60.96)=61 ; round(−0.8/s)=−41 ; round(2.5/s)=127 ; round(−1.7/s)=−86
  codes = [61, −41, 127, −86]   (đo khớp)   dequant 127·s = 2.500 (đỉnh map đúng lên 127)
```

**Số đo THẬT** (`measure_s5.py`, seed 0–3; canonical `bench/RESULTS.md` trong ngoặc):
```
[5.1a] INT8 sym per-tensor Gaussian SQNR = 38.39 dB   (RESULTS: 40.50 dB, cùng lớp ~40, dưới 44 lý tưởng)
[5.1a] LUẬT 6dB/bit: nhân đôi scale (−1 bit) 38.39 → 32.37 dB, delta = −6.02  (lý thuyết −6.02) ✓
[5.1a] book example scale=0.019685 codes=[61, −41, 127, −86]  (khớp RESULTS)
[5.1b] skewed(all-positive): sym 37.11  →  asym 43.71 dB   (+6.60)   ← asym giành nửa range
[5.1c] per-tensor 36.29  →  per-channel 42.64 dB   (+6.35)          ← co crest cục bộ
[5.1d] W8A8 int-GEMM rel Frobenius = 1.02e-2 ; accumulator dtype = torch.int32 ; s_x:(32,1) s_w:(128,1)
```

**Frontier / cổng.** W8A8 (SmoothQuant / TensorRT-LLM) = recipe INT8 chuẩn: per-token dynamic
activation × per-channel static weight — chính vì scale hoist. **Interview gate:** "Vì sao weight
quantize theo *output* channel chứ không input?" → factor-out khỏi contraction axis `k` (mất integer
GEMM nếu sai). Trait FOP: **first-principles** (dẫn 6 dB/bit từ `s²/12`, không "vì paper"). Bucket:
table-stakes numerics, nhưng là nền cho inference-scarce.

---

## 5.2 · Group-INT4 — 2-nibble packing bit-exact + group scaling

**Câu hỏi.** Ở 4 bit, hai bài toán **tách rời** phải đúng đồng thời: (1) *lưu* — nhét 2 code 4-bit vào
1 byte mà round-trip **bit-exact**, kể cả two's-complement âm và độ dài lẻ; (2) *numeric* — chọn scale
mịn tới đâu để 16 mức `{−8..7}` còn dùng được?

**Sự thật nền tảng.** INT4 chỉ có **16 mức**. Hai áp lực khác bản chất:
- *Storage là bài toán bit-exact TUYỆT ĐỐI* — lệch 1 bit → mọi số sau đó rác. Không có "gần đúng".
- *Numeric là bài toán granularity* — 4 bit quá ít cho 1 scale phủ cả tensor (crest phạt nặng, xem
  5.1). Chia weight thành **group 128 elem liền nhau dọc contraction axis**, mỗi group 1 scale
  `absmax/7` → lưới bám dynamic range cục bộ. Đây là điều làm 4-bit weight *dùng được*.

**Dẫn xuất — packing (two's-complement nibble).** Muốn lưu `-8..7` trong 4 bit: lấy **low 4 bit** của
biểu diễn two's-complement, tức `q & 0xF`. Phép này map `0..7 → 0..7` và `−8..−1 → 8..15` (vì
`−1 = ...11111111`, low nibble `1111 = 0xF = 15`). Pack: `byte = (hi << 4) | lo`. Unpack **sign-extend**:
nibble `≥ 8 → trừ 16` (khôi phục dấu): `0xF=15 → 15−16 = −1`, `0x8=8 → 8−16 = −8`. Độ lẻ: append 1
nibble 0, drop khi biết `n` gốc. **Nếu quên sign-extend:** `0xF` decode thành `+15` thay vì `−1` →
sai hoàn toàn. Đó là hợp đồng bit-level test 500-case găm.

**Dẫn xuất — group quant & sàn SQNR.** Reshape `[out, in] → [out, n_groups, g]`, `absmax` theo trục
cuối, `s = absmax/7`, `round(w/s).clamp(−8,7)`. Sàn mid-tread uniform `≈ 6.02·4 = 24 dB`; trên Gaussian
non-clip nó rơi thấp hơn (18.6 dB) vì crest-penalty (5.1) — nhưng **group NÂNG so per-tensor** vì mỗi
group có crest nhỏ hơn (ít mẫu hơn → max cục bộ nhỏ hơn). Đánh đổi: thêm scale (overhead) đổi độ mịn.

**Neo code** (`src/scratch_llm/quant/int4_group.py`):
```python
def pack_int4(q):                                  # :62   q 1-D trong [−8,7]
    nibbles = (q.to(int64) & 0xF)                  # :76   −8..−1 → 8..15 (two's-comp low nibble)
    if n % 2 == 1: nibbles = cat([nibbles, 0])     # :77-78 pad lẻ 1 nibble 0
    packed = (hi << 4) | lo                         # :81   hi-nibble TRƯỚC
def unpack_int4(packed, n):                        # :85
    out = stack([hi, lo], 1).reshape(-1)[:n]       # :98-99 interleave hi,lo → drop pad
    out = where(out >= 8, out - 16, out)           # :101  SIGN-EXTEND (quên = 0xF→15 sai)
def quantize_groupwise_int4(w, group_size=128):    # :108
    wf = w.reshape(out, n_groups, group_size)      # :124  chia group dọc input axis
    absmax = wf.abs().amax(dim=-1, keepdim=True)    # :125  1 absmax / group
    scales = (absmax / 7).clamp_min(tiny)          # :127  s = absmax/7
    q = round(wf/scales).clamp_(-8, 7).to(int8)    # :128
```
Bug-review: `scales` để dtype `int8` container cho `q` (giá trị `[−8,7]` vừa int8) nhưng **chưa pack**
— `quantize_groupwise_int4` trả `q` *chưa đóng gói* (mỗi elem 1 byte int8); packing là bước riêng
`pack_int4`. Tách concern sạch: numeric ⟂ storage. `clamp_min(finfo.tiny)` chặn all-zero group.

**Hình ảnh — 1 byte chứa 2 nibble (bit layout THẬT):**
```
q = [−8, −1]                       −8 & 0xF = 1000₂ = 8      −1 & 0xF = 1111₂ = 15
byte = (8 << 4) | 15 = 1000_1111₂ = 0x8F = 143   ◄── ĐO: pack([−8,−1]) = [143]
        └hi─┘ └lo─┘
unpack:  hi = 143>>4 & 0xF = 8   → 8≥8  → 8−16 = −8   ✓
         lo = 143    & 0xF = 15  → 15≥8 → 15−16 = −1  ✓

shape journey (weight [256, 512], g=128):
  w:(256,512) fp32 ─reshape─► (256, 4, 128) ─absmax dim=-1─► scales:(256,4,1) fp32
                            ─round/clamp──► q:(256,512) int8 ∈[−8,7]
  storage: 256·512 codes → pack → 256·256 bytes (2/byte) + 256·4 fp32 scales
```

**Số đo THẬT** (`measure_s5.py` seed 4–5):
```
[5.2a] 500-case fuzz bit-exact = True ; adversarial [−8,−1,7,0,−8,7] round-trip = [−8,−1,7,0,−8,7]
[5.2a] pack([−8,−1]) = [143] (0x8F) ; unpack → [−8,−1]        ← two's-comp + sign-extend đúng
[5.2b] group(g=128) SQNR 18.67 dB  >  per-row(g=512) 17.52 dB  (+1.15)   (RESULTS: 18.64 / analytic 18.60)
[5.2b] số scale fp32:  g=128 → 1024   g=32 → 4096   g=512 → 256   ← mịn hơn = nhiều scale hơn
```

**Frontier / cổng.** W4A16 group-128 = default suy luận LLM 2024–2026 (GPTQ/AWQ/Marlin). Marlin kernel
unpack nibble **inline** trong shared memory rồi dequant tại chỗ — chính hợp đồng bit-order này. **Gate:**
"Vì sao pack hi-nibble trước lại quan trọng?" → phải khớp thứ tự unpack; "0xF unpack ra gì nếu quên
sign-extend?" → +15 (sai) vs −1 (đúng). **Modify-and-predict:** g 128→32, SQNR ↑ nhưng số scale ×4
(1024→4096) — độ mịn mua bằng byte scale. Trait: **spec-with-falsifiers** (bit-exact contract test).

---

## 5.3 · NVFP4 vs MXFP4 — E2M1/E4M3/E8M0 codec, two-level scaling

**Câu hỏi.** Khi element format **giống hệt nhau** (cả hai đều FP4 = E2M1), thì cái gì quyết định format
nào chính xác hơn? Trả lời: **codec của block-scale** và **kích thước block** — không gì khác.

**Sự thật nền tảng.** FP4 = E2M1 (1 sign, 2 exp, 1 mantissa) chỉ biểu diễn **8 độ lớn**
`{0, .5, 1, 1.5, 2, 3, 4, 6}` — quá thô để phủ range của một tensor. Cả hai format đắp thêm 1 **scale
theo block nhỏ** rồi mới round element lên lưới E2M1. Khác biệt **CHỈ** ở hai chỗ:
1. **Scale codec:** MXFP4 lưu block-scale là **E8M0 = số mũ thuần** (power-of-two, bias 127) → làm tròn
   scale lý tưởng về **octave gần nhất**, lệch tới ~2×. NVFP4 dùng **hai tầng**: `s_global` fp32
   per-tensor × `s_block` **E4M3** (fp8, non-power-of-two) per-16-block → bám scale lý tưởng tới ~1
   phần 16 (3-bit mantissa relative precision).
2. **Block size:** NVFP4 `k=16` < MXFP4 `k=32` → mỗi block ít elem hơn → spread nội-block nhỏ hơn → mịn.

**Dẫn xuất — recipe NVFP4 (TensorRT-Model-Optimizer).** Vì sao `s_global = amax/(448·6)` và `/6`?
```
s_global   = amax / (E4M3_MAX · FP4_MAX) = amax / (448·6)
s_block    = quantize_E4M3( block_amax / 6 / s_global )
codes      = round_E2M1( x / (s_global · s_block) )
x̂          = s_global · s_block · codes
```
Chốt quan trọng — **vì sao `448·6` fold vào global đảm bảo `s_block` luôn E4M3-biểu-diễn-được:**
```
block_amax / 6 / s_global = (block_amax/6) · (448·6)/amax = block_amax · 448 / amax  ≤  448
```
vì `block_amax ≤ amax`. Nên `s_block ∈ (0, 448]` → luôn nằm trong range E4M3. `/6` chia cho `FP4_MAX`
để block max map lên đỉnh lưới E2M1 (giá trị 6). Đây là hằng số recipe, không phải magic.

**Dẫn xuất — MXFP4 no-overflow (ceil, KHÔNG round-nearest).**
```
s_block = 2^ceil(log2(block_amax / 6))     ← mọi |x|/s_block ≤ 6, KHÔNG element nào clip
```
Chọn **ceil** (không round-nearest) là để MXFP4 thành **baseline MẠNH, fair**: round-nearest E8M0 đôi
khi round scale *xuống* → clip lẻ 1 outlier → phạt oan block mịn. No-overflow ceil = bản trung thực.
Giá của ceil: block max rơi bất kỳ đâu trong `(3, 6]` tùy `amax/6` nằm đâu trong octave → **phí tới 2×
dynamic range**. Chính cái coarseness đó là thứ E4M3-block-scale của NVFP4 tránh được.

**Vì sao NVFP4 thắng — hai cơ chế, tách được bằng đo.** Nếu MXFP4 *ever* tie/thắng thì đó là **bug đặt
scale**, không phải "loss chấp nhận được". Đo cô lập (§Số đo): cơ chế-1 (E4M3 vs E8M0, cùng k=32) cho
+1.12 dB; cơ chế-2 (k=16 vs k=32, cùng codec) cho +0.57 dB (E4M3). Cộng lại ⟹ tổng gap MSE **1.48×**.

**Neo code** (`src/scratch_llm/quant/nvfp4_mxfp4.py`):
```python
FP4_MAGNITUDES = (0.0, .5, 1., 1.5, 2., 3., 4., 6.)     # :65   8 độ lớn E2M1, max 6
def quantize_e4m3(v):                                    # :135  block-scale codec của NVFP4
    exp  = floor(log2(v)).clamp(-6, 8)                   # :142-143 binade
    step = exp2(exp - 3)                                 # :144  ULP = 2^(exp−mantissa_bits)
    q    = round(v/step) * step                          # :145  round-half-even trong binade
def _e8m0_no_overflow_scale(amax):                       # :162  block-scale codec của MXFP4
    exp = ceil(log2(amax / 6))                           # :172  CEIL → no clip, phí ≤2× range
    return exp2(exp)                                      # :174  power-of-two
def nvfp4_quantize(x, block=16):                         # :212
    s_global = (amax / (448·6)).clamp_min(tiny)          # :218  đảm bảo s_block ≤ 448
    s_block  = quantize_e4m3(block_amax / 6 / s_global)  # :220  E4M3 non-pow2
    codes    = e2m1_round_to_code(xb / scale)            # :222  scale = s_global·s_block
def mxfp4_quantize(x, block=32):                         # :257
    s_block = _e8m0_no_overflow_scale(block_amax)        # :263  E8M0 pow2, block 32
def quant_linear_nvfp4(x, weight, block=16):             # :296  block-scaled GEMM
    w_hat = e2m1_decode(codes) * scale                   # :322  scale gấp sẵn vào w_hat mỗi block
    for kb in range(n_kb): y += x_blk @ w_hat[:,kb,:].t() # :326-328  accumulate BLOCK-BY-BLOCK dọc K
```
Bug-review: `e2m1_round_to_code` (:85) dùng `argmin` trên 8 anchor, ties→lower index (measure-zero
trên reals — vô hại). `quant_linear_nvfp4` gấp scale vào `w_hat` *trước* rồi cộng dồn — đúng ngữ nghĩa
block-scaled tensor-core (scale sống trong accumulation). Element `x` giữ fp32 (weight-only quant).

**Hình ảnh — two-level vs single-level (một block):**
```
NVFP4 (k=16, TWO-level):                         MXFP4 (k=32, SINGLE-level):
  x_block (16 elem) ─amax─► block_amax             x_block (32 elem) ─amax─► block_amax
    s_global (fp32, cả tensor)                       s_block = 2^ceil(log2(amax/6))   ← chỉ octave
    s_block  = E4M3(amax/6/s_global)  ← non-pow2       (phí tới 2× range mỗi block)
    codes    = E2M1(x/(s_global·s_block))            codes = E2M1(x/s_block)
  scale bám ideal tới ~6% (3-bit mantissa)         scale lệch tới 2× (chỉ lũy thừa 2)

ideal block-scale = 5.0:   E8M0 → 4.0 (round-nearest, 20% err)   |   E4M3 → 5.0 (0% err)  ◄── ĐO
  no-overflow (ceil): amax/6=5.0 → 2^ceil(log2 5)=2^3=8.0 → max |x|/s = 5/... = 3.75 ≤ 6 (no clip)

shape journey (nvfp4_quantize, x:(4096,32)):
  x ─_blockify(16)─► xb:(8192, 16) fp32 ─amax dim=1─► block_amax:(8192,) 
    s_global: scalar fp32 ; s_block:(8192,) E4M3 ; codes:(8192,16) uint8 (bit3 sign|bits2-0 mag)
```

**Số đo THẬT** (`bench/a5_nvfp4_mxfp4.py`, canonical, khớp RESULTS):
```
NVFP4 : MSE 9.048e-03   SQNR 20.43 dB          MXFP4 : MSE 1.336e-02   SQNR 18.74 dB
MSE(MXFP4)/MSE(NVFP4) = 1.476×   (>1 ⟹ NVFP4 thắng, load-bearing)
  cơ chế-1 scale format @ k=32 (E4M3 vs E8M0): 19.86 vs 18.74 dB  → +1.12 dB
  cơ chế-2 block size (k=16 vs k=32), per codec: E4M3 20.43>19.86 (+0.57) ; E8M0 18.95>18.74 (+0.20)
ideal 5.0 → E8M0 4.0 (20% err)  vs  E4M3 5.0 (0% err)                       ← micro-demo measure_s5.py
FP4 nibble pack/unpack bit-exact = True
block-scaled GEMM: 4.13e-7 vs fp32-dequant (accumulate EXACT) ; 0.286% vs bf16-same-W (<1% oracle) ;
                   9.53% vs bf16-original-W (= honest 4-bit cost, đúng ~20 dB SQNR)
```

**Frontier / cổng.** NVFP4 (Blackwell) vs MXFP4 (OCP microscaling, MI300/Blackwell) là mặt trận
format 4-bit 2025–2026. **Gate:** "two-level scaling giải quyết gì mà single-level E8M0 không?" →
non-power-of-two block scale không phí ≤2× dynamic range mỗi block. **Modify-and-predict:** MXFP4 block
32→16 (bằng NVFP4) nhưng giữ E8M0 → gap thu hẹp phần cơ-chế-2, nhưng cơ-chế-1 (E4M3 codec) VẪN thắng.
Trait: **claims-honesty** (verifier xác nhận baseline MXFP4 là bản ceil *mạnh*, không rigged). Bucket:
**kernels/inference = scarce differentiator 2026.**

---

## 5.4 · FP8 E4M3 KV — per-channel-K vs per-token-V, clamp-before-cast

**Câu hỏi.** KV cache là **bộ nhớ runtime thống trị** của decode (lớn tuyến tính theo seq len, đọc mỗi
bước). Lưu FP8 thay BF16 halves nó — nhưng (a) *trục scale nào* cho K, cho V, (b) vì sao E4M3 phải
**clamp trước khi cast**?

**Sự thật nền tảng.** KV tensor `(B, H_kv, T, head_dim)`. K của Transformer có **outlier channel** — vài
`head_dim` channel biên độ lớn hơn hẳn phần còn lại, ổn định qua mọi token. V thì không có cấu trúc đó.
Và E4M3 là **float NaN-only ở đỉnh**: `>448 → NaN chứ KHÔNG saturate` như INT.

**Dẫn xuất — vì sao K per-channel, V per-token.** Nhìn qua lăng kính *fixed-point* (INT):
- **per-token** (một scale/row = max theo channel): channel outlier có mặt ở *mọi* token → set scale
  cho mọi token → bước lượng tử khổng lồ nuốt các channel nhỏ.
- **per-channel** K (một scale/`head_dim` column = max theo token): channel outlier có scale riêng lớn,
  channel nhỏ có scale riêng nhỏ → mỗi channel giữ precision thật.

Định luật granularity: `MSE(per-token-K)/MSE(per-channel-K) ≈ 2.49` khi K có outlier, `~0.60–0.95` khi
KHÔNG — chứng minh nó bám **structure**, không bám axis.

**Dẫn xuất — E4M3 & clamp-before-cast.** E4M3: 4 exp + 3 mantissa, bias 7. Max finite = `S.1111.110`
`= 1.110₂ · 2^(15−7) = 1.75 · 256 = 448`; `S.1111.111` reserved **NaN** (không ±inf). Nên `|x| > 448`
cast thẳng → **NaN** (không clip xuống 448) → phải `clamp(±448)` **TRƯỚC** `.to(float8_e4m3fn)`. Scale
`= amax/448` map đỉnh group lên đỉnh range để dùng hết 3-bit mantissa.

**Byte accounting (vì sao FP8 THẬT SỰ nửa BF16).** FP8 = codes·1 B + scales·4 B. Bí quyết: per-channel-K
scale **calibrate MỘT LẦN từ prefill** rồi tái dùng cho mọi 1-token append → amortize ~0 B/elem qua
chuỗi dài → FP8 (1 B) thật sự nửa BF16 (2 B). **Nếu tính lại scale mỗi append** → mỗi elem 1 scale
(4 B/elem) → thua luôn cái lợi. Đo E2E: FP8 **0.552×** bytes, INT4 0.302×.

**Neo code** (`src/scratch_llm/quant/fp8_kv.py`):
```python
def quantize_fp8_e4m3(x, axis):                          # :65
    amax  = x.abs().amax(dim=axis, keepdim=True).clamp_min(eps)  # :72
    scale = amax / 448.0                                 # :73
    codes = (x/scale).clamp(-448, 448).to(float8_e4m3fn) # :75  CLAMP TRƯỚC CAST (E4M3 NaN>448)
class QuantizedKVCache:                                  # :131  duck-type model.KVCache → no model edit
    self._k_scale = [None]*n_layers                      # :157  per-channel-K scale, calibrate-once
    def _quant_v(self, x): quantize_fp8_e4m3(x, 3)       # :165  V: per-token (reduce head_dim=3)
    def _quant_k(self, layer, x):                        # :175  K:
        if scale is None:                                # :191  CALIBRATE trên prefill block
            amax = x.abs().amax(dim=2, keepdim=True)      # :192  reduce TOKEN axis → per-channel
            self._k_scale[layer] = amax / 448.0          # :194-195  lưu, tái dùng
        codes = (x/scale).clamp(-448,448).to(f8_e4m3fn)  # :198
```
Bug-review: `QuantizedKVCache` **KHÔNG** subclass `SlotKVCache` — cố ý, để đi đường decode phẳng
(`length`/`append`/`get`/`advance`), tránh route ragged-batch. Đây là lý do nó "wired vào A1 decoder
mà không sửa model.py". `kv_bytes` (:214) cộng từ storage width THẬT (codes + scale one-time) → byte
đo, không phán.

**Hình ảnh — trục scale (K vs V) trên tensor (B,H,T,D):**
```
K: (1, 4, 256, 32) — outlier ở vài COLUMN (head_dim), ổn định qua token
          head_dim →                          per-channel-K:  amax reduce dim=2 (token)
  token   [ 12.0  0.3  0.2  9.8 ...]            → scale shape (1,4,1,32) — 1 scale / COLUMN
    ↓     [ 11.7  0.4  0.1  9.9 ...]            → outlier col scale riêng, col nhỏ scale riêng ✓
          [ 12.3  0.2  0.3  9.5 ...]
          └outlier col┘  └outlier col┘         per-token-K:  amax reduce dim=3 (head_dim)
                                                → scale shape (1,4,256,1) — 1 scale / ROW
                                                → mọi row bị outlier col kéo → col nhỏ bị nghiền ✗
V: không outlier col → per-token (dim=3) rẻ & đủ.

clamp-before-cast:  500.0 ──.to(float8_e4m3fn)──► NaN          (E4M3 >448 = NaN, KHÔNG saturate)
                    500.0 ──clamp(448)──.to(...)─► 448.0       (an toàn)   ◄── ĐO cả hai
```

**Số đo THẬT — VÀ CÁI GAP TRUNG THỰC (FOP-4).**
Canonical `bench/a5_fp8_kv.py` (khớp RESULTS):
```
tensor SQNR   FP8(E4M3) 31.81 dB   >   INT4 18.61 dB
INT4 per-channel-K law   MSE(per-token)/MSE(per-channel) = 2.49×      ◄── ĐO TRÊN INT4, không FP8
E2E logit vs BF16-KV   FP8: SQNR 24.45 dB MSE 2.39e-3  |  INT4: SQNR 19.86 dB MSE 6.86e-3 (2.88× tệ hơn)
KV bytes   BF16=36864   FP8=20352 (0.552×)   INT4=11136 (0.302×)
```
**Cái gap** (harness `measure_s5.py`, cùng dữ liệu outlier nhưng đo trên **FP8**):
```
[5.4b] K WITH outliers, FP8:  per-token/per-channel MSE ratio = 0.29×   ← per-channel THUA! (bench seed-1 K, FP8 path)
[5.4b] K WITHOUT outliers, FP8:                        ratio = 0.83×   ← gần 1
[5.4a] 500.0 → f8 WITHOUT clamp = nan ;  clamp(448) → 448.0
[5.4c] FP8 SQNR 31.75 dB  >  INT4-KV 18.35 dB ;  bytes 0.516×
```
**Vì sao gap:** `bench` cố ý đo per-channel-K law bằng `quantize_int4` (:48–49), **không** FP8. Lý do
first-principles: **E4M3 là FLOAT** — mantissa cho ~3–4 bit *relative* precision *bất kể magnitude
trong-token*, nên nó tự nuốt dynamic-range mà per-token gây ra → luật granularity yếu hẳn. Với **INT4
(fixed-point)** outlier thật sự "steal codes" → per-channel thắng 2.49×. **Bài học:** luật per-channel
là hiện tượng của lưới **fixed-point**; với float format hãy đo lại chứ đừng giả định. Production vẫn
chọn per-channel-K FP8 vì (a) không bao giờ hại E2E (24.45 dB) và (b) calibrate-once amortize scale byte
→ giữ 0.5× bill. Ta dạy *code thật*, và gọi tên gap này.

**Frontier / cổng.** FP8-KV là cầu A5→A1: halve KV byte bill mà cả perf track (paged, MLA) đang đánh.
vLLM/TensorRT-LLM dùng per-channel-K static-calibrated. **Gate:** "Vì sao E4M3 không saturate như INT8,
buộc bạn làm gì?" → NaN-only top → clamp-before-cast. **Gate sâu:** "per-channel-K thắng bao nhiêu?" →
"tùy lưới: INT ~2.5×, FP8 gần như hòa vì mantissa" (câu trả lời này phân biệt người đọc code với người
đọc slide). Trait: **claims-honesty + measured>implied.**

---

## 5.5 · AWQ — activation-aware scaling, bảo vệ salient channel

**Câu hỏi.** Cùng bit-width (INT4) và cùng group size, làm sao *recover accuracy* mà **không thêm một
bit nào**? Trả lời: chọn *chỗ tiêu* dynamic-range của lưới — bảo vệ channel mà **activation** lớn,
KHÔNG phải channel mà **weight** lớn.

**Sự thật nền tảng.** Sai số weight-quant lan ra output:
```
y_q − y = x @ (Ŵ − W)ᵀ      ⟹  đóng góp của input channel j  =  x_j · (Ŵ − W)_{:,j}
```
Nó bị **nhân với biên độ activation `x_j`**. Nên một nhúm channel bị activation lớn (LLM "activation
outlier") **thống trị** output error *dù weight của chúng bình thường*. Round-to-nearest đối xử mọi
channel như nhau → tiêu error budget **sai chỗ**.

**Dẫn xuất — đẳng thức bất biến + vì sao scale UP.** Với scale dương per-input-channel `s` (đường chéo):
```
(x / s) @ (W · s)ᵀ  ==  x @ Wᵀ        ← bất biến TUYỆT ĐỐI (đo: max diff 5.96e−7)
```
vì `W·s` scale *cột* weight, `x/s` scale *cùng channel* nghịch → contraction triệt tiêu. AWQ scale
**cột weight salient LÊN** trước round: đẩy nó cao trên lưới INT4 → error **tương đối** co lại, mà group
step `absmax/7` **gần như không đổi** vì nhiều channel non-salient vẫn giữ absmax. Rồi fold `1/s` vào
activation. Sau fold, noise cột đó chia `s` → share output error (đã bị activation khuếch đại) giảm
**~s²**. Non-salient bị scale *xuống* chút bởi geo-mean normalize → mang thêm chút error nhưng activation
bé nên bỏ qua.

**Dẫn xuất — knob α và vì sao interior optimum.** `s = act_scaleᵅ`, geo-mean normalized
(`s/√(max·min)` → pin trung điểm hình học = 1). Grid-search `α ∈ [0,1]` min **calib output MSE** — no
gradient, no retrain:
- `α = 0` → `s ≡ 1` → **đúng bằng naive INT4** (search không bao giờ tệ hơn baseline).
- `α` quá lớn → salient col scale cực đại → **chính nó set absmax của group** → group step nở → *cả
  group* mất precision → MSE tăng. Đo: `α=1.0 → MSE 5.55e-2` (**6× TỆ hơn** naive!).
- ⟹ có **interior optimum** `α ≈ 0.25` (protect-vs-inflate trade-off). Đây là dấu hiệu một trade-off
  thật, không phải "càng nhiều càng tốt".

**Neo code** (`src/scratch_llm/quant/awq.py`):
```python
def act_scale(x):                                        # :94   E[|x_j|] per-input-channel = saliency
    return x.abs().reshape(-1, x.shape[-1]).mean(dim=0)  # :101
def _normalize_scale(s):                                 # :104  geo-mean-center → step ~không đổi
    return s / (s.max() * s.min()).sqrt()                # :112
def _awq_dequant_weight(w, chan_scale, group_size):      # :115  quant(W·s) → dequant → fold /s
    w_scaled = w * chan_scale[None, :]                   # :117  scale cột LÊN
    q, gs = quantize_groupwise_int4(w_scaled, group_size)# :118  CÙNG lưới INT4 với naive (reuse 5.2!)
    return dequantize_groupwise_int4(q, gs)/chan_scale   # :119  fold 1/s về
def search_awq_scale(w, x, grid=20):                     # :122
    y_ref = xf @ wf.t()                                  # :139  reference
    for i in range(grid+1):                              # :143  α = i/20 ∈ [0,1]
        s   = _normalize_scale(a.pow(i/grid))            # :145
        cur = mse(y_ref, xf @ _awq_dequant_weight(...).t())  # :147  đo calib MSE
        if cur < best.calib_mse: best = ...              # :148  giữ min
```
Bug-review: AWQ **reuse** `int4_group.quantize_groupwise_int4` (:51 import) — *cùng lưới INT4 với
naive*, **chỉ khác chỗ đặt scale**. Đó là bằng chứng "không thêm bit". `α=0` cho `s=1` → bit-exact
naive (đo: MSE trùng khít). Grid coarse 21 điểm — production AWQ có thể mịn hơn, nhưng interior optimum
đã lộ rõ.

**Hình ảnh — dời độ mịn tới chỗ đau:**
```
input channel:      [ non  non  SAL  non  SAL  non ]     SAL = activation outlier (x_j lớn)
activation |x_j|:   [ 1.0  0.9  12.  1.1  12.  0.8 ]     act_scale = E[|x_j|]
                            AWQ scale s = act_scaleᵅ (α≈0.25), geo-mean normalized:
weight col scale:   [ ↓    ↓    ↑↑   ↓    ↑↑   ↓  ]     salient LÊN, non-salient xuống chút
  → W·s: salient col cao trên lưới INT4 → rel round-err ↓ ; group absmax ~giữ (non-sal set nó)
  → fold 1/s vào x: noise salient col / s → share output err (×act) ↓ ~s²

α grid (ĐO):  0.0→9.03e-3(=naive)  0.2→5.54e-3  0.3→5.54e-3  0.5→8.20e-3  1.0→5.55e-2(6× tệ!)
                    └baseline┘        └── interior optimum α≈0.25 ──┘         └inflate cả group┘

shape: W:(256,512) · x:(64,512) → chan_scale:(512,) → q:(256,512) int4 + group_scales:(256,4)
```

**Số đo THẬT** (`bench/a5_awq.py`, canonical, khớp RESULTS):
```
5 seeds: naive MSE ~9.0e-3 (18.7 dB)  vs  AWQ ~5.3e-3 (21.0 dB)   mean recovery 1.71× (min 1.65×)
held-out: naive 9.25e-3 | AWQ 5.49e-3 | recovery 1.69×            ← win transfers, không calib-overfit
α grid interior optimum α≈0.25 ; α=0 ≡ naive bit-exact ; α=1.0 → 5.55e-2 (6× tệ hơn)
activation-weighted col-err:  salient 1.604 → 0.560 (2.87× thấp hơn) ; other 0.716 → 0.845 (nhích lên)
identity (x/s)@(W·s).ᵀ == x@Wᵀ  max abs diff = 5.96e-7                ← measure_s5.py
```

**Frontier / cổng.** AWQ (arXiv:2306.00978, MLSys 2024 best paper) = PTQ W4A16 default 2024–2026, cặp
với NVFP4/group-INT4 + Marlin kernel. **Gate:** "Weight-quant, sao lại nhìn activation?" → error
propagation nhân activation magnitude (`y_q−y = x@(Ŵ−W)ᵀ`); "α làm gì?" → protect-vs-inflate, α=0=naive,
interior optimum. **Modify-and-predict:** α=1 → salient được bảo vệ tối đa NHƯNG group step bị nó kéo
lên → MSE tổng ↑ (đo 6× tệ). Trait: **research-as-MDP** (grid-search 1 knob, kill nếu không beat naive;
α=0 là floor an toàn). Bucket: **inference/PTQ = differentiator.**

---

## Bảng số đo THẬT (chạy lại được — DoD-là-profile)

| Concept | Đo | Kết quả | Nguồn | Ý nghĩa |
|---|---|---|---|---|
| 5.1 | INT8 sym per-tensor Gaussian SQNR | **38.39 dB** (RESULTS 40.50) | measure_s5 | ~40 lớp, dưới 44 lý tưởng (crest penalty) |
| 5.1 | LUẬT 6dB/bit (2× scale) | 38.39→32.37, **Δ=−6.02** | measure_s5 | +1 bit = +6.02 dB, dẫn từ s²/12 |
| 5.1 | asym vs sym (skewed) | 37.11→**43.71 (+6.60)** | measure_s5 | asym giành nửa range âm |
| 5.1 | per-channel vs per-tensor | 36.29→**42.64 (+6.35)** | measure_s5 | co crest cục bộ |
| 5.1 | W8A8 int-GEMM | rel **1.02e-2**, acc **int32** | measure_s5 | scale hoist khỏi Σ_k |
| 5.2 | pack/unpack bit-exact | **True** (500 fuzz + adversarial) | measure_s5 | two's-comp + sign-extend |
| 5.2 | pack([−8,−1]) | **[143] = 0x8F** | measure_s5 | hi=0x8, lo=0xF |
| 5.2 | group g=128 vs g=512 | 18.67 > 17.52 (**+1.15**) | measure_s5 | group co crest per-block |
| 5.3 | NVFP4 vs MXFP4 MSE | **1.48×** (20.43 vs 18.74 dB) | bench | two-level + k=16 thắng |
| 5.3 | cơ chế-1 (E4M3 vs E8M0 @k=32) | **+1.12 dB** | bench | non-pow2 scale bám ideal |
| 5.3 | ideal 5.0 → scale | E8M0 4.0 (20%) vs E4M3 5.0 (0%) | measure_s5 | octave rounding vs mantissa |
| 5.3 | block-scaled GEMM vs bf16 | **0.286%** (<1% oracle) | bench | scale sống trong accumulation |
| 5.4 | FP8 tensor SQNR vs INT4 | **31.81 vs 18.61 dB** | bench | 8-bit float ≫ 4-bit int |
| 5.4 | per-channel-K law (INT4) | **2.49×** | bench | fixed-point granularity bites |
| 5.4 | per-channel-K law (FP8) | **0.29×** (collapse!) | measure_s5 | GAP: E4M3 mantissa nuốt range |
| 5.4 | E4M3 500.0 no-clamp | **NaN** (clamp→448) | measure_s5 | NaN-only top → clamp-before-cast |
| 5.4 | KV bytes FP8 vs BF16 | **0.552×** | bench | calibrate-once amortize scale |
| 5.5 | AWQ recovery (5 seed) | **1.71×** (min 1.65) | bench | dời scale, cùng bit |
| 5.5 | held-out recovery | **1.69×** | bench | không calib-overfit |
| 5.5 | α=1.0 vs optimum | **6× tệ** (interior α≈0.25) | bench | protect-vs-inflate |
| 5.5 | identity (x/s)@(W·s)ᵀ | max diff **5.96e-7** | measure_s5 | bất biến đại số |

---

## Checklist recall COLD (che phần trên, tự trả lời — đây là cách re-learn)

1. **5.1** Dẫn luật 6.02 dB/bit từ `s²/12`. Vì sao INT8 Gaussian rơi ~38 chứ không ~44 dB? (crest factor)
2. **5.1** Vì sao asym thắng sym +6 dB trên tensor toàn dương? Vì sao per-channel co crest?
3. **5.1** Viết `Y[m,n]` và chỉ ra vì sao `s_X, s_W` kéo ra ngoài `Σ_k`. Vì sao PHẢI per-*output*-channel?
4. **5.2** `pack([−8,−1])` = byte nào? Vì sao `& 0xF` rồi sign-extend? `0xF` decode ra gì nếu quên?
5. **5.2** Vì sao group thắng per-tensor ở INT4? Giá của g 128→32 là gì (đếm scale)?
6. **5.3** Hai cơ chế NVFP4 thắng MXFP4? Vì sao `s_global=amax/(448·6)` đảm bảo `s_block ≤ 448`?
7. **5.3** E8M0 vs E4M3 bám scale lý tưởng 5.0 ra sao? Vì sao MXFP4 dùng ceil (no-overflow) không round?
8. **5.4** Vì sao K per-channel, V per-token? Vì sao E4M3 buộc clamp-before-cast (NaN>448)?
9. **5.4** "per-channel-K law 2.49×" đo trên INT4 hay FP8? Vì sao trên FP8 nó collapse? (mantissa float)
10. **5.5** Dẫn `y_q−y = x@(Ŵ−W)ᵀ`. Vì sao bảo vệ channel *activation*-lớn? α=0 và α=1 cho gì? Vì sao interior optimum?

> Trả lời cold được cả 10 = **S5 thật sự OWNED** (interview-grade). Vấp câu nào → mở đúng mục đó, hoặc
> blank-slate hàm tương ứng (`raise NotImplementedError` → test đỏ → tự dẫn → xanh) để re-own bằng tay.
> Đặc biệt câu 9 là câu phân biệt "đọc code" với "đọc slide".

---

*Cross-ref: `roadmap/S5_quantization.md` (spine) · `bench/RESULTS.md` §"Perf track (A5)" (ledger đo) ·
`performance/notes/A5_design_note.md` (design) · `docs/learning/PROGRESS.md` (ledger 89 Bài) · sibling
derivation `M2_transformer_forward.md`. Concept kế: perf serving track (KV-cache, paged attention, MLA)
— nơi FP8-KV (5.4) trở thành cầu A5→A1.*
