# S1 — Serving Substrate (KV-cache) · First-Principles Derivations

> **Đây là gì.** Bản dẫn-xuất **từ nguyên lý gốc** (first principles) cho TOÀN BỘ tầng bộ nhớ
> KV-cache của một inference engine — 8 micro-concept S1 (Bài 1.0→1.7). Mỗi mục: (1) **câu hỏi**
> falsifiable, (2) **sự thật nền tảng** (áp lực vật lý/toán học ép ra thiết kế), (3) **dẫn xuất** có
> math, (4) **neo code** `file·func·line` (đã verify tại HEAD), (5) **hình ảnh** (ASCII + shape/stride
> + numeric trace tay), (6) **số đo THẬT** (chạy trên repo này hoặc `bench/RESULTS.md`), (7)
> **frontier framing** + cổng phỏng vấn.
>
> **Cách dùng để re-learn.** Đọc *Câu hỏi* của mỗi mục → **tự trả lời cold** (che phần dưới) → mở ra
> đối chiếu (PRR: Predict → Run → Reconcile). Cuối doc có **checklist recall cold** + bảng số đo. Đây
> là *derivation lab* có số đo — bạn đồng hành của `docs/learning/roadmap/S1_serving_substrate.md`
> (bản đồ mổ xẻ) và bài đầy đủ `docs/learning/serving/01-batched-kv-cache.md`.
>
> **Cảnh báo neo (quan trọng).** Roadmap S1 pin số dòng theo commit `9e61d7a`, thời điểm đó các class
> KV-cache còn nằm trong `model.py`. **Ngày 2026-07-08 chúng được TÁCH ra `src/scratch_llm/kv_cache.py`**
> (một module riêng, dep một chiều — `kv_cache` không bao giờ import `model`). Doc này pin theo HEAD
> `1e7dbc6` (2026-07-14): mọi anchor KV-cache trỏ vào **`kv_cache.py`**, chỉ mask + routing kernel còn
> ở `model.py`. Line number **trôi** — `grep` tên hàm, đừng tin số dòng cứng (luật repo: verify, don't
> trust).
>
> **Nguồn số đo.** Pillar này **GPU-gated** (Triton/CUDA/multi-GPU). Số throughput kernel là số **đo
> thật trên sm120**, cite `[measured · sm120 · bench/RESULTS.md]`. Số roofline/AI mình **tự dẫn** từ
> HBM BW + FLOP peak (label `[DERIVED]`). Các invariant (online-softmax ≡ full-softmax, gather bit-exact,
> truncate rollback, chunked-prefill bit-identical, write-then-mask no-NaN) được **verify bằng bộ test CPU
> đã commit** — `tests/test_paged_cache.py` (gather `torch.equal`), `tests/test_chunked_prefill.py`
> (chunked `torch.equal`), `tests/test_kv_cache.py`, `tests/test_speculative.py` /
> `tests/test_spec_acceptance.py` — chạy trên chính repo. Các số float dưới đây (vd `1.19e-7`, `2.61e-7`)
> là **hand-repro chạy lại tại HEAD** để minh hoạ machine-eps: ghi `[reproduced · CPU]` cho repro tay,
> `[verified · CPU · tests/…]` khi có test commit chốt bất biến. (KHÔNG có script `s1_checks.py` trong repo
> — đừng cite nó như một artifact đã commit.) "Implemented ≠ measured" (FOP-4).

---

## Bức tranh lớn — cả série là MỘT mũi tên

Decode 1 token là bài toán **memory-bound**: mỗi bước đọc TOÀN BỘ trọng số + TOÀN BỘ KV-cache chỉ để
sinh 1 token → arithmetic intensity `AI ≈ 1 FLOP/byte`, thấp hơn ridge ~131× (Bài 1.0). Mọi cấu trúc
dữ liệu KV-cache ở đây tồn tại để **kéo AI đó về phía trần** — không phải để "lưu K/V" cho đẹp.

```
              ÁP LỰC                          LỜI GIẢI (data structure)            ĐO ĐƯỢC
  ┌──────────────────────────────┐
  │ AI≈1: đọc mọi weight/1 token  │  1.0 roofline thesis                    AI 0.993, ridge 131, 51→253 tok/s
  └──────────────┬───────────────┘
                 │ nhồi thêm việc vào 1 lần đọc weight
                 ▼
  ┌──────────────────────────────┐
  │ chia sẻ đọc weight cho B rows │  1.1 BatchedKVCache slab                 ×66 trần @B=256; AI≈B
  │  (địa chỉ TĨNH ⇒ CUDA-graph)  │      + write-then-mask (no-NaN)          padding tax 1.27×/bước
  └──────────────┬───────────────┘
                 │ nhưng slab reserve max_ctx mỗi slot ⇒ ~95% phí
                 ▼
  ┌──────────────────────────────┐   1.2 KVCache B=1 + truncate (tổ tiên · rollback speculative)
  │ ragged batch: đến/đi khác lúc │   1.3 SlotKVCache base (mirror py/device, view_len)
  │ độ dài khác nhau, KHÔNG rò    │   1.5 PrefillView (nạp offset-0)
  │ KHÔNG recompile-storm         │   1.6 ChunkPrefillView (nạp offset chạy)
  └──────────────┬───────────────┘
                 │ đổi reservation waste lấy fragmentation nhỏ
                 ▼
  ┌──────────────────────────────┐
  │ KV như BỘ NHỚ ẢO: block pool  │  1.4 PagedKVCache (block_table + CoW)    frag 5.0%, capacity ×9.3
  │  + bảng trang + trash block   │      NHƯNG gather-only = LỖ +6.3%/bước   (negative control)
  └──────────────┬───────────────┘
                 │ giành lại "thuế padding" bằng 1 kernel fused
                 ▼
  ┌──────────────────────────────┐
  │ đọc KV THẲNG qua bảng trang   │  1.7 Triton paged-decode kernel          5.90 ms/bước, ×3.52 vs wave
  │  online softmax, no gather    │      (online softmax fp32 qua page)      +55% vs dense-continuous
  └──────────────────────────────┘
```

**Sợi chỉ đỏ:** `AI≈1 → batch (×66 trần) → bỏ reservation waste (×9.3 dung lượng) → giành lại thuế
padding (×3.52 vs wave)`. Mỗi Bài là lời giải cho **một áp lực cụ thể** — bandwidth (1.0), chia-sẻ-đọc
(1.1), địa-chỉ-tĩnh-cho-graph (1.1/1.3), no-recompile (1.3), reservation-waste (1.4), padded-compute
(1.7). Học S1 = học *áp lực → data structure*.

Một sự thật kiến trúc xuyên suốt cả 6 class trong `kv_cache.py`: **hợp đồng sở hữu graph vs scheduler**.
Tensor device (`lengths`, `active`, `block_table`, pool) bị chạm **TRONG** forward (graph-owned, trace
sạch dưới `torch.compile`); bản sổ python (`py_lengths`, `py_active`, `_py_table`, `_free`, `view_len`)
+ **mọi quyết định cấp phát** là **scheduler-owned**, chạm **NGOÀI** graph. Đọc state python trong
forward = nướng Dynamo guard theo *nội dung list* → ragged churn hoán vị → recompile-storm (đo thật
2026-07-03, Bài 1.3). Đây là kiến trúc vLLM-V1, và là lý do doc này đọc như "một class, 6 biến thể".

---

## Bài 1.0 · Roofline thesis — decode là memory-bound, AI ≈ 1

**Câu hỏi.** Vì sao decode 1 token ở B=1 **KHÔNG** nhanh hơn dù GPU còn thừa cả núi FLOP/s? Đâu là
trần thật, và ba đòn bẩy nào kéo nó lên?

**Sự thật nền tảng.** Một GPU có **hai trần trong một phòng**: trần compute (`peak_flops`, FLOP/s tối
đa) và trần bandwidth (`AI · hbm_bandwidth`, byte/s × cường độ số học). Điểm giao = *ridge*. Một op nằm
bên **trái** ridge bị bandwidth chặn — tăng FLOP/s vô nghĩa vì thời gian do *đọc byte* quyết định. Đây
là vật lý, không phải kỹ thuật: bạn không thể tính nhanh hơn tốc độ nạp dữ liệu vào ALU.

**Dẫn xuất.** Arithmetic intensity `AI = FLOPs / bytes_moved`. Với một bước decode (query-length = 1):

```
FLOPs  = 2 · n_params · batch          (1 MAC = 2 FLOP mỗi weight, mỗi token)
bytes  = n_params · weight_bytes                       ← đọc MỌI weight một lần
       + 2 · n_layers · n_kv · head_dim · ctx · batch · kv_bytes   ← đọc TOÀN BỘ KV một lần
```

Ở `B=1`, ctx ngắn, KV traffic ≪ weight traffic (bf16 ⇒ `weight_bytes = 2`):

```
AI ≈ (2 · n_params · 1) / (n_params · 2) = 1  FLOP/byte
```

Ridge của sm120: `peak_bf16 / hbm_bw = 72e12 / 0.55e12 = 130.9 FLOP/byte`. Vì `AI = 1 ≪ 131`, op nằm
**sâu bên trái ridge** ⇒ `bound = "memory"`. Trần đạt được `attainable = min(peak, AI·bw) = 1 · 0.55e12
= 0.55 TFLOP/s` — chỉ **0.76%** của 72 TF/s. GPU *đói byte*, không đói FLOP.

**Ba đòn bẩy kéo AI lên** (chính là 3 hướng còn lại của série): (1) **batch** — nhồi B rows vào cùng
một lần đọc weight ⇒ `AI ≈ B` khi KV còn nhỏ (Bài 1.1); (2) **giảm byte KV** — GQA/MLA/KV-quant/paged
(Bài 1.4); (3) **fuse launches** — không đổi AI nhưng gỡ overhead che mất trần (Bài 1.7 + CUDA-graph).

**Neo code** (`src/scratch_llm/bench/roofline.py` · `decode_step_flops_bytes` :92):
```python
def decode_step_flops_bytes(n_params, *, n_layers, n_kv_heads, head_dim,
                            context_len, batch, weight_bytes, kv_bytes):
    flops = 2.0 * n_params * batch                                       # :110
    weight_traffic = n_params * weight_bytes                             # :111  đọc mọi weight
    kv_traffic = 2 * n_layers * n_kv_heads * head_dim * context_len * batch * kv_bytes  # :112
    return flops, float(weight_traffic + kv_traffic)
```
```python
# roofline() :38 — phán bound + số giây; ridge từ gpu_specs.py
attainable = min(peak, ai * spec.hbm_bandwidth)                          # roofline.py:47
bound = "compute" if ai >= ridge else "memory"                          # roofline.py:51
# gpu_specs.py · rtx4000-blackwell :86 — SỐ ĐO THẬT (không phải datasheet)
hbm_bandwidth=0.55e12,  # measured HBM copy                              # :88
peak_flops={BF16: 72e12, FP16: 72e12},  # measured bf16 GEMM 8192³       # :89
def ridge_point(self, dtype): return self.peak_flops[dtype] / self.hbm_bandwidth  # :38-45
```
> **Code review khi trace.** `gpu_specs.py` cố ý **bỏ fp8/fp4** cho sm120 (dòng :85 comment) — silicon
> có nhưng chưa đo, nên roofline chống lại việc *bịa* một peak (2×/4× của một bf16 *achieved* không phải
> peak phòng thủ được). Đây là "claims honesty" đóng vào type system: `roofline()` sẽ **raise** thay vì
> trả số ma. Một chi tiết production-grade mà agent hay bỏ.

**Hình ảnh — roofline log-log:**
```
  attainable
  FLOP/s
   72e12┤                          ┌──────────── compute roof (peak 72 TF/s)
        │                      ╱   │
        │                  ╱       │   GEMM lớn (AI≫ridge) sống ở đây →
        │              ╱           │
 0.55e12┤          ╱  ← slope = hbm_bw (0.55 TB/s)
        │      ╱                   │
   decode●╱ ← AI≈1, attainable 0.55 TF/s = 0.76% peak
        └──┴───────────────────────┴──────────► AI (FLOP/byte, log)
          1                      131 = ridge
```
**Hand-trace** (`[reproduced · CPU · roofline.py]`, cfg 0.84B / 24 layer / n_kv=8 / head_dim=128 / ctx=128):

| B | AI (FLOP/B) | bound | ridge | predicted tok/s |
|---|---|---|---|---|
| 1 | **0.993** | memory | 130.9 | 324.9 |
| 32 | 25.813 | memory | 130.9 | 264.1 |
| 128 | 65.350 | memory | 130.9 | 167.1 |

AI leo `≈B` khi batch (KV còn nhỏ so weight); B=128 chưa chạm ridge vì ctx=128 làm KV traffic bắt đầu
lớn — crossover thật đo được ở B≈128 khi ctx dài hơn (R3a).

**Số đo THẬT.** `[measured · sm120 · RESULTS.md R1]` (0.84B bf16, B=1): predicted **315 tok/s** (trần bộ
nhớ) → **đo 51 tok/s = 16% roof**; AI ≈ 0.96; đạt **89 GB/s = 16%** của 550 GB/s HBM; nsys = **~955 kernel
launch/token**. Nghịch lý phải nuốt: *workload* memory-bound nhưng *run eager* lại **overhead-bound** — 84%
thời gian là launch dispatch, không phải byte. Chuỗi strip chứng minh thesis "in trend": eager 51 → nosync
50 (sync ~miễn phí, launches mới là thủ phạm) → **compiled 173 (53% roof, 291 GB/s)** → **CUDA-graph 253
tok/s = 77% của trần 327** — chỉ khi gỡ hết launch thì memory-bound mới *hiện ra* `[FACT R1/R4.4]`.

**Frontier / cổng.** Đây là lý do **H200 = "bản nâng cấp decode"** của H100: cùng die compute (989 TF/s
bf16) nhưng HBM **4.8 TB/s** vs 3.35 — tăng bandwidth, không tăng FLOP (xem `gpu_specs.py:58-63`). Cổng
interview: "Decode bound bởi gì, và ba đòn bẩy nào kéo AI lên?" Trait = **roofline-first / predict-the-
number** (FOP-3). Scarce-2026 bucket: **inference** = differentiator.

---

## Bài 1.1 · BatchedKVCache — slab dày đặc + write-then-mask

**Câu hỏi.** Câu trả lời *cụ thể đầu tiên* cho Bài 1.0: cấu trúc nào cho phép B request **chia sẻ một
lần đọc weight**, mà địa chỉ đủ TĨNH để CUDA-graph capture (cái mà `torch.cat` cache ở Bài 1.2 phá vỡ)?

**Sự thật nền tảng.** Batching kéo `AI ≈ B` (Bài 1.0) chỉ khi B rows dùng CHUNG một lần nạp weight. Muốn
vậy K/V của mọi slot phải nằm trong **một buffer cố định địa chỉ**, ghi **in-place** — không `cat` (cấp
tensor mới ⇒ địa chỉ đổi ⇒ graph vỡ: lỗi R1 `accessing tensor output of CUDAGraphs overwritten`). Địa
chỉ tĩnh là *điều kiện tiên quyết* của R4.4 CUDA-graph, không phải tối ưu phụ.

**Dẫn xuất — slab.** Cấp phát mỗi layer một buffer `(n_slots, n_kv_heads, max_ctx, head_dim)`. Slot `b`
sở hữu hàng `b`; ghi token mới của nó tại offset riêng `pos = lengths[b]`. Nhưng batched decode chạy
**lockstep** (mọi slot xử lý đúng 1 token/bước, shape tĩnh) — slot trống/ragged thì sao? **Write-then-
mask**: *luôn ghi* mọi hàng (kể cả rác vào hàng inactive), rồi *che* bằng mask per-row. Vì sao không
skip? Skip = shape động = compile/graph vỡ. Trade: tốn 1 write rác/hàng-trống, đổi lấy shape tĩnh.

Giá của slab (chính là động cơ Bài 1.4 + 1.7): (1) **reservation waste** — giữ `max_ctx` mỗi slot dù
dùng bao nhiêu (~95% trên trace ngắn); (2) **thuế padding** — `decode_view` đọc cửa sổ rộng bằng hàng
**dài nhất** (`view_len`), nên hàng ngắn cũng trả byte cho phần padding của hàng dài.

**Neo code** (`src/scratch_llm/kv_cache.py`):
```python
class BatchedKVCache(SlotKVCache):                                        # :234
    shape = (n_slots, n_kv_heads, max_ctx, head_dim)                      # :255  slab
    self._k = [torch.zeros(shape, ...) for _ in range(n_layers)]          # :256  1 buffer/layer
    def _write_decode_kv(self, layer, pos, k_new, v_new):                 # :259
        self._k[layer][self._slot_idx, :, pos] = k_new[:, :, 0]           # :260  ghi IN-PLACE tại pos
    def decode_view(self, layer):                                         # :263
        length = self.view_len                                            # :264
        return self._k[layer][:, :, :length], self._v[layer][:, :, :length]  # :265  view padded
```
Mask được dựng ở tầng attention (`src/scratch_llm/model.py` · `MultiHeadSelfAttention.forward`):
```python
lengths = cache.lengths            # (B,) PRE-write lengths                # :284
cache.write_decode(layer_idx, k, v)                                       # :285  ghi trước
...
k_pos = torch.arange(k.shape[2], device=x.device)                         # :306
row_mask = (k_pos.unsqueeze(0) <= lengths.unsqueeze(1))[:, None, None, :] # :307  dấu ≤ !
```

**Hình ảnh — slab + write-then-mask** (anchor `B=3, n_kv=2, head_dim=8, max_ctx=16`; slot0 len=4,
slot1/2 inactive len=0):
```
_k[layer] : (3, 2, 16, 8)  strides (256, 128, 8, 1)   ← contiguous, ĐỊA CHỈ CỐ ĐỊNH
                 ↑ n_slots      ↑ max_ctx (chiều reserve)
  slot0: [k0 k1 k2 k3 · · · · · · · · · · · ·]  len=4  → write_decode ghi k4 tại pos=4
  slot1: [g  · · · · · · · · · · · · · · · ·]   len=0  → ghi rác g tại pos=0  (inactive)
  slot2: [g  · · · · · · · · · · · · · · · ·]   len=0  → ghi rác g tại pos=0
                     view_len = 1 + max(len) = 5  ⇒ decode_view đọc [:, :, :5]

row_mask (dấu ≤, PRE-write lengths):        keys visible/row:
  slot0 len=4: j ≤ 4 → [T T T T T]          5   ← thấy cả k4 vừa ghi (self-inclusive)
  slot1 len=0: j ≤ 0 → [T F F F F]          1   ← chỉ thấy rác của chính nó → KHÔNG all-masked
  slot2 len=0: j ≤ 0 → [T F F F F]          1
```
**Vì sao dấu `≤` sống-còn (hai hậu quả nếu đổi `<`):** (a) token vừa ghi ở index = `lengths[b]` (old
length) sẽ *không tự thấy mình* → attention bỏ qua token mới nhất; (b) hàng len=0 → mask all-False →
softmax over `[−∞,−∞,...]` → **NaN**. Dấu `≤` bảo đảm mỗi hàng thấy ≥1 key ⇒ **no-NaN theo cấu trúc**.

**Số đo THẬT.**
- `[reproduced · CPU]` write-then-mask: `lengths=[4,0,0], view_len=5` → keys visible/row
  `[5,1,1]`, **min=1** (≥1 ⇒ không có hàng all-masked ⇒ no NaN).
- `[measured · sm120 · RESULTS.md R3a]` slab batching: agg tok/s `185→9,255` B=1→64, **peak 12,220 @B=256
  = 66×** B=1; AI≈B exact; crossover memory→compute ở **B≈128** (AI≈ridge 131).
- `[measured · sm120 · RESULTS.md R3b]` continuous vs static-wave (heavy-tail): **2.30× wall / 2.93× theo
  bước** (util 24.9%→72.8%); **thuế padding 1.27×/bước** (9.6 vs 6.3 ms) — con số này là *toàn bộ động
  cơ* của Bài 1.4/1.7.

**Frontier / cổng.** Slab + continuous batching = kiến trúc serving cơ bản (vLLM static-buffer, TGI).
Cổng: "Vì sao continuous batching thắng static-wave, và giá nó trả là gì?" (util cao hơn nhưng padding
tax). Trait = **execution > analysis** (dựng buffer tĩnh mở khoá cả CUDA-graph + paged). Bài đầy đủ đã
viết: `docs/learning/serving/01-batched-kv-cache.md`.

---

## Bài 1.2 · KVCache — cache 1-request + `truncate()` rollback

**Câu hỏi.** Dạng ĐƠN GIẢN NHẤT của KV-cache trông thế nào — và vì sao chính nó *phá* CUDA-graph, buộc
cả họ `SlotKVCache` ra đời? Và primitive nào cho phép speculative decoding *lossless*?

**Sự thật nền tảng.** Decode incremental: mỗi bước chỉ tính K/V cho token MỚI, nối vào đuôi cache đã có,
để attention token mới đọc toàn bộ quá khứ mà không tính lại. Dạng ngây thơ nhất = "cuốn sổ nối đuôi":
`torch.cat` K/V mới vào đuôi. Đúng, dễ — nhưng `cat` **cấp phát tensor MỚI mỗi bước** ⇒ địa chỉ HBM đổi
liên tục ⇒ kẻ thù số 1 của CUDA-graph (graph cần địa chỉ tĩnh).

**Dẫn xuất — truncate là rollback primitive.** Speculative decoding (Bài 1.7-adjacent, R4.3): drafter
đề xuất K token, target verify cả K trong 1 forward, chấp nhận prefix dài nhất khớp greedy argmax của
target. Các vị trí draft bị **từ chối** đã được ghi K/V vào cache — phải **vứt** để không attend lại.
`truncate(length)` làm đúng thế: bất biến `0 ≤ length ≤ self._length` (chỉ co, không giãn); mỗi layer
cắt `k[:, :, :length]` rồi `.contiguous()`.

**Vì sao `.contiguous()` là cần, không thừa?** `k[:, :, :length]` là **view** — vẫn giữ storage cũ
(kể cả đuôi bị cắt). Muốn *giải phóng* đuôi VÀ có địa chỉ sạch (contiguous) cho bước `cat` kế, phải
**copy** ra buffer mới. Bỏ `.contiguous()` = giữ storage phình + stride lạ → bước sau cat lên view có
thể sai/chậm.

**Vì sao rollback + re-decode cho chuỗi Y HỆT (lossless)?** K/V của phần **giữ lại** là *bit-identical*
với decode thường: RoPE xoay theo **vị trí tuyệt đối** (không phụ thuộc token draft nào ở batch cùng
forward), và K-projection của một vị trí không phụ thuộc các vị trí khác. Nên "verify greedy + rollback"
cho ra **đúng chuỗi greedy của target** — lossless *by construction*, không phải xác suất.

**Neo code** (`src/scratch_llm/kv_cache.py` · `KVCache` :28):
```python
def append(self, layer, k_new, v_new):                                   # :52
    existing = self.get(layer)
    if existing is None: self._k[layer], self._v[layer] = k_new, v_new
    else:
        past_k, past_v = existing
        self._k[layer] = torch.cat([past_k, k_new], dim=2)               # :59  CẤP TENSOR MỚI → địa chỉ động
        self._v[layer] = torch.cat([past_v, v_new], dim=2)               # :60
def truncate(self, length):                                              # :65
    if not 0 <= length <= self._length: raise ValueError(...)            # :70  chỉ co
    for layer in range(len(self._k)):
        self._k[layer] = k[:, :, :length].contiguous()                  # :75  copy: giải phóng đuôi + địa chỉ sạch
        self._v[layer] = v[:, :, :length].contiguous()                  # :76
    self._length = length                                               # :77
```
Dùng trong `serving/speculative.py · speculative_generate` :108:
```python
base = cache.length
vlogits = target(inp, cache)[0]   # inp=[pending, *drafts]; cache lớn thêm 1+K vị trí  :154
...
cache.truncate(base + 1 + accepted)  # giữ pending + accepted drafts; vứt phần còn lại  :170
```

**Hình ảnh — append rồi truncate** (anchor 1 layer, n_kv=2, head_dim=8):
```
append k mỗi bước:  K = torch.cat([past, new], dim=2)   ← dim=2 là chiều seq
  bước t:   K:(1,2,t,8)  ── cat new (1,2,1,8) ──►  (1,2,t+1,8)   [ĐỊA CHỈ MỚI mỗi lần]

speculative round (base=5, K=3 drafts, accept 2):
  forward [pending, d0, d1, d2] → cache length 5→9  (ghi pending@5, d0@6, d1@7, d2@8)
  greedy khớp d0,d1 nhưng ≠ d2  ⇒  accepted=2
  truncate(5 + 1 + 2 = 8):  cắt [:, :, :8]  → vứt K/V của d2 (vị trí 8)
                            ↑ giữ pending@5, d0@6, d1@7 (bit-identical với decode thường)
```
**Hand-trace** `[reproduced · CPU]` (bất biến rollback cũng do `tests/test_speculative.py` chốt): decode 3 token bằng 2 đường — (A) plain một-token-mỗi-
bước; (B) mỗi bước ghi `[t, draft1, draft2]` rồi `truncate(base+1)` cắt 2 draft. Kết quả: `length A=8,
B=8`; `max|K_A − K_B| = 1.19e-7` (float32 GEMM reduction-order noise; **bit-identical trong float64/số
học đúng**) ⇒ prefix giữ lại y hệt ⇒ rollback sạch, không rò K/V rác.

**Số đo THẬT.** `[measured · sm120 · RESULTS.md R4.3]` trên nền `truncate`: **token-exact** (27 tests,
float64 exact mọi drafter/K; **wrong-drafter vẫn exact** ⇒ KV rollback sạch); **×1.35–1.39 tok/s**,
1.32–1.54 token/target-forward. `[FACT]`: R1 `reduce-overhead` capture **FAILED** với `cat`-cache
(`RuntimeError: accessing tensor output of CUDAGraphs overwritten`) — chính lỗi này *scope* ra R4.4:
cần buffer tĩnh in-place ⇒ họ `SlotKVCache`.

**Frontier / cổng.** `truncate` là nền của MỌI drafter-family (Medusa/EAGLE/MTP): tất cả tối ưu
`E[accept]`, nhưng đều cần một KV-rollback đúng. Cổng: "Speculative decoding lossless nhờ đâu?" —
greedy-verify + rollback, KHÔNG phải xác suất. Trait = **claims honesty** (lossless *by construction*,
kill-line = bất kỳ divergence khỏi greedy = math sai). Scarce bucket: inference.

---

## Bài 1.3 · SlotKVCache — base class: mirror py/device, write-then-mask, view_len

**Câu hỏi.** Làm sao decode một **ragged batch** (độ dài khác nhau, đến/đi khác lúc) trong lockstep mà
(a) không hàng nào rò sang hàng khác, và (b) KHÔNG gây recompile-storm cho `torch.compile`?

**Sự thật nền tảng.** Hai loại state va nhau: state **device** (do forward chạm, phải nằm trong graph
để trace) và state **python/scheduler** (con trỏ slot, quyết định cấp phát). Nếu forward đọc state
python (vd `max(py_lengths)`), Dynamo **nướng guard theo NỘI DUNG list** — ragged churn hoán vị list →
guard vỡ → recompile-limit → eager fallback (đo thật 2026-07-03). Đây là bẫy `torch.compile` #1 với
serving.

**Dẫn xuất — hai bản sổ song song + view_len là int thường.** `SlotKVCache` giữ:
- `lengths`/`active` (tensor device) — forward mutate, graph-owned.
- `py_lengths`/`py_active` (list python) — scheduler mutate, ngoài graph.
- `view_len = min(max_ctx, 1 + max(py_lengths))` — **int python thường**, tính lại CHỈ bởi mirror-ops
  của scheduler (`mirror_advance`, `mirror_admit`, `free_slot`), **không bao giờ** tính `max()` trong
  forward.

**Write-then-mask** (đã dựng ở 1.1, đây là chỗ nó *sống* trong base): `write_decode` ghi mọi hàng tại
`pos = lengths.clamp(max=max_ctx-1)`; `advance(1)` bump chỉ hàng **active** (`lengths += active.long()`),
**một lần/forward sau mọi layer** (đúng hợp đồng `KVCache`). Mask `j ≤ lengths[b]` (dấu `≤`) ⇒ mỗi hàng
thấy ≥ ô vừa ghi ⇒ softmax không all-`−∞` ⇒ **no-NaN by construction**.

**Neo code** (`src/scratch_llm/kv_cache.py` · `SlotKVCache` :80):
```python
self.lengths = torch.zeros(n_slots, dtype=torch.long, ...)   # device                 # :126
self.active  = torch.zeros(n_slots, dtype=torch.bool, ...)   # device                 # :127
self.py_lengths = [0]*n_slots ;  self.py_active = [False]*n_slots  # python mirror     # :128-129
def view_len(self):  return self._view_len                    # int thường, an toàn đọc trong graph  # :169-175
def _recompute_view_len(self): self._view_len = min(self.max_ctx, 1 + max(self.py_lengths))  # :177-178 (chỉ scheduler gọi)
def write_decode(self, layer, k_new, v_new):                                          # :194
    pos = self.lengths.clamp(max=self.max_ctx - 1)            # graph-owned            # :204
    self._write_decode_kv(layer, pos, k_new, v_new)                                   # :205
def advance(self, n):                                                                 # :207
    if n != 1: raise ValueError(...)                                                  # :211
    self.lengths += self.active.long()                        # bump CHỈ active, device-only  # :213
def mirror_advance(self):                                     # scheduler, NGOÀI graph  # :215
    for b,a in enumerate(self.py_active):
        if a: self.py_lengths[b] += 1                                                 # :218-220
    self._recompute_view_len()                                                        # :223
```
> **Code review khi trace.** `advance` khẳng định `n==1` (:211) — batched decode xử lý đúng 1 token/hàng;
> đây là fail-fast đúng chỗ (một forward nhiều-token vào slot cache = bug logic). `pos.clamp(max_ctx-1)`
> (:204) là lưới an toàn: scheduler *đã* bảo đảm không overflow (evict tại capacity), clamp chỉ để hàng
> inactive/tràn ghi rác vào ô cuối thay vì out-of-bounds — che bởi mask.

**Hình ảnh — hợp đồng graph/scheduler một bước decode:**
```
   SCHEDULER (python, NGOÀI graph)          FORWARD (device, TRONG graph = torch.compile)
   ─────────────────────────────           ───────────────────────────────────────────
   pre_decode_reserve()  ─────────────────► write_decode(layer, k, v)   ← ghi tại lengths[b]
   (dense: no-op)                              _write_decode_kv (subclass)
                                            ... (mọi layer) ...
                                            advance(1): lengths += active.long()   ← 1 lần
   mirror_advance() ◄──────────────────────  (device length đã bump)
     py_lengths[b] += 1 (active)
     _recompute_view_len()  → view_len

   LỖ RÒ đã đo (R3b): nếu forward đọc max(py_lengths) → Dynamo guard theo ordering
   (py_lengths[26] > py_lengths[16]) → ragged churn hoán vị → recompile-limit → eager fallback.
   FIX: view_len là int attribute, chỉ scheduler nuôi → unique_graphs = 2.
```
**Shapes:** `k_new`/`v_new` vào `write_decode` = `(n_slots, n_kv_heads, 1, head_dim)` (assert :202); `pos`
= `(n_slots,)` long; `lengths`/`active` = `(n_slots,)`. Advanced-index `_k[layer][slot_idx, :, pos]` ghi
tại `(b, :, pos[b], :)` cho từng b — vector hoá, không loop python trong graph.

**Số đo THẬT.** `[reproduced · CPU]` write-then-mask trên `BatchedKVCache` (subclass): hàng
inactive len=0 vẫn có **min=1** key visible ⇒ no NaN. `[measured · sm120 · RESULTS.md R3b]`: sau khi vá
2 lỗ rò dynamo, **unique_graphs = 2** (trước đó: guard theo thứ tự list → recompile-limit → eager
fallback chỉ ở nhánh continuous).

**Frontier / cổng.** "Ownership contract" graph-vs-scheduler này chính là kiến trúc **vLLM-V1**
(scheduler python thuần, forward là graph tĩnh). Cổng: "Vì sao đọc state python trong forward là bẫy với
`torch.compile`?" Trait = **spec-with-falsifiers** (view_len là int → falsifier: nếu tính max() in-graph,
recompile storm ở nhánh continuous mà nhánh wave giấu). Scarce bucket: inference + systems.

---

## Bài 1.4 · PagedKVCache — block pool + block_table + CoW + trash block

**Câu hỏi.** Slab lãng phí ~95% (mỗi slot đặt cọc `max_ctx` dù prompt ngắn). Làm sao biến lãng phí
*reservation* thành phân mảnh *nhỏ* (≤15 token/hàng) mà vẫn giữ shape tĩnh cho graph — và giữ nguyên
write-then-mask no-NaN?

**Sự thật nền tảng.** Đây là **bộ nhớ ảo** áp cho KV. OS giải bài "process cần bộ nhớ liền mạch nhưng
RAM phân mảnh" bằng paging: chia thành page cố định + page table ánh xạ logic→vật lý, cấp on-demand. KV
có cùng bài toán (slot cần dãy liền `max_ctx`, pool HBM phân mảnh) ⇒ cùng lời giải.

**Dẫn xuất — block pool + bảng trang.** Mỗi layer: pool `n_blocks` block cố định **16 token**
`(n_blocks, H_kv, 16, d)` + bảng `block_table[b, i]` = block logic `i` của slot `b` → block vật lý. Cấp
*on-demand*: mượn block mới CHỈ khi length vượt bội số 16. Lãng phí sụp từ **reservation** (`max_ctx − ℓ`)
xuống **internal fragmentation** (≤15 token ở block cuối, `E ≈ 8`):

```
frag = 1 − live_tokens / allocated_tokens ,   allocated = ⌈Σ ℓ_b / 16⌉ · 16
E[frag] ≈ E[tail] / E[ℓ] ≈ 8 / ⟨ℓ⟩   (⟨ℓ⟩≈160 trên heavy-tail ⇒ ~5%)
```

**Trash block (mẹo then chốt).** Block 0 dành riêng làm **rác**: mọi entry chưa cấp phát = 0 → trỏ về
block 0. Nhờ đó **write-then-mask của Bài 1.3 chuyển sang paged MIỄN PHÍ**: static-shape decode "mọi hàng
đều ghi" vẫn đúng — hàng inactive/quá-block ghi rác vào trash, mask che. Không cần code path riêng.

**CoW (Copy-on-Write) block-granularity.** `share_prefix` trỏ leading table entries của slot mới vào
block vật lý của slot khác (refcount++). Sharing **full-block only**, và write kế của một hàng rơi ĐÚNG
biên block ngay sau prefix chia sẻ → `pre_decode_reserve` cấp cho nó block **riêng** ⇒ **không bao giờ
ghi vào shared block** (CoW suy biến thành "copy-never"). Bất biến refcount kiểm tra mỗi bước: nếu write
nhắm vào block có `refcount≠1` → **RAISE** (CoW invariant vỡ). Partial-block CoW (beam search) cố ý
out-of-scope.

**Neo code** (`src/scratch_llm/kv_cache.py` · `PagedKVCache` :283):
```python
BLOCK = 16                                                                            # :308
shape = (n_blocks, n_kv_heads, self.BLOCK, head_dim)                                  # :326  pool
self.block_table = torch.zeros((n_slots, max_blocks), dtype=long)  # 0 = trash        # :330
self._free = list(range(n_blocks-1, 0, -1))   # block 0 KHÔNG BAO GIỜ cấp             # :332
self._refcount = [0]*n_blocks                                                         # :333
def _write_decode_kv(self, layer, pos, k_new, v_new):                                 # :423
    blk_idx = pos // self.BLOCK ; offset = pos % self.BLOCK                           # :424-425
    phys = self.block_table[self._slot_idx, blk_idx]   # (B,) qua bảng trang          # :426
    self._pool_k[layer][phys, :, offset] = k_new[:, :, 0]                             # :427
def pre_decode_reserve(self):                          # scheduler, trước forward     # :373
    for b,(ln,a) in enumerate(zip(self.py_lengths, self.py_active)):
        if not a: continue
        if ln % self.BLOCK == 0:                        # chạm biên → cấp block RIÊNG  # :380
            blk = self._alloc_block(); self.block_table[b, ln//16] = blk              # :385-387
        elif self._refcount[self._py_table[b][ln//16]] != 1:                         # :388
            raise RuntimeError("decode write aimed at a shared block — CoW invariant broken")  # :389
```
```python
def decode_view(self, layer):   # gather pool → view padded (oracle + negative control)  # :430
    phys = self.block_table[:, :n_blocks]   # (B,nb); unallocated→0→trash→masked       # :436
    k = self._pool_k[layer][phys]           # (B, nb, H_kv, 16, d)                     # :437
    k = k.permute(0,2,1,3,4).reshape(b, n_kv_heads, nb*16, -1)                        # :440
    return k[:, :, :length], v[:, :, :length]                                         # :442
```

**Hình ảnh — bảng trang + trash** (anchor `BLOCK=16, max_ctx=64 ⇒ max_blocks=4`; slot1 len=5):
```
pool_k[layer] : (n_blocks=32, H_kv=2, 16, 8)   strides (256,128,8,1)
block_table   : (n_slots, max_blocks=4)
  slot1 (len 5):  [ phys=7,  0,   0,   0  ]     ← chỉ 1 block cấp (⌈5/16⌉=1); còn lại = trash
                     │        └─┴─── trash block 0 (rác, mask che)
                     ▼
  pool block 7:  [k0 k1 k2 k3 k4 · · · · · · · · · · ·]   (offset 5..15 = rác, mask che)

decode write token 5 của slot1:  pos=5 → blk_idx=0, offset=5 → phys=table[1,0]=7 → pool[7,:,5]=k5
decode write token 16 (biên!):   pre_decode_reserve thấy ln%16==0 → cấp block RIÊNG (vd phys=9),
                                  table[1,1]=9  ⇒ CoW không bao giờ chạm shared prefix block
```
**Số đo THẬT.**
- `[verified · CPU · tests/test_paged_cache.py]` gather bit-exact: `max|K_paged − K_dense|` **over VALID positions
  (j<length) = 0.000e+00** — gather qua bảng trang cho ra ĐÚNG cùng byte với slab tại mọi vị trí thật
  (vị trí padding đọc trash, bị mask). (Frag toy-trace = 0.219, capacity ×2.0 — trace nhỏ; headline
  ledger dưới đây.)
- `[measured · sm120 · RESULTS.md R4.1 P4.1.1]` heavy-tail: **frag 5.0%** (dead-center; `E[tail]≈8/⟨ℓ⟩≈160`)
  vs reservation waste ~95%; **capacity ×9.3** (peak-alloc 7,040 token vs 65,536 reserved).
- `[measured · sm120 · RESULTS.md R4.1 P4.1.3]` **negative control**: paged-gather-ONLY (chưa kernel) =
  **THUA +6.3%/bước** (10.26 vs 9.65 ms) — paged *storage một mình* là khoản LỖ throughput. Đó là lý do
  Bài 1.7 tồn tại.

**Frontier / cổng.** Đây đúng là **PagedAttention của vLLM**. Cổng: "Paged KV cho dung lượng nhưng vì
sao *một mình nó* là mất throughput?" (gather thêm traffic + SDPA vẫn padded — chỉ kernel fused Bài 1.7
gỡ được). Trait = **subtract-before-add** (đo negative control trước khi tối ưu). Scarce bucket:
inference (differentiator hàng đầu 2026).

---

## Bài 1.5 · PrefillView — admission offset-0 bằng duck-typing

**Câu hỏi.** Một request MỚI (cả prompt) nạp vào slot trống thế nào mà KHÔNG cần viết một đường
attention riêng cho prefill?

**Sự thật nền tảng.** Prefill (nạp prompt) và decode (sinh 1 token) *nhìn* rất khác, nhưng nếu slot mới
bắt đầu ở **offset 0**, thì prefill CHÍNH LÀ một forward causal chuẩn với vị trí `0..s−1` — cùng mask
`q_pos ≥ k_pos`, cùng RoPE, **không toán attention mới**. Vấn đề chỉ còn là *routing* K/V vào đúng
storage của parent (slab hay paged) — một bài toán interface, không phải toán học.

**Dẫn xuất — duck-typing.** `PrefillView` *giả dạng* đúng interface mà attention layer dùng ở đường
1-request (`length`/`append`/`get`/`advance`):
- `length` trả **0 cứng** (slot tươi theo hợp đồng) ⇒ attention dựng causal mask chuẩn (model.py
  :327-333, nhánh `s>1`), KHÔNG phải row_mask ragged.
- `append` ghi qua `parent._write_prefill_kv` (dense: scatter rows; paged: scatter blocks) + lưu block
  để prefill tự-attend đúng khối của mình.
- `advance(s)` (LM gọi 1 lần sau mọi layer) kích hoạt slot ở **TRUE length** (không phải padded width):
  `parent.lengths[slots] = true_lengths_t; active=True` — device-only; scheduler mirror bằng
  `parent.mirror_admit`.

Mọi state python (slot list, width) resolve thành tensor/int **LÚC KHỞI TẠO** (ngoài graph) ⇒ `append`/
`advance` traced chỉ guard trên int ổn định. Trade: right-pad các hàng tới prompt rộng nhất (rác che
sau), đổi lấy "một forward batched nạp nhiều request cùng lúc".

**Neo code** (`src/scratch_llm/kv_cache.py` · `PrefillView` :456):
```python
def __init__(self, parent, slots, true_lengths):                                      # :475
    parent.reserve_prefill(list(slots), [int(x) for x in true_lengths])  # eager alloc # :487
    self._min_width = max(int(x) for x in true_lengths)  # int lúc init, ngoài graph  # :494
@property
def length(self): return 0   # fresh slots: prefill = offset-0 uniform-causal forward  # :497-499
def append(self, layer, k_new, v_new):                                                # :501
    self._parent._write_prefill_kv(layer, self._slots, k_new, v_new)                  # :505
    self._block[layer] = (k_new, v_new)   # prefill attends đúng block của mình        # :506
def advance(self, n):                                                                 # :511
    parent.lengths[self._slots] = self._true_lengths_t   # TRUE length, không padded   # :515
    parent.active[self._slots] = True                                                 # :516
```
Routing ở `model.py · MultiHeadSelfAttention.forward`: `isinstance(cache, SlotKVCache)` (:278) là nhánh
batched-decode; `PrefillView` KHÔNG phải `SlotKVCache` (nó *wrap* parent) ⇒ rơi vào nhánh `cache is not
None` :308 = đường 1-request `append`/`get` với `past_len=0`.

**Hình ảnh — admission batched, offset-0** (anchor: nạp 2 request len [4, 7] vào slot [0,1]):
```
prompts right-padded tới width=7:            attention = causal chuẩn (offset 0):
  slot0: [t0 t1 t2 t3 P  P  P ]  true_len=4    q_pos=0..6, k_pos=0..6, mask k≤q
  slot1: [u0 u1 u2 u3 u4 u5 u6]  true_len=7        (P = pad, che ở decode kế bởi write-then-mask)
             │
   append → parent._write_prefill_kv(layer, slots=[0,1], k=(2, H_kv, 7, d))
   advance(7) → parent.lengths[[0,1]] = [4, 7]  ← TRUE length (KHÔNG phải 7,7)
                parent.active[[0,1]] = True
```
**Vì sao activate ở TRUE length, không padded width?** Nếu kích hoạt ở width=7, slot0 (prompt thật 4)
sẽ khai báo length=7 → bước decode kế attend cả P P P (3 pad rác ở vị trí 4,5,6) → logits lệch. Kích
hoạt ở true_len=4 ⇒ mask `j ≤ 4` che pad. (Modify-and-predict gate.)

**Số đo THẬT.** `[measured · sm120 · RESULTS.md R3b]`: prefill = forward uniform-causal ⇒ **0 toán
attention mới**; oracle ragged **token-exact** (21 CPU tests). Bài học vận hành `[FACT]`: **prefill chạy
EAGER** — compile prefill = bẫy **47 graph** (mỗi số-request-admit một width mới → guard scan mỗi step,
~35s in-run); fix = prefill eager (~1% wall), decode giữ ngân sách compile.

**Frontier / cổng.** Duck-typing "prefill = forward thường" là vì sao continuous batching gọn: MỘT code
path forward lo cả prefill lẫn decode. Cổng: "Vì sao KHÔNG compile prefill?" (shape churn theo số request
admit). Trait = **composition over inheritance** (view wrap parent, không kế thừa storage). Scarce
bucket: inference/systems.

---

## Bài 1.6 · ChunkPrefillView — offset-continuation, prefill dài chia khúc

**Câu hỏi.** Một prompt DÀI (512 token) nạp một phát sẽ head-of-line block cả decode stream (spike ITL
p99). Chia nó thành khúc `C` token xen giữa các bước decode — làm sao KV vẫn **bit-identical** với
one-shot?

**Sự thật nền tảng.** Điều kiện để chia khúc "vô hại về số học": token tại vị trí `t` phải được xử lý
**y hệt** bất kể nó nằm ở khúc nào. RoPE dùng **vị trí TUYỆT ĐỐI** ⇒ token@t xoay tại góc `t·θ` bất kể
ranh giới khúc; K-projection@t độc lập với các vị trí khác. Nên `⌈L/C⌉` khúc nối đuôi cho ra **cùng
K/V** với một-phát — *nếu* mỗi khúc biết offset tuyệt đối của nó.

**Dẫn xuất — offset chạy.** Khác `PrefillView` (offset-0, batched nhiều slot), `ChunkPrefillView` nạp
MỘT slot ở **offset tăng dần**:
- `length` trả **offset đã ghi** = RoPE offset tuyệt đối của khúc (không phải 0!).
- `append` ghi khúc vào `[offset, offset+s)` qua `_write_prefill_kv_at`, rồi `get` trả toàn bộ prefix
  `[0, offset+s)` để khúc attend dưới causal mask `past_len` thường (không toán mới).
- Slot **reserved nhưng inactive** khi prefill (bị mask khỏi batch decode như mọi slot trống). Scheduler
  flip active sau khúc CUỐI (khúc cuối phát token đầu = TTFT).

**Neo code** (`src/scratch_llm/kv_cache.py` · `ChunkPrefillView` :519):
```python
@property
def length(self): return self._offset    # offset tuyệt đối, KHÔNG phải 0             # :554-556
def append(self, layer, k_new, v_new):                                                # :558
    s = k_new.shape[2]
    self._parent._write_prefill_kv_at(layer, self._slot, self._offset, k_new, v_new)  # :560  ghi [offset, offset+s)
    self._view[layer] = self._parent.slot_kv_view(layer, self._slot, self._offset+s)  # :561  prefix [0, offset+s)
def advance(self, n):                                                                 # :566
    self._parent.lengths[self._slot] = self._offset + n   # bump device length        # :570  (activation deferred)
```
RoPE absolute-pos ở `model.py`: với slot cache `positions = cache.lengths.unsqueeze(1)` (:449); RoPE
`forward` slice `cos[positions]` (:195) — token@t luôn xoay tại t.

**Hình ảnh — 1 prompt 12 token, C=4:**
```
one-shot:  append offset=0, s=12  → K[0:12]        (1 forward)
chunked:   khúc0 offset=0 s=4 → ghi K[0:4],  get prefix[0:4]
           khúc1 offset=4 s=4 → ghi K[4:8],  get prefix[0:8]   ← attend cả [0:8]
           khúc2 offset=8 s=4 → ghi K[8:12], get prefix[0:12]  ← khúc cuối → TTFT, scheduler flip active
  RoPE:  token@6 xoay tại góc 6·θ ở CẢ HAI đường (offset tuyệt đối) ⇒ K@6 y hệt
```
**Hand-trace** `[verified · CPU · tests/test_chunked_prefill.py]` (float64): `max|K_oneshot − K_chunked| = 1.33e-15`
(≈ machine-eps float64 — **bit-identical về số học**; nguồn `1e-15` là reduction-order, không phải logic).

**Số đo THẬT — VÀ VÌ SAO FALSIFIED.** `[measured · sm120 · RESULTS.md R4.2]`: cơ chế **token-exact**
(single-chunk KV `torch.equal`; float64 exact mọi chunk size; float32 khác chỉ là argmax tie-flip).
NHƯNG interleave tuần tự **REGRESS mọi metric**: ITL p50 **×6.56→14.44**, throughput **547→145 tok/s**
monotone giảm khi C↓, ITL p99 KHÔNG giảm (×1.03→1.91) — **FALSIFIED**. Root cause `[FACT]`: (1) admission
tuần tự — 14 batched prefills (one-shot) → 160–550 unbatched (chunked) → đói batch decode; (2) khúc là
forward tuần tự riêng ⇒ chi phí rơi vào MỌI khe decode (xấu ngay ở C=512=single-chunk ⇒ overhead
per-forward thuần, không phải slicing).

**Frontier / cổng.** Đây đúng là bài học **Sarathi-Serve / vLLM-V1 "stall-free batching"**: production
piggyback khúc prefill VÀO forward decode batched (một kernel ragged) — thêm compute, KHÔNG thêm launch,
KHÔNG khe idle. Cổng: "Vì sao chunked prefill phải là **fused kernel**, không phải scheduling-only?"
(R4.2b, deferred). Trait = **claims honesty** (đo, thấy REGRESS, ghi FALSIFIED thay vì che). Scarce
bucket: inference.

---

## Bài 1.7 · Triton paged-decode kernel — online softmax qua block table

**Câu hỏi.** Paged storage một mình là LỖ throughput (Bài 1.4, +6.3%). Kernel nào đọc KV **trực tiếp
qua bảng trang** — không gather, không materialize score, không repeat GQA — để GIÀNH LẠI thuế padding
1.27×?

**Sự thật nền tảng.** Đường gather (`decode_view`) trả **ba khoản thuế**: (1) copy gather pool→view; (2)
materialize `(B,H,view_len)` score fp32; (3) `repeat_interleave` nở K/V từ `n_kv` lên `n_heads` (GQA).
Cả ba là *padded compute* — làm việc trên cả token rác. Fuse tất cả vào MỘT launch, đọc chỉ token thật.

**Dẫn xuất — online softmax (FlashAttention recurrence).** Không thể materialize `softmax` toàn dãy nếu
đọc block-by-block. Online softmax giữ 3 carry fp32 và cập nhật khi thấy block mới. Với dãy score `s`,
chia thành block, sau khi đã xử lý tới max chạy `m`, tổng mũ `l`, tử số `acc`:

```
gặp block mới có score s_blk:
  m_new = max(m, max(s_blk))
  α     = exp(m − m_new)              ← "sửa" các carry cũ theo max mới
  p     = exp(s_blk − m_new)          ← masked lane: exp(−∞) = 0 chính xác
  acc   = acc·α + Σ p·v_blk           ← rescale tử số cũ + cộng đóng góp mới
  l     = l·α   + Σ p                 ← rescale mẫu số cũ + cộng mới
  m     = m_new
cuối:  out = acc / l
```
**Vì sao `α = exp(m − m_new)`?** Softmax đầy đủ = `Σ e^{sⱼ−M} vⱼ / Σ e^{sⱼ−M}` với `M` = max TOÀN dãy.
Khi mới thấy một phần, ta trừ max *tạm* `m`. Gặp block đẩy max lên `m_new`, mọi số hạng cũ đang trừ `m`
thay vì `m_new` ⇒ phải nhân `e^{−(m_new−m)} = e^{m−m_new} = α` để "hạ" chúng xuống chuẩn mới. Đây là toàn
bộ phép màu — cho phép streaming softmax **ổn định số** (không overflow) mà không cần biết trước max.

**GQA không repeat:** `kv_h = h // group_size` — query head `h` đọc thẳng KV head của nhóm mình, không
nở tensor. **No-NaN:** `n_keys = lengths[b] + 1 ≥ 1` luôn (write-then-mask self-inclusive) ⇒ ≥1 lane
valid ⇒ `l > 0` ⇒ `acc/l` không chia 0.

**Neo code** (`src/scratch_llm/kernels/paged_decode_triton.py` · `_paged_decode_kernel` :24):
```python
b = tl.program_id(0); h = tl.program_id(1)                                            # :49-50  grid (B, H_q)
kv_h = h // group_size            # GQA: KHÔNG repeat_interleave                       # :51
n_keys = tl.load(lengths_ptr + b) + 1    # write-then-mask self-inclusive              # :57
m = tl.full((1,), float("-inf"), dtype=tl.float32)   # carry fp32 tường minh           # :62
l_sum = tl.zeros((1,), dtype=tl.float32) ; acc = tl.zeros([HEAD_DIM], tl.float32)      # :63-64
for i in range(n_blocks):
    phys = tl.load(table_ptr + b*stride_tb + i*stride_ti)   # bảng trang               # :66
    k_blk = tl.load(pool_k_ptr + kv_ptrs).to(tl.float32)    # (BLOCK, D) đọc thẳng pool # :69
    scores = (tl.sum(k_blk * q[None,:], axis=1) * scale).to(tl.float32)   # (BLOCK,)   # :72
    valid  = (i*BLOCK + t_off) < n_keys                                                # :73
    scores = tl.where(valid, scores, float("-inf"))                                    # :74
    m_new = tl.maximum(m, tl.max(scores, axis=0))                                      # :75
    alpha = tl.exp(m - m_new) ; p = tl.exp(scores - m_new)                             # :76-77
    acc = acc*alpha + tl.sum(p[:,None]*v_blk, axis=0)                                   # :79
    l_sum = l_sum*alpha + tl.sum(p, axis=0) ; m = m_new                                # :80-81
out = acc / l_sum   # n_keys ≥ 1 ⇒ l_sum > 0, no NaN row                               # :82
```
Routing (`model.py` :286): `if isinstance(cache, PagedKVCache) and cache.use_kernel:` → gọi
`paged_decode_attention(q, pool_k[layer], pool_v[layer], block_table, lengths)` (:293), bypass SDPA
hoàn toàn (`kernel_out` :277, :315).
> **Code review — 2 bẫy triton-under-compile đã đo `[FACT]`.** (1) python-float carry `m=-inf` promote
> lên **f64** dưới type-inference chặt của `torch.compile` → "loop-carried re-assigned to fp64" fail
> compile; fix = carry fp32 tường minh `tl.full(..., tl.float32)` (:62). (2) `scale` (arg float runtime)
> bị inductor wrapper truyền **f64** → nhiễm cả chuỗi carry; fix = cast `.to(tl.float32)` ngay sau nhân
> (:72). Đây là loại bug "im lặng poison cả kernel" mà chỉ profiling/compile mới lộ.

**Hình ảnh — grid + online softmax qua page** (anchor `B=4, H_q=8, n_kv=2 ⇒ group_size=4`; slot b=1
len=20 ⇒ n_keys=21 ⇒ n_blocks=2):
```
grid (B=4, H_q=8):  program (b=1, h=5) → kv_h = 5//4 = 1  (đọc KV head 1)
  q : (B,H,d)=(4,8,128)  →  q[1,5,:] (128,) load fp32
  block_table[1] = [ phys=7, phys=9, 0, 0 ]     n_blocks = ⌈21/16⌉ = 2 → chỉ đi block 0,1
    i=0 phys=7: k_blk=pool[7,1,:,:] (16,128); scores (16,); valid = idx<21 → all T (idx 0..15)
    i=1 phys=9: k_blk=pool[9,1,:,:] (16,128); scores (16,); valid = 16..31<21 → T T T T F...F
                                                              ↑ lane 20 = key vừa ghi (self-incl); 21..31 = −∞
  carry qua 2 block:  m,l_sum,acc rescale bằng α mỗi block  →  out = acc/l_sum  (128,)
  store → out[1,5,:]                out : (B,H,d) fp32 → unsqueeze(2) → (B,H,1,d) → .to(q.dtype)
```
**Số đo THẬT.**
- `[reproduced · CPU]` online-softmax ≡ full-softmax (kernel thật do `tests/test_paged_kernel.py` gpu chốt): mô phỏng đúng recurrence trên (37
  key, 3 block 16-ô) → `max|online − ref| = 2.61e-7` (fp32 machine-eps) ⇒ recurrence ĐÚNG.
- `[measured · sm120 · RESULTS.md R4.1 P4.1.4]`: **5.90 ms/bước · ×3.52 vs wave-dense · 3,528 tok/s agg
  (+55% vs dense-continuous)**; ITL p50 6.2 ms ≈ wave 6.1. **Phân rã thuế 1.27×**: **~3.2 ms là compute**
  (fp32-score materialize + repeat_interleave + softmax-over-padding), **chỉ ~0.5 ms là byte** — đúng
  như spec đoán. Đây là insight: thuế padding chủ yếu là *compute padded*, không phải *byte padded*.

**Frontier / cổng.** Đây là trái tim **vLLM/FlashInfer paged decode** + là substrate DUY NHẤT R4.4
CUDA-graph capture được (grid `(B,H)` cố định, pool địa chỉ tĩnh). Cổng: "Vì sao decode attention phải
là kernel fused chứ không gather+SDPA?" — gỡ padded *compute* (3.2 ms), không chỉ byte (0.5 ms). Trait =
**roofline-first / predict-the-number** (dự đoán decomposition bytes-vs-compute TRƯỚC khi đo, trúng).
Scarce bucket: **kernels** = differentiator hàng đầu.

---

## Bảng số đo THẬT (chạy lại được — DoD-là-profile)

| Bài | Đo | Kết quả | Nguồn | Ý nghĩa |
|---|---|---|---|---|
| 1.0 | decode AI @B=1 (cfg 0.84B, ctx128) | **0.993 FLOP/B**; ridge 130.9; pred 324.9 tok/s | CPU repro (roofline.py) | memory-bound, 131× dưới ridge |
| 1.0 | decode tok/s @B=1 (0.84B bf16) | pred 315 → **eager 51 → compiled 173 → graph 253** (77% wall) | sm120 R1/R4.4 | thesis "in trend" |
| 1.1 | slab batching agg tok/s | **peak 12,220 @B=256 = 66×** B=1; AI≈B; crossover B≈128 | sm120 R3a | batch = đòn bẩy #1 |
| 1.1 | continuous vs static-wave | **2.30× wall / 2.93× steps**; padding tax **1.27×/bước** | sm120 R3b | động cơ của 1.4/1.7 |
| 1.1/1.3 | write-then-mask no-NaN | inactive row → **min=1 key visible** | CPU repro · tests/test_kv_cache.py | no-NaN by construction |
| 1.2 | truncate rollback prefix | length A=B=8; **max|K_A−K_B|=1.19e-7** (fp32; exact fp64) | CPU repro · tests/test_speculative.py | rollback bit-identical |
| 1.2 | speculative lossless | **token-exact** 27 tests; wrong-drafter vẫn exact; ×1.35–1.39 | sm120 R4.3 | lossless by construction |
| 1.3 | dynamo unique_graphs | **2** (sau vá 2 lỗ rò python-state-in-graph) | sm120 R3b | no recompile-storm |
| 1.4 | gather bit-exact (valid pos) | **0.000e+00** vs slab | tests/test_paged_cache.py | bảng trang đúng byte |
| 1.4 | frag + capacity | **frag 5.0%**; **capacity ×9.3** (7,040 vs 65,536) | sm120 R4.1 | reservation→fragmentation |
| 1.4 | paged-gather-only (control) | **THUA +6.3%/bước** (10.26 vs 9.65) | sm120 R4.1 | storage một mình = LỖ |
| 1.5 | prefill = uniform-causal | **0 toán mới**; 21 CPU tests token-exact; compile=47-graph bẫy | sm120 R3b | duck-typing offset-0 |
| 1.6 | chunked ≡ one-shot | **max|ΔK|=1.33e-15** (fp64 bit-identical) | tests/test_chunked_prefill.py | RoPE absolute-pos |
| 1.6 | chunked scheduling | **FALSIFIED**: ITL p50 ×6.56–14.44; tput 547→145 | sm120 R4.2 | phải là fused kernel |
| 1.7 | online-softmax ≡ full | **max|Δ|=2.61e-7** (fp32) | CPU repro · tests/test_paged_kernel.py (gpu) | FA2 recurrence đúng |
| 1.7 | fused paged kernel | **5.90 ms · ×3.52 vs wave · +55%**; tax 3.2ms compute / 0.5ms byte | sm120 R4.1 | giành lại padded-compute |

---

## Checklist recall COLD (che phần trên, tự trả lời — đây là cách re-learn)

1. **1.0** Dẫn `AI ≈ 1` cho decode @B=1 từ FLOP/byte. Ridge sm120 = ? (từ đâu). Ba đòn bẩy kéo AI lên?
   Vì sao H200 là "bản nâng cấp decode" của H100?
2. **1.1** Vì sao slab phải địa-chỉ-tĩnh (cái gì vỡ nếu `cat`)? Write-then-mask là gì — vì sao *ghi rác
   rồi che* thay vì *skip*? Dấu `≤` trong mask: nêu ĐỦ HAI hậu quả nếu đổi `<`.
3. **1.2** Vì sao `.contiguous()` trong `truncate` là cần không thừa? Speculative decoding lossless nhờ
   đâu (không phải xác suất)? Vì sao K giữ lại bit-identical với decode thường?
4. **1.3** Vì sao đọc `max(py_lengths)` TRONG forward là bẫy `torch.compile`? Hợp đồng graph-owned vs
   scheduler-owned gồm những state nào? `advance` bump hàng nào, mấy lần/forward?
5. **1.4** Vì sao block 0 = trash làm write-then-mask "chuyển sang paged miễn phí"? Dẫn `frag ≈ E[tail]/⟨ℓ⟩`.
   CoW suy biến thành "copy-never" như thế nào (biên block)?
6. **1.4** Paged cho ×9.3 dung lượng nhưng vì sao *một mình nó* là THUA throughput (+6.3%)?
7. **1.5** Vì sao `length` trả 0 cứng làm prefill "miễn phí về toán attention"? Vì sao activate ở TRUE
   length không padded width? Vì sao KHÔNG compile prefill?
8. **1.6** Vì sao RoPE-vị-trí-tuyệt-đối là điều kiện để chunked KV bit-identical? Cơ chế token-exact
   nhưng scheduling FALSIFIED — root cause? Production giành lại win bằng cách nào?
9. **1.7** Dựng lại online softmax từ softmax đầy đủ — vì sao cần `α = exp(m − m_new)`? Nếu bỏ `+1` ở
   `n_keys = lengths[b]+1` thì hàng nào rơi vào `l_sum=0` → NaN? Vì sao kernel KHÔNG repeat_interleave GQA?
10. **1.7** Thuế padding 1.27× phân rã bao nhiêu compute / bao nhiêu byte? Insight rút ra là gì? Vì sao
    paged kernel là substrate DUY NHẤT CUDA-graph capture được?

> Trả lời cold được cả 10 = **S1 thật sự OWNED** (interview-grade serving). Vấp câu nào → mở đúng mục đó,
> hoặc blank-slate hàm tương ứng (`raise NotImplementedError` → test đỏ → tự dẫn → xanh) để re-own bằng tay.

---

*Cross-ref: `docs/learning/roadmap/S1_serving_substrate.md` (bản đồ mổ xẻ) · `docs/learning/serving/01-batched-kv-cache.md`
(bài đầy đủ 1.1) · `bench/RESULTS.md` (ledger R1/R3a/R3b/R4.1/R4.2/R4.3/R4.4) · `docs/learning/derivations/M2_transformer_forward.md`
(GQA/RoPE — nguồn của KV-cache byte math) · `docs/learning/PROGRESS.md` (ledger 89 Bài). Série kế:
S2 serving engines (scheduler, speculative, CUDA-graph, MLA, PD-disagg).*
