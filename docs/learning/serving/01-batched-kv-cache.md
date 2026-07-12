# Bài 1 — `BatchedKVCache`: bộ đệm slot tĩnh và nguyên tắc write-then-mask

> **Series 1 · Serving substrate** · số dòng re-pin theo commit **`d7e4603`** (2026-07-05)
> Code: `src/scratch_llm/model.py` — `SlotKVCache` (:255), `BatchedKVCache` (:409)
> Tests găm bất biến: `tests/test_batched_cache.py` · Số đo gốc: `bench/RESULTS.md` §A1-R3b
> Tiên quyết: hiểu attention + KV-cache đơn request (`model.py:203 KVCache`, spec
> `docs/design/L2_kv_cache_SPEC.md`).

---

## 1. Feynman trước — bài toán bằng lời, không code

Tưởng tượng một **khách sạn có B=3 dãy tủ khóa**, mỗi dãy `max_ctx=8` ô đánh số `0..7`. Mỗi
*request* (khách) thuê **một dãy**; mỗi lượt sinh token, khách bỏ **một món đồ (cặp K,V)** vào
**ô kế tiếp** của dãy mình. Sổ lễ tân ghi hai thứ: `lengths[b]` = dãy `b` đã dùng mấy ô,
`active[b]` = dãy có khách hay trống.

Bài toán lớp này giải: **khách đến/đi lúc khác nhau, độ dài khác nhau** (*ragged batch*). Trước
R3b, `KVCache` cũ (`model.py:202`) chỉ có **một** con số `length` chung cho cả batch và nới kho
bằng `torch.cat` — nghĩa là (a) mọi khách buộc cùng độ dài, (b) địa chỉ kho **đổi liên tục**
(chính `cat` này làm CUDA-graph capture fail ở rung R1 — xem `bench/RESULTS.md` §A1-R1). Lớp mới
trả lời: kho **cấp phát một lần, địa chỉ cố định**, mỗi hàng một **con trỏ riêng**.

Và một quyết định nghe ngược đời: **dãy trống cũng bỏ đồ (rác) vào tủ mỗi lượt**, rồi dùng
**tấm rèm (mask)** che để không ai nhìn thấy rác. Ghi-trước-che-sau = **write-then-mask**.
Vì sao không "bỏ qua hàng trống"? Vì bỏ qua = *shape đổi theo bước* → `torch.compile` /
CUDA-graph vỡ. Ghi thừa một ô rác thì rẻ; đổi shape thì đắt. **Đây là đánh đổi số 1 của cả hệ.**

## 2. Code thật — đúng file, đúng dòng, đúng thứ tự chạy

Cấu trúc lớp sau refactor R4.1 (`src/scratch_llm/model.py`):

```
SlotKVCache (:240)  ── "sổ lễ tân" dùng chung: lengths/active + mirror python + view_len
   ├── BatchedKVCache (:380) ── kho DÀY ĐẶC: mỗi slot một dãy liền max_ctx ô   ◄ BÀI NÀY
   └── PagedKVCache  (:419)  ── kho PHÂN TRANG: block 16 ô + bảng ánh xạ       (bài 5)
```

### 2a. Trạng thái — hai bộ sổ song song (thiết kế, không phải thừa)

```python
# SlotKVCache.__init__ (model.py ~:270)
self.lengths = torch.zeros(n_slots, dtype=torch.long, device=device)  # sổ TRÊN GPU
self.active  = torch.zeros(n_slots, dtype=torch.bool, device=device)
self.py_lengths: list[int]  = [0] * n_slots                           # sổ PYTHON (mirror)
self.py_active:  list[bool] = [False] * n_slots
self._view_len = 1                                                    # int thường — §2d
```

**Hợp đồng sở hữu** (docstring `model.py:240`): tensor GPU do **forward** (vùng compile) chạm;
bản python do **scheduler** cập nhật *ngoài* đồ thị — `mirror_advance` (:361), `mirror_admit`
(:371). Lý do *đo được*: nếu forward đọc list python, Dynamo "nướng" giá trị thành guard; batch
ragged đổi liên tục → recompile-storm → rơi về eager (trận đánh đầy đủ ở bài 4). Nhớ một câu:
**GPU-tensor của đồ thị, python-list của scheduler, không đọc chéo trong forward.**

### 2b. Ghi — `write_decode` (base :340) → `_write_decode_kv` (dense :405)

```python
# model.py:340 (SlotKVCache) — kiểm shape rồi giao storage hook
pos = self.lengths.clamp(max=self.max_ctx - 1)     # (B,) — MỖI HÀNG MỘT VỊ TRÍ
self._write_decode_kv(layer, pos, k_new, v_new)

# model.py:405 (BatchedKVCache) — một lệnh làm "B con trỏ khác nhau"
self._k[layer][self._slot_idx, :, pos] = k_new[:, :, 0]
```

Dòng cuối là **advanced indexing** đáng học thuộc: `_k[layer]` shape `(B, H_kv, max_ctx, d)`;
index dim 0 bằng `_slot_idx=[0..B-1]`, dim 2 bằng `pos` (đều tensor `(B,)`), dim 1 giữ `:`.
Quy tắc PyTorch: *hai advanced index không kề nhau → chiều broadcast của chúng nhảy lên ĐẦU* →
vế trái shape `(B, H_kv, d)`, khớp `k_new[:, :, 0]`. Một lệnh, B lần ghi vào B ô khác nhau —
không vòng lặp python, không host-sync.

### 2c. Đọc — `decode_view` (:409): "cửa sổ nhìn", 0 byte copy

```python
length = self.view_len                                                 # int python — §2d
return self._k[layer][:, :, :length], self._v[layer][:, :, :length]   # VIEW, không copy
```

### 2d. `view_len` (:316) — vì sao là int python do mirror nuôi

`view_len = min(max_ctx, 1 + max(py_lengths))` — cửa sổ phải phủ **cả ô vừa ghi** của hàng dài
nhất (`+1` vì ghi tại `lengths[b]`). Chỉ được tính lại trong `_recompute_view_len` (:323), gọi từ
`mirror_advance` / `mirror_admit` / `free_slot` (:330) — toàn chỗ của scheduler. Nếu tính
`max(py_lengths)` *ngay trong* forward → Dynamo guard theo **thứ tự phần tử** của list
(`py_lengths[26] > py_lengths[16]` — lỗi thật đã đo, log ở `bench/RESULTS.md` §R3b findings) →
storm. Int có sẵn thì Dynamo tự nâng thành SymInt động — êm.

### 2e. Nhịp thời gian một bước decode — ai gọi gì, thứ tự nào

```
             │  [scheduler — python, NGOÀI đồ thị]      [forward — TRONG đồ thị]
bước t       │
 evict/admit │  free_slot(b) / mirror_admit(...)
 reserve     │  cache.pre_decode_reserve()  (:310 — dense: no-op; paged: bài 5)
 forward ────┼───────────────────────────────►  layer 0: write_decode tại pos=lengths
             │                                           decode_view + mask(j ≤ lengths)
             │                                  layer 1: write_decode — CÙNG lengths!
             │                                  ...
             │                                  advance(1) (:353): lengths += active  ← MỘT lần
 mirror ─────┼──  cache.mirror_advance() (:361): py_lengths+=1 (hàng active), view_len tính lại
```

Bất biến then chốt: **mọi layer trong một forward thấy CÙNG một `lengths`** — `advance(1)` chạy
một lần sau layer cuối (`TransformerLM.forward` gọi; hợp đồng y hệt `KVCache.advance` cũ :236).
Và `advance` chỉ cộng hàng active: `self.lengths += self.active.long()` — hàng trống ghi rác tại
ô 0 mãi, con trỏ không nhích.

## 3. Ví dụ số tay — B=3, H_kv=1, max_ctx=8, d=2 (bút và giấy)

Trạng thái trước bước t: `lengths=[3,0,5]`, `active=[T,F,T]`, `view_len = 1+max(3,0,5) = 6`.
Token mới chiếu ra K: hàng 0→`[10,11]`, hàng 1→`[20,21]` (rác — hàng trống), hàng 2→`[30,31]`.

**Ghi** — `pos = [3,0,5]`, một lệnh advanced-indexing đặt:

```
              ô:   0        1      2      3         4      5         6   7
slot 0 (len 3): [k00]   [k01]  [k02]  [★10,11]   [ · ]  [ · ]     [·] [·]
slot 1 (len 0): [★20,21][rác]  [rác]  [rác]      [rác]  [rác]     [·] [·]
slot 2 (len 5): [k20]   [k21]  [k22]  [k23]      [k24]  [★30,31]  [·] [·]
                                 ★ = vừa ghi bước này, tại đúng lengths[b]
```

**Đọc** — `decode_view` trả `(3, 1, 6, 2)` (cắt tới `view_len=6`, là *view*, 0 byte copy).

**Rèm che** — attention dựng mask `j ≤ lengths[b]` (`model.py:825–826`; chi tiết đường attention
ở bài 2):

```
                 j=0  1  2  3  4  5
slot 0 (len 3):  [✓][✓][✓][✓][✗][✗]   3 ô cũ + Ô VỪA GHI (j=3): token mới TỰ THẤY MÌNH
slot 1 (len 0):  [✓][✗][✗][✗][✗][✗]   chỉ thấy đúng cục rác vừa ghi → softmax CÓ 1 phần tử
slot 2 (len 5):  [✓][✓][✓][✓][✓][✓]   5 cũ + 1 mới
```

Đây là "**không-NaN theo cấu trúc**": nếu hàng 1 bị che *hết* (all ✗), mọi score = −∞ →
softmax = 0/0 = **NaN**, NaN luồn qua `o_proj` làm bẩn cả bước. Vì hàng nào cũng *vừa ghi* một ô
và mask *luôn cho thấy ô đó* (dấu `≤`, không phải `<`), mỗi hàng softmax có ≥1 số hữu hạn.
Output hàng trống là rác-hữu-hạn → scheduler vứt. Không cần một dòng `if` đặc biệt nào.

**Sau forward** — `advance(1)`: `lengths = [3,0,5] + [1,0,1] = [4,0,6]`. Scheduler gọi
`mirror_advance()`: `py_lengths=[4,0,6]`, `view_len=7`. Ô rác của slot 1? Bước sau **ghi đè
chính nó** tại ô 0 — không rò rỉ, không tích tụ.

**Giá phải trả (mồi cho bài 5):** kho giữ `3 × 8 = 24` ô, chỉ dùng `4+0+6=10`; mọi hàng đọc/tính
trên cửa sổ rộng `view_len` của **hàng dài nhất**. Trên bench thật (B=32, max_ctx=2048): ~95% chỗ
"đặt cọc" không dùng + **thuế padding 1.27×/bước** (9.65 vs 6.3 ms — đo ở R3b). Hai con số đó là
*toàn bộ lý do tồn tại* của `PagedKVCache` + kernel Triton (bài 5–6, nơi 9.65 ms/bước → 5.90).

## 4. Test nào găm bất biến nào (`tests/test_batched_cache.py`)

| Bất biến | Test |
|---|---|
| Ghi đúng ô riêng từng hàng | `test_write_decode_lands_at_per_row_offsets` |
| Cửa sổ phủ ô vừa ghi | `test_view_len_covers_every_written_key` |
| Chỉ hàng active nhích; mirror khớp device | `test_advance_bumps_only_active_rows` |
| Ragged batch == decode đơn từng request, token-chính-xác | `test_ragged_batched_decode_matches_single_stream` |
| Đổ rác slot trống không đổi 1 bit logits hàng active | `test_padding_invariance_under_poisoned_slots` |
| Trường hợp suy biến (đều độ dài) == đường R3a cũ | `test_uniform_lengths_match_r3a_static_batch` |

## 5. Cổng teach-back (chưa dạy lại được thì chưa sang bài 2)

1. **Giải thích bằng lời của bạn:** vì sao hàng *trống* vẫn ghi K/V mỗi bước thay vì "skip"?
   Câu trả lời phải chạm được hai chữ: *shape* và *capture*.
2. **Vì sao dấu `≤` trong mask là sống-còn?** Nêu đủ **hai** hậu quả nếu đổi thành `<`.
3. **Modify-and-predict** (dùng đúng ví dụ §3): nếu chuyển `advance(1)` chạy **sau mỗi layer**
   thay vì một lần cuối forward — với `lengths=[3,0,5]` ban đầu, layer 1 ghi K/V hàng 0 vào ô số
   mấy, mask của nó nhìn tới đâu, và lệch gì so với layer 0?

<details>
<summary>Phác đáp án (chỉ mở SAU khi đã tự trả lời)</summary>

1. Skip hàng trống ⇒ batch co giãn theo bước ⇒ shape động: `torch.compile` recompile liên tục
   (guard theo shape) và CUDA-graph (R4.4) không capture nổi vì graph cần shape + địa chỉ tĩnh.
   Ghi rác 1 ô/bước là chi phí ~0 so với việc giữ **toàn bộ đồ thị tĩnh**.
2. (a) Token mới không tự thấy mình: attention của query hiện tại thiếu đúng key/value của chính
   nó ⇒ sai số học (mất số hạng self-attention của vị trí hiện tại). (b) Hàng trống `lengths=0`:
   `j < 0` không có j nào ⇒ hàng mask all-False ⇒ softmax(−∞ toàn hàng) = NaN ⇒ NaN lan qua
   `o_proj`. Dấu `≤` cho mỗi hàng *ít nhất* một ô hợp lệ — chính ô vừa ghi.
3. Layer 0 ghi hàng 0 tại ô 3, rồi `advance` đẩy `lengths[0]=4` ⇒ layer 1 ghi tại **ô 4** (lệch
   1 so với layer 0) và mask layer 1 cho nhìn `j ≤ 4` — tức là nhìn thêm ô 3... nhưng K/V *của
   layer 1* tại ô 3 là dữ liệu CŨ/không tồn tại (layer 1 chưa từng ghi ô 3 ở bước này) ⇒ mỗi
   layer một hệ tọa độ khác nhau, cache giữa các layer lệch pha 1 ô ⇒ logits sai ngay bước đó.
   Đây chính là lý do hợp đồng "advance MỘT lần, sau TẤT CẢ layer" tồn tại từ `KVCache` gốc.
</details>

---

*Tiếp theo:* **Bài 2 — RoPE per-row + per-row mask trong `MultiHeadSelfAttention.forward`
(`model.py:774–860`)**: vì sao mỗi hàng cần *góc xoay riêng*, `positions =
cache.lengths.unsqueeze(1)` (:936) làm điều đó thế nào — kèm ví dụ xoay vector 2-D bằng tay.
