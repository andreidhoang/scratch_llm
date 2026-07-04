# SÉRIE 1 — A1 serving SUBSTRATE (bộ máy bộ nhớ KV-cache)

> **Số dòng pin theo commit `9e61d7a`.** Roadmap = bản đồ mổ xẻ + trace để dẫn đường cho teach-back
> sâu về sau, KHÔNG phải bài giảng đầy đủ mỗi mục. Mỗi Bài ~một màn hình, tự chứa.
> Số đo: `bench/RESULTS.md` (R1/R2/R3b/R4.1/R4.2/R4.3/R4.4). Code: `src/scratch_llm/`.

**Vì sao série này.** Decode một token là bài toán *memory-bound*: mỗi bước đọc TOÀN BỘ trọng số +
TOÀN BỘ KV-cache chỉ để sinh 1 token → arithmetic intensity AI ~ 1 FLOP/byte, thấp hơn ridge ~136 lần
(Bài 1.0). Cả série là *một mũi tên duy nhất*: mọi cấu trúc dữ liệu KV-cache ở đây tồn tại để **kéo AI
đó về phía trần** — batch để đọc trọng số một lần cho B request (slab dày đặc, Bài 1.1), rồi bỏ lãng phí
reservation của slab (paged, Bài 1.4), rồi giành lại "thuế padding" bằng kernel fused (Bài 1.7). Ta đi từ
cái đơn giản nhất (cache 1-request, Bài 1.2) lên cái phức tạp nhất, mỗi bài tựa lên bài trước.

Thứ tự đọc: **1.0 (vì sao) → 1.1 (câu trả lời cụ thể: slab, đã có bài sâu) → 1.2 tổ tiên B=1 → 1.3 base
class chung → 1.4 sibling paged → 1.5/1.6 admission → 1.7 kernel giành lại thuế.**

---

## Bài 1.0 — Roofline thesis: decode là memory-bound, AI ~ 1 (`src/scratch_llm/bench/roofline.py` · `decode_step_flops_bytes` :92 · `gpu_specs.py` `rtx4000-blackwell` :86)
> **Câu hỏi first-principles:** vì sao decode 1 token ở B=1 KHÔNG nhanh hơn dù GPU còn thừa cả núi FLOP/s?
> **Số đo (aha):** B=1 predicted 315 tok/s (trần bộ nhớ) → **đo 51 tok/s = 16% roof**; AI ≈ 0.96, **thấp
> hơn ridge 136×**; đạt 89 GB/s = **16% của 550 GB/s** HBM; ~955 kernel launch/token (R1).

**1. Feynman — bài toán bằng lời.** Roofline = "hai trần trong một phòng": trần compute (FLOP/s tối đa) và
trần bandwidth (byte/s × AI). Điểm giao = *ridge* (sm120 đo được ≈131 FLOP/byte). Một op nằm bên TRÁI ridge
thì bị bandwidth chặn: tăng FLOP/s không giúp gì, vì thời gian bị *đọc byte* quyết định. Decode 1 token là ví
dụ kinh điển: bạn đọc mọi trọng số (n_params byte) để làm đúng 2·n_params FLOP → AI ≈ 1. Đánh đổi cốt lõi của
cả série: **decode nghèo AI vì query-length = 1**; muốn giàu AI phải nhồi thêm việc (batch) vào cùng một lần
đọc trọng số.

**2. Cơ chế (derive từ đầu).** `decode_step_flops_bytes` (:110–113): `flops = 2·n_params·batch`;
`bytes = n_params·weight_bytes + 2·n_layers·n_kv_heads·head_dim·context_len·batch·kv_bytes`. Ở B=1, ctx ngắn:
bytes ≈ trọng số, `AI = 2·n_params / (n_params·2) = 1` (bf16 2 byte). `roofline()` (:38) tính
`attainable = min(peak, AI·bandwidth)`; AI=1 ≪ ridge 131 ⇒ bound="memory", `predicted_seconds = flops/attainable`.
Điểm mấu chốt: `attainable = 1·0.55e12 = 0.55 TFLOP/s` — chỉ 0.76% của 72 TF/s peak. GPU *đói byte*, không đói FLOP.

**3. Trace code.** `gpu_specs.py:86` `rtx4000-blackwell`: hbm 0.55e12, bf16 peak 72e12 (đo thật 2026-06-29, KHÔNG
phải datasheet) → `ridge_point()` (:38) = 72e12/0.55e12 ≈ 131. `decode_step_flops_bytes()` đếm byte → `roofline()`
(:38) phán bound + số giây → `predicted_tokens_per_s` (:33) = 1/giây. R1 đo 51 tok/s: *workload* memory-bound
(AI≈1) nhưng *run* eager lại bị **overhead-bound** — 89 GB/s = 16% HBM, 84% thời gian là ~955 launch/token, KHÔNG
phải byte. Đó là nghịch lý phải nuốt: memory-bound là trần lý thuyết, launch overhead che mất nó.

**4. Cổng teach-back.** (a) Giải thích vì sao AI của decode ≈ 1 mà của GEMM lớn thì ≫ ridge — chạm được chữ
"query-length" và "chia sẻ dimension". (b) *Modify-and-predict:* nếu batch B=32 (ctx ngắn), AI mới ≈ bao nhiêu,
và bound đổi sang gì? (Gợi: AI≈B khi KV còn nhỏ; B=128≈ridge → crossover sang compute — R3a đo đúng B≈128.)

**5. Frontier.** Đây là lý do H200 (cùng die compute H100 nhưng HBM 4.8 TB/s) là "bản nâng cấp decode": tăng
bandwidth, không tăng FLOP. Câu interview: "Decode bound bởi gì, và ba đòn bẩy nào kéo AI lên?" (batch, KV-quant,
paged/GQA/MLA — tất cả giảm byte hoặc chia sẻ đọc trọng số).

---

## Bài 1.1 — BatchedKVCache: slab dày đặc + write-then-mask  ▸ **BÀI ĐẦY ĐỦ ĐÃ VIẾT**
> **Đã có bài sâu, KHÔNG lặp lại ở đây:** `docs/learning/serving/01-batched-kv-cache.md`.
> Code: `model.py` · `BatchedKVCache` (:408), base `SlotKVCache` (:254).
> **Số đo (aha):** continuous vs static-wave **2.30× wall / 2.93× theo bước** (util 24.9%→72.8%); trần
> throughput slab **12,220 tok/s @B=256 = 66×** B=1 (R3a); **thuế padding 1.27×/bước** (9.65 vs 6.3 ms).

**Vị trí trong série (đọc bài kia rồi quay lại):** slab là *câu trả lời cụ thể đầu tiên* cho Bài 1.0 — cấp phát
một buffer `(n_slots, n_kv_heads, max_ctx, head_dim)` cố định địa chỉ, ghi mỗi hàng tại offset riêng
(`_write_decode_kv` :433), che bằng mask per-row `j ≤ lengths[b]` (`model.py:824`). Địa chỉ cố định = điều kiện
để R4.4 CUDA-graph capture (cái mà cache `torch.cat` cũ ở Bài 1.2 phá vỡ). Nhưng slab trả giá: giữ `max_ctx` mỗi
slot dù dùng bao nhiêu (**reservation waste ~95%**) + đọc view rộng bằng hàng dài nhất (**thuế padding 1.27×**).
Hai con số đó là *toàn bộ động cơ* của Bài 1.4 (paged) và Bài 1.7 (kernel). Bài 1.3 mổ base class `SlotKVCache`
mà bài này để dành.

---

## Bài 1.2 — KVCache: cache 1-request + `truncate()` rollback (`model.py` · `KVCache` :202 · `truncate` :239)
> **Câu hỏi first-principles:** dạng đơn giản nhất của KV-cache trông thế nào, và vì sao chính nó *phá* CUDA-graph?
> **Số đo (aha):** speculative decoding trên nền `truncate` → **lossless token-exact**, **×1.35–1.39 tok/s**,
> 1.3–1.5 token/target-forward; drafter cố tình SAI vẫn ra đúng chuỗi greedy ⇒ rollback sạch (R4.3).

**1. Feynman — bài toán bằng lời.** Đây là "cuốn sổ nối đuôi": mỗi bước decode, K/V mới `torch.cat` vào đuôi
(`append` :226). Một `length` chung (:214), bump một lần/forward sau mọi layer (`advance` :236). Đơn giản, đúng —
nhưng `cat` **cấp phát tensor MỚI mỗi bước** → địa chỉ HBM đổi liên tục. Đánh đổi: dễ viết, nhưng địa chỉ động là
kẻ thù số 1 của CUDA-graph (graph cần địa chỉ tĩnh) — chính lỗi R1 `accessing tensor output of CUDAGraphs
overwritten`. Đó là lý do cả họ `SlotKVCache` (Bài 1.3) ra đời: buffer cố định, ghi in-place thay vì `cat`.

**2. Cơ chế (derive từ đầu).** `truncate(length)` (:239–251) là primitive rollback cho speculative decoding
(R4.3): sau một forward xác minh ghi thêm K+1 vị trí draft, các vị trí bị *từ chối* phải bị vứt để không attend
lại. Bất biến: `0 ≤ length ≤ self._length` (chỉ co, không giãn); mỗi layer cắt `k[:, :, :length].contiguous()`
rồi đặt lại `_length`. Vì K/V của phần giữ lại là *bit-identical* với decode thường (RoPE xoay theo vị trí tuyệt
đối, không phụ thuộc draft), rollback + re-decode cho chuỗi y hệt — đó là chứng minh lossless.

**3. Trace code.** `append` (:226) → nếu trống thì gán, else `torch.cat([past, new], dim=2)`. `advance` (:236)
cộng `_length`. `truncate` (:239) lặp mọi layer, cắt `[:, :, :length]`, `.contiguous()` (ép cấp phát lại để bỏ
đuôi), set `_length=length`. Trong speculative (`serving/speculative.py`): draft K vị trí → verify → nếu chấp
nhận m<K+1 thì `cache.truncate(base+m)` cuốn về, đúng bằng "pending-token invariant". Test wrong-drafter vẫn exact
⇒ đường rollback không rò rỉ K/V rác.

**4. Cổng teach-back.** (a) Vì sao `.contiguous()` trong `truncate` là cần, không thừa? (Gợi: view cắt vẫn giữ
storage cũ; muốn *giải phóng* đuôi và có địa chỉ sạch cho bước sau phải copy.) (b) *Modify-and-predict:* nếu bỏ
`truncate` và chỉ giảm `_length` (không cắt tensor) — speculative còn lossless không? Ghi/attend gì sai ở bước kế?

**5. Frontier.** `truncate` là nền của MỌI drafter-family (Medusa/EAGLE/MTP): tất cả tối ưu E[accept], nhưng đều
cần một KV-rollback đúng. Câu interview: "Speculative decoding lossless nhờ đâu?" — greedy-verify + rollback, không
phải xác suất.

---

## Bài 1.3 — SlotKVCache: base class chung — write-then-mask, mirror py/device, view_len (`model.py` · `SlotKVCache` :254)
> **Câu hỏi first-principles:** làm sao decode một *ragged batch* (độ dài khác nhau, đến/đi khác lúc) trong
> lockstep mà không hàng nào rò sang hàng khác, và KHÔNG gây recompile-storm cho `torch.compile`?
> **Số đo (aha):** sau khi sửa 2 lỗ rò dynamo, **unique_graphs = 2** (trước đó: guard theo thứ tự list →
> recompile-limit → eager fallback chỉ ở nhánh continuous, R3b findings).

**1. Feynman — bài toán bằng lời.** `SlotKVCache` là "sổ lễ tân" trừu tượng hóa RA KHỎI Bài 1.1: nó giữ
`lengths`/`active` (con trỏ + trạng thái mỗi slot) và *hai bản sổ song song* — tensor GPU (do forward chạm, trong
graph) và list python `py_lengths`/`py_active` (do scheduler chạm, ngoài graph). Storage để cho subclass
(`BatchedKVCache` dày đặc / `PagedKVCache` phân trang). Đánh đổi cốt lõi: **ghi rác vào slot trống mỗi bước rồi
che (write-then-mask)** thay vì skip — vì skip = shape động = compile/graph vỡ.

**2. Cơ chế (derive từ đầu).** *Write-then-mask:* `write_decode` (:368) ghi mọi hàng tại `pos =
lengths.clamp(max=max_ctx-1)` (:378); mask attention nhận `j ≤ lengths[b]` (`model.py:824`, dấu `≤` không phải `<`)
→ mỗi hàng thấy ÍT NHẤT ô vừa ghi ⇒ softmax không bao giờ all-−∞ ⇒ **không-NaN theo cấu trúc**. *view_len:*
`view_len = min(max_ctx, 1 + max(py_lengths))` (`_recompute_view_len` :351) — cửa sổ phải phủ ô vừa ghi của hàng
dài nhất. Bất biến sống-còn: view_len là **int python thường**, chỉ tính lại bởi mirror-ops của scheduler
(`mirror_advance` :389, `mirror_admit` :399, `free_slot` :358) — KHÔNG BAO GIỜ tính `max()` trong forward.

**3. Trace code — hợp đồng sở hữu graph/scheduler.** Một bước decode: scheduler gọi `pre_decode_reserve` (:338,
dense no-op) NGOÀI graph → forward gọi `write_decode` (:368) → `_write_decode_kv` (subclass) rồi `advance(1)` (:381)
`lengths += active.long()` (device-only, một lần sau mọi layer) → scheduler gọi `mirror_advance` (:389) bump
`py_lengths` + `_recompute_view_len`. Lỗ rò đã đo (R3b): nếu forward đọc `max(py_lengths)`, Dynamo nướng guard theo
*thứ tự phần tử* (`py_lengths[26] > py_lengths[16]`); ragged churn hoán vị → recompile-limit → eager fallback. Sửa:
view_len là attribute int, chỉ scheduler nuôi → unique_graphs 2.

**4. Cổng teach-back.** (a) Vì sao dấu `≤` trong mask là sống-còn — nêu ĐỦ HAI hậu quả nếu đổi thành `<`? (token
mới không tự thấy mình; hàng trống len=0 → mask all-False → NaN.) (b) *Modify-and-predict:* nếu tính `view_len` NGAY
TRONG forward bằng `1 + cache.lengths.max().item()` — token vẫn đúng, nhưng đo được gì hỏng ở nhánh continuous mà
nhánh wave (độ dài đều) lại giấu? (Gợi: `.item()` sync + guard theo ordering → storm chỉ khi list hoán vị.)

**5. Frontier.** "Ownership contract" graph-vs-scheduler này chính là kiến trúc vLLM-V1 (scheduler python thuần,
forward là graph tĩnh). Câu interview: "Vì sao trạng thái python đọc trong forward là bẫy với torch.compile?"

---

## Bài 1.4 — PagedKVCache: block pool + block_table + CoW + guard block committed (`model.py` · `PagedKVCache` :457)
> **Câu hỏi first-principles:** slab lãng phí ~95% (mỗi slot đặt cọc `max_ctx` dù prompt ngắn) — làm sao biến
> lãng phí *reservation* thành phân mảnh *nhỏ* (≤15 token/hàng) mà vẫn giữ shape tĩnh cho graph?
> **Số đo (aha):** **frag 5.0%** vs reservation waste ~95%; **capacity ×9.3** (peak-alloc 7,040 token vs 65,536
> reserved). NHƯNG paged-gather-only (chưa có kernel) là **THUA +6.3%/bước** (10.26 vs 9.65 ms) — negative control (R4.1).

**1. Feynman — bài toán bằng lời.** KV-cache như *bộ nhớ ảo*: thay vì mỗi slot một dãy liền `max_ctx` ô, ta có
**pool các block cố định 16 ô** (`BLOCK=16` :482) + một *bảng trang* `block_table[b, i]` ánh xạ block logic i của
slot b → block vật lý. Cấp phát *on-demand*: chỉ mượn block mới khi length vượt bội số 16. Lãng phí sụp từ
"reservation" (max_ctx−ℓ) xuống "internal fragmentation" (≤15 ô ở block cuối, E≈8). Đánh đổi: đổi *gather* (đọc rải
rác qua bảng) lấy *dung lượng* — và gather-không-kernel là một khoản LỖ throughput (Bài 1.7 mới giành lại).

**2. Cơ chế (derive từ đầu).** *Trash block:* block 0 dành riêng làm rác (:497); mọi entry chưa cấp phát = 0 → trỏ
về block 0 → write-then-mask của Bài 1.3 vẫn đúng nguyên văn (ghi rác vào trash, mask che). *Allocator:* free-list
`_free` (:506, block 0 không bao giờ cấp), `_refcount` (:507). `_write_decode_kv` (:597): `blk_idx = pos//16`,
`offset = pos%16`, `phys = block_table[slot_idx, blk_idx]` → ghi vào `pool_k[phys, :, offset]`. *CoW guard:*
`pre_decode_reserve` (:547) chạy trước forward — nếu `ln%16==0` (chạm biên) thì cấp block *riêng* mới; else assert
`_refcount[block] == 1` (:562) — "decode write nhắm vào shared block ⇒ CoW invariant vỡ". Vì sharing là *full-block
only* (`share_prefix` :565) và write kế luôn rơi ĐÚNG biên sau prefix chia sẻ → CoW suy biến thành "copy-never",
refcount cân bằng mỗi bước (kiểm tra P4.1.2).

**3. Trace code.** Admission: `reserve_prefill` (:535) cấp ⌈ℓ/16⌉ block, ghi vào `block_table`. Decode:
`pre_decode_reserve` (:547) cấp block biên → forward `write_decode`→`_write_decode_kv` (:597) qua bảng →
`decode_view` (:604) *gather* pool thành view dày đặc padded (đường oracle + negative control P4.1.3: chính copy này
là giá phải trả khi chưa có kernel). `waste_stats` (:587) trả (allocated, live, frag) cho P4.1.1. `share_prefix`
(:565) refcount++ block prefix chung.

**4. Cổng teach-back.** (a) Vì sao block 0 = trash làm write-then-mask "chuyển sang paged miễn phí"? (b)
*Modify-and-predict:* nếu cho `share_prefix` chia sẻ *nửa block* (partial, cho beam search) — bất biến refcount ở
`pre_decode_reserve` :562 sẽ bắn ra sao khi hàng chia sẻ decode tiếp, và vì sao repo *cố tình bỏ* partial-block CoW?

**5. Frontier.** Đây đúng là PagedAttention của vLLM. Câu interview: "Paged KV cho dung lượng nhưng vì sao *một mình
nó* là mất throughput?" (gather thêm traffic + SDPA vẫn padded — chỉ kernel fused Bài 1.7 gỡ được).

---

## Bài 1.5 — PrefillView: admission offset-0 bằng duck-typing (`model.py` · `PrefillView` :630)
> **Câu hỏi first-principles:** một request MỚI (cả prompt) nạp vào slot trống thế nào mà KHÔNG cần viết một
> đường attention riêng cho prefill?
> **Số đo (aha):** prefill = forward uniform-causal thường ⇒ **0 toán attention mới**; oracle ragged token-exact
> (21 CPU tests, R3b). Bài học vận hành: **prefill chạy EAGER** (compile prefill = bẫy 47 graph, ~35s in-run).

**1. Feynman — bài toán bằng lời.** `PrefillView` *giả dạng* (duck-type) đúng interface mà attention layer dùng ở
đường 1-request (`length`/`append`/`get`/`advance`). Mẹo: slot mới bắt đầu ở **offset 0** ⇒ prefill CHÍNH LÀ forward
causal chuẩn với vị trí `0..s−1` — không mask mới, không toán mới. K/V ghi xuyên qua storage hook của parent (dày
đặc hay paged). Đánh đổi: right-pad các hàng tới prompt rộng nhất (rác được che sau), đổi lấy "một forward batched
nạp nhiều request cùng lúc".

**2. Cơ chế (derive từ đầu).** `length` (:672) trả **0** cứng (slot tươi theo hợp đồng) → attention dựng causal mask
chuẩn `q_pos ≥ k_pos` (`model.py:848–849`), không phải row_mask ragged. `append` (:675) ghi qua
`parent._write_prefill_kv` + lưu block để prefill tự-attend đúng khối của mình. `advance(s)` (:685) kích hoạt slot ở
**true length** (không phải padded width): `parent.lengths[slots] = true_lengths_t`, `active=True` — device-only,
một lần sau mọi layer; scheduler mirror bằng `parent.mirror_admit` (:399). Mọi state python (slot list, width) resolve
thành tensor/int LÚC KHỞI TẠO — ngoài graph — nên `append`/`advance` traced chỉ guard trên int ổn định.

**3. Trace code.** Scheduler tìm `free_slots` (:354) → `PrefillView(parent, slots, true_lengths)` (:649):
`reserve_prefill` (dense no-op / paged cấp block) + đóng băng slot/width thành tensor. Forward LM chạy như thường,
gọi `append` mỗi layer (ghi padded prefill), `get` (:682) trả block để tự-attend, cuối cùng `advance(s)` một lần
→ scheduler `mirror_admit`. Lỗ rò đã đo (R3b): admission steady-state đến với n=1,2,3… mỗi width compile một graph
mới (47 graph, guard scan mỗi step) → **fix: prefill eager** (nó ~1% wall; decode giữ ngân sách compile).

**4. Cổng teach-back.** (a) Vì sao `length` trả 0 cứng làm prefill "miễn phí về toán attention"? (b)
*Modify-and-predict:* nếu `advance` kích hoạt slot ở *padded width* thay vì `true_lengths` — hàng prompt ngắn sẽ
attend tới đâu ở bước decode kế, và logits lệch thế nào?

**5. Frontier.** Duck-typing để "prefill = forward thường" là vì sao continuous batching gọn: một code path forward
lo cả prefill lẫn decode. Câu interview: "Vì sao KHÔNG compile prefill?" (shape churn theo số request admit).

---

## Bài 1.6 — ChunkPrefillView: offset-continuation, prefill dài chia khúc (`model.py` · `ChunkPrefillView` :693)
> **Câu hỏi first-principles:** một prompt DÀI (512 token) nạp một phát sẽ head-of-line block cả decode stream
> (spike ITL p99) — chia nó thành khúc C token xen giữa các bước decode, làm sao KV vẫn bit-identical?
> **Số đo (aha):** **token-exact** (single-chunk KV `torch.equal`; RoPE vị trí tuyệt đối). NHƯNG interleave tuần
> tự **REGRESS mọi metric**: ITL p50 **×6.56–14.44**, throughput **547→145 tok/s**, ITL p99 KHÔNG giảm (×1.03–1.91) — FALSIFIED (R4.2).

**1. Feynman — bài toán bằng lời.** `ChunkPrefillView` nạp MỘT slot đang prefill, s token tại một **offset chạy**.
Prompt dài = ⌈L/C⌉ khúc, xen giữa decode. Khác `PrefillView` (offset-0, batched nhiều slot), cái này *nối đuôi* một
slot ở offset tăng dần. Điểm sống: `length` (:728) trả **offset đã ghi** = RoPE offset tuyệt đối của khúc → token
tại vị trí t xoay tại t bất kể ranh giới khúc ⇒ **KV chunked bit-identical với prefill một-phát** (oracle token-exact).
Đánh đổi *đã đo và bị bác*: chia khúc *đúng về cơ chế* nhưng scheduler tuần tự này **thua** — bài học đắt giá.

**2. Cơ chế (derive từ đầu).** `append` (:732) ghi khúc vào `[offset, offset+s)` qua `_write_prefill_kv_at` (:318),
rồi `get` (:737) trả `slot_kv_view(layer, slot, offset+s)` — toàn bộ prefix `[0, offset+s)` để khúc attend dưới causal
mask `past_len` thường (không toán mới). Slot **reserved nhưng inactive** trong khi prefill (bị mask khỏi batch
decode như mọi slot trống — write-then-mask). `advance(s)` (:740) chỉ bump device length; scheduler flip active +
`mirror_admit` sau khúc CUỐI (khúc cuối phát token đầu của request = TTFT). Bit-identical vì RoPE absolute-pos: đại
số giống hệt prefill một-phát (float64 exact mọi chunk size; float32 khác chỉ là argmax tie-flip).

**3. Trace code + vì sao FALSIFIED.** Đường: scheduler dựng `ChunkPrefillView(parent, slot, offset)` mỗi khúc →
forward → `append`/`get`/`advance` → sau khúc cuối scheduler kích hoạt. Đo R4.2: (1) **admission tuần tự** — scheduler
đẩy MỘT slot prefill/iteration ⇒ 160 request thành 160–550 forward prefill *không batched* vs one-shot **14 batched**
→ đói batch decode. (2) **khúc là forward tuần tự riêng** ⇒ chi phí nó rơi vào MỌI khe decode; ITL p50 xấu 6.6× ngay
cả ở C=512 (một khúc = one-shot) ⇒ đây là overhead per-forward thuần, không phải slicing.

**4. Cổng teach-back.** (a) Vì sao RoPE-vị-trí-tuyệt-đối là điều kiện để chunked KV bit-identical (nếu RoPE dùng
vị trí *tương đối trong khúc* thì hỏng gì)? (b) *Modify-and-predict:* biết win của chunked prefill là *kernel/batching
property* chứ không phải scheduling — mô tả một forward batched giành lại nó (piggyback khúc prefill VÀO batch decode)
sẽ đổi bức tranh ITL thế nào so với đường tuần tự này?

**5. Frontier.** Đây đúng là bài học Sarathi-Serve / vLLM-V1 "stall-free batching": production piggyback khúc prefill
vào forward decode batched (một kernel ragged) — thêm compute, KHÔNG thêm launch, KHÔNG khe idle. Câu interview: "Vì
sao chunked prefill phải là fused kernel, không phải scheduling-only?" (R4.2b, deferred).

---

## Bài 1.7 — Kernel Triton paged decode: online softmax qua block table (`kernels/paged_decode_triton.py` · `_paged_decode_kernel` :24 · `paged_decode_attention` :86)
> **Câu hỏi first-principles:** paged storage một mình là LỖ throughput (Bài 1.4, +6.3%) — kernel nào đọc KV
> *trực tiếp qua bảng trang*, không gather, không materialize score, không repeat GQA, để GIÀNH LẠI thuế padding 1.27×?
> **Số đo (aha):** **5.90 ms/bước · ×3.52 vs wave-dense · 3,528 tok/s agg (+55% vs dense-continuous)**; ITL p50
> 6.2 ms ≈ wave 6.1. Thuế 1.27× phân rã: **~3.2 ms là compute** (fp32-score + repeat_interleave + softmax-over-padding),
> **chỉ ~0.5 ms là byte** — đúng như spec đoán (R4.1 P4.1.4).

**1. Feynman — bài toán bằng lời.** Đường gather (Bài 1.4 `decode_view`) vật chất hóa một view padded `(B,H,view_len)`
rồi SDPA — trả ba khoản: copy gather, score fp32 materialize, repeat_interleave GQA. Kernel này gộp tất cả vào MỘT
launch: grid `(B, H_q)`, mỗi program đi qua các block của hàng mình bằng **online softmax fp32** (đúng recurrence FA2,
chuyên biệt cho 1 query row trên page 16 ô). Đọc thẳng `pool_k[phys]` qua bảng — không bao giờ dựng view padded. Đánh
đổi: mất tính tổng quát của SDPA, đổi lấy đọc *chỉ token thật*.

**2. Cơ chế (derive từ đầu).** Online softmax (:62–82): giữ 3 carry fp32 — `m` (max chạy, :62), `l_sum` (mẫu số,
:63), `acc` (tử số, :64). Mỗi block: `phys = table[b,i]` (:66) → load `k_blk` (:69) → `scores = sum(k·q)·scale` (:72)
→ mask `valid = i·16+t < n_keys` (:73, với `n_keys = lengths[b]+1` :57 — write-then-mask self-inclusive) → cập nhật
`m_new = max(m, max(scores))`, `alpha = exp(m−m_new)`, `p = exp(scores−m_new)`, rồi
`acc = acc·alpha + sum(p·v)`, `l_sum = l_sum·alpha + sum(p)` (:75–81). Cuối: `out = acc/l_sum` (:82) — `n_keys ≥ 1`
luôn (write-then-mask) ⇒ `l_sum > 0` ⇒ **không NaN row**. GQA: `kv_h = h // group_size` (:51) — query head đọc KV head
của nhóm mình, KHÔNG repeat_interleave.

**3. Trace code.** `paged_decode_attention` (:86) nhận q `(B,H,1,d)`, pool `(n_blocks,H_kv,16,d)`, block_table
`(B,max_blocks)`, lengths `(B,)` → launch `_paged_decode_kernel[grid=(B,H)]` (:103) với strides + `scale=1/√d`. Được
bật qua `PagedKVCache.use_kernel` (`model.py:508`); attention forward (`model.py:805–813`) rẽ vào kernel thay SDPA khi
`isinstance(cache, PagedKVCache) and cache.use_kernel`. Hai bẫy triton-under-compile đã đo: (1) python-float carry
`m=-inf` promote lên f64 → "loop-carried re-assigned to fp64"; fix carry fp32 tường minh (:62). (2) `scale` (arg float
runtime) truyền f64 bởi inductor wrapper → nhiễm cả chuỗi carry; fix cast `.to(tl.float32)` trong kernel (:72).

**4. Cổng teach-back.** (a) Vì sao online softmax cần `alpha = exp(m − m_new)` — nó "sửa lại" `acc`/`l_sum` cũ theo
max mới thế nào (dựng lại công thức từ softmax đầy đủ)? (b) *Modify-and-predict:* nếu để `n_keys = lengths[b]` (bỏ
`+1`, :57) — hàng vừa-ghi mất gì, và có hàng nào rơi vào `l_sum = 0` → NaN không?

**5. Frontier.** Đây là trái tim vLLM/FlashInfer paged decode + là substrate DUY NHẤT R4.4 CUDA-graph capture được
(grid `(B,H)` cố định, pool địa chỉ tĩnh). Câu interview: "Vì sao decode attention phải là kernel fused chứ không
gather+SDPA?" — gỡ padded compute (thuế 3.2 ms), không chỉ byte (0.5 ms).

---

*Đóng série:* mũi tên AI ~ 1 → batch (slab, ×66 trần) → bỏ reservation waste (paged, ×9.3 dung lượng) → giành lại
thuế padding (kernel fused, ×3.52 vs wave, +55%). Series sau: RoPE per-row + serve() scheduler (`docs/learning/INDEX.md`
serving 02/04), rồi R4.3–R4.6 (speculative, CUDA-graph, MLA, PD-disagg) trong study-queue.
