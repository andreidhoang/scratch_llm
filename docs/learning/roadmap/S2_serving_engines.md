# SÉRIE 2 — Serving engines (A1) + các tối ưu R4.x

> **Số dòng pin theo commit `9e61d7a`.** Mỏ neo chính là *file · class · hàm*; số dòng là tiện tra nhanh.
> Số đo lấy từ `bench/RESULTS.md` (§A1 R0…R4.6). Prose tiếng Việt, thuật ngữ giữ tiếng Anh.

## Vì sao série này

Cả série là MỘT câu chuyện: **leo bức tường bộ nhớ (memory wall) của decode** — từ 16% → 53% → 77% của
mức trần HBM đo được trên chính card (RTX PRO 4000 Blackwell, sm120, 0.55 TB/s). B=1 eager chỉ chạm 51
tok/s = **16%** trần vì launch-overhead che tường (R0/R1); `torch.compile` fuse các launch → 173 tok/s =
**53%** (R1 strip); CUDA-graph gom mọi launch còn lại → 253 tok/s = **77%** (R4.4). Mỗi Bài dưới đây là
một đòn bẩy khác nhau lên cùng bức tường đó — batching (nâng AI≈B), continuous scheduling (nâng
utilization), speculative (nâng token/forward), paged+cudagraph (bỏ overhead), MLA (co cache), disagg
(tách hai profile trái ngược). Trục xuyên suốt: **mọi tối ưu phải token-exact so với oracle
`sampling.generate` greedy** — đo tốc độ trên một engine sai là vô nghĩa. Ta đi từ oracle (Bài 2.1) và
thước đo (2.2) trước, rồi mới tới từng đòn bẩy.

---

## Bài 2.1 — Oracle giải mã token-exact (`sampling.py` · `_decode`/`generate` :77,127)

> **Câu hỏi first-principles:** làm sao biết một engine serving "nhanh hơn" mà KHÔNG đổi output? Cần một
> nguồn sự thật duy nhất, đơn giản đến mức không thể sai, để mọi rung sau đối chiếu từng token.
> **Số đo (aha):** đường `use_cache=True` và `use_cache=False` phải cho **output y hệt** — đó là bất
> biến neo cả 9 rung (mọi R3b/R4.x đều test "token-exact vs generate greedy").

**1. Feynman — bài toán bằng lời:** đây là *chiếc thước chuẩn cất trong tủ kính*. Nó cố tình chậm và
ngây thơ: giải mã **một request, một token mỗi lần**, không batch, không paged, không graph. Đánh đổi:
bỏ hết tốc độ để lấy *tính đúng không cãi được*. Hai đường trong cùng một hàm — `use_cache=False` tính
lại toàn prefix mỗi bước (không thể sai vì không có state), `use_cache=True` dùng `KVCache`; hai đường
BẮT BUỘC ra cùng token. Nếu lệch → cache bug, sửa cache chứ không đổ cho model.

**2. Cơ chế (derive từ đầu):** greedy = `argmax(logits)` khi `temperature==0` (`_sample_next` :58-64) —
xác định, không RNG. Logprob của token lấy từ *chính logits nó được sample ra* qua `log_softmax`
(`_logprob_of` :67) → khớp một lần re-score teacher-forced (bất biến cho RL sau này). Dừng: gặp
`stop_ids` (:108) hoặc đủ `max_tokens` (:104) — chính xác, không lố.

**3. Trace code:** `generate` (:127) → `_decode` (:77). Nếu `use_cache`: (a) cấp `KVCache(len(blocks))`
(:101); (b) prefill cả prompt trong 1 forward, lấy logits vị trí cuối `[0,-1]` (:102-103); (c) vòng
decode :104-111 — `_sample_next` → append → nếu stop thì break → forward token mới `[[next_id]]` vào
cache (:110-111). Đường no-cache :114-123 recompute `ids[-context_length:]` mỗi bước. Cả hai gọi CHUNG
`_sample_next`/`_logprob_of` → cùng một quyết định số học.

**4. Cổng teach-back:** (a) Vì sao logprob phải đọc từ logits chưa scale temperature/top-p, không phải
từ `probs` đã lọc? (b) *Sửa-và-đoán:* nếu ở đường cache ta prefill nhưng LẤY NHẦM logits `[0,0]` (vị trí
đầu) thay vì `[0,-1]` — token đầu tiên sinh ra sẽ dựa trên context nào, và test token-exact vs no-cache
vỡ ở token thứ mấy?

**5. Frontier:** đây là "loss-at-init/round-trip oracle" mà lab nào cũng dựng trước khi tối ưu. Interview:
"bạn refactor KV-cache/paged/graph — chứng minh không đổi output thế nào?" → câu trả lời là *chiếc thước
này* + so từng token ở float64 (float32 chỉ khác do argmax tie-flip, không phải logic — xem R4.2/R4.3).

---

## Bài 2.2 — Thước đo serving: TTFT/ITL/goodput percentiles (`serving/metrics.py` · `summarize` :119)

> **Câu hỏi first-principles:** "nhanh hơn" nghĩa là gì cho một LLM server? Một con số throughput che
> mất trải nghiệm người dùng. Phải đo ĐÚNG bốn số quyết định ship hay không.
> **Số đo (aha):** goodput có thể = **0 tok/s** dù throughput cao ngất — R4.2 đo cả bốn arm goodput=0 vì
> SLO ITL bị vi phạm trên trace bão hòa. Throughput biết nói dối; goodput thì không.

**1. Feynman — bài toán bằng lời:** bốn cái đồng hồ, mỗi cái đo một nỗi đau khác nhau. **TTFT**
(time-to-first-token) = prompt bấm enter tới chữ đầu — do prefill gate. **ITL** (inter-token latency) =
độ mượt khi chữ chảy ra — do decode gate. **Throughput** = tổng tok/s cả hệ. **Goodput** = throughput
*chỉ tính request đạt SLO* (DistServe). Đánh đổi cốt lõi: throughput cao mà ITL giật cục → mất khách;
goodput là con số "trả tiền". Identity neo tất cả: `latency ≈ TTFT + ITL × num_output_tokens`.

**2. Cơ chế (derive từ đầu):** `RequestRecord` giữ `token_times_s` trên MỘT clock đơn điệu (:36).
`ttft_s = token_times_s[0] − start_s` (:47). `itls_s` = hiệu liên tiếp (:53). Percentile theo
**nearest-rank**: `idx = round(p·(n−1))` (:103-105) — cùng quy ước `bench.harness`, không nội suy.
Goodput: một request "đạt" iff `ttft ≤ SLO.ttft` VÀ `max(itls) ≤ SLO.itl` (`request_meets_slo` :113) —
**worst gap**, không phải trung bình (một cú giật cũng là fail).

**3. Trace code:** `summarize` (:119) → `wall_s = max(token_times[-1]) − min(start_s)` (:132-134) →
`throughput = Σ output_len / wall_s` (:137) → `goodput = Σ output_len CỦA request đạt SLO / wall_s`
(:139-140) → TTFT percentile theo *per-request* (:142), ITL percentile theo *pool GỘP mọi gap* (:145-150).
Điểm tinh tế: ITL gộp toàn bộ gap của mọi request rồi mới lấy p99 → một admission spike ở bất kỳ request
nào cũng nhô lên p99.

**4. Cổng teach-back:** (a) Vì sao goodput dùng `max(itls)` chứ không `mean(itls)` để phán một request?
(b) *Sửa-và-đoán:* nếu đổi ITL percentile từ "pool gộp" sang "p99 của từng request rồi lấy trung bình" —
admission spike (một gap 200 ms giữa hàng nghìn gap 6 ms) sẽ biến mất hay còn? Con số nào che mất pha?

**5. Frontier:** goodput-under-SLO là ngôn ngữ chuẩn 2026 (DistServe/vLLM). Interview: "TTFT p99 vs ITL
p99 — tối ưu nào đánh vào cái nào?" → prefill-side (chunked, disagg) đánh TTFT/ITL-spike; decode-side
(cudagraph, batching) đánh ITL p50 + throughput.

---

## Bài 2.3 — R0 baseline: decode tuần tự có đo (`serving/baseline.py` · `decode_record` :44)

> **Câu hỏi first-principles:** trước khi tối ưu, phải ĐO cái ngây thơ và biết nó bị bound bởi gì. Decode
> B=1 có phải memory-bound không? Nếu có sao chỉ đạt 16% trần?
> **Số đo (aha):** **51 tok/s = 16% của trần 315 tok/s; 89 GB/s = 16% của 550 GB/s** (R1). Workload là
> memory-bound (AI≈0.96) nhưng eager run **overhead-bound**: ~955 kernel launch/token — 84% wall là
> overhead, chỉ 16% là traffic bộ nhớ thật.

**1. Feynman — bài toán bằng lời:** R0 KHÔNG viết lại decoder — nó **gắn đồng hồ** lên
`sampling.generate` sẵn có. Đây là bài học honesty: "naive baseline" đúng nghĩa là chạy tuần tự từng
request một, đo từng token trên clock chuẩn, rồi để `metrics.summarize` phán. Đánh đổi: mọi thứ chậm
nhất có thể, nhưng số đo *thật* và làm mốc so cho mọi rung sau.

**2. Cơ chế (derive từ đầu):** timing đúng trên CUDA async đòi `torch.cuda.synchronize()` TRƯỚC mỗi
`clock()` — nếu không, timestamp đo lúc *enqueue* kernel chứ không phải *hoàn tất* (`get_time` :65-68).
`start_s` đọc ngay trước prefill; `token_times_s[i]` ngay sau token i sẵn sàng → `ttft_s` chính là thời
gian prefill.

**3. Trace code:** `decode_record` (:44): `start_s = get_time()` (:77) → `KVCache` + prefill
`model(x,cache)[0,-1]` (:79-81) → vòng :86-93 gọi `_sample_next` (dùng chung Bài 2.1) → `token_times.append(get_time())`
→ forward token mới. `run_baseline` (:103) chạy list prompt **tuần tự** (:113) — one at a time, đó là cái
"tĩnh" mà continuous batching (2.5) sẽ đánh bại. Token-exact với `generate` được test pin.

**4. Cổng teach-back:** (a) Vì sao workload memory-bound (AI≈1) mà achieved BW chỉ 16% của HBM peak — hai
cái đó mâu thuẫn không? (b) *Sửa-và-đoán:* nếu bỏ `torch.cuda.synchronize()` trong `get_time`, TTFT đo ra
sẽ lớn hơn hay nhỏ hơn thực tế, và ITL p50 bị nhiễu theo hướng nào?

**5. Frontier:** phân biệt "workload bound" (roofline của bài toán) vs "run bound" (cái đang thật sự giới
hạn) là câu hỏi phễu ở interview kernel. Ở đây workload memory-bound nhưng run overhead-bound → lever
đúng là *fuse launch*, không phải kernel bộ nhớ nhanh hơn.

---

## Bài 2.4 — R3a static batch: đòn bẩy AI≈B (`serving/batched.py` · `batched_greedy_decode` :26)

> **Câu hỏi first-principles:** B=1 đọc toàn bộ weight mỗi token cho MỘT token — lãng phí. Nếu đọc weight
> một lần rồi áp cho B sequence thì arithmetic intensity thành bao nhiêu, và tok/s scale thế nào theo B?
> **Số đo (aha):** **agg(B=32)/agg(B=1) = 24.9×**, đỉnh **12,220 tok/s @B=256 = 66×**; AI≈B chính xác;
> roofline cắt memory→compute tại **B≈128** (AI≈ridge 131).

**1. Feynman — bài toán bằng lời:** đọc một quyển sách (weight, 1.68 GB) cho một học sinh là phí; đọc to
cho cả lớp B đứa cùng lúc thì chi phí đọc chia cho B. Đó là *weight amortization*: bytes/step ≈ 2P + B·KV
(không phải B·2P). Đánh đổi: đòi mọi row **cùng độ dài** (RoPE positions chia sẻ) — nên đây mới là *static*
batch; ragged + join/leave là R3b. Vì sao batching mới là câu trả lời cho memory wall, không phải kernel
nhanh hơn: kernel nhanh không cứu nổi workload AI≈1; batching *nâng chính AI lên B*.

**2. Cơ chế (derive từ đầu):** AI = 2PB/(2P + B·KV). Khi B nhỏ → mẫu ≈ 2P → AI ≈ B (memory regime,
per-stream tok/s ~hằng, aggregate tuyến tính). Khi B·KV vượt 2P → AI bão hòa, cắt sang compute-bound tại
B≈ridge/… ≈128. Plateau ~12K tok/s = ~28% của 72 TF/s (naive `@` matmul).

**3. Trace code:** `batched_greedy_decode` (:26): chặn prompt lệch độ dài (:41-45) → 1 `KVCache` chung →
prefill cả batch `model(x,cache)[:,-1]` shape `(B,vocab)` (:50) → vòng :52-57: `logits.argmax(dim=-1)`
cho `(B,)` — **một weight read phục vụ cả B row** (:53) → `x = nid.unsqueeze(1)` `(B,1)` → forward. Row b
== single-stream greedy của prompt b (batch independence, test pin).

**4. Cổng teach-back:** (a) Vì sao khi B<128 per-stream tok/s gần như hằng còn aggregate tuyến tính? (b)
*Sửa-và-đoán:* eager control cũng scale 30.7× (RESULTS R3a) — nếu batching amortize weight thì tại sao
eager (không compile, overhead-bound) VẪN scale gần tuyến tính? Nó đang amortize cái gì khác?

**5. Frontier:** "batch size vs latency/throughput" là câu hỏi capacity-planning kinh điển. B≈128 crossover
= nơi lab chuyển từ lo bandwidth sang lo FLOP → quyết định GQA/MQA (để KV nhỏ, batch to hơn vẫn fit) và
chọn kernel tensor-core.

---

## Bài 2.5 — R3b continuous + hợp đồng sở hữu + nhánh R4.2 chunked (`serving/continuous.py` · `serve` :90)

> **Câu hỏi first-principles:** static batch chạy cả wave tới khi ROW DÀI NHẤT xong — row ngắn ngồi
> không. Nếu quyết định lại *mỗi bước* (đuổi row xong, nạp row mới vào slot trống) thì được bao nhiêu, và
> giá phải trả là gì?
> **Số đo (aha):** **2.30× wall throughput** (2,283 vs 995 tok/s), **2.93× theo số step** (1,396 vs 4,088);
> util **72.8% vs 24.9%**. Khoảng cách 2.93→2.30 = **thuế padding dense buffer 1.27× (9.6 vs 6.3 ms/step)**.

**1. Feynman — bài toán bằng lời:** khách sạn B slot. *Wave* (static): đợi cả đoàn checkout mới nhận đoàn
mới → slot trống ngồi không. *Continuous* (Orca): slot nào trống là nhận ngay người kế trong hàng đợi.
Cùng một engine, cùng cache, cùng forward — chỉ khác **policy**, nên Δ đo được là *thuần scheduling*
(:10-12). Đánh đổi honesty: continuous nâng throughput NHƯNG ITL p50 xấu đi (9.6 vs wave 6.3 ms) và có
spike admission — batch trộn tuổi giữ `L_view` cao → đọc nhiều padding hơn.

**2. Cơ chế (derive từ đầu):** `speedup ≈ max_len/(mean_len + B·τ_p/τ)`; util wave = mean_len/max_len.
Trace heavy-tail util wave = 24.9% (== analytic), continuous 72.8% (drain-tail-limited). **Hợp đồng sở
hữu graph/scheduler** (bài học dynamo, RESULTS findings): tensor GPU của forward, python-list của
scheduler, KHÔNG đọc chéo trong forward — `mirror_admit`/`mirror_advance` (:282,:361) là nửa python chạy
*ngoài* đồ thị; `view_len` là int thường, không đọc `max(py_lengths)` in-graph (nếu đọc → Dynamo guard
theo thứ tự slot → recompile-storm → eager fallback chỉ ở arm continuous, thiên vị baseline). Prefill chạy
**eager** (prefill_model, :99,:274) vì admission tới lắt nhắt n=1,2,3… mỗi width là 1 graph mới (47 graph,
~35s compile) — decode mới xứng ngân sách compile. Sau 2 fix: unique_graphs=2.

**3. Trace code (một vòng `while`, :250):** (1) **evict** row hết budget → `finish` (:227,:252-254); (2)
**admit** — one-shot: gom slot free, MỘT prefill right-padded cho mọi row admit (`_prefill` :446, tránh
stall n_slots/mean_len), rồi `mirror_admit` (:274-298); (3) **một decode step** lockstep trên slot active,
shape tĩnh, row inactive tính masked rồi vứt (:349-373): `pre_decode_reserve` → `model(last_ids.unsqueeze(1),cache)`
→ argmax → `mirror_advance`. **Nhánh R4.2 chunked** (`prefill_chunk_size`, :299-345): prompt L chia
`⌈L/C⌉` chunk ≤C token, **một chunk/iteration xen giữa các decode step** qua `ChunkPrefillView`
(`_prefill_chunk` :472); slot reserved-nhưng-inactive (masked khỏi decode batch) tới chunk cuối mới emit
token đầu (=TTFT) rồi join. Token-exact vì RoPE xoay mỗi token tại vị trí tuyệt đối bất kể ranh giới chunk.

**4. R4.2 — MEASURED NEGATIVE (honest):** dự đoán ITL p99 GIẢM ≥2×; đo được **RỘT NGƯỢC**: ITL p99 ×1.03→1.91,
ITL p50 ×6.56→14.44, throughput **547→145 tok/s monotone GIẢM** khi C↓. Root cause chẩn từ cột
`prefills`/`steps`: (1) **admission tuần tự hóa** — scheduler đẩy MỘT slot prefilling mỗi iteration →
160 request thành 160-550 prefill *không batch* vs one-shot **14 prefill batch**, bỏ đói decode; (2) chunk
là **forward tuần tự riêng** → chi phí đầy đủ rơi vào mỗi gap decode (p50 xấu 6.6× ngay cả C=512 = single
chunk). **Bài học principal-level:** lợi ích chunked prefill là *kernel/batching property*, KHÔNG phải
scheduling-only. Sản xuất (Sarathi/vLLM-V1) **piggyback chunk VÀO forward decode batch** (một fused ragged
kernel) — thêm compute, không thêm launch, không slot idle. Scoped R4.2b (deferred, không chặn R4.3-4.6).

**5. Cổng teach-back:** (a) Vì sao wall-ratio (2.30) < step-ratio (2.93)? "Padding traffic" nghĩa là gì
trên dense buffer? (b) *Sửa-và-đoán:* R4.2 chunked C=512 (một chunk = one-shot prefill) ITL p50 vẫn xấu
6.6× — điều đó chứng minh vấn đề nằm ở *slicing* hay ở *per-forward overhead*? Nếu piggyback vào decode
forward thì thành phần nào biến mất?

**6. Frontier:** continuous batching là default 2026. Interview: "chunked prefill giúp gì và cạm bẫy
scheduling-only là gì?" → chính là measured-negative này: tách forward = mất nhiều hơn spike bỏ được.

---

## Bài 2.6 — R4.3 speculative decoding lossless (`serving/speculative.py` · `speculative_generate` :108)

> **Câu hỏi first-principles:** decode là 1 weight-read/token (AI≈1). Có cách nào commit >1 token mỗi
> forward mà KHÔNG đổi distribution output không?
> **Số đo (aha):** **token-exact** (float64, mọi drafter/K; drafter cố tình SAI vẫn ra đúng greedy →
> rollback KV sạch); speedup **×1.21-1.29 wall / 1.33-1.39 token/target-forward** (n-gram zero-cost).

**1. Feynman — bài toán bằng lời:** một *drafter* rẻ đoán K token tới; target verify CẢ K trong MỘT
forward; chấp nhận prefix dài nhất khớp argmax của chính target, lấy token target tại mismatch đầu. Chuỗi
commit ĐÚNG BẰNG cái target tự sinh — **lossless cho greedy by construction**. Đánh đổi: nếu draft trật,
tốn một slot verify (miễn phí về correctness). Nâng AI từ phía *khác* batching: nhiều token cùng verify.

**2. Cơ chế (derive từ đầu) — bất biến `pending` + KV rollback:** đầu mỗi round, `pending` = token committed
kế mà K/V CHƯA vào cache và == argmax greedy của target (đúng by construction). Draft K guess cho các
token SAU nó; forward `[pending, *drafts]`; `greedy[i]` = token target sau `inp[i]`; accept prefix draft
khớp greedy; `pending ← greedy[accepted]` (correction/continuation free). **KV rollback:** cache mọc thêm
`1+len(drafts)`, nhưng `cache.truncate(base + 1 + accepted)` (:170, `model.py:239`) *cắt bỏ K/V draft bị
từ chối* → giữ đúng pending + accepted. Mỗi round commit `1+accepted` token cho 1 target forward = speedup.

**3. Trace code:** `speculative_generate` (:108): prefill → `pending=argmax` (:139-140). Vòng :147: dựng
`context = prompt+generated+[pending]` → `drafter.propose(context,k)` (:148-149) → `base=cache.length`
(:152) → forward `[[pending,*drafts]]` (:153-154) → `greedy=argmax` (:156) → accept loop :160-164 →
`generated += [pending]+drafts[:accepted]`, `pending=greedy[accepted]` (:167-169) → **`cache.truncate`
:170**. `NGramDrafter.propose` (:46): tìm lần xuất hiện gần nhất của last-n-gram, đề xuất k token theo sau
(training-free). `Drafter` là Protocol (:30) → n-gram / ModelDrafter (:60) hoán đổi.

**4. Hai dự đoán FALSIFIED (một lý do honest):** acceptance random dự đoán ~0, đo **54-72%**; speedup random
dự đoán <1× loss, đo **×1.35-1.39 WIN**. Nguyên nhân: model 0.84B *chưa train* → greedy output degenerate-
repetitive (entropy thấp) → n-gram hit cả trên prompt random. Acceptance bám **output-entropy của MODEL**,
không bám prompt. Model đã train sẽ phụ thuộc prompt/domain như dự đoán gốc.

**5. Cổng teach-back:** (a) Chứng minh losslessness: vì sao drafter SAI hoàn toàn vẫn cho đúng chuỗi greedy?
`truncate` đóng vai gì? (b) *Sửa-và-đoán:* nếu QUÊN `cache.truncate` (giữ nguyên K/V mọi draft) — round
kế token đầu tiên tính attention lên các key nào thừa, và output lệch ở token thứ mấy so với oracle?

**6. Frontier:** cả họ drafter (Medusa/EAGLE-2/3 feature-tree/MTP ~85-90% 2nd-token) đều tối ưu E[accept]
— số hạng chi phối công thức speedup `E[accept+1]/(1+K·c_draft/c_target)`. Interview: "spec decode giữ
distribution thế nào?" → accept-longest-prefix + rollback; với sampling (không greedy) thì thêm bước
modified-rejection (Leviathan/Chen).

---

## Bài 2.7 — R4.4 CUDA-graph decode: bỏ launch overhead (`serving/cudagraph.py` · `CudaGraphDecoder` :31)

> **Câu hỏi first-principles:** sau khi compile fuse pointwise, decode vẫn còn ~54-68 launch/step — CPU
> dispatch từng cái. Nếu *ghi* cả step thành MỘT submission rồi replay thì chạm được trần chưa?
> **Số đo (aha):** B=1 step **−74.3% (15.38→3.96 ms)**; **253 tok/s = 77% của trần** (eager 65=20%,
> compiled 53%); 54-68 `cudaLaunchKernel` → **1 `cudaGraphLaunch`**. Token-exact.

**1. Feynman — bài toán bằng lời:** CUDA graph = "băng ghi" toàn bộ chuỗi kernel của một decode step, ghi
MỘT LẦN, sau đó *replay* bằng một lệnh duy nhất — CPU thôi phải xếp hàng trăm launch giữa các kernel, GPU
hết chờ host. Đây là mảnh cuối leo từ 53%→77% trần.

**2. Cơ chế (derive từ đầu) — vì sao paged kernel là substrate DUY NHẤT capture được:** (a) cache `torch.cat`
(R1) mọc địa chỉ MỚI mỗi step → graph ghi con trỏ cố định → capture ILLEGAL. Block pool paged là
**fixed-address**, ghi in-place. (b) dense `BatchedKVCache` đọc slice `[:,:,:view_len]` có SHAPE mọc mỗi
step → graph shape tĩnh không phủ nổi. Kernel paged launch **grid `(B,H)` cố định**; số key mỗi row là
runtime loop-bound đọc từ tensor `lengths` trên device → MỘT capture phục vụ mọi độ dài. Thêm nữa
`torch.compile` reduce-overhead TỪ CHỐI path này (`lengths += active` in-place = "mutated input"); manual
capture tự làm chủ mutation.

**3. Trace code:** `CudaGraphDecoder.__init__` (:39) đòi `PagedKVCache` `use_kernel=True` + `_static_in`
fixed-address (:45). `capture` (:50): warmup trên side stream (prime Triton autotune + allocator) →
snapshot `lengths` → `torch.cuda.graph(g)` bọc `model(_static_in, cache)` (:67-69) → **restore lengths**
(:71-73, undo advance của warmup+capture, để state prefilled nguyên vẹn). `step` (:76): `pre_decode_reserve`
(host, cấp block nếu row vượt biên 16) → `_static_in.copy_(last)` → `graph.replay()` → argmax →
`mirror_advance` (host) — nửa host bao quanh replay, mutate block table tại địa chỉ cố định để kernel captured
thấy update.

**4. Cổng teach-back:** (a) Vì sao dense buffer (shape mọc) KHÔNG capture được mà paged kernel (grid cố
định, loop-bound runtime) thì được? (b) *Sửa-và-đoán:* nếu QUÊN restore `lengths` sau capture (:71) — replay
đầu tiên đọc `lengths` bằng bao nhiêu, và token đầu tiên sai thế nào so với eager?

**5. Frontier:** cudagraph decode là default vLLM-V1. Dự đoán gốc −20-28% (H100 vLLM path) bị FALSIFIED
tốt (−74.3%) vì baseline eager của ta launch-bound nặng hơn nhiều. Interview: "graph capture yêu cầu gì?"
→ static address + static shape/grid; mutation phải do mình own. Mở rộng deferred: graph pool per batch
size cho continuous batching.

---

## Bài 2.8 — R4.5 MLA weight-absorption identity (`mla.py` · `forward_absorbed` :111)

> **Câu hỏi first-principles:** thay vì cache K,V mỗi head, có thể cache MỘT latent low-rank `c_KV`/token
> rồi vẫn attend đúng không? Đại số nào cho phép, và điều kiện gì?
> **Số đo (aha):** identity đúng tới **1.4e-15 (float64 = machine eps)**; MLA cache **1152 B = 1.8% của
> MHA (65536 B), 3.56× < GQA-8**.

**1. Feynman — bài toán bằng lời:** thay vì lưu K,V đã "bung" ra mỗi head, lưu *bản nén* `c_KV` (dim
d_latent) + một rotary key nhỏ `k_R` (dim d_rope) chia sẻ mọi head. Mẹo: đừng bung ra rồi attend — *gập*
ma trận up-projection vào query/output và attend THẲNG trong latent space. Đánh đổi: đòi RoPE **decoupled**
(tách riêng đường d_rope), vì rotation phụ-thuộc-vị-trí trên content-K sẽ phá phép gập (W_UK phải static).

**2. Cơ chế (derive từ đầu) — identity thuần đại số:** content score `q_c·K^C = q_c·(W_UK c_KV) =
(W_UK^T q_c)·c_KV`. Gập W_UK vào query → `q_abs` attend `c_KV` trong d_latent. Output
`Σ a_j V_j = W_UV(Σ a_j c_KV_j)` → gập W_UV vào output path. Decode chỉ đọc `c_KV`(+`k_R`), không bao giờ
đọc K,V đầy đủ. Điều kiện: RoPE decoupled giữ content-K position-free → W_UK là static fold.

**3. Trace code:** `_project` (:76) cho `q_c,q_rope,c_kv,k_rope` (rope CHỈ trên q_rope/k_rope, :83,:86).
`forward_naive` (:92, oracle): bung `k_c=up_k(c_kv)`, `v=up_v(c_kv)` rồi attend chuẩn (:97-108).
`forward_absorbed` (:111): `wk=up_k.weight.view(H,dh,dc)` → `q_abs=einsum("bhsd,hdc->bhsc",q_c,wk)` (:118-119,
gập W_UK vào query) → score `q_abs·c_KV^T + q_rope·k_R^T` (:121-124) → `latent_out=attn·c_KV` (:128) →
`out=einsum("bhsc,hdc->bhsd",latent_out,wv)` (:130-131, gập W_UV). `test_mla.py` pin naive==absorbed
float64.

**4. Cổng teach-back:** (a) Viết lại `(W_UK^T q)·c = q·(W_UK c)` và chỉ ra vì sao nó cho phép cache
`c_KV` thay K,V *zero quality change*. (b) *Sửa-và-đoán:* nếu áp RoPE lên content-K (bỏ decoupled, xoay
k_c) — phép gập W_UK còn static được không? `forward_absorbed` lệch `forward_naive` vì đâu?

**5. Frontier:** MLA (DeepSeek-V2/V3) là KV story 2026 — 671B có cache/token NHỎ hơn 70B GQA-8. Bridges
A5 (FP8 latent). Interview: "MLA tiết kiệm cache thế nào lúc decode?" → absorb + cache latent; decoupled
RoPE là điều kiện load-bearing.

---

## Bài 2.9 — R4.6 PD-disaggregation demo (`bench/disagg.py` · `measure_kv_transfer`/`measure_itl_contrast` :58,:101)

> **Câu hỏi first-principles:** prefill (compute-bound, GEMM cả prompt) và decode (memory-bound GEMV, một
> token) profile TRÁI NGƯỢC. Co-locate → prefill burst cướp cycle của decode → ITL spike. Tách ra đáng
> giá bao nhiêu, giá là gì?
> **Số đo (aha):** decode-worker ITL p99 **20.2 vs 61.5 ms = 3.0× tốt hơn**; đổi lấy KV transfer MỘT LẦN
> **0.180 ms (16.8 MB, 187 GB/s)**; agg 1707 vs 1038 tok/s.

**1. Feynman — bài toán bằng lời:** hai công việc trái tính nết ở chung một GPU thì giẫm chân nhau: mỗi
lần prefill một prompt dài, luồng decode đang chảy bị khựng (spike ITL). Disagg: prefill chạy worker riêng,
luồng decode KHÔNG bao giờ bị ngắt; giá là **chuyển KV cache một lần** từ worker prefill sang worker decode.
Đánh đổi: bỏ spike lặp lại (~40 ms) đổi lấy transfer một lần (0.18 ms) — lời to.

**2. Cơ chế (derive từ đầu):** single-GPU không đặt nổi 2 worker thật → demo HAI nửa đo được: (1) KV-transfer
tax = D2D copy các paged block prefilled vs bound analytic (bytes/HBM_BW); (2) ITL mà decode worker sạch
(chỉ short-prompt decode) thấy vs co-located (decode xen prefill burst). Goodput-under-SLO nghiêng về disagg.
KV bytes = `2(K,V) × n_layers × live_blocks × BLOCK × kv_heads × head_dim × 2(bf16)` (:64-72).

**3. Trace code:** `_prefill_one` (:37) prefill 512-tok vào paged cache (worker output). `measure_kv_transfer`
(:58): `_copy` lặp `dst._pool_k[layer].copy_(src._pool_k[layer])` mọi layer (:74-78), median 20 lần (:82-88);
đo 0.180 ms vs analytic 0.061 — chậm hơn vì **32 copy nhỏ per-layer → launch-bound dưới HBM peak** (RESULTS
root cause). `measure_itl_contrast` (:101): `decode_only` = 96 short (:114); `co_located` = chèn long mỗi 6
request (:116); chạy `serve_continuous` cả hai, `summarize` ra ITL percentile (:119-126).

**4. Cổng teach-back:** (a) Vì sao disagg cải thiện ITL p99 (spike) chứ không nhất thiết ITL p50? Nối về
`request_meets_slo` dùng max-gap (Bài 2.2). (b) *Sửa-và-đoán:* nếu prompt dài 2048 thay vì 512 — KV-transfer
tax (0.18 ms) scale thế nào, và điểm hòa vốn (transfer < spike bỏ được) dịch về đâu?

**5. Frontier:** disagg (DistServe/Mooncake) là kiến trúc serving node-scale 2026. Honest scope: đây single-GPU
sim (D2D proxy cho NVLink/RDMA thật). Interview: "khi nào disaggregate prefill/decode?" → khi ITL SLO chặt +
prompt dài burst; giá thật là bandwidth chuyển KV ở node scale (P6).

---

*Đóng série:* R0-R4.6 khép vòng memory wall **16%→53%→77%** trên chính card. Gap thật tới vLLM (~1.3×
latency / ~1.5-2× throughput) = chunked-prefill piggyback (R4.2b) + prefix cache — xem
`performance/notes/A1_design_note.md`.
