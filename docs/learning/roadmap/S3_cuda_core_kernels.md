# Série 3 — A2 CUDA-core kernel ladder (Triton): leo tới trần Hopper

> **Số dòng pin theo commit `9e61d7a`.** Số đo: `bench/RESULTS.md` §A2 (dòng 483–528). Design note:
> `performance/notes/A2_design_note.md` ("Climbing from coalescing to the Hopper ceiling").
> Card đứng: RTX PRO 4000 Blackwell **sm120**, peaks đo được **0.551 TB/s HBM · 72.1 TF/s bf16 ·
> ridge 131 FLOP/byte**.

**Vì sao série này.** Série 1 dạy *serving substrate* (KV-cache, batching) — nơi decode chạm **tường
memory** ở tầng hệ thống. Série 3 đi xuống một tầng: **chính các kernel** tạo nên forward, và câu hỏi
first-principles duy nhất xuyên suốt là *"kernel này bị chặn bởi cái gì — HBM bandwidth hay tensor-core
throughput?"*. Roofline trả lời câu đó bằng MỘT con số: arithmetic intensity (AI = FLOP/byte). Through-line
của cả série: **4 trong 5 kernel memory-bound VĨNH VIỄN** (AI≈1, trần = HBM BW, không mẹo nào vượt được —
chỉ có "chạm tường"); **chỉ GEMM băng qua ridge** thành compute-bound (AI 1365) và mới có chuyện "leo tới
cuBLAS". Ta bắt đầu bằng cái thước (harness đo peak), rồi leo thang từ kernel AI thấp nhất (GEMV) tới kernel
duy nhất vượt ridge (GEMM). Một chú ý xuyên suốt (ADR-0011): **Triton lo coalescing/float4/swizzle**, nên
mỗi bài phải trung thực *cái gì compiler làm hộ* vs *cái raw-CUDA ladder phải viết tay*.

---

## Bài 3.0 — Roofline harness: cái thước đo mọi kernel (`bench/_harness.py` · `Roofs` :67 · `bench/kernel_roofline.py` · `profile` :48)

> **Câu hỏi first-principles:** làm sao *trước khi* tối ưu, biết một kernel nên bị chặn bởi memory hay
> compute — và "tốt" nghĩa là bao nhiêu phần trăm của trần nào?
> **Số đo (aha):** copy anchor **551 GB/s = 100.1% mem**, gemm anchor **72.1 TF/s = 100.0% cmp**, spread
> <1.2% — thước tự đặt đúng cả hai mái nhà. ncu **BLOCKED** (`ERR_NVGPUCTRPERM`) → 5 metric ncu-debt.

**1. Feynman — bài toán bằng lời.** Roofline là một *mái nhà hai dốc*: trục hoành là AI (FLOP/byte của
kernel), trục tung là FLOP/s đạt được. Dốc trái nghiêng lên = **memory roof** (BW × AI): kernel AI thấp bị
BW kéo xuống. Dốc phải phẳng = **compute roof** (peak FLOP/s). Chỗ hai dốc gặp nhau là **ridge** (AI =
compute/BW = 131 trên card này). Kernel nào rơi bên TRÁI ridge → memory-bound; bên PHẢI → compute-bound.
Đánh đổi cốt lõi của harness: **đo peak trên CHÍNH card này, không lấy số datasheet** — vì "% của trần"
chỉ có nghĩa khi trần là cái card thật đạt được, không phải con số marketing.

**2. Cơ chế (derive từ đầu).** Hai peak, đo bằng chính torch:
- **HBM BW** (`measure_mem_bw_bytes_s` :50): copy 256 MB device-to-device, `2·numel·elem_size / sec`
  (đọc + ghi = 2 byte/phần tử). Ra 0.551 TB/s.
- **Compute** (`measure_compute_peak_flops_s` :58): GEMM vuông 8192³, `2·n³ / sec`. Ra 72.1 TF/s bf16.
- **ridge** (`Roofs.ridge` :79) = compute/BW = 131 FLOP/byte.
- **attainable(flops, bytes)** (:83): `mem_bound = AI·BW`; nếu `mem_bound < compute` → trả `(mem_bound,
  "mem")`, ngược lại `(compute, "cmp")`. **%roof = achieved/attainable** = headroom tuyệt đối tới trần
  ĐANG chặn. Đây là toàn bộ toán roofline, gói trong 8 dòng.

**3. Trace code.** `Roofs.measure` (:74) gọi hai hàm đo → cache 2 peak. `profile(name, fn, flops, bytes_,
roofs)` (`kernel_roofline.py` :48): (a) `bench_ms(fn)` (`_harness.py` :21) — `triton.testing.do_bench` với
**L2 flush mỗi rep** + quantile [0.5,0.2,0.8], trả median + spread (KHÔNG lấy mean — mean bị interrupt làm
bẩn); (b) `achieved = flops/sec`; (c) `roofs.attainable(...)` cho trần + bound; (d) đóng gói `KernelProfile`
(:36) với `pct_roof`, `ai`, `bound`. `main()` (:114) chạy 2 anchor — `copy(256MB)` và `gemm(8192³)` — in
bảng + `write_csv` + `plot_roofline`. Gate R0 (:127): BW ∈ [0.45,0.62] TB/s, compute ∈ [60,85] TF/s.
**ncu-debt** (`A2_ncu_debt` :27): 5 metric (SpeedOfLight, MemoryWorkload sectors/request, BankConflicts,
WarpStalls, TensorPipe) — ncu bị chặn trên box unprivileged nên mỗi kernel *đăng ký nợ* metric sẽ soi vào
ngày H100.

**4. Cổng teach-back.** (a) Giải thích tại sao `pct_roof` dùng `attainable` (trần đang chặn) chứ không phải
peak compute cố định — cho một kernel AI=1 thì hai mẫu số khác nhau bao nhiêu? (b) **Modify-and-predict:**
nếu đổi `measure_mem_bw_bytes_s` từ copy 256 MB xuống copy 256 KB, con số BW đo được sẽ *cao hơn hay thấp
hơn* 0.551 TB/s, và điều đó làm %roof của GEMV *đúng lên hay sai đi*? (Gợi ý: L2 cache 256 KB vừa khít.)

**5. Frontier.** Roofline là ngôn ngữ chung của mọi buổi review kernel ở lab 2026: "kernel này ở đâu trên
roofline, nghẽn ở đâu, fix là gì". ncu-debt = kỷ luật trung thực: khi counter bị chặn, bạn NÊU RÕ metric
mình *không* đo được thay vì bịa. Câu interview: "AI của decode-GEMV là bao nhiêu, và tại sao batching lại
đẩy nó sang phải trên roofline?" (nối về Série 1).

---

## Bài 3.1 — GEMV: kernel AI≈1, đáy của cái thang (`src/scratch_llm/kernels/gemv_triton.py` · `gemv_blockrow` :120)

> **Câu hỏi first-principles:** với kernel chỉ đọc mỗi byte MỘT lần (AI≈1 FLOP/byte, cách ridge 131 hai bậc
> độ lớn), làm sao "tốt hơn" ngoài việc *stream A ở đúng HBM peak*?
> **Số đo (aha):** naive một-warp/row **78% HBM** (431 GB/s) → blockrow coalesced **528.8 GB/s = 95.9% HBM**
> (**107% của torch.mv**). Deliverable là **GB/s, không phải FLOP/s**.

**1. Feynman — bài toán bằng lời.** GEMV = `y = A @ x`, A là `(M,N)`. Đọc A đúng một lần, mỗi phần tử một
multiply-add ⇒ `bytes = M·N·2` (bf16 A áp đảo), `FLOP = 2·M·N` ⇒ **AI ≈ 1**. Đây là shape của decode-GEMV
(Série 1 R1). Vì AI cố định ở đáy roofline, kernel giỏi = kernel *lấp đầy bus HBM*. Đánh đổi cốt lõi giữa 3
stage: **độ rộng của load** (coalescing) đấu với **số CTA lấp GPU**. Naive thua ở cả hai.

**2. Cơ chế (derive từ đầu).** Coalescing: 32 lane của một warp đọc 32 địa chỉ *liền kề* → gộp thành 1
transaction 128-byte. Nếu mỗi lane đọc rời rạc → 32 transaction → BW hụt. Ba stage:
- **naive** (:74): grid=(M,), **1 warp/row**, tile hẹp `BLOCK_N=64`. Đúng nhưng under-fill SM + load hẹp →
  78% HBM.
- **blockrow** (:120): vẫn 1 program/row nhưng tile **rộng** (`BLOCK_N` tới 4096) + `num_warps` autotune
  (`_BLOCKROW_CONFIGS` :87). Accumulator fp32 element-wise, reduce một lần cuối. Lane đọc A liền kề → 128-byte
  coalesced → **95.9% HBM**, chạm tường.
- **split** (:164): grid=(M,S), mỗi row chia S program, mỗi program `atomic_add` partial fp32 (:161). Chỉ
  thắng khi M quá ít CTA (tall-skinny N≫M); shape vuông thì chỉ *hòa* blockrow + trả giá atomic.

**3. Trace code.** `gemv_blockrow(a,x)` (:120) → `_gemv_blockrow_kernel` (:93, có `@triton.autotune`):
`row=program_id(0)`; vòng `for n0 in range(0,n_size,BLOCK_N)` (:110) load tile A + tile x → `acc += a*x`
(fp32) → `tl.sum(acc)` (:116) → store `y[row]`. Accumulator element-wise (lane-aligned qua các tile), reduce
một lần — rẻ hơn `tl.sum` mỗi vòng. **Honesty (docstring :26–32):** float4 vectorization (LDG.E.128) là
**compiler-managed** — Triton tự emit vectorized load cho `tl.load` liền mạch trên `tl.arange` rộng. Raw-CUDA
ladder sẽ thêm: `float4`/`__nv_bfloat162` packed load tay, warp-shuffle tree reduction (thay `tl.sum`),
`__ldg`. Không cái nào đổi *bound* (đã HBM-limited), chỉ đổi *bao gần tường*.

**4. Cổng teach-back.** (a) Vì sao 95.9% đã là "xong" — cái gì về mặt vật lý cấm blockrow đạt 130% HBM? (b)
**Modify-and-predict:** shape đổi từ vuông 8192² sang tall-skinny M=8, N=1e6. Stage nào (blockrow vs split)
thắng, và tại sao naive/blockrow lại *đói CTA*? (Gợi ý: grid=(M,)=8 CTA trên ~48 SM.)

**5. Frontier.** GEMV chính là forward của decode ở B=1 — "quyết định 3.4 của Série 1" (batching đẩy AI≈B
sang phải). Câu interview: "tại sao GEMV không bao giờ nhanh hơn HBM peak, còn GEMM thì có?" — câu trả lời
là AI, và nó mở đúng sang Bài 3.5.

---

## Bài 3.2 — Softmax: online recurrence, đếm số PASS qua HBM (`src/scratch_llm/kernels/softmax_triton.py` · `_softmax_online_kernel` :82)

> **Câu hỏi first-principles:** khi kernel memory-bound, runtime = số byte / BW. Vậy làm sao *giảm số byte*
> mà vẫn đúng softmax, không NaN?
> **Số đo (aha):** fused **~100% HBM (551 GB/s)**; online **3N** byte vs twopass **4N** = **1.33× ít byte
> hơn, 1.17× nhanh hơn** (khi N vượt L2).

**1. Feynman — bài toán bằng lời.** Softmax mỗi phần tử làm 1 exp + vài add ⇒ AI vài FLOP/byte, tận trái
ridge ⇒ **thuần memory-bound**. Nên "thuật toán" ở đây không phải FLOP mà là *số lần quét (pass) qua row*.
Softmax an toàn ngây thơ cần 3 pass đọc + 1 pass ghi = **4N**. Milakov–Gimelshein (arXiv:1805.02867) gộp
max + denominator vào MỘT pass = **3N**. Nếu cả row nằm gọn trong 1 tile register thì chỉ đọc 1 + ghi 1 =
**2N** (ideal). Đây CHÍNH là recurrence FA2 chạy trên key-tile — softmax là flash-attention với V=I.

**2. Cơ chế (derive từ đầu).** Online recurrence giữ `(m, d)` running và *rescale* khi max nhảy:
khi thấy tile mới có max `m_new`, mass cũ phải nhân `factor = exp(m − m_new)` để đổi gốc: `d = d·factor +
Σ exp(x − m_new)`. Bất biến: sau mọi tile, `d = Σ exp(x_i − m)` đúng với `m` = max toàn cục hiện tại. Numerics
(docstring :19–24): mọi accumulate fp32; `m_safe = 0` khi running-max = −inf (tránh `exp(nan)`); row toàn
−inf (fully masked) → `d==0` → phát **uniform 1/N** thay vì 0/0 NaN.

**3. Trace code.** `_softmax_online_kernel` (:82): pass 1 (:98) vòng tile: `m_new = max(m, max(x))`, `factor
= exp(m − m_new_safe)` (:104, =0 khi chưa có mass), `p = exp(x − m_new_safe)`, `d = d·factor + p_sum` (:107)
— đây là dòng recurrence lõi; pass 2 (:115) normalize + write. So sánh: `_softmax_twopass_kernel` (:42) có 3
pass rõ rệt (max :55, denom :63, normalize :72). `_softmax_fused_kernel` (:123) load cả row 1 lần
(`BLOCK_N≥N`), max/exp/sum trong register — 2N, stage đạt peak. Dispatcher `softmax_triton` (:157) chọn mode;
fused dùng `BLOCK_N=next_pow2(N)`. **Honesty (:26–32):** coalescing, float4, bank-conflict swizzle của
reduction đều compiler-managed; raw-CUDA thêm warp-shuffle tree reduction + float4 row load tay.

**4. Cổng teach-back.** (a) Giải thích dòng `d = d*factor + p_sum` — nếu bỏ `factor` (không rescale) thì kết
quả sai thế nào, và tại sao twopass *không* cần rescale? (b) **Modify-and-predict:** một row có outlier
`+1e4` ở tile CUỐI. twopass và online cho ra cùng đáp số — nhưng online phải làm gì ở tile cuối mà twopass
không? (Gợi ý: `m` nhảy vọt, mọi mass trước đó bị nhân `exp(m_old − 1e4)≈0`.)

**5. Frontier.** Online-softmax rescale là trái tim của FlashAttention (`corr = exp(m − m_new)` — Série sau,
A4). Câu interview: "softmax và flash-attention khác nhau chỗ nào?" — đáp: FA = softmax online **× V** fuse
vào tile-loop, không materialize N×N. 1.33× ít byte = tại sao production dùng online, không twopass.

---

## Bài 3.3 — Norms: cái giá của reduction thứ hai (`src/scratch_llm/kernels/norm_triton.py` · `_rmsnorm_kernel` :43 · `_layernorm_kernel` :68)

> **Câu hỏi first-principles:** RMSNorm có 1 reduction, LayerNorm có 2 (μ rồi σ²). Reduction thứ hai có
> làm norm CHẬM hơn không — hay nó "miễn phí" vì đã memory-bound?
> **Số đo (aha):** **cả hai ~100–101% HBM @ N≥4096**; RMS vs LN chênh **±1%** (small-N edge là noise, trung
> thực).

**1. Feynman — bài toán bằng lời.** Cả hai norm là whole-row-in-a-block: một program đọc N phần tử của row
MỘT lần vào register, reduce fp32, normalize tile đang giữ, nhân weight, ghi ra. Ideal traffic = `2·M·N·2B`
(đọc 1 + ghi 1) cho *cả hai*. Câu hỏi rung này: reduction thứ hai của LN là *extra compute trên tile đã
on-chip*, KHÔNG phải extra HBM read (x đã đọc rồi). Nên ở N lớn (memory-bound), hai norm phải đo **gần bằng
GB/s**; lợi thế 1-reduction của RMS chỉ là compute/latency edge, thấy rõ ở N nhỏ. Đây là bài học "reduction
tự do khi đã chạm tường".

**2. Cơ chế (derive từ đầu).** RMSNorm: `x / sqrt(mean(x²)+eps) · w` — một reduction (sum of squares).
LayerNorm: `(x−μ)/sqrt(σ²+eps) · w + b` — reduction 1 (μ), reduction 2 (σ² = `mean((x−μ)²)` trên *cùng tile
đang giữ*, không đọc lại HBM). Chọn dạng `mean((x−μ)²)` chứ KHÔNG dùng `E[x²]−E[x]²` (dạng sau bị cancellation
với outlier lớn). eps NẰM DƯỚI sqrt → row toàn 0 cho `sqrt(eps)>0` → không NaN.

**3. Trace code.** `_rmsnorm_kernel` (:43): load row 1 lần (:58), `ms = sum(x*x)/N` (:60 — MỘT reduction),
`rstd = 1/sqrt(ms+eps)` (:61), `y = x·rstd·w`, store. `_layernorm_kernel` (:68): load (:85), `mean =
sum(x)/N` (:87 — reduction 1), `xc = where(mask, x−mean, 0)` (:88 — tail masked không bẩn variance sum), `var
= sum(xc*xc)/N` (:89 — reduction 2 trên tile giữ, KHÔNG re-read), `rstd`, `y = xc·rstd·w (+b)`. Cả hai
`@triton.autotune` chỉ trên geometry (`_CONFIGS` :40: warps 1–16, stages 1–2), `BLOCK_N=next_pow2(N)` truyền
per-shape để cả row ở 1 block. Wrapper `rmsnorm_triton` (:106) / `layernorm_triton` (:127) flatten về (M,N)
qua `_as_2d` (:99).

**4. Cổng teach-back.** (a) Tại sao reduction thứ hai của LN "miễn phí" ở N=8192 nhưng thấy được ở N=64? Nối
tới AI. (b) **Modify-and-predict:** nếu LN đọc lại x từ HBM cho reduction σ² (thay vì dùng tile đang giữ),
traffic thành mấy N, và %HBM đo được sẽ tụt về đâu so với RMS? (Gợi ý: 3N read vs 1N → không còn ~100%.)

**5. Frontier.** RMSNorm thắng LayerNorm trong LLM hiện đại (Llama/GPT-NeoX) một phần vì 1 reduction + không
bias — nhưng bài này chứng minh **ở N lớn cả hai chạm tường như nhau**, nên lý do thật là numerics/simplicity,
không phải tốc độ raw. Câu interview: "fuse RMSNorm vào đâu để khỏi round-trip HBM?" (đáp: vào residual add +
matmul kế tiếp — đúng tinh thần Bài 3.4 fusion).

---

## Bài 3.4 — TopK: rung "đo cái thất bại" + cứu chuộc bằng fusion (`src/scratch_llm/kernels/topk_triton.py` · `_topk_kernel` :35 · `fused_softmax_topk` :127)

> **Câu hỏi first-principles:** không phải op nào cũng hợp GPU. Một selection op (chọn k lớn nhất) — tại sao
> nó KHÔNG chạm được tường HBM, và khi kernel dở thì cứu bằng cách nào?
> **Số đo (aha):** iter-max **258 GB/s = 46.9% peak** (poor GPU fit, trung thực) — nhưng **fused
> softmax+topk 3.39× nhanh hơn / 3.00× ít traffic**.

**1. Feynman — bài toán bằng lời.** TopK trả về k giá trị lớn nhất mỗi row + index. Thiết kế: 1 program/row,
load row 1 lần, rồi **k pass iterative max-extraction** — mỗi pass là một tree reduction toàn row để rút max
hiện tại rồi mask nó đi. Vấn đề: k pass tạo **chuỗi phụ thuộc tuần tự** (pass i+1 phải đợi pass i), và AI ≈ 0
(so sánh, không phải FLOP). Kernel này **occupancy/latency-bound**, %HBM thấp *by construction* — và đó là
kết quả ĐÚNG, đề bài yêu cầu "measure the failure". Cứu chuộc: đừng làm TopK nhanh hơn, hãy **fuse** để tiết
kiệm round-trip HBM.

**2. Cơ chế (derive từ đầu).** Vì sao 46.9%: mỗi pass rút 1 max phải reduce toàn N, k pass = k lần reduce
tuần tự trên cùng dữ liệu → không stream được, SM idle chờ dependency. Tie-break ĐỊNH NGHĨA rõ (torch.topk
không định nghĩa tie): **lowest column index wins** — trong các lane bằng max, lấy index nhỏ nhất qua
`tl.min`. Fusion: softmax là *monotone tăng*, nên top-k của softmax = top-k của logits **theo index**. Fuse
= đọc row 1 lần, tính denominator softmax toàn row từ chính load đó, rồi chỉ emit k probability — bỏ được
vòng ghi-softmax-rồi-đọc-lại mà đường "softmax rồi topk" riêng phải trả.

**3. Trace code.** `_topk_kernel` (:35): load row `other=-inf` (:50, lane pad không bao giờ được chọn); vòng
`for i in range(K)` (:52): `maxv = tl.max(r)` (:53), `is_max = r==maxv`, `idx = tl.min(where(is_max, offs,
n_cols))` (:56 — tie-break lowest-index), store val+idx, `r = where(offs==idx, -inf, r)` (:59 — xóa winner
cho pass sau). `_fused_softmax_topk_kernel` (:62): load 1 lần, `m = max(r)`, `denom = sum(where(mask, exp(r−m),
0))` (:80 — denominator trên TOÀN N, không chỉ top-k), rồi cùng vòng k-extraction nhưng emit `prob =
exp(maxv−m)/denom` (:87). Wrapper `_launch` (:93) chọn `num_warps` theo `block_n`; `topk_last_dim` (:118) vs
`fused_softmax_topk` (:127).

**4. Cổng teach-back.** (a) Giải thích vì sao 46.9% là số ĐÚNG chứ không phải bug — cái gì về k-pass tuần tự
cấm nó chạm 100% HBM? (b) **Modify-and-predict:** nếu k tăng từ 8 lên 64, %HBM của iter-max đi lên hay xuống,
và tỉ số fusion-win (3.4×) đổi thế nào? (Gợi ý: fusion tiết kiệm round-trip cố định ~N, còn k pass là chi phí
tuyến tính theo k.)

**5. Frontier.** TopK là MoE router (chọn top-k expert) + sampling. Bài học frontier: **khi một op dở trên
GPU, đừng đánh bóng nó — hãy fuse nó vào hàng xóm** (softmax+topk, hay router+gather). Câu interview: "tại sao
top-k của MoE router lại rẻ dù selection dở?" — đáp: k nhỏ (2–8), và fuse vào softmax + dispatch. Nối tới
A6 EP-MoE all-to-all.

---

## Bài 3.5 — GEMM: kernel DUY NHẤT băng qua ridge (`src/scratch_llm/kernels/gemm_triton.py` · `_gemm_tiled_kernel` :101 · `gemm_autotuned` :232)

> **Câu hỏi first-principles:** GEMM là kernel duy nhất AI vượt ridge (compute-bound). Vậy craft ở đây KHÔNG
> phải "chạm tường HBM" mà là "leo tới tensor-core throughput" — làm sao?
> **Số đo (aha):** **0.2% → 128.5% → 134.3% của cuBLAS-proxy** (naive 0.2 → autotuned **101.9 TF/s**);
> AI 1365, ridge 134. Ladder tái hiện đúng SHAPE siboehm.

**1. Feynman — bài toán bằng lời.** GEMM vuông N³: AI = `2·M·N·K / ((MN+NK+MK)·elem) = N/3 ≈ 1365` FLOP/byte
ở N=4096 — **xa bên phải ridge 131** ⇒ compute-bound. Nghĩa là mỗi byte A/B được *tái sử dụng* N lần, nên
trần không phải HBM mà là **tensor-core throughput**. Craft = data reuse (SMEM tile) + đưa vào tensor core
(`tl.dot`) + tune block/warp/stage. Đánh đổi: block lớn = nhiều reuse nhưng ít CTA; đây là bài toán tune.

**2. Cơ chế (derive từ đầu).** Naive: 1 program/output-element, K-loop scalar qua global memory — mỗi A-row
đọc lại N lần, mỗi B-col M lần, **0 reuse** ⇒ ~0.2% cuBLAS (L1/issue-bound). Tiled: output tile
BLOCK_M×BLOCK_N tích lũy qua K-block bằng `tl.dot` — compiler stage qua SMEM + phát lên tensor core, mỗi
tile được reuse. GROUP_M swizzle: duyệt pid theo block GROUP_M×N để các B-column một group chạm ở lại
L2-resident (giảm HBM read cho phần bên phải AI-model). Autotune trên BLOCK_M/N/K + num_warps + num_stages =
register block-tiling + software-pipelining. **%roof >100% được disclose thẳng:** cuBLAS-proxy ở đây là
`torch.matmul` default, và autotuned `tl.dot` *đánh bại heuristic mặc định* ở shape 4096³ — không phải claim
vượt super-peak.

**3. Trace code.** `_gemm_naive_kernel` (:43): `m=pid//N, n=pid%N`, K-loop `acc += sum(a*b)` (:69), 0 tiling.
`_gemm_tiled_kernel` (:101, thân dùng chung cho stage 2+3): tính GROUP_M swizzle (:123–129), `a_ptrs/b_ptrs`
block 2D (:134), vòng `for k0 in range(cdiv(K,BLOCK_K))` (:138): load A/B tile mask remainder → `acc +=
tl.dot(a,b)` (:142 — bf16×bf16→fp32 tensor-core MMA), advance con trỏ; store mask biên (:150).
`_gemm_autotuned_kernel` (:173) = cùng thân bọc `@triton.autotune` trên `_AUTOTUNE_CONFIGS` (:153, 10 config).
`gemm_tiled` (:216) = 1 config cố định (128×128×64, 8 warp, 3 stage); `gemm_autotuned` (:232) = tune;
`gemm = gemm_autotuned` (:237) là entry point. **Honesty (:19–25):** float4 global load, register
block-tiling, XOR-swizzle chống bank conflict — **compiler làm hết**: `tl.load` block 2D được vectorize/coalesce,
`tl.dot` chọn micro-tiling + SMEM layout (swizzle), `num_stages` lái async-copy pipeline. Stage 3 là "autotune
knob của compiler", KHÔNG phải "viết swizzle tay".

**4. Cổng teach-back.** (a) Tại sao naive rơi 0.2% mà không phải ~50% — cái gì cấm reuse khi 1 program lo 1
output element? (b) **Modify-and-predict:** shape đổi sang M=1 (thành GEMV). `tl.dot` với BLOCK_M=1 còn ý
nghĩa không, kernel này còn compute-bound không, và ta nên rơi về kernel nào của série? (Gợi ý: AI sụp về ≈1,
băng qua ridge NGƯỢC lại sang memory-bound → Bài 3.1.)

**5. Frontier.** GEMM là 90% FLOP của training/prefill. Trần thật: sm120 dừng ở ~102 TF/s (WGMMA là
sm_90a-only); leo tiếp là ngày H100 — **CUDA-core 32 → WGMMA 318 → warp-spec 531 → CUTLASS 630 TF/s** (design
note §4). Bài học série (design note §4): *trên memory-bound kernel, craft là chạm tường — tường LÀ trần;
trên GEMM, craft là tiling + tensor core, và 10% cuối tới cuBLAS đắt hơn giá trị của nó trừ khi bạn LÀ người
viết cuBLAS.* Câu interview: "GEMM 4096³ bị chặn bởi gì, và WGMMA/TMA thêm gì so với tl.dot?"

---

*Nối tiếp:* các kernel ISA-gated (WGMMA sm_90a, tcgen05 sm_100a) chỉ **compile-verified** trên box này —
runtime + TF/s là nợ của ngày H100/B200 (`bench/RESULTS.md` §"ISA-gated kernels"; A3 tensor-core ladder
`performance/notes` là série kế). Through-line giữ nguyên: 4 kernel memory-bound đã chạm tường 96–100% HBM;
chỉ GEMM còn đường leo, và đường đó dẫn thẳng lên tensor core Hopper.
