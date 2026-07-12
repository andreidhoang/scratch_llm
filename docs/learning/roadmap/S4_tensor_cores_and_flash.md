# Série 4 — Tensor cores (A3, raw CUDA) + Flash Attention (A4)

> **Số dòng pin theo commit `9e61d7a`.** Số đo lấy từ `bench/RESULTS.md` §A3 (dòng 686–703) và §A4
> (dòng 559–575, 658–680). Design notes gốc: `performance/notes/A3_design_note.md`,
> `performance/notes/A4_design_note.md`.

**Vì sao série này.** Série 2 (A2) đã leo GEMM bằng *register-tiling* trên CUDA core tới ~134% cuBLAS-proxy
ở shape memory-bound. Série 4 đi tiếp hai bước không thể tách rời: **(A3)** đóng băng chính cái
register-tiling ấy vào **silicon** — tensor core — và trèo từ sàn CUDA-core **4.1% → 38.9% → 81.9%** của
cuBLAS bằng cách đưa operand mỗi rung một bậc gần ALU hơn (register → SMEM → tensor-core fragment) và issue
từ sync → async; **(A4)** *fuse* A2+A3 vào một kernel attention **không bao giờ ghi ma trận N×N** — flash
attention. Mạch xuyên suốt một câu: **tensor core là register-tiling hoá đá; flash là GEMM-tiling + online
softmax fuse lại để né bức tường O(N²).** Sáu bài đi từ sàn CUDA-core lên tới FA2 fused + backward.

---

## Bài 4.1 — Naive SMEM GEMM: cái sàn CUDA-core mà tensor core trèo lên từ đó (`kernels/gemm_smem_cuda.cu` · `sgemm_smem_kernel` :16)

> **Câu hỏi first-principles:** một GEMM viết "đúng sách" (SMEM-tiled, fp32-accumulate, KHÔNG tensor core)
> chạm được bao nhiêu phần trăm cuBLAS — và vì sao con số đó là *sàn*, không phải thất bại?
> **Số đo (aha):** **3.5 TF/s = 4.1% của cuBLAS** (RESULTS.md :694). Tensor core sẽ nhân số này lên ~8×.

**1. Feynman — bài toán bằng lời.** Mỗi thread "nhận nuôi" đúng **một** ô output `C[row][col]` và tự tay
cộng dồn tích trong (inner product) dài K cho ô đó. Để không đọc lại HBM K lần, block hợp tác kéo một
**tile 32×32** của A và B vào SMEM, mọi thread trong block xài chung. Đây là *toàn bộ* bài học A2 GEMM —
nhưng phép cộng dồn là **FMA vô hướng trên CUDA core**, một lần một cặp số. Đánh đổi cốt lõi: đúng và đơn
giản (một ô/thread, một vòng `k`), nhưng thông lượng bị chặn bởi số FMA vô hướng mà scalar ALU nuốt được
mỗi chu kỳ — đó chính là lý do tồn tại của tensor core.

**2. Cơ chế (derive từ đầu).** `C[M,N] = A[M,K] @ B[K,N]`. Chia K thành `numTiles = ceil(K/32)` slab. Mỗi
slab: nạp `As[32][32]`, `Bs[32][32]`, `__syncthreads()`, rồi `acc += As[ty][k]*Bs[k][tx]` cho `k=0..31`.
Bất biến remainder: lane ngoài biên nạp `0.0f` (`gr<M && gc<K ? ... : 0`) → tích của nó = 0, không cần
epilogue riêng — cùng công thức lo cả M=1 lẫn K lẻ. FP32 accumulate với input bf16: `acc` là `float`, chỉ
**làm tròn một lần** về bf16 khi ghi ra (:48) — headroom giữ reduction dài-K ổn định.

**3. Trace code.** `gemm_smem_cuda.cu` · `sgemm_smem_kernel`: thread claim ô `(row,col)` (:22–23) → vòng
slab `t` (:27) → masked cooperative load A/B vào SMEM (:34–37) → `__syncthreads()` (:38) → inner MAC
`acc += As*Bs` unrolled (:40–43, **đây là FMA CUDA-core, dòng WMMA sẽ thay**) → `__syncthreads()` (:44) →
sau mọi slab: `C = __float2bfloat16(acc)` bounds-checked (:47–48). Launcher `sgemm_smem` (:52) set grid
`(ceil(N/32), ceil(M/32))`, block `32×32`. Wrapper Python `gemm_smem_cuda.py` · `gemm_smem` :32 JIT-compile
`-arch=sm_120`.

**4. Cổng teach-back.** (a) Vì sao mỗi thread một ô + tile SMEM 32×32 chạm HBM ~K/32 lần thay vì K lần —
tính arithmetic intensity thô. (b) *Modify-and-predict:* nếu bỏ `As`/`Bs` (đọc thẳng A,B từ global trong
inner loop), %HBM và TF/s đi đâu, và vì sao con số 4.1% vẫn là *trần* của đường CUDA-core dù có tile hay
không?

**5. Frontier.** Đây là "siboehm kernel 1–3" trong một file. Câu phỏng vấn: "GEMM của anh scalar-FMA đạt X%
cuBLAS — cuBLAS lấy 20× còn lại ở đâu?" Trả lời: tensor core (bài sau) + async pipelining (`cp.async`) + tuned
tiling. 4.1% không phải bug — nó là *định nghĩa* của sàn.

---

## Bài 4.2 — WMMA GEMM: fragment machinery + cú nhảy ~8× + vì sao FP32-accum (`kernels/wmma_gemm.py` · `wmma_gemm_kernel` :77)

> **Câu hỏi first-principles:** một lệnh `mma_sync` "16×16×16" khác một FMA vô hướng ở chỗ nào, và tại sao
> nó cho ~8× dù cùng thuật toán GEMM?
> **Số đo (aha):** **28.3 TF/s = 38.9% của cuBLAS (~8× trên sàn CUDA-core)** (RESULTS.md :695). Và FP16-accum
> là anti-example: sai số **gr%c dần theo K** — chính là lý do FP32-accum.

**1. Feynman — bài toán bằng lời.** Thay vì mỗi thread một ô, giờ **một warp (32 thread) cùng làm một tile
16×16×16 MMA trong một lệnh**. WMMA phát cho mỗi lane một "phiếu phân công" (fragment) — bạn lập trình
*tile*, không lập trình *phần tử*: `load_matrix_sync → mma_sync → store_matrix_sync`. Layout thread→value
bên trong fragment là **cố tình mờ**. Đánh đổi: bạn được ~8× throughput miễn phí (silicon làm 16³ MAC/lệnh),
nhưng mất quyền kiểm soát layout — và chính bức tường mờ đó là lý do WMMA *không* chạm được WGMMA/tcgen05
(cần descriptor tay). Đó là điểm mở của Bài 4.3.

**2. Cơ chế (derive từ đầu).** Block tile 128×128 do **8 warp** (2×4) sở hữu; mỗi warp giữ lưới **4×2**
accumulator fragment 16×16 → phủ 64×32. K stream theo slab BK=32 qua SMEM (padding APAD/BPAD=8 half để ldm là
bội của 8 → giết bank conflict). **Hai fact correctness:** (i) FP32 accumulate — `acc` fragment kiểu `float`
dù input `half`; biến thể `fp16acc` làm mỗi `mma_sync` round tổng chạy về half → sai số reduction K-sâu **lớn
dần theo K** (test drive K lên và thấy fp16-acc vượt fp32-acc). (ii) Element-exact vs `torch.matmul` ở
fp32-accum tolerance trên K-remainder + outlier — một validity bit sai sẽ *âm thầm* cho zeros.

**3. Trace code.** `wmma_gemm.py` — CUDA src trong `_CUDA_SRC`. `wmma_gemm_kernel<AccT>` (:77): khai báo
SMEM `As/Bs` padded (:82–83) + staging per-warp cho edge store (:85) → `fill_fragment(acc,0)` lưới 4×2
(:95–100) → vòng K slab `kk += BK` (:102): stage A tile (:104–109) + B tile (:110–116) zero-fill OOB →
`__syncthreads()` → vòng `kf` trên BK/WMMA_K: `load_matrix_sync` A-frag (:126) & B-frag (:131) → **`mma_sync`
lưới 4×2** (:133–137, *đây là cú 8× thay dòng FMA của Bài 4.1*) → epilogue: fast-path `store_matrix_sync` khi
tile full + FP32-acc + N chẵn (:153–156), else guarded scalar drain qua `stage` SMEM (:157–169). Hai
instantiation: `wmma_gemm` (FP32-acc, shipping) vs `wmma_gemm_fp16acc` (anti-example) (:189–192). Python
`wmma_gemm` :220, `wmma_gemm_fp16acc` :228.

**4. Cổng teach-back.** (a) Giải thích "lập trình tile, không lập trình phần tử": vì sao fragment layout mờ
lại vừa là sức mạnh (8×) vừa là trần (không lên WGMMA nổi)? (b) *Modify-and-predict:* chạy `wmma_gemm_fp16acc`
ở K=512 vs K=4096 — rel-err đường nào lớn hơn, gấp cỡ mấy lần, và vì sao FP32-accum chữa được (nghĩ theo số
bit thấp mất mỗi lần round tổng chạy)?

**5. Frontier.** Mọi tensor-core GEMM (và promotion FP8 two-level của A5) accumulate FP32 vì lý do đo được
này. Câu phỏng vấn: "vì sao WMMA đủ cho 39% mà không hơn?" → sync load (chưa `cp.async` double-buffer) + layout
do compiler quản → Bài 4.3 mở lid.

---

## Bài 4.3 — mma.sync + ldmatrix + XOR swizzle: mở lid, chạm 82% cuBLAS (`kernels/gemm_mma_sync.cu` · `gemm_mma_sync_kernel` :82)

> **Câu hỏi first-principles:** khi tự tay phát PTX `mma.sync` (bỏ WMMA), bạn phải tự lo hai thứ WMMA giấu:
> nạp operand vào đúng register layout của tensor core, và tránh bank conflict khi nạp. Làm sao?
> **Số đo (aha):** **59.0 TF/s = 81.9% của cuBLAS, rel err 6.6e-6** (RESULTS.md :696) — vượt DoD ≥60%.

**1. Feynman — bài toán bằng lời.** Bài 4.2 gọi `mma_sync` qua WMMA; giờ ta phát thẳng PTX
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`. Nhưng tensor core đòi operand đã nằm sẵn theo **layout
register warp-collective rất cụ thể** — nạp bằng tay sẽ sai. `ldmatrix.sync` là lệnh nạp SMEM→register
*collective cả warp* đúng vào layout đó. Và vì 32 lane đọc 8 hàng 8×8 cùng lúc, nếu SMEM đặt naive thì các
hàng đập cùng bank → serialize. **XOR swizzle** hoán vị chunk để hàng liền kề rơi bank khác. Đánh đổi: cực
nhọc (chỉ số fragment tay, swizzle tay) nhưng đây là craft mà WGMMA sau này *tự động hoá trong hardware*.

**2. Cơ chế (derive từ đầu).** Block 64×64, 4 warp (2×2), mỗi warp 32×32 sub-tile; per-warp atom lưới 2(m)×4(n)
với 2 k-atom/BK; `acc[2][4][4]` FP32 register sống suốt vòng K. **Swizzle:** chunk 16B (8×f16); chunk logic `c`
ở hàng `r` lưu tại physical `c XOR (r & (C-1))` — As: C=4 (mask 3), Bs: C=8 (mask 7). XOR-per-row là **song
ánh trên mỗi hàng** (invertible → giữ đúng) và map các hàng của *cùng* chunk logic sang chunk vật lý *khác nhau*
→ khác bank set. `ldmatrix` đọc 8 row-pointer; 8 hàng khác nhau ở `(r & mask)` low bit → fan across bank. Metric
đóng (ncu-debt, ncu bị chặn ở đây): `l1tex__data_bank_conflicts_...op_ld.sum ≈ 0`.

**3. Trace code.** `gemm_mma_sync.cu`: PTX wrappers `ldmatrix_x4` (:52), `ldmatrix_x2_trans` (:62, `.trans`
cho B K-major), `mma_m16n8k16` (:70, `+f` in-place FP32 accumulate). Kernel `gemm_mma_sync_kernel` (:82): SMEM
swizzled `As/Bs` (:84–85) → zero acc (:96–102) → vòng K `kt` (:105): global→SMEM 16B chunk vector 128-bit,
bounds-masked, **lưu swizzled** `pchunk = cInRow ^ (row & 3/7)` (A :117, B :138) → `__syncthreads()` (:151) →
`ldmatrix_x4` A-frag đọc SMEM swizzled (:169) + `ldmatrix_x2_trans` B-frag (:182) → **16 warp-MMA
`mma_m16n8k16`** cộng cả 2 k-atom (:187–195) → epilogue map acc→C bounds-checked theo layout m16n8 (:199–221).
Python `gemm_mma_sync` :38.

**4. Cổng teach-back.** (a) Vẽ vì sao `c XOR (r & mask)` là song ánh trên mỗi hàng và vì sao nó tách bank —
không được đổi *dữ liệu*, chỉ đổi *địa chỉ*. (b) *Modify-and-predict:* nếu bỏ swizzle (lưu `pchunk = cInRow`
thẳng), correctness còn đúng không, và metric bank-conflict + TF/s đi đâu (còn 82% không)?

**5. Frontier.** 82% = `ldmatrix` + conflict-free swizzle trên sync-load; 18% còn lại là `cp.async`
double-buffer (register→SMEM→**async** arc). Trần thật: WGMMA+TMA (Hopper sm_90a, ~318→618 TF/s) — descriptor
đã hand-decode ở `performance/artifacts/wgmma_descriptor_manual.md`, H100-gated. Câu phỏng vấn kinh điển:
"giải thích một SMEM swizzle chống bank conflict cho `ldmatrix`."

---

## Bài 4.4 — Naive attention: bức tường O(N²) mà flash tồn tại để phá (`kernels/attention_naive.py` · `naive_attention` :34)

> **Câu hỏi first-principles:** attention "đúng sách" tốn *bao nhiêu* HBM, và vì sao chi phí đó là **memory**
> O(N²) chứ không phải compute?
> **Số đo (aha):** naive peak-mem **8.1→32.1→128.2 MB** ở N=1024→2048→4096 (**×4 mỗi lần gấp đôi = N²**), fused
> phẳng 0.1→0.3 MB → **>400× nhỏ hơn @4096**; @16K score matrix = **1.0 GiB = 65536× on-chip scratch**
> (RESULTS.md :571–572).

**1. Feynman — bài toán bằng lời.** Attention sách giáo khoa là **ba op rời**, mỗi op vật chất hoá một tensor
đầy đủ ra HBM rồi đọc lại: `S = QKᵀ/√d` (N×N), `P = softmax(S)` (N×N), `O = P@V` (N×d). Kẻ giết là bước 1–2:
ma trận **N×N** phình **bậc hai** theo seq. Compute của attention là O(N²·d) *dù thế nào*; nhưng *memory* có
thể là O(N) nếu **không bao giờ ghi S**. Đó là toàn bộ tiền đề của flash — bài này dựng cái strawman để đo
tường.

**2. Cơ chế (derive từ đầu).** Đếm byte: một ma trận `(batch_heads, N, N)` = `batch_heads·N·N·itemsize`.
Gấp đôi N → ×4 byte (N²). @16K fp32 1-head = 16384²·4 = 1.0 GiB — không thể sống trong ~KB SRAM và *không cần*:
kernel tiled chỉ giữ **một block tile²** một lúc. Đối lập: fused working set = O(N·d) output + O(N) (m,ℓ) stats
+ O(tile²) scratch — số hạng N×N biến mất. `onchip_blowup_ratio = naive_score_bytes / tile_scratch_bytes` là
con số 65536×. Correctness oracle: `naive_attention` vs `F.scaled_dot_product_attention` <1e-3 (đo 6.4e-7,
:667); claim robust là **growth-ratio** (peak tuyệt đối phụ thuộc allocator).

**3. Trace code.** `attention_naive.py` · `naive_attention` (:34): scale=1/√d → **kernel 1** `s = einsum(qkᵀ)*scale`
(:53, N×N chạm HBM) → causal mask `k_idx > q_idx` fill −inf (:54–57) → **kernel 2** `p = softmax(s, dim=-1)`
(:59, N×N nữa) → **kernel 3** `o = einsum(p,v)` collapse về N×d (:62). Model memory giải tích:
`attention_memory_footprint` (:107) → `MemoryFootprint.onchip_blowup_ratio` (:99, chạy cả trên CPU gate, không
alloc). Test `tests/test_attention_naive.py` găm cả match SDPA lẫn tăng-trưởng N².

**4. Cổng teach-back.** (a) Vì sao compute là O(N²·d) *cả hai đường* nhưng chỉ memory mới O(N)-hoá được? (b)
*Modify-and-predict:* N=4096→8192, naive score-matrix byte ×mấy, fused working set ×mấy — và con số nào giữ
"flat" trên đường fused?

**5. Frontier.** Đây là "vì sao FA" made falsifiable. Câu phỏng vấn: "ước lượng HBM của attention naive ở
context 128K" → N²·d·bytes, thấy ngay OOM. Mọi thứ Bài 4.5–4.6 xây là để đường 128.2 MB kia phẳng lại.

---

## Bài 4.5 — Online softmax: recurrence Milakov cô lập (`kernels/online_softmax.py` · `online_softmax_normalizer` :43)

> **Câu hỏi first-principles:** softmax sách là **3 pass** (max, sum, chia) — mỗi pass đọc cả hàng. Làm sao
> tính softmax **đúng bit** khi chỉ thấy hàng *một tile một lúc*, không lưu cả hàng?
> **Số đo (aha):** online **khớp 3-pass 0–7e-18 (fp64)**, kể cả **+50 outlier tới ở tile CUỐI** và ×1000
> overflow-bait (RESULTS.md :573, :668). Đây là lõi số học FA fuse vào vòng tile.

**1. Feynman — bài toán bằng lời.** FA thấy score theo *tile*, không được giữ cả hàng → không thể làm pass tìm
max trước. Online softmax gộp pass 1+2 thành **một pass streaming**: giữ running-max `m` và running-denominator
`d`, và **re-base tổng cũ theo max mới** mỗi khi tile mới nâng max. Chốt: hệ số `exp(m_old − m_new)` — khi
outlier muộn đội max lên, mọi số hạng đã cộng được kéo về max mới trong O(1), *không đọc lại*. Đánh đổi: một
phép nhân chỉnh (`corr`) mỗi tile đổi lấy việc **không bao giờ lưu cả hàng** → đúng thứ cho phép né N×N.

**2. Cơ chế (derive từ đầu).** Recurrence:
```
mᵢ = max(mᵢ₋₁, xᵢ)
dᵢ = dᵢ₋₁ · exp(mᵢ₋₁ − mᵢ)  +  exp(xᵢ − mᵢ)
      └── rescale tổng cũ ──┘    └── cộng phần tử này ──┘
```
Bất biến: sau tile bất kỳ, `m` = max thật của phần đã thấy, `d` = Σ exp(xⱼ − m) trên phần đã thấy → cuối
stream cho đúng denominator an toàn. Tile đầu `m=−inf` → `corr = exp(−inf − m_new) = 0`, zero hoá `d` khởi tạo
đúng. Kill: sai dấu/thiếu `exp(m_old−m_new)` → hỏng đúng ca outlier-muộn (bug online-softmax phổ biến nhất).

**3. Trace code.** `online_softmax.py`: oracle `three_pass_softmax` (:32, cố tình *không* dùng `torch.softmax`
— so term-by-term với thuật toán sách). `online_softmax_normalizer` (:43): khởi tạo `m=−inf`, `d=0` (:51–52) →
vòng tile `start` (:54): `m_tile = xi.amax` → `m_new = max(m, m_tile)` → **`corr = exp(m − m_new)`** (:58) →
**`d = corr*d + exp(xi − m_new).sum`** (:59) → `m = m_new`. `online_softmax` (:64) gọi normalizer rồi
`exp(x−m)/d`. Test `tests/test_online_softmax.py` drive tile size + adversarial +50.

**4. Cổng teach-back.** (a) Chứng minh bằng lời bất biến "`d` = Σexp(xⱼ−m) trên phần đã thấy" được recurrence
bảo toàn qua một tile. (b) *Modify-and-predict:* nếu bỏ `corr` (thay `d = d + exp(xi−m_new).sum`) — với stream
`[1, 2, 51]` (outlier 51 ở tile cuối), `d` sai hướng nào (quá lớn hay quá nhỏ), và output softmax lệch ra sao?

**5. Frontier.** Cùng một `corr` xuất hiện y hệt trong FA2 (Bài 4.6). Đây là "streaming reduction with a moving
normalizer" — mẫu tái dùng ở FP8 attention, MoE router, log-sum-exp phân tán. Câu phỏng vấn: "viết online
softmax và nói vì sao rescale là O(1)."

---

## Bài 4.6 — Flash Attention FA2: Triton forward + backward (D-vector + atomic dQ) và model integration (`kernels/flash_attention_triton.py` · `_fa2_fwd_kernel` :37 · `_fa2_bwd_kernel` · `TritonFlashAttention`)

> **Câu hỏi first-principles:** ghép GEMM-tiling (A2) + online softmax (Bài 4.5) thế nào để một kernel tính
> attention mà **không tile nào ghi S ra HBM** — và backward tránh N×N ra sao?
> **Số đo (aha):** FA2-Triton **50.0% của SDPA (causal 48.3%) @ seq4096**, **44× nhẹ hơn naive @8K không OOM**,
> causal speedup **1.11→1.74×**; backward **gradcheck 5/5 vs SDPA autograd**; GQA KV **32→4 MB (8×)**
> (RESULTS.md :669–670). Model integration test `test_triton_attention_forward_backward` chạy thành công trên GPU.

**1. Feynman — bài toán bằng lời.** Đây là nơi A2 và A3 *fuse*: một program lo **một query tile** cho một
(batch·head), nạp Q tile một lần, **loop qua key tile** giữ online-softmax `(m, ℓ, acc)` trong fp32, chỉ ghi ra
O tile + L. Vì S sinh ra trong SRAM rồi tiêu ngay trong cùng vòng, **N×N không bao giờ chạm HBM** — SMEM hằng
số. Win của FA là **memory (không ghi S), không phải FLOP** (compute vẫn O(N²·d)). Đánh đổi: recompute rẻ để né
traffic đắt — cùng chủ đề memory-first của série serving A1.

**2. Cơ chế (derive từ đầu).** Fwd: với mỗi key tile, `S = QKᵀ·scale` → mask (padded −inf; causal
`offs_q ≥ offs_k`) → `m_new = max(m, rowmax S)` → `p = exp(S − m_new)` → `corr = exp(m − m_new)` →
`ℓ = corr·ℓ + Σp` → `acc = corr·acc + p@V` → cuối: `O = acc/ℓ`, `L = m + log(ℓ)` (logsumexp cho backward).
Causal skip: query tile không attend key vượt index mình → `k_end = q_start + BLOCK_Q`, bỏ nguyên tam giác trên
(nguồn của 1.11→1.74×). **Backward recompute-vs-store:** fwd chỉ lưu `(Q,K,V,O,L)` — toàn O(N·d), *không* P.
Backward tái tạo `S,P` từ `L` (`p = exp(s − L)`), và Jacobian-softmax gom về **D-vector**
`D_i = Σ_d O_id·dO_id = Σ_j P_ij·dP_ij` — reduction d-wide rẻ trên tensor đã có, **không cần dP** để lập → không
byte O(N²) nào băng qua ranh fwd→bwd.

**3. Trace code.**
*   *Forward Triton* `flash_attention_triton.py` · `_fa2_fwd_kernel` (:37): load Q tile fp32 (:74) → init `m_i=−inf, l_i=0, acc=0` (:76–78) → `k_end` causal (:82) → loop key tile (:84): `s = dot(q,kᵀ)·scale` (:96) → mask padded + causal (:97–99) → **online recurrence** `m_new/p/corr/l_i/acc` (:101–106, *đối chiếu Bài 4.5*) → cuối `o = acc/l_i` (:108), store O + `L = m_i + log(l_i)` (:110–112). Autotune configs (:27–35) chọn BLOCK/warps/stages theo seq (fair vs SDPA tuned). Host `flash_attention_triton_forward` :115.
*   *Backward Triton* `flash_attention_triton.py` · `_fa2_bwd_kernel`: MAP block tới Key tile `BLOCK_K`, loop ngoài qua Query tile `BLOCK_Q`. SRAM chứa `dk, dv` tích lũy cục bộ. Loop Q nạp `Q, dO, L, D` → tính `S = (Q @ K.T) * scale` → `p = exp(S - L)` → `dp = dO @ V.T` → `ds = p * (dp - D) * scale` → tích lũy `dk += ds.T @ q`, `dv += p.T @ do`, và `tl.atomic_add` cho `dQ` global (`dq_ptrs += ds @ k`). Host launcher `flash_attention_triton_backward` tính vector `df = sum(of * dof, dim=-1)` (D-vector) rồi launch grid `(cdiv(n, BLOCK_K), b)`.
*   *Autograd Wrapper & Wiring* `flash_attention_triton.py` · `TritonFlashAttention`: PyTorch autograd wrapper. Model `model.py` nhận `use_triton_attention` config trong `ModelConfig` và tự động dispatch `TritonFlashAttention.apply(q, k, v, is_causal)` trong `MultiHeadSelfAttention.forward` khi chạy trên CUDA.

**4. Cổng teach-back.** (a) Chỉ ra dòng nào trong `_fa2_fwd_kernel` ứng đúng với dòng nào của
`online_softmax_normalizer` (Bài 4.5) — và vì sao `acc` cũng phải nhân `corr` chứ không chỉ `ℓ`. (b)
*Modify-and-predict:* vì sao `dQ` lại cần `tl.atomic_add` trong khi `dK` và `dV` chỉ cần `tl.store` thông thường ở cuối kernel? (Hint: xem chiều giảm của từng gradient).

**5. Frontier.** Triton FA2 fwd+bwd hoàn chỉnh vượt qua kiểm thử autograd gradcheck 5/5. Trần thật: **FA3-class Hopper** (warp-spec producer/consumer + TMA + ping-pong + FP8, ~75% util / ~740 TF/s) — kernel đã compile-verified ở `performance/rental/kernels/` (8× wgmma + 3× TMA + 14 mbarrier + setmaxnreg, RESULTS.md :724), H100-gated. Câu phỏng vấn: "FA tiết kiệm memory hay FLOP?" → memory; "backward lưu gì?" → (Q,K,V,O,L) + recompute + D-vector; "vì sao dQ cần atomic_add?" → các block Key song song cùng ghi đè lên các hàng của dQ.

---

*Kết série:* sàn CUDA-core **4.1%** → WMMA **38.9%** → mma.sync+swizzle **81.9%** (A3), rồi tường O(N²)
(128.2 MB @4K) → online softmax (0–7e-18) → FA2 fused (**44× nhẹ hơn, 50% SDPA**, gradcheck 5/5) (A4). Mạch:
tensor core = register-tiling hoá đá; flash = A2+A3 fuse để né HBM. Bậc kế: `cp.async` double-buffer + WGMMA/TMA
(H100 day) và FA3 warp-spec.
