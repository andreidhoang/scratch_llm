# S4 — Tensor Cores + Flash Attention · First-Principles Derivations

> **Đây là gì.** Bản dẫn-xuất **từ nguyên lý gốc** cho TOÀN BỘ tầng "đá silicon + fuse" của một LLM —
> 6 micro-concept S4 (Bài 4.1→4.6): leo GEMM từ **sàn CUDA-core (4.1%)** lên **WMMA (38.9%)** lên
> **mma.sync + ldmatrix + XOR-swizzle (81.9%)** của cuBLAS, rồi phá **bức tường O(N²)** của attention
> bằng **online softmax** fuse vào **Flash Attention 2** (fwd không-ghi-S + recomputation backward). Đây
> là *derivation lab có số đo* của série `roadmap/S4_tensor_cores_and_flash.md` (reference chung). Mỗi
> mục: (1) **Câu hỏi** falsifiable, (2) **Sự thật nền tảng** (áp lực vật lý/toán ép ra thiết kế), (3)
> **Dẫn xuất** có công thức, (4) **Neo code** `file·func·line`, (5) **Hình ảnh** (ASCII + shape / stride
> / dtype + micro-example hand-traced), (6) **Số đo THẬT**, (7) **Frontier framing** + cổng phỏng vấn.
>
> **Cách dùng để re-learn (PRR).** Đọc phần *Câu hỏi* → **tự trả lời cold** (che phần dưới) → mở ra đối
> chiếu **cái gap**. Cuối doc có **checklist recall cold** + bảng số đo. Đây đúng vòng Predict → Run →
> Reconcile: bạn **commit một con số TRƯỚC** (roofline %, byte, số bit mất), rồi để nó va vào số đo thật;
> *gap* chính là bài học. Understanding = số bạn đoán ≈ số đo được; khoảng cách gọi tên constraint bạn quên.
>
> **Cảnh báo neo.** Line number **trôi** theo commit. Cite ở đây pin theo HEAD `1e7dbc6` (2026-07-14).
> Nếu lệch, `grep` tên hàm — đừng tin số dòng cứng (luật repo: verify, don't trust). Files traced:
> `src/scratch_llm/kernels/{gemm_smem_cuda.py+.cu, wmma_gemm.py, gemm_mma_sync.py+.cu, attention_naive.py,
> online_softmax.py, flash_attention_triton.py, flash_attention.py}`.
>
> **Nguồn số đo (honesty — FOP-4).** Pillar này **GPU-gated** (CUDA C++ / Triton, sm120 Blackwell). Con
> số throughput là **đo thật trên chính card này** — kéo từ `bench/RESULTS.md` §A3 (dòng ~822–831) và §A4
> (dòng ~795–808) và ghi rõ `[measured · sm120 · RESULTS.md]`. Con số **correctness / bit / byte** trong doc
> là **đo lại tươi trên CPU** bằng `python -c` (equivalence online-softmax, naive-attn vs SDPA, footprint
> arithmetic, D-vector identity, gradcheck) — chạy được không cần GPU, ghi `[measured · CPU · fresh]`. Số
> **trần H100** (WGMMA/TMA, FA3) là `[CEILING · H100-gated]` (compile-verified, chưa run) hoặc `[PREDICTED]`
> (dẫn từ roofline). "Implemented ≠ measured" — mỗi số dán nhãn.

---

## Bức tranh lớn — S4 là hai cú fuse cùng một chủ đề: "kéo operand gần ALU + đừng ghi cái N×N"

Série 3 (S3) leo GEMM bằng *register/SMEM tiling* trên CUDA core tới ~134% cuBLAS-proxy ở shape
memory-bound. S4 đi tiếp **hai bước không tách rời**, cùng một câu: **tensor core = register-tiling hoá
đá; flash = GEMM-tiling + online softmax fuse lại để né HBM.**

```
                       ÁP LỰC             LỜI GIẢI S4                      SỐ ĐO (aha)
 ┌── GEMM ladder (A3) ─────────────────────────────────────────────────────────────────────┐
 │  scalar FMA quá chậm    4.1 naive SMEM GEMM (CUDA core, fp32-accum)     3.5 TF/s = 4.1%   │
 │      ▼ operand xa ALU        │  mỗi thread 1 ô, tile 32×32 SMEM                  cuBLAS    │
 │  cần MAC 16³/lệnh       4.2 WMMA fragment MMA (m16n16k16, FP32-accum)   28.3 TF/s = 38.9% │
 │      ▼ layout do compiler    │  warp lập trình TILE không PHẦN TỬ            (~8× sàn)     │
 │  bức tường "mờ" WMMA    4.3 mma.sync + ldmatrix + XOR-swizzle (PTX tay)  59.0 TF/s = 81.9%│
 │                              │  tự nạp register-layout + né bank conflict   rel-err 6.6e-6│
 └──────────────────────────────────────────────────────────────────────────────────────────┘
                         (fuse GEMM-tiling + online softmax ↓)
 ┌── Flash ladder (A4) ─────────────────────────────────────────────────────────────────────┐
 │  attention ghi S,P N×N  4.4 naive 3-kernel attention (strawman)   128 MB @4K = ×4/doubling │
 │      ▼ O(N²) HBM wall        │  S=QKᵀ, P=softmax(S), O=P@V rời      16K → 1 GiB = 65536× SRAM│
 │  không lưu nổi cả hàng   4.5 online softmax (Milakov recurrence)   khớp 3-pass 0 (fp64)     │
 │      ▼ streaming (m,d)       │  corr=exp(m_old−m_new) re-base O(1)  late +50 outlier PASS   │
 │  fuse 3 op, đừng ghi S   4.6 Flash Attention 2 (fwd fused + bwd)   50% SDPA · 44× nhẹ @8K   │
 │                              │  (Q,K,V,O,L)+recompute+D-vector      gradcheck 5/5           │
 └──────────────────────────────────────────────────────────────────────────────────────────┘
```

**Một sự thật xuyên suốt — mỗi thiết kế là lời giải cho một *áp lực* cụ thể:**
- **A3 (compute)** — GEMM là compute-bound ở shape lớn; sàn CUDA-core bị chặn bởi *số FMA vô hướng/chu kỳ*.
  Mỗi rung kéo operand **một bậc gần ALU hơn** (register → SMEM → **tensor-core fragment**) và đổi issue
  từ **sync → async**. Con số 4.1% → 39% → 82% là *đường leo* đó, đo thật.
- **A4 (memory)** — attention là *memory*-bound O(N²) *chỉ vì* nó ghi ma trận N×N ra HBM. Compute là O(N²·d)
  *dù thế nào*; nhưng *memory* có thể là O(N) nếu **không bao giờ vật chất hoá S**. Flash = fuse để né đúng
  bức tường đó. Win của FA là **memory, không phải FLOP** — câu này là cổng phỏng vấn kinh điển.

Hai ladder gặp nhau ở Bài 4.6: FA2 kernel *bên trong* dùng `tl.dot` (tensor core của A3) để tính `QKᵀ` và
`P@V` theo tile, *bọc ngoài* bằng online softmax (4.5) để không tile nào chạm HBM với cái N×N. **Học S4 =
học áp lực → lời giải** cho cả compute (tensor core) lẫn memory (flash).

**Peak của card này (mọi % quy về đây)** `[measured · sm120 · RESULTS.md:25]`: bf16 GEMM 8192³ =
**72 TF/s**, HBM copy **0.55 TB/s**, ridge point **≈130 FLOP/B**. "cuBLAS proxy" = `torch.matmul` ở
4096³ f16→fp32, ~72–84 TF/s. Mọi "% cuBLAS" bên dưới là *của con số này*, KHÔNG phải datacenter H100.

---

## Bài 4.1 · Naive SMEM GEMM — cái sàn CUDA-core mà tensor core trèo lên từ đó

**Câu hỏi.** Một GEMM viết "đúng sách" (SMEM-tiled, fp32-accumulate, **không** tensor core) chạm được bao
nhiêu % cuBLAS — và vì sao con số đó là *sàn*, không phải thất bại?

**Sự thật nền tảng.** `C[M,N] = A[M,K] @ B[K,N]` là compute-bound ở shape lớn. Nhưng phép cộng dồn ở đây
là **FMA vô hướng trên CUDA core** — một lần một cặp `float`. Thông lượng bị chặn cứng bởi *số FMA vô
hướng scalar-ALU nuốt được mỗi chu kỳ*. Đó chính là *lý do tồn tại* của tensor core: nó làm 16³ MAC/lệnh
thay vì 1.

**Dẫn xuất — hai câu hỏi con.**

*(a) Vì sao phải tile SMEM?* Nếu mỗi thread đọc thẳng `A`,`B` từ global trong inner loop, mỗi phần tử `A`
bị đọc lại **N lần** (một lần cho mỗi cột output), `B` đọc lại **M lần** → HBM traffic khổng lồ → memory
wall. Chặn khối `TILE×TILE` (=32×32) vào SMEM để cả block xài chung: mỗi phần tử `A` giờ đọc từ HBM
`K/TILE` lần thay vì `K` lần → giảm traffic **×TILE**. Đây *toàn bộ* là bài học S3 GEMM.

*(b) Vì sao 4.1% là SÀN, không phải bug?* Tính **arithmetic intensity** của một output block `BM×BN` stream
qua K `[PREDICTED · dẫn từ roofline]`:
```
FLOPs(block)  = 2 · BM · BN · K            (mỗi ô: K nhân + K cộng)
HBM bytes     = (BM·K + K·BN) · itemsize   (đọc tile A và B qua toàn K, mỗi phần tử 1 lần)
AI = FLOP/byte = 2·BM·BN·K / ((BM+BN)·K·itemsize) = BM·BN / ((BM+BN)·itemsize/2)
              → với fp16 (2 byte):  AI = BM·BN/(BM+BN)
   TILE=32×32:  1024/64  = 16  FLOP/B
   128×128:     16384/256= 64  FLOP/B
```
Ridge của card = **130 FLOP/B**. Tile 32×32 cho AI=**16 ≪ 130** → **memory-bound**: kernel này *không đủ*
tái sử dụng dữ liệu để bão hoà compute, bất kể nó dùng CUDA core hay tensor core. Bậc kế (WMMA, 128×128)
nâng AI lên 64 (gần ridge hơn) *và* thay scalar-FMA bằng tensor-core MAC — đó là hai đòn của cú 8×. Vậy
4.1% = giao của "tile nhỏ ⇒ AI thấp" và "scalar ALU ⇒ MAC/chu kỳ thấp"; nó là *định nghĩa của sàn*.

**Fp32-accumulate với input bf16.** `acc` là `float` suốt reduction; chỉ **làm tròn MỘT lần** về bf16 khi
ghi ra. Headroom fp32 giữ reduction dài-K ổn định — luật này thành load-bearing ở Bài 4.2 (anti-example fp16).

**Neo code** (`gemm_smem_cuda.cu · sgemm_smem_kernel :16`):
```cpp
const int row = blockIdx.y * TILE + threadIdx.y;   // :22  thread "nhận nuôi" 1 ô C[row][col]
const int col = blockIdx.x * TILE + threadIdx.x;   // :23
float acc = 0.0f;                                  // :25  accumulator FP32 (input bf16)
for (int t = 0; t < numTiles; ++t) {               // :27  slab K, mỗi slab 32 sâu
  As[ty][tx] = (row<M && aCol<K) ? __bfloat162float(A[row*K+aCol]) : 0.0f;  // :34-35 masked load
  Bs[ty][tx] = (bRow<K && col<N) ? __bfloat162float(B[bRow*N+col]) : 0.0f;  // :36-37
  __syncthreads();                                 // :38
  #pragma unroll
  for (int k=0;k<TILE;++k) acc += As[ty][k]*Bs[k][tx];  // :40-43  FMA CUDA-core — dòng WMMA sẽ THAY
  __syncthreads();                                 // :44
}
if (row<M && col<N) C[row*N+col] = __float2bfloat16(acc);  // :47-48  round FP32→bf16 MỘT lần, cuối
```
Launcher `sgemm_smem :52` set `grid=(⌈N/32⌉,⌈M/32⌉)`, `block=32×32`; wrapper Python
`gemm_smem_cuda.py · gemm_smem :32` JIT-compile `-arch=sm_120` (`:24-29`).

**Bất biến remainder (không cần epilogue riêng).** Lane ngoài biên nạp `0.0f` (`:34-37`) → tích của nó = 0
→ không đóng góp reduction. Một công thức lo cả `M=1`, `K` lẻ, tile dư ở biên M/N. Đây là "the entire
remainder story", tái dùng y hệt ở 4.2 và 4.3.

**Hình ảnh — một thread, một ô, slab K qua SMEM:**
```
  A[M,K] bf16                    B[K,N] bf16
  ┌──────────────┐              ┌──────┐
  │ row ────────►│ slab t       │      │        SMEM: As[32][32], Bs[32][32] (fp32)
  └──────────────┘  (32 sâu)    │ col  │        acc(row,col) += Σ_{k=0..31} As[ty][k]·Bs[k][tx]
        │                       │  │   │                       └─ FMA vô hướng CUDA-core ─┘
        └──── nạp 32×32 tile ──►└──┼───┘        lặp numTiles = ⌈K/32⌉ slab → C[row][col] (bf16)
   dtype:  A,B bf16 → SMEM fp32 → acc fp32 → C bf16   (round MỘT lần ở :48)
   shape:  block 32×32 threads · grid (⌈N/32⌉, ⌈M/32⌉)
```
*Hand-trace* `M=N=K=32` (1 tile, 1 block): mỗi thread chạy 32 FMA fp32 rồi round → `C = A@B` element-exact
ở fp32-accum tolerance (test `test_gemm_smem_cuda.py`, gpu).

**Số đo THẬT.** `[measured · sm120 · RESULTS.md:822]` A3 R0, 4096³ f16(bf16)→out, cuBLAS proxy ~72–84 TF/s:
**3.5 TF/s = 4.1% của cuBLAS**, element-exact vs `torch.matmul`. Tensor core (4.2) nhân số này ~8×.
AI-derivation ở trên `[PREDICTED]`: tile 32×32 → AI 16 FLOP/B ≪ ridge 130 → memory-bound.

**Frontier / cổng.** Đây là "siboehm kernel 1–3" gộp một file. Câu phỏng vấn: *"GEMM scalar-FMA của anh đạt
X% cuBLAS — 20× còn lại cuBLAS lấy ở đâu?"* → tensor core (4.2) + async pipelining (`cp.async`
double-buffer) + tuned tiling. 4.1% không phải bug — nó là *sàn*. **Trait:** roofline-first / predict-the-number.

---

## Bài 4.2 · WMMA GEMM — fragment machinery, cú nhảy ~8×, và vì sao FP32-accum

**Câu hỏi.** Một lệnh `mma_sync` "16×16×16" khác một FMA vô hướng ở chỗ nào, và tại sao nó cho ~8× *dù cùng
thuật toán GEMM*?

**Sự thật nền tảng.** Tensor core là **register-tiling hoá đá**: thay vì mỗi thread một ô, giờ **một warp
(32 thread) cùng làm một tile 16×16×16 MMA trong MỘT lệnh** — silicon làm `16³ = 4096` MAC/lệnh. Bạn lập
trình *tile*, không lập trình *phần tử*. Đổi lại: layout thread→value bên trong fragment là **cố tình mờ**
(WMMA giấu) — và chính bức tường mờ đó là *trần* của WMMA (không chạm được WGMMA/tcgen05, cần descriptor
tay). Đó là điểm mở của Bài 4.3.

**Dẫn xuất — cú 8× đến từ đâu (hai đòn).**
1. **MAC/lệnh:** scalar FMA = 1 MAC/lệnh/thread; `mma_sync` m16n16k16 = `16·16·16 = 4096` MAC/lệnh/warp.
   Ngay cả chia cho 32 lane thì vẫn **128 MAC/lane/lệnh** vs 1 — đây là bậc độ lớn của tốc.
2. **AI:** block tile 128×128 (không phải 32×32) → AI = `128·128/(128+128) = 64 FLOP/B` (Bài 4.1) — gấp 4×
   sàn, gần ridge 130 hơn → ít memory-bound hơn.
   Tổng hai đòn ⇒ đo được **~8×** (3.5 → 28.3 TF/s). *Chưa* tới trần vì: (i) load còn **sync** (chưa
   `cp.async` double-buffer, ALU đợi SMEM), (ii) layout do compiler quản (không squeeze được như 4.3).

**Vì sao FP32-accum — anti-example đo được.** `mma_sync` cho chọn kiểu accumulator. FP32-accum: fragment
`acc` là `float`, reduction cả K giữ 23-bit mantissa. FP16-accum: mỗi `mma_sync` **round tổng chạy về
FP16 (10-bit mantissa)** → mỗi lần round mất ~13 bit thấp → sai số reduction **lớn dần theo K** (càng nhiều
lần round, càng dồn). Kernel biên dịch **hai instantiation từ MỘT body templated** để chứng minh:
`wmma_gemm` (FP32-acc, shipping) vs `wmma_gemm_fp16acc` (anti-example). Test drive K lên (512 → 4096) và
thấy đường FP16-acc vượt FP32-acc — đó là "why FP32 accumulate" *đo được*, không phán.

**Neo code** (`wmma_gemm.py`, CUDA trong string `_CUDA_SRC`; line = dòng file `.py`):
```cpp
template <typename AccT>
__global__ void wmma_gemm_kernel(const half* A, const half* B, float* C, int M,int N,int K) { // :78
  __shared__ __align__(16) half As[BM][BK+APAD];    // :82  APAD=8 half → ldm bội 8 → giết bank conflict
  __shared__ __align__(16) half Bs[BK][BN+BPAD];    // :83
  wmma::fragment<accumulator,16,16,16,AccT> acc[WM_FRAGS][WN_FRAGS];   // :95  lưới 4×2 per warp (64×32)
  for (i,j) wmma::fill_fragment(acc[i][j], 0);       // :96-100
  for (int kk=0; kk<K; kk+=BK) {                     // :102  slab BK=32
    // stage A tile (BM×BK) + B tile (BK×BN) vào SMEM, zero-fill OOB
    As[r][c] = (gr<M&&gc<K)? A[gr*K+gc] : 0;          // :108  (masked, như 4.1)
    Bs[r][c] = (gr<K&&gc<N)? B[gr*N+gc] : 0;          // :115
    __syncthreads();
    for (int kf=0; kf<BK/WMMA_K; ++kf) {              // :120
      wmma::load_matrix_sync(a_frag[i], &As[aRow][kf*16], BK+APAD);   // :126  SMEM→fragment
      wmma::load_matrix_sync(b_frag[j], &Bs[kf*16][bCol], BN+BPAD);   // :131
      for (i,j) wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]); // :133-137 cú 8× (thay :40-43 của 4.1)
    }
  }
  // epilogue: store_matrix_sync fast-path (tile full & FP32-acc & N chẵn) :153-156, else guarded drain :157-169
}
```
Hai kernel: `wmma_gemm` (FP32-acc) `:197` · `wmma_gemm_fp16acc` (anti-example) `:198`; Python wrapper
`wmma_gemm :220`, `wmma_gemm_fp16acc :228`.

**Hình ảnh — warp lập trình TILE, block 128×128 do 8 warp giữ:**
```
  BLOCK 128×128 output           8 warp (WARPS_M=2 × WARPS_N=4)      mỗi warp: lưới 4×2 acc-fragment 16×16
  ┌──────┬──────┬──────┬──────┐  warp giữ (BM/2)×(BN/4)=64×32       ┌──┬──┐
  │ w0   │ w1   │ w2   │ w3   │                                     │f │f │  WM_FRAGS=4 (dọc)
  ├──────┼──────┼──────┼──────┤  K stream slab BK=32 qua SMEM       ├──┼──┤  WN_FRAGS=2 (ngang)
  │ w4   │ w5   │ w6   │ w7   │  load_matrix_sync → mma_sync ×(4×2) │f │f │  → phủ 64×32
  └──────┴──────┴──────┴──────┘  → store_matrix_sync (epilogue)     └──┴──┘  = 8 fragment/warp
  dtype: A,B half(fp16) → SMEM half(+PAD) → fragment → acc FP32 → C fp32
  layout thread→value TRONG fragment = MỜ (WMMA giấu) → sức mạnh (8×) VÀ trần (không lên WGMMA)
```
*Hand-trace fp16 vs fp32 accum:* stream K=4096 với các số ~1.0; FP16-acc round tổng ~4096 lần về 10-bit →
rel-err phình ~√K·2⁻¹⁰; FP32-acc giữ 23-bit → rel-err ~2⁻²³·√K, nhỏ hơn nhiều lần. Test khẳng định thứ tự này.

**Số đo THẬT.** `[measured · sm120 · RESULTS.md:823]` A3 R1, 4096³: **28.3 TF/s = 38.9% của cuBLAS (39.4%
peak) — ~8× trên sàn CUDA-core**; FP16-accum error grows with K (FP32-accum justified, tested);
element-exact FP32-acc vs `torch.matmul`. Load còn sync (chưa `cp.async`).

**Frontier / cổng.** Mọi tensor-core GEMM production accumulate FP32 vì lý do đo được này (kể cả FP8
two-level của quant §S5). Câu phỏng vấn: *"vì sao WMMA đủ 39% mà không hơn?"* → sync load + layout compiler-
managed → Bài 4.3 mở lid. *"lập trình tile không phần tử nghĩa là gì?"* → fragment mờ. **Trait:** claims
honesty (anti-example đo được) + roofline-first.

---

## Bài 4.3 · mma.sync + ldmatrix + XOR swizzle — mở lid, chạm 82% cuBLAS

**Câu hỏi.** Khi tự tay phát PTX `mma.sync` (bỏ WMMA), bạn phải tự lo hai thứ WMMA giấu: **(i)** nạp
operand vào đúng *register layout warp-collective* của tensor core, và **(ii)** tránh **bank conflict** khi
nạp. Làm sao?

**Sự thật nền tảng.** Tensor core đòi operand đã nằm sẵn theo layout register rất cụ thể (lane nào giữ phần
tử nào). Nạp bằng vòng lặp thường sẽ *sai layout* → kết quả rác. `ldmatrix.sync` là lệnh nạp SMEM→register
**collective cả warp** đúng vào layout đó. Và vì 32 lane đọc 8 hàng của tile 8×8 cùng lúc, nếu SMEM đặt
naive thì các hàng đập **cùng bank** → serialize (bank conflict).

**Dẫn xuất — XOR swizzle là bijection tách bank.**

SMEM chia 32 bank 4-byte. Một "chunk" = 16 byte = 8 half (đúng một hàng 8×8 mà `ldmatrix` đọc). Đặt naive:
hàng `r`, chunk logic `c` ở địa chỉ `r·(chunks/row) + c` → nhiều hàng của *cùng* chunk `c` rơi *cùng*
offset-mod-bank → conflict. **Swizzle:** lưu chunk logic `c` của hàng `r` tại chunk vật lý
```
pchunk = c XOR (r & (C-1))          As: C=4 chunks/row (mask 3);  Bs: C=8 chunks/row (mask 7)
```
Hai tính chất *ép* ra correctness + no-conflict:
1. **Bijection trên mỗi hàng.** Với `r` cố định, `c ↦ c XOR (r&mask)` là hoán vị của `{0..C-1}` (XOR với
   hằng là song ánh) → **invertible** → chỉ đổi *địa chỉ*, không đổi *dữ liệu* → đọc lại đúng nếu dùng cùng
   công thức. (Đây là tại sao correctness giữ nguyên.)
2. **Tách hàng liền kề sang bank khác.** Các hàng `r` khác nhau ở low-bit `(r&mask)` → cùng chunk logic `c`
   map sang chunk vật lý *khác nhau* → khác bank set. `ldmatrix` đọc 8 row-pointer; 8 hàng fan across bank
   thay vì đập một bank → conflict ≈ 0.

**Vì sao 82%, 18% còn lại ở đâu?** 82% = `ldmatrix` (đúng layout) + swizzle (no-conflict) trên **sync**
load. 18% cuối là `cp.async` double-buffer (register→SMEM→**async** arc, ALU không đợi load). Trần thật:
WGMMA+TMA (Hopper, descriptor hoá silicon).

**Neo code** (`gemm_mma_sync.cu · gemm_mma_sync_kernel :82`; PTX m16n8k16, khác WMMA m16n16k16):
```cpp
// PTX wrappers:
ldmatrix_x4(...)        // :52  nạp 4× tile 8×8 f16 → A-operand register layout
ldmatrix_x2_trans(...)  // :62  .trans cho B K-major → B-operand (col) layout
mma_m16n8k16(...)       // :70  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32, "+f" = FP32 in-place

__shared__ __half As[BM*BK];  // :84  64×32, swizzled 16B chunk (C=4)
__shared__ __half Bs[BK*BN];  // :85  32×64, swizzled 16B chunk (C=8)
float acc[2][4][4]; ... = 0;  // :96-102  2 m-atom × 4 n-atom × 4 FP32 reg/lane (m16n8 = 128 out/32 lane)
for (int kt=0; kt<nKtiles; ++kt) {                 // :105  slab K, BK=32
  int pchunk = cInRow ^ (row & 3);  As[row*BK+pchunk*8] = ...128-bit float4 load  // :117  STORE SWIZZLED (A)
  int pchunk = cInRow ^ (row & 7);  Bs[row*BN+pchunk*8] = ...                     // :138  STORE SWIZZLED (B)
  __syncthreads();                                 // :151
  int pchunk = aChunk ^ (aRow & 3);  ldmatrix_x4(a_frag..., &As[aRow*BK+pchunk*8]);       // :168-170  READ swizzled
  int pchunk = bChunk ^ (bRow & 7);  ldmatrix_x2_trans(b_frag..., &Bs[bRow*BN+pchunk*8]); // :181-182
  for (mi,ni,ki) mma_m16n8k16(acc[mi][ni]..., a_frag..., b_frag...);  // :187-195  16 warp-MMA, cộng 2 k-atom
}
// epilogue: acc → C row-major FP32, bounds-checked theo layout m16n8 (2 hàng cách 8, 2 cột)  :199-221
```
Python `gemm_mma_sync.py · gemm_mma_sync :38` (float16 in, fp32 out).

**Hình ảnh — XOR swizzle, hàng 0..3 của cùng chunk logic c=1 (As, mask=3):**
```
 logic:  row r, chunk c=1        physical  pchunk = 1 XOR (r & 3)     bank set (4B) khác nhau
   r=0 → 1 XOR 0 = 1  ┐          r=0: chunk 1                          ┐  ldmatrix đọc 8 row-ptr
   r=1 → 1 XOR 1 = 0  │ cùng     r=1: chunk 0    ← các hàng CÙNG chunk  │  → 8 hàng rơi 8 bank set
   r=2 → 1 XOR 2 = 3  │ logic    r=2: chunk 3       logic rơi chunk     │  → conflict ≈ 0 (thay vì ×8)
   r=3 → 1 XOR 3 = 2  ┘ c=1      r=3: chunk 2       VẬT LÝ khác nhau    ┘
 KHÔNG đổi dữ liệu (bijection/hàng, đọc lại cùng công thức) — chỉ đổi ĐỊA CHỈ để tách bank.
 dtype: A,B f16 → SMEM f16 swizzled → ldmatrix → register fragment → mma → acc FP32 → C fp32
```

**Số đo THẬT.** `[measured · sm120 · RESULTS.md:824]` A3 R2, 4096³: **59.0 TF/s = 81.9% của cuBLAS (82.0%
peak); rel err 6.6e-6** — vượt DoD ≥60%. bank-conflict ≈ 0 là **ncu-debt** (ncu bị chặn trên box này; metric
để đóng H100 day: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` target ~0). Modify-and-predict
(bỏ swizzle → `pchunk=cInRow`): correctness *vẫn đúng* (chỉ địa chỉ đổi), nhưng conflict tăng ~×8 → TF/s rớt
khỏi 82%.

**Frontier / cổng.** Trần thật: **WGMMA+TMA** (Hopper sm_90a) descriptor-based, ~318→618 TF/s
`[CEILING · H100-gated]` — descriptor đã hand-decode ở `performance/artifacts/wgmma_descriptor_manual.md`,
kernel compile-verified ở `performance/rental/kernels/wgmma_gemm_sm90a.cu`. Câu phỏng vấn kinh điển:
*"giải thích một SMEM swizzle chống bank conflict cho `ldmatrix`."* **Trait:** first-principles + kernel-craft.
**Scarce-2026 bucket:** kernels = differentiator.

---

## Bài 4.4 · Naive attention — bức tường O(N²) mà flash tồn tại để phá

**Câu hỏi.** Attention "đúng sách" tốn *bao nhiêu* HBM, và vì sao chi phí đó là **memory** O(N²), *không*
phải compute?

**Sự thật nền tảng.** Attention sách giáo khoa là **ba op rời**, mỗi op vật chất hoá một tensor đầy đủ ra
HBM rồi đọc lại:
```
1. S = QKᵀ/√d      (…, N, N)   ← O(N²) byte, chạm HBM
2. P = softmax(S)  (…, N, N)   ← O(N²) byte nữa
3. O = P @ V       (…, N, d)   ← collapse về N×d
```
Kẻ giết là bước 1–2: ma trận **N×N** phình **bậc hai** theo seq. **Compute** của attention là O(N²·d) *dù
thế nào* (cả naive lẫn flash); nhưng **memory** có thể là O(N) nếu **không bao giờ ghi S**. Đó là toàn bộ
tiền đề của flash — bài này dựng strawman để *đo* bức tường.

**Dẫn xuất — đếm byte.** Một ma trận `(batch_heads, N, N)` = `batch_heads · N · N · itemsize` byte. Gấp
đôi N → **×4 byte** (N²). @16K fp32 1-head = `16384² · 4 = 1.00 GiB` — không sống nổi trong ~KB SRAM *và
không cần*: kernel tiled chỉ giữ **một block tile²** một lúc. Đối lập, fused working set:
```
onchip_blowup_ratio = naive_score_bytes / tile_scratch_bytes
                    = (N²·itemsize) / (tile²·itemsize)
   N=16384, tile=64:  16384² / 64²  =  65536×
```
Số hạng N×N **biến mất** trên đường fused — chỉ còn O(N·d) output + O(N) (m,ℓ) stats + O(tile²) scratch.
Correctness oracle: `naive_attention` vs `F.scaled_dot_product_attention` <1e-3 (claim robust là
**growth-ratio**, peak tuyệt đối phụ thuộc allocator).

**Neo code** (`attention_naive.py · naive_attention :34`):
```python
scale = 1.0 / math.sqrt(d)                                    # :48
s = torch.einsum("...qd,...kd->...qk", qf, kf) * scale         # :53  KERNEL 1 — N×N chạm HBM
if is_causal:
    s = s.masked_fill(_causal_mask(n_q,n_k,q.device), -inf)    # :54-57  k_idx > q_idx = tương lai
p = torch.softmax(s, dim=-1)                                   # :59  KERNEL 2 — N×N nữa
o = torch.einsum("...qk,...kd->...qd", p, vf)                  # :62  KERNEL 3 — collapse N×d
```
Model giải tích chạy cả trên CPU gate (không alloc): `attention_memory_footprint :107` →
`naive_score_bytes :123` (`batch_heads·N·N·itemsize`), `tile_scratch_bytes :125` (`tile²·itemsize`),
`flash_working_bytes :127`; `MemoryFootprint.onchip_blowup_ratio :99`.

**Hình ảnh — ba tensor N×N chạm HBM (naive) vs một tile² on-chip (fused):**
```
 NAIVE (3 kernel, mỗi cái ghi/đọc HBM):        FUSED FA (một kernel, tile² on-chip):
   Q,K ─► [ S : N×N ] ──HBM──►                   Q tile ─┐
          [ P : N×N ] ──HBM──►  ─► O               loop K tile: S_tile (tile²) sinh & TIÊU trong SRAM
   HBM footprint ∝ N²  (×4/doubling)               HBM footprint ∝ N·d  (PHẲNG theo N cho số hạng N×N)
   shape S,P: (batch_heads, N, N) fp32             working set: O(N·d)+O(N)+O(tile²), N×N BIẾN MẤT
```
*Hand-trace* `N=1024, d=16, 1 head, fp32`: score matrix = `1024²·4 = 4.19 MB` (một tensor N×N). Peak CUDA
đo được ~2× (vì **S và P cùng sống** + allocator). Cả hai tăng ×4/doubling = N².

**Số đo THẬT.**
- `[measured · sm120 · RESULTS.md:795,555]` peak CUDA mem (d=16, 1 head): naive **8.1 → 32.1 → 128.2 MB**
  @N=1024→2048→4096 (increments **×3.96/×3.98/doubling = N²**); fused oracle **0.1 → 0.3 MB** (phẳng) →
  **>400× nhỏ hơn @4096**. matches SDPA **6.4e-7**.
- `[measured · CPU · fresh]` footprint arithmetic (score-only, một tensor N×N): **4.2 / 16.8 / 67.1 MB**
  @1024/2048/4096 (**×4 chính xác**); N=16384 → **1.0 GiB**; `onchip_blowup_ratio = 65536.0`.
- `[measured · CPU · fresh]` `naive_attention` == SDPA (fp32, causal): max|Δ| = **3.87e-7**.

> Ghi chú honesty: analytic 4.2 MB (một N×N) < measured-peak 8.1 MB (S+P cùng sống + allocator). Cả hai là
> **clean N²**; growth-ratio là claim robust (peak tuyệt đối = allocator-state-dependent, ledger nói rõ).

**Frontier / cổng.** Đây là "vì sao FA" made falsifiable. Câu phỏng vấn: *"ước lượng HBM của attention
naive ở context 128K"* → `N²·d·bytes`, thấy ngay OOM. Mọi thứ 4.5–4.6 xây là để đường 128 MB kia phẳng lại.
**Trait:** predict-the-number (byte-counting) + memory-first.

---

## Bài 4.5 · Online softmax — recurrence Milakov cô lập

**Câu hỏi.** Softmax sách là **3 pass** (max, sum, chia) — mỗi pass đọc cả hàng. Làm sao tính softmax **đúng
bit** khi chỉ thấy hàng *một tile một lúc*, không lưu cả hàng?

**Sự thật nền tảng.** FA thấy score theo *tile*, không được giữ cả hàng → không thể làm pass tìm max trước.
Online softmax (Milakov & Gimelshein 2018) gộp pass 1+2 thành **một pass streaming**: giữ running-max `m` và
running-denominator `d`, **re-base tổng cũ theo max mới** mỗi khi tile mới nâng max.

**Dẫn xuất — recurrence + bất biến.**
```
mᵢ = max(mᵢ₋₁, xᵢ)
dᵢ = dᵢ₋₁ · exp(mᵢ₋₁ − mᵢ)  +  exp(xᵢ − mᵢ)
      └── rescale tổng cũ ──┘    └── cộng phần tử này ──┘
```
**Bất biến (chứng minh bằng lời):** giả sử sau tile `i−1`, `dᵢ₋₁ = Σ_{j đã thấy} exp(xⱼ − mᵢ₋₁)`. Sau tile
`i`, max thành `mᵢ`. Mỗi số hạng cũ mang base `exp(xⱼ − mᵢ₋₁)`; nhân cả tổng cũ với `exp(mᵢ₋₁ − mᵢ)` biến
nó thành `Σ exp(xⱼ − mᵢ)` (đúng base mới), rồi cộng số hạng mới `exp(xᵢ − mᵢ)`. Vậy `dᵢ = Σ_{đã thấy}
exp(xⱼ − mᵢ)` — bất biến giữ. Cuối stream: `m` = max thật, `d` = denominator an toàn.

**Vì sao rescale là O(1)?** Khi outlier muộn đội max lên, ta *không đọc lại* các số hạng cũ — chỉ nhân *một
số vô hướng* `corr = exp(m_old − m_new)` vào tổng đã cộng. Đó chính là thứ cho phép né N×N: bạn stream tile,
không giữ hàng. Tile đầu `m=−inf` → `corr = exp(−inf − m_new) = 0` → zero-hoá `d` khởi tạo đúng.

**Kill (bug phổ biến nhất).** Bỏ / sai dấu `corr`: các số hạng cũ **không bị crush** về base mới → tổng cũ
lingering ở giá trị to → `d` **quá LỚN** → mọi xác suất bị pha loãng, token max bị đánh giá *thấp*. Hỏng
đúng ca outlier-muộn.

**Neo code** (`online_softmax.py · online_softmax_normalizer :43`):
```python
m = torch.full((*lead,1), -inf)     # :51  running max
d = torch.zeros((*lead,1))          # :52  running denominator
for start in range(0, n, tile):     # :54  ONE streaming pass (không pass max riêng)
    xi = x[..., start:start+tile]                       # :55
    m_tile = xi.amax(dim=-1, keepdim=True)              # :56
    m_new = torch.maximum(m, m_tile)                    # :57
    corr = torch.exp(m - m_new)                         # :58  = exp(m_old − m_new); 0 ở tile đầu (m=−inf)
    d = corr * d + torch.exp(xi - m_new).sum(-1, keepdim=True)  # :59  rescale cũ + cộng mới
    m = m_new                                           # :60
```
Oracle `three_pass_softmax :32` (cố tình *không* dùng `torch.softmax` — so term-by-term với thuật toán
sách). `online_softmax :64` → `exp(x−m)/d`.

**Hình ảnh — outlier +50 tới ở tile CUỐI, stream `[1, 2, 51]` tile=1:**
```
 tile [1]:  m=1,  d = 0·0 + exp(1−1)             = 1
 tile [2]:  m=2,  d = exp(1−2)·1 + exp(2−2)      = 0.368+1 = 1.368   ← cũ (1) crushed ×exp(−1)
 tile [51]: m=51, d = exp(2−51)·1.368 + exp(51−51)= ~0 + 1 = 1.0     ← cũ crushed ×exp(−49)≈0
   p(token=51) = exp(51−51)/d = 1/1.0 = 1.0  ✓  (đúng: 51 áp đảo)
 BỎ corr (d = d + exp(xi−m_new)):  d = 1 → 2 → 3.0  ← cũ KHÔNG crush → d=3.0 (quá LỚN)
   p(token=51) = 1/3.0 = 0.333  ✗  (outlier bị pha loãng)
 dtype fp64; shape (…, 1) cho m,d
```

**Số đo THẬT** `[measured · CPU · fresh]`:
- online == 3-pass (fp64), random 4×500 rows: max|Δ| = **0.0**.
- **ADVERSARIAL** +50 outlier ở tile cuối (tile=2): max|Δ| = **0.0** — `corr` rescale làm nó đúng.
- modify-and-predict, bỏ `corr` trên `[1,2,51]`: buggy `d = 3.0` vs correct `d = 1.0` → **quá LỚN** (khớp
  dẫn xuất "stale terms không crush").
- `[measured · CPU · RESULTS.md:555]` khớp 3-pass ref <1e-6 fp64 cho mọi tile size.

**Frontier / cổng.** Cùng `corr` xuất hiện *y hệt* trong FA2 (4.6). "Streaming reduction với moving
normalizer" tái dùng ở FP8 attention, MoE router, log-sum-exp phân tán. Câu phỏng vấn: *"viết online softmax
và nói vì sao rescale là O(1)."* **Trait:** first-principles + invariant-driven correctness.

---

## Bài 4.6 · Flash Attention 2 — fwd fused (no N×N) + recomputation backward (D-vector, atomic dQ)

**Câu hỏi.** Ghép GEMM-tiling (A3, `tl.dot`) + online softmax (4.5) thế nào để một kernel tính attention mà
**không tile nào ghi S ra HBM** — và backward tránh N×N ra sao?

**Sự thật nền tảng.** Đây là nơi A3 và A4 *fuse*: một program lo **một query tile** cho một (batch·head),
nạp Q tile MỘT lần, **loop qua key tile** giữ online-softmax `(m, ℓ, acc)` trong fp32, chỉ ghi ra O tile +
L. Vì S sinh trong SRAM rồi tiêu ngay trong cùng vòng, **N×N không bao giờ chạm HBM**. Win của FA là
**memory (không ghi S), KHÔNG phải FLOP** (compute vẫn O(N²·d)). Đổi: recompute rẻ để né traffic đắt.

**Dẫn xuất — forward fused.** Với mỗi key tile:
```
S     = QKᵀ · scale                    (BLOCK_Q, BLOCK_K)   ← tl.dot (tensor core A3)
mask  : padded key → −inf; causal offs_q ≥ offs_k → −inf
m_new = max(m, rowmax S)
p     = exp(S − m_new)                  (BLOCK_Q, BLOCK_K)
corr  = exp(m − m_new)                  ← Y HỆT online softmax 4.5
ℓ     = corr·ℓ + Σⱼ p                    (rescale denominator)
acc   = corr·acc + p @ V                (rescale NUMERATOR — xem teach-back a)
cuối : O = acc / ℓ ;  L = m + log(ℓ)     (logsumexp cho backward)
```
**Vì sao `acc` CŨNG phải nhân `corr`, không chỉ `ℓ`?** `acc` giữ `Σⱼ exp(sⱼ − m_old)·vⱼ` (numerator chưa
chuẩn hoá). Khi max lên `m_new`, mọi số hạng đã cộng mang base cũ `exp(sⱼ − m_old)`; để re-base sang
`exp(sⱼ − m_new)` phải nhân `exp(m_old − m_new) = corr` — *đúng cùng* `corr` như `ℓ`. Nếu chỉ rescale `ℓ`,
tử số và mẫu số dùng base *khác nhau* → `O = acc/ℓ` sai. (Đây là cổng teach-back.)

**Causal skip (nguồn của 1.11→1.74×).** Query tile `[q_start, q_start+BLOCK_Q)` không attend key vượt index
mình → `k_end = q_start + BLOCK_Q` → bỏ nguyên **tam giác trên**. Càng nhiều tile, tam giác bỏ càng lớn →
speedup tăng theo seq.

**Dẫn xuất — backward recompute-vs-store + D-vector.** Forward chỉ lưu `(Q,K,V,O,L)` — *toàn* O(N·d), *không*
P. Backward tái tạo `P` từ `L`: `p = exp(s − L)` (vì `L = logsumexp(s)`). Softmax-Jacobian:
```
dsᵢⱼ = pᵢⱼ · ( dpᵢⱼ − Σₖ pᵢₖ dpᵢₖ )        (Jacobian của softmax hàng i)
```
Số hạng recentering `Σₖ pᵢₖ dpᵢₖ` gom về **D-vector** — và đây là mẹo then chốt:
```
Σₖ pᵢₖ dpᵢₖ = Σₖ pᵢₖ (dOᵢ · vₖ) = dOᵢ · (Σₖ pᵢₖ vₖ) = dOᵢ · Oᵢ
⟹  Dᵢ = Σⱼ Pᵢⱼ dPᵢⱼ = Σ_d Oᵢ_d · dOᵢ_d
```
Vế phải là **reduction d-wide rẻ trên tensor ĐÃ CÓ** (`O`, `dO`) — *không cần* materialize `dP` để lập `D`.
Vậy **không byte O(N²) nào băng qua ranh fwd→bwd**. Rồi `dV = Pᵀ dO`, `dP = dO Vᵀ`, `ds = P⊙(dP − D)`, và
`dQ = ds K · scale`, `dK = dsᵀ Q · scale`.

**Vì sao `dQ` cần `tl.atomic_add` mà `dK`,`dV` chỉ cần `tl.store`?** Backward kernel này parallelize theo
**KEY tile** (`pid_k = program_id(0)`), loop *ngoài* qua Query tile. `dK,dV` của một key tile được tích luỹ
*cục bộ* trong SRAM (`dk, dv` register) suốt Q-loop rồi ghi **một lần** ở cuối → mỗi key tile do đúng một
program viết → không đua → `tl.store`. Nhưng với query `i` cố định, `dQᵢ = Σ_{mọi key tile} dsᵢⱼ @ kⱼ` —
**mọi** program (mọi key tile) cùng cộng vào *cùng* hàng `dQᵢ` → đua → phải `tl.atomic_add`. (Chiều reduction
của `dQ` là chiều bị parallelize; `dK/dV` reduction dọc Q nằm gọn trong một program.)

**Neo code.**
*Forward* (`flash_attention_triton.py · _fa2_fwd_kernel :37`):
```python
q = tl.load(q_ptrs, ...).to(tl.float32)             # :74   Q tile nạp MỘT lần
m_i = tl.full((BLOCK_Q,), -inf); l_i = 0; acc = 0    # :76-78  online state fp32
k_end = (q_start+BLOCK_Q) if IS_CAUSAL else n_ctx    # :82   causal skip tam giác trên
for k_start in range(0, k_end, BLOCK_K):             # :84   loop KEY tile
    s = tl.dot(q, tl.trans(k)) * scale               # :96   QKᵀ qua tensor core (A3)
    s = tl.where(k_mask, s, -inf)                    # :97   padded key
    if IS_CAUSAL: s = tl.where(offs_q>=offs_k, s,-inf) # :98-99
    m_new = tl.maximum(m_i, tl.max(s, axis=1))       # :101  ↔ online_softmax :57
    p = tl.exp(s - m_new[:,None])                    # :102  ↔ :59 (exp shift)
    corr = tl.exp(m_i - m_new)                       # :103  ↔ :58 corr
    l_i = corr*l_i + tl.sum(p, axis=1)               # :104  ↔ :59 denominator
    acc = corr[:,None]*acc + tl.dot(p, v)            # :105  acc CŨNG ×corr (P@V qua tensor core)
    m_i = m_new                                      # :106
o = acc / l_i[:,None]                                # :108
tl.store(o_ptrs, o, ...)                             # :109-110
tl.store(l_ptrs, m_i + tl.log(l_i), ...)             # :112  L = logsumexp cho backward
```
*Backward* (`flash_attention_triton.py · _fa2_bwd_kernel :173`, grid theo KEY tile):
```python
dk = tl.zeros((BLOCK_K, D)); dv = tl.zeros((BLOCK_K, D))   # :236-237  tích luỹ CỤC BỘ trong SRAM
for q_idx in range(q_start, n_ctx, BLOCK_Q):               # :243  loop ngoài qua Query tile
    s  = tl.dot(q, tl.trans(k)) * scale                    # :266  recompute S (không lưu ở fwd)
    p  = tl.exp(s - lse[:,None])                           # :272  P từ L (logsumexp saved)
    dp = tl.dot(do, tl.trans(v))                           # :276  dP = dO Vᵀ
    ds = p * (dp - d_vec[:,None]) * scale                  # :279  softmax-bwd qua D-vector (scale folded)
    dk += tl.dot(tl.trans(ds), q)                          # :283  dK += dsᵀ Q  (cục bộ)
    dv += tl.dot(tl.trans(p), do)                          # :284  dV += Pᵀ dO  (KHÔNG scale)
    tl.atomic_add(dq_ptrs, tl.dot(ds, k), ...)             # :292-293  dQ += ds K  ← ATOMIC (đua giữa key tile)
tl.store(dk_ptrs, dk, ...); tl.store(dv_ptrs, dv, ...)     # :302-303  store MỘT lần (không đua)
```
Host `flash_attention_triton_backward :306` tính `df = Σ (of·dof)` (D-vector, `:329`) trước, launch grid
`(cdiv(n,BLOCK_K), b)` (`:340`). Autograd wrapper `TritonFlashAttention :394` (`save_for_backward(Q,K,V,O,L)`
`:407`). Oracle thuần torch: `flash_attention.py · flash_attention_forward :26` (fwd tiled) +
`FlashAttentionPyTorch.backward :118` (recompute `s :132`, `p :135`, `d_vec :137`, `ds :140`, `dq :141`,
`dk :142`) — chú ý oracle giữ `scale` ở `dq/dk` thay vì fold vào `ds`; hai cách tương đương, `dv` không bao
giờ có scale.

**Hình ảnh — fwd fused (không N×N chạm HBM) + bwd recompute:**
```
 FORWARD (1 program = 1 query tile):        BACKWARD (1 program = 1 KEY tile):
   Q_tile ──load once──►                       load K,V tile
   for K_tile:                                 for Q_tile:  (loop ngoài)
     S = QKᵀ (SRAM) ─┐ online (m,ℓ,acc)          recompute S,P từ (Q,K,V,L)
     P = exp(S−m)    │ corr=exp(m−m_new)          ds = P⊙(dP−D)·scale
     acc=corr·acc+PV │ ← S TIÊU trong SRAM        dk += dsᵀQ ┐ cục bộ SRAM → tl.store cuối (1 writer)
   O = acc/ℓ ; L=m+log ℓ                          dv += PᵀdO ┘
   HBM out: O (N·d) + L (N) — N×N BIẾN MẤT        dQ += dsK → tl.atomic_add (nhiều key tile đua 1 hàng)
   saved cho bwd: (Q,K,V,O,L) toàn O(N·d)         D-vector Dᵢ=ΣO·dO = ΣP·dP (không cần dP để lập)
```
*Hand-trace L→P:* `L = m + log(ℓ) = logsumexp(s)` ⇒ `p = exp(s − L) = exp(s)/Σexp(s)` = đúng softmax, tái
tạo từ *một* số/hàng (L) thay vì cả hàng P. Đó là recompute-vs-store: đổi FLOP recompute lấy 0 byte N×N.

**Số đo THẬT.**
- `[measured · sm120 · RESULTS.md:797]` FA2-Triton fwd: **50.0% của SDPA (causal 48.3%) @ seq4096**; FA
  peak **40–96 MB** vs naive **105–4256 MB = 44× nhẹ hơn @8K, no OOM**; causal speedup **1.11→1.74×**.
  (Prior 4090 = 53%; nay 50% pinned trên card này.)
- `[measured · sm120 · RESULTS.md:798]` backward **gradcheck 5/5 vs SDPA autograd**; GQA KV **32→4 MB** (n_kv
  4 vs 32), runtime flat.
- `[measured · CPU · fresh]` flash oracle == SDPA (fp64, causal, ragged N=130, tile=64): max|Δ| = **6.66e-16**.
- `[measured · CPU · fresh]` D-vector identity `Σ_d O·dO == Σ_j P·dP`: max|Δ| = **1.78e-15**.
- `[measured · CPU · fresh]` `FlashAttentionPyTorch.backward` gradcheck vs autograd (fp64): dQ **6.9e-16**,
  dK **1.6e-15**, dV **8.9e-16**.

**Frontier / cổng.** Trần thật: **FA3-class Hopper** (warp-spec producer/consumer + TMA + ping-pong + FP8,
~75% util / **~740 TF/s**) `[CEILING · H100-gated]` — kernel compile-verified ở
`performance/rental/kernels/fa3_attention_hopper.cu` (8× wgmma + 3× TMA + 14 mbarrier + setmaxnreg,
RESULTS.md:852). Câu phỏng vấn: *"FA tiết kiệm memory hay FLOP?"* → memory; *"backward lưu gì?"* → (Q,K,V,O,L)
+ recompute + D-vector; *"vì sao dQ cần atomic_add?"* → key tile song song cùng ghi các hàng dQ. **Trait:**
memory-first + recompute-vs-store. **Scarce-2026 bucket:** inference/kernels = differentiator.

---

## Bảng số đo THẬT (chạy lại được — DoD-là-profile)

| Bài | Đo | Kết quả | Nguồn | Ý nghĩa |
|---|---|---|---|---|
| — | card peaks (bf16 GEMM / HBM / ridge) | **72 TF/s · 0.55 TB/s · 130 FLOP/B** | sm120 · RESULTS.md:25 | mọi % quy về đây |
| 4.1 | naive SMEM GEMM (CUDA core) | **3.5 TF/s = 4.1% cuBLAS** | sm120 · :822 | sàn; scalar FMA |
| 4.1 | AI của tile T×T (fp16) | 32×32 → **16 FLOP/B ≪ 130** | PREDICTED (roofline) | tile nhỏ ⇒ memory-bound |
| 4.2 | WMMA GEMM (m16n16k16) | **28.3 TF/s = 38.9% cuBLAS (~8×)** | sm120 · :823 | tensor core = 8× sàn |
| 4.3 | mma.sync+ldmatrix+swizzle | **59.0 TF/s = 81.9% cuBLAS · rel-err 6.6e-6** | sm120 · :824 | mở lid → 82% |
| 4.4 | naive attn peak-mem vs N | **8.1→32.1→128.2 MB (×~4/doubling)** | sm120 · :795 | O(N²) memory wall |
| 4.4 | score matrix @16K vs SRAM | **1.0 GiB = 65536× tile² scratch** | CPU fresh + :795 | không sống nổi on-chip |
| 4.4 | footprint arithmetic (score-only) | 4.2/16.8/67.1 MB (**×4**); 16K=**1.0 GiB** | CPU fresh | clean N² |
| 4.4 | naive_attention == SDPA (fp32) | max\|Δ\| = **3.87e-7** | CPU fresh | oracle đúng |
| 4.5 | online == 3-pass (fp64) | max\|Δ\| = **0.0** (random + late +50) | CPU fresh + :555 | recurrence đúng bit |
| 4.5 | bỏ corr trên [1,2,51] | buggy d=**3.0** vs correct **1.0** (quá LỚN) | CPU fresh | corr là cả bài |
| 4.6 | FA2-Triton fwd | **50.0% SDPA (causal 48.3%) @4096** | sm120 · :797 | win = memory |
| 4.6 | FA peak vs naive @8K | **44× nhẹ hơn, no OOM**; causal 1.11→1.74× | sm120 · :797 | constant SMEM |
| 4.6 | backward gradcheck | **5/5 vs SDPA autograd**; GQA KV 32→4 MB | sm120 · :798 | recompute+D-vector |
| 4.6 | flash oracle == SDPA (fp64) | max\|Δ\| = **6.66e-16** | CPU fresh | tiled == full |
| 4.6 | D-vector identity | max\|Δ\| = **1.78e-15** | CPU fresh | ΣO·dO=ΣP·dP |
| 4.6 | FA bwd gradcheck (fp64) | dQ **6.9e-16** · dK **1.6e-15** · dV **8.9e-16** | CPU fresh | không cần dP |
| ceiling | WGMMA+TMA / FA3 Hopper | ~318→618 TF/s · ~740 TF/s (~75% util) | CEILING · H100-gated | trần thật |

Chạy lại số CPU-fresh: `PYTHONPATH=src python` với `online_softmax`, `attention_naive`, `flash_attention`
(script trong scratchpad phiên này — mọi số là đo, không phán).

---

## Checklist recall COLD (che phần trên, tự trả lời — đây là cách re-learn)

1. **4.1** Tính AI của tile GEMM `BM×BN` (fp16). Vì sao 32×32 → 16 FLOP/B ⇒ memory-bound? Vì sao 4.1% là
   *sàn* chứ không phải bug?
2. **4.1** Vì sao mỗi thread một ô + tile SMEM 32×32 chạm HBM ~K/32 lần thay vì K? Vì sao zero-fill OOB lo
   được cả M=1 và K lẻ mà không cần epilogue riêng?
3. **4.2** `mma_sync` m16n16k16 cho bao nhiêu MAC/lệnh vs scalar FMA? Cú ~8× đến từ hai đòn nào?
4. **4.2** Vì sao FP32-accum? FP16-accum sai *theo hướng nào* khi K tăng, và vì sao (nghĩ theo bit mất mỗi
   lần round tổng)?
5. **4.3** `pchunk = c XOR (r & mask)`: chứng minh là bijection *trên mỗi hàng* (⇒ correctness giữ) VÀ tách
   bank (⇒ conflict≈0). Bỏ swizzle → correctness còn đúng không, TF/s đi đâu?
6. **4.3** `ldmatrix.sync` giải bài toán gì mà load thường không? Vì sao 82%, 18% cuối ở đâu?
7. **4.4** Vì sao compute là O(N²·d) *cả hai đường* nhưng chỉ *memory* mới O(N)-hoá được? N=16K → score
   matrix bao nhiêu GiB, = mấy× on-chip scratch?
8. **4.5** Dẫn bất biến `d = Σexp(xⱼ−m) trên phần đã thấy` qua một tile. Bỏ `corr` trên `[1,2,51]` → `d` sai
   *hướng nào* (to/nhỏ), output softmax lệch ra sao?
9. **4.6** Chỉ dòng nào của `_fa2_fwd_kernel` ứng dòng nào của `online_softmax_normalizer`. Vì sao `acc`
   *cũng* phải ×`corr`, không chỉ `ℓ`?
10. **4.6** Backward lưu gì (không lưu gì)? Dẫn `Dᵢ = Σ_d Oᵢ_d dOᵢ_d = Σⱼ Pᵢⱼ dPᵢⱼ`. Vì sao `dQ` cần
    `atomic_add` mà `dK/dV` chỉ `store`?

> Trả lời cold được cả 10 = **S4 thật sự OWNED** (interview-grade). Vấp câu nào → mở đúng mục đó, hoặc
> blank-slate hàm tương ứng (`raise NotImplementedError`/xoá kernel body → test đỏ → tự dẫn → xanh + đo lại)
> để re-own tay. **Một ✅ Bài là OWNED — không re-derive** (FOP-7) trừ khi re-own bằng blank-slate protocol.

---

*Cross-ref: `docs/learning/roadmap/S4_tensor_cores_and_flash.md` (spine reference · 6 Bài) ·
`docs/learning/PROGRESS.md` (ledger 89 Bài) · `docs/learning/roadmap/README.md` (perf roadmap · 41 Bài) ·
`bench/RESULTS.md` §A3 (dòng ~822–831) + §A4 (dòng ~795–808) + §compile-verified (dòng ~841–852, H100
ceilings) · `performance/notes/A3_design_note.md` + `A4_design_note.md` (verdict floor/ceiling) · sibling:
`derivations/S3_cuda_core_kernels.md` (CUDA-core floor này leo lên từ đó) ·
`derivations/M2_transformer_forward.md` (attention/softmax SỐNG ở đây). Concept kế: A5 quant (NVFP4/AWQ,
FP32-accum promotion) + S6 distributed/ISA (WGMMA/TMA/FA3 H100 day).*
