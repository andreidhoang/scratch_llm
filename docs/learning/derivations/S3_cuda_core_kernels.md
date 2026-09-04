# S3 — CUDA-core Kernels · First-Principles Derivations

> **Đây là gì.** Bản dẫn-xuất **từ nguyên lý gốc** cho TOÀN BỘ tầng kernel CUDA-core của forward pass —
> 6 micro-concept S3 (Bài 3.0→3.5). Đây là *derivation lab có số đo* của série
> `roadmap/S3_cuda_core_kernels.md` (reference chung). Mỗi mục: (1) **Câu hỏi** falsifiable, (2) **Sự
> thật nền tảng** (áp lực vật lý/toán ép ra thiết kế), (3) **Dẫn xuất** có công thức, (4) **Neo code**
> `file·func·line`, (5) **Hình ảnh** (ASCII + shape/stride/dtype + micro-example), (6) **Số đo THẬT**,
> (7) **Frontier framing** + cổng phỏng vấn.
>
> **Cách dùng để re-learn.** Đọc phần *Câu hỏi* → **tự trả lời cold** (che phần dưới) → mở ra đối chiếu
> **cái gap**. Cuối doc có **checklist recall cold** + bảng số đo. Cách này đúng vòng PRR (Predict → Run →
> Reconcile): bạn commit một con số TRƯỚC, rồi để nó va vào số đo thật; *gap* chính là bài học.
>
> **⚠️ NEO ĐÃ DI DỜI (sửa 31/08).** Commit `8030db6` (29/07) tái cấu trúc kernels từ dạng phẳng
> sang cây theo họ phép toán. Bảng dịch — dùng nó khi doc dưới nhắc đường dẫn cũ:
> `kernels/gemv_triton.py` → `kernels/gemm/triton/gemv.py` · `gemm_triton.py` →
> `kernels/gemm/triton/tiled.py` · `softmax_triton.py` → `kernels/reduce/softmax.py` ·
> `topk_triton.py` → `kernels/reduce/topk.py` · `norm_triton.py` → `kernels/norm/normalize.py` ·
> `paged_decode_triton.py` → `kernels/attention/decode/paged.py`. Các bench `bench/{gemv,gemm,norm,softmax}.py`
> gộp vào `bench/kernels/` (`run.py` + thư mục con cùng tên họ). Số đo trong doc KHÔNG đổi — cùng kernel,
> cùng card.
>
> **Cảnh báo neo.** Line number **trôi** theo commit. Cite ở đây pin theo HEAD `1e7dbc6` (2026-07-14).
> Nếu lệch, `grep` tên hàm — đừng tin số dòng cứng (luật repo: verify, don't trust). Files:
> `bench/_harness.py`, `bench/kernel_roofline.py`, `src/scratch_llm/kernels/{gemv,softmax,norm,topk,gemm}_triton.py`,
> và các bench `bench/{gemv,softmax,norm,topk,gemm}.py`.
>
> **Nguồn số đo (honesty — FOP-4).** Pillar này **GPU-gated** (Triton/CUDA, sm120). Số throughput là số
> **đo thật trên chính card này** — kéo từ `bench/RESULTS.md` §A2 và `performance/notes/A2_design_note.md`,
> gắn nhãn `[measured · sm120 · RESULTS.md]`. Vài con số roofline/equivalence là **CPU-measured trong
> chính doc này** (`python -c`, gắn nhãn `[CPU-measured · doc này]`). Con số chưa có trong ledger (leo lên
> trần Hopper) là **[PREDICTED]** / design-target, ghi rõ. "Implemented" ≠ "measured": chỉ số đo/profiled
> mới là kết quả.

---

## Bức tranh lớn — 5 kernel, MỘT câu hỏi

Toàn bộ forward pass của một LLM, khi mổ xuống tầng kernel, chỉ gồm **năm hình dạng tính toán**. Với mỗi
kernel, câu hỏi first-principles **duy nhất** là: *"kernel này bị chặn bởi cái gì — HBM bandwidth hay
tensor-core throughput?"* Roofline trả lời câu đó bằng MỘT con số — **arithmetic intensity** `AI = FLOP/byte`
— và một cái ngưỡng — **ridge** `= compute_peak / HBM_BW`.

```
                    TFLOP/s (log)
   compute roof  ┤━━━━━━━━━━━━━━━━━━━━━━━━━━●  gemm(8192³) 72.1 TF/s = 100% cmp
   72.1 TF/s     ┤                       ╱ ▲
                 ┤                     ╱   │  bên PHẢI ridge = COMPUTE-BOUND
                 ┤       mem roof    ╱     │  (chỉ 1 kernel: GEMM, AI≈1365)
                 ┤     (slope=BW)  ╱       │
                 ┤              ╱ ← ridge = 131 FLOP/byte (compute/BW)
                 ┤    ●───────╱ copy 551 GB/s = 100% mem
                 ┤  ╱ ▲     │
                 ┤╱   │     │  bên TRÁI ridge = MEMORY-BOUND (4 kernel: AI≈1)
                 └────┴─────┴──────────────────────────► AI = FLOP/byte (log)
                     GEMV  softmax/norm      GEMM
                     AI≈1   AI≈few            AI≈1365

  4/5 kernel nằm ĐÁY roofline (AI≈1) → memory-bound VĨNH VIỄN, không mẹo nào vượt HBM peak.
  1/5 kernel (GEMM) băng qua ridge → compute-bound, và MỚI có chuyện "leo lên cuBLAS/tensor-core".
```

**Năm hình dạng — và áp lực mỗi cái trả lời:**

| Bài | Kernel | AI | Bound | Áp lực (pressure) → thiết kế |
|---|---|---|---|---|
| 3.0 | roofline harness | — | — | **đo lường**: "% của trần" vô nghĩa nếu trần là số datasheet → đo peak trên CHÍNH card |
| 3.1 | GEMV `y=Ax` | ≈1 | mem | **bandwidth**: đọc A một lần → chỉ có thể *lấp đầy bus HBM* (coalescing) |
| 3.2 | softmax | ≈vài | mem | **byte**: runtime = bytes/BW → giảm *số pass* qua row (online recurrence) |
| 3.3 | RMSNorm/LayerNorm | ≈1 | mem | **byte**: reduction thứ 2 của LN có "miễn phí" khi đã chạm tường không? |
| 3.4 | TopK | ≈0 | occupancy | **latency**: selection là chuỗi phụ thuộc tuần tự → op DỞ trên GPU, cứu bằng *fusion* |
| 3.5 | GEMM `C=AB` | ≈1365 | compute | **compute**: mỗi byte reuse N lần → craft = tiling + tensor core, không phải HBM |

Một throughline (design note §1): **trên memory-bound kernel, craft là *chạm tường* — tường LÀ trần.
Trên GEMM, craft là tiling + tensor core.** Học S3 = học *đọc AI → biết trần → biết "tốt" là bao nhiêu %*.

Một chú ý honesty xuyên suốt (ADR-0011, Triton-primary): **Triton lo coalescing / float4 / bank-conflict
swizzle**. Mỗi bài phải trung thực *cái gì compiler làm hộ* vs *cái raw-CUDA ladder phải viết tay* — không
bịa ra một "CUDA ladder" ta không viết.

---

## 3.0 · Roofline harness — cái thước đo CẢ HAI mái nhà

**Câu hỏi.** Làm sao — *trước khi* tối ưu — biết một kernel *nên* bị chặn bởi memory hay compute, và
"tốt" nghĩa là **bao nhiêu phần trăm của trần NÀO**?

**Sự thật nền tảng.** Một con số throughput không có nghĩa gì nếu không có mẫu số. Mẫu số đúng KHÔNG phải
peak datasheet (số marketing, không đạt được) mà là **peak card này thật sự đạt** — đo bằng chính torch.
Và mẫu số phải là *trần đang chặn kernel này*, không phải một trần cố định: một kernel AI=1 so với compute
peak sẽ mãi mãi "1% peak" và con số đó vô dụng; so với *memory roof tại AI của nó* mới cho headroom thật.

**Dẫn xuất.** Roofline là một **mái nhà hai dốc** trong toạ độ log-log `(AI, FLOP/s)`:
```
attainable(AI) = min( compute_peak ,  AI · HBM_BW )
                       └─ mái phải phẳng ┘  └─ mái trái dốc (BW × AI) ┘
```
Hai dốc gặp nhau ở **ridge** — nơi hai giới hạn bằng nhau:
```
AI · HBM_BW = compute_peak   ⟹   ridge = compute_peak / HBM_BW
```
Với card này: `ridge = 72.1 TF/s / 0.551 TB/s ≈ 131 FLOP/byte`. Kernel có `AI < ridge` → rơi bên TRÁI →
memory-bound; `AI > ridge` → bên PHẢI → compute-bound.

Hai peak, đo bằng chính torch (không lấy datasheet):
- **HBM BW**: copy 256 MB device→device, `2·numel·elem_size / sec` (đọc + ghi = **2** byte/phần tử).
- **Compute**: GEMM vuông 8192³ qua cuBLAS, `2·n³ / sec` — cuBLAS đã tuned nên đây là trần *thật đạt được*.

**%roof = achieved / attainable(AI)** = headroom tuyệt đối tới trần ĐANG chặn. Đó là toàn bộ toán roofline.
*Vì sao dùng `attainable` chứ không phải compute peak cố định?* Cho kernel AI=1 (GEMV): `attainable ≈ 1·BW ≈
0.55 TF/s`, còn compute peak = 72 TF/s — chênh **131×**. Nếu lấy compute peak làm mẫu số, GEMV mãi "0.76%"
và ta không bao giờ biết nó đã *chạm tường HBM*. `attainable` cho đúng "95.9% của cái nó CÓ THỂ đạt".

**Neo code** (`bench/_harness.py`):
```python
def measure_mem_bw_bytes_s() -> float:                       # :50
    a = torch.empty(1 << 26, device="cuda", dtype=torch.float32)  # 64Mi = 256 MB
    b = torch.empty_like(a)
    ms = bench_ms(lambda: b.copy_(a))[0]
    return 2 * a.numel() * a.element_size() / (ms * 1e-3)     # đọc+ghi = 2 byte/elem

def measure_compute_peak_flops_s(dtype, n=8192) -> float:    # :58
    a = torch.randn(n, n, device="cuda", dtype=dtype); b = torch.randn(n, n, ...)
    ms = bench_ms(lambda: torch.matmul(a, b))[0]
    return 2.0 * n**3 / (ms * 1e-3)                           # cuBLAS = trần thật-đạt

@dataclass(frozen=True)
class Roofs:                                                  # :67
    @property
    def ridge(self): return self.compute_flops_s / self.bw_bytes_s          # :79
    def attainable(self, flops, bytes_):                                    # :83
        mem_bound = (flops / bytes_) * self.bw_bytes_s
        return (mem_bound, "mem") if mem_bound < self.compute_flops_s else (self.compute_flops_s, "cmp")
```
Và `profile` (`bench/kernel_roofline.py:48`) đóng gói phép đo:
```python
def profile(name, fn, flops, bytes_, roofs):                 # :48
    med, lo, hi = bench_ms(fn)                               # median (không mean!)
    achieved = flops / (med * 1e-3)
    attainable, bound = roofs.attainable(flops, bytes_)
    return KernelProfile(..., ai=flops/bytes_, pct_roof=100.0*achieved/attainable, bound=bound)  # :54-63
```
`bench_ms` (`_harness.py:21`) dùng `triton.testing.do_bench` với **L2 flush mỗi rep** + quantile
`[0.5, 0.2, 0.8]` → trả **median** (typical) + spread p20–p80 (trust). *Không lấy mean* — mean bị OS
interrupt làm bẩn.

**Hình ảnh — dữ liệu chảy qua thước:**
```
fn (một kernel) ──► bench_ms ──► (median ms, p20, p80)   [do_bench, L2-flush/rep]
                                        │
      flops (FLOP), bytes_ (byte) ──────┤
                                        ▼
     achieved = flops / (ms·1e-3)   [FLOP/s]
     ai       = flops / bytes_      [FLOP/byte]  ── vị trí trên trục hoành
     attainable, bound = roofs.attainable(flops, bytes_)   ── trần đang chặn
     pct_roof = 100 · achieved / attainable    ── % headroom tới trần đó
                                        ▼
     KernelProfile(name, ms, spread_pct, gflops, gbps, ai, pct_roof, bound)  [dataclass :36]
```
Hai anchor trong `main()` (`kernel_roofline.py:140-155`) tự-kiểm cái thước: `copy(256MB)` (flops=numel,
bytes=2·numel·4) phải rơi ~100% **mem**; `gemm(8192³)` (flops=2n³, bytes=3n²·2) phải rơi ~100% **cmp**.
Nếu thước đặt sai, hai anchor sẽ lệch — đó là gate R0 (`:128-129`: BW ∈ [0.45, 0.62] TB/s, compute ∈
[60, 85] TF/s).

**Số đo THẬT.**
- Peaks: **HBM 0.551 TB/s · bf16 72.1 TF/s · ridge 131** — gate PASS `[measured · sm120 · RESULTS.md]`.
- Anchor: **copy 551 GB/s = 100.1% mem · gemm 72.1 TF/s = 100.0% cmp**, spread < 1.2% — thước tự đặt đúng
  CẢ hai mái nhà `[measured · sm120 · RESULTS.md]`.
- CPU-check của toán ridge: `72.1e12 / 0.551e12 = 130.9 FLOP/byte` `[CPU-measured · doc này]` ≈ 131 ✓.
- **ncu-debt**: `A2_ncu_debt` (`kernel_roofline.py:27`) đăng ký **5 metric** (SpeedOfLight,
  MemoryWorkload sectors/request, BankConflicts, WarpStalls, TensorPipe) — ncu **BLOCKED** trên box
  unprivileged (`ERR_NVGPUCTRPERM`) → mỗi kernel *đăng ký nợ* metric để soi ngày H100.

> **Neo → gap thường gặp.** Nếu đo BW bằng copy 256 **KB** thay vì 256 MB, con số sẽ *cao hơn giả tạo*
> (256 KB < L2 48 MB → copy chạy ở L2 bandwidth, không phải DRAM) → làm %roof của mọi memory-bound kernel
> *sai lệch xuống* (mẫu số phồng). Đó là lý do harness chọn 256 MB: đủ lớn để tràn L2, đo DRAM thật.

**Frontier / cổng.** Roofline là **ngôn ngữ chung** của mọi buổi kernel review ở lab 2026: "kernel này ở
đâu trên roofline, nghẽn ở đâu, fix là gì". ncu-debt = kỷ luật honesty: khi counter bị chặn, NÊU RÕ metric
không đo được thay vì bịa. Gate = "AI của decode-GEMV là bao nhiêu, và vì sao batching đẩy nó sang phải?".
Trait = **roofline-first / predict-the-number** (FOP-3). Scarce bucket: kernels (differentiator 2026).

---

## 3.1 · GEMV — kernel AI≈1, đáy của cái thang

**Câu hỏi.** Với kernel đọc mỗi byte đúng MỘT lần (`AI ≈ 1 FLOP/byte`, cách ridge 131 tận **hai bậc độ
lớn**), làm sao "tốt hơn" ngoài việc *stream A ở đúng HBM peak*?

**Sự thật nền tảng.** `y = A @ x` với `A: (M, N)`. Đọc A đúng một lần, mỗi phần tử một multiply-add:
```
bytes = M·N·2 (A, bf16 áp đảo) + N·2 (x) + M·2 (y)          ≈ M·N·2
FLOP  = 2·M·N   (một mul + một add / phần tử)
AI    = 2MN / (MN·2) = 1 FLOP/byte                          ← ĐÁY roofline
```
Vì AI **cố định** ở đáy, không thuật toán nào đổi được *bound* (đã memory-bound). Kernel "giỏi" = kernel
*lấp đầy bus HBM*. Deliverable là **GB/s, không phải FLOP/s** — báo FLOP/s cho kernel AI=1 là giấu sự thật.

**Dẫn xuất — coalescing.** 32 lane của một warp đọc 32 địa chỉ. Nếu 32 địa chỉ đó *liền kề* → GPU gộp
thành **1 transaction 128-byte** (coalesced). Nếu rời rạc → **32 transaction** → BW hụt 32×. Ba stage là
ba mức lấp bus khác nhau — đấu giữa **độ rộng load** (coalescing) và **số CTA lấp GPU**:
1. **naive** — grid `(M,)`, **1 warp/row**, tile hẹp `BLOCK_N=64`. Đúng, nhưng 1 warp/CTA under-fill SM +
   load hẹp → chỉ **78% HBM**.
2. **blockrow** — vẫn 1 program/row nhưng tile **rộng** (`BLOCK_N` tới 4096) + `num_warps` autotune. Lane
   đọc A liền kề → 128-byte coalesced → **95.9% HBM**, chạm tường.
3. **split** — grid `(M, S)`, mỗi row chia S program, mỗi program `atomic_add` partial fp32. Chỉ thắng khi
   M quá ít CTA (tall-skinny `N ≫ M`); shape vuông thì chỉ *hòa* blockrow + trả giá atomic.

*Vì sao accumulator element-wise `(BLOCK_N,)` fp32, reduce một lần cuối — không `tl.sum` mỗi vòng?* Vì
lane giữ **lane-aligned** qua các tile (`acc[j] += a[j]*x[j]`), một reduction cuối cùng rẻ hơn reduce mỗi
iteration; và fp32 tránh mất đuôi khi cộng dồn N số hạng bf16.

**Neo code** (`src/scratch_llm/kernels/gemv_triton.py`):
```python
@triton.autotune(configs=_BLOCKROW_CONFIGS, key=["n_size"])   # :92  BLOCK_N∈{512,1024,2048,4096}×warps{2,4,8}
@triton.jit
def _gemv_blockrow_kernel(a_ptr, x_ptr, y_ptr, m_size, n_size, stride_am, stride_an, BLOCK_N):
    row = tl.program_id(0)                                     # :104  một program / row
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)               # :109  element-wise fp32 accumulator
    for n0 in range(0, n_size, BLOCK_N):                       # :110
        offs = n0 + tl.arange(0, BLOCK_N)
        mask = offs < n_size
        a = tl.load(a_ptr + row*stride_am + offs*stride_an, mask=mask, other=0.0).to(tl.float32)  # :113  coalesced
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += a * x                                           # :115  lanes stay lane-aligned
    y = tl.sum(acc, axis=0)                                    # :116  MỘT reduction cuối
    tl.store(y_ptr + row, y.to(y_ptr.dtype.element_ty))
```
`gemv_naive` (`:74`): `BLOCK_N=64, num_warps=1` (fixed, cố tình *không* autotune — đó là floor).
`gemv_split` (`:164`): `atomic_add` ở `:161`, `y` phải là fp32 (atomics yêu cầu). Byte model của bench:
`_bytes` (`bench/gemv.py:29`) = `m*n*elem + n*elem + m*elem`.

**Honesty (docstring `:26-32`).** float4 vectorization (`LDG.E.128` — nạp 8 bf16/lane thành một load
128-bit) là **compiler-managed**: Triton tự emit vectorized load cho `tl.load` liền mạch trên `tl.arange`
rộng. Raw-CUDA ladder sẽ thêm: `float4`/`__nv_bfloat162` packed load tay, warp-shuffle tree reduction (thay
`tl.sum`), `__ldg`. Không cái nào đổi *bound* (đã HBM-limited), chỉ đổi *bao gần tường*.

**Hình ảnh — data journey (8192², bf16):**
```
A: (8192, 8192) bf16    row-major, stride=(8192, 1)      x: (8192,) bf16
      │
grid=(8192,)  ── program row r đọc A[r, :] một lần ──►  warp lanes đọc A[r, offs]
      │                                                  offs LIỀN KỀ ⇒ 1 txn 128B  (coalesced)
   BLOCK_N=4096 tile ── acc[(4096,)] fp32 += a·x ── lặp 2 tile (8192/4096)
      │
   tl.sum(acc) ─► y[r]  scalar  ─► y: (8192,) bf16

  bytes = 8192·8192·2 + 8192·2 + 8192·2 ≈ 1.343e8   FLOP = 2·8192² = 1.342e8   ⇒ AI = 1.000
```

**Số đo THẬT.**
- **naive 431 GB/s = 78% HBM** (one-warp/row, load hẹp) `[measured · sm120 · RESULTS.md]`.
- **blockrow 528.8 GB/s = 95.9% HBM** — **107% của torch.mv** (vượt vendor GEMV) `[measured · sm120 ·
  RESULTS.md]`. Đây là "xong": vật lý cấm vượt HBM peak, 95.9% là chạm tường.
- AI của shape 8192²: `1.342e8 / 1.343e8 = 1.000 FLOP/byte` `[CPU-measured · doc này]`.

> **Neo → gap.** Vì sao 95.9% *đã là xong*? Vì `attainable = 1·BW = HBM peak`; không có cách nào đọc A
> nhanh hơn bus HBM. 4.1% còn lại là launch/tail overhead, không phải "thuật toán tồi". Blockrow đạt 130%
> HBM là **bất khả** (vi phạm bảo toàn: byte phải qua bus).

**Frontier / cổng.** GEMV chính là **forward của decode ở B=1** (Série 1 R1) — mỗi bước sinh một token đọc
lại toàn bộ weight một lần. Batching đẩy `AI ≈ B` sang phải trên roofline (weight đọc một lần, amortize qua
B row) → đó là vì sao B≈128 mới crossing sang compute-bound. Gate = "vì sao GEMV không bao giờ nhanh hơn
HBM peak, còn GEMM thì có?" — câu trả lời là AI, mở thẳng sang Bài 3.5. Trait = **roofline-first**.

---

## 3.2 · Softmax — online recurrence, đếm số PASS qua HBM

**Câu hỏi.** Khi kernel memory-bound, `runtime = bytes / BW`. Vậy làm sao *giảm số byte* mà vẫn đúng
softmax (an toàn số học, không NaN)?

**Sự thật nền tảng.** Softmax mỗi phần tử làm 1 exp + vài add ⇒ AI vài FLOP/byte, tận trái ridge ⇒ **thuần
memory-bound**. Nên "thuật toán" ở đây KHÔNG phải FLOP mà là *số lần quét (pass) qua row*. Ít pass = ít
byte = ít thời gian.

**Dẫn xuất — đếm pass (Milakov–Gimelshein, arXiv:1805.02867).** Safe-softmax ngây thơ cần biết max toàn
row (chống overflow `exp`) rồi denominator rồi normalize:
```
twopass:  pass1 max │ pass2 Σexp(x−m) │ pass3 normalize+write   = 3 read + 1 write = 4N
online:   pass1 (max + denom cùng lúc, rescale) │ pass2 write    = 2 read + 1 write = 3N
fused:    load row 1 lần vào register, max/exp/sum on-chip │ write = 1 read + 1 write = 2N (ideal)
```
Tỉ số: `online / twopass = 3N / 4N = 3/4` → **online moves 4/3 = 1.33× ít byte hơn**.

**Dẫn xuất — online recurrence (trái tim FA2).** Vấn đề: gộp max + denom vào một pass, nhưng `Σ exp(x−m)`
cần `m` = max toàn cục *đã biết trước*. Mẹo: giữ `(m, d)` **running** và **rescale** khi max nhảy. Khi thấy
tile mới có max `m_new`, mọi mass cũ tính theo gốc `m` phải đổi gốc:
```
d_new = d · exp(m − m_new) + Σ_tile exp(x − m_new)
              └─ factor rescale ─┘
```
**Bất biến:** sau mọi tile, `d = Σ_{đã thấy} exp(x_i − m)` đúng với `m` = max hiện tại. Chứng minh factor:
mass cũ là `Σ exp(x_i − m_old)`; muốn đổi sang gốc `m_new`, nhân `exp(m_old − m_new)` cho mỗi số hạng vì
`exp(x_i − m_old)·exp(m_old − m_new) = exp(x_i − m_new)`. QED. *Bỏ factor = kết quả SAI* khi max nhảy giữa
chừng; twopass **không** cần rescale vì nó đã biết `m` toàn cục ở pass 2.

**Numerics (adversarial-hardened, docstring `:19-24`).** Mọi accumulate fp32. `m_safe = 0` khi running-max
= `−inf` (tránh `exp(nan)` cho tile toàn masked). Row toàn `−inf` (fully masked) → `d == 0` → phát **uniform
1/N** thay vì `0/0 = NaN`.

**Neo code** (`src/scratch_llm/kernels/softmax_triton.py`):
```python
@triton.jit
def _softmax_online_kernel(x_ptr, y_ptr, stride_row, n_cols, BLOCK_N):     # :83 (@jit :82)
    m = tl.full((), float("-inf"), tl.float32); d = tl.zeros((), tl.float32)  # :96-97
    for start in range(0, n_cols, BLOCK_N):                                # :98  MỘT read pass
        x = tl.load(x_row + offs, mask=offs<n_cols, other=float("-inf")).to(tl.float32)  # :100
        m_new = tl.maximum(m, tl.max(x, axis=0))                           # :101
        m_new_safe = tl.where(m_new == float("-inf"), 0.0, m_new)          # :102
        factor = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_new_safe)) # :104  rescale mass cũ
        p = tl.exp(x - m_new_safe)                                         # :105  tail lane exp(−inf)=0
        p_sum = tl.where(m_new == float("-inf"), 0.0, tl.sum(p, axis=0))
        d = d * factor + p_sum                                             # :107  ← DÒNG recurrence lõi
        m = m_new
    # ... pass 2 (:113) normalize + write
```
So sánh: `_softmax_twopass_kernel` (`def :43`) có **3 pass rõ rệt** — max (`:55-60`), denom (`:63-68`),
normalize (`:72-79`). `_softmax_fused_kernel` (`def :124`) load cả row 1 lần (`BLOCK_N ≥ N`), max/exp/sum trong
register — 2N, stage đạt peak; dispatcher chọn `BLOCK_N = next_pow2(N)` (`:176`).

**Honesty — the L2 caveat (load-bearing, `bench/softmax.py` docstring `:17-24`).** 1.33× online-vs-twopass
là bound **DRAM**: chỉ hiện khi re-read *miss cache*. Card này có **48 MB L2**; ở M=N=4096, cả tensor 32 MB
nằm gọn trong L2 → re-read của twopass là **L2 hit, không phải DRAM** → tỉ số *đo* (1.17×) *understate* tỉ
số analytic (1.33×). Bench in CẢ hai: byte ledger analytic (exact) + latency đo (cái L2 cho phép), không
lẫn lộn. Đây chính là insight đẻ ra FlashAttention: online softmax cần khi KV *không* fit SRAM.

**Hình ảnh — recurrence với outlier ở tile cuối:**
```
row = [ tile0: 1,2,3 ][ tile1: 2,1,0 ][ tile2: 1e4, 0, 0 ]   (N=9, BLOCK_N=3)

tile0:  m=−inf→3,  factor=0 (chưa có mass),  d = 0 + Σexp(x−3)      = ~1.5
tile1:  m=3,       factor=exp(3−3)=1,          d = 1.5·1 + Σexp(x−3)  = ~1.6
tile2:  m=3→1e4,   factor=exp(3−1e4)≈0,        d = 1.6·0 + exp(0)+...  = ~1.0
        └─ mọi mass trước bị nhân ≈0 vì outlier "reset" gốc — ĐÚNG, vì exp(3−1e4)≈0 thật.
   twopass KHÔNG cần bước này: pass1 đã tìm m=1e4 trước, pass2 dùng luôn.
```
Data journey: `x: (M, N) bf16, stride=(N,1)` → grid `(M,)` → mỗi row `(m,d)` scalar fp32 running → `y: (M,
N)` cùng dtype. Ideal traffic `= 2·M·N·2B`.

**Số đo THẬT.**
- **fused ~100% HBM (551 GB/s)** — stage đạt peak `[measured · sm120 · RESULTS.md]`.
- **online 3N vs twopass 4N = 1.33× ít byte, 1.17× nhanh hơn** (khi N vượt L2) `[measured · sm120 ·
  RESULTS.md]`.
- **online recurrence == `torch.softmax` bit-exact**: `max|Δ| = 0.0` trên row có outlier `+1e4` ở tile
  CUỐI (tile=256, N=4096) `[CPU-measured · doc này]` — chứng minh factor-rescale là cái làm nó đúng.
- byte ratio analytic: `4N/3N = 1.333×` `[CPU-measured · doc này]`.

> **Neo → gap.** Nếu bỏ `factor` (không rescale): khi max nhảy ở tile cuối, `d` vẫn giữ mass tính theo `m`
> cũ (gốc sai) → denominator sai → softmax sai. twopass thoát vì biết `m` toàn cục trước khi cộng. Online
> *trade* một lần rescale để đổi lấy một pass ít hơn — 1.33× ít byte.

**Frontier / cổng.** Online-softmax rescale (`corr = exp(m − m_new)`) là **trái tim FlashAttention** (A4):
FA = softmax online **× V** fuse vào tile-loop, KHÔNG materialize ma trận score N×N. "Softmax và FA khác
nhau chỗ nào?" → FA là softmax online nhân V, tránh N×N HBM. Production dùng online, không twopass, vì 1.33×
ít byte. Trait = **predict-the-number** (đếm pass trước khi đo). Scarce bucket: kernels/inference.

---

## 3.3 · Norms — cái giá của reduction thứ hai

**Câu hỏi.** RMSNorm có **1** reduction (Σx²), LayerNorm có **2** (μ rồi σ²). Reduction thứ hai có làm norm
CHẬM hơn không — hay nó "miễn phí" vì đã memory-bound?

**Sự thật nền tảng.** Cả hai norm là *whole-row-in-a-block*: một program đọc N phần tử của row **một lần**
vào register, reduce fp32, normalize tile đang giữ, nhân weight, ghi ra. Ideal traffic = `2·M·N·2B` (đọc 1
+ ghi 1) cho **cả hai**. Reduction thứ hai của LN là *extra compute trên tile đã on-chip*, **KHÔNG phải
extra HBM read** — x đã ở register rồi. Nên ở N lớn (memory-bound), hai norm phải đo **gần bằng GB/s**;
lợi thế 1-reduction của RMS chỉ là compute/latency edge, thấy rõ ở N nhỏ (nơi latency > byte time).

**Dẫn xuất.**
```
RMSNorm:  y = x / sqrt(mean(x²) + eps) · w              — 1 reduction: Σx²
LayerNorm: y = (x − μ) / sqrt(σ² + eps) · w + b         — 2 reduction: μ = Σx/N, rồi σ² = Σ(x−μ)²/N
```
*Vì sao dạng `mean((x−μ)²)` chứ KHÔNG `E[x²] − E[x]²`?* Dạng sau (hai reduction rời) bị **catastrophic
cancellation** với outlier lớn: `E[x²]` và `E[x]²` đều khổng lồ, hiệu của chúng mất chính xác. Dạng
`Σ(x−μ)²` (two-pass thật trên *cùng tile đang giữ*, không re-read HBM) giữ chính xác. *Vì sao eps DƯỚI
sqrt?* Row toàn 0 → `sqrt(0 + eps) = sqrt(eps) > 0` → không chia 0, không NaN.

**Neo code** (`src/scratch_llm/kernels/norm_triton.py`):
```python
@triton.jit
def _rmsnorm_kernel(...):                                    # :45 (@autotune :43, @jit :44)
    x = tl.load(x_ptr + row*stride_xm + cols, mask=mask, other=0.0).to(tl.float32)  # :58  1 HBM read
    ms = tl.sum(x * x, axis=0) / N                           # :60  ← THE ONE reduction
    rstd = 1.0 / tl.sqrt(ms + eps)                           # :61  eps under sqrt → zero row safe
    y = (x * rstd) * w; tl.store(...)

@triton.jit
def _layernorm_kernel(...):                                  # :70 (@autotune :68, @jit :69)
    x = tl.load(...).to(tl.float32)                          # :85  1 HBM read (giống RMS!)
    mean = tl.sum(x, axis=0) / N                             # :87  reduction 1
    xc = tl.where(mask, x - mean, 0.0)                       # :88  tail masked, không bẩn variance sum
    var = tl.sum(xc * xc, axis=0) / N                        # :89  reduction 2 trên TILE GIỮ (no re-read)
    rstd = 1.0 / tl.sqrt(var + eps)                          # :90
    y = (xc * rstd) * w; (+b if HAS_BIAS)                    # :93-95
```
Cả hai `@triton.autotune` chỉ trên geometry (`_CONFIGS :40`: warps 1–16, stages 1–2), `BLOCK_N =
next_pow2(N)` truyền per-shape để cả row ở 1 block. Wrapper `_as_2d` (`:99`) flatten `(..., N) → (M, N)`.
Byte model bench (`bench/norm.py:54`): `(2·M·N + weight[+bias])·elem`.

**Hình ảnh — reduction thứ 2 sống ở đâu:**
```
     HBM                    REGISTER (on-chip)                 HBM
  x[row,:] ──1 read──►  [x0 x1 ... x_{N-1}]  ──► μ = Σx/N   (reduction 1, on tile)
   (N·2B)                     │  giữ nguyên tile   └─► σ²= Σ(x−μ)²/N (reduction 2, CÙNG tile)
                              ▼
                       y = (x−μ)·rstd·w + b  ──1 write──► y[row,:] (N·2B)

  reduction 2 KHÔNG chạm HBM lần nữa (x còn trong register) ⇒ "miễn phí" khi byte-time áp đảo.
  Ở N=64: tile nhỏ, byte-time ~ latency-time → reduction 2 lộ ra như compute edge.
```

**Số đo THẬT.**
- **cả RMS + LN ~100–101% HBM @ N ≥ 4096** — reduction thứ 2 KHÔNG làm chậm khi đã chạm tường `[measured ·
  sm120 · RESULTS.md]`.
- **RMS vs LN chênh ±1%** (small-N edge là noise, trung thực — không claim RMS nhanh hơn khi nó không)
  `[measured · sm120 · RESULTS.md]`.

> **Neo → gap.** Nếu LN *đọc lại* x từ HBM cho reduction σ² (thay vì dùng tile đang giữ): traffic thành 3N
> read + 1N write = **4N** thay vì 2N → %HBM đo được **tụt ~½** so với RMS. Cái làm reduction 2 "miễn phí"
> chính là *giữ x trong register* — mất tính chất đó là mất luôn tính miễn phí.

**Frontier / cổng.** RMSNorm thắng LayerNorm trong mọi LLM 2026 (Llama/Qwen/DeepSeek/Gemma) — nhưng bài
này chứng minh **ở N lớn cả hai chạm tường như nhau**, nên lý do thật là **numerics/simplicity** (1
reduction, không bias, không centering — Bài M2.4), KHÔNG phải tốc độ raw. "Fuse RMSNorm vào đâu để khỏi
round-trip HBM?" → vào residual-add + matmul kế tiếp (tinh thần fusion Bài 3.4). Trait = **claims honesty**
(±1% = noise, không thổi phồng). Scarce bucket: kernels.

---

## 3.4 · TopK — đo cái THẤT BẠI, rồi cứu chuộc bằng fusion

**Câu hỏi.** Không phải op nào cũng hợp GPU. Một *selection* op (chọn k lớn nhất) — vì sao nó KHÔNG chạm
được tường HBM, và khi kernel dở thì cứu bằng cách nào?

**Sự thật nền tảng.** TopK trả về k giá trị lớn nhất mỗi row + index. Thiết kế: 1 program/row, load row 1
lần, rồi **k pass iterative max-extraction** — mỗi pass là một tree reduction toàn row để rút max hiện tại
rồi mask nó đi. Hai thứ giết throughput: (a) k pass tạo **chuỗi phụ thuộc tuần tự** (pass `i+1` phải đợi
pass `i` xong mới biết mask ai), (b) `AI ≈ 0` (so sánh, không phải FLOP). Kernel này **occupancy/latency-
bound**, %HBM thấp *by construction* — và đó là kết quả **ĐÚNG**, đề bài yêu cầu "measure the failure".

**Dẫn xuất — vì sao 46.9% là con số đúng.** Mỗi pass rút 1 max phải reduce toàn N; k pass = k lần reduce
**tuần tự** trên cùng dữ liệu → SM idle chờ dependency (không stream được như GEMV). Bandwidth hiệu dụng
`= bytes / time`; time bị latency của chuỗi phụ thuộc đè, không phải byte → GB/s rơi xa dưới peak.

**Dẫn xuất — cứu chuộc bằng fusion.** Đừng làm TopK nhanh hơn (bất khả, bản chất op tuần tự) — **fuse** để
tiết kiệm round-trip HBM. Softmax **monotone tăng** ⇒ top-k của softmax = top-k của logits **theo index**.
Nên: đọc row 1 lần, tính denominator softmax toàn row từ *chính load đó*, rồi chỉ emit k probability. Bỏ
được vòng "ghi softmax N → đọc lại N" mà đường "softmax rồi topk" riêng phải trả:
```
2-pass:  softmax(read N, write N) + topk(read N, write k)  ≈ 3N + k  moved/row
fused:   read N once, write 2k (k prob + k idx)            ≈ N + 2k   moved/row
ratio ≈ 3N / N = 3×  (khi k ≪ N)
```

**Neo code** (`src/scratch_llm/kernels/topk_triton.py`):
```python
@triton.jit
def _topk_kernel(...):                                       # :36 (@jit :35)
    r = tl.load(x_ptr + row*stride_xm + offs, mask=mask, other=float("-inf")).to(tl.float32)  # :50  pad −inf
    for i in range(K):                                       # :52  k pass TUẦN TỰ (serial dep chain)
        maxv = tl.max(r, axis=0)                             # :53  full-row tree reduction
        is_max = r == maxv
        idx = tl.min(tl.where(is_max, offs, n_cols), axis=0) # :56  tie-break: LOWEST index wins
        tl.store(val_ptr + ... + i, maxv); tl.store(idx_ptr + ... + i, idx)
        r = tl.where(offs == idx, float("-inf"), r)          # :59  xóa winner cho pass sau

@triton.jit
def _fused_softmax_topk_kernel(...):                         # :63 (@jit :62)
    r = tl.load(...)                                         # :76  ĐỌC ROW 1 LẦN
    m = tl.max(r, axis=0)                                    # :79
    denom = tl.sum(tl.where(mask, tl.exp(r - m), 0.0), axis=0)  # :80  denom trên TOÀN N (không chỉ top-k)
    for i in range(K):                                       # :83  cùng k-extraction
        maxv = tl.max(r, axis=0); ...
        prob = tl.exp(maxv - m) / denom                      # :87  emit softmax prob của logit chọn
        ...
```
*Vì sao tie-break lowest-index?* `torch.topk` KHÔNG định nghĩa thứ tự tie → ta ĐỊNH NGHĨA rõ: trong các
lane bằng max, lấy index nhỏ nhất qua `tl.min` (`:56`). Bench de-tie bằng `+ arange·1e-6` để so bit-exact.

**Hình ảnh — chuỗi phụ thuộc tuần tự (k=3):**
```
r = [5, 9, 2, 9, 1]   (N=5, tie ở index 1 và 3)

pass0:  max=9, is_max@{1,3}, idx=min(1,3)=1 → emit (9, 1);  r=[5,−∞,2,9,1]
          │ (pass1 PHẢI đợi pass0 ghi −∞ mới đúng)
pass1:  max=9, idx=3 → emit (9, 3);  r=[5,−∞,2,−∞,1]
          │
pass2:  max=5, idx=0 → emit (5, 0)

  ── mỗi mũi tên ▼ là một dependency: SM không thể chạy pass1 trước khi pass0 xong ⇒ latency-bound.
  GEMV/softmax stream liên tục; TopK khựng k lần ⇒ 46.9%, KHÔNG phải bug.
```
Data journey: `x: (M, N)` → `values: (M, k)` (x's dtype hoặc fp32 cho fused) + `indices: (M, k)` int64.

**Số đo THẬT.**
- **iter-max 258 GB/s = 46.9% peak** — the honest poor-GPU-fit `[measured · sm120 · RESULTS.md]`.
- torch.topk (radix-select) chỉ **15.7% peak** — kể cả vendor cũng dở, không phải lỗi code ta (design note
  §2 table) `[measured · sm120 · A2_design_note.md]`.
- **fused softmax+topk: 3.39× nhanh hơn / 3.00× ít traffic** `[measured · sm120 · RESULTS.md]`.
- fusion ideal-traffic ratio (N=32768, k): `3.00×` (201.4 MB → 67.1 MB) `[CPU-measured · doc này]`.

> **Neo → gap.** Nếu k tăng 8 → 64: %HBM của iter-max *xuống* (nhiều pass tuần tự hơn, latency chồng thêm);
> fusion-win (3.4×) *giảm* vì fusion tiết kiệm round-trip cố định ~N, còn k pass là chi phí tuyến tính theo
> k — cả hai đường đắt thêm theo k, tỉ số co lại.

**Frontier / cổng.** TopK là **MoE router** (chọn top-k expert) + sampling. Bài học frontier: **khi một op
dở trên GPU, đừng đánh bóng nó — fuse nó vào hàng xóm** (softmax+topk, router+gather). "Vì sao top-k của
MoE router rẻ dù selection dở?" → k nhỏ (2–8) + fuse vào softmax + dispatch. Nối tới A6 EP-MoE all-to-all.
Trait = **subtract-before-add / research-taste** (nhận ra op nào không đáng tối ưu, chuyển sang fusion).

---

## 3.5 · GEMM — kernel DUY NHẤT băng qua ridge

**Câu hỏi.** GEMM là kernel duy nhất AI vượt ridge (compute-bound). Vậy craft ở đây KHÔNG phải "chạm tường
HBM" mà là "leo tới tensor-core throughput" — làm sao?

**Sự thật nền tảng.** GEMM vuông N³:
```
AI = 2·M·N·K / ((MN + NK + MK)·elem) = N/3 ≈ 1365 FLOP/byte tại N=4096
```
`1365 ≫ ridge 131` ⇒ **compute-bound**. Nghĩa là mỗi byte A/B được *tái sử dụng* ~N lần, nên trần KHÔNG
phải HBM mà là **tensor-core throughput**. Craft = data reuse (SMEM tile) + đưa vào tensor core (`tl.dot`)
+ tune block/warp/stage. (Spec A2 quote ~512 dưới model traffic nặng hơn đếm re-read; cả hai đều ≫ ridge.)

**Dẫn xuất — vì sao naive rơi 0.2%, không phải 50%.** Naive: 1 program/output-element, K-loop scalar qua
global memory. Mỗi A-row đọc lại **N lần** (mỗi output cột), mỗi B-col đọc lại **M lần** → **0 reuse**. Nó
không chạm tensor core (`tl.sum(a*b)` scalar, không `tl.dot`), bị L1/issue-bound → ~0.2% cuBLAS. Con số
thấp thảm không phải vì HBM mà vì **không reuse + không MMA**.

**Dẫn xuất — leo lên.** Tiled: output tile `BLOCK_M × BLOCK_N` tích lũy qua K-block bằng `tl.dot`. Compiler
stage A/B tile qua SMEM + phát lên tensor core; mỗi tile được reuse `BLOCK_K` lần. **GROUP_M swizzle**:
duyệt pid theo block `GROUP_M × N` để các B-column một group chạm nhau ở lại **L2-resident** (giảm HBM read
cho phần bên phải của AI-model). Autotune trên `BLOCK_M/N/K + num_warps + num_stages` = register block-
tiling + software-pipelining (async-copy).

**Neo code** (`src/scratch_llm/kernels/gemm_triton.py`):
```python
@triton.jit
def _gemm_naive_kernel(...):                                 # :44 (@jit :43)
    m = pid // N; n = pid % N                                # :60-61  1 program / output element
    for k0 in range(0, K, BLOCK_K):                          # :64  K-loop scalar qua global
        acc += tl.sum(a * b, axis=0)                         # :69  KHÔNG tl.dot, 0 reuse ⇒ 0.2%

@triton.jit
def _gemm_tiled_kernel(...):                                 # :102 (@jit :101, thân chung stage 2+3)
    # GROUP_M swizzle: giữ B-columns L2-resident                :123-129
    a_ptrs = a_ptr + offs_m[:,None]*stride_am + offs_k[None,:]*stride_ak   # :134  block 2D
    b_ptrs = b_ptr + offs_k[:,None]*stride_bk + offs_n[None,:]*stride_bn   # :135
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)     # :137
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):                 # :138
        a = tl.load(a_ptrs, mask=...); b = tl.load(b_ptrs, mask=...)
        acc += tl.dot(a, b)                                  # :142  ← bf16×bf16 → fp32 tensor-core MMA
        a_ptrs += BLOCK_K*stride_ak; b_ptrs += BLOCK_K*stride_bk
    tl.store(c_ptrs, acc.to(...), mask=c_mask)               # :150  mask biên

_gemm_autotuned_kernel = triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M","N","K"])(_gemm_tiled_kernel)  # :173
gemm = gemm_autotuned                                        # :237  shipping entry point
```
`gemm_tiled` (`:216`) = 1 config cố định (128×128×64, 8 warp, 3 stage); `_AUTOTUNE_CONFIGS` (`:153`) = 10
config. Byte model bench (`bench/gemm.py:54`): `3·N²·2` (đọc A,B + ghi C, minimal-traffic).

**Honesty (docstring `:19-25`).** float4 global load, register block-tiling, XOR-swizzle chống bank
conflict — **compiler làm hết**: `tl.load` block 2D được vectorize/coalesce, `tl.dot` chọn micro-tiling +
SMEM layout (swizzle), `num_stages` lái async-copy pipeline. Stage 3 là "autotune knob của compiler",
KHÔNG phải "viết swizzle tay". **%roof > 100% được disclose thẳng:** cuBLAS-proxy ở đây là `torch.matmul`
default, và autotuned `tl.dot` *đánh bại heuristic mặc định* ở shape 4096³ — không phải claim vượt super-peak.

**Hình ảnh — reuse là toàn bộ câu chuyện:**
```
NAIVE (0.2%):  C[m,n] = Σ_k A[m,k]·B[k,n]     1 program lo 1 output
   A[m,:] đọc lại cho MỌI n (N lần)  ·  B[:,n] đọc lại cho MỌI m (M lần)  ⇒ 0 reuse, scalar, no MMA

TILED (128%):  output tile 128×128 tích lũy qua K-block
   ┌──BLOCK_K──┐         A tile (128×64) ─┐
   │  A tile   │  ×  B tile (64×128)  ──► tl.dot ──► acc(128×128) fp32  [tensor core MMA]
   └───────────┘         mỗi A/B tile reuse 128 lần trong tile ⇒ compute-bound
   GROUP_M swizzle: nhóm 8 row-block chạm chung B-cols ⇒ L2-resident ⇒ ít HBM read

  AI = N/3 = 1365 ≫ ridge 131 ⇒ trần là tensor core, KHÔNG phải HBM.
```
Data journey: `A: (4096,4096) bf16 · B: (4096,4096) bf16 → C: (4096,4096) bf16`, acc fp32 in-register.

**Số đo THẬT.**
- ladder: **naive 0.2% → tiled 128.5% → autotuned 134.3% của cuBLAS-proxy** (autotuned **101.9 TF/s**)
  `[measured · sm120 · RESULTS.md]`. Tái hiện đúng SHAPE siboehm (0.2% → cao) trên card này.
- AI 4096³: `2·4096³ / (3·4096²·2) = 1365.3 FLOP/byte` `[CPU-measured · doc này]`; ridge run này đo 134
  (noise của per-run peak; R0 đo 131 — cùng ~131 ± noise).
- **[PREDICTED] trần Hopper (design note §4, ISA-gated, chỉ compile-verified trên sm120):** CUDA-core 32
  → WGMMA 318 → warp-spec 531 → CUTLASS 630 TF/s. WGMMA là **sm_90a-only**; runtime TF/s là nợ ngày
  H100/B200 (`RESULTS.md` §ISA-gated: PTX artifact `4× wgmma.mma_async.m64n64k16` compile-verified).

> **Neo → gap.** Đổi shape sang M=1 (thành GEMV): `AI` sụp về ≈1, băng qua ridge NGƯỢC lại → memory-bound
> → `tl.dot` với `BLOCK_M=1` vô nghĩa (tensor core cần tile), ta nên rơi về **kernel Bài 3.1**. Đây là vì
> sao decode (M=1) dùng GEMV kernel, prefill (M lớn) dùng GEMM kernel.

**Frontier / cổng.** GEMM là **90% FLOP** của training/prefill. Trần thật: sm120 dừng ~102 TF/s (WGMMA
sm_90a-only); leo tiếp là ngày H100 (WGMMA/TMA warp-spec → CUTLASS). Bài học série (design note §4): *trên
memory-bound kernel, craft là chạm tường — tường LÀ trần; trên GEMM, craft là tiling + tensor core, và 10%
cuối tới cuBLAS đắt hơn giá trị của nó trừ khi bạn LÀ người viết cuBLAS.* Gate = "GEMM 4096³ bị chặn bởi
gì, và WGMMA/TMA thêm gì so với tl.dot?". Trait = **roofline-first + claims honesty** (%roof>100 disclose).
Scarce bucket: kernels (top differentiator 2026).

---

## Bảng số đo THẬT (chạy lại được — DoD-là-profile)

| Bài | Đo | Kết quả | Nguồn |
|---|---|---|---|
| 3.0 | peaks card này | HBM **0.551 TB/s** · bf16 **72.1 TF/s** · ridge **131** | measured · RESULTS.md |
| 3.0 | anchor copy / gemm | **551 GB/s = 100.1% mem** · **72.1 TF/s = 100.0% cmp** (spread <1.2%) | measured · RESULTS.md |
| 3.0 | ridge arithmetic | 72.1/0.551 = **130.9 FLOP/byte** ≈ 131 | CPU-measured · doc |
| 3.1 | GEMV naive | **431 GB/s = 78% HBM** | measured · RESULTS.md |
| 3.1 | GEMV blockrow | **528.8 GB/s = 95.9% HBM** (107% torch.mv) | measured · RESULTS.md |
| 3.1 | GEMV AI @8192² | **1.000 FLOP/byte** | CPU-measured · doc |
| 3.2 | softmax fused | **551 GB/s ~100% HBM** | measured · RESULTS.md |
| 3.2 | online vs twopass | **1.33× ít byte · 1.17× nhanh (>L2)** | measured · RESULTS.md |
| 3.2 | online == torch.softmax | **max\|Δ\| = 0.0** (+1e4 outlier tile cuối) | CPU-measured · doc |
| 3.3 | RMS + LN @N≥4096 | **cả hai ~100–101% HBM** | measured · RESULTS.md |
| 3.3 | RMS vs LN | **±1%** (small-N = noise) | measured · RESULTS.md |
| 3.4 | TopK iter-max | **258 GB/s = 46.9% peak** (honest fail) | measured · RESULTS.md |
| 3.4 | torch.topk | **15.7% peak** (vendor cũng dở) | measured · A2_design_note |
| 3.4 | fused softmax+topk | **3.39× nhanh · 3.00× ít traffic** | measured · RESULTS.md |
| 3.4 | fusion ideal ratio | **3.00×** (201→67 MB) | CPU-measured · doc |
| 3.5 | GEMM ladder | **0.2% → 128.5% → 134.3% cuBLAS** (autotuned 101.9 TF/s) | measured · RESULTS.md |
| 3.5 | GEMM AI @4096³ | **1365.3 FLOP/byte** (N/3) | CPU-measured · doc |
| 3.5 | Hopper climb | 32 → 318 → 531 → 630 TF/s | **[PREDICTED]** · design note §4 |

---

## Checklist recall COLD (che phần trên, tự trả lời — đây là cách re-learn)

1. **3.0** Định nghĩa `ridge`. Vì sao `%roof` dùng `attainable(AI)` chứ không compute peak cố định — cho
   kernel AI=1 thì hai mẫu số chênh bao nhiêu lần? Vì sao đo peak trên card này, không lấy datasheet?
2. **3.0** Đo BW bằng copy 256 KB thay vì 256 MB → con số cao hơn hay thấp hơn, và làm %roof của GEMV
   sai kiểu gì? (Gợi ý: L2 48 MB.)
3. **3.1** Dẫn `AI = 1` cho GEMV từ byte + FLOP. Vì sao 95.9% HBM *đã là xong* — cái gì vật lý cấm 130%?
4. **3.1** Shape đổi sang tall-skinny M=8, N=1e6: stage nào (blockrow vs split) thắng, vì sao naive/blockrow
   *đói CTA*?
5. **3.2** Đếm pass: twopass / online / fused = mấy N? Dẫn tỉ số 1.33×.
6. **3.2** Viết dòng recurrence `d = d·factor + p_sum`. `factor = ?`. Bỏ factor thì sai thế nào, và vì sao
   twopass KHÔNG cần rescale?
7. **3.2** Vì sao đo được chỉ 1.17× mà analytic là 1.33×? (L2 caveat.)
8. **3.3** Reduction thứ 2 của LN sống ở đâu (HBM hay register)? Vì sao "miễn phí" @N=8192 nhưng lộ @N=64?
9. **3.3** Nếu LN re-read x từ HBM cho σ² → traffic mấy N, %HBM tụt về đâu so với RMS?
10. **3.4** Vì sao 46.9% là con số ĐÚNG chứ không phải bug — cái gì về k-pass tuần tự cấm nó chạm 100% HBM?
11. **3.4** Vì sao fuse softmax+topk hợp lệ (index)? Nó tiết kiệm cái gì? Ratio ≈?
12. **3.5** Dẫn `AI = N/3` cho GEMM. Vì sao naive rơi 0.2% (không phải 50%)? tl.dot + SMEM tile sửa gì?
13. **3.5** GEMM đổi sang M=1 → còn compute-bound không, rơi về kernel nào của série? Vì sao?

> Trả lời cold được cả 13 = **S3 thật sự OWNED** (interview-grade). Vấp câu nào → mở đúng mục đó, hoặc
> blank-slate hàm tương ứng (`raise NotImplementedError` → test đỏ → tự dẫn → xanh + đo lại) để re-own tay.

---

*Cross-ref: `docs/learning/roadmap/S3_cuda_core_kernels.md` (spine reference) · `docs/learning/PROGRESS.md`
(ledger 89 Bài) · `docs/learning/roadmap/README.md` (perf roadmap · 41 Bài) · `bench/RESULTS.md` §A2
(số đo nguồn) · `performance/notes/A2_design_note.md` (verdict floor/ceiling) · sibling:
`derivations/M2_transformer_forward.md` (forward pass — nơi các op này SỐNG). Concept kế: A3 tensor-core
ladder (CUDA C++ WMMA sm120 · A4 flash-attention = softmax online × V fuse vào tile-loop).*
