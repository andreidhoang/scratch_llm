# S6 — Distributed Primitives + ISA-gated Kernels · First-Principles Derivations

> **Đây là gì.** Bản dẫn-xuất **từ nguyên lý gốc** (first principles) cho TOÀN BỘ pillar S6 — 7
> micro-concept (Bài 6.1→6.7): bốn primitive song song (TP · PP · EP · MFU) + ba kernel tensor-core
> theo thế hệ ISA (WGMMA → FA3 → tcgen05). Mỗi mục: (1) **Câu hỏi** falsifiable, (2) **Sự thật nền
> tảng** (áp lực vật lý/toán ép ra thiết kế), (3) **Dẫn xuất** có math, (4) **Neo code** `file·func·line`,
> (5) **Hình ảnh** (ASCII + shape/stride/dtype + ví dụ số hand-traced), (6) **Số đo THẬT**, (7)
> **frontier framing** + cổng phỏng vấn.
>
> **Cách dùng để re-learn.** Đọc phần *Câu hỏi* của mỗi mục → **tự trả lời COLD** (che phần dưới) → mở
> ra đối chiếu. Cuối doc có **checklist recall COLD** + bảng số đo. Đây là *derivation lab* có số đo —
> bạn đồng hành của `roadmap/S6_distributed_and_isa.md` (reference chung, pin commit `9e61d7a`).
>
> **Cảnh báo neo.** Line number **trôi** theo commit. Cite ở đây pin theo HEAD ngày 2026-07-14
> (`utils/{tp_mlp,pipeline_schedule,ep_moe,mfu}.py` + `performance/rental/kernels/*.cu`). Nếu lệch,
> `grep` tên hàm — đừng tin số dòng cứng (luật repo: verify, don't trust).
>
> **Nguồn số đo — luật honesty (FOP-4).** Bài 6.1–6.4 (distributed) là **correctness measured** trên
> gloo/CPU + số học thuần: mọi số dưới đây chạy lại được (script `scratchpad/verify_s6.py`), khớp
> `bench/RESULTS.md §"Perf track (A6…)"`. Bài 6.5–6.7 (ISA kernel) là **compile-verified ONLY** trên box
> sm120 — TF/s là *pre-registered target* của paper/book, **runtime DEFERRED** tới H100/B200; đánh dấu
> `[PREDICTED]` khắp nơi. "Compiles + PTX chứa đúng ISA + descriptor bit-exact" là số THẬT; TF/s thì
> không. **Implemented ≠ measured.**

---

## Bức tranh lớn — S6 trả lời một câu duy nhất: *collective nào chạy trên link nào?*

Bốn primitive + cái cân đo (MFU) + ba kernel ISA đều quy về **một** áp lực kỹ thuật: **map the
chattiest collective onto the fattest link** (đưa collective ồn nhất lên đường băng thông béo nhất).

```
                 MỘT MODEL QUÁ TO CHO MỘT GPU — chẻ nó theo trục nào?
                              │
   ┌──────────────┬──────────┴──────────┬──────────────────┐
   ▼              ▼                      ▼                  ▼
 DP/DDP         TP (6.1)              PP (6.2)           EP (6.3)
 chẻ BATCH      chẻ MỘT layer        chẻ theo ĐỘ SÂU    chẻ EXPERT
 (đã có,S5)     (column→row GEMM)    (p stage, m µbatch) (256 expert/W rank)
   │              │                     │                  │
 all-reduce     2 all-reduce/layer    p2p activation     all-to-all ×2
 grad (sparse)  (g fwd, f bwd)/block   (SPARSE, kề nhau)  (DENSE, mọi→mọi)
   │              │  CHATTIEST          │  THƯA             │  CHATTIEST NHẤT
   └──inter-node──┘  → intra-node       └──inter-node──     └──→ intra-node
      IB ~50GB/s        NVLink ~900GB/s     IB (rẻ băng thông)   NVLink
                    └──────────── ~18× cliff NVLink→IB ────────────┘
                              │
                     MFU (6.4) = cái SCOREBOARD: C=6ND
                     "bao nhiêu % hardware bạn TRẢ TIỀN mà làm việc có ích"
                     sáu killer gọi tên cả TP/PP/EP ở trên
                              │
   ┌──────────────────────────┴──────────────────────────┐
   ▼  Bên TRONG một GPU: tensor-core nào phát MMA thế nào? ▼
 WGMMA (6.5)          FA3 (6.6)              tcgen05 (6.7)
 warpgroup(128)       warp-specialized       single-thread(1) issue
 async MMA            producer/consumer      accumulator vào TMEM
 sm_90a Hopper        TMA + ping-pong        sm_100a Blackwell
 └──────── collective-widening: 32(warp) → 128(warpgroup) → 1(thread) ────────┘
```

**Áp lực → lời giải** (mỗi thiết kế S6 giải một áp lực cụ thể):
- **TP** ← *một matmul không vừa một GPU*: chẻ column-then-row để activation elementwise rơi vào chỗ
  không cần đồng bộ, dồn comm về đúng một điểm.
- **PP** ← *độ sâu không vừa*: chẻ theo layer, nhưng phải fill/drain → **bubble** `(p-1)/m`.
- **EP** ← *quá nhiều expert*: shard expert, bay token tới chủ bằng **all-to-all** (collective nặng nhất).
- **MFU** ← *đồng hồ nói "6.2 s/step", nhưng tốt hay xấu?*: quy về `6ND/peak`, và **sáu killer** giải
  thích vì sao run thật rơi 40–55% chứ không 100%.
- **WGMMA/FA3/tcgen05** ← *silicon mỗi thế hệ nới rộng "ai phát MMA"*: từ warp → warpgroup → thread, và
  accumulator dời từ register → TMEM. Async (issue tách completion) là thứ mở khoá overlap.

Xuyên suốt: **primitive thì exact & test được ngay hôm nay** (gloo/CPU/số học); chỉ **băng thông** (busbw
line-rate, cliff ~18×) và **TF/s kernel** là bị khoá sau máy thuê. Đó là đối xứng của cả pillar.

---

## 6.1 · Megatron TP MLP — column→row GEMM, đúng 1 all-reduce/forward

**Câu hỏi.** Một matmul quá to cho một GPU — chẻ nó qua `W` rank thế nào để hai nửa (GEMM-1, GEMM-2)
**ghép lại không cần comm ở giữa**, và forward chỉ tốn **đúng một** collective?

**Sự thật nền tảng.** DDP chẻ *batch* — mỗi rank vẫn chạy cả layer. TP chẻ *một layer*. Chìa khoá: **GeLU
là elementwise**. Nếu ta chẻ sao cho hoạt hình GeLU tính được trên shard local mà không cần thấy shard
của rank khác, thì ta *hoãn* được mọi giao tiếp về đúng một điểm — sau GEMM-2.

**Dẫn xuất — hai trục chẻ liên hợp (column-then-row).** FFN: `Y = GeLU(X @ A) @ B`, với `A: d_model×d_ff`,
`B: d_ff×d_model`.

*Bước 1 — chẻ `A` theo CỘT (output).* Rank `r` giữ `A_r = A[:, r·d_ff/W : (r+1)·d_ff/W]`. Từ **toàn bộ**
`X` (replicated), rank tính `X @ A_r` = một dải `d_ff/W` cột của pre-activation. Vì GeLU elementwise:
```
GeLU(X @ A_r) = ( GeLU(X @ A) )[:, r·d_ff/W : (r+1)·d_ff/W]     ← đúng dải của GeLU đầy đủ, 0 comm
```

*Bước 2 — chẻ `B` theo HÀNG (input).* Rank `r` giữ `B_r = B[r·d_ff/W : (r+1)·d_ff/W, :]`. Nhân dải GeLU
local với `B_r` → một output **partial** full-width `d_model`. Tổng các partial = output thật:
```
Y = GeLU(X @ A) @ B = Σ_{r=0}^{W-1} GeLU(X @ A_r) @ B_r     ← ĐÚNG một all-reduce SUM gộp lại
```
Đây là toàn bộ phép màu: chọn *hai trục liên hợp* (column rồi row) để GeLU (điểm cần đồng bộ nếu chẻ
sai) rơi vào chỗ *đã* có đủ dữ liệu local ⇒ comm dồn về **một** all-reduce ở cuối.

**Cặp op liên hợp `f`/`g` — làm autograd đối xứng fwd↔bwd.** Đây là phần tinh tế nhất:
- `f` (`_CopyToModelParallel`) — **cửa vào**: `fwd = identity` (X đã replicated, không cần làm gì);
  `bwd = all-reduce`. Vì sao bwd cần cộng? Mỗi rank sinh `∂L/∂X` *riêng* qua `A_r` của nó; `∂L/∂X` thật =
  tổng các đóng góp → phải all-reduce.
- `g` (`_ReduceFromModelParallel`) — **cửa ra**: `fwd = all-reduce` (cộng partial); `bwd = identity`
  (gradient của output đầy đủ broadcast nguyên vẹn về mọi nhánh `B_r`, không đổi).

Nên: **forward = 1 collective** (`g`), **backward = 1 collective** (`f`). `f`/`g` là *conjugate*:
`(identity, all-reduce)` và `(all-reduce, identity)`.

**Đếm all-reduce cho ĐÚNG (bẫy phỏng vấn).** MLP một mình: 1 fwd (`g`) + 1 bwd (`f`) = **2 all-reduce/MLP
mỗi step fwd+bwd**. Attention block (QKV column-parallel → output row-parallel) cho **2 cái nữa**. Nên một
transformer layer đầy đủ = **4 all-reduce/step** (2 fwd: attn-g + mlp-g; 2 bwd: attn-f + mlp-f). Câu "2
all-reduce/layer" trong tài liệu Megatron thường chỉ **2 trong forward pass** (attn-g + mlp-g); nói cho
rõ luôn: *fwd 2, bwd 2, tổng 4*.

**Bias không shard.** GEMM-2 bias cộng **một lần SAU** all-reduce. Nếu cộng *trong* parallel region (trước
`g`) → mỗi rank cộng bias → all-reduce SUM đếm bias **W lần** thay vì 1 → output thừa `(W−1)×` bias (output
đúng đã có 1× bias rồi, nên phần *dư* là `W−1`, không phải `W`).

**Neo code** (`src/scratch_llm/utils/tp_mlp.py`):
```python
class _CopyToModelParallel(Function):              # :50   f
    def forward(ctx, x, group):  return x          # :54   fwd = identity
    def backward(ctx, grad):                       # :59
        dist.all_reduce(grad, op=SUM, ...)         # :61   bwd = all-reduce
class _ReduceFromModelParallel(Function):          # :65   g
    def forward(ctx, x, group):
        dist.all_reduce(x, op=SUM, group=group)    # :71   fwd = all-reduce (partial→full)
        return x
    def backward(ctx, grad):  return grad, None    # :75   bwd = identity

class ColumnParallelLinear(nn.Module):             # :96
    def forward(self, x):
        x = copy_to_model_parallel(x, self.group)  # :126  f: id fwd / all-reduce bwd
        return F.linear(x, self.weight, self.bias) # :127  dải d_ff/W cột
class RowParallelLinear(nn.Module):                # :139
    def forward(self, x):
        partial = F.linear(x, self.weight)         # :170  partial full-width, CHƯA bias
        out = reduce_from_model_parallel(partial)  # :171  g: THE collective duy nhất
        if self.bias is not None: out = out + self.bias  # :172-173  bias SAU reduce, MỘT lần

class TensorParallelMLP(nn.Module):                # :190
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))       # :215  col-par → GeLU (0 comm) → row-par
```

**Hình ảnh — data journey** (anchor `d_model=8, d_ff=16, W=4`, `X:(3,8)` fp32):
```
X (3,8) replicated mọi rank
  │  ColumnParallelLinear: weight_r = (d_ff/W, d_model) = (4,8)   ← rank giữ 4 cột output
  ▼  F.linear(X, weight_r) → (3,4)   ‖ concat 4 rank = (3,16) đầy đủ
GeLU (3,4) elementwise  ── 0 comm ──  = đúng slice [4r:4r+4] của GeLU(X@A) full
  │  RowParallelLinear: weight_r = (d_model, d_ff/W) = (8,4)      ← rank giữ 4 hàng input
  ▼  F.linear(gelu, weight_r) → (3,8) PARTIAL full-width
      rank0 partial + rank1 + rank2 + rank3
              └────── all-reduce SUM ──────┘  → (3,8) output THẬT (giống hệt mọi rank)
  + bias (8,)  ← cộng MỘT lần sau reduce
```
Ví dụ số (hand-traced, script `verify_s6.py`): manual-shard 4 rank rồi `stack(partials).sum(0)` cho
`max|tp − ref| = 1.78e-15` (= sai số làm tròn fp64 ⇒ **numerically identical**). Cộng bias trong region →
`(wrong − tp)/bias = 3.0` = `W−1` (bias bị đếm `W=4` lần, output đúng có 1 lần ⇒ dư `W−1`).

**Số đo THẬT.**
```
[measured · fp64 · verify_s6.py]  max|tp − ref| = 1.78e-15  ⇒ TP == single-GPU (rtol 1e-9)
[measured · fp64 · verify_s6.py]  bias-in-region overshoot = 3.0 = W−1  (bias đếm W lần, dư W−1; vì sao bias phải cộng SAU reduce)
[measured · gloo/CPU · RESULTS.md §A6]  exactly 1 all-reduce/fwd, 2 fwd+bwd (world_size 2 và 4);
      mutation test bias-double-count + missing-all-reduce đều FAIL (oracle có răng)
```

**Frontier / cổng.** TP chỉ **intra-node** (≤8, NVLink domain) — *chính vì* all-reduce của nó chattiest,
phải cưỡi fattest link. Megatron-LM / DeepSpeed dùng đúng cấu trúc này; DeepSeek-V3 dùng TP=1 (thay bằng
EP+DP) để né NVLink pressure ở scale MoE. **Gate phỏng vấn:** "Megatron một transformer layer tốn mấy
all-reduce, ở đâu?" → 4 (2 fwd g, 2 bwd f, mỗi block attn+MLP). **Trait FOP:** roofline-first (biết
collective nào ở đâu). **Scarce-2026:** distributed = differentiator (inference+training).

---

## 6.2 · Pipeline GPipe vs 1F1B — bubble `(p−1)/m`, peak-act `min(p,m)`

**Câu hỏi.** Chẻ model theo *chiều sâu* qua `p` stage, bơm `m` microbatch — pipeline phải **fill** rồi
**drain**, idle đó (bubble) bằng bao nhiêu, và 1F1B đổi được **cái gì** so với GPipe?

**Sự thật nền tảng.** `p` công nhân xếp hàng, mỗi người một công đoạn. Sản phẩm đầu phải đi hết `p` công
đoạn thì công nhân cuối mới có việc → đầu ca ai đó rảnh (fill), cuối ca cũng vậy (drain). Bubble = tổng
idle đó. **Áp lực thứ hai — memory:** activation phải giữ đến lúc backward dùng; giữ bao nhiêu *cùng lúc*
= peak activation memory. Đó là hai trục 1F1B đánh đổi.

**Dẫn xuất bubble `(p−1)/m`.** Với `t_f = t_b = t`: mỗi device làm `m` forward + `m` backward, busy thật =
`m·(t_f+t_b) = 2mt`. Makespan = fill + steady + drain: forward wave clear hết `p` stage mất `(m+p−1)`
op-time, backward wave thêm `(m+p−1)`:
```
makespan = 2·(m + p − 1)·t
bubble   = makespan − busy = 2(m+p−1)t − 2mt = 2(p−1)t
bubble_fraction = bubble / busy = 2(p−1)t / 2mt = (p − 1) / m      ← t và số 2 triệt tiêu
```
Deeper pipeline (`p↑`) → bubble dày; nhiều microbatch (`m↑`) → mỏng. **GPipe và 1F1B có bubble Y HỆT** —
1F1B *không* mua gì về throughput.

**Dẫn xuất 1F1B mua GÌ — peak activation `min(p,m)` thay vì `m`.**
- **GPipe:** làm **hết `m` forward rồi mới `m` backward**. Stage 0 phải stash activation của cả `m`
  microbatch trước khi backward đầu tiên chạy → **peak = m**.
- **1F1B:** stage `s` warmup `w = min(p−1−s, m)` forward, rồi **một-forward-một-backward** ở steady state
  → giải phóng activation nhanh bằng lúc tạo → live activation chặn ở **độ sâu pipeline**: **peak =
  min(p, m)**, *bất biến theo `m`*. Đó là toàn bộ điểm: chọn `m ≫ p` cho bubble mỏng, rồi 1F1B để *dám*
  giữ nổi activation.

**Interleaved v-chunk.** Chia mỗi device thành `v` chunk xen kẽ → mỗi pipeline op ngắn `1/v` → bubble
`(p−1)/(v·m)`, đổi lấy `v×` comm stage-to-stage + peak-act nở về `~v·p`.

**Neo code** (`src/scratch_llm/utils/pipeline_schedule.py`):
```python
def one_f_one_b_schedule(p, m):                    # :128
    for s in range(p):
        w = max(0, min(p - 1 - s, m))              # :140  warmup: stage sâu warmup ÍT
        for _ in range(w):        ops.append(Op(s, f, FORWARD)); f += 1        # :143  fill
        for _ in range(m - w):    ...F... ...B...  # :146  steady 1F1B
        while b < m:              ops.append(Op(s, b, BACKWARD)); b += 1       # :151  drain

def _predecessors(sched):                          # :163  3 họ cạnh DAG
    # device-order :182 · forward-flow F(i,s)←F(i,s-1) :186-193 ·
    # backward-flow B(i,s) cần F(i,s) :195-198 VÀ B(i,s+1) :199-201
def simulate(sched, t_f=1.0, t_b=1.0):             # :205  Kahn longest-path
    end[op] += dur[op.kind]                         # :238  start(=max pred end) + duration
    live += 1 if F else -1; peak = max(peak, live)  # :255  peak-act walk
    bubble_fraction = bubble_time / busy            # :261  đo được, pin vào (p-1)/m

def gpipe_bubble_fraction(p, m):        return (p - 1) / m          # :299
def one_f_one_b_bubble_fraction(p, m):  return (p - 1) / m          # :305  Y HỆT
def gpipe_peak_activations(p, m):       return m                   # :328
def one_f_one_b_peak_activations(p, m): return min(p, m)           # :338  điểm của 1F1B
```

**Hình ảnh — 1F1B steady state** (`p=4, m=8`):
```
      fill →     ← steady 1F1B →              ← drain
dev0:  F0 F1 F2  F3 B0 F4 B1 F5 B2 F6 B3 F7 B4  B5 B6 B7   ← warmup w=3, giữ ≤4 activation
dev1:  .  F0 F1  F2 B0 F3 B1 ...                 ← warmup w=2
dev2:  .  .  F0  F1 B0 F2 B1 ...                 ← warmup w=1
dev3:  .  .  .  F0  B0 F1 B1 ...                 ← warmup w=0, backprop NGAY
       └── bubble (idle) = 2(p-1)t = 6t; busy = 2mt = 16t ⇒ 6/16 = 0.375 ──┘
peak activation: dev0 giữ tối đa min(p,m)=4  (GPipe: dev0 giữ m=8)
```
Op dataclass: `Op(stage:int, microbatch:int, kind:"F"/"B")` (:52) — schedule là `tuple[tuple[Op,...],...]`
per-device (:80). `simulate` trả `bubble_fraction` (float) đo bằng longest-path, khớp analytic.

**Số đo THẬT** (script `verify_s6.py`, module thật):
```
[measured · CPU · verify_s6.py + RESULTS.md §A6]  bubble sim == analytic:
   p=4  m=8  → 0.3750    peak: GPipe 8  vs 1F1B 4  (min=4)
   p=8  m=16 → 0.4375    peak: GPipe 16 vs 1F1B 8  (min=8)
   p=4  m=16 → 0.1875    peak: GPipe 16 vs 1F1B 4  (min=4, BẤT BIẾN theo m!)
[measured · RESULTS.md §A6]  77 tests; Kahn longest-path DAG reproduces all 9 (p,m);
   mutations (warmup off-by-one, dropped B-edge) đều bị bắt
```
Chú ý `p=4, m=16`: bubble mỏng đi (0.1875) nhờ `m↑`, nhưng **1F1B peak vẫn = 4** trong khi GPipe = 16 —
đó là lý do 1F1B là default (bạn *dám* chọn `m ≫ p`).

**Frontier / cổng.** PP/DP cưỡi **inter-node** link (rẻ băng thông) vì comm của chúng *thưa* (p2p
activation giữa stage kề nhau) — ngược TP/EP. DeepSeek DualPipe overlap bubble với comm; Megatron
interleaved-1F1B chia bubble `/v`. **Gate:** "pipeline `p` stage `m` microbatch, bubble bao nhiêu,
GPipe→1F1B đổi gì?" → `(p−1)/m` cả hai; 1F1B giữ throughput, hạ peak-act `O(m)→O(p)`. **Trait:**
predict-the-number (bubble = number, không phải cảm giác). **Scarce-2026:** distributed = differentiator.

---

## 6.3 · Expert-parallel MoE — all-to-all + zero token-divergence

**Câu hỏi.** 256 expert không nhét nổi một GPU — shard expert qua rank, giữ token tại chỗ, rồi *đưa token
tới rank sở hữu expert của nó* thế nào để output **bằng bit** với một reference dense một-process?

**Sự thật nền tảng.** Dense DDP replica chạy *mọi* expert trên *mọi* device — không scale (256 DeepSeek-V3
expert không vừa). EP shard expert: rank `r` giữ `E/W` expert, giữ token tại chỗ. Correctness dựa trên
**một identity duy nhất:** *expert nào xử lý token nào, với gate weight nào, ĐỘC LẬP với expert đó nằm ở
đâu vật lý.* All-to-all chỉ **dời byte**, không đổi số học.

**Dẫn xuất — hai chuyến bay all-to-all.** Router (replicated bit-identical mọi rank) chọn top-K expert/
token. Token phải "bay" tới rank chủ expert (dispatch), được xử lý, rồi "bay về" (combine) = **hai
all-to-all**. Nhưng split-size **data-dependent** (không biết trước bao nhiêu token tới mỗi rank) → phải
trả *thêm một* all-to-all tí hon trên **counts** trước:
```
(1) metadata all-to-all trên in_splits (W long)  → out_splits: mỗi rank nhận bao nhiêu
(2) DISPATCH all-to-all trên feature với split đó → recv_feat (recv_total, d)
(3) local expert-GEMM: mỗi expert owned chạy trên token nó nhận
(4) COMBINE all-to-all: split ĐẢO NGƯỢC (inverse permutation) → back về slot gốc
```
Hai điều kiện để identity cắn: **(a) routing determinism** — router bit-replicated ⇒ mọi rank cùng top-K
(một nhiễu gate-logit `1e-6` có thể lật top-K tie → sai expert → invariant đòi **zero token-divergence**,
map token→expert khớp *token-for-token*, không chỉ output allclose); **(b) weight identity** —
`build_experts` seed-deterministic ⇒ expert `e` cùng weight mọi rank.

**Đánh đổi.** All-to-all là collective **nặng nhất** (mọi rank gửi mọi rank). **Load lệch** (hot expert)
là bottleneck throughput — rank bận nhất định step-time (busiest rank sets the barrier).

**Neo code** (`src/scratch_llm/utils/ep_moe.py`):
```python
def route(logits, top_k):                          # :95
    affinity = torch.sigmoid(logits)               # :104  affinity per expert (độc lập)
    topk_idx = affinity.topk(top_k, -1).indices    # :105
    gate = gate_sel / gate_sel.sum(-1, keepdim=True)  # :107  normalize sum-to-1

class ExpertParallelMoE(nn.Module):                # :148
    def forward(self, x_local):                    # :169
        topk_idx, gate = route(self.router(x_local), k)      # :181-182  replicated
        flat_expert = topk_idx.reshape(-1)                    # :186  (S,) expert id/slot
        dest_rank   = flat_expert // per                      # :192  owner rank
        local_expert= flat_expert % per                       # :193  idx trong shard
        perm = torch.argsort(dest_rank, stable=True)          # :196  block liền per-rank
        in_splits = torch.bincount(dest_rank, minlength=w)    # :199
        dist.all_to_all_single(out_splits, in_splits)         # :203  METADATA a2a (counts)
        dist.all_to_all_single(recv_feat, send_feat, out_list, in_list)   # :209  DISPATCH
        recv_out[mask] = self.experts[le](recv_feat[mask])    # :225  local expert-GEMM
        dist.all_to_all_single(back, recv_out, in_list, out_list)  # :229  COMBINE (split SWAPPED)
        slot_out[perm] = back                                 # :235  un-permute (inverse argsort)
        y = (slot_out.view(n,k,d) * gate.unsqueeze(-1)).sum(1)  # :236  gate-weight + sum-K
```
Oracle: `dense_moe_reference` (:111) trả `(y, topk_idx)` — cái thứ hai là **zero-divergence oracle**
(:133).

**Hình ảnh — dispatch** (`N=6, K=2, E=8, W=4, per=E/W=2`):
```
route(x): topk_idx (6,2) → flat_expert (12,) = [7,5,6,4,2,7,2,1,3,4,6,5]
                            dest = flat // per(=2) = [3,2,3,2,1,3,1,0,1,2,3,2]
argsort(dest,stable): gom block liền  rank0│rank1│rank2│rank3
bincount(dest) = in_splits = [1, 3, 4, 4]      ← rank0 gửi 1 slot, rank1 gửi 3, ...
   ├─ metadata a2a → out_splits (mỗi rank học nhận bao nhiêu)
   ├─ DISPATCH a2a (feat (S,d), local_expert (S,)) → recv_feat (recv_total, d)
   ├─ expert[le](recv_feat[mask])  ← chỉ token của expert owned
   └─ COMBINE a2a (in/out SWAPPED) → back → slot_out[perm]=back → ×gate, Σ over K → y (6,8)
```
Ví dụ số (hand-traced): `gate` row-sum = `[1.0, 1.0, 1.0]` (chuẩn hoá đúng); `dest_rank` counts
`[1,3,4,4]` ⇒ **traffic cross-rank THẬT** (rank0 chỉ nhận 1 slot off-rank — không degenerate-local).

**Số đo THẬT** (script `verify_s6.py`, `route`/`dense_moe_reference` thật — không cần dist):
```
[measured · CPU · verify_s6.py]  gate row-sum = 1.0 mỗi token (normalize đúng)
[measured · CPU · verify_s6.py]  dest_rank counts = [1, 3, 4, 4]  ⇒ real cross-rank traffic
[measured · gloo/CPU · RESULTS.md §A6]  EP output == single-process dense-gather (allclose);
      ZERO token-divergence; mutation (remove inverse-permute :235) bị bắt (kết quả về sai slot)
```

**Frontier / cổng.** EP như TP: **intra-node** (all-to-all chattiest → NVLink). DeepEP/DualPipe overlap
all-to-all sau compute cho MoE training multi-node (rental-gated). DeepSeek-V3: 256 expert, EP=64.
**Gate:** "MoE dispatch, làm sao biết split-size cho all-to-all?" → metadata exchange trên counts TRƯỚC.
**Trait:** claims-honesty (zero-divergence oracle, không chỉ allclose). **Scarce-2026:** MoE+distributed =
differentiator cao nhất.

---

## 6.4 · MFU / HFU 6ND + sáu killer

**Câu hỏi.** Một số đọc đồng hồ (`s/step`) → biến thành *phần trăm hardware bạn thực sự trả tiền mà làm
việc có ích* thế nào, và vì sao run thật rơi 40–55% chứ không 100%?

**Sự thật nền tảng.** Hai utilization cách nhau đúng **một FLOP-count**. **MFU** = FLOP *toán học đòi* /
peak — dùng so *model* (không thưởng công vô ích). **HFU** = FLOP *phần cứng chạy* / peak — cộng thêm
recompute. `HFU ≥ MFU` luôn; khoảng cách = **thuế rematerialization**.

**Dẫn xuất `C ≈ 6·N·D`.** Vì sao 6? Mỗi param, mỗi token:
```
forward  : 1 matmul = 1 multiply + 1 add    = 2 FLOP/param/token   (FLOPS_FWD_PER_PARAM)
backward : 2 matmul cùng shape:
             input-grad  (∂L/∂x = ∂L/∂y · Wᵀ)  = 2 FLOP
             weight-grad (∂L/∂W = xᵀ · ∂L/∂y)  = 2 FLOP           = 4 FLOP/param/token   (BWD)
                                               ───────────────
                                               6 FLOP/param/token  ⇒ C = 6·N·D
```
**HFU / recompute.** Gradient checkpointing chạy forward **hai lần** (throw activation, tính lại lúc bwd).
Full-recompute (`r=1`) cộng một forward nữa = `+2ND` → hardware = `(6+2)ND = 8ND`:
```
HFU/MFU = (6 + 2r) / 6      r=0 → 1.0 (không checkpoint)   r=1 → 8/6 = 4/3
```
`r` là **knob DUY NHẤT** tách hai utilization.

**Dẫn xuất sáu killer — vì sao ≠ 100%.** Realized MFU = ideal · ∏ η_i (mỗi killer η ∈ (0,1] độc lập, scale
achieved-FLOP/s):
```
MFU_KILLERS (thứ tự canonical): unoverlapped_comm · pipeline_bubble · memory_bound_kernels
                              · small_per_gpu_batch · moe_imbalance · stragglers
realized = ideal · ∏ η_i
```
Attribution **log-additive** (chia gap cho từng killer, *exact* không Shapley vì model là pure product):
```
E = ∏ η_i          (gap tổng, multiplicative)
−ln E = Σ (−ln η_i)  (thành additive trong log-space)
share(i) = (−ln η_i) / (−ln E)     ⇒ Σ share = 1 CHÍNH XÁC
```
Ba killer đầu **gọi tên** ba bài trước: `unoverlapped_comm` → 6.1 TP all-reduce trên critical path;
`pipeline_bubble` → 6.2 `(p−1)/m`; `moe_imbalance` → 6.3 hot expert.

**Neo code** (`src/scratch_llm/utils/mfu.py`):
```python
FLOPS_FWD_PER_PARAM = 2.0; FLOPS_BWD_PER_PARAM = 4.0; FLOPS_PER_PARAM = 6.0   # :42-44
def training_flops(N, D):  return 6.0 * N * D                 # :83   numerator MFU
def hardware_flops(N, D, recompute_fraction=1.0):
    return model + recompute_fraction * 2.0 * N * D           # :106  +forward → 8ND ở r=1
def mfu(N, D, step_time_s, n_dev, peak):
    return (training_flops(N,D)) / step_time_s / (n_dev * peak)   # :137-140
def hfu_over_mfu(r):  return (6.0 + 2.0*r) / 6.0              # :183  4/3 ở r=1
MFU_KILLERS = (unoverlapped_comm, pipeline_bubble, ...)      # :191  6 killer canonical
def compose_mfu(ideal, eff):  return ideal * ∏ η             # :210-217
def mfu_gap_attribution(eff): return {i: -ln η_i / Σ}        # :228-238  shares sum to 1
def unexplained_factor(...):  return measured / compose      # :253  1.0 iff 6 killer giải hết
```

**Hình ảnh — PaLM 540B scoreboard:**
```
achieved model FLOP/s = 6 · N · (tok/s) = 6 · 540e9 · 238,300 = 7.72e17 FLOP/s
peak cluster          = 6144 chips · 275 TFLOP/s              = 1.69e18 FLOP/s
MFU = 7.72e17 / 1.69e18 = 0.4570   (published 0.462 — Δ 0.005 = attention term 6N bỏ)
HFU = MFU · (6+2·0.75)/6 = 0.4570 · 1.25 = 0.5712   (published 0.578, r≈0.753 backed từ 57.8/46.2)
gap decomposition (ideal 0.6, η_comm 0.9, η_bubble 0.8):
   realized = 0.6 · 0.9 · 0.8 = 0.432
   share_comm   = ln(1/0.9)/(ln(1/0.9)+ln(1/0.8)) = 0.3207
   share_bubble = 0.6793                             Σ = 1.000 ✓
```

**Số đo THẬT** (script `verify_s6.py`, module thật):
```
[measured · CPU · verify_s6.py + RESULTS.md §A6]  PaLM 540B: MFU = 0.4570 (Δ0.005 vs 0.462);
      HFU@r=0.75 = 0.5712 (ledger 0.5717 @r≈0.753; published 0.578); hfu/mfu@r=1 = 4/3 = 1.3333
[measured · CPU · verify_s6.py]  compose_mfu(0.6, .9×.8) = 0.432;
      attribution {comm 0.3207, bubble 0.6793}  sum = 1.000000  (log-additive exact)
[measured · RESULTS.md §A6]  29 tests: N·D invariance, HFU≥MFU monotone in recompute
```

**Frontier / cổng.** Đây là scoreboard mọi scaling run bị chấm (Llama-3 report ~40% MFU, PaLM 46.2%). Số
lớn hơn 100% ⇒ peak/FLOP-count sai. **Gate kinh điển:** "6.2 s/step, 512 H100, 70B, 4M tok/step — tốt
không, nửa còn lại đâu?" → `mfu()` cho số, `mfu_gap_attribution` gọi tên nửa mất (sáu killer nhân lại đúng
bằng nó). **Trait:** predict-the-number + claims-honesty (`unexplained_factor` = "cái thứ bảy còn ăn
FLOP"). **Scarce-2026:** perf instrumentation = table-stakes cao.

---

## 6.5 · WGMMA GEMM — warpgroup-async MMA + 64-bit SMEM descriptor (compile-only sm_90a)

**Câu hỏi.** Làm sao **128 thread (warpgroup)** cùng phát *một* instruction MMA lên tile lớn hơn bất kỳ
thread/warp nào giữ nổi, **bất đồng bộ** (issue tách completion), và operand SMEM được mô tả bằng **một con
số 64-bit** thay vì con trỏ?

**Sự thật nền tảng.** `mma.sync` (Ampere) là *warp*(32)-collective, **blocking**. WGMMA (Hopper) nới thành
*warpgroup*(128)-collective, **async**: 128 thread cùng *phóng* MMA (`64·64·16 = 65,536 FMA`/instruction)
rồi đi tiếp; tensor core đọc A/B và ghi accumulator ở background theo timeline riêng — `D` chỉ hợp lệ *sau*
`wgmma.wait_group`. Tách issue↔completion *chính là* thứ cho phép producer TMA overlap consumer MMA (nền
của FA3, 6.6).

**Dẫn xuất — descriptor 64-bit KHÔNG phải con trỏ mà là cách *walk* tile ra khỏi SMEM.** Tensor core walk
B thẳng từ shared memory (B **không bao giờ** chiếm register). Descriptor pack:
```
[ 0,14) start_address = byte_addr >> 4     (16-byte granular)
[16,30) LBO (leading byte offset)  >> 4    (bước giữa core-matrix theo K)
[32,46) SBO (stride byte offset)   >> 4    (bước theo M/N)
[49,52) base_offset (swizzle phase)
[62,64) layout_type {0 none, 1 128B, 2 64B, 3 32B}
encode: matrix_descriptor_encode(x) = (x & 0x3FFFF) >> 4   (mask 18 bit, drop low 4 → 14-bit field)
```
`>>4` ép mọi tile **16-byte-aligned**. Vì sao vậy? SMEM bank + tensor-core fetch granularity là 16 byte;
địa chỉ không cần 4 bit thấp.

**Dẫn xuất — accumulator size ép trần N≤256.** Warpgroup m64n64: `32 f32 reg/thread = 64·64/128`. Nếu
N=256: `128 reg/thread = 64·256/128` — sát trần **255-register** kiến trúc ⇒ đó *là* lý do `N ≤ 256`.

**Async glue (4 op, verbatim artifact §1.10):** `wgmma.fence` (order register writes trước async read) →
`wgmma.mma_async` (scale-d=0 tile đầu = zero-init free, 1 sau = accumulate) → `commit_group` (gói thành
group) → `wait_group N` (N>0 = overlap knob; N=0 = drain hết).

**Neo code** (`performance/rental/kernels/wgmma_gemm_sm90a.cu`):
```c
uint64_t matrix_descriptor_encode(uint64_t x){ return (x & 0x3FFFF) >> 4; }   // :73-75
uint64_t make_smem_desc(const half* p, int base_offset){                       // :81
    desc |= encode(addr);                        //  :86  [0,14) start = addr>>4
    desc |= encode(16)   << 16;                  //  :87  LBO=16B → 1
    desc |= encode(1024) << 32;                  //  :88  SBO=1024B → 64
    desc |= (base_offset & 0x7) << 49;           //  :89  swizzle phase
    desc |= 1ull << 62;                          //  :90  128B swizzle
}
void wgmma_m64n64k16_ss(float d[32], uint64_t a_desc, uint64_t b_desc){         // :112
    asm("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0..%31}, "
        "%32, %33, 1, 1, 1, 0, 0;" ...);         //  :118  imms: scale-d,a,b=1, trans-a,b=0
}
extern "C" __global__ void wgmma_gemm_sm90a(...){                               // :146
    float d[32]; for(...) d[i]=0.0f;             //  :164-166  acc 32 reg zero-init MỘT lần
    for(int k0=0; k0<K; k0+=BK){                 //  :169  mainloop
        /* GMEM→SMEM swizzled store */ swz128(...);  // :179  128B swizzle XOR bit[7,10)→[4,7)
        wgmma_fence();                           //  :201  MỘT lần sau zero-init
        for(int kk=0; kk<BK/16; ++kk){           //  :203  4 K-strip của 16
            uint64_t a_desc = make_smem_desc(As + kk*16, phase);   // :212
            wgmma_m64n64k16_ss(d, a_desc, b_desc);                 // :214
        }
        wgmma_commit(); wgmma_wait<0>();         //  :218-219  drain trước khi overwrite SMEM
    }
    /* epilogue :223-252: 32 reg → C[64×64], warp w giữ rows [16w,16w+16) */
}
```

**Hình ảnh — descriptor bit-exact + collective width:**
```
warp(32) mma.sync        →  warpgroup(128) wgmma          [collective-widening]
 blocking                   ASYNC (issue ≠ completion)
                            128 thread → 1 instruction → 64×64×16 tile

descriptor worked example (addr=0x2000, 128B swizzle):
  start = 0x2000>>4 = 0x200        [0,14)
  LBO   = 16>>4     = 1     <<16 = 0x0000_0001_0000
  SBO   = 1024>>4   = 64    <<32 = 0x0040_0000_0000
  swz   = 1         <<62 =         0x4000_0000_0000_0000
  ─────────────────────────────────────────────
  desc  = 0x4000_0040_0001_0200   (khớp CUTLASS make_smem_desc / Colfax)
```

**Số đo THẬT.**
```
[measured · CPU arithmetic · verify_s6.py]  descriptor = 0x4000004000010200 bit-exact
      (khớp CUTLASS GmmaDescriptor: start>>4, LBO 16, SBO 1024, 128B swizzle)
[measured · CPU arithmetic]  acc regs/thread: m64n64 = 32,  m64n256 = 128 (sát trần 255 → N≤256)
[measured · compile · RESULTS.md §ISA-gated]  nvcc -arch=sm_90a -ptx exit 0;
      PTX có 4× wgmma.mma_async.m64n64k16.f32.f16.f16 + fence/commit/wait
[PREDICTED · book §7.3.1]  ~318 TFLOPS H100 4096³ FP16 (4.5× WMMA 71) — runtime DEFERRED, KHÔNG claim
```

**Frontier / cổng.** WGMMA là atom Hopper; `-arch=sm_90a` (chữ `a` **bắt buộc** — plain `sm_90` im lặng
thiếu instruction). CUTLASS 3.x/CuTe, flash-attn v3 build trên nó. **Gate:** "wgmma async nghĩa là gì cho
`D` register, ai reclaim completion?" → `D` không valid tới khi `wait_group` retire group; B ở SMEM (walk
qua descriptor), A được phép ở register. **Trait:** roofline-first (biết accumulator RF cap ép N). **Scarce
-2026:** kernels = differentiator cao nhất (RL/inference/kernels).

---

## 6.6 · FA3 attention — warp-specialized producer/consumer + TMA + ping-pong (compile-only sm_90a)

**Câu hỏi.** Dựng trên WGMMA async, làm sao *chuyên hoá warpgroup* thành **producer** (TMA load) vs
**consumer** (MMA + online-softmax), double-buffer K/V ping-pong, để load HBM luôn overlap compute?

**Sự thật nền tảng.** FA2 dùng cả block để load *và* tính, xen kẽ — tensor core rảnh khi chờ HBM. FA3
**chia vai**: producer chỉ phóng TMA (một thread elected, `setmaxnreg.dec` nhả register về 24), consumer
chỉ MMA + softmax (`setmaxnreg.inc` xin 240 register). K/V stage thành **ping-pong double-buffer**: producer
fill stage này *trong khi* consumer drain stage kia. Async WGMMA + TMA cho phép **hoàn toàn** giấu latency
HBM sau tensor-core math (FA2 không async được).

**Dẫn xuất — mbarrier handshake.** Producer/consumer bắt tay qua **mbarrier** (async barrier, PTX §9.7.13):
```
producer: mbar_expect_tx(bar, bytes)  — báo "phase này sẽ giao `bytes` qua TMA" rồi arrive
consumer: mbar_wait(bar, phase)       — spin trên phase-bit tới khi TMA completion lật nó
TMA: cp.async.bulk.tensor.2d  — load tile 2D HBM→SMEM một-thread-issue, completion signal trên mbarrier
```

**Dẫn xuất — online-softmax là ENGINE CHÍNH XÁC (không xấp xỉ).** Mỗi K-tile, cập nhật running state:
```
m_tile   = row-max của score tile
m_new    = max(m_old, m_tile)
alpha    = exp2((m_old − m_new)·log2e)        ← rebasing factor e^{m_old − m_new}
acc_o   *= alpha    ← RESCALE RUNNING OUTPUT (bug kinh điển: quên dòng NÀY, chỉ rescale l)
row_l   *= alpha    ← rescale running denom
p        = exp2((s − m_new)·log2e)
row_l   += Σ p
O~      += P @ V     (deferred divide: chia l ở epilogue — trick FA2)
```
Vì sao `exp2` không `exp`? Hardware có `ex2.approx` một instruction; `exp(x) = exp2(x·log2e)` → nhân
`log2e` một lần rồi dùng `ex2` (nhanh + đủ chính xác vì đã trừ max).

**Neo code** (`performance/rental/kernels/fa3_attention_hopper.cu`):
```c
void warpgroup_reg_alloc()  { asm("setmaxnreg.inc.sync.aligned.u32 240;"); }   // :149 consumer xin regs
void warpgroup_reg_dealloc(){ asm("setmaxnreg.dec.sync.aligned.u32 24;"); }    // :150 producer nhả regs
extern "C" __global__ void fa3_attention_fwd(...){                             // :157
  if (is_producer){                                                            // :204
    warpgroup_reg_dealloc();                                                   // :205
    tma_load_2d(sQ, &tmap_Q, 0, q_row0, bar_q);                                // :211 Q load MỘT lần
    for(kt...){ mbar_wait(&bar_empty[stage], ...);                             // :221 back-pressure
                tma_load_2d(&sK[...], &tmap_K, ...); tma_load_2d(&sV[...],...); } // :227-228
    return; }
  warpgroup_reg_alloc();                                                       // :237 consumer
  for(kt...){ mbar_wait(&bar_full[stage], ...);                                // :260 chờ K,V ready
    /* WGMMA #1 S=Q@K^T */ for(ks<KSTEPS_QK) wgmma_m64n64k16(acc_s,..,ks==0?0:1); // :268-271
    wgmma_commit(); wgmma_wait<0>();                                           // :273-274 (FA3: wait<1>)
    /* online-softmax :281-333 */
    alpha[r] = exp2f((row_m[r]-m_new)*LOG2E);                                  // :303
    for(i) acc_o[i] *= alpha[i&1];   // RESCALE O — must-not-forget            // :310
    /* WGMMA #2 O~ += P@V */ for(ks<KSTEPS_PV) wgmma_m64n64k16(acc_o,..,1);    // :340-343
    mbarrier.arrive(bar_empty[stage]);   // release stage → producer           // :349-351
  }
  for(i) acc_o[i] *= (row_l[r]>0 ? 1/row_l[r] : 0);  // epilogue normalize O=O~/l  // :356-360
}
```

**Hình ảnh — warp-specialization ping-pong:**
```
PRODUCER WG (24 reg)          CONSUMER WG (240 reg)
  │ TMA Q (resident)            │ mbar_wait(bar_q)
  │ ┌──stage 0──┐               │ ┌── while consumer drain stage0 ──┐
  │ TMA K0,V0 ──┼──bar_full[0]──► WGMMA S=Q@K0ᵀ → softmax → O~+=P@V0
  │ mbar_wait(bar_empty[1])     │ mbarrier.arrive(bar_empty[0]) ◄──┘
  │ TMA K1,V1 ──┼──bar_full[1]──► WGMMA S=Q@K1ᵀ → ...  (stage kia)
  └─ overlap: fill stage kế TRONG KHI consumer tính stage này ─┘
```
Shape: Q `[Br=64, D=64]` f16 resident; K/V ping-pong `[STAGES=2][Bc=64, D=64]`; acc_o 32 reg/thread f32.
`m_tile[2], row_m[2], row_l[2]` — mỗi thread giữ 2 owned query row (WGMMA D-fragment layout).

**Số đo THẬT.**
```
[measured · compile · RESULTS.md §ISA-gated]  nvcc -arch=sm_90a -ptx AND -cubin exit 0 (ptxas ĐẦY ĐỦ);
      PTX có 8× wgmma + 3× cp.async.bulk.tensor (TMA) + 14 mbarrier + setmaxnreg (warp-spec)
[PREDICTED · FA3 paper]  ~75% util / ~740 TF/s FP16; FP8 (E4M3) ~1.2 PFLOP/s
      — runtime + fragment-map/causal-mask flagged DEFER, KHÔNG claim measured
```

**Frontier / cổng.** FA3 là chuẩn 2026 attention trên Hopper (warp-spec + TMA + FP8 incoherent-processing
Hadamard). **Gate:** "FA3 nhanh hơn FA2 nhờ đâu trên Hopper?" → async WGMMA + TMA cho warp-specialization
giấu HBM sau math (FA2 không async được). "Bug kinh điển?" → quên `acc_o *= alpha` (:310): chỉ rescale `l`
mà không rescale running output → K-tile trước bị weight sai theo max cũ. **Trait:** roofline-first
(memory-bound → overlap). **Scarce-2026:** kernels = top differentiator.

---

## 6.7 · tcgen05 GEMM — TMEM accumulator + full-warpgroup drain (compile-only sm_100a)

**Câu hỏi.** Thế hệ Blackwell datacenter — accumulator dời khỏi register *vào Tensor Memory (TMEM)*, MMA
thu về **single-thread issue**; vậy ai drain 128 lane TMEM ra, và cạm bẫy nào làm 3/4 tile stale?

**Sự thật nền tảng — nghịch lý collective-widening.** Throughline: `mma.sync` = warp(32), `wgmma` =
warpgroup(128), **tcgen05 = single-thread(1)** issue. Nhưng một thread phát MMA, còn accumulator **không
nằm ở register nữa** mà ở **TMEM** (128 lane × 512 col, on-chip). Vì sao? TMEM giải phóng register file
khổng lồ (accumulator to hơn nhiều), MMA issue rẻ (một thread). Đổi lại: **drain phải cả warpgroup**.

**Dẫn xuất — vì sao drain cần 4 warp (bug tcgen05 kinh điển).** TMEM address pack `[31:16]=lane, [15:0]=
col`. Một warp `tcgen05.ld` chỉ chạm **32/128 lane**. Nên:
```
drain một warp  → đọc 32/128 lane → 3/4 tile 128-row STALE (chỉ 1/4 output đúng)
drain 4 warp    → warp w địa chỉ lane-quadrant [32w, 32w+32) → phủ đủ 128 lane
```

**Dẫn xuất — spine 5 bước.**
```
(1) tcgen05.alloc  TMEM accumulator (NCOL pow2 ≥32, MỘT warp issue, rồi relinquish permit)
                   atom lớn nhất f16 = 128×256×16
(2) TMA stage A/B K-strip vào SMEM, mbarrier-tracked
(3) MỘT thread tcgen05.mma.cta_group::1.kind::f16 [d-tmem], a-desc, b-desc, idesc, enable-d
        enable-input-d = predicate: false atom đầu (D:=A·B), true sau (D+=A·B)
(4) tcgen05.commit → arrive trên mbarrier (completion là MBARRIER, KHÔNG wait_group như WGMMA)
(5) full-warpgroup drain: warp w OR (32·w)<<16 vào base; tcgen05.ld.32x32b.x4 → 4 reg/thread; loop col
```
Descriptor SMEM **chung với WGMMA** (LBO/SBO >>4, swizzle bit[62:64)) — cùng artifact.

**Neo code** (`performance/rental/kernels/tcgen05_gemm_sm100a.cu`):
```c
void tcgen05_mma_f16(uint32_t d_tmem, uint64_t a,uint64_t b, uint32_t idesc, bool accum){  // :149
  asm("setp.ne.b32 p, %4, 0;"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;" ...);  // :155 single-thread issue
}
void drain_tmem(uint32_t tmem_base, float* D, int ld_D){                // :186 FULL warpgroup
  const int warp = tid >> 5;                                           // :189 lane-quadrant owner
  const uint32_t warp_lane_base = tmem_base | (32*warp) << 16;         // :192 OR lane-quadrant
  for(int col=0; col<NCOL; col+=4){
    asm("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];"..);// :198-200 32 lane×4 col→4 reg
    asm("tcgen05.wait::ld.sync.aligned;");                             // :201 ld retire trước dùng
    const int row = 32*warp + lane;                                    // :206 phủ đủ 128 lane
  }
}
extern "C" __global__ void tcgen05_gemm_1sm(...){                       // :222
  if (warp==0) tmem_alloc(tslot, NCOL);                                // :242 alloc từ warp 0
  uint32_t d_tmem = *tmem_slot;                                        // :245 base addr alloc ghi về
  for(int k=0; k<n_kiter; ++k){                                        // :258 K-mainloop
    /* TMA stage :260-266 */  mbar_wait(bar_full, phase&1);            // :267
    tcgen05_fence_before();
    if (tid==0) tcgen05_mma_f16(d_tmem, a_desc, b_desc, idesc, k!=0);  // :275 accum = k≠0
    if (tid==0) tcgen05_commit(bar_mma);                              // :277
    mbar_wait(bar_mma, phase&1);                                       // :280 retire trước overwrite
  }
  drain_tmem(d_tmem, D, ld_D);                                         // :287 full-warpgroup drain
  if (warp==0) tmem_dealloc(d_tmem, NCOL);                            // :290
}
```

**Hình ảnh — TMEM drain lane-quadrant:**
```
accumulator KHÔNG ở register — ở TMEM (128 lane × 512 col on-chip)
  MMA issue: 1 thread (tid==0)  ── nghịch lý: rộng ra thì thu về single-thread
  drain:
    warp0 → lane [ 0, 32)  ┐
    warp1 → lane [32, 64)  │  4 warp phủ 128 lane
    warp2 → lane [64, 96)  │  tcgen05.ld.32x32b.x4 → 4 reg/thread, loop NCOL step 4
    warp3 → lane [96,128)  ┘
  drain SAI (1 warp): đúng 32/128 = 1/4; 3/4 STALE  ← verify per-quadrant, không chỉ top-left
```
Ví dụ số: `MMA_M=128, MMA_N=256, MMA_K=16, NCOL=256`; single-warp drain phủ `32/128 = 25%` output.

**Số đo THẬT.**
```
[measured · CPU arithmetic]  single-warp drain coverage = 32/128 = 1/4 (3/4 stale) — cạm bẫy structural
[measured · compile · RESULTS.md §ISA-gated]  nvcc -arch=sm_100a -ptx AND -cubin exit 0;
      PTX có 21× tcgen05 incl. tcgen05.mma.cta_group::1.kind::f16 + TMA;
      SASS (nvdisasm) = UTCHMMA + LDTM.x4
[PREDICTED · book/PERF_ENGINEERING_SPEC §4·A3]  ~1209 TF/s = 54% dense B200 (2250 TF/s); 2-SM cta_group::2 win
      HONEST ~8% (1209→1302, SMEM-bandwidth relief, KHÔNG raw math) — runtime DEFERRED
```

**Frontier / cổng.** tcgen05/UMMA là tensor-core B200 (`-arch=sm_100a`, chữ `a` bắt buộc như sm_90a).
2-SM `cta_group::2` win chỉ ~8% (honest framing — bandwidth relief, không phải math). **Gate:** "Blackwell
đổi gì so với Hopper ở accumulator?" → dời từ register (WGMMA) vào TMEM; MMA issue một thread; drain cần cả
warpgroup vì lane-quadrant. **Trait:** claims-honesty (idesc flagged `[INFERENCE]`; 2-SM win không thổi
phồng). **Scarce-2026:** kernels = top differentiator.

---

## Bảng số đo THẬT (chạy lại được — DoD-là-profile)

| Bài | Đo | Kết quả | Nguồn | Nhãn |
|---|---|---|---|---|
| 6.1 | TP output vs single-GPU | `max diff = 1.78e-15` → identical | verify_s6.py (fp64) | measured |
| 6.1 | bias-in-region overshoot | `3.0 = W−1` | verify_s6.py | measured |
| 6.1 | all-reduce count | 1/fwd, 2 fwd+bwd (MLP) · 4/layer | RESULTS.md §A6 gloo | measured |
| 6.2 | bubble p4m8 / p8m16 / p4m16 | `0.3750 / 0.4375 / 0.1875` (sim==analytic) | verify_s6.py | measured |
| 6.2 | peak-act GPipe vs 1F1B | `8/16/16` vs `4/8/4` (=min(p,m)) | verify_s6.py | measured |
| 6.3 | gate row-sum | `1.0` mỗi token | verify_s6.py | measured |
| 6.3 | dest_rank counts | `[1,3,4,4]` real cross-rank | verify_s6.py | measured |
| 6.3 | EP == dense-gather | allclose, zero token-divergence | RESULTS.md §A6 gloo | measured |
| 6.4 | PaLM MFU / HFU | `0.4570` (Δ0.005) / `0.5712`@r=0.75 | verify_s6.py | measured |
| 6.4 | hfu/mfu @ r=1 | `4/3 = 1.3333` | verify_s6.py | measured |
| 6.4 | six-killer attribution sum | `1.000000` (comm 0.3207, bubble 0.6793) | verify_s6.py | measured |
| 6.5 | WGMMA descriptor | `0x4000004000010200` bit-exact | verify_s6.py | measured |
| 6.5 | acc regs m64n64 / m64n256 | `32 / 128` (→ N≤256) | verify_s6.py | measured |
| 6.5 | PTX wgmma count | `4× wgmma.mma_async` (sm_90a exit 0) | RESULTS.md §ISA | measured (compile) |
| 6.5 | H100 4096³ FP16 | ~318 TFLOPS | book §7.3.1 | **[PREDICTED]** |
| 6.6 | PTX warp-spec | `8× wgmma + 3× TMA + 14 mbarrier + setmaxnreg` (-cubin exit 0) | RESULTS.md §ISA | measured (compile) |
| 6.6 | FA3 util / TF/s | ~75% / ~740 TF/s FP16; FP8 ~1.2 PFLOP/s | FA3 paper | **[PREDICTED]** |
| 6.7 | single-warp drain coverage | `32/128 = 1/4` (3/4 stale) | verify_s6.py | measured |
| 6.7 | PTX tcgen05 count | `21× tcgen05 + TMA`; SASS UTCHMMA+LDTM.x4 (-cubin exit 0) | RESULTS.md §ISA | measured (compile) |
| 6.7 | B200 1-SM / 2-SM | ~1209 TF/s (54% dense) / +8% | SPEC §4·A3 | **[PREDICTED]** |

**Tổng: 17 số MEASURED (14 CPU/gloo/arithmetic correctness + 3 compile-verified) · 3 [PREDICTED] (TF/s,
runtime DEFERRED tới H100/B200).**

---

## Checklist recall COLD (che phần trên, tự trả lời — đây là cách re-learn)

1. **6.1** Vì sao GeLU nằm *giữa* column-parallel và row-parallel mà **không** cần all-reduce trước nó?
   (chạm hai chữ: *elementwise* + *output-column*). Một transformer layer tốn mấy all-reduce, ở đâu?
2. **6.1** Nếu cộng bias `fc2` **trước** `reduce_from_model_parallel` với `W=4` — output lệch mấy lần bias?
   Test nào bắt?
3. **6.2** Dẫn `bubble = (p−1)/m` (makespan, busy). GPipe và 1F1B bubble giống hệt — vậy 1F1B mua gì?
4. **6.2** `p=4, m=8 → 0.375`. Tăng `m=16` giữ `p=4`: bubble mới =? peak-act 1F1B đổi từ `min(4,8)` sang?
5. **6.3** Vì sao cần all-to-all *metadata trên counts* TRƯỚC all-to-all trên feature? Vì sao invariant đòi
   **zero token-divergence** chứ không chỉ output allclose?
6. **6.3** Bỏ un-permute `slot_out[perm]=back` (:235) — output còn allclose không? Mutation-test nào bắt?
7. **6.4** Dẫn `C=6ND` (2 fwd + 4 bwd — 4 đó từ đâu?). `HFU/MFU` khác nhau đúng *một* knob nào? `4/3` từ đâu?
8. **6.4** Run `0.9-comm × 0.8-bubble`, ideal 0.6 → realized? share bubble = bao nhiêu %? (log-additive)
9. **6.5** Descriptor 64-bit encode `addr=0x2000, 128B swizzle` = ? (bit-trace). Vì sao B *phải* ở SMEM còn
   A được ở register? Vì sao `N ≤ 256`? (acc reg cap)
10. **6.6** Bug "quên rescale O" — cụ thể quên dòng nào (:310), output sai thế nào? FA3 nhanh hơn FA2 nhờ gì?
11. **6.7** Vì sao MMA single-thread-issue nhưng drain phải cả 4 warp? Drain từ 1 warp → mấy phần output
    đúng, ở đâu của tile? Blackwell đổi gì so Hopper ở accumulator?

> Trả lời COLD được cả 11 = **S6 thật sự OWNED** (interview-grade). Vấp câu nào → mở đúng mục đó, hoặc
> blank-slate hàm tương ứng (`raise NotImplementedError` → test đỏ → tự dẫn → xanh; distributed dùng gloo,
> kernel dùng `nvcc -ptx` grep-ISA) để re-own bằng tay. **Honesty rail (FOP-4):** với 6.5–6.7, "OWNED" =
> dẫn được cấu trúc + descriptor + PTX-ISA; TF/s vẫn `[PREDICTED]` tới ngày rental — đừng claim measured.

---

*Cross-ref: `docs/learning/roadmap/S6_distributed_and_isa.md` (reference chung, commit `9e61d7a`) ·
`performance/notes/A6_design_note.md` (map-the-chattiest-collective) ·
`performance/artifacts/wgmma_descriptor_manual.md` (§2 descriptor bit-map, §5 wgmma_ss) ·
`bench/RESULTS.md §"Perf track (A6…)" + §"ISA-gated kernels"` (số đo gốc) · `PROGRESS.md` (ledger 89 Bài) ·
sibling: `M2_transformer_forward.md` (forward pass). Script số đo: `scratchpad/verify_s6.py`. Rental
runbooks: `H100_day_runbook.md` · `B200_day_runbook.md` · `serving_day_8xH200_runbook.md`.*
