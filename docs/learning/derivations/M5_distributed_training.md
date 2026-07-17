# M5 — Distributed Training · First-Principles Derivations

> **Đây là gì.** Bản dẫn-xuất **từ nguyên lý gốc** (first principles) cho TOÀN BỘ nửa distributed-training
> của A2 — 5 micro-concept M5 (Bài 5.1→5.5): làm sao chẻ *một training step* ra nhiều GPU mà vẫn giữ đúng
> **một identity đại số**. Mỗi mục: (1) **Câu hỏi** falsifiable, (2) **Sự thật nền tảng** (áp lực vật lý/toán
> ép ra thiết kế), (3) **Dẫn xuất** kèm math, (4) **Neo code** `file·func·line`, (5) **Hình ảnh** (ASCII +
> shape/stride/dtype + micro-example tay), (6) **Số đo THẬT** (chạy trên chính repo này), (7) **Frontier**
> + cổng phỏng vấn.
>
> **Cách dùng để re-learn.** Đọc *Câu hỏi* của mỗi mục → **tự trả lời COLD** (che phần dưới) → mở code đối
> chiếu → chỉ đọc kỹ phần *delta* giữa dự đoán và thực tế. Cuối doc có **checklist recall COLD** + **bảng số
> đo**. Đây là *derivation lab* có số đo, bạn đồng hành của `roadmap_model/M5_distributed_training.md` (spine).
>
> **Cảnh báo neo.** Line number **trôi** theo commit. Cite ở đây pin theo HEAD `1e7dbc6` (2026-07-14). Nếu
> lệch, `grep` tên hàm — đừng tin số dòng cứng (luật repo: *verify, don't trust*).
>
> **Nguồn số đo.** Correctness DDP/ZeRO-1/FSDP = **MEASURED** trên gloo 2-rank (15/15 test pass, `tests/
> test_{ddp,zero1,fsdp}.py`). Memory + comms = **arithmetic exact**, pin bởi `test_{memory_math,comms_calc}.py`
> — số 100B/xl là *đại số đóng*, không phải training run. Real multi-GPU NCCL busbw / MFU-at-scale / cliff
> NVLink→IB là **[PREDICTED] rental-gated** (8×H200 day). Through-line honesty (FOP-4): *identity + closed
> form exact BÂY GIỜ; bandwidth measured ở rental.*

---

## Bức tranh lớn — chẻ một step ra N GPU mà vẫn = một single-process

M1–M4 dựng *một* model chạy trên *một* GPU. Nhưng một model frontier **không nhét nổi một card**: 100B param
ở mixed-precision = **2.0 TB state** *trước cả một activation* — gấp **25×** một H100-80GB. M5 derive cái
thang chẻ nó ra, và mọi rung của thang mọc từ **một identity**:

```
                     MỘT IDENTITY GỐC (5.1)
   loss = mean over B ⇒ ∇(1/B·Σlᵢ) = avg_ranks[ per-rank mean grad ]   (EXACT, không xấp xỉ)
   ⇒ mọi replica step GIỐNG HỆT một single-process chạy cả batch, MIỄN LÀ:
        (a) mọi rank khởi từ CÙNG weight (broadcast rank0)   (b) grad all-reduce SUM rồi /N

   Từ identity đó mọc CÁI THANG SHARDING — mỗi rung bỏ đi một thứ replicated:

   ┌─────────────┬──────────────┬──────────────┬───────────────┬──────────────────────┐
   │             │ weight (2-4) │ grad (2-4)   │ optim m+v(+M)│ per-rank @100B,W=64  │
   ├─────────────┼──────────────┼──────────────┼──────────────┼──────────────────────┤
   │ DDP  (st0)  │  replicated  │  replicated  │  replicated  │  2000 GB   ✗          │ 5.1
   │ ZeRO-1(st1) │  replicated  │  replicated  │  SHARDED /W  │   425 GB   ✗          │ 5.2
   │ ZeRO-2(st2) │  replicated  │  SHARDED /W  │  SHARDED /W  │   228 GB   ✗          │ 5.2
   │ ZeRO-3=FSDP │  SHARDED /W  │  SHARDED /W  │  SHARDED /W  │    31 GB   ✓ (fit 80) │ 5.3
   └─────────────┴──────────────┴──────────────┴──────────────┴──────────────────────┘
              ↑ chẻ state              ↑ nhưng activation là TB THỨ HAI:
                                         651 GB/stack vanilla → flash 114 → ckpt 6.7   5.4

   VÀ một câu hỏi xuyên suốt: COLLECTIVE NÀO CHẠY TRÊN LINK NÀO, scale tới đâu thì network chặn?  5.5
        DP chết ở  N ≤ B·W/C   ·   TP chết ở  N ≤ (3/2)·D_ff·W/C   ·   2D = (3/2)·B·D_ff·(W/C)²
```

**Sợi chỉ xuyên M5:** *map the chattiest collective onto the fattest link, và biết TRƯỚC — bằng closed
form — khi thêm GPU ngừng giúp.* Mỗi thiết kế M5 là lời giải cho **một áp lực**: throughput (DDP), bộ nhớ
state (ZeRO/FSDP), bộ nhớ activation (flash/checkpointing), băng thông interconnect (comms algebra). Học M5 =
học *áp lực → collective → closed-form bound*.

**Ranh giới với S6 (perf-roadmap).** M5 mổ *training data-parallel*: DDP · ZeRO · FSDP · memory · comms
algebra. Serving-parallelism primitives — TP-MLP · 1F1B-pipeline · EP-MoE · MFU · WGMMA — ở `roadmap/
S6_distributed_and_isa.md`. Khi chạm TP/PP/EP ta **cross-link** sang S6, không giải lại.

---

## 5.1 · Data-parallel + ring all-reduce — vì sao average = full-batch, và byte bounded

**Câu hỏi.** (a) Vì sao *trung bình* gradient per-rank **bằng ĐÚNG** (không xấp xỉ) gradient full-batch? (b)
Một all-reduce chuyển **bao nhiêu byte per-device** — bounded, hay tăng theo N?

**Sự thật nền tảng.** N công nhân, mỗi người nhận một *phần* batch (data parallel: **chẻ data, replicate
model**). Ai cũng tính grad trên phần mình rồi *phải đồng ý* trước khi bước — nếu không, các replica trôi ra
xa nhau và bạn có N model khác nhau thay vì một. "Đồng ý" = all-reduce: cộng grad mọi rank rồi chia N. Áp lực
cốt lõi: DDP cho throughput (N× tokens/step) *miễn phí về capacity*, nhưng mỗi step phải trả một all-reduce
toàn bộ gradient — và câu hỏi sống-còn là all-reduce đó tốn *bao nhiêu byte* và có **giấu được dưới backward**
không.

**Dẫn xuất.**

*Correctness identity (a) — vì sao EXACT.* Loss là mean over B example: `L = (1/B)·Σᵢ lᵢ`. Gradient tuyến
tính: `∇L = (1/B)·Σᵢ ∇lᵢ`. Chẻ B thành N shard bằng nhau (`B/N` mỗi rank). Rank `r` tính **mean grad local**:
```
g_r = (N/B) · Σ_{i ∈ shard_r} ∇lᵢ
```
Trung bình N cái đó:
```
(1/N)·Σ_r g_r = (1/N)·(N/B)·Σ_all ∇lᵢ = (1/B)·Σ_all ∇lᵢ = ∇L    ← full-batch gradient, EXACT
```
Không có "≈" ở đâu cả — chỉ là tuyến tính của `∇` + phân phối tổng. **Hai điều kiện** để identity giữ:
(a) mọi replica khởi từ *cùng weight* (broadcast rank 0 lúc init), (b) grad **all-reduce SUM rồi chia N**
trước step. Bỏ điều kiện (b) — quên `/N` — thì grad bị SUM chứ không mean → lớn gấp N× → learning-rate hiệu
dụng gấp N → trajectory trôi ngay step 1.

*Busbw derive (b) — vì sao bounded.* Ring all-reduce = **reduce-scatter + all-gather**. Mỗi device giữ chunk
`S/N`; ring có `N−1` bước, mỗi bước forward một chunk `S/N` cho hàng xóm phải → mỗi phase gửi `(N−1)/N·S`
byte/device. Cộng hai phase:
```
all-reduce/device = 2·(N−1)/N·S    →  N→∞:  2S   (BOUNDED, độc lập N)
```
Đây LÀ định nghĩa **bus-bandwidth**: `busbw = algbw · 2(N−1)/N` (algbw = `S/time` "thuật toán thấy", busbw =
"link thật cõng"). *Bounded 2S là toàn bộ lý do data-parallel scale được* — thêm GPU chia *compute* nhưng
**không** tăng wire volume/device. So với **naive** (mỗi device gửi full tensor tới N−1 peer = `(N−1)·S`):
ring/naive = `2/N` — ring thắng bằng cách chunk sao cho mỗi byte của reduction qua mỗi link *đúng một lần*.

*Ladder ba rung — cùng numerics, bóc dần idle byte-movement:*
- **`"naive"`** — all-reduce từng grad tensor riêng → trả latency floor `2(N−1)·α` cho *mỗi* tensor.
- **`"flat"`** — concat mọi grad vào một buffer → **một** all-reduce → amortize latency per-call.
- **`"overlap"`** — bắn all-reduce từ `register_post_accumulate_grad_hook` ngay khi mỗi grad sẵn sàng, để
  comm giấu dưới backward *đang chạy*. Đây là production DDP + graded deliverable.

**Neo code** (`utils/ddp.py`):
```python
class DDP(nn.Module):
    def __init__(self, module, mode="overlap"):                       # :35
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)                             # :42  điều kiện (a): cùng weight
        if mode == "overlap":
            for p in self.module.parameters():
                if p.requires_grad:
                    p.register_post_accumulate_grad_hook(self._async_all_reduce)  # :50

    def _async_all_reduce(self, p):                                   # :52
        handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)  # :54  bắn ngay, non-blocking

    def finish_gradient_synchronization(self):                       # :60  gọi sau backward, trước step
        if self.mode == "overlap":
            for handle, p in self._handles:
                handle.wait()                                         # :65
                p.grad /= self.world_size                             # :67  điều kiện (b): /N ⇒ mean
        elif self.mode == "flat":
            flat = torch.cat([g.reshape(-1) for g in grads])          # :71  một buffer
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)               # :72  MỘT collective
            flat /= self.world_size                                   # :73
        else:  # naive                                                # :79
            for p in self.module.parameters():
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM); p.grad /= self.world_size  # :82-83
```

**Hình ảnh — ring all-reduce, N=4, S=4 chunk (mỗi ô = 1 chunk `S/N`):**
```
   reduce-scatter (N−1=3 bước): mỗi device tích luỹ MỘT chunk-sum khác nhau
   dev0 [a₀ b₀ c₀ d₀]   step1→ dev1 nhận a, cộng → ... → sau 3 bước:
   dev1 [a₁ b₁ c₁ d₁]          dev0 giữ Σd, dev1 giữ Σa, dev2 giữ Σb, dev3 giữ Σc
   dev2 [a₂ b₂ c₂ d₂]          (mỗi device gửi 3 chunk = (N−1)/N·S = 3/4·S)
   dev3 [a₃ b₃ c₃ d₃]
   all-gather (N−1=3 bước): quay vòng phát chunk-sum đã có → mọi device có [Σa Σb Σc Σd]
                            (lại 3/4·S) ⇒ tổng 2·(N−1)/N·S = 3/2·S per device  ✓ khớp số đo N=4
   grad journey: p.grad (fp32, shape param) → all_reduce SUM (in-place) → /world_size ⇒ mean grad
```
Hand-trace `/N`: 2 rank, grad rank0 = 3.0, rank1 = 1.0. SUM → 4.0 trên cả hai. `/2` → 2.0 = mean(3,1). ✓
Nếu quên `/2`: cả hai giữ 4.0 = SUM → step gấp đôi LR → drift.

**Số đo THẬT** (ring fractions + latency floor = closed form `utils/comms_calc.py`, pinned
`tests/test_comms_calc.py`; DDP==single-proc = gloo `tests/test_ddp.py`, atol 1e-5; `S=1 GB`):
```
ring all-reduce/device:  N=2→1.000·S   N=8→1.750·S   N=64→1.969·S   N=1024→1.998·S   N→∞→2S (bounded)
reduce-scatter = all-gather = (N−1)/N·S:  N=8 → 0.875·S mỗi cái
naive N=8 = 7.0·S ;  ring/naive = 0.2500 = 2/W ;  alternate-ring N=8 = 7.0·S (W/2 tệ hơn)
latency floor: 200 tensor all-reduce riêng = 3.150 ms  vs  flat (1 all-reduce) = 0.364 ms → 8.7× nhanh hơn
DDP gloo 2-rank == single-process full-batch, 3 step:  naive/flat/overlap đều max|Δparam| = 1.49e-08  PASS
```

**Frontier / cổng.** DDP cưỡi *inter-node link* (IB, băng thông rẻ) được vì comm của nó — grad all-reduce —
*giấu được* dưới backward (overlap hook). Đối lập TP (S6 Bài 6.1): all-reduce *activation* của TP nằm trên
critical path, không giấu được → phải cưỡi NVLink. **Interview:** "một DDP step chuyển bao nhiêu byte/device,
vì sao thêm GPU không tăng byte đó?" → `2(N−1)/N·S ≈ 2` bản grad, bounded 2S; thêm GPU chỉ chia *compute*,
không chia wire volume (trần DP ở 5.5). Trait = **predict-the-number** (busbw closed form). Ref: ZeRO paper
(1910.02054) §baseline. Table-stakes systems.

---

## 5.2 · ZeRO-1/2/3 — mô hình 16-psi + chẻ state theo bucket

**Câu hỏi.** DDP replicate *toàn bộ* 16 B/param state trên *mọi* rank. Cái nào *bắt buộc* replicated, cái
nào chẻ được — và chẻ nó có tốn thêm **byte-on-wire** không?

**Sự thật nền tảng.** AdamW mang *bốn* fp32 buffer mỗi param: weight 4 + grad 4 + moment `m` 4 + variance `v`
4 = **16 B/param** ("16-psi", ψ = số param). Ở 100B param đó là **1.6 TB state** *trước cả một activation*.
DDP thảm hoạ: replicate cả 16 B/param trên *mọi* rank → thêm GPU tăng throughput, **không bao giờ** tăng
capacity. Quan sát ZeRO: update Adam là **elementwise** — param `i` chỉ cần state của chính `i`. Nên *phân
vùng* param: mỗi rank *sở hữu* ~1/W param, giữ optimizer state chỉ cho shard mình, step shard đó, rồi
**broadcast** param đã update từ chủ. Cuối step mọi rank vẫn cầm full model giống hệt; chỉ *bộ nhớ* `m/v` chia
W. Đánh đổi tuyệt vời: memory ÷W gần như **miễn phí về comm**.

**Dẫn xuất.**

*16 → 20 B/param dưới mixed-precision.* Đừng nhầm "mixed precision tiết kiệm memory" — với *state* nó làm
**to hơn**. fp32 master 4 + fp32 grad 4 + `m` 4 + `v` 4 (optimizer-side 16) **+** bf16 compute-copy weight 2
**+** bf16 grad buffer 2 = **20 B/param** (hoặc 18 nếu fuse upcast, không materialize bf16-grad). Mixed làm
*matmul* rẻ + halve *activation* dtype — không shrink state.

*Ladder ZeRO chẻ theo ranh giới bucket tự nhiên* (`_zero_buckets` :123 trả `(weights, grads, optimizer)`
byte/param):
- **ZeRO-1** chẻ *optimizer bucket* (master + fp32-grad + m + v = `4·4 = 16` B/param mixed); weight + backward-
  grad **vẫn replicated** (grad phải replicated để all-reduce). Per-rank = `(w+g)·N + opt·⌈N/W⌉`.
- **ZeRO-2** chẻ thêm *grad* (reduce-scatter thay all-reduce → mỗi rank chỉ giữ grad cho shard mình).
- **ZeRO-3 / FSDP** chẻ thêm *weight* → mọi thứ `~1/W`, giá là all-gather weight on-demand mỗi fwd/bwd (5.3).

*100B arithmetic, mixed, W=64* (`zero_shard_bytes`): buckets = weights 2, grads 2, optimizer 16 B/param.
```
stage 0 (DDP):  (2+2+16)·100B          = 2000 GB/rank   ✗
stage 1:  (2+2)·100B + 16·⌈100B/64⌉    =  400 + 25       =  425.00 GB/rank  ✗
stage 2:  2·100B + (2+16)·⌈100B/64⌉    =  200 + 3.1 + 25 =  228.12 GB/rank  ✗
stage 3:  (2+2+16)·⌈100B/64⌉ = 20·1.5625B                =   31.25 GB/rank  ✓ (fit 80 GB)
```
**Corollary (tested):** cho 100B trên 80 GB, ZeRO-1 và ZeRO-2 fail ở *mọi* world size — replicated bucket một
mình đã vượt HBM (stage1 replicated = `4·100B = 400 GB`; stage2 = `2·100B = 200 GB`). Chỉ ZeRO-3 xuống dưới.
Floor state-only = **25 GPU** (mixed, `gpus_needed` :185) / 20 (fp32).

*Vì sao ZeRO-1 comms-free.* DDP trả *một ring all-reduce* = reduce-scatter + all-gather (5.1). ZeRO-1: mỗi
rank chỉ cần summed-grad cho optimizer-shard của nó → **reduce-scatter** (`(N−1)/N·P·b`); rồi republish
param-shard đã update → **all-gather** (`(N−1)/N·P·b`). *Đúng bằng hai nửa của ring all-reduce* → cùng wire
volume DDP. Nên ZeRO-1 = memory ÷W ở comm cost *bằng* DDP.

*Greedy partition.* `partition_by_numel` gán mỗi param cho rank *least-loaded* (tie → rank thấp); đảm bảo
`max−min load ≤ max(numel)`.

**Neo code** (`utils/zero1.py`):
```python
def partition_by_numel(numels, world_size):                          # :31
    loads = [0] * world_size
    for n in numels:
        rank = min(range(world_size), key=loads.__getitem__)         # :39  least-loaded, tie→thấp
        owners.append(rank); loads[rank] += n

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):             # :66
        owners = partition_by_numel([p.numel() for p in flat], self.world_size)  # :81
        self._owner = dict(zip(flat, owners, strict=True))           # :82
        mine = [p for p in group["params"] if self._owner[p] == self.rank]  # :86  chỉ param của rank mình
        self.inner = optimizer_cls(local_groups, **kwargs)           # :90  inner optim = local shard

    @torch.no_grad()
    def step(self, closure=None):                                    # :103
        if self.inner is not None: self.inner.step()                 # :109  step CHỈ shard local
        if self.world_size > 1:
            handles = [dist.broadcast(p.data, src=self._owner[p], async_op=True)  # :112  republish param
                       for group in self.param_groups for p in group["params"]]
            for handle in handles: handle.wait()                     # :118  mọi rank lại đồng bộ full model
```
`training_state_breakdown` :48 và `zero_shard_breakdown` :136 (`memory_math.py`) là các số 100B ở trên —
sharded bucket charge `⌈N/W⌉` param (:161).

**Hình ảnh — thang chẻ, per-rank ở W=64 (mỗi khối tỉ lệ byte):**
```
        weight  grad          optimizer (master+fp32grad+m+v)
DDP   [██     ][██     ][████████████████]  = 2000 GB   ← mọi thứ replicated ×64 lãng phí
ZeRO1 [██     ][██     ][▌]                  =  425 GB   ← opt ÷64, weight+grad còn full
ZeRO2 [██     ][▌      ][▌]                  =  228 GB   ← + grad ÷64
ZeRO3 [▌      ][▌      ][▌]                  =   31 GB   ← + weight ÷64 (mọi thứ /W)  ✓
        │ replicated (200GB)   │ sharded /64 (25→0.4GB)
```
Hand-trace greedy `partition_by_numel([100,50,50,25,25], W=2)`:
```
loads=[0,0]  n=100→rank0 [100,0]   n=50→rank1 [100,50]   n=50→rank1 [100,100]
n=25→rank0(tie→thấp) [125,100]   n=25→rank1 [125,125]  ⇒ owners=[0,1,1,0,1]  loads cân 125/125 ✓
```

**Số đo THẬT:**
```
fp32 state  = 16 B/param → 100B = 1.60 TB    mixed = 20 B/param → 2.00 TB  (state TO HƠN dưới mixed)
mixed breakdown/param: master4 grad4 m4 v4 bf16w2 bf16g2                 (= 20)
W=64 per-rank:  stage0=2000  stage1=425.00  stage2=228.12  stage3=31.25 GB  (chỉ stage3 fit 80GB)
gpus_needed(100B,80GB): stage3 mixed=25  fp32=20 ;  stage1/stage2 → RAISE "replicated exceeds HBM"
partition_by_numel([100,50,50,25,25], 2) = [0, 1, 1, 0, 1]   (load 125/125 cân)
ZeRO-1 wire/step == DDP wire/step (ZeRO1/DDP ratio = 1.0000, xl N=8: cả hai 11.744 GB)  ← comms-free
```

**Frontier / cổng.** ZeRO stages = Rajbhandari et al. **1910.02054** (2019), abstraction chuẩn của DeepSpeed.
**Interview kinh điển** (chính docstring `zero1.py`): "100B param, AdamW — memory đi đâu, ZeRO-1 mua gì ở
world size W?" → 1.6 TB @ 16 B/param; 8 B/param moment (mixed: 16 B optimizer) → `÷W`, ở *near-DDP comm cost*.
Trait = **first-principles memory accounting**. Nối 5.3: ZeRO-3 là chỗ nó thành FSDP thật. Scarce-2026 bucket:
table-stakes (mọi lab lớn train ở ZeRO-3/FSDP), nhưng *giải thích được comms-free* là differentiator.

---

## 5.3 · FSDP (ZeRO-3) — flat-param all-gather/reduce-scatter, shard matrices / replicate norms

**Câu hỏi.** Chẻ *param* (không chỉ state) ra W rank — full model chỉ tồn tại *thoáng qua* — thì fwd/bwd phải
gọi collective gì, và vì sao norm/bias **không** nên shard?

**Sự thật nền tảng.** DDP giữ full model mọi rank. FSDP: mỗi rank chỉ giữ *một lát 1/W* của mỗi ma trận (fp32
master shard). Khi `forward()` cần một layer, nó **all-gather** các shard lại thành full weight *tạm thời*,
tính, rồi (sau backward) **reduce-scatter** full gradient để mỗi rank giữ đúng grad của shard mình. Full model
không bao giờ resident lâu — nó *materialize thoáng qua*. Nhưng **không phải mọi param đều shard**: ma trận
lớn (`ndim≥2`: Linear/Embedding) shard; norm/bias (`ndim≤1`) **replicate**.

**Dẫn xuất.**

*Vì sao replicate 1-D param là BẮT BUỘC, không cosmetic — hai lý do:*
1. **Memory:** shard một RMSNorm 64-element mua **gần-zero** memory (64/W byte tiết kiệm) mà vẫn trả *full
   comm* (một collective). Không đáng.
2. **Correctness (quan trọng hơn):** official `test_fsdp_gradient_sync` assert grad của mọi non-Linear param
   **bit-identical across ranks**. Một reduce-scattered shard-grad, theo construction, *khác nhau per-rank*
   (mỗi rank sở hữu lát khác) → **fail** check đó. Path *replicate-and-all-reduce* là cái làm norm/bias grad
   khớp. Phân loại theo `ndim` trùng khít predicate "parent không phải Linear/Embedding" (mọi matrix 2-D, mọi
   norm 1-D) — và là convention ZeRO-3 chung.

*Shard mechanics.* Param sharded: flatten → pad tới bội của W (`shard_len = ⌈numel/W⌉`) → giữ lát
`[rank·shard_len : (rank+1)·shard_len]`. `p.data` từ đó trỏ *chỉ shard*; `.data` swap sang full-gathered copy
trong forward, về master shard ở grad-sync.

*Wire cost derive.* `fsdp_step_bytes` model **production FSDP**: param sharded → all-gather full weight cho
forward (`(N−1)/N·P·b`), all-gather **lại** cho backward (đã free sau dùng), reduce-scatter grad về chủ shard
(`(N−1)/N·P·b`) → **3 one-way pass** vs DDP 2 → ratio đúng **3/2** ở mọi N.

> **⚠️ Gap code-vs-textbook (dạy code THẬT):** container `utils/fsdp.py` này là **FSDP1-minimal** — forward
> all-gather full copy rồi **GIỮ resident xuyên backward** (autograd's saved tensors pin nó), *không* re-gather
> cho backward. Nên container *thật sự* chuyển **2 leg = DDP-equivalent** cho sharded param, không phải 3/2.
> Con số 3/2 là closed-form của *true FSDP* (free-after-forward + re-gather). Peak resident của container này
> = full model + shards trong fwd/bwd — freeing sau forward ở đây là "accounting theater" vì autograd giữ
> gathered weight tới backward. **ADR trigger** (docstring :47): nếu gloo pass nhưng GPU bench thấy all-gather
> không overlap compute → nâng wrap granularity (per-param → per-block) để free thật + prefetch.

**Neo code** (`utils/fsdp.py`):
```python
def __init__(self, module, compute_dtype=None):                      # :97
    for p in module.parameters(): dist.broadcast(p.data, src=0)      # :105  cùng weight (điều kiện a)
    for name, p in module.named_parameters():
        if p.ndim >= 2:                                              # :113  SHARDED: ma trận
            shard_len = -(-numel // self.world_size)                 # :116  ⌈numel/W⌉ pad bội W
            master = flat[self.rank*shard_len:(self.rank+1)*shard_len].clone()  # :119  lát rank mình
            p.data = master                                          # :120  từ đây chỉ giữ shard
        else:                                                        # :122  REPLICATED: norm/bias (ndim≤1)
            self._replicas.append(_Replica(p, name, p.data))         # :124  giữ full fp32

def forward(self, *args, **kwargs):                                  # :133
    for s in self._shards:
        s.param.data = self._all_gather_full(s, dtype or s.master.dtype)  # :140  materialize full THOÁNG QUA
    return self.module(*args, **kwargs)

def finish_gradient_synchronization(self):                           # :145
    for s in self._shards:                                           # sharded → reduce-scatter
        dist.reduce_scatter_tensor(shard_grad, flat, op=dist.ReduceOp.SUM)  # :160
        shard_grad /= self.world_size                                # :161  ⇒ mean, shard-shaped
    for r in self._replicas:                                        # replicated → all-reduce (như DDP)
        dist.all_reduce(g, op=dist.ReduceOp.SUM); g /= self.world_size  # :171-172  ⇒ IDENTICAL mọi rank
```

**Hình ảnh — vòng đời một sharded param (Linear weight (16,8), numel=128, W=2):**
```
  init:   flatten 128 → shard_len=⌈128/2⌉=64 → rank0 giữ flat[0:64], rank1 giữ flat[64:128]  (fp32 master)
          p.data → shard [64] fp32                                         (resident giữa step: 64/rank)
  forward: all_gather([64]×2) → cat → [:128].view(16,8)  → p.data = full (16,8)  (thoáng qua, compute_dtype)
  backward: autograd tính grad theo FULL (16,8) → p.grad shape (16,8) fp32
  sync:   reduce_scatter_tensor(full_grad→[64]) SUM /W → shard_grad [64]  → p.data=master[64], p.grad=[64]
          ⇒ grad shard-shaped khớp .data ✓                       (norm 1-D: all_reduce → full, giống mọi rank)
  numel not divisible: numel=130,W=4 → shard_len=⌈130/4⌉=33, pad→132, rank3 có 2 zero-padding
```

**Số đo THẬT:**
```
FSDP classification (model thật nn.Linear(8,16)+SiLU+Linear(16,4)):
   SHARDED (ndim≥2):  0.weight (16,8) numel 128 ,  2.weight (4,16) numel 64
   REPLICATED (ndim≤1): 0.bias (16,) , 2.bias (4,)                    ← đúng policy
FSDP/DDP wire ratio (xl 3.355B param, bf16):  N=2→1.5000  N=8→1.5000  N=64→1.5000  (3/2 EXACT mọi N)
   xl N=8:  DDP=11.744  ZeRO1=11.744  FSDP=17.616 GB/step
gloo correctness: test_ddp + test_zero1 + test_fsdp = 15/15 PASS (loss+full-grad+trajectory == single-proc,
   3 seed × ≥3 step, MLP và TransformerLM, grad non-Linear bit-identical across ranks)
```

**Frontier / cổng.** *FSDP1 vs FSDP2* (FRONTIER_PRACTICE_2026): impl này FSDP1-style (flatten per-param).
**FSDP2** = PyTorch `fully_shard`: per-parameter **DTensor** sharding thay FlatParameter, ~7% lower per-GPU
memory / ~1.5% higher throughput, *compose được* với TP/PP (shard một weight hai chiều) — TorchTitan default
(**arXiv:2410.06511**). **Interview:** "DDP all-reduce ~2(N−1)/N·S; FSDP chuyển gì, memory đi đâu?" → 3 leg
(1.5×) để cắt resident param `S→S/W`; memory đổi lấy một all-gather thừa. Trait = **subtract-before-add**
(shard cái đáng, replicate cái nhỏ). Cross-link TP-side sharding một matmul: S6 Bài 6.1. Ref: 1910.02054.

---

## 5.4 · Activation checkpointing + mixed-precision memory — the 100B one-pager

**Câu hỏi.** State đã chẻ xong (5.2–5.3) nhưng *activation* là **TB thứ hai** — recompute đổi được bao nhiêu
memory lấy bao nhiêu FLOP, và vì sao mixed-precision làm state *to hơn* chứ không nhỏ?

**Sự thật nền tảng.** Sharding cứu *state*; nhưng forward lưu *activation* để backward dùng, và ở seq 4k một
block 100B lưu **~8.1 GB** — ×80 layer = **651 GB**, một TB thứ hai còn to hơn cả state ZeRO-3 chừa lại
(31 GB/rank). Hai đòn bẩy: (1) **FlashAttention** không bao giờ materialize hai ma trận score (seq×seq) → bỏ
đúng term bậc hai. (2) **Gradient checkpointing**: lưu *chỉ input residual của mỗi block*, recompute nội thất
block trong backward → activation memory từ `O(depth·per_block)` xuống `O(depth·residual)`, giá ≈ một forward
thừa.

**Dẫn xuất.**

*Activation formula* (Korthikanti 2022 §4.1, `activation_bytes_per_layer` :95):
```
act_bytes/layer = 34·s·b·h  +  5·a·s²·b
                  └ linear ┘    └ quadratic ┘
```
- Term **tuyến tính** `34·s·b·h` = mọi thứ tỉ lệ residual stream (LN input, Q/K/V, attn out, 8h của MLP,
  mask) — bf16 activations, 4h MLP, no TP/SP.
- Term **bậc hai** `5·a·s²·b` = hai ma trận attention materialized (score fp16 2 + softmax fp16 2 + mask 1).
- `flash=True` bỏ **đúng** term bậc hai (:111) — FlashAttention recompute score tile-wise trong backward.

*Vì sao flash bắt buộc ở long-context.* Term bậc hai `∝ s²`; seq 4k→8k (gấp đôi) → vanilla `×~4` (bậc hai
thống trị), flash chỉ còn tuyến tính `∝ s` → `×2`. Đó *là* lý do flash bắt buộc khi context dài.

*Mixed-precision state — TO HƠN, không nhỏ* (`training_state_breakdown` :48): fp32 = weight 4 + grad 4 + m 4
+ v 4 = **16**. Mixed = fp32 master 4 + fp32 grad 4 + m 4 + v 4 + **bf16 weight compute copy 2** (+ bf16 grad
2 nếu không fuse upcast) = **18–20**. Mixed tiết kiệm *activation-dtype* + *matmul-time*, **không** state.

*Recompute trade* (`checkpointing.py`): `"full"` lưu chỉ block-input, recompute cả block; `"selective"` (SAC,
default 2026) lưu cái *store-cheap/recompute-expensive* = **matmul output**, recompute phần rẻ (softmax/
elementwise) → ~70% activation memory ở ~2.7% FLOP thừa. `_sac_policy` :56: op ∈ `MATMUL_OPS` → `MUST_SAVE`,
else `PREFER_RECOMPUTE`.

*Fp16-accumulation cạm bẫy* (`mixed_precision.py`): running sum fp16 **đứng im** khi increment < nửa-ulp ở
magnitude hiện tại. Cộng `0.01` vào fp16 kẹt ở `32.0` vì `ulp@32 = 2⁻⁵ = 0.03125`, `0.01 < ulp/2 = 0.0156`
→ round-to-no-op. *Vì sao* optimizer state / loss / reduction accumulate **fp32** dù compute bf16.

**Neo code:**
```python
# memory_math.py
linear    = 34 * seq * batch * d_model                               # :110
quadratic = 0 if flash else 5 * n_heads * seq * seq * batch          # :111  flash bỏ ĐÚNG term bậc hai
# residual_stream_bytes :115 = 2·s·b·d ← cái checkpointing giữ thay 34·s·b·h nội thất

# checkpointing.py
def _sac_policy(_ctx, op, *a, **k):                                  # :56
    return CheckpointPolicy.MUST_SAVE if op in _SAVE_OPS else CheckpointPolicy.PREFER_RECOMPUTE
def run_block(block, mode, *args):                                   # :63
    if mode == "full":      return checkpoint(block, *args, use_reentrant=False)          # :69
    if mode == "selective": return checkpoint(block, *args, use_reentrant=False,          # :71
                                context_fn=lambda: create_selective_checkpoint_contexts(_sac_policy))

# mixed_precision.py
def naive_accumulate(value, count, dtype):                           # :28
    acc = torch.zeros((), dtype=dtype)
    for _ in range(count): acc += torch.tensor(value, dtype=dtype)   # sequential ⇒ phơi small-addend underflow
```

**Hình ảnh — hai TB, và trade recompute:**
```
  STATE (đã chẻ 5.2-5.3)            ACTIVATION (TB thứ hai, 5.4)
  100B mixed = 2.0 TB               seq4k, 100B shape:
  ZeRO3 W=64 → 31 GB/rank           vanilla  [████████ 8.14 GB/layer] ×80 = 651 GB   ← 5·a·s²·b thống trị 82%
                                    flash    [██ 1.43 GB/layer]       ×80 = 114 GB   ← bỏ term bậc hai
                                    +ckpt    [▏ 84 MB residual/layer] ×80 = 6.7 GB   ← chỉ giữ 2·s·b·d, recompute nội thất
  recompute trade (compute axis):  full recompute cả block (+1 forward)  ·  selective giữ matmul-out, recompute softmax
```
Hand-trace ulp: fp16 mantissa 10 bit. Ở `[32,64)` số mũ cho spacing `2^(5−10) = 2⁻⁵ = 0.03125`. `32.0 + 0.01`
= `32.01`, làm tròn về bội gần nhất của `0.03125` quanh 32 → `32.0` (vì `0.01 < 0.0156`). Kẹt vĩnh viễn.

**Số đo THẬT** (canonical L=80, d_model=10240, n_heads=80, seq=4096, b=1):
```
vanilla act/layer = 8,136,949,760 B = 8.14 GB   ×80 = 650,955,980,800 = 651 GB
flash   act/layer = 1,426,063,360 B = 1.43 GB   ×80 = 114,085,068,800 = 114 GB   (bỏ 82% = term bậc hai)
ckpt residual/layer = 83,886,080 B = 83.9 MB    ×80 = 6.71 GB      (cái checkpointing giữ)
seq 4096→8192:  vanilla ×3.65 (→×4, s² thống trị)   flash ×2.00 (linear s)  ← vì sao flash bắt buộc long-ctx
mixed state = 20 B/param (16→20) : state TO HƠN, không nhỏ
fp16 Σ(0.01 ×10000) = 32.0000 (đóng băng)   |   fp32 = 100.00 (đúng)   ← vì sao reduction/optim ở fp32
checkpointing saved-activation bytes: none=212992 , selective=16384 , full=16384  (full<sel<none, 13× cắt)
matmul executions (fwd+bwd): none=6 , selective=6 , full=7  ← "full" recompute → +1 matmul (compute axis)
```
> Chú ý gap đo được: `full ≈ selective` trên **byte axis** (SAC cache matmul-out *bên trong* checkpoint region,
> vô hình với outer `saved_tensors_hooks`); khác biệt selective-vs-full hiện trên **compute axis** (matmul
> exec 6 vs 7). Đúng như docstring `count_saved_activation_bytes` :81 cảnh báo.

**Frontier / cổng.** SAC (selective) = default Megatron/TorchTitan 2026 (Korthikanti 2022). **Interview:**
"train fp16 loss plateau/NaN — vì sao?" → accumulation precision (fp16 nuốt small-addend) + autocast policy
(matmul bf16, norm/softmax/reduction fp32) + FP8 amax/scale overflow một tầng thấp hơn. Đây cũng là gốc cơ chế
**train-vs-serve logit drift** (`utils/monitors.py` kl_train_infer): hai engine reduce softmax/PV ở precision
khác → logit khác trên cùng weight. Trait = **claims-honesty** (mixed "tiết kiệm memory" là *sai* cho state).
Flash + checkpointing *compose*: flash shrink cái recomputed-block chạm, checkpointing quyết bao nhiêu block
live cùng lúc. Scarce-2026: kernels/precision = differentiator.

---

## 5.5 · Comms algebra — collective nào tốn gì, map lên interconnect, khi nào scaling dừng

**Câu hỏi.** Có `B` token/step, model rộng `D_ff`, `C` FLOP/s/chip, `W` byte/s egress — *closed form* nào nói
được scale tới bao nhiêu GPU trước khi network thành bottleneck, cho từng chiến lược?

**Sự thật nền tảng.** Câu hỏi kỹ thuật **đắt nhất** của distributed: "thêm GPU tới khi nào ngừng giúp?" — trả
*trước khi thuê một card*, bằng closed form. Model: N device, mỗi cái egress `W` byte/s + `C` FLOP/s; comm và
compute *overlap*, nên một step **comms-bound** khi thời-gian-wire per-device vượt thời-gian-compute per-device.
Mọi collective quy về ring primitive; mỗi chiến lược đụng tường trên *một trục khác*: DP trên **batch**, TP
trên **width**, 2D nhân hai trần đó.

**Dẫn xuất.**

*Bảng ring* (từ 5.1): `all_gather = reduce_scatter = (N−1)/N·S`; `all_reduce = 2(N−1)/N·S → 2S` khi N→∞.

*Per-scheme wire/step:*
```
DDP   = 1 all-reduce grad         = 2(N−1)/N·P·b        (ddp_step_bytes :125)
ZeRO1 = reduce-scatter + all-gather = 2(N−1)/N·P·b       (zero1_step_bytes :134)  == DDP, comms-free
FSDP  = 2·all-gather + reduce-scatter = 3(N−1)/N·P·b     (fsdp_step_bytes :146)   = 3/2× DDP
TP    = 4·L all-reduce ACTIVATION (không weight!)        (tp_step_bytes :159)     ∝ B·s·D, trên critical path
```

*Scaling bound derive (large-N, `(N−1)/N→1`):*
- **DP:** comm = all-reduce của fp16 grad `≈ 12·D·D_ff/W`; compute = `12·B·D·D_ff/(N·C)`. `comm ≤ compute` ⇔
  **`N ≤ B·W/C`** (`dp_max_world` :214). `D, D_ff` **triệt tiêu** → trần DP thuần = tokens/step × ratio W/C.
- **FSDP:** backward comm `≈ 12·D·D_ff/W` vs compute `12·B·D·D_ff/(N·C)` → **cùng `B·W/C`**; forward tương tự.
  FSDP mua `W×` memory ở `1.5×` wire nhưng *trần scaling BẤT ĐỘNG* → **memory là lý do chọn FSDP, không
  throughput** (`fsdp_max_world` :226).
- **TP fwd:** comm = một all-reduce `(B,D)` `≈ 4BD/W` vs compute `6BDD_ff/(NC)` ⇒ **`N ≤ (3/2)·D_ff·W/C`**
  (`tp_max_world_fwd` :236). `B, D` triệt tiêu → trần set bởi **width** → TP `~8` trong node bất kể job.
- **2D overlapped:** hai trục operate trên tensor đã sharded bởi trục kia → comm = **max** hai nhánh → mỗi
  nhánh cho một bound độc lập, *nhân nhau*: **`N ≤ (3/2)·B·D_ff·(W/C)²`** (`fsdp_tp_max_world` :258).
  Sequential = **sum** → minimize over split → **`(3/8)·B·D_ff·(W/C)²`** = đúng `1/4` của overlapped.

*Solver.* `comms_bound_world_size` :281 = exponential-bracket + bisection (giả định wire nondecreasing theo N
→ gap monotone), tìm world size nhỏ nhất `comm > compute`. `crossover_link_bandwidth` :320 = inverse: giữ N
rank compute-bound cần bao nhiêu egress.

**Neo code** (`utils/comms_calc.py` — mọi hàm pure closed form, derivation ở docstring, `A2_COMMS_ALGEBRA.md`
là output *generated*, test lock):
```python
def ring_allreduce_bytes(size_bytes, world):                         # :57
    return reduce_scatter_bytes(size_bytes, world) + all_gather_bytes(size_bytes, world)  # 2(N−1)/N·S
def dp_max_world(tokens, link_bw, gpu_flops):                        # :214
    return tokens * link_bw / gpu_flops                              # N ≤ B·W/C  (D, D_ff triệt tiêu)
def tp_max_world_fwd(d_ff, link_bw, gpu_flops):                      # :236
    return 1.5 * d_ff * link_bw / gpu_flops                          # N ≤ (3/2)·D_ff·W/C  (B, D triệt tiêu)
def fsdp_tp_max_world(tokens, d_ff, link_bw, gpu_flops, overlapped=True):  # :258
    coeff = 1.5 if overlapped else 3.0/8.0
    return coeff * tokens * d_ff * (link_bw / gpu_flops) ** 2        # 2D = (3/2)·B·D_ff·(W/C)²  (bình phương!)
```

**Hình ảnh — ba trần, ba trục (xl: D=2560, D_ff=10240, B=65536, C=1 PFLOP/s, W=400 GB/s):**
```
   N_max │
    161 ─┤ ● 2D-overlapped (3/2)·B·D_ff·(W/C)²  ← nhân hai trần, W vào BÌNH PHƯƠNG
         │ │
     40 ─┤ ● 2D-sequential  = 1/4 của overlapped
     26 ─┤ ● DP / FSDP  N ≤ B·W/C  (trục BATCH)
      6 ─┤ ● TP-fwd  N ≤ (3/2)D_ff·W/C  (trục WIDTH → ~8 trong node)
         └────────────────────────────────
   cliff W/C: rớt NVLink 400 → IB 50 GB/s (×1/8):  DP ∝W → ×1/8 (26→3.3)   2D ∝(W/C)² → ×1/64 (161→2.5)
   ⇒ 2D/TP PHẢI ở trong NVLink island; DP/FSDP mới cưỡi được IB inter-node
```

**Số đo THẬT** (xl worked example, pin bởi `test_comms_calc.py`):
```
DP-max      = 26.2   (closed B·W/C ;  solver comms_bound_world_size = 28, khớp brute-force)
FSDP-max    = 26.2   (== DP: extra comm khớp extra FLOP mỗi pass → trần bất động)
TP-fwd-max  =  6.1   ((3/2)·D_ff·W/C, width-bound)
2D overlap  = 161.1  ·  2D sequential = 40.3   (đúng 1/4: 161.1/40.3 = 4.0×)
wire/step xl N=8:  DDP = ZeRO1 = 11.744 GB  <  FSDP = 17.616 GB  (3/2)
crossover_link_bw giữ 64 DP rank compute-bound = 961 GB/s  (vượt mọi single-NIC ⇒ chính là trần DP)
NVLink→IB cliff (×1/8 W): DP 26.2→3.3 (×1/8)  ·  2D 161.1→2.52 (×1/64)   ← [PREDICTED] rental-gated (busbw thật)
```

**Frontier / cổng.** DualPipe / zero-bubble PP (DeepSeek-V3 **2412.19437**, zero-bubble **2401.10241**):
reorder fwd/bwd micro-batch để overlap all-to-all EP sau compute, ăn gần hết bubble (S6 Bài 6.2 mổ 1F1B
bubble `(p−1)/m`). 4D/5D parallelism + EP all-to-all (Megatron/TorchTitan **2410.06511**): compose
DP+TP+SP+PP+EP trên device-mesh; MoE dùng *all-to-all* (mỗi token routing tới expert trên rank khác) không
all-reduce (S6 Bài 6.3). **Interview** (chính docstring): "B token/step, D_ff-wide, C FLOP/s, W byte/s — mấy
chip trước khi network bottleneck, xoay knob nào?" → DP chết `B·W/C`, TP chết `(3/2)D_ff·W/C`, 2D =
`(3/2)B·D_ff·(W/C)²`; quá đó chỉ interconnect nhanh hơn hoặc critical-batch to hơn cứu. Trait =
**roofline-first / predict-the-number**. **Honesty:** busbw line-rate + cliff NVLink→IB `~18×` là
**rental-gated** (chưa measured); closed form + gloo-correctness là cái measured NOW.

---

## Bảng số đo THẬT (chạy lại được — DoD-là-profile)

| Concept | Đo | Kết quả | Ý nghĩa |
|---|---|---|---|
| 5.1 | ring all-reduce/dev, N→∞ | 1.0 → 1.75 → 1.969 → **2S bounded** | vì sao DP scale |
| 5.1 | ring/naive N=8 | **0.2500 = 2/W** | chunking thắng |
| 5.1 | flat vs 200 tensor riêng | 0.364 ms vs 3.150 ms → **8.7×** | latency floor |
| 5.1 | DDP gloo == single-proc (3 mode) | **1.49e-08** PASS | identity đúng |
| 5.2 | state/param fp32 vs mixed | 16 → **20 B/param** (state TO hơn) | 100B = 1.6→2.0 TB |
| 5.2 | 100B W=64 per-rank | 2000→425→228→**31.25 GB** | chỉ ZeRO-3 fit 80 |
| 5.2 | gpus_needed 100B/80GB | ZeRO3 **25** ; ZeRO1/2 **RAISE** | replicated bucket > HBM |
| 5.2 | partition greedy | [100,50,50,25,25]→**[0,1,1,0,1]** (125/125) | least-loaded balance |
| 5.2 | ZeRO1/DDP wire | **1.0000** (comms-free) | RS+AG = ring all-reduce |
| 5.3 | FSDP classify | Linear w SHARD, bias REPLICATE | ndim≥2 policy |
| 5.3 | FSDP/DDP wire ratio | **1.5000** mọi N (2/8/64) | 3 leg vs 2 |
| 5.3 | gloo suite | **15/15 PASS** | correctness measured |
| 5.4 | vanilla act/layer×80 | 8.14 GB → **651 GB** | TB thứ hai |
| 5.4 | flash / ckpt | 114 GB / **6.71 GB** (bỏ 82%) | s² term / residual only |
| 5.4 | seq 4k→8k | vanilla **×3.65** flash **×2.00** | vì sao flash long-ctx |
| 5.4 | fp16 Σ(0.01×10⁴) | **32.0** đóng băng (fp32=100) | reduction fp32 |
| 5.4 | ckpt saved bytes | none 212992 → **16384** (13×) | recompute trade |
| 5.5 | xl DP / FSDP-max | **26.2 == 26.2** (solver 28) | trần B·W/C, FSDP không dời |
| 5.5 | xl TP-fwd / 2D | 6.1 / **161.1** (seq 40.3 = 1/4) | width vs (W/C)² |
| 5.5 | crossover_link_bw @64 | **961 GB/s** | > single-NIC = trần DP |

---

## Checklist recall COLD (che phần trên, tự trả lời — đây là cách re-learn)

1. **5.1** Dẫn `avg_ranks[per-rank mean grad] = full-batch grad` — điều kiện gì trên loss khiến nó EXACT?
   Quên `/world_size` với N=4 → loss/step lệch thế nào, test nào bắt?
2. **5.1** Một all-reduce chuyển bao nhiêu byte/device? Vì sao bounded khi N→∞? ring/naive ratio =?
3. **5.2** 100B AdamW: memory đi đâu (16 vs 20 B/param)? ZeRO-1/2/3 chẻ *cái gì*? W=64 per-rank mỗi stage?
4. **5.2** Vì sao ZeRO-1 tiết kiệm optimizer-memory mà *không* tốn thêm byte-on-wire so DDP? (một câu)
5. **5.2** Vì sao ZeRO-1 và ZeRO-2 fail 100B/80GB ở *mọi* world size, chỉ ZeRO-3 fit?
6. **5.3** Vì sao *không* shard norm/bias — hai lý do (memory + correctness)? Nếu shard RMSNorm → test nào fail?
7. **5.3** FSDP forward gọi collective gì, grad-sync gọi gì (sharded vs replicated)? Wire ratio FSDP/DDP =?
   Gap: container `utils/fsdp.py` thật sự chuyển 2 leg hay 3 leg, vì sao?
8. **5.4** Activation formula? `flash` bỏ term nào? seq gấp đôi → vanilla ×? flash ×? Vì sao?
9. **5.4** Mixed-precision "tiết kiệm memory" — đúng/sai, cái gì nó grow vs shrink? fp16 Σ(0.01) đóng băng ở đâu?
10. **5.5** DP/TP/2D chết ở world size nào (closed form)? Vì sao FSDP cùng trần DP dù chuyển 1.5× byte?
    Rớt NVLink→IB (×1/8 W): DP-max và 2D-max đổi mấy lần mỗi cái?

> Trả lời COLD được cả 10 = **M5 thật sự OWNED** (interview-grade). Vấp câu nào → mở đúng mục, hoặc blank-slate
> hàm tương ứng (`raise NotImplementedError` → test đỏ → tự dẫn → xanh) để re-own bằng tay.

---

*Cross-ref: `roadmap_model/M5_distributed_training.md` (spine chung) · `roadmap/S6_distributed_and_isa.md`
(TP/PP/EP/WGMMA serving-side, twin của M5) · `PROGRESS.md` (ledger 89 Bài) · sibling derivations
`M2_transformer_forward.md`, `M3_optimization.md` · design docs `A2_100B_MEMORY_ONEPAGER.md`,
`A2_COMMS_ALGEBRA.md`. Honesty (FOP-4): correctness gloo-MEASURED, memory/comms arithmetic-exact pinned;
real NCCL busbw / MFU / NVLink→IB cliff = **[PREDICTED] rental-gated** (8×H200 day).*
