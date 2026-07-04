# Série M5 — Distributed training (systems, derived) — nửa A2 của CS336

> **Series M5 · Distributed training** · số dòng pin theo commit **`4ad0ac5`**
> Zone: `src/scratch_llm/utils/{ddp,zero1,fsdp,comms_calc,memory_math,checkpointing,mixed_precision}.py`
> + `docs/design/A2_100B_MEMORY_ONEPAGER.md` + `docs/design/A2_COMMS_ALGEBRA.md`
> Số đo gốc: `bench/RESULTS.md` §"Perf track (A6…)" (gloo/CPU correctness) + các test-invariant
> `tests/test_{ddp,zero1,fsdp,memory_math,comms_calc,checkpointing,mixed_precision}.py`.
> Đây là **TWIN train-side** của perf-roadmap `docs/learning/roadmap/S6_distributed_and_isa.md`:
> S6 mổ *serving-parallelism primitives* (TP MLP · 1F1B · EP-MoE · MFU · WGMMA/FA3/tcgen05); M5 mổ
> *training data-parallel derivation* (DDP · ZeRO · FSDP · memory · comms algebra). **Không lặp** — mỗi
> khi chạm TP/PP/EP ta cross-link sang S6 thay vì tự giải lại.

**Vì sao série này.** M1–M4 dựng *một* model chạy trên *một* GPU: byte→BPE→Transformer→optim→train
loop→post-training. Nhưng một model frontier (100B param) *không nhét nổi* một card — 1.6–2.0 TB state
trước cả activation, gấp 20–25× một H100-80GB. Série này derive cách **chẻ một training step ra nhiều
GPU** mà vẫn giữ đúng **một identity đại số**: với mean-loss và batch chia đều, gradient full-batch =
trung bình gradient per-rank, nên mọi replica step *giống hệt* một single-process chạy trên cả batch.
Từ identity đó mọc ra cả ladder — DDP (chẻ data) → ZeRO (chẻ state) → FSDP (chẻ cả param) — và một câu
hỏi kỹ thuật *xuyên suốt cả M5 lẫn S6*: **collective nào chạy trên link nào**. Sợi chỉ: *map the
chattiest collective onto the fattest link, và biết trước — bằng closed form — khi thêm GPU ngừng giúp.*

Thứ tự học (mỗi Bài tựa Bài trước): DDP + ring all-reduce (nguyên tử comm) → ZeRO (chẻ state, dùng chính
ring đó) → FSDP (chẻ param, all-gather/reduce-scatter = hai nửa của ring) → memory + recompute (100B
one-pager: state vs activation) → comms algebra (đóng closed-form: khi nào scaling dừng).

---

## Bài 5.1 — Data parallel + ring all-reduce: busbw = algbw·2(N−1)/N, naive→flat→overlap (`utils/ddp.py` · `DDP.finish_gradient_synchronization` :60)

> **Câu hỏi first-principles:** vì sao trung bình các gradient per-rank *bằng đúng* gradient full-batch
> (không xấp xỉ), và một all-reduce chuyển **bao nhiêu byte per-device** — bounded hay tăng theo N?
> **Neo (invariant / số đo):** MEASURED — với **cả ba mode** (naive/flat/overlap), param DDP 2-rank gloo
> == single-process full-batch tới fp tolerance qua nhiều optimizer step (`tests/test_ddp.py`, gloo ×5,
> RESULTS.md §A6). Analytic: ring all-reduce chuyển **2(N−1)/N·S** byte/device → **bounded 2S khi N→∞**.

**1. Feynman — bài toán bằng lời.** N công nhân, mỗi người nhận một *phần* batch (data parallel: chẻ
data, replicate model). Ai cũng tính gradient trên phần của mình rồi *phải đồng ý* trước khi bước — nếu
không, các replica trôi ra xa nhau và bạn có N model khác nhau thay vì một. "Đồng ý" = all-reduce: cộng
gradient của mọi rank rồi chia N → mọi người có cùng mean gradient → step giống hệt. Đánh đổi cốt lõi:
DDP cho throughput (N× tokens/step) *miễn phí về capacity* — nhưng mỗi step phải trả một all-reduce toàn
bộ gradient, và câu hỏi sống-còn là all-reduce đó tốn *bao nhiêu byte* và có giấu được dưới backward không.

**2. Dẫn xuất từ đầu (derive).** *Correctness identity:* loss là mean over B example. ∇(1/B·Σᵢ lᵢ) =
(1/B)Σᵢ ∇lᵢ. Chẻ B thành N shard bằng nhau (B/N mỗi rank); rank r tính grad-mean-local = (N/B)Σ_shard ∇lᵢ.
Trung bình N cái đó: (1/N)Σ_r (N/B)Σ_shard ∇lᵢ = (1/B)Σ_all ∇lᵢ = **gradient full-batch, exact**. Hai
điều kiện: (a) mọi replica khởi từ *cùng weight* (broadcast rank 0 lúc init), (b) grad được **all-reduce
SUM rồi chia N** trước step. *Busbw derive:* ring all-reduce = reduce-scatter + all-gather. Mỗi device
giữ chunk S/N; ring có N−1 bước, mỗi bước forward một chunk S/N cho hàng xóm → mỗi phase gửi (N−1)/N·S
byte/device → all-reduce = **2(N−1)/N·S**. Đây LÀ định nghĩa bus-bandwidth: busbw = algbw·2(N−1)/N (algbw
= S/time là "thuật toán thấy", busbw là "link thật cõng"). Khi N→∞ → 2S — *bounded, độc lập N* — đó chính
là **lý do data parallel scale được**. Ladder ba rung bóc dần idle byte-movement, **cùng numerics**:
- `"naive"` :79 — all-reduce từng grad tensor riêng (một collective/param) → phải trả latency floor 2(N−1)·α
  cho *mỗi* tensor.
- `"flat"` :69 — concat mọi grad vào một buffer → **một** all-reduce → amortize latency per-call.
- `"overlap"` :47 — bắn all-reduce từ `register_post_accumulate_grad_hook` ngay khi mỗi grad sẵn sàng, để
  comm giấu dưới backward *đang chạy*. Đây là production DDP + graded deliverable.

**3. Trace code.** `DDP.__init__` :35 → broadcast param + buffer từ src=0 :41-44 (điều kiện (a)); nếu
overlap, đăng ký hook `_async_all_reduce` cho mọi param requires_grad :48-50. Backward chạy → hook :52 bắn
`dist.all_reduce(p.grad, SUM, async_op=True)` :54, cất `(handle,p)` :55. Sau `loss.backward()`,
`finish_gradient_synchronization` :60: overlap → `handle.wait()` + `p.grad /= world_size` :64-67 (chia N =
điều kiện (b)); flat → cat mọi grad :71 → một all-reduce :72 → chia + copy ngược :74-77; naive → loop
all-reduce từng grad :80-83. Test pin: bất kỳ drift ⇒ thiếu init-broadcast, quên /world_size, hoặc grad bị
tiêu thụ trước khi async all-reduce xong.

**4. Cổng teach-back.** (a) Vì sao trung bình gradient per-rank *bằng đúng* (không xấp xỉ) gradient
full-batch — điều kiện gì trên loss khiến nó exact? (một câu: mean-loss + batch chia đều → tuyến tính của ∇).
(b) **Sửa-và-đoán:** nếu quên `p.grad /= world_size` trong overlap mode với N=4 — loss/step lệch thế nào,
và test nào bắt? (đáp: gradient bị SUM chứ không mean → lớn gấp 4× → step gấp 4 lần learning-rate hiệu dụng
→ trajectory trôi ngay step 1; test_ddp equivalence FAIL).

**5. Frontier.** DDP cưỡi *inter-node link* (IB, băng thông rẻ) được vì comm của nó — gradient all-reduce —
*giấu được* dưới backward (overlap hook). Đối lập TP (roadmap/S6 Bài 6.1): all-reduce activation của TP nằm
*trên critical path*, không giấu được → phải cưỡi NVLink. Interview: "một DDP step chuyển bao nhiêu byte
per-device, và tại sao thêm GPU không tăng byte đó?" — 2(N−1)/N·S ≈ 2 bản gradient, bounded 2S; thêm GPU
chỉ chia *compute*, không chia wire volume (đó là trần DP ở Bài 5.5). Ref: ZeRO paper (1910.02054) §baseline.

---

## Bài 5.2 — ZeRO-1/2/3: mô hình 16-psi + sharding optimizer/grad/param (`utils/zero1.py` · `ShardedOptimizer.step` :102; `utils/memory_math.py` · `zero_shard_bytes` :169)

> **Câu hỏi first-principles:** DDP replicate *toàn bộ* 16 B/param state trên mọi rank — cái nào *bắt buộc*
> phải replicated, cái nào chẻ được, và chẻ nó có tốn thêm byte-on-wire không?
> **Neo (invariant / số đo):** MEASURED — sharded-optimizer trajectory == unsharded qua nhiều step/seed,
> **per-rank state numel ≈ total/W** trong bound greedy-partition (`tests/test_zero1.py`, gloo). Arithmetic
> (pinned `test_memory_math.py`): ở W=64, per-rank state **2000→425→228→31.25 GB** (stage 0→1→2→3);
> ZeRO-1 cắt optimizer bucket **đúng 64×**, ZeRO-3 chia cả 2.0 TB đúng cho W.

**1. Feynman — bài toán bằng lời.** AdamW mang *bốn* fp32 buffer mỗi param: weight 4 + grad 4 + moment m 4
+ variance v 4 = **16 B/param** ("16-psi" — psi = số param). Ở 100B param đó là 1.6 TB state *trước cả một
activation*. DDP thảm hoạ: replicate cả 16 B/param trên *mọi* rank → thêm GPU tăng throughput, không bao
giờ tăng capacity. Quan sát ZeRO: update Adam là *elementwise* — param i chỉ cần state của chính param i.
Nên *phân vùng* param: mỗi rank *sở hữu* ~1/W param, giữ optimizer state chỉ cho shard của mình, step
shard đó, rồi **broadcast** param đã update từ chủ sở hữu. Cuối step mọi rank vẫn cầm full model giống hệt;
chỉ *bộ nhớ* m/v chia cho W. Đánh đổi tuyệt vời: memory ÷W gần như **miễn phí về comm**.

**2. Dẫn xuất từ đầu (derive).** *Ladder ZeRO chẻ theo ranh giới bucket tự nhiên* (`_zero_buckets` :123):
- **ZeRO-1** chẻ *optimizer bucket* (master+fp32-grad+m+v); weight + backward-grad *vẫn replicated* (grad
  phải replicated để all-reduce). fp32: bucket = 2·4 = 8 B/param (m+v) → 8/W.
- **ZeRO-2** chẻ thêm *grad* (reduce-scatter thay all-reduce → mỗi rank chỉ giữ grad cho shard mình).
- **ZeRO-3 / FSDP** chẻ thêm *weight* → mọi thứ ~1/W, giá là all-gather weight on-demand mỗi fwd/bwd (Bài 5.3).

*Vì sao ZeRO-1 comms-free:* DDP trả *một ring all-reduce* = reduce-scatter + all-gather (Bài 5.1). ZeRO-1:
mỗi rank chỉ cần summed-grad cho optimizer-shard của nó → **reduce-scatter** ((N−1)/N·P·b); rồi republish
param-shard đã update → **all-gather** ((N−1)/N·P·b). *Đúng bằng hai nửa của ring all-reduce* → cùng wire
volume DDP. Nên ZeRO-1 = memory ÷W ở comm cost *bằng* DDP (`zero1_step_bytes` :134, RESULTS xl: cả hai
11.74 GB/step). *100B arithmetic* (`zero_shard_bytes`): mixed-precision W=64 → stage 0 = 2000 GB (không
fit), stage 1 = 425 GB (không), stage 2 = 228 GB (không), stage 3 = **31.25 GB (fit 80 GB)**. Corollary
tested: cho 100B trên 80 GB, ZeRO-1 và ZeRO-2 fail ở *mọi* world size — replicated bucket một mình đã vượt
HBM; chỉ ZeRO-3 xuống dưới. Floor state-only = **25 GPU** (mixed) / 20 (fp32) (`gpus_needed` :185).

**3. Trace code.** `ShardedOptimizer.__init__` :66: `partition_by_numel` :31 gán mỗi param cho rank
*least-loaded* (greedy, tie→rank thấp; max−min load ≤ max(numel)) → `self._owner` :82; build inner optimizer
chỉ cho param *của rank mình* :84-92. `step` :102 → inner step chỉ shard local :108-109 → nếu W>1, broadcast
mỗi param `src=self._owner[p]` async :112-115, wait tất cả :116-118 (mọi rank lại đồng bộ full model).
`state_numel_on_rank` :98 = số đo ZeRO-1 memory claim (≈ total/W). `zero_shard_breakdown` :136 (memory_math)
= per-rank byte theo 3 bucket, sharded bucket charge `ceil(N/W)` param :161.

**4. Cổng teach-back.** (a) Vì sao ZeRO-1 tiết kiệm optimizer-memory mà *không* tốn thêm byte-on-wire so với
DDP? (một câu: reduce-scatter+all-gather = đúng hai nửa của cùng ring all-reduce DDP vốn đã trả). (b)
**Sửa-và-đoán:** nếu partition đặt *tất cả* param lên rank 0 (thay greedy) — per-rank state của rank 0 và
rank 1 là bao nhiêu, correctness còn giữ không? (đáp: rank0 = full 16 B/param, rank1..W = 0 → memory không
tiết kiệm nữa nhưng *trajectory vẫn đúng* — partition chỉ đổi memory, không đổi số học; test equivalence vẫn
PASS, test state-numel≈total/W FAIL).

**5. Frontier.** ZeRO stages = Rajbhandari et al. **1910.02054** (2019), abstraction chuẩn của DeepSpeed.
Interview kinh điển (chính docstring zero1.py): "100B param, AdamW — memory đi đâu, ZeRO-1 mua gì ở world
size W?" → 1.6 TB @ 16 B/param; 8 B/param moment → 8/W, ở *near-DDP comm cost*. Nối tiếp Bài 5.3: ZeRO-3
là chỗ nó thành FSDP thật.

---

## Bài 5.3 — FSDP (ZeRO-3): flat-param all-gather/reduce-scatter + shard-matrices/replicate-norms (`utils/fsdp.py` · `FSDP.finish_gradient_synchronization` :145)

> **Câu hỏi first-principles:** chẻ *param* (không chỉ state) ra W rank — full model chỉ tồn tại *thoáng
> qua* — thì fwd/bwd phải gọi collective gì, và vì sao norm/bias *không* nên shard?
> **Neo (invariant / số đo):** MEASURED — sharded run (loss + full grad reassembled + trajectory) ==
> single-process full-batch tới fp32 tol, 2-rank gloo **3 seed × ≥3 step**, MLP *và* TransformerLM
> (`tests/test_fsdp.py`); official `test_fsdp_gradient_sync` bắt grad của mọi non-Linear param **bit-identical
> across ranks**. Analytic: FSDP:DDP wire ratio = **3/2 exact** ở mọi N.

**1. Feynman — bài toán bằng lời.** DDP giữ full model mọi rank. FSDP: mỗi rank chỉ giữ *một lát 1/W* của
mỗi ma trận (fp32 master shard). Khi `forward()` cần một layer, nó **all-gather** các shard lại thành full
weight *tạm thời*, tính, rồi (sau backward) **reduce-scatter** full gradient để mỗi rank giữ đúng grad của
shard mình. Full model không bao giờ resident lâu — nó *materialize thoáng qua*. Nhưng *không phải mọi param
đều shard*: ma trận lớn (ndim≥2: Linear/Embedding) shard; norm/bias (ndim≤1) **replicate** — shard một norm
64-element mua gần-zero memory mà vẫn trả full comm. Đánh đổi: đổi memory (S→S/W resident param) lấy *một
all-gather thừa* mỗi step.

**2. Dẫn xuất từ đầu (derive).** *Vì sao replicate 1-D param là bắt buộc, không cosmetic:* official test
assert grad của mọi non-Linear param **bit-identical across ranks**. Một reduce-scattered shard-grad, theo
construction, *khác nhau per-rank* (mỗi rank sở hữu lát khác) → fail check đó. Path replicate-and-all-reduce
là cái làm norm/bias grad khớp. Phân loại theo `ndim` trùng khít predicate "parent không phải Linear/Embedding"
(mọi matrix 2-D, mọi norm 1-D) — và là convention ZeRO-3 chung. *Wire cost derive* (`fsdp_step_bytes` :146):
param sharded → mỗi step all-gather full weight cho forward ((N−1)/N·P·b), all-gather lại cho backward (đã
free sau dùng), reduce-scatter grad về chủ shard ((N−1)/N·P·b) → **3 one-way pass** vs DDP 2 → ratio đúng
**3/2** ở mọi N (RESULTS xl: 17.62 vs 11.74 GB). *Dtype invariant:* `compute_dtype` chọn dtype của bản compute
*thoáng qua* (gathered shard / cast full copy); fp32 master + grad reduction *luôn* fp32 → sau sync mọi
`.grad` là fp32 shape-match `.data` (shard-shaped cho sharded, full cho replicated).

**3. Trace code.** `FSDP.__init__` :97: broadcast rank0 :104-107; phân loại mỗi param :110-124 — ndim≥2 →
flatten, pad tới bội của W, giữ lát rank mình `p.data = master` :113-121 (`_Shard`); ndim≤1 → `_Replica`
giữ full :122-124. `forward` :133: mỗi shard `_all_gather_full` :126 → `p.data` = full copy :139-140; mỗi
replica cast :141-142 → chạy module (full copy resident tới backward). `finish_gradient_synchronization`
:145: shard → `reduce_scatter_tensor(SUM)` :160 → `/world_size` :161 → `p.data = master` (shard-shaped) rồi
gán shard_grad :162-164; replica → `all_reduce(SUM)/W` :171-172 → restore master :174-175. `resident_param_numel`
:205 = số đo footprint (dedup theo data_ptr): giữa step ≈ sharded/W + replicated full; trong fwd/bwd = full
+ shard.

**4. Cổng teach-back.** (a) Vì sao *không* shard norm/bias — hai lý do (memory + correctness)? (một câu:
shard 64-element mua ~0 memory nhưng vẫn full comm, VÀ reduce-scattered shard-grad khác per-rank → fail
test bit-identical). (b) **Sửa-và-đoán:** nếu classify RMSNorm weight (1-D) là SHARDED thay REPLICATED —
`test_fsdp_gradient_sync` fail hay pass, và grad của norm khác nhau thế nào giữa rank? (đáp: FAIL; mỗi rank
giữ reduce-scattered lát khác → grad không bit-identical across ranks, đúng cái test cấm).

**5. Frontier.** *FSDP1 vs FSDP2* (FRONTIER_PRACTICE_2026.md:255-256): impl này là FSDP1-style — flatten
per-param, gather-all-ở-forward-free-ở-sync (container tối giản, không per-block overlap). **FSDP2** =
PyTorch `fully_shard`: per-parameter **DTensor** sharding thay FlatParameter, ~7% lower per-GPU memory /
~1.5% higher throughput, và *compose được* với TP/PP (shard một weight hai chiều) — TorchTitan default
(**arXiv:2410.06511**). ADR-trigger đã ghi trong docstring :47: nếu gloo pass nhưng GPU bench thấy all-gather
không overlap compute → nâng wrap granularity (per-param → per-block). Interview: "DDP all-reduce chuyển
~2(N−1)/N·S; FSDP chuyển gì, memory đi đâu?" → 3 leg (1.5×) để cắt resident param S→S/W; memory đổi lấy một
all-gather thừa. Ref gốc ZeRO-3: **1910.02054**. Cross-link: TP-side sharding một matmul ở roadmap/S6 Bài 6.1.

---

## Bài 5.4 — Activation checkpointing + mixed-precision memory: the 100B one-pager (`utils/checkpointing.py` · `run_block` :63; `utils/memory_math.py` · `activation_bytes_per_layer` :95)

> **Câu hỏi first-principles:** state đã chẻ xong (Bài 5.2-5.3) nhưng *activation* là TB thứ hai — recompute
> đổi được bao nhiêu memory lấy bao nhiêu FLOP, và vì sao mixed-precision làm state *to hơn* chứ không nhỏ?
> **Neo (invariant / số đo):** MEASURED (CPU) — grad dưới `"full"`/`"selective"` == eager `"none"` tới fp
> tol; saved-activation bytes ordered **full < selective < none** (`tests/test_checkpointing.py`). Arithmetic
> (pinned): seq-4096 100B shape, vanilla **8.14 GB/layer → ×80 = 651 GB**; flash bỏ đúng term bậc hai →
> **114 GB**; checkpointing giữ chỉ residual **~84 MB/layer → 6.7 GB** stack.

**1. Feynman — bài toán bằng lời.** Sharding cứu *state*; nhưng forward lưu *activation* để backward dùng,
và ở seq 4k một block 100B lưu ~8.1 GB — ×80 layer = 651 GB, một TB thứ hai còn to hơn cả state ZeRO-3 chừa
lại (31 GB/rank). Hai đòn bẩy: (1) **FlashAttention** không bao giờ materialize hai ma trận score (seq×seq)
→ bỏ đúng term bậc hai, recompute tile-wise trong backward. (2) **Gradient checkpointing**: lưu *chỉ input
residual của mỗi block*, recompute nội thất block trong backward → activation memory từ O(depth·per_block)
xuống O(depth·residual), giá ≈ một forward thừa. *Song song đó* mixed-precision: matmul chạy bf16 (throughput
tensor-core) nhưng master weight + m + v + reduction *ở lại fp32* → state **to hơn** (18–20 B/param), không nhỏ.

**2. Dẫn xuất từ đầu (derive).** *Activation formula* (Korthikanti 2022, `activation_bytes_per_layer` :95):
`s·b·h·34 + 5·a·s²·b` byte/layer. Term tuyến tính 34·s·b·h = mọi thứ tỉ lệ residual stream (LN input, QKV,
attn out, 8h của MLP, mask); term bậc hai 5·a·s²·b = hai ma trận attention materialized (score 2 + softmax 2
+ mask 1). `flash=True` bỏ **đúng** term bậc hai :111 — ở seq 4k đó là ~82% activation của layer, và nó lớn
theo s². *Mixed-precision state* (`training_state_breakdown` :48): fp32 = weight 4 + grad 4 + m 4 + v 4 = 16.
Mixed = fp32 master 4 + fp32 grad 4 + m 4 + v 4 + **bf16 weight compute copy 2** (+ bf16 grad 2 nếu không fuse
upcast) = 18–20. Mixed làm *matmul* rẻ + halve *activation* dtype — không shrink state, grow nó. *Recompute
trade* (`checkpointing.py`): "full" lưu chỉ block-input, recompute cả block; "selective" (SAC, default 2026)
lưu cái *store-cheap/recompute-expensive* = matmul output, recompute phần rẻ (softmax/elementwise) → ~70%
activation memory ở ~2.7% FLOP thừa. *Fp16-accumulation cạm bẫy* (`mixed_precision.py` :28): running sum fp16
đứng im khi increment < nửa-ulp (0.01 vào fp16 kẹt ở 32.0: ulp@32 = 2⁻⁵) → *vì sao* optimizer state/loss/
reduction accumulate fp32 dù compute bf16.

**3. Trace code.** `run_block` :63: "none" → block thẳng; "full" → `checkpoint(block, use_reentrant=False)`
:69; "selective" → checkpoint + `create_selective_checkpoint_contexts(_sac_policy)` :71-76. `_sac_policy`
:56: op ∈ `MATMUL_OPS` :43 → `MUST_SAVE`, else `PREFER_RECOMPUTE`. Số đo memory axis: `count_saved_activation_bytes`
:80 (saved_tensors_hooks, dedup storage-ptr) → full < selective < none; compute axis: `count_op_executions`
:120 (TorchDispatchMode đếm matmul; giữ *cả* fwd+bwd vì recompute ở backward). memory_math one-pager:
`residual_stream_bytes` :115 = 2·s·b·d ≈ 84 MB/layer (cái checkpointing giữ thay 34·s·b·h nội thất).

**4. Cổng teach-back.** (a) Mixed-precision "tiết kiệm memory" — đúng hay sai, và *cái gì* nó tiết kiệm vs
*cái gì* nó grow? (đáp: sai cho state — state 16→18-20 B/param; nó tiết kiệm activation-dtype + matmul-time,
không state). (b) **Sửa-và-đoán:** ở seq 8192 (gấp đôi 4096), vanilla activation/layer đổi bao nhiêu lần, và
sau flash thì đổi bao nhiêu? (đáp: vanilla ~×4 vì term bậc hai 5·a·s²·b thống trị và ∝s²; flash chỉ còn term
tuyến tính ∝s → ~×2. Đó *là* lý do flash bắt buộc ở long-context).

**5. Frontier.** SAC (selective) = default Megatron/TorchTitan 2026 (Korthikanti 2022). Interview: "train
fp16 loss plateau/NaN — vì sao?" → accumulation precision (fp16 nuốt small-addend) + autocast policy (matmul
bf16, norm/softmax/reduction fp32) + FP8 amax/scale overflow một tầng thấp hơn. Đây cũng là gốc cơ chế của
train-vs-serve logit drift (`utils/monitors.py` kl_train_infer): hai engine reduce softmax/PV ở precision
khác → logit khác trên cùng weight. Checkpointing + flash *compose*: flash shrink cái một recomputed-block
chạm, checkpointing quyết bao nhiêu block live cùng lúc; memory chừa ra tiêu vào micro-batch lớn hơn (cũng
giữ pipeline bubble mỏng — Bài 5.5 / S6 Bài 6.2).

---

## Bài 5.5 — Comms algebra: collective nào tốn gì, map lên interconnect, khi nào scaling dừng (`utils/comms_calc.py` · `ring_allreduce_bytes` :57, `fsdp_tp_max_world` :258)

> **Câu hỏi first-principles:** có B token/step, model rộng D_ff, C FLOP/s/chip, W byte/s egress — *closed
> form* nào nói được scale tới bao nhiêu GPU trước khi network thành bottleneck, cho từng chiến lược?
> **Neo (số đo / prediction):** pinned `test_comms_calc.py::test_xl_worked_example_regression` — xl model
> (D=2560, D_ff=10240, L=32, B=65536, C=1 PFLOP/s): DP-max **26.2** @400 GB/s (exact solver **28**), TP-fwd
> **6.1**, 2D-overlapped **161.1**; per-step wire DDP **11.74 GB** = ZeRO-1 (comms-free) < FSDP **17.62 GB**
> (1.5×). NVLink→IB cliff ~18× = **PREDICTION rental-gated** (8×H200 day), chưa measured.

**1. Feynman — bài toán bằng lời.** Câu hỏi kỹ thuật đắt nhất của distributed: "thêm GPU tới khi nào ngừng
giúp?" — trả *trước khi thuê một card*, bằng closed form. Model: N device, mỗi cái egress W byte/s + C FLOP/s;
comm và compute *overlap*, nên một step *comms-bound* khi thời-gian-wire per-device vượt thời-gian-compute
per-device. Mọi collective quy về ring primitive; mỗi chiến lược đụng tường trên *một trục khác*: DP trên
*batch* (chết ở N ≈ BW/C), TP trên *width* (chết ở (3/2)D_ff·W/C), 2D nhân hai trần đó. Đánh đổi cuối cùng
(closed form nói ra): với B ghim bởi critical-batch và D_ff ghim bởi scaling law, lever duy nhất còn lại là
tỉ số phần cứng W/C — và nó vào công thức *bình phương*.

**2. Dẫn xuất từ đầu (derive).** *Bảng ring* (`all_gather_bytes` :37 / `reduce_scatter_bytes` :47 /
`ring_allreduce_bytes` :57): all-gather = reduce-scatter = (N−1)/N·S; all-reduce = 2(N−1)/N·S → 2S khi N→∞.
*Per-scheme wire/step:* DDP = một all-reduce grad = 2(N−1)/N·P·b (`ddp_step_bytes` :125); ZeRO-1 = RS+AG =
*cùng* (comms-free, :134); FSDP = 2·AG + RS = 3/2× (:146); TP = 4·L all-reduce *activation* (không weight!)
:159 → scale với B·s·D, không param. *Scaling bound derive* (large-N, (N−1)/N→1): DP comm = all-reduce
grad ≈ 12·D·D_ff/W; compute = 12·B·D·D_ff/(N·C); comm≤compute ⇔ **N ≤ BW/C** (`dp_max_world` :214, D và D_ff
triệt tiêu → trần DP = token/step × W/C, thuần). FSDP = *cùng* BW/C (extra comm mỗi pass khớp FLOP pass đó —
memory là lý do chọn FSDP, không throughput). TP fwd = một all-reduce (B,D) ≈ 4BD/W vs 6BDD_ff/(NC) ⇒ **N ≤
(3/2)D_ff·W/C** (`tp_max_world_fwd` :236; B,D triệt tiêu → trần set bởi *width* → TP ~8 trong node bất kể job).
2D overlapped: hai trục operate trên tensor đã sharded bởi trục kia → comm = max hai nhánh → mỗi nhánh cho một
bound độc lập, *nhân nhau*: **N ≤ (3/2)·B·D_ff·(W/C)²** (`fsdp_tp_max_world` :258); sequential = sum → 1/4 số đó.

**3. Trace code.** Mọi hàm là pure closed form, derivation ở docstring, doc `A2_COMMS_ALGEBRA.md` là output
*generated* từ chúng (không gõ tay), test lock. `ring_allreduce_time` :89 = α-β model: 2(N−1)·hops·α +
2(N−1)/N·S/β — latency floor 2(N−1)·α *per tensor* là *vì sao* DDP flatten grad (Bài 5.1 "flat"). Solver
`comms_bound_world_size` :281 = exponential-bracket + bisection (giả định wire nondecreasing theo N → gap
monotone), tìm world size nhỏ nhất comm>compute; xl DP → 28 (closed 26.2 + slack (N−1)/N). `crossover_link_bandwidth`
:320 = inverse: giữ 64 DP rank compute-bound cần **~961 GB/s** egress (vượt mọi single-NIC → chính là fact
trần DP). Worked xl: 2D-overlapped 161.1 vs sequential 40.3 (đúng ×4).

**4. Cổng teach-back.** (a) Vì sao FSDP có *cùng* scaling ceiling BW/C như DP dù chuyển 1.5× byte — memory hay
throughput là lý do chọn nó? (một câu: extra all-gather mỗi pass khớp FLOP của pass đó → trần bất động;
FSDP mua memory ÷W, không throughput). (b) **Sửa-và-đoán:** rớt egress từ NVLink 400 GB/s xuống IB 50 GB/s
(×1/8 W) — DP-max và 2D-overlapped-max đổi mấy lần mỗi cái? (đáp: DP-max ∝ W → ×1/8 (26.2→3.3); 2D-max ∝ (W/C)²
→ ×1/64 (161.1→2.5). Đó *là* NVLink→IB cliff: trần 2D sập bình phương → 2D/TP phải ở trong NVLink island).

**5. Frontier.** DualPipe / zero-bubble PP (DeepSeek-V3 **2412.19437**, zero-bubble **2401.10241**): reorder
fwd/bwd micro-batch để overlap all-to-all EP sau compute, ăn gần hết bubble (S6 Bài 6.2 mổ 1F1B bubble
(p-1)/m). 4D/5D parallelism + EP all-to-all (Megatron/TorchTitan 2410.06511, FRONTIER_PRACTICE_2026.md:374):
compose DP+TP+SP+PP+EP trên device-mesh; MoE dùng *all-to-all* (mỗi token routing tới expert trên rank khác)
không all-reduce (S6 Bài 6.3). Interview (chính docstring): "B token/step, D_ff-wide, C FLOP/s, W byte/s — mấy
chip trước khi network bottleneck, xoay knob nào?" → DP chết BW/C, TP chết (3/2)D_ff·W/C, 2D = (3/2)B·D_ff·(W/C)²;
quá đó chỉ interconnect nhanh hơn hoặc critical-batch to hơn cứu. **Honesty:** busbw line-rate + NVLink→IB
cliff ~18× là **rental-gated** (chưa measured, `serving_day_8xH200_runbook.md`); closed form + gloo-correctness
là cái measured NOW.

---

*Ghi chú honesty (FOP-4).* M5 là nửa **BUILT + gloo/CPU-verified** của A2: correctness của DDP/ZeRO-1/FSDP là
**MEASURED** trên gloo (2-rank, nhiều seed/step, RESULTS §A6 + test suite); memory + comms là **arithmetic
exact** pinned bởi test (`test_memory_math`, `test_comms_calc`) — số 100B/xl là *đại số*, không phải training
run. **Chưa có** real multi-GPU NCCL busbw hay MFU-at-scale hay NVLink→IB cliff đo được — chúng **rental-gated**
(8×H200 day). Through-line đúng: *identity + closed form exact bây giờ, bandwidth rental-gated.* Cross-link:
serving-parallelism primitives (TP/PP/EP/WGMMA) ở perf-roadmap `S6_distributed_and_isa.md`; MLA weight-absorption
ở `S2_serving_engines.md` Bài 2.8. Design docs: `A2_100B_MEMORY_ONEPAGER.md`, `A2_COMMS_ALGEBRA.md`.
