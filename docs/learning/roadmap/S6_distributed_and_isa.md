# Série 6 — A6 distributed primitives + ISA-gated kernels + measurement apparatus

> **Series 6 · Distributed & ISA** · số dòng pin theo commit **`9e61d7a`**
> Zone: `src/scratch_llm/utils/{tp_mlp,pipeline_schedule,ep_moe,mfu}.py` +
> `performance/rental/kernels/{wgmma_gemm_sm90a,fa3_attention_hopper,tcgen05_gemm_sm100a}.cu` +
> `performance/artifacts/wgmma_descriptor_manual.md`
> Số đo gốc: `bench/RESULTS.md` §"Perf track (A6…)" + §"ISA-gated kernels" · Design note:
> `performance/notes/A6_design_note.md`

**Vì sao série này.** Bốn primitive song song (TP · PP · EP) + cái cân đo (MFU) trả lời *một* câu hỏi
kỹ thuật: **collective nào chạy trên link nào**. Cái lõi lý thuyết của chúng — TP 2 all-reduce/layer,
1F1B bubble (p-1)/m, EP all-to-all, MFU 6ND — là **chính xác và test được ngay hôm nay** trên một box
(gloo/CPU hoặc số học thuần); chỉ *băng thông* (busbw ở line-rate, cliff NVLink→IB ~18×) là bị khoá
sau máy thuê. Nửa sau série là ba kernel tensor-core theo thế hệ ISA (WGMMA → FA3 → tcgen05): chúng
**compile được** cho arch đích ngay đây, nhưng *chạy* thì phải thuê H100/B200 — đúng đối xứng với
distributed: primitive exact bây giờ, bandwidth rental-gated. Xuyên suốt: **map the chattiest
collective onto the fattest link.**

Thứ tự học (đơn giản → phức tạp, bài sau tựa bài trước): TP (shard một matmul) → PP (bubble depth-wise)
→ EP (all-to-all, collective nặng nhất) → MFU (scoreboard, sáu killer *gọi tên* cả ba bài trên) → WGMMA
(atom tensor-core) → FA3 (dựng trên WGMMA + TMA) → tcgen05 (thế hệ TMEM).

---

## Bài 6.1 — Megatron TP MLP: f/g conjugate + 2-AllReduce/layer (`utils/tp_mlp.py` · `TensorParallelMLP.forward` :215)

> **Câu hỏi first-principles:** một matmul quá to cho một GPU — chẻ nó qua W rank thế nào để hai nửa
> (GEMM-1 và GEMM-2) *ghép lại không cần comm ở giữa*, và forward chỉ tốn **đúng một** collective?
> **Số đo (aha):** TP output + mọi grad **numerically identical** single-GPU (rtol 1e-5); forward phát
> **đúng 1 all-reduce**, fwd+bwd **đúng 2** — không hơn không kém (RESULTS.md §A6, world_size 2 và 4).

**1. Feynman — bài toán bằng lời.** DDP chẻ *batch*; TP chẻ *một layer*. Hình dung FFN `Y = GeLU(X@A)@B`
với `A` là `d_model×d_ff`, `B` là `d_ff×d_model`. Chẻ `A` theo **cột** (output) thì mỗi rank giữ một dải
`d_ff/W` cột → tính được một dải pre-activation từ *toàn bộ* X. GeLU là elementwise nên GeLU trên dải
local = đúng dải của GeLU đầy đủ — **không comm**. Rồi chẻ `B` theo **hàng** (input): mỗi rank nhân dải
GeLU của mình với `B_r` → ra một output *partial* full-width; tổng các partial = output thật, gộp bằng
**một all-reduce SUM**. Đánh đổi cốt lõi: chọn *hai trục chẻ liên hợp* (column-then-row) để hoạt hình
GeLU rơi vào chỗ *không cần đồng bộ*, dồn toàn bộ comm về đúng một điểm.

**2. Cơ chế (derive từ đầu).** Cặp op liên hợp `f`/`g` làm autograd đối xứng fwd↔bwd:
- `f` (`_CopyToModelParallel` :50) — cửa vào: **fwd = identity** (X đã replicated), **bwd = all-reduce**.
  Mỗi rank sinh grad-w.r.t.-X *dùng chung* qua `A_r`; phải cộng lại mới ra `∂L/∂X` thật.
- `g` (`_ReduceFromModelParallel` :65) — cửa ra: **fwd = all-reduce** (cộng partial), **bwd = identity**
  (grad của output đầy đủ broadcast nguyên vẹn về mọi nhánh `B_r`).

Nên: forward = **1 collective** (g), backward = **1 collective** (f). Bias của GEMM-2 *không* shard —
cộng **một lần sau** all-reduce (nếu cộng trong parallel region → cộng W lần → sai). Đây là "MLP's share"
của Megatron 2-all-reduce/layer (attention block cho 2 cái còn lại → 4/layer tổng).

**3. Trace code.** `TensorParallelMLP.forward` :215 → `fc2(F.gelu(fc1(x)))`:
1. `ColumnParallelLinear.forward` :125 → `copy_to_model_parallel(x)` (f: identity fwd) rồi `F.linear(x, weight_r)` → dải pre-act `(…, d_ff/W)`.
2. `F.gelu` elementwise trên dải — **0 comm**.
3. `RowParallelLinear.forward` :169 → `F.linear(x, weight_r)` ra partial → `reduce_from_model_parallel` :171 (g: **all-reduce SUM** — đây LÀ collective duy nhất) → cộng bias *sau* :172-173.
4. `load_reference` :219 shard cùng full weight ra rank (column-slice :133, row-slice :184) → so với `ReferenceMLP` :225.

**4. Cổng teach-back.** (a) Vì sao GeLU nằm *giữa* column-parallel và row-parallel mà không cần một
all-reduce trước nó? (chạm hai chữ: *elementwise* và *output-column*). (b) **Modify-and-predict:** nếu cộng
bias của `fc2` **trước** `reduce_from_model_parallel` thay vì sau — với W=4, output lệch đi *bao nhiêu lần*
bias, và test nào bắt? (đáp: bias bị cộng 4 lần; mutation-test bias-double-count FAIL).

**5. Frontier.** TP chỉ intra-node (≤8, NVLink domain) — *chính vì* all-reduce của nó là chattiest, phải
cưỡi fattest link. Interview: "Megatron một transformer layer tốn mấy all-reduce, và ở đâu?" (4: 2 MLP +
2 attention; fwd g / bwd f mỗi block).

---

## Bài 6.2 — Pipeline schedule: GPipe vs 1F1B, bubble (p-1)/m, peak-act min(p,m) (`utils/pipeline_schedule.py` · `one_f_one_b_schedule` :128)

> **Câu hỏi first-principles:** chẻ model theo *chiều sâu* qua p stage, bơm m microbatch — pipeline phải
> **fill** rồi **drain**, thời gian idle đó (bubble) bằng bao nhiêu, và 1F1B đổi được *cái gì* so với GPipe?
> **Số đo (aha):** bubble = **(p-1)/m** cho *cả hai* (p4m8 → **0.375**, p8m16 → **0.4375**); 1F1B đổi lấy
> **peak activation min(p,m)** thay vì GPipe **m** — same bubble, ít memory hơn (RESULTS.md §A6, 77 tests).

**1. Feynman — bài toán bằng lời.** p công nhân xếp hàng, mỗi người làm một công đoạn. Sản phẩm đầu tiên
phải đi hết p công đoạn thì công nhân cuối mới có việc → đầu ca ai đó rảnh (fill); cuối ca cũng vậy (drain).
Bubble = tổng thời gian rảnh đó. GPipe: làm **hết m forward rồi mới m backward** → mọi activation của m
microbatch phải giữ cùng lúc (peak = m). 1F1B: **một-tới-một-lui** ở steady state → giải phóng activation
nhanh bằng lúc tạo → peak chặn ở độ sâu pipeline. Đánh đổi: **1F1B KHÔNG cải thiện throughput** (bubble y
hệt) — nó chỉ hạ memory từ O(m) xuống O(p), để bạn dám chọn m ≫ p làm bubble mỏng.

**2. Cơ chế (derive từ đầu).** Với t_f = t_b = t: mỗi device làm m·(t_f+t_b) việc thật. Makespan =
fill + steady + drain: forward wave clear hết stage mất (m+p-1) op-time, backward wave thêm (m+p-1) →
makespan = 2(m+p-1)t. Bubble = makespan − busy = 2(p-1)t; chia busy 2mt (t và số 2 triệt tiêu) → **(p-1)/m**.
Deeper pipeline (p↑) → bubble dày; nhiều microbatch (m↑) → mỏng. Interleaved v-chunk chia bubble thêm /v
đổi lấy v× comm (`interleaved_1f1b_bubble_fraction` :308).

**3. Trace code.** Hai đường: *sinh schedule* và *kiểm bằng sim độc lập*.
1. `one_f_one_b_schedule` :128 → stage s có warmup `w = min(p-1-s, m)` forward (stage sâu warmup ít, stage cuối backprop ngay), rồi steady F,B, rồi cooldown drain w backward.
2. `_predecessors` :163 dựng DAG 3 họ cạnh: device-order, forward-flow `F(i,s)←F(i,s-1)`, backward-flow `B(i,s)` cần `F(i,s)` **và** `B(i,s+1)`.
3. `simulate` :205 — Kahn longest-path: op start = max end của predecessor; makespan = max end. Peak-act = walk stream mỗi device (+1 forward, −1 backward, track max) :250-257.
4. `bubble_fraction` đo được so với analytic `gpipe_bubble_fraction` :296 / `one_f_one_b_bubble_fraction` :302; peak so với `one_f_one_b_peak_activations` :331 = `min(p,m)`.

**4. Cổng teach-back.** (a) 1F1B và GPipe bubble giống hệt — vậy vì sao 1F1B là default? (một câu: memory
min(p,m) vs m). (b) **Modify-and-predict:** p=4, m=8 → bubble 0.375. Nếu tăng m lên 16 giữ p=4, bubble mới
= ? và peak-act 1F1B đổi từ min(4,8)=4 sang bao nhiêu? (đáp: (4-1)/16 = 0.1875; peak vẫn min(4,16)=4 — *bất
biến theo m* khi m≥p, đó là toàn bộ điểm của 1F1B).

**5. Frontier.** PP/DP cưỡi inter-node link (rẻ băng thông) vì comm của chúng *thưa* (point-to-point activation
giữa stage kề nhau) — ngược TP/EP. Interview: "pipeline p stage m microbatch, bubble bao nhiêu, GPipe→1F1B đổi
gì?" — đáp thẳng (p-1)/m cả hai; 1F1B giữ throughput, hạ peak-act O(m)→O(p).

---

## Bài 6.3 — Expert-parallel MoE: all-to-all + token-divergence oracle (`utils/ep_moe.py` · `ExpertParallelMoE.forward` :169)

> **Câu hỏi first-principles:** 256 expert không nhét nổi một GPU — shard expert qua rank, giữ token tại
> chỗ, rồi *đưa token tới rank sở hữu expert của nó* thế nào để output **bằng bit** với một reference dày đặc?
> **Số đo (aha):** EP output == single-process dense-gather (allclose) với **zero token-divergence** — một
> nhiễu gate-logit 1e-6 có thể lật top-K tie → invariant bắt map (token→expert) khớp *token-for-token*
> (RESULTS.md §A6; rank0 gửi 3/12, 5/10 off-rank — traffic cross-rank thật, không degenerate-local).

**1. Feynman — bài toán bằng lời.** Router (replicated bit-identical mọi rank) chọn top-K expert/token. Nhưng
expert bị chia: rank r giữ E/W expert. Token phải "bay" tới rank chủ expert, được xử lý, rồi "bay về". Hai
chuyến bay = hai **all-to-all** (dispatch + combine). Correctness dựa trên *một* identity: **expert nào xử lý
token nào, với gate weight nào, độc lập với expert đó nằm ở đâu vật lý** — all-to-all chỉ dời byte, không đổi
số học. Đánh đổi: all-to-all là collective nặng nhất (mọi rank gửi mọi rank), và **load lệch** (hot expert)
là bottleneck throughput — rank bận nhất định step-time.

**2. Cơ chế (derive từ đầu).** Split-size của all-to-all *data-dependent* (không biết trước bao nhiêu token
tới mỗi rank). Nên trả *hai* all-to-all: (i) một `all_to_all_single` tí hon trên **counts** (W long) để học
mỗi rank nhận bao nhiêu; (ii) all-to-all thật trên feature với split đó. Combine leg **tái dùng dispatch split
đảo ngược** (đúng inverse permutation). Hai điều kiện để identity cắn: router bit-replicated (mọi rank cùng
top-K), và weight identity (`build_experts` :57 seed-deterministic → expert e cùng weight mọi rank).

**3. Trace code.** `forward` :169:
1. Route (replicated) :181-182 → `route` :95 (sigmoid affinity, topk, gate normalize sum-to-1).
2. Flatten (token,k)→slot :186-190; `dest_rank = flat_expert // per` :192; stable-sort theo dest → block liền :196; `in_splits = bincount` :199.
3. **Metadata all-to-all** :202-203 (counts) → biết `recv_total`.
4. **DISPATCH all-to-all** :208-215 (feature + local-expert-id).
5. Local expert-GEMM :219-225 (mỗi owned expert chạy trên token nhận được).
6. **COMBINE all-to-all** :228-231 (split swapped = inverse) → un-permute :235 → gate-weight + sum-over-K :236.
7. Oracle: `dense_moe_reference` :111 trả `(y, expert_of_slot)` — cái thứ hai là zero-divergence oracle.

**4. Cổng teach-back.** (a) Vì sao cần all-to-all *metadata trên counts* trước all-to-all trên feature? (một câu:
split data-dependent, không biết a-priori). (b) **Modify-and-predict:** nếu bỏ bước un-permute `slot_out[perm] =
back` :235 (dùng thẳng `back`) — output còn allclose không, và mutation-test nào bắt? (đáp: không — kết quả về
sai slot; mutation "remove inverse-permute" FAIL).

**5. Frontier.** EP như TP: intra-node (all-to-all chattiest → NVLink). DeepEP/DualPipe overlap all-to-all sau
compute là depth training multi-node (rental-gated). Interview: "MoE dispatch, làm sao biết split-size cho
all-to-all?" — metadata exchange trên counts trước.

---

## Bài 6.4 — MFU/HFU 6ND + sáu killer (`utils/mfu.py` · `mfu` :117, `MFU_KILLERS` :191)

> **Câu hỏi first-principles:** một số đọc đồng hồ (s/step) → biến thành *phần trăm hardware bạn thực sự
> trả tiền mà làm việc có ích* thế nào, và vì sao run thật rơi 40–55% chứ không 100%?
> **Số đo (aha):** tái tạo **PaLM 540B 46.2% MFU** (tính ra **45.70%**, Δ0.005) / **57.8% HFU** (57.17%) từ
> C=6ND — một reproduction số ngoài thật, không phải tautology (RESULTS.md §A6, 29 tests).

**1. Feynman — bài toán bằng lời.** Hai utilization cách nhau đúng một FLOP-count. **MFU** = FLOP *toán học
đòi* / peak — dùng so *model* (không thưởng công vô ích). **HFU** = FLOP *phần cứng chạy* / peak — cộng thêm
recompute (gradient checkpointing chạy forward hai lần). HFU ≥ MFU luôn; khoảng cách = **thuế
rematerialization**. HFU nói chip bận cỡ nào; MFU nói bao nhiêu cái bận đó là *cần thiết*. Đánh đổi: report
MFU để so model công bằng, report HFU để biết mình recompute nhiều cỡ nào.

**2. Cơ chế (derive từ đầu).** `C ≈ 6·N·D`: **2** FLOP/param/token forward (một multiply + một add) + **4**
backward (input-grad + weight-grad, hai matmul cùng shape) = 6 (`FLOPS_PER_PARAM` :44). Full-recompute cộng
một forward nữa → +2ND → hardware = 8ND (`hardware_flops` :86); tỉ lệ HFU/MFU = (6+2r)/6, r=1 → **4/3**
(`hfu_over_mfu` :171). Sáu killer nhân nhau: realized MFU = ideal · ∏ η_i (`compose_mfu` :201). Attribution
log-additive: −ln E = Σ(−ln η_i) → share killer i = (−ln η_i)/(−ln E), **cộng đúng bằng 1** (exact, không
Shapley approx — vì model là pure product) (`mfu_gap_attribution` :220).

**3. Trace code.** `mfu` :117 = `(training_flops :71 + extra) / step_time / (n_devices · peak)`;
`training_flops` = 6·N·D. Sáu killer (`MFU_KILLERS` :191, thứ tự canonical): unoverlapped_comm (→ Bài 6.1 TP
all-reduce trên critical path), pipeline_bubble (→ Bài 6.2 (p-1)/m), memory_bound_kernels, small_per_gpu_batch,
moe_imbalance (→ Bài 6.3 hot expert), stragglers. `unexplained_factor` :241 = measured / compose — 1.0 iff sáu
killer giải thích hết gap; <1 = còn "cái thứ bảy" ăn FLOP. PaLM anchor: 6144 TPU-v4 @ 275 TFLOP/s bf16
(`PEAK_FLOPS_BF16_DENSE` :52), r≈0.75 khớp split 46.2/57.8.

**4. Cổng teach-back.** (a) MFU và HFU khác nhau đúng *một* knob — knob nào, và HFU lớn hơn nghĩa là gì tốt hay
xấu? (đáp: recompute_fraction; HFU>MFU = đang trả thuế rematerialization). (b) **Modify-and-predict:** một run
0.9-comm × 0.8-bubble, ideal MFU 60% — realized bao nhiêu, và bubble chiếm bao nhiêu % của gap? (đáp:
0.6·0.72=43.2%; share bubble = ln(1/0.8)/(ln(1/0.9)+ln(1/0.8)) ≈ 68%).

**5. Frontier.** Đây là scoreboard mọi scaling run bị chấm. Interview kinh điển: "6.2 s/step, 512 H100, 70B,
4M tok/step — tốt không, nửa còn lại đâu?" → `mfu()` cho số, `mfu_gap_attribution` gọi tên nửa mất (sáu killer
nhân lại đúng bằng nó).

---

## Bài 6.5 — WGMMA GEMM: warpgroup-async MMA + 64-bit SMEM descriptor (`performance/rental/kernels/wgmma_gemm_sm90a.cu` · `wgmma_m64n64k16_ss` :112; artifact `wgmma_descriptor_manual.md`)

> **Câu hỏi first-principles:** làm sao **128 thread (warpgroup)** cùng phát *một* instruction MMA lên tile
> lớn hơn bất kỳ thread/warp nào giữ nổi, *bất đồng bộ* (issue tách khỏi completion), và operand SMEM được
> mô tả bằng một **con số 64-bit** thay vì con trỏ?
> **Số đo (aha):** PTX có **4× wgmma.mma_async.m64n64k16.f32.f16.f16** + fence/commit/wait, compile sm_90a
> exit 0; descriptor encode khớp CUTLASS GmmaDescriptor (start>>4, LBO 16, SBO 1024, 128B swizzle); target
> book **~318 TFLOPS** H100 4096³ FP16 (RESULTS.md §ISA-gated; **runtime deferred — không claim measured**).

**1. Feynman — bài toán bằng lời.** `mma.sync` là *warp*(32)-collective, blocking. WGMMA nới thành
*warpgroup*(128)-collective, **async**: 128 thread cùng *phóng* MMA (65,536 FMA = 64·64·16 mỗi instruction)
rồi đi tiếp; tensor core đọc A/B và ghi accumulator ở background theo timeline riêng — D chỉ hợp lệ *sau*
`wgmma.wait_group`. Đánh đổi/mở khoá: tách issue↔completion là *chính xác* thứ cho phép producer TMA copy
overlap consumer MMA (nền của FA3, Bài 6.6). B **buộc ở SMEM** — tensor core walk B thẳng từ shared memory qua
descriptor, B không bao giờ chiếm register.

**2. Cơ chế (derive từ đầu).** Descriptor 64-bit *không phải con trỏ* mà là cách *walk* tile ra khỏi SMEM
(artifact §2.1): `[0:14)` start_address = addr>>4 (16-byte granular), `[16:30)` LBO = leading byte offset >>4
(bước giữa core-matrix theo K), `[32:46)` SBO = stride byte offset >>4 (bước theo M/N), `[62:64)` swizzle
{0 none,1 128B,2 64B,3 32B}. Encode: `matrix_descriptor_encode(x) = (x & 0x3FFFF) >> 4` — mask 18 bit, drop
low 4 → 14-bit field, ép mọi tile 16-byte-aligned. Worked example (tile 128×64 f16, 128B swizzle @0x2000):
start=0x200, LBO=16→1, SBO=1024→64, swizzle=1 → **descriptor = 0x4000004000010200** (khớp Colfax
`make_smem_desc`). Accumulator: 32 f32 reg/thread = 64·64/128; warpgroup = 4096 reg = một tile 64×64 (đây là
sức ép RF cap N≤256). Async glue (artifact §1.10): `wgmma.fence` (order register writes trước async read) →
`wgmma.mma_async` (scale-d=0 tile đầu = zero-init free, 1 sau) → `commit_group` (gói thành một group) →
`wait_group N` (N>0 = overlap knob).

**3. Trace code.** `wgmma_gemm_sm90a` kernel :146, mainloop over K :169:
1. Stage GMEM→SMEM với 128B-swizzled store `swz128` :101 (XOR bit [7,10) vào [4,7) — phải khớp swizzle=1 descriptor khai, §3.3).
2. `wgmma_fence()` :201 (một lần, sau zero-init 32 reg).
3. Loop 4 K-strip :202-215 → `make_smem_desc` :81 dựng a_desc/b_desc → `wgmma_m64n64k16_ss` :112 (asm string verbatim từ artifact §5: 32 "+f", "l"(a),"l"(b), imm "1,1,1,0,0").
4. `wgmma_commit()` + `wgmma_wait<0>()` :218-219 drain trước khi overwrite SMEM.
5. Epilogue :230-252 map 32 reg → C[64×64] (reg↔coord flagged DEFER, rental byte-diff).

**4. Cổng teach-back.** (a) Vì sao B *phải* ở SMEM còn A được phép ở register? (một câu: core walk B qua
descriptor, B không vào RF). (b) **Modify-and-predict:** đổi shape sang `m64n256k16` — accumulator vector rộng
bao nhiêu reg/thread, và vì sao N=256 là max? (đáp: 128 reg/thread = 64·256/128; sát trần 255-register kiến
trúc — đó *là* lý do N≤256).

**5. Frontier.** WGMMA là atom Hopper; `-arch=sm_90a` (chữ `a` bắt buộc, plain sm_90 im lặng thiếu instruction).
Interview: "wgmma async nghĩa là gì cho D register, và ai reclaim completion?" — D không valid tới khi
`wait_group` retire group.

---

## Bài 6.6 — FA3 attention: warp-specialized producer/consumer + TMA + ping-pong (`performance/rental/kernels/fa3_attention_hopper.cu` · `fa3_attention_fwd` :157)

> **Câu hỏi first-principles:** dựng trên WGMMA async, làm sao *chuyên hoá warpgroup* thành producer (TMA
> load) vs consumer (MMA + online-softmax), double-buffer K/V ping-pong, để load HBM luôn overlap compute?
> **Số đo (aha):** PTX (compile sm_90a, **-cubin exit 0 ptxas đầy đủ**) có **8× wgmma + 3× cp.async.bulk.tensor
> (TMA) + 14 mbarrier + setmaxnreg** (warp-spec); target paper **~75% util / ~740 TF/s** FP16, FP8 ~1.2 PFLOP/s
> (RESULTS.md §ISA-gated; **runtime + fragment-map/causal-mask flagged DEFER**).

**1. Feynman — bài toán bằng lời.** FA2 dùng cả block để load *và* tính, xen kẽ. FA3 **chia vai**: một
warpgroup producer chỉ phóng TMA load HBM→SMEM (một thread elected, `setmaxnreg.dec` nhả register), các
warpgroup consumer chỉ MMA + softmax (`setmaxnreg.inc` xin thêm register). K/V stage thành ping-pong
double-buffer: producer fill stage này trong khi consumer drain stage kia, bắt tay qua mbarrier. Đánh đổi:
async WGMMA + TMA cho phép *hoàn toàn* giấu latency HBM sau tensor-core math — nhưng phải quản parity mbarrier
+ register realloc thủ công.

**2. Cơ chế (derive từ đầu).** Producer/consumer handshake bằng **mbarrier** (async barrier, PTX §9.7.13):
`mbar_expect_tx` (producer báo phase này giao `bytes` qua TMA rồi arrive) + `mbar_wait` (consumer spin trên
phase-bit tới khi TMA completion lật). **TMA** (`cp.async.bulk.tensor.2d`) load tile 2D HBM→SMEM một-thread-issue,
completion signal trên mbarrier. Online-softmax là *engine chính xác* (không xấp xỉ): mỗi K-tile — row-max
m_tile → m_new = max(m_old, m_tile) → rescale factor alpha = exp2((m_old−m_new)·log2e) áp lên **cả acc_o lẫn
row_l** (bug kinh điển: quên rescale O, chỉ rescale l) → p = exp2((s−m_new)) → row_l += Σp. Deferred divide
(chia l ở epilogue, trick FA2).

**3. Trace code.** `fa3_attention_fwd` :157, block = producer WG + consumer WG(s):
1. Producer :204 — `warpgroup_reg_dealloc` :205, load Q một lần :211, loop K/V tile: `mbar_wait(bar_empty)` back-pressure :221 → `tma_load_2d` K,V vào ping-pong stage :227-228.
2. Consumer :237 — `warpgroup_reg_alloc`, loop K/V tile: `mbar_wait(bar_full)` :260.
3. **WGMMA #1** S=Q@K^T :262-274 (contract D qua KSTEPS_QK=4 strip, scale_d=0 strip đầu) → `wgmma_wait<0>` (FA3 thật để PV in-flight: wait<1>).
4. Online-softmax :281-333 (rescale acc_o *alpha :310 — cái must-not-forget).
5. **WGMMA #2** O~ += P@V :337-346.
6. Release stage về producer :349-351 (ping-pong parity toggle).
7. Epilogue normalize O = O~/l :354-360.

**4. Cổng teach-back.** (a) Bug kinh điển "quên rescale O" — cụ thể quên dòng nào, và output sai thế nào? (đáp:
`acc_o[i] *= alpha` :310; nếu chỉ rescale l mà không rescale running output → các K-tile trước bị weight sai theo
max cũ). (b) **Modify-and-predict:** nếu để `wgmma_wait<1>` thay `<0>` sau WGMMA#1 — overlap gì được mở, và rủi
ro đọc gì? (đáp: PV của tile trước còn in-flight khi bắt đầu QK tile sau = overlap thật; rủi ro đọc D register
trước khi group retire nếu quản parity sai).

**5. Frontier.** FA3 là chuẩn 2026 attention trên Hopper (warp-spec + TMA + FP8 incoherent-processing Hadamard).
Interview: "FA3 nhanh hơn FA2 nhờ đâu trên Hopper?" — async WGMMA + TMA cho warp-specialization giấu HBM sau
math (FA2 không async được).

---

## Bài 6.7 — tcgen05 GEMM: TMEM accumulator + full-warpgroup drain (`performance/rental/kernels/tcgen05_gemm_sm100a.cu` · `tcgen05_gemm_1sm` :222)

> **Câu hỏi first-principles:** thế hệ Blackwell datacenter — accumulator dời khỏi register *vào Tensor
> Memory (TMEM)*, MMA thu về **single-thread issue**; vậy ai drain 128 lane TMEM ra, và cạm bẫy nào làm 3/4
> tile stale?
> **Số đo (aha):** PTX (compile sm_100a, **-cubin exit 0**) có **21× tcgen05 incl. tcgen05.mma.cta_group::1.kind::f16**
> + TMA; SASS (nvdisasm) = **UTCHMMA + LDTM.x4**; target **~1209 TF/s = 54% dense** B200 (2250 TF/s), 2-SM win
> honest **~8% (1209→1302)** (RESULTS.md §ISA-gated; **idesc/descriptor consts flagged [INFERENCE], runtime deferred**).

**1. Feynman — bài toán bằng lời.** Xuyên suốt throughline collective-widening: `mma.sync` = warp(32),
`wgmma` = warpgroup(128), **tcgen05 = single-thread(1)** issue. Nhưng nghịch lý: một thread phát MMA, còn
accumulator không nằm ở register nữa mà ở **TMEM** (128 lane × 512 col, on-chip). Đánh đổi: TMEM giải phóng
register file khổng lồ (accumulator to hơn nhiều), MMA issue rẻ (một thread) — nhưng **drain phải cả
warpgroup**: một warp `tcgen05.ld` chỉ chạm 32/128 lane, nên drain một warp = 3/4 tile 128-row **STALE** (bug
tcgen05 kinh điển). Bốn warp phải drain, warp w địa chỉ lane-quadrant [32w, 32w+32).

**2. Cơ chế (derive từ đầu).** Spine: (1) `tcgen05.alloc` TMEM accumulator (NCOL pow2 ≥32, **một warp** issue,
rồi relinquish permit) — atom lớn nhất f16 là 128×256×16. (2) TMA stage A/B K-strip vào SMEM, mbarrier-tracked.
(3) **một thread** `tcgen05.mma.cta_group::1.kind::f16 [d-tmem], a-desc, b-desc, idesc, enable-d` — accumulate
trong TMEM cả K-loop (enable-input-d = predicate: false atom đầu D:=A·B, true sau D+=A·B). (4) `tcgen05.commit`
arrive trên mbarrier (completion là mbarrier, *không* wait_group như WGMMA). (5) full-warpgroup drain: TMEM
address pack [31:16]=lane, [15:0]=col; warp w OR (32·w)<<16 vào base, `tcgen05.ld.32x32b.x4` đọc 32 lane × 4 col
→ 4 reg/thread, loop col. Descriptor SMEM **chung với WGMMA** (LBO/SBO >>4, swizzle bit [62:64)) — cùng artifact.

**3. Trace code.** `tcgen05_gemm_1sm` :222:
1. `tmem_alloc` :132 từ warp 0 :242 → base addr ghi về SMEM slot :245.
2. K-mainloop :257 — TMA stage :260-266 → `mbar_wait(bar_full)` :267.
3. `tcgen05_fence_before` :271 → một thread `tcgen05_mma_f16` :149 (accum = k≠0) → `tcgen05_commit` :277 → `mbar_wait(bar_mma)` :280 (retire trước khi overwrite SMEM).
4. **`drain_tmem` :186** — full warpgroup 128 thread, warp = tid>>5, `warp_lane_base = tmem_base | (32·warp)<<16` :192, loop NCOL step 4: `tcgen05.ld.32x32b.x4` :198 → `tcgen05.wait::ld` :201 → store row = 32·warp+lane :206 (bao phủ đủ 128 lane — điểm structural của rung).
5. `tmem_dealloc` :290 từ warp 0.

**4. Cổng teach-back.** (a) Vì sao MMA single-thread-issue nhưng drain phải cả 4 warp? (một câu: `tcgen05.ld`
mỗi warp chỉ chạm 32/128 lane TMEM). (b) **Modify-and-predict:** nếu `drain_tmem` chạy từ *một* warp (bỏ vòng
quadrant) — bao nhiêu phần output đúng, và nó xuất hiện ở đâu của tile? (đáp: đúng 32/128 = 1/4; chỉ 32 row
top-của-quadrant-warp-đó đúng, 3/4 còn lại stale — verify per-quadrant, không chỉ top-left).

**5. Frontier.** tcgen05/UMMA là tensor-core B200 (`-arch=sm_100a`, chữ `a` bắt buộc như sm_90a). 2-SM
`cta_group::2` win chỉ ~8% (SMEM-bandwidth relief, *không* raw math) — honest framing. Interview: "Blackwell
đổi gì so với Hopper ở accumulator?" — dời từ register (WGMMA) vào TMEM; MMA issue một thread; drain cần cả
warpgroup vì lane-quadrant.

---

*Ghi chú honesty (FOP-4):* mọi số distributed (Bài 6.1–6.4) là **correctness measured** trên gloo/CPU (112 CPU
tests); mọi số ISA-kernel (Bài 6.5–6.7) là **compile-verified only** — TF/s là *pre-registered target* của
paper/book, **runtime deferred** tới H100/B200 (`H100_day_runbook.md`, `B200_day_runbook.md`, `serving_day_8xH200_runbook.md`).
Bandwidth (busbw line-rate, cliff NVLink→IB ~18×) là rental-gated. Đúng through-line: primitive exact bây giờ,
bandwidth rental-gated. Design note: `performance/notes/A6_design_note.md`.
