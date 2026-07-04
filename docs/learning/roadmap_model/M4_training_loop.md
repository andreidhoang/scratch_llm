# SÉRIE M4 — THE TRAINING LOOP + efficiency, derived (vòng lặp huấn luyện, dẫn xuất từ đầu)

> **Số dòng pin theo commit `4ad0ac5`.** Đây là roadmap DẪN-XUẤT (derive từ blank + trace) để dẫn đường
> teach-back sâu về sau, KHÔNG phải bài giảng đầy đủ mỗi mục. Mỗi Bài ~một màn hình, tự chứa: (1) bài toán
> bằng lời, (2) dẫn xuất toán/thiết kế từ đầu, (3) trace code file·hàm·dòng, (4) cổng teach-back, (5) frontier.
> Số đo: `bench/RESULTS.md` (§Frontier ablations F1/F4 dòng 636–673). Code: `src/scratch_llm/train.py` +
> `utils/{mixed_precision,checkpointing,seeding,monitors}.py`. TWIN perf: `docs/learning/roadmap/` (serving+kernel).

**Vì sao série này.** M1–M3 dựng *cái model* (byte→BPE→Transformer block: RMSNorm·RoPE·SwiGLU·GQA) và *cái
optimizer* (AdamW/Muon) — nhưng một model với trọng số ngẫu nhiên KHÔNG biết gì. Série này là **cái động cơ
biến trọng số ngẫu nhiên thành policy đã học**: một vòng lặp `forward→loss→backward→clip→step→zero_grad` lặp
hàng nghìn bước, cộng bốn đòn bẩy *efficiency* để nó chạy được ở quy mô thật (mixed precision cho throughput,
activation checkpointing cho bộ nhớ, `torch.compile` cho launch-overhead) và *đáng tin* (fixed seed + monitors).
Sợi chỉ xuyên suốt: **mọi thứ ở đây là "plumbing" — nhưng plumbing SAI thì mọi thứ phía trên vô nghĩa**, nên
mỗi component có một *invariant-oracle* đo được (overfit→<1e-2, checkpoint round-trip, seed-reproducible,
grad-parity) là tuyến phòng thủ đầu tiên mà lab dùng để screen. Série này tựa lên M2 (optimizer: `build_optimizer`,
`cosine_lr`, `gradient_clipping`) và khép vòng ở Phase-0 speedrun (BPE→pretrain→eval→sample, `val_bpb 0.02` đo thật).

Thứ tự đọc: **4.1 (bộ xương: một bước train + overfit oracle) → 4.2 mixed precision (đòn bẩy numerics) →
4.3 activation checkpointing (đòn bẩy bộ nhớ) → 4.4 torch.compile + F4 NaN (đòn bẩy launch-overhead) →
4.5 seed + monitors (tin cậy + quan sát).** Mỗi Bài tựa lên Bài trước — đừng nhảy cóc.

---

## Bài 4.1 — Một bước train: forward→loss→backward→clip→step→zero_grad + memmap + overfit oracle (`train.py` · `train` :114 · `get_batch` :37)
> **Câu hỏi first-principles:** thứ tự SÁU thao tác trong một bước SGD là gì, vì sao ĐÚNG thứ tự đó (đổi chỗ
> hai cái bất kỳ hỏng ra sao), và vì sao *overfit-one-batch* là oracle số 1 để biết wiring này đúng?
> **Neo (invariant đo thật):** overfit-one-batch → **loss < 1e-2** trong ngân sách step của AdamW (test
> `test_train.py`); hybrid MuonAdamW cũng đẩy 1 batch xuống **<0.05 trong 300 step** (RESULTS.md:620, MEASURED).
> Đồng thời: loss-at-init ≈ **log V** (CE đều), checkpoint round-trip khôi phục step + mọi param (test).

**1. Feynman — bài toán bằng lời.** Một bước train là "sáu động tác một nhịp": (1) lấy một batch (input, next-token),
(2) `zero_grad` xoá gradient cũ, (3) `forward` ra logits, (4) `loss = cross_entropy`, (5) `backward` tính grad,
(6) `clip` rồi `step` cập nhật trọng số. Analogy: leo núi trong sương — mỗi bước bạn (a) đo dốc tại chỗ đứng
(grad), (b) đi một bước theo dốc (step), rồi (c) *quên* dốc cũ (`zero_grad`) trước khi đo lại. Đánh đổi cốt lõi
của cả série: vòng lặp này **đơn giản đến mức dễ SAI ngầm** — một dấu shift lệch trong `get_batch`, một
`zero_grad` đặt sai chỗ, một loss cộng nhầm aux — không crash, chỉ *học chậm hoặc không học*. Vì thế nghề là
gắn mỗi mắt xích một **invariant đo được**, không tin "nó chạy" là "nó đúng".

**2. Dẫn xuất từ đầu (derive).** *Next-token alignment:* mô hình học P(x_{t+1} | x_{≤t}); nên `targets` PHẢI là
`inputs` dịch một vị trí. `get_batch` (:37) chọn `starts` ngẫu nhiên (:53), rồi `inputs = data[s:s+L]`,
`targets = data[s+1:s+1+L]` (:54–55) — bất biến `targets[b,t] == inputs[b,t+1]`. Lệch một ô = model học dịch
identity, loss kẹt. *Vì sao memmap:* `data` là `np.memmap` — corpus lớn hơn RAM không bao giờ nạp trọn; OS phân
trang lazy. *Thứ tự sáu động tác (derive vì sao):* `zero_grad` (:156) TRƯỚC backward vì `.backward()` *cộng dồn*
vào `.grad` (không ghi đè) — quên xoá = grad của nhiều batch chồng lên nhau. `clip` (:170) GIỮA backward và step
vì clip tác động lên `.grad` đã tính xong nhưng TRƯỚC khi step đọc nó — global-ℓ2 clip (M2) chặn một batch outlier
làm nổ một bước. `step` (:171) cuối. *Loss-at-init oracle:* model tươi đoán đều trên V token ⇒ CE ≈ log V; lệch ⇒
bug head/embedding/mask. *Overfit oracle (derive vì sao là #1):* nếu tối ưu KHÔNG kéo được loss của MỘT batch cố
định về ~0, thì lỗi ở *wiring* (optimizer/loss/grad-flow), KHÔNG phải ở data — vì một batch luôn "học thuộc lòng"
được nếu grad chảy đúng. Nó *cô lập* lỗi wiring khỏi lỗi data trước khi đốt compute thật.

**3. Trace code.** `train` (:114): `seed_everything(cfg.seed)` (:132) → `build_optimizer` (:134, M2) → vòng
`for step` (:150): `cosine_lr` (:151) ghi `group["lr"]` (:152–153) → `get_batch` (:155) → `zero_grad` (:156) →
`forward_model(inputs)` trong `amp_ctx` → `cross_entropy(logits, targets)` (:168; nhánh MoE :163–166 cộng
`aux.total`) → `loss.backward()` (:169) → `gradient_clipping(model.parameters(), cfg.grad_clip)` (:170) →
`optimizer.step()` (:171) → nếu MoE thì `moe_update_biases()` (:174, aux-loss-free nudge SAU update). Log mỗi
`log_every`: `loss.item()` (:177) — *một* host-sync/interval (bài học R1). Checkpoint: `save_checkpoint` (:62)
lưu `{model, optim, step}`; `load_checkpoint` (:74) khôi phục (`weights_only=False` :83 vì ta nạp checkpoint
tin cậy có metadata optimizer). Test pin: `test_train.py` — next-token align, round-trip, overfit<1e-2, học <log V.

**4. Cổng teach-back.** (a) Vì sao `zero_grad` phải TRƯỚC `backward`, không phải sau `step`? (Gợi: `.backward`
cộng dồn; đặt sai chỉ hỏng khi có >1 batch/step — micro-batching che mất lỗi.) (b) *Sửa-và-đoán:* nếu đổi
`targets = data[s+1:...]` thành `data[s:...]` (bỏ shift) — loss-at-init đổi không? loss overfit hội tụ về đâu?
(Gợi: init vẫn ≈ log V; overfit vẫn về ~0 nhưng model học *copy* input → sinh văn bản lặp token hiện tại, oracle
next-token-align bắt được ngay.) (c) Vì sao overfit-one-batch cô lập lỗi *wiring* khỏi lỗi *data*?

**5. Frontier.** Ba oracle này (loss-at-init≈log V, overfit-one-batch, fixed-seed) là **CS336 Engineering
disciplines #1–3** (CLAUDE.md) = đúng bộ "first-line triage" mà FRONTIER_PRACTICE_2026 §babysit (dòng 76) liệt
kê: lab screen ứng viên bằng chính chúng — "loss-at-init bắt bug head/embedding/mask *trước khi* tốn compute;
overfit-one-batch tách bug wiring khỏi bug data". So với nanochat `speedrun.sh`: cùng bộ xương forward→loss→
backward→step, nhưng ta thêm loud-NaN-guard (Bài 4.3) + MoE aux nudge. Câu interview: "Model không học — ba
thứ bạn kiểm tra theo thứ tự nào, và mỗi thứ loại trừ giả thuyết gì?"

---

## Bài 4.2 — Mixed precision: bf16 autocast + fp32 accum/master, và vì sao KHÔNG cần GradScaler (`train.py` :157 · `utils/mixed_precision.py` · `naive_accumulate` :28)
> **Câu hỏi first-principles:** vì sao huấn luyện low-precision cần fp32 ở ĐÚNG vài chỗ (accum/master/loss),
> và vì sao bf16 *không* cần loss-scaling trong khi fp16 thì cần?
> **Neo (invariant đo thật):** fp16 naive accumulate 0.01 **đóng băng ở 32.0** (ulp@32 = 2^-5 = 0.03125 >
> 2·0.01 ⇒ cộng làm tròn về no-op) — invariant unit-test được, `naive_accumulate` :28. muon_adamw · **bf16** ·
> eager → **loss 9e-4 (LEARNED, không GradScaler)** (RESULTS.md:645, MEASURED sm120).

**1. Feynman — bài toán bằng lời.** Số dấu phẩy động có *độ phân giải tương đối*: gần 0 thì mịn, xa 0 thì thô
(ulp — khoảng cách hai số kề — lớn dần theo độ lớn). bf16/fp16 có **ít bit mantissa** (bf16: 7, fp16: 10) nên
ulp thô. Analogy: cái thước chỉ có vạch cm — cộng 3mm vào 40cm, kết quả làm tròn về 40cm, "3mm bốc hơi". Đánh
đổi cốt lõi: matmul ở bf16 nhanh gấp bội (tensor core), NHƯNG *tổng chạy dài* (optimizer momentum, loss, reduction)
ở bf16 mất số hạng nhỏ → hỏng. Lời giải: **bf16 cho GEMM throughput, fp32 cho những chỗ cộng dồn dài**. autocast
mã hoá đúng policy per-op này — ta KHÔNG chọn dtype bằng tay.

**2. Dẫn xuất từ đầu (derive).** *Vì sao accum phải fp32 (small-addend underflow):* tổng tuần tự `acc += inc`
đóng băng khi `inc < ulp(acc)/2` — cộng làm tròn về `acc`. Ví dụ đo được: fp16 cộng 0.01, tại `acc=32` thì
ulp = 2^-5 = 0.03125, 0.01 < 0.031/2 ⇒ **kẹt vĩnh viễn ở 32.0**. Đây là *lý do* optimizer state (momentum/variance),
loss, reduction giữ fp32 dù model chạy bf16 (`naive_accumulate` :28–37 phơi bày lỗi này bằng vòng *tuần tự* cố ý —
`torch.sum` pairwise/tree chính xác hơn nhiều, vòng tuần tự CHÍNH LÀ bài học). *Autocast dtype rule:* dưới
`torch.autocast`, matmul → bf16 (tensor core), còn LayerNorm/softmax/reduction + master weights → fp32 (ổn định).
*Vì sao bf16 KHÔNG cần GradScaler còn fp16 CẦN:* fp16 có exponent 5 bit → dải ~6e-5..65504; gradient nhỏ *underflow
về 0* → GradScaler nhân loss lên trước backward, chia lại sau, để đẩy grad vào dải biểu diễn được. bf16 có exponent
**8 bit** (bằng fp32) → cùng *dải mũ*, chỉ mất *mantissa precision*, không underflow — nên không cần scaling. Đó là
vì sao `train.py` guard `amp_dtype ∈ {None, "bf16"}` (:128), từ chối fp16 với thông điệp "fp16 needs a GradScaler".

**3. Trace code.** `TrainConfig.amp_dtype` (:110): `None ⇒ fp32` (mặc định = đường A1), `"bf16" ⇒ autocast`.
`train` (:128) raise nếu không phải hai giá trị đó. `amp_dtype = torch.bfloat16 if ... else None` (:146);
`device_type` (:147). Trong vòng: `amp_ctx = torch.autocast(device_type, dtype=amp_dtype)` (:157–161) hoặc
`nullcontext()` nếu fp32; forward + loss chạy *bên trong* `with amp_ctx` (:162–168) — chỉ forward/loss autocast,
`backward`/`clip`/`step` (:169–171) NGOÀI context (grad accumulate ở dtype param = fp32 master). `naive_accumulate`
(:28): `acc = zeros(dtype)`, vòng `acc += inc` (:35–37) — 0-d tensor phơi underflow. Cross-link perf: bf16 GEMM
peak trên card này = 72 TF/s (roadmap/S3, RESULTS.md:493) — đó là *lý do kinh tế* của cả bài này.

**4. Cổng teach-back.** (a) Dựng lại con số: fp16 cộng 0.01, đóng băng ở đâu và VÌ SAO đúng chỗ đó? (ulp@x =
2^(floor(log2 x)−10); tìm x nhỏ nhất mà ulp/2 > 0.01.) (b) Vì sao bf16 khỏi GradScaler mà fp16 cần — chạm hai chữ
"exponent bits" và "underflow" vs "precision". (c) *Sửa-và-đoán:* nếu để momentum của AdamW ở bf16 thay vì fp32,
overfit-one-batch (batch nhỏ, ~vài trăm step) có còn về <1e-2 không, hay chỉ hỏng ở run dài? (Gợi: batch nhỏ,
grad còn lớn → chưa chạm underflow; lỗi lộ ở tail của run dài — vì sao overfit *không* bắt được lỗi này.)

**5. Frontier.** Câu interview kinh điển (mixed_precision.py docstring): "*bạn train fp16 và loss plateau/NaN — vì
sao?*" → accumulation precision + autocast policy (và một bậc thấp hơn: FP8 amax/scale tràn). Đây cũng là gốc cơ
học của **train↔serve logit drift** (`monitors.py` `kl_train_infer`, Bài 4.5): hai engine reduce softmax/PV ở
precision khác nhau → logits khác trên CÙNG trọng số. So frontier: FRONTIER_2026 §F4 (dòng 133) đặt bf16-autocast
là default 2026; bậc kế FP8-GEMM gated bởi train↔serve KL. DeepSeek-V3 (2412.19437) train FP8 với per-tile/per-128
block scaling — chính là "fp32 ở đúng chỗ" đẩy xuống một precision nữa.

---

## Bài 4.3 — Activation checkpointing: đòn bẩy recompute-vs-store cho bộ nhớ (`utils/checkpointing.py` · `run_block` :63 · `_sac_policy` :56)
> **Câu hỏi first-principles:** bộ nhớ activation lúc train là O(depth · per_block) — làm sao giảm nó xuống
> ~O(depth · residual) mà autograd vẫn ra *đúng* gradient, và tại sao "selective" là default 2026?
> **Neo (invariant đo thật):** grad dưới `"full"`/`"selective"` **bằng eager `"none"` tới fp-tolerance**
> (`test_checkpointing.py`, recompute phải trong suốt với autograd); saved-bytes **full < selective < none**
> đo qua `saved_tensors_hooks`. SAC (Korthikanti/Megatron): **~70% activation-mem cho ~2.7% extra FLOPs** @GPT-3.

**1. Feynman — bài toán bằng lời.** Forward lưu MỌI activation trung gian để backward dùng lại (chain rule cần
chúng) — đó là phần bộ nhớ *lớn nhất* lúc train, tỉ lệ với depth × width × batch × seq. Đánh đổi: thay vì *lưu*
activation, ta *vứt* chúng ở forward rồi **tính lại** ở backward. Analogy: giải toán nhiều bước — thay vì giữ mọi
kết quả trung gian trên giấy (tốn giấy), chỉ giữ đề mỗi phần rồi *làm lại* phần đó khi cần kiểm. Trả thêm compute
(một forward nữa) để đổi lấy bộ nhớ. Ba mức span đúng trade này: `none` (lưu hết), `full` (chỉ lưu input mỗi block,
tính lại cả block), `selective` (lưu cái *đắt tính lại, rẻ lưu* = matmul output; tính lại cái rẻ = softmax/elementwise).

**2. Dẫn xuất từ đầu (derive).** *Vì sao đúng gradient:* checkpoint là một `autograd.Function` mà forward chạy
`no_grad` (không dựng graph, không lưu activation), backward *chạy lại* forward *có* graph rồi backprop qua nó.
Điều kiện sống-còn: forward phải **deterministic** (cùng input → cùng activation) — nếu không (dropout seed khác,
non-determinism) thì grad tính lại ≠ grad thật. *Vì sao selective là điểm ngọt:* recompute một GEMM tốn nhiều FLOP
(đắt), nhưng *lưu* output nó rẻ (một tensor); ngược lại softmax/elementwise rẻ tính lại. Nên **MUST_SAVE matmul
output, PREFER_RECOMPUTE phần còn lại** = tối đa tiết kiệm bộ nhớ với tối thiểu FLOP thêm. Đây chính là cùng trade
như FA2-backward (roadmap/S4: không lưu ma trận score N×N, tính lại) nhưng ở tầng *cả block*, cao hơn một nấc.
*Hai trục đo:* bộ nhớ (bytes activation lưu qua ranh giới checkpoint) và compute (số matmul chạy — recompute làm
nó tăng ở backward).

**3. Trace code.** `MATMUL_OPS` (:43) = {mm, addmm, bmm, matmul}. `_sac_policy` (:56): `op in _SAVE_OPS →
MUST_SAVE`, else `PREFER_RECOMPUTE`. `run_block(block, mode, *args)` (:63): `none → block(*args)`; `full →
checkpoint(block, use_reentrant=False)` (:69); `selective → checkpoint(..., context_fn=lambda:
create_selective_checkpoint_contexts(_sac_policy))` (:71–76). Đo bộ nhớ: `count_saved_activation_bytes` (:80)
dùng `saved_tensors_hooks` — `pack` dedup theo storage `data_ptr` (:93–98), cộng bytes distinct tensor lưu qua
*ranh giới ngoài* (SAC cache matmul-output *bên trong* region, ẩn với hook ngoài ⇒ `full ≈ selective` ở TRỤC BỘ
NHỚ; khác biệt lộ ở TRỤC COMPUTE). Đo compute: `count_op_executions` (:120) qua `TorchDispatchMode` đếm matmul —
phải bọc CẢ forward LẪN backward vì recompute xảy ra ở backward (:126). Test: grad-parity (recompute trong suốt)
+ bytes-order full<selective<none.

**4. Cổng teach-back.** (a) Vì sao forward-phải-deterministic là điều kiện để checkpointing ra đúng grad — dropout
phá nó thế nào, và fix ra sao (rng-state save)? (b) Vì sao ở TRỤC BỘ NHỚ `full ≈ selective` nhưng ở TRỤC COMPUTE
thì khác? (SAC cache matmul-out trong region, hook ngoài không thấy; khác biệt = số matmul recompute.) (c) *Sửa-
và-đoán:* nếu `_sac_policy` đổi thành MUST_SAVE *mọi* op (không chỉ matmul) — bộ nhớ và số matmul recompute đổi
về đâu, và nó suy biến thành mode nào? (Gợi: MUST_SAVE-all ≡ "none" về hiệu ứng.)

**5. Frontier.** `selective` = default 2026 (Korthikanti et al., dùng ở Megatron/TorchTitan): "~70% activation
memory cho ~2.7% extra FLOP ở GPT-3 scale" (docstring :14). So chuẩn: TorchTitan để bạn chọn `full`/`selective`
per-layer; DeepSeek/Llama pipeline dùng để nhét batch lớn hơn vào cùng HBM. Cross-link perf: cùng recompute-vs-store
là bản chất **FA2 backward** (roadmap/S4) — chỉ khác chỗ đặt ranh giới. Câu interview: "Bạn OOM ở microbatch lớn
nhất — hai đòn bẩy bộ nhớ không đụng tới model, và mỗi cái đổi compute/comm ra sao?" (activation checkpointing +
ZeRO/FSDP shard — roadmap/S6).

---

## Bài 4.4 — torch.compile + phát hiện F4: bf16✓, compile✓, nhưng combo NaN (loud guard) (`train.py` · `torch.compile` :145 · NaN guard :178)
> **Câu hỏi first-principles:** `torch.compile` giúp gì (đòn bẩy launch-overhead), vì sao trên box này bf16
> ĐÚNG-riêng và compile ĐÚNG-riêng nhưng KẾT HỢP thì NaN, và một NaN-guard đúng chỗ trông thế nào?
> **Neo (đo thật, RESULTS.md:645–647):** muon_adamw·bf16·eager → **loss 9e-4 LEARNED**; muon_adamw·fp32·**compile**
> → **8.3e-4 LEARNED**; **any·bf16·compile → NaN @~step5–10, guard RAISES** (`[FACT]` sm120/torch-2.12 inductor,
> *reproduces với plain AdamW* — không phải logic ta). Prediction chưa đo: compile+bf16 = **+10–20% MFU** (F4 spec).

**1. Feynman — bài toán bằng lời.** Eager PyTorch phóng *một CUDA launch mỗi op* — hàng trăm launch/step, mỗi cái
tốn ~µs dispatch. `torch.compile` (inductor) *fuse* nhiều pointwise op thành ít kernel + sinh code Triton → ít
launch, ít HBM round-trip. Analogy: thay vì gọi thợ riêng cho từng viên gạch, giao cả bức tường một lần. Đó là đòn
bẩy **launch-overhead** (bổ sung cho 4.2 numerics, 4.3 memory). Đánh đổi *đắt* đã đo: compile + bf16 *riêng lẻ*
đều đúng + có giá trị, nhưng *tích* của chúng trên **inductor sm120/torch-2.12** sinh mã NaN — một codegen bug,
KHÔNG phải lỗi logic. Bài học vận hành: đừng tin "mỗi mảnh đúng ⇒ combo đúng"; và *fail-loud* thay vì đốt compute
trên NaN im lặng.

**2. Dẫn xuất từ đầu (derive).** *Vì sao compile giảm thời gian:* decode/step eager bị *overhead-bound* (roadmap/S1
đo: ~955 launch/token ở serving; train tương tự nhiều launch nhỏ). Fuse → collapse launch → BW đo được tăng (perf
R1: eager 51 → compiled 173 tok/s = 15%→53% HBM, RESULTS.md:217). *Vì sao +10–20% MFU (prediction):* MFU = FLOP
model / (FLOP peak · wall); compile giảm wall (overhead) không đổi FLOP model ⇒ MFU lên. *Vì sao combo NaN (predict-
vs-measure honesty):* F4 *dự đoán* bf16+compile = combo MFU rẻ; *đo* ra nó diverge trên inductor này. Verify:
reproduces với plain AdamW (loại trừ Muon), với `matmul_precision="high"` vẫn NaN (loại trừ giả thuyết precision-mode)
⇒ là codegen bug của inductor khi lower autocast-bf16 region, không phải logic repo. Mỗi feature ĐÚNG một mình
(bf16-eager 9e-4, fp32-compile 8.3e-4) ⇒ tích mới hỏng. *NaN-guard derive:* guard phải fire *không thêm host-sync*
— nên đặt tại nhịp log (`loss.item()` vốn đã sync :177), `math.isfinite` (:178), raise với thông điệp chẩn đoán.

**3. Trace code.** `TrainConfig.compile` (:111). `train`: `forward_model = torch.compile(model) if cfg.compile
else model` (:145) — GIỮ `model` gốc cho `.cfg`/`.moe_update_biases`/checkpoint (compiled module có prefix
`_orig_mod.`, và split Muon phải thấy submodule thật) — đây là lý do có HAI biến `model`/`forward_model`. Forward
gọi `forward_model(inputs)` (:165/:168) nhưng clip/step/moe-bias dùng `model` (:170/:171/:174). NaN-guard (:176–186):
tại `step % log_every == 0`, `loss_value = loss.item()` (:177, one sync), `if not math.isfinite(loss_value)` (:178)
→ `raise RuntimeError(f"non-finite loss ({loss_value}) at step {step}...avoid bf16 + torch.compile together on
sm120 (an inductor codegen bug).")` (:183–186). Comment (:181–182) ghi rõ trigger + fix: bf16-eager HOẶC fp32-compile;
bf16+compile là **đường H100-rental**. Đo: RESULTS.md bảng F1/F4 (:642–647).

**4. Cổng teach-back.** (a) Vì sao `torch.compile` giúp một step *overhead-bound* mà không giúp step đã
*memory-saturated*? (b) Chuỗi verify nào cho phép kết luận "codegen bug của inductor" chứ không phải "logic ta sai
hay Muon sai"? (reproduces plain-AdamW + mỗi feature đúng riêng + matmul_precision không cứu.) (c) *Sửa-và-đoán:*
nếu bỏ NaN-guard và chạy bf16+compile 10k step — cái gì được ghi vào `history`, checkpoint lưu gì, và bạn phát hiện
hỏng ở đâu (và mất bao nhiêu compute) so với fail-loud @step5? Vì sao guard đặt ở nhịp log là "free".

**5. Frontier.** F4 (FRONTIER_2026 §F4, dòng 133–140): bf16-autocast + compile là default MFU 2026; NaN-guard là
kỷ luật "silent-divergence triage" (FRONTIER_PRACTICE §babysit dòng 76: "failure mode chased = SILENT DIVERGENCE và
LOW-PRECISION BLOWUP"). Neo MFU: Llama-3 báo ~40–50% MFU dense trên H100; PaLM anchor **46.2% MFU** (reproduced by
`utils/mfu.py`, RESULTS-adjacent — perf roadmap/S6). Cross-link perf: launch-collapse chi tiết = roadmap/S3 (kernel
fusion) + S1 R1 (eager 51→compiled 173→cudagraph 253 tok/s = 20%→53%→77% wall). Câu interview: "compile+bf16 NaN
trên box A nhưng không box B — bạn debug thế nào để phân biệt lỗi-của-bạn với codegen bug của compiler?"

---

## Bài 4.5 — Fixed-seed reproducibility + monitors: tin cậy và quan sát (`utils/seeding.py` · `seed_everything` :15 · `utils/monitors.py` · `build_snapshot` :131)
> **Câu hỏi first-principles:** vì sao KHÔNG có fixed-seed thì không ablation nào đáng tin, và tối thiểu phải
> LOG những gì để một RL run *đọc được* (không phải "chạy xong rồi đoán")?
> **Neo (invariant + đo thật):** cùng seed → **loss history identical** (test `test_train.py` reproducibility);
> Phase-0 speedrun (BPE→pretrain→eval→sample) đo **val_bpb 0.0206** (RESULTS.md:666, MEASURED); monitor
> `kl_train_infer` HALT threshold = **0.10** (`monitors.py` :32) — trên ngưỡng ⇒ run KHÔNG đo policy đang serve.

**1. Feynman — bài toán bằng lời.** *Seed:* mọi RNG (init trọng số, sampling batch, dropout, shuffle) phải cố
định để một re-run tái tạo *đúng con số*. Không có nó, bạn không phân biệt được "hiệu ứng thật" với "RNG drift" —
mọi ablation vô nghĩa. *Monitors:* một RL run mà *không log* thì như lái xe đêm không đèn — loss xuống nhưng bạn
không biết model đang *reward-hack* (viết dài để ăn điểm), hay train-engine và serve-engine đã *lệch policy*. Đánh
đổi: log tốn vài phép numpy/step nhưng biến run từ "uninterpretable" thành "chẩn đoán được". Nguyên tắc: **thiếu
MỘT guardrail bắt buộc = run không đọc được**, nên type ép đủ mọi field.

**2. Dẫn xuất từ đầu (derive).** *Seed:* `seed_everything` seed CẢ ba nguồn — `random` (python), `np.random`
(batch sampling ở `get_batch`), `torch.manual_seed` (init + dropout) + `cuda.manual_seed_all` (kernel RNG). Thiếu
một nguồn = một khe non-determinism. Caveat đã ghi (train.py docstring :124–126): `train` reseed RNG *data-sampling*,
nhưng *weight init xảy ra TRƯỚC `train`* — muốn e2e-reproducible phải seed TRƯỚC khi dựng `model`. *Monitors (derive
tối thiểu-phải-log):* ba KL riêng biệt vì mỗi cái đo một drift khác: `KL(current‖ref)` = độ lệch khỏi model gốc (KL
penalty), `KL(current‖old)` = độ lệch off-policy (PPO trust region), `kl_train_infer = KL(train‖infer)` = train-engine
vs serve-engine trên CÙNG trọng số — cái *load-bearing* nhất: nếu >0.10, mọi gradient tính trên một policy KHÁC policy
đang served ⇒ HALT. Cộng: IS-ratio (histogram + ESS — off-policy blow up khi ESS→1), reward-dist, length-dist (tell
của verbosity reward-hacking). `MonitorSnapshot` (:107) là frozen dataclass **mọi field required** — vắng một cái =
run uninterpretable, nên *type* ép, không để quên ngầm.

**3. Trace code.** `seed_everything(seed)` (:15): `random.seed` → `np.random.seed` → `torch.manual_seed` →
`if cuda.is_available(): cuda.manual_seed_all` (:20–21). Gọi ở `train` :132. Monitors (`monitors.py`): `log_softmax`
(:35, stable) → `kl_divergence(log_p, log_q)` (:42, Σ p·(log p − log q), enforce 0·log0=0 :50) → `mean_kl` (:54).
`importance_ratios` (:59) = exp(logp_cur − logp_old); `effective_sample_size` (:67) = (Σw)²/Σw²; `normalized_ess`
(:79) ∈[0,1] (<0.1 ⇒ gradient không tin được). `build_snapshot` (:131, keyword-only all-required) bundle →
`MonitorSnapshot` (:107) với `.halt` property (:126) = `kl_train_infer > 0.10`. Phase-0 loop: `speedrun.py` +
`eval/` báo `val_bpb` (RESULTS.md:656–673) — khép vòng BPE→pretrain→eval→sample, seed cố định để tái lập.

**4. Cổng teach-back.** (a) Vì sao PHẢI log ba KL *riêng* — mỗi cái loại trừ chế độ hỏng nào, và vì sao
`kl_train_infer` là cái đầu tiên khiến bạn HALT? (b) Vì sao `seed_everything` trong `train` là *chưa đủ* cho e2e-
reproducibility (init ở đâu)? (c) *Sửa-và-đoán:* nếu reward tăng đều nhưng `length_p90` cũng phình và `reward_std`
co lại — model đang học gì, và monitor nào bạn nhìn TRƯỚC? (Gợi: verbosity reward-hack; length-dist là tell.) Nếu
`is_ratio_ess` (normalized) tụt về 0.05 — gradient estimate còn dùng được không?

**5. Frontier.** Đây là **CS336 disciplines #3 (fixed-seed) + #4 (mandatory RL logging)** (CLAUDE.md) = đúng bộ
GDM-aligned FRONTIER_PRACTICE §babysit (dòng 76): "fixed seed làm một spike reproducible; log entropy + KL riêng +
IS-ratio + length". `kl_train_infer` = gốc cơ học của train↔serve drift (Bài 4.2: hai engine reduce softmax/PV khác
precision → logits khác) — chính lỗ hổng mà `bench/RESULTS.md:383` phơi bày (serving đo trên random-weight toy).
So frontier: DeepSeek-R1 (2501.12948) GRPO log chính bộ này; Dr.GRPO sửa length-bias mà `length_p90` monitor bắt.
Câu interview: "RL run của bạn reward tăng nhưng eval tệ đi — ba thứ bạn nhìn trong log, và mỗi thứ chẩn đoán gì?"

---

*Đóng série:* vòng lặp = sáu-động-tác-một-nhịp (4.1) + ba đòn bẩy efficiency (bf16 numerics 4.2 · recompute-vs-store
bộ nhớ 4.3 · compile launch-overhead 4.4, với F4 NaN honesty) + tin-cậy/quan-sát (4.5). Mỗi mắt xích có một
invariant-oracle đo được — đó là *toàn bộ* lý do lab screen bằng chúng. Neo khép vòng: overfit<1e-2 (wiring đúng) →
bf16 loss 9e-4 (numerics đúng) → val_bpb 0.02 Phase-0 (LOOP CLOSES, model biết nói). Série sau M5: post-training
(SFT→EI→GRPO/Dr.GRPO) tựa TRỰC TIẾP lên monitors 4.5 + optimizer M2; frontier ablations F1 (Muon iso-FLOP) → F9
dùng chính `train.py` + seed + monitors này làm substrate đo.
