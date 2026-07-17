# M4 — Training Loop + Efficiency · First-Principles Derivations

> **Đây là gì.** Bản dẫn-xuất **từ nguyên lý gốc** (first principles) cho TOÀN BỘ vòng lặp huấn luyện +
> bốn đòn bẩy efficiency — 5 micro-concept M4 (Bài 4.1→4.5). Mỗi mục: (1) **câu hỏi** falsifiable, (2)
> **sự thật nền tảng** phải giải thích, (3) **dẫn xuất** từ đó (có toán), (4) **neo code** `file·func·line`,
> (5) **hình ảnh** (ASCII + shape/stride/dtype + micro-example), (6) **số đo THẬT** (chạy trên chính repo
> này, không phán), (7) **frontier framing** + cổng phỏng vấn.
>
> **Cách dùng để re-learn.** Đọc phần *Câu hỏi* của mỗi mục → **tự trả lời cold** (che phần dưới) → mở ra
> đối chiếu. Đây là PRR loop (Predict → Run → Reconcile): bạn *đoán con số trước*, rồi chạy `python -c`,
> rồi chỉ học *phần lệch*. Cuối doc có **checklist recall cold** + bảng số đo. Đây là bạn đồng hành của
> `roadmap_model/M4_training_loop.md` (reference chung, dẫn-xuất-vắn) — doc này là *derivation lab* có số đo.
>
> **Cảnh báo neo.** Line number **trôi** theo commit. Cite ở đây pin theo **HEAD ngày 2026-07-14**
> (`train.py`, `utils/{mixed_precision,checkpointing,seeding,monitors}.py`). Roadmap gốc pin `4ad0ac5`
> (số dòng KHÁC — vd roadmap ghi `train` :114, HEAD là :208). Nếu lệch, `grep` tên hàm — đừng tin số dòng
> cứng (luật repo: verify, don't trust).
>
> **Nguồn số đo.** Mọi số CPU trong doc này chạy lại được bằng `python -c`/`scratchpad/measure_m4.py`; số
> GPU (bf16+compile NaN, val_bpb, iso-FLOP) là ledger-cited từ `bench/RESULTS.md` §Frontier ablations
> (sm120 Blackwell) và đánh dấu **[FACT sm120]** hoặc **[PREDICTED]**. Đó là "DoD là một profile, không
> phải test xanh" (FOP-3) áp cho việc học.

---

## Bức tranh lớn — vòng lặp train là cái động cơ gì?

M1–M3 dựng *cái model* (byte→BPE→block: RMSNorm·RoPE·SwiGLU·GQA) và *cái optimizer* (AdamW/Muon). Nhưng
model với trọng số ngẫu nhiên **không biết gì**. M4 là **động cơ biến trọng số ngẫu nhiên → policy đã học**:
một vòng lặp `forward→loss→backward→clip→step→zero_grad` lặp hàng nghìn bước, cộng bốn đòn bẩy để nó *chạy
được ở quy mô thật* và *đáng tin*.

```
        ┌──────────────────────── train(cfg, data, model) ────────────────────────┐
        │  seed_everything(cfg.seed)  [4.5 tin cậy]  ─► build_optimizer (M2)        │
        │  forward_model = compile(model) if cfg.compile else model  [4.4 launch]   │
        │                                                                            │
        │  for step in range(max_steps):                                             │
        │    lr = cosine_lr(step,…)  ──► group["lr"]=lr        (M2 schedule)          │
        │    inputs,targets = get_batch(data,…)   [4.1 memmap · next-token align]    │
        │    zero_grad()                    ◄── (1) xoá grad cũ (.backward CỘNG DỒN)  │
        │    with autocast(bf16):           [4.2 mixed precision, KHÔNG GradScaler]   │
        │        loss = cross_entropy(forward_model(inputs), targets)  ◄── (3)(4)     │
        │    loss.backward()                ◄── (5) grad (activation ckpt [4.3] ở đây) │
        │    gradient_clipping(params, clip) ◄── (5.5) chặn outlier batch (M2)         │
        │    optimizer.step()               ◄── (6) cập nhật trọng số                 │
        │    if step % log_every == 0:                                                │
        │        loss_value = loss.item()   ◄── one host-sync/interval                │
        │        if not isfinite(loss_value): RAISE   [4.4 NaN guard, fail-loud]      │
        │        history.append((step, loss_value))    [4.5 reproducible history]     │
        └────────────────────────────────────────────────────────────────────────────┘
              │
              ▼  overfit-one-batch <1e-2 (wiring đúng) → bf16 loss 9e-4 (numerics đúng)
                 → val_bpb 0.0206 Phase-0 (LOOP CLOSES — model biết nói)
```

**Sáu-động-tác-một-nhịp (4.1)** là bộ xương; **ba đòn bẩy efficiency** treo lên nó:

- **4.2 mixed precision** = đòn bẩy **numerics/throughput**: bf16 cho GEMM (tensor core), fp32 cho *cộng dồn
  dài* (accum/loss/master). Áp lực: HBM bandwidth + tensor-core FLOP.
- **4.3 activation checkpointing** = đòn bẩy **bộ nhớ**: vứt activation ở forward, tính lại ở backward. Áp
  lực: HBM capacity (activation O(depth·per_block) là phần lớn nhất lúc train).
- **4.4 torch.compile** = đòn bẩy **launch-overhead**: fuse nhiều op → ít CUDA launch. Áp lực: kernel-launch
  latency (step nhỏ là overhead-bound).
- **4.5 seed + monitors** = **tin cậy + quan sát**: không có seed → không ablation nào đáng tin; không có
  monitor → RL run "uninterpretable".

Sợi chỉ xuyên suốt: **mọi thứ ở đây là "plumbing" — nhưng plumbing SAI thì mọi tầng phía trên vô nghĩa**.
Nên mỗi component có một **invariant-oracle đo được** (overfit<1e-2, checkpoint round-trip, seed-reproducible,
grad-parity, NaN-guard) — đó là *chính xác* bộ "first-line triage" mà lab dùng để screen ứng viên. Học M4 =
học *áp lực vật lý → đòn bẩy → oracle bảo vệ*.

---

## 4.1 · Một train step + memmap + overfit-one-batch oracle

**Câu hỏi.** Thứ tự SÁU thao tác trong một bước SGD là gì, vì sao ĐÚNG thứ tự đó (đổi chỗ hai cái bất kỳ hỏng
ra sao), và vì sao *overfit-one-batch* là oracle số 1 để biết wiring này đúng — trước khi đốt một giây compute?

**Sự thật nền tảng.** Vòng lặp này **đơn giản đến mức dễ SAI NGẦM**: một dấu shift lệch trong `get_batch`, một
`zero_grad` đặt sai chỗ, một loss cộng nhầm aux — **không crash**, chỉ *học chậm hoặc không học*. Không có
exception nào bắt lỗi ngữ nghĩa. Nên nghề không phải "viết vòng lặp" mà là **gắn mỗi mắt xích một invariant đo
được** và không bao giờ tin "nó chạy" = "nó đúng".

**Dẫn xuất.**

*(a) Next-token alignment — vì sao `targets` phải dịch một ô.* LM học phân phối `P(x_{t+1} | x_{≤t})`. Vậy nhãn
tại vị trí `t` phải là *token kế* của input tại `t`:
```
inputs [b,t]  =  data[s+t]
targets[b,t]  =  data[s+t+1]        ⟹   invariant:  targets[b,t] == inputs[b,t+1]
```
Lệch một ô = model học ánh xạ *identity* (copy input), loss kẹt ở một mức vô nghĩa. Đây là bug im lặng cổ điển.

*(b) Vì sao `np.memmap`.* Corpus pretraining (hàng chục–trăm tỉ token) **lớn hơn RAM**. `np.memmap` ánh xạ file
trên đĩa vào không gian địa chỉ ảo; OS phân trang **lazy** — chỉ trang nào `data[s:s+L]` chạm mới nạp. Ta không
bao giờ `np.load` trọn corpus. `get_batch` sinh `starts` ngẫu nhiên rồi slice — mỗi slice là một page-fault nhỏ.

*(c) Thứ tự sáu-động-tác — derive vì sao KHÔNG hoán vị được.*
```
(1) get_batch      lấy (inputs, targets)
(2) zero_grad      TRƯỚC backward — vì .backward() CỘNG DỒN vào .grad (không ghi đè). Quên xoá
                   = grad của batch này chồng lên batch trước → bước đi theo hướng trộn hai batch.
(3) forward        logits = model(inputs)
(4) loss           cross_entropy(logits, targets)
(5) backward       điền .grad cho mọi param (chain rule)
(5.5) clip         GIỮA backward và step — clip đọc .grad ĐÃ tính xong nhưng TRƯỚC khi step dùng nó.
                   global-ℓ2 clip (M2) chặn một batch outlier làm nổ một bước.
(6) step           optimizer đọc .grad, cập nhật trọng số.
```
Vì sao `zero_grad` trước `backward` chứ không sau `step`? Về mặt hiệu ứng, "sau step" cũng đúng NẾU mỗi step
đúng một backward. Nhưng khoảnh khắc bạn micro-batch (nhiều backward/step để accumulate grad), "sau step" xoá
mất phần tích luỹ. Đặt `zero_grad` ngay đầu step là **bất biến an toàn** với mọi cấu hình. `clip` phải nằm *giữa*
vì nó là phép biến đổi *trên* `.grad`: trước backward thì chưa có grad để clip; sau step thì đã cập nhật xong,
clip vô nghĩa.

*(d) Overfit-one-batch — derive vì sao là oracle #1.* Lấy MỘT batch cố định, chạy vòng lặp lặp đi lặp lại chỉ
trên nó. Nếu tối ưu KHÔNG kéo được loss về ~0, thì lỗi ở **wiring** (optimizer/loss/grad-flow), **KHÔNG** ở data
— vì một batch cố định *luôn học thuộc lòng được* nếu gradient chảy đúng (đủ tham số để nội suy 4×16 nhãn). Nó
**cô lập** lỗi wiring khỏi lỗi data *trước khi* đốt compute thật. Cặp với loss-at-init ≈ `log V` (Bài 1.1/2.6):
init đúng → CE đều → `−log(1/V) = log V`; lệch xa ⇒ bug head/embedding/mask.

**Neo code** (`src/scratch_llm/train.py`):
```python
def get_batch(data, batch_size, context_length, device="cpu"):     # :40
    max_start = len(data) - context_length - 1                     # :51  chừa 1 cho shift
    starts = np.random.randint(0, max_start + 1, size=batch_size)  # :56  RNG (seed ở 4.5)
    inputs  = np.stack([data[s : s + context_length]     for s in starts])  # :57
    targets = np.stack([data[s + 1 : s + 1 + context_length] for s in starts])  # :58  shift +1
    return torch.from_numpy(inputs).long().to(device), torch.from_numpy(targets).long().to(device)
# --- trong train() ---
inputs, targets = get_batch(train_data, cfg.batch_size, cfg.context_length, cfg.device)  # :266
optimizer.zero_grad()                                              # :267  (2) TRƯỚC backward
with amp_ctx:                                                      # :273  (4.2)
    loss = cross_entropy(forward_model(inputs), targets)          # :279  (3)(4)
loss.backward()                                                   # :280  (5)
gradient_clipping(model.parameters(), cfg.grad_clip)             # :281  (5.5) GIỮA
optimizer.step()                                                 # :282  (6)
```
`save_checkpoint` (:65) lưu `{model, optim, step, model_config}` (:82–88); `load_checkpoint` (:93) dùng
`weights_only=False` (:102) vì ta nạp checkpoint *tin cậy* của chính mình có metadata optimizer. Test pin:
`tests/test_train.py` — next-token align, round-trip, overfit<1e-2, học <log V.

**Hình ảnh — data journey của một batch:**
```
np.memmap file (ONDISK, int64)  data = [.. 172 173 174 175 176 177 ..]   len N ≫ RAM
        │  starts = randint  →  s=172
        ▼
inputs [b, :]  = data[172:177]  →  [172,173,174,175,176]   shape (B=2, L=5) long   (page-fault lazy)
targets[b, :]  = data[173:178]  →  [173,174,175,176,177]   shape (B=2, L=5) long
        │  torch.from_numpy(...).long()               dtype int64, contiguous
        ▼
forward_model(inputs) → logits (B, L, V) float32   ── cross_entropy ──►  loss (scalar)
        invariant CHECK:  targets[b,t] == inputs[b,t+1]   (173==173, 174==174, …)  ✓
```

**Số đo THẬT** (`scratchpad/measure_m4.py`, CPU):
```
memmap dtype/shape: int64 (200,)  ON-DISK (not RAM-loaded)
inputs[0]  = [172, 173, 174, 175, 176]
targets[0] = [173, 174, 175, 176, 177]      targets[b,t]==inputs[b,t+1] ∀ : True
loss@init  = 6.9877   log(V=1000) = 6.9078   ratio = 1.012          ⇒ init đúng, CE đều
overfit-one-batch: loss 4.4050 (init) → 8.91e-04 (step 300)         ⇒ <1e-2 oracle PASS (wiring đúng)
```
GPU ledger (RESULTS.md, sm120): hybrid MuonAdamW đẩy 1 batch xuống **<0.05 trong 300 step** — cùng budget/threshold.

**Frontier / cổng.** Ba oracle (loss-at-init≈log V, overfit-one-batch, fixed-seed) = **CS336 Engineering
disciplines #1–3** = đúng bộ "first-line triage" `FRONTIER_PRACTICE_2026 §babysit`. So nanochat `speedrun.sh`:
cùng bộ xương forward→loss→backward→step; ta thêm NaN-guard (4.4) + MoE aux nudge. **Gate phỏng vấn:** "Model
không học — ba thứ bạn kiểm theo thứ tự nào, mỗi thứ loại trừ giả thuyết gì?" (loss@init loại bug head/mask;
overfit loại bug wiring; seed-repro loại RNG drift). Trait = **cheapest-oracle discipline + first-principles**.
Table-stakes (nhưng thiếu nó = silent reject).

---

## 4.2 · Mixed precision — bf16 autocast, vì sao KHÔNG cần GradScaler

**Câu hỏi.** Vì sao huấn luyện low-precision cần fp32 ở ĐÚNG vài chỗ (accum/master/loss), và vì sao **bf16
KHÔNG cần loss-scaling** trong khi **fp16 thì CẦN**?

**Sự thật nền tảng.** Số dấu phẩy động có **độ phân giải TƯƠNG ĐỐI**: gần 0 thì mịn, xa 0 thì thô. Khoảng cách
hai số kề (ulp — *unit in the last place*) tỉ lệ với độ lớn: `ulp(x) = 2^(floor(log2 x) − mantissa_bits)`. bf16
có **7 bit mantissa**, fp16 có **10**, fp32 có **23**. Ít mantissa ⇒ ulp thô ⇒ *cộng số nhỏ vào số lớn bị làm
tròn về no-op*. Analogy: thước chỉ có vạch cm; cộng 3mm vào 40cm → làm tròn về 40cm, "3mm bốc hơi".

**Dẫn xuất.**

*(a) Vì sao accum phải fp32 — small-addend underflow.* Tổng tuần tự `acc += inc` **đóng băng** khi
`inc < ulp(acc)/2` (cộng làm tròn về `acc`). Với fp16, tại `acc = 32`:
```
ulp(32) = 2^(floor(log2 32) − 10) = 2^(5−10) = 2^-5 = 0.03125
half-ulp = 0.015625  >  inc = 0.01   ⟹  32 + 0.01  ⟶  round  ⟶  32   (KẸT VĨNH VIỄN)
```
Đây là *lý do* optimizer state (momentum/variance), loss, và reduction giữ **fp32** dù model chạy bf16: chúng là
những *tổng chạy dài* mất số hạng nhỏ nếu ở low-precision. Điều thú vị đo được: **bf16 (7 mantissa) đóng băng
SỚM HƠN (ở 4.0) so với fp16 (10 mantissa, ở 32.0)** — càng ít mantissa, ulp càng thô, freeze càng sớm ⇒ bf16
càng *cần* fp32-accum. (`naive_accumulate` :28 cố ý dùng vòng *tuần tự* để phơi lỗi; `torch.sum` dùng pairwise
tree reduction chính xác hơn nhiều — vòng tuần tự CHÍNH LÀ bài học.)

*(b) Vì sao bf16 KHÔNG cần GradScaler còn fp16 CẦN — exponent range.* Đây là điểm mấu chốt, và nó KHÔNG phải về
mantissa mà về **số bit exponent** (dải mũ = dải số biểu diễn được):
```
        exponent bits   smallest_normal      max          mantissa bits
fp16          5           6.10e-05          6.55e+04           10
bf16          8           1.18e-38          3.39e+38            7
fp32          8           1.18e-38          3.40e+38           23
```
fp16 có 5 bit exponent → smallest normal ~6.1e-5. Gradient trong mạng sâu thường ~1e-7…1e-8 → **underflow về 0**
trong fp16. GradScaler *nhân loss lên* (vd ×2^16) trước backward để đẩy grad vào dải biểu diễn được, rồi *chia
lại* sau khi unscale — một vũ điệu chỉ để cứu dải mũ hẹp của fp16. **bf16 có 8 bit exponent = ĐÚNG BẰNG fp32**
→ cùng dải mũ (tới ~1e-38), chỉ mất *precision* (mantissa), **không underflow** → **khỏi cần scaling**. Đó là lý
do `train.py` guard `amp_dtype ∈ {None, "bf16"}` và từ chối fp16 với thông điệp "fp16 needs a GradScaler".

*(c) Autocast dtype rule.* Dưới `torch.autocast`, PyTorch chọn dtype *per-op* theo policy có sẵn: matmul → bf16
(throughput tensor-core), còn LayerNorm/softmax/reduction + master weights → fp32 (ổn định). Ta **không** chọn
dtype bằng tay. Trong `train()`, chỉ *forward + loss* nằm trong `with amp_ctx`; `backward/clip/step` NGOÀI context
→ grad tích luỹ ở dtype của param = **fp32 master**.

**Neo code:**
```python
# utils/mixed_precision.py
def naive_accumulate(value, count, dtype):              # :28  vòng TUẦN TỰ cố ý (phơi underflow)
    acc = torch.zeros((), dtype=dtype)                  # :33
    inc = torch.tensor(value, dtype=dtype)              # :34
    for _ in range(count):                              # :35
        acc += inc                                      # :36  freeze khi inc < ulp(acc)/2
    return acc                                          # :37
# train.py
if cfg.amp_dtype not in (None, "bf16"):                 # :231  fp16 bị từ chối
    raise ValueError("amp_dtype must be None or 'bf16' (fp16 needs a GradScaler); …")  # :232
amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else None   # :257
amp_ctx = (torch.autocast(device_type=device_type, dtype=amp_dtype)  # :268
           if amp_dtype is not None else nullcontext())            # :271
with amp_ctx:                                           # :273  CHỈ forward+loss autocast
    loss = cross_entropy(forward_model(inputs), targets)           # :279
loss.backward()                                        # :280  NGOÀI autocast → grad ở fp32 master
```

**Hình ảnh — vì sao 0.01 bốc hơi ở 32.0 (fp16):**
```
acc:  0 ──+0.01──► 0.01 ──…──► 2.0 ──…──► 16.0 ──…──► 32.0
                                                        │  ulp(32)=0.03125, half=0.0156
                                                        │  inc=0.01 < 0.0156
                                                        ▼
                                       32.0 + 0.01 = 32.0  (round-to-nearest-even → no-op)
                                                        │  ∞ lần cộng nữa
                                                        ▼
                                                     32.0  ◄── ĐÓNG BĂNG

dtype  mantissa  freeze@   |  autocast(cpu,bf16):  matmul → bf16  (tensor-core)
fp16     10       32.0     |                       layer_norm → fp32 (stability)
bf16      7        4.0     |  exponent: bf16(8)=fp32(8) ⟹ no underflow ⟹ no GradScaler
```

**Số đo THẬT** (CPU):
```
naive_accumulate(0.01 ×100000):  fp16 = 32.000000   bf16 = 4.000000   fp32 = 1000.665039   (exact 1000.0)
fp16 running-sum FREEZES at 32.0   ulp(32)=2^-5=0.03125   half=0.0156 > inc 0.01 ⇒ add no-op
grad 1e-8:  fp16 → 0.000e+00 (UNDERFLOW)   bf16 → 1.001e-08 (survives)   [fp16 subnormal floor ~5.96e-8]
smallest_normal: fp16 6.10e-05   bf16 1.18e-38 (= fp32)   ⇒ vì sao bf16 khỏi GradScaler
autocast(cpu,bf16): matmul out dtype = torch.bfloat16   layer_norm out dtype = torch.float32
```
GPU ledger: muon_adamw · **bf16** · eager → **loss → 9e-4 (LEARNED, không GradScaler)** [FACT sm120, RESULTS.md].

**Frontier / cổng.** Câu interview kinh điển (docstring `mixed_precision.py`): *"bạn train fp16 và loss
plateau/NaN — vì sao?"* → accumulation precision + autocast policy (+ một bậc thấp hơn: FP8 amax/scale tràn).
Đây cũng là gốc cơ học của **train↔serve logit drift** (`monitors.py::kl_train_infer`, 4.5): hai engine reduce
softmax/PV ở precision khác → logits khác trên CÙNG trọng số. Frontier: bf16-autocast = default 2026; DeepSeek-V3
(`2412.19437`) train **FP8** với per-tile/per-128 block scaling — chính là "fp32 ở đúng chỗ" đẩy xuống một
precision nữa. Trait = **predict-the-number** (bạn dựng lại 32.0 từ ulp trước khi chạy). Scarce (kernels/perf-adj).

---

## 4.3 · Activation checkpointing — recompute-vs-store cho bộ nhớ

**Câu hỏi.** Bộ nhớ activation lúc train là O(depth · per_block) — làm sao giảm xuống ~O(depth · residual) mà
autograd **vẫn ra ĐÚNG gradient**, và tại sao "selective" là default 2026?

**Sự thật nền tảng.** Forward lưu MỌI activation trung gian để backward dùng lại (chain rule cần chúng: `∂(Wx)/∂W`
cần `x`, `∂softmax/∂z` cần output softmax…). Đó là phần bộ nhớ **lớn nhất** lúc train, tỉ lệ `depth × width ×
batch × seq`. Ở microbatch lớn nhất, chính activation — không phải weight — làm OOM.

**Dẫn xuất.**

*(a) Trade cốt lõi: đổi compute lấy bộ nhớ.* Thay vì *lưu* activation, ta **vứt** chúng ở forward rồi **tính lại**
ở backward. Analogy: giải toán nhiều bước — thay vì giữ mọi kết quả trung gian trên giấy (tốn giấy), chỉ giữ đề
mỗi phần rồi *làm lại* phần đó khi cần. Trả thêm ~một forward để đổi lấy bộ nhớ. Ba mức span đúng trade:
```
none       lưu HẾT activation                 max memory,  min compute   (eager default)
full       chỉ lưu INPUT mỗi block, tính lại  ~O(depth·residual),  +≈1 forward
           cả block ở backward
selective  lưu cái ĐẮT-tính-lại-RẺ-lưu (matmul output), tính lại cái RẺ (softmax/elementwise)
```

*(b) Vì sao ra ĐÚNG gradient.* Checkpoint là một `autograd.Function`: forward chạy **`no_grad`** (không dựng
graph, không lưu activation); backward **chạy lại forward CÓ graph** rồi backprop qua nó. Điều kiện sống-còn:
forward phải **deterministic** (cùng input → cùng activation). Dropout với seed khác giữa forward gốc và forward
recompute ⇒ grad tính lại ≠ grad thật ⇒ SAI. (Fix chuẩn: lưu + restore RNG state quanh recompute; model này
không có dropout nên vô hại. `use_reentrant=False` là đường modern compose được với `saved_tensors_hooks`.)

*(c) Vì sao "selective" là điểm ngọt.* Xét từng op:
```
recompute một GEMM (matmul):  TỐN nhiều FLOP (đắt),  nhưng LƯU output nó RẺ (một tensor)
recompute softmax/elementwise: RẺ FLOP,             lưu cũng rẻ
```
⇒ chính sách tối ưu: **MUST_SAVE matmul output, PREFER_RECOMPUTE phần còn lại** = tiết kiệm bộ nhớ tối đa với
FLOP thêm tối thiểu. Korthikanti/Megatron đo: **~70% activation-memory cho ~2.7% extra FLOP** @GPT-3. Đây *cùng
bản chất* với FA2-backward (không lưu ma trận score N×N, tính lại) — chỉ khác **chỗ đặt ranh giới** (cả block vs
trong attention).

*(d) Hai TRỤC đo tách biệt.* (i) **Bộ nhớ**: bytes activation lưu qua *ranh giới ngoài* checkpoint. (ii)
**Compute**: số matmul chạy (recompute làm nó *tăng ở backward*). Điểm tinh tế: SAC cache matmul-output *bên
TRONG* region → ẩn với hook ngoài → ở TRỤC BỘ NHỚ `full ≈ selective`; khác biệt full-vs-selective lộ ở TRỤC
COMPUTE (số matmul recompute). Phải đo cả forward LẪN backward vì recompute xảy ra ở backward.

**Neo code** (`src/scratch_llm/utils/checkpointing.py`):
```python
MATMUL_OPS = frozenset({aten.mm, aten.addmm, aten.bmm, aten.matmul})   # :43  one cheap out, expensive GEMM
def _sac_policy(_ctx, op, *a, **k):                                     # :56
    if op in _SAVE_OPS:  return CheckpointPolicy.MUST_SAVE             # :58  matmul-out: lưu
    return CheckpointPolicy.PREFER_RECOMPUTE                           # :60  còn lại: tính lại
def run_block(block, mode, *args):                                     # :63
    if mode == "none":      return block(*args)                        # :67
    if mode == "full":      return checkpoint(block, *args, use_reentrant=False)   # :69
    if mode == "selective": return checkpoint(block, *args, use_reentrant=False,   # :71
                context_fn=lambda: create_selective_checkpoint_contexts(_sac_policy))  # :75
def pack(t):                                                           # :93  đo TRỤC bộ nhớ
    ptr = t.untyped_storage().data_ptr()                              # :94  dedup theo storage
    if ptr not in seen: total["bytes"] += t.untyped_storage().nbytes()  # :97
# count_op_executions (:121) bọc CẢ fwd+bwd (:128) — recompute ở backward mới đếm được
```

**Hình ảnh — ba mức trên một stack 3 block:**
```
              forward giữ gì ở boundary?              backward làm gì?
none      [x0 A0 B0 x1 A1 B1 x2 A2 B2 …] LƯU HẾT     đọc thẳng          → 767,808 B, 81 matmul
full      [x0]      [x1]      [x2]  chỉ INPUT block   RE-RUN cả block    →  12,352 B, 105 matmul (+24 recompute)
selective [x0 +matmul-outs bên trong region ẩn]      re-run softmax/elt →  12,352 B,  81 matmul (0 extra)
                    ▲ cache trong region, hook ngoài không thấy ⇒ bộ nhớ ≈ full,
                      nhưng KHÔNG re-run GEMM ⇒ compute = none
```

**Số đo THẬT** (`d_model=32, n_layers=3, seq=8`, fp64, CPU):
```
saved-activation bytes (fwd→bwd boundary):  none 767,808   full 12,352   selective 12,352   ⇒ 62× reduction
matmul executions (fwd+bwd):                none 81        full 105 (+24 recompute)   selective 81 (0 extra)
grad parity vs eager:  full max|Δ| = 0.00e+00   selective max|Δ| = 0.00e+00   ⇒ trong suốt với autograd
```
Ba số này khớp `tests/test_checkpointing.py`: bytes `full<selective<none` + grad-parity fp-tolerance.

**Frontier / cổng.** `selective` = default 2026 (Megatron/TorchTitan; TorchTitan cho chọn full/selective
per-layer). DeepSeek/Llama pipeline dùng để nhét batch lớn hơn vào cùng HBM. Cross-link perf: cùng
recompute-vs-store là bản chất **FA2 backward** (roadmap/S4). **Gate phỏng vấn:** "Bạn OOM ở microbatch lớn nhất
— hai đòn bẩy bộ nhớ KHÔNG đụng model, mỗi cái đổi compute/comm ra sao?" (activation checkpointing +2.7% FLOP ·
ZeRO/FSDP shard +comm). Trait = **trade-off-aware + roofline-first** (đặt tên trục memory vs compute). Scarce (systems).

---

## 4.4 · torch.compile + F4 bf16+compile NaN guard

**Câu hỏi.** `torch.compile` giúp gì (đòn bẩy launch-overhead), vì sao trên box này **bf16 đúng-riêng và compile
đúng-riêng nhưng KẾT HỢP thì NaN**, và một NaN-guard *đúng chỗ* trông thế nào?

**Sự thật nền tảng.** Eager PyTorch phóng **một CUDA launch mỗi op** — hàng trăm launch/step, mỗi cái ~µs dispatch
+ một round-trip HBM cho input/output. Với step nhỏ (hoặc decode), thời gian bị *overhead-bound*, không phải
compute-bound. `torch.compile` (inductor) **fuse** nhiều pointwise op thành ít kernel Triton → ít launch, ít HBM
round-trip. Analogy: thay vì gọi thợ riêng cho từng viên gạch, giao cả bức tường một lần.

**Dẫn xuất.**

*(a) Vì sao compile giảm wall-time.* Step overhead-bound: `wall ≈ Σ launch_latency`. Fuse K op thành 1 kernel →
collapse K launch thành 1 → BW đo được tăng. (perf R1: eager 51 → compiled 173 tok/s = 15%→53% HBM; cudagraph
253 = 77% wall — RESULTS.md.) *Vì sao +MFU:* `MFU = FLOP_model / (FLOP_peak · wall)`; compile giảm `wall`
(overhead) mà không đổi `FLOP_model` ⇒ MFU lên. **[PREDICTED]** bf16+compile = **+10–20% MFU** (F4 spec, chưa đo
được vì combo NaN trên box này).

*(b) Vì sao combo NaN — predict-vs-measure honesty.* F4 *dự đoán* bf16+compile là combo MFU rẻ; *đo* ra nó
diverge trên **inductor sm120 / torch-2.12**. Chuỗi verify để kết luận "codegen bug của inductor", KHÔNG phải
logic ta:
```
bf16 · eager          → LEARNED (9e-4)        ⟹ bf16 riêng ĐÚNG
fp32 · compile        → LEARNED (8.3e-4)      ⟹ compile riêng ĐÚNG
any  · bf16 · compile → NaN @~step5–10        ⟹ chỉ TÍCH mới hỏng
  ├─ reproduces với plain AdamW               ⟹ loại trừ Muon
  └─ matmul_precision="high" vẫn NaN          ⟹ loại trừ giả thuyết precision-mode
```
Kết luận: inductor lower sai vùng autocast-bf16 → codegen bug. Bài học vận hành: **đừng tin "mỗi mảnh đúng ⇒
combo đúng"**.

*(c) NaN-guard — derive vì sao đặt ở nhịp log là "free".* Kiểm `isfinite(loss)` cần đọc giá trị loss về host =
một `.item()` = một **host-sync** (chặn GPU). Nếu kiểm mỗi step → mỗi step một sync → giết throughput. Nhưng vòng
lặp **đã** có `loss.item()` ở nhịp `log_every` (để log). Đặt guard *ngay đó* ⇒ **0 sync thêm**. `math.isfinite`
bắt cả `nan` LẪN `inf` (divergence do LR nổ cũng ra inf trước nan). Fail-loud > đốt compute trên NaN im lặng.

**Neo code:**
```python
# train.py
forward_model = torch.compile(model) if cfg.compile else model     # :256  GIỮ model gốc riêng cho
#   .cfg/.moe_update_biases/checkpoint (compiled module có prefix `_orig_mod.`, Muon-split cần submodule thật)
loss = cross_entropy(forward_model(inputs), targets)               # :279  forward qua compiled
gradient_clipping(model.parameters(), cfg.grad_clip)             # :281  clip/step/bias dùng model GỐC
if cfg.log_every and step % cfg.log_every == 0:                   # :292
    loss_value = loss.item()   # the one host sync per log interval  # :293
    if not math.isfinite(loss_value):                            # :294  bắt nan/inf
        raise RuntimeError(                                       # :299
            f"non-finite loss ({loss_value}) at step {step}: training diverged. Check the LR, "
            "or avoid bf16 + torch.compile together on sm120 (an inductor codegen bug).")  # :301
```
Chú ý HAI biến `model` / `forward_model`: forward gọi `forward_model` (:279), nhưng `clip`(:281)/`step`(:282)/
`moe_update_biases`(:290) dùng `model` gốc — vì compiled module bọc state_dict bằng prefix `_orig_mod.` và Muon
phải thấy submodule thật để split 2D-matrix vs 1D-param.

**Hình ảnh — launch collapse + guard placement:**
```
eager:   op1▶launch  op2▶launch  op3▶launch … opN▶launch    (N µs-dispatch, overhead-bound)
compile: [op1·op2·op3 … fused]▶1 kernel                     (1 launch → wall ↓ → MFU ↑)

step:  fwd → loss → bwd → clip → step ──►  step % log_every == 0 ?
                                              │ yes → loss.item()  ◄── sync ĐÃ có sẵn (để log)
                                              │        isfinite? ── no ──► RAISE (fail-loud, 0 sync thêm)
                                              │        yes → history.append
                                              └ no  → (không sync — throughput giữ nguyên)
```

**Số đo THẬT:**
```
CPU:  math.isfinite(nan)=False  isfinite(inf)=False  isfinite(3.14)=True
CPU:  train() với max_lr=1e4 → loss NaN → guard RAISED @ step 2: "non-finite loss (nan) … diverged"
CPU:  torch.compile: eager == compiled forward  (max|Δ| = 5.07e-07)   ⇒ compile đúng khi đứng riêng
GPU ledger [FACT sm120, RESULTS.md]:  bf16·eager 9e-4 ✓ | fp32·compile 8.3e-4 ✓ | bf16·compile → NaN@~step5–10 guard RAISES
```

**Frontier / cổng.** F4 (`FRONTIER_2026 §F4`): bf16-autocast + compile là default MFU 2026; NaN-guard = kỷ luật
"silent-divergence triage". Neo MFU: Llama-3 ~40–50% MFU dense trên H100; PaLM anchor **46.2% MFU** (repro
`utils/mfu.py`). **Gate phỏng vấn:** "compile+bf16 NaN trên box A nhưng không box B — debug thế nào để phân biệt
lỗi-của-bạn với codegen bug của compiler?" (ablate từng feature riêng + reproduce với optimizer khác + loại
precision-mode). Trait = **claims-honesty (predict-vs-measure) + fail-fast**. Scarce (perf/systems).

---

## 4.5 · Fixed-seed reproducibility + monitors

**Câu hỏi.** Vì sao KHÔNG có fixed-seed thì không ablation nào đáng tin, và tối thiểu phải LOG những gì để một
RL run *đọc được* (không phải "chạy xong rồi đoán")?

**Sự thật nền tảng.** *Seed:* mọi RNG (init trọng số, sampling batch, dropout, shuffle) phải cố định để một
re-run tái tạo **đúng con số**. Không có nó, bạn không phân biệt được "hiệu ứng thật" với "RNG drift" → mọi
ablation vô nghĩa. *Monitors:* một RL run *không log* thì như lái xe đêm không đèn — loss xuống nhưng bạn không
biết model đang *reward-hack* (viết dài ăn điểm) hay train-engine đã *lệch policy* khỏi serve-engine.

**Dẫn xuất.**

*(a) Seed — phải seed CẢ BA nguồn.* `seed_everything` seed: `random` (python — vd shuffle), `np.random` (batch
sampling ở `get_batch` :56), `torch.manual_seed` (init trọng số + dropout) + `cuda.manual_seed_all` (kernel RNG).
Thiếu một nguồn = một khe non-determinism. **Caveat load-bearing** (train.py docstring :227–229): `train` reseed
RNG *data-sampling*, nhưng **weight init xảy ra TRƯỚC `train`** (lúc `TransformerLM(cfg)`). Muốn e2e-reproducible
phải `seed_everything` *TRƯỚC* khi dựng `model`, không chỉ trong `train`.

*(b) Monitors — derive "tối thiểu phải log".* **Ba KL riêng biệt** vì mỗi cái đo một drift khác:
```
KL(current‖ref)   = độ lệch khỏi model gốc         → KL-penalty (đừng đi quá xa base)
KL(current‖old)   = độ lệch off-policy             → PPO trust-region (importance-sampling còn hợp lệ?)
kl_train_infer    = KL(train-engine ‖ serve-engine) trên CÙNG trọng số  ← LOAD-BEARING nhất
                    nếu >0.10 ⇒ mọi gradient tính trên một policy KHÁC policy đang served ⇒ HALT
```
`kl_train_infer` là gốc cơ học của train↔serve drift (nối thẳng 4.2: hai engine reduce softmax/PV ở precision
khác → logits khác). Cộng: **IS-ratio** (histogram + ESS — off-policy blow up khi `ESS→1`), **reward-dist**,
**length-dist** (tell của verbosity reward-hacking). `MonitorSnapshot` (:107) là frozen dataclass **mọi field
required** — vắng một cái = run uninterpretable, nên *type* ép, không để quên ngầm.

*(c) Toán của KL + ESS.*
```
KL(p‖q) = Σ_x p(x)·(log p(x) − log q(x)) ≥ 0,   0·log0 ≡ 0,   ASYMMETRIC (KL(p‖q) ≠ KL(q‖p))
ESS     = (Σw)² / Σ(w²),   w = IS-ratio.   Uniform w ⇒ ESS = N;   một w áp đảo ⇒ ESS → 1.
normalized_ess = ESS/N ∈ [0,1];   < 0.1 ⇒ gradient estimate KHÔNG tin được.
```
`log_softmax` (:35) trừ max trước exp (ổn định số); `kl_divergence` (:50) dùng `np.where(p>0, …, 0)` để ép
`0·log0=0`.

**Neo code:**
```python
# utils/seeding.py
def seed_everything(seed):                              # :15
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)   # :17–19  BA nguồn
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)      # :20–21
# train.py: seed_everything(cfg.seed) :240 — reseed data RNG; init đã xảy ra TRƯỚC (caveat :227)
# utils/monitors.py
KL_TRAIN_INFER_HALT = 0.10                               # :32
def kl_divergence(log_p, log_q, axis=-1):                # :42
    p = np.exp(log_p)
    terms = np.where(p > 0, p * (log_p - log_q), 0.0)   # :50  0·log0 = 0
    return terms.sum(axis=axis)
def effective_sample_size(weights):                      # :67
    s2 = float((w**2).sum());  return float(w.sum()**2 / s2)   # :73–76  (Σw)²/Σw²
@dataclass(frozen=True)
class MonitorSnapshot:                                    # :107  MỌI field required
    @property
    def halt(self):  return self.kl_train_infer > KL_TRAIN_INFER_HALT   # :125–128
```

**Hình ảnh — ESS collapse + halt gate:**
```
IS-ratios w (N=100)         ESS = (Σw)²/Σw²        diagnosis
uniform  [1,1,…,1]      →    100 (=N)  norm 1.00    off-policy khoẻ
one big  [100,ε,…,ε]    →    1.0       norm 0.01    ◄ gradient dựa 1 mẫu → KHÔNG tin

kl_train_infer:  0.05 ──► halt=False   |   0.15 ──► halt=True (>0.10)
                         (đo policy served)   (HALT: gradient KHÔNG đo policy đang serve)
```

**Số đo THẬT** (CPU):
```
seed 0 run A: [4.4771, 4.3585, 4.3891, 4.4234]     seed 0 run B: [4.4771, 4.3585, 4.3891, 4.4234]  ⇒ identical
seed 1 run C: [4.4128, 4.4386, 4.4700, 4.4173]     same-seed identical? True   diff-seed differs? True
KL(p‖p) = 0.000000    KL(p‖q) = 0.337026   KL(q‖p) = 0.440012   ⇒ ≥0 và ASYMMETRIC
ESS uniform(N=100) = 100.0 (norm 1.000)   ESS one-dominant = 1.0000 (norm 0.0100 < 0.1 unreliable)
snapshot: kl_train_infer 0.05 → halt=False   |   0.15 → halt=True   (thr 0.10)
```
GPU ledger: Phase-0 speedrun (BPE→pretrain→eval→sample) đo **val_bpb 0.0206** (seed cố định để tái lập) [FACT sm120].

**Frontier / cổng.** = **CS336 disciplines #3 (fixed-seed) + #4 (mandatory RL logging)** = GDM-aligned
`FRONTIER_PRACTICE §babysit`. `kl_train_infer` = lỗ hổng mà `RESULTS.md` phơi bày (serving đo trên random-weight
toy). Frontier: DeepSeek-R1 (`2501.12948`) GRPO log chính bộ này; Dr.GRPO sửa length-bias mà `length_p90` monitor
bắt. **Gate phỏng vấn:** "RL run của bạn reward tăng nhưng eval tệ đi — ba thứ bạn nhìn trong log, mỗi thứ chẩn
đoán gì?" (length_p90 ↑ ⇒ verbosity hack; kl_current_ref ↑ ⇒ trôi khỏi base; ESS ↓ ⇒ off-policy vỡ). Trait =
**observability-first + reproducibility discipline**. RL-logging = **scarce differentiator 2026**.

---

## Bảng số đo THẬT (chạy lại được — DoD-là-profile)

| Concept | Đo | Kết quả | Ý nghĩa |
|---|---|---|---|
| 4.1 | next-token align | `targets[b,t]==inputs[b,t+1]` ∀ → **True** | wiring alignment đúng |
| 4.1 | loss@init | **6.9877 ≈ log1000 = 6.9078** (r 1.012) | CE đều ⇒ init/mask đúng |
| 4.1 | overfit-one-batch | 4.4050 → **8.91e-04** @300 | <1e-2 oracle: wiring đúng |
| 4.2 | naive_accumulate 0.01 | fp16 **32.0** · bf16 **4.0** · fp32 1000.67 | small-addend underflow |
| 4.2 | fp16 freeze point | **32.0** (ulp/2=0.0156 > 0.01) | vì sao accum = fp32 |
| 4.2 | grad 1e-8 | fp16 **→0** · bf16 **1.001e-8** | exponent range ⇒ no GradScaler |
| 4.2 | autocast per-op | matmul **bf16** · layer_norm **fp32** | dtype policy tự động |
| 4.3 | saved bytes | none 767,808 · full/sel **12,352** | **62×** memory cut |
| 4.3 | matmul fwd+bwd | none 81 · full **105** · sel 81 | full recompute +24; sel 0 extra |
| 4.3 | grad parity | full/sel max\|Δ\| = **0.00e+00** | trong suốt với autograd |
| 4.4 | NaN guard | `isfinite(nan)=False`; guard **RAISED@step2** | fail-loud, 0 sync thêm |
| 4.4 | compile alone | eager==compiled max\|Δ\| **5.07e-7** | compile đúng khi riêng |
| 4.4 | bf16+compile | **NaN@~step5–10** [FACT sm120] | codegen bug, không phải logic |
| 4.5 | seed reproducibility | same seed → **identical history** | ablation đáng tin |
| 4.5 | KL | p‖p **0** · p‖q **0.337** · q‖p **0.440** | ≥0, asymmetric |
| 4.5 | ESS | uniform **100** · peaked **1.0** (norm 0.01) | off-policy collapse |
| 4.5 | halt gate | 0.05→False · 0.15→**True** (thr 0.10) | train↔serve drift HALT |
| loop | Phase-0 val_bpb | **0.0206** [FACT sm120] | LOOP CLOSES, model biết nói |

---

## Checklist recall COLD (che phần trên, tự trả lời — đây là cách re-learn)

1. **4.1** Thứ tự sáu-động-tác? Vì sao `zero_grad` TRƯỚC `backward`, `clip` GIỮA `backward` và `step`? (dẫn từ
   ".backward CỘNG DỒN" và "clip là phép trên `.grad`").
2. **4.1** *Sửa-và-đoán:* bỏ shift (`targets = data[s:…]`) — loss@init đổi không? overfit hội tụ về đâu? oracle
   nào bắt được?
3. **4.1** Vì sao overfit-one-batch cô lập lỗi *wiring* khỏi lỗi *data*?
4. **4.2** Dựng lại: fp16 cộng 0.01 đóng băng ở **đâu** và VÌ SAO đúng chỗ đó? (ulp(x)=2^(floor(log2 x)−10)).
5. **4.2** Vì sao bf16 khỏi GradScaler mà fp16 cần? (chạm "exponent bits" + "underflow" vs "precision" — KHÔNG
   phải mantissa).
6. **4.2** Dưới autocast: matmul dtype? layer_norm dtype? Vì sao khác nhau?
7. **4.3** Vì sao checkpointing ra ĐÚNG grad? Điều kiện gì (dropout phá nó thế nào)? Vì sao ở TRỤC BỘ NHỚ
   `full≈selective` nhưng TRỤC COMPUTE thì khác?
8. **4.3** *Sửa-và-đoán:* nếu `_sac_policy` MUST_SAVE *mọi* op — bộ nhớ + số matmul recompute về đâu? suy biến
   thành mode nào?
9. **4.4** Vì sao NaN-guard đặt ở nhịp `log` là "free"? Chuỗi verify nào ⇒ "codegen bug inductor" chứ không phải
   "logic/Muon sai"?
10. **4.5** Vì sao PHẢI log ba KL *riêng* — mỗi cái loại trừ chế độ hỏng nào? Vì sao `kl_train_infer` là cái đầu
    tiên khiến HALT? Vì sao `seed_everything` trong `train` là *chưa đủ* cho e2e-reproducibility?

> Trả lời cold được cả 10 = **M4 thật sự OWNED** (interview-grade). Vấp câu nào → mở đúng mục đó, hoặc
> blank-slate hàm tương ứng (`raise NotImplementedError` → test đỏ → tự dẫn → xanh) để re-own bằng tay.

---

*Cross-ref: `roadmap_model/M4_training_loop.md` (reference chung, dẫn-xuất-vắn) · `derivations/M2_transformer_forward.md`
(forward pass — CE/loss@init nối từ 2.6) · `PROGRESS.md` (ledger 89 Bài) · `CURRICULUM.md` (con đường) ·
`bench/RESULTS.md` §Frontier ablations F1/F4 (số GPU sm120). Concept trước: M3 — objective + optimization
(CE→AdamW→cosine→Muon). Concept kế: M5 — distributed (DDP→ZeRO→FSDP), tựa lên `gradient_clipping` + mixed
precision + activation checkpointing của série này.*
