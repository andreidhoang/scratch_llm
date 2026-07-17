# S2 — Serving Engines · First-Principles Derivations

> **Đây là gì.** Bản dẫn-xuất **từ nguyên lý gốc** (first principles) cho TOÀN BỘ serving stack của một
> LLM decode-only — 9 micro-concept S2 (Bài 2.1→2.9), leo bức tường bộ nhớ decode từ **16% → 53% → 77%**
> mức trần HBM đo trên chính card. Mỗi mục: (1) **Câu hỏi** falsifiable, (2) **Sự thật nền tảng** (áp lực
> vật lý/toán buộc ra thiết kế), (3) **Dẫn xuất** có math, (4) **Neo code** `file·func·line` (verify tại
> HEAD), (5) **Hình ảnh** (ASCII + shape/stride/dtype + numeric micro-trace), (6) **Số đo THẬT**, (7)
> **frontier framing** + cổng phỏng vấn.
>
> **Cách dùng để re-learn.** Đọc *Câu hỏi* của mỗi mục → **tự trả lời COLD** (che phần dưới) → mở code
> đối chiếu → chỉ đọc phần *reconcile the gap*. Cuối doc có **checklist recall cold** + bảng số đo. Đây là
> *derivation lab* có số đo, bạn đồng hành của `roadmap/S2_serving_engines.md` (spine chung, reference).
>
> **Cảnh báo neo.** Line number **trôi** theo commit. Cite ở đây pin theo HEAD ngày 2026-07-14. Roadmap
> spine pin theo commit `9e61d7a` và đã lệch ở vài chỗ — ví dụ `cache.truncate` KHÔNG ở `model.py:239`
> mà ở `kv_cache.py · KVCache.truncate · :65` (KV-cache đã tách khỏi `model.py` từ 2026-07-08). Luôn
> `grep` tên hàm, đừng tin số dòng cứng (luật repo: verify, don't trust).
>
> **Nguồn số đo.** Số GPU-gated (Triton/CUDA/multi-GPU) **[measured · sm120 Blackwell · bench/RESULTS.md]**
> — trích từ ledger, KHÔNG bịa throughput kernel. Số CPU-safe (token-exact oracle, MLA identity float64,
> metrics percentile, AI≈B toy) chạy lại được bằng `scratchpad/s2_checks.py` — đó là "DoD là một profile"
> (FOP-3/4) áp cho việc học. Số roofline chưa đo cụ thể ghi **[PREDICTED]** kèm cách dẫn.

---

## Bức tranh lớn — serving là leo một bức tường bộ nhớ

Một inference engine là hàm `f: (danh sách request) → (token stream + số đo TTFT/ITL/throughput/goodput)`.
Toàn bộ S2 dựng hàm đó, và cả série là MỘT câu chuyện: **decode B=1 bị chặn bởi bức tường bộ nhớ HBM**,
mỗi Bài là một *đòn bẩy khác nhau lên cùng bức tường*.

```
                       TRẦN HBM = 0.551 TB/s ; compute 72 TF/s ; ridge AI≈131 ; wall ≈ 315-327 tok/s
   ┌───────────────────────────────────────────────────────────────────────────── 100%
   │                                                          ██ R4.4 cudagraph 253 tok/s = 77%
   │                                        ██ R1 compiled 173 tok/s = 53%
   │  ▁▁ R0/R1 eager 51 tok/s = 16%   (84% wall = launch overhead, KHÔNG phải traffic bộ nhớ)
   └───────────────────────────────────────────────────────────────────────────── 0%

   ĐÒN BẨY (mỗi cái tấn công một áp lực khác):
   2.1 oracle  ── token-exact vs generate greedy   ← BẤT BIẾN neo mọi rung (đo tốc trên engine sai = vô nghĩa)
   2.2 metrics ── TTFT / ITL / throughput / GOODPUT ← thước: throughput biết nói dối, goodput thì không
   2.3 R0      ── decode tuần tự có đo             ← honesty: workload memory-bound nhưng run overhead-bound
   2.4 R3a     ── static batch  → nâng AI≈B        ← ÁP LỰC BỘ NHỚ: đọc weight 1 lần, dùng cho B stream
   2.5 R3b     ── continuous batching             ← ÁP LỰC UTILIZATION: đuổi row xong, nạp row mới mỗi step
   2.6 R4.3    ── speculative decode (lossless)    ← nâng AI từ phía KHÁC: nhiều token / 1 target forward
   2.7 R4.4    ── CUDA-graph decode               ← ÁP LỰC LAUNCH: gom hàng trăm launch → 1 submission
   2.8 R4.5    ── MLA weight-absorption           ← ÁP LỰC CACHE: cache latent low-rank thay K,V per-head
   2.9 R4.6    ── PD-disaggregation               ← ÁP LỰC SCHEDULING: tách 2 profile trái ngược (prefill vs decode)
```

**Sự thật xuyên suốt (khác M2 một bậc).** Ở M2, áp lực là *số học/độ sâu* của một forward. Ở S2, áp lực
là **kinh tế phần cứng của autoregressive decode**: mỗi token cần đọc *toàn bộ weight* (P params) để sinh
*một* token → arithmetic intensity `AI = FLOP/byte ≈ 1` → **memory-bound**. Kernel nhanh hơn KHÔNG cứu nổi
`AI≈1`; mọi đòn bẩy S2 hoặc **nâng AI** (batching 2.4/2.5, speculation 2.6), **bỏ overhead che tường**
(cudagraph 2.7), **co byte phải đọc** (MLA 2.8), hoặc **tách profile để không giẫm chân** (disagg 2.9).
Học S2 = học *áp lực phần cứng → đòn bẩy → số đo phần trăm-của-trần*.

**Prefill vs decode — hai profile trái ngược (nền của cả série):**

```
PREFILL  (prompt L token, MỘT forward):  FLOP ~ 2·P·L  ,  bytes ~ 2P   →  AI ~ L  →  COMPUTE-bound (GEMM)
DECODE   (1 token, MỖI forward):          FLOP ~ 2·P·1  ,  bytes ~ 2P   →  AI ~ 1  →  MEMORY-bound (GEMV)
```

---

## 2.1 · Token-exact decode oracle — chiếc thước chuẩn trong tủ kính

**Câu hỏi.** Làm sao biết một engine serving "nhanh hơn" mà **KHÔNG đổi output**? Cần một nguồn sự thật duy
nhất, đơn giản đến mức *không thể sai*, để mọi rung sau (batch, paged, graph, spec) đối chiếu **từng token**.

**Sự thật nền tảng.** Tối ưu = phép biến đổi bảo toàn *ngữ nghĩa* (output) nhưng đổi *cách tính*. Muốn phát
biểu "bảo toàn" cần một hàm tham chiếu cố tình ngây thơ đến mức đúng-không-cãi-được. Hai đường trong **cùng
một hàm** phải cho **cùng token**: `use_cache=False` tính lại toàn prefix mỗi bước (không có state → không
thể sai); `use_cache=True` dùng `KVCache`. Nếu lệch → **cache bug**, sửa cache, không đổ cho model.

**Dẫn xuất.** Greedy = `argmax(logits)` khi `temperature==0` — **xác định**, không RNG (đó là điều kiện để
token-exact có nghĩa; với sampling thì so distribution/seed, không so token). Logprob của token lấy từ
**chính logits nó được sample ra** qua `log_softmax` (raw model dist, *không* scale temperature/top-p):
```
log π(a_t | s_t) = log_softmax(z_t)[a_t] = z_t[a_t] − logsumexp(z_t)
```
Đọc từ `probs` đã lọc top-p sẽ SAI: mẫu số đã bị renormalize trên nucleus, không còn là `π` thật mà PPO/GRPO
cần cho IS-ratio. Dừng: gặp `stop_ids` (break *sau* khi append token stop) hoặc đủ `max_tokens` — chính xác,
không lố một token.

**Neo code** (`src/scratch_llm/sampling.py`):
```python
def _sample_next(logits, params):                       # :58
    if params.temperature == 0.0:
        return int(logits.argmax().item())              # :61  greedy = argmax, xác định
    probs = softmax(logits / params.temperature, ...)   # :62
    probs = _top_p_filter(probs, params.top_p)          # :63
    return int(torch.multinomial(probs, 1).item())      # :64

def _logprob_of(logits, token_id):                      # :67
    return float(torch.log_softmax(logits, -1)[token_id])  # :73  RAW dist, không scale
# _decode :76 — HAI đường CHUNG _sample_next / _logprob_of:
if use_cache:                                           # :100
    cache = KVCache(len(model.blocks))                  # :101
    logits = model(x, cache)[0, -1]                     # :103  prefill, lấy vị trí cuối [0,-1]
    for _ in range(params.max_tokens):                  # :104
        next_id = _sample_next(logits, params)          # :105
        ...
        x = torch.tensor([[next_id]], ...)              # :110  decode MỘT token vào cache
        logits = model(x, cache)[0, -1]                 # :111
    return generated, logprobs                          # :112  đường cache RETURN sớm ở đây
# fall-through (KHÔNG phải else) — nhánh no-cache: recompute prefix MỖI bước
for _ in range(params.max_tokens):                      # :114
    window = ids[-context_length:]                      # :115
    logits = model(x)[0, -1]                            # :117
```

**Hình ảnh — hai đường phải hội tụ:**
```
prompt=[1,2,3,4,5]                     tensor dtype=long
   │
   ├── use_cache=True  : prefill (1,5)→KVCache ; decode (1,1) vào cache mỗi bước  ── O(1) K/V write/step
   │                     logits = model(x,cache)[0,-1]  shape (vocab,)
   └── use_cache=False : mỗi bước forward window (1, ≤ctx) — KHÔNG state         ── O(L) recompute/step
                         logits = model(x)[0,-1]        shape (vocab,)
                     │
   BẤT BIẾN NEO ─────┴──►  argmax(logits)  →  MỖI BƯỚC RA CÙNG token  (nếu lệch = cache bug)
```
Micro-trace (`s2_checks.py`, tiny model seed=0, vocab=64): cả hai đường ra
`[30, 63, 30, 63, 19, 23, 19, 23, 2, 45, …]` — **IDENTICAL: True**.

**Số đo THẬT** (CPU, `scratchpad/s2_checks.py`):
```
use_cache=True  vs  use_cache=False   →   token-exact IDENTICAL: True
```
Ledger: bất biến này neo **cả 9 rung** — mọi R3b/R4.x test "token-exact vs generate greedy"; float64 exact,
float32 lệch chỉ do argmax tie-flip (không phải logic) `[measured · bench/RESULTS.md · R4.2/R4.3]`.

**Frontier / cổng.** Đây là "loss-at-init/round-trip oracle" mà lab nào cũng dựng TRƯỚC khi tối ưu. Interview:
"bạn refactor KV-cache/paged/graph — chứng minh không đổi output thế nào?" → *chiếc thước này* + so từng token
ở float64. Trait = cheapest-oracle discipline (FOP-2). Table-stakes.

---

## 2.2 · Metrics TTFT / ITL / throughput / GOODPUT — throughput biết nói dối

**Câu hỏi.** "Nhanh hơn" nghĩa là gì cho một LLM server? Một con số throughput che mất trải nghiệm người dùng.
Phải đo ĐÚNG **bốn** số quyết định ship-hay-không.

**Sự thật nền tảng.** Serving có hai pha chi phí trái ngược (Bức tranh lớn): **prefill** gate cái *đầu tiên*,
**decode** gate cái *mượt*. Một số tổng che mất pha nào đau. Bốn đồng hồ, mỗi cái đo một nỗi đau:
- **TTFT** (time-to-first-token) = enter → chữ đầu — **prefill-gated**.
- **ITL / TPOT** (inter-token latency) = độ mượt chữ chảy ra — **decode-gated**.
- **Throughput** = tổng tok/s cả hệ.
- **Goodput** = throughput *chỉ tính request đạt SLO* (DistServe) — con số "trả tiền".

**Dẫn xuất.** Identity neo tất cả:
```
latency ≈ TTFT + ITL × num_output_tokens
```
→ nói ngay *đòn bẩy nào đánh vào số nào*: prefill-side (chunked, disagg) đánh TTFT; decode-side (cudagraph,
batching) đánh ITL·N. **Vì sao goodput dùng `max(itls)` không `mean`?** SLO là hợp đồng *worst-case* — một cú
giật 200 ms giữa hàng nghìn gap 6 ms vẫn làm hỏng trải nghiệm; `mean` bôi trơn nó đi. **Percentile nearest-rank**
`idx = round(p·(n−1))` (không nội suy) — cùng convention `bench.harness`, tái lập được. **ITL percentile gộp
POOL mọi gap** của mọi request → một admission spike ở *bất kỳ* request nào cũng nhô lên p99 (nếu lấy "p99
từng request rồi trung bình" thì spike biến mất — che mất pha admission).

**Neo code** (`src/scratch_llm/serving/metrics.py`):
```python
class RequestRecord:                                    # :29  token_times_s trên 1 clock đơn điệu
    def ttft_s(self):  return token_times_s[0]-start_s  # :45-47  prefill time
    def itls_s(self):  return (t[i]-t[i-1] ...)         # :50-56  hiệu liên tiếp (len = output_len−1)
def percentiles(values):                                # :96
    idx_p99 = round(0.99*(n-1))                         # :105  nearest-rank, KHÔNG nội suy
def request_meets_slo(record, slo):                     # :113
    has_bad_itl = max(record.itls_s) > slo.itl_s        # :115  WORST gap, không mean
    return record.ttft_s <= slo.ttft_s and not has_bad_itl
def summarize(records, slo):                            # :119
    throughput = total_output_tokens / wall_s           # :137
    goodput = Σ output_len CỦA request đạt SLO / wall_s # :139-140
    ttft_ms = percentiles([r.ttft_s ...])               # :142-143  per-request
    all_itls = [itl for r in records for itl in r.itls_s]  # :145-149  POOL gộp
```

**Hình ảnh — vì sao goodput ≠ throughput:**
```
request A: [t0]──6ms──[t1]──6ms──[t2]──230ms(SPIKE)──[t3]      SLO: itl ≤ 50 ms
                                     ▲ max(itls)=230 > 50  ⇒  request A FAIL SLO
throughput = 4 tok / 0.26 s = 15.38 tok/s   (đếm hết)
goodput    = 0 tok / 0.26 s =  0.00 tok/s   (A fail ⇒ 0 request đạt)   ← throughput nói dối
```
Micro-trace (`s2_checks.py`): record `token_times=(0.01,0.02,0.25,0.26)` → `itls=[0.01,0.23,0.01]`,
`max=0.230`; SLO(itl≤0.05) → `meets = False`; `throughput=15.38`, **`goodput=0.00`**.
Percentile n=5 `[10,20,30,40,50]`: p50 idx=`round(.5·4)=2`→**30**, p95/p99 idx=`round(.95·4)=4`→**50**.

**Số đo THẬT** (CPU, `s2_checks.py`) + ledger:
```
percentiles([10..50])  →  p50=30  p95=50  p99=50   (nearest-rank)
1 spike request        →  throughput 15.38 tok/s   goodput 0.00 tok/s
R4.2 saturated trace   →  goodput = 0 tok/s ALL arms (ITL vi phạm SLO)  [measured · bench/RESULTS.md · R4.2]
```

**Frontier / cổng.** Goodput-under-SLO là ngôn ngữ chuẩn 2026 (DistServe/vLLM). Interview: "TTFT p99 vs ITL
p99 — tối ưu nào đánh cái nào?" → dẫn từ `latency = TTFT + ITL·N`. Trait = measure-the-right-number. Scarce
(serving) differentiator.

---

## 2.3 · R0 baseline — decode tuần tự có đo (workload-bound vs run-bound)

**Câu hỏi.** Trước khi tối ưu, phải ĐO cái ngây thơ và biết nó bị chặn bởi *gì*. Decode B=1 có phải
memory-bound không? Nếu có, sao chỉ đạt **16%** trần HBM?

**Sự thật nền tảng.** Phải phân biệt hai khái niệm hay bị lẫn: **workload bound** (roofline của *bài toán*:
decode có `AI≈1` ⇒ memory-bound *về nguyên tắc*) vs **run bound** (cái đang *thật sự* giới hạn wall-clock: ở
đây là **launch overhead** — CPU dispatch ~955 kernel/token). Hai cái KHÔNG mâu thuẫn: workload muốn 315 tok/s
nhưng run chỉ đạt 51 vì 84% wall là overhead *che* traffic bộ nhớ thật. R0 KHÔNG viết lại decoder — nó **gắn
đồng hồ** lên `sampling.generate` sẵn có (honesty: naive = chạy tuần tự từng request một).

**Dẫn xuất — timing đúng trên CUDA async.** CUDA kernel là *bất đồng bộ*: `clock()` ngay sau một lệnh GPU đo
lúc **enqueue**, không phải **complete**. Phải `torch.cuda.synchronize()` TRƯỚC mỗi timestamp:
```
sai:   enqueue kernel ─► clock()          → đo thời gian XẾP HÀNG (nhỏ hơn thực tế)
đúng:  enqueue kernel ─► synchronize() ─► clock()   → đo thời gian GPU HOÀN TẤT
```
`start_s` đọc ngay trước prefill; `token_times_s[i]` ngay sau token i sẵn sàng ⇒ `ttft_s` = thời gian prefill.

**Neo code** (`src/scratch_llm/serving/baseline.py`):
```python
def decode_record(model, prompt_ids, params, device, *, clock=time.perf_counter):  # :44
    def get_time():                                     # :65
        if torch.device(device).type == "cuda":         # :66  khớp "cuda" và "cuda:N"
            torch.cuda.synchronize()                     # :67  ĐỒNG BỘ trước clock
        return clock()                                   # :68
    start_s = get_time()                                 # :77
    cache = KVCache(len(model.blocks))                   # :79
    logits = model(x, cache)[0, -1]                      # :81  prefill
    for _ in range(params.max_tokens):                   # :86
        next_id = _sample_next(logits, params)           # :87  DÙNG CHUNG Bài 2.1
        token_times.append(get_time())                   # :89
        x = torch.tensor([[next_id]], ...)               # :92
        logits = model(x, cache)[0, -1]                  # :93
def run_baseline(model, prompts, params, device):        # :103
    return [decode_record(...) for prompt in prompts]    # :113  TUẦN TỰ, one at a time
```

**Hình ảnh — 84% wall là overhead:**
```
một decode step (eager, B=1), wall = 19.6 ms/token:
  CPU: launch k1│gap│launch k2│gap│ … │launch k955│gap        ← ~955 dispatch tuần tự = 84% wall
  GPU:      [k1]     [k2]        …        [k955]                ← traffic bộ nhớ thật = 16% wall (89 GB/s)
                                                     ▲ GPU CHỜ host giữa các kernel
achieved BW 89/550 = 16%   ⇔   51/315 tok/s = 16%   (cùng một 16%: run overhead-bound, chưa chạm tường)
```
`run_baseline` chạy tuần tự → utilization của "batch" = 1/B → chính cái "tĩnh" mà 2.4/2.5 đánh bại.
Micro-trace (`s2_checks.py`): `decode_record == generate` (IDENTICAL: True), `ttft_s(prefill)=0.000547 s`.

**Số đo THẬT** `[measured · sm120 Blackwell · bench/RESULTS.md · A1 R1]`:
```
decode B=1 eager:  51 tok/s = 16% của 315 tok/s trần ; 89 GB/s = 16% của 550 GB/s HBM
nsys:              ~955 kernel launch/token (dominant = GEMV) + 1 .item() host-sync/token
strip A (nosync):  48→50 tok/s  → per-token sync ~FREE (confound là dispatch, không phải sync)
```

**Frontier / cổng.** Phân biệt "workload bound" vs "run bound" là câu hỏi phễu ở interview kernel. Ở đây
workload memory-bound nhưng run overhead-bound → **lever đúng là *fuse launch*** (R1 compile → R4.4 graph),
KHÔNG phải kernel bộ nhớ nhanh hơn. Trait = predict-then-measure, name-the-real-constraint (FOP-3).

---

## 2.4 · R3a static batch — đòn bẩy AI ≈ B

**Câu hỏi.** B=1 đọc toàn bộ weight (1.68 GB) mỗi token cho MỘT token — lãng phí. Nếu đọc weight *một lần*
rồi áp cho B sequence thì arithmetic intensity thành bao nhiêu, và tok/s scale thế nào theo B?

**Sự thật nền tảng.** Decode memory-bound vì `AI≈1` (đọc 2P byte weight cho 1 token). Kernel nhanh không cứu
nổi `AI≈1`. **Batching nâng CHÍNH AI**: đọc quyển sách (weight) to cho cả lớp B đứa cùng lúc → chi phí đọc
*chia cho B* (weight amortization). Đánh đổi: đòi mọi row **cùng độ dài** (RoPE positions chia sẻ) → đây là
*static* batch; ragged + join/leave là R3b (2.5).

**Dẫn xuất — roofline batched decode.** Mỗi decode step:
```
FLOP  ≈ 2·P·B        (mỗi param một MAC = 2 FLOP, cho B token)
bytes ≈ 2P + B·KV    (đọc weight MỘT LẦN = 2P ; cộng KV-read mỗi stream = B·KV)
                              2·P·B
   AI  =  ───────────────────────────
                          2P + B·KV
```
- **B nhỏ** (`B·KV ≪ 2P`): mẫu ≈ 2P ⇒ **`AI ≈ B`** (memory regime; per-stream tok/s ~hằng, aggregate tuyến tính).
- **B lớn** (`B·KV ≫ 2P`): `AI → 2P/KV` = hằng (bão hòa) → cắt sang **compute-bound**.
- Crossover khi `AI = ridge = compute_peak/BW = 72e12/0.551e12 ≈ 131` ⇒ **B ≈ 128** (đo được).

**Vì sao PHẢI batching, không phải kernel nhanh hơn?** Roofline: dưới ridge, wall ~ `bytes/BW`. `AI≈1` nghĩa
là ta ở tận đáy trục hoành — kernel nhanh (nâng trần compute) vô dụng; chỉ **dịch phải trên trục AI** (batching)
mới leo. Đó là toàn bộ triết lý serving.

**Neo code** (`src/scratch_llm/serving/batched.py`):
```python
def batched_greedy_decode(model, prompts, max_tokens, device):   # :26
    if len({len(p) for p in prompts}) != 1:                       # :41  ÉP equal-length (RoPE align)
        raise ValueError("requires equal-length prompts (R3a)")  # :42
    cache = KVCache(len(model.blocks))                           # :48  MỘT cache chung
    x = torch.tensor([list(p) for p in prompts], ...)            # :49  (B, L)
    logits = model(x, cache)[:, -1]                              # :50  prefill → (B, vocab)
    for _ in range(max_tokens):
        nid = logits.argmax(dim=-1)                              # :53  (B,) — MỘT weight read/B row
        x = nid.unsqueeze(1)                                     # :56  (B, 1)
        logits = model(x, cache)[:, -1]                          # :57
```

**Hình ảnh — weight amortization + shape journey:**
```
B=1:  weight(1.68GB) ──► 1 token   (đọc 1.68GB / token)          bytes/tok = 2P
B=32: weight(1.68GB) ──► 32 token  (đọc 1.68GB / 32 token)       bytes/tok = 2P/32 + KV   ← amortized

decode step shapes:  x (B,1) long ──model+cache──► logits (B, vocab) fp32 ──argmax(-1)──► nid (B,) long
                                                                                            │ unsqueeze(1)
                                                                                         x (B,1) → lặp
row b độc lập:  batch axis KHÔNG trộn thông tin (causal per-row) ⇒ row b == single-stream greedy prompt b
```
Micro-trace (`s2_checks.py`): 3 prompt equal-length, mọi row `batched[b] == generate(prompt b)` →
**batch-independent: True**. AI toy `2PB/(2P+B·KV)` với P=0.84e9, KV=4096: `B=1→1.00, 8→8.00, 32→32.00,
128→127.96, 256→255.84` — **`AI ≈ B` chính xác trong memory regime**.

**Số đo THẬT** `[measured · sm120 Blackwell · bench/RESULTS.md · A1 R3a]`:
```
agg(B=32)/agg(B=1) = 24.9×  ;  peak 12,220 tok/s @B=256 = 66× ;  185→9,255 tok/s (B=1→64)
roofline crosses memory→compute tại B≈128 (AI≈ridge 131) ; plateau ~12K = ~28% của 72 TF/s
eager control CŨNG scale 30.7× — amortize cả overhead ~20 ms/step (không chỉ weight bytes)
```

**Frontier / cổng.** "Batch size vs latency/throughput" là capacity-planning kinh điển. B≈128 crossover = nơi
lab chuyển từ lo bandwidth sang lo FLOP → quyết định GQA/MQA (KV nhỏ → batch to hơn vẫn fit) + kernel tensor-core.
Trait = roofline-first, predict-the-crossover (FOP-3). Scarce (inference).

---

## 2.5 · R3b continuous batching + hợp đồng sở hữu graph/scheduler

**Câu hỏi.** Static batch chạy cả wave tới khi ROW DÀI NHẤT xong — row ngắn ngồi không. Nếu quyết định lại
*mỗi bước* (đuổi row xong, nạp row mới vào slot trống) thì được bao nhiêu, và giá phải trả là gì?

**Sự thật nền tảng.** Static wave: utilization = `mean_len / max_len` — heavy-tail (vài row rất dài) → phần
lớn slot ngồi không suốt đuôi. Continuous (Orca): slot trống là nhận ngay request kế trong hàng đợi → batch
luôn bão hòa. Cùng engine/cache/forward — chỉ khác **policy** → Δ đo được là *thuần scheduling*.

**Dẫn xuất.** Với queue bão hòa:
```
speedup ≈ max_len / (mean_len + admit_tax)      util_wave = mean_len / max_len
```
Nhưng wall-ratio < step-ratio vì **thuế padding của dense buffer**: batch trộn tuổi (row mới lẫn row già) giữ
`view_len = 1 + max(py_lengths)` cao → mỗi decode step đọc nhiều **padding** hơn (dense buffer slice
`[:, :, :view_len]` bao cả vùng chưa dùng của row ngắn).

**Hợp đồng sở hữu graph/scheduler — bài học load-bearing (Dynamo).** Đây là điểm principal-level. Có HAI
loại state:
- **graph-owned** (device tensor): `lengths`, `active`, K/V buffer — mutate *trong* forward, trace sạch dưới
  `torch.compile`.
- **scheduler-owned** (python): `py_lengths`, `py_active`, free-list, mọi quyết định allocation — mutate
  *ngoài* graph (`mirror_admit` sau prefill, `mirror_advance` sau decode, `pre_decode_reserve` trước decode).

**Vì sao tách?** Nếu đọc `max(py_lengths)` *trong* forward, Dynamo bake guard theo **thứ tự** của python-list;
ragged churn permute list → recompile-storm → **eager fallback CHỈ ở arm continuous** (wave uniform không churn)
→ thiên vị baseline, số đo bẩn. `view_len` phải là **int thường**, recompute chỉ bởi mirror-op. Prefill chạy
**eager** (`prefill_model` uncompiled) vì admission tới lắt nhắt n=1,2,3… mỗi width = 1 graph mới (đo: 47 graph,
~35s compile) — decode mới xứng ngân sách compile. Sau fix: `unique_graphs=2`.

**Neo code** (`src/scratch_llm/serving/continuous.py`, `serve` :90; view_len ở `kv_cache.py`):
```python
while queue or any(r is not None for r in slot_req):    # :250  MỘT vòng scheduler
    for b, req in enumerate(slot_req):                  # :252  (1) EVICT row hết budget
        if len(slot_tokens[b]) >= req.max_new_tokens: finish(b)   # :253-254
    # (2) ADMIT — continuous: slot free bất kỳ ; wave: chỉ khi tất cả slot free
    if queue and free and (policy=="continuous" or len(free)==n_slots):  # :262
        first = _prefill(prefill_model or model, cache, admits, slots, lens, dev)  # :274  MỘT prefill right-pad
        cache.mirror_admit(slots, lens)                 # :282  python half NGOÀI graph
    # (3) một DECODE step lockstep trên slot active (shape tĩnh; row inactive tính masked rồi vứt)
    logits = model(last_ids.unsqueeze(1), cache)        # :357
    cache.mirror_advance()                              # :361  python half NGOÀI graph
```
```python
# kv_cache.py — vì sao view_len KHÔNG được đọc in-graph:
def view_len(self):  return self._view_len              # :169-175  int thường (no .item, no max in-graph)
def _recompute_view_len(self):                          # :177
    self._view_len = min(self.max_ctx, 1 + max(self.py_lengths))   # :178  chỉ scheduler gọi
def advance(self, n):  self.lengths += self.active.long()  # :213  device-only (graph-owned)
def mirror_advance(self): ... self._recompute_view_len()   # :215-223  python-only (scheduler-owned)
```

**Hình ảnh — wave vs continuous (heavy-tail 24×64 + 6×256 + 2×512):**
```
WAVE (static):      [ admit cả đoàn ]───decode tới ROW DÀI NHẤT (512)───[ drain ]   util 24.9%
                    slot ngắn (64) xong ở step 64 → NGỒI KHÔNG 448 step
CONTINUOUS (Orca):  [admit][decode][evict+admit][decode]…  slot trống nạp NGAY        util 72.8%
                    ↑ nhưng batch trộn tuổi → view_len cao → padding traffic 9.6 vs 6.3 ms/step (×1.27)
```
`step-ratio 2.93×` (thuần scheduling) → `wall-ratio 2.30×` (sau thuế padding 1.27×). Đó là vì sao
**R4.1 PagedAttention** tồn tại: đọc chỉ real token qua block-table, bỏ padding tax.

**Số đo THẬT** `[measured · sm120 Blackwell · bench/RESULTS.md · A1 R3b/R4.2]`:
```
continuous vs wave (heavy-tail): 2.30× wall (2,283 vs 995 tok/s) ; 2.93× by step (1,396 vs 4,088)
util 72.8% vs 24.9% (== analytic) ; padding tax 9.6 vs 6.3 ms/step = 1.27× (giải thích 2.93→2.30)
TTFT p95 4.9× (852 vs 4,146 ms) ; ITL p50 5.4→9.6-10.8 ms (giá của batching) ; unique_graphs=2
R4.2 chunked prefill = MEASURED NEGATIVE (honest): ITL p99 RỘT NGƯỢC ×1.03→1.91, throughput 547→145 GIẢM
   — chunk là forward tuần tự riêng, KHÔNG piggyback vào decode batch (Sarathi/vLLM-V1 mới đúng)
```

**Frontier / cổng.** Continuous batching là default 2026 (vLLM/TGI). Interview: "chunked prefill giúp gì và cạm
bẫy scheduling-only?" → chính measured-negative này: tách forward = mất nhiều hơn spike bỏ được; lợi ích chunk
là *kernel/batching property*, không phải scheduling-only. Trait = claims-honesty (ship cả kết quả âm) +
ownership-boundary (FOP-4). Scarce (inference).

---

## 2.6 · R4.3 speculative decoding — lossless by construction

**Câu hỏi.** Decode là 1 weight-read/token (`AI≈1`). Có cách nào **commit >1 token mỗi target forward** mà
KHÔNG đổi distribution output không?

**Sự thật nền tảng.** Verify **rẻ hơn** generate: kiểm tra K token đề xuất tốn *một* forward K+1-wide (cùng
chi phí bộ nhớ như 1-wide decode vì vẫn đọc weight một lần); sinh K token *tuần tự* tốn K forward. Một
**drafter** rẻ đoán K token; target verify CẢ K trong MỘT forward; chấp nhận prefix dài nhất **khớp argmax
của chính target**. Chuỗi commit ĐÚNG BẰNG cái target tự sinh → **lossless cho greedy by construction**.

**Dẫn xuất — bất biến `pending` + KV rollback.** Đầu mỗi round, `pending` = token committed kế mà K/V CHƯA
vào cache và `== argmax` greedy của target (đúng by construction). Draft K guess cho các token SAU nó; forward
`[pending, *drafts]`; `greedy[i]` = token target sau `inp[i]`. Accept prefix draft khớp greedy:
```
accept longest j sao cho  drafts[0:j] == greedy[0:j]
commit = [pending] + drafts[0:accepted]        (1 + accepted token cho 1 target forward)
pending_next = greedy[accepted]                (correction/continuation MIỄN PHÍ)
```
**KV rollback (mấu chốt losslessness).** Forward `[pending, *drafts]` làm cache mọc `1 + len(drafts)` vị trí.
Nếu drafts bị từ chối, K/V của chúng đã vào cache → round kế sẽ attend lên key SAI. `cache.truncate(base + 1 +
accepted)` cắt bỏ K/V draft bị từ chối → cache giữ đúng `pending + accepted`, **bit-identical với plain decode**.
Vì thế **drafter SAI hoàn toàn vẫn ra đúng greedy** — nó chỉ tốn slot verify, không đổi correctness.

Speedup:
```
speedup ≈ E[accepted + 1] / (1 + K·c_draft/c_target)      (n-gram: c_draft ≈ 0 → ≈ E[accept+1])
```

**Neo code** (`src/scratch_llm/serving/speculative.py`; `truncate` ở `kv_cache.py`):
```python
def speculative_generate(target, drafter, prompt_ids, max_new_tokens, device, k=4):  # :108
    logits = target(x, cache)[0, -1]                    # :139  prefill
    pending = int(logits.argmax())                      # :140  pending = argmax (đúng by construction)
    while len(generated) < max_new_tokens:              # :147
        context = prompt + generated + [pending]        # :148
        drafts = list(drafter.propose(context, k))[:k]  # :149
        base = cache.length                             # :152  điểm rollback
        inp = torch.tensor([[pending, *drafts]], ...)   # :153  (1, 1+len(drafts))
        vlogits = target(inp, cache)[0]                 # :154  MỘT forward verify; cache mọc theo
        greedy = vlogits.argmax(dim=-1).tolist()        # :156  greedy[i] = target sau inp[i]
        for i, d in enumerate(drafts):                  # :160  accept longest matching prefix
            if d == greedy[i]: accepted += 1
            else: break                                 # :163-164
        generated.append(pending); generated.extend(drafts[:accepted])  # :167-168
        pending = int(greedy[accepted])                 # :169  correction/continuation
        cache.truncate(base + 1 + accepted)             # :170  → kv_cache.py:65  DROP rejected drafts' K/V
# NGramDrafter.propose :46 — tìm last-n-gram xuất hiện gần nhất, đề xuất k token theo sau (training-free)
```

**Hình ảnh — một round (K=4, accepted=2):**
```
cache.length = base                     drafter đề xuất [d0 d1 d2 d3]
inp = [ pending  d0  d1  d2  d3 ]        forward 1 lần → greedy = [g0 g1 g2 g3 g4]
         │        ✓   ✓   ✗              d0==g0, d1==g1, d2≠g2 → accepted = 2
commit = [pending, d0, d1]              pending_next = g2 (correction, MIỄN PHÍ)
cache mọc +5 vị trí ─► truncate(base + 1 + 2 = base+3) ─► drop K/V của d2,d3  (bit-identical plain decode)
   3 token commit / 1 target forward  =  speedup
```
Micro-trace (`s2_checks.py`, prompt có n-gram lặp): drafter **n-gram** → exact=True, accept 16.67%;
drafter **ALWAYS-WRONG** (`[0]*k`) → **exact=True**, accept 0.00% — chứng minh losslessness: sai draft KHÔNG
làm hỏng output, chỉ 1.000 tok/forward (= plain decode).

**Số đo THẬT** (CPU `s2_checks.py` + ledger):
```
CPU: n-gram drafter exact=True (accept 16.67%) ; ALWAYS-WRONG exact=True (accept 0%) ← losslessness proof
GPU: token-exact float64 mọi drafter/K ; wrong-drafter vẫn exact ⇒ KV rollback clean  [measured · R4.3]
     speedup ×1.21-1.29 wall / 1.33-1.39 tok/target-forward (n-gram zero-cost)         [measured · R4.3]
     accept: repetitive 43-63%, random 54-72% (FALSIFIED: acceptance bám ENTROPY của model, không prompt)
```

**Frontier / cổng.** Cả họ drafter (Medusa/EAGLE-2/3 feature-tree, MTP ~85-90% 2nd-token) tối ưu `E[accept]`.
Interview: "spec decode giữ distribution thế nào?" → accept-longest-prefix + KV rollback; với sampling (không
greedy) thêm **modified-rejection sampling** (Leviathan/Chen) để lossless. Trait = prove-the-invariant
(losslessness). Scarce (inference).

---

## 2.7 · R4.4 CUDA-graph decode — giết launch overhead

**Câu hỏi.** Sau khi `torch.compile` fuse pointwise, decode vẫn còn ~54-68 launch/step — CPU dispatch từng
cái. Nếu *ghi* cả step thành MỘT submission rồi replay thì chạm được trần chưa?

**Sự thật nền tảng.** R1 đã chứng minh decode **launch-bound**, không memory-bound (Bài 2.3). Mảnh cuối leo
53%→77% trần là **per-launch CPU dispatch**. CUDA graph = "băng ghi" toàn bộ chuỗi kernel của một step, ghi
MỘT LẦN, replay bằng *một* lệnh `cudaGraphLaunch` — CPU thôi xếp hàng trăm launch, GPU hết chờ host.

**Dẫn xuất — vì sao paged kernel là substrate DUY NHẤT capture được.** Graph capture đòi **địa chỉ tĩnh** và
**shape tĩnh**:
- (a) `KVCache` R1 dùng `torch.cat` (`kv_cache.py:59`) → mọc **địa chỉ MỚI** mỗi step → graph ghi con trỏ cố
  định → capture **ILLEGAL** (`RuntimeError: accessing tensor output overwritten by a subsequent run`).
- (b) Dense `BatchedKVCache` đọc slice `[:, :, :view_len]` (`decode_view` :263-265) có **SHAPE mọc** mỗi step
  → graph shape tĩnh không phủ nổi.
- (c) Block pool paged = **fixed-address**, ghi in-place; kernel launch **grid `(B,H)` cố định**, số key mỗi
  row là *runtime loop-bound* đọc từ tensor `lengths` trên device → **MỘT capture phục vụ mọi độ dài**.
- (d) `torch.compile` reduce-overhead TỪ CHỐI path này (`lengths += active` in-place = "mutated input");
  **manual capture** tự làm chủ mutation (host mutate block table tại địa chỉ cố định giữa các replay).

**Vì sao snapshot + restore `lengths`?** Warmup (prime Triton autotune + allocator) và bản thân capture đều
gọi `advance` → làm mọc `lengths`. Phải snapshot trước, restore sau, để state prefilled nguyên vẹn cho replay
đầu tiên (garbage K/V ghi quá length restored bị mask + ghi đè bởi decode thật).

**Neo code** (`src/scratch_llm/serving/cudagraph.py`):
```python
class CudaGraphDecoder:                                  # :31
    def __init__(self, model, cache):                    # :39
        if not isinstance(cache, PagedKVCache) or not cache.use_kernel:  # :40  ĐÒI paged + kernel
            raise ValueError(...)                        # :41
        self._static_in = torch.zeros((n_slots,1), ...)  # :45  fixed-address staging
    def capture(self, warmup_steps=3):                   # :50
        saved = self.cache.lengths.clone()               # :56  SNAPSHOT
        with torch.cuda.stream(stream):                  # :61  warmup side stream (prime autotune)
            for _ in range(warmup_steps): self.model(self._static_in, self.cache)  # :62-63
        self.cache.lengths.copy_(saved)                  # :65  undo warmup advance
        self._graph = torch.cuda.CUDAGraph()             # :67
        with torch.cuda.graph(self._graph):              # :68  CAPTURE
            self._static_logits = self.model(self._static_in, self.cache)  # :69
        self.cache.lengths.copy_(saved)                  # :71  undo capture's OWN advance
    def step(self, last_tokens):                         # :76
        self.cache.pre_decode_reserve()                  # :80  host: cấp block nếu qua biên 16
        self._static_in.copy_(last_tokens.view(-1,1))    # :81  copy vào fixed-address
        self._graph.replay()                             # :82  MỘT cudaGraphLaunch (thay 54-68 launch)
        nxt = self._static_logits[:,-1].argmax(-1).clone()  # :83
        self.cache.mirror_advance()                      # :84  host half NGOÀI graph
```

**Hình ảnh — launch collapse:**
```
EAGER step:     CPU: launch×54-68 (mỗi cái 1 dispatch + gap)   GPU chờ host giữa kernel     15.38 ms
CUDA-GRAPH:     CPU: cudaGraphLaunch ×1  ──────────────────►    GPU chạy liền mạch            3.96 ms  (−74.3%)

host mutation quanh replay (KHÔNG in graph):
   pre_decode_reserve (cấp block) ─► copy_(_static_in) ─► REPLAY ─► argmax ─► mirror_advance (bump py_lengths)
                                     ▲ địa chỉ cố định để kernel captured thấy update
```

**Số đo THẬT** `[measured · sm120 Blackwell · bench/RESULTS.md · A1 R4.4]`:
```
B=1 step: −74.3% (15.38 → 3.96 ms) ; 253 tok/s = 77% của wall (eager 65=20%, compiled 53%)
54-68 cudaLaunchKernel → 1 cudaGraphLaunch ; B=8/B=32: −71.3% / −68.4% (agg 1749 / 6307 tok/s)
token-exact (gpu test B∈{1,4,8}, qua ranh 16-block) — capture đổi HOW launch, không đổi WHAT compute
```
(Dự đoán gốc −20-28% FALSIFIED tốt: −74.3% vì baseline eager của ta launch-bound nặng hơn H100-vLLM path.)

**Frontier / cổng.** Cudagraph decode là default vLLM-V1. Interview: "graph capture yêu cầu gì?" → static
address + static shape/grid; mutation phải do mình own. Trait = name-the-unmodeled-constraint (launch), FOP-3.
Scarce (kernels/inference).

---

## 2.8 · R4.5 MLA weight-absorption — cache latent thay K,V (serving)

**Câu hỏi.** Thay vì cache K,V mỗi head, có thể cache MỘT latent low-rank `c_KV`/token rồi vẫn attend *đúng*
không? Đại số nào cho phép, và điều kiện gì?

**Sự thật nền tảng.** KV-cache là memory-wall của long-context serving (Bài 2.4: cache chặn batch × ctx). GQA
co bằng *giảm số K/V head*; MLA co bằng **nén** K,V thành latent `c_KV` (dim `d_latent`) + rotary key nhỏ `k_R`
(dim `d_rope`) chia sẻ mọi head. Mẹo: đừng bung `c_KV` ra rồi attend — **gập** ma trận up-projection vào
query/output và attend THẲNG trong latent space.

**Dẫn xuất — identity thuần đại số.** Content score:
```
q_c · K^C = q_c · (W_UK c_KV) = (W_UK^T q_c) · c_KV
```
Gập `W_UK` vào query → `q_abs = W_UK^T q_c` attend `c_KV` trong `d_latent`. Output:
```
Σ_j a_j V_j = Σ_j a_j (W_UV c_KV_j) = W_UV (Σ_j a_j c_KV_j)
```
Gập `W_UV` vào output path. Decode chỉ đọc `c_KV`(+`k_R`) từ cache — **không bao giờ** vật chất hoá K,V đầy đủ.

**Điều kiện load-bearing — decoupled RoPE.** RoPE là phép quay phụ-thuộc-vị-trí. Nếu áp lên content-K
(`k_c = W_UK c_KV`), score thành `q_c · R_pos (W_UK c_KV)` — `R_pos` xen giữa nên **không gập** `W_UK` static
được (fold phải là ma trận cố định). DeepSeek tách riêng đường `d_rope`: content-K **position-free** (giữ
`W_UK` static fold), chỉ `k_R` mang vị trí. Đó là *lý do* MLA có nhánh decoupled.

**Neo code** (`src/scratch_llm/mla.py`):
```python
def _project(self, h, positions):                        # :76  rope CHỈ trên q_rope/k_rope (:83,:86)
    q_c   = self.q_c(h).view(b,s,H,dh).transpose(1,2)     # :81  (B,H,S,dh) content query (no RoPE)
    c_kv  = self.down_kv(h)                               # :84  (B,S,dc) — LATENT được cache
    k_rope= self.rope(self.k_r(h)...)                    # :85-86 (B,1,S,dr) shared mọi head
def forward_absorbed(self, h, positions):                # :111  path decode (cache saving)
    wk = self.up_k.weight.view(H, dh, dc)                # :118  W_UK
    q_abs = torch.einsum("bhsd,hdc->bhsc", q_c, wk)      # :119  gập W_UK vào query → (B,H,S,dc)
    scores = (matmul(q_abs, c_kv...T)                    # :122  attend TRONG latent space
            + matmul(q_rope, k_rope_h...T)) * self.scale # :123-124  + decoupled RoPE term
    latent_out = torch.matmul(attn, c_kv.unsqueeze(1))   # :128  (B,H,S,dc) weighted sum LATENT
    wv = self.up_v.weight.view(H, dh, dc)                # :130  W_UV
    out = torch.einsum("bhsc,hdc->bhsd", latent_out, wv) # :131  gập W_UV vào output
# forward_naive :92 (oracle): bung k_c=up_k(c_kv), v=up_v(c_kv) rồi attend chuẩn → PHẢI == absorbed
```

**Hình ảnh — naive (train) vs absorbed (decode):**
```
NAIVE (oracle, :92):   c_KV ──up_k──► K^C (B,H,S,dh) ──┐
                       c_KV ──up_v──► V   (B,H,S,dh) ──┤ attend chuẩn → out   (cache K,V per-head = to)
ABSORBED (decode,:111): q_c ──⊗W_UK──► q_abs (B,H,S,dc) ─► attend c_KV THẲNG ─► latent_out ─⊗W_UV─► out
                        chỉ đọc c_KV (dc) + k_R (dr) từ cache  ⇒  cache NHỎ
   identity: (W_UK^T q_c)·c_KV == q_c·(W_UK c_KV)   ← chỉ là kết hợp ma trận, ZERO quality change
```
Micro-trace (`s2_checks.py`, float64, d_model=256, H=8, dh=64, dc=512, dr=64): `forward_naive` và
`forward_absorbed` cùng shape `(2,16,256)`, `max|naive − absorbed| = 3.33e-15` (≈ machine eps).

**Số đo THẬT** (CPU float64 `s2_checks.py` + ledger):
```
CPU float64: max|naive − absorbed| = 3.33e-15  (machine eps ~2.2e-16)          ← identity holds
GPU/CPU:     identity 1.4e-15 (float64) ; float32 <1e-4                          [measured · R4.5]
KV bytes/token (DeepSeek-V3 dims d_latent=512, d_rope=64, verify bằng s2_checks):
   MLA   = (512+64)·2 = 1152 B  =  1.8% của MHA (2·128·128·2 = 65536 B)          [measured · R4.5]
   GQA-8 = 2·8·128·2 = 4096 B  →  MLA nhỏ hơn GQA-8 3.56×                         [measured · R4.5]
```

**Frontier / cổng.** MLA (DeepSeek-V2/V3) là KV story 2026: 671B có cache/token NHỎ hơn 70B GQA-8 → kinh tế
long-context rơi ra từ *kiến trúc*. Bridges A5 (FP8 latent). Interview: "MLA tiết kiệm cache thế nào lúc
decode?" → absorb `W_UK/W_UV` + cache latent; decoupled RoPE là điều kiện load-bearing. Trait =
derive-the-identity + name-the-condition. Scarce (inference/arch).

---

## 2.9 · R4.6 PD-disaggregation — tách hai profile trái ngược

**Câu hỏi.** Prefill (compute-bound, GEMM cả prompt) và decode (memory-bound GEMV, một token) profile TRÁI
NGƯỢC. Co-locate → prefill burst cướp cycle của decode → **ITL spike**. Tách ra đáng giá bao nhiêu, giá là gì?

**Sự thật nền tảng.** Hai job trái tính nết ở chung một GPU thì giẫm chân: mỗi prefill prompt dài, luồng
decode đang chảy bị khựng (spike ITL p99). Disagg: prefill chạy worker riêng, luồng decode KHÔNG bao giờ bị
ngắt; giá là **chuyển KV cache một lần** từ prefill-worker sang decode-worker. Đánh đổi: bỏ spike lặp lại
(~40 ms) đổi lấy transfer một lần (0.18 ms) — lời to. Nối về `request_meets_slo` dùng `max`-gap (Bài 2.2):
disagg cải thiện **ITL p99** (spike) chứ không nhất thiết p50.

**Dẫn xuất — điểm hòa vốn.** Single-GPU không đặt nổi 2 worker thật → demo HAI nửa đo được:
- (1) **KV-transfer tax** = D2D copy các paged block prefilled. Bound analytic = `bytes / HBM_BW`.
- (2) **ITL contrast**: decode worker *sạch* (chỉ short-prompt decode) vs *co-located* (decode xen prefill burst).
```
KV bytes = 2(K,V) × n_layers × live_blocks × BLOCK × kv_heads × head_dim × 2(bf16)
disagg lời khi:   transfer_cost (một lần)  <  Σ spike_ITL bỏ được (lặp mỗi prefill)
```
Prompt dài hơn → `live_blocks` tăng → transfer tax tăng tuyến tính; điểm hòa vốn dịch (nhưng spike cũng tăng).

**Neo code** (`bench/disagg.py` — 2.9 là demo, không phải một `serving/` module riêng):
```python
def _prefill_one(model, plen, device):                   # :37  prefill 512-tok vào paged cache (worker output)
def measure_kv_transfer(model, device):                  # :58
    live_blocks = len(src._py_table[0])                  # :63
    kv_bytes = 2 * n_layers * live_blocks * BLOCK * kv_heads * head_dim * 2  # :64-72
    def _copy():                                          # :74
        for layer in range(n_layers):
            dst._pool_k[layer].copy_(src._pool_k[layer])  # :76  32 copy nhỏ per-layer → launch-bound
            dst._pool_v[layer].copy_(src._pool_v[layer])  # :77
    # median 20 lần :82-88
def measure_itl_contrast(model, prefill_model, device):  # :101
    decode_only = [short(i) for i in range(96)]          # :114  disagg decode worker (chỉ short)
    co_located  = [long(i) if i%6==5 else short(i) ...]  # :116  chèn long mỗi 6 request
    # serve_continuous cả hai → summarize → ITL percentile :119-126
```

**Hình ảnh — co-located spike vs disagg clean:**
```
CO-LOCATED (1 GPU):  decode…decode…[PREFILL long burst]…decode…   ITL: 6│6│6│ 61.5 │6│6  ← p99 spike
DISAGG:              prefill-worker: [prefill]──KV(16.8MB, 0.18ms one-time)──► decode-worker
                     decode-worker:  decode…decode…decode…decode          ITL: 6│6│6│6│6   p99 = 20.2 ms
   goodput @ SLO(ITL p99 ≤ 30ms):  disagg MEETS (20.2)  ;  co-located VIOLATES (61.5)   ← 2.2 goodput
```
KV bytes = `2 × n_layers × live_blocks × 16 × kv_heads × head_dim × 2` — 16.8 MB cho prompt 512-tok.

**Số đo THẬT** `[measured · sm120 Blackwell · bench/RESULTS.md · A1 R4.6]`:
```
KV-transfer tax:  0.180 ms (16.8 MB, 187 GB/s ; analytic 0.061) — 32 copy nhỏ/layer → launch-bound
ITL p99 disagg vs co-located:  20.2 vs 61.5 ms = 3.0× tốt hơn  ; agg 1707 vs 1038 tok/s
goodput @ ITL p99 ≤ 30ms:  disagg MEETS (20.2) ; co-located VIOLATES (61.5)
```
Honest scope: single-GPU sim (D2D proxy cho NVLink/RDMA thật ở node scale).

**Frontier / cổng.** Disagg (DistServe/Mooncake) là kiến trúc serving node-scale 2026. Interview: "khi nào
disaggregate prefill/decode?" → khi ITL SLO chặt + prompt dài burst; giá thật là bandwidth chuyển KV ở node
scale. Trait = trade-off-explicit + honest-scope (single-GPU sim). Scarce (inference/systems).

---

## Bảng số đo THẬT (chạy lại được — DoD-là-profile)

| Concept | Đo | Kết quả | Nguồn |
|---|---|---|---|
| 2.1 | oracle use_cache T vs F | token-exact IDENTICAL | CPU `s2_checks.py` |
| 2.2 | percentiles [10..50] nearest-rank | p50=30, p95=p99=50 | CPU `s2_checks.py` |
| 2.2 | 1 spike request | throughput 15.38 vs **goodput 0.00** tok/s | CPU `s2_checks.py` |
| 2.3 | decode B=1 eager | **51 tok/s = 16%** roof; 89 GB/s = 16% HBM; ~955 launch/tok | measured · R1 |
| 2.3 | strip A nosync | 48→50 (sync ~free) | measured · R1 |
| 2.4 | AI toy 2PB/(2P+B·KV) | B=1→1.0, 32→32.0, 128→127.96 (**AI≈B**) | CPU `s2_checks.py` |
| 2.4 | agg(B=32)/agg(B=1) | **24.9×**; peak 12,220 @B=256=66×; crossover B≈128 | measured · R3a |
| 2.5 | continuous vs wave heavy-tail | **2.30× wall / 2.93× step**; util 72.8 vs 24.9% | measured · R3b |
| 2.5 | padding tax | 9.6 vs 6.3 ms/step = **1.27×** (2.93→2.30) | measured · R3b |
| 2.5 | R4.2 chunked (honest neg) | ITL p99 ×1.03→1.91, throughput 547→145 DOWN | measured · R4.2 |
| 2.6 | spec ALWAYS-WRONG drafter | **exact=True**, accept 0% (losslessness) | CPU `s2_checks.py` |
| 2.6 | spec speedup (n-gram) | ×1.21-1.29 wall / 1.33-1.39 tok/fwd | measured · R4.3 |
| 2.7 | cudagraph B=1 step | **−74.3%** (15.38→3.96 ms); **253 tok/s = 77%** roof | measured · R4.4 |
| 2.7 | launch collapse | 54-68 cudaLaunchKernel → **1** cudaGraphLaunch | measured · R4.4 |
| 2.8 | MLA identity float64 | max\|naive−absorbed\| = **3.33e-15** (eps) | CPU `s2_checks.py` |
| 2.8 | MLA KV bytes (DS-V3 dims) | **1152 B = 1.8%** MHA; 3.56× < GQA-8 | CPU + measured · R4.5 |
| 2.9 | disagg ITL p99 | **20.2 vs 61.5 ms = 3.0×**; KV-transfer 0.180 ms | measured · R4.6 |

*CPU numbers: `PYTHONPATH=src python scratchpad/s2_checks.py`. GPU numbers: `[measured · sm120 Blackwell ·
bench/RESULTS.md]` (Triton/CUDA/multi-GPU không tái lập trên card này ⇒ ledger-cited, không bịa).*

---

## Checklist recall COLD (che phần trên, tự trả lời — đây là cách re-learn)

1. **2.1** Vì sao logprob đọc từ logits *chưa* scale temperature/top-p? Hai đường `use_cache` True/False phải cho gì, lệch = bug ở đâu?
2. **2.2** Dẫn `latency ≈ TTFT + ITL·N`. Vì sao goodput dùng `max(itls)` không `mean`? Vì sao ITL percentile gộp POOL?
3. **2.3** Workload memory-bound (AI≈1) mà achieved BW chỉ 16% HBM — mâu thuẫn không? "Workload bound" vs "run bound"?
4. **2.4** Dẫn `AI = 2PB/(2P+B·KV)`. Vì sao `AI≈B` khi B nhỏ? Crossover memory→compute ở B nào và vì sao (ridge)?
5. **2.4** Vì sao batching (không phải kernel nhanh hơn) là lever cho workload AI≈1? (dẫn từ roofline)
6. **2.5** Vì sao wall-ratio (2.30) < step-ratio (2.93)? "Padding traffic" là gì? Vì sao `view_len` KHÔNG được đọc in-graph?
7. **2.5** R4.2 chunked prefill: vì sao đo được kết quả ÂM (ITL p99 tăng)? Sản xuất (Sarathi) làm khác gì?
8. **2.6** Chứng minh losslessness: vì sao drafter SAI hoàn toàn vẫn cho đúng chuỗi greedy? `truncate(base+1+accepted)` đóng vai gì?
9. **2.7** Vì sao dense buffer (shape mọc) + `torch.cat` cache KHÔNG capture được mà paged kernel (grid `(B,H)` cố định) thì được? Vì sao restore `lengths`?
10. **2.8** Viết `(W_UK^T q)·c = q·(W_UK c)` và chỉ ra vì sao nó cho cache `c_KV` thay K,V *zero quality change*. Vì sao PHẢI decoupled RoPE?
11. **2.9** Vì sao disagg cải thiện ITL p99 (spike) chứ không nhất thiết p50? (nối `request_meets_slo` max-gap). Điểm hòa vốn dịch đâu khi prompt dài hơn?

> Trả lời cold được cả 11 = **S2 thật sự OWNED** (interview-grade). Vấp câu nào → mở đúng mục đó, hoặc
> blank-slate hàm tương ứng (`raise NotImplementedError` → test đỏ → tự dẫn → xanh) để re-own bằng tay.

---

## Xuyên suốt — bức tường bộ nhớ, ba lần leo

```
16% (R0/R1 eager, launch-bound)  ──torch.compile fuse──►  53% (R1 compiled)  ──CUDA graph──►  77% (R4.4)
                                                                                              │
        song song: batching nâng AI≈B (2.4/2.5) ; speculation nâng token/forward (2.6) ;      │
        MLA co cache (2.8) ; disagg tách profile (2.9) — mỗi cái một đòn bẩy KHÁC lên cùng tường
```
Gap còn tới vLLM (~1.3× latency / ~1.5-2× throughput) = chunked-prefill *piggyback* (R4.2b, đo được là hướng
đúng sau khi R4.2 scheduling-only FALSIFIED) + prefix cache. Đó là node kế của série.

---

*Cross-ref: `roadmap/S2_serving_engines.md` (spine reference, 9 Bài) · `derivations/M2_transformer_forward.md`
(forward pass, KV-cache/GQA sinh ra bức tường này) · `PROGRESS.md` (ledger 89 Bài) · `CURRICULUM.md` (con
đường) · `performance/notes/A1_*` (design notes R2→R4.6) · `bench/RESULTS.md` §A1 R0-R4.6 (số đo gốc). Concept
kế: S3 — CUDA-core kernel ladder (A2: GEMV→softmax→RMSNorm→GEMM, leo từ trần HBM lên trần compute).*
