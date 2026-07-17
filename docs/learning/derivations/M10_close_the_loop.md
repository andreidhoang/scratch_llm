# M10 — Close the Loop + F1–F9 Ablations · First-Principles Derivations

> **Đây là gì.** Bản dẫn-xuất **từ nguyên lý gốc** cho *capstone của phía model*: sau khi M1–M9 tiện xong
> từng bánh răng (byte → BPE → Transformer → MoE → optim → train → post-training → frontier), M10 **lắp cỗ
> máy lại và lên dây cót** — một script biến corpus thành model biết nói (10.1), một report card làm oracle
> đo "tốt lên" bằng MỘT con số (10.2), một thang tier `C=6ND` định cỡ run trả tiền (10.3), và một *ablation
> study* pre-registered biến "nghe nói nó tốt" thành kết luận nhân quả (10.4). Mỗi mục: (1) **câu hỏi**
> falsifiable, (2) **sự thật nền tảng**, (3) **dẫn xuất** có math, (4) **neo code** `file·func·line`, (5)
> **hình ảnh** (ASCII + shape/stride/dtype + số hand-traced), (6) **số đo THẬT** chạy trên chính repo này,
> (7) **frontier framing** + cổng phỏng vấn.
>
> **Cách dùng để re-learn.** Đọc phần *Câu hỏi* của mỗi mục → **tự trả lời cold** (che phần dưới) → mở ra
> đối chiếu. Cuối doc có **checklist recall cold** + bảng số đo. Đây là bạn đồng hành của
> `roadmap_model/M10_close_the_loop_and_ablations.md` (reference chung) — doc này là *derivation lab* có số đo.
>
> **Cảnh báo neo.** Line number **trôi** theo commit. Cite ở đây pin theo HEAD ngày 2026-07-14 (`1e7dbc6`).
> Nếu lệch, `grep` tên hàm — đừng tin số dòng cứng (luật repo: verify, don't trust).
>
> **Nguồn số đo (FOP-4, claims honesty).** Đây là pillar **CPU-runnable**: mọi số trong doc chạy lại được
> bằng các `python -c` ghi trong từng mục — đó là "DoD là một profile, không phải test xanh". **Ranh giới
> trung thực:** số ở *unit/nano level* là **[FACT]** (đo thật, ghi rõ device); số *headline* của run trả
> tiền (CORE d20, Muon iso-FLOP ≥15%) là **[PREDICTED]** rental-gated — đánh dấu rõ, không trộn.

---

## Bức tranh lớn — từ "viện bảo tàng component" đến "một model biết nói"

M1–M9 cho ta 461 test xanh + một serving stack tiên tiến. Nhưng **test-component-xanh KHÔNG chứng minh
cả pipeline *hợp* thành một thứ sinh được văn bản mạch lạc.** M10 là mũi tên đóng vòng đó:

```
                        ┌──────────────────── M10: ĐÓNG VÒNG ─────────────────────┐
  corpus (bytes)        │                                                          │
      │                 │  10.1  speedrun spine — one script, one --depth knob     │
      ▼                 │  ┌──────────┬──────────┬────────┬──────┬────────┐        │
  ┌────────────┐        │  │ tokenizer│ pretrain │[midtr] │[sft] │ eval   │  sample│
  │ train_bpe  │───────►│  │ byte-BPE │MuonAdamW │ A4 slot│A5slot│ report │───────►│──► "the quick brown
  └────────────┘        │  └──────────┴──────────┴────────┴──────┴───┬────┘  talk  │    fox jumps over…"
   M1 tokenizer         │   M1        M3/M4       (skip@0) (skip@0)   │            │
                        │                                            ▼            │
                        │        10.2  REPORT CARD = oracle của vòng lặp          │
                        │        ┌─────────────┬────────────┬──────────────┐      │
                        │        │ val_bpb     │ multiple-  │ generative   │      │
                        │        │ (intrinsic) │ choice (LL)│ (gen+verify) │      │
                        │        └─────────────┴────────────┴──────────────┘      │
                        │              │  → CORE-style aggregate (một số)         │
                        └──────────────┼──────────────────────────────────────────┘
                                       ▼
        10.3  THANG TIER — C=6ND định cỡ run nào đáng trả tiền
        nano($0,giây,plumbing) → d20(~$100,biết nói) → ablation(standing 24GB,$0) → stretch
                                       │
                                       ▼
        10.4  ABLATION STUDY = phương pháp DẪN XUẤT (F1–F9)
        đổi ĐÚNG một biến · giữ C=6ND cố định (iso-FLOP) · pre-register số falsifiable + KILL · đo
        F1 Muon · F2 MTP · F3 de-confound spec-decode · F4 bf16/compile · F5 MLA · F6 aux-loss-free · F7 GRPO · F9 QK-Clip
```

**Áp lực xuyên suốt M10** — mỗi thiết kế là lời giải cho một áp lực cụ thể (đúng như M2 dạy *áp lực → lời giải*):

| Mục | Áp lực | Lời giải |
|---|---|---|
| 10.1 | **Composition risk** — 34 component có thể *không* ghép được | một run end-to-end thật, không nhánh giả lập |
| 10.2 | **Comparability** — "tốt hơn" phải là MỘT số so được | bpb (byte-denominator, tokenizer-invariant) + LL-scored MC + CORE-style |
| 10.3 | **Compute economics** — GPU-hour là tiền thật, N-vs-D là câu hỏi $$ | Chinchilla `C=6ND` làm cân bắt buộc TRƯỚC khi trả tiền |
| 10.4 | **Causal attribution** — bê hết kỹ thuật vào 1 run ⇒ kết luận zero | iso-FLOP + one-variable + pre-registered falsifier + KILL |

Một sự thật frontier: **kỷ luật (chứ không phải kết quả) là kỹ năng lab tuyển.** M10 dạy cách *chứng minh*
một thứ hoạt động, không phải cách *đồn* nó hoạt động.

---

## 10.1 · Close the loop — khe hở "bảo tàng vs model" + speedrun spine

**Câu hỏi.** Vì sao 461 test xanh + một serving stack tiên tiến **VẪN không chứng minh** ta có một model
biết nói — và cái gì là *bằng chứng tối thiểu* rằng cả pipeline hợp thành một thứ hoạt động?

**Sự thật nền tảng.** Component test là "bánh răng đạt dung sai"; "model biết nói" là "đồng hồ chạy đúng
giờ". Hai điều **khác loại**: bánh răng đúng lẻ KHÔNG kéo theo cỗ máy chạy đúng — khớp nối, thứ tự, scale,
dtype có thể sai ở chỗ không test nào lẻ chạm tới. Bằng chứng duy nhất rằng hàm hợp `decode ∘ eval ∘ train ∘
encode` *hợp được* là **chạy đúng chuỗi hàm đó một lần, end-to-end, không nhánh nào giả lập**, và ra một
sample mạch lạc + một metric chạy được.

**Dẫn xuất — vì sao spine 4-stage là tối thiểu-đủ.** Một LM là hàm hợp
```
f = decode ∘ eval ∘ train ∘ encode
```
Để chứng minh nó *composes*, ta phải hiện thực hoá đúng chuỗi đó, không stub: (1) `train_bpe` trên corpus →
(2) pretrain Transformer bằng MuonAdamW → (3) report card (`val_bpb`) → (4) sample. Midtrain (A4) + SFT (A5)
bị **cố ý bỏ** khỏi pre-flight (chúng tái dùng cùng `train` loop; mục tiêu Phase 0 là "pipeline compose",
KHÔNG phải "model giỏi") — trong code chúng là *slot* skip ở 0 step.

*Vì sao một knob `--depth` duy nhất?* nanochat aspect-ratio scaling: `d_model = 64·depth`, `head_dim` ghim
128 ⇒ `n_heads = d_model//128`. Một số điều khiển **cả chiều rộng lẫn sâu**, nên "nano" (depth 4) và "d20
headline" (depth 20, ~564M) là *cùng một code path, khác một số* — bất biến bảo vệ run trả tiền: **nano
pre-flight phải xanh TRƯỚC khi tiêu $100 cho d20.**

*Cái giá của việc KHÔNG đóng vòng — đo được.* `bench/RESULTS.md:383`: speculative-decode acceptance **54–72%
trên random-weight target** (lẽ ra ~0). Đó không phải bug — nó là **confound**: model random-weight có
entropy đầu ra thấp → greedy suy biến lặp → n-gram drafter trúng đậm. Ta đang đo *tính chất của
model-chưa-train*, tưởng đang đo prompt. Chỉ một model THẬT mới gỡ được (F3, §10.4).

**Neo code** (`src/scratch_llm/speedrun.py`):
```python
def run_speedrun(cfg) -> SpeedrunResult:                              # :350  chain 4+ stage
    tokenizer, tokens, prompt_ids, stage_name = stage_tokenizer(cfg)  # :355  train_bpe / shards
    ...
    if tokens.size <= cfg.context_length + 1:                         # :358  guard "corpus quá ngắn"
        raise ValueError(...)
    model, stage_name = stage_pretrain(cfg, tokens, len(tokenizer.vocab))  # :366  MuonAdamW
    model = stage_midtrain(cfg, model)   # :369  A4 slot — 0 step = skip
    model = stage_sft(cfg, model, tokenizer)  # :372  A5 slot — 0 step = skip
    card = stage_eval(cfg, model, tokenizer, tokens)   # :376  report card
    sample = stage_sample(cfg, model, tokenizer, prompt_ids)  # :379  the talking artifact

def model_config_for_depth(depth, vocab_size, context_length):       # :109  aspect-ratio scaling
    d_model = 64 * depth                                             # :112  width ← depth
    n_heads = max(1, d_model // 128)                                 # :113  head_dim pinned 128
    return ModelConfig(..., n_layers=depth, tie_embeddings=False)    # :117,:120  untied (F1 routing)

def stage_midtrain(cfg, model):                                      # :228  A4 slot
    if cfg.midtrain_steps == 0: return model                         # :230  skip = identity
    raise NotImplementedError("midtrain is the A4 rung …")           # :232  fail-loud nếu bật
```
`stage_pretrain` (:196) gọi `train(TrainConfig(optimizer=cfg.optimizer, warmup=max(1, steps//20), …))`
(:207–219); `cfg.optimizer` mặc định `"muon_adamw"` (`SpeedrunConfig` :59). `_nano_config()` (:404) =
tier pre-flight (depth 4, vocab 384, ctx 64, 60 step, CPU).

**Hình ảnh — data journey qua spine** (nano: depth 4, vocab 384, ctx 64):
```
_BUILTIN_CORPUS (str, ~180 char) ── ×reps ──► text (str, ≥5000 char)          [_load_corpus :124]
   │ train_bpe → Tokenizer(vocab 384 = 256 byte + 128 merge)                   [stage_tokenizer :170]
   ▼
tokens : np.ndarray shape (1682,) dtype int64      ← encode(text)              [:192]
   │ get_batch → inputs (B=16, S=64) long, targets (16,64) long
   ▼
TransformerLM(d_model 256, 4 layer, 2 head, head_dim 128) ── train 60 step ──► weights học được
   │ stage_eval: val = tokens[-256:] (int64), num_bytes = len(decode(val).encode('utf-8'))
   ▼
ReportCard(val_bpb ≈ 0.011)          [stage_eval :317]
   │ stage_sample: prompt_ids = encode(text[:24]) → generate(48 tok) → decode
   ▼
sample : str  " jumps over the lazy dog. a language model learns to predict the next token…"
```
Hand-trace kích thước: nano vocab 384 = 256 byte-token + 128 merge (`_nano_config` :410 comment); `d_model =
64·4 = 256`; `n_heads = max(1, 256//128) = 2`; `head_dim = 256//2 = 128` (pinned). `n_params` đếm thật ở
`run_speedrun` (:394) `sum(p.numel())`.

**Số đo THẬT** (`python -m` phần này chạy `run_speedrun(_nano_config())` — CPU, Mac, HEAD 1e7dbc6):
```
stages     : tokenizer → pretrain → eval → sample      ⇒ LOOP CLOSES  [FACT]
n_params   : 3,410,176        (= 3.41M, khớp ledger RESULTS.md:721)
n_tokens   : 1,682            (khớp ledger)
val_bpb    : 0.0113           nats/tok 0.0359   (in-sample repeated corpus — metric CHẠY, không phải model tốt)
wall       : 117 s            (CPU-Mac; ledger GPU sm120 = 8.3 s @150 step, val_bpb 0.0206 — RESULTS.md:721,722)
sample     : ' jumps over the lazy dog. a language model learns to predict the next token from'
```
Sample **nối đúng corpus** (không phải nhiễu) ⇒ pipeline compose. `val_bpb` là *in-sample* (corpus lặp) —
nó chứng minh **metric chạy được**, KHÔNG phải model tốt. Run thật cần held-out split + $100 d20 (rental).

**Frontier / cổng.** Reference: **nanochat của Karpathy** (`speedrun.sh` + `--depth`). Ta *giống* spine
4-stage + knob depth; *mạnh hơn* ở green-CI + adapter-oracle; *khác* ở `tie_embeddings=False` (:120) trong
khi nanochat tie (`modeling_nanochat.py:434`) — untie để routing Muon rõ (F1: tied 2-D tensor phải đi
AdamW, gây mập mờ nếu tie). Gate = "vì sao 'component test xanh' KHÔNG đủ để tuyên bố có model — bằng chứng
tối thiểu của 'close the loop' là gì?" (một lần chạy end-to-end thật, không nhánh giả lập, ra sample mạch
lạc + metric chạy). Trait = **execution > analysis** (ship cỗ máy chạy, không thêm component thứ 34).

---

## 10.2 · Report card — oracle của vòng lặp (bpb · MC · generative)

**Câu hỏi.** Một model "tốt lên" nghĩa là gì bằng MỘT con số so sánh được — và vì sao **bits-per-byte**
(không phải per-token loss) là metric đúng, còn multiple-choice phải chấm **bằng likelihood** chứ không phải
bằng generation?

**Sự thật nền tảng.** Không có oracle thì "đóng vòng" vô nghĩa. Report card gộp ba loại tín hiệu: (1)
**intrinsic** — `val_bpb`, model nén văn bản tốt cỡ nào; (2) **multiple-choice** — ARC/MMLU, chọn đáp án
đúng, chấm bằng log-prob; (3) **generative** — GSM8K/HumanEval, sinh lời giải rồi chấm bằng verifier.

### 10.2a — bpb: vì sao mẫu số là *byte*, không *token*

**Dẫn xuất.** Từ định nghĩa cross-entropy, tổng negative log-likelihood teacher-forced của một stream:
```
NLL_nats = Σ_t −ln p(x_t | x_<t)          (nats)
bpb      = NLL_nats / ln 2 / n_bytes       (bits per UTF-8 byte)
```
*Vì sao byte, không token?* Nếu mẫu số là **token**, một tokenizer "gian" nhồi nhiều byte/token sẽ có ít
token hơn cho cùng văn bản → per-token loss thấp hơn **miễn phí**, không phải vì model giỏi hơn. Chia cho
**số byte UTF-8** mà stream giải mã ra ⇒ metric so sánh **model**, không so sánh **tokenizer**
(tokenizer-invariant). Đây là lý do nanochat report `val_bpb` làm headline pretrain number.

*Invariant kiểm chứng (cheapest oracle của vòng lặp):* model **đều** (all-equal logits trên V) → `p = 1/V`
mỗi token → `−ln p = ln V` nats/token → `log₂(V)` bit/token. Nếu stream 1 byte/token thì bpb = log₂(V).

**Neo code** (`src/scratch_llm/eval/metrics.py`):
```python
def bits_per_byte(model, token_ids, num_bytes, *, context_length=None, device="cpu"):  # :37
    ...
    total_nll = 0.0  # nats
    for start in range(0, n - 1, ctx):          # :70  cửa sổ KHÔNG chồng (nanochat/GPT-2 chunked bpb)
        inp = ids[start : start + ctx]
        tgt = ids[start + 1 : start + 1 + int(inp.numel())]   # :72  mỗi transition x_<t→x_t chấm 1 lần
        logits = model(inp[:m].unsqueeze(0).to(device))[0]    # :76  (m, V)
        logp = torch.log_softmax(logits.float(), dim=-1)      # :77  fp32
        nll = -logp.gather(-1, tgt[:m]…).squeeze(-1).sum()    # :78  Σ −ln p(token đúng)
        total_nll += float(nll)
    bpb = total_nll / math.log(2) / num_bytes                 # :81  → bits / byte
    return BpbResult(bpb, total_nll / n_tokens, n_tokens, num_bytes)  # :82  nats/token = total/scored
```

**Hình ảnh — cửa sổ không chồng + boundary token** (stream 200 byte-token, ctx 64):
```
ids: [t0 t1 t2 … t63 | t64 … t127 | t128 … t191 | t192 … t199]
       └──window 0───┘ └─window 1─┘ └─window 2─┘ └─win 3─┘
         chấm t1..t63    t65..t127    t129..t191   t193..t199   ← MỖI transition đúng 1 lần
       ▲
       t0 KHÔNG được chấm (không có gì đứng trước) ⇒ scored = 199, không phải 200
```
`total_nll` (fp32 scalar) / `math.log(2)` / `num_bytes` (int). `nats/token = total_nll / n_tokens` chia cho
**số transition đã chấm** (199), còn `bpb` chia cho **số byte** (200) — chênh chính là boundary token.

**Số đo THẬT** (uniform model, V=256, 200 byte-token, 1 byte/token):
```
log₂(V)             = 8.000000
measured nats/token = 5.545177   =  ln(256) = 5.545177   ⇒ EXACT (invariant per scored token)  [FACT]
measured bpb        = 7.960000   =  (199/200)·8.0        ⇒ boundary token: 199 scored / 200 byte
```
**Gap được đặt tên (code review):** docstring `:13–14` nói "1 byte/token → log₂(V) bpb" — chính xác *tới
boundary token*: per-token là log₂(V) **đúng tuyệt đối**, còn bpb = `(scored/bytes)·log₂(V)` → log₂(V) chỉ
khi stream dài (199/200 → 1). Đây là xấp xỉ chunked, hơi bi quan, đúng như docstring `:48–52` thừa nhận
("first token of each window predicts with truncated context"). Đo được ⇒ hiểu đúng.

### 10.2b — MC scored by likelihood, length-normalized

**Dẫn xuất.** Model bé **chưa biết tuân format** "trả lời A/B/C/D". Nhưng likelihood của chuỗi đáp án vẫn
tách được tín hiệu: chấm mỗi option bằng `log p(option | prompt)` teacher-forced, chọn argmax. *Vì sao
length-normalize?* Raw summed log-prob = `Σⱼ log p(oⱼ)` — mỗi số hạng **âm** ⇒ option **dài hơn có nhiều
số hạng âm hơn** ⇒ tổng âm hơn ⇒ raw sum **thiên vị option NGẮN**. Chia cho số token đáp án ⇒ *trung bình
log-prob mỗi token* ⇒ khử bias độ dài (convention ARC/MMLU).

**Neo code** (`src/scratch_llm/eval/multiple_choice.py`):
```python
def option_logprob(model, prompt_ids, option_ids, *, length_normalize=True):  # :28
    seq = torch.tensor([list(prompt_ids)+list(option_ids)], …)   # :42  prompt ⊕ option
    logp = torch.log_softmax(model(seq)[0].float(), dim=-1)      # :44  (T, V) fp32
    for j, token in enumerate(option_ids):
        total += float(logp[p + j - 1, token])   # :48  logit ở pos (p+j−1) dự đoán option token j
    return total / len(option_ids) if length_normalize else total  # :49  ← chia độ dài = khử bias
```

**Số đo THẬT** (uniform model, mỗi token log-prob = −ln256; short=1 token, long=4 token, cùng nội dung/token):
```
raw  sum : short(1tok) = −5.5452   long(4tok) = −22.1807   ⇒ short THẮNG giả bởi 16.6355 nats
norm mean: short       = −5.5452   long        = −5.5452    ⇒ TIE (bias khử, diff = 0.000000)   [FACT]
```
Với model đều (mọi option ngang nhau per-token), *đúng đắn* là hoà; raw sum lại chấm short thắng đậm — chính
là bias length-normalize sinh ra để chặn. Hại nhất khi độ dài đáp án tương quan với đúng/sai.

### 10.2c — CORE-style aggregate

**Dẫn xuất.** Accuracy thô không so được giữa task khác số lựa chọn (chance của task 4-option là 0.25, của
2-option là 0.5). Centering đặt **chance = 0, perfect = 1**:
```
core_style = mean_task ( (acc − base) / (1 − base) ),   base = 1/n_options = MCTask.random_baseline
```

**Neo code** (`src/scratch_llm/eval/report_card.py`):
```python
def core_style_score(task_accuracy, baselines):        # :39
    centered = [(acc - baselines[name]) / (1 - baselines[name])   # :42-46  (acc−base)/(1−base)
                for name, acc in task_accuracy.items() if baselines[name] < 1.0]
    return sum(centered)/len(centered) if centered else 0.0        # :47
```
`build_report_card` (:82) gọi `bits_per_byte` (:102), vòng MC → `evaluate_multiple_choice` (:109) nạp
`mc[name]`+`baselines[name]`, vòng gen → `evaluate_generative` (:115), rồi `ReportCard(..., core_style=
core_style_score(mc, baselines) if mc else None)` (:120–125). Naming trung thực (docstring `:1–8`):
`core_style_score` là **công thức aggregate** của DCLM-CORE — chỉ là điểm CORE *chính thức* khi cho đúng bộ
task CORE chính thức; trên task tuỳ ý nó là "CORE-*style*", không phải leaderboard metric.

**Số đo THẬT** (3 task: arc4 base .25 acc .25 / mmlu4 base .25 acc .50 / bool2 base .50 acc .75):
```
raw mean acc = 0.5000       ← trộn task khác chance ⇒ vô nghĩa
core_style   = 0.2778   [FACT]   arc4→(.25−.25)/.75=0.000 (đúng chance)
                                 mmlu4→(.50−.25)/.75=0.333
                                 bool2→(.75−.50)/.50=0.500   mean[0,.333,.5]=0.2778
```

**Frontier / cổng.** Reference: **nanochat report card** (`val_bpb` headline + DCLM-CORE 2406.11794 aggregate);
generative reuse `sampling.generate` + verifiable-reward grader trong `rewards/` (nối thẳng A5). Gate =
"chấm multiple-choice bằng likelihood vs generation khác nhau ra sao, vì sao model nhỏ CHỈ chấm được bằng
likelihood?" + "vì sao mẫu số bpb là byte?". Trait = **claims honesty** (nhãn "CORE-style ≠ CORE chính
thức"; boundary-token gap nêu rõ). Scarce bucket = eval discipline (table-stakes nhưng silently-screened).

---

## 10.3 · Model-size tiers + cross-check token `C = 6ND`

**Câu hỏi.** Cho một ngân sách compute cố định, chọn model bao to (N) và train bao nhiêu token (D) thế nào
— và vì sao `C = 6·N·D` là cái cân **bắt buộc** trước khi trả tiền?

**Sự thật nền tảng.** Ngân sách compute là **tiền thật** (GPU-hour). Cùng một số FLOP, nên đổ vào *model to
hơn* (N ↑) hay *nhiều dữ liệu hơn* (D ↑)? Đây là câu hỏi Chinchilla. `C = 6ND` là **hối suất** biến hai
trục thành một, cho ta ước lượng D từ (C, N) *trước khi* thuê GPU.

**Dẫn xuất — vì sao hằng số là 6, không phải 2.** Một forward qua model dense: mỗi weight tham gia đúng
**1 multiply + 1 add** cho mỗi token ⇒ `2·N` FLOP/token. Backward tốn ~**2× forward**: cần grad theo
*input* (chain rule qua mỗi matmul) *và* grad theo *weight* — mỗi cái ~`2N` ⇒ backward ~`4N`. Tổng train:
```
C_per_token = 2N (fwd) + 4N (bwd) = 6N        ⇒   C = 6·N·D
```
(Nếu chỉ inference/forward, hằng số là 2, không 6 — đó là bẫy interview thường gặp.)

*Cross-check token của d20 (headline).* Cho `C ≈ 4×10¹⁹` (8×H100 vài giờ) và `N ≈ 564M`:
```
D = C / (6N) = 4e19 / (6 · 5.643e8) = 1.18e10 ≈ 11.8B token
D / N = 20.9 tok/param   ⇒ đúng vùng Chinchilla-optimal (~20)
```

**Neo code** (`src/scratch_llm/speedrun.py` + `src/scratch_llm/scaling/isoflop.py`):
```python
# speedrun.py — aspect-ratio scaling: một số depth quyết cả N
d_model = 64 * depth                          # :112
n_heads = max(1, d_model // 128)              # :113  guard max(1,·) chống head=0 khi d_model<128
# isoflop.py — the compute identity
FLOPS_PER_PARAM_TOKEN = 6.0                    # :28   2 fwd + 4 bwd
def compute_from_params_tokens(n, d): return 6.0 * n * d    # :73  C = 6ND
def tokens_from_compute_params(c, n): return c / (6.0 * n)  # :78  D = C/(6N) — bridge D_opt law
def check_exponent_sum(a, b, tol=0.05):        # :103  a+b≈1 gate: C=6ND FORCES exponents to sum to 1
    if abs(a + b - 1.0) > tol: raise ValueError(...)        # :110  fail-loud trước extrapolation
```

**Hình ảnh — thang tier (rủi ro-tiền) + aspect-ratio journey:**
```
depth ──► d_model=64·depth ──► n_heads=d_model//128 ──► head_dim=128 (pinned)
  4    ──►     256          ──►      2               ──►    128     ⇒ nano  N=3.41M   ($0, giây)
 20    ──►    1280          ──►     10               ──►    128     ⇒ d20   N=564M    (~$100, biết nói)

TIER LADDER (docs/FRONTIER_2026_ABLATIONS.md §2):
  nano  ($0, giây, CPU)     → chứng minh PLUMBING (10.1)                      [FACT]
  d20   (~$100, 8×H100 2-4h)→ model biết nói + report card công khai, CORE≈GPT-2  [PREDICTED]
  ablation (standing 24GB $0)→ nơi KHOA HỌC sống (10.4)
  stretch d26/d32           → chỉ khi một ablation biện minh cho scale
```

**Số đo THẬT** (param đếm thật bằng `sum(p.numel())` trên module THẬT):
```
depth  4  → d_model  256  n_heads  2  head_dim 128  →  N = 3,410,176        [FACT]
depth 20  → d_model 1280  n_heads 10  head_dim 128  →  N = 564,317,440 (~564M) [FACT]
C=6ND cross-check (d20): C=4.0e19, N=564M → D = C/(6N) = 1.181e10 (~11.8B tok), D/N = 20.9  [FACT]
reverse: 6·N·D = 4.000e19 == C  (identity đóng)
target: CORE ≈ 0.256–0.269 (GPT-2 grade), ~$48–100, 8×H100      [PREDICTED — rental-gated]
```
(N₂₀ đo được 564M > roadmap "~561M" vì `vocab 65536` untied embed+head chiếm phần lớn — số đo thắng số ước.)

**Frontier / cổng.** Reference: **nanochat d20** (~$100, CORE≈GPT-2). Đối chiếu 2026: DeepSeek-V3
671B/37B-active, GLM-4.5, Kimi-K2 1.04T/32B — MoE deep-narrow ⇒ `N_total ≫ N_active`; `C=6ND` dùng
`N_active` cho FLOP nhưng KV/serve theo tổng. Gate = "dẫn `C=6ND` từ đếm FLOP fwd/bwd — vì sao hằng số 6?"
+ "khi nào ta *cố tình rời* Chinchilla-optimal?" (inference-aware: nếu serve nghìn tỉ token, over-train một
model NHỎ hơn — Llama-3 ~1875 tok/param — vì serve chi phối lifetime cost; Sardana 2401.00448). Trait =
**roofline-first / predict-the-number** (đoán D trước khi thuê GPU). Scarce bucket = scaling (differentiator).

---

## 10.4 · Ablation study như một phương pháp DẪN XUẤT: F1–F9

**Câu hỏi.** Làm sao *chứng minh* (không phải đồn) rằng một kỹ thuật 2026 (Muon, MTP, aux-loss-free, MLA…)
thực sự **load-bearing** — cấu trúc thí nghiệm nào biến "nghe nói nó tốt" thành kết luận nhân quả bảo vệ
được trước hội đồng?

**Sự thật nền tảng.** Bê hàng chục kỹ thuật lấp lánh vào một run to ⇒ biến chồng biến, tiền đốt, kết luận
**zero**. Ablation study là kỷ luật gỡ chuyện đó: đổi **đúng một biến**, giữ **compute cố định (iso-FLOP)**,
**pre-register** số falsifiable + ngưỡng **KILL** *trước khi chạy*, rồi đo. Đánh đổi: chậm hơn (một run mỗi
biến) đổi lấy *nhân quả* — chính kỷ luật này (không phải kết quả) là kỹ năng frontier-lab tuyển.

**Dẫn xuất — bốn trụ của một ablation bảo vệ được:**

**(1) Iso-FLOP là điều kiện KHÔNG-thể-bỏ.** Nếu Muon "thắng" nhưng dùng nhiều FLOP hơn, cái thắng có thể chỉ
là "train lâu hơn" — vô nghĩa. Giữ `C = 6ND` cố định (cùng N, cùng D) biến câu hỏi thành *loss-per-FLOP*,
cái duy nhất so được. Trong code, hai arm **share N** (same config, same seeded init) và **D** (same
step/batch/context) ⇒ C cố định *by construction*, không cần bookkeeping.

**(2) Pre-register + KILL chặn p-hacking.** Không pre-register thì ta *post-hoc* chọn ngưỡng để "thắng". Một
prediction falsifiable (F1: "Muon đạt loss của AdamW với ≥15% ít token, hoặc ≥0.02 nats thấp hơn @iso-FLOP")
+ KILL ("token saving <5% hoặc diverge ở LR tái dùng của AdamW") ràng buộc ta trung thực TRƯỚC khi thấy số.

**(3) Tuned baseline bắt buộc.** Deflation literature (arXiv 2509.02046): các headline Muon 1.4–2× đến từ
**baseline under-tuned**. Nên `sweep_lr` là first-class — race ở LR *tối ưu* của AdamW, không phải LR tuỳ ý.

**(4) Convergent-defaults là bằng chứng MẠNH NHẤT.** Bằng chứng mạnh nhất một kỹ thuật load-bearing KHÔNG
phải một run của ta, mà là **nhiều lab độc lập** adopt: MTP (V3 *và* GLM-4.5), Muon (nanochat + Moonlight +
Kimi-K2), aux-loss-free MoE (V3 + GLM + Qwen3), QK-norm (Qwen3/Gemma3/OLMo2). Nhiều lab hội tụ = *load-bearing*,
không phải *thời trang*. Ablation của ta *xác nhận cục bộ* tín hiệu convergent đó ở scale sub-1B.

**Neo code — hai harness thật của front này:**

*(a) F1 iso-FLOP race* (`src/scratch_llm/eval/optimizer_race.py`) — pure metrics unit-testable không cần train:
```python
def tokens_to_match(baseline, challenger):        # :59  token challenger cần để chạm loss CUỐI của baseline
    target = baseline[-1][1]
    for i,(x,y) in enumerate(challenger):
        if y <= target:
            return float(x_prev + (x-x_prev)*(y_prev-target)/(y_prev-y))  # :75  nội suy tuyến tính
    return None
def token_saving_fraction(b, c):                   # :79  1 − tokens_to_match / D_baseline
def nats_delta_at_budget(b, c):                    # :87  challenger − baseline @ SHARED budget
    if abs(bx - cx) > 1e-6*…:                       # :97  guard: budget khác nhau ⇒
        raise ValueError("… not an iso-FLOP race")  # :98  KHÔNG phải iso-FLOP ⇒ RAISE (so sánh void)
def run_race(...):                                 # :269  hai arm cùng N,D,seed,data — chỉ optimizer khác
    baseline   = run_arm(..., optimizer="adamw", …)         # :283
    challenger = run_arm(..., optimizer="muon_adamw", …)    # :292
    compute_flops = compute_from_params_tokens(n_params, total_tokens)  # :310  C=6ND đóng dấu
```
`run_arm` (:132) `seed_everything` TRƯỚC khi dựng model ⇒ hai arm share **cả init lẫn batch stream**; chỉ
optimizer khác. `sweep_lr` (:223) chọn argmin final-val-loss = tuned baseline. `_val_loss` (`train.py:173`)
chấm trên **fixed sequential windows, không RNG** (:193–195) ⇒ bật eval không perturb batch stream.

*(b) F3 de-confound spec-decode* (`src/scratch_llm/eval/spec_acceptance.py`) — gỡ confound §10.1:
```python
def measure_domain_acceptance(model, prompts, drafter, …):  # :133  đo BY DOMAIN trên ckpt THẬT
    committed, stats = speculative_generate(model, drafter, p.ids, …)  # :162
    greedy = generate(model, p.ids, SamplingParams(temperature=0.0, …))  # :165  ORACLE đi kèm
    # PromptResult.lossless: committed_ids == greedy_ids  (:59) ← losslessness không GIẢ ĐỊNH, đo thật
```

*(c) F1 Muon Newton–Schulz* (`src/scratch_llm/optim.py`):
```python
def _zeropower_via_newtonschulz5(g, steps=5):     # :137  quintic (3.4445,−4.7750,2.0315) trong bf16
    x = g.to(torch.bfloat16); x = x / (x.norm()+1e-7)   # :156,:160  Frobenius ≥ spectral ⇒ mọi σ≤1
    for _ in range(steps):
        aa = x @ x.T; bb = b*aa + c*(aa@aa)         # :162-163  X ← a·X + (b·A + c·A²)·X, A=XXᵀ
        x = a*x + bb @ x
    return x.to(g.dtype)                            # ≈ U Vᵀ (hướng của g, spectrum ~phẳng)
```
`build_optimizer` (:357) phân vùng: Muon cho 2-D block matrices, AdamW cho phần còn lại (mọi 1-D + tied 2-D).

**Hình ảnh — cấu trúc một ablation + NS spectrum:**
```
ISO-FLOP RACE (F1):                          NEWTON–SCHULZ nén spectrum (5 step):
  N,D,seed,data ═══ CỐ ĐỊNH ═══► chỉ optimizer khác     σ input (κ=721)        σ output
      │                                        │  ██                          ▁▁▂▅█▇▃▁
   baseline (AdamW, tuned LR) ──► val_curve    │  █                    5 NS    ▁▁▂▅███▇▃
   challenger (MuonAdamW)     ──► val_curve    │  ▁▁▁▁▁▂▂▃▅█  ──────►          band [0.08, 1.20]
      │                                        │ 0.04     31.75               median 0.86
   tokens_to_match / saving / nats_delta       └─ trải rộng ────────►  ─ nén về ~1 (không lift σ_min) ─
```
Data journey race: mỗi `val_curve` = `list[(tokens_seen, val_ce)]` float; `tokens_to_match` nội suy điểm
đầu tiên challenger ≤ `baseline[-1][1]`. NS journey: `g` (2-D Tensor, fp32) → bf16 → chuẩn hoá Frobenius →
5 vòng quintic → `.to(g.dtype)`.

**Số đo THẬT:**
```
─ F1 iso-FLOP race pure metrics (synthetic curves; baseline chạm loss 3.00 @ 1000 tok) ─   [FACT]
  tokens_to_match       = 600.0        (challenger chạm 3.00 sớm, nội suy)
  token_saving_fraction = 0.4000       (40% ít token hơn)
  nats_delta_at_budget  = −0.1500      (âm = challenger tốt hơn @ shared budget)
  non-isoFLOP guard     = RAISES "final token budgets differ (baseline 1000 vs challenger 999)…"

─ F1 Newton–Schulz spectrum (5 step, square 256×256 Gaussian) ─                             [FACT]
  INPUT  σ: min 0.044  max 31.75   cond 721.4
  OUTPUT σ: min 0.078  max 1.201   median 0.861   q90/q10 1.62
  ⇒ NÉN vào band ~[0.08,1.20] (không INFLATE); median<1 ⇒ falsifies "all∈[0.7,1.3], median≈1"

─ F1 hybrid param partition (nano, untied) ─                                                [FACT]
  Muon group : 28 params, ndims={2}          (block 2-D matrices ONLY)
  AdamW group: 11 params, has 1-D = True     (norms + embed/head)
  disjoint cover: overlap 0, Muon+Adam numel 3,410,176 == total  ⇒ exact partition

─ headline (rental-gated) ─
  F1: Muon ≥15% ít token @iso-FLOP · F3: spec-accept prose <10% code/JSON 40–60%    [PREDICTED]
  confound bằng chứng: spec-accept 54–72% trên RANDOM-weight (RESULTS.md:383)       [FACT ledger-cited]
```
**Câu chuyện pre-registration-bị-falsify-rồi-sửa (FOP-2/4):** F1 pre-register "σ ∈ [0.7,1.3], median≈1" đã
bị **falsify** — một ma trận **square Gaussian** có `σ_min ≈ 0.044` gần 0 mà 5-step quintic *không lift lên
1 được* (chỉ tới 0.078), median 0.86 < 1. Vì sao điều này **KHÔNG giết Muon?** Muon chỉ cần *hướng*
orthogonalized của momentum (`≈ U Vᵀ`), **không cần spectrum phẳng tuyệt đối** — cond number sập từ 721 →
~15 và bulk tight (q90/q10=1.62) là đủ để update có "hình dạng" đều. Over-claim bị sửa trong ledger
(`RESULTS.md:619`), test assert invariant *đã sửa*. Đó là claims-honesty in action.

**Frontier / cổng.** Reference cụ thể: **F1 Muon** — Moonlight 2502.16982 (RMS = `1/√max(A,B)`), Kimi-K2
2507.20534 (MuonClip @1T). **F2 MTP** — DeepSeek-V3 2412.19437 (seam `model.py`) + GLM-4.5 2508.06471.
**F5 MLA** — DeepSeek-V2 2405.04434 (weight-absorption). **F6** aux-loss-free — V3 (bias γ, `moe.py`).
**F7 GRPO** — R1 2501.12948. Gate = "thiết kế một ablation chứng minh Muon load-bearing — kể đủ: biến gì
cố định (N,D,seed,data), prediction falsifiable, KILL, tuned baseline, negative control, và vì sao 'nhiều
lab độc lập adopt' là bằng chứng mạnh hơn một run của bạn?". Trait = **research-as-MDP / spec-with-falsifiers**
(pre-register + KILL + iso-FLOP). Scarce bucket = experimental rigor (differentiator — labs screen on this).

---

## Bảng số đo THẬT (chạy lại được — DoD-là-profile)

| Concept | Đo | Kết quả | Ý nghĩa |
|---|---|---|---|
| 10.1 | nano speedrun end-to-end | stages `tok→pretrain→eval→sample`, **3.41M** param, **1682** tok | **LOOP CLOSES** [FACT] |
| 10.1 | nano `val_bpb` (CPU / GPU ledger) | **0.0113** (CPU 117 s) / 0.0206 (sm120 8.3 s) | metric chạy (in-sample) |
| 10.1 | confound (spec-accept random-weight) | **54–72%** (lẽ ra ~0) | vì sao close-the-loop cần model THẬT |
| 10.2a | bpb invariant (uniform, V=256) | nats/tok **5.545177 = ln256** exact; bpb **7.96** = 199/200·8 | cheapest oracle + boundary-token gap |
| 10.2b | MC length bias (uniform) | raw short −5.55 vs long −22.18 (**short thắng giả**); norm **diff 0** | length-normalize khử bias |
| 10.2c | CORE-style centering | raw mean 0.50 → **core_style 0.2778** | chance=0, perfect=1 |
| 10.3 | param count depth 4 / 20 | **3,410,176** / **564,317,440 (~564M)** | aspect-ratio scaling |
| 10.3 | `C=6ND` cross-check d20 | C=4e19,N=564M → **D=11.8B**, D/N=**20.9** | Chinchilla-optimal |
| 10.4 | iso-FLOP race pure metrics | ttm **600**, saving **40%**, nats_delta **−0.15**, non-iso guard **RAISES** | loss-per-FLOP comparable |
| 10.4 | Newton–Schulz spectrum | in κ=721 → out band **[0.08,1.20]** median **0.86** | NÉN về ~1 (falsify sửa) |
| 10.4 | F1 hybrid partition | Muon 28 (2-D) / AdamW 11 (1-D), overlap **0**, numel match | exact disjoint partition |

---

## Checklist recall COLD (che phần trên, tự trả lời — đây là cách re-learn)

1. **10.1** Vì sao 461 test xanh KHÔNG chứng minh có model biết nói? Bằng chứng tối thiểu của "close the loop" là gì?
2. **10.1** `RESULTS.md:383` (spec-accept 54–72% trên random) là *confound* hay *bug*? Ta đang đo tính chất của gì?
3. **10.1** Một knob `--depth`: dẫn `d_model, n_heads, head_dim` cho depth 4 và depth 20. Guard `max(1,·)` cứu gì?
4. **10.2a** Vì sao mẫu số bpb là **byte** không **token**? Một tokenizer nén tốt "thắng gian" thế nào nếu đổi sang token?
5. **10.2a** Uniform model V=256: nats/token = ? (exact). Vì sao bpb đo được **7.96** chứ không đúng **8.0**?
6. **10.2b** Raw summed log-prob thiên vị option dài hay ngắn — vì sao? `length_normalize` sửa bằng cách nào?
7. **10.2c** `core_style` centering công thức? base = ? Vì sao raw-mean-accuracy không so được giữa task 4-option và 2-option?
8. **10.3** Dẫn `C = 6ND` từ đếm FLOP fwd/bwd — vì sao hằng số **6** chứ không **2**? Inference thì hằng số là mấy?
9. **10.3** Cho C=4e19, N=564M: D = ? D/N = ? Khi nào ta *cố tình rời* Chinchilla-optimal (21 tok/param)?
10. **10.4** Bốn trụ của một ablation bảo vệ được? Vì sao **iso-FLOP** không-thể-bỏ (cái "thắng giả" nếu bỏ)?
11. **10.4** F1 pre-register "σ∈[0.7,1.3], median≈1" bị falsify (square Gaussian σ_min≈0.04, median 0.86). Vì sao **không giết Muon**?
12. **10.4** Vì sao "nhiều lab độc lập adopt" (MTP: V3+GLM, Muon: nanochat+Kimi) là bằng chứng MẠNH HƠN một run của bạn?

> Trả lời cold được cả 12 = **M10 thật sự OWNED** (interview-grade, đóng cả phía model). Vấp câu nào → mở
> đúng mục đó, hoặc blank-slate hàm tương ứng (`raise NotImplementedError` → test đỏ → tự dẫn → xanh) để
> re-own bằng tay.

---

*Cross-ref: `roadmap_model/M10_close_the_loop_and_ablations.md` (reference chung) ·
`docs/FRONTIER_2026_ABLATIONS.md` (§2 tier ladder · §3 F1–F11 · §7 report card) ·
`docs/FRONTIER_2026_TASKSPEC.md` (25 rung buildable) · `bench/RESULTS.md` (§Frontier ablations :604 · Phase 0
:712) · `PROGRESS.md` (ledger 89 Bài) · sibling derivations M2 (transformer forward) · M3 (optim: AdamW→Muon)
· M4 (training loop) · M6 (scaling laws — IsoFLOP fit). Đây là concept CUỐI của phía model: M1–M9 dựng bánh
răng, M10 lắp cỗ máy + lên dây cót + làm khoa học trên nó. Perf-roadmap S2 (MLA/speculative) + S6 (TP/PP/EP)
là phía serving của chính F2/F5; DELTA tối ưu decode kernel chúng sinh — giờ trên model THẬT.*
