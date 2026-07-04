# SÉRIE 10 — CLOSE THE LOOP + THE ABLATION STUDY (capstone của phía model)

> **Số dòng pin theo commit `4ad0ac5` (HEAD).** Đây là bản sinh đôi (twin) của perf-roadmap
> (`docs/learning/roadmap/`, phần serving+kernel) nhưng cho *chính con model*: sau khi M1–M9 dựng
> từng component (byte → BPE → Transformer → MoE → optim → train → post-training → frontier), série
> này **đóng vòng**: một script duy nhất biến corpus thành model biết nói, một report card làm oracle,
> và một *ablation study* pre-registered đo xem kỹ thuật 2026 nào thực sự load-bearing.
> Neo trung thực (FOP-4): phía này **đã build + đã toy-test**, các run thật bị **rental-gate** — số
> ĐO thật duy nhất là **Phase 0 val_bpb 0.0206 + sample mạch lạc** (`bench/RESULTS.md:656`) và
> bộ test; mọi target CORE / bảng tier là **PREDICTION/target**, đánh dấu rõ.

**Vì sao série này.** M1–M9 cho ta một *viện bảo tàng component* chất lượng cao — nhưng bảo tàng
không phải một model biết nói: component test xanh KHÔNG chứng minh cả pipeline *hợp* thành một thứ
sinh được văn bản mạch lạc. Tệ hơn, mọi số serving của phía perf đo trên **random-weight toy** (bằng
chứng: `bench/RESULTS.md:383`, acceptance-rate của speculative decode bị confound bởi entropy thấp
của model chưa train). Série này là *mũi tên đóng vòng*: (10.1) vì sao phải đóng vòng + cái speedrun
spine; (10.2) report card = oracle của vòng lặp; (10.3) bậc thang model-size + cross-check token
`C=6ND`; (10.4) ablation study như một **phương pháp dẫn xuất** — cách một thí nghiệm EV-ranked,
iso-FLOP, một-biến *chứng minh* một kỹ thuật có gánh sức nặng của nó hay không. Đọc theo thứ tự
**10.1 → 10.2 → 10.3 → 10.4**: vòng lặp trước, oracle của nó, thang đo nó chạy trên, rồi khoa học ta
làm với nó.

---

## Bài 10.1 — Close the loop: khe hở "bảo tàng vs model" + speedrun spine (`src/scratch_llm/speedrun.py` · `run_speedrun` :116 · `model_config_for_depth` :80)
> **Câu hỏi first-principles:** vì sao 461 test xanh + một serving stack tiên tiến VẪN không chứng minh
> ta có một model biết nói — và cái gì là bằng chứng tối thiểu rằng cả pipeline *hợp* thành một thứ hoạt động?
> **Neo (số đo — ĐO THẬT):** Phase 0 nano (depth 4, vocab 384, ctx 64, 150 step MuonAdamW, 8.3 s trên
> sm120) → **val_bpb 0.0206** + sample MẠCH LẠC nối đúng corpus (`bench/RESULTS.md:656,666`). "The loop
> closes" = `[FACT]`. Con số val_bpb này là *in-sample* (corpus lặp) — nó chứng minh **metric chạy được**,
> không phải model tốt (run thật cần held-out split + $100 d20, rental-gated).

**1. Feynman — bài toán bằng lời.** Hình dung một xưởng đồng hồ: mỗi bánh răng (tokenizer, attention,
RoPE, SwiGLU, MoE, AdamW, KV-cache) đã tiện xong + đo lẻ đạt dung sai. Nhưng *chưa ai lắp cả cỗ máy
lại rồi lên dây cót xem kim có chạy đúng giờ không*. Đó chính là khe hở: component test là "bánh răng
đạt chuẩn", còn "model biết nói" là "đồng hồ chạy đúng". Đánh đổi/quyết định cốt lõi: thay vì viết
thêm component thứ 34, ta *đóng vòng* — chấp nhận một model bé xíu, miễn nó đi hết **tokenizer →
pretrain → eval → sample** trong một lần chạy. nanochat (Karpathy) gọi đây là triết lý "one script,
whole loop": một `speedrun.sh`, một cái knob `--depth`. Cái giá của việc KHÔNG đóng vòng đã đo được:
`RESULTS.md:383` — speculative-decode acceptance 54–72% trên "random text" (lẽ ra ~0), bởi vì model
random-weight có entropy đầu ra thấp → greedy suy biến lặp → n-gram drafter trúng đậm. Đó là một *kết
quả bị nhiễm confound*: ta đang đo tính chất của model-chưa-train, tưởng là đo prompt. Chỉ một model
THẬT mới gỡ được (F3).

**2. Dẫn xuất từ đầu (derive).** *Vì sao spine này là tối thiểu-đủ?* Một model ngôn ngữ là hàm hợp
`decode ∘ eval ∘ train ∘ encode`; để chứng minh nó *hợp được*, ta phải chạy đúng chuỗi hàm đó end-to-end
một lần, không nhánh nào giả lập. speedrun chọn 4 stage: (1) train BPE trên corpus → (2) pretrain
Transformer bằng MuonAdamW → (3) report card (val_bpb) → (4) sample greedy/temp. Midtrain/SFT/RL bị
*cố ý bỏ* khỏi pre-flight (chúng tái dùng cùng `train` loop + `algos/`, F2/F7) — vì mục tiêu Phase 0
là "pipeline compose", không phải "model giỏi". *Vì sao knob `--depth` duy nhất?* nanochat aspect-ratio
scaling: `d_model = 64·depth`, `head_dim` ghim 128 (`model_config_for_depth` :83–84) ⇒ `n_heads =
d_model//128`. Một số điều khiển cả chiều rộng lẫn sâu, nên "nano" (depth 4) và "d20 headline"
(depth 20 ⇒ d_model 1280 / 10 heads / ~561M) là *cùng một code path, khác một số*. Bất biến bảo vệ
run trả tiền: nano pre-flight phải xanh (compose được) TRƯỚC khi tiêu $100 cho d20.

**3. Trace code.** `run_speedrun(cfg)` (:116): `_load_corpus` (:95, lặp corpus built-in cho đủ token)
→ `_train_tokenizer` (:104, byte-BPE `train_bpe`) → `stages.append("tokenizer")` → `tokens =
tokenizer.encode(text)` (:125, guard "corpus quá ngắn" :126) → dựng `TransformerLM(model_config_for_depth
(...))` (:132) → `train(TrainConfig(optimizer=cfg.optimizer, ...))` (:133, `optimizer` mặc định
`"muon_adamw"` :51, warmup = steps//20 :139) → `stages.append("pretrain")` → val slice đuôi (:154) →
`build_report_card(...)` (:157) → `stages.append("eval")` → prompt 24 char đầu → `generate(...)`
(:169) → `sample = tokenizer.decode(gen_ids)` → trả `SpeedrunResult` (:178, đếm `n_params` :180). CLI
`main()` (:203) map `--nano` → `_nano_config()` (:189, depth 4 / vocab 384 / 60 step) hoặc build từ
flag. `scripts/speedrun.sh` chỉ là vỏ CLI gọi `python -m scratch_llm.speedrun`. Test ghim: Phase-0 run
thật (`RESULTS.md:656`) — 3.41M params, 1682 tok, 8.3 s, val_bpb 0.0206, sample nối đúng "the quick
brown fox jumps over the lazy dog…".

**4. Cổng teach-back.** (a) Giải thích chính xác vì sao `RESULTS.md:383` là một *confound*, không phải
một bug — chạm được chữ "entropy đầu ra của model random" và "acceptance đo model không đo prompt".
(b) *Sửa-và-đoán:* nếu ta chạy speedrun với `--optimizer adamw` (baseline A1) thay vì `muon_adamw`,
Phase-0 "loop compose" còn đúng không, và con số nào (val_bpb hay chỉ "chạy được") mới là thứ pre-flight
thực sự bảo chứng? (Gợi: pre-flight bảo chứng *plumbing*, không bảo chứng chất lượng — cả hai optimizer
đều phải compose.)

**5. Frontier.** Reference: **nanochat của Karpathy** (`speedrun.sh` + `--depth`). Vendored
`modeling_nanochat.py:434` để **tie** lm_head↔embed (`_tied_weights_keys`); ta *khác* — `tie_embeddings=False`
mặc định (`model.py:46`, speedrun ghim untied `:91`) để routing Muon rõ ràng (F1: tensor tied 2-D phải
đi AdamW, gây mập mờ). Ta *giống* nanochat ở spine 4-stage + knob depth, *mạnh hơn* ở green-CI +
adapter-oracle. Câu interview: "Vì sao 'component test xanh' không đủ để tuyên bố có một model — bằng
chứng tối thiểu của 'close the loop' là gì?" (một lần chạy end-to-end thật, không nhánh giả lập, ra
sample mạch lạc + metric chạy).

---

## Bài 10.2 — Report card: oracle của vòng lặp (`src/scratch_llm/eval/report_card.py` · `build_report_card` :82 · `core_style_score` :39; `metrics.py` `bits_per_byte` :36; `multiple_choice.py` `option_logprob` :27; `generative.py` `evaluate_generative` :27)
> **Câu hỏi first-principles:** một model "tốt lên" nghĩa là gì bằng MỘT con số so sánh được — và vì sao
> **bits-per-byte** (không phải per-token loss) là metric đúng, còn accuracy multiple-choice phải chấm
> **bằng likelihood** chứ không phải bằng generation?
> **Neo (invariant — ĐO qua test):** model đều (logit all-equal trên vocab V) cho **đúng log2(V) bit/token**
> (`metrics.py` docstring :13–14, tested); nếu stream 1 byte/token thì = log2(V) bpb. Phase-0 val_bpb
> 0.0206 là số ĐO (in-sample). CORE-style aggregate & target ARC/MMLU/GSM8K là **PREDICTION/target**.

**1. Feynman — bài toán bằng lời.** Không có oracle thì "đóng vòng" vô nghĩa: ta cần một *thẻ điểm*
biến "model này khá hơn model kia" thành một con số. Report card gộp ba loại tín hiệu: (1) **intrinsic**
— val_bpb, model nén văn bản tốt cỡ nào; (2) **multiple-choice** — ARC/MMLU, chọn đáp án đúng; (3)
**generative** — GSM8K/HumanEval, sinh lời giải rồi chấm bằng verifier. Đánh đổi cốt lõi ở bpb: chọn
mẫu số là **byte** (không phải token) để một tokenizer "gian" (nhồi nhiều byte/token) KHÔNG được thưởng
free — metric so sánh *model*, không so sánh *tokenizer*. Ở multiple-choice, quyết định là chấm bằng
**log-prob của đáp án** (teacher-forced) chứ không bắt model *sinh* ra chữ cái — vì model bé chưa biết
tuân theo format "trả lời A/B/C/D", nhưng likelihood của chuỗi đáp án vẫn phân biệt được.

**2. Dẫn xuất từ đầu (derive).** *bpb:* từ định nghĩa cross-entropy, tổng negative log-likelihood
teacher-forced của stream = `Σ_t −ln p(x_t | x_<t)` nats; đổi sang bit (chia `ln 2`) rồi chuẩn hóa
theo số **byte UTF-8** mà stream giải mã ra: `bpb = (Σ_t −ln p(x_t|x_<t)) / ln2 / n_bytes`
(`metrics.py:5–8`). Kiểm chứng bằng giới hạn: model đều → `p = 1/V` mỗi token → `−ln p = ln V` nats/token
→ `log2(V)` bit/token; nếu 1 byte/token thì bpb = log2(V) — chính là invariant test. *Length-normalize
cho MC:* raw summed log-prob thiên vị đáp án ngắn (ít token → tổng ít âm hơn); chia cho số token đáp án
(`option_logprob` :49) khử bias độ dài — đúng convention ARC/MMLU. *CORE-style:* accuracy thô không so
được giữa task có số lựa chọn khác nhau; centering `(acc − base)/(1 − base)` (`core_style_score` :42–46)
đặt chance = 0, perfect = 1, với `base = 1/n_options` (`MCTask.random_baseline` :28). Trung thực trong
đặt tên: đây là *công thức aggregate* của DCLM-CORE — chỉ là điểm CORE *chính thức* khi cho đúng bộ task
CORE chính thức; trên task tùy ý nó là "CORE-*style*" (docstring :1–8).

**3. Trace code.** `build_report_card(model, tokenizer, val_tokens, val_num_bytes, mc_tasks, gen_tasks)`
(:82): nếu có `val_tokens`+`val_num_bytes` → `bits_per_byte(...)` (:102). `bits_per_byte` (:36) chấm
theo **cửa sổ không chồng** `ctx` (nanochat/GPT-2 chunked bpb :48–52): mỗi transition `x_<t→x_t` chấm
đúng một lần, `total_nll += −logp.gather(tgt)` (:78), trả `BpbResult(bpb, nats/token, ...)`. Vòng MC
(:108–111): mỗi task → `evaluate_multiple_choice` (:69) → `predict_choice` (:52, argmax
`option_logprob`) → nạp `mc[name]` + `baselines[name]`. Vòng generative (:114–118):
`evaluate_generative` (:27) sample completion bằng `generate` rồi `grade_fn(completion, reference)`
(mặc định greedy temp 0 :44 để điểm tất định). Cuối: `ReportCard(val_bpb, nats_per_token, mc, generative,
core_style)` (:120), `core_style` chỉ tính khi có MC (:125). `to_markdown` (:67) in bảng. Test ghim:
report-card harness 8 test (`RESULTS.md:672`); invariant log2(V) trong `metrics.py`.

**4. Cổng teach-back.** (a) Vì sao mẫu số của bpb là *byte* chứ không *token* — nêu đủ hậu quả nếu đổi
sang per-token (một tokenizer nén tốt sẽ "thắng gian" thế nào)? (b) *Sửa-và-đoán:* nếu tắt
`length_normalize` (`option_logprob` :49) trên MMLU với các đáp án dài ngắn khác nhau — accuracy lệch
theo hướng nào, và task nào (đáp án dài đều / đáp án ngắn đều / lẫn lộn) bị hại nhất? (Gợi: raw sum thiên
vị đáp án ngắn ⇒ hại nhất khi độ dài đáp án tương quan với đúng/sai.)

**5. Frontier.** Reference: **nanochat report card** — val_bpb là headline pretrain number (tokenizer-
invariant), + DCLM-CORE (2406.11794) làm aggregate. Ta *giống* triết lý (bpb headline + CORE-style),
*trung thực hơn* ở nhãn "CORE-style ≠ CORE chính thức" (`report_card.py:1–8`). Đối chiếu vendored:
`modeling_nanochat.py:499` cắt `hidden_states[:, slice_indices]` trước `lm_head` — cùng seam
teacher-forced ta dùng để chấm log-prob. Câu interview: "Chấm multiple-choice bằng likelihood vs bằng
generation khác nhau ra sao, và vì sao model nhỏ CHỈ chấm được bằng likelihood?" (model chưa biết tuân
format sinh; log-prob của đáp án vẫn tách được tín hiệu).

---

## Bài 10.3 — Model-size tiers + cross-check token `C=6ND` (`docs/FRONTIER_2026_ABLATIONS.md` §2 · `speedrun.py` `model_config_for_depth` :80 · `_nano_config` :189)
> **Câu hỏi first-principles:** cho một ngân sách compute cố định, chọn model bao to (N) và train bao nhiêu
> token (D) thế nào — và vì sao `C = 6·N·D` là cái cân bắt buộc trước khi trả tiền?
> **Neo (PREDICTION/target — CHƯA đo):** d20 headline = depth 20 / d_model 1280 / 10 heads / **~561M**,
> **≈11–12B token** (`C≈4e19 ⇒ D≈11.9B`, ~21 tok/param, Chinchilla-optimal), 8×H100 ~2–4h, **~$48–100**,
> target **CORE ≈ 0.256–0.269 (GPT-2 grade)**. Đây là **target rental-gated**, KHÔNG phải số đo. Số ĐO duy
> nhất ở tier này: nano pre-flight (depth 4, 3.41M, 8.3 s, val_bpb 0.0206) — `bench/RESULTS.md:656`.

**1. Feynman — bài toán bằng lời.** Ngân sách compute là tiền thật (GPU-hour). Câu hỏi kinh tế: cùng
một số FLOP, nên đổ vào *model to hơn* (nhiều tham số N) hay *nhiều dữ liệu hơn* (nhiều token D)? Đây
là câu hỏi Chinchilla. Đánh đổi: N to → model "thông minh" hơn per token nhưng đắt cả train lẫn serve;
D to → dạy kỹ hơn nhưng tốn thời gian. `C = 6ND` là "hối suất" biến hai trục thành một: mỗi tham số ×
mỗi token ≈ 6 FLOP (2 cho forward matmul, 4 cho backward). Bậc thang tier (§2) là một *thang rủi ro-tiền*:
nano ($0, giây, chứng minh plumbing) → d20 (~$100, model biết nói + report card công khai) → ablation
science (standing Blackwell 24GB, $0, nơi khoa học sống) → stretch d26/d32 (chỉ khi một ablation biện
minh cho scale).

**2. Dẫn xuất từ đầu (derive).** *Vì sao 6ND?* Một forward qua model dense tốn ~`2N` FLOP/token (mỗi
weight tham gia 1 multiply + 1 add). Backward ~2× forward (grad theo input + grad theo weight) ⇒ tổng
train ~`6N` FLOP/token; nhân D token: `C = 6·N·D`. *Cross-check token của d20:* cho `C ≈ 4e19` (ngân sách
8×H100 vài giờ) và `N ≈ 561M`, giải `D = C/(6N) = 4e19/(6·5.61e8) ≈ 1.19e10 ≈ 11.9B` token ⇒
`D/N ≈ 21 tok/param` — đúng vùng Chinchilla-optimal (~20). *Vì sao aspect-ratio scaling?* nanochat ghim
`head_dim = 128` và cho `d_model = 64·depth` (`:83`), nên `n_heads = depth/2` (:84). Điều này giữ tỉ lệ
rộng/sâu ổn định khi scale — một số `depth` quyết cả N (qua `d_model` và `n_layers=depth`). Ở nano,
`d_model = 256 < 128·2` nên `n_heads = max(1, 256//128) = 2` (guard `max(1,…)` :84 chống head=0 khi
model tí hon). Kết nối perf-roadmap: bài toán "N vs D cho compute-optimal" chính là IsoFLOP/Chinchilla
mà `scaling/isoflop.py` fit (F1 dùng nó để *giữ C cố định* khi so Muon vs AdamW).

**3. Trace code.** `model_config_for_depth(depth, vocab, ctx)` (:80): `d_model = 64*depth` (:83),
`n_heads = max(1, d_model//128)` (:84), `n_layers = depth` (:88), `tie_embeddings=False` (:91). Bảng
tier sống ở `docs/FRONTIER_2026_ABLATIONS.md` §2 (:59–64): nano depth~4–6 → headline d20 561M/~11.9B
tok/$48–100/CORE≈0.256 → ablation 30–300M standing → stretch d26/d32. `_nano_config()` (:189) = tier
pre-flight: depth 4, vocab 384 (256 byte + 128 merge :195), ctx 64, 60 step. `n_params` đếm thật ở
`run_speedrun` (:180) `sum(p.numel())`. Test ghim: Phase-0 (`RESULTS.md:656`) xác nhận nano tier chạy
thật 8.3 s; **d20 tier PENDING rental** (§5 Phase 1). Trung thực: mọi ô "CORE ≈…" trong bảng §2 là
**target pre-registered**, chỉ thành `[FACT]` khi điền cột measured.

**4. Cổng teach-back.** (a) Dẫn `C = 6ND` từ đếm FLOP forward/backward — vì sao hằng số là 6 chứ không 2?
(b) *Sửa-và-đoán:* nếu một model sẽ **serve nghìn tỉ token** (inference chi phối lifetime), Chinchilla-
optimal (21 tok/param) còn là lựa chọn đúng không, hay ta cố tình over-train một model NHỎ hơn (kiểu
Llama-3 ~1875 tok/param)? N và D dịch về đâu? (Gợi: inference-aware allocation — `docs/FRONTIER_PRACTICE_2026.md`
pillar scaling, Sardana 2401.00448; N↓, D↑ khi serve chi phối.)

**5. Frontier.** Reference: **nanochat d20** (~$100, ~561M, CORE≈GPT-2) là tier headline; d32 (~1.9B,
~$800) là stretch; d26 chỉ là "passing suggestion" chưa benchmark (verifier UNCERTAIN, §8). Đối chiếu
2026 landscape (§6): DeepSeek-V3 671B/37B, GLM-5.2 ~744B/40B, Kimi-K2 1.04T/32B — tất cả là MoE
deep-narrow, nên `N` (total) ≫ `N_active`; `C=6ND` dùng `N_active` cho FLOP nhưng KV/serve theo tổng.
Câu interview: "Cho ngân sách C, chọn N và D thế nào — và khi nào ta *cố tình rời* Chinchilla-optimal?"
(training-compute-optimal vs amortized-serving-cost; over-train khi serve chi phối lifetime budget).

---

## Bài 10.4 — Ablation study như một phương pháp DẪN XUẤT: F1–F9 (`docs/FRONTIER_2026_ABLATIONS.md` §3 · `bench/RESULTS.md` §Frontier ablations :604)
> **Câu hỏi first-principles:** làm sao *chứng minh* (không phải đồn) rằng một kỹ thuật 2026 (Muon, MTP,
> aux-loss-free, MLA…) thực sự load-bearing — cấu trúc thí nghiệm nào biến "nghe nói nó tốt" thành một
> kết luận nhân quả bảo vệ được trước hội đồng?
> **Neo (số đo ĐO + PREDICTION, phân loại rõ):** ĐÃ ĐO ở unit level — F1 NS singular values *nén* vào
> band ~[0.68,1.14] (over-claim "all∈[0.7,1.3]" bị **falsify → sửa**, `RESULTS.md:619`); F1 hybrid
> overfit-one-batch <0.05 (:620); F1 param-partition đúng (:621); F1/F4 train wiring learns loss→8.3e-4,
> **bf16+compile NaN trên sm120/torch-2.12 inductor** (:647). PENDING (rental): F1 iso-FLOP "Muon ≥15%
> ít token hơn" — **PREDICTION**, chưa đo (:622).

**1. Feynman — bài toán bằng lời.** Có hàng chục kỹ thuật 2026 lấp lánh; nếu bê hết vào một run to, ta
KHÔNG bao giờ biết cái nào thực sự giúp — biến chồng biến, tiền đốt, kết luận zero. Ablation study là
kỷ luật gỡ chuyện đó: đổi **đúng một biến**, giữ **compute cố định (iso-FLOP)**, **pre-register** con số
falsifiable + ngưỡng KILL *trước khi chạy*, rồi đo. Đánh đổi cốt lõi: chậm hơn (một run mỗi biến) nhưng
đổi lấy *nhân quả* — chính kỷ luật này (chứ không phải kết quả) là kỹ năng frontier-lab tuyển. Ẩn dụ:
một bác sĩ không kê 5 thuốc cùng lúc rồi đoán thuốc nào chữa; họ thử nghiệm đối chứng, một biến, có
placebo (ở đây: negative control như neural-RM reward-hack ở F7, hay no-balancing collapse ở F6).

**2. Dẫn xuất từ đầu (derive).** *Vì sao iso-FLOP là điều kiện bắt buộc?* Nếu Muon "thắng" nhưng dùng
nhiều FLOP hơn, cái thắng có thể chỉ là "train lâu hơn" — vô nghĩa. Giữ `C=6ND` cố định (qua
`scaling/isoflop.py`) biến câu hỏi thành *loss-per-FLOP*, cái duy nhất so được. *Vì sao pre-register +
KILL?* Không pre-register thì ta *post-hoc* chọn ngưỡng để "thắng" (p-hacking); một prediction falsifiable
(F1: "Muon đạt loss của AdamW với ≥15% ít token, hoặc ≥0.02 nats thấp hơn @iso-FLOP") + KILL ("token
saving <5% hoặc diverge ở LR tái dùng của AdamW") ràng buộc ta trung thực TRƯỚC khi thấy số. *Xếp hạng
EV:* leverage ÷ effort, #1–#4 là 80/20 (F1 Muon → F2 MTP → F3 de-confound → F4 bf16/compile). *Lập luận
convergent-defaults (§3, mạnh nhất):* bằng chứng mạnh nhất một kỹ thuật load-bearing KHÔNG phải một run
của ta, mà là nó được **nhiều lab độc lập** adopt — MTP (V3 **và** GLM-4.5), Muon (nanochat + Moonlight +
Kimi-K2), aux-loss-free MoE (V3 + GLM + Qwen3), QK-norm (Qwen3/Gemma3/OLMo2). Nhiều lab hội tụ cùng một
lựa chọn = tín hiệu nó là *load-bearing*, không phải *thời trang*. Ablation của ta *xác nhận cục bộ* cái
tín hiệu convergent đó ở scale sub-1B.

**3. Trace code.** Ledger `bench/RESULTS.md:604` (§Frontier ablations): mỗi rung có bảng
`predicted | KILL if | measured`, `[FACT]` chỉ khi cột measured điền. F1 (:610–634): NS orthogonality
(:619 — over-claim falsify: NS *nén* band, không *lạm phát*, `σ_max<1.35`, immaterial vì Muon chỉ cần
*hướng* update), hybrid wiring (:620 PASS), param-partition (:621 PASS — tensor tied 2-D + mọi 1-D đi
AdamW; toán RMS-match `0.2·√max(A,B)` sửa từ `1/max`→`1/√max`, §8), iso-FLOP (:622 PENDING). F1/F4 train
wiring (:636–654): fp32/bf16/compile mỗi cái học loss→~8e-4, **bf16+compile NaN** trên inductor box này
(:647 `[FACT]` box-specific, reproduces cả với plain AdamW ⇒ codegen bug, không phải logic ta) → NaN
guard loud ở `train.py`. Các rung F2–F9 (§3 :79–86 + rung cards :92–159) đều pre-registered, mở seam
code: F2 MTP dùng `x`-before-`lm_head` (`model.py:938`/`:941`), F5 MLA-real wire `mla.py`→`ModelConfig`,
F6 `moe.py` balancing, F7 `algos/grpo.py`. Cross-link perf-roadmap: F5 MLA weight-absorption → S2 Bài 2.8;
MTP/MLA decode kernel feed DELTA.

**4. Cổng teach-back.** (a) Vì sao **iso-FLOP** là điều kiện KHÔNG-thể-bỏ để một ablation optimizer có ý
nghĩa — nêu cái "thắng giả" nếu bỏ nó? (b) *Sửa-và-đoán:* F1 pre-register "σ ∈ [0.7,1.3], median≈1" đã bị
falsify (NS 5 bước nén vào [0.68,1.14], median≈0.77, ma trận Gaussian vuông có σ_min≈0.08). Vì sao điều
này **không giết Muon** — Muon dùng gì từ update mà việc "σ không về đúng 1" trở nên vô hại? (Gợi: Muon
chỉ cần *hướng* orthogonalized của momentum, không cần spectrum phẳng tuyệt đối.)

**5. Frontier.** Reference cụ thể + đối chiếu: **F1 Muon** — Moonlight `2502.16982` (Lemma 1: RMS =
`1/√max(A,B)`, ta sửa đúng theo đó), Kimi-K2 `2507.20534` (MuonClip ở 1T scale). **F2 MTP** — DeepSeek-V3
`2412.19437` (MTP module sequential D=1) + GLM-4.5 `2508.06471`; V3 dùng MTP head học *và* làm draft
speculative — ta giống (seam `model.py:938`), khác vì ta D=1 sequential (không phải parallel Gloeckle
`2404.19737`). **F5 MLA** — DeepSeek-V2 `2405.04434` (weight-absorption). **F6** aux-loss-free — V3
`2412.19437` (bias γ update, `moe.py`). **F7 GRPO** — R1 `2501.12948`. Vendored đối chiếu: nanochat tie
weights (`modeling_nanochat.py:434`) trong khi ta untie (F1 cần routing rõ). Câu interview: "Thiết kế
một ablation chứng minh Muon load-bearing — kể đủ: biến gì cố định, prediction falsifiable, KILL, negative
control, và vì sao 'nhiều lab độc lập adopt' là bằng chứng mạnh hơn một run của bạn?"

---

*Đóng série (và đóng cả phía model).* M1–M9 dựng từng bánh răng; M10 lắp cỗ máy + lên dây cót: **loop
closes** (Phase 0 `[FACT]`: val_bpb 0.0206 + sample mạch lạc) → report card làm oracle → thang tier
`C=6ND` định cỡ run trả tiền → ablation study F1–F9 pre-registered là *khoa học* ta làm trên model thật.
Ranh giới trung thực (FOP-4): mọi thứ ở unit/nano level ĐÃ ĐO + xanh; mọi headline (CORE d20, Muon iso-FLOP
≥15%, MTP ≥1.5× accept) là **PREDICTION rental-gated** — 3 ngày rental là cái sẽ biến chúng thành `[FACT]`.
Cross-link: perf-roadmap S2 (MLA/speculative) + S6 (TP/PP/EP) là *phía serving* của chính những kỹ thuật
này; F2/F5 sinh ra đúng decode kernel mà DELTA tối ưu — giờ trên model THẬT, không phải random-weight toy.
