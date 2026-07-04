# SÉRIE 6 — SCALING LAWS, dẫn xuất (CS336 A3: IsoFLOP · Chinchilla · budget planner)

> **Số dòng pin theo commit `4ad0ac5`** (HEAD). Đây là ROADMAP dẫn-xuất-trước: mỗi Bài DERIVE component
> từ first principles rồi trace xuống `file·hàm·dòng`, KHÔNG phải bài giảng đầy đủ. Mỗi Bài ~một màn hình.
> Code: `src/scratch_llm/scaling/{isoflop,planner}.py`. Số đo: `bench/RESULTS.md` §W5. Test:
> `tests/test_scaling.py` (8), `tests/test_planner.py` (18).

**Vì sao série này.** Các série trước dựng *một* model rồi *một* vòng train cho nó — nhưng chưa trả lời
câu hỏi đắt nhất của cả pipeline: **với X FLOP trong tay, model nên TO bao nhiêu và ăn bao nhiêu token?**
Đoán sai một bậc là đốt cả cụm GPU vào một model quá to huấn luyện quá ít (đúng lỗi GPT-3). Scaling laws là
câu trả lời *định lượng, ngoại suy được*: đo vài chục run NHỎ, fit một power-law, rồi tiên đoán cấu hình
tối ưu cho run LỚN mà không bao giờ chạy nó. Série này rất khác các série kernel — không có GPU nào chạy
ở đây; toàn bộ là một *phương pháp thống kê* + một *bộ máy kế toán ngân sách*. Nó tựa lên cái đếm FLOP
`C=6ND` (cross-link perf roadmap/S6 Bài 6.4 `mfu.py`) và cho ra con số đầu vào của MỌI run pretraining
thật (bao gồm headline nanochat d20 ở front frontier). Thứ tự đọc: **6.1 (vì sao power-law tồn tại →
ngoại suy được) → 6.2 (phương pháp IsoFLOP: fit (N*,D*) từ họ run) → 6.3 (Chinchilla: quy tắc ~20
tok/param, vì sao GPT-3 under-trained) → 6.4 (planner: biến FLOP-budget thành cấu hình + kỷ luật ngân sách).**

---

## Bài 6.1 — Vì sao scaling law TỒN TẠI: loss ~ power-law của compute, và vì sao nó ngoại suy được (`scaling/isoflop.py` · module docstring :1–19 · `PowerLaw.predict` :38)
> **Câu hỏi first-principles:** vì sao final loss của một LM lại là một *đường thẳng trong log-log* theo
> compute — và vì sao MỘT sự thật thực nghiệm đó cho phép ta tiên đoán model tỉ-tham-số từ model triệu-tham-số?
> **Neo (invariant / số đo):** power-law `y = coeff·x^exponent` fit trong log-space phục hồi ĐÚNG luật gieo
> tới độ chính xác fp — test `test_fit_powerlaw_recovers_exact_law` (:62): gieo exponent 0.42/coeff 3.7,
> fit ra 0.42 (abs 1e-10). [FACT, MEASURED — bất biến test.]

**1. Feynman — bài toán bằng lời.** Sự thật thực nghiệm gây sốc (Kaplan 2020, Hoffmann 2022): nếu bạn
plot *final loss* của các LM đã train tối ưu theo *compute C* trên trục log-log, bạn được gần như một
**đường thẳng** trải qua **bảy bậc độ lớn** của C. Đường thẳng trong log-log = **power-law** trong không
gian thường: `L(C) ≈ E + a·C^(-α)`. Không có lý thuyết nào bắt buộc điều này — nó là *quy luật* đo được,
bền vững đến mức thành một công cụ kỹ thuật. Analogy: như định luật khí lý tưởng, ta không cần cơ học
thống kê của từng phân tử để dùng PV=nRT dự đoán; ta không cần hiểu vì sao loss giảm mượt để *dùng* độ dốc
đó ngoại suy. Đánh đổi cốt lõi của cả série: **đo rẻ ở scale nhỏ, đặt cược đắt ở scale lớn** — power-law
là chiếc cầu, và nó chỉ đúng *trong khi regime còn giữ* (cùng data, cùng kiến trúc, cùng optimizer).

**2. Dẫn xuất từ đầu (derive).** Vì sao fit phải làm trong **log-space, không bao giờ raw-space**? Giả sử
`y = k·x^p`. Lấy log hai vế: `log y = log k + p·log x` — một đường thẳng, độ dốc `p`, chặn `log k`. Đây là
lý do fit tuyến tính `np.polyfit(log x, log y, 1)` (`fit_powerlaw` :69) phục hồi exponent chính xác. Nhưng
tại sao KHÔNG fit `y = k·x^p` trực tiếp bằng least-squares raw? Vì raw-space least-squares là
**heteroscedastic**: residual của điểm `x` lớn nhất (giá trị `y` cỡ tỉ) áp đảo tổng bình phương, kéo
exponent lệch theo đúng một điểm. Log-space cho MỌI điểm trọng số bằng nhau (sai số *tương đối* đều). Test
`test_fit_is_log_space_not_raw_space` (:72) chứng minh bằng phản-ví-dụ gieo sẵn: đặt một outlier nhân-giảm
ở `x` lớn nhất → log-fit lệch <0.03, raw-fit lệch >0.15 (rơi về ~0.31), hai estimator *khác nhau chứng
minh được*. `PowerLaw` (:31) đóng gói đúng cặp (exponent, coeff), `predict` (:38) ngoại suy bằng mũ hoá.

**3. Trace code.** `PowerLaw` frozen dataclass (:31–39): hai field `exponent`/`coeff`, `predict(x) =
coeff·x^exponent` (:39). `fit_powerlaw(xs, ys)` (:57): guard ≥2 điểm + strictly-positive (log cần >0,
:67–68) → `np.polyfit(np.log(x), np.log(y), 1)` trả (slope, intercept) → `PowerLaw(exponent=slope,
coeff=exp(intercept))` (:70). Test-oracle pin nó: `test_fit_powerlaw_recovers_exact_law` (:62) gieo luật
exact → đòi fit khớp 1e-10; `test_fit_powerlaw_rejects_bad_input` (:94) đòi ValueError trên 1-điểm và trên
y âm. Đây là *nguyên thủy đo lường* mà Bài 6.2 xây toàn bộ phương pháp lên trên.

**4. Cổng teach-back.** (a) Giải thích vì sao "loss là đường thẳng trong log-log" cho phép ngoại suy tỉ-param
từ triệu-param — chạm được chữ "power-law = tự-đồng-dạng qua các scale" và điều kiện "regime phải giữ". (b)
*Sửa-và-đoán:* nếu ta fit trong raw-space thay vì log, với một họ điểm trải 6 bậc độ lớn có một outlier ở
đầu to — exponent lệch về phía nào, và vì sao log-space *miễn nhiễm*? (Gợi: raw đuổi theo điểm thống trị
residual; log cho sai-số-tương-đối đều.)

**5. Frontier.** Đây là Kaplan et al. (2001.08361) và Chinchilla (Hoffmann, 2203.15556) — hai power-law
loss-vs-compute nổi tiếng nhất; Chinchilla *sửa* Kaplan (Bài 6.3). Frontier 2026 mở rộng ý này thành
**hyperparameter transfer / muP** (`docs/FRONTIER_PRACTICE_2026.md:170–176`, Yang 2203.03466): không chỉ
loss mà cả *learning rate/init tối ưu* cũng transfer qua width — tune một lần trên proxy nhỏ, zero-shot
sang model lớn, với "coordinate-check" (activation RMS width-invariant) làm cổng đúng-sai. Câu interview:
"Vì sao fit scaling law làm trong log-space? Nếu tôi cho bạn 20 run nhỏ, bạn tiên đoán loss của run to
1000× thế nào — và điều gì có thể *phá* ngoại suy đó?" (regime shift: đổi data/arch/optimizer → power-law
mới).

---

## Bài 6.2 — Phương pháp IsoFLOP: fit (N*,D*) tối ưu từ họ run ở C=6ND cố định (`scaling/isoflop.py` · `isoflop_min` :42 · `check_exponent_sum` :103)
> **Câu hỏi first-principles:** cho một họ run trải nhiều kích thước N ở CÙNG budget compute C, làm sao
> rút ra cấu hình compute-optimal (N*,D*) — và làm sao *bắt lỗi* trước khi ngoại suy?
> **Neo (invariant / số đo):** 9 budget → đúng 9 cặp (C, N*); exponent phục hồi **a=0.46868 · b=0.53132 ·
> a+b=1.0000** trên course-data (regression-locked, `test_course_json_regression_locked` :140, abs 5e-4).
> [FACT, MEASURED — fit của course JSON.]

**1. Feynman — bài toán bằng lời.** IsoFLOP = "lát cắt đẳng-compute". Cố định budget C (ví dụ 6e18 FLOP);
train một *họ* model kích thước N khác nhau, mỗi cái đúng số token `D = C/(6N)` để tiêu HẾT C. Plot final
loss theo N: đường cong hình **parabola** (chén). N quá nhỏ → model không đủ dung lượng hấp thụ C (under-
capacity); N quá lớn → phải cắt bớt D nên train quá ít bước (under-training). **Đáy chén = (C, N*)** — một
điểm compute-optimal cho budget đó. Lặp cho nhiều C → một họ điểm `(C, N*)`, và ĐÓ là dữ liệu vào cho
power-law Bài 6.1. Analogy: mỗi budget là một "túi tiền cố định", ta hỏi *chia bao nhiêu cho size vs bao
nhiêu cho data* để loss thấp nhất — parabola trả lời, đáy là tỉ lệ vàng của túi đó.

**2. Dẫn xuất từ đầu (derive).** Ba nước đi (module docstring :4–9). **(1) Min-picking:** repo dùng
biến thể CS336 — *argmin loss per budget* (`isoflop_min` :42), không fit parabola bậc-2 kiểu Hoffmann; đơn
giản hơn, đủ khi lưới N dày. **(2) Fit power-law trong log-log** (Bài 6.1): `N* ∝ C^a`, và độc lập `D* ∝
C^b`. **(3) Cầu compute-identity** `C ≈ 6·N·D` (2 FLOP/param/token forward + 4 backward): `D_opt = C/(6·N*)`
được *dẫn xuất*, không đo riêng (`tokens_from_compute_params` :78). Bất biến giả-chứng (falsifiable):
`C = 6ND` ÉP `a + b ≈ 1`. Vì sao? `log C = log 6 + log N + log D`; nếu `N ∝ C^a` và `D ∝ C^b` thì thay vào
`C = 6·C^a·C^b` ⇒ `C^1 = C^(a+b)` ⇒ **a+b=1 về mặt cấu trúc**. `check_exponent_sum` (:103) bắn ValueError
nếu `|a+b-1| > tol` — cổng này bắt lỗi min-picking hoặc lỗi cầu *trước* khi ngoại suy. Trên course data
a≈b≈0.5 (Chinchilla: scale N và D gần bằng nhau).

**3. Trace code.** `isoflop_min(runs)` (:42): group theo `compute_budget`, giữ run `final_loss` nhỏ nhất
mỗi budget (:52), trả `[(C, N*)]` sort theo C tăng (:54). Pipeline đầy đủ (`test_course_json_regression_
locked` :140): `isoflop_min` → tách `budgets`/`n_opts` → `fit_powerlaw(budgets, n_opts)` cho `a` →
`d_opts = [tokens_from_compute_params(c,n) for ...]` → `fit_powerlaw(budgets, d_opts)` cho `b` →
`check_exponent_sum(a,b)`. Regression-lock đo một lần rồi đóng băng: a=0.46868, b=0.53132, a+b=1.0 exact
(1e-9 — vì D dẫn từ CÙNG các hàng (C,N) nên identity là cấu trúc). Test răng-cưa `test_exponent_sum_gate_
fires_on_corrupted_data` (:123): ghép D với N* *lệch hàng* (đảo thứ tự) → mỗi fit trông hợp lý riêng nhưng
`a+b` la làng → gate bắn. `test_isoflop_min_nine_budgets` (:49) pin: 9 budget, budget nhỏ nhất 6e18 có
argmin N=762093419.

**4. Cổng teach-back.** (a) Dựng lại vì sao `a+b=1` là hệ quả TẤT YẾU của `C=6ND` (làm phép đại số log), và
vì sao cổng đó bắt được lỗi mà mỗi fit riêng lẻ *giấu*. (b) *Sửa-và-đoán:* nếu lưới N của bạn ở một budget
chỉ có các N *lớn* (thiếu nhánh trái parabola) — `isoflop_min` trả điểm nào, N* bị lệch về hướng nào, và
`a` đo được cao hay thấp hơn thật? (Gợi: min-picking chọn N nhỏ nhất còn lại → N* thiên lệch lên → a lệch.)

**5. Frontier.** IsoFLOP là "Approach 2" trong Chinchilla (2203.15556) — họ chạy CHÍNH phương pháp lát-cắt
này. Nhưng cross-reference đắt nhất: **method IsoFLOP CHÍNH LÀ tư duy của EV-ranked ablation study**
(`docs/FRONTIER_2026_ABLATIONS.md` §3 "leverage ÷ effort"): mỗi rung ablation F1–F9 chạy ở **iso-FLOP**
(giữ `C=6ND` cố định qua `scaling/isoflop.py`) để so *một biến* công bằng — F1 Muon "đạt loss của AdamW với
≥15% ít token hơn ở iso-FLOP" là *đúng một điểm parabola so hai optimizer*. Scaling-law-thinking = research-
as-MDP: đo rẻ nhiều lát, đặt cược một lần. Câu interview: "Derive và bảo vệ một scaling law: vì sao fit
trong log-space, vì sao hai exponent phải cộng thành một, và ngoại suy của bạn *thành thật* với tới đâu?"
(nêu factor — fit chỉ đúng khi regime đúng).

---

## Bài 6.3 — Chinchilla: quy tắc ~20 token/param, và vì sao GPT-3 under-trained (`scaling/isoflop.py` · `compute_from_params_tokens` :73 · `nonembed_params` :83 · `propose_shape` :88)
> **Câu hỏi first-principles:** cho a≈b≈0.5, tỉ lệ token/param tối ưu ra bao nhiêu — và tại sao một model
> 175B huấn luyện trên 300B token là một sai lầm *đo được* chứ không phải ý kiến?
> **Neo (số đo / prediction):** ngoại suy course-fit ra **N*≈7.01e10 (~70B) @1e23 FLOP, D*≈2.38e11 (~238B
> token)** → **D*/N* ≈ 3.4** ở scale này; shape đề xuất n_layer=71/d_model=9068 (`propose_shape`, aspect 128).
> [PREDICTION/INFERENCE — ngoại suy ×333 quá budget lớn nhất 3e21; a/b là [FACT] fit course data.]

**1. Feynman — bài toán bằng lời.** Kaplan 2020 kết luận: khi có thêm compute, đổ *phần lớn* vào **size**
(model to hơn nhiều, data chỉ nhích). GPT-3 làm đúng lời khuyên đó: 175B param, chỉ 300B token (~1.7
token/param). Chinchilla 2022 chạy lại cẩn thận hơn (IsoFLOP, Bài 6.2) và lật ngược: N và D nên scale
**gần bằng nhau** (a≈b≈0.5), nghĩa là tỉ lệ token/param tối ưu **~xấp xỉ hằng số ~20** ở scale họ đo. Hệ
quả bàng hoàng: GPT-3 **under-trained** nặng — cùng compute đó, một model 4× NHỎ hơn train trên 4× NHIỀU
token hơn (Chinchilla 70B, 1.4T token) *thắng nó*. Analogy: bạn có ngân sách nấu ăn cố định; Kaplan bảo
"mua nồi thật to"; Chinchilla đo ra "nồi vừa phải + nhiều nguyên liệu hơn" mới no. GPT-3 mua cái nồi khổng
lồ rồi bỏ đói nó.

**2. Dẫn xuất từ đầu (derive).** Quy tắc 20 tok/param từ đâu? `D/N = (C/(6N))/N = C/(6N²)`. Với `N* ∝ C^a`,
`D*/N* = C/(6·N*²) ∝ C^(1-2a)`. Nếu a=0.5 CHÍNH XÁC → `C^0 = hằng số` → **tỉ lệ token/param KHÔNG đổi theo
scale** — đó là toàn bộ nội dung "20 tok/param" của Chinchilla. (Fit course data cho a=0.4687 hơi <0.5 nên
tỉ lệ *tăng chậm* theo C; ở 1e23 ta đo D*/N*≈3.4, thấp hơn 20 vì đây là fit của *course dataset* nhỏ, không
phải luật Chinchilla gốc — thành thật ghi rõ.) Số param: `N ≈ 12·L·d²` (`nonembed_params` :83 — attention
4d² + MLP 8d² mỗi block). Đảo lại để ra shape cụ thể: cố định aspect `d/L = r` (mặc định 128, regime GPT-2/3
mà Kaplan thấy loss phẳng), thay `d=rL` vào `N=12r²L³` → giải `L` từ căn bậc ba, rồi khôi phục `d = sqrt(N/
(12L))` để dồn sai-số-làm-tròn vào một chỗ (`propose_shape` :88–100). Cầu `C=6ND` (`compute_from_params_
tokens` :73) đóng vòng: cho N* và D*, `C = 6·N*·D*` phải bằng budget mục tiêu.

**3. Trace code.** `nonembed_params(L, d) = 12·L·d²` (:85). `propose_shape(N, aspect=128)` (:88): `L =
round((N/(12·aspect²))^(1/3))` (:98), `d = round((N/(12·L))^0.5)` (:99). `compute_from_params_tokens(N, D)
= 6·N·D` (:75), nghịch đảo `tokens_from_compute_params(C, N) = C/(6N)` (:80). Test `test_nonembed_params_
and_inverse` (:111): round-trip `nonembed_params(propose_shape(target)) ≈ target` rel 2%, và aspect giữ
trong [0.6, 1.5]×128. Số ngoại suy (RESULTS.md §W5, `test_course_json_regression_locked` :161–164): N*
7.005e10 @1e23 / 2.061e11 @1e24; D* 2.379e11 / 8.086e11; consistency `6·N*·D* ≈ target` rel 2% (:166–168).
Honesty-lock: đây là **ngoại suy ×33–×333** quá budget fit lớn nhất (3e21) — [INFERENCE], không [FACT].

**4. Cổng teach-back.** (a) Dẫn xuất "20 token/param ≈ hằng số" từ a=0.5 và `D/N = C/(6N²)` — vì sao chính
a=0.5 (không phải a=0.7) làm tỉ lệ độc lập scale? (b) *Sửa-và-đoán:* GPT-3 (175B, 300B token) — dùng
`C=6ND` tính compute của nó, rồi hỏi: ở CÙNG compute đó, Chinchilla-optimal ra N* bao nhiêu (nhỏ hơn hay
lớn hơn 175B), và điều đó nói gì về việc nó "phí" bao nhiêu phần compute vào size thay vì data?

**5. Frontier.** Chinchilla (Hoffmann 2203.15556) là kết quả gốc; headline front frontier ở đây tuân đúng
nó: **nanochat d20 ~561M param trên ~11.9B token = ~21 tok/param, "Chinchilla-optimal"**
(`FRONTIER_2026_ABLATIONS.md:62`). Cross-reference perf: cầu `C=6ND` là *chính* cái mà `utils/mfu.py` dùng
để tính MFU (perf roadmap/S6 Bài 6.4 — tái tạo PaLM 540B 46.2% MFU từ 6ND) — cùng một identity, một bên
dùng để *lập kế hoạch*, một bên để *chấm điểm hiệu suất*. Câu interview: "GPT-3 sai ở đâu về scaling, và
bạn chứng minh nó under-trained bằng số thế nào?" (tính compute, so N* Chinchilla, chỉ ra tỉ lệ token/param
lệch bậc độ lớn).

---

## Bài 6.4 — Budget planner: biến FLOP-budget thành (N,D,steps,LR) dưới ngân sách đối kháng (`scaling/planner.py` · `QueryPlanner` :140 · `declare_grid` :178 · `RunConfig.train_flops` :114)
> **Câu hỏi first-principles:** bạn có 12 B200-giờ query lên training-API để đặt MỘT cược compute-optimal
> cho run 48-giờ — tiêu chúng thế nào để không bao giờ bị API từ chối hay rút cạn budget giữa chừng?
> **Neo (invariant / số đo):** ledger identity `spent + reserved + remaining == total` đúng SAU mọi thao
> tác (`_assert_ledger_identity`, `test_ledger_identity_holds_through_mixed_lifecycle` :188); và end-to-end
> planner phục hồi optimum gieo sẵn trong **1.35×** (`test_end_to_end...` :321). [FACT, MEASURED — bất biến test.]

**1. Feynman — bài toán bằng lời.** Bài 6.1–6.3 cho *phương pháp*; Bài này là *kỷ luật vận hành* để chạy
nó dưới một ngân sách thật với một API **đối kháng**. Stanford training-API tính wall-clock giây vào cap
cứng 12 B200-giờ (43_200 s), luật kế toán tàn nhẫn: submit **giữ trọn `max_runtime_seconds`** (reserve bi
quan); complete **hoàn tiền xuống runtime thực** (sàn 1 s, lạc quan); timeout **tính đủ reservation**;
config trùng → 409; reservation vượt remaining → 400. Planner này LÀ kế toán đó dưới dạng logic thuần —
HTTP client sống ở scaffold khác. Analogy: đặt bàn nhà hàng phải *đặt cọc trọn suất* khi giữ chỗ, chỉ hoàn
lại phần không ăn khi rời; đến muộn quá giờ giữ chỗ mất cọc. Đánh đổi cốt lõi: **reserve bi quan, refund
lạc quan, không bao giờ đua với cap** — một lưới predeclared có worst-case (mọi run timeout) vẫn lọt
remaining thì không thứ tự hoàn thành nào có thể mắc kẹt kế hoạch nửa chừng.

**2. Dẫn xuất từ đầu (derive).** Bất biến sống-còn: **tiền chỉ chảy `remaining → reserved` (submit) →
`spent`/về lại (finish), không bao giờ tự sinh** (docstring :14–16). Từ đó `remaining = total - spent -
reserved` (:164) luôn đúng. Phí terminal của một run hoàn thành = `clamp(actual, 1, max_runtime)` (:231 —
gương biểu thức case của server); run in-flight giữ đúng `max_runtime`. Kỷ luật chống-đối-kháng:
`declare_grid` (:178) cộng worst-case = Σ`max_runtime` (:197) và từ chối CẢ lưới nếu vượt remaining (:198)
— all-or-nothing, nên timeout-tính-đủ không thể làm cạn giữa chừng. `RunConfig.train_flops = 6·N·D` (:117,
lại `C=6ND`) neo mỗi cell của lưới vào một budget IsoFLOP (Bài 6.2). LR: mặc định 3e-4 (`RunConfig` :93) —
một hằng số hợp lý ở scale này; frontier dùng muP để *transfer* LR qua width thay vì cố định (mục 5).
`propose_shape`/`nonembed_params` (Bài 6.3) sinh ứng viên N; `total_train_tokens` chia hết `seq_len·batch`
(:104) để ra số optimizer step nguyên (steps = tokens/(512·batch)).

**3. Trace code.** Quy trình runbook (docstring :24–27, test end-to-end :272): (1) **calibrate** secs-per-
FLOP từ 2 run rẻ đã complete (:277–285). (2) `declare_grid(grid)` (:178) — validate + reject trùng +
kiểm worst-case lọt cap, trả worst-case; lưới immutable (:186). (3) tiêu budget: `submit(config, max)`
(:206) reserve trọn max → `complete(rid, actual, loss)` (:225) hoàn xuống `clamp(actual,1,max)` (:231). (4)
`results()` (:170) trả (config, loss) → nạp vào `isoflop.isoflop_min` + `fit_powerlaw` (Bài 6.2) → ngoại
suy N* ở C_target → `nearest_arch` chọn cell ladder gần nhất trong log-space (:324). Bất biến ledger kiểm
mỗi bước (`_assert_ledger_identity` :55). Các test HTTP-semantics pin từng luật: reserve-full (:60),
refund-to-actual (:69), floor 1 s (:81), timeout-đủ (:93), 400 over-budget (:103), 409 duplicate (:117),
422 invalid (:139), undeclared→refuse (:170). End-to-end (:272) gieo mặt loss Chinchilla `L=E+A/N^α+B/D^β`,
tiêu cả 43_200 s, phục hồi N* gieo trong 1.35× và khớp `nearest_arch` (:329).

**4. Cổng teach-back.** (a) Vì sao "reserve trọn max, refund xuống actual" + "worst-case lưới phải lọt cap"
KẾT HỢP làm kế hoạch *không thể* bị strand nửa chừng — bất kể thứ tự run xong/timeout? (b) *Sửa-và-đoán:*
nếu planner reserve `actual` (lạc quan) thay vì `max` (bi quan) lúc submit — kịch bản nào làm `remaining`
tụt âm và API bắn 400 giữa lưới? (Gợi: một run timeout bị tính `max` > `actual` đã trừ → thâm hụt.)

**5. Frontier.** Đây là kỷ luật "research-as-MDP" (FOP-5) mã hoá thành code: một active experiment, ngân
sách hữu hạn, kill-criterion. Cross-reference frontier đắt nhất: LR/init trong `RunConfig` là **hằng số cố
định** — frontier 2026 KHÔNG cố định chúng mà dùng **muP / μTransfer** (Yang 2203.03466,
`FRONTIER_PRACTICE_2026.md:481–486`): tune HP một lần trên proxy hẹp, transfer zero-shot sang run lớn, với
coordinate-check (activation RMS width-invariant) làm cổng đúng-sai; nếu không có nó, ladder loss bị *nhiễu
bởi retune HP mỗi bậc*. Essential AI (2505.02222) còn cho thấy Muon *nới* (không thay) muP tới ~4B param
(`FRONTIER_2026_ABLATIONS.md:218–219`). Câu interview: "Có 12 B200-giờ query để đặt một cược compute-optimal
48-giờ — tiêu thế nào?" (predeclare lưới IsoFLOP worst-case-lọt-cap, calibrate từ 2 run rẻ, argmin loss/
budget, fit `N*∝C^a` log-log, ngoại suy MỘT lần, submit MỘT lần — và transfer LR bằng muP thay vì đoán).

---

*Đóng série:* power-law loss-vs-compute là *sự thật thực nghiệm* ngoại suy được (6.1) → phương pháp IsoFLOP
rút (N*,D*) từ họ run iso-compute, gác bằng `a+b=1` (6.2) → Chinchilla đọc ra ~20 tok/param và chỉ mặt
GPT-3 under-trained (6.3) → planner biến budget thành cấu hình dưới ngân sách đối kháng (6.4). Neo cứng của
cả série: **a=0.46868 · b=0.53132 · a+b=1.0000** (fit course data, MEASURED) + identity `C=6ND` (dùng ở
cả `mfu.py` perf roadmap/S6). Số N*/D* ở 1e23/1e24 là *ngoại suy ×333* — [INFERENCE], không phải run thật.
Série sau (post-training/RL) rời khỏi câu hỏi "to bao nhiêu" sang "align thế nào".
