# M6 — Scaling Laws · First-Principles Derivations

> **Đây là gì.** Bản dẫn-xuất **từ nguyên lý gốc** (first principles) cho TOÀN BỘ pillar M6 —
> scaling laws (CS336 A3): 4 micro-concept (Bài 6.1→6.4). Mỗi mục: (1) **câu hỏi** falsifiable,
> (2) **sự thật nền tảng** — áp lực toán/vật-lý ép thiết kế, (3) **dẫn xuất** kèm math, (4) **neo
> code** `file·func·line`, (5) **hình ảnh** (ASCII + shape/dtype + ví dụ số tay-trace), (6) **số đo
> THẬT** (chạy trên chính repo này, không phán), (7) **frontier framing** + cổng phỏng vấn.
>
> **Cách dùng để re-learn.** Đọc phần *Câu hỏi* của mỗi mục → **tự trả lời COLD** (che phần dưới) →
> mở code thật ra đối chiếu → *reconcile cái gap*. Cuối doc có **checklist recall cold** + bảng số
> đo. Đây là bạn đồng hành của `roadmap_model/M6_scaling_laws.md` (reference chung) — doc này là
> *derivation lab* có số đo THẬT.
>
> **Cảnh báo neo.** Line number **trôi** theo commit. Cite ở đây pin theo HEAD `1e7dbc6` (2026-07-14),
> file `scaling/isoflop.py` + `scaling/planner.py`. Nếu lệch, `grep` tên hàm — đừng tin số dòng cứng
> (luật repo: verify, don't trust).
>
> **Nguồn số đo.** Mọi số trong doc này chạy lại được bằng `python - <<'PY' … PY` ghi trong từng mục
> (import `from scratch_llm.scaling.isoflop import …`, package editable-installed). Đó là "DoD là một
> profile, không phải test xanh" (FOP-3) áp cho việc học. Pillar này **CPU-runnable toàn bộ** — không
> có GPU nào chạy ở đây; nó là một *phương pháp thống kê* + một *bộ máy kế toán ngân sách*.

---

## Bức tranh lớn — pillar này trả lời câu hỏi ĐẮT nhất của cả pipeline

Mọi série M1→M5 dựng *một* model rồi *một* vòng train. M6 hỏi câu ở **tầng trên** tất cả:

> **Với `C` FLOP trong tay, model nên TO bao nhiêu (`N` params) và ăn bao nhiêu token (`D`)?**

Đoán sai một bậc là đốt cả cụm GPU vào một model quá to train quá ít — đúng lỗi GPT-3. Nhưng bạn
**không thể** chạy run tỉ-đô để biết đáp án. Scaling law là cây cầu: **đo rẻ ở scale nhỏ → ngoại suy
đắt ở scale lớn.**

```
   NHIỀU run NHỎ (rẻ)                        MỘT cược LỚN (đắt, chạy 1 lần)
   72 run @ 9 budget                                    │
        │                                               │
        ▼   [6.2] isoflop_min: mỗi budget → argmin loss ▼
   (C, N_opt) × 9  ────────────────────────────►  ngoại suy N*,D* @ C_target
        │                                               ▲
        │  [6.1] fit power-law TRONG LOG-SPACE           │  [6.3] propose_shape:
        ▼        N_opt = coeff·C^a   (đường thẳng log-log) N → (n_layer, d_model)
   a = 0.46868   ├─ [6.2] cổng a+b=1  (C=6ND ép cấu trúc) │
   b = 0.53132   ┘        check_exponent_sum → bắt bug     │
        │                                                 │
        └─ [6.3] Chinchilla: a≈b≈0.5 ⇒ tok/param ≈ hằng ──┘
                          GPT-3 (1.71 tok/param) = under-trained ĐO ĐƯỢC
                                                                 │
   [6.4] planner: biến budget → (N,D,steps,LR) DƯỚI API ĐỐI KHÁNG (12 B200-giờ cap)
         reserve bi quan · refund lạc quan · worst-case-lọt-cap · không đua với cap
```

**Áp lực xuyên suốt = KINH TẾ COMPUTE.** Mỗi thiết kế M6 là lời giải cho một áp lực cụ thể:
- **6.1 log-space fit** ← áp lực *thống kê*: power-law trải 7 bậc độ lớn; raw-space least-squares
  bị điểm lớn nhất áp đảo (heteroscedastic). Log-space cho mọi điểm trọng số bằng nhau.
- **6.2 IsoFLOP + cổng a+b=1** ← áp lực *nhất-quán*: `C=6ND` là ràng buộc cứng; hai exponent PHẢI
  cộng thành 1, nếu không thì có bug — cổng bắt trước khi ngoại suy.
- **6.3 Chinchilla ~20 tok/param** ← áp lực *phân bổ*: túi tiền cố định, chia bao nhiêu cho size vs
  data. a≈b≈0.5 ⇒ chia gần đều ⇒ tỉ lệ token/param ~hằng.
- **6.4 budget planner** ← áp lực *vận hành*: API tính tiền tàn nhẫn, cap cứng; kế toán sai một
  bước là bị từ chối hoặc cạn budget giữa chừng.

Một sự thật nền: **`C = 6·N·D`** (compute identity) là trục xương sống — nó xuất hiện ở cả 4 mục và
cả `utils/mfu.py` (perf roadmap). Học M6 = học *áp lực → lời giải*, quanh identity đó.

---

## 6.1 · Vì sao scaling law TỒN TẠI — và vì sao fit trong log-space, không bao giờ raw

**Câu hỏi.** Vì sao *final loss* của một LM lại là **đường thẳng trong log-log** theo compute `C` —
và vì sao MỘT sự thật thực nghiệm đó cho phép tiên đoán model tỉ-tham-số từ model triệu-tham-số?

**Sự thật nền tảng.** Kaplan 2020 / Hoffmann 2022 đo được: plot final loss của các LM đã train
tối-ưu theo compute `C` trên trục log-log → gần như một **đường thẳng** trải qua **bảy bậc độ lớn**
của `C`. Không lý thuyết nào bắt buộc điều này — nó là *quy luật thực nghiệm*, bền đến mức thành công
cụ kỹ thuật. Analogy: như `PV=nRT` — ta không cần cơ học thống kê từng phân tử để *dùng* nó dự đoán;
ta không cần hiểu vì sao loss giảm mượt để *dùng* độ dốc đó ngoại suy.

**Dẫn xuất — vì sao PHẢI log-space.** Đường thẳng trong log-log = **power-law** trong không gian
thường:
```
y = k · x^p          (power-law)
log y = log k + p · log x     ← lấy log hai vế: một ĐƯỜNG THẲNG, độ dốc p, chặn log k
```
Vậy fit power-law = fit tuyến tính trên `(log x, log y)` bằng `np.polyfit(log x, log y, 1)` → phục
hồi exponent `p` chính xác. Nhưng vì sao **KHÔNG** fit `y = k·x^p` trực tiếp bằng least-squares
raw-space? Vì raw-space least-squares là **heteroscedastic**:
```
minimize  Σᵢ (yᵢ − k·xᵢ^p)²
```
Residual của điểm `x` lớn nhất (giá trị `y` cỡ tỉ) có bình phương **áp đảo** cả tổng → nó một mình
kéo exponent theo nó. Log-space cho MỌI điểm **trọng số bằng nhau** (vì nó tối thiểu hoá sai số
*tương đối*, không tuyệt đối): `(log yᵢ − log ŷᵢ)² = (log(yᵢ/ŷᵢ))²`. Đây chính là điều một scaling
law cần — mỗi bậc độ lớn đóng góp như nhau, không phải bậc lớn nhất nuốt trọn.

**Vì sao ngoại suy được?** Power-law là hàm **tự-đồng-dạng** (self-similar): `y(λx)/y(x) = λ^p` —
tỉ lệ chỉ phụ thuộc *tỉ số* scale, không phụ thuộc scale tuyệt đối. Nên độ dốc đo ở `10⁶` param
*giống* độ dốc ở `10⁹` — miễn là **regime còn giữ** (cùng data, cùng arch, cùng optimizer). Đó là
điều kiện sống-còn: đổi data/arch/optimizer → power-law MỚI, ngoại suy cũ vỡ.

**Neo code** (`src/scratch_llm/scaling/isoflop.py`):
```python
@dataclass(frozen=True)
class PowerLaw:                                             # :32 (decorator :31)
    exponent: float
    coeff: float
    def predict(self, x: float) -> float:                  # :38
        return self.coeff * x**self.exponent               # :39  ngoại suy = mũ hoá

def fit_powerlaw(xs, ys) -> PowerLaw:                       # :57
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    if x_arr.size < 2 or x_arr.size != y_arr.size:         # :65  cần ≥2 điểm
        raise ValueError(...)
    if np.any(x_arr <= 0) or np.any(y_arr <= 0):           # :67  log cần strictly >0
        raise ValueError("power-law fit needs strictly positive xs and ys (log-log space)")
    slope, intercept = np.polyfit(np.log(x_arr), np.log(y_arr), 1)   # :69  FIT TRONG LOG-SPACE
    return PowerLaw(exponent=float(slope), coeff=float(np.exp(intercept)))  # :70  coeff = e^intercept
```
Chú ý cái **gap thật giữa textbook và code**: textbook viết `L(C)=E+a·C^(-α)` (có *irreducible loss*
`E`, một tiệm cận). Code này fit **thuần power-law** `y=coeff·x^p` (không có `E`), vì trục `y` ở đây
KHÔNG phải loss mà là **`N_opt` / `D_opt`** (số param/token tối ưu) — hai đại lượng đó *là* power-law
sạch của `C`, không có tiệm cận. `E` chỉ xuất hiện khi fit **loss** trực tiếp (mục 6.3 dùng nó ở dạng
`L=E+A/N^α+B/D^β` trong end-to-end test). Đừng lẫn hai loại fit.

**Hình ảnh — data journey.**
```
xs = [C₁, C₂, …, C₉]   list[float]  (9 budget FLOP, ví dụ 6e18 … 3e21)
ys = [N₁, N₂, …, N₉]   list[float]  (9 N_opt, ví dụ 7.6e8 … 1.2e10)
   │  np.asarray float64            shape (9,)   (9,)
   ▼  np.log(x), np.log(y)          → log-space: giờ là đám điểm ~thẳng hàng
   │
   ▼  np.polyfit(logx, logy, deg=1) → (slope, intercept) = (0.46868, 0.15…)
   │
   ▼  PowerLaw(exponent=slope, coeff=exp(intercept))
   predict(1e23) = coeff · (1e23)^0.46868  → một số float (ngoại suy)

Raw-space (SAI):        log-space (ĐÚNG):
 y                        log y
 │            ●(áp đảo)    │        ●
 │                        │      ●
 │       ●                │    ●
 │   ●                    │  ●          mỗi điểm trọng số =, đường thẳng
 │ ●___________ x         │●___________ log x
 fit nghiêng về ● lớn      độ dốc = p ổn định
```

**Số đo THẬT** (chạy `python`, HEAD 1e7dbc6):
```
# (1) fit phục hồi ĐÚNG luật gieo tới fp precision:
planted exponent 0.42, coeff 3.7  →  fit exponent=0.42 (abs err 0.00e+00), coeff=3.6999999999999
# (2) log vs raw với MỘT outlier giảm ×3 ở x lớn nhất (y=2·x^0.5, 20 điểm):
log-fit exp = 0.47842  (gap 0.0216 so với 0.5)   ← barely moves
raw-fit exp = 0.30600  (gap 0.1940)              ← điểm lớn nhất kéo lệch, hai estimator KHÁC NHAU
```
Test-oracle pin: `test_fit_powerlaw_recovers_exact_law` (:62, abs 1e-10),
`test_fit_is_log_space_not_raw_space` (:72 — log_gap<0.03, raw_gap>0.15).

**Frontier / cổng.** Kaplan (2001.08361) + Chinchilla (2203.15556) là hai power-law loss-vs-compute
nổi tiếng nhất. Frontier 2026 mở rộng thành **hyperparameter transfer / muP** (Yang 2203.03466):
không chỉ *loss* mà cả *LR/init tối ưu* cũng transfer qua width — tune một lần trên proxy nhỏ,
zero-shot sang model lớn, với "coordinate-check" (activation RMS width-invariant) làm cổng đúng-sai.
**Interview gate:** "Vì sao fit scaling law làm trong log-space? Cho 20 run nhỏ, bạn tiên đoán loss
của run to 1000× thế nào — và điều gì *phá* ngoại suy đó?" (regime shift). **Trait:** first-principles
+ claims-honesty (nêu factor ngoại suy). **Bucket:** table-stakes (mọi RE phải biết), nhưng "biết
regime nào phá nó" là dấu hiệu senior.

---

## 6.2 · Phương pháp IsoFLOP — fit `(N*, D*)` từ họ run + cổng `a+b=1`

**Câu hỏi.** Cho một họ run trải nhiều kích thước `N` ở CÙNG budget `C`, làm sao rút ra cấu hình
compute-optimal `(N*, D*)` — và làm sao *bắt lỗi* trước khi ngoại suy đắt tiền?

**Sự thật nền tảng.** "IsoFLOP" = lát cắt đẳng-compute. Cố định budget `C` (ví dụ 6e18 FLOP); train
một *họ* model `N` khác nhau, mỗi cái đúng `D = C/(6N)` token để tiêu HẾT `C`. Plot final loss theo
`N` → đường cong hình **parabola** (cái chén):
- `N` quá nhỏ → model không đủ dung lượng hấp thụ `C` (under-capacity, loss cao),
- `N` quá lớn → phải cắt `D` nên train quá ít bước (under-training, loss cao),
- **đáy chén = `(C, N*)`** — compute-optimal cho budget đó.

Lặp cho nhiều `C` → họ điểm `(C, N*)`, và ĐÓ là data vào cho power-law 6.1.

**Dẫn xuất — vì sao có đáy (parabola có min).** Dùng dạng Chinchilla `L(N,D) = E + A·N^(-α) +
B·D^(-β)`. Thay `D = C/(6N)`:
```
L(N) = E + A·N^(-α) + B·(6N/C)^β
dL/dN = −α·A·N^(-α-1) + β·B·(6/C)^β·N^(β-1) = 0
⟹  α·A·N^(-α-1) = β·B·(6/C)^β·N^(β-1)
⟹  N^(α+β) = (α·A)/(β·B) · (C/6)^β
⟹  N* = [ (α·A)/(β·B) · (C/6)^β ]^(1/(α+β))          ← đáy chén, dạng đóng
```
Đây chính xác là `_n_true` trong test end-to-end (`tests/test_planner.py:238`). Một *dốc lên* (term
`N^β`, under-training) + một *dốc xuống* (term `N^-α`, under-capacity) giao nhau → min duy nhất.

**Dẫn xuất — cổng `a+b=1` là HỆ QUẢ TẤT YẾU của `C=6ND`.** Giả sử fit ra `N* ∝ C^a` và độc lập
`D* ∝ C^b`. Lấy log identity:
```
C = 6·N·D   ⟹   log C = log 6 + log N + log D
thay N ∝ C^a, D ∝ C^b:   C^1 = const · C^a · C^b = const · C^(a+b)
so khớp mũ hai vế       ⟹   a + b = 1     (về mặt CẤU TRÚC)
```
Nên nếu fit độc lập cho ra `a+b` lệch xa 1 → CÓ BUG (min-picking hỏng, hoặc cầu `D=C/(6N)` ghép
lệch hàng). `check_exponent_sum(a, b, tol=0.05)` bắn `ValueError` — cổng bắt lỗi *trước* ngoại suy.

> **GAP THẬT giữa code và textbook (phải teach cái này).** Trong repo, pipeline KHÔNG fit `D*` độc
> lập từ đo riêng — nó **dẫn** `D_opt = C/(6·N_opt)` từ CÙNG các hàng `(C, N_opt)` (`isoflop.py:79`,
> `tokens_from_compute_params`). Hệ quả: `a+b = 1.0` **CHÍNH XÁC tới 1e-9**, không phải "≈1 thực
> nghiệm". Vì sao? Nếu `D = C/(6N)` thì `log D = log(C/6) − log N`; fit tuyến tính log-log của `D`
> theo `C` cho slope `b`, và vì `N ∝ C^a` (đã fit), nên `log D = log(C/6) − a·log C − … = (1−a)·log C
> + const`, tức `b = 1−a` **về mặt đại số** dưới một linear fit. Cổng `a+b=1` ở đây KHÔNG kiểm
> "identity có đúng không" (nó luôn đúng) — nó kiểm **hàng có ghép đúng không**: nếu ai đó pha `D`
> với `N_opt` *lệch hàng* (đảo thứ tự budget), mỗi fit riêng vẫn trông hợp lý nhưng `a+b` la làng.
> Đó là bug-class thật mà test `test_exponent_sum_gate_fires_on_corrupted_data` (:123) dựng lên. Ở
> Chinchilla GỐC (Approach 2) thì `a` và `b` fit từ *hai đo độc lập*, và `a+b≈1` là một *xác nhận
> thực nghiệm* của phương pháp — đó là bản "textbook". Repo dùng bản "derived", rẻ hơn, nhưng cổng
> đổi ý nghĩa: từ "validate method" → "validate wiring".

**Neo code:**
```python
def isoflop_min(runs):                                     # :42
    by_budget = {}
    for run in runs:
        budget = run["compute_budget"]
        best = by_budget.get(budget)
        if best is None or run["final_loss"] < best["final_loss"]:   # :52  argmin loss/budget
            by_budget[budget] = run
    return [(c, by_budget[c]["parameters"]) for c in sorted(by_budget)]   # :54  (C, N_opt) sort ↑

def tokens_from_compute_params(compute, n_params):         # :78  cầu C=6ND đảo ngược
    return compute / (FLOPS_PER_PARAM_TOKEN * n_params)    # :80  D = C/(6N) — DẪN, không đo riêng

def check_exponent_sum(a, b, tol=0.05):                    # :103
    if abs(a + b - 1.0) > tol:                             # :108
        raise ValueError(f"exponent-sum gate failed: a={a:.4f}, b={b:.4f}, a+b={a+b:.4f} …")  # :109
```
Chú ý `isoflop_min` dùng **min-picking** (argmin loss trực tiếp), KHÔNG fit parabola bậc-2 kiểu
Hoffmann — biến thể CS336, đơn giản hơn, đủ khi lưới `N` dày. Đánh đổi: nếu lưới `N` ở một budget
*thiếu nhánh trái* (chỉ có `N` lớn), min-picking chọn `N` nhỏ nhất còn lại → `N*` thiên lệch → `a`
lệch. Đó là kill-mode phải biết.

**Hình ảnh — parabola + pipeline.**
```
Một budget C=6e18:  loss theo N (8 run trong fixture)
 loss
 7.19 ●                                        argmin loss
 6.75  ●                                          │
 6.41   ●                                         ▼
 6.15    ●                                    (C, N*=762,093,419)
 5.99     ●                                       │  đáy chén
 5.90      ●___●____________________              │
            N nhỏ  N* đáy   N lớn →               │
                                                  ▼
 Lặp 9 budget → 9 điểm (C, N*) → fit_powerlaw log-log → a=0.46868
 D_opt = C/(6N*) từ CÙNG hàng → fit → b=0.53132 → a+b check → PASS
```
Data journey shapes: `runs` = `list[dict]` (72 phần tử, mỗi dict `{parameters, compute_budget,
final_loss}`) → `isoflop_min` → `list[tuple[float,float]]` (9 phần tử `(C, N_opt)`) → tách thành hai
`list[float]` (9,) → `fit_powerlaw` → `PowerLaw(exponent, coeff)`.

**Số đo THẬT** (fixture `tests/fixtures/isoflops_curves.json`, 72 run · 9 budget × 8 run):
```
isoflop_min → 9 (C, N_opt) pairs:
   C=6.0e18  N_opt=  762,093,419      C=1.0e21  N_opt= 6,859,328,563
   C=1.0e19  N_opt=  806,647,749      C=3.0e21  N_opt=12,148,905,329
   C=3.0e19  N_opt=1,536,852,354     (… 9 hàng, budget tăng dần)
N_opt = 1.1634 · C^0.46868
D_opt = 0.1433 · C^0.53132
a = 0.46868 · b = 0.53132 · a+b = 1.0000000000   → check_exponent_sum PASS
# cổng FIRES khi ghép D lệch hàng (đảo n_opts):
  b_corrupt = 1.46180 → a+b = 1.9305 → ValueError "exponent-sum gate failed …"
```
Test-oracle: `test_isoflop_min_nine_budgets_on_course_data` (:49, pin N=762093419 @6e18),
`test_course_json_regression_locked` (:140, a/b lock abs 5e-4, a+b abs 1e-9).

**Frontier / cổng.** IsoFLOP là "Approach 2" trong Chinchilla. Cross-reference đắt nhất: **method
IsoFLOP CHÍNH LÀ tư duy của EV-ranked ablation** (`FRONTIER_2026_ABLATIONS.md` §3) — mỗi rung F1–F11
chạy ở **iso-FLOP** (`C=6ND` cố định) để so *một biến* công bằng (F1 Muon "đạt loss AdamW với ≥15%
ít token hơn ở iso-FLOP" = đúng một điểm parabola so hai optimizer). **Interview gate:** "Derive và
bảo vệ scaling law: vì sao fit log-space, vì sao hai exponent cộng thành một, ngoại suy *thành thật*
tới đâu?" **Trait:** spec-with-falsifiers (cổng `a+b=1` là một pre-registered falsifier). **Bucket:**
table-stakes + research-taste (biết cổng bắt bug gì).

---

## 6.3 · Chinchilla — quy tắc ~20 token/param, và GPT-3 under-trained ĐO ĐƯỢC

**Câu hỏi.** Cho `a≈b≈0.5`, tỉ lệ token/param tối ưu ra bao nhiêu — và tại sao một model 175B train
trên 300B token là sai lầm *đo được* chứ không phải ý kiến?

**Sự thật nền tảng.** Kaplan 2020 kết luận: có thêm compute thì đổ *phần lớn* vào **size** (model to
hơn nhiều, data chỉ nhích). GPT-3 làm đúng lời khuyên đó: 175B param, chỉ 300B token (~1.7
token/param). Chinchilla 2022 chạy lại cẩn thận hơn (IsoFLOP) và **lật ngược**: `N` và `D` nên scale
**gần bằng nhau** (`a≈b≈0.5`) ⇒ tỉ lệ token/param tối ưu ~**hằng số ~20** ở scale họ đo. Hệ quả bàng
hoàng: GPT-3 **under-trained** nặng — cùng compute đó, model 4× NHỎ train trên 4× NHIỀU token *thắng
nó* (Chinchilla 70B, 1.4T token).

**Dẫn xuất — "20 tok/param ≈ hằng" từ đâu.**
```
D/N = (C/(6N)) / N = C/(6N²)
với N* ∝ C^a:   D*/N* = C/(6·N*²) ∝ C / C^(2a) = C^(1−2a)
```
Nếu `a = 0.5` **CHÍNH XÁC** → `C^(1−1) = C^0 = hằng số` → **tỉ lệ token/param KHÔNG đổi theo scale**.
Đó là toàn bộ nội dung "20 tok/param" của Chinchilla. (Course fit cho `a=0.46868 < 0.5` nên tỉ lệ
*tăng chậm* theo `C`: `C^(1−2·0.46868) = C^0.0626`. Ở 1e23 đo `D*/N* ≈ 3.4`, thấp hơn 20 vì đây là
fit của *course dataset nhỏ*, KHÔNG phải luật Chinchilla gốc — **thành thật ghi rõ**: con số 20 là
kết quả run thật của Chinchilla, con số 3.4 là artefact của fixture giáo trình.)

**Dẫn xuất — từ `N` ra shape cụ thể `(n_layer, d_model)`.** Param non-embedding:
```
N ≈ 12·L·d²      (mỗi block: attention 4d² [q,k,v,o] + MLP 8d² [w1,w2 với d_ff=4d])
```
Chú ý: SwiGLU `d_ff=8/3·d` (M2.5) cũng cho `3·d·(8/3 d) = 8d²` — nên `8d²` đúng cho *cả hai* FFN, đây
là lý do iso-param M2.5 khớp M6. Đảo `N=12Ld²` để ra shape: cố định aspect `r = d/L` (mặc định 128 —
regime GPT-2/3 Kaplan thấy loss *phẳng* qua đó). Thay `d = r·L`:
```
N = 12·r²·L³   ⟹   L = (N / (12·r²))^(1/3)      ← giải L từ căn bậc ba
d = sqrt(N / (12·L))                              ← khôi phục d, dồn sai-số-làm-tròn vào MỘT chỗ
```

**Dẫn xuất — chứng minh GPT-3 under-trained bằng SỐ.** GPT-3: `N=175e9, D=300e9`.
```
C = 6·N·D = 6 · 175e9 · 300e9 = 3.15e23 FLOP
tok/param thực = D/N = 300/175 = 1.71     ← so với ~20 tối ưu ⇒ thiếu data ~12×
```
Ở CÙNG compute `3.15e23`, đọc từ course-fit: `N* ≈ 120B` (< 175B) trên `D* ≈ 438B` token. Nghĩa là
GPT-3 nên *nhỏ hơn* và ăn *nhiều token hơn*. (Con số 120B/438B là [INFERENCE] — ngoại suy course-fit;
kết luận định tính "under-trained, nên nhỏ hơn + nhiều data hơn" là [FACT] từ Chinchilla gốc.)

**Neo code:**
```python
def nonembed_params(n_layer, d_model):                     # :83
    return 12 * n_layer * d_model**2                       # :85  N ≈ 12·L·d²

def propose_shape(n_params, aspect_ratio=128.0):           # :88
    if n_params <= 0 or aspect_ratio <= 0: raise ValueError(...)          # :96
    n_layer = max(1, round((n_params / (12.0 * aspect_ratio**2)) ** (1/3)))  # :98  L từ ∛
    d_model = max(1, round((n_params / (12.0 * n_layer)) ** 0.5))            # :99  d = √(N/12L)
    return n_layer, d_model                                # :100

def compute_from_params_tokens(n_params, n_tokens):        # :73
    return FLOPS_PER_PARAM_TOKEN * n_params * n_tokens      # :75  C = 6·N·D
```

**Hình ảnh — Kaplan vs Chinchilla + shape inversion.**
```
Cùng compute C:
  Kaplan   → "mua NỒI thật to"  → N khổng lồ, D nhỏ  (GPT-3: 175B / 300B, 1.7 tok/param)
  Chinchilla → "nồi vừa + nhiều nguyên liệu" → N vừa, D lớn (70B / 1.4T, ~20 tok/param) ✓ thắng

Shape inversion (N → L, d):
  N=7.005e10 ──► L = (N/(12·128²))^(1/3) = 71
             ──► d = √(N/(12·71))       = 9068     (check d/L = 127.7 ≈ 128 ✓)
             ──► recon 12·71·9068² = 7.006e10 (khớp N tới rounding)
```

**Số đo THẬT:**
```
# ngoại suy course-fit (×33–×333 quá budget lớn nhất 3e21 — [INFERENCE]):
C=1e23: N_opt=7.005e10 (70.1B)  D_opt=2.379e11 (238B)  D/N=3.40  shape L=71  d=9068   6ND=1.000e23 ✓
C=1e24: N_opt=2.061e11 (206B)   D_opt=8.086e11 (809B)  D/N=3.92  shape L=102 d=12977  6ND=1.000e24 ✓
extrapolation factor 1e24 / 3e21 = 333×
nonembed_params(24, 2048) = 1,207,959,552 = 12·24·2048²   (đúng công thức)
# GPT-3 under-training, ĐO:
GPT-3: N=175e9 D=300e9 → C=6ND=3.150e23,  tok/param = 1.71
  ở CÙNG C, course-fit N_opt=1.199e11 (120B), D_opt=4.377e11 (438B), tok/param=3.6   [INFERENCE]
# nanochat d20 (front frontier repo này) TUÂN Chinchilla:
nanochat d20: ~561M param / ~11.9B token → 21.2 tok/param  ≈ "Chinchilla-optimal" ~20 ✓ [FACT]
```
Test-oracle: `test_nonembed_params_and_inverse` (:111, round-trip rel 2%, aspect trong [0.6,1.5]×128),
`test_course_json_regression_locked` (:161–164, N*/D* @1e23/1e24 pin rel 1e-3).

**Frontier / cổng.** Chinchilla (Hoffmann 2203.15556) là kết quả gốc; headline front frontier repo
này tuân đúng nó (nanochat d20 = 21.2 tok/param). Cross-reference perf: cầu `C=6ND` là *chính* cái
`utils/mfu.py` dùng tính MFU (tái tạo PaLM 540B 46.2% MFU) — cùng identity, một bên *lập kế hoạch*,
một bên *chấm hiệu suất*. **Interview gate:** "GPT-3 sai ở đâu về scaling, chứng minh under-trained
bằng số thế nào?" (tính `C=6ND`, so `N*` Chinchilla, chỉ ra tok/param lệch bậc độ lớn). **Trait:**
predict-the-number + claims-honesty (phân biệt 3.4 course-fit vs 20 Chinchilla-gốc). **Bucket:**
table-stakes; nhưng đọc được "20 là fit thật, 3.4 là artefact fixture" là dấu hiệu senior.

---

## 6.4 · Budget planner — biến FLOP-budget thành `(N,D,steps,LR)` dưới API ĐỐI KHÁNG

**Câu hỏi.** Bạn có 12 B200-giờ query lên training-API để đặt MỘT cược compute-optimal cho run
48-giờ — tiêu chúng thế nào để KHÔNG bao giờ bị API từ chối hay rút cạn budget giữa chừng?

**Sự thật nền tảng.** Bài 6.1–6.3 cho *phương pháp*; Bài này là *kỷ luật vận hành* để chạy nó dưới
một ngân sách thật với API **đối kháng**. Stanford training-API tính wall-clock giây vào cap cứng 12
B200-giờ (`43_200 s`), luật kế toán tàn nhẫn:
- **submit** → giữ trọn `max_runtime_seconds` (reserve **bi quan**),
- **complete** → hoàn tiền xuống runtime thực, sàn 1 s (refund **lạc quan**),
- **timeout** → tính đủ reservation (adversary),
- config trùng → HTTP 409; reservation > remaining → HTTP 400; config sai → 422.

Planner này LÀ kế toán đó dưới dạng logic thuần (HTTP client sống ở scaffold khác). Analogy: đặt bàn
nhà hàng phải *đặt cọc trọn suất* khi giữ chỗ, chỉ hoàn phần không ăn khi rời; đến muộn quá giờ giữ
chỗ mất cọc.

**Dẫn xuất — bất biến sống-còn.** Tiền chỉ chảy `remaining → reserved` (submit) → `spent`/về lại
(finish), **không bao giờ tự sinh**. Từ đó:
```
spent + reserved + remaining == total_budget_seconds       (đúng SAU mọi thao tác)
remaining = total − spent − reserved                       (planner.py:165)
```
Phí terminal của một run hoàn thành:
```
charge = clamp(actual, 1, max_runtime) = min(max(actual, 1), max_runtime)   (:231)
```
gương biểu thức case của server (`greatest(1.0, used_runtime)`, capped at reservation). Run in-flight
giữ đúng `max_runtime` (:161).

**Dẫn xuất — vì sao "worst-case-lọt-cap" làm kế hoạch KHÔNG THỂ bị strand.** Đây là mấu chốt. Kỷ
luật chống-đối-kháng: `declare_grid` cộng worst-case = `Σ max_runtime` (mọi run timeout) và **từ
chối CẢ lưới** nếu vượt remaining (all-or-nothing):
```
Σ max_runtimeᵢ  ≤  remaining     ⟹  declare OK, ngược lại InsufficientBudgetError (:198)
```
Vì sao điều này đủ? Vì worst-case (mọi run tính đủ `max_runtime`) là **cận trên** của mọi kịch bản:
một run *hoàn thành* chỉ tốn `clamp(actual,1,max) ≤ max`, tức **rẻ hơn hoặc bằng** worst-case. Nên
nếu worst-case lọt cap thì **không thứ tự hoàn thành/timeout nào** có thể làm `remaining` tụt âm giữa
chừng. Đây là một chứng minh bất-biến kiểu *monotone*: reserve theo cận trên → thực chi luôn ≤ dự trù
→ không bao giờ vỡ cap. Đảo lại (sửa-và-đoán): nếu planner reserve `actual` (lạc quan) lúc submit,
thì một run *timeout* bị tính `max > actual` đã trừ → **thâm hụt** → API 400 giữa lưới. Đó là lý do
reserve PHẢI bi quan.

**Dẫn xuất — nối về 6.1–6.3.** Mỗi cell của lưới neo vào một budget IsoFLOP:
```
RunConfig.nonembed_params = 12 · n_layer · d_model²        (:112, = N của 6.3)
RunConfig.train_flops     = 6 · N · D                       (:117, = C của 6.2)
total_train_tokens % (SEQ_LEN · batch) == 0                 (:104, để steps nguyên)
steps = total_train_tokens / (512 · batch)
```
Runbook đầy đủ: (1) **calibrate** secs-per-FLOP từ 2 run rẻ đã complete; (2) `declare_grid` (validate
+ reject trùng + worst-case lọt cap); (3) tiêu budget (`submit` reserve trọn max → `complete` hoàn
xuống clamp); (4) `results()` → `isoflop.isoflop_min` + `fit_powerlaw` (6.2) → ngoại suy `N*` @
`C_target` → `nearest_arch` chọn cell ladder gần nhất trong log-space.

**Neo code:**
```python
@property
def remaining_seconds(self):                               # :163
    return self.total_budget_seconds - self.spent_seconds - self.reserved_seconds   # :165

def declare_grid(self, grid):                              # :178
    ...
    worst_case = 0.0
    for run in grid:
        run.config.validate()                              # :191  422 nếu sai
        if run.config in configs or run.config in self._seen:
            raise DuplicateConfigError(...)                # :195  409
        worst_case += run.max_runtime_seconds              # :197  Σ max_runtime
    if worst_case > self.remaining_seconds:                # :198  gate
        raise InsufficientBudgetError(...)                 # :199  all-or-nothing
    self._declared = configs; return worst_case

def submit(self, config, max_runtime_seconds):             # :206
    ...
    if max_runtime_seconds > self.remaining_seconds:       # :215
        raise InsufficientBudgetError(...)                 # :216  400
    self._entries.append(LedgerEntry(run_id, config, float(max_runtime_seconds)))  # :221 reserve MAX

def complete(self, run_id, actual_runtime_seconds, final_loss):   # :225
    entry = self._reserved_entry(run_id)
    charge = min(max(actual_runtime_seconds, MIN_CHARGE_SECONDS), entry.max_runtime_seconds)  # :231
    entry.status = "completed"; entry.charged_seconds = charge; entry.final_loss = float(final_loss)
    return charge

def fail_timeout(self, run_id):                            # :237
    entry = self._reserved_entry(run_id)
    entry.status = "timed_out"; entry.charged_seconds = entry.max_runtime_seconds  # :241  tính đủ
    return entry.charged_seconds
```

**Hình ảnh — dòng tiền của một run.**
```
                     total = 43,200 s
   remaining ──submit(max=600)──► reserved(600)
        │  (bi quan: giữ trọn max)      │
        │                               ├─ complete(actual=42.5) ──► spent(42.5), refund 557.5 về remaining
        │                               │      charge = clamp(42.5, 1, 600) = 42.5   (lạc quan)
        │                               └─ fail_timeout ──────────► spent(600)        (đủ, adversary)
        ▼
   BẤT BIẾN mọi lúc:  spent + reserved + remaining == 43,200

   declare_grid: Σ max_runtime = worst-case (mọi run timeout)
      worst-case ≤ remaining ? → OK (không kịch bản nào strand)  :  reject cả lưới
```

**Số đo THẬT:**
```
TOTAL_BUDGET_SECONDS = 43,200.0  (= 12 B200-giờ)
submit(max=600):    reserved=600  spent=0   remaining=42,600   identity sum = 43,200 ✓
complete(actual=42.5): charge=42.5  spent=42.5  reserved=0  remaining=43,157.5     (refund 557.5)
complete floor:  actual=0.05  → charge=1.0     (sàn 1 s)
complete cap:    actual=1e6   → charge=600.0   (trần = max reservation)
timeout:         fail_timeout → charge=600.0   (đủ reservation)
adversarial:  400 over-budget (600 > remaining 400) · 409 duplicate · 422 heads%kv!=0
declare_grid: worst-case 1500 > 1000 → REJECT cả lưới; worst-case 900 ≤ 1000 → OK;
              submit config ngoài lưới → 400 UndeclaredConfigError
train_flops:  cfg L=24 d=2048 batch=256 tokens=13,107,200 → N=1,207,959,552, C=6ND=9.500e16
```
Test-oracle: `test_ledger_identity_holds_through_mixed_lifecycle` (:188),
`test_submit_reserves_full_max_runtime` (:60), `test_completion_refunds_down_to_actual_runtime` (:69),
`test_declared_grid_worst_case_must_fit_the_cap` (:153),
`test_end_to_end_planner_recovers_planted_optimum` (:272 — phục hồi optimum gieo trong 1.35×).

**Frontier / cổng.** Đây là kỷ luật "research-as-MDP" (FOP-5) mã hoá thành code: một active
experiment, ngân sách hữu hạn, kill-criterion. Cross-reference đắt nhất: `RunConfig.learning_rate`
là **hằng số cố định** (3e-4, :93) — frontier 2026 KHÔNG cố định mà dùng **muP / μTransfer** (Yang
2203.03466): tune HP một lần trên proxy hẹp, transfer zero-shot sang run lớn, với coordinate-check
làm cổng; nếu không có nó, ladder loss bị *nhiễu bởi retune HP mỗi bậc*. Essential AI (2505.02222)
còn cho thấy Muon *nới* (không thay) muP tới ~4B param. **Interview gate:** "Có 12 B200-giờ query để
đặt một cược compute-optimal 48-giờ — tiêu thế nào?" (predeclare lưới IsoFLOP worst-case-lọt-cap,
calibrate từ 2 run rẻ, argmin loss/budget, fit `N*∝C^a` log-log, ngoại suy MỘT lần, submit MỘT lần —
và transfer LR bằng muP thay vì đoán). **Trait:** research-as-MDP + spec-with-falsifiers (bất biến
ledger là một invariant pre-registered). **Bucket:** research-taste differentiator (kỷ luật budget
là thứ phân biệt RE senior).

---

## Bảng số đo THẬT (chạy lại được — DoD-là-profile · HEAD 1e7dbc6)

| Concept | Đo | Kết quả | Ý nghĩa |
|---|---|---|---|
| 6.1 | fit phục hồi luật gieo | exp 0.42 (abs err **0.00e+00**), coeff 3.7 | log-space polyfit exact |
| 6.1 | log vs raw + 1 outlier | log-exp **0.478** (gap .022) vs raw-exp **0.306** (gap .194) | heteroscedastic ⇒ phải log |
| 6.2 | isoflop_min | **9** cặp (C, N_opt); @6e18 N*=**762,093,419** | min-picking per budget |
| 6.2 | fit exponents | a=**0.46868** · b=**0.53132** · a+b=**1.0000000000** | C=6ND ép a+b=1 (structural) |
| 6.2 | cổng trên data hỏng | ghép lệch hàng → a+b=**1.9305** → ValueError | gate bắt wiring bug |
| 6.3 | ngoại suy @1e23 | N*=**7.005e10** (70B), D*=**2.379e11** (238B), D/N=**3.40** | Chinchilla-optimal (course fit) |
| 6.3 | ngoại suy @1e24 | N*=**2.061e11** (206B), D*=**8.086e11**, ×**333** quá data | [INFERENCE] honesty |
| 6.3 | propose_shape(N=7e10) | L=**71**, d=**9068** (d/L=127.7≈128), recon 7.006e10 | N=12Ld² inversion |
| 6.3 | GPT-3 under-train | C=6ND=**3.15e23**, tok/param=**1.71** (vs ~20) | under-trained ĐO ĐƯỢC |
| 6.3 | nanochat d20 | 561M/11.9B = **21.2 tok/param** | Chinchilla-optimal thật |
| 6.4 | ledger identity | submit600→ **43,200** = spent+reserved+remaining | tiền không tự sinh |
| 6.4 | refund/floor/cap | actual 42.5→**42.5** · 0.05→**1.0** · 1e6→**600** | clamp(actual,1,max) |
| 6.4 | timeout | fail_timeout → charge **600** (đủ) | adversary → reserve bi quan |
| 6.4 | declare_grid | worst 1500>1000 REJECT; 900≤1000 OK | all-or-nothing chống strand |

---

## Checklist recall COLD (che phần trên, tự trả lời — đây là cách re-learn)

1. **6.1** Vì sao fit power-law làm trong **log-space**? Điều gì hỏng nếu fit raw-space (dẫn từ
   heteroscedasticity)? Vì sao power-law *ngoại suy được* (self-similar) — và điều gì *phá* nó?
2. **6.1** `PowerLaw.predict` ngoại suy thế nào? Code fit **loss** hay fit **N_opt/D_opt** — khác gì
   (có `E` tiệm cận hay không)?
3. **6.2** Dẫn `a+b=1` từ `C=6ND` bằng đại số log. Trong REPO nó exact 1e-9 (không "≈1") — vì sao?
   (D dẫn từ cùng hàng). Vậy cổng `check_exponent_sum` thực chất kiểm gì?
4. **6.2** `isoflop_min` dùng min-picking hay fit parabola? Nếu lưới `N` một budget *thiếu nhánh
   trái*, `N*` lệch hướng nào, `a` cao hay thấp hơn thật?
5. **6.3** Dẫn "20 tok/param ≈ hằng" từ `a=0.5` và `D/N = C/(6N²)`. Vì sao chính `a=0.5` (không
   `a=0.7`) làm tỉ lệ độc-lập-scale? Course fit cho D/N=3.4 chứ không 20 — vì sao (thành thật)?
6. **6.3** Dẫn `N = 12·L·d²` (attention 4d² + MLP 8d²). SwiGLU `d_ff=8/3·d` có phá `8d²` không?
   `propose_shape` đảo ngược ra `(L,d)` thế nào?
7. **6.3** GPT-3 (175B, 300B token): tính `C=6ND`, tok/param, và ở CÙNG `C` thì Chinchilla-optimal
   `N*` nhỏ hơn hay lớn hơn 175B? Nói gì về "phí compute vào size"?
8. **6.4** Vì sao "reserve trọn max" + "worst-case lưới lọt cap" KẾT HỢP làm kế hoạch *không thể* bị
   strand — bất kể thứ tự run xong/timeout? (chứng minh monotone: thực chi ≤ dự trù).
9. **6.4** `charge` của một run hoàn thành = ? (clamp). Timeout tính bao nhiêu? Nếu reserve `actual`
   thay vì `max` lúc submit — kịch bản nào làm `remaining` tụt âm?
10. **6.4** `RunConfig.train_flops` = ? Nối `nonembed_params`/`train_flops`/`total_train_tokens` về
    `C=6ND` và về pipeline 6.2 thế nào (mỗi cell lưới = một điểm IsoFLOP)?

> Trả lời cold được cả 10 = **M6 thật sự OWNED** (interview-grade). Vấp câu nào → mở đúng mục đó,
> hoặc blank-slate hàm tương ứng (`raise NotImplementedError` → test đỏ → tự dẫn → xanh) để re-own
> bằng tay. Đề xuất blank-slate: `fit_powerlaw` (6.1), `check_exponent_sum` (6.2), `propose_shape`
> (6.3), `declare_grid`/`complete` (6.4).

---

*Cross-ref: `roadmap_model/M6_scaling_laws.md` (reference chung, SÉRIE 6) · `PROGRESS.md` (ledger 89
Bài, M6 6.1–6.4) · `bench/RESULTS.md` §W5 (fit gốc + PNG) · sibling: `M2_transformer_forward.md`
(N=12Ld² từ arch), `M3_optimization.md` (AdamW/cosine — LR mà planner cố định) ·
`FRONTIER_2026_ABLATIONS.md` §3 (iso-FLOP ablation) · `utils/mfu.py` (cùng identity C=6ND, bên chấm
hiệu suất). Concept kế: M7 — data pipeline (feed the compute-optimal D tokens).*
