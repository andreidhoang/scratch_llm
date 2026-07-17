# M7 — Data Pipeline · First-Principles Derivations

> **Đây là gì.** Bản dẫn-xuất **từ nguyên lý gốc** (first principles) cho TOÀN BỘ data pipeline của
> CS336 A4 — 4 micro-concept M7 (Bài 7.1→7.4) + 1 mục đóng (7.5) trace hai file còn lại
> (`decontaminate.py` + `shards.py`). Mỗi mục: (1) **Câu hỏi** falsifiable, (2) **Sự thật nền tảng**
> (áp lực vật lý/toán buộc thiết kế), (3) **Dẫn xuất** có công thức, (4) **Neo code** `file·func·line`,
> (5) **Hình ảnh** (ASCII + shape/dtype + numeric micro-trace), (6) **số đo THẬT** (chạy trên chính repo
> này), (7) **frontier framing** + cổng phỏng vấn.
>
> **Cách dùng để re-learn.** Đọc phần *Câu hỏi* của mỗi mục → **tự trả lời cold** (che phần dưới) → mở
> ra đối chiếu. Cuối doc có **checklist recall cold** + bảng số đo. Đây là *derivation lab* có số đo,
> bạn đồng hành của `roadmap_model/M7_data_pipeline.md` (bản đồ trace) — doc này OWN dẫn xuất cuối.
>
> **Cảnh báo neo.** Line number **trôi** theo commit. Cite ở đây pin theo HEAD ngày 2026-07-14
> (`1e7dbc6`). Nếu lệch, `grep` tên hàm — đừng tin số dòng cứng (luật repo: verify, don't trust). Roadmap
> gốc pin theo `4ad0ac5` nên vài số dòng của nó đã dịch; số dưới đây đọc lại từ HEAD hiện tại.
>
> **Nguồn số đo.** Mọi số đánh dấu **[MEASURED]** chạy được lại bằng `scratchpad/measure_m7.py` (import
> từ `scratch_llm.data.*`). `fasttext`/`resiliparse` (`[data]` extras) KHÔNG cài trên card này ⇒ nhánh
> classifier + extraction được đánh **[PREDICTED]** hoặc **[test-invariant]** / **[ADR]**, không phán là đo.
> Đây là "DoD là một profile, không phải test xanh" (FOP-3) áp cho việc học — và **claims honesty** (FOP-4):
> "machinery xanh CI" ≠ "recipe giảm loss X%"; số recipe chỉ tồn tại trong paper ta *cite* (DCLM/FineWeb),
> không *đo*.

---

## Bức tranh lớn — pipeline là một hàm gì, và nó giải áp lực nào?

Série model+train (M1–M6) ngầm giả định **token nào cũng như token nào**. Sai. Ở *fixed compute* (D token
cố định, Chinchilla: D≈20N), thứ quyết định loss cuối không phải thêm layer hay tinh optimizer — mà là
*token NÀO*. M7 là hàm `g: (biển CommonCrawl thô) → (corpus mà mỗi token ĐÁNG train)`. Toàn bộ M7 dựng
dây chuyền lọc quặng đó, chạy **đúng thứ tự pipeline chạy**:

```
CommonCrawl WARC bytes  (HTML rác · spam · trùng lặp · PII · leak eval)
   │  extract (resiliparse)                                   [7.2 · corpus entry]
   ▼
text (str)  ── mỗi doc từ đây là 1 ứng viên corpus
   │
   ├─► language-ID   (fastText lid.176)   cheap, cắt volume LỚN nhất  →  DROP   [7.2]
   ├─► Gopher        (4 ngưỡng structural) cheap, minh bạch, phá-hủy  →  DROP   [7.2]
   ├─► NSFW / toxic  (Jigsaw fastText)     classifier, thấy RAW page  →  DROP   [7.2]
   ├─► PII mask      (regex email/phone/ip) TRANSFORM — không drop!   →  MASK   [7.2]
   ├─► quality       (wiki-vs-cc fastText)  ĐẮT nhất → chạy trên ÍT doc nhất → DROP  [7.3]
   │        (mỗi doc bị charge cho stage ĐẦU TIÊN drop nó — sổ kế toán kép)
   ▼
alive[]  ── survivors
   │
   ├─► dedup LAST    (exact-hash → MinHash/LSH S-curve, trên survivors)  →  DROP  [7.4]
   ├─► decontaminate (13-gram gate vs eval sets, A0)                     →  DROP  [7.5]
   ▼
build_dataset → BPE → tokenize_to_shard → shard_%05d.bin (uint16 memmap + <|eot|>)  [7.5]
   │
   ▼
train.py::get_batch  ── cùng substrate mọi rung downstream (F1-run, SFT, d20) ăn
```

**Bốn khối, mỗi khối một áp lực:**
- **Pipeline order** = *áp lực CHI PHÍ*: sắp trạm theo `c_i/(1−p_i)` — rẻ+cắt-nhiều đi trước. ← 7.1
- **Heuristic (Gopher/C4)** = *áp lực RẺ + MINH BẠCH*: chạy trên MỌI doc nên phải whiteboard được, không cần train. ← 7.2
- **Learned classifier** = *áp lực KHÔNG-CÓ-GROUND-TRUTH*: "quality" vô định nghĩa ⇒ định nghĩa nó bằng *label source*. ← 7.3
- **Dedup** = *áp lực O(N²)*: N triệu doc, không quét mọi cặp ⇒ MinHash ước lượng Jaccard rẻ + LSH banding gom near-linear. ← 7.4
- **Decontaminate + shards** = *áp lực TÍNH TOÀN VẸN + BỘ NHỚ*: eval không được lọt vào train; corpus phải memmap-able. ← 7.5

Một sự thật xuyên suốt: **"garbage in, garbage out" ở đây là TOÁN, không phải khẩu hiệu** — một token spam
tốn ĐÚNG bằng FLOP của một token vàng nhưng dạy model gần như không gì. Học M7 = học *áp lực → lời giải*,
và mỗi lời giải kèm một **bất biến audit** (`total_in == kept + Σ discarded`) — nếu bạn không kế toán được
*filter nào* định hình corpus, bạn không thể debug data recipe.

---

## 7.1 · Pipeline order + discard accounting — vì sao thứ tự là quyết định engineering

**Câu hỏi.** Ở compute budget cố định, (a) vì sao *thứ tự* chạy filter là một quyết định engineering (không
phải chi tiết vặt), và (b) làm sao *chứng minh* mọi document được kế toán đúng một lần?

**Sự thật nền tảng.** Mỗi stage có **chi-phí-per-doc `c_i`** và **tỉ-lệ-giữ `p_i`** (fraction sống sót). Số
doc mà stage `i` phải xử lý = `N · Π_{j<i} p_j` — *phụ thuộc mọi stage đứng trước*. Đây là áp lực chi phí:
đặt trạm đắt sau các trạm cắt-nhiều thì nó thấy ít doc hơn.

**Dẫn xuất (a) — cost-order.** Tổng chi phí pipeline:
```
C = Σ_i  c_i · N · Π_{j<i} p_j
```
Muốn cực tiểu `C`, xét hoán đổi hai stage kề nhau `i, i+1` (argument exchange, kiểu Smith's rule cho
scheduling): giữ nguyên nếu
```
c_i / (1 − p_i)  ≤  c_{i+1} / (1 − p_{i+1})
```
⇒ **sắp stage theo `c_i/(1−p_i)` TĂNG dần**. Trực giác: `1−p_i` = fraction bị *cắt*; ta muốn "chi phí trên
mỗi đơn vị volume cắt được" nhỏ trước. Language-ID rẻ nhất + cắt volume lớn nhất (non-English là phần khổng
lồ của CC) → **đầu tiên**. Quality classifier đắt nhất → **cuối** trong các per-doc stage, để nó chỉ thấy
doc đã sống sót mọi cắt rẻ.

**Hai ngoại lệ có chủ đích (KHÔNG theo cost-order):**
1. **PII mask nằm GIỮA harmful và quality** — nó *transform*, không *drop* → không đổi doc count nên
   cost-order không áp. Đặt sau harmful (để classifier thấy trang RAW; mask sẽ *giấu* signal độc hại) và
   trước quality (để quality + downstream thấy đúng text sẽ được train).
2. **Dedup CUỐI** — dedup sớm = phí signature lên doc mà filter sau vứt anyway; và (exact-line dedup) đếm
   line count *trước filter* bị nhiễm bởi doc chẳng bao giờ vào corpus.

**Dẫn xuất (b) — discard accounting.** Bất biến audit = "sổ kế toán kép" của data: mỗi doc bị charge cho
stage **ĐẦU TIÊN** drop nó (loop `continue` ngay khi fail), nên
```
total_in  ==  kept  +  Σ_stage discarded[stage]
```
Không có bất biến này bạn không thể nói *filter nào* đã định hình corpus — data recipe thành hộp đen.

**Neo code** (`src/scratch_llm/data/pipeline.py`):
```python
STAGE_ORDER = ("extract","language","gopher","nsfw","toxic","quality","dedup")  # :40  cost-ordered
def filter_data(docs, config=None, *, doc_ids=None, work_dir=None):             # :173
    discarded = dict.fromkeys(STAGE_ORDER, 0)                                   # :194  sổ kế toán
    for doc_id, raw in zip(ids, docs, strict=True):                            # :198
        text = _extract(raw)                                                    # :200
        if text is None or not text.strip():
            discarded["extract"] += 1; continue                                # :202  charge FIRST-drop
        if config.language_classifier is not None and not filters.classifier_keep(...):
            discarded["language"] += 1; continue                               # :208
        if config.gopher and not filters.gopher_quality_filter(text):
            discarded["gopher"] += 1; continue                                 # :212
        # ... nsfw :218 · toxic :223 ...
        if config.mask_pii:                                                     # :226  TRANSFORM, no continue
            text, n = filters.mask_emails(text); ...                           # :227
        if config.quality_classifier is not None and not filters.classifier_keep(...):
            discarded["quality"] += 1; continue                                # :237
        alive.append(DocResult(doc_id, text))                                  # :239
    if config.dedup and alive:                                                  # :242  DEDUP LAST
        alive, n_dropped = _dedup_survivors(alive, config.minhash, work_dir)
        discarded["dedup"] = n_dropped                                          # :244
```
`FilterReport.discard_table()` (:118) in bảng per-stage `in/discarded/kept/discard%` + dòng TOTAL + dòng PII;
được log ở INFO (:249) — audit trail là *deliverable*, không phải debug aid.

**Hình ảnh — funnel + accounting** (số đo THẬT dưới, 7 doc):
```
in │ stage      │ drop │ kept │  ── mỗi doc charge cho stage ĐẦU TIÊN drop nó
 7 │ extract    │  1   │  6   │     doc00001 = "" (empty) ─┐
 6 │ language   │  1   │  5   │     doc00002 = "Bonjour…" (fr) ─┐   FIRST-drop
 5 │ gopher     │  1   │  4   │     doc00003 = "too short" (<50w) │   ⇒ mỗi doc
 4 │ nsfw       │  0   │  4   │                                   │   đúng 1 lần
 4 │ toxic      │  1   │  3   │     doc00004 = "…SLUR…"            │
 3 │ quality    │  0   │  3   │                                   │
 3 │ dedup      │  1   │  2   │     doc00006 == doc00005 (dup) ───┘
───┴────────────┴──────┴──────┴──  TOTAL: 7 = 2 kept + 5 dropped ✓ (identity holds)
```

**Số đo THẬT** (`filter_data`, fake lang/toxic classifiers để chạy full STAGE_ORDER; 7 docs):
```
[MEASURED] total_in=7  discarded={extract:1, language:1, gopher:1, nsfw:0, toxic:1, quality:0, dedup:1}
[MEASURED] kept=2  survivors=[doc00000, doc00005]   identity total_in==kept+Σdiscarded → True
[MEASURED] discard_table: extract 14.3% → language 16.7% → gopher 20.0% → toxic 25.0% → dedup 33.3% → TOTAL 71.4%
[illustrative] cost-order c_i/(1−p_i) (c_i là chi phí GIẢ ĐỊNH — code KHÔNG có cost model; số minh hoạ nguyên lý, không đo từ run):  language 1.67  <  gopher 4.00  <  quality 71.43   ⇒ đúng thứ tự STAGE_ORDER
```
(Bất biến này được pin cứng ở `tests/test_data_filters.py::test_pipeline_accounting_identity_dedup_last_and_discard_table_logged` :290 và `…canonical_order…` :235.)

**Frontier / cổng.** Đây đúng là "data-ablation loop" một senior data RE 2026 sống trong đó
(`FRONTIER_PRACTICE_2026.md` §data-curation): pull vài CC shard → extract→filter→quality→dedup→decontam →
train proxy model nhỏ (~0.5–2B, FineWeb 1.7B rig) ở iso-compute → eval trên suite đóng băng → so win-rate.
"Explicit per-filter discard accounting" được liệt nguyên văn là dấu hiệu "Good looks like". DCLM (Li 2024,
**2406.11794**): filtering recipe > raw token count. Gate = "bạn chạy corpus filter theo thứ tự nào và vì
sao dedup đi cuối?" — cheap/destructive trước, transform-không-drop cho PII, dedup trên survivors; "cho tôi
xem discard accounting". Trait = **spec-with-falsifiers** (bất biến kế toán) + **claims honesty** (audit).

---

## 7.2 · Heuristic quality filter (Gopher/C4) + PII — cheap, transparent, destructive

**Câu hỏi.** Trước khi có model, làm sao vứt trang "rõ ràng vô dụng" (nav-menu, keyword-spam, table dump)
bằng luật *minh bạch, rẻ, không cần training* — và vì sao chúng chạy SỚM?

**Sự thật nền tảng.** Heuristic filter = "người gác cổng mù chữ nhưng có thước": không *hiểu* trang, chỉ đo
đại lượng bề mặt. Rae et al. 2021 (Gopher, App. A) quan sát: text người-viết-tử-tế rơi vào một *dải hẹp*
của các thống kê này; ra ngoài dải gần như luôn là rác máy sinh. Áp lực: nó chạy trên **MỌI** doc nên phải
CỰC RẺ, và nó **phá-hủy** (drop thẳng) nên đặt SỚM để classifier đắt (7.3) không bao giờ thấy trang tệ hiển nhiên.

**Dẫn xuất — bốn dao mổ.** `gopher_quality_filter` = **AND của bốn check**, mỗi cái một *giả thuyết về text
người*, mỗi cái giết một failure-mode CỤ THỂ:
```
1. 50 ≤ n_words ≤ 100_000        quá ngắn = fragment/nav-menu; quá dài = dump/concat
2. 3.0 ≤ mean_word_length ≤ 10.0  <3 = gibberish/token rác;  >10 = URL-soup/base64
3. ≤ 30% dòng kết thúc "…"/"..."   nhiều ellipsis = search-result snippet list (truncated)
4. ≥ 80% từ có ≥1 alpha char       nhiều token thuần số = table/log dump
```
Mỗi ngưỡng đơn giản đến mức whiteboard được — đó CHÍNH LÀ giá trị: minh bạch, tái lập, không cần data để train.

**Sự thật (PII).** PII mask là *transform*, KHÔNG drop — nên nó không theo cost-order, nằm giữa harmful và
quality (7.1). Ba regex `subn` ra `(new_text, n_masked)`, **idempotent**: token thay thế `|||EMAIL_ADDRESS|||`
không chứa `@` nên mask lại = no-op. Digit lookaround `(?<!\d)…(?!\d)` chặn match bên trong digit-run dài
(order id, timestamp). IP octet `(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)` giới hạn 0–255. *Chi phí có ghi nhận:*
version-string `1.2.3.4` cũng khớp IP-regex → bị mask (documented, accepted).

**Sự thật (một shape cho mọi classifier).** `classifier_keep((label, score), keep_label, threshold)` = luật
ngưỡng DÙNG CHUNG cho MỌI classifier filter (language/nsfw/toxic/quality): `keep iff label == keep_label AND
score ≥ threshold`. Verdict "tốt nhưng không chắc" (score thấp) bị coi là **DROP** — corpus-building nghiêng
về vứt. Internalize shape `(label, score) → threshold` một lần, bốn filter thành một.

**Neo code** (`src/scratch_llm/data/filters.py`):
```python
def gopher_quality_filter(text) -> bool:                       # :233
    words = text.split(); n_words = len(words)
    if not 50 <= n_words <= 100_000: return False              # :246  check 1
    mean_len = sum(len(w) for w in words) / n_words
    if not 3.0 <= mean_len <= 10.0: return False               # :250  check 2
    lines = text.splitlines()
    if lines:
        n_ellipsis = sum(1 for l in lines if l.rstrip().endswith(("...","…")))
        if n_ellipsis / len(lines) > 0.30: return False        # :256  check 3
    n_alpha = sum(1 for w in words if any(c.isalpha() for c in w))
    return n_alpha / n_words >= 0.80                           # :260  check 4
def classifier_keep(classification, keep_label, threshold) -> bool:  # :126
    label, score = classification
    return label == keep_label and score >= threshold          # :131  uncertain "good" → drop
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")     # :198
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")  # :202
def mask_emails(text): return _EMAIL_RE.subn("|||EMAIL_ADDRESS|||", text)  # :209  (new_text, n)
```
Extraction (`extract_text_from_html_bytes` :139): resiliparse `extract_plain_text`, UTF-8 strict trước, rồi
`detect_encoding` + `errors="replace"` fallback (CC bytes không đáng tin UTF-8). `fasttext_classify` (:110)
gọi thẳng pybind `model.f.predict` (wrapper Python vỡ dưới numpy≥2), remap code (`eng→en`, `zh-*→zh`, :55).

**Hình ảnh — bốn dao mổ + PII transform:**
```
              ┌───────────── gopher_quality_filter (AND) ─────────────┐
  text ─────► │ [50,100k] words · [3,10] mean_len · ≤30% "…" · ≥80% α │ ─► keep? bool
              └───────────────────────────────────────────────────────┘
   "hi there"(2w)→F    "a b c "×60 (mean_len=1)→F    URL-soup(mean_len>10)→F    "123456"×N(α=0)→F

  PII (transform, count nhưng KHÔNG drop):
   "…a@b.com…c.d@e.org…" ──mask_emails──► "…|||EMAIL_ADDRESS|||…|||EMAIL_ADDRESS|||…", n=2
                          ──re-mask─────► (…, n=0)   ◄── idempotent: |||…||| không có '@'
```

**Số đo THẬT** (`gopher_quality_filter`, `mask_*`, `classifier_keep` — pure regex/python, không cần model):
```
[MEASURED] gopher: good(104w)=True · "hi there"(2w)=False · "a b c"×60(mean_len<3)=False
[MEASURED]         URL-soup(mean_len>10)=False · ellipsis 10/12 lines=False · numeric α<80%=False
[MEASURED] mask_emails 2 addr → ('…|||EMAIL_ADDRESS|||…|||EMAIL_ADDRESS|||…', 2);  re-mask → (…, 0)  idempotent
[MEASURED] mask_phone "(415) 555-1234"→1 · trong "12345678901234"→0 (digit-run lookaround chặn)
[MEASURED] mask_ips "192.168.0.1"→1 · "999.1.1.1"→0 (octet>255) · "1.2.3.4"→1 (version-string, cost có ghi)
[MEASURED] classifier_keep: (en,0.9|thr0.65)=True · (en,0.5)=False (uncertain→drop) · (fr,0.99)=False (sai label)
```
(Bốn Gopher check có test biên riêng `tests/test_data_filters.py:98–124`; extraction moby fixture khớp
official adapter *byte-for-byte* `test_extract_text_matches_official_moby_fixture` :157 — **[test-invariant]**,
cần resiliparse nên không đo trên card này.)

**Frontier / cổng.** Gopher/C4 vẫn là tầng-1 mọi pipeline 2026 (FineWeb dùng biến thể), NHƯNG 2026 chuyển
trọng tâm sang model-based quality (7.3): heuristic có TRẦN — chỉ bắt "rõ ràng rác", không phân biệt "đọc
được nhưng vô bổ" vs "đọc được và hữu ích". Gate = "cho tôi ba heuristic filter và chính xác trang nào mỗi
cái giết?" + "vì sao heuristic chạy trước model filter?" (rẻ + phá hủy → giảm tải classifier đắt). Trait =
**first-principles** (mỗi ngưỡng ↔ một failure-mode) + **subtract-before-add** (regex trước neural).

---

## 7.3 · Learned quality classifier — signal design là deliverable (DCLM)

**Câu hỏi.** "Quality" KHÔNG có ground truth — vậy làm sao *định nghĩa* nó vận hành được, và vì sao một
classifier học được đánh bại mọi heuristic (7.2)?

**Sự thật nền tảng.** Heuristic hỏi "trang có *hình dạng* text tốt không?". Classifier hỏi sâu hơn: "trang
*giống* nguồn tôi tin không?". Vấn đề cốt lõi: **không ai có nhãn "quality"**. Câu trả lời KHÔNG phải "cố
định nghĩa quality tốt hơn" — mà là **từ chối định nghĩa nó trực tiếp**, thay bằng một *proxy nguồn tin cậy*.

**Dẫn xuất — label source IS the recipe.** Chọn:
- **Positive** = text từ trang được nguồn-tin-cậy link tới. CS336 A4: URL trong Wikipedia external
  references (trò "linked-by-Reddit-karma" của GPT-2/WebText nhưng đổi trust anchor). Label canonical `wiki`.
- **Negative** = **random CommonCrawl** (label `cc`) — RÚT TỪ CÙNG phân phối filter sẽ chạy. *Điểm chí mạng:*
  negative từ nơi khác dạy model **domain gap** thay vì **quality gap** — classifier học "wiki vs reddit"
  thay vì "tốt vs rác", vô dụng.

Train một classifier RẺ (`fastText supervised`, không neural ranker — phải chạy được trên MỌI doc sống sót)
phân biệt hai tập → nó **amortize phán xét về nguồn** lên toàn corpus với giá fastText. **Insight load-bearing:
chọn positive set CHÍNH LÀ quyết định modeling.** Đổi nguồn positive = định nghĩa corpus khác; không lượng
capacity nào của model undo được điều đó.

Vì sao classifier > heuristic? Heuristic là hàm CỐ ĐỊNH của thống kê bề mặt — không thể học "register hữu
ích". Classifier học một *decision boundary* trong không gian n-gram/từ tách "giống-nguồn-tin" khỏi "web-thô",
bắt tín hiệu heuristic mù (từ vựng học thuật, cấu trúc lập luận, mật độ thông tin).

**Núm = score threshold trên label `wiki`**, qua đúng interface `classifier_keep` chung. Predict-before-run
(ADR-0016 §Justification): nâng threshold → drop nhiều hơn → mean quality corpus TĂNG, size GIẢM → precision↑,
recall-của-keep↓. Đó chính là **invariant monotone**: keep-fraction **đơn điệu không-tăng** theo threshold.

**Neo code** (`src/scratch_llm/data/quality.py`):
```python
POSITIVE_LABEL = "wiki"   # :33   NEGATIVE_LABEL = "cc"  # :34   (label source = định nghĩa quality)
class QualityClassifier:
    @classmethod
    def train(cls, pos_texts, neg_texts, *, dim=64, epoch=10, lr=0.5,       # :51  signal-design step
              word_ngrams=2, min_count=1, seed=0):
        # pos_texts CARRY the entire definition of "quality"; neg từ CÙNG dist (random CC)
        model = fasttext.train_supervised(input=..., dim=dim, epoch=epoch, lr=lr,
                    wordNgrams=word_ngrams, minCount=min_count, thread=1, seed=seed)  # :85  reproducible
    def classify(self, text): return fasttext_classify(self._model, text)   # :116  (label, score)
def classify_quality(text, model=None):                                     # :141
    if model is None:
        model = _default_classifier(str(models_dir()/DEFAULT_QUALITY_MODEL_FILENAME))  # :150
        # raises FileNotFoundError + lệnh train nếu vắng (:132) — KHÔNG bundle model,
        # shipping default sẽ GIẤU quyết định signal-design mà ADR-0016 tồn tại để ghi lại
    return model.classify(text)
```

**Hình ảnh — signal design + threshold dial:**
```
  trusted source (Wikipedia refs) ──► pos texts "__label__wiki …"  ┐
                                                                    ├─► fastText supervised (dim=64, ngram=2)
  random CommonCrawl (SAME dist)  ──► neg texts "__label__cc …"    ┘        │
                                                                            ▼   classify(doc) → (label, score)
   threshold sweep (núm precision/recall):     ┌──────────────────────────────┐
     thr↑  ─────────────────────────────────►  │ keep_fraction ĐƠN ĐIỆU KHÔNG-TĂNG │
     0.0    0.5    0.75   0.9    1.0            │  0.60 → 0.50 → 0.20 → 0.10 → 0.0  │
     corpus lớn+bẩn ──────► nhỏ+sạch            └──────────────────────────────┘
```

**Số đo THẬT** (fastText KHÔNG cài trên card ⇒ đo *cơ chế núm* bằng synthetic scored corpus + `classifier_keep`,
đúng cái mà invariant test pin):
```
[MEASURED] threshold → keep_frac (label=='wiki', N=10): thr0.0→0.60  0.3→0.60  0.5→0.50  0.6→0.40
[MEASURED]                                              thr0.75→0.20  0.9→0.10  1.0→0.00   → monotone non-increasing: True
[test-invariant] tests/test_data_quality.py::test_threshold_keep_fraction_monotone_non_increasing :94 (trên real fastText output)
[HONESTY/FOP-4] model thật quality_wiki_cc.bin CHƯA train (ADR-0016 §Consequences: machinery xong + test synthetic; run thật data-gated)
[cited, KHÔNG đo] DCLM 2406.11794: fastText(OpenHermes-2.5+r/ELI5 vs web) = 30.2 Core > perplexity 29.0 > SemDeDup 27.1 > PageRank 26.1; +6.6pt MMLU ở 40% ÍT compute
```

**Frontier / cổng.** DCLM là biên lai chính xác: positive dạng **instruction/QA-formatted** (OpenHermes +
r/ELI5) bias keeper-set về register *hữu ích downstream*, không chỉ "trông giống Wikipedia". Rung kế
(FineWeb-Edu, Penedo 2024, **2406.17557**): thay URL-proxy label bằng **LLM-as-judge rubric 0–5**
(Llama-3-70B chấm 460k trang) rồi *distill* vào linear head trên frozen embedding — pattern "label→distill"
là kỹ năng thật 2026. Gate = "định nghĩa 'high quality' cho pretraining data khi không có label thế nào?" —
*bạn không*; bạn chọn nguồn positive tin cậy, train classifier rẻ vs random negative, sweep threshold.
**Label source IS the recipe.** Trait = **research-as-MDP** (chọn positive = pull high-variance node) +
**claims honesty** (không giả vờ đã đo recipe). Đây là **scarce-2026 differentiator** (data curation).

---

## 7.4 · Dedup — exact-hash → MinHash/LSH, derive the S-curve

**Câu hỏi.** Làm sao tìm cặp document *gần trùng* trong N triệu doc mà không quét `O(N²)` cặp — và derive
được xác suất một cặp similarity `s` được LSH đề xuất?

**Sự thật nền tảng.** Web đầy bản sao: cùng bài báo trên 50 site, cùng license trong 10k repo, cùng template
đổi vài từ. Train nhiều lần trên cùng nội dung = memorize + phí compute (Lee 2021: dedup makes LMs better).
Hai bài toán: **exact dup** (byte-hệt) dễ — hash rồi so. **Near-dup** (khác whitespace/case/vài từ) KHÓ —
không hash trực tiếp được. Áp lực trung tâm: **`O(N²)` cặp là bất khả thi** ở web-scale.

**Dẫn xuất — ba tầng.**

**(1) Exact line dedup** — 2-pass streaming, bounded memory. Pass 1 đếm `BLAKE2b` 16-byte digest của mỗi
line (hash chứ không raw → O(#distinct lines) memory). Pass 2 rewrite, giữ line iff count corpus-wide == 1.
Bất biến: line sống sót ở đâu đó thì xuất hiện ĐÚNG một lần toàn corpus — dup drop **CẢ HAI** bản (chủ đích
cho boilerplate).

**(2) MinHash — ước lượng Jaccard O(k) space.** Định nghĩa near-dup qua **Jaccard của tập n-gram** của
*normalized* text. Identity trung tâm: với hàm hash `h_i` random,
```
P[ min_{x∈A} h_i(x)  ==  min_{y∈B} h_i(y) ]  =  |A∩B| / |A∪B|  =  Jaccard(A, B)
```
*Vì sao:* `h_i` cảm sinh một thứ tự ngẫu nhiên gần-đều trên `A∪B`; min của union rơi vào MỘT phần tử; hai
set có min bằng nhau **iff** phần tử đạt min-của-union nằm trong *intersection* — xác suất đúng bằng `|A∩B|/
|A∪B|`. Nên `sig[i] = min_x h_i(x)` cho `k` hàm → *fraction hàng bằng nhau giữa hai signature là unbiased
estimate của Jaccard*, chỉ **O(k) space** thay vì lưu cả set. Hàm hash: `(a·x + b) mod (2⁶¹−1)` (Mersenne
prime; Python int không overflow ⇒ exact, không dùng `hash()` salted → determinism).

**(3) LSH banding — tránh O(N²).** Chia `k` hàng signature thành `b` band × `r` hàng (`k = b·r`). Hai doc
vào cùng bucket nếu *trùng khít cả r-tuple của ≥1 band*. Derive xác suất candidate cho cặp similarity `s`:
```
P[1 band khớp]       = s^r          (r hàng độc lập, mỗi hàng khớp prob s)
P[1 band KHÔNG khớp] = 1 − s^r
P[cả b band trượt]   = (1 − s^r)^b
P[≥1 band khớp]      = 1 − (1 − s^r)^b     ← S-CURVE, knee gần s ≈ (1/b)^(1/r)
```
Đây là công thức Neo. Chọn `(b, r)` để **knee rơi ngay Jaccard threshold**: ADR-0015 chốt `k=100, b=10,
r=10` → knee `(1/10)^(1/10) ≈ 0.794` ≈ threshold 0.8. Quy tắc: giữ `r=10`, scale `b` — tăng band ở fixed r
chỉ *thêm* recall (monotone in b).

**(4) Confirm + cluster.** LSH candidate CHƯA phải dup: tính **true** Jaccard trên full n-gram set, giữ nếu
`≥ threshold`. Rồi union-find gộp transitive (A~B, B~C ⇒ {A,B,C} một cluster), `union` gắn root lớn dưới
root nhỏ ⇒ **root = min input index** ⇒ survivor = "first in input order" rơi ra tự nhiên. Slogan:
**LSH proposes, Jaccard disposes, union-find drops.**

**Neo code** (`src/scratch_llm/data/dedup.py`):
```python
def minhash_signature(items, num_hashes, seed=0):                      # :165
    base = [_stable_hash64(item) for item in set(items)]               # :177  BLAKE2b-64 base hash
    if not base: raise ValueError("cannot MinHash an empty set")       # :179
    return [min((a*x+b) % _MERSENNE_61 for x in base)                  # :180-182  sig[i]=min h_i
            for a, b in _hash_family(num_hashes, seed)]
def lsh_candidates(signatures, num_bands):                             # :185
    rows = k // num_bands                                              # :201
    for doc_idx, sig in enumerate(signatures):
        for band in range(num_bands):
            key = (band, tuple(sig[band*rows:(band+1)*rows]))          # :206  band r-tuple bucket
            buckets[key].append(doc_idx)                              # within-bucket pair = candidate
def lsh_collision_probability(similarity, num_bands, rows_per_band):   # :226
    return 1.0 - (1.0 - similarity**rows_per_band) ** num_bands        # :232  the S-curve
def minhash_dedup(paths, num_hashes, num_bands, ngrams, jaccard_threshold, out_dir, *, seed=0):  # :274
    hashable = [i for i, g in enumerate(doc_ngrams) if g]              # :314  chỉ doc có evidence vào index
    for local_i, local_j in lsh_candidates(signatures, num_bands):
        i, j = hashable[local_i], hashable[local_j]                   # :319  remap local→global (test :297 pin)
        if jaccard(doc_ngrams[i], doc_ngrams[j]) >= jaccard_threshold: # :320  true-Jaccard CONFIRM
            uf.union(i, j)
    ... uf.find(idx) != idx → skip (survivor = cluster root = min input idx :325)
```
`DEFAULT_MINHASH_PARAMS` (:52) = `{num_hashes:100, num_bands:10, ngrams:5, jaccard_threshold:0.8}`.
`normalize_text` (:117): NFD → lowercase → bỏ combining marks (`Mn/Mc/Me`) + punctuation (`P*`) → collapse
whitespace — nó *ĐỊNH NGHĨA* near-dup (khác case/accent/dấu câu → Jaccard 1.0).

**Hình ảnh — MinHash identity + S-curve + banding:**
```
  A = {ngrams}   B = {ngrams}           k=100 signature, b=10 bands × r=10 rows
  sig_A: [ 7 3 9 | 2 5 1 | ... ]        band0  band1
  sig_B: [ 7 3 9 | 8 5 4 | ... ]        band0 khớp (7,3,9)=(7,3,9) → cùng bucket → CANDIDATE
          ↑match  ↑không                 (candidate ≠ dup: confirm bằng true Jaccard ≥ 0.8)

  fraction hàng khớp ≈ Jaccard(A,B)          P=1−(1−s^r)^b, (b=10,r=10):
                                              s:  0.5   0.7   0.8   0.9   0.95
  S-curve:   P│        ___----             P:  0.01  0.25  0.68  0.99  1.00
             1┤     __/                         └knee 0.794 = threshold 0.8┘
             0┤__--/      s→
              └─0.5──0.8──1.0
```
Numeric micro-trace (transitive cluster, unigram): a=tok0..19, b=tok5..24, c=tok10..29 →
`J(a,b)=15/25=0.6`, `J(b,c)=0.6`, `J(a,c)=10/30≈0.33 < 0.5`. Threshold 0.5 ⇒ union {a,b}, {b,c} ⇒ transitive
{a,b,c} một cluster; d riêng ⇒ survivors = **[a, d]** (a = min index của cluster).

**Số đo THẬT** (pure python, seed=0):
```
[MEASURED] S-curve P=1−(1−s^r)^b @(b=10,r=10):  s=0.5→0.0097  0.7→0.2491  0.8→0.6789  0.85→0.8884  0.9→0.9863  0.95→0.9999
[MEASURED] knee (1/10)^(1/10) = 0.7943  ≈ threshold 0.8  |  aggressive (b=50): knee 0.6762, P(0.8)=0.9966 (5× signature cost)
[MEASURED] MinHash identity (k=2000): trueJ=0.5→match_frac 0.4870 (|diff|0.0130 < 5σ 0.0559); trueJ=0.2→0.2060 (|diff|0.0060 < 0.0447)
[MEASURED] normalize_text("Héllo,  WORLD!\n\tfoo—bar") = "hello world foobar"; variants case/accent/punct → identical
[MEASURED] transitive (thr=0.5, high-recall bands e.g. b=50 r=2 để LSH propose cặp J≈0.6; default b=10 P(0.6)=0.059 sẽ KHÔNG bắt): J(a,b)=0.60 J(b,c)=0.60 J(a,c)=0.333 → minhash_dedup survivors = [a.txt, d.txt] (keep FIRST)
[MEASURED] exact_line_dedup: a.txt → "unique to A\n"; within-file dup "repeat me"×2 drops BOTH copies
[ADR-0015] pre-registered table match ✓ (0.010/0.249/0.679/0.986/0.9999 — chênh lệch = làm tròn ADR)
```

**Frontier / cổng.** MinHash 5-gram @0.8 là chuẩn công nghiệp, NHƯNG 2026 học hai điều: (i) **dedup SCOPE** —
FineWeb (2406.17557) thấy per-snapshot MinHash (5-gram, 0.75) đánh bại global cross-dump dedup: global xóa
*quá nhiều* + phân phối survivor *tệ hơn* (upsample data cũ/kém). Dedup KHÔNG monotonically tốt — scope là
quyết định phải log. (ii) **Semantic dedup** (SemDeDup): hai doc là dup ngữ nghĩa (rephrase/back-translation)
với Jaccard THẤP → n-gram dedup provably MISS; embed + cluster + cosine bắt được, xóa ~50% web-scale mất mát
tối thiểu — rung thứ ba của thang (exact → fuzzy/MinHash → semantic). Gate = "walk me through MinHash+LSH và
cách set band count" — signature = per-hash-fn min trên n-gram set; `(b,r)` split là núm precision/recall
(nhiều band → knee trái → recall cao, nhiều false candidate phải confirm); chọn `(b,r)` để knee sit ở Jaccard
threshold, quote `1−(1−s^r)^b`. Trait = **roofline-first / predict-the-number** (predict P trước, đo sau).

---

## 7.5 · Decontamination (A0 gate) + token shards — đóng data path

> Mục này trace hai file còn lại (`decontaminate.py`, `shards.py`): **cổng toàn vẹn cuối** (eval không lọt
> vào train) + **serialization** (corpus sạch → memmap shard mà `train.py::get_batch` ăn). Chúng dùng lại
> đúng primitive n-gram của 7.4 và đóng vòng data → substrate cho mọi rung downstream.

**Câu hỏi.** (a) Làm sao *chứng minh* pretraining corpus không chứa text eval (nếu không, mọi ablation number
vô nghĩa)? (b) Vì sao serialize corpus thành `uint16` memmap với `<|eot|>` separator?

**Sự thật nền tảng (a) — n=13.** Nếu eval item lọt vào train, model memorize nó → eval number thổi phồng,
ablation nói dối. Áp lực: cần một gate bắt **verbatim leak** mà bỏ qua **overlap ngẫu nhiên tự nhiên**. GPT-3
App. C / nanochat / Llama-3 practice: **n=13 exact-collision n-gram**. Vì sao 13? 13 từ liên tiếp đủ DÀI để
match ngẫu nhiên trong natural text là *vanishingly rare*, đủ NGẮN để một eval item leak verbatim gần chắc
chứa ít nhất một. Threshold `0.0` ⇒ MỘT 13-gram trùng là drop; paraphrase (đổi vài từ, phá mọi 13-window) qua.

**Dẫn xuất (a) — symmetric normalization.** Gate **im lặng fail** nếu hai phía normalize khác nhau. Một
normalization DÙNG CHUNG cả hai phía: lowercase → mọi non-alphanumeric thành space (`[^\w]|_`) → whitespace-
split (collapse runs). Robust với case/punct/whitespace giữa eval file và bản crawl. Eval item ngắn hơn `n`
từ đóng góp FULL word-tuple làm một guard entry (chỉ guard doc ngắn identical, không substring của doc dài —
narrow có chủ đích). **Kill check** (surfaced, không auto-fail): nếu >`SUSPICIOUS_OVERLAP_RATE`=20% một slice
bị flag → eval có lẽ leak wholesale → set `suspicious` + log warning để human điều tra (đừng im lặng vứt 20%).

**Sự thật nền tảng (b) — shards.** Format headerless (nanoGPT/nanochat): raw native-little-endian token id,
`<|eot|>` id append sau MỖI document, dtype `uint16` khi mọi id < 2¹⁶ — với **kill-switch sang uint32** ngay
khi id vượt (no silent wraparound). Vì sao EOT sau MỖI doc (không phải GIỮA)? Nó cho model một boundary rõ +
làm concatenated shard seamless — không có nó, window sample xuyên hai doc không liên quan dạy **spurious
long-range dependency**. Metadata ở sidecar `.meta.json` để `.bin` đúng `itemsize·n_tokens` bytes (memmap-able).

**Dẫn xuất (b) — doc_filter seam.** A0 gate nối vào shard build qua `doc_filter` (keep-predicate) áp TRƯỚC
tokenize (và trong `build_dataset` áp MỘT LẦN trước cả BPE training — eval text không được shape merges).

**Neo code**:
```python
# decontaminate.py
DEFAULT_NGRAM_N = 13                       # :60  GPT-3/nanochat collision length
SUSPICIOUS_OVERLAP_RATE = 0.20             # :64  kill check (surfaced, không auto-fail)
_NON_WORD = re.compile(r"[^\w]|_")         # :70
def _normalize_words(text):                # :73  ONE normalization, cả hai phía (symmetry load-bearing)
    return _NON_WORD.sub(" ", text.lower()).split()
def build_eval_ngrams(eval_texts, n=13):   # :79  guard set = mọi 13-gram của mọi eval text
    if len(words) < n: grams.add(tuple(words))            # :93-94  short item → full tuple
    else: grams.update(tuple(words[i:i+n]) for i in ...)  # :96
def ngram_overlap(doc, eval_ngrams, n=13): # :100  fraction 13-window của doc nằm trong guard
    hits = sum(1 for i in range(windows) if tuple(words[i:i+n]) in eval_ngrams); return hits/windows  # :114-115
def decontaminate_docs(docs, eval_ngrams, n=13, threshold=0.0):  # :130  keep iff overlap ≤ threshold
# shards.py
def tokenize_to_shard(docs, tokenizer, eot_id, out_path, *, doc_filter=None):  # :74
    for doc in docs:
        if doc_filter is not None and not doc_filter(doc): n_docs_filtered += 1; continue  # :98  A0 seam
        ids.extend(tokenizer.encode(doc)); ids.append(eot_id)      # :101-102  EOT sau MỖI doc
    dtype = np.uint16 if max_token_id <= _UINT16_MAX else np.uint32 # :114  kill-switch
    arr.astype(dtype).tofile(out_path)                             # :116  headerless
def load_shard(path):                                              # :130
    if actual != itemsize * meta.n_tokens: raise ValueError(...)   # :141  headerless-size invariant
```

**Hình ảnh — A0 gate + shard layout:**
```
  eval sets ──build_eval_ngrams(n=13)──► guard {13-grams}
                                              │  ngram_overlap(doc, guard)
  train doc ──normalize (LOWER, punct→space)──┤  clean: 0.0 → KEEP
                                              │  planted(UPPER+punct): 0.5714 → DROP (verbatim leak caught)
                                              └  paraphrase: 0.0 → KEEP (đổi từ phá 13-window)

  shard_00000.bin (uint16, headerless):   [id id id … <eot> id id … <eot> id … <eot>]
                                            └─ doc0 ──┘      └─ doc1 ─┘     └ doc2 ┘
     bytes = itemsize(2) · n_tokens        <|eot|> count == n_docs      .meta.json sidecar (n_tokens,dtype,eot_id)
```

**Số đo THẬT** (pure python + BPE, không cần model):
```
[MEASURED] DEFAULT_NGRAM_N=13; eval sentence 16 words → 4 guard 13-grams (16−13+1)
[MEASURED] ngram_overlap: clean=0.0 (KEEP) · planted UPPER+punct=0.5714 (DROP) · paraphrase=0.0 (KEEP)
[MEASURED] decontaminate_docs([clean,planted,para]) → kept=2 dropped=1 suspicious=True (1/3=33%>20% kill check fires)
[MEASURED] build_dataset(3 docs, vocab=300) → shard n_tokens=73, dtype=uint16, eot_id=0, max_id=299
[MEASURED] file bytes=146 == itemsize(2)·n_tokens(73) ✓ (headerless);  <|eot|> xuất hiện 3 lần == n_docs ✓
```
(Bất biến pin: `tests/test_decontaminate.py::test_verbatim_planting_dropped_paraphrase_passes` :118,
`test_uppercase_and_whitespace_plantings_still_caught` :185; `tests/test_shards.py::
test_shard_file_is_headerless_uint16` :84, `test_uint32_kill_switch` :94, `test_shard_round_trips_docs_with_eot` :66.)

**Frontier / cổng.** Decontamination reuse đúng primitive n-gram của dedup (7.4) *giữa* train/eval corpora:
Llama-3 dùng 8-gram gate; >4% overlap là **launch blocker**. Shard format = nanoGPT/nanochat headerless memmap
— chuẩn de-facto cho speedrun. Gate = "how do you decontaminate pretraining data against eval sets, và vì sao
n=13?" (đủ dài → accidental match rare, đủ ngắn → verbatim leak chắc chứa một) + "why memmap uint16 shards với
EOT, và gì hỏng nếu train xuyên doc boundary không có nó?" (spurious long-range dependency). Trait =
**execution > analysis** (đóng data→substrate loop) + **claims honesty** (kill check surface, không im lặng vứt).

---

## Bảng số đo THẬT (chạy lại được — `scratchpad/measure_m7.py` — DoD-là-profile)

| Concept | Đo | Kết quả | Ý nghĩa |
|---|---|---|---|
| 7.1 | accounting identity | 7 = 2 kept + 5 discarded → **True** | sổ kế toán kép mọi doc 1 lần |
| 7.1 | discard_table | extract 14.3%→…→dedup 33.3%→TOTAL 71.4% | audit "filter nào định hình corpus" |
| 7.1 | cost-order `c/(1−p)` (c_i minh hoạ) | language **1.67** < gopher **4.00** < quality **71.43** | rẻ+cắt-nhiều đi trước |
| 7.2 | gopher 4 check | good=True; short/gibberish/URL/ellipsis/numeric=**False** | mỗi ngưỡng giết 1 failure-mode |
| 7.2 | mask_emails idempotent | 2 addr → n=2; re-mask → **n=0** | transform, không drop |
| 7.2 | mask_ips octet range | 192.168.0.1→1 · 999.1.1.1→**0** · 1.2.3.4→1 | octet 0–255 (version-string cost) |
| 7.2 | classifier_keep | (en,0.5)→**False** (uncertain→drop) | corpus nghiêng về vứt |
| 7.3 | threshold dial | keep_frac 0.60→0.50→0.20→0.10→**0.00** monotone | núm precision/recall |
| 7.4 | S-curve @(10,10) | 0.5→**0.0097** · 0.8→**0.6789** · 0.9→**0.9863** | knee = threshold |
| 7.4 | knee | (1/10)^(1/10) = **0.7943** ≈ 0.8 | chọn (b,r) đặt knee ở threshold |
| 7.4 | MinHash identity k=2000 | J=0.5→**0.4870** (<5σ) · J=0.2→**0.2060** | fraction hàng khớp ≈ Jaccard |
| 7.4 | transitive cluster | J(a,b)=J(b,c)=0.6 → survivors **[a, d]** | keep FIRST of cluster |
| 7.4 | exact_line_dedup | "repeat me"×2 → drop **BOTH** | corpus-unique lines only |
| 7.5 | decontaminate 13-gram | planted UPPER+punct=**0.5714** DROP · paraphrase=0.0 KEEP | verbatim leak caught |
| 7.5 | shard headerless | bytes 146 == 2·73 · eot count **3 == n_docs** | uint16 memmap + EOT |

---

## Checklist recall COLD (che phần trên, tự trả lời — đây là cách re-learn)

1. **7.1** Derive `c_i/(1−p_i)` order từ exchange argument. Vì sao PII mask *phá* quy tắc đó có chủ đích? Vì sao dedup cuối?
2. **7.1** Viết bất biến accounting. Nếu chuyển dedup lên TRƯỚC quality, identity còn đúng không? Con số nào đổi, vì sao corpus cuối khác?
3. **7.2** Bốn ngưỡng Gopher — mỗi cái giết trang THẬT nào? Nếu bỏ check `≥80% alpha`, loại trang nào lọt vào quality classifier?
4. **7.2** Shape `(label,score)+threshold` — vì sao verdict "tốt nhưng score thấp" bị DROP? PII idempotent nhờ đâu?
5. **7.3** Vì sao "quality" định nghĩa bởi *label source*, không bởi model? Vì sao negative PHẢI từ cùng phân phối filter chạy?
6. **7.3** Nâng threshold → keep-fraction đi hướng nào và vì sao? Đổi positive "Wikipedia-linked"→"instruction/QA" (DCLM) đẩy corpus về register nào?
7. **7.4** Derive `P[sig row match] = Jaccard` từ blank (argument "min-của-union nằm trong intersection"), rồi lên `P = 1−(1−s^r)^b`.
8. **7.4** Muốn recall cao hơn ở s=0.8: tăng `b` hay giảm `r`? Knee dịch hướng nào, giá là gì? Vì sao KHÔNG chỉ hạ jaccard_threshold?
9. **7.4** "LSH proposes, Jaccard disposes, union-find drops" — mỗi mệnh đề làm gì? Vì sao survivor = min input index?
10. **7.5** Vì sao n=13 (không phải 5 hay 50)? Vì sao normalization phải symmetric cả hai phía? Gì hỏng nếu train xuyên doc boundary không có `<|eot|>`?

> Trả lời cold được cả 10 = **M7 thật sự OWNED** (interview-grade). Vấp câu nào → mở đúng mục đó, hoặc
> blank-slate hàm tương ứng (`raise NotImplementedError` → test đỏ → tự dẫn → xanh) để re-own bằng tay
> (`minhash_signature`, `gopher_quality_filter`, `ngram_overlap` là ba ứng viên có teeth nhất).

---

*Cross-ref: `roadmap_model/M7_data_pipeline.md` (bản đồ trace, pin `4ad0ac5`) · `docs/adr/ADR-0015` (dedup
params) · `docs/adr/ADR-0016` (quality signal) · `PROGRESS.md` (ledger 89 Bài) · `CURRICULUM.md` (con đường)
· sibling: `M2_transformer_forward.md` (forward pass). Série trước: M6 scaling (IsoFLOP — *bao nhiêu* token).
Série sau: M8 post-training A5 (corpus sạch → SFT→ExpertIteration→GRPO, nơi reward là tín hiệu). Thành thật
(FOP-4): machinery xanh CI trên fixture/synthetic; run full-CommonCrawl + train quality_wiki_cc.bin thật vẫn
rental/data-gated.*
