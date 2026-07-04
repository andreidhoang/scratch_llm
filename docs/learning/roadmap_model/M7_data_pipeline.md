# SÉRIE 7 — THE DATA PIPELINE, derived (CS336 A4)

> **Số dòng pin theo commit `4ad0ac5`.** Roadmap = bản đồ dẫn xuất + trace để dẫn đường cho teach-back
> sâu về sau, KHÔNG phải bài giảng đầy đủ mỗi mục. Mỗi Bài ~một màn hình, tự chứa, *derivation-first*.
> Code: `src/scratch_llm/data/{pipeline,filters,quality,dedup}.py`. ADR: `docs/adr/ADR-0015` (dedup
> params), `ADR-0016` (quality signal). Frontier: `docs/FRONTIER_PRACTICE_2026.md` §data-curation.

**Vì sao série này.** Série trước (scaling / IsoFLOP, A3) trả lời *bao nhiêu* token cho một compute
budget — Chinchilla nói D ≈ 20·N. Nhưng ba série model+train (M1–M6) ngầm giả định một điều: **token nào
cũng như token nào**. Sai. Ở *fixed compute* (D cố định), thứ quyết định loss cuối không phải thêm
layer hay tinh chỉnh optimizer nữa — mà là *token nào* bạn cho model ăn. Đây là série "garbage in": một
data recipe tồi làm hỏng mọi thứ M1–M6 dựng lên, và không lượng capacity nào cứu được. Cả série là **một
mũi tên**: từ CommonCrawl thô (HTML rác, spam, trùng lặp, PII) → một corpus mà mỗi token *đáng* được
train. Ta đi theo đúng thứ tự pipeline chạy — extract → filter cheap → classify → dedup — mỗi Bài tựa
lên Bài trước, vì thứ tự đó CHÍNH LÀ nội dung cần derive.

Thứ tự đọc: **7.1 (vì sao data thắng ở fixed compute + thứ tự pipeline + discard accounting) → 7.2
(heuristic filter: Gopher/C4 — cheap, destructive) → 7.3 (learned quality classifier: vì sao classifier
đánh bại heuristic — DCLM) → 7.4 (dedup: exact-hash rồi MinHash/LSH near-dup, derive S-curve).**

> **Trạng thái thành thật (FOP-4).** Toàn bộ *machinery* đã build + xanh CI (50 CPU test trong
> `tests/test_data_{pipeline,filters,quality,dedup}.py`), nhưng **CHƯA có run full-CommonCrawl** — nó
> rental-gated + data-gated. Không có "loss giảm X% nhờ recipe này" nào ở đây là số đo; những số như thế
> chỉ tồn tại trong các paper (DCLM/FineWeb) mà ta *cite*, không *đo*. Cụ thể quality classifier: cỗ máy
> `train_quality_classifier(...).save(...)` xong + test trên synthetic corpora, nhưng model thật
> `quality_wiki_cc.bin` (train trên scraped Wikipedia-reference positives) CHƯA train (ADR-0016
> §Consequences). Mỗi Neo dưới đây là **MEASURED test-invariant** hoặc **ADR pre-registered number** —
> không phải kết quả training.

---

## Bài 7.1 — Vì sao data thắng ở fixed compute: pipeline order + discard accounting (`data/pipeline.py` · `filter_data` :173 · `STAGE_ORDER` :40 · `FilterReport.discard_table` :118)
> **Câu hỏi first-principles:** ở compute budget cố định, vì sao *thứ tự* chạy filter là một quyết định
> engineering (không phải chi tiết vặt), và làm sao *chứng minh* mọi document được kế toán đúng một lần?
> **Neo (invariant · MEASURED):** discard-accounting identity `total_in == kept + Σ discarded[stage]`
> — đo cứng trong `test_pipeline_accounting_identity_...` (:321): 7 docs vào → `discarded =
> {language:1, gopher:1, toxic:1, dedup:1}`, `kept == 3`, survivors `[doc00000, doc00005, doc00006]`
> (dedup giữ FIRST của cặp near-dup, deterministic theo input order).

**1. Feynman — bài toán bằng lời.** Bạn có một compute budget cố định = train trên đúng D token. Bạn
KHÔNG được thêm compute. Câu hỏi duy nhất còn lại: *chọn D token nào từ một biển CommonCrawl rác?*
"Garbage in, garbage out" ở đây không phải khẩu hiệu mà là toán: một token spam/boilerplate/trùng lặp
tiêu tốn đúng bằng FLOP của một token vàng, nhưng dạy model gần như không gì (hoặc dạy sai). Pipeline
này là *dây chuyền lọc quặng*: quặng thô (HTML bytes) vào một đầu, kim loại sạch (text đáng train) ra
đầu kia. Đánh đổi cốt lõi — **thứ tự các trạm lọc**: chạy trạm rẻ+phá-hủy trước (cắt khối lượng lớn với
giá rẻ), classifier đắt sau (chạy trên ít document nhất), và dedup *cuối cùng*.

**2. Dẫn xuất từ đầu (derive).** Thứ tự `STAGE_ORDER` (:40) = `extract → language → gopher → nsfw →
toxic → quality → dedup` không tùy tiện; nó là nghiệm của một bài tối ưu chi phí. Gọi mỗi stage có
chi-phí-per-doc `c_i` và tỉ-lệ-giữ `p_i`. Số doc stage i xử lý = `N·Π_{j<i} p_j`. Tổng chi phí =
`Σ c_i · N·Π_{j<i} p_j`. Để cực tiểu: **sắp stage theo `c_i / (1 − p_i)` tăng dần** — trạm nào *rẻ và
cắt nhiều* (low c, low p → high 1−p) đi trước. Language-ID là rẻ nhất + cắt volume lớn nhất → đầu tiên.
Quality classifier đắt nhất → cuối trong các per-doc stage, để nó chỉ thấy các doc đã sống sót mọi cắt
rẻ. Hai ngoại lệ có lý do riêng, KHÔNG theo cost-order:
- **PII mask nằm GIỮA harmful và quality** (:225–232): nó *transform*, không *drop* → không đổi doc
  count. Đặt nó sau harmful classifier (để classifier thấy trang RAW — mask sẽ giấu signal độc hại) và
  trước quality (để quality + mọi thứ downstream thấy đúng text sẽ được train).
- **Dedup CUỐI** (:242): dedup earlier = phí signature lên các doc mà filter sau vứt đi anyway; và
  (với exact-line dedup) đếm line count *trước filter* sẽ bị nhiễm bởi doc chẳng bao giờ vào corpus.

*Discard accounting* là bất biến audit: mỗi doc bị charge cho stage ĐẦU TIÊN drop nó (`continue` ngay,
:208/213/218…), nên `total_in = kept + Σ discarded`. Đây là "sổ kế toán kép" của data — không có nó bạn
không thể nói *filter nào* đã định hình corpus.

**3. Trace code.** `filter_data` (:173): validate `doc_ids` unique (:191) → khởi tạo `discarded =
dict.fromkeys(STAGE_ORDER, 0)` (:194). Vòng lặp per-doc (:198): `_extract` (:199, Bài 7.2) → nếu rỗng
`discarded["extract"]++; continue` → language `classifier_keep` (:205, Bài 7.3) → `gopher_quality_filter`
(:211, Bài 7.2) → nsfw/toxic (:215/220) → PII mask 3 lần (:226–232, cộng `pii_masked` nhưng KHÔNG
continue) → quality (:234) → sống sót thì `alive.append(DocResult(...))` (:239). Sau vòng lặp: dedup trên
`alive` (:242, `_dedup_survivors` :147 ghi ra tmp dir rồi gọi `minhash_dedup`, Bài 7.4). `FilterReport`
(:102) giữ invariant trong docstring; `discard_table()` (:118) in bảng per-stage `in/discarded/kept/
discard%` + dòng TOTAL + dòng PII. **Test pin:** `test_pipeline_...canonical_order...` xác minh stage
chạy đúng thứ tự + charge first-dropping-stage; `test_pipeline_accounting_identity...` (:290) pin cả
identity lẫn survivor set + rằng bảng được log ở INFO.

**4. Cổng teach-back.** (a) Derive lại vì sao sắp stage theo `c_i/(1−p_i)` — và vì sao PII mask *phá vỡ*
quy tắc đó một cách có chủ đích. (b) *Sửa-và-đoán:* nếu chuyển dedup lên TRƯỚC quality, `total_in == kept
+ Σ discarded` còn đúng không? Con số `discarded` nào đổi, và *tại sao* corpus cuối có thể khác (gợi:
signature phí + doc bị quality-drop vẫn được dùng làm "bản gốc" để drop near-dup của nó).

**5. Frontier.** Đây đúng là "data-ablation loop" mà một senior data RE 2026 sống trong đó
(`FRONTIER_PRACTICE_2026.md` §data-curation): pull vài CC shard → extract→filter→quality→dedup→
decontaminate → train một proxy model nhỏ (~0.5–2B, kiểu FineWeb 1.7B rig) ở iso-compute → eval trên
suite đóng băng → so win-rate. "Explicit per-filter discard accounting" được liệt kê nguyên văn là một
trong các dấu hiệu "Good looks like". DCLM (Li et al. 2024, **2406.11794**) là biên lai quy mô lớn:
recipe lọc đánh bại thêm token — filtering recipe > raw token count. Câu interview: "bạn chạy corpus
filter theo thứ tự nào và vì sao dedup đi cuối?" — cheap/destructive trước, transform-không-drop cho
PII, dedup trên survivors; "cho tôi xem discard accounting" là cách audit mọi data recipe.

---

## Bài 7.2 — Heuristic quality filter: Gopher/C4 structural checks (`data/filters.py` · `gopher_quality_filter` :233 · `classifier_keep` :126 · PII masks :209–225)
> **Câu hỏi first-principles:** trước khi có model, làm sao vứt trang "rõ ràng vô dụng" (nav-menu,
> keyword-spam, list rác) bằng các luật *minh bạch, rẻ, không cần training* — và vì sao chúng chạy SỚM?
> **Neo (invariant · MEASURED):** Gopher là bốn ngưỡng structural cứng, mỗi ngưỡng có test riêng
> (`test_gopher_word_count_bounds` / `_mean_word_length_bounds` / `_ellipsis_line_fraction` /
> `_alpha_word_fraction`, `tests/test_data_filters.py:98–124`); và moby fixture: extraction khớp official
> adapter *byte-for-byte* (`test_extract_text_matches_official_moby_fixture` :157).

**1. Feynman — bài toán bằng lời.** Heuristic filter = "người gác cổng mù chữ nhưng có thước": nó không
*hiểu* trang, chỉ đo các đại lượng bề mặt (đếm từ, độ dài từ trung bình, tỉ lệ dòng kết thúc bằng "…",
tỉ lệ từ có chữ cái) và loại thẳng cái nào lệch. Rae et al. 2021 (Gopher, Appendix A) quan sát: text
người-viết-tử-tế rơi vào một *dải hẹp* của các thống kê này; ra ngoài dải gần như luôn là rác máy sinh
(SEO spam, table dump, lorem ipsum). Đánh đổi: **precision thấp, recall của "rác" cao, giá cực rẻ** — nó
không tinh, nhưng chạy trên MỌI doc nên phải rẻ, và nó phá-hủy (drop thẳng) nên đặt SỚM để classifier
đắt (Bài 7.3) không bao giờ thấy trang tệ hiển nhiên.

**2. Dẫn xuất từ đầu (derive).** `gopher_quality_filter` (:233) = AND của bốn check, mỗi cái là một
*giả thuyết về text người*:
1. `50 ≤ n_words ≤ 100_000` (:246) — quá ngắn = fragment/nav; quá dài = dump/concatenation.
2. `3.0 ≤ mean_word_length ≤ 10.0` chars (:249) — <3 = gibberish/token rác; >10 = URL-soup/mã hoá.
3. `≤ 30%` dòng kết thúc bằng `...`/`…` (:255) — nhiều ellipsis = truncated snippet list (kiểu
   search-result page).
4. `≥ 80%` từ chứa ít nhất một alpha char (:259) — nhiều token thuần số/ký hiệu = table/log dump.
Mỗi ngưỡng là một *dao mổ*: đơn giản đến mức bạn whiteboard được, và mỗi cái diệt một failure-mode cụ
thể. Đây là "C4-style" (Raffel et al.) + Gopher subset. Song song, `classifier_keep` (:126) định nghĩa
*một* luật ngưỡng dùng chung cho MỌI classifier filter: `keep iff label == keep_label AND score ≥
threshold` — verdict "tốt nhưng không chắc" bị coi là DROP (corpus-building nghiêng về vứt). Internalize
shape `(label, score) → threshold` một lần, bốn filter (language/nsfw/toxic/quality) thành một. PII mask
(:209–225) là *transform*: `_EMAIL_RE`/`_PHONE_RE`/`_IP_RE` (:198/202/206) `subn` ra `(new_text,
n_masked)`, idempotent (token `|||EMAIL_ADDRESS|||` không chứa `@` nên mask lại = no-op).

**3. Trace code.** Extraction: `extract_text_from_html_bytes` (:139) — resiliparse `extract_plain_text`,
UTF-8 strict trước, `detect_encoding` + `errors="replace"` fallback (CC bytes không đáng tin UTF-8). Vào
pipeline qua `_extract` (`pipeline.py:141`). Gopher `text.split()` → 4 check tuần tự, fail bất kỳ → False.
`identify_language` (:169) là *instance đầu tiên* của pattern `(label,score)`: fastText lid.176 →
`fasttext_classify` (:110, gọi thẳng pybind `model.f.predict` vì wrapper Python vỡ dưới numpy≥2) →
remap `zh-*/yue/wuu → zh`, `eng → en` (:55) để threshold so đúng một code canonical. **Test pin:** Gopher
4 test biên; PII `test_mask_*` (email nhiều+idempotent, phone ignore digit-run dài, IP octet 0–255);
extraction moby fixture byte-exact.

**4. Cổng teach-back.** (a) Derive lại vì sao mỗi trong bốn ngưỡng Gopher bắt một failure-mode CỤ THỂ —
đặt tên trang thật mỗi check giết. (b) *Sửa-và-đoán:* nếu bỏ check `≥80% alpha` (:259), loại trang nào
bắt đầu lọt qua vào quality classifier, và nó có nguy hiểm hơn (đắt hơn) không so với để Gopher chặn?

**5. Frontier.** Gopher/C4 heuristic vẫn là tầng-1 của mọi pipeline 2026 (FineWeb dùng biến thể), NHƯNG
2026 chuyển trọng tâm sang model-based quality (Bài 7.3) vì heuristic có trần: nó chỉ bắt "rõ ràng rác",
không phân biệt được "đọc được nhưng vô bổ" vs "đọc được và hữu ích". `FRONTIER_PRACTICE_2026.md`
§data-curation liệt kê heuristic là bước đầu bắt buộc rồi mới tới classifier. Câu interview: "cho tôi ba
heuristic filter và chính xác trang nào mỗi cái giết?" — và "vì sao heuristic chạy trước model filter?"
(rẻ + phá hủy → giảm tải cho classifier đắt).

---

## Bài 7.3 — Learned quality classifier: signal design (DCLM) (`data/quality.py` · `QualityClassifier.train` :51 · `classify_quality` :141 · ADR-0016)
> **Câu hỏi first-principles:** "quality" không có ground truth — vậy làm sao *định nghĩa* nó vận hành
> được, và vì sao một classifier học được đánh bại mọi heuristic (Bài 7.2)?
> **Neo (invariant · MEASURED):** keep-fraction **đơn điệu không-tăng** theo score threshold
> (`test_threshold_keep_fraction_monotone_non_increasing`, `tests/test_data_quality.py:94`, trên synthetic
> pos/neg/mixed set) — threshold LÀ cái núm precision/recall. **Thành thật:** model thật (`quality_wiki_cc.bin`)
> CHƯA train (ADR-0016 §Consequences: machinery xong + test synthetic; run thật data-gated).

**1. Feynman — bài toán bằng lời.** Heuristic (Bài 7.2) hỏi "trang này có *hình dạng* của text tốt
không?". Classifier hỏi câu sâu hơn: "trang này *giống* nguồn tôi tin không?". Vấn đề: KHÔNG ai có nhãn
"quality". Câu trả lời không phải "cố gắng định nghĩa quality tốt hơn" — mà là **từ chối định nghĩa nó
trực tiếp**, và thay bằng một *proxy nguồn tin cậy*: chọn một tập positive từ nguồn ta *tin* là chất
lượng (CS336 A4: trang được link từ Wikipedia external references — trò "linked-by-Reddit-karma" của
GPT-2/WebText nhưng đổi trust anchor), negative = random CommonCrawl. Train một classifier rẻ phân biệt
hai tập → nó *amortize phán xét về nguồn* lên toàn corpus với giá fastText. **Insight load-bearing: chọn
positive set CHÍNH LÀ quyết định modeling.** Đổi nguồn positive = định nghĩa một corpus khác; không lượng
capacity nào của model undo được điều đó.

**2. Dẫn xuất từ đầu (derive).** Vì sao classifier > heuristic? Heuristic là hàm cố định của thống kê bề
mặt — nó không thể học "register hữu ích". Classifier học một *decision boundary* trong không gian
n-gram/từ mà tách "giống-nguồn-tin" khỏi "web-thô", bắt được tín hiệu heuristic mù (từ vựng học thuật,
cấu trúc lập luận, mật độ thông tin). ADR-0016 chốt thiết kế:
- **Positive = trusted-source-linked** (label `wiki`, :33); **negative = random CC** (label `cc`, :34) —
  RÚT TỪ CÙNG phân phối filter sẽ chạy. Negative từ nơi khác dạy model *domain gap* thay vì *quality gap*
  (bẫy chí mạng: classifier học "wiki vs reddit" thay vì "tốt vs rác").
- **Model = fastText supervised** (`dim=64, epoch=10, lr=0.5, wordNgrams=2, minCount=1`, `thread=1` +
  seed cố định → reproducible, :64–92), KHÔNG phải neural ranker: nó phải rẻ đủ để chạy trên MỌI doc
  sống sót.
- **Núm = score threshold trên label `wiki`**, qua đúng interface `classifier_keep` chung. Predict-before-
  run (ADR-0016 §Justification): nâng threshold → drop nhiều hơn → mean quality corpus tăng, size giảm
  → precision↑, recall-của-keep↓. Đó chính là invariant monotone đo được ở Neo.
- **KHÔNG bundle model** (:32): `classify_quality` raise với hướng dẫn train khi thiếu file — ship model
  mặc định sẽ *giấu* quyết định signal-design mà ADR này tồn tại để ghi lại.

**3. Trace code.** `QualityClassifier.train` (:51): ghi `__label__{pos} …` + `__label__{neg} …` ra tmp
file (:77–83) → `fasttext.train_supervised` single-thread (:85) → wrap. `.classify` (:116) →
`fasttext_classify` (dùng lại của filters.py) → `(label, score)`. `classify_quality` (:141): dùng model
truyền vào, else `_default_classifier` (:129, lru_cache) load `models_dir()/quality_wiki_cc.bin` —
raise `FileNotFoundError` kèm lệnh train nếu vắng (:132). Trong pipeline: `FilterConfig.with_default_models`
(:74) wire `quality.classify_quality` vào stage quality (:88). **Test pin:** `test_labels_and_separation`
(pos→`wiki`, neg→`cc`), `test_threshold_keep_fraction_monotone_non_increasing` (:94, invariant núm),
`test_train_rejects_empty_corpus`. Trap ghi lại: fasttext 0.9.3 `wordNgrams=1` NaN trên tiny-vocab →
mặc định `=2` (ADR-0016 §Consequences).

**4. Cổng teach-back.** (a) Derive lại vì sao "quality" được định nghĩa bởi *label source*, không bởi
model — và vì sao negative PHẢI từ cùng phân phối filter chạy (nếu không, model học gì?). (b) *Sửa-và-đoán:*
nếu đổi positive từ "Wikipedia-linked" sang "instruction/QA-formatted" (DCLM), corpus giữ lại nghiêng về
register nào, và discard% dịch hướng nào? (predict TRƯỚC khi train — đây đúng là ablation DCLM.)

**5. Frontier.** DCLM (Li et al. 2024, **2406.11794**) là biên lai chính xác của bài này:
`FRONTIER_PRACTICE_2026.md` §DCLM-classifier ghi — fastText trên positives **OpenHermes-2.5 + r/ELI5**
(instruction/QA-formatted) vs random web đạt **30.2 Core** accuracy, đánh bại perplexity-filter 29.0,
SemDeDup 27.1, PageRank 26.1; và **+6.6pt MMLU ở 40% ÍT compute hơn** SOTA open-data trước đó. Cơ chế =
insight load-bearing: positive dạng instruction/answer bias keeper-set về register *hữu ích downstream*,
không chỉ "trông giống Wikipedia". Rung tiếp theo (FineWeb-Edu, Penedo et al. 2024, **2406.17557**):
thay URL-proxy label bằng LLM-as-judge rubric 0–5 (Llama-3-70B chấm 460k trang) rồi *distill* vào một
linear head trên frozen embedding — pattern "label→distill" là kỹ năng thật 2026. Câu interview: "định
nghĩa 'high quality' cho pretraining data khi không có label thế nào?" — bạn không; bạn *chọn nguồn
positive tin cậy*, train classifier rẻ vs random negative, và sweep threshold. Label source IS the recipe.

---

## Bài 7.4 — Dedup: exact-hash → MinHash/LSH near-dup, derive the S-curve (`data/dedup.py` · `minhash_signature` :165 · `lsh_candidates` :185 · `lsh_collision_probability` :226 · ADR-0015)
> **Câu hỏi first-principles:** làm sao tìm cặp document *gần trùng* trong N triệu doc mà không quét
> O(N²) cặp — và derive được xác suất một cặp similarity s được LSH đề xuất?
> **Neo (formula + MEASURED):** LSH S-curve `P[candidate] = 1 − (1 − s^r)^b` (`lsh_collision_probability`
> :226); ADR-0015 pre-registered table: `P(0.5)=0.010, P(0.8)=0.679, P(0.9)=0.986, P(0.95)=0.9999` ở
> `(b=10, r=10)`. MEASURED: (i) match-fraction của signature rows ≈ true Jaccard trong dải 5σ binomial
> (`test_row_match_probability_estimates_jaccard` :125, k=2000, J∈{0.5,0.2}); (ii) recall đơn điệu theo
> band count, `recalls[0]<0.1 → recalls[-1]>0.9` (:164); (iii) transitive cluster giữ ĐÚNG một survivor,
> `[a.txt, d.txt]` (:216); (iv) true-Jaccard confirm bác LSH candidate J=0.6 ở threshold 0.8 (:240).

**1. Feynman — bài toán bằng lời.** Web đầy bản sao: cùng bài báo trên 50 site, cùng license file trong
10k repo, cùng template với vài từ đổi. Train nhiều lần trên cùng nội dung = memorize + phí compute (Lee
et al. 2021: dedup makes LMs better). Hai bài toán khác nhau: **exact dup** (byte-hệt) dễ — hash rồi so.
**Near-dup** (khác whitespace/case/vài từ) khó — không hash trực tiếp được. Ý tưởng: (1) biến "gần giống"
thành một *con số* — Jaccard của tập n-gram; (2) ước lượng Jaccard *rẻ* — MinHash; (3) tìm cặp Jaccard
cao mà không quét O(N²) — LSH banding. Đánh đổi xuyên suốt: **LSH proposes, Jaccard disposes, union-find
drops** — LSH chỉ giảm cặp phải kiểm, KHÔNG quyết định trùng; confirm bằng true Jaccard giữ precision.

**2. Dẫn xuất từ đầu (derive).** Ba tầng, derive từ blank:

*Exact line dedup* (:76): 2-pass streaming. Pass 1 đếm BLAKE2b 16-byte digest của mỗi line (bounded
memory: hash chứ không raw line). Pass 2 rewrite, giữ line iff count corpus-wide == 1. Invariant: line
sống sót ở đâu đó thì xuất hiện đúng một lần toàn corpus (dup drop CẢ HAI bản — đó là chủ đích cho
boilerplate).

*MinHash — ước lượng Jaccard O(k) space.* Định nghĩa near-dup qua tập n-gram của *normalized* text
(`normalize_text` :117: NFD → lowercase → bỏ combining marks + punctuation → collapse whitespace; nên
hai text khác nhau chỉ ở case/accent/dấu câu → Jaccard 1.0). Identity trung tâm (`minhash_signature`
:165): với hàm hash h_i random, `P[min_x∈A h_i(x) == min_x∈B h_i(x)] = |A∩B|/|A∪B| = Jaccard(A,B)`.
*Vì sao:* min của union rơi vào một phần tử; hai set có min bằng nhau iff phần tử đạt min-của-union nằm
trong *intersection* — xác suất đúng bằng Jaccard. Nên `sig[i] = min h_i(item)` cho k hàm → *fraction
hàng bằng nhau giữa hai signature là unbiased estimate của Jaccard*, chỉ O(k) space thay vì lưu cả set.
Hàm hash: `(a·x + b) mod (2^61−1)` (Mersenne prime :45) trên BLAKE2b-64 base hash — Python int không
overflow nên exact, không dùng `hash()` salted (determinism).

*LSH banding — tránh O(N²).* Chia k hàng signature thành `b` band × `r` hàng (`k = b·r`,
`lsh_candidates` :185). Hai doc vào cùng bucket nếu *trùng khít cả r-tuple của ≥1 band*. Derive xác
suất: một band (r hàng) khớp với xác suất `s^r` (r hàng độc lập, mỗi hàng khớp với prob s). Band KHÔNG
khớp: `1 − s^r`. Cả b band đều trượt: `(1 − s^r)^b`. Ít nhất một band khớp (→ candidate):
`P = 1 − (1 − s^r)^b` — **S-curve** với knee gần `s ≈ (1/b)^(1/r)`. Đây là công thức Neo. Chọn `(b, r)`
để knee rơi ngay Jaccard threshold: ADR-0015 chốt `k=100, b=10, r=10` → knee `(1/10)^(1/10) ≈ 0.794` ≈
threshold 0.8. Aggressive (`b=50`): knee → 0.676, recall@0.8 lên 0.9966 (5× signature cost). Quy tắc:
giữ `r=10`, scale `b` — tăng band ở fixed r chỉ *thêm* recall (monotone in b, đo ở test :164).

*Confirm + cluster.* LSH candidate CHƯA phải dup: `minhash_dedup` (:274) tính true `jaccard` (:219) trên
full n-gram set, giữ nếu `≥ threshold` (:320). Rồi `_UnionFind` (:240) gộp transitive (A~B, B~C ⇒
{A,B,C} một cluster) — `union` (:258) gắn root lớn dưới root nhỏ nên **root = min input index** → survivor
= "first in input order" rơi ra tự nhiên từ `find()`. Giữ đúng một doc/cluster (không drop-both).

**3. Trace code.** `minhash_dedup` (:274): validate `b | k` (:296) → đọc `ngram_set` mỗi doc (:311) →
chỉ doc có evidence (n-gram set khác rỗng) vào index (:314) → `minhash_signature` (:315) → `lsh_candidates`
(:318) → mỗi candidate `jaccard ≥ threshold` thì `uf.union` (:320) → survivor = doc mà `uf.find(idx)==idx`
(:325), copy byte-identical (:328). **Test pin (đo cứng):** `test_row_match_probability_estimates_jaccard`
(:125) — match-frac trong 5σ của J (kill nếu hash-family mất unbiasedness); `test_s_curve_recall_monotone
_in_bands_at_fixed_k` (:164); `test_transitive_closure_keeps_exactly_one` (:216) survivors `[a,d]`;
`test_true_jaccard_confirm_rejects_lsh_candidate` (:240) — LSH đề xuất J=0.6 pair (P≈1−1e−10) nhưng
confirm@0.8 bác → cả hai sống (kill một confirm-step union-mọi-candidate). Fixture thật (ADR-0015):
rails/react MIT license (khác whitespace+attribution) → true Jaccard ≈0.92 (merged); pytorch 3-clause →
≈0.007 — threshold 0.8 tách chúng hai bậc độ lớn.

**4. Cổng teach-back.** (a) Derive lại `P[sig row match] = Jaccard` từ blank (argument "min-của-union
nằm trong intersection"), rồi từ đó lên `P = 1 − (1 − s^r)^b`. (b) *Sửa-và-đoán:* muốn recall cao hơn ở
s=0.8 (bắt paraphrase nhẹ hơn) — bạn tăng `b` hay giảm `r`? Knee dịch hướng nào, và cái giá là gì (gợi:
candidate volume → confirm work; ADR-0015 aggressive row). Vì sao KHÔNG chỉ hạ `jaccard_threshold`?

**5. Frontier.** MinHash 5-gram @0.8 là chuẩn công nghiệp, NHƯNG 2026 học hai điều
(`FRONTIER_PRACTICE_2026.md`): (i) **dedup SCOPE** — FineWeb (Penedo et al. 2024, **2406.17557**) thấy
per-snapshot MinHash (5-gram, 0.75) đánh bại global cross-dump dedup: global dedup xóa *quá nhiều* và
phân phối sống sót *tệ hơn* (nó upsample data cũ/kém). Dedup KHÔNG monotonically tốt — scope là quyết
định phải log. (ii) **Semantic dedup** — SemDeDup: hai doc là dup ngữ nghĩa (rephrase, back-translation)
với Jaccard THẤP → n-gram dedup provably miss; embed + cluster + cosine bắt được, xóa ~50% web-scale với
mất mát tối thiểu. Rung thứ ba của thang dedup (exact → fuzzy/MinHash → semantic). Và decontamination
(Lee et al. 2021, **2107.06499**; Llama-3 8-gram gate): dùng lại đúng primitive n-gram này *giữa*
train/eval corpora — >4% overlap là launch blocker. Câu interview: "walk me through MinHash+LSH và cách
set band count" — signature = per-hash-fn min trên n-gram set; `(b,r)` split là núm precision/recall
(nhiều band → knee trái → recall cao, nhiều false candidate phải confirm); chọn `(b,r)` để knee sit ở
Jaccard threshold, quote `1 − (1 − s^r)^b` ở các similarity phải bắt.

---

*Đóng série.* Mũi tên: token nào cũng tốn FLOP như nhau ⇒ ở fixed compute, *chọn token* là đòn bẩy duy
nhất còn lại. Ta dựng dây chuyền lọc quặng — extract → cheap heuristic (Gopher, phá-hủy, sớm) → learned
classifier (quality, đắt, muộn — "label source IS the recipe") → dedup (exact rồi MinHash/LSH S-curve,
cuối cùng, trên survivors) — với discard accounting `total_in == kept + Σ discarded` làm sổ kế toán kép.
Toàn bộ machinery xanh CI trên fixture/synthetic; **run full-CommonCrawl + train model quality thật vẫn
rental/data-gated** (FOP-4). Série sau (M8, post-training A5): corpus sạch này thành SFT→ExpertIteration→
GRPO — nơi reward, không data-quality-classifier, là tín hiệu.
