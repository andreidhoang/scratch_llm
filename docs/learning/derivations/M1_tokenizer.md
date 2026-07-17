# M1 — Byte-level BPE Tokenizer · First-Principles Derivations

> **Đây là gì.** Bản dẫn-xuất **từ nguyên lý gốc** (first principles) cho TOÀN BỘ tầng nền của stack —
> tokenizer byte-level BPE — 4 micro-concept M1 (Bài 1.1→1.4). Mỗi mục: (1) **câu hỏi** falsifiable, (2)
> **sự thật nền tảng** (áp lực vật lý/toán học buộc thiết kế này), (3) **dẫn xuất** có toán, (4) **neo code**
> `file·func·line`, (5) **hình ảnh** (ASCII + byte-layout/shape thật), (6) **số đo THẬT** (chạy trên chính
> repo này, không phán), (7) **frontier framing** + cổng phỏng vấn.
>
> **Cách dùng để re-learn.** Đọc phần *Câu hỏi* của mỗi mục → **tự trả lời cold** (che phần dưới) → mở ra
> đối chiếu. Cuối doc có **checklist recall cold** + bảng số đo. Đây là bạn đồng hành của
> `roadmap_model/M1_tokenizer.md` (reference chung) — doc này là *derivation lab* CÓ SỐ ĐO.
>
> **Cảnh báo neo.** Line number **trôi** theo commit. Cite ở đây pin theo HEAD `1e7dbc6` (2026-07-14,
> `src/scratch_llm/tokenizer.py`, 302 dòng). Roadmap `roadmap_model/M1_tokenizer.md` pin commit cũ `4ad0ac5`
> nên **số dòng của nó đã lệch** (vd nó ghi `_bpe :219`; HEAD thật là `_bpe :246`) — luật repo: `grep` tên
> hàm, đừng tin số dòng cứng.
>
> **Nguồn số đo.** Mọi số trong doc này chạy lại được bằng một script `python` gọi thẳng `train_bpe` /
> `_compute_merges` / `Tokenizer` từ `src/scratch_llm/tokenizer.py` (editable-installed). Đó là "DoD là một
> profile, không phải test xanh" (FOP-3) áp cho việc học.

---

## Bức tranh lớn — tokenizer là hàm gì, và nó chốt cái gì cho cả stack?

Tokenizer là **cặp hàm nghịch đảo** `encode: str → list[int]` và `decode: list[int] → str`, cùng một
**thủ tục train** dựng ra bảng tra cứu (`vocab`, `merges`) từ corpus. Nó là *sàn nhà* của mọi thứ phía sau:

```
                        ┌──────────────── TRAIN (một lần, offline) ─────────────────┐
   corpus.txt (str) ──► _pretokenize_counts ──► _compute_merges ──► (vocab, merges)  │  [1.2 · 1.3]
        │                  │ regex GPT-2 chẻ       │ greedy argmax                    │
        │                  │ special = ranh giới   │ + incremental                    │
        └──────────────────┴───────────────────────┴──────────────────────────────────┘
                                          │
                                          ▼   chốt V = vocab_size  ◄── NÚM NÉN-vs-PHỦ  [1.1]
                       ┌──────────────────────────────────────────────────┐
   "the cat 🌌"  ──►   │  encode: split_special → GPT-2 regex → _bpe(rank) │  ──► [258, 293, 262, 141]
   (str, UTF-8)        │  decode: b"".join(vocab[id]).decode("utf-8")      │  ◄──   (list[int])
                       └──────────────────────────────────────────────────┘        [1.4 round-trip = identity]
                                          │
                                          ▼
        V định nghĩa:  token_emb (V, d_model) ·  lm_head (d_model, V) ·  cross_entropy đáy = log V
                        ────────────────────────────────────────────────────────────────  [1.1 → M2 · M3]
```

**Một sự thật xuyên suốt M1: tokenizer là một cái NÚM giữa hai áp lực đối nghịch.**

- **Áp lực COMPRESSION (nén)** — chuỗi càng ngắn (token/text ít) → attention O(L²) rẻ hơn, decode ít bước
  hơn, mỗi token đặc thông tin hơn. Đẩy V *lên*.
- **Áp lực COVERAGE + COST (phủ + chi phí)** — V nhỏ thì `lm_head` V×d nhỏ, mỗi token thấy nhiều data hơn,
  và **256 byte là sàn no-OOV**. Đẩy V *xuống*, nhưng KHÔNG được xuống dưới 256.
- **Áp lực DETERMINISM (xác định)** — merge phải replay Y HỆT lúc encode, nếu không model thấy token lạ →
  vô nghĩa. Buộc phải *lưu thứ tự merge* + tie-break xác định.
- **Áp lực BOUNDARY (ranh giới)** — không được merge xuyên qua ranh giới từ (regex prior) hay ranh giới tài
  liệu (special token). Vocab sạch, có cấu trúc.

Học M1 = học *áp lực → lời giải*: 1.1 chốt cái núm V; 1.2 là thuật toán vặn núm (greedy merge); 1.3 là hàng
rào giữ vocab sạch; 1.4 là hợp đồng round-trip + no-OOV mà mọi thứ tin cậy.

**Con số load-bearing đầu tiên của cả stack là V.** Nó set chiều rộng logit, FLOP/token khi decode, và
*điểm gốc của mọi loss* (`log V`). Đó là lý do M1 đứng trước M2 — bạn không thể dựng `token_emb` hay đo
`loss@init` trước khi tokenizer chốt V.

---

## 1.1 · Vì sao subword — V là núm nén-vs-phủ, loss@init = log V

**Câu hỏi.** Tại sao KHÔNG dùng thẳng byte (`V=256`, phủ mọi text) hay thẳng word (`V=`cả từ điển, chuỗi
ngắn)? Chọn `V` ở giữa để tối ưu CÁI GÌ — và tại sao `V` phải được chốt TRƯỚC khi có model?

**Sự thật nền tảng.** Model là một hàm chọn 1-trong-`V` ở mỗi bước. Hai cực đều hỏng: byte thuần cho chuỗi
**quá dài** (attention O(L²) nổ, thông tin loãng); word thuần cho **V nổ** (hàng triệu) + **OOV** (từ lạ/typo/
tên riêng không có id). Subword là điểm cân bằng — nhưng cân bằng đó là một *đường cong đánh đổi*, không phải
một điểm ma thuật.

**Dẫn xuất.** Đặt bài toán như **mã hoá tối thiểu**: cho corpus, chọn tập ký hiệu `S` (`|S| ≤ V`) để tối
thiểu tổng số token khi mã hoá corpus. Ký hiệu đáng thêm = chuỗi con **xuất hiện nhiều** (thêm nó rút ngắn
được nhiều chỗ). BPE (Bài 1.2) là lời giải *greedy* của bài toán này. Ba cực trên trục `V`:

```
 V=256   (byte thuần)      V ~ 10³–10⁵  (BPE subword)        V = |từ điển| (word thuần)
 ├───────────────────────────────┼──────────────────────────────────────┤
 phủ 100%, KHÔNG OOV             núm chỉnh: từ hay gặp→1 token           chuỗi ngắn nhất,
 chuỗi DÀI nhất (~1 B/tok)       (nén); từ lạ→mảnh→byte (phủ)            V nổ + OOV (chết)
 lm_head bé                      lm_head vừa                             lm_head khổng lồ
```

**Loss@init = log V — dẫn từ đâu.** Một LM *tươi* (chưa học, weight nhỏ ngẫu nhiên) trên input bất kỳ phải
đoán **đều** trên `V` lựa chọn ⇒ `p(token) = 1/V` mỗi lớp ⇒ cross-entropy:
```
CE_init = −log p(token đúng) = −log(1/V) = log V
```
Đây là **oracle rẻ nhất của cả stack** (CLAUDE.md §Engineering disciplines #1): model tươi mà CE **lệch xa**
`log V` ⇒ có bug ở head / embedding / mask / vocab. Và nó giải thích một cái bẫy: `V=256` cho `log 256 ≈ 5.55`
nats, `V=65536` cho `log 2¹⁶ ≈ 11.09` nats — loss "cao hơn" của vocab lớn **KHÔNG** nghĩa model tệ hơn, chỉ là
**trục rộng hơn**. Muốn so công bằng qua hai tokenizer → chuẩn hoá về **bits-per-byte** (`val_bpb`), không phải
per-token loss. Vì `log V` là điểm gốc, `V` phải chốt TRƯỚC model — nó *đặt gốc toạ độ* cho mọi loss về sau.

**Vì sao 256 là sàn không thể bỏ.** Mọi text là chuỗi byte UTF-8; giữ đủ 256 byte đơn ⇒ ký tự lạ nhất
(emoji chưa từng thấy) vẫn phân rã được về byte → **no-OOV bởi cấu trúc** (Bài 1.4). Nên `train_bpe` từ chối
`V < len(special) + 256`.

**Neo code** (`src/scratch_llm/tokenizer.py`):
```python
def train_bpe(input_path, vocab_size, special_tokens=None):     # :130
    base_size = len(special_tokens) + 256                        # :142  sàn = special + 256 byte
    if vocab_size < base_size:                                   # :143  V < sàn ⇒ chết (no-OOV cần 256)
        raise ValueError(f"vocab_size={vocab_size} too small: need >= {base_size} ...")
    ...
    merges = _compute_merges(dict(word_freqs), vocab_size - base_size)  # :160  #merge = V − base = núm nén
```
`vocab_size − base_size` **chính là cái núm**: mỗi merge biến một cặp byte hay-gặp thành 1 token, rút ngắn
chuỗi tương lai. `V` lớn = nhiều merge = nén mạnh hơn.

**Hình ảnh — layout id trong `vocab`** (`base_size = 1 special + 256 byte`):
```
 id:   0            1        2      ...    256          257      258   ...
      ┌──────────┐ ┌──────┐ ┌──────┐      ┌────────┐   ┌──────┐ ┌──────┐
      │<|eot|>   │ │\x00  │ │\x01  │ ...  │\xff    │   │merge0│ │merge1│ ...
      └──────────┘ └──────┘ └──────┘      └────────┘   └──────┘ └──────┘
       special      256 byte (sàn no-OOV, id 1..256)     V−base merge (núm nén)
```

**Số đo THẬT.**
```
log V (điểm gốc loss, nats):  V=256→5.5452  V=400→5.9915  V=1000→6.9078  V=50257→10.8249  V=65536→11.0904

Nén (cùng câu 70 byte, train trên corpus varied English, V tăng):
  V= 257 (  0 merge):  70 token   1.00 byte/token   ← byte thuần, chuỗi dài nhất
  V= 307 ( 50 merge):  47 token   1.49 byte/token
  V= 407 (150 merge):  31 token   2.26 byte/token
  V= 657 (282 merge):  13 token   5.38 byte/token   ← nén 5.4× so với byte thuần
  (V>657 vẫn 282 merge/13 token: corpus tí HON đã BÃO HOÀ — núm bị chặn bởi entropy corpus, honest ceiling)

loss@init = log V, đo bằng torch CE trên logits ĐỀU (uniform):
  V=256  → CE(uniform)=5.5452  log V=5.5452  diff=1.45e-06
  V=1000 → CE(uniform)=6.9078  log V=6.9078  diff=1.81e-06     ⇒ CE_init = log V (khớp tới 1e-6)

base_size reject:  train_bpe(V=100, 1 special) → ValueError "too small: need >= 257"
```
> **Ghi chú honesty (FOP-4).** Con số nén bão hoà ở 282 merge là *artifact của corpus tí hon*, không phải
> tính chất của BPE — trên corpus lớn, đường cong nén tiếp tục đi lên tới hàng chục nghìn merge. Nêu ra để
> không overclaim. Số `loss@init` full-model thật (7.04 ≈ log 1000) đo ở M2 §2.6c; entropy≈log256 lúc init
> đo ở `bench/RESULTS.md:280` (GRPO smoke) — M1 *chốt* V, M2/M3 *hiện thực hoá* log V.

**Frontier / cổng.** nanochat/GPT-4 dùng Rust BPE `V=65536=2¹⁶` (power-of-two: căn thẳng tensor dim, id vừa
`uint16`). Của ta toy `V=400–512` trên TinyStories — **giống cơ chế, khác scale** (ADR-0001: BPE from-scratch
= artifact mastery; pipeline RL thật dùng tokenizer native của model pretrained để hai engine index CÙNG V).
Gate = "vì sao `vocab_size` là *modeling decision* không phải hyperparameter tuỳ tiện — nó chạm trục nào?"
(logit width · FLOP/token · loss origin · OOV). Trait = first-principles + cheapest-oracle discipline.

---

## 1.2 · Thuật toán merge BPE — count-pair → greedy argmax → incremental

**Câu hỏi.** Cho corpus đã chẻ thành từ, quy tắc *tham lam* nào xây được `V−base` ký hiệu subword? Và làm sao
chạy nhanh mà **không đếm lại cả corpus mỗi vòng**? Vì sao tie-break phải *xác định*?

**Sự thật nền tảng.** BPE = "nén bằng cách dán cặp kề nhau hay gặp nhất, lặp lại". Trạng thái: mỗi *word* là
tuple các symbol (khởi đầu = từng byte). Mỗi vòng: tìm cặp kề nhau tần suất cao nhất trong TOÀN corpus, dán
thành 1 symbol mới, ghi lại merge, lặp. Sau `K` vòng có `K` merge **theo thứ tự** — và thứ tự đó là
load-bearing (Bài 1.4 phải replay đúng).

**Dẫn xuất — objective + greedy.** Đại lượng cần: `count(pair) = Σ freq(word) với word chứa pair kề nhau`.
Vòng lặp greedy:
```
best = argmax_pair count(pair)          # cặp hay gặp nhất
merged = best[0] + best[1]              # symbol mới (nối bytes)
# tie: nhiều cặp cùng count → chọn cặp LỚN HƠN theo thứ tự từ điển (byte so lexicographically)
```
Greedy **không** tối ưu toàn cục (một chuỗi merge khác có thể cho tổng token ít hơn), nhưng đổi lấy **xác định
+ rẻ** — và quan trọng hơn: kết quả *tái lập được*. Tie-break "lexicographically greater" là điều kiện để
**REPRODUCIBLE**: khi nhiều cặp đồng tần, một quy tắc xác định (không phụ thuộc thứ tự dict/hash) đảm bảo hai
lần train cho cùng merges.

**Mẹo incremental (cốt lõi hiệu năng).** Naive = O(vòng × corpus): mỗi vòng quét lại toàn bộ để đếm cặp. Thay
vào đó giữ **`pair_to_words[pair]` = tập chỉ số word chứa pair**. Khi merge `best`, CHỈ quét lại những word
trong `pair_to_words[best]`. Với mỗi word đổi, tính **delta** cục bộ:
```
delta(p) = (count p trong word MỚI − count p trong word CŨ) × freq(word)
pair_counts[p] += delta(p)
```
Chi phí scale theo *số word bị chạm*, không theo corpus. Đây là điểm mấu chốt biến BPE từ "chậm không dùng
được" thành "train trong giây trên corpus vừa".

**Vì sao delta = new − old là ĐỦ (không sót cặp).** Merge chỉ đổi các word chứa `best`. Trong một word đó,
mọi cặp *không chạm* vị trí merge giữ nguyên (`new_count = old_count` ⇒ delta 0). Chỉ các cặp *tại/kề* chỗ
merge đổi. Vì ta duyệt `old_pairs.keys() | new_pairs.keys()` (hợp của cả hai), mọi cặp sinh ra HOẶC mất đi
đều được cập nhật. Word không chứa `best` không đổi ⇒ không cần quét. QED — kết quả identical với đếm-lại-toàn-bộ.

**Neo code** (`_compute_merges` :76, `_merge_word` :61):
```python
# initial scan: dựng pair_counts + pair_to_words (:89–93)
for i, word in enumerate(words):
    for a, b in zip(word, word[1:], strict=False):
        pair_counts[(a, b)] += freqs[i]
        pair_to_words[(a, b)].add(i)

for _ in range(num_merges):                                       # :96
    if not pair_counts: break                                     # :97  hết cặp ⇒ bão hoà (xem 1.1)
    best = max(pair_counts, key=lambda p: (pair_counts[p], p))    # :101  (count, THEN pair) ⇒ tie→pair lớn hơn
    merged = best[0] + best[1]                                    # :102
    merges.append(best)                                           # :103

    for i in list(pair_to_words[best]):                           # :105  CHỈ word chứa best
        new = _merge_word(old, best, merged)                      # :108  thay non-overlapping trái→phải
        old_pairs = Counter(zip(old, old[1:], strict=False))      # :111
        new_pairs = Counter(zip(new, new[1:], strict=False))      # :112
        for p in old_pairs.keys() | new_pairs.keys():             # :113  hợp ⇒ không sót cặp
            delta = (new_pairs[p] - old_pairs[p]) * f              # :114  incremental
            if delta: pair_counts[p] += delta                     # :116
            # ... bảo trì pair_to_words + dọn cặp về 0 (:117–122)
    pair_counts.pop(best, None); pair_to_words.pop(best, None)    # :124–125
```
`_merge_word` (:61) thay **non-overlapping trái→phải**: khớp cặp thì `i += 2`, không thì `i += 1` — nên trên
`"aaa"` với cặp `(a,a)` cho `[aa, a]` (không chồng lấn), đúng chuẩn BPE.

**Hình ảnh — hai vòng đầu trên `bpe_example`** (`{low:5, lower:2, widest:3, newest:6}`):
```
 word (byte tuple)      freq   round-0 pair_counts (count, pair):
  l o w                  5       (s,t)=9  ← widest×3 + newest×6
  l o w e r              2       (e,s)=9        tie 9=9 → chọn (s,t)? KHÔNG:
  w i d e s t            3       (w,e)=8        max theo (count, pair): (s,t) vs (e,s),
  n e w e s t            6       (o,w)=7        's'>'e' ⇒ 'st'>'es' ⇒ argmax=(s,t) ✓
                                              ─────────────────────────────────────────
 merge0 = (s,t)→'st'.  CHỈ widest,newest chứa (s,t) → quét lại 2 word đó:
  w i d e st  (mất (s,t),(e,s); thêm (e,st),(st, ·))    delta cập nhật cục bộ
  n e w e st                                             pair_to_words[(s,t)] pop
 → round-1 argmax = (e,st)=9  → merge1 = (e,st)→'est'   ... → [st,est,ow,low,west,ne]
```

**Số đo THẬT.**
```
reference bpe_example (num_merges=6):
  → [(s,t), (e,st), (o,w), (l,ow), (w,est), (n,e)]     == spec CS336/Sennrich ? TRUE
  (test_compute_merges_reproduces_bpe_example — oracle bắt mọi lỗi đếm/tie-break)

round-0 pair_counts (top-4):  (s,t)=9  (e,s)=9  (w,e)=8  (o,w)=7
  argmax = (s,t)  (tie 9=9 với (e,s), 'st' > 'es' lexicographically ⇒ chọn (s,t))

tie-break thuần (4 cặp đều freq 1: A-B, A-C, B-ZZ, BA-A):
  → [(BA, A)]     (chọn cặp LỚN NHẤT: 'BA' > 'B' > 'A')   == spec ? TRUE
```

**Frontier / cổng.** Đúng thuật toán Sennrich 2016 (BPE gốc cho NMT), lõi của tiktoken (GPT-2/3/4) + Rust BPE
nanochat. Khác biệt frontier = *tốc độ train* không phải thuật toán: tiktoken/nanochat viết vòng này bằng
**Rust** + song song hoá pretokenization để train `V=2¹⁶` trên hàng chục GB trong phút; của ta Python
incremental — đúng nhưng chậm (kill-criterion A1: nếu train TinyStories vỡ ngân sách thời gian → `multiprocessing`
pretokenization TRƯỚC khi làm gì fancy). Gate = "BPE train là greedy trên objective gì, và vì sao thứ tự merge
phải lưu chứ không chỉ tập token?" Trait = first-principles + reproducibility discipline.

> **Code review (teaching-as-review).** `_compute_merges` là O(#merge × #word-chạm × len-word) — đúng và
> gọn. Điểm tinh tế đã xử đúng: `pair_counts.pop(p)` khi về ≤0 (:121) tránh rác Counter phình. Một *cải tiến*
> khả dĩ: thay `list(pair_to_words[best])` copy set mỗi vòng bằng iterate trực tiếp — nhưng cần copy vì set bị
> mutate trong vòng, nên copy là ĐÚNG (không phải bug). Không tìm thấy lỗi correctness ở đây.

---

## 1.3 · Pretokenization regex GPT-2/4 + special tokens

**Câu hỏi.** Vì sao phải CHẺ text bằng regex `\p{L}/\p{N}/\p{P}` **trước** khi merge, thay vì thả BPE lên cả
dòng thô? Và special token đứng ở đâu trong luồng — vì sao nó KHÔNG được đếm vào BPE?

**Sự thật nền tảng.** Thả BPE lên câu thô ⇒ nó học merge **xuyên qua** khoảng trắng/dấu câu: `"dog."` và
`"dog"` thành token khác nhau, `"the "` dính từ sau tuỳ ngữ cảnh → vocab **bẩn, phân mảnh vô lý**.
Pretokenization là **hàng rào**: chẻ text thành pre-token (mảnh từ/số/dấu/space) và **cấm merge vượt ranh giới
pre-token**. BPE chỉ chạy TRONG mỗi pre-token. Đây là một *prior của con người* áp lên BPE (BPE tự nó không
biết "từ") — đổi tính "học thuần data" lấy vocab sạch, có cấu trúc.

**Dẫn xuất — đọc từng nhánh pattern GPT-2.** Pattern (:31–33) là một **alternation** (thử trái→phải, nhánh đầu
khớp thắng):
```
'(?:[sdmt]|ll|ve|re)   → contraction tiếng Anh:  's 'll 've 're 'd 'm 't  (một token riêng)
 ?\p{L}+               → một cụm CHỮ, space-dẫn-đầu TUỲ CHỌN dính vào từ sau  ⇒ " the" = MỘT pre-token
 ?\p{N}+               → một cụm SỐ
 ?[^\s\p{L}\p{N}]+     → một cụm dấu câu/ký hiệu (không phải space/chữ/số)
\s+(?!\S)|\s+          → khoảng trắng đuôi (trailing) / khoảng trắng còn lại
```
**Vì sao space-dẫn-đầu-dính-từ-sau (`" ?\p{L}+"`)?** Nếu tách space thành token riêng, mỗi từ tốn 2 token
(space + từ). Dính space vào từ sau ⇒ `" the"` là 1 pre-token: model biết **ranh giới từ** (space đứng trước =
đầu từ mới) mà KHÔNG phí token space riêng → nén tốt hơn + tín hiệu ranh giới rõ. Cần module `regex` (không
phải `re` stdlib) vì `\p{L}/\p{N}` là **Unicode property class** (`re` không có).

**Special token là chuyện KHÁC.** Nó là **ranh giới tài liệu** (`<|endoftext|>`), không phải pre-token. Phải:
(a) chẻ TRƯỚC, (b) **longest-first** (để `<|eot|><|eot|>` khớp trước `<|eot|>`), (c) **KHÔNG đếm vào BPE** (nó
có id riêng cấp ở Bài 1.1) — để không merge nào dính hai tài liệu. Đối xứng train↔encode nhưng **ngược mục
đích**: train *bỏ* special (không capture group → delimiter rớt khỏi chunks); encode *giữ* special (có capture
group → giữ làm segment riêng).

**Neo code** (`_pretokenize_counts` :41, `_GPT2_PAT` :31):
```python
delimiter = "|".join(regex.escape(s) for s in sorted(special_tokens, key=len, reverse=True))  # :48 longest-first
chunks = regex.split(delimiter, text)          # :49  KHÔNG capture group ⇒ special BỊ BỎ khỏi chunks
for chunk in chunks:
    for match in _GPT2_PAT.finditer(chunk):    # :55  BPE regex chỉ chạy TRONG chunk (giữa hai special)
        token_bytes = match.group().encode("utf-8")            # :56
        counts[tuple(bytes([b]) for b in token_bytes)] += 1    # :57  pre-token → tuple từng-byte
```
Phía encode, `_split_on_specials` (:288) dùng CÙNG regex nhưng **CÓ** capture group `(...)` (:292–296) để GIỮ
special làm segment riêng — đối xứng nhưng ngược.

**Hình ảnh — data flow qua pretokenization:**
```
 " the cat123 sat!!  世界"
        │ _GPT2_PAT.finditer  (mỗi match = 1 pre-token)
        ▼
 [' the'] [' cat'] ['123'] [' sat'] ['!!'] [' '] [' 世界']
   ▲space  ▲space  ▲số      ▲space  ▲dấu   ▲sp   ▲space+CJK
   dính từ         nguyên khối (\p{N}+)    câu          (mỗi match .encode utf-8 → tuple byte)
        │
        ▼  BPE (1.2) chỉ merge TRONG mỗi ô, KHÔNG xuyên ô

 "ab<|endoftext|>cd" ── split special (longest-first) ──► ["ab", "cd"]  (delimiter rớt)
                        ⇒ không word nào chứa '<'; special có id riêng, không vào BPE
```

**Số đo THẬT.**
```
GPT-2 pretokens của " the cat123 sat!!  世界":
  [' the', ' cat', '123', ' sat', '!!', ' ', ' 世界']
  ⇒ ' the' giữ space-dẫn-đầu là MỘT pre-token; '123' nguyên khối; dấu '!!' tách khỏi chữ

digit-grouping (GPT-2 vs GPT-4/Llama-3) trên "12345":
  \p{N}+     → ['12345']        (GPT-2: cả con số thành 1 pre-token → số kỳ dị)
  \p{N}{1,3} → ['123', '45']    (GPT-4/Llama-3: nhóm ≤3 chữ số → arithmetic tốt hơn)

special boundary — _pretokenize_counts("ab<|endoftext|>cd", ["<|endoftext|>"]):
  keys = ['ab', 'cd']          không key nào chứa '<' ? TRUE   (special không lọt vào đếm BPE)
```

**Frontier / cổng.** Đây là điểm phân kỳ GPT-2 vs GPT-4/Llama-3 rõ nhất và là *modeling decision* có niên đại:
cl100k (GPT-4) và Llama-3 dùng `\p{N}{1,3}` — chẻ số thành nhóm ≤3 để BPE không nuốt cả con số thành token kỳ
dị (số học tệ). Của ta hiện `\p{N}+` (GPT-2, `tokenizer.py:32`) ⇒ digit-grouping là **một dòng sửa** + retrain
tiny BPE (`FRONTIER_PRACTICE_2026 §digit-grouping`). Gate = "pretokenization giải bug gì mà BPE thuần không
thấy, và vì sao digit-grouping là default 2026?" (merge xuyên ranh giới → vocab bẩn; số kỳ dị → arithmetic
kém). Trait = frontier-referenced + know-the-dated-default.

---

## 1.4 · Encode/decode round-trip + byte-fallback (no OOV)

**Câu hỏi.** Encode phải replay merge thế nào để KHỚP train? Bất biến `decode(encode(s)) == s` giữ được nhờ
đâu? Vì sao **KHÔNG BAO GIỜ OOV** kể cả với emoji/chữ lạ chưa từng thấy?

**Sự thật nền tảng.** Encode = "phát lại lịch sử nén": lấy pre-token (bytes), áp các merge đã học *theo đúng
thứ tự train* tới khi hết merge; ra chuỗi id. Decode = ngược tầm thường: nối bytes của từng id rồi giải UTF-8.
Bí mật no-OOV: **sàn của mọi thứ là 256 byte đơn** — mọi id cuối cùng phân rã về byte, và MỌI text UTF-8 là
một chuỗi byte, nên ký tự lạ nhất vẫn mã hoá được thành các byte của nó. BPE chỉ *nén* các byte đó khi có
merge; không merge thì chúng ở dạng byte thô — vẫn hợp lệ. Đó là **byte-fallback**: coverage bởi cấu trúc,
không phải may mắn.

**Dẫn xuất — encode phải chọn rank NHỎ NHẤT.** Train áp merge theo thứ tự (merge0 sớm nhất = tần suất cao
nhất). Để encode cho CÙNG kết quả như thể pre-token có mặt lúc train, ở mỗi bước phải dán cặp có **rank nhỏ
nhất** (merge sớm nhất = ưu tiên cao nhất), KHÔNG phải "cặp trái nhất" hay "cặp dài nhất". `rank[pair] = vị trí
trong merges` (:188). Đây là điểm tinh tế nhất của encode.

**Chứng minh round-trip = identity.** `decode` chỉ nối `vocab[id]`:
- `vocab[id]` cho một merge = `a + b` (nối bytes, `train_bpe :163`); cho một byte = chính byte đó.
- Nối lại toàn bộ id của một pre-token = đúng chuỗi byte gốc của pre-token (merge chỉ *nhóm* byte, không
  thêm/bớt byte nào).
- Ghép mọi pre-token = đúng bytes gốc của text — vì **pretokenization chỉ chẻ, không xoá byte nào** (kể cả
  space, do `" ?\p{L}+"` giữ space).
- ⇒ `b"".join(...).decode("utf-8") == s`. QED.

**Vì sao no-OOV (theo dõi một emoji CHƯA TỪNG THẤY).** `🌌` = UTF-8 `[f0 9f 8c 8c]`. Corpus train chỉ có `🌍` =
`[f0 9f 8c 8d]` (khác byte cuối!). Encode `🌌`: byte đầu gặp merge đã học từ `🌍` (`f0 9f 8c` là prefix chung) →
1 token; byte cuối `8c` không ghép được với gì → **rơi về byte thô** (id byte). Không có nhánh nào "không tìm
thấy id" → **không thể OOV**. `errors="replace"` trong decode (:286) chỉ cứu chuỗi id THỦ CÔNG cắt giữa ký tự
multi-byte — round-trip hợp lệ không bao giờ kích hoạt nó.

**Neo code** (`_bpe` :246, `encode` :269, `decode` :284):
```python
def _bpe(self, token_bytes):                                    # :246
    parts = [bytes([b]) for b in token_bytes]                   # :252  khởi đầu = từng byte (sàn no-OOV)
    while len(parts) >= 2:                                      # :253
        best_rank, best_i = None, -1
        for i in range(len(parts) - 1):
            rank = self._ranks.get((parts[i], parts[i + 1]))    # :257  rank = thứ tự train
            if rank is not None and (best_rank is None or rank < best_rank):  # :258  RANK NHỎ NHẤT
                best_rank, best_i = rank, i
        if best_i < 0: break                                    # :261  không cặp nào có merge ⇒ dừng (byte thô)
        parts[best_i:best_i+2] = [parts[best_i] + parts[best_i+1]]  # :263  dán
    ids = [self._bytes_to_id[p] for p in parts]                 # :265  parts → ids (mọi p CHẮC có id: byte hoặc merge)

def decode(self, ids):                                          # :284
    data = b"".join(self.vocab[i] for i in ids)                 # :285  nối bytes
    return data.decode("utf-8", errors="replace")               # :286  giải UTF-8
```

**Hình ảnh — byte-fallback của emoji chưa thấy `🌌`:**
```
 '🌌'  ── .encode("utf-8") ──►  bytes  [ f0 , 9f , 8c , 8c ]
                                          └──── merge học từ 🌍 (f0 9f 8c) ────┘  └ byte thô
                                 _bpe:    [f0 9f 8c]  = id 262 (merge)          8c = id 141 (byte 0x8c)
                                 ids   =  [262, 141]
 decode: vocab[262] + vocab[141] = b'\xf0\x9f\x8c' + b'\x8c' = b'\xf0\x9f\x8c\x8c' = '🌌'   ✓ round-trip
         │
         └─ id→byte: byte value v → id v+1 (special id 0, byte 0x00→id 1, ... 0x8c=140→id 141)
```

**Số đo THẬT.**
```
round-trip decode(encode(s)) == s cho MỌI UTF-8:
  'the cat sat'      → True     'hello, world!'    → True
  '世界 🌍 emoji'      → True     '  leading spaces' → True     '' → True

byte-fallback — emoji CHƯA THẤY '🌌' (train chỉ có '🌍'):
  UTF-8 bytes = [0xf0, 0x9f, 0x8c, 0x8c]   (4 byte)
  encode → ids [262, 141]   trong đó  262 = b'\xf0\x9f\x8c' (merge 3-byte học từ 🌍),  141 = b'\x8c' (byte thô)
  decode → '🌌'   round-trip → True     ⇒ KHÔNG OOV dù model chưa từng thấy 🌌

special = MỘT id, không bị chẻ:
  encode('hello<|endoftext|>world') → eot_id=0, count(eot)=1, decode == gốc → True
```

**Frontier / cổng.** Byte-fallback + round-trip là hợp đồng chung của tiktoken/nanochat/HF; "sớm-rank thắng"
đúng thuật toán encode tiktoken. Khác biệt implementation: tiktoken/Rust BPE encode bằng **priority queue trên
rank** O(n log n) thay vì quét O(n²) như `_bpe :256` của ta (của ta có `_cache` per-pre-token :248 để
amortize). ADR-0001: chính vì SỞ HỮU round-trip + special stability này mà ta *đọc/patch được token id* và tin
ranh giới special khi A5 chuyển sang tokenizer native — contract khởi động phải assert `vocab_size` + mọi
special id KHỚP giữa train-engine và serve-engine (lệch → abort; so KL per-token trên hai trục vocab khác nhau
là vô nghĩa). Gate = "vì sao byte-level BPE không bao giờ OOV, và round-trip identity CHỨNG minh điều gì — nó
có ĐỦ để nói encode 'đúng như train' không?" (round-trip là điều kiện CẦN: bytes bảo toàn nên decode luôn
đúng; nhưng nếu encode chọn sai cặp, chuỗi ID khác train → model thấy token lạ → nén/loss tệ; round-trip vẫn
pass ⇒ **không đủ** để chứng minh "đúng như train"). Trait = frontier-referenced + invariant-first.

> **Code review (teaching-as-review).** Hai điểm đáng ghi: (1) `decode` dùng `errors="replace"` (:286) ⇒
> **nuốt lỗi im lặng** khi id list cắt giữa ký tự multi-byte (trả `U+FFFD` thay vì raise). Với round-trip hợp
> lệ không kích hoạt, nhưng với id THỦ CÔNG thì che bug — hơi lệch fail-fast (FOP/PRINCIPLES §Error Handling).
> Đánh đổi có chủ đích: decode phải luôn trả str (dùng cho sampling/serving), raise giữa vòng generate sẽ tệ
> hơn. (2) `_bpe` O(n²) là hot path encode — với pre-token dài (URL, chuỗi base64) chi phí phình; priority
> queue là nâng cấp thực. Cả hai KHÔNG phải bug correctness — round-trip test có teeth và pass.

---

## Bảng số đo THẬT (chạy lại được — DoD-là-profile)

| Concept | Đo | Kết quả | Ý nghĩa |
|---|---|---|---|
| 1.1 | log V (điểm gốc loss) | 256→5.5452 · 1000→6.9078 · 65536→11.0904 nats | loss@init = log V |
| 1.1 | nén (70-byte, V tăng) | 257→70tok(1.0) · 407→31(2.26) · 657→13(**5.38 B/tok**) | V là núm nén |
| 1.1 | CE(uniform) vs log V | diff ≤ **1.8e-06** (V=256,1000,50257) | oracle loss@init |
| 1.1 | base_size reject | V=100 → **ValueError** "need >= 257" | 256-byte floor no-OOV |
| 1.2 | reference bpe_example | **[st,est,ow,low,west,ne]** == spec | merge core đúng |
| 1.2 | round-0 argmax | (s,t)=9 thắng (e,s)=9 vì 'st'>'es' | greedy + tie-break |
| 1.2 | tie-break thuần | 4 cặp freq 1 → **(BA, A)** | lexicographically greater |
| 1.3 | GPT-2 pretokens | `' the'` là 1 pre-token (giữ space) | boundary prior |
| 1.3 | digit-grouping | "12345": `\p{N}+`→['12345'] vs `\p{N}{1,3}`→['123','45'] | GPT-2 vs GPT-4 default |
| 1.3 | special boundary | keys=['ab','cd'], no '<' | special không vào BPE |
| 1.4 | round-trip UTF-8 | '世界 🌍 emoji' + 4 khác → **True** | identity |
| 1.4 | byte-fallback | '🌌' chưa thấy → [262,141] → **round-trip True** | no-OOV |
| 1.4 | special single id | count(eot)=**1**, decode ok | ranh giới tài liệu |

---

## Checklist recall COLD (che phần trên, tự trả lời — đây là cách re-learn)

1. **1.1** Vì sao KHÔNG dùng byte thuần (V=256) hay word thuần? `V` là núm giữa hai áp lực nào?
2. **1.1** Dẫn `loss@init = log V`. Vì sao model V=256 và V=50k có loss@init khác nhau KHÔNG nói lên model nào
   tốt hơn — chuẩn hoá thế nào để so công bằng?
3. **1.1** Vì sao 256 là sàn không thể bỏ? `train_bpe` từ chối `V < ?`
4. **1.2** BPE greedy trên objective gì? Tie-break chọn cặp nào và VÌ SAO (điều kiện gì được đảm bảo)?
5. **1.2** Mẹo incremental: vì sao chỉ quét-lại-word-bị-chạm cho ĐÚNG `pair_counts` như đếm-lại-toàn-bộ?
   `delta = new − old` vì sao đủ, không sót cặp?
6. **1.3** Vì sao phải pretokenize TRƯỚC khi merge — bug gì nếu thả BPE lên câu thô? `" ?\p{L}+"` tiết kiệm gì?
7. **1.3** Special token: chẻ trước hay sau? Longest-first vì sao? Vì sao KHÔNG đếm vào BPE?
8. **1.3** `\p{N}+` vs `\p{N}{1,3}` trên "12345" ra gì? Vì sao digit-grouping là default 2026?
9. **1.4** Encode chọn cặp theo tiêu chí gì (rank nhỏ nhất / trái nhất / dài nhất)? Vì sao đúng cái đó?
10. **1.4** Theo dõi '🌌' (chưa thấy) qua encode→decode: nó dừng ở đâu (byte thô)? Vì sao no-OOV? Round-trip
    identity có ĐỦ chứng minh encode "đúng như train" không?

> Trả lời cold được cả 10 = **M1 thật sự OWNED** (interview-grade). Vấp câu nào → mở đúng mục đó, hoặc
> blank-slate hàm tương ứng (`_compute_merges` hoặc `_bpe` → `raise NotImplementedError` → test đỏ
> `test_compute_merges_reproduces_bpe_example` / `test_encode_decode_round_trip_ascii_and_unicode` → tự dẫn → xanh) để re-own
> bằng tay (proves tests có teeth).

---

*Cross-ref: `roadmap_model/M1_tokenizer.md` (reference chung, pin commit cũ — số dòng đã trôi) ·
`docs/learning/PROGRESS.md` (ledger 89 Bài) · `docs/learning/CURRICULUM.md` (con đường) ·
`derivations/M2_transformer_forward.md` (concept kế: V → token_emb → forward pass) · test oracle
`tests/test_tokenizer.py`. Concept kế: M2 — Transformer forward (V chốt ở đây → chiều rộng logit + loss origin ở đó).*
