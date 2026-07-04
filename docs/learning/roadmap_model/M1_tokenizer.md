# SÉRIE 1 (MODEL) — TOKENIZER: byte → token

> **Số dòng pin theo commit `4ad0ac5`.** Đây là roadmap *derivation-first*: mỗi Bài DẪN XUẤT component
> từ gốc (bài toán → toán từ tờ giấy trắng → vì sao thiết kế NÀY), rồi trace xuống file·hàm·dòng thật, rồi
> nối tới frontier. TWIN của roadmap performance (`docs/learning/roadmap/` — serving+kernels); phía này là
> chính cái MODEL. Scope série: `src/scratch_llm/tokenizer.py` (đọc trọn 276 dòng). Test pin bất biến:
> `tests/test_tokenizer.py`. Ngôn ngữ Việt; thuật ngữ (BPE, pretokenization, byte-fallback, tiktoken…) giữ Anh.

**Vì sao série này — tầng nền của cả model roadmap.** Tokenizer là *trục đầu tiên* mọi thứ khác dựng lên:
nó định nghĩa V = vocab_size, và V CHÍNH LÀ chiều rộng của mỗi vector logit, chiều của `token_emb` và
`lm_head`, và cơ số của mọi cross-entropy về sau. Không có série nào trước nó — đây là *sàn nhà*: embedding
(série sau), Transformer block, training loss, RL reward đều index vào cái vocab này. Sợi chỉ xuyên série:
**tokenizer là một cái núm nén-vs-phủ (compression vs coverage)** — chọn V và chọn cách chẻ text quyết định
sequence dài bao nhiêu token (chi phí compute/FLOP, xem roadmap perf S1 Bài 1.0: decode đọc CẢ `lm_head` V×d
mỗi bước — V lớn = byte đọc nhiều hơn) và model "nhìn" thế giới ở hạt nào. Neo trung tâm của série, dùng lại ở
mọi Bài: **encode/decode round-trip là identity** (`decode(encode(s)) == s`, test đo thật) và **loss-at-init
≈ log(V)** (đĩa cân correctness rẻ nhất của cả stack — CLAUDE.md §Engineering disciplines #1).

Thứ tự đọc: **1.1 vì sao subword (byte/char/word trade-off, V là núm) → 1.2 thuật toán merge BPE (train) →
1.3 pretokenization regex + special tokens → 1.4 encode/decode round-trip + byte-fallback (no OOV).**

---

## Bài 1.1 — Vì sao subword: byte-vs-char-vs-word, V là núm nén/phủ (`src/scratch_llm/tokenizer.py` · `train_bpe` :130 · vocab ordering :151–164)
> **Câu hỏi first-principles:** tại sao KHÔNG dùng thẳng byte (V=256) hay thẳng word (V=cả từ điển)? Chọn V nằm ở giữa để tối ưu CÁI GÌ?
> **Neo (invariant):** loss-at-init ≈ log(V) — MEASURED: `algos/grpo` smoke đo response-token entropy ≈ log 256 ở init (bench/RESULTS.md:280); model tươi trên nhiễu phải cho CE ≈ log(V) đều. Sai ⇒ bug head/embedding/mask. Đây là lý do V (do tokenizer chốt) là con số load-bearing đầu tiên.

**1. Feynman — bài toán bằng lời.** Model chỉ nói được ngôn ngữ của một *bảng chữ cái hữu hạn* gồm V ký hiệu;
mỗi bước nó chọn 1 trong V. Câu hỏi thiết kế: bảng đó nên là gì? Ba cực. **(a) Byte thuần (V=256):** phủ MỌI
text (UTF-8) hoàn hảo, không bao giờ OOV — nhưng chuỗi cực dài (chữ Anh trung bình ~4 byte/từ, tiếng có dấu/emoji
tệ hơn) → attention là O(L²), decode tốn L bước, thông tin loãng. **(b) Word thuần (V = cả từ điển):** chuỗi ngắn,
giàu nghĩa mỗi token — nhưng V nổ (hàng triệu), `lm_head` V×d khổng lồ, và **OOV**: từ lạ/typo/tên riêng không có
id. **(c) Subword/BPE (V ~ 10³–10⁵):** núm chỉnh GIỮA — từ hay gặp thành 1 token (nén), từ lạ vỡ thành mảnh
subword hoặc tận cùng thành byte (phủ, không OOV). Đánh đổi cốt lõi: **compression (token/text ít) vs coverage +
V nhỏ**. Lớn V = nén tốt hơn (chuỗi ngắn hơn, ít FLOP hơn) nhưng head to hơn + mỗi token thấy ít data hơn.

**2. Dẫn xuất từ đầu (derive).** Đặt bài toán như *mã hoá*: cho corpus, chọn tập ký hiệu S để tối thiểu tổng số
token khi mã hoá, với |S| ≤ V. Word-freq gợi ý: ký hiệu đáng thêm vào S là chuỗi con XUẤT HIỆN NHIỀU. BPE là lời
giải *greedy* của bài toán đó (Bài 1.2). Điểm neo lượng hoá: một LM tươi (chưa học) trên input ngẫu nhiên phải
đoán *đều* trên V lựa chọn ⇒ p = 1/V mỗi token ⇒ cross-entropy = −log(1/V) = **log(V)**. Đó là lý do V phải được
chốt TRƯỚC khi có model: nó đặt điểm gốc của mọi loss. V nhỏ (256) ⇒ log 256 ≈ 5.55 nats; V lớn (65536) ⇒ log 2¹⁶
≈ 11.09 nats — loss "cao" hơn KHÔNG phải model tệ hơn, chỉ là trục rộng hơn. Đây là bẫy cổ điển khi so loss giữa
hai vocab: phải chuẩn hoá về **bits-per-byte** (val_bpb, bench/RESULTS.md:666) để so công bằng qua các tokenizer.

**3. Trace code.** `train_bpe` (:130) chốt V và bố cục id theo đúng thứ tự CS336: `base_size = len(special)+256`
(:142); nếu `vocab_size < base_size` → raise (:143, test `test_train_bpe_rejects_too_small_vocab`) — vì **256 byte
là sàn không thể bỏ** (điều kiện no-OOV, Bài 1.4). Gán id: special trước (:151–153), rồi 256 byte (:154–155), rồi
mỗi merge một id (:162–164). Test `test_train_bpe_vocab_ordering_and_size` (:47) pin: `vocab[0]==b"<|endoftext|>"`,
`vocab[1]==bytes([0])`, `vocab[256]==bytes([255])`, `len(vocab)==256+1+len(merges)`. Số merge = `V − base_size` =
núm nén: mỗi merge biến một cặp byte hay-gặp thành 1 token, rút ngắn chuỗi tương lai.

**4. Cổng teach-back.** (a) Giải thích vì sao loss-at-init của model với V=256 và V=50k KHÁC nhau nhưng KHÔNG nói
lên model nào tốt hơn — cách chuẩn hoá để so? (b) *Sửa-và-đoán:* nếu tăng V gấp đôi (thêm 30k merge) trên cùng
corpus, đoán 3 thứ đổi: (i) độ dài chuỗi token trung bình, (ii) loss-at-init, (iii) kích thước `lm_head` + byte
đọc mỗi bước decode (nối perf S1 Bài 1.0 — decode memory-bound, đọc cả lm_head).

**5. Frontier.** nanochat dùng GPT-4-style **Rust BPE, vocab 65,536 = 2¹⁶** (docs/FRONTIER_2026_ABLATIONS.md:66;
xác nhận `nanochat/tokenizer.py` lúc build). Chọn 2¹⁶ không ngẫu nhiên: power-of-two căn thẳng với kernel/tensor
dimension và cho id vừa uint16. Của ta V nhỏ hơn nhiều (toy: 400–512 trên TinyStories) — **giống về cơ chế BPE,
khác về scale**: ADR-0001-tokenizer-of-record ghi rõ BPE from-scratch này là *artifact mastery + tiny-LM*, còn
pipeline thật (A5 RL) dùng tokenizer native của model pretrained để hai engine index CÙNG một V (nếu không, mọi
so sánh per-token KL(train‖infer) là vô nghĩa). Câu interview: "Vì sao vocab-size là một *modeling decision*, không
phải hyperparameter tuỳ tiện — nó chạm những trục nào của model?" (logit width, FLOP/token, loss origin, OOV).

---

## Bài 1.2 — Thuật toán merge BPE: đếm cặp → merge tham lam nhất → lặp (`tokenizer.py` · `_compute_merges` :76 · `_merge_word` :61)
> **Câu hỏi first-principles:** cho một corpus đã chẻ thành từ, quy tắc *tham lam* nào xây được V−base ký hiệu subword, và làm sao chạy nhanh mà không đếm lại cả corpus mỗi vòng?
> **Neo (invariant):** MEASURED — `_compute_merges` trên `bpe_example` chuẩn (Sennrich/CS336) tái tạo ĐÚNG chuỗi merge `[st, est, ow, low, west, ne]` (test `test_compute_merges_reproduces_bpe_example` :21–37) + tie-break chọn cặp lexicographically GREATER (`(BA,A)`, test :40–45). Bắt được mọi lỗi đếm/tie-break.

**1. Feynman — bài toán bằng lời.** BPE = "nén bằng cách dán cặp hay gặp nhất, lặp lại". Analogy: bạn có một đống
từ viết bằng ký tự rời; bạn nhìn cặp ký hiệu KỀ NHAU xuất hiện nhiều nhất trong toàn corpus (ví dụ `s`+`t`), dán
chúng thành một ký hiệu mới `st`, ghi lại merge đó, rồi làm lại — giờ `e`+`st` có thể lên ngôi. Sau K vòng bạn có
K merge, theo THỨ TỰ. Đánh đổi: greedy (không tối ưu toàn cục) đổi lấy *xác định + rẻ*; và thứ tự merge là
load-bearing — nó phải được replay Y HỆT lúc encode (Bài 1.4).

**2. Dẫn xuất từ đầu (derive).** Trạng thái: mỗi "word" là tuple các symbol (ban đầu = từng byte). Đại lượng cần:
`count(pair)` = tổng freq của từ chứa cặp đó kề nhau. Vòng lặp: `best = argmax count`, tie → **cặp lớn hơn theo
thứ tự từ điển** (quy ước CS336; byte so lexicographically). Merge = `best[0]+best[1]`, ghi vào danh sách. Ngây thơ
= O(vòng × corpus). **Mẹo incremental (cốt lõi):** giữ `pair_to_words[pair]` = tập chỉ số từ chứa cặp; khi merge
`best`, CHỈ quét lại những từ ĐÓ. Với mỗi từ đổi, tính delta = (đếm cặp mới) − (đếm cặp cũ), cập nhật `pair_counts`
cục bộ. Chi phí scale theo *số từ bị chạm*, không theo corpus. Vì sao tie-break "greater" quan trọng: khi nhiều
cặp đồng tần, chọn xác định (không phụ thuộc thứ tự dict) là điều kiện để merge REPRODUCIBLE — chính cái test :40 gác.

**3. Trace code.** `_compute_merges` (:76): khởi tạo — `words` list-of-list (:84), `freqs` (:85); quét lần đầu dựng
`pair_counts` + `pair_to_words` (:89–93). Vòng chính (:96): `best = max(pair_counts, key=lambda p:(pair_counts[p], p))`
(:101) — tuple `(count, pair)` cho "count cao nhất, tie → pair lớn hơn"; `merged = best[0]+best[1]` (:102); append
(:103). Cập nhật incremental (:105–125): với mỗi từ trong `pair_to_words[best]`, `_merge_word` (:61) thay MỌI lần
xuất hiện non-overlapping trái→phải của cặp (:67, `i+=2` khi khớp); rồi so `old_pairs` vs `new_pairs` (:111–112),
cộng `delta*f` vào `pair_counts` (:114–116), bảo trì `pair_to_words` (:117–120), dọn cặp về 0 (:121). Cuối vòng
pop `best` (:124–125). `train_bpe` gọi nó với `num_merges = vocab_size − base_size` (:160).

**4. Cổng teach-back.** (a) Dựng lại vì sao chỉ quét-lại-từ-bị-chạm cho ĐÚNG `pair_counts` như đếm-lại-toàn-bộ —
tại sao delta = new−old đủ, không sót cặp nào? (b) *Sửa-và-đoán:* đổi tie-break từ "greater" sang "smaller" (đổi
`key` thành `(count, [thứ tự đảo])`), đoán `bpe_example` giờ ra chuỗi merge gì ở bước đầu — có còn `[st, est, ...]`
không, và test :21 hỏng ở merge số mấy?

**5. Frontier.** Đây đúng là thuật toán Sennrich et al. 2016 (BPE gốc cho NMT), và là lõi của tiktoken (GPT-2/3/4)
+ Rust BPE của nanochat. Khác biệt frontier là *tốc độ train*, không phải thuật toán: nanochat/tiktoken viết vòng
này bằng **Rust** (song song hoá pretokenization + đếm) để train vocab 2¹⁶ trên hàng chục GB trong phút; của ta
là Python incremental — đúng nhưng chậm, kill-criterion của A1 guide: nếu BPE train trên TinyStories vỡ ngân sách
thời gian thì `multiprocessing` pretokenization TRƯỚC khi làm gì fancy (docstring :14–15). Câu interview: "BPE train
là greedy trên cái objective gì, và vì sao thứ tự merge phải được lưu chứ không chỉ tập token?" (encode phải replay
đúng thứ tự — Bài 1.4).

---

## Bài 1.3 — Pretokenization regex GPT-2/GPT-4 + special tokens (`tokenizer.py` · `_GPT2_PAT` :31 · `_pretokenize_counts` :41)
> **Câu hỏi first-principles:** vì sao phải CHẺ text bằng regex \p{L}/\p{N}/\p{P} TRƯỚC khi merge, thay vì thả BPE lên cả dòng thô? Và special token đứng ở đâu trong luồng?
> **Neo (invariant):** MEASURED — pretokenize KHÔNG bao giờ vượt ranh giới special (`test_pretokenize_does_not_cross_special_boundary` :98–103: `<|endoftext|>` bị tách ra, "ab"/"cd" đếm riêng, không word nào chứa `<`); special là MỘT id, không bị chẻ (`test_special_token_is_single_id_and_not_split` :84–89).

**1. Feynman — bài toán bằng lời.** Nếu thả BPE lên cả câu thô, nó sẽ học merge XUYÊN qua khoảng trắng và dấu câu:
`"dog."` và `"dog"` thành token khác nhau, `"the "` dính với từ sau tuỳ ngữ cảnh — vocab bẩn, phân mảnh vô lý.
Pretokenization là *hàng rào*: chẻ text thành "pre-token" (mảnh từ/số/dấu/khoảng trắng) và **cấm merge vượt ranh
giới pre-token**. BPE chỉ chạy TRONG mỗi pre-token. Analogy: bạn cắt câu thành các viên gạch theo loại ký tự trước,
rồi mới cho phép dán bên trong từng viên. Đánh đổi: regex là một *prior* của con người áp lên BPE (BPE tự nó không
biết "từ") — đổi tính "học thuần data" lấy vocab sạch, có cấu trúc.

**2. Dẫn xuất từ đầu (derive).** Pattern GPT-2 (:31–33) là một alternation, đọc từng nhánh: `'(?:[sdmt]|ll|ve|re)`
gom contraction tiếng Anh (`'s`, `'ll`…); ` ?\p{L}+` = một cụm CHỮ, *khoảng trắng dẫn đầu tuỳ chọn dính vào từ sau*
(nên `" the"` là MỘT pre-token — mẹo để model biết ranh giới từ mà không phí token space riêng); ` ?\p{N}+` = một
cụm SỐ; ` ?[^\s\p{L}\p{N}]+` = cụm dấu câu/ký hiệu; `\s+(?!\S)|\s+` = khoảng trắng đuôi. Cần `regex` (không phải
`re` stdlib) vì `\p{L}/\p{N}` là Unicode property class (:29–30). Special token là chuyện KHÁC pretokenization: nó
là *ranh giới tài liệu* (`<|endoftext|>`) — phải chẻ TRƯỚC, longest-first, và **KHÔNG đếm vào BPE** (nó có id riêng
được cấp ở Bài 1.1), để không merge nào dính hai tài liệu.

**3. Trace code.** `_pretokenize_counts` (:41): nếu có special → dựng delimiter `|`.join escape, **sorted len giảm
dần** (:48, longest-first để `<|eot|><|eot|>` khớp trước `<|eot|>`), `regex.split` KHÔNG capture group ⇒ delimiter
BỊ BỎ khỏi `chunks` (:49) — special không lọt vào đếm. Với mỗi chunk, `_GPT2_PAT.finditer` (:55) → mỗi match
`.encode("utf-8")` → thành tuple từng-byte `tuple(bytes([b]) for b in ...)` (:57) → `counts[...] += 1`. Đây là input
cho `_compute_merges` (Bài 1.2). Phía encode (Bài 1.4) `_split_on_specials` (:261) dùng CÙNG regex nhưng CÓ capture
group `(...)` (:265–269) để GIỮ special làm segment riêng — đối xứng với train nhưng ngược mục đích (giữ vs bỏ).

**4. Cổng teach-back.** (a) Vì sao " ?\p{L}+" cho khoảng-trắng-dẫn-đầu-dính-từ-sau lại là thiết kế tốt hơn tách
space thành token riêng — nó tiết kiệm/mã hoá gì? (b) *Sửa-và-đoán:* đổi nhánh số ` ?\p{N}+` → ` ?\p{N}{1,3}`
(pattern Llama-3/GPT-4, digit-grouping), đoán `"12345"` chẻ thành mấy pre-token và thành gì; và vì sao điều này
GIÚP số học (nối FRONTIER_PRACTICE_2026 §digit-grouping :130–136) — round-trip có còn giữ không?

**5. Frontier.** Đây là điểm phân kỳ GPT-2 vs GPT-4/Llama-3 rõ nhất và là một *modeling decision* dated: cl100k
(GPT-4/tiktoken) và Llama-3 dùng `\p{N}{1,3}` — chẻ số thành nhóm ≤3 chữ số để BPE không nuốt cả con số thành token
kỳ dị (số học tệ hơn). Của ta hiện là `\p{N}+` (GPT-2, tokenizer.py:32) ⇒ digit-grouping là *một dòng sửa* +
retrain tiny BPE, invariant: `encode("12345")` không pre-token nào >3 chữ số, round-trip vẫn pass
(FRONTIER_PRACTICE_2026:134). nanochat/tiktoken cũng xử special token đúng cách này (id riêng, tách trước merge).
Câu interview: "Pretokenization giải quyết bug gì mà BPE thuần không thấy, và tại sao digit-grouping là mặc định
2026?" (merge xuyên ranh giới → vocab bẩn; số kỳ dị → arithmetic kém).

---

## Bài 1.4 — Encode/decode round-trip + byte-fallback (no OOV) (`tokenizer.py` · `Tokenizer.encode` :242 · `_bpe` :219 · `decode` :257)
> **Câu hỏi first-principles:** encode phải replay merge thế nào để KHỚP train? Và bất biến `decode(encode(s))==s` giữ được nhờ đâu — vì sao KHÔNG BAO GIỜ OOV kể cả với emoji/chữ lạ chưa từng thấy?
> **Neo (invariant):** MEASURED — `decode(encode(s)) == s` cho MỌI UTF-8 (`test_encode_decode_round_trip_ascii_and_unicode` :78–81, gồm "世界 🌍 emoji", " leading spaces", ""); `encode_iterable` khớp `encode` (:92–95). Đây là identity round-trip — Neo trung tâm của série.

**1. Feynman — bài toán bằng lời.** Encode = "phát lại lịch sử nén": lấy pre-token (bytes), rồi áp CÁC merge đã học
*theo đúng thứ tự train* cho tới khi không merge được nữa; kết quả là chuỗi id. Decode = ngược lại tầm thường: nối
bytes của từng id rồi giải UTF-8. Bí mật "không bao giờ OOV": sàn của mọi thứ là **256 byte đơn** — id nào cũng
cuối cùng phân rã về byte, và MỌI text UTF-8 là một chuỗi byte, nên ký tự lạ nhất (emoji chưa từng thấy) vẫn mã hoá
được thành các byte của nó. BPE chỉ *nén* các byte đó khi có merge; không có merge thì chúng ở dạng byte thô —
vẫn hợp lệ. Đó là **byte-fallback**: coverage được bảo đảm bởi cấu trúc, không phải may mắn.

**2. Dẫn xuất từ đầu (derive).** Encode một pre-token phải cho CÙNG kết quả như thể pre-token đó có mặt lúc train.
Train áp merge theo thứ tự tần suất giảm dần; nên encode phải: ở mỗi bước, tìm cặp kề nhau có **rank NHỎ NHẤT**
(merge sớm nhất = ưu tiên cao nhất), dán nó, lặp. `rank[pair] = vị trí trong danh sách merges` (:188). Đây là điểm
tinh tế: không phải "merge cặp trái nhất" hay "cặp dài nhất", mà **cặp có rank sớm nhất** — đúng thứ tự train.
Round-trip đúng vì: decode chỉ nối bytes (`vocab[id]`); `vocab[id]` cho mỗi merge = `a+b` (Bài 1.1 :163), cho byte
= chính byte đó; nối lại toàn bộ = đúng chuỗi byte gốc của pre-token; ghép mọi pre-token = đúng bytes gốc của text
(pretokenization chỉ chẻ, không xoá byte nào — kể cả space, vì " ?\p{L}+" giữ space) ⇒ `.decode("utf-8")` trả về s.

**3. Trace code.** `Tokenizer.__init__` (:177): `_bytes_to_id` (:187, đảo vocab), `_ranks` (:188, merge→rank),
special có thể thêm sau nếu chưa trong vocab (:189–197), `_cache` (:198). `encode` (:242): `_split_on_specials`
(:244) tách segment; special → append thẳng id (:246); text thường → `_GPT2_PAT.finditer` (:248, CÙNG regex train)
→ mỗi match `_bpe(bytes)`. `_bpe` (:219): cache hit (:221); else `parts = [từng byte]` (:225); vòng (:226): quét
tìm `best_i` với `rank` NHỎ NHẤT (:229–233, `rank < best_rank`); nếu không cặp nào có rank → break (:234); dán
`parts[i:i+2] = [a+b]` (:236); cuối map `parts → ids` qua `_bytes_to_id` (:238), cache (:239). `decode` (:257):
`b"".join(vocab[i])` rồi `.decode("utf-8", errors="replace")` (:259 — `replace` chỉ cứu chuỗi id THỦ CÔNG cắt giữa
ký tự multi-byte; round-trip hợp lệ không kích hoạt nó). `from_files` (:200) dùng latin-1 (:210,216) để mọi byte
0–255 round-trip qua text file.

**4. Cổng teach-back.** (a) Chứng minh bằng lời vì sao byte-fallback ⇒ KHÔNG BAO GIỜ OOV: theo dõi một emoji chưa
từng thấy qua encode→decode, chỉ ra nó dừng ở đâu (byte thô) và vì sao decode vẫn dựng lại đúng. (b) *Sửa-và-đoán:*
trong `_bpe` đổi tiêu chí từ "rank nhỏ nhất" sang "cặp trái nhất bất kể rank" — round-trip decode(encode(s))==s có
CÒN giữ không? Cái gì hỏng (gợi: round-trip *vẫn* đúng vì bytes bảo toàn, nhưng chuỗi ID khác train ⇒ model thấy
token lạ ⇒ loss/compression tệ — round-trip là điều kiện CẦN, không ĐỦ cho "đúng như train").

**5. Frontier.** Byte-fallback + round-trip là hợp đồng chung của tiktoken/nanochat/HF: id sớm-rank thắng là chính
xác thuật toán encode của tiktoken. Một khác biệt implementation frontier: tiktoken/Rust BPE encode bằng cấu trúc
ưu tiên (priority queue trên rank) O(n log n) thay vì quét O(n²) như `_bpe` :229 của ta — cùng KẾT QUẢ, khác tốc
độ (của ta có `_cache` per-pretoken :221 để amortize). ADR-0001 nhấn: chính vì SỞ HỮU round-trip + special-token
stability này mà ta *đọc/patch được token id* và tin được ranh giới special khi A5 chuyển sang tokenizer native —
contract khởi động phải assert `vocab_size` + mọi special id KHỚP giữa train-engine và serve-engine, lệch là abort
(so sánh KL per-token trên hai trục vocab khác nhau là vô nghĩa). Câu interview: "Vì sao byte-level BPE không bao
giờ OOV, và round-trip identity CHỨNG minh điều gì — nó có đủ để nói encode 'đúng như train' không?"

---

*Đóng série.* Tokenizer chốt V (Bài 1.1, núm nén-vs-phủ, loss-at-init=log V) bằng thuật toán merge greedy
incremental (Bài 1.2, reproduces `[st,est,ow,low,west,ne]`), có hàng rào pretokenization + special tokens (Bài 1.3,
không merge xuyên ranh giới), và bảo đảm round-trip identity + no-OOV bằng byte-fallback (Bài 1.4). Série sau
(embedding + Transformer block) index thẳng vào cái V này: mỗi id → một hàng `token_emb` (d chiều), và `lm_head`
V×d trả về đúng trục logit mà cross-entropy đo bằng log(V) làm gốc. Nối perf: V là một phần *byte đọc mỗi bước
decode* (roadmap S1 Bài 1.0) và *chiều rộng GEMM cuối* — tokenizer là quyết định model ĐẦU TIÊN có hệ quả xuống
tận kernel.
