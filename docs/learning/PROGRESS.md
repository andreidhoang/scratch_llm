Learning-node: M3 Bài 3.3 — cosine schedule + warmup + gradient clipping (`optim.py` cosine_lr/grad clip). ✅ M3 3.2 AdamW (2026-07-14). 📚 **DERIVATION DOCS 16/16 PILLAR COMPLETE (2026-07-14)** — `docs/learning/derivations/` (9,782 dòng, mọi neo code verified + math re-derived + số đo chạy lại được; 12 FIXED/3 SOUND, 0 rewrite). *Viết-doc = substrate re-learn, KHÁC teach-back:* cột dưới vẫn track **cold defense** của Navigator (mastery thật). Teach-back đã defend cold: M1 1.1–1.4 · M2 2.1–2.3 (2.4–2.6 Driver-explained) · M3 3.1–3.2. Node teach-back kế = 3.3. Nợ re-own: 2.4/2.5/2.6/3.1/3.2 + mờ 2.3 (i)/(ii). Mở 3.3: spaced recall (1 step AdamW?) → cold predictions cho warmup + cosine + grad clip. **Pointer (2026-08-02):** hàng đợi mastery CHÍNH giờ là K3 hand-build order `situ → kda → gated_mla → latent_moe → attn_res` (`k3/core/`, ROADMAP:56–61) — K2/KDA là "the rung where the mastery happens" (ROADMAP:156).

# Learning progress — sổ cái mastery 89 Bài (track theo teach-back)

> **Đây là gì.** Nguồn-sự-thật DUY NHẤT cho *Navigator đang ở đâu* trên con đường 89 Bài của
> [`CURRICULUM.md`](CURRICULUM.md). Song sinh với `performance/PERF_PLAN.md` "Current Node" của nửa build:
> một session mới đọc dòng `Learning-node:` ở đầu file (được `session-start.sh` tiêm vào context) + bảng
> dưới → biết ngay Bài kế tiếp, KHÔNG cần đoán, KHÔNG re-derive cái đã hiểu. Trải nghiệm học liền mạch
> xuyên session — đúng cách một senior AI RE track kiến thức từ đầu.
>
> **Framework (cập nhật 2026-07-06):** Bạn có thể master trực tiếp trên code đã ship hoặc tự tay viết lại (re-implement by hand) thông qua thư mục [mastery/](file:///Users/danghuyhoang/Desktop/cs336/scratch_llm/mastery/README.md) (được gitignore để đảm bảo an toàn). Thư mục này chứa bản sao cấu trúc codebase với skeletons và các file tests độc lập giúp bạn code tay và chạy `PYTHONPATH=mastery/src pytest mastery/tests/` trực tiếp mà không ảnh hưởng tới code chính hay CI. Xem code trong `src/scratch_llm/` là **Oracle** để đối chiếu và sửa lỗi. Mọi ghi chép sâu vẫn lưu tại `docs/learning/`.
>
> **Phương pháp học = PRR loop (Predict → Run → Reconcile), mặc định (2026-07-05).** KHÔNG đọc monologue
> — internalize là hàm của *nỗ lực tự-truy-hồi của Navigator*, không phải chất lượng giải thích của Driver
> (đọc derivation hay = *fluency illusion*: thấy hiểu nhưng không encode). Mỗi **micro-concept** (nhỏ hơn
> một Bài, tôn trọng giới hạn working-memory ~4 chunk): (1) **Predict COLD** — Navigator viết dự đoán/dòng
> dẫn xuất đầu tiên TRƯỚC mọi giải thích (đoán sai là tốt); (2) **Run** — chạy code thật, in shape/số/token
> hoặc test red→green trên GPU; (3) **Reconcile** — Driver chỉ giải thích ĐÚNG chỗ lệch, 3 dòng, không phải
> tường; (4) **Re-derive + tự VẼ** trên ví dụ MỚI; (5) **Teach-back GATE + modify-and-predict**. Depth
> (code-anchor + Feynman VN) nằm ở bước reconcile/re-derive, KHÔNG phải wall-of-text bước 1. Chi tiết đầy
> đủ: `CLAUDE.md` §How we build → "Master understanding — the PRR loop".
>
> **Giao thức ledger (gắn với cổng teach-back ở bước 5):**
> 1. Qua cổng teach-back một Bài ⇒ set Bài đó **✅** kèm **ngày + neo đã defend** (bằng chứng đã master
>    thật, không phải tick suông). **Neo NÊN kèm neo tuyển dụng:** gate phỏng vấn nào Bài này ăn + FOP
>    trait nào phô ra (theo [`FRONTIER_HIRING_MAP.md`](FRONTIER_HIRING_MAP.md) §8/§9) — ledger đồng thời
>    là bản đồ sẵn-sàng-phỏng-vấn. Blank-slate MỘT hàm để re-derive lạnh = tập dượt gate "from scratch 45'".
> 2. **Advance dòng `Learning-node:`** ở đầu file sang Bài kế (theo thứ tự CURRICULUM 8 chặng).
> 3. Cũng tick ở [`INDEX.md`](INDEX.md) study-queue nếu Bài đó có trong đó.
> 4. **Spaced callback:** mở session KẾ bằng 1 câu recall 20 giây từ một Bài ✅ TRƯỚC đó (spacing +
>    interleaving thắng đường quên) rồi mới vào Bài mới.
> 5. **Một Bài ✅ là ĐÃ SỞ HỮU — không bao giờ re-derive** (FOP-7: đừng dựng lại việc đã defend cold được),
>    trừ khi user chủ động xin re-own (blank-slate protocol ở `CLAUDE.md` §Reference-as-oracle).
>
> **Trạng thái:** ⬜ chưa học · 🔄 đang học (gate mở) · ✅ đã teach-back.
> **Lưu ý:** "lesson đã viết" (file roadmap tồn tại) ≠ "đã master". Bảng này track **teach-back của
> Navigator**, không phải sự tồn tại của doc. Vì vậy nó bắt đầu gần như toàn ⬜ dù roadmap đã viết đầy đủ.

Thứ tự = con đường đơn nhất của [`CURRICULUM.md`](CURRICULUM.md) (đan model+perf theo mối nối tự nhiên).

## Chặng 0 · Định hướng
| Bài | Concept | TT | Ngày | Neo đã defend |
|---|---|---|---|---|
| 0 | Claude Code vận hành trên repo này (meta) | ⬜ | | |

## Chặng 1 · MODEL M1–M4 — dựng model chạy được trên 1 GPU
### M1 · Tokenizer (`tokenizer.py`)
| Bài | Concept | TT | Ngày | Neo đã defend |
|---|---|---|---|---|
| 1.1 | Vì sao subword; V = núm nén-vs-phủ; loss-at-init=log V | ✅ | 2026-07-05 | loss-at-init=log V (`model.py:983-986`); 256-byte floor → no-OOV (`tokenizer.py:142,154-156`); double-V: seq↓ / log V +0.69 / lm_head params×2 & decode-bytes×2; so sánh công bằng = bits-per-byte |
| 1.2 | Thuật toán merge BPE (count-pair → greedy → incremental) | ✅ | 2026-07-05 | greedy max-count + tie-break lexicographic (`:99-101`, cần cho reproducibility); incremental delta chỉ đụng pair cạnh merge-site + inverted index `pair_to_words` (`:88,105-125`); trace `the cat sat on the mat` |
| 1.3 | Pretokenization regex GPT-2/4 + special tokens | ✅ | 2026-07-05 | fence chặn merge cross word/doc-boundary; leading-space rides với word (`_GPT2_PAT` :31-33); special split FIRST longest-first, không đếm, own ids (`:45-49`) |
| 1.4 | Encode/decode round-trip + byte-fallback (no-OOV) | ✅ | 2026-07-06 | PRR-verified vs real `tokenizer.py`: (P1) `'世界 🌍'`→**11 ids** round-trip True — #token = f(training corpus), English-BPE học 0 merge cho CJK/emoji ⇒ full byte-fallback (`_bpe:225`, decode byte-concat `:257-259`); (P2) `encode('a<|endoftext|>b')`=`[98,0,99]` special id=**0 FRONT** (spec order `:151-156`), isolated không merge (`_split_on_specials`→`encode:244-249`); (P3) round-trip ⊥ merge-policy (holds under leftmost), cái hỏng = **fidelity to training segmentation** (encode phải replay min-rank `:230-233` để boundaries==train-time, nếu không → OOD tokens, KHÔNG phải byte-explosion). **Hiring:** gate = BPE-from-scratch + round-trip invariant as correctness oracle (test-first/sanity family); trait = epistemic precision (round-trip-holds ≠ segmentation-matches). Table-stakes. |
### M2 · Transformer forward (`model.py`)
| Bài | Concept | TT | Ngày | Neo đã defend |
|---|---|---|---|---|
| 2.1 | Attention = differentiable KV-retrieval | ✅ | 2026-07-06 | Differentiable key-value lookup. Softmax(Q K^T / sqrt(d_k)) V. Division by sqrt(d_k) stabilizes gradients by preventing softmax saturation. Causal masking blocks future tokens. |
| 2.2 | Multi-head + GQA/MQA, vì sao share KV | ✅ | 2026-07-14 | GQA = **decouple query-resolution (n_heads) khỏi KV-resolution (n_kv)**: q_proj→n_heads·head_dim, k/v_proj→n_kv·head_dim (`model.py:238-240`). Defend COLD (anchor 32/8, lại trên 64/8): repeats=n_heads//n_kv=4 = GROUP SIZE (không phải n_kv); repeat_interleave dim=1 → khối liền kề ⇒ head i→kv i//repeats (`:319-320`); k=(1,8,s,128) NHỎ hơn q=(1,32,s,128) ⇒ cache 4× nhỏ (đo 4096 vs 16384 B/tok/layer bf16). QKᵀ không repeat **RAISES** (32 vs 8 không broadcast). MQA n_kv=1 = 1 retrieval-subspace (hại chất lượng+ổn định), GQA n_kv=8 qua knee (Ainslie 2023); n_kv=3 → ValueError fail-fast `:81`. **Hiring:** gate = giải thích KV-cache memory-wall & GQA/MLA cắt thế nào (serving = scarce differentiator); trait = predict-the-number (đoán đúng 4×/8× cache ratio) + trade-off. *Mechanics defend cold; why-A/why-B Driver-explained → re-derive lạnh ở spaced-callback 2.3.* |
| 2.3 | RoPE từ yêu cầu relative-position | ✅ | 2026-07-14 | Rotate mỗi cặp (x_even,x_odd) góc pos·θₖ (`model.py:206-207`), θₖ=1/theta^(2k/d) (`:186`). Defend COLD (đo trên RoPE thật, head_dim=8): (1) **rotation** bảo toàn chuẩn 3.395326→3.395327; (2) q_rot·k_rot chỉ phụ thuộc **(m−n)** — (5,3)=(7,5)=(12,10)=0.24372 đo được, Δ=0→raw q·k; (3) spectrum [1,0.1,0.01,0.001] = đồng hồ đa-tần (pair0 nhanh/cận, pair cuối chậm/xa). Derivation R_{mθ}ᵀR_{nθ}=R_{(n−m)θ} (Rᵀ=R⁻¹ + cộng-góc) + perturbation theta 1e4→1e6 (bước sóng chậm nhất 6.283→198.692 tok ⇒ long-context) = Driver-explained → re-derive lạnh spaced-callback 2.4. **Hiring:** gate = vì sao RoPE cho relative-pos + theta/YaRN long-context (table-stakes arch + serving-adjacent); trait = first-principles derivation. |
| 2.4 | RMSNorm + pre-norm residual | ✅ | 2026-07-14 | RMSNorm out=x/RMS·weight ⇒ RMS(out)=1 đo 6.53→1.000000 (`:149-150`), không centering/bias; pre-norm `x=x+attn(norm(x))` (`:409`) giữ identity path ⇒ grad qua 50 block = 298 vs post-norm 7.4e-5 (**4 triệu×**), đổi lại ||stream|| phình 36 ⇒ cần final_norm `:435`; RMSNorm bỏ mean-centering của LayerNorm (Zhang&Sennrich 2019: re-scaling mới là cái có ích) ⇒ rẻ hơn (memory-bound). **Hiring:** gate = pre-norm vs post-norm stability + norm ở đâu (arch table-stakes); trait = predict-the-number (grad ratio). *Driver-explained, chốt theo quyết định Navigator — KHÔNG cold-defend; ứng viên blank-slate re-own nếu muốn bản interview-grade.* |
| 2.5 | SwiGLU gated MLP vs ReLU/GELU | ✅ | 2026-07-14 | FFN = xử lý per-token, ~2/3 param model. SwiGLU=w2(silu(w1 x)⊙w3 x) 3 ma trận (`:220-225`) vs ReLU-MLP 2 ma trận. **Iso-param:** 3·d·d_ff=8d² ⇒ d_ff=8/3·d (đo d=4096: 134,221,824≈134,217,728 ratio 1.0000; Llama-2-7B 11008=2.688d). Gate = tích 2 phép chiếu ⇒ tương tác NHÂN/data-dependent gating (họ GLU, Dauphin17/Shazeer20) mà ReLU-MLP (piecewise-linear 1 proj) không làm được. SiLU=x·sigmoid(x): mượt, grad≠0 khi x<0 (no dead unit), self-gate mềm vs ReLU zero-cứng. **Hiring:** gate = FFN design + iso-param/FLOP comparison (arch table-stakes); trait = iso-param honesty. *Driver-explained, Navigator-closed — blank-slate re-own nếu muốn interview-grade.* |
| 2.6 | QK-norm + tied/untied embeddings + cross_entropy seam | ✅ | 2026-07-14 | **QK-norm** (`:267-268`, TRƯỚC RoPE `:272`): RMSNorm(head_dim) ⇒ ‖q‖=√head_dim cố định (đo 364→5.657=√32) ⇒ logit q·kᵀ/√d bị chặn ≤√head_dim, chống bf16-instability #1 (Qwen3/Gemma3/OLMo2; Kimi-K2 QK-Clip=F9 `:247-251`). **Tied emb** (`:437-438` lm_head.weight=token_emb.weight): tiết kiệm đúng V·d_model (đo 128,000=1000×128; scale thật ~512M); repo untie mặc định (`:68` ADR-0004) giữ loss@init sạch. **CE seam** (`:496-499`): CE=logsumexp(z)−z_t, log∘exp triệt tiêu, fp32 chống overflow; loss@init≈log V (đo 7.04≈6.91) = oracle rẻ nhất, full circle Bài 1.1. **Hiring:** gate = loss-at-init sanity + bf16 stability (arch+debug table-stakes); trait = cheapest-oracle discipline. *Driver-explained, Navigator-closed — blank-slate re-own nếu muốn interview-grade.* |
### M3 · Objective + optimization (`optim.py`, `model.py`)
| Bài | Concept | TT | Ngày | Neo đã defend |
|---|---|---|---|---|
| 3.1 | cross_entropy từ MLE, log V at init, logsumexp | ✅ | 2026-07-14 | **MLE→NLL→CE:** học = maximize Πₜ p_θ(xₜ|x_<t) → log → minimize NLL=−Σ log p; với q=onehot, H(q,p)=−log p_t = NLL ⇒ CE **là** MLE viết lại (`cross_entropy` `:490-499`). **Gradient đẹp nhất:** ∂CE/∂z_j = p_j−[j==t] = **softmax−onehot** (đo autograd vs công thức: khớp allclose True, V=5); = "sai số dự đoán", bounded [−1,1], log∘exp triệt tiêu ⇒ no vanishing (vì sao KHÔNG MSE: softmax+MSE có p(1−p) vanishing). MLE trực quan đo: logit[t]+=0/2/5 → p 0.015/0.10/0.69, CE 4.21/2.30/0.37. log V@init từ p≈1/V (đo 7.04≈6.91, khép Bài 1.1). **Hiring:** gate = dẫn ∂CE/∂z=softmax−onehot cold + vì sao không MSE (RL/optim table-stakes); trait = first-principles + cheapest-oracle. *Driver-explained, Navigator-closed.* |
| 3.2 | AdamW: SGD→momentum→Adam→decoupled WD | ✅ | 2026-07-14 | **SGD** 1 lr/mọi chiều → kẹt ill-conditioned (đo f=0.5(100x²+y²) 200 step: dist 0.108, lr≤2/a=0.02 ghì chiều thoải). **Momentum** m=β₁m+(1−β₁)g (`:84`): EMA triệt zigzag chiều dốc, tích luỹ chiều thoải → 0.0001. **Adam** v=β₂v+(1−β₂)g² (`:85`), θ−=α_t·m/(√v+eps) (`:88`): chia √v = per-dim adaptive LR → đi thẳng, hội tụ @lr 0.1 (5× SGD, dist 0.0003). **Bias-correct** α_t=lr√(1−β₂ᵗ)/(1−β₁ᵗ) (`:87`). **Decoupled WD** θ−=lr·λ·θ (`:91`): tách khỏi grad → shrink đều (1−lrλ) (đo grad=0: 10→9.9); L2-in-Adam sai vì λθ bị chia √v → decay không đều (Loshchilov&Hutter19). betas=(.9,.95) wd=.1. **Hiring:** gate = dẫn Adam từ SGD + vì sao decoupled WD (optim table-stakes); trait = first-principles. Doc: `derivations/M3_optimization.md`. *Driver-explained.* |
| 3.3 | cosine schedule + warmup + gradient clipping | ⬜ | | |
| 3.4 | Muon: Newton-Schulz orthogonalization, RMS=1/√max | ⬜ | | |
| 3.5 | Param-split hybrid + TIED-2D trap | ⬜ | | |
### M4 · Training loop + efficiency (`train.py`, `utils/`)
| Bài | Concept | TT | Ngày | Neo đã defend |
|---|---|---|---|---|
| 4.1 | One train step + memmap + overfit-one-batch oracle | ⬜ | | |
| 4.2 | Mixed precision bf16 autocast, no GradScaler | ⬜ | | |
| 4.3 | Activation checkpointing: recompute-vs-store | ⬜ | | |
| 4.4 | torch.compile + F4 (bf16+compile NaN guard) | ⬜ | | |
| 4.5 | Fixed-seed reproducibility + monitors | ⬜ | | |

## Chặng 2 · PERF S3–S4 — đưa compute về tốc độ ánh sáng
### S3 · CUDA-core kernels (`kernels/*_triton.py`, `bench/`)
| Bài | Concept | TT | Ngày | Neo đã defend |
|---|---|---|---|---|
| 3.0 | Roofline harness: đo cả hai roof trên chính card | ⬜ | | |
| 3.1 | GEMV: bậc AI≈1 dưới đáy | ⬜ | | |
| 3.2 | Softmax: online recurrence, đếm HBM pass | ⬜ | | |
| 3.3 | Norms: giá của reduction thứ hai | ⬜ | | |
| 3.4 | TopK: "đo cái thất bại" + cứu bằng fusion | ⬜ | | |
| 3.5 | GEMM: kernel DUY NHẤT vượt ridge | ⬜ | | |
### S4 · Tensor cores + Flash (`kernels/`)
| Bài | Concept | TT | Ngày | Neo đã defend |
|---|---|---|---|---|
| 4.1 | Naive SMEM GEMM: cái sàn CUDA-core | ⬜ | | |
| 4.2 | WMMA GEMM: fragment machinery, vì sao FP32-accum | ⬜ | | |
| 4.3 | mma.sync + ldmatrix + XOR swizzle | ⬜ | | |
| 4.4 | Naive attention: bức tường O(N²) | ⬜ | | |
| 4.5 | Online softmax: Milakov recurrence cô lập | ⬜ | | |
| 4.6 | FA2: fused fwd (no N×N) + recomputation backward | ⬜ | | |

## Chặng 3 · Trải ra nhiều GPU — M5, S6
### M5 · Distributed training (`utils/{ddp,zero1,fsdp,memory_math,comms_calc}.py`)
| Bài | Concept | TT | Ngày | Neo đã defend |
|---|---|---|---|---|
| 5.1 | Data-parallel + ring all-reduce | ⬜ | | |
| 5.2 | ZeRO-1/2/3: 16-psi model + sharding | ⬜ | | |
| 5.3 | FSDP (ZeRO-3): all-gather/reduce-scatter flat-param | ⬜ | | |
| 5.4 | Activation-ckpt + mixed-prec memory: 100B one-pager | ⬜ | | |
| 5.5 | Comms algebra: khi nào scaling comms-bound | ⬜ | | |
### S6 · Distributed primitives + ISA (`utils/`, `performance/rental/kernels/`)
| Bài | Concept | TT | Ngày | Neo đã defend |
|---|---|---|---|---|
| 6.1 | Megatron TP MLP: 2 all-reduce/layer | ⬜ | | |
| 6.2 | Pipeline GPipe vs 1F1B, bubble (p-1)/m | ⬜ | | |
| 6.3 | Expert-parallel MoE: all-to-all | ⬜ | | |
| 6.4 | MFU/HFU 6ND + sáu killer | ⬜ | | |
| 6.5 | WGMMA GEMM (compile-only sm_90a) | ⬜ | | |
| 6.6 | FA3 warp-specialized (compile-only) | ⬜ | | |
| 6.7 | tcgen05 GEMM (compile-only sm_100a) | ⬜ | | |

## Chặng 4 · Khoa học & nhiên liệu — M6, M7
### M6 · Scaling laws (`scaling/{isoflop,planner}.py`)
| Bài | Concept | TT | Ngày | Neo đã defend |
|---|---|---|---|---|
| 6.1 | Vì sao scaling law; log-space fit | ⬜ | | |
| 6.2 | IsoFLOP method: fit (N*,D*), cổng a+b=1 | ⬜ | | |
| 6.3 | Chinchilla ~20 tok/param | ⬜ | | |
| 6.4 | Budget planner dưới API đối kháng | ⬜ | | |
### M7 · Data pipeline (`data/`)
| Bài | Concept | TT | Ngày | Neo đã defend |
|---|---|---|---|---|
| 7.1 | Vì sao data thắng ở fixed-compute; pipeline order | ⬜ | | |
| 7.2 | Heuristic quality filter (Gopher/C4) | ⬜ | | |
| 7.3 | Learned quality classifier (DCLM signal design) | ⬜ | | |
| 7.4 | Dedup: exact-hash → MinHash/LSH S-curve | ⬜ | | |

## Chặng 5 · Gợi năng lực — post-training (M8)
### M8 · Post-training / RL (`algos/`, `rewards/`, `envs/`, `rollout/`)
| Bài | Concept | TT | Ngày | Neo đã defend |
|---|---|---|---|---|
| 8.1 | Vì sao post-training: lifecycle + objective-shift | ⬜ | | |
| 8.2 | SFT substrate: MLE + per-token loss masking | ⬜ | | |
| 8.3 | Policy-grad: REINFORCE→baseline→PPO-clip→GRPO→Dr.GRPO | ⬜ | | |
| 8.4 | RLVR grader r1_zero: format×correctness | ⬜ | | |
| 8.5 | Expert Iteration = STaR/RFT | ⬜ | | |
| 8.6 | DPO: contrastive, reward-model-free, Z(x) cancel | ⬜ | | |
| 8.7 | Rollout seam + train↔infer KL + mandatory logging | ⬜ | | |

## Chặng 6 · Kiến trúc frontier (M9)
### M9 · MoE · MLA · MTP (`moe.py`, `mla.py`, `model.py`)
| Bài | Concept | TT | Ngày | Neo đã defend |
|---|---|---|---|---|
| 9.1 | Vì sao sparse: decouple params khỏi FLOPs | ⬜ | | |
| 9.2 | Router + top-k gating + aux-loss-free bias | ⬜ | | |
| 9.3 | MLA: low-rank KV + weight-absorption identity | ⬜ | | |
| 9.4 | MTP: denser training signal + free draft head | ⬜ | | |

## Chặng 7 · PERF phục vụ model frontier — S1, S2, S5
### S1 · Serving substrate (`model.py`, `kernels/paged_decode_triton.py`)
| Bài | Concept | TT | Ngày | Neo đã defend |
|---|---|---|---|---|
| 1.0 | Roofline thesis: decode memory-bound, AI≈1 | ⬜ | | |
| 1.1 | BatchedKVCache: slab + write-then-mask (bài đầy đủ: `serving/01`) | ⬜ | | |
| 1.2 | KVCache 1-request + truncate() rollback | ⬜ | | |
| 1.3 | SlotKVCache base: write-then-mask, py/device mirror | ⬜ | | |
| 1.4 | PagedKVCache: block pool + block_table + CoW | ⬜ | | |
| 1.5 | PrefillView: offset-0 admission duck-typing | ⬜ | | |
| 1.6 | ChunkPrefillView: offset-continuation chunked prefill | ⬜ | | |
| 1.7 | Triton paged-decode kernel: online softmax qua block table | ⬜ | | |
### S2 · Serving engines (`sampling.py`, `serving/`, `mla.py`)
| Bài | Concept | TT | Ngày | Neo đã defend |
|---|---|---|---|---|
| 2.1 | Token-exact decode oracle | ⬜ | | |
| 2.2 | Serving metrics: TTFT/ITL/throughput/goodput | ⬜ | | |
| 2.3 | R0 baseline: sequential decode đo được | ⬜ | | |
| 2.4 | R3a static batch: đòn bẩy AI≈B | ⬜ | | |
| 2.5 | R3b continuous batching + ownership contract | ⬜ | | |
| 2.6 | R4.3 speculative decoding, lossless | ⬜ | | |
| 2.7 | R4.4 CUDA-graph decode: giết launch overhead | ⬜ | | |
| 2.8 | R4.5 MLA weight-absorption (serving) | ⬜ | | |
| 2.9 | R4.6 PD-disaggregation demo | ⬜ | | |
### S5 · Quantization (`quant/`)
| Bài | Concept | TT | Ngày | Neo đã defend |
|---|---|---|---|---|
| 5.1 | INT8: SQNR floor, 6 dB/bit, scale factor-out | ⬜ | | |
| 5.2 | Group-INT4: 2-nibble packing + group scaling | ⬜ | | |
| 5.3 | NVFP4 vs MXFP4: codec + two-level scaling | ⬜ | | |
| 5.4 | FP8 E4M3 KV: per-channel-K vs per-token-V | ⬜ | | |
| 5.5 | AWQ: activation-aware scaling | ⬜ | | |

## Chặng 8 · Đóng vòng & chứng minh (CAPSTONE — M10)
### M10 · Close-the-loop + F1–F9 ablations
| Bài | Concept | TT | Ngày | Neo đã defend |
|---|---|---|---|---|
| 10.1 | Close the loop: museum-vs-model gap + speedrun spine | ⬜ | | |
| 10.2 | Report card: oracle của vòng (bpb, MC, generative) | ⬜ | | |
| 10.3 | Model-size tiers + C=6ND cross-check | ⬜ | | |
| 10.4 | Ablation study AS derivation: F1–F9 | ⬜ | | |

---
*Tổng: 89 Bài (48 MODEL + 41 PERF) + Bài 0 meta.*
*📚 **Derivation docs: 16/16 pillar COMPLETE** (2026-07-14) — `docs/learning/derivations/` (9,782 dòng, verified). Đây là substrate re-learn đầy đủ cho CẢ 89 Bài.*
*✅ Teach-back (cold defense — mastery thật): 12 Bài (M1 1.1–1.4 + M2 2.1–2.6 + M3 3.1–3.2) · 🔄: 0 · ⬜: 77. Node teach-back kế: M3 Bài 3.3.*
*Con đường + vì sao thứ tự này: [`CURRICULUM.md`](CURRICULUM.md). Nội dung Bài: [`roadmap_model/`](roadmap_model/README.md) + [`roadmap/`](roadmap/README.md).*
