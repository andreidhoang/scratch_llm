# Frontier practice (2026) — deepening the CS336 from-scratch build

> **What this is.** A 2026 frontier-engineering layer on top of the five CS336 assignment guides.
> For each pillar it answers: *what does a senior research engineer at a frontier lab actually do
> here, and which modern techniques should the from-scratch build adopt?* It does **not** replace
> the fundamentals — the per-assignment guides + the official `../../../lectures/` scaffolds remain
> the spine. This layer **deepens** them.
>
> **Provenance.** Produced 2026-06-09 by a multi-agent pass: five senior-RE researchers (one per
> assignment) read the PDF + the clean guide + scaffold and web-researched current practice, each
> adversarially fact-checked by a second agent against primary sources. Of ~65 proposals: **56
> confirmed, 8 corrected, 0 rejected**, + the missing items the fact-checkers caught (folded in
> below). Every item from that pass is grounded in a dated source; corrections are shown inline
> as **✎ Fact-check (2026)**.
>
> **Later batch — GDM alignment (2026-06-16).** 5 items tagged **➕ *(GDM-alignment addition)***
> (knowledge distillation + its scaling law, INT4 weight/activation quant, MoE pipeline-prefill,
> TPU/XLA literacy) were added to align the build to the Vlad Feinberg / GDM interview. They are
> individually cited from standard literature + the interview, but were **not** run through the
> 2026-06-09 multi-agent fact-check — treat their citations as unverified until checked against the
> primary sources. A second 2026-06-16 batch tagged **➕ *(frontier-judgment addition)*** (multi-turn /
> agentic tool-use RL; the PEFT/LoRA cut-line) was harvested from a Vizuara-bootcamp triage under a
> first-principles rule — keep only what a 2026 frontier lab screens for, cut the GPU-poor / applied
> noise (LoRA-as-default, QLoRA/NF4, and fine-tune-vs-RAG were cut). Same caveat: cited, not fact-checked.

## How to use the three tiers

- **🟢 Modern default** — a small, high-value upgrade a 2026 from-scratch build *should* adopt as its
  default (cheap, teaches a fundamental). Build these into the assignment as you do it. *(GQA, RoPE,
  RMSNorm, SwiGLU are already in the repo — these are the next ones.)*
- **🔵 Build lab** — a clean, scoped, genuinely-buildable opt-in module/experiment. Do these *after*
  the assignment's core is green, to go deeper.
- **⚪ Know-it** — production-scale infra or active research you must be able to discuss in an
  interview but should **not** build from scratch first.

**Discipline (unchanged):** every build keeps the from-scratch fundamentals first, ships with its
correctness invariant as a test, and stays green-CI. No research-project scope-creep — these are
engineering skills, not a paper to ship.

**Reading the sources (a screened skill).** Each item is grounded in a dated paper — use them to practice
*citation-tree traversal*, the skill frontier labs say standard backend engineers most often lack: from a
root paper, walk who-cites-it forward, and triage a paper's worth in minutes (the claim, the method, the one
figure that matters) *without* reading cover-to-cover. Traversing humanity's existing edge is the precondition
for advancing it.

---


## A1 · Basics — substrate

> **What a senior RE actually does here (2026):**
>
> A senior RE on the pretraining-architecture/"basics" pillar in 2026 spends very little time inventing algorithms and most of it (a) babysitting runs and (b) porting deltas from the latest tech reports into the house stack. Concretely:
>
> - BABYSIT & TRIAGE. Watch loss curves, global grad-norm, per-layer activation/residual-stream RMS, attention-logit magnitude, and (for MoE) router entropy + per-expert load — live, on dashboards. The first-line triage tools are exactly the CS336 disciplines: loss-at-init ≈ log(vocab) catches head/embedding/mask bugs before compute is spent; overfit-one-batch isolates wiring bugs from data bugs; a fixed seed makes a spike reproducible. The failure modes chased are SILENT DIVERGENCE and LOW-PRECISION (bf16/fp8) BLOWUPS — a slow loss spike at step 40k, a NaN from an exploding attention logit — not novel research. This is why the 2026 stabilizer toolkit (QK-norm, output z-loss, depth-scaled/variance-controlled init, grad-clip) is bread-and-butter.
>
> - ABLATE AT A PROXY, THEN TRANSFER UP. Run small-scale architecture/optimizer ablations at a proxy width/depth (QK-norm on/off, pre-vs-peri-norm, init scaling, LR/schedule), read a coordinate-check or a short IsoFLOP, pick the winner, and transfer it to the big run. "Does this change keep activation RMS flat across width/depth?" is the daily question; the win must survive the scale-up.
>
> - PORT THE DELTA. Each new report (Qwen3, Gemma 3, OLMo 2, SmolLM3, DeepSeek-V3, Kimi K2) ships one or two concrete knobs. The job is to read it, decide if it is load-bearing or lab-specific, and port just the delta — e.g. "Gemma 3 dropped soft-capping for QK-norm; adopt QK-norm, skip soft-capping." Knowing who-uses-what (MLA = DeepSeek/Kimi only; most labs stayed on GQA) is the literacy that separates a competent RE from a hype-follower.
>
> - OWN THE TOKENIZER & SAMPLER ENDS. Profile tokenizer compression ratio and encode throughput; decide digit/whitespace handling (digit-grouping is now standard for arithmetic); keep the training-time sampler bit-identical to the inference engine's (temperature/top-p/min-p semantics) so eval-vs-train gaps are model-real, not config artifacts. Keep the eval harness honest (decontamination, fixed decoding).
>
> "Good" looks like: a green-CI, reproducible substrate where every primitive has a falsifiable invariant test, a run that can be debugged from its curves alone, and architecture choices each justified by a dated report or a proxy-scale ablation — not by vibes.


*Fact-check verdict: ACCURATE: Verified every enhancement against primary sources AND the actual target repo (/scratch_llm). All headline attributions check out — Qwen3 QK-norm + removed QKV-bias 'for stable training' (2505.09388); Gemma 3 explicitly replaced Gemma 2 soft-capping with QK-norm and uses 5:1 local:global with a 1024 window (2503.19786); MiniCPM originated WSD (2404.06395); PaLM z-loss=1e-4 on the output softmax (2204.02311); Llama 3 tokenizer's \\p{N}{1,3} (meta-llama/llama3); min-p (2407.01082, ICLR 2025) WITH the Schaeffer critique (2506.13681); GPT-OSS learned per-head attention sink (model card, Aug 5 2025); Peri-LN (2502.02732); muP/μTransfer (2203.03466) + u-μP (2407.17465); MLA decoupled-RoPE (DeepSeek-V2 2405.04434); MuonClip pretraining Kimi K2 1T on 15.5T tokens with zero loss spikes (2507.20534) + Muon at scale/Moonlight (2502.16982); YaRN/PI (2309.00071 / 2306.15595); SmolLM3 NoPE every 4th layer. The buildability claims are real: the repo genuinely has log_z=torch.logsumexp in cross_entropy (model.py:407, so output z-loss is byte-identical when c_z=0 and 'effectively free'), the router z-loss twin (moe.py:209), the GPT-2 PAT with ?\\p{N}+ (tokenizer.py:32, so digit-grouping is a one-line edit), tie_embeddings in ModelConfig (model.py:45), _top_p_filter and no min_p yet (sampling.py:38), and ADR-0003 sampler-parity. Only ONE real slip: the depth-init item mis-describes the current init as 'plain trunc-normal' (it is truncated-Xavier) and loosely cites GPT-2 '1/√N' vs the 1/√(2N) it actually builds — flagged NEEDS_FIX, but the load-bearing premise (no residual down-scaling present) is verified TRUE, so the enhancement is valid. CURRENT: Yes — sources span Jan-Jul 2025 flagships and the tiering tracks who-actually-uses-what (MLA = DeepSeek/Kimi only and correctly AWARENESS_ONLY; GQA the broad default; min-p flagged contested per the 2025 re-analysis; Muon explicitly NOT dismissed as 'marginal'). Two CONFIRMED items carry nuances worth recording: (a) z-loss as CORE is the weakest tiering since the proposal itself calls it near-inert at TinyStories/fp32 — but CORE is defensible because it is free (reuses the existing logsumexp), future-proofs the bf16/fp8 move, and completes the pair with the already-present router z-loss; (b) the peri-LN item lists OLMo 2 as an adopter — OLMo 2 actually uses output-only 'reordered norm' (Liu et al. 2021), not the both-sides sandwich the item describes, but the citation holds because the Peri-LN paper itself groups Gemma 2 + OLMo 2. FUNDAMENTALS-FIRST & FREE OF SCOPE-CREEP: Yes. Every item is CPU-buildable on TinyStories with a falsifiable invariant (loss-at-init ≈ log V, bounded logits, flat-vs-fanning RMS, degenerate-case sampler equivalence). The two items that could over-scope — full muP and Muon — are both correctly tiered (muP scoped to a coordinate-check toy lab with a kill-criterion; Muon AWARENESS_ONLY with 'own AdamW first'), so the proposal disciplines its own scope rather than smuggling research projects into the substrate. Net: ship it after the one depth-init wording/source fix; the main omission is an in-pillar cheap default (intra-document attention masking) that fits this set's ethos and should be added.*


### 🟢 Modern defaults — adopt these into the from-scratch build


#### QK-norm (RMSNorm on queries & keys before RoPE)

- **What:** Apply a small RMSNorm to the per-head query and key vectors (over head_dim) just after the Q/K projections and before RoPE, so the dot-product that feeds softmax cannot produce runaway-large logits.
- **Why (2026):** Load-bearing and now near-universal in the open-weight class: Qwen3 explicitly removed the old QKV-bias and added QK-Norm 'to ensure stable training'; Gemma 3 REPLACED Gemma 2's logit soft-capping with QK-norm; OLMo 2 uses it too. It is the cheapest fix for the #1 large-model failure mode (exploding attention logits / bf16 overflow) — the same failure Kimi K2's QK-Clip targets. The CS336 baseline has no logit-magnitude control, so this is the obvious next architectural default after GQA/RoPE/RMSNorm/SwiGLU.
- **Build:** model.py MultiHeadSelfAttention: add self.q_norm = RMSNorm(head_dim), self.k_norm = RMSNorm(head_dim); after reshaping to (B,H,S,head_dim) and BEFORE self.rope(...), do q = self.q_norm(q); k = self.k_norm(k). Gate behind a ModelConfig.qk_norm flag so it composes with GQA/KVCache (k_norm uses n_kv heads). Convention to build = RMSNorm over head_dim, pre-RoPE (Qwen3); note OLMo 2 pairs it with a post-norm block (a documented variant), so make placement a one-line choice, not a hardcode. Minimal version: just the two norms. INVARIANT (test_model): loss-at-init still ≈ log(vocab); and the max |attention logit| at init is materially smaller than the no-QK-norm baseline on the same random input (assert a bound).
- **Source:** Qwen3 Technical Report arXiv:2505.09388 (May 2025); OLMo 2 arXiv:2501.00656 (Jan 2025); Gemma 3 arXiv:2503.19786 (Mar 2025); original QK-norm: Henry et al. arXiv:2010.04245 (2020).
- **Cost:** CPU-buildable on TinyStories; ~1-2 h. Two RMSNorm modules + 2 lines in attention. Kill-criterion: if loss-at-init moves off log(vocab) the norm is on the wrong axis (normalize head_dim, not seq).

#### Depth-scaled residual init (1/√(2·n_layers) on output projections)

- **What:** Initialize the residual-writing projections (attention o_proj and FFN down-projection w2) with their std divided by √(2·n_layers), so the sum of residual branches does not blow the residual-stream variance up with depth.
- **Why (2026):** A genuine fundamental the repo is currently missing: model.py uses plain trunc-normal with no depth awareness, which is fine at 4 layers but wrong as a default and a known cause of deep-model instability. The 1/√(2N) rule is the canonical GPT-2 trick, carried through GPT-NeoX and the Llama lineage; it teaches the residual-stream-variance mental model that every later stability tool (z-loss, peri-norm) builds on. Load-bearing for depth; cheap and exactly the kind of 'know your init' question a lab probes.
- **Build:** model.py: give _trunc_normal_linear_weight an optional residual_layers: int|None; when set, scale std by 1/math.sqrt(2*residual_layers). In MultiHeadSelfAttention pass n_layers for o_proj; in SwiGLU pass it for w2; plumb cfg.n_layers through. Minimal version: scale only o_proj and w2. INVARIANT: at init, measure each block's output-vs-input residual RMS — with the scaling it stays roughly flat across the stack; without it, it grows with layer index (write the per-block RMS as the falsifiable number first).
- **Source:** GPT-2 (Radford et al. 2019, residual scaling 1/√N); GPT-NeoX-20B arXiv:2204.06745 (2022); discussed in Small-scale proxies for large-scale Transformer training instabilities arXiv:2309.14322 (2023).
- **Cost:** CPU-buildable; ~1 h. Pure init change, zero runtime cost. Kill-criterion: if loss-at-init drifts from log(vocab), the scaling was applied to the wrong matrices (it must be only the residual-writing projections).

> ✎ **Fact-check (2026):** Reword the repo claim to: 'model.py uses truncated-Xavier init (std=√(2/(in+out))) with NO depth/residual awareness — the residual-writing projections (o_proj, w2) are not down-scaled, which the verified absence of any residual_layers/√(2·n_layers) term confirms.' The load-bearing premise (no 1/√(2N) residual scaling present) is TRUE — verified by grep — so the enhancement itself stands as a valid CORE default. Source: keep GPT-2 (Radford et al. 2019) for the residual-scaling idea but state the built rule as 1/√(2N) per GPT-NeoX-20B (arXiv:2204.06745) and 'Small-scale proxies' (arXiv:2309.14322), which is exactly what the code change implements.

#### WSD (warmup–stable–decay) learning-rate schedule

- **What:** A three-phase LR schedule: linear warmup → long constant 'stable' phase at peak LR → short final decay to ~0 (often linear) over the last ~10-20% of steps. Unlike cosine, it does not need the total step count fixed up front.
- **Why (2026):** The de-facto modern default for from-scratch runs and the natural partner to A3 scaling laws: you can branch a run, decay early, and read the result without committing the horizon — and the 'stable-phase checkpoint' is reusable for continual pretraining. Adopted by MiniCPM (which introduced it), SmolLM2/SmolLM3, and DeepSeek-V3. The repo only has cosine; WSD is a small, high-value branch that teaches why the loss stays high then drops sharply in decay (the 'river-valley' picture).
- **Build:** optim.py: add wsd_lr(step, max_lr, warmup_steps, stable_steps, decay_steps, min_lr=0.0) — linear ramp in warmup, flat max_lr through stable, linear (or 1-sqrt) decay to min_lr in the last decay_steps. Wire a schedule='wsd'|'cosine' switch in TrainConfig. Match SmolLM3's recipe as the default shape (2000 warmup, decay over final ~10%). INVARIANT: returns exactly max_lr for every step in the stable window; equals min_lr at the end; monotone in each phase; a unit test pins the three phase boundaries.
- **Source:** MiniCPM arXiv:2404.06395 (2024, WSD originator); SmolLM3 (HuggingFace blog, 2025: AdamW β=(0.9,0.95), wd 0.1, WSD 2000 warmup + linear decay final 10%); DeepSeek-V3 arXiv:2412.19437 (2024).
- **Cost:** CPU-buildable; ~0.5 h. Pure scalar function + one config switch. Kill-criterion: none — it is a closed-form schedule; just unit-test the phase boundaries.

#### Output z-loss (final-softmax logit regularizer)

- **What:** Add a tiny auxiliary penalty on the LM head's log-partition: c_z · mean(logsumexp(logits)²), with c_z = 1e-4, to stop the output logits from drifting to large magnitudes that destabilize low-precision softmax.
- **Why (2026):** Honest scope: at the TinyStories/17M/fp32 scale this run uses it is near-inert — its value is at scale and in bf16/fp8, where it is a standard anti-divergence tool (PaLM applied it to the output softmax; ST-MoE generalized it to the router; OLMo 2 uses it). It belongs as a CORE default because it is almost free and it makes the model robust when you later move to low precision — and the repo already has the sibling router z-loss in moe.py, so adding the output-side twin closes the pair. Pedagogically it cements the logsumexp/partition-function view from cross-entropy.
- **Build:** model.py cross_entropy already computes log_z = torch.logsumexp(logits, -1) — z-loss is literally c_z * log_z.pow(2).mean(), so reuse that value (zero extra compute). Return (ce, z_loss) or fold c_z*zloss into the training loss in train.py behind a TrainConfig.z_loss_coef (default 1e-4; 0 disables). INVARIANT: with c_z=0 the loss is byte-identical to today; loss-at-init still ≈ log(vocab); and across a few steps logsumexp(logits) stays bounded near log(vocab) instead of drifting upward (log it).
- **Source:** PaLM arXiv:2204.02311 (Chowdhery et al. 2022, z-loss 1e-4 on output softmax); ST-MoE arXiv:2202.08906 (2022, router z-loss); OLMo 2 arXiv:2501.00656 (2025).
- **Cost:** CPU-buildable; ~0.5 h. Reuses existing logsumexp — effectively free. Kill-criterion: if c_z=0 changes any number, the coefficient is mis-plumbed.

### 🔵 Build labs — scoped, opt-in, after the core is green


#### Digit-grouping pre-tokenizer (\p{N}{1,3})

- **What:** Change the BPE pre-tokenization regex so runs of digits are split into groups of at most three (the \p{N}{1,3} alternative), instead of letting BPE merge whole numbers into idiosyncratic tokens.
- **Why (2026):** A concrete, dated tokenizer-design default: Llama 3's tiktoken pattern uses exactly \p{N}{1,3}, and digit-aware splitting is standard across modern tokenizers because arbitrary number tokens hurt arithmetic and number understanding. It is a clean way to show that tokenizer design is a modeling decision (the token axis is the model's output axis), with a trivially testable behavior — and it spreads the enhancement set onto the tokenizer pillar, not just the model.
- **Build:** tokenizer.py: the GPT-2 PAT used for pre-tokenization currently has \p{N}+; swap that clause for \p{N}{1,3} (Llama 3's pattern), keeping everything else. Make it a from_files/train option so the original GPT-2 behavior stays available. INVARIANT: pre-tokenizing '12345' yields ['123','45'] (or per the regex, max-3 groups), and decode(encode(s)) still round-trips on a numeric-heavy fixture; assert no pre-token contains >3 consecutive digits.
- **Source:** Llama 3 tokenizer.py (meta-llama/llama3, regex contains \p{N}{1,3}); GPT-4/tiktoken cl100k uses the same digit-grouping; Qwen2.5 tokenizer report (2024).
- **Cost:** CPU-buildable; ~1 h including a re-trained tiny BPE to see the vocab change. One regex edit. Kill-criterion: round-trip test must still pass.

#### Min-p sampling (with the honest critique)

- **What:** A truncation sampler that keeps only tokens whose probability ≥ min_p · p_max (a fraction of the top token's prob), so the kept set is wide when the model is unsure and narrow when it is confident — a confidence-adaptive alternative to top-p.
- **Why (2026):** Worth building and worth being honest about. Min-p is implemented in essentially every deployed inference engine (vLLM/HF/llama.cpp) and is a useful contrast to top-p for understanding truncation samplers. BUT a 2025 critical re-analysis found the original paper's evidence does not robustly support its quality/diversity superiority over a well-tuned top-p — so present it as 'know it, it's everywhere, its edge over top-p is contested,' not as a strict upgrade. Good lesson in reading sampling claims skeptically.
- **Build:** sampling.py: add a _min_p_filter mirroring _top_p_filter — compute probs, p_max = probs.max(-1), keep mask probs >= min_p*p_max, renormalize; expose min_p in SamplingParams (compose after temperature, consistent with ADR-0003 sampler-parity so train and infer agree). INVARIANT: min_p=0 reduces to plain temperature sampling; min_p→1 collapses to greedy (argmax); kept-set is always a superset-by-confidence (a fixture asserts the set membership).
- **Source:** Min-p: Nguyen et al. arXiv:2407.01082 (ICLR 2025); critique: Schaeffer et al. 'Min-p, Max Exaggeration' arXiv:2506.13681 (2025).
- **Cost:** CPU-buildable; ~1 h. One filter function. Kill-criterion: degenerate cases (min_p=0 / →1) must equal plain-sampling / greedy or the threshold math is wrong.

#### Learned attention sink (per-head softmax denominator bias)

- **What:** Add one learnable scalar per attention head into the softmax denominator (an implicit always-present 'sink' logit), letting a head attend to effectively nothing — equivalent to a virtual sink token without a real one.
- **Why (2026):** A small, current architectural knob: GPT-OSS (2025) ships exactly this learned per-head sink, building on the StreamingLLM observation that models park excess attention on early tokens. It improves stability and long-context/streaming behavior and is now in HuggingFace and TensorRT-LLM. As a from-scratch exercise it makes the 'softmax must sum to 1, so where does spare attention go?' phenomenon concrete and falsifiable.
- **Build:** model.py: give MultiHeadSelfAttention a sink = nn.Parameter(zeros(n_heads)); in scaled_dot_product_attention (or a variant), after computing row-max-stabilized scores, include exp(sink - rowmax) as an extra term in the denominator only (not in the numerator/values). Keep behind a config flag. INVARIANT: with sink fixed at -inf (or the param absent) the output is bit-identical to plain attention; with a finite sink, each row's attention weights over real tokens sum to <1 by the sink's share (assert the deficit equals the sink term).
- **Source:** GPT-OSS model card / release (OpenAI, 2025, learnable per-head attention sink); StreamingLLM: Xiao et al. arXiv:2309.17453 (ICLR 2024); 'When Attention Sink Emerges' arXiv:2410.10781 (2024).
- **Cost:** CPU-buildable; ~1-2 h. One parameter + denominator term. Kill-criterion: sink=-inf must recover plain attention exactly, else the term leaked into the numerator.

#### Sliding-window (local) attention with periodic global layers

- **What:** Restrict most layers' causal attention to a fixed window of the last W tokens (a banded mask), interleaving occasional full-'global' layers, to cut the KV-cache and attention cost at long context.
- **Why (2026):** A live efficiency default: Gemma 3 uses a 5:1 local:global ratio with a 1024-token window specifically to shrink the long-context KV cache; Mistral popularized sliding-window attention. It is a natural extension of the causal mask the repo already builds and directly teaches the KV-cache economics that pair with the existing GQA work — a clean opt-in module rather than a rewrite.
- **Build:** model.py: in the mask construction add a window option — mask = (k_pos <= q_pos) & (k_pos > q_pos - W); add ModelConfig.sliding_window: int|None and a layer pattern (e.g. every 6th layer global). Compose with the existing KVCache by trimming cached K/V older than the window. INVARIANT: with W >= context_length the result is identical to full causal attention (assert allclose on logits); the banded mask still prevents any future-token leak (causal-no-leak test holds).
- **Source:** Gemma 3 arXiv:2503.19786 (2025, 5:1 local:global, window 1024); Mistral 7B arXiv:2310.06825 (2023, sliding-window attention); Longformer arXiv:2004.05150 (2020).
- **Cost:** CPU-buildable; ~2-3 h (mask is easy; correct KV-cache trimming is the fiddly part). Kill-criterion: W≥seq must equal full attention exactly, else the band offset is off by one.

#### Normalization placement: pre-LN vs Peri-LN / sandwich-norm (residual-variance lab)

- **What:** Make the block's norm placement a config choice: standard pre-norm (x + Mod(Norm(x))) vs Peri-LN / 'sandwich' (x + Norm(Mod(Norm(x)))), which adds a norm on each sublayer's OUTPUT before the residual add.
- **Why (2026):** A real, recently-adopted knob: Gemma 2 and OLMo 2 'silently' added output normalization, and the Peri-LN analysis shows why — it turns the residual-stream variance growth from exponential (plain pre-LN) into roughly linear with depth, improving deep-model stability. Building it as an A/B makes the residual-stream-variance story (the same one behind depth-scaled init and z-loss) measurable. Replaces the much thinner 'weight tying' idea, whose flag already exists in ModelConfig — though folding a loss-at-init-with-tying check into this lab is a nice add.
- **Build:** model.py TransformerBlock: add ModelConfig.norm_placement = 'pre'|'peri'; for 'peri' wrap each sublayer output in an extra RMSNorm before the residual add (an out_norm per sublayer), keeping pre-norm as default. INVARIANT (the lab's point): at init, log per-block residual RMS across the stack and show pre-LN grows ~exponentially while peri-LN stays ~linear; loss-at-init ≈ log(vocab) for both. (Optional tying check: with tie_embeddings=True, lm_head.weight IS token_emb.weight and loss-at-init still ≈ log(vocab).)
- **Source:** Peri-LN: Kim et al. arXiv:2502.02732 (2025, names Gemma 2 & OLMo 2 as adopters); Gemma 2 arXiv:2408.00118 (2024, sandwich norm); OLMo 2 arXiv:2501.00656 (2025).
- **Cost:** CPU-buildable; ~2 h. Extra norms + the variance-measurement script (which is the deliverable). Kill-criterion: both placements must hit loss-at-init ≈ log(vocab); if peri-LN doesn't flatten the RMS curve, the output norm is misplaced.

#### muP minimal coordinate-check (width-transfer parameterization)

- **What:** Implement just the maximal-update (muP) scaling rules for init std and per-layer LR as functions of width, and verify a 'coordinate check': per-layer activation magnitudes stay ~constant as you widen the model (so an LR tuned small transfers to large).
- **Why (2026):** Frontier labs use muP/μTransfer to tune hyperparameters on a tiny model and zero-shot transfer to the big one (Cerebras, parts of the Qwen/Yi lineage). FULL muP is too much to own from scratch and is tiered AWARENESS — but the buildable, high-value slice is the coordinate check on a toy width sweep: it makes 'what does width-stable feature learning even mean?' concrete. Scope it as a tiny lab, not a framework.
- **Build:** New tools/mup_coord_check.py using the existing TransformerLM: build models at widths d_model ∈ {128,256,512} applying muP rules (input/output/hidden init and LR scaled by width ratios), run a few steps on random data, and plot the mean abs activation per layer vs width. INVARIANT (coordinate check): under muP the per-layer activation magnitudes are ~flat across widths (overlapping curves); under standard parameterization they fan out with width — that divergence IS the falsifiable result.
- **Source:** Tensor Programs V / μTransfer: Yang et al. arXiv:2203.03466 (2022); u-μP (unit-scaled muP): Blake et al. arXiv:2407.17465 (2024).
- **Cost:** CPU-buildable on toy widths; ~3-4 h (the scaling bookkeeping is the work). Kill-criterion: if muP curves don't flatten relative to standard, a scaling exponent is wrong — that's the whole signal, so debug it, don't ship it broken.

#### Intra-document attention masking (packed-sequence document masking) ➕ *(fact-check addition)*

- **What:** When documents are concatenated to fill the context window, restrict attention so a token only attends within its own document — a block-diagonal extension of the causal mask.
- **Why (2026):** A real, widely-adopted training default (Llama-3 / OLMo-style packing): cross-document attention is a subtle leak that measurably hurts. Fundamentals-first and falsifiable — your causal-no-leak invariant now also becomes document-no-leak.
- **Build:** model.py — pass a per-token document-id segment vector into attention and AND a block-diagonal segment mask with the causal mask. Test: a token cannot attend across a document boundary (extend the causal-no-leak test).
- **Source:** Zhao et al., "Analysing the Impact of Sequence Composition on LM Pre-Training", arXiv:2402.13991 (2024).
- **Cost:** CPU-buildable; ~1–2 h.

### ⚪ Know-it — discuss in interviews, do NOT build from scratch


#### Multi-head Latent Attention (MLA)

- **What:** An attention variant that projects K/V into a small shared latent vector that is cached instead of full per-head K/V (with a decoupled RoPE component), shrinking the KV cache far more than GQA while aiming to keep quality.
- **Why (2026):** Be able to discuss, do NOT build from scratch here. MLA is real and load-bearing — but only at DeepSeek (V2/V3) and Kimi (K2/Linear); essentially every other 2026 open-weight flagship (Llama 4, Qwen3, Gemma 3, gpt-oss, OLMo 2) stayed on GQA. So in an interview the correct framing is 'MLA is DeepSeek/Kimi's KV-compression bet; GQA is the broad default,' plus the tradeoff (extra up-projection compute and a fiddly decoupled-RoPE path vs. a much smaller cache). The repo's GQA already covers the buildable KV-economics lesson.
- **Build:** No build. Understand only: be able to whiteboard how MLA caches a latent c_KV instead of K,V, why RoPE has to be decoupled (you can't rotate a position-free latent), and the cache-size-vs-compute tradeoff vs GQA — and to say honestly that most labs did not adopt it.
- **Source:** DeepSeek-V2 arXiv:2405.04434 (2024, introduces MLA); DeepSeek-V3 arXiv:2412.19437 (2024); Kimi K2 arXiv:2507.20534 (2025).
- **Cost:** No build / awareness only. Reading + ability to whiteboard the latent-cache and decoupled-RoPE tradeoff.

#### Muon / MuonClip optimizer (orthogonalized momentum)

- **What:** An optimizer for 2-D hidden weights that orthogonalizes the momentum update via a cheap Newton–Schulz iteration (equalizing the update's singular values); MuonClip adds QK-Clip to cap attention logits during training.
- **Why (2026):** Be able to discuss, and be accurate: do NOT call it 'marginal.' Muon dominates the nanoGPT/CIFAR speedruns, and as MuonClip it was used to pretrain the 1T-parameter Kimi K2 on 15.5T tokens with zero loss spikes (Moonshot claims ~2× token efficiency vs AdamW) — the first Muon-family optimizer proven at 1T scale. The honest framing: AdamW is still THE default and the edge-over-a-well-tuned-AdamW at general scale is still debated, but Muon is a serious, validated contender, not a toy. (Nice connection: MuonClip's QK-Clip targets the same exploding-attention-logit failure as the CORE QK-norm item.)
- **Build:** No build required for this assignment. If curious it is small (a ~10-line Newton–Schulz orthogonalization wrapped around SGD-momentum, applied to 2-D weights only, AdamW for embeddings/norms) — but own AdamW first; treat Muon as a comparison you can describe, not a from-scratch substrate piece.
- **Source:** Muon: Keller Jordan, 'Muon: An optimizer for hidden layers' (kellerjordan.github.io, 2024) + KellerJordan/Muon repo; Kimi K2 / MuonClip arXiv:2507.20534 (2025); Moonlight/Muon-at-scale arXiv:2502.16982 (2025).
- **Cost:** No build (awareness). Note: it IS CPU-buildable later (~10-line Newton–Schulz) if you want the comparison; kill-criterion if you ever do — it must beat a well-tuned AdamW on overfit-one-batch wall-clock, else it's not worth the complexity.

#### Long-context RoPE scaling (Position Interpolation / NTK / YaRN)

- **What:** Post-hoc methods to extend a model's context beyond its trained length by rescaling RoPE frequencies — linear position interpolation, NTK-aware base scaling, and YaRN's per-frequency interpolation with attention-temperature correction.
- **Why (2026):** Be able to discuss. These are the standard recipes for turning a 4k/8k model into a 32k–128k one (Llama, Qwen, Code Llama all use a variant), and the concepts (which RoPE frequencies to interpolate vs extrapolate, why high-freq dims break first) are exactly what a long-context interview probes. But it is a fine-tuning/serving-time concern layered on the RoPE the repo already owns, not a from-scratch substrate build — and validating it needs long-context evals you won't run on TinyStories.
- **Build:** No build. Understand: PI divides positions by a scale factor; NTK scales the RoPE base θ; YaRN interpolates per-frequency and adds a softmax temperature. Be able to explain why naive extrapolation fails and the high-vs-low-frequency-dim intuition. (If ever demoed, it's a one-line change to RoPE's inv_freq — but it's awareness here.)
- **Source:** Position Interpolation: Chen et al. arXiv:2306.15595 (2023); YaRN: Peng et al. arXiv:2309.00071 (2023); NTK-aware scaling (bloc97, 2023, community).
- **Cost:** No build / awareness only. Reading + ability to explain the frequency-interpolation intuition; real validation needs long-context evals (out of scope at this scale).

#### NoPE / selective position-encoding removal

- **What:** Dropping explicit positional encoding (no RoPE) on some or all layers, relying on the causal mask alone to encode order — used selectively (e.g. every Nth layer) to improve length generalization.
- **Why (2026):** Be able to discuss. SmolLM3 omits RoPE on every 4th layer (NoPE) and reports better long-context behavior, and there is a line of work showing decoder-only models can learn position from the causal mask alone. It is a cheap, real 2026 architecture knob worth knowing — but it is a small ablation/awareness item, not a CORE default (RoPE-everywhere remains standard), and its payoff is a length-generalization claim best evaluated at longer context than this assignment trains.
- **Build:** No build needed; if demoed it's trivial (skip the rope(...) call on selected layers behind a config flag). The point is awareness: explain the NoPE-vs-RoPE length-generalization tradeoff and why a mix can beat all-RoPE.
- **Source:** SmolLM3 (HuggingFace blog, 2025, NoPE every 4th layer); 'The Impact of Positional Encoding on Length Generalization' Kazemnejad et al. arXiv:2305.19466 (2023).
- **Cost:** No build / awareness only. Trivial to toggle if ever wanted; understanding the length-gen tradeoff is the deliverable.

#### Multi-Token Prediction (MTP) ➕ *(fact-check addition)*

- **What:** Auxiliary heads (sharing embedding/head) that predict tokens 2..D ahead during pretraining, densifying the training signal; the heads double as a self-speculative-decoding draft at inference.
- **Why (2026):** The single most-discussed 2024–25 training-objective delta (DeepSeek-V3). Heavier than a cheap architectural knob (it changes the objective + adds heads), so know it and its inference-draft link rather than building it from scratch first.
- **Build:** Awareness: adds auxiliary heads + loss terms; note the connection to A2 speculative decoding (the MTP heads ARE a draft model).
- **Source:** DeepSeek-V3, arXiv:2412.19437 (Dec 2024).
- **Cost:** Awareness (objective-level change).

---


## A2 · Systems

> **What a senior RE actually does here (2026):**
>
> A senior systems RE lives in a tight measure-fix-remeasure loop and reports in numbers, never adjectives. The loop: (1) Run a step under a profiler — Nsight Systems for the cross-kernel timeline + comms gaps, torch.profiler/Kineto for per-op CPU/GPU traces, always with cuda.synchronize() around timed regions because CUDA is async. (2) Classify the bottleneck. For a single kernel: compute arithmetic intensity (FLOPs/byte) and place it on the roofline vs the GPU ridge point — below ridge = memory-bound (fuse, reduce HBM traffic, raise tile reuse), above = compute-bound (need a better MMA schedule / higher precision throughput). For a distributed step: is the timeline comms-bound (all-reduce/all-gather/reduce-scatter/all-to-all not hidden under compute) or compute-bound? (3) Fix the dominant term: write or fuse a Triton/CUDA/CuTeDSL kernel, change the activation-recompute policy, fire collectives earlier from grad-ready hooks to overlap with backward, shard optimizer/grads/params (ZeRO/FSDP), or drop precision (BF16→FP8) on the GEMMs. (4) Re-profile and confirm the gap closed. (5) Numerics gate every change: loss-curve parity vs the fp32/bf16 baseline, no FP8/FP4 overflow→NaN, and train-engine vs serve-engine logit parity (the kl_train_infer check) when the inference kernels/precision differ. 'Good' is a target number: % MFU (Model FLOPs Utilization, ~40–55% dense on H100 is strong), tokens/s/GPU, $/M-tokens for serving, p99 latency under an SLO, and KV-cache hit rate. The failure modes they hunt daily: silent low-precision overflow/underflow (a single layer's amax blows the FP8 scale → loss spike or NaN), comms bubbles in the Nsight timeline (the all-gather that didn't prefetch), activation-memory OOM at the largest microbatch/sequence, KV-cache fragmentation/eviction thrash under load, and train-vs-serve drift (the served policy ≠ the trained one because the engines fuse softmax/PV reductions differently or quantize the KV cache). On the inference side specifically, the P&L is wall-clock and $/token: the daily levers are batching policy (continuous batching, chunked prefill), KV-cache management (paging, quantization), speculative decoding acceptance rate, and prefill/decode resource split. The deliverable artifact is almost always a profile + a before/after number with a one-line root cause, not prose.


*Fact-check verdict: Accurate, current, and impressively disciplined — this is a fundamentals-first proposal that survives adversarial fact-checking almost intact. I verified every arXiv ID, repo, and dated 2026 source against primary sources, including the high-hallucination-risk recent ones, and they all resolve correctly: arXiv:2603.07685 (MoE Megatron Core, Mar 2026), the vLLM FP8-KV blog (Apr 22 2026), and the PyTorch FlexAttention+FA4 blog (Mar 5 2026) are all real and correctly described. Two load-bearing quantitative claims are not just plausible but EXACT: selective recomputation is '70% (and 65%) activation-memory saved for only 2.7% (and 1.6%) FLOPs' verbatim from Korthikanti et al., and the FSDP2 '~7% lower per-GPU memory / ~1.5% higher throughput vs FSDP1' is reported precisely in the TorchTitan paper and correctly attributed to it (the PyTorch docs' softer 'comparable throughput' line is a different source, so this is not a contradiction). The CORE_MODERN_DEFAULT tiering is honest — selective recompute and per-parameter FSDP2 are genuinely the cheap modern defaults over what CS336 already teaches — and the AWARENESS_ONLY gating of FA3/FA4, Transformer Engine FP8/FP4, PD-disaggregation, and 4D/5D+EP correctly avoids research-project scope-creep while keeping the buildable CPU/gloo equivalence-test labs (ring-CP, TP, paged-KV, speculative, fake-quant FP8) that carry the actual mastery. The single defect is a narrow stat error: KIVI's '~8x batch / ~2x throughput' should read '~4x batch / 2.35-3.47x throughput,' which slightly overstates the batch win and understates the throughput win — a footnote-level fix, not a structural problem. The only real omission is MLA/FlashMLA, the KV-compression lever frontier serving stacks ship and the proposal's KV section conspicuously lacks. Net: ship it after correcting the KIVI numbers and adding an MLA awareness item (ideally with a small latent-KV buildable lab).*


### 🟢 Modern defaults — adopt these into the from-scratch build


#### Selective activation recomputation (store-cheap / recompute-expensive)

- **What:** Instead of full-layer checkpointing (save only the block input, recompute everything) or the assignment's O(sqrt N) recursive strategy, selectively store the activations that are cheap to store but expensive to recompute (e.g. the attention softmax/dropout intermediates) and recompute the rest. This is the policy Megatron/TorchTitan use by default.
- **Why (2026):** Load-bearing. Full recomputation costs ~30%+ extra FLOPs; the CS336 assignment teaches the full/O(sqrt N) lever but the modern default is selective: ~70% activation-memory reduction for only ~2.7% recompute FLOPs on GPT-3-scale (Korthikanti et al.). It is the cleanest 'next default' over what the assignment already covers, and it forces the learner to reason per-tensor about the recompute-vs-store cost in bytes and FLOPs, which is the exact mental model used to size a real training run.
- **Build:** Extends the gradient_checkpointing deliverable in model.py / the TransformerBlock. Build a `selective_checkpoint` wrapper using torch.utils.checkpoint with a custom `context_fn` / a SAC (Selective Activation Checkpointing) policy that decides per-op whether to save or recompute (PyTorch exposes `torch.utils.checkpoint.create_selective_checkpoint_contexts`). Minimal buildable version is CPU-buildable on the toy model: a policy that saves matmul outputs but recomputes the (cheap) elementwise/softmax. Invariant (correctness test): gradients with selective checkpointing must equal eager gradients to fp tolerance on a fixed-seed batch (allclose on every .grad), and peak-activation memory measured via torch.cuda.memory must sit strictly between full-recompute and no-recompute. Predict the per-block activation MiB first, then verify.
- **Source:** Korthikanti et al., 'Reducing Activation Recomputation in Large Transformer Models', arXiv:2205.05198 (May 2022); PyTorch SAC API in torch.utils.checkpoint (docs.pytorch.org, 2024–2025); used as default in TorchTitan, arXiv:2410.06511 (2024).
- **Cost:** CPU-buildable for the correctness test (gradient + memory invariant on the toy model); ~0.5 day on top of the existing checkpointing work. GPU only needed to reproduce the MFU/throughput number. Kill criterion: if selective grads don't match eager, the recompute boundary is wrong — stop and bisect the policy.

#### FSDP2 per-parameter (DTensor dim-0) sharding as the FSDP design

- **What:** Build the FSDP deliverable using per-parameter sharding (each parameter chunked on dim-0 across ranks, represented as a DTensor) rather than FSDP1's FlatParameter (flatten+concat a group of tensors, then chunk the flat buffer). Same ZeRO-3 communication pattern (all-gather params for fwd/bwd, reduce-scatter grads), different sharding unit.
- **Why (2026):** This is the *design choice* the from-scratch FSDP should adopt, because FSDP1's FlatParameter is now legacy: PyTorch's `fully_shard` (FSDP2) ships per-parameter DTensor sharding and TorchTitan uses it as the default 1D parallelism. Per-parameter sharding is what makes FSDP compose with tensor/pipeline parallelism (you can shard an individual weight two ways), gives ~7% lower per-GPU memory and ~1.5% higher throughput vs FSDP1, and yields a sharded state_dict with no extra communication. Building the assignment's FSDP this way teaches the strictly more modern abstraction at the same effort. Honest caveat: for a 2-rank toy the memory delta is invisible; the value is the abstraction and interview currency, not a perf win at this scale.
- **Build:** Shapes the `get_fsdp` / `fsdp_on_after_backward` / `fsdp_gather_full_params` deliverables in utils/. Instead of a flat buffer, store each parameter as its own dim-0 shard (rank r holds rows [r*n/W : (r+1)*n/W]); forward pre-hook all-gathers the full param, forward post-hook frees it; backward reduce-scatters the grad so each rank keeps only its shard's grad in fp32 (master), with bf16/fp16 compute weights. The official scaffold's test IS the oracle: test_fsdp_correctness asserts the FSDP-trained params (after fsdp_gather_full_params) equal a non-parallel model step-for-step (atol 1e-6 fp32 / 1e-4 fp16), and test_fsdp_gradient_sync asserts grad.shape==data.shape and grad.dtype==fp32. Run the 2-rank gloo equivalence ×5. CPU/gloo-buildable in full.
- **Source:** PyTorch `torch.distributed.fsdp.fully_shard` docs (2.7–2.9, 2025); HuggingFace 'FSDP1 vs FSDP2' concept guide (2025); TorchTitan, arXiv:2410.06511 (2024). Original ZeRO-3: Rajbhandari et al., arXiv:1910.02054 (2019).
- **Cost:** CPU/gloo-buildable and fully covered by the scaffold's existing FSDP tests; this reframes effort the assignment already requires, +~0.25 day for the per-parameter shard bookkeeping vs flat. No extra GPU. Kill: if the 2-rank gathered params drift from the single-process baseline, the all-gather/reduce-scatter boundary or the fp32-master/bf16-compute split is wrong.

### 🔵 Build labs — scoped, opt-in, after the core is green


#### Paged KV-cache (block table) extension of the contiguous KV-cache

- **What:** Replace the contiguous per-sequence KV-cache with a paged one: KV stored in fixed-size blocks, a per-sequence block table mapping logical positions to physical blocks. This is the PagedAttention idea that lets a server pack many sequences without per-request contiguous reservation.
- **Why (2026):** Paging is the single biggest serving memory win — it nearly eliminates KV fragmentation and is the foundation under vLLM/SGLang/TensorRT-LLM. For a from-scratch learner it is the on-ramp to understanding why serving throughput is KV-bound and how block tables enable prefix sharing and copy-on-write. Honest scope: the production kernel (a Triton/CUDA paged-attention gather) is AWARENESS; the *block-table indexing logic and its equivalence to contiguous attention* is genuinely buildable and is the real lesson.
- **Build:** Extends the existing `KVCache` in model.py and the decode path. Build a `PagedKVCache`: allocate KV in blocks of size B (e.g. 16 tokens), keep a block_table per sequence, and in cache-aware attention gather K/V via the block table before the SDPA/oracle call. Minimal version is pure-PyTorch and CPU-buildable. Invariant (the oracle): for a fixed prompt+continuation, paged decode logits == contiguous-KV decode logits to fp tolerance (the existing 'cached == recompute' test, now 'paged == contiguous'); add a fragmentation test — N short sequences fit in paged memory that contiguous reservation would OOM. Reuse the benchmarking_script harness to report tokens/s.
- **Source:** Kwon et al., 'Efficient Memory Management for LLM Serving with PagedAttention' (vLLM), arXiv:2309.06180 (SOSP 2023); vLLM paged-attention docs (docs.vllm.ai, 2024–2026).
- **Cost:** CPU-buildable for the block-table logic + the paged==contiguous oracle; ~0.5–1 day. The fast gather kernel needs a GPU and is explicitly out of scope. Kill: if paged decode logits ever diverge from contiguous, the block-table position mapping or the gather order is wrong.

#### Lossless speculative decoding (n-gram / tiny-draft) on the rollout seam

- **What:** Accelerate decode by proposing K tokens cheaply (a small draft model or an n-gram/prompt-lookup proposer), verifying them in one target forward pass, and accepting the longest prefix that matches the target distribution. Greedy speculative output is bit-identical to plain greedy — it only changes speed.
- **Why (2026):** Speculative decoding is standard in every serving stack (vLLM, SGLang, TensorRT-LLM) and EAGLE-3 reaches 3–6.5x with ~80% acceptance — the frontier reference. The from-scratch value is the *verification/acceptance algorithm* and the losslessness invariant (the modified-rejection-sampling correction that keeps the output distribution exact). Honest scope: training an EAGLE-3 draft head is AWARENESS; the simple n-gram/prompt-lookup proposer + exact verification is fully buildable on infra this repo already has, and teaches the core idea.
- **Build:** Sits on the rollout/ + sampling.py seam (generate_with_logprobs already exposes per-token logprobs). Build a `speculative_generate`: proposer emits K candidate tokens (prompt-lookup n-gram is the zero-extra-model version), one target forward scores positions p..p+K, accept via greedy match (or the rejection-sampling rule for temperature>0). Invariant (the oracle): with greedy decoding, speculative output token IDs are IDENTICAL to target-only greedy on a fixed prompt (lossless); for sampling, accepted-token empirical distribution matches target within tolerance over many seeds. Log mean acceptance length (the speedup proxy). CPU-buildable with the existing TransformerLM as both target and (smaller-config) draft.
- **Source:** Leviathan et al., 'Fast Inference from Transformers via Speculative Decoding', arXiv:2211.17192 (ICML 2023); Chen et al., arXiv:2302.01318 (2023); EAGLE-3, arXiv:2503.01840 (Mar 2025) as the frontier draft-head reference.
- **Cost:** CPU-buildable; ~0.5–1 day. No GPU needed for the losslessness test (the whole point). Kill: if greedy speculative output differs from greedy target-only by even one token, the verification/acceptance logic is wrong — this is a hard, binary invariant.

#### Ring/context-parallel attention toy (with zigzag load balancing)

- **What:** Shard the sequence across ranks; each rank holds a Q/K/V chunk and computes attention by passing K/V around a ring, accumulating with the same online-softmax rescale as FlashAttention. Zigzag assignment (interleaved chunks) balances the causal triangle so no rank idles.
- **Why (2026):** Context parallelism is how 1M-token training/inference happens (Llama 3, DeepSeek). It is the most direct payoff of having already built the FA2 online-softmax recurrence: ring attention IS that recurrence distributed over a ring. The buildable lesson is two-fold — the running-(m,l,acc) state survives across ring steps, and zigzag fixes the causal load imbalance (~10% over naive striping). Honest scope: an overlap-tuned production ring kernel is AWARENESS; the correctness of the distributed recurrence is buildable and high-value.
- **Build:** New small module in utils/ or kernels/, reusing the flash_attention_forward oracle as the per-step primitive. Each of W ranks holds seq/W tokens; loop W steps, each step rank r computes partial attention of its Q against the currently-held K/V chunk (use the (O,L) return to rescale-combine across steps), then sends K/V to rank r+1 (gloo send/recv or all-gather in the toy). Invariant: 2-rank (and 4-rank) gloo ring-attention output == single-process full attention on the same sequence, to fp tolerance, causal and non-causal, run ×5. Add a load-balance check: with zigzag chunking, per-rank FLOPs are within ~10% of each other under causal masking, vs the lopsided naive split.
- **Source:** Liu, Zaharia, Abbeel, 'Ring Attention with Blockwise Transformers', arXiv:2310.01889 (2023); zigzag/striped balancing: zhuzilin/ring-flash-attention (GitHub, 2024) and the long-context-attention/USP repo (Fang & Zhao, 2024); inference CP: arXiv:2411.01783 (2024).
- **Cost:** CPU/gloo-buildable in full (the recurrence + the equivalence test); ~1–1.5 days. GPU only to measure real comms overlap. Kill: if the ring output doesn't equal full attention, the cross-step (m,l) rescale combine is wrong — the same online-softmax bug class as the FA2 oracle, so the oracle is your debugger.

#### Tensor-parallel toy (column/row-parallel Linear, 2-rank equivalence)

- **What:** Split a Linear's weight across ranks: column-parallel (shard output dim, no input comm, all-gather output) then row-parallel (shard input dim, all-reduce output). Chaining column→row parallel is the Megatron MLP/attention TP pattern, with one all-reduce per block.
- **Why (2026):** TP is half of every 'how do you train a 100B model' answer and the partner to the assignment's tp_calcs algebra. Building a 2-rank toy turns the comms-vs-compute math into something measured, and shows exactly where the all-reduce lands (and why sequence parallelism is needed to shard the LayerNorm/dropout activations TP leaves replicated). Honest scope: the guide already flags full TP/PP as 'do the math, skip the impl unless slack after FSDP' — so this is explicitly an OPTIONAL lab gated on FSDP being green, not a default.
- **Build:** Small module in utils/, on the toy model's Linear (cs336_basics Linear). Implement ColumnParallelLinear (shard weight cols, optional all-gather) and RowParallelLinear (shard weight rows, all-reduce output); compose them as an MLP. Invariant: 2-rank gloo TP MLP output and gradients == single-process replicated MLP on the same input, to fp tolerance, run ×5. Then measure: count collectives per forward (should be one all-reduce per column→row pair) and connect the measured comms volume to the tp_calcs prediction.
- **Source:** Shoeybi et al., 'Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism', arXiv:1909.08053 (2019); sequence-parallel companion: Korthikanti et al., arXiv:2205.05198 (2022).
- **Cost:** CPU/gloo-buildable in full; ~1 day. Gate on FSDP green first (guide's rule). Kill: TP output/grads must match the replicated baseline — any mismatch is a shard-axis or all-reduce-placement bug.

#### FlexAttention custom-mask lab (sliding-window / ALiBi / document mask)

- **What:** Use PyTorch's torch.nn.attention.flex_attention to express attention variants (causal, sliding-window, ALiBi bias, document/block-diagonal masking) via small score_mod / mask_mod Python callbacks plus a block_mask, getting a fused kernel without writing CUDA/Triton.
- **Why (2026):** This is the production-research workflow for *new attention variants* in 2026: prototype the mask/bias in Python, get a FlashAttention-backed fused kernel for free (FlexAttention now has an FA4/CuTeDSL backend, 1.2–3.2x over its Triton path on Blackwell). Crucially it is OPTIONAL, not CORE: FlexAttention's value is that it *hides* the kernel — the opposite of what the from-scratch FA2 deliverable teaches. So it earns its place only as the 'now that you've hand-written FA2, here's how you'd actually ship a variant' lab, and it gives you a second independent oracle.
- **Build:** New bench/lab using kernels/. Define mask_mod for sliding-window (|q-kv|<=w) and a score_mod for ALiBi (subtract slope*(q-kv)); build the block_mask with create_block_mask. Invariant: FlexAttention output == a from-scratch reference for the same variant — for sliding-window/ALiBi, write the explicit masked/biased SDPA and assert allclose; for plain causal, assert it matches the existing flash_attention_forward oracle. Then benchmark FlexAttention vs the hand-rolled FA2 Triton kernel on the roofline to contextualize the 53%-of-SDPA number. GPU needed for the fused-kernel path; the mask/score reference is CPU-checkable.
- **Source:** PyTorch blog 'FlexAttention + FlashAttention-4: Fast and Flexible' (pytorch.org, Mar 5 2026) and the original FlexAttention blog (2024); API in torch.nn.attention.flex_attention.
- **Cost:** Reference/equivalence is CPU-buildable; the fused kernel + roofline needs a GPU (Hopper/Blackwell for the FA4 backend, any CUDA for the Triton backend); ~0.5 day. Kill: if FlexAttention output != the explicit masked reference, the score_mod/mask_mod indexing (b,h,q,kv args) is wrong.

#### Simulated FP8 fake-quant numerics study (extends the mixed-precision lesson)

- **What:** A CPU 'fake quant' that simulates FP8 e4m3/e5m2 (round to the format's mantissa/exponent, apply a per-tensor or per-block scale) and measures the resulting drift in a GEMM and in accumulation, mirroring the assignment's fp16-accumulation lesson one precision lower.
- **Why (2026):** FP8 is the current frontier training precision (DeepSeek-V3 trained 671B in FP8 with fine-grained tile/block scaling), and the fundamental lesson — why per-tensor scaling clips outliers and per-block/per-tile scaling fixes it, why you accumulate in higher precision, why e5m2 for grads and e4m3 for fwd — is exactly the mixed_precision_accumulation lesson generalized. Honest scope: real FP8 GEMMs need a Hopper/Blackwell tensor core and Transformer Engine (AWARENESS); the *numerics intuition* is buildable and CPU-testable with fake-quant, which is what makes it a clean lab rather than a hardware exercise.
- **Build:** Extends mixed_precision_accumulation / utils. Write `fake_quant_fp8(x, fmt, scale)` (clamp to format max, round mantissa) and a per-block scaling variant; compute a GEMM as quant(A)@quant(B) accumulated in fp32, compare to fp32 reference. Invariant/study: (1) per-block scaling has strictly lower max relative error than per-tensor when one block holds an outlier (construct such a tensor) — a falsifiable, predict-first result; (2) accumulating the fake-FP8 products in fp16 vs fp32 reproduces the assignment's small-addend loss, now worse — directly extends the fp16-accumulation finding. Pure CPU/numpy/torch.
- **Source:** DeepSeek-V3 Technical Report, arXiv:2412.19437 (Dec 2024) — FP8 with per-tile (1x128) / per-block (128x128) scaling at scale; NVIDIA 'Per-Tensor and Per-Block Scaling Strategies for Effective FP8 Training' (developer.nvidia.com, 2024–2025); FP8 formats: Micikevicius et al., arXiv:2209.05433 (2022).
- **Cost:** Fully CPU-buildable; ~0.5 day. No GPU. Kill: if per-block doesn't beat per-tensor on the outlier-constructed tensor, the block-scale computation (amax per block) is wrong.

#### INT8/INT4 weight + activation fake-quant (the TCO lever) ➕ *(GDM-alignment addition)*

- **What:** The integer companion to the FP8 study: symmetric affine quant `q = round(clip(x/s, qmin, qmax)); x̂ = q·s` (INT4 ⇒ 16 levels, −8..7), weight-only first, then the harder, higher-value **per-token activation** quantization. Measures perplexity-vs-bit-width on the substrate.
- **Why (2026):** ~99% of serving TCO is the electricity to power the chips, so dropping operand size cuts power, latency, and $/request — and since the decode roofline is memory-bound, weight + KV + activation bits dominate `$/token`. The "true miracle" (Feinberg/GDM) is quantizing **runtime activations**, not just weights: per-token activation outliers are exactly why naive INT4-activation degrades, which motivates per-channel scaling / SmoothQuant / AWQ.
- **Build:** Extend the FP8 fake-quant util with integer formats. Quantize the substrate `Linear` weights INT8→INT4; then add per-token activation fake-quant; report perplexity Δ vs bit-width on a fixed eval batch and tie it to the decode roofline (bits ↓ ⇒ HBM traffic ↓ ⇒ power ↓).
- **Invariant (predict-first):** bits→16 recovers the fp16 perplexity; INT8 weight-only ⇒ small Δ; INT4 **activation** exposes the outlier blowup (state the predicted degradation before running) ⇒ motivates per-channel scaling.
- **Source:** Dettmers et al., 'LLM.int8()', arXiv:2208.07339 (2022); Xiao et al., 'SmoothQuant', arXiv:2211.10438 (2022); Lin et al., 'AWQ', arXiv:2306.00978 (2023).
- **Cost:** Fully CPU-buildable (fake-quant); ~0.5–1 day. Kill: if bits→16 doesn't recover the fp baseline, the quant/dequant scale is wrong.

#### Multi-head Latent Attention (MLA) as a KV-compression serving lever (+ FlashMLA) ➕ *(fact-check addition)*

- **What:** Low-rank latent compression of the KV cache: project K/V into a shared per-token latent vector, cache only the latent, up-project on read. ~93% KV-cache reduction, ~5.76× max generation throughput.
- **Why (2026):** The architectural third KV lever (alongside paging and quantization) that frontier stacks actually ship; FlashMLA is its production Hopper decode kernel (the MLA analogue of FlashAttention).
- **Build:** A CPU module: down-project K/V to d_latent, cache only the latent, up-project on read. Invariant: latent-cached decode logits match full-KV decode to fp tolerance; measure KV-byte savings vs the contiguous/paged cache.
- **Source:** DeepSeek-V2, arXiv:2405.04434 (May 2024); FlashMLA (DeepSeek, open-sourced Feb 2025).
- **Cost:** CPU-buildable; medium.

#### Pipeline-parallel schedule lab (zero-bubble / DualPipe) ➕ *(fact-check addition)*

- **What:** A pipeline micro-batch schedule that eliminates (most of) the pipeline bubble by reordering forward/backward across micro-batches.
- **Why (2026):** DP/TP/CP each get a from-scratch toy in A2, but PP is otherwise awareness-only; bubble elimination is the PP frontier and a clean scheduling exercise.
- **Build:** A toy scheduler that simulates forward/backward micro-batch ordering across stages and reports the bubble fraction; compare naive 1F1B vs zero-bubble.
- **Source:** Zero-bubble PP, arXiv:2401.10241 (ICLR 2024); DualPipe (DeepSeek-V3, arXiv:2412.19437).
- **Cost:** CPU simulation; medium.

### ⚪ Know-it — discuss in interviews, do NOT build from scratch


#### FlashAttention-3 / FlashAttention-4 (Hopper warp-spec/TMA; Blackwell CuTeDSL)

- **What:** FA3 rewrites the FA2 kernel for Hopper using warp specialization, TMA async copies, and FP8 forward; FA4 moves to NVIDIA's CuTeDSL (Python→CUDA JIT via Cutlass) with Blackwell-specific 2-CTA/UMMA/persistent-scheduling paths and BF16+FP8.
- **Why (2026):** This is the literal frontier of the attention kernel and the correct *target awareness* for the roofline discussion — but it is hardware-gated (FA3 needs Hopper, FA4 targets Hopper/Blackwell via CuTeDSL) and is thousands of lines of schedule-specialized code. The from-scratch deliverable correctly targets FA2 on A100/4090 and states the hardware honestly; you must be able to explain in an interview *what* FA3/FA4 add (async warp-group pipelining, TMA, low-precision tensor cores) without reimplementing them.
- **Build:** Do not build. Awareness deliverable: in the roofline writeup, state the FA2 '% of SDPA' number AND name precisely why FA3/FA4 go faster (warp specialization overlaps softmax with the next tile's MMA; TMA removes address-gen overhead; FP8 doubles tensor-core throughput) and why they are Hopper/Blackwell-only. Optionally read the Modal FA4 reverse-engineering writeup to see the schedule. The repo's existing 'FA2-only on this GPU' ADR is the correct artifact.
- **Source:** Shah et al., 'FlashAttention-3', arXiv:2407.08608 (2024) + tridao.me/blog/2024/flash3; FA4: Dao-AILab/flash-attention repo (flash-attn-4 pip pkg, from flash_attn.cute import flash_attn_func) and PyTorch FlexAttention+FA4 blog (Mar 5 2026); Modal 'We reverse-engineered Flash Attention 4' (2026).
- **Cost:** Awareness only — needs Hopper/Blackwell + deep Cutlass/CuTeDSL expertise; explicitly not a from-scratch build. Zero build cost; the cost is reading + being able to whiteboard the schedule.

#### Production FP8/FP4 training stacks (Transformer Engine, per-block scaling, MXFP)

- **What:** Real low-precision training via NVIDIA Transformer Engine: FP8 e4m3/e5m2 GEMMs with current vs delayed (amax-history) scaling, fine-grained per-block/per-tile scaling, and emerging FP4/MXFP4 (microscaling block formats) on Blackwell.
- **Why (2026):** FP8 training is now standard at the frontier (DeepSeek-V3 in FP8; Blackwell adds FP4), and the scaling-recipe choices (current scaling now preferred over delayed for convergence; per-block to handle SwiGLU/attention outliers) are real interview material. But the kernels, the amax bookkeeping, and the convergence validation are a library's job (Transformer Engine); a from-scratch reimplementation is scope-creep with no mastery carry beyond the fake-quant numerics lab above.
- **Build:** Do not build. Awareness: be able to explain delayed vs current scaling (delayed breaks the amax→scale data dependency for speed but risks precision; current is JIT-accurate), per-tensor vs per-block (block fixes intra-tensor outlier variance), and the e4m3-fwd/e5m2-grad split. The buildable companion is the simulated-FP8 lab; this item is the production context for it.
- **Source:** NVIDIA Transformer Engine (github.com/NVIDIA/TransformerEngine, FP8+FP4, 2024–2026) and its FP8 primer docs; DeepSeek-V3, arXiv:2412.19437 (Dec 2024); 'Towards Fully FP8 GEMM LLM Training at Scale', arXiv:2505.20524 (2025); MXFP/OCP microscaling spec (2023).
- **Cost:** Awareness only — needs Hopper/Blackwell tensor cores; reimplementation is a non-goal. Zero build cost.

#### Prefill/decode disaggregation + continuous batching (serving architecture)

- **What:** Split LLM serving into a compute-bound prefill phase and a memory-bound decode phase on separate GPU pools, streaming the KV cache between them; plus continuous (iteration-level) batching and chunked prefill that keep decode latency flat under prefill bursts.
- **Why (2026):** By mid-2025 PD disaggregation became the default architecture in vLLM, SGLang, TensorRT-LLM, and NVIDIA Dynamo — it is *the* serving-systems interview topic because prefill and decode have opposite resource profiles and co-locating them causes interference. But it is multi-node orchestration (KV transfer over the network, independent TP/batch tuning per phase); building it from scratch is an infra project, not a learning module. The repo's rollout seam is the right on-ramp to *discuss* it.
- **Build:** Do not build the disaggregated system. Awareness: be able to explain why prefill is compute-bound and decode is memory/KV-bound, what continuous batching and chunked prefill buy (higher goodput, flat p99), and the KV-transfer cost that makes disaggregation a tradeoff. The buildable adjacent skills are the paged KV-cache and speculative-decoding labs above, which exercise the same KV/throughput intuitions in-process.
- **Source:** Zhong et al., 'DistServe', arXiv:2401.09670 (OSDI 2024); Agrawal et al., 'Sarathi-Serve' / chunked prefill, arXiv:2403.02310 (2024); Orca continuous batching (OSDI 2022); vLLM/SGLang/NVIDIA Dynamo docs (2025–2026).
- **Cost:** Awareness only — multi-GPU/multi-node serving infra; not a from-scratch build. Zero build cost.

#### Quantized KV cache at production quality (KIVI 2-bit, FP8 KV)

- **What:** Shrink the KV cache with low-bit quantization: KIVI's tuning-free asymmetric 2-bit (key per-channel, value per-token) or FP8 KV (per-tensor / per-attention-head), trading a little accuracy for much larger batch/context.
- **Why (2026):** The KV cache dominates serving memory and wall-clock past ~1M context, so KV quantization is a primary cost lever (KIVI: ~8x batch, ~2x throughput; FP8 KV is built into vLLM). Knowing the per-channel-key / per-token-value asymmetry and the accuracy-vs-memory tradeoff is real systems knowledge — and it's also the concrete mechanism behind train-vs-serve logit drift (a server may silently quantize KV). But correct, fast quantized-KV kernels are a serving-engine concern; the buildable piece is the numerics intuition, not the kernel.
- **Build:** Do not build a production quantized-KV kernel. Awareness + a tiny optional probe: if extending the paged-KV lab, simulate FP8/int8 KV with fake-quant and measure the resulting kl_train_infer between full-precision-KV decode and quantized-KV decode on the same prompt — connecting the serving lever to the repo's existing drift diagnostic. The full skill (real low-bit kernels, accuracy benchmarks) is awareness.
- **Source:** Liu et al., 'KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache', arXiv:2402.02750 (ICML 2024); vLLM quantized-KV docs + 'The State of FP8 KV-Cache and Attention Quantization in vLLM' (vLLM blog, Apr 2026).
- **Cost:** Awareness; the fake-quant KV drift probe is a small CPU add-on (~0.25 day) if desired. Production kernels need a GPU and are out of scope.

> ✎ **Fact-check (2026):** Per the KIVI paper/ICML poster: KIVI enables up to ~4x larger batch size (not 8x) and 2.35x-3.47x throughput (not ~2x; the proposal actually understates it), with ~2.6x peak-memory reduction. Fix the numbers to 'up to ~4x batch, 2.35-3.47x throughput, ~2.6x peak-memory reduction.' Everything else (per-channel-key / per-token-value asymmetry, FP8-KV in vLLM, the kl_train_infer drift link, vLLM Apr 2026 blog) is accurate.

#### 4D/5D parallelism + MoE expert-parallel all-to-all (Megatron-LM / TorchTitan)

- **What:** Composing data + tensor + sequence + pipeline + expert parallelism (4D/5D) on a device mesh to train frontier-scale (and MoE) models, where expert parallelism routes tokens to experts on other GPUs via all-to-all collectives.
- **Why (2026):** Every frontier flagship is trained with composed parallelism, and MoE EP all-to-all overlap is one of the hottest, hardest comms problems (the repo's own MoE module notes EP comms as a GPU follow-up). You must be able to reason about the 5D mesh and why all-to-all (not all-reduce) is the MoE collective. But implementing a composed-parallel trainer is what Megatron-LM/TorchTitan/DeepSpeed *are* — a from-scratch reimplementation has no mastery carry beyond the toy DP/TP/CP/FSDP pieces already proposed.
- **Build:** Do not build. Awareness: be able to draw the device mesh (which dim shards what), explain why EP uses all-to-all (each token's chosen experts live on different ranks) and why hiding that all-to-all under expert compute is the bottleneck, and place ZeRO/FSDP within the DP dimension. The from-scratch toys (TP, ring-CP, FSDP) give the per-axis intuition; this item is how they compose at scale. Pair with the assignment's tp_calcs/fsdp_tp_calcs comms algebra.
- **Source:** Megatron-LM: Narayanan et al., arXiv:2104.04473 (2021) + Korthikanti et al., arXiv:2205.05198 (2022, sequence parallel); TorchTitan 4D parallel, arXiv:2410.06511 (2024); 'Scalable Training of Mixture-of-Experts Models with Megatron Core', arXiv:2603.07685 (2026); DeepSpeed ZeRO, arXiv:1910.02054 (2019).
- **Cost:** Awareness only — production framework territory; not a from-scratch build. Zero build cost; the cost is being able to whiteboard the mesh and the EP all-to-all.

#### MoE serving: expert-parallel vs pipeline-prefill (the Flash-2.0 latency unlock) ➕ *(GDM-alignment addition)*

- **What:** Two ways to place an MoE on N chips. **Expert-parallel** shards N experts across N chips, so a token routed mid-forward to an expert on another chip triggers an all-to-all whose latency scales terribly in N — the HBM/latency crisis that keeps low-latency products on dense models. **Pipeline-prefill** instead pipelines *layers* across chips so experts stay resident on one machine: layer 2 processes the first 1k prefill tokens on chip B while layer 1 processes the next 1k on chip A, hiding the comms behind compute.
- **Why (2026):** The exact pivot that unlocked MoE for a low-latency Flash-class model (Feinberg/GDM war story) — the win was a *parallelization-strategy swap*, not a kernel. Being able to whiteboard why expert-parallel all-to-all is latency-bound and how pipeline-prefill hides it is high-signal serving-systems literacy.
- **Build:** Do not build. Awareness; it is the synthesis of the PP-schedule lab (bubble/overlap) and the EP all-to-all item above — the two halves of this tradeoff.
- **Source:** Flash-2.0 development account (Feinberg/GDM interview, 2026); GShard expert-parallel all-to-all, arXiv:2006.16668 (2020); GPipe pipeline parallelism, arXiv:1811.06965 (2018).
- **Cost:** Awareness only.

#### TPU / XLA / JAX serving reality (the GDM substrate) ➕ *(GDM-alignment addition)*

- **What:** The from-scratch build is pragmatically GPU/Triton/PyTorch (rental GPUs; TPU access is gated), but GDM-class pre-training and serving run on **TPU via JAX/XLA**: `jit` traces the program to an XLA graph, fuses elementwise+reduction ops into the matmul, and shards via `pjit`/`shard_map` on a device mesh. "Golfing the XLA compiler" — coaxing fusions, fixing layout, avoiding silent recompiles and memory drops — is the real serving-eng day job (and exactly what the Flash-2.0 40-day rotation did by hand).
- **Why (2026):** XLA/JAX literacy is a concrete GDM-specific signal. The from-scratch GPU work transfers (same roofline, same parallelism algebra), but you must be able to discuss the XLA fusion/sharding model and why a compiler-fused graph behaves differently from eager PyTorch.
- **Build:** Do not build. Awareness: be able to explain `jit`/op-fusion, `pjit`/`shard_map` sharding, and the recompile trap; optionally `jax.jit` a tiny function and read its HLO.
- **Source:** JAX docs (jit/pjit/shard_map); OpenXLA; *How to Scale Your Model* (the JAX/TPU "Scaling Book").
- **Cost:** Awareness only.

---


## A3 · Scaling

> **What a senior RE actually does here (2026):**
>
> A senior RE on the pretraining/scaling pillar in 2026 spends most of their time turning a fixed GPU budget into a defended bet on a single large run. Concretely:
>
> (1) RUNNING LADDERS. They design and launch IsoFLOP / IsoToken ladders: a grid of small-to-medium runs (often 8-30 models, 1e18-1e21 FLOPs) on a real cluster (TorchTitan/Megatron + W&B/internal dashboards), where the deliverable is not a model but a *clean (C, N, D, loss) table*. The craft is choosing the grid so the fit is well-conditioned and the extrapolation factor to the target run stays defensible (typically <~30x past the largest ladder point).
>
> (2) FITTING WITH UNCERTAINTY. They fit power laws / the parametric L(N,D) using robust objectives (Huber in log-space) and report bootstrap confidence intervals on the exponents and on the predicted optimum — never a bare point estimate. After Besiroglu's 2024 replication, "your fit has a CI and survives a bootstrap" is table stakes; a fit without error bars gets bounced in review.
>
> (3) CHOOSING THE *RIGHT* BUDGET. The signature 2026 move: deciding whether to optimize training-compute (Chinchilla) or amortized-serving cost (inference-aware / over-training). For a model that will serve trillions of tokens they deliberately over-train a smaller model far past Chinchilla (Llama-3 style: 8B on 15T = ~1875 tok/param) because inference cost dominates the lifetime budget. They compute the crossover demand and pick tokens/param accordingly.
>
> (4) HYPERPARAMETER TRANSFER. They tune LR/init/etc. on a cheap proxy width under μP and transfer zero-shot to the big run, so the ladder isn't confounded by per-scale HP retuning. They run μP coordinate-checks (activation RMS width-invariant) as a correctness gate before trusting any transfer.
>
> (5) LOSS-vs-CAPABILITY. They predict downstream benchmark performance via loss-to-loss / two-stage (FLOPs->loss->task) mappings, and are explicit that the law predicts *loss*, not accuracy — emergence/saturation can break the loss->capability map.
>
> The failure modes they chase: extrapolating past the regime where the law holds (data wall, batch-size > critical, precision floor), a fit dominated by one noisy point, confusing train-optimal with serve-optimal, and HP confounds. "Good" looks like: a predicted optimal (N,D,HPs) for the big run with a stated CI, the extrapolation factor in every plot caption, and a one-line statement of what would invalidate the law.


*Fact-check verdict: Unusually disciplined and accurate. Every one of the 13 cited sources is real and correctly attributed (authors, arXiv IDs, years, venues all verified), the techniques are described faithfully with no hallucinations or garbled mashups, and the tier discipline is honest — the CPU-buildable defaults (robust fit, inference-aware allocator, free-fit diagnostic) are correctly CORE, and the GPU-bound laws (precision, MoE, downstream-capability, data-mixing) are correctly tiered down to AWARENESS with accurate 'why you can't build this on CPU' reasoning, kill-criteria, and clean falsifiable invariants. It is genuinely fundamentals-first with no research-project scope-creep. Two corrections keep it from being rubber-stamped: (1) item 1's 'bootstrap CI strictly narrows when adding points' is not a deterministically sound assert (a high-leverage point can widen a finite-sample CI) — soften to an averaged/interior-point claim; (2) item 4's currency lags the field — the March-2026 Czech et al. result (2603.22339) shows the IsoFLOP min-picking the proposal elevates to CORE (items 1, 3) is systematically biased even on noise-free data and recommends the Variable-Projection form of parametric Approach 3, so the parametric fit deserves promotion (or at least a bias caveat on the min-pick path) rather than sitting as a lesser OPTIONAL. The one true gap is conceptual, not factual: for a pillar titled 'inference-aware allocation,' test-time/inference-compute scaling (Snell 2024; coupled with overtraining in the 2026 paper 2604.01411) is a whole missing axis and would be the highest-value next add.*


### 🟢 Modern defaults — adopt these into the from-scratch build


#### Robust log-space fit with Huber loss + bootstrap confidence intervals

- **What:** Replace the naive least-squares log-log line with a Huber-loss fit (robust to one bad IsoFLOP point) and attach bootstrap confidence intervals to the exponents a, b and to every extrapolated prediction (N_opt, D_opt at the target budget).
- **Why (2026):** Load-bearing. After Besiroglu et al. (2024) showed the original Chinchilla parametric fit was non-converged with implausibly narrow CIs, 'report a CI, use a robust objective, survive a bootstrap' became the lab default — a scaling fit without error bars gets bounced in review. The Huber loss is exactly what fixes a single outlier run dominating the slope, and the bootstrap CI is the honest way to state how far you trust the extrapolation. This is the single highest-value, lowest-cost upgrade to the CS336 fitter.
- **Build:** In src/scratch_llm/scaling/: add fit_powerlaw(xs, ys, *, loss='huber') using scipy.optimize.least_squares with f_scale (or a hand-rolled Huber on log-residuals), and bootstrap_powerlaw(xs, ys, n=1000) that resamples the (C_i, N_opt) pairs with replacement and returns percentile CIs for a and predict(target). Invariant test: on isoflops_curves.json the median bootstrap a and b recover ~0.5 each with a+b in [0.95,1.05]; corrupt one run's loss by 5x and assert the Huber slope moves <2x less than the plain-lstsq slope; assert the bootstrap CI on predict(1e24) strictly narrows when you add more ladder points (monotone shrinkage of std).
- **Source:** Besiroglu, Erdil, Barnett, You, 'Chinchilla Scaling: A replication attempt', arXiv:2404.10102, 2024 (Epoch AI). Baseline: Hoffmann et al., 'Training Compute-Optimal LLMs', arXiv:2203.15556, 2022.
- **Cost:** CPU-buildable on the existing isoflops_curves.json (zero new data); ~2-3 h. Pure numpy/scipy. No kill-criterion — it strictly improves the existing fitter.

> ✎ **Fact-check (2026):** Keep tier CORE_MODERN_DEFAULT and the source. Soften the test to a statistical/averaged claim: 'the bootstrap CI on predict(target) narrows ON AVERAGE across random seeds as more well-conditioned interior ladder points are added,' or assert shrinkage only when the added points lie inside the existing FLOP range (interpolation, not high-leverage extrapolation). The other two invariants (a,b~0.5 with a+b in [0.95,1.05]; Huber slope moves <2x of lstsq under a 5x-corrupted point) are sound.

#### Inference-aware allocation (compute-optimal vs serve-optimal over-training)

- **What:** A second allocator that, given an expected lifetime inference token demand D_inf, minimizes total (training + inference) compute instead of just training loss — recovering the Llama-3 'over-train a smaller model' answer: pick N smaller and D larger than Chinchilla when serving dominates.
- **Why (2026):** Load-bearing and the senior distinction the bare Chinchilla fit hides. Every frontier model that actually ships (Llama 3: 8B on 15T tokens = ~1875 tok/param) is over-trained relative to Chinchilla because inference cost is paid forever. Knowing *which budget you optimize* — one-time train vs amortized serve — is the line that separates a candidate who can fit a power law from one who can make a deployment decision. Sardana et al. formalize it; this is the most important conceptual add to the assignment.
- **Build:** In scaling/: add optimal_allocation(C_train_budget, D_inference, ...) that grids N, derives D_train = C/(6N), evaluates total cost = 6*N*D_train (train) + 2*N*D_inference (inference fwd) subject to a Chinchilla-style L(N,D) surface (use the parametric law from the optional item, or the IsoFLOP-fitted exponents), and returns argmin total cost. Invariant test (the limit check the advisor named): as D_inference -> 0 the returned (N,D) converges to the pure Chinchilla compute-optimal point from the existing fitter (within tolerance); as D_inference grows, N_opt monotonically decreases and tokens/param monotonically increases. Sanity-print the crossover D_inference where over-training starts to pay.
- **Source:** Sardana, Portes, Doubov, Frankle, 'Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws', arXiv:2401.00448, ICML 2024. Empirical anchor: 'The Llama 3 Herd of Models', arXiv:2407.21783, Meta 2024 (8B trained on ~15T tokens).
- **Cost:** CPU-buildable on synthetic/fitted surface; ~2-3 h. No GPU. Reuses the fitter. Marginal only if you ignore deployment — load-bearing the moment a model is meant to be served.

#### Free-fit a+b diagnostic + mandatory extrapolation-factor reporting

- **What:** Deepen the existing a+b approximately 1 check from a pass/fail assert into a reported diagnostic: fit a and b independently, report the free-fit a+b (its deviation from 1 is a data-quality signal), and emit the extrapolation factor (target_C / max_ladder_C) in every prediction and plot caption.
- **Why (2026):** Low-cost, high-signal, and it teaches the fundamental honesty discipline that the build guide and every scaling interview reward. The C=6ND identity forces a+b=1, so the *free-fit* deviation is a built-in lie detector for your min-picking / 6ND constant; collapsing it to a silent assert throws away that diagnostic. Stating the extrapolation factor ('this prediction reaches 31x past my largest data point') is the senior tell that distinguishes a defensible exponent from a number.
- **Build:** In scaling/: PowerLaw.fit returns both exponents and the free-fit a+b; add a ScalingReport dataclass carrying a, b, a+b, the per-prediction extrapolation_factor, and the bootstrap CI. Make the existing a+b check a *warning with the measured value* plus a separate hard ConstrainedFit option that imposes b=1-a (one fewer free parameter, lower variance) — and report both so the learner sees the variance/diagnostic trade-off. Invariant test: on isoflops_curves.json free-fit a+b is in [0.95,1.05]; deliberately mislabel the 6ND constant (use 2ND) and assert a+b departs from 1 by >0.1 (the diagnostic fires); assert every predict() return carries a finite extrapolation_factor.
- **Source:** Hoffmann et al., arXiv:2203.15556, 2022 (the C=6ND identity and a~b~0.5). Methodology rationale: Besiroglu et al., arXiv:2404.10102, 2024.
- **Cost:** CPU, trivial (~1 h), pure refactor of the fitter you already build. No new deps.

#### Compute-optimal inference (test-time-compute scaling) allocator ➕ *(fact-check addition)*

- **What:** Allocate a fixed *inference* FLOP budget across model size vs samples / search depth — the inference-side mirror of the compute-optimal *training* allocator.
- **Why (2026):** An entire allocation axis for reasoning models, and it shifts the train-vs-serve crossover from the over-training item: once test-time scaling is accounted for, over-training becomes compute-optimal in more regimes. A pillar titled "inference-aware allocation" is incomplete without it.
- **Build:** CPU: given an inference-FLOP budget per query, grid model-size × number-of-samples against a synthetic accuracy(N, k) surface and return the argmax. The inference-side twin of the training allocator.
- **Source:** Snell et al., "Scaling LLM Test-Time Compute Optimally…", arXiv:2408.00724 (2024); "Test-Time Scaling Makes Overtraining Compute-Optimal", arXiv:2604.01411 (2026).
- **Cost:** CPU-buildable; ~2–3 h.

### 🔵 Build labs — scoped, opt-in, after the core is green


#### Full parametric L(N,D)=E+A/N^alpha+B/D^beta fit (Hoffmann 'approach 3', done right)

- **What:** Fit the five-parameter Chinchilla loss surface (E, A, B, alpha, beta) directly to all 72 runs via Huber-loss optimization over log L, instead of the IsoFLOP min-picking shortcut — then derive the compute-optimal frontier analytically from the fitted surface.
- **Why (2026):** The build guide tags this SKIP *for passing the assignment* (the PDF says use min-picking), but as a frontier enhancement it is exactly the method labs use when they have a full (N,D,loss) grid, and it is the object every other scaling result (inference-aware, data-constrained) modifies. Building it once — and reproducing Besiroglu's finding that the fit is sensitive to the loss scale and optimizer — teaches why the 'cheap shortcut' exists and where the real method bites. Genuinely buildable on the existing data.
- **Build:** In scaling/: fit_parametric(runs) using scipy.optimize over theta=(E, log A, log B, alpha, beta) minimizing Huber(log L_pred - log L_obs) with a multi-start grid (Hoffmann's failure was a single bad init). Derive N_opt(C), D_opt(C) by minimizing L s.t. C=6ND (closed form: N_opt ∝ C^(beta/(alpha+beta))). Invariant test: the parametric-derived exponent a = beta/(alpha+beta) agrees with the IsoFLOP min-pick a within the bootstrap CI on isoflops_curves.json (two independent methods must agree); E (irreducible loss) is positive and below the smallest observed loss; multi-start reduces final Huber loss vs single-start (reproduces the convergence lesson).
- **Source:** Hoffmann et al., 'Training Compute-Optimal LLMs', arXiv:2203.15556, 2022 (their third estimation procedure). Convergence pitfalls: Besiroglu et al., arXiv:2404.10102, 2024.
- **Cost:** CPU on existing data; ~3-4 h (the multi-start + conditioning is the work). scipy only. Kill-criterion: if multi-start still won't converge to a fit consistent with the min-pick exponent, fall back and document it as the Besiroglu lesson.

> ✎ **Fact-check (2026):** Keep the item buildable but upgrade why_2026 to cite Czech et al. 2603.22339 (2026) and recommend the Variable Projection formulation of Approach 3 (reduce the 5-D fit to a 2-D optimization over alpha,beta with E,A,B solved by linear least-squares inside — this is what fixes the conditioning Hoffmann/Besiroglu struggled with, more robustly than naive multi-start). Consider promoting this nearer to CORE, or at minimum add a caveat to items 1/3 that the min-pick path carries a known systematic bias the parametric+VarPro fit corrects. Baseline source Hoffmann 2203.15556 is correct.

#### Data-constrained scaling: effective-tokens closed form (repeated-token decay)

- **What:** Add the Muennighoff 'effective data' transform D' = U + U*R*(1 - exp(-R/R*)) (U = unique tokens, R = extra epochs, R* = a fitted decay half-life) so the allocator uses *effective* tokens, capturing that repeated tokens past a few epochs buy almost nothing.
- **Why (2026):** Load-bearing once you hit the data wall — which frontier pretraining has. A Chinchilla law fit under unique-token assumptions over-promises the moment you re-use data; Muennighoff showed up to ~4 epochs is nearly free, then returns decay sharply. This is *why* A4 (data curation) matters and where the constant exponent stops being constant. Cleanly buildable as a pure transform feeding the existing fitter.
- **Build:** In scaling/: effective_tokens(unique, total, R_star) returning D' per the formula, and a tiny fit_repetition_decay helper that fits R* from synthetic (epochs, loss) points generated from the law (you don't have real repeat-run data, so generate it from L(N,D') with a chosen R* and recover it). Wire D' into optimal_allocation so the data budget saturates. Invariant test (advisor's limit check): as R -> 0 (one epoch), D' -> unique exactly; D' is monotone increasing and concave in R; D' -> unique + U*R_star as R -> infinity (bounded payoff); the fitter recovers the planted R_star within tolerance.
- **Source:** Muennighoff, Rush, Barak, Le Scao, Tazi, Piktus, Pyysalo, Wolf, Raffel, 'Scaling Data-Constrained Language Models', arXiv:2305.16264, NeurIPS 2023.
- **Cost:** CPU, synthetic data (generate-and-recover); ~2-3 h. Pure numpy. No real repeat-token corpus needed for the buildable version.

#### muP coordinate-check + zero-shot learning-rate transfer mini-lab

- **What:** A small lab that re-parametrizes the existing scratch_llm Transformer in Maximal Update Parametrization (muP: width-dependent init scales and per-tensor LR multipliers) and demonstrates (a) the coordinate-check — activation/logit RMS at init is width-invariant — and (b) that the optimal LR found on a narrow proxy transfers to a wider model.
- **Why (2026):** The CS336 A3 PDF itself lists Yang et al. 2022 as an in-scope reference, because muP is how labs avoid re-tuning HPs at every rung of the scaling ladder — without it, your ladder loss is confounded by per-scale LR. 'Tune once on a proxy, transfer zero-shot' is standard practice (the paper transfers HPs to GPT-3-6.7B from a 40M proxy at ~7% of the cost). The coordinate-check is a clean, falsifiable correctness gate that teaches the abc-parametrization fundamentals.
- **Build:** Add scaling/mup.py (or a lab notebook) that wraps model.py: scale embedding/output/hidden init by the muP rules and attach per-group LR multipliers in optim.py. Coordinate-check test (CPU, tiny, the load-bearing invariant): instantiate the model at widths d=128,256,512 with muP, run one forward on fixed input, assert per-layer activation RMS is constant across widths within ~10% (under standard parametrization it would scale with width — assert that the non-muP control *fails* this). LR-transfer demo: overfit-one-batch a tiny model at two widths, show the loss-vs-LR curve's argmin is approximately width-invariant under muP and shifts under SP.
- **Source:** Yang, Hu, Babuschkin, Sidor, Liu, Farhi, Ryder, Pachocki, Chen, Gao, 'Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer', arXiv:2203.03466, 2022. (Cited directly in the A3 handout.)
- **Cost:** CPU-buildable for the coordinate-check + tiny LR-transfer demo (~3-5 h); full multi-width LR sweep is cheap on CPU at toy width. Touches model.py/optim.py. Kill-criterion: if the coordinate-check won't go width-invariant after fixing init/LR rules, the parametrization is wrong — debug before claiming transfer.

#### Kaplan-vs-Chinchilla reconciliation lab

- **What:** A short analysis that fits both the Kaplan (N-heavy, a~0.73) and Chinchilla (a~0.5) compute-optimal exponents on the same data and shows the discrepancy is driven by fitting choices (counting embedding vs non-embedding params, including warmup/LR-schedule artifacts, the last-step vs smoothed loss).
- **Why (2026):** Marginal-but-clarifying. Both papers are cited in the A3 handout, and the 'why did two careful teams get different exponents?' question is a classic scaling interview probe. The answer — that the headline exponent is sensitive to seemingly-innocuous methodology (param counting, schedule, fit objective) — is the deepest version of the build guide's 'know when the law breaks' lesson, and it directly motivates the robust-fit core item.
- **Build:** In a scaling/ analysis script: re-fit the isoflops data (or the parametric surface) under two conventions — (1) N = total params vs N = non-embedding params via nonembed = 12*n_layer*d_model^2, and (2) loss = final step vs min-over-tail. Report how a moves between conventions. Invariant test: switching N from total to non-embedding params shifts the fitted a by a measurable, signed amount (assert the direction matches the literature — embedding-inclusion biases a upward at small N); document which convention the CS336 data implies.
- **Source:** Kaplan et al., arXiv:2001.08361, 2020 vs Hoffmann et al., arXiv:2203.15556, 2022. Reconciliation framing corroborated by 'Reconciling Kaplan and Chinchilla Scaling Laws' (arXiv:2406.12907, 2024).
- **Cost:** CPU on existing data; ~2 h. Pure analysis, no new deps. Low effort, mostly a writeup with two fits.

#### Critical-batch-size feasibility guard in the QueryPlanner

- **What:** Add a critical-batch-size estimate (CBS grows with data/token budget, ~sqrt-like) as a feasibility constraint in the budget-aware QueryPlanner, so the planner doesn't propose configs whose train_batch_size exceeds the regime where larger batches stay compute-efficient.
- **Why (2026):** Directly relevant to the A3 leaderboard (the API exposes train_batch_size and charges wall-clock). Zhang/Kakade (ICLR 2025) found CBS scales primarily with *data size, not model size* — so the right batch for a small fitting run differs from the big run, and picking batch > CBS wastes the wall-clock budget on data-inefficient steps. This makes the query planner's budget discipline physically grounded rather than arbitrary.
- **Build:** In scaling/ (QueryPlanner): add critical_batch_size(total_tokens) returning a CBS estimate B_crit ∝ D^p (p~0.3-0.5, a documented default), and in the planner reject or down-rank any candidate config with train_batch_size > B_crit, and flag when a small fitting run's batch can't be naively reused at the 48-B200-hr scale. Invariant test (CPU, no API needed): CBS is monotone increasing in total_train_tokens and (per the paper) approximately invariant to model size N at fixed tokens — assert both on a synthetic grid; assert the planner filters out a deliberately-oversized batch config.
- **Source:** Zhang, Morwani, Vyas, Wu, Zou, Ghai, Foster, Kakade, 'How Does Critical Batch Size Scale in Pre-training?', arXiv:2410.21676, ICLR 2025.
- **Cost:** CPU, no API/GPU for the buildable invariant (~2 h). Pure function + planner hook. The real CBS measurement needs runs, so the buildable version uses the published scaling form.

### ⚪ Know-it — discuss in interviews, do NOT build from scratch


#### Scaling laws for precision (low-precision training effective parameters)

- **What:** Kumar et al.'s precision-aware scaling law: training in lower precision reduces a model's *effective* parameter count, and post-train quantization degrades a model more the more it was trained — so precision is a third axis alongside N and D, and training a bigger model in lower precision can be compute-optimal.
- **Why (2026):** You must be able to discuss this in a pretraining interview — it's why FP8/low-precision training decisions are made — but it is NOT a from-scratch CPU build: validating it requires real low-precision (FP8/INT) training runs across a precision grid on GPU. The API in A3 exposes a dtype knob (float32/bfloat16), which is the natural place to *note* the dependency, but you cannot fit the precision scaling law without hardware and many runs. Tier down: discuss, don't build.
- **Build:** Do NOT build from scratch. Awareness deliverable: a 1-page note in scaling/ docs stating the law's shape (effective params decrease with bit-width; PTQ damage grows with training tokens), tying it to the API's dtype field, and to the A2 mixed-precision work already in the repo. If ever resourced on GPU, the minimal experiment is a 3-point bf16/fp16/fp8 ladder at one (N,D) — explicitly a rented-GPU follow-up, not a CPU module.
- **Source:** Kumar, Ankner, et al., 'Scaling Laws for Precision', arXiv:2411.04330, 2024 (ICLR 2025).
- **Cost:** Awareness only. Buildable version needs GPU + FP8-capable hardware (Hopper/Blackwell) and many runs — out of scope for a CPU from-scratch build.

#### Distillation scaling laws (teacher size · student size · distillation tokens) ➕ *(GDM-alignment addition)*

- **What:** A scaling law for the *student's* loss as a function of student size, distillation-token budget, and teacher capability — with a "capacity gap" regime (too-strong a teacher can hurt a small student) and a compute-optimal teacher/student/token allocation. The quantitative complement to the A5 distillation build-lab.
- **Why (2026):** This is the law a pre-training lead is fitting when "a distillation-infra rewrite uncovers new scaling laws that enable a Flash-class model" (Feinberg/GDM): it tells you *when distilling a big teacher into a served student beats training that student directly*, given a fixed serving-FLOP target. The interview signal is being able to state the allocation tradeoff and the capacity-gap caveat.
- **Build:** Do NOT fit from scratch (needs a teacher/student family across many runs). Awareness deliverable: a note in `scaling/` connecting `fit_powerlaw` to the (N_student, D_distill, teacher) axes, and stating the compute-optimal-distillation and capacity-gap findings. The CPU-buildable slice is the *mechanism* (the A5 KD lab), not the fitted law.
- **Source:** Busbridge et al., 'Distillation Scaling Laws', arXiv:2502.08606 (2025, Apple); capacity gap: Cho & Hariharan, 'On the Efficacy of Knowledge Distillation', ICCV 2019.
- **Cost:** Awareness only. The KD mechanism is the buildable companion (A5 🔵); fitting the law needs GPU + many runs.

#### MoE scaling laws: granularity and optimal sparsity

- **What:** Sparse-model scaling laws that add a granularity axis (G = FFN_size / expert_size; the common 'expert = FFN' choice is suboptimal at almost every budget — Krajewski) and an optimal-sparsity axis (a sweet-spot ratio of active-to-total params per compute/param constraint — Abnar). The dense<->MoE efficiency gap widens with scale.
- **Why (2026):** Highly relevant given the repo already ships a DeepSeek-V3-style MoE (moe.py) — but the *scaling law* for MoE is an AWARENESS item: fitting it requires training many MoE models across (N_total, N_active, G, D) on real hardware. You should be able to whiteboard granularity and optimal sparsity in an interview (every frontier model since ~2024 is sparse), and point at moe.py as where the ladder *would* attach, but you cannot fit these laws on CPU.
- **Build:** Do NOT fit from scratch on CPU. Awareness deliverable: a note mapping the granularity parameter G onto the existing MoEConfig (expert_d_ff / n_routed_experts already let you vary G in moe.py) and stating the two findings (G=1 / expert=FFN is suboptimal; there is an optimal active/total sparsity per budget). If resourced on GPU, the minimal experiment is a small fixed-FLOP sweep over G and sparsity at one compute budget — a rented-GPU follow-up. The buildable-now slice is purely the awareness mapping, not a fitted law.
- **Source:** Krajewski, Ludziejewski, et al., 'Scaling Laws for Fine-Grained Mixture of Experts', arXiv:2402.07871, ICML 2024; Abnar, Shah, Busbridge, et al., 'Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for MoE', arXiv:2501.12370, 2025 (Apple).
- **Cost:** Awareness only. moe.py exposes the G/sparsity knobs, but fitting the law needs GPU + many MoE runs. Out of scope for CPU.

#### Downstream-capability prediction (two-stage FLP) and loss-as-proxy honesty

- **What:** The two-stage 'FLOPs -> pretraining loss -> downstream task metric' prediction method (Chen et al.): predict loss with a scaling law, then map loss to benchmark accuracy with a separate fitted curve, because predicting accuracy directly from FLOPs fails near emergence thresholds.
- **Why (2026):** The thing you ship is task performance, not loss — and the build guide explicitly names 'upstream loss vs downstream capability' as the scarce senior skill. You must be able to explain that your scaling law predicts *loss* (a proxy), that the loss->capability map is non-linear and can break (emergence/saturation), and that the standard fix is the two-stage decomposition with soft metrics. But building/validating it needs a model family evaluated on real benchmarks — not a CPU from-scratch deliverable.
- **Build:** Do NOT build from scratch (no model family + benchmark harness on CPU). Awareness deliverable: in the writeup/plot captions for the core fitter, state explicitly 'this law predicts validation loss, which is a proxy for capability; the loss->task map is non-linear and may exhibit emergence.' Optionally, a toy demo: fit a sigmoid loss->accuracy curve on *synthetic* (loss, accuracy) pairs to illustrate why hard-accuracy thresholds look 'emergent' while a soft metric is smooth — clearly labeled as illustrative, not a real downstream law.
- **Source:** Chen, Huang, Gao, Wang, Yang, Ji, 'Scaling Laws for Predicting Downstream Performance in LLMs' (FLP), arXiv:2410.08527, 2024. Emergence-as-metric-artifact context: Schaeffer et al., 'Are Emergent Abilities a Mirage?', NeurIPS 2023.
- **Cost:** Awareness only (optional synthetic sigmoid illustration is ~1 h CPU). Real downstream scaling needs a model family + eval harness — out of scope.

#### Data-mixing / domain scaling laws

- **What:** Scaling laws that predict loss as a function of the domain-weight vector h (the mixture proportions) in addition to N and D, enabling you to pick an optimal data mixture by fitting a small proxy and extrapolating (DoReMi / RegMix / Data-Mixing-Laws / the Apple 'Scaling Laws for Optimal Data Mixtures' formulation).
- **Why (2026):** This is where A3 (scaling) meets A4 (data) and it is a real lever frontier teams pull — but it is AWARENESS for a from-scratch build: fitting a mixture law requires a *multi-domain tokenized corpus* and many runs over different h, neither of which the A3 single-corpus DCLM setup or a CPU build provides. A known sharp edge to be able to discuss: optimal mixtures are often *not* scale-invariant, so weights tuned on a small proxy can fail at the target scale (AutoScale/Aioli critiques).
- **Build:** Do NOT build from scratch in A3 (single-corpus, no domain labels). Awareness deliverable: a note that the same fit_powerlaw machinery generalizes to L(N, D, h) by adding domain-weight features, flagged as the natural bridge to the A4 data pillar where multi-domain corpora exist; state the scale-invariance caveat. The buildable slice belongs in A4 (with real domain mixtures), not here.
- **Source:** Shukor, Bethune, Busbridge, Grangier, Fini, El-Nouby, Ablin, 'Scaling Laws for Optimal Data Mixtures', arXiv:2507.09404, 2025 (Apple); RegMix (arXiv:2407.01492, ICLR 2025); DoReMi (Xie et al., NeurIPS 2023).
- **Cost:** Awareness only. Needs a multi-domain labeled corpus + many runs. The CPU-buildable version lives in the A4 pillar, not A3.

---


## A4 · Data

> **What a senior RE actually does here (2026):**
>
> A senior data RE in 2026 lives in the data-ablation loop, not in any one filter. The loop: pull a few CommonCrawl shards (or an internal crawl), run extract -> heuristic/Gopher filter -> model-based quality filter -> dedup -> decontaminate, then train a FIXED small proxy model (~0.5-2B, e.g. the 1.7B FineWeb ablation rig) on the candidate mix for a fixed token/compute budget, eval on a FROZEN held-out suite (MMLU/ARC/HellaSwag + a perplexity benchmark like Paloma C4-100), and compare win-rate vs the current baseline mix at iso-compute. The unit of progress is "this curation change beat baseline on N of M frozen evals at the same FLOPs," exactly the DCLM/FineWeb methodology. Critically, they ALSO read the data: hand-inspect 20-50 docs the new filter dropped vs kept, because aggregate metrics hide failure modes (a quality classifier that secretly downweights code, a dedup pass that erased legitimate variety). Day-to-day knobs: the quality-classifier positive/negative label source (the single highest-leverage decision — "quality" is defined by the label set), classifier score thresholds, MinHash (b,r) / n-gram / Jaccard settings and dedup SCOPE (per-shard vs global), per-domain mixing weights, and the end-of-run annealing/mid-training mix (upsampling high-quality + long-context data). They own decontamination as a release gate: every model ships with a documented n-gram overlap check against all eval benchmarks, and they treat a contamination finding as a launch blocker. "Good" looks like: reproducible curation code, an ablation ledger (each change -> proxy-eval delta), explicit per-filter discard accounting, and a decontamination report. The failure modes they actively chase: benchmark contamination (verbatim or paraphrased eval leakage inflating scores), over-filtering that collapses diversity / hurts tail domains, dedup false-positives that delete valid content, classifier bias toward one register (formal English) at the expense of code/multilingual/dialogue, and "synthetic data feedback loops" where rephrased/LLM-generated data drifts the distribution. Most of the throughput is on Spark/Ray + GPU clusters (NeMo-Curator-class infra); the RE designs the recipe and the ablation, not the cluster plumbing.


*Fact-check verdict: Accurate and current. This is an unusually clean proposal: I verified all 12 enhancements against primary sources and every cited paper is real, correctly attributed (right authors, right arXiv ID, right year), and every headline number is exact — DCLM's 30.2/29.0/27.1/26.1 Core, the 6.6pt MMLU at 40% less compute, FineWeb's per-snapshot-beats-global finding, FineWeb-Edu's Llama-3-70B/Snowflake-arctic/>=3 recipe, Llama-3's +24.0% GSM8K / +6.4% MATH annealing (negligible at 405B), Lee et al.'s >4% validation overlap, Llama-3's 8-gram/50%-token-ratio gate, DoReMi's 30x transfer / 2.6x steps, and Nemotron-CC's 0-19 binning + 3-classifier max-ensemble all check out. No hallucinations, no garbled mashups, no stale citations. It is genuinely fundamentals-first and free of research-project scope-creep: each buildable item reuses an existing primitive (MinHash shingling, the fastText path, embedding+clustering), ships a correct correctness oracle, and the honest caveats (n-gram misses paraphrase; SemDeDup < fastText in DCLM; per-shard win is scale-dependent; annealing benefit needs a full run) are accurate and appropriately placed. The tiering is mostly honest, with one real adversarial weakness: the data-MIXING items lean on DoReMi (2023) as the canonical automated answer and demote RegMix (ICLR 2025) to a parenthetical, when in 2026 RegMix is the more current and more buildable method (small-model regression at ~10% of DoReMi's compute) and deserves first-class OPTIONAL treatment — hence the two NEEDS_FIX on items 8-9. The only substantive content gap is a curation METHOD for the quality-vs-diversity trade-off (QuaDMix), which the proposal names as a fear but never addresses algorithmically. Fix the RegMix tiering and add QuaDMix, and this pillar is ship-ready.*


### 🟢 Modern defaults — adopt these into the from-scratch build


#### Train<->eval decontamination (n-gram overlap gate)

- **What:** A pass that flags/removes any training document containing an n-gram (e.g. 8- to 13-gram) that also appears in an evaluation benchmark, run ACROSS the train/eval boundary. Reuses the exact n-gram/hashing primitive already built for MinHash dedup, just applied between corpora instead of within one.
- **Why (2026):** Load-bearing and non-negotiable: no credible 2026 model ships without a documented decontamination report, because train-test overlap silently inflates benchmark scores. Lee et al. found >4% of standard validation sets overlap with common web training data; Llama-3 gates every eval on an 8-gram overlap check. CS336-A4 deliberately scopes this OUT (its only rule is 'never copy Paloma val into train'), so this DEEPENS past A4 — it is the engineering discipline that turns 'don't copy' into a verifiable invariant. Honest caveat: n-gram matching misses paraphrased/translated contamination, which is exactly why it's a floor, not a ceiling.
- **Build:** New module src/scratch_llm/data/decontaminate.py reusing the n-gram shingling + mmh3 hashing from the MinHash build. Build an n-gram set (or Bloom/hashset) from the eval corpus; stream train docs, flag any doc whose token n-gram intersects the eval set above a token-ratio threshold (Llama-3 style: contaminated if a ratio of the doc's tokens fall inside an eval n-gram). Minimal buildable version: pure-CPU on the test fixtures. Correctness invariant/test: inject a known eval 13-gram into one synthetic train doc -> the gate flags EXACTLY that doc; a clean held-out set yields ~0 flags (false-positive floor). Second test: a doc sharing only a 7-gram is NOT flagged when n=13 (threshold sensitivity).
- **Source:** Lee et al. 2021, 'Deduplicating Training Data Makes Language Models Better', arXiv:2107.06499 (train-test overlap >4%, n-gram matching finds it). Grattafiori/Dubey et al. 2024, 'The Llama 3 Herd of Models', arXiv:2407.21783 (8-gram overlap decontamination, token-ratio threshold per dataset). Soldaini et al. 2024, 'Dolma', arXiv:2402.00159 (production pipeline decontaminates against eval sets).
- **Cost:** CPU-buildable, low effort (~half a day; the n-gram primitive already exists). Kill-criterion: if false-positive rate on a clean held-out set is non-trivial, raise n or the token-ratio threshold before trusting it.

#### DCLM-style quality-classifier positive set (instruction/QA-formatted positives)

- **What:** Keep the A4 fastText quality classifier and its (label, score)+threshold shape, but swap the trusted-positive label source from 'Wikipedia-reference-linked pages' to 'instruction/QA-formatted high-quality text' (OpenHermes-2.5 + high-karma r/ExplainLikeImFive answers) vs random web negatives — the DCLM-Baseline recipe.
- **Why (2026):** Directly deepens the 15-point load-bearing deliverable whose lesson is 'quality IS the label source.' DCLM showed this exact positive set is the single biggest lever: fastText on OH-2.5+ELI5 positives scored 30.2 Core accuracy vs 29.0 for perplexity filtering, 27.1 for SemDeDup, 26.1 for PageRank — and gave a 6.6pt MMLU gain at 40% less compute than prior open-data SOTA. The mechanism is the load-bearing insight: instruction/answer-formatted positives bias the keeper set toward downstream-useful register, not just 'looks like Wikipedia.' Cheap, drop-in, same dependency (fastText), same test oracle.
- **Build:** In the curate quality-classifier module, parameterize the positive corpus. Add a second training config: positives = a sample of OpenHermes-2.5 + ELI5-style Q/A text (or any instruction/answer-formatted set), negatives = random CC; ~400k docs split 50/50, mirroring DCLM. The existing run_classify_quality adapter is unchanged. Correctness invariant/test: (a) the existing test_classify_quality still passes (returns (label, float>0)); (b) a NEW comparison test — on a small labeled held-out set, the instruction-positive classifier ranks held-out 'useful' docs above 'boilerplate' docs with higher AUC than the Wikipedia-positive baseline, AND you log per-class agreement between the two classifiers (the signal-design ablation). Predict-before-you-run: write expected discard% shift before training.
- **Source:** Li et al. 2024, 'DataComp-LM: In search of the next generation of training sets for language models', arXiv:2406.11794 (DCLM-Baseline: fastText on OpenHermes-2.5 + r/ELI5 positives vs random RefinedWeb negatives, ~400k docs, top-10% threshold; 30.2 Core vs 29.0/27.1/26.1).
- **Cost:** CPU-buildable, low effort (~half a day; reuses the fastText training path). Only new cost is sourcing the positive corpora (public HF datasets).

#### Per-shard dedup as the default + cluster-size logging

- **What:** Make the default dedup SCOPE per-shard (deduplicate within each CommonCrawl snapshot/file group), not naively global across the whole corpus, and record each duplicate cluster's size in the kept document's metadata so the corpus can later be re-hydrated (upsampled by cluster size) if desired.
- **Why (2026):** Teaches the fundamental that dedup is NOT monotonically good and that SCOPE changes the resulting distribution — a subtlety the A4 baseline (global line/MinHash dedup) hides. FineWeb found per-snapshot MinHash (5-grams, 0.75 Jaccard) beat global cross-dump dedup: global dedup disproportionately removed content and the surviving distribution was lower-quality (it effectively upsampled older/worse data). This is a small change to the dedup driver but it converts 'I ran MinHash' into 'I reasoned about dedup scope,' which is exactly the data-engineering judgment a frontier RE is screened on. Honest caveat: the win is corpus/scale-dependent — the point is to make scope an explicit, logged decision.
- **Build:** In the MinHash dedup driver, add a `scope` parameter ('per_shard' default vs 'global') controlling whether candidate-collision buckets are formed within a shard group or across all inputs; when dropping all-but-one per cluster, write `cluster_size` into the kept doc's sidecar metadata. Correctness invariant/test: (a) per-shard mode never merges duplicates that live in different shard groups (a planted cross-shard near-dup pair survives as two docs under per_shard, collapses to one under global) — this proves scope is actually respected; (b) the kept representative's logged cluster_size equals the true cluster member count on a hand-built cluster. The existing test_minhash_deduplication tests still pass (default behavior on a single fixture dir is unchanged).
- **Source:** Penedo et al. 2024, 'The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale', arXiv:2406.17557 (per-snapshot MinHash with 5-grams at 0.75 beat global dedup; global iterative dedup removed disproportionately good data; FineWeb-2 re-hydrates by stored cluster size).
- **Cost:** CPU-buildable, low effort (~half a day; a scope flag + a metadata field on the existing dedup code). No new dependency.

### 🔵 Build labs — scoped, opt-in, after the core is green


#### LLM-as-judge educational-value labels -> distilled lightweight classifier (FineWeb-Edu recipe)

- **What:** Replace heuristic/URL-proxy labels with a strong-LLM rubric score (educational value 0-5 via an additive prompt), then DISTILL those labels into a cheap classifier (linear/logistic head on a frozen sentence-embedding model) that scores the whole corpus. This is the FineWeb-Edu recipe and the modern successor to the fastText URL-proxy classifier.
- **Why (2026):** This is where quality filtering is actually heading in 2026: LLM-as-judge rubric labels capture 'is this educational/useful' far better than 'was this page linked from Wikipedia,' and distilling into an embedding+linear head makes corpus-scale scoring cheap. FineWeb-Edu (Llama-3-70B scoring 460k pages 0-5, then a linear regressor on Snowflake-arctic-embed-m fine-tuned on 410k labels) produced dramatically better MMLU/ARC than unfiltered FineWeb at equal tokens. It deepens the exact signal-design lesson ('the label source IS the recipe') one rung past the fastText URL trick. Honest framing: the load-bearing skill is the label->distill pattern; the LLM-judge labels need an API or a local instruct model, so it's an opt-in lab not a CPU default.
- **Build:** New submodule under curate: edu_classifier.py. Step 1 (labels): prompt an instruct LLM (API or small local model) with an additive 0-5 educational rubric over a few hundred docs; cache labels to disk. Step 2 (distill): embed each doc with a frozen sentence-transformer, fit a logistic/linear head on the cached labels (sklearn or a tiny torch head). Step 3: expose run_classify_quality-shaped (label, score) so it slots into the same pipeline. Minimal buildable version: a few hundred cached labels is enough to demonstrate the pipeline. Correctness invariant/test: on a held-out slice of LLM-labeled docs, the distilled classifier's ranking AUC vs the held-out labels clears a stated bar (e.g. >0.75), AND you report agreement (e.g. Spearman) with the URL-proxy fastText classifier from the A4 core — the explicit signal-design comparison. Predict-before-you-run: state which docs you expect the two classifiers to disagree on.
- **Source:** Penedo et al. 2024, 'The FineWeb Datasets', arXiv:2406.17557 (FineWeb-Edu: Llama-3-70B-Instruct scored 460k pages 0-5 on an additive scale; linear regression head on Snowflake-arctic-embed-m fine-tuned on 410k annotations, 20 epochs, lr 3e-4, frozen encoder; threshold>=3).
- **Cost:** Hybrid: distillation step is CPU-buildable; the LLM-judge labels need an API call or a local instruct model (small GPU or paid API for a few hundred labels — cheap). Medium effort (~1-2 days). Kill-criterion: if distilled AUC vs held-out LLM labels is near chance, the embedding model or rubric is wrong — fix before scaling labels.

#### Semantic (embedding) deduplication — SemDeDup-minimal

- **What:** Dedup at the MEANING level, beyond exact and MinHash fuzzy dedup: embed each document, cluster (k-means), and within each cluster drop documents whose pairwise cosine similarity exceeds a threshold — removing paraphrases/near-semantic-duplicates that share few literal n-grams.
- **Why (2026):** Closes the gap MinHash leaves: two docs can be semantic duplicates (a rephrase, a translation-back, templated content with swapped surface tokens) with low Jaccard, so n-gram dedup misses them. SemDeDup showed you can remove ~50% of web-scale data (C4, LAION) with minimal performance loss and faster training. It teaches the third rung of the dedup ladder (exact -> fuzzy/MinHash -> semantic) and the embedding+clustering primitive that recurs in retrieval and data curation. Honest framing: marginal ON TOP of good MinHash for most pipelines and embedding the whole corpus is costly at scale — hence OPTIONAL, not CORE; the value is understanding when literal dedup is insufficient.
- **Build:** New module curate/semantic_dedup.py: embed docs with a frozen sentence-transformer; k-means cluster (sklearn/faiss-cpu); within each cluster, for pairs with cosine > tau keep one, drop the rest (cluster-and-drop, same transitive-closure logic as MinHash). Minimal buildable version: a few hundred docs, small k, CPU. Correctness invariant/test: two hand-written paraphrases (low literal n-gram overlap, high semantic similarity) land in the same cluster and collapse to one (cosine > tau), while two unrelated docs both survive — i.e. it catches what MinHash provably misses (verify the paraphrase pair has low Jaccard first). Reuse the cluster-and-drop test scaffold from the MinHash build.
- **Source:** Abbas et al. 2023, 'SemDeDup: Data-efficient learning at web-scale through semantic deduplication', arXiv:2303.09540 (embeddings + k-means + cosine threshold; remove 50% of data with minimal loss on C4/LAION). Cross-check: in DCLM (arXiv:2406.11794) SemDeDup alone underperformed the fastText filter (27.1 vs 30.2 Core), so it is a complement, not a replacement.
- **Cost:** CPU-buildable at toy scale (faiss-cpu/sklearn + a small embedding model); medium effort (~1 day). At real scale needs GPU embedding — keep the build to a slice.

#### Perplexity-based quality filter (CCNet / KenLM head-middle-tail)

- **What:** Score each document by the perplexity of a cheap n-gram language model trained on a high-quality reference corpus (Wikipedia), and bucket documents head/middle/tail — low perplexity (looks like Wikipedia) = likely cleaner, very high perplexity = likely gibberish/boilerplate. The classic CCNet quality signal.
- **Why (2026):** A second, orthogonal quality axis to the fastText classifier: instead of 'does a classifier think this is like my positives,' it asks 'how surprising is this under a clean-text model.' It is cheap, transparent, and still in production lineages (RedPajama/CCNet filtering). Teaches that 'quality' has multiple operational definitions and that combining a perplexity filter with a classifier filter is a real design choice. Honest framing: in head-to-head ablations the model-based fastText filter beats perplexity filtering (DCLM: 30.2 vs 29.0 Core), so this is a complementary/awareness-adjacent signal — valuable to understand and cheap to build, but not the primary lever.
- **Build:** New module curate/perplexity_filter.py. Train a small n-gram LM on a Wikipedia sample (kenlm if available, or a pure-Python n-gram model with add-k smoothing to keep it dependency-light and CPU-only); score docs by per-token perplexity; expose (label, score)+threshold like the other filters, with head/middle/tail bucketing. Correctness invariant/test: a clean English fixture (e.g. the Moby-Dick extract already in the test fixtures) gets LOW perplexity (head bucket) while a gibberish/random-token fixture gets HIGH perplexity (tail bucket) — i.e. the ordering is correct on a known-clean vs known-garbage pair. Predict-before-you-run: state the expected bucket for each fixture.
- **Source:** Wenzek et al. 2019, 'CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data', arXiv:1911.00359 (KenLM perplexity vs Wikipedia, head/middle/tail buckets). Comparison point: DCLM (arXiv:2406.11794) found perplexity filtering (29.0 Core) below the fastText filter (30.2).
- **Cost:** CPU-buildable, low-medium effort (~1 day; a pure-Python n-gram model avoids the kenlm build entirely). No GPU.

#### Synthetic rephrasing of web text — WRAP-minimal

- **What:** Augment (not replace) raw web docs by rephrasing them into a cleaner target style (e.g. 'rewrite as a clear encyclopedia entry' or as Q/A) with an instruct model, then train on the mix of original + rephrased. The WRAP recipe; also the mechanism behind Nemotron-CC's synthetic subset.
- **Why (2026):** Synthetic/rephrased data is a major 2026 lever for squeezing more learning per token and per-domain coverage: WRAP reported ~3x pretraining speedup and >10% perplexity improvement on the Pile by rephrasing C4; Nemotron-CC generated 1.9T synthetic tokens by Wikipedia-style rephrasing of low-quality docs. It teaches the curation-as-generation shift and, crucially, the failure mode to guard against: rephrasing can fabricate or drop facts, and rephrased docs can become near-duplicates of their source (distribution collapse). Honest framing: needs an LLM, and the quality/factuality risk is real — hence a scoped opt-in lab.
- **Build:** New module curate/rephrase.py: take a doc, prompt an instruct LLM (API or small local model) to rewrite it in a target style, cache outputs. Minimal buildable version: rephrase a few dozen fixture docs. Correctness invariant/test (this is the load-bearing safety check): (a) the rephrased doc preserves the source's key facts/entities — assert a high overlap of named entities / numbers between source and rephrase (a cheap keyword-preservation test); AND (b) the rephrase is NOT a MinHash near-duplicate of the source under your existing dedup threshold (it genuinely rephrased, not copied) — reuse the MinHash Jaccard from the dedup build. Together these catch the two real failure modes (fact drift and copy-collapse).
- **Source:** Maini et al. 2024, 'Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling' (WRAP), arXiv:2401.16380 (~3x speedup, >10% Pile perplexity improvement). Su et al. 2024, 'Nemotron-CC', arXiv:2412.02595 (1.9T synthetic tokens via Wikipedia-style rephrasing of low-quality docs).
- **Cost:** Needs an LLM API or small local instruct model (cheap for a few dozen docs); medium effort (~1 day). The invariant tests are CPU-only. Kill-criterion: if rephrases fail the entity-preservation test, the prompt/model is hallucinating — stop before scaling.

#### Manual domain reweighting via proxy validation loss (toy mixing, not DoReMi proper)

- **What:** Make data MIXING an explicit, measured knob: assemble the corpus from labeled domains (e.g. web / code / math / wiki), train a tiny proxy model under a few hand-chosen mixture weightings, and pick weights by per-domain proxy validation loss / downstream proxy-eval — the manual, buildable core of data-mixture optimization.
- **Why (2026):** Data mixing is a first-class frontier lever (DoReMi reached baseline accuracy with 2.6x fewer steps just by reweighting Pile domains), and the buildable, fundamentals-teaching version is the manual ablation: vary weights, measure, choose. It connects A4 (which has no mixing component) to the real pretraining-recipe surface and teaches the proxy-model methodology that underlies the whole field. Honest framing: this is explicitly NOT DoReMi proper (no Group DRO, no learned reference) — automated mixture optimization is tiered AWARENESS below; this is the hand-tuned version whose VALUE is the measurement loop, not the algorithm.
- **Build:** New module scaling/data_mixing.py (this touches the data<->scaling seam): given domain-labeled shards and a weight vector, produce a sampled mix; train the existing small from-scratch model for a fixed budget under each of a few weightings; record per-domain held-out loss. Minimal buildable version: 2-3 synthetic 'domains' (e.g. two distinct text distributions), a handful of weightings, tiny model, few steps. Correctness invariant/test: (a) the sampler reproduces the requested weights (empirical domain proportions match the target vector within tolerance) — a pure-CPU determinism test; (b) overfit-one-batch still holds under the mixed loader (wiring is intact). The 'which mix wins' result is GPU/compute-dependent and goes in the ablation ledger, not the unit test.
- **Source:** Xie et al. 2023, 'DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining', arXiv:2305.10429 (proxy model + Group DRO; 2.6x fewer steps to baseline, +6.5pt few-shot on the Pile) — cited as the motivation; the buildable item is the manual reweighting subset, not the DRO algorithm.
- **Cost:** Sampler + invariant tests are CPU-buildable, low effort. The actual mix-selection ablation needs a GPU slice (rent per the repo's follow-the-plan rule). Medium effort (~1-2 days incl. a small GPU ablation).

> ✎ **Fact-check (2026):** Keep the manual item as-is, but add RegMix (arXiv:2407.01492) as the cited modern bridge from this manual loop to automated mixing: 'the manual ablation you build IS the small-model-sweep kernel that RegMix automates via regression.' This makes the source story current and the conceptual ladder (manual sweep -> RegMix regression -> DoReMi DRO) explicit.

#### RegMix — data mixture as regression ➕ *(fact-check addition)*

- **What:** Train N tiny models on N sampled domain mixtures, fit a lightweight regressor (mixture → proxy loss), and predict the optimal mixture. The modern, practical automated-mixing method.
- **Why (2026):** Matches DoReMi at ~10% of the compute (~2% of final-run FLOPs) with only a few short small-model runs, and is the buildable bridge from the manual mixing loop (the manual sweep IS the kernel RegMix automates).
- **Build:** Extend the manual domain-reweighting lab: sample mixtures, train tiny proxies, fit LightGBM/linear on (mixture → proxy loss), validate the predicted optimum beats the hand-tuned mixes.
- **Source:** Liu et al., "RegMix: Data Mixture as Regression for LM Pre-training", arXiv:2407.01492 (ICLR 2025 spotlight).
- **Cost:** Cheap-GPU / CPU proxies; medium.

#### QuaDMix — quality-diversity balanced selection ➕ *(fact-check addition)*

- **What:** A single parameterized sampling function over per-doc quality labels (multiple classifiers) AND diversity labels (domains), with parameters tuned via small-proxy simulations + LightGBM.
- **Why (2026):** Operationalizes the quality-vs-diversity trade-off the pillar otherwise only *defends against* with tests — ~7.2% average benchmark gain over optimizing quality and diversity independently.
- **Build:** Parameterize sampling on (quality_score, domain_class); tune the parameters on small proxy models. Builds directly on your quality classifier + dedup outputs.
- **Source:** Liu et al., "QuaDMix: Quality-Diversity Balanced Data Selection…", arXiv:2504.16511 (April 2025).
- **Cost:** Cheap-GPU; medium.

### ⚪ Know-it — discuss in interviews, do NOT build from scratch


#### Automated data-mixture optimization (DoReMi / Group DRO) at scale

- **What:** Learn domain mixture weights automatically: train a small reference model, then a proxy model with group distributionally robust optimization to up-weight high-loss domains, and transfer the resulting weights to a much larger training run.
- **Why (2026):** You must be able to discuss it — 'how do you choose data mixture weights?' is a standard interview probe, and DoReMi/Group-DRO and its successors (RegMix, online data mixing) are the canonical answers. But the load-bearing benefit only appears at real scale (the value is transferring proxy-derived weights to a 30x-larger model), the proxy+reference training loop is the expensive part, and a CPU toy doesn't teach the actual win — so it is correctly NOT built from scratch here. Build the manual reweighting version (OPTIONAL above) to internalize the measurement loop; understand DoReMi as the automated generalization.
- **Build:** Do NOT build from scratch. Awareness deliverable: be able to whiteboard the two-stage loop (reference model -> proxy with Group DRO that minimizes worst-case excess loss across domains -> apply weights to the big run) and explain why it transfers across scale, and contrast with the manual ablation you DID build. If ever exercised, it would slot onto the scaling/data_mixing seam — but as a rented multi-run experiment, not a unit-tested module.
- **Source:** Xie et al. 2023, 'DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining', arXiv:2305.10429 (proxy 280M -> 8B transfer; 2.6x fewer steps; Group DRO over Pile domains).
- **Cost:** Not buildable on CPU; needs multiple full proxy + reference training runs (multi-GPU). Awareness/discussion item only.

> ✎ **Fact-check (2026):** Promote RegMix to a first-class OPTIONAL_EXTENSION (cite Liu et al. 2024, 'RegMix: Data Mixture as Regression for Language Model Pre-training', arXiv:2407.01492, ICLR 2025) — buildable kernel: train N tiny models on N sampled mixtures, fit a lightweight regressor (e.g. LightGBM/linear) on (mixture -> proxy loss), predict the optimum, validate it beats the hand-tuned mixes. Keep DoReMi as the AWARENESS generalization (Group-DRO worst-case excess loss), but present RegMix as the modern default automated method and contrast the two.

#### Annealing / mid-training: high-quality + long-context data upsampling at end of pretraining

- **What:** The end-of-pretraining 'decay/anneal' phase (Warmup-Stable-Decay style): as the learning rate decays, switch the data mix to upsample small amounts of high-quality domain data (math, code, instruction-style) and long-context documents, which disproportionately lifts target benchmarks.
- **Why (2026):** Arguably THE most load-bearing data-curriculum practice in 2026 — essentially every frontier run ends with a high-quality + long-context annealing phase, and labs even use annealing as a cheap probe for 'is this small dataset valuable?' Llama-3 found annealing on high-quality code/math lifted the 8B model +24.0% on GSM8K and +6.4% on MATH (negligible on 405B — an honest scale caveat); MiniCPM's WSD scheduler is built around this decay phase. Must be discussable in interview. Tiered AWARENESS because the benefit is a property of a full multi-billion-token training run with an LR schedule — you cannot demonstrate it on a CPU toy. The DATA-PREP half (assembling a disjoint, higher-quality anneal mix) is, however, exactly the curation skill the rest of this pillar builds.
- **Build:** Do NOT build the training-side benefit from scratch. Buildable data-prep stub (optional): assemble a held-out 'anneal mix' from the highest-classifier-score docs + long documents, kept DISJOINT from the bulk corpus. Invariant if you build the stub: the anneal set is provably disjoint from the bulk mix (no doc-id overlap) AND its mean quality-classifier score / mean document length is strictly higher than the bulk mix (it really is the high-quality/long tail). The actual loss/benchmark lift is a rented-GPU run, not a unit test. Awareness: explain WSD and why upsampling high-quality data under LR decay helps.
- **Source:** Grattafiori/Dubey et al. 2024, 'The Llama 3 Herd of Models', arXiv:2407.21783 (annealing on high-quality code/math: +24.0% GSM8K, +6.4% MATH on 8B; annealing as a small-dataset value probe). Hu et al. 2024, 'MiniCPM', arXiv:2404.06395 (Warmup-Stable-Decay scheduler with a high-quality decay phase).
- **Cost:** Training benefit not CPU-buildable (needs a full LR-scheduled run). The data-prep stub + its disjointness/quality invariant is CPU-buildable, low effort. Primarily an awareness item.

#### Large-scale synthetic pretraining corpora (Cosmopedia / Persona-Hub / Nemotron-synth)

- **What:** Generating pretraining-scale synthetic data: persona- or seed-driven prompting of an LLM to produce billions of tokens of diverse synthetic textbooks/QA/knowledge text (Cosmopedia, Persona-Hub), and the synthetic subsets that now make up large fractions of frontier corpora (Nemotron-CC's 1.9T synthetic tokens).
- **Why (2026):** A major and growing slice of 2026 pretraining data is synthetic, and 'how would you generate diverse synthetic pretraining data, and what are the risks?' is a live interview topic. Persona-Hub drives diversity via ~1B personas as knowledge carriers; Cosmopedia produced 25B tokens of synthetic textbooks with Mixtral; Nemotron-CC ships 1.9T synthetic tokens. But generating at pretraining scale requires large LLM inference budgets and careful diversity/contamination/factuality control, and the from-scratch lesson (the rephrase loop + its safety invariants) is already captured by the WRAP-minimal OPTIONAL above. So the at-scale corpus generation is awareness, not a build. Honest caveat: diversity collapse and model-collapse feedback loops are unresolved risks here.
- **Build:** Do NOT build at scale. The buildable micro-version (rephrase + entity-preservation + anti-copy invariants) is the WRAP-minimal OPTIONAL item. Awareness deliverable: explain Text-to-Persona / Persona-to-Persona for diversity, the topic-clustering seed approach (Cosmopedia), why synthetic data needs its own dedup + decontamination + factuality checks, and the model-collapse risk of training on too much self-generated data.
- **Source:** Ge/Chan et al. 2024, 'Scaling Synthetic Data Creation with 1,000,000,000 Personas' (Persona-Hub), arXiv:2406.20094. Cosmopedia (HuggingFaceTB, 2024; Mixtral-8x7B, ~25B synthetic tokens, HF dataset card + blog). Su et al. 2024, 'Nemotron-CC', arXiv:2412.02595 (1.9T synthetic tokens).
- **Cost:** Pretraining-scale generation needs large LLM inference budgets (heavy GPU/API). Awareness/discussion item; the buildable kernel is the separate WRAP-minimal OPTIONAL.

#### GPU-scale curation infra: classifier ensembles + distributed fuzzy/semantic dedup (Nemotron-CC / NeMo-Curator)

- **What:** The production data-engineering stack: ensembles of model-based quality classifiers (each scoring a different facet of quality, scores binned 0-19), GPU-accelerated MinHash/connected-components fuzzy dedup and semantic dedup over trillions of tokens, distributed across many nodes (Spark/Ray + RAPIDS), e.g. NVIDIA NeMo-Curator and the Nemotron-CC pipeline. Includes the honest finding that dedup is not monotonically beneficial.
- **Why (2026):** This is what 'build data pipelines for LLM training' means at a frontier lab in 2026 — but it is infrastructure, not an algorithm to re-derive. The mastery payload (MinHash math, the (b,r) S-curve, classifier signal design, dedup scope) is exactly what the CS336-A4 core + the CORE items above teach from scratch; the at-scale version is plumbing (RAPIDS connected-components, multi-node sharding) that should NOT be reimplemented. Must be discussable: classifier ENSEMBLES (Nemotron-CC) beat any single quality signal, and — the credibility nuance — FineWeb showed GLOBAL cross-dump dedup can HURT (it upsamples lower-quality older content), so 'more dedup' is not always better. Tiered AWARENESS because it only exists at trillion-token / multi-node scale.
- **Build:** Do NOT build from scratch. Awareness deliverable: explain how the from-scratch primitives (MinHash+LSH, the quality classifier, decontamination) map onto distributed infra (GPU MinHash + connected components for clustering, classifier ensembles for quality, Spark/Ray sharding), and articulate the dedup-scope nuance (per-shard vs global; global can hurt). The from-scratch CORE/OPTIONAL items ARE the buildable, tested versions of these same primitives — that is the intended division of labor.
- **Source:** Su et al. 2024, 'Nemotron-CC', arXiv:2412.02595 (classifier ensemble, scores binned 0-19; 6.3T tokens incl. 1.9T synthetic; NeMo-Curator pipeline). Penedo et al. 2024, 'FineWeb', arXiv:2406.17557 (global cross-dump dedup HURT vs per-snapshot — dedup is not monotonically beneficial). NVIDIA NeMo-Curator (official repo/docs, 2024-2025: GPU fuzzy + semantic dedup, 30+ heuristic filters).
- **Cost:** Multi-node GPU infrastructure (RAPIDS/Spark/Ray); trillion-token scale. Awareness/interview item only — the buildable equivalents are the from-scratch CORE/OPTIONAL items.

#### "Quality is a domain selector" + multilingual curation (FineWeb-2) ➕ *(fact-check addition)*

- **What:** Empirically, classifier-based quality filters behave more like *domain* selectors than quality measurers; FineWeb-2 is the per-language curation + cluster-size re-hydration pipeline for multilingual data.
- **Why (2026):** Sharpens the "quality is defined by the label source" lesson your two CORE classifier items rest on, and addresses the multilingual classifier-bias failure mode the pillar names but does not build for.
- **Build:** Awareness: a caution to interpret your quality classifier as a domain prior, and a pointer for multilingual pipelines.
- **Source:** "The Data-Quality Illusion", arXiv:2510.00866 (Oct 2025); FineWeb-2, arXiv:2506.20920 (June 2025).
- **Cost:** Awareness.

---


## A5 · Alignment / post-training

> **What a senior RE actually does here (2026):**
>
> A senior RE on the post-training/RLVR pillar in 2026 spends most of the day keeping an RL run interpretable and from diverging, not inventing algorithms. Concretely: (1) Babysit RLVR runs (GRPO-family on verifiable math/code/agentic tasks) on a verl/OpenRLHF/slime stack where rollouts come from a fast inference engine (vLLM/SGLang) and the gradient step runs on FSDP/Megatron — the two engines are SEPARATE processes, so the first thing you stare at every morning is the dashboard: token entropy (collapse = dead run), KL(policy‖ref) and KL(policy‖old), the importance-sampling-ratio histogram, reward mean/std split by format-vs-answer, and response-length-by-correctness (the verbosity reward-hack tell). (2) Chase specific failure modes: entropy collapse (fix with clip-higher / KL / lowering LR), length explosion on wrong answers (the Dr.GRPO/length-norm bug, or overlong-shaping), reward hacking (the grader/judge gets gamed — e.g. a model emitting a bare delimiter token that fools an LLM-judge), and train-inference logprob mismatch (the rollout engine and trainer disagree on logprobs for the same checkpoint, silently biasing the gradient — diagnosed via the train↔infer KL, fixed with truncated importance sampling or, recently, FP16 rollouts). (3) Curate and grade data: write/repair verifiers and reward functions (string-normalized math graders, code unit-test harnesses, format checkers), build difficulty-filtered prompt sets (dynamic sampling drops all-correct/all-wrong groups), and run dedup/decontamination against eval sets. (4) Own the recipe: SFT → preference optimization (DPO/variants) → RLVR, in the Tülu-3/OLMo-2 mold, and decide per-stage hyperparameters (β, clip range, KL coeff, group size, off-policy epochs). (5) Evaluate honestly: pass@1 vs maj@k/self-consistency vs best-of-N-with-a-verifier, on held-out sets, with contamination checks; an RE is judged on whether the curve is real, not whether it went up. "Good" looks like: a reproducible, fully-logged run whose reward gains transfer to held-out evals without length/format hacking, and whose every hyperparameter choice you can defend from first principles. The scarcest skill is debugging WHY an RL run died, which is a logging-and-stability discipline, not a new-loss discipline.


*Fact-check verdict: Strong, current, and unusually disciplined — but with one load-bearing factual error of exactly the 'plausible-but-wrong that ships' kind this review exists to catch. Of 14 enhancements, 12 are CONFIRMED and 0 are REJECTED: every cited paper resolves to a real, correctly-attributed, non-stale primary source (DeepSeekMath/DAPO/Dr.GRPO/GSPO/RLOO/DPO-family/s1/Cobbe/Gao/FP16/R1/OpenRLHF all verified by ID, title, author, and the specific technical claim), the formulas check out (k3=exp(r)−r−1; RLOO=(G/(G−1))(r_i−r̄); GSPO length-normalized geometric-mean ratio), and the tiering is honest — including the two I pressure-tested (RLOO=CORE as a teaching default, preference-family=OPTIONAL because the DPO primitive already exists). Scope discipline is excellent: PRM, async infra, and the train-inference correction are correctly held at AWARENESS with explicit 'do not build from scratch' bounds, and every CORE/OPTIONAL item is genuinely CPU-buildable and snapshot-guarded, with no research-project creep. Two NEEDS_FIX: (1) the flagship KL entry wrongly says DeepSeek-R1 REMOVES the KL term — R1 (2501.12948 Eq.1) demonstrably KEEPS −β·D_KL; the removal belongs to DAPO and Dr.GRPO (the k3/Schulman parts are correct); and (2) the train-inference-mismatch entry mis-states the FP16 fix as 'FP16 rollouts' — the paper (2510.26788) shows BF16 rounding is the root cause and the fix is UNIFORM FP16 across train+inference (rollout-only FP16 against a BF16 trainer would not converge the logprobs). Both are sub-claim fixes inside otherwise-sound entries, not rejections. The most significant gap is CISPO/MiniMax-M1 (2506.13585), which completes the IS-variant axis the proposal half-builds; the entropy-mechanism (2505.22617) and online/iterative DPO (2401.10020) are secondary verified misses. Fix the two attributions and add CISPO, and this is an accurate, fundamentals-first 2026 curriculum.*


### 🟢 Modern defaults — adopt these into the from-scratch build


#### KL penalty as an optimized term (k1/k2/k3 estimators), not just a logged metric

- **What:** Add an optional per-token KL(policy‖ref) penalty to the GRPO objective, computed with the unbiased low-variance k3 estimator exp(r)−r−1 where r = logπ_ref − logπ_θ. Expose a kl_coeff knob (0 disables it). The repo already LOGS KL; this lets the loss OPTIMIZE against it.
- **Why (2026):** Load-bearing fundamental. KL-to-reference is the single most-discussed RLHF stability lever in interviews and the thing that stops a policy from collapsing onto the reward's blind spots. Honest nuance: DeepSeek-R1's GRPO and DAPO actually REMOVE the KL term for pure-reasoning RLVR (it caps exploration), while Tülu-3 RLVR and all RLHF-with-a-reward-model keep it — so the load-bearing skill is knowing the k1/k2/k3 estimators cold AND knowing WHEN to turn KL off. Owning the toggle is the senior signal.
- **Build:** In algos/, add compute_kl_penalty(policy_log_probs, ref_log_probs) returning the k3 estimator per token, and a kl_coeff arg to compute_policy_gradient_loss / grpo_microbatch_train_step that adds kl_coeff * kl to the per-token loss before masked aggregation. Reuse the existing get_response_log_probs to score the frozen ref model. Invariant test: k3 is non-negative elementwise; E[k3] under samples from π_θ approximates true KL and has lower variance than k1=(−r); with kl_coeff=0 the loss exactly equals the current GRPO loss (snapshot-identical). Schulman's three-estimator note is the reference.
- **Source:** Schulman, 'Approximating KL Divergence' (joschu.net, 2020); KL term in GRPO: DeepSeekMath arXiv:2402.03300 (2024); KL removed: DAPO arXiv:2503.14476 (2025) and Dr.GRPO arXiv:2503.20783 (2025).
- **Cost:** CPU-buildable, ~2-3h. Pure tensor math; tested against the existing GRPO snapshot with kl_coeff=0. Kill-criterion: none — it is a strict superset of the current loss.

> ✎ **Fact-check (2026):** Replace 'DeepSeek-R1's GRPO ... REMOVE the KL term' with: 'DeepSeek-R1 (2501.12948, Eq.1) and DeepSeekMath GRPO (2402.03300, Eq.3-4) KEEP the −β·D_KL term and use the k3 estimator; DAPO (2503.14476) and Dr.GRPO (2503.20783) are the ones that DROP KL for pure-reasoning RLVR.' The 'know when to turn KL off' senior-signal framing is correct once the attribution is fixed.

#### RLOO advantage (REINFORCE leave-one-out) as a 4th entry in the advantage/loss ablation

- **What:** Add the leave-one-out baseline: for a group of G rollouts, A_i = r_i − mean(r_{j≠i}) = (G/(G−1))·(r_i − mean(r_all)). It is the exactly-UNBIASED group baseline, in contrast to GRPO's biased mean/std normalization.
- **Why (2026):** Load-bearing as a teaching contrast, marginal as a production default. RLOO (Ahmadian et al.) showed plain REINFORCE-with-a-good-baseline beats PPO and DPO for RLHF, and it is the cleanest way to SEE why GRPO's /std and self-inclusion introduce bias. It is ~5 lines on top of the existing compute_group_normalized_rewards and gives a third advantage estimator (REINFORCE-baseline / GRPO / RLOO) for the same ablation harness the assignment already runs.
- **Build:** In algos/, extend compute_group_normalized_rewards with a baseline='loo' branch (or a sibling compute_rloo_advantages) that subtracts the leave-one-out mean. Invariant test: per group, sum of LOO advantages need NOT be zero (unlike mean-subtraction) but E[advantage]=0 over the data; verify A_i = (G/(G−1))(r_i − r̄) algebraically against a hand-computed 3-sample group; confirm it plugs into the existing reinforce_with_baseline loss path unchanged.
- **Source:** Ahmadian et al., 'Back to Basics: Revisiting REINFORCE Style Optimization for RLHF in LLMs', arXiv:2402.14740 (2024).
- **Cost:** CPU-buildable, ~1h. Trivial extension of existing code with a closed-form correctness check.

#### Off-policy epochs>1 with cached old_log_probs (make the clip actually do something)

- **What:** Run >1 gradient epoch per rollout batch by caching π_old logprobs once after sampling and reusing them across epochs, so the GRPO-clip ratio ρ_t = exp(logπ_θ − logπ_old) departs from 1 and the clip becomes load-bearing. The scaffold's own defaults flag epochs_per_rollout_batch as the toggle.
- **Why (2026):** Fundamental and currently MISSING in spirit: with the repo's on-policy default (epochs=1) ρ_t≡1 and the entire GRPO-clip machinery is a no-op — the clip only earns its keep off-policy. Every real RLVR stack is mildly off-policy (sample once, update several times) because rollouts dominate cost. This is the cheapest way to make the trust-region concept real instead of decorative.
- **Build:** In the grpo_train_loop, after rollout+grade, run get_response_log_probs ONCE under the current (now 'old') policy, detach and cache it; loop epochs, recomputing only π_θ logprobs and calling grpo_clip with the cached old_log_probs. Invariant tests: (a) with epochs=1, first-step ρ_t==1 and loss == on-policy loss (snapshot-identical); (b) old_log_probs carries no grad (assert .requires_grad is False); (c) clip-fraction in the metadata is 0 on epoch 1 and >0 on later epochs once ρ drifts.
- **Source:** CS336 A5 §7.2 (epochs_per_rollout_batch, cache old logprobs once); PPO clip mechanics: Schulman et al. PPO arXiv:1707.06347 (2017); GRPO-clip: DeepSeekMath arXiv:2402.03300 (2024).
- **Cost:** CPU-buildable to test (toy env), GPU for a real curve. ~3-4h. Kill-criterion: if epoch=1 snapshot changes, the caching is wired wrong.

#### Self-consistency (maj@k) as the default reasoning-eval metric

- **What:** Sample k chains at temperature>0, extract the answer from each, and report the majority-vote answer (maj@k) alongside greedy pass@1. The simplest, training-free inference-time-scaling baseline.
- **Why (2026):** Load-bearing as an evaluation honesty upgrade. Reporting only greedy pass@1 understates a reasoning model and hides variance; maj@k is the universal cheap baseline every reasoning-RL result is compared against (and the floor that best-of-N-with-a-verifier must beat). It reuses the rollout + grader you already built and turns the eval into a real test-time-compute curve (accuracy vs k).
- **Build:** In envs/ + a small eval util: for each task, decode k responses via the existing rollout seam, parse answers with the rewards/ grader's normalizer, take the mode. Invariant test: maj@1 == greedy-pass@1 on a fixed seed; on a hand-built toy where 2/3 chains agree on the correct answer, maj@3 == correct even if one chain is wrong; accuracy is monotone-ish increasing in k on a real model.
- **Source:** Wang et al., 'Self-Consistency Improves Chain of Thought Reasoning', arXiv:2203.11171 (2022, ICLR 2023).
- **Cost:** CPU-testable on a toy env, cheap GPU for real numbers. ~2h. No new model needed.

### 🔵 Build labs — scoped, opt-in, after the core is green


#### DAPO Clip-Higher (decoupled ε_low / ε_high) + token-mean loss aggregation

- **What:** Two scoped upgrades from DAPO: (1) Clip-Higher splits the single cliprange into ε_low and ε_high (e.g. 0.2 / 0.28) so up-clipping is looser than down-clipping, preserving probability mass on rare exploratory tokens; (2) token-mean aggregation pools the loss over ALL response tokens in the batch (sum/total-token-count) instead of per-sequence mean-then-batch-mean.
- **Why (2026):** Directly extends what the repo already has. Clip-Higher is DAPO's headline anti-entropy-collapse fix and a one-line change to compute_grpo_clip_loss; token-mean is exactly the masked_mean global-vs-per-sequence choice the assignment's length-norm study already foregrounds, so DAPO's aggregation is a natural fourth point in that study (per-token mean / per-sequence mean / Dr.GRPO sum-over-constant / DAPO token-mean). Honest scope: the full DAPO win (50 AIME on 32B) needs scale + dynamic sampling + overlong-shaping; the two pieces here are the cheap, transferable, CPU-testable fundamentals.
- **Build:** In algos/: give compute_grpo_clip_loss separate clip_low/clip_high args (default both to the old cliprange → backward-compatible) and add a loss_agg_mode='token_mean' path in grpo_microbatch_train_step that divides the summed masked loss by the total response-token count across the batch. Invariant tests: with clip_low==clip_high the snapshot is unchanged; an advantage>0 token with ρ above 1+ε_high is clipped at 1+ε_high (not 1+ε_low); token_mean == masked_mean(dim=None) over the batch's response tokens; reproduce DAPO's batch-of-2 length-bias example showing token-mean ≠ per-sequence mean.
- **Source:** Yu et al., 'DAPO: An Open-Source LLM Reinforcement Learning System at Scale', arXiv:2503.14476 (2025); recipe params (clip 0.2/0.28, loss_agg_mode=token-mean) from the verl DAPO docs.
- **Cost:** CPU-buildable, ~2-3h. Both are small, snapshot-guarded loss tweaks; the headline accuracy gain is GPU+scale-bound (awareness).

#### DAPO Dynamic Sampling (drop zero-gradient groups) + overlong filtering/shaping

- **What:** (1) Dynamic Sampling: filter out prompt-groups whose rollouts are all-correct or all-wrong (group reward std=0 ⇒ all advantages=0 ⇒ zero gradient), resampling until the batch has G usable groups. (2) Overlong shaping: apply a linear length penalty inside a buffer near max_response_length (e.g. last 4096 tokens) instead of a hard truncation, so over-length samples give a soft signal rather than reward noise.
- **Why (2026):** A genuine efficiency + stability lever you can SEE: all-same-reward groups waste a forward/backward pass for exactly zero learning, and length-truncation injects reward noise that destabilizes long-CoT RL. Both are buildable and CPU-testable on the verifiable env. Honest: marginal at toy scale, real at long-CoT scale — but the concept (zero-variance groups are wasted compute) is a fundamental every RLVR RE invokes.
- **Build:** In algos/envs rollout path: after grading, drop groups where raw-reward variance is 0 and keep sampling (cap retries, e.g. max_num_gen_batches=10); add an overlong penalty in rewards/ that subtracts a value rising linearly from 0→penalty_factor over the buffer window. Invariant tests: a batch of all-correct groups yields an empty post-filter batch (and the loop resamples, not crashes); the advantage tensor after filtering has no all-zero group; a response exactly at max_len−buffer has 0 penalty, one at max_len has full penalty, linear in between.
- **Source:** Yu et al., DAPO, arXiv:2503.14476 (2025); buffer/penalty config (overlong_buffer.len=4096, penalty_factor=1.0; filter groups all-1-or-0) from the verl DAPO docs.
- **Cost:** CPU-buildable, ~3h. Dynamic sampling touches the loop; overlong shaping is local to rewards/. No GPU needed to verify correctness.

#### GSPO sequence-level importance ratio as a loss variant (token- vs sequence-level IS)

- **What:** An importance ratio defined on the WHOLE sequence — the length-normalized geometric mean s(θ) = (π_θ(y|x)/π_old(y|x))^(1/|y|) — with clipping/optimization done at the sequence level, instead of GRPO's per-token ratio. One clipped scalar per response.
- **Why (2026):** The cleanest 'token-level vs sequence-level IS' contrast, which is THE 2025 stability debate. GSPO's thesis: per-token ratios accumulate high-variance noise over long responses (and break under MoE expert-routing drift), causing collapse on gigantic models; sequence-level IS fixed Qwen3's large-scale/MoE RL. Honest scope: the win is at large/MoE scale — at 1.5B dense it is roughly a wash with token-level GRPO, so build it to OWN the variance argument, not because it beats GRPO on your toy run.
- **Build:** In algos/, add loss_type='gspo': compute one sequence ratio per response from the masked sum of (logπ_θ − logπ_old) divided by response length, exponentiate, clip to [1−ε,1+ε], multiply by the (scalar) advantage. Invariant tests: for a length-1 response GSPO reduces to the token-level ratio; the ratio is invariant to padding (masked length only); gradient flows to all response tokens equally through the geometric mean; with old==θ the ratio is exactly 1 and the loss == −A.
- **Source:** Zheng et al. (Qwen), 'Group Sequence Policy Optimization', arXiv:2507.18071 (2025); Qwen GSPO blog (qwenlm.github.io/blog/gspo).
- **Cost:** CPU-buildable, ~3h. A self-contained loss variant with closed-form length-1 and identity checks; large-scale/MoE benefit is awareness-only.

#### Offline preference-loss family: extend the DPO primitive to IPO / SimPO / KTO

- **What:** Generalize the existing per-instance DPO loss into a small family sharing the same log-prob machinery: IPO (squared-loss on the margin, fixes DPO overfitting), SimPO (reference-FREE, length-normalized average-logprob reward + target margin γ), and KTO (learns from unpaired binary good/bad signals via a prospect-theory value function).
- **Why (2026):** Preference optimization is the other half of post-training and a guaranteed interview block ('DPO vs its variants'). The high-value lesson is structural: SimPO/ORPO drop the reference model (cheaper, but can over-shorten), IPO swaps the BT log-sigmoid for a bounded squared loss, KTO escapes paired data. Honest caveat the literature itself states: many DPO variants do NOT reliably beat well-tuned DPO — so the deliverable is understanding the design axes (reference-free? length-norm? paired?), not a leaderboard claim.
- **Build:** In algos/, refactor the DPO loss into compute_preference_loss(loss_type) reusing get_response_log_probs: DPO = −logσ(β·(Δ_θ − Δ_ref)); IPO = (Δ_θ − Δ_ref − 1/(2β))² ; SimPO = −logσ(β/|y|·(logπ_θ(y_w) − logπ_θ(y_l)) − γ) (no ref); KTO from the per-example value function on unpaired labels. Invariant tests: DPO branch still hits the scaffold's tiny-gpt2 fixture (loss ≈ 0.5785 at β=0.5); SimPO/ORPO branches never call lm_ref (assert it can be None); IPO loss is bounded and its minimizer matches the DPO optimum on a 2-point synthetic; KTO with a 50/50 good/bad toy gives a finite, sign-correct gradient.
- **Source:** DPO: Rafailov et al. arXiv:2305.18290 (2023); IPO: Azar et al. arXiv:2310.12036 (2023); SimPO: Meng et al. arXiv:2405.14734 (2024); KTO: Ethayarajh et al. arXiv:2402.01306 (2024); ORPO: Hong et al. arXiv:2403.07691 (2024).
- **Cost:** CPU-buildable, ~3-4h. All are closed-form losses tested on tiny fixtures; the DPO snapshot pins the refactor.

#### Generative reward model / LLM-as-judge as a swappable RewardFn (with the reward-hacking caveat)

- **What:** A grader that conforms to the existing RewardFn protocol but, instead of string-matching, prompts a judge model to verify the response and reads a Yes/No (or score) — optionally via a CoT-then-verdict generative verifier whose 'Yes'-token probability is the scalar reward.
- **Why (2026):** This is how RLVR escapes the math/code sandbox into non-verifiable tasks (the Tülu-3/RLHF frontier). The pillar-defining lesson is the FAILURE mode, not the success: judges are reward-hackable — a model can learn to emit a single delimiter/affirmation token that fools an LLM-judge into a high score ('One Token to Fool LLM-as-a-Judge'). Building one teaches both why generative RMs scale (they convert inference compute into verifier accuracy) and why your length/format/judge-confidence logging is non-negotiable. NOT a research project — a swappable grader plus the standard guardrails.
- **Build:** In rewards/, implement a JudgeRewardFn(*, response_text, ground_truth) -> RewardDict satisfying envs/protocol.py's RewardFn: render a verification prompt, decode a verdict, map to {reward, format_reward, answer_reward}. Keep it backend-agnostic (text-in/dict-out). For CI, stub the judge with a deterministic rule so the loop is testable. Invariant tests: the judge grader is drop-in interchangeable with r1_zero_reward_fn in compute_group_normalized_rewards (same dict keys, same shapes); a known adversarial string (bare '</answer>' or 'Yes') is flagged by a sanity check, demonstrating the hack; on a held-out set, judge agreement with the rule-based grader is reported (don't trust it blindly).
- **Source:** Mahan et al., 'Generative Reward Models', arXiv:2410.12832 (2024); Zheng et al., 'Judging LLM-as-a-Judge' (MT-Bench/Chatbot Arena), arXiv:2306.05685 (2023); reward-hacking caveat: 'One Token to Fool LLM-as-a-Judge', arXiv:2507.08794 (2025).
- **Cost:** CPU-buildable with a stubbed judge, ~3h; a real judge needs a second model (GPU/API). Scoped strictly to a grader + guardrail tests, no novel-research scope.

#### s1 budget forcing — a test-time-compute controller in the decoder

- **What:** Control the thinking budget at decode time: cap thinking tokens (force the end-of-thinking + answer), or EXTEND it by suppressing the end-of-thinking token and appending 'Wait' to make the model continue and self-correct. A training-free way to trade compute for accuracy.
- **Why (2026):** The cheapest, most surprising inference-time-scaling result: SFT on 1K traces + budget forcing matched o1-preview-class math, purely by manipulating decode. It cleanly separates 'inference-time scaling' from 'RL training' and slots straight into the existing sampler. Honest: it is most effective on a model already SFT'd for long CoT; it is a controller, not a trainer.
- **Build:** In sampling.py, add a budget-forcing decode wrapper: track a thinking-token counter; on hitting max_thinking, inject the end-of-think delimiter and switch to answer mode; to extend, when the model emits end-of-think before min_thinking, replace it with 'Wait' and continue. Invariant tests: with cap = +inf and no extension it equals ordinary decoding (byte-identical at fixed seed); the forced delimiter appears exactly at the cap; appending 'Wait' increases generated length monotonically; on a toy 'self-correcting' stub the extended trace flips a wrong answer to right.
- **Source:** Muennighoff et al., 's1: Simple test-time scaling', arXiv:2501.19393 (2025).
- **Cost:** CPU-buildable as a decode controller (logic-testable with a stub LM), GPU for real accuracy lift. ~2-3h.

#### Best-of-N with a verifier (reward-ranked sampling) as an inference-time-scaling lab

- **What:** Sample N candidates, score each with the verifiable grader (or a reward/judge model), and return the top-scoring one — the verifier-guided counterpart to maj@k, and the inference-time mirror of Expert Iteration's filter-and-keep step.
- **Why (2026):** Closes the loop between the EI/STaR baseline (filter-correct then SFT) and inference-time scaling (filter-best at decode): same machinery, different time. It is the standard way to convert a good verifier into accuracy without more training, and the honest baseline that exposes reward-model quality — best-of-N is only as good as the scorer, and it is the canonical setting where a weak RM gets over-optimized (Goodhart).
- **Build:** In envs/ + eval util: reuse the rollout seam to draw N responses, score with the rewards/ grader, argmax. Invariant tests: best-of-1 == greedy at fixed seed; with the oracle verifiable grader, best-of-N accuracy is monotone non-decreasing in N (it can only pick a correct one if present); pass@N (any-correct) upper-bounds best-of-N, which upper-bounds maj@N when the verifier is perfect; with a deliberately noisy scorer, best-of-N accuracy can DROP as N grows (the over-optimization demo).
- **Source:** Cobbe et al., 'Training Verifiers to Solve Math Word Problems' (GSM8K, verifier reranking), arXiv:2110.14168 (2021); over-optimization: Gao et al., 'Scaling Laws for Reward Model Overoptimization', arXiv:2210.10760 (2022).
- **Cost:** CPU-testable on a toy env (with the oracle grader the monotonicity check is exact), cheap GPU for real numbers. ~2h.

#### CISPO — clipped IS-weight policy optimization ➕ *(fact-check addition)*

- **What:** Clip the importance-sampling *weight* but retain *every* token's gradient — vs GRPO's per-token clip-and-mask (which can zero a token) and GSPO's sequence-level clip. No rare/exploratory "fork" token is ever dropped from the update.
- **Why (2026):** Completes the token-vs-sequence-vs-weight importance-sampling design axis the GSPO item half-builds; the algorithm behind MiniMax-M1. A from-scratch RE should be able to whiteboard why it differs from both clip-and-mask and sequence-clip.
- **Build:** A loss variant alongside GRPO/GSPO in algos/: clip the IS weight, keep all per-token gradients. CPU-buildable; the third point in the IS-clipping ablation.
- **Source:** CISPO / MiniMax-M1, arXiv:2506.13585 (June 2025).
- **Cost:** CPU loss variant; small.

#### Online / iterative preference optimization (Self-Rewarding) ➕ *(fact-check addition)*

- **What:** Iterate LLM-as-judge labeling + DPO so preferences are collected on-policy — the online counterpart to the offline DPO/IPO/SimPO/KTO family.
- **Why (2026):** The offline-vs-online preference-optimization axis is a guaranteed interview topic, and this is a one-loop extension of the DPO primitive you already build.
- **Build:** Wrap the DPO primitive in a loop: sample pairs from the current policy, label with an LLM-as-judge (or your generative RM), run a DPO step, repeat.
- **Source:** Yuan et al., "Self-Rewarding Language Models", arXiv:2401.10020 (ICML 2024).
- **Cost:** Medium (needs a judge).

#### Knowledge distillation — logit KD · on-policy reverse-KL · sequence-level (the serve-cheap lever) ➕ *(GDM-alignment addition)*

- **What:** Compress a trained teacher into a small, cheap-to-serve student — three variants on one seam: **logit KD** (Hinton: `L = α·CE(y,z_s) + (1−α)·T²·KL(softmax(z_t/T) ‖ softmax(z_s/T))`, the `T²` restoring the soft-target gradient scale; forward KL ⇒ mass-covering); **on-policy / reverse-KL KD** (GKD: `KL(student ‖ teacher)` on student-sampled sequences ⇒ mode-seeking, kills the train/inference distribution shift); **sequence-level KD** (Kim & Rush: SFT the student on teacher samples ⇒ KD as Expert-Iteration with a teacher).
- **Why (2026):** The single highest-leverage infra lever a frontier pre-training lead names — distillation transfers a giant teacher's statistics into the *served* student, and one distillation-infra rewrite "uncovered new scaling laws that directly enabled a Flash-class model" (Feinberg/GDM). The served model is usually a distilled student; owning the three KL directions and the `T²` scale is the from-scratch value, not the trillion-token infra.
- **Build:** New `algos/distill.py`. Teacher = the green A1 substrate (or a larger config); student = a smaller `ModelConfig`. Logit KD reuses `cross_entropy`; on-policy KD reuses the **rollout seam** (sample from the student, score teacher logprobs); sequence-level KD reuses the **A5 SFT step**. Pair with the A3 fitter for the distillation-scaling mini-fit (the A3 know-it above).
- **Invariant (test-first, falsifiable):** `T=1, α=1` ⇒ KD loss is byte-identical to `cross_entropy` (degenerate equivalence); overfit-one-batch drives student logits → teacher logits as the KD weight → 1; loss-at-init ≈ log V still holds on the student head; on a hand-built 2-mode target, forward-KL is mass-covering while reverse-KL is mode-seeking (predict which before running).
- **Source:** Hinton et al., 'Distilling the Knowledge in a Neural Network', arXiv:1503.02531 (2015); Kim & Rush, 'Sequence-Level Knowledge Distillation', arXiv:1606.07947 (2016); Agarwal et al., 'On-Policy Distillation (GKD)', arXiv:2306.13649 (2023); Busbridge et al., 'Distillation Scaling Laws', arXiv:2502.08606 (2025).
- **Cost:** CPU-buildable (tiny teacher→student); medium. Kill: if `T=1,α=1` doesn't reduce to plain CE, the temperature/scale wiring is wrong — fix before trusting any KD number.

#### Multi-turn / agentic tool-use RL on the env protocol ➕ *(frontier-judgment addition)*

- **What:** Extend single-turn verifiable-reward RL (GRPO on math/Countdown) to **multi-turn, tool-using episodes** — the policy interleaves reasoning with tool calls (code-exec · search · calculator), consumes the tool result, and continues over several turns to a terminal verifiable reward. Trajectory-level GRPO/Dr.GRPO advantage over the whole episode.
- **Why (2026):** This is where post-training actually moved — agentic coding, computer-use, and deep-research agents are all multi-turn tool-use RL. Single-turn RLVR (math) is the warm-up; the long-horizon tool-interleaved regime is the live frontier and the natural post-"aha" next step. Highest-signal A5 direction, and it reuses what you already own.
- **Build:** Extend `VerifiableEnv` to a multi-turn episode (state → action {text + optional tool-call} → tool result → … → terminal grade); the **rollout seam** collects trajectories; GRPO over a trajectory-level advantage with per-turn masking. Start with one deterministic tool (calculator or a code-exec sandbox) on the existing Countdown/math env.
- **Invariant (predict-first):** a 1-turn episode (no tool call) reduces **exactly** to the current single-turn `grpo_train_loop` (degenerate-case equivalence); per-turn masking sums to the single-turn loss; reward/length logging fires per turn.
- **Source:** the 2026 agentic-RL direction (tool-use / long-horizon RLVR) — verify specific citations before relying on them.
- **Cost:** CPU-buildable on a deterministic tool; medium. The forward edge of the A5 crown.

### ⚪ Know-it — discuss in interviews, do NOT build from scratch


#### Train-inference logprob mismatch: measure it, then correct it with truncated importance sampling

- **What:** When rollouts come from a fast inference engine (vLLM/SGLang) and gradients from the trainer (FSDP/Megatron), the two compute slightly different logprobs for the SAME checkpoint, biasing the policy gradient. Fixes: a sequence/token-level truncated importance-sampling correction (re-weight by π_train/π_infer, clipped), or — recently — running rollouts in FP16 to make the engines agree.
- **Why (2026):** This is a top day-to-day production failure mode in 2026 async RL, and a sharp interview discriminator — but the FULL correction only matters when you actually have two engines, which a from-scratch single-process loop does not. So: AWARENESS for the correction, with one cheap thing you SHOULD build now. The repo's discipline #4 already mandates logging the train↔inference KL drift; that diagnostic is the buildable 20%. The TIS correction factor and FP16-rollout fix are awareness until a real two-engine rollout path exists.
- **Build:** Buildable-minimal: in utils/monitors.py, when (if) the rollout engine differs from the trainer, log per-token KL(π_infer‖π_train) and the IS-ratio histogram — the drift signal. The correction itself is awareness: multiply the per-token loss by clip(exp(logπ_train − logπ_infer), max=C) (truncated IS). Invariant for the minimal version: same-engine rollouts give train↔infer KL ≈ 0 and IS ratio ≈ 1 (the null check that proves the diagnostic is wired right). Do NOT build the engine split from scratch.
- **Source:** 'Defeating the Training-Inference Mismatch via FP16', arXiv:2510.26788 (2025); truncated IS for off-policy/async LLM RL is standard in verl/slime/AReaL stacks (see verl docs and the async-RL literature).
- **Cost:** Awareness; the minimal drift-log is the same code as discipline #4 (already required). Full correction is GPU + two-engine, out of from-scratch scope.

> ✎ **Fact-check (2026):** Replace 'FP16 rollouts' / 'running rollouts in FP16' with 'switching BOTH training and inference to uniform FP16 (BF16's wide-range rounding error is the root cause; FP16's higher mantissa precision removes it) — Qi/Liu et al., Defeating the Training-Inference Mismatch via FP16, arXiv:2510.26788 (2025).' Truncated-IS attribution to verl/slime/AReaL stacks is fine.

#### Process reward models (PRM): outcome vs process supervision and why pure-PRM-RL fell out of favor

- **What:** PRMs score each intermediate reasoning STEP rather than only the final answer (outcome RM). Step labels come from humans (PRM800K) or automatically via Monte-Carlo rollout completion (Math-Shepherd / OmegaPRM). Used for verifier-reranking (best-of-N) and, historically, step-level PPO.
- **Why (2026):** Must-discuss in any reasoning interview, but you should NOT build a PRM-RL pipeline from scratch — it needs step-label annotation (expensive or MC-heavy), a separate trained model, and it is the textbook reward-hacking magnet (DeepSeek-R1's report explicitly tried and dropped PRM-driven RL because the dense PRM gets gamed and is hard to train at scale, favoring the simple verifiable outcome reward). The honest 2026 takeaway: outcome RLVR won for math/code; PRMs survive mainly as verifiers/rerankers and a research area, not as the RL signal.
- **Build:** Awareness-first. The buildable toy (optional, clearly labeled) is a step-level reward SHAPING in rewards/: split a response on a step delimiter and assign a per-step shaped reward, then show in the logs that a model can inflate step count to farm shaped reward (the hacking demo) — illustrating WHY R1 dropped it. Do not train a real PRM. Invariant for the toy: the shaped per-step rewards sum to the outcome reward when steps are not gamed; a padded-step response gets caught by the length-by-correctness log.
- **Source:** Lightman et al., 'Let's Verify Step by Step' (PRM800K), arXiv:2305.20050 (2023); Wang et al., 'Math-Shepherd', arXiv:2312.08935 (2023); DeepSeek-R1 PRM discussion, arXiv:2501.12948 (2025).
- **Cost:** Awareness; optional toy shaping is CPU-buildable (~1-2h) and explicitly a concept demo, not a PRM. No PRM training from scratch.

#### Async / distributed RL infrastructure (verl · OpenRLHF · slime · TRL) — read the seams, don't rebuild them

- **What:** The production RLVR stack: a disaggregated learner (FSDP/Megatron) + sampler (vLLM/SGLang) connected by a weight-sync + experience-queue layer (often Ray), running rollouts and gradient steps asynchronously for throughput. verl, OpenRLHF, slime, and TRL are the reference implementations.
- **Why (2026):** This is where the pillar actually runs at frontier scale, and being able to whiteboard the learner/sampler split, weight broadcast, off-policy staleness, and the resulting train-inference mismatch is core interview signal. But it is months of systems engineering and explicitly OUT of from-scratch scope — the from-scratch value is the ALGORITHM and the LOGGING, which you own; the orchestration you should be able to discuss and use, not reimplement.
- **Build:** Do not build. Map it onto what you have: your envs/protocol.py + rollout seam IS the learner/sampler boundary in miniature (the CPU LocalBackend = sampler, algos/ = learner). The awareness deliverable is one paragraph in an ADR connecting your single-process grpo_train_loop to the disaggregated async design, naming the staleness/IS-correction it would require. Optionally read verl's GRPO/DAPO configs to see your hyperparameters in their config schema.
- **Source:** verl (Volcano Engine RL, github.com/volcengine/verl, 2024–2025); OpenRLHF arXiv:2405.11143 (2024); slime (THUDM, 2025); TRL (huggingface/trl).
- **Cost:** Awareness only — zero from-scratch build. Use the libraries for real multi-GPU runs (the repo's plan already rents GPUs for RL); reimplementing the infra carries no mastery for this pillar.

#### Entropy-collapse mechanism (Clip-Cov / KL-Cov) ➕ *(fact-check addition)*

- **What:** Entropy change is ∝ covariance(action-probability, logit-change); Clip-Cov / KL-Cov target the high-covariance tokens directly.
- **Why (2026):** The theory *behind* DAPO's clip-higher and a stronger anti-collapse lever — know the mechanism, not just the symptom-fix.
- **Build:** Awareness: explains why entropy collapses and what a principled fix targets.
- **Source:** Cui et al., "The Entropy Mechanism of RL for Reasoning LMs", arXiv:2505.22617 (May 2025).
- **Cost:** Awareness.

#### PEFT — LoRA / QLoRA: know when NOT to use it ➕ *(frontier-judgment addition)*

- **What:** Low-rank adaptation freezes the base weights and trains `ΔW = BA` (rank r ≪ d); QLoRA adds a 4-bit NF4-quantized frozen base + double-quant + paged optimizers so a large model fine-tunes on one GPU.
- **Why (2026), and the cut:** This is **not** a frontier-flagship technique — labs full-fine-tune their flagships (they have the compute, and PEFT leaves quality on the table). The senior signal is the *judgment*: LoRA/QLoRA is the right tool for **multi-tenant adapter serving** (one base, many cheap task adapters) and **GPU-poor / on-device** fine-tuning, not for the model you train from scratch. Derive the low-rank update; state precisely where it pays and where it doesn't; do **not** cargo-cult it into the substrate.
- **Build:** Do not build into the substrate. Awareness; a `ΔW=BA` adapter on one `Linear` is a ~1-hour demo if ever wanted, clearly labeled non-default.
- **Source:** Hu et al., 'LoRA', arXiv:2106.09685 (2021); Dettmers et al., 'QLoRA', arXiv:2305.14314 (2023).
- **Cost:** Awareness. The value is the discrimination, not the technique.

---

## Capstone · DELTA — linear-attention decode (the inference/kernel axis)

> The one frontier topic the A1–A5 pillars are light on, and the axis the **DELTA capstone** targets
> (design doc + 4-week plan in the workspace root; tracked in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §7).
> All ⚪ **know-it / interview-awareness** — the talking points `/master` connects each DELTA concept to
> in its "connect to frontier + interview" step.
>
> **Extended landscape (reference, 2026-06-22):** the fuller serving-stack / hardware-roadmap / frontier-model map around this inference axis (with a compute-tiered curriculum) is in [`../../reference/Frontier_Inference_2026_Research_Brief.md`](../../reference/Frontier_Inference_2026_Research_Brief.md). Reference only; this doc + [`../../DELTA.md`](../../DELTA.md) remain canonical.

#### Why batch-1 decode is memory-bound (and why 100% MFU is an anti-goal)
- **What / why (2026):** the full recurrent state round-trips HBM every token; GDN, DeltaNet, Mamba(-2) all sit **<1 FLOP/byte** on the H100 roofline at decode — *more* memory-bound than attention. The contribution is framed as % of the bandwidth ceiling, not a speedup multiplier.
- **MFU is not the target.** 100% Model-FLOPs-Utilization is *impossible by construction* — it would require the accelerator to do nothing but back-to-back `matmul` with zero HBM reads; real nets must run activation functions, attention, and intermediate HBM writes that are inherently slower than tensor-core matmul. The job is *inference co-design*: pick layer shapes / matrix topologies that saturate the units, balancing predictable scaling quality against MFU — low-tens MFU is not "wasted compute," and chasing 100% is the wrong objective.
- **Interview:** *"what's the arithmetic intensity of a linear-attention decode step, and why can't more FLOPs fix it?"* · *"why is 100% MFU an anti-goal?"*  *(USC arXiv 2603.05931.)*

#### GatedDeltaNet-2 — erase/write decoupling
- **What / why (2026):** a channel-wise erase gate `b` (key axis) + write gate `w` (value axis) generalize GDN/KDA; the **erase gate carries most of the gain**. Recovers KDA, then GDN, as the gates and decay collapse.
- **Interview:** *"derive the gated delta rule; why decouple erase from write?"*  *(NVIDIA arXiv 2605.22791.)*

#### Fused state-resident decode kernel
- **What / why (2026):** restructure ~5 HBM passes → **1 read + 1 write**, `S` resident in SMEM/registers; **Triton-first** (the win is traffic, not tensor cores). NVIDIA ships training/prefill kernels only — no decode kernel, no checkpoint — so the decode regime is open.
- **Interview:** *"fuse this recurrence; which passes leak to HBM; where's the roofline ceiling?"*

#### The free-decoupling thesis + the prediction discipline
- **What / why (2026):** the 2 extra gates are O(d) loads amortized against O(d²) state traffic → **~free at decode** (P2); every claim pre-registered with a falsifier (P1–P7), kills committed up front.
- **Interview:** *"what would falsify your result?"* — the research-taste gate that separates a research engineer from a kernel typist.
- **Cost:** Know-it / interview-awareness. The build *is* the capstone; `/master` each concept the week you implement it.

---


> **Pointers:** the fundamentals live in the per-assignment guides ([`assignment_guides/INDEX.md`](assignment_guides/INDEX.md)); the build order is [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md); the official spec + test oracle is `../../lectures/`.
