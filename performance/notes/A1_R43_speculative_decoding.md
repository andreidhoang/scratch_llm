# A1 Rung 4.3 — Speculative decoding (lossless) · engineering spec (pre-registration)

> **Spec-with-falsifiers (FOP-2).** Written BEFORE building (2026-07-04, delegate mode ADR-0013).
> Parent: `performance/A1_transformer_inference.md` §4.3. Oracle = R1 single-stream greedy
> (`sampling.generate`, temperature 0). Gate = **with a greedy target, spec-decode output is
> token-IDENTICAL to non-spec greedy** (losslessness is the whole point — prove it); measure
> acceptance rate and end-to-end speedup. Kill criterion: any divergence from greedy target ⇒ the
> rejection rule is wrong; stop and fix the math before measuring speedup.

## 1. Mechanism (first principles) — raise decode arithmetic intensity from the other side

Decode is memory-bound: one weight read per token (AI ≈ 1). Batching raises AI by sharing a weight
read across requests; **speculation** raises it by verifying *many candidate tokens in ONE target
forward*. A cheap **drafter** proposes K tokens; the target runs a single forward over the K+1
positions (the last committed token + the K drafts) and emits greedy argmax logits at each; we
accept the longest prefix of drafts that matches the target's own greedy argmax, and at the first
mismatch we take the target's argmax token instead (a "free" correct token). This produces EXACTLY
the sequence the target would have produced alone — **lossless for greedy by construction** (for
sampling, the Leviathan/Chen rejection rule `accept if u ≤ p_target/p_draft, else resample from
`(p_target − p_draft)_+`` gives distribution-identity; greedy is the degenerate case we test first).

Speedup ≈ `E[accepted+1] / (1 + K·c_draft/c_target)` where `c_draft/c_target` is the per-token cost
ratio — so a cheap, well-correlated drafter with high acceptance wins; a mis-correlated or expensive
drafter LOSES (spec overhead with no accepted tokens). That tradeoff is the measurement.

**Drafter choice (from-scratch, training-free):** we have no distilled draft model, so the honest
drafter is **prompt-lookup / n-gram speculation** (Saxena 2023): find the most recent occurrence of
the last n committed tokens in the context and propose the tokens that followed. Training-free,
model-agnostic, genuinely lossless, and gives REAL acceptance on repetitive/structured text
(code, JSON, repeated phrases) — the exact regime the spec says acceptance is highest. The engine is
drafter-agnostic (a `Drafter` protocol: `propose(committed_ids, k) -> list[int]`), so a smaller
draft `TransformerLM` slots in later; prompt-lookup is the measurable default here.

## 2. Build (serving-layer, not a Mode-3 kernel)

`serving/speculative.py`:
- `Drafter` protocol + `NGramDrafter(n, k)` (prompt-lookup) + optional `ModelDrafter(draft_model)`.
- `speculative_generate(target, drafter, prompt_ids, max_new_tokens, device) -> (tokens, SpecStats)`.
  Loop: drafter proposes ≤K tokens → one target forward over [last_committed, draft_0..draft_{K-1}]
  with the KV cache (s = accepted_context + K) → greedy-verify prefix → commit accepted + 1 → **roll
  the KV cache back** to the committed length (rejected drafts' KV discarded). Uses the single-request
  `KVCache` (batch-1); the verification forward is the existing s>1 cached-attention path.
- `SpecStats(n_target_forwards, n_tokens, n_drafted, n_accepted, acceptance_rate, mean_accept_len)`.

KV rollback: append the K+1 verification KV, then truncate to committed length (a `KVCache.truncate`
/ length reset — the rejected suffix is never attended again). This is the one new cache primitive.

## 3. Predictions (pre-registered)

Standing GPU sm_120, `RUNG1_CONFIG` (~1B bf16), single-stream, K=4.

| # | experiment | predicted | bound / rationale |
|---|---|---|---|
| P4.3.1 | greedy losslessness | **spec output == `generate` greedy, token-exact**, every prompt | correctness gate — kill line: any mismatch is a rejection-rule/rollback bug |
| P4.3.2 | acceptance rate, repetitive prompt (n-gram hits) | **40–75%** mean accept fraction | prompt-lookup lands on real repeats in structured text (code/JSON/repeated phrases) |
| P4.3.3 | acceptance rate, random/unstructured prompt | **near 0** | no n-gram structure ⇒ drafts miss; the honest negative that bounds the method |
| P4.3.4 | end-to-end speedup, repetitive prompt | **1.3–2.0×** tok/s vs plain greedy (or explicitly killed) | E[accept+1]/(1+K·c_draft/c_target); n-gram drafter is ~free so c_draft≈0 ⇒ speedup≈E[accept+1]/1 capped by target-forward-over-K+1 cost |
| P4.3.5 | speedup, random prompt | **<1× (net LOSS)** | zero acceptance + the K+1-wide verify forward > 1-wide decode; register the loss, it proves the AI tradeoff |

**Kill criteria:** (a) losslessness fails once ⇒ fix rejection/rollback before any speedup number.
(b) acceptance >0 but speedup <1 on repetitive text ⇒ the verify forward isn't amortizing (profile
the K+1 forward cost). (c) prompt-lookup acceptance ~0 even on hand-crafted repetitive input ⇒
n-gram match/propose bug.

## 4. DoD

- [ ] Losslessness proven: spec greedy == `generate` greedy, token-exact, across prompts incl. adversarial (empty-ish, highly repetitive, random).
- [ ] Acceptance rate + speedup measured on repetitive vs random prompts (the structured-text contrast) → `bench/RESULTS.md`.
- [ ] Mechanism-literacy writeup (design-note paragraph): Medusa (target-side extra heads, tree-verified), EAGLE-2/3 (feature-level autoregression + dynamic draft trees, current lossless SOTA), MTP (DeepSeek-V3 next-2-token heads, ~85–90% 2nd-token acceptance → ~1.8× TPS).
- [ ] KV-rollback primitive tested (rejected-suffix KV never attended; committed KV bit-identical to plain decode).

## Mechanism literacy (DoD writeup) — the lossless-speculation family

All four are LOSSLESS (target verifies; output = target's distribution) and differ only in how the
K candidate tokens are produced — cheaper/better drafts ⇒ higher acceptance ⇒ more tokens/forward.

- **Draft model (this rung's `ModelDrafter`):** a separate small model autoregresses K tokens. Simple,
  but the draft's own K forwards cost `K·c_draft`; the win needs `c_draft ≪ c_target` AND high
  agreement (usually a distilled/aligned draft). The classic Leviathan/Chen formulation.
- **Prompt-lookup / n-gram (this rung's measured `NGramDrafter`):** draft is a string match in the
  context — **zero model cost**, so speedup ≈ E[accept+1]. Wins on repetitive/structured text (code,
  JSON, long-context copy); ~0 acceptance on novel text. The training-free baseline.
- **Medusa:** bolt **K extra decoding heads** onto the *target* itself, each predicting the token at
  offset +1..+K from the same hidden state; a tree of candidates is verified in one target forward.
  No separate model; heads are cheap-trained. Acceptance limited by the heads' independence
  (they don't see each other's tokens).
- **EAGLE-2/3:** autoregress the DRAFT at the **feature (hidden-state) level**, not the token level,
  re-using the target's second-to-top features + a light head, with a **dynamic draft TREE** whose
  branches are pruned by confidence. Current lossless SOTA — feature-level drafting tracks the target
  far better than token-level, pushing acceptance/depth up (EAGLE-3 adds training-time test-time-scaling).
- **MTP (DeepSeek-V3):** train **multi-token-prediction heads** (next-2) jointly with the model; at
  serve time the 2nd-token head drafts, ~**85–90% 2nd-token acceptance → ~1.8× TPS**. The draft is
  a first-class part of the model, so agreement is high by construction.

The through-line: speculation raises decode arithmetic intensity by verifying many tokens per weight
read; the research frontier is entirely about **making the K drafts cheaper and more likely to be
accepted** (better drafts = higher E[accept]), the term that dominates the speedup formula.
