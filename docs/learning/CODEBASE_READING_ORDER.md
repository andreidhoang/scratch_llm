# Codebase Reading Order — the frontier-RE pre-flight

> **What this is.** How a senior AI research engineer at a frontier lab (or Karpathy — whose own
> rule is *"loss-at-init = log(vocab), then overfit one batch, then trace one example end to end"*)
> reads this codebase **before spending a GPU-hour**. It is a *reading order*, not a re-derivation:
> the exact files, in the order a tensor flows, each with what to verify and the oracle that proves
> it. For the deep first-principles derivation of each component, that lives in
> [`roadmap_model/`](roadmap_model/README.md) + [`roadmap/`](roadmap/README.md); this doc is the
> **orientation pass** that precedes a real run.
>
> **Anchors** are `file · symbol · line`, verified at HEAD `56dd1e3` (2026-07-13). Line numbers
> drift — trust the symbol name, re-grep the line. Pair this with
> [`../../deploy/runbooks/frontier_gpu_day.md`](../../deploy/runbooks/frontier_gpu_day.md) (the
> commands) and [`../../bench/RESULTS.md`](../../bench/RESULTS.md) (the pre-registered numbers).

## The method (why this order, not alphabetical)

1. **Read the correctness oracles before the code.** The cheapest bugs to catch are the ones the
   tests already encode. Read what *correct* means first, then read the implementation knowing what
   it must satisfy.
2. **Follow a tensor, not a directory.** Trace one token: `byte → id → batch → forward → loss →
   backward → optimizer step → checkpoint → sample`. Flow-order files compose in your head;
   alphabetical files don't.
3. **Read the thing you'll actually run *last*, against its pre-registered number.** Know the
   predicted number before the run — the driver + the ledger are the final read, not the first.

---

## Pass 0 — Know what you're about to run (15 min)

| File | Why |
|---|---|
| `docs/STATUS.md` | single source of truth: what's built / measured / green |
| `deploy/runbooks/frontier_gpu_day.md` | the exact commands you'll run + the kill gates |
| `bench/RESULTS.md` §Frontier ablations | the **pre-registered predictions** — loss bands, F1's 1.1–1.4× band, KILL thresholds |

The frontier tell: `RESULTS.md` labels everything **measured vs implemented**. If a number is not in
the ledger it has not been measured — treat it as a hypothesis (FOP-4).

## Pass 1 — The correctness oracles (read the *tests*, 30 min)

These four **are the spec**. If they are green, the wiring is sound — the highest-leverage half hour.

| Test · anchor | The oracle it encodes |
|---|---|
| `tests/test_model.py · test_loss_at_init_is_log_vocab · 46` | fresh LM CE ≈ `log(V)` — cheapest bug-catcher (off ⇒ head/embed/mask bug) |
| `tests/test_optim.py · test_overfit_one_batch · 67` | one batch → loss <0.05; can't ⇒ optimizer/data/loss *wiring* broken, not the data |
| `tests/test_train.py` | next-token alignment (`targets = inputs+1`), checkpoint round-trip, fixed-seed reproducibility |
| `tests/test_optimizer_race.py` | the F1 harness invariants — the val-eval hook is a byte-identical no-op |

## Pass 2 — Trace one token through the forward (the spine, 2–3 h)

Read `model.py` in **dependency order**, predicting the tensor shape at each hop.

| # | File · symbol · line | Verify cold |
|---|---|---|
| 1 | `tokenizer.py · train_bpe · 130` → `encode · 269` / `decode · 284` | byte→id, round-trip invariant, specials stay single ids |
| 2 | `model.py · RMSNorm · 137` | why RMS not LayerNorm (one reduction, no mean-subtract) |
| 3 | `model.py · RotaryPositionalEmbedding · 173` | relative position via rotation, applied to q/k pre-attention |
| 4 | `model.py · SwiGLU · 215` | gated MLP, the `8/3·d` width |
| 5 | `model.py · MultiHeadSelfAttention · 228` (`.forward · 253`) | GQA KV-sharing, causal mask, qk-norm path, `use_triton_attention` seam |
| 6 | `model.py · TransformerBlock · 378` (`.forward · 402`) | pre-norm residual `x + attn(norm(x))` |
| 7 | `model.py · TransformerLM · 418` | embed → N blocks → final norm → head |
| 8 | `model.py · cross_entropy · 490` | logsumexp form (cancels `log∘exp`) — what Pass 1's `log(V)` oracle checks |

Cross-ref: `kv_cache.py` (the decode-time KV substrate, extracted from `model.py`).

## Pass 3 — The training step (make it *learn*, 1–2 h)

| # | File · symbol · line | Verify |
|---|---|---|
| 1 | `optim.py · AdamW · 25` | decoupled weight decay, bias-corrected step |
| 2 | `optim.py · _zeropower_via_newtonschulz5 · 137` → `Muon · 170` | NS5 orthogonalization + Moonlight RMS-match (why one LR serves both) |
| 3 | `optim.py · split_muon_adamw_params · 280` | **silent-corruption risk**: tied embed/head + 1-D → AdamW, 2-D blocks → Muon; a mis-partition trains wrong AND passes loss-at-init |
| 4 | `optim.py · build_optimizer · 357` / `CombinedOptimizer · 318` / `apply_qk_clip · 392` | how the hybrid is assembled + stepped; F9's QK-clip |
| 5 | `train.py · get_batch · 40` → `train · 208` → `_val_loss · 173` → `save_checkpoint · 65` / `build_model_from_checkpoint · 114` | the loop you'll run for hours; the NaN guard + config-carrying resume (rental safety-net) |

## Pass 4 — The data pipeline (garbage in = wasted GPU, 1 h)

| File · symbol · line | Verify |
|---|---|
| `data/shards.py · tokenize_to_shard · 74` / `build_dataset · 193` / `download_fineweb_slice · 255` | memmap shard integrity, next-token alignment into the loader |
| `data/decontaminate.py` (13-gram gate) | **eval leakage kills your headline number** — the filter must run *before* BPE training (`build_dataset` applies it up front) |

## Pass 5 — What you'll actually run (the driver + its number, 1 h)

| File · symbol | Why last |
|---|---|
| `eval/report_card.py` + `eval/metrics.py` | know exactly what `val_bpb` measures |
| `eval/optimizer_race.py` (`run_race` / `sweep_lr` / `tokens_to_match`) | the F1 logic; the **mandatory tuned-AdamW sweep** is the honest-science core (2509.02046) |
| `bench/optimizer_race.py` | the CLI you type; re-read `RESULTS.md`'s F1 pre-registration beside it |
| `sampling.py · generate · 127` | the talking artifact; greedy = argmax, stop at eot |

## Passes 6–7 — Only what you're ablating / post-training

- **Frontier deltas** (read the one you're running): `dsa.py` (F8.1 sparse attn), `linear_attn.py`
  (F10.1 Gated-DeltaNet), `kernels/flash_attention_triton.py` (FA2 fwd+bwd + profile in
  `bench/flash_bwd_roofline.py`), `mla.py` (perf-owned).
- **RL** (if post-training): `algos/sft.py` → `algos/chat_sft.py` → `algos/grpo.py`
  (`compute_group_normalized_rewards · 107` / `compute_grpo_clip_loss · 174` /
  `grpo_microbatch_train_step · 255`) → `rewards/r1_zero.py` → `envs/`. Load-bearing detail: the
  **response-mask alignment** — off-by-one ⇒ every RL number silently corrupt.
- **Chat close-the-loop**: `chat.py` (template + single-id specials) → `algos/chat_sft.py`
  (assistant-masked SFT) → `chat_cli.py` (`ChatSession` / `batch_reply`) → `speedrun.py` (the spine
  that chains tokenizer → pretrain → midtrain → sft → eval → sample → chat).

---

## If you have one hour — the irreducible 6

`tokenizer.py` · `model.py` · `optim.py` · `train.py` · `data/shards.py` · `bench/RESULTS.md`.
Everything else is a delta on those.

## The two-minute pre-flight (before you type the run command)

```bash
pytest tests/test_model.py::test_loss_at_init_is_log_vocab tests/test_optim.py::test_overfit_one_batch -q
```
Green ⇒ the model predicts uniform at init and the optimizer can actually drive a batch to zero —
Karpathy's whole "don't trust, verify" gate in two tests. Then follow
`deploy/runbooks/frontier_gpu_day.md` top to bottom.
