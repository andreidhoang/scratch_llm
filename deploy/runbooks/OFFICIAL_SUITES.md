# Official-scaffold acceptance — CS336 A2/A4/A5 (exec-spec node W9)

> The from-scratch `scratch_llm` modules were wired behind the **official Stanford CS336 scaffold
> adapters** (`/workspace/lectures/assignment{2,4,5}-*/tests/adapters.py`) and run against the
> course's own `tests/` + snapshot fixtures — the objective correctness oracle (repo `CLAUDE.md`:
> "the adapter tests verify your implementation is correct"). Adapters are thin glue in the
> throwaway scaffold checkouts (not committed here); **no `scratch_llm` source was modified to make
> a test pass** — the one real gap found (A2 FSDP) was fixed in the source and re-run.
>
> Reproduce: `PYTHONPATH=/workspace/scratch_llm/src[:<scaffold cs336-basics>] CUDA_VISIBLE_DEVICES=''
> python -m pytest tests/ -o addopts=''` from each scaffold root (light pure-python deps —
> einops/einx/jaxtyping — into a private dir on PYTHONPATH, never the shared venv).

## Scoreboard (2026-07-04)

| Assignment | Passed | Failed | Blocked | Notes |
|---|---|---|---|---|
| **A2 systems** | **10** | **0** | 4 | 4 blocked = Triton FA2 (GPU-only `@skipif(not cuda)`). FSDP gap fixed (47f49a1). |
| **A4 data** | **20** | **0** | 1 | 1 blocked = `test_classify_quality` needs a trained model artifact (not bundled, ADR-0016). |
| **A5 alignment** | **20** | **0** | 6 | 6 blocked = out-of-scope course-rote (MMLU/GSM8K parsers, packed-SFT dataloader — never built). |
| **Total** | **50** | **0** | 11 | 0 real defects. Blocks are GPU-tier / out-of-load-bearing-scope, each explained. |

## A2 — systems (`assignment2-systems`) · 10P / 0F / 4 blocked

| Test | Result | scratch_llm symbol |
|---|---|---|
| `test_sharded_optimizer.py` (ToyModel, TiedWeights) | 2 PASS | `utils.zero1.ShardedOptimizer` (ZeRO-1) |
| `test_ddp.py` (ToyModel, TiedWeights) | 2 PASS | `utils.ddp.DDP` (overlap) |
| `test_attention.py` pure-torch FA2 (fwd O/L + bwd dQ/dK/dV to 1e-2) | PASS | `kernels.flash_attention.FlashAttentionPyTorch` |
| `test_fsdp.py::test_fsdp_correctness[fp32/fp16]` | 2 PASS | `utils.fsdp.FSDP` |
| `test_fsdp.py::test_fsdp_gradient_sync[fp32/fp16]` | **2 PASS (after 47f49a1)** | `utils.fsdp.FSDP` |
| `test_attention.py` Triton (×4) | 4 BLOCKED | GPU-only; deferred to the rented box (see `A2_multigpu_nccl_bench.md`) |

**The one real finding (fixed).** Our FSDP originally sharded *all* requires-grad params (incl.
1-D RMSNorm weights / biases), so their grads differed across ranks — `test_fsdp_gradient_sync`
expects small params **replicated** with all-reduced identical grads. Numerically harmless
(`test_fsdp_correctness` always passed), but a real policy divergence from the official ZeRO-3
contract. **Fixed in `src/scratch_llm/utils/fsdp.py` (47f49a1):** classify by `ndim` — `≥2-D`
matrices sharded (reduce-scatter), `≤1-D` params replicated with mean-all-reduced grads. Both
gradient-sync tests now pass; correctness stays green.

## A4 — data (`assignment4-data`) · 20P / 0F / 1 blocked

All 11 adapters delegate to `data.{dedup,filters,quality}` as thin passthroughs (signatures already
match). Ran hermetically (~1.7 s); the three fastText models (`lid.176.bin`, Jigsaw NSFW, Jigsaw
hatespeech) were present in `/workspace/.data_models`, so langid + NSFW + toxic run live.

- **dedup** 3/3 (exact · minhash-exact · minhash-fuzzy) · **extract** 1/1 (resiliparse byte-for-byte
  Moby-Dick match) · **langid** 2/2 (en, zh) · **pii** 5/5 (emails/phones/ips) · **gopher** 7/7 ·
  **toxicity** 2/2 (nsfw, toxic).
- **BLOCKED** `test_classify_quality` — needs a trained quality model at
  `/workspace/.data_models/quality_wiki_cc.bin` (not bundled by design, ADR-0016). Contract
  confirmed: a throwaway model trained via `data.quality.train_quality_classifier` on the fixtures
  returns `("wiki", >0)` / `("cc", >0)` — the official `(label, score)` shape. Unblock by training +
  saving a model (see `A4_full_slice.md`).

## A5 — alignment (`assignment5-alignment`) · 20P / 0F / 6 blocked

Wired to `algos.{sft,grpo,dpo}` + `rewards`. Ran offline (`HF_HUB_OFFLINE=1`, all fixtures on disk).

- **test_grpo.py** 19 PASS — `tokenize_prompt_and_output`, `get_response_log_probs`,
  `compute_rollout_rewards`, group-normalized **GRPO** (unbiased-std) + **Dr.GRPO** (mean-center),
  policy-gradient (none / grpo_clip), aggregate (sequence + constant), `grpo_train_step`
  (standard_on_policy · grpo_constant · dr_grpo · off_policy) — all snapshot-exact (rtol 1e-4 /
  train-step atol 1e-6).
- **test_dpo.py** 1 PASS — `per_instance_dpo_loss` = **0.9104**, reproducing the checked-out 2026
  scaffold's pinned value exactly (the older ≈0.5785 hint is stale). tiny-gpt2 loads in eval mode.
- **BLOCKED (6, out of load-bearing scope, grep-confirmed absent):** `run_parse_mmlu_response` /
  `run_parse_gsm8k_response` (drive 4 `test_metrics.py` eval-parser tests) and
  `get_packed_sft_dataset` / `run_iterate_batches` (drive 2 `test_data.py` packed-SFT tests) — the
  course-rote eval-harness + dataloader the A5 guide tags COURSE-ROTE / not load-bearing.

**Transparency (passed but adapter-supplied, not scratch-native):** the 2026 scaffold added GRPO
*variant* modes beyond our GRPO/Dr.GRPO scope (baseline=`none`/RFT, normalizer=`mean`/MaxRL, `noclip`,
`gspo`); the adapter filled these by arithmetic and matched the oracle — they pass the official
contract but don't validate a from-scratch implementation of those specific variants (ADR-0017 scopes
us to GRPO + Dr.GRPO).

**Convention note (for anyone re-wiring):** the 2026 `run_aggregate_loss_across_microbatch`
`constant` mode = `total_sum/const` applied once globally, so constant-mode microbatch losses must
**not** be divided by `grad_accum` again (our `grpo_microbatch_train_step` divides uniformly and
aggregates `constant` as a per-batch mean). The adapter handles this; a direct primitive re-wire
needs the adjustment.
