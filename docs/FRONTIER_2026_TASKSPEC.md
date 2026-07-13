# Frontier 2026 — Task Spec (the buildable next-phase DAG)

> **What this is.** The implementation-grade task breakdown for the close-the-loop / frontier-ablation
> front (ADR-0018). The strategy + falsifiers live in
> [`FRONTIER_2026_ABLATIONS.md`](FRONTIER_2026_ABLATIONS.md); **this doc is the buildable spec** —
> per rung: exact interfaces (file → change, grounded in the real code), config additions,
> CPU-green tests, a pre-registered measured falsifier + kill, dependencies, and the perf-front
> zone note. Loop status: **CLOSED** (F1 Muon + train-wiring/F4 + eval harness + speedrun spine
> shipped, GPU-verified talking sample). This is what comes next.
>
> **Provenance.** Produced 2026-07-04 by an 8-agent workflow (`w77bbp4pb`): one deep-spec agent per
> rung reading the ACTUAL files it would touch + verifying primary sources, plus a sequencing critic.
> 18 rung specs, 0 agent errors. The critic's genuinely-missing items are folded in as §B added
> rungs; its scope corrections are §D.
>
> **Reference oracle available:** the venv vendors `transformers/models/{deepseek_v2,deepseek_v3,
> deepseek_v32,glm4_moe,nanochat,qwen3}` — read these as implementation oracles (re-own, don't copy).

<!-- Next-node: F1-run GPU DAY on the sm120 box — harness ✅ 2026-07-12 + F9 observer ✅ 2026-07-13: run `python bench/optimizer_race.py --data-dir <shards> --depth 8 --tokens 7e8 --amp bf16` (bf16 EAGER, never +compile on sm120); F3 spec-acceptance + F9 max-logit read off the same trained ckpt/run. Build-next on CPU (batch 2): A5 ✅ + A6 ✅ 2026-07-13 (chat-SFT + REPL — the loop now TALKS via chat template) → NEXT: F7a aha → F2a MTP → A7 DDP wiring → A4 midtrain (spec now complete) → F10.2/F8.2 wiring → F11. Shipped 2026-07-13: A0 · A3 · F3-harness · F8.1 · F9 · F10.1 · A5 · A6. · UPDATE this line when a rung ships -->

> ▶ **START HERE (fresh session).** The loop is **CLOSED** (F1 Muon · train-wiring/F4 · eval report
> card · speedrun spine shipped, GPU-verified talking sample). **A1 real-corpus shards ✅ 2026-07-09**
> (`data/shards.py`: memmap uint16/uint32 shards + staged tokenizer + FineWeb slice + shard-backed
> `speedrun --data-dir`). **A2 checkpoint chaining ✅ 2026-07-09** (config-carrying `save_checkpoint`
> / `build_model_from_checkpoint` + `Tokenizer.save/load` + the speedrun stage spine with
> `work_dir`/`resume` + the pinned stage-transition optimizer policy; measured in `bench/RESULTS.md`
> §Frontier). **F1-run HARNESS ✅ 2026-07-12** (`eval/optimizer_race.py` + `bench/optimizer_race.py`:
> pure metrics + A/B driver on the real `train()` + the MANDATORY LR-tuned-AdamW sweep arm + NS
> instrument; sweep protocol pre-registered in RESULTS.md). **SIX-RUNG CPU BATCH ✅ 2026-07-13** —
> A0 decontamination (`data/decontaminate.py`, 13-gram gate + shard `doc_filter` seam) · A3 chat
> template (`chat.py` + `SpeedrunConfig.chat`) · F3 acceptance harness (`eval/spec_acceptance.py` +
> CLI; measured falsifier awaits a trained ckpt) · F8.1 DSA core (`dsa.py`, top-k==dense + indexer
> recall 0.99 vs 0.30 random) · F9 QK-clip guard (observer + `apply_qk_clip` + train wiring — **run
> F1 with `track_attn_logits=True`; its falsifier rides that run**) · F10.1 Gated-DeltaNet
> (`linear_attn.py`, chunkwise==recurrent==ref, state 1024× < GQA-8 KV @4k). A4's truncated spec is
> now COMPLETE in §A (ids-only shards; mask reconstructed at SFT collate). **Next node → the F1-run
> GPU day** (sm120 box; see the Next-node marker above for the exact command), and on CPU →
> **batch 2: A5 → A6 → F7a → F2a → A7** . Full order + deps in §0; near-term picks in §E.
>
> **Build protocol — every rung, no exceptions:** ① pre-register the rung's falsifier in
> `bench/RESULTS.md` §Frontier ablations *before* running (predict-before-run) → ② build **test-first**
> → ③ green-CI (`ruff check` + `ruff format --check` + `pyright` + `pytest -m "not gpu"`) → ④ commit
> (`<area>: <imperative>`) → ⑤ **push origin main** (standing policy; `gh` is authed). **Zone:** never
> edit perf-owned files (`mla.py`, `serving/`, `kernels/`, `quant/`, `utils/{tp_mlp,pipeline_schedule,
> ep_moe,mfu}.py`) — satisfy their Protocols (`Drafter`/`RewardFn`/`VerifiableEnv`) from F-front code;
> pull-rebase + precise `git add` (never `-A`) on shared files (three fronts share this checkout).
> **On ship, advance the `Next-node:` marker above.**

## §0 — EV-ranked critical path (build in this order)

Two tracks interleave: **A** = complete the loop into a *chat* model (the d20 artifact); **B** =
the frontier ablation study (the differentiating research). `[S/M/L]` = effort.

| # | Rung | Track | Eff | Deps | One-line |
|---|---|---|---|---|---|
| 1 | **A1** real-corpus shards | A | M | — | FineWeb-EDU → memmap uint16 token shards (the toy corpus can't make a d20) |
| 2 | **A0** decontamination gate | A | M | A1 | n-gram-overlap strip of eval sets (GSM8K/MMLU/Countdown/report-card) from train shards |
| 3 | **A2** checkpoint chaining | A | M | — | config-carrying save/resume — the rental safety-net + stage spine A4/A5/A6 hang off |
| 4 | **F1-run** iso-FLOP Muon vs AdamW | B | M | A2 | the **pending headline** — tokens-to-match + NS overhead on the real loop |
| 5 | **A3** chat specials + template | A | S | — | `<\|bos\|>/<\|user\|>/<\|assistant\|>/<\|eot\|>` + `render_conversation` (unlocks chat) |
| 6 | **F2a** MTP train head | B | L | — | DeepSeek-V3 D=1 aux head — additive/byte-identical; bake into the single d20 pretrain |
| 7 | **F3** de-confound serving | B | S | A2 | re-run n-gram acceptance on a *trained* ckpt → restore the falsified prompt-dependence |
| 8 | **A4** midtrain stage | A | M | A1,A3 | SmolTalk + MC + tool-use adapters → masked shards |
| 9 | **A5** SFT stage | A | M | A2,A3 | assistant-only masked-CE over the chat template (reuses `algos/sft.py`) |
| 10 | **A6** chat REPL / serve | A | M | A2,A3,A5 | point the serving stack at the SFT'd ckpt → **a model you talk to** |
| 11 | **F7a** GRPO aha harness | B | M | — | wire `grpo_train_loop` over a real base + Countdown/GSM (RLVR spine) |
| 12 | **F6** MoE balancing ablation | B | S | — | aux-loss-free vs seq-aux vs none × fine/coarse (measurement over shipped code) |
| 13 | **F2b** MTPDrafter | B | M | F2a,F3 | the trained MTP head as a lossless self-speculative drafter (train↔serve payoff) |
| 14 | **F7b** aha-curve detector | B | S | F7a | reward↑ + correct-length-growth + KL-bounded → the aha oracle |
| 15 | **F5** MLA-for-real | B | L | — | trainable+servable MLA + latent KV cache (**ship the first-slice first**, §D) |
| 16 | **F7c** neural-RM control | B | M | — | a hackable proxy that proves why R1-Zero refuses a learned RM (**CPU-proof only**, §D) |
| 17 | **F9** MuonClip guard | B | S | F1-run | QK-Clip gated OFF sub-1B + the qk-norm-bounds-logit test |
| 18 | **F8.1 / F8.2** DSA sparse attn | B | M/L | — | lightning indexer + top-k gather + KL warm-up (**PROMOTED to core — §10; scarce 2026 signal**) |
| 19 | **F10.1 / F10.2** hybrid linear attn 🆕 | B | M/L | (F5 seam) | Gated-DeltaNet block (chunkwise==recurrent==ref) → `attn_schedule` 3:1 iso-param quality/KV ablation — **the 2026 attn frontier; model-side twin of DELTA GDN-2** |
| 20 | **F11** agentic / tool-use RL 🆕 | B | L | — | multi-turn `ToolEnv`(VerifiableEnv) + masked multi-turn rollout + format-only reward-hack control — **the #1 stated 2026 lab priority** |
| — | **A7** distributed d20 pretrain | A | L | A2 | wire `utils/{ddp,fsdp,zero1}` into `train()` — 561M needs data-parallel on 8×H100 |
| — | **A8** d20 rental runbook + guardrails | A | M | A2,A7 | resume/off-box-sync + $/token cap + divergence kill-switch + repro manifest |
| — | **A9** public release surface | A | S | A6 | model card + reproducible weights/tokenizer/config/transcript bundle |

> **⚠ EV re-ranked 2026-07-09 (ABLATIONS §10).** The numeric order above is build-dependency order, not
> EV. **Scarce-2026 EV order = F8(+F10) · F7-reframed(+F11) · F1 · F2 · F4**, then F3/F5/F6/F9. Attention
> efficiency (F8 DSA + F10 linear-hybrid) and RL honesty/agentic (F7-reframed + F11) carry the
> differentiating hiring signal; F1/F2/F4 are table-stakes done well. F7's falsifier is **reframed** to a
> *random-reward debunk control* (length-growth is a GRPO-bias artifact, not the "aha"); F1's kill band is
> **recalibrated** to the tuned-baseline 1.1–1.4× (see the F1-run DoD note + `bench/RESULTS.md`).

**The d20 gate:** everything through A6 + the ablations you choose to *bake into the single d20
pretrain* (F2a MTP is the one worth baking; F1-run picks the optimizer) + A7/A8 must be green before
the $100 8×H100 run. F5/F6/F7/F8/F9/F10/F11 run at 30–300M on the **standing sm120 box** independently.

---

## §A — Track A: complete the loop into a chat model

### A1 · Real-corpus shards `[M]`
- **Interfaces:** NEW `data/shards.py` — `tokenize_to_shard(docs, tok, eot_id)`, `load_shard()`+`ShardMeta`, `download_fineweb_slice()` (network, CI-skipped); `data/__init__.py` additive export; `speedrun.py` shard-backed pretrain path.
- **Config:** `SpeedrunConfig.data_dir: str|None=None`, `shard_glob='*.bin'`.
- **DoD:** memmap round-trips `encode(d0)+[eot]+encode(d1)+[eot]`; file size == `2·n_tokens` (uint16); `get_batch` consumes the shard next-token-aligned. **Kill:** any id ≥ 2¹⁶ → switch to uint32.
- **Zone:** `data/` is main-track zone — additive new file only.
- **d20 extension (§D):** add a shuffled multi-shard sampler/streamer (~11B tokens); the base spec is nano-only.

### A0 · Decontamination gate `[M]` *(added by critic)*
- **Goal:** strip train documents that n-gram-overlap the eval sets (GSM8K, MMLU, Countdown, the report-card val) before A1 shards feed a scored run — else every B-track number is contaminated.
- **Interfaces:** NEW `data/decontaminate.py` — `ngram_overlap(doc, eval_ngrams, n=13)`, `decontaminate_shard(...)`; hook into A1's `tokenize_to_shard`.
- **DoD:** a planted eval string in a train doc is dropped/flagged; clean docs pass; overlap rate logged. **Kill:** >X% of a real slice flagged → the eval set leaked into pretrain data, investigate.
- *Needs a deep spec before build (outline only here).*

### A2 · Checkpoint chaining `[M]`
- **Interfaces:** `train.py` extend `save_checkpoint` to persist `asdict(model.cfg)` + `build_model_from_checkpoint()`; `tokenizer.py` `Tokenizer.save()`; `speedrun.py` refactor `run_speedrun` into chained stage fns.
- **Config:** `SpeedrunConfig.work_dir`, `midtrain_steps=0`, `sft_steps=0` (0 = stage skipped).
- **DoD:** save→rebuild every param `torch.equal`, `step` restored; resumed model's step-0 loss ≪ log V while a fresh model ≈ log V; config round-trips; tokenizer save/load round-trips a special-token string. **Kill:** reconstructed `ModelConfig` shape-mismatches the weights.
- **d20 extension (§D):** intra-run periodic checkpoint + `resume-at-step` + RNG/dataloader-position restore + off-box (rclone/HF-hub) sync; **stage-transition optimizer policy** — resuming across `adamw↔muon_adamw` is undefined; pin it (fresh optimizer state at each stage switch, LR re-warmup).

### A3 · Chat template + specials `[S]`
- **Interfaces:** NEW `chat.py` — `CHAT_SPECIAL_TOKENS`, `Message`, `render_conversation`, `render_for_completion`; resolve special ids via the public encode path; `speedrun.py` wires specials through `train_bpe`.
- **Config:** `SpeedrunConfig.chat: bool=False` (True → tokenizer trains with the specials).
- **DoD:** each special encodes to exactly 1 id; a turn round-trips `<\|user\|>…<\|eot\|><\|assistant\|>…<\|eot\|>` in order; mask covers **only** assistant tokens + its eot; completion prompt ends with the `<\|assistant\|>` id, no trailing eot. **Kill:** `len(encode(special))>1` (specials weren't threaded into both `train_bpe` and the constructor).
- **§D note:** if tool-use is trained (A4), add tool-boundary specials so tool structure is tokenizable.

### A4 · Midtrain stage `[M]` — deps A1,A3 *(deep spec completed 2026-07-12, replacing the truncated body)*
- **Interfaces:** NEW `data/chat_adapters.py` —
  `adapt_smoltalk(row) -> list[Message]` (accepts the SmolTalk `{role, content}` messages shape, network
  loader optional + CI-skipped); `adapt_mc(question, choices, gold_letter) -> list[Message]` (user =
  question + lettered options, assistant = **exactly the gold letter**); `adapt_tool_use(expr, result)
  -> list[Message]` (assistant carries expression + result inline; tool-boundary *specials* stay
  deferred to A3 §D/F11); `build_midtrain_shard(convos, tokenizer, out_path, context_length) ->
  tuple[ShardMeta, MidtrainStats]` — render each convo via A3 `render_conversation` and persist **ids
  only, NO mask**: midtrain is *continued pretraining* on chat-shaped data (full CE over every token —
  the model learns the specials' statistics); the assistant-only mask is A5's business, reconstructed
  from special positions at collate time (single source of truth = the token structure). Conversations
  whose rendered ids exceed `context_length` are **dropped whole + counted** (never truncate through a
  special); concatenate survivors into the A1 shard writer. `DEFAULT_MIDTRAIN_MIX`: a deterministic
  seeded synthetic mix (MC + tool-use; SmolTalk when the network loader is allowed).
  `speedrun.py` `stage_midtrain`: chains from the pretrain ckpt under the pinned A2 stage-transition
  policy (fresh optimizer + LR re-warmup); `midtrain_steps=0` ⇒ byte-identical skip.
- **Config:** `SpeedrunConfig.midtrain_steps=0`, `midtrain_mix: str|None`.
- **DoD (as tests):** MC assistant span decodes to exactly the gold letter; tool-use span contains
  expression + output; the shard round-trips next-token-aligned through `get_batch`; oversize convos
  dropped AND counted in `MidtrainStats`; the mix is deterministic under a seed; `midtrain_steps=0` is
  byte-identical. **Kill:** >30% of the mix exceeds `context_length` (can't include without cutting a
  special) — shrink the sources or raise ctx, don't truncate.

### A5 · SFT stage `[M]` — deps A2,A3
- **Interfaces:** NEW `algos/chat_sft.py` — `collate_chat_batch()`, `chat_sft_epoch()` (imports `algos/sft.py` primitives, no edits to it); `speedrun.py` `stage_sft`.
- **Config:** `SpeedrunConfig.sft_steps`, `sft_set: str|None`.
- **DoD:** response_mask True exactly at assistant positions + eot (hand-derived); masked loss-at-init ≈ log V; ≤200 steps overfit one chat batch to NLL<0.1; after overfit, greedy `generate(render_for_completion([user]), stop=eot)` decodes the trained assistant string. **Kill:** overfit NLL plateaus >0.1 (mask off-by-one after the shift).

### A6 · Chat REPL / serve `[M]` — deps A2,A3,A5
- **Interfaces:** NEW `chat_cli.py` — `ChatSession`, `repl()`+`__main__`, optional batched path via the public serving API; `speedrun.py` final `chat` preview → `SpeedrunResult.chat_reply`.
- **DoD:** `reply('hi')` returns str + grows history 0→2; SFT'd nano model reproduces the trained reply and **stops at eot** (len < max_tokens); returned text has no leaked specials; batched replies == per-turn greedy. **Kill:** never emits `<\|eot\|>` (turn terminator not learned or `stop_ids` unplumbed).
- **Zone:** consumes perf-owned `serving/` via public functions only.

### A7 · Distributed d20 pretrain `[L]` *(added by critic)* — deps A2
- **Goal:** `train()` is single-process; `utils/{ddp,fsdp,zero1}.py` exist but are never called from the loop. A 561M d20 on 8×H100 needs a data-parallel wrapper. Wire DDP/FSDP into `train()` behind a flag; validate gloo-equivalence on CPU + a 2-GPU smoke.
- *Needs a deep spec (outline only). Reuses the CS336-A2 distributed modules the perf front built.*

### A8 · d20 rental runbook + guardrails `[M]` *(added by critic)* — deps A2,A7
- **Goal:** `deploy/runbooks/d20_speedrun_8xH100.md` — provision → stage decontaminated shards → launch distributed pretrain → periodic off-box checkpoint sync → teardown. Plus: a **$/token projection** against the $100 cap, a **loss-divergence kill-switch** (the `train.py` NaN guard + a slope check), a **reproducibility manifest** (global seed, deterministic shard order, config + git-SHA + torch/CUDA version), and **recipe re-validation on H100/sm90** (the bf16+compile NaN is an sm120 dev-box fact — re-test the required bf16/compile recipe on the rental arch *before* the paid run).

### A9 · Public release surface `[S]` *(added by critic)* — deps A6
- **Goal:** a **model card** (config, data provenance + decontamination statement, eval-vs-baseline table, intended use, license) + a reproducible bundle (weights + tokenizer + config + a sample transcript). This is the portfolio artifact.

---

## §B — Track B: the frontier ablation study

### F1-run · Iso-FLOP Muon vs AdamW `[M]` — the pending headline
- **Interfaces:** NEW `eval/optimizer_race.py` (pure metric fns `tokens_to_match`/`token_saving_fraction` + the A/B driver); `train.py` additive val-eval hook + `TrainConfig.eval_every` (default byte-identical); `optim.py` `Muon(profile_ns=False)` NS wall-time instrument (additive, off); NEW `bench/optimizer_race.py` CLI → appends the row to `RESULTS.md`; NEW `tests/test_optimizer_race.py`.
- **DoD:** pure-metric unit tests; both arms hold `C=6ND` constant; the additive hook is a no-op on the default path (identical loss history). **Measured falsifier:** at N≈35M / D≈700M (C≈1.5e17) on sm120, same seed/LR/cosine, only the optimizer differs → Muon `tokens_to_match ≤ 0.85·D` (≥15% saving) or ≥0.02 nats lower at iso-FLOP; NS overhead <1%. **Kill:** saving <5% AND nats_lower <0.02, or divergence at the reused AdamW LR.
- **⚠ Recalibrated 2026-07-09** (`FRONTIER_2026_ABLATIONS.md` §10 + `bench/RESULTS.md` F1 pre-run note): Muon is **deflated + scale-dependent** (1.4×@0.1B→1.1×@1.2B vs a *tuned* AdamW; `2509.02046`) and superseded by MuonH (`2606.16899`). **New mandatory arm: an independently LR-tuned AdamW baseline** — the `tokens_to_match ≤ 0.85·D` bar is *only* meaningful against it (an untuned baseline is a fake win). Predict the **1.1–1.4× band**, not a fixed ≥15%. The tuned-baseline *methodology* is the hireable artifact.

### F2a · MTP training head `[L]`
- **Interfaces:** NEW `mtp.py` `class MTPHead` (`RMSNorm(h) ⊕ RMSNorm(emb_next) → eh_proj Linear(2d→d) → 1 TransformerBlock(moe=None) → final_norm`, shared `lm_head`); `model.py` `ModelConfig.mtp_depth=0`, build `mtp_head` in `__init__` (RNG last → base bit-identical), **extract `_trunk`** from `forward` (embed→blocks→final_norm, external behavior identical), add `forward_train(ids,targets)→(logits,aux,mtp_logits)`; `train.py` `TrainConfig.mtp_loss_weight=0.3` + `λ·L_MTP` (`cross_entropy(mtp_logits[:,:-1], targets[:,1:])`).
- **DoD:** MTP-head loss-at-init ≈ log V; `forward_train` main logits == `forward()`; `mtp_depth∈{0,1}` give identical base forward under shared seed; existing test_model/test_kv_cache/test_speculative stay green after the `_trunk` refactor. **Measured falsifier:** Δ = val_CE(λ=0.3) − val_CE(no-MTP, same seed) ∈ [−0.02, +0.01] nats. **Kill:** MTP aux degrades base val loss by >+0.01 nats.
- **Bake into the d20:** MTP is additive + neutral-to-better → the one ablation worth in the single d20 pretrain (it *is* the free draft head).

### F2b · MTPDrafter `[M]` — deps F2a, F3
- **Interfaces:** `mtp.py` `@dataclass MTPDrafter` satisfying the `Drafter` Protocol (`serving/speculative.py:30`) **structurally** (no edit to perf-owned serving); `tests/test_mtp.py` (F-front-owned, not `test_speculative.py`); optional F-front bench.
- **DoD:** `propose` returns ≤2 valid ids (main a1 + MTP a2); `speculative_generate(MTPDrafter)` is token-exact to greedy (float64); raises on a model without an MTP head. **Measured falsifier:** on the F3 trained model, MTPDrafter a2-acceptance on open prose ≥0.30 and strictly > `NGramDrafter(n=3)` (≈0). **Kill:** a2-acceptance ≤ n-gram's (the head is a useless drafter).
- **Zone:** structural Protocol satisfaction from `mtp.py` — the endorsed reuse pattern.

### F3 · De-confound serving `[S]` — deps A2
- **Interfaces:** NEW `eval/spec_acceptance.py` (domain-acceptance on top of the unchanged serving Drafter Protocol); NEW `bench/f3_deconfound_acceptance.py` CLI; NEW `tests/test_spec_acceptance.py`.
- **DoD:** harness well-formed; **mechanism test** — overfit a tiny model then structured-prompt acceptance > random-prompt acceptance; committed tokens == greedy `generate` (losslessness preserved). **Measured falsifier:** on a trained ckpt (val_bpb ≪ log2 V), n-gram(n=3,k=4) acceptance: prose <10% (predict 5–8%), code/JSON 40–60%. **Kill:** prose ≥ code/JSON (still confounded), or every domain >30% (checkpoint under-trained).

### F5 · MLA-for-real `[L]`
- **Interfaces:** NEW `mla_attn.py` — `LatentKVCache` (single-request latent cache, duck-types `KVCache`) + `MLASelfAttention` (decode-capable, cache-integrated); `model.py` `ModelConfig.attn: Literal['gqa','mla']='gqa'` + MLA dims + `to_mla_config()` bridge + `TransformerBlock` MLA path + `AnyKVCache` union + `new_kv_cache()`; `sampling.py` generate servable for MLA; NEW `tests/test_mla_model.py`; NEW `bench/mla_decode.py`.
- **Config:** `attn`, `mla_d_head`, `mla_d_latent` (default 4·d_head), `mla_d_rope` (default head_dim//2, even).
- **DoD:** `MLASelfAttention.forward` == `mla.py` `forward_absorbed` oracle (float64, after state_dict copy); loss-at-init ≈ log V; cached==recompute decode parity + token parity; latent KV bytes == `n_layers·(d_latent+d_rope)·T` (single latent, no ×2) and ≥3× < GQA-8. **Measured falsifier:** iso-param MLA within +0.02 val loss of GQA-8; decode ≥1.2× at 16k. **Kill:** Δ>+0.05 nats (wiring/absorption bug).
- **Zone:** `mla.py` is **perf-owned — do NOT edit it**; treat its math as the oracle, wire in `mla_attn.py`+`model.py`. Coordinate any mla.py change with the perf front.
- **Scope (§D):** ship the **first-slice** (`MLASelfAttention` no-cache + oracle-match test) as its own commit before the LatentKVCache + 16k bench.

### F6 · MoE balancing ablation `[S]`
- **Interfaces:** NEW `eval/moe_ablation.py` — `AblationArm`(BIAS_FREE/SEQ_AUX/NONE) + `Granularity`(coarse/fine) + `build_moe_config` + `evaluate_val_loss` + `router_diagnostics` + `train_arm` + `run_moe_ablation`; `eval/__init__.py` re-export; NEW `tests/test_moe_ablation.py`; NEW `scripts/f6_moe_ablation.py` CLI; pre-register in `RESULTS.md`.
- **Config:** NO new `MoEConfig`/`ModelConfig` fields — arms differ only via existing `bias_update_speed`/`aux_loss_alpha`/`expert_d_ff`/`n_routed_experts`/`n_experts_per_tok`. **Pinned design:** SEQ_AUX uses a **large** α∈{1e-3,1e-2} (not V3's tiny 1e-4 — the point is a strong aux trades off loss); `z_loss_coef=0` across all arms to isolate balancing.
- **DoD:** coarse/fine are iso-param + iso-active-FLOP; `build_moe_config` maps arms correctly; **discriminating test** — an induced router-bias preference is overcome by a fast balancer (entropy recovers >0.9·log N) while the `bias_update_speed=0` control stays collapsed. **Measured falsifier:** BIAS_FREE final val CE ≤ SEQ_AUX with entropy >0.9·log N_r. **Kill:** BIAS_FREE worse by >0.02 nats, or entropy <0.9·log N_r after training.
- **Zone:** pure `eval/` — reads the public MoE API only.

### F7a · GRPO aha harness `[M]`
- **Interfaces:** NEW `eval/aha.py` — `EnvFamily`+`AhaConfig`, `build_env(cfg, codec)`, `run_aha`+`AhaRun`, `from_speedrun`; `speedrun.py` expose `SpeedrunResult.model/.tokenizer`; `eval/__init__.py` export; additive `deploy/runbooks/A5_countdown_aha.md` $0 standing-box variant.
- **DoD:** fresh vocab-256 model + ByteTextCodec Countdown → `run_aha` 3 steps, all `GRPOStepMetrics` finite, `history[0].entropy ≈ log(256)` (loss-at-init oracle); `from_speedrun` composes; vocab-mismatch raises; jsonl logging. **Kill:** the base→env→loop bridge raises device/vocab errors or any metric non-finite.
- **Note:** a real "aha" needs ~0.5–1.5B (rental) — this is the box-runnable harness + the rental hook.

### F7b · Aha-curve detector `[S]` — deps F7a
- **Interfaces:** `eval/aha.py` add threshold constants + `AhaSummary`+`summarize_aha` + `aha_dashboard`+`_sparkline`; `eval/__init__.py` export.
- **DoD:** detects aha on a synthetic rise (reward↑ + correct-length-growth + KL-bounded); **rejects** the flat control (zero false positives) and unbounded-KL; dashboard renders all mandatory channels (reward/len/kl/entropy/is_ess). **Kill:** aha_detected=True on the flat negative control (thresholds miscalibrated).

### F7c · Neural-RM reward-hacking control `[M]`
- **Interfaces:** NEW `rewards/neural_rm_control.py` — `length_format_proxy_reward`, `make_length_proxy_reward_fn`, `RewardHackingResult`+`run_reward_hacking_ab`; `rewards/__init__.py` export.
- **DoD:** proxy is **answer-blind** (same reward for right/wrong same-length; r1_zero gives 1.0 vs 0.0) and **monotone in length**; malformed → all-zeros; conforms to the `RewardFn` shape. **Kill:** under the rule-based r1_zero grade, length *also* grows unbounded with answer_reward flat (then rules hack too — the point collapses).
- **Scope (§D):** ship the **CPU-green by-construction proof** (the two incentive tests); the full deep-copy A/B training harness is optional.

### F9 · MuonClip guard `[S]` — deps F1-run
- **Interfaces:** `model.py` additive per-head max-logit observer in `MultiHeadSelfAttention`, gated by `ModelConfig.track_attn_logits=False` (byte-identical off); `optim.py` `apply_qk_clip()` (post-step per-head W_q/W_k rescale by `min(1,τ/S_max)`); `train.py` wire after `optimizer.step()`, off by default.
- **Config:** `track_attn_logits`, `TrainConfig.qk_clip=False`, `qk_clip_tau=100.0`.
- **DoD:** qk_norm=True bounds max logit (<~30) and < qk_norm=False; `apply_qk_clip` rescales only over-τ heads (exact `√(τ/S_max)`, others byte-identical); no-op under τ; re-forward logit ≤ τ. **Measured falsifier:** with qk_norm=True on the F1 run (N≤1B), max per-head logit <30 at every step → QK-Clip γ==1 (confirms qk_norm suffices sub-1B). **Kill:** sustained S_max>30 (qk_norm insufficient earlier than claimed).
- **Zone:** do NOT edit the shared `scaled_dot_product_attention` — add the observer in `MultiHeadSelfAttention` only.

### F8.1 · DSA core mechanism `[M]` — stretch
- **Interfaces:** NEW `dsa.py` — `DSAConfig`, `LightningIndexer`, `dense_attention_target`, `indexer_kl_loss`, `select_topk_mask`+`topk_sparse_attention`, `attention_mass_recall`+`dsa_attention_flops`.
- **DoD:** top-k == dense at k≥L (the exact O(L²)→O(L·k) identity); KL-trained indexer recovers ≥0.95 held-out attention mass (random-init ~0.33 proves teeth); KL zero at match; recall monotone in k. **Kill:** held-out recall <0.95 after 300 KL steps, or top-k≠dense at k≥L.
- **Zone:** fully outside perf zone (new file, imports only model.py primitives).

### F8.2 · DSA wired + measured `[L]` — deps F8.1
- **Interfaces:** `model.py` `ModelConfig.dsa: DSAConfig|None=None` + indexer in `MultiHeadSelfAttention.__init__` + `dsa_active` flag + sparse forward branch + `dense_target_and_scores`; `dsa.py` `train_indexer`+`set_dsa_active`; pre-register `RESULTS.md`.
- **DoD:** dense path bit-identical when `dsa_active=False`; sparse==dense at top_k≥ctx; loss-at-init ≈ log V with DSA active; `train_indexer` KL drops <0.5× first. **Measured falsifier:** at 8k, sparse (top_k=2048) within +0.03 nats/token of dense + a FLOP crossover. **Kill:** gap >0.1 nats at 8k, or recall <0.95 at k=2048.

### F10 · Hybrid linear attention (Gated-DeltaNet / KDA-style) `[M/L]` 🆕 *(2026-07-09, ABLATIONS §10)*
> **Why (verified 2026).** The attention frontier moved *past* MLA to **hybrid linear attention**:
> Kimi Linear (Moonshot, Oct-25 — KDA extends Gated DeltaNet) **beats full attention** across
> short/long/RL with **75% KV cut, 6× decode @1M**; Qwen3-Next/Qwen3.5 interleave **Gated-DeltaNet :
> full-attention 3:1**. Attention is *contested, not settled* (MLA vs DSA vs linear-hybrid vs plain-GQA)
> ⇒ high ablation signal. **This is the model-side twin of the DELTA GDN-2 decode-kernel spike** — F10
> builds the trainable/servable torch reference + the quality/KV ablation; DELTA (perf/capstone zone)
> builds the fused low-precision decode kernel on top. Ship them as one story.
- **Interfaces:** NEW `linear_attn.py` — `LinearAttnConfig` (`d_head`, `n_heads`, `expand_v`, `chunk_size`), `GatedDeltaNet` block = the delta-rule recurrence `S_t = S_{t-1}(diag α_t − β_t k_t k_tᵀ) + β_t k_t v_tᵀ`, `y_t = S_t q_t` with a **chunkwise-parallel** forward (the training path) + a **recurrent-scan** forward (the decode path, single-step state update), a `linear_attn_state_bytes()` measurement, and a float64 reference loop the chunkwise path must match; `model.py` additive **per-layer attention schedule** `ModelConfig.attn_schedule: tuple[Literal['full','gqa','mla','gdn'],...] | None = None` (None ⇒ current uniform behavior, byte-identical) consumed in the block loop so a 3:1 `('gdn','gdn','gdn','full')·k` interleave is expressible; a `LinearAttnState` that duck-types the `KVCache`/`SlotKVCache` contract for the recurrent decode path.
- **Config:** `attn_schedule` (default None = uniform, base bit-identical), `linear_attn: LinearAttnConfig|None`, `expand_v` (default 2, the GDN value-expansion), `chunk_size` (default 64).
- **DoD:** chunkwise forward == recurrent-scan forward == float64 reference (rel ≤1e-5); loss-at-init ≈ log V with a GDN layer in the schedule; `attn_schedule=None` gives byte-identical base forward (RNG-neutral construction); recurrent decode step-by-step == chunkwise teacher-forced (token-exact greedy); **state bytes/token measured** vs GQA-8 KV/token. **Measured falsifier:** iso-param 3:1 `('gdn':full)` hybrid within **+0.03 val loss** of an all-full-attention baseline at ≤300M, with **state/token ≥2× smaller** than GQA-8 KV and a decode-throughput crossover as ctx grows. **Kill:** val gap >0.1 nats, OR no state/decode win (the linear block is dead weight at this scale).
- **Zone:** NEW `linear_attn.py` (F-front-owned) + **additive** `model.py` (`attn_schedule` + block-loop branch — RNG-last, base-neutral). **Do NOT** write the fused GDN decode *kernel* here — that is the perf/DELTA zone (`kernels/`); F10 ships the torch reference + the model-quality/KV ablation, and exposes the recurrent state contract the DELTA kernel accelerates. Coordinate the shared attention seam with the perf front (satisfy `KVCache`/`SlotKVCache` Protocols, don't edit `serving/`/`mla.py`).
- **Scope (ship first-slice first):** commit ① the `GatedDeltaNet` block + chunkwise==recurrent==reference test (no model wiring) as its own commit; ② the `attn_schedule` wiring + iso-param quality/KV ablation second. Deps: none (independent of F5/F8, but shares the `attn`-variant seam — land after F5's `attn` field exists to avoid a merge conflict, or introduce `attn_schedule` as the superset).
- **Interview.** "Why did 2026 labs move to linear-attention *hybrids* over pure MLA — what does the 3:1 ratio buy, why keep *any* full-attention layers, and what makes the *decode* kernel (state materialization at low precision) the hard part?"

### F11 · Agentic / tool-use RL `[L]` 🆕 *(2026-07-09, ABLATIONS §10 — the #1 stated 2026 lab priority)*
> **Why (verified 2026).** Agentic + long-horizon tool-use RL is the **#1 stated post-training
> priority** of DeepSeek (V3.2 agentic task synthesis), Moonshot (K2 joint real+synthetic-env RL, SOTA
> open agentic), and Qwen (Qwen3.5 SWE-bench 76.4 > GPT-5.2 75.4); 2026 eval shifted chatbot→agentic.
> A verifiable multi-turn tool env is **higher-signal than F6/F9** and turns the RL stack from a
> single-turn RLVR toy into the actual 2026 frontier shape.
- **Interfaces:** NEW `envs/tool_env.py` — a `ToolEnv` implementing the existing `VerifiableEnv` Protocol but **multi-turn**: `step(state, action) → (obs, reward, done)` where an action containing a tool call (e.g. `<tool>python: …</tool>`) is executed by a sandboxed evaluator (start with a pure-python calculator / arithmetic-expression evaluator — no network, no arbitrary exec) and the result is fed back as the next observation; a verifiable task family (multi-step arithmetic / unit-conversion whose gold answer is checkable). NEW multi-turn rollout in `algos/` (reuse `algos/grpo.py` primitives — a `multi_turn_rollout(policy, env, max_turns)` that concatenates turns into one trajectory for the GRPO/Dr.GRPO update, masking tool-result tokens out of the loss). NEW `rewards/tool_format_control.py` — a **format-only reward** (rewards well-formed tool-call *shape*, answer-blind) as the reward-hacking control.
- **Config:** `max_turns` (default 4), `tool_timeout`, `mask_tool_output: bool=True` (tool results are observations, not learned tokens).
- **DoD:** the sandboxed evaluator is **safe by construction** (no `eval`/network — an AST-restricted arithmetic evaluator, fuzz-tested to reject non-arithmetic input); a multi-turn trajectory round-trips (turns concatenated, tool-output tokens masked from CE); the mandatory RL logging is present + finite (entropy, KL(cur‖ref)/KL(cur‖old) **separately**, IS-ratio histogram, reward stats, **turns-used + length stats** — the `rl-run-auditor` gate); the format-only control **provably reward-hacks** (drives tool-call rate up with task success-rate flat). **Measured falsifier (rental/box-runnable at tiny scale):** on the synthetic tool env, verifiable-reward RL raises task success-rate with **turns-used bounded** (no tool-spam) while the **format-only control's success-rate stays flat** (proving the verifiable reward is load-bearing). **Kill:** success-rate flat under the true reward (env too hard / base too weak → shrink the task), OR the format-only control is *not* caught by the logging (monitoring is uninterpretable — the real failure).
- **Zone:** `envs/`, `algos/` (additive multi-turn rollout — reuse, don't edit, `grpo.py`'s update), `rewards/`, `utils/monitors.py` — all F-front / main-track owned. No perf-owned files.
- **Scope (§D-style):** ship the **CPU-green harness + the by-construction control proof** first (safe evaluator + multi-turn masking + the format-only-hacks incentive test); the real "does it learn tool-use" run is rental-gated (needs a ~0.5–1.5B base, like F7).
- **Interview.** "Why is agentic/tool-use RL the 2026 frontier over single-turn RLVR — how do you mask tool outputs from the loss, credit-assign across turns, and stop the agent from reward-hacking the tool-call *format* instead of solving the task?"

---

## §C — Whole-front definition of done (the d20 artifact)

1. **The loop closes on REAL data, reproducibly:** real BPE (chat specials trained in) → chained
   pretrain(MuonAdamW) → midtrain → SFT on **decontaminated** FineWeb-EDU shards → eval → chat.
2. **The $100 d20 (8×H100, ~561M) beats a PRE-REGISTERED report-card bar** (> nano **and** > a stated
   small-model CORE target — not just the random baseline) **and** holds a coherent multi-turn chat.
3. **The rental is safe + bounded:** preemption-resumable with periodic off-box-synced checkpoints, a
   documented token budget + $/token projection under $100, a loss-divergence kill-switch, a
   provisioning→teardown runbook, and the bf16/compile recipe re-validated on the H100 arch.
4. **Every B-track ablation is pre-registered** (ABLATIONS.md + RESULTS.md) and returns
   confirm/kill/inconclusive with the measured number filled against prediction **on a real (F3) base**.
5. **Green-CI floor holds** — ruff + pyright + `pytest -m "not gpu"`, including every new test file.
6. **Measurement discipline** — every GPU claim is predicted-before-run; no speedup/quality number
   ships off an untrained model.
7. **Zone discipline held** — no perf-owned files edited (mla.py, serving/, kernels/, quant/,
   utils/{tp_mlp,pipeline_schedule,ep_moe,mfu}.py); RESULTS.md + node pointer additive.
8. **Public artifact ships a model card + reproducible bundle** (weights + tokenizer + config +
   transcript + eval-vs-baseline table + license).

## §D — Scope corrections (from the critic) — read before building

- **F5 over-scoped `[L]`:** commit the first-slice (`MLASelfAttention` no-cache + oracle-match) alone
  first; defer `LatentKVCache` + 16k decode bench to a second commit.
- **F7c over-scoped `[M]`:** the rule-based-reward point is proven **CPU-green by construction**
  (answer-blind + length-monotone); ship that; the full A/B training harness is optional.
- **A1 under-scoped for the d20:** it feeds nano but defers the multi-shard shuffle/stream needed for
  ~11B tokens — add the shuffled sampler as a d20 extension.
- **A2 under-scoped for the rental:** stage-boundary save only — add intra-run periodic checkpoint,
  resume-at-step, RNG/dataloader-position restore, off-box sync, and the **adamw↔muon_adamw
  stage-transition optimizer policy** (undefined today).
- **A3 tool specials under-scoped:** A4 renders tool spans inline with no tool-boundary tokens — add
  them if tool-use is to be trainable-as-tokens.
- ⚠️ **A4 spec is TRUNCATED** — its `build_midtrain_shard` body/DoD/falsifier are unspecified; finish
  the spec before building A4.

## §E — First actions

1. **A1 + A2** (the foundation + the rental safety-net) — start here; both `[M]`, no deps.
2. **F1-run** — the *pending headline* Muon-vs-AdamW number; `[M]`, needs A2, runs on the standing box.
3. **A3 + F2a** in parallel (chat surface + the MTP head to bake into the d20 pretrain).

Every rung: pre-register its falsifier in `bench/RESULTS.md` §Frontier ablations **before** running,
build test-first to green-CI, then measure. Node pointer lives in
[`FRONTIER_2026_ABLATIONS.md`](FRONTIER_2026_ABLATIONS.md) §3.
