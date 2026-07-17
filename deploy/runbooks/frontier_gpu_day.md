# Frontier GPU Day — one session, pure execution

**What this is.** Every GPU-executable task of the close-the-loop / frontier front, sequenced so a
single GPU session is *pure execution* — no code is written on the box. All the code, tests, drivers
and CLIs referenced here are shipped + green (2026-07-13, re-audited + fixed 2026-07-16 — the
39-agent GPU-readiness review, 9 confirmed findings; table in `bench/RESULTS.md` §*Pre-run
amendment — F1 recalibration + GPU-readiness review*). This runbook is the "press play" list: each
step has the exact command, the pre-registered predicted number, the pass/kill gate, and a rough
wall-clock.

**Where it runs.** The standing **sm120 box** (RTX PRO 4000 Blackwell, 25 GB) runs *all* of this —
**with the driver defaults as now shipped (`--attention sdpa`)**. Honesty correction (2026-07-16
review): the OLD eager attention path measured **~30 GB peak** at the F1 config (depth 8 /
vocab 32768 / batch 32 / ctx 1024 — saved-tensors accounting: each layer retains fp32
`(B, H, S, S)` scores for backward) — it did **NOT** fit; this runbook's previous "fits in 25 GB
with room to spare" was false. The now-default fused-SDPA path never materializes the S×S scores;
predicted peak is well under 25 GB `[INFERENCE — watch nvidia-smi during the first sweep arm]`. So
**no rental is needed** for the frontier GPU day, but only on the SDPA default — never pass
`--attention eager` at this config. (The separate $100 **d20** run — 480.4M measured at vocab 32768,
on 8×H100 — and the perf datacenter days are their own rentals under [ADR-0012]; see
`deploy/runbooks/A2_multigpu_nccl_bench.md` and the A5 runbooks.)

**Golden rule (sm120).** Run bf16 **eager**. `bf16 + torch.compile` NaNs on this box's
torch-2.12 inductor (`bench/RESULTS.md` F4 — reproduces with plain AdamW, not our logic). `--compile`
is the H100-tier path only. (`--attention sdpa` is orthogonal: a fused kernel *inside* eager
execution, not `torch.compile` — the golden rule stands unchanged.)

---

## Step 0 — env sanity (2 min)

```bash
cd ~/scratch_llm && source .venv/bin/activate     # or scripts/bootstrap-pod.sh on a fresh pod
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
pytest -m gpu -q                                   # the whole GPU suite must be green BEFORE measuring
```
**Gate:** `torch.cuda.is_available()` True, sm120 named, `pytest -m gpu` green. If the GPU suite is
red, stop — a broken kernel invalidates every number below.

---

## Step 1 — build the F1-scale shard corpus, decontaminated (A1 + A0) (CPU/network — run FIRST, or the night before)

**Why the old command is gone (2026-07-16 review):** the rows-API build this step used to
pre-commit (`--n-docs 6000`) measured **≈4.8M tokens** — the 7e8-token race budget over that corpus
is **~146 epochs**: a memorization benchmark, not an optimizer signal (voided science; the driver's
new epochs guard now ABORTS exactly this). F1 needs **≥ 7e8 tokens** — the parquet bulk path
(`--fineweb-parquet`; needs the pyarrow extra: `uv pip install -e ".[data]"`).

**Dry-run the download plan first** (prints file list + GiB + estimated tokens; downloads nothing):

```bash
python -m scratch_llm.data.shards \
  --out data/fineweb_edu --fineweb-parquet --target-tokens 7e8 --dry-run
```

- **Gate:** the plan's `est_tokens` must cover the 7e8 target (expect ~2.4 GiB of parquet at the
  2.95 B/token estimate × 1.25 safety). An under-supplied plan fails loudly at the accounting stage
  anyway — catch it here, before any bytes move.

Then the real build:

```bash
python -m scratch_llm.data.shards \
  --out data/fineweb_edu --vocab-size 32768 \
  --fineweb-parquet --target-tokens 7e8 --num-workers 8 --decontaminate
```

- Streams hub parquet → BPE (trained on a 16 MiB capped sample; `--max-train-bytes` to raise) →
  ~42 × 16.8M-token uint16 shards + `tokenizer.json` + sidecars, bounded RAM. Downloads are
  resumable at file granularity (`.part` rename + size-checked cache) — a killed run re-uses what
  it fetched.
- `--decontaminate` = the A0 13-gram gate (strips train docs overlapping GSM/Countdown/report-card
  eval before BPE training). Add `--eval-file gsm8k_test.txt --eval-file mmlu.txt` to guard the real
  test splits (one item/line) — **do this if you will score MMLU/GSM8K**, else they leak.
- **Kill:** the final accounting line (`… docs, … filtered → data/fineweb_edu`) shows >20% of docs
  filtered ⇒ the eval set leaked into the corpus slice — investigate before training.
- **Wall `[INFERENCE]`:** download minutes; BPE training tens of minutes at vocab 32768; the
  pure-Python encode of ~2.7 GB text dominates — expect **hours** even at `--num-workers 8`. It
  needs **no GPU**: run it before the GPU day (or overnight) so Step 2 starts on staged shards.
- For the d20's ~10B tokens you still need the multi-shard *shuffled* streamer (§D extension / d20
  gate P2), not this endpoint — this build is the F1-scale run.

---

## Step 2 — F1 iso-FLOP Muon vs tuned-AdamW (+ F9 ride-along) (~4–10 h `[INFERENCE]`)

The pending headline. The sweep→race→ledger driver was CPU-smoke-verified end-to-end (2026-07-13;
re-verified after the 2026-07-16 review fixes).

```bash
python bench/optimizer_race.py \
  --data-dir data/fineweb_edu --depth 8 --tokens 7e8 \
  --batch-size 32 --context-length 1024 --amp bf16 --device cuda \
  --sweep-lrs 1e-4,2e-4,3e-4,6e-4,1e-3 --sweep-frac 0.2 \
  --out results/f1_race.json
```
- **Defaults carry the registered regime (2026-07-16) — state, don't assume:** the driver builds the
  model with **qk_norm ON** (the F9-registered regime; the opt-out is `--no-qk-norm` — do NOT pass
  it, the pre-registered falsifier "S_max < 30 *under qk_norm*" is conditioned on it) and
  **`--attention sdpa`** (the fused training path — the F1-OOM fix; `eager` is the reference path
  that measured ~30 GB and does not fit this box). Both are ledgered per-run in the JSON row
  (`qk_norm`, `attention`).
- **No `mkdir results/` needed:** the driver creates `--out`'s parent at startup — an unwritable
  path fails in second 1, not after the arms.
- **Epochs guard:** the driver prints `epochs = D / corpus` and ABORTS above `--max-epochs`
  (default 1.5). On the Step-1 corpus (≥7e8 tokens) this reads ≈1.0 and passes; if it aborts, the
  corpus is undersized — rebuild Step 1, don't override the guard.
- **Mandatory tuned baseline:** the driver runs the 5-point AdamW LR sweep at 20% horizon FIRST, picks
  argmin val CE, and only then races Muon at that LR (the 2509.02046 lesson — an untuned baseline is a
  fake win). The sweep curves ship in `f1_race.json` for audit. Resume past a finished sweep with
  `--baseline-lr <winner>`.
- **Recalibrated scale (2026-07-16, pre-run):** measured **N = 59,253,248** at depth 8 / vocab 32768
  (the registered "N≈35M" assumed a smaller vocab — the tied 32768×512 embed/head alone is 16.8M),
  so C = 6ND ≈ **2.49e17** at D = 7e8 (realized D:N ≈ 11.8). Verdict unaffected — both arms share
  identical N, D, seed by construction.
- **Pre-registered predict (recalibrated 2026-07-09, `bench/RESULTS.md` F1):** Muon
  `token_saving_fraction` in the **1.1–1.4× band** (≈15–25% at this N, shrinking with N) vs the
  *tuned* AdamW, or ≥0.02 nats lower at iso-FLOP. NS overhead <1%.
- **KILL:** saving <5% vs the tuned baseline, OR divergence at the reused LR, OR NS overhead >3% —
  all three branches are wired into the printed verdict (`verdict_string`), and a KILL is a result.
- **Crash/divergence containment (2026-07-16):** every completed stage — sweep, baseline arm,
  challenger arm, final metrics — is atomically persisted to `--out` the moment it exists (`status`
  in the JSON names the newest stage on disk), so a crash at hour 6 keeps everything finished; a
  diverged Muon arm returns a **recorded KILL row** (persisted + printed), not a lost day. Note
  `muon_wall_s` is NS-profiler-perturbed (`muon_wall_perturbed_by_profiler: true`) — never compare
  it to `baseline_wall_s` directly.
- **F9 rides along for free:** the max-attn-logit observer is ON by default (opt-out
  `--no-track-logits`; it observes training forwards only), so the row prints the max per-head
  attention logit. **F9 predict:** with qk_norm, S_max < 30 at every step ⇒ QK-Clip γ≡1 (qk_norm
  suffices sub-1B). **F9 KILL:** sustained S_max > 30.
- **Wall `[INFERENCE]` (pending the run):** total = 3 × 7e8 tokens (5 sweep runs @ 20% + 2 full
  arms) ≈ 64k steps @ 32,768 tok/step. The review's honest recompute of the OLD eager path was
  **5.5–19 h** (this runbook's previous "5–7 h" had no basis); SDPA cuts the attention-dominated
  step — expect **~4–10 h**. Calibrate early: one sweep arm is 1/15 of the total, so measure its
  wall and multiply by 15 before committing the day.
- The CLI prints two paste-ready ledger rows (F1 iso-FLOP + F9 ride-along). Copy them into
  `bench/RESULTS.md` §Frontier ablations.

---

## Step 3 — the trained checkpoint the ablations ride (1–2 h)

F1 decides the *optimizer + LR*; `speedrun` produces the config-carrying **artifact** F3/A5/A6/release
consume (single-responsibility — F1 doesn't emit a checkpoint by design). Use the F1-winning LR:

```bash
python -m scratch_llm.speedrun \
  --data-dir data/fineweb_edu --depth 8 --steps 20000 \
  --optimizer muon_adamw --lr <F1_WINNING_LR> --bf16 --device cuda \
  --work-dir runs/d8_base
```
- **Gate:** `val_bpb` in the report card ≪ log2(vocab); the sample is coherent. Writes
  `runs/d8_base/pretrain.pt` (config-carrying, resumable — the rental safety-net).
- Optional chat model: add `--chat` (once the tokenizer is rebuilt with specials) then a second run
  with `--sft-steps N` to get an A6-talkable checkpoint.

---

## Step 4 — F3 de-confound speculative acceptance (10 min)

On the *trained* checkpoint (random-weight models falsified prompt-dependence — the whole point of F3):

```bash
python bench/f3_deconfound_acceptance.py \
  --ckpt runs/d8_base/pretrain.pt --tokenizer-dir data/fineweb_edu \
  --ngram-n 3 --k 4 --device cuda --out results/f3_acceptance.json
```
- **Predict:** n-gram(n=3,k=4) acceptance — prose <10% (≈5–8%), code/JSON 40–60%. Losslessness
  (committed == greedy) must hold or the row is void (harness bug).
- **KILL:** prose ≥ code/JSON (still confounded), OR every domain >30% (checkpoint under-trained).

---

## Step 5 — FA2 backward: grad-parity + profile (15 min)

The Triton FA2 backward shipped implemented-not-measured. Correctness first, then the profile
(FOP-3: a kernel's DoD is a profile, not a green test).

```bash
pytest -m gpu tests/test_flash_attention_triton.py tests/test_model.py -q   # grad parity vs SDPA autograd
python bench/flash_bwd_roofline.py --seqs 512 1024 2048 4096 8192          # the profile
```
- **Predict (fwd+bwd):** Triton reaches ≥45% of SDPA-flash at seq 4096, d=64, causal, bf16 (≈50%,
  below the forward's ~55–65% because our bwd accumulates dQ via atomics — the named cost).
- **KILL:** <35% of SDPA at 4096, or the correctness gate fails (grads ≠ SDPA). If slow, the
  bottleneck to name is atomic contention on dQ (the next optimization: split-K dQ or a dedicated
  dQ kernel).
- Paste the roofline row into `bench/RESULTS.md` (perf A4 section).

---

## Step 6 — record + advance (10 min)

1. Flip the four pending F1/F9 rows in `bench/RESULTS.md` from "pending" to the measured numbers
   (predicted-vs-measured, verdict PASS/KILL — honest either way; a KILL is a result).
2. Append the F3 + FA2-bwd rows.
3. Advance the `Next-node:` marker in `docs/FRONTIER_2026_TASKSPEC.md` and the STATUS/RESULTS lines.
4. Update the auto-memory `f1-gpu-day-protocol.md` (mark the GPU day done; next = the d20 rental or
   CPU batch-2 continuation).
5. `git add -p` the ledger/docs (never `-A` — shared checkout), commit, push; confirm remote CI green.

---

## Time + cost

| Step | Wall | Notes |
|---|---|---|
| 0 env + gpu suite | ~5 min | |
| 1 shards (parquet, ≥7e8 tok) | **hours `[INFERENCE]`** | CPU/network only — run before the GPU day (or overnight); pure-Python encode dominates |
| 2 F1 race (+F9) | **~4–10 h `[INFERENCE]`** | SDPA default; old eager recompute was 5.5–19 h; calibrate = 15 × the first sweep arm |
| 3 trained ckpt | 1–2 h | |
| 4 F3 | ~10 min | |
| 5 FA2 bwd | ~15 min | |
| **total (GPU day proper)** | **~6–13 h `[INFERENCE]`** | one standing-box day with shards staged in advance; **$0 rental** at this scale |

**Not in this runbook (separate rentals, own runbooks):** the $100 d20 (480.4M measured at vocab
32768, 8×H100 — needs A7 distributed wiring first, still CPU-buildable), and the perf datacenter
days (H100/B200/8×H200, [ADR-0012]). Those are gated on their own go/no-go.
