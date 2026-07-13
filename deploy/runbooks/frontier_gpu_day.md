# Frontier GPU Day — one session, pure execution

**What this is.** Every GPU-executable task of the close-the-loop / frontier front, sequenced so a
single GPU session is *pure execution* — no code is written on the box. All the code, tests, drivers
and CLIs referenced here are already shipped + green on `main` (2026-07-13). This runbook is the
"press play" list: each step has the exact command, the pre-registered predicted number, the
pass/kill gate, and a rough wall-clock.

**Where it runs.** The standing **sm120 box** (RTX PRO 4000 Blackwell, 25 GB) runs *all* of this —
F1 at depth 8 (~50M) fits in 25 GB with room to spare, so **no rental is needed** for the frontier
GPU day. (The separate $100 **d20** run — 561M on 8×H100 — and the perf datacenter days are their
own rentals under [ADR-0012]; see `deploy/runbooks/A2_multigpu_nccl_bench.md` and the A5 runbooks.)

**Golden rule (sm120).** Run bf16 **eager**. `bf16 + torch.compile` NaNs on this box's
torch-2.12 inductor (`bench/RESULTS.md` F4 — reproduces with plain AdamW, not our logic). `--compile`
is the H100-tier path only.

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

## Step 1 — build decontaminated real shards (A1 + A0) (5–15 min, network)

```bash
python -m scratch_llm.data.shards \
  --out data/fineweb_edu --vocab-size 32768 --n-docs 6000 --decontaminate
```
- `--decontaminate` = the A0 13-gram gate (strips train docs overlapping GSM/Countdown/report-card
  eval before BPE training). Add `--eval-file gsm8k_test.txt --eval-file mmlu.txt` to guard the real
  test splits (one item/line) — **do this if you will score MMLU/GSM8K**, else they leak.
- **Kill:** the "A0 decontamination: kept X/Y" line shows >20% dropped ⇒ the eval set leaked into the
  corpus slice — investigate before training.
- Produces `data/fineweb_edu/{tokenizer.json, shard_*.bin, *.meta.json}`. For the d20's ~11B tokens
  you need the multi-shard streamer (§D extension), not this endpoint — this slice is the F1-scale run.

---

## Step 2 — F1 iso-FLOP Muon vs tuned-AdamW (+ F9 ride-along) (5–7 h)

The pending headline. The sweep→race→ledger driver was CPU-smoke-verified end-to-end (2026-07-13).

```bash
python bench/optimizer_race.py \
  --data-dir data/fineweb_edu --depth 8 --tokens 7e8 \
  --batch-size 32 --context-length 1024 --amp bf16 --device cuda \
  --sweep-lrs 1e-4,2e-4,3e-4,6e-4,1e-3 --sweep-frac 0.2 \
  --out results/f1_race.json
```
- **Mandatory tuned baseline:** the driver runs the 5-point AdamW LR sweep at 20% horizon FIRST, picks
  argmin val CE, and only then races Muon at that LR (the 2509.02046 lesson — an untuned baseline is a
  fake win). The sweep curves ship in `f1_race.json` for audit.
- **Pre-registered predict (recalibrated 2026-07-09, `bench/RESULTS.md` F1):** Muon
  `token_saving_fraction` in the **1.1–1.4× band** (≈15–25% at ~50M, shrinking with N) vs the *tuned*
  AdamW, or ≥0.02 nats lower at iso-FLOP. NS overhead <1%.
- **KILL:** saving <5% vs the tuned baseline, OR divergence at the reused LR, OR NS overhead >3%.
- **F9 rides along for free:** `--track-logits` is ON by default, so the row prints the max per-head
  attention logit. **F9 predict:** with qk_norm, S_max < 30 at every step ⇒ QK-Clip γ≡1 (qk_norm
  suffices sub-1B). **F9 KILL:** sustained S_max > 30.
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
| 1 shards | 5–15 min | network-bound |
| 2 F1 race (+F9) | **5–7 h** | 5 sweep runs @ ~20 min + 2 full arms @ ~2 h |
| 3 trained ckpt | 1–2 h | |
| 4 F3 | ~10 min | |
| 5 FA2 bwd | ~15 min | |
| **total** | **~8–10 h** | one standing-box day; **$0 rental** at this scale |

**Not in this runbook (separate rentals, own runbooks):** the $100 d20 (561M, 8×H100 — needs A7
distributed wiring first, still CPU-buildable), and the perf datacenter days (H100/B200/8×H200,
[ADR-0012]). Those are gated on their own go/no-go.
