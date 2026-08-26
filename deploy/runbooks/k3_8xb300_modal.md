# K9 runbook — serve the real Kimi K3 on 8×B300 (Modal + vLLM), measured by us

> ⚠️ **ERRATA-G binding (2026-08-26).** Pre-audit planning generation — NOT current law. The sealed
> measure-first corpus governs (Desktop `plan/` + memory tracker; precedence G > F > E > D > C > B >
> body). K3-from-scratch = post-gate season W9–W16. Full binding: ERRATA-G blocks in
> `docs/k3/ROADMAP.md` and `docs/k3/MERGED_KERNELS_K3_ROADMAP.md`.

> Purpose: convert the hosting book's unverifiable claims (0.93 s TTFT · 92.1 tok/s ·
> $190.13/M tokens · 27-min cold boot · "fp8 KV trap") into **measured-by-us** ledger entries
> (`bench/RESULTS.md §K3` + `docs/k3/FACTS.md` S5/S6). Discipline: ADR-0012 — this runbook is
> written before any rental; budget cap is set before launch; teardown is a checklist, not a
> hope. Cross-check against hosting-book capsules 11–18 when they open (2026-08-03).
>
> **Budget cap: 3 GPU-hours ≈ $170.** Node burns **$56.79/h** (8 × B300 @ $0.001972/s/GPU,
> modal.com/pricing, FACTS S4) = **$1,363/day if forgotten**. Scale-to-zero is the whole game.

## Phase 0 — pre-flight (free, CPU, no GPU rented)

- [ ] **Tensor-level closure** (completes FACTS A18): script fetches each of the 96 shard
  headers via HTTP range reads (`model-0000{1..96}-of-000096.safetensors`, first ~10 MB each),
  parses the safetensors JSON header, sums params per tensor-name pattern, diffs against
  `k3/param_count.py` output. Target: residual 2,208 → 0, or a named explanation per tensor.
- [ ] **Download plan**: 1,561 TB total. Check Xet behavior on shard 1 first (FACTS S10 —
  Unsloth docs warn about stalls; have `HF_HUB_DISABLE_XET=1` fallback ready). Modal volume
  ≥ 2 TB. Verify `model.safetensors.index.json` + `config.json` checksums before paying for
  GPU time to read corrupt weights.
- [ ] **Tokenizer probe**: load the real tiktoken 160K tokenizer; round-trip a long document;
  confirm bos 163584 / eos 163586 / pad 163839 (FACTS A4).
- [ ] **Image + flags pinned**: vLLM day-0 K3 support ships **Docker images only** (branch
  `kimi-k3`, pre-release FlashInfer — FACTS S1). Pin the exact digest in this runbook before
  launch. Baseline flags (verified from the vLLM day-0 blog):
  ```
  --tensor-parallel-size 8 --trust-remote-code --load-format fastsafetensors \
  --enable-prefix-caching --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 --reasoning-parser kimi_k3
  ```
  benchmark recipe adds: `--kv-cache-dtype fp8 --max-model-len auto`
  (prefix caching is OFF by default for K3 — the `--enable-prefix-caching` is deliberate;
  known bug vllm#50235: miss at 1536-token boundary).

## Phase 1 — launch & cold boot (clock starts)

- [ ] Modal 8×B300 (2,304 GB HBM — weights 1,561 GB + KV headroom; min practical ~1,680 GB).
- [ ] **Measure cold boot**: volume-read start → server ready. Book claims 27 min — verify.
- [ ] Watch for: FP8 KV "no scaling factor, falling back to 1.0" log line (GPUStack saw it on
  both engines — FACTS S6). Record whether it appears.

## Phase 2 — measurements (the ledger entries)

Single-stream first (book comparability), then a small concurrency sweep. Log every number
with config + method in `bench/RESULTS.md §K3`.

- [ ] **TTFT** (short prompt, warm): book 0.93 s · vLLM-team bs1 context: 111 tok/s TP8 decode.
- [ ] **Steady-state decode tok/s, bs=1**: book 92.1 · vLLM 111 (GB300 NVL72) · SGLang ~113.
- [ ] **Concurrency sweep** 1→8→32: throughput curve + where TTFT degrades.
- [ ] **fp8-KV A/B** (the "trap" question, FACTS S6): same eval prompts with and without
  `--kv-cache-dtype fp8`; spot-check quality (e.g., 20 long-context QA pairs) + tok/s delta.
  The official recipe USES fp8 KV — measure, don't assume.
- [ ] **Long-context spot check** at 128K+ (1M if budget allows): KDA state vs KV memory
  behavior, TPOT vs context length (Kimi Linear claim: ~6× vs full MLA at 1M).
- [ ] Optional (if ≥1 h budget remains): DSpark speculative config — expected ~3× decode
  (vLLM measured 331–370 tok/s; book capsule 30 labels it "reported").
- [ ] **Cost derivation**: $/M output tokens from OUR measured steady-state tok/s at
  $56.79/h (book: $190.13/M; arithmetic says $171/M @ 92 tok/s single-stream — FACTS S5).

## Phase 3 — teardown (mandatory)

- [ ] Terminate the Modal app; verify `$0` running (console + `modal app list`).
- [ ] Snapshot volume only if a second session is pre-approved; otherwise delete (re-download
  is cheaper than idle storage at this size).
- [ ] Ledger entries written same day: three-way table — book's numbers vs vLLM/SGLang
  reported vs **ours** — in FACTS.md S5/S6 and `bench/RESULTS.md §K3`.

## Failure playbook (the book's "five failures" cross-check, capsules 11–18)

- OOM at load → check `gpu_memory_utilization` + that the volume mount is the fast path.
- Slow cold boot → measure read throughput from volume; `fastsafetensors` flag present?
- fp8 KV quality regression → fall back to bf16 KV, record the delta (that IS the data).
- Tool-call parse errors → confirm `--tool-call-parser kimi_k3` + `--reasoning-parser kimi_k3`.
- Anything else: capture the exact error text, add to this playbook + FACTS.md.
