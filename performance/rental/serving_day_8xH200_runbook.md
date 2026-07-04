# Serving Day — 8×H200 NVLink node — EXECUTION RUNBOOK

> **What this is.** The copy-paste script a person runs on rental day. It operationalizes
> `performance/PERF_PLAN.md` §"Phase 4 — THE SERVING DAY" and `docs/adr/ADR-0012-inference-rental-tiers.md`.
> Read those two first for the *why*; this file is the *how* — exact commands, in order, with the
> gates and kill lines inline. Model: **DeepSeek-R1 671B FP8** on one **8×H200 SXM NVLink** node.
> Engine: **SGLang** (chosen for R1-FP8 — it publishes a single-node R1 config and the MTP/EP/PD
> knobs we measure; vLLM equivalents are noted inline as fallback).
>
> **Claim-label legend** (per FOP-4, used on every predicted number below):
> `[FACT: derivation]` = arithmetic from published specs/configs · `[INFERENCE]` = a pre-registered
> hypothesis to be measured · `[UNCERTAIN: market]` = a price that moves, re-check at session time.
> All math is plain text (no LaTeX).
>
> **The one rule that governs the whole day:** the node is released with **numbers in
> `bench/RESULTS.md`**, or the day is wasted (FOP-1/FOP-4). Log as you go, not at the end.

---

## 0. Session constants (fill these in first, then everything below just works)

Paste this block into the node shell **once** at login; every later command references these.

```bash
# ---- pinned engine image (spec §2 rule 6: pin the image, smoke the command BEFORE the node) ----
export SGLANG_IMAGE="lmsysorg/sglang:v0.4.6.post1-cu124"   # [UNCERTAIN: pin the exact tag you smoke-tested; engine flags change fast]
export MODEL="deepseek-ai/DeepSeek-R1"                     # native FP8 (fp8_e4m3) weights, ~671 GB
export FALLBACK_MODEL="Qwen/Qwen3-235B-A22B-FP8"           # kill-criteria swap (fits 8×H100 too)

# ---- paths (WORKSPACE is the volume if this instance has one — verify below) ----
export WS="${WORKSPACE:-/workspace}"
export HF_HOME="${WS}/.hf_home"                            # 671 GB lands here
export PORT=30000                                          # SGLang default; bench client points here
export HOST=127.0.0.1                                      # drive load ON the node → localhost, no Caddy/token
export RESULTS="${WS}/scratch_llm/bench/RESULTS.md"        # log here BEFORE release

# ---- budget alarm (spec: hard alarm at 12 node-hrs) ----
export T0=$(date +%s); echo "T0=$T0 ($(date))"            # session start; every block prints elapsed
```

Elapsed-time one-liner (run any time; the day is planned to 8.5 hrs, alarm at 12):

```bash
awk -v t0="$T0" 'BEGIN{d=systime()-t0; printf "elapsed %.2f node-hrs (alarm @12)\n", d/3600}'
```

Hard budget alarm — fire a loud marker at 12 node-hrs so a hung bring-up cannot silently burn money:

```bash
nohup bash -c 'sleep 43200; echo "!!! 12 NODE-HRS — KILL OR JUSTIFY !!!" | tee '"${WS}"'/BUDGET_ALARM' >/dev/null 2>&1 &
```

---

## 1. PRE-RENTAL CHECKLIST (do all of this BEFORE spending a cent)

Everything here happens on Tier 0 (standing sm_120) or a cheap Tier-1 box — **never on node time.**

```
□ ENGINE IMAGE PINNED. $SGLANG_IMAGE tag is fixed (not :latest). Reason: SGLang/vLLM TP/EP/disagg
  flags change release-to-release (spec §2 rule 6). Pin the tag whose --help you read below.

□ EXACT LAUNCH COMMAND SMOKE-TESTED ON A SMALL MODEL FIRST. On any single GPU (Tier 0 sm_120 works),
  run the §3.3 launch line with MODEL=Qwen/Qwen2.5-0.5B-Instruct and --tp 1, confirm:
    (a) server reaches "The server is fired up and ready to roll!" / logs "max_total_num_tokens";
    (b) a /generate curl returns tokens;
    (c) the §4 bench client connects and prints a report.
  This proves the command PARSES and the client WIRES UP — the two things you must not debug on the
  node. Then re-read `python -m sglang.launch_server --help` for THIS image and confirm every flag
  in §3.3 / §6 / §7 / §8 still exists (rename-proofing).

□ nccl-tests BUILD REHEARSED (§2.2 has the build; rehearse it compiles). If the image lacks MPI,
  build with MPI=0 (single-process, 8 GPUs via -g 8 — sufficient for our busbw sweep).

□ `nvidia-smi topo -m` PARSE REHEARSED. Accept rule: EVERY GPU-pair cell reads NV8 or NV18
  (NVLink). REJECT if any pair reads PIX / PHB / NODE / SYS (that is PCIe, not NVLink — TP/EP will
  be comms-bound and the day's physics is invalid). See §2.1.

□ DISK + BANDWIDTH GATE. Listing must advertise ≥1.5 TB disk and ≥5 Gbps down. The R1 pull is
  671 GB: at 5 Gbps that is ~18–20 min ideal; kill the listing if the measured ETA > 90 min
  (§2.3 computes ETA from a real 60 s sample). [FACT: derivation] 671 GB × 8 bit/B ÷ 5e9 bit/s ≈ 1073 s ≈ 18 min.

□ P1–P7 PRE-REGISTRATION ROWS COPIED INTO bench/RESULTS.md (§5 table, verbatim, measured column = —).
  Do this at D5, BEFORE the session, so the prediction cannot be edited to fit the result (FOP-2).

□ TWO-LOAD TRACE ADAPTED to the bench client (§4): heavy-tail saturated (throughput) + shallow
  (TTFT). R3b finding: one trace cannot measure both.

□ ON-DEMAND instance only — NOT interruptible (spec §2 rule 5: a preempted TP/EP bring-up burns the
  day). Hard budget alarm armed at 12 node-hrs (§0).

□ PUBLISHED REFERENCE OPEN: the SGLang "DeepSeek Usage" doc single-node 8×H200 R1 config
  (github.com/sgl-project/sglang, docs/references/deepseek.md) — our P2/P3 numbers get checked
  against it. Honesty anchor (ADR-0012 §"Why R1 is the teacher" #4).
```

---

## 2. THE DAY — Hour 0–0.75 · PRE-FLIGHT (topo · start the pull · busbw sweep)

Rent the node, SSH in, paste §0. Then, in this order (the pull runs in the background while you verify).

### 2.1 Topology verify — ACCEPT/REJECT gate (do this in the first 5 minutes)

```bash
nvidia-smi topo -m
```

**ACCEPT** iff every GPU×GPU off-diagonal cell reads `NV8` or `NV18` (all pairs NVLink-connected —
one NVLink domain, 900 GB/s/GPU). **REJECT and kill the listing** if any pair reads `PIX`, `PHB`,
`NODE`, or `SYS` (PCIe path — not the NVLink node we paid for; sunk ≤ $30, spec §2 rule 6 / kill
criteria). Auto-check:

```bash
nvidia-smi topo -m | awk '
  /^GPU[0-9]/ { for(i=2;i<=9;i++) if($i!="X" && $i!~/^NV(8|18)$/){bad=$i" at row "$1}
} END { if(bad) print "REJECT — non-NVLink link: "bad; else print "ACCEPT — all pairs NV8/NV18" }'
```

Also confirm this is a volume (irreplaceable-data question) and see the GPUs:

```bash
vast-capabilities | jq '.instance.workspace_is_volume, .hardware.gpu.name, .hardware.gpu.count'
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
```

### 2.2 Build nccl-tests (2 min) — needed for the P1 busbw sweep

```bash
cd "$WS"
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
# MPI=0 → single-process launcher, all 8 GPUs via -g 8 (no mpirun needed for one node)
make -j MPI=0 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
ls ./build/all_reduce_perf   # built binary
```

### 2.3 START THE 671 GB WEIGHT PULL IMMEDIATELY — and gate its ETA ≤ 90 min

Kick this off NOW so it downloads while you run busbw (§2.4). Do not wait for it to bring up the engine.

```bash
source /venv/main/bin/activate
uv pip install -U "huggingface_hub[hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1                 # multi-connection fast path
mkdir -p "$HF_HOME"
# background pull; log to a file we sample for the ETA gate
nohup huggingface-cli download "$MODEL" \
  --repo-type model --local-dir "${WS}/models/DeepSeek-R1" \
  > "${WS}/pull.log" 2>&1 &
echo "PULL_PID=$!"
```

**ETA gate — sample real throughput after 60 s, kill if ETA > 90 min** [FACT: derivation]
(remaining_GB ÷ observed_GB_per_min):

```bash
before=$(du -sm "${WS}/models/DeepSeek-R1" 2>/dev/null | cut -f1); sleep 60
after=$(du -sm "${WS}/models/DeepSeek-R1" 2>/dev/null | cut -f1)
awk -v a="$after" -v b="$before" 'BEGIN{
  rate=(a-b);                       # MB/min
  gbmin=rate/1024;
  remain=671-a/1024;
  eta=(gbmin>0)?remain/gbmin:9999;
  printf "rate=%.2f GB/min, downloaded=%.1f GB, ETA=%.1f min\n", gbmin, a/1024, eta;
  print (eta>90)?"KILL — ETA>90min (spec kill criterion)":"OK — pull ETA within gate"
}'
```

If KILL: destroy the instance now (`vastai destroy instance $CONTAINER_ID --api-key $CONTAINER_API_KEY`),
sunk ≤ $30. Otherwise proceed — the pull continues in the background.

### 2.4 nccl-tests busbw sweep 1 KB → 1 GB — WHILE downloading → **P1**

```bash
cd "${WS}/nccl-tests"
# AllReduce, 1KB..1GB, doubling (-f 2), all 8 GPUs (-g 8); ring is the large-msg algo
NCCL_ALGO=Ring  ./build/all_reduce_perf -b 1K -e 1G -f 2 -g 8 | tee "${WS}/nccl_ring.log"
# crossover check: tree wins small, ring wins large (A6 Rung-0 note; optional if time-boxed)
NCCL_ALGO=Tree  ./build/all_reduce_perf -b 1K -e 1G -f 2 -g 8 | tee "${WS}/nccl_tree.log"
```

Read from the `busbw (GB/s)` column of the output:
- **Large message (≥256 MB):** the `busbw` at the 256 MB / 512 MB / 1 GB rows → the P1 large-msg number.
- **Small message (1–8 MB, decode-sized):** the `time (us)` at the 1 MB row → the P1 latency floor
  (this is the input to P2's per-token AllReduce cost).

**P1 gate** [INFERENCE]: large-msg busbw **≥ 720 GB/s** (80% of the 900 GB/s NVLink line rate
[FACT: derivation]); small-msg floor **15–40 μs**. Log both to RESULTS.md now (don't wait). If
large-msg busbw < 600 GB/s the interconnect is degraded — note it; every downstream number inherits it.

---

## 3. Hour 0.75–1.75 · BRING-UP (TP-8 up · correctness smoke · B=1 decode → P2)

**Bring-up hard clock: > 2 h from here → swap to the fallback model (§9 kill criteria). Set a timer.**

### 3.1 Wait for the pull to finish

```bash
until [ -f "${WS}/models/DeepSeek-R1/config.json" ] && ! pgrep -f "huggingface-cli download" >/dev/null; do
  sleep 30; du -sh "${WS}/models/DeepSeek-R1" 2>/dev/null
done
echo "weights present: $(du -sh ${WS}/models/DeepSeek-R1 | cut -f1)"
```

### 3.2 Launch the SGLang server via the pinned image (Docker)

If the base image already has SGLang, skip Docker and call `python -m sglang.launch_server` directly.
Otherwise run the pinned container (this is the "arrive with a tested binary" rule, spec §2 rule 2):

```bash
docker run --gpus all --shm-size 32g --network host --ipc host \
  -v "${WS}/models:/models" -v "${HF_HOME}:/root/.cache/huggingface" \
  --name sglang-r1 -d "$SGLANG_IMAGE" \
  python3 -m sglang.launch_server \
    --model-path /models/DeepSeek-R1 \
    --tp 8 \
    --trust-remote-code \
    --host 0.0.0.0 --port ${PORT} \
    --mem-fraction-static 0.90 \
    --context-length 65536
docker logs -f sglang-r1   # watch until "The server is fired up and ready to roll!"
```

### 3.3 The EXACT baseline launch command (the published single-node R1-FP8 config)

This is the TP-8 baseline config from the SGLang DeepSeek reference (verbatim shape; verify flag
names against the pinned image's `--help`, spec §2 rule 6). **This is the exact line smoke-tested
pre-rental** (§1) with a small model and `--tp 1`:

```bash
python3 -m sglang.launch_server \
  --model-path /models/DeepSeek-R1 \
  --tp 8 \                        # tensor-parallel across the 8-GPU NVLink domain (fit forces this: TP-8 shard 83.9 GB/GPU)
  --trust-remote-code \
  --host 0.0.0.0 --port ${PORT} \
  --mem-fraction-static 0.90 \    # leave headroom for KV; raise toward 0.93 if KV pool too small
  --context-length 65536 \
  --enable-torch-compile \        # published perf flag; adds compile time to bring-up
  --torch-compile-max-bs 256
# FP8 is native in the R1 weights (fp8_e4m3) — no --quantization flag needed.
# vLLM fallback:  vllm serve /models/DeepSeek-R1 --tensor-parallel-size 8 --trust-remote-code \
#                   --max-model-len 65536 --port ${PORT}
```

At startup the log prints the KV pool — **capture it for P5 now**:
```bash
docker logs sglang-r1 2>&1 | grep -iE "max_total_num_tokens|KV cache|memory pool"
curl -s http://${HOST}:${PORT}/get_server_info | jq '.max_total_num_tokens, .max_running_requests'
```

### 3.4 Correctness smoke — 5 greedy prompts + logprob sanity (gate before ANY perf number)

Greedy (temp 0) on 5 prompts, eyeball for coherence + no repetition/garbage:

```bash
for p in \
 "The capital of France is" \
 "2+2=" \
 "def fibonacci(n):" \
 "The three laws of thermodynamics are" \
 "Translate to French: good morning"; do
  echo "=== $p ==="
  curl -s http://${HOST}:${PORT}/generate -H 'Content-Type: application/json' \
    -d "{\"text\":\"$p\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":48}}" \
    | jq -r '.text'
done
```

Logprob sanity (the cheap numerics oracle — greedy top-1 logprob should be finite, ≤ 0, and the
token stream deterministic across two identical calls):

```bash
curl -s http://${HOST}:${PORT}/generate -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":8},
       "return_logprob":true,"top_logprobs_num":5}' \
  | jq '.meta_info.output_token_logprobs[:3]'
```

**Gate:** outputs coherent + logprobs finite/≤0 + two identical greedy calls return identical text.
Garbage here = FP8/TP-shard/rope wiring bug — stop, do not measure. (If it persists past the 2 h
bring-up clock → fallback model §9.)

### 3.5 B=1 decode tok/s → **P2**

Single stream, one request at a time, measure decode rate:

```bash
python3 -m sglang.bench_serving \
  --backend sglang --host ${HOST} --port ${PORT} \
  --dataset-name random --num-prompts 8 --max-concurrency 1 --request-rate inf \
  --random-input-len 512 --random-output-len 256 --random-range-ratio 1.0 \
  --output-file "${WS}/p2_b1.jsonl"
# read: "Output token throughput (tok/s)" at concurrency 1 = the B=1 decode rate
```

**P2 prediction** [INFERENCE]: **30–60 tok/s — LATENCY-bound, not bandwidth-bound.**
Derivation [FACT: derivation]: naive active-weight roofline ~37 GB active FP8 ÷ 8 GPUs ÷ 4.8 TB/s ≈
0.96 ms → ~1,000 tok/s; the REAL bound is 61 layers × (2 AllReduces × 20–40 μs + launches + routing)
≈ 4–8 ms/token. This is the node-scale B=1 lesson: on sm_120 the arc was overhead→memory; here it's
**latency** (the P1 small-msg floor × 2 AllReduces × 61 layers is the dominant term). Log tok/s AND
the bound verdict.

---

## 4. THE BENCHMARK CLIENT — two traces (this section is referenced by §3, §5, §6, §7)

**R3b finding (bench/continuous.py): one trace cannot measure both throughput and TTFT.** So we drive
TWO loads, both adapted from `bench/continuous.py::make_trace`:

- **Heavy-tail SATURATED** (throughput surface): the make_trace `heavy` mix per wave =
  `24×64 + 6×256 + 2×512` output tokens, prompt 32, queue kept full (`--request-rate inf`, high
  `--max-concurrency`). Measures aggregate tok/s and the knee.
- **SHALLOW** (TTFT surface): small output lengths, low arrival rate, small concurrency so queue-wait
  does NOT dominate — TTFT reflects admission, not backlog.

### 4.1 High-fidelity path — generate the exact make_trace distribution as a bench dataset

Run on the node (writes a ShareGPT-style JSONL the bench client reads via `--dataset-path`):

```bash
cat > "${WS}/gen_trace.py" <<'PY'
import json, random, sys
# mirrors bench/continuous.py::make_trace TRACES + PROMPT_LEN
PROMPT_LEN = 32
MIX = {"heavy": [64]*24 + [256]*6 + [512]*2,      # saturated / throughput
       "shallow": [8]*28 + [16]*4}                 # short outputs / TTFT surface
kind, waves, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rng = random.Random(0); rows=[]
for _ in range(waves):
    w = MIX[kind][:]; rng.shuffle(w)
    for m in w:
        rows.append({"prompt_len": PROMPT_LEN, "output_len": m})
with open(out,"w") as f:
    for r in rows: f.write(json.dumps(r)+"\n")
print(f"{kind}: {len(rows)} requests -> {out}")
PY
python3 "${WS}/gen_trace.py" heavy   16 "${WS}/trace_heavy.jsonl"    # ~512 reqs, saturated
python3 "${WS}/gen_trace.py" shallow 8  "${WS}/trace_shallow.jsonl"  # ~256 reqs, shallow
```

Drive with the generated distribution (use `--dataset-name random` with matched lengths if the
pinned image's bench tool lacks a custom-jsonl loader — check `sglang.bench_serving --help`):

```bash
# THROUGHPUT (saturated): queue full, wide concurrency
python3 -m sglang.bench_serving --backend sglang --host ${HOST} --port ${PORT} \
  --dataset-name random --num-prompts 2048 --max-concurrency ${C} --request-rate inf \
  --random-input-len 32 --random-output-len 256 --random-range-ratio 0.25 \
  --output-file "${WS}/thru_C${C}.jsonl"

# TTFT (shallow): low arrival rate, small concurrency, short outputs
python3 -m sglang.bench_serving --backend sglang --host ${HOST} --port ${PORT} \
  --dataset-name random --num-prompts 256 --max-concurrency 8 --request-rate 4 \
  --random-input-len 32 --random-output-len 8 --random-range-ratio 0.5 \
  --output-file "${WS}/ttft_shallow.jsonl"
```

Placeholders: `${C}` = concurrency (the §5 sweep sets it), `${PORT}`/`${HOST}` from §0.
vLLM fallback client: `vllm bench serve --model /models/DeepSeek-R1 --dataset-name random
--num-prompts ... --max-concurrency ... --request-rate ...` (same flags, `vllm bench serve` shape).

Key output fields to record: **Output token throughput (tok/s)**, **Median/P99 TTFT (ms)**,
**Median/P99 ITL (ms)**, **Request throughput (req/s)**.

---

## 5. Hour 1.75–3.25 · THROUGHPUT (concurrency sweep 1→256 → P3)

Sweep concurrency on the saturated trace; record aggregate tok/s + ITL/TTFT percentiles at each point;
find the knee (where aggregate tok/s stops rising).

```bash
for C in 1 2 4 8 16 32 64 128 256; do
  echo "=== concurrency $C ==="
  python3 -m sglang.bench_serving --backend sglang --host ${HOST} --port ${PORT} \
    --dataset-name random --num-prompts $((C*8)) --max-concurrency $C --request-rate inf \
    --random-input-len 32 --random-output-len 256 --random-range-ratio 0.25 \
    --output-file "${WS}/sweep_C${C}.jsonl"
done
grep -H "Output token throughput" "${WS}"/sweep_C*.jsonl 2>/dev/null || \
  for f in "${WS}"/sweep_C*.jsonl; do echo "$f:"; jq '{tok_s:.output_throughput, ttft_p99:.p99_ttft_ms, itl_p99:.p99_itl_ms}' "$f"; done
```

**Experts-hit-vs-B curve** (the P3 mechanism — why the MoE knee is later than dense). Turn on the
expert-distribution recorder and read how many of the 256 experts are touched per batch:

```bash
# [verify flag/endpoint names against the pinned image --help — engine flags change fast (spec §2 rule 6)]
curl -s http://${HOST}:${PORT}/start_expert_distribution_record
# ... run one sweep point at a given C ...
curl -s http://${HOST}:${PORT}/stop_expert_distribution_record
curl -s http://${HOST}:${PORT}/dump_expert_distribution_record > "${WS}/experts_C${C}.json"
```

**P3 prediction** [INFERENCE]: **aggregate decode @ concurrency ≥128 ≥ 3,000 tok/s; the scaling knee
arrives LATER than dense.** Derivation [FACT: derivation]: MoE amortization is weaker than dense —
E[experts hit] = 256·(1−(1−8/256)^B); at B=32 that is ≈163/256, so expert-weight reads keep growing
with B until B ≫ E/k = 32 (dense R3a saw AI≈B and knee at B≈128). **Register the bent curve** — the
aggregate-tok/s-vs-B plot should bend later than the dense R3a curve. Cross-check the peak against the
published SGLang R1 8×H200 number (§1 reference).

---

## 6. Hour 3.25–4.75 · EP vs TP (B∈{32,64,128} + per-expert load histogram → P4)

Two server configs, same trace. Baseline = TP-8 MoE (already up, §3.3). EP arm = expert-parallel.

```bash
# --- EP-8 arm: restart the server with expert parallelism ---
docker rm -f sglang-r1
docker run --gpus all --shm-size 32g --network host --ipc host -v "${WS}/models:/models" \
  --name sglang-r1-ep -d "$SGLANG_IMAGE" \
  python3 -m sglang.launch_server --model-path /models/DeepSeek-R1 --tp 8 \
    --enable-ep-moe --ep-size 8 \      # [verify: newer images use --enable-deepep-moe --deepep-mode auto]
    --trust-remote-code --host 0.0.0.0 --port ${PORT} --mem-fraction-static 0.90 --context-length 65536
docker logs -f sglang-r1-ep   # wait for ready
```

Run BOTH configs at B ∈ {32, 64, 128} (rerun §5 sweep points for each), record throughput + the
per-expert load histogram (`dump_expert_distribution_record`, §5) — compute max/mean load ratio to
quantify imbalance:

```bash
for B in 32 64 128; do
  python3 -m sglang.bench_serving --backend sglang --host ${HOST} --port ${PORT} \
    --dataset-name random --num-prompts $((B*8)) --max-concurrency $B --request-rate inf \
    --random-input-len 32 --random-output-len 256 --output-file "${WS}/ep_B${B}.jsonl"
done
# imbalance from the dumped histogram:
jq '[.[]]|{max:max,mean:(add/length),ratio:(max/(add/length))}' "${WS}/experts_C128.json"
```

**P4 prediction** [INFERENCE]: **EP-8 ≥ 1.2× TP-8 at B≥64, with the imbalance histogram logged.**
Derivation [FACT: derivation]: wire/token — EP all-to-all ≈ top-k·h·(1 B fp8 dispatch + 2 B bf16
combine)·(7/8) ≈ **150 KB** vs TP-MoE AllReduce ≈ 61 × 2·(7/8)·7168·2 B ≈ **1.5 MB** — ~10× less wire
and no replicated expert reads. Risk = hot experts (per-expert load ties to `moe.py`'s aux-loss-free
balancing); the histogram max/mean ratio quantifies it.

### A6 Rung-1 TP micro (20 min side-quest): the 2-AllReduce/layer count check

Confirm the per-layer collective count that P2 assumed: 1 AllReduce after attention + 1 after MLP =
**2 AllReduces/layer**. Either read it off an `nsys` decode-step capture, or from SGLang's comm
counters if exposed. This closes A6 Rung-1 (the "count the collectives" check).

```bash
# best-effort nsys trace of ~20 decode steps (if nsys is in the image):
nsys profile -o "${WS}/tp_micro" --duration 8 \
  python3 -m sglang.bench_serving --backend sglang --host ${HOST} --port ${PORT} \
    --dataset-name random --num-prompts 4 --max-concurrency 1 --random-output-len 64 || \
  echo "nsys unavailable — read AllReduce count from engine comm counters instead"
```

---

## 7. Hour 4.75–5.5 · MLA KV AUDIT (engine-reported KV/token → P5)

Read the engine's actual KV/token and pool size; compare to the analytic 70.3 KB/token.

```bash
curl -s http://${HOST}:${PORT}/get_server_info | jq '{max_total_num_tokens, max_running_requests}'
docker logs sglang-r1 2>&1 | grep -iE "KV cache|max_total_num_tokens|token pool|#tokens"
```

Compute KV/token from the reported pool: `KV_bytes_per_token = free_KV_bytes / max_total_num_tokens`.
Free KV bytes ≈ (per-GPU HBM − weights_shard − activations) × 8. Cross-check against the analytic:

```bash
# analytic MLA KV/token: (c_kv 512 + k_pe 64) × 61 layers × 2 B (BF16)
python3 -c "print('analytic MLA KV/token =', (512+64)*61*2, 'B =', (512+64)*61*2/1024, 'KB')"
# capacity check: pool_tokens vs ~6.4M analytic (e.g. 128 concurrent × 50K ctx)
```

**P5 prediction** [INFERENCE]: **engine-reported KV ≈ 70 KB/token BF16 (±10%).**
Derivation [FACT: derivation]: (512+64)×61×2 B = 70,272 B = 70.3 KB/token. Capacity:
free-KV-GB ÷ 70.3 KB ≈ the ~6.4 M-token pool (ADR-0012 KV economics; 4.7× smaller than a 70B GQA-8's
327.7 KB/token). Log both the per-token number AND the max concurrent×ctx the pool implies.

---

## 8. Hour 5.5–7.0 · PD-DISAGGREGATION (prefill/decode split → P6)

Compare co-located (default, §3.3) vs prefill/decode-disaggregated on the TWO-load trace. On one node,
PD-disagg splits the 8 GPUs (e.g. 4 prefill + 4 decode) with a mini load balancer.

```bash
# [verify exact disagg flags against pinned image --help — this is the fastest-moving surface]
# prefill server (GPUs 0-3):
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/DeepSeek-R1 \
  --tp 4 --disaggregation-mode prefill --trust-remote-code --port 30001 --mem-fraction-static 0.90 &
# decode server (GPUs 4-7):
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m sglang.launch_server --model-path /models/DeepSeek-R1 \
  --tp 4 --disaggregation-mode decode  --trust-remote-code --port 30002 --mem-fraction-static 0.90 &
# mini load balancer fronting both (routes prefill→decode via KV transfer):
python3 -m sglang.srt.disaggregation.mini_lb --prefill http://127.0.0.1:30001 \
  --decode http://127.0.0.1:30002 --host 0.0.0.0 --port ${PORT} &
```

Then drive BOTH traces (§4) at matched throughput, disagg vs co-located:

```bash
# shallow trace → TTFT p95 (the disagg win surface)
python3 -m sglang.bench_serving --backend sglang --host ${HOST} --port ${PORT} \
  --dataset-name random --num-prompts 256 --max-concurrency 8 --request-rate 4 \
  --random-input-len 512 --random-output-len 8 --output-file "${WS}/disagg_ttft.jsonl"
# saturated trace → goodput under SLO
python3 -m sglang.bench_serving --backend sglang --host ${HOST} --port ${PORT} \
  --dataset-name random --num-prompts 2048 --max-concurrency 128 --request-rate inf \
  --random-input-len 512 --random-output-len 256 --output-file "${WS}/disagg_thru.jsonl"
```

**P6 prediction** [INFERENCE]: **TTFT p95 ≥ 2× better at matched throughput (or goodput ≥ 1.3× under
SLO).** Derivation [FACT: derivation]: co-located prefill bursts evict decode from the batch (the
R4.2/R4.6 admission-spike effect at scale — measured ITL p99 ~29 ms admission spikes on sm_120);
disagg isolates them. Measured on the two-load trace — **shallow for TTFT, saturated for throughput**
(R3b lesson). If the disagg bring-up itself blocks → do NOT debug on node-time (§9); record co-located
numbers, file the gap.

---

## 9. Hour 7.0–7.75 · STRETCH · MTP spec-decode (→ P7)

R1 ships an MTP (multi-token-prediction / NEXTN) head. Relaunch with speculative decoding and measure
acceptance + speedup vs the §3.5 / §5 baselines.

```bash
# [verify spec-decode flag names against pinned image --help]
python3 -m sglang.launch_server --model-path /models/DeepSeek-R1 --tp 8 --trust-remote-code \
  --host 0.0.0.0 --port ${PORT} --mem-fraction-static 0.88 --context-length 65536 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2
# re-run the B=1 (§3.5) and a mid-concurrency (§5) point; read acceptance from the server logs/metrics:
curl -s http://${HOST}:${PORT}/get_server_info | jq '.'   # look for accept length / spec stats
```

**P7 prediction** [INFERENCE, stretch]: acceptance **60–80%**, decode **×1.5–2.0**. Acceptance is
domain-dependent — register the band, measure. Re-run anything noisy from the day in this window.

---

## 10. Hour 7.75–8.25 · LEDGER + RELEASE (the day is not done until this is)

1. **Every number → `bench/RESULTS.md`** (§11 pre-registration rows get their `measured` filled).
2. **Postmortem skeleton → `performance/notes/A6_serving_day.md`** (1 page, peer-review quality;
   §12 DoD). What did the node teach that sm_120 could not?
3. **Release.** Confirm all six headline numbers (§12) are in RESULTS.md, THEN destroy:

```bash
grep -c "P[1-7]" "$RESULTS"    # sanity: the P-rows are present and filled
vastai destroy instance $CONTAINER_ID --api-key $CONTAINER_API_KEY
```

---

## 11. P1–P7 PRE-REGISTRATION TABLE (copy VERBATIM into bench/RESULTS.md BEFORE the session, D5)

> **Copy these rows into `bench/RESULTS.md` BEFORE the session (D5).** The `measured` column stays `—`
> until the run fills it — pre-registration means the prediction is frozen before the number is seen
> (FOP-2). Predicted + derivation are verbatim from `PERF_PLAN.md` §"Phase 4"; the `[FACT]/[INFERENCE]`
> labels follow FOP-4 (the derivations are `[FACT: derivation]`; the predicted headline is the
> `[INFERENCE]` under test).

| # | experiment | predicted | derivation / bound | measured |
|---|---|---|---|---|
| P1 | AllReduce busbw, large msg (≥256 MB) | **≥720 GB/s** (80% of the 900 GB/s line rate); small msg (1–8 MB, decode-sized): **latency floor 15–40 μs** | ring efficiency at line rate; the small-msg floor is the input to P2 | — |
| P2 | R1 B=1 decode tok/s | **30–60 — LATENCY-bound, not bandwidth-bound** | naive active-weight roofline: ~37 GB active FP8 ÷ 8 GPUs ÷ 4.8 TB/s ≈ 0.96 ms → ~1,000 tok/s. Real bound: 61 layers × (2 AllReduces × 20–40 μs + launches + routing) ≈ 4–8 ms/token. THE node-scale B=1 lesson (contrast the sm_120 arc: overhead → memory; here: latency) | — |
| P3 | aggregate decode @ concurrency ≥128 | **≥3,000 tok/s**; the scaling knee arrives LATER than dense | MoE amortization is weaker than dense: E[experts hit] = 256·(1−(1−8/256)^B) → B=32 hits ≈163/256, so expert-weight reads keep growing with B until B ≫ E/k = 32 — dense R3a saw AI≈B; register the bent curve | — |
| P4 | EP-8 vs TP-8, B ≥ 64 | **EP ≥1.2×**, with the imbalance histogram logged | wire/token: EP a2a ≈ top-k·h·(1 B fp8 dispatch + 2 B bf16 combine)·(7/8) ≈ **150 KB** vs TP-MoE AllReduce ≈ 61 × 2·(7/8)·7168·2 B ≈ **1.5 MB** — ~10× less wire and no replicated expert reads; risk = hot experts (per-expert load ties to `moe.py`'s aux-loss-free balancing) | — |
| P5 | MLA KV/token (engine-reported) | **≈70 KB/token BF16 (±10%)** | (512+64)×61×2 B; capacity: free-KV-GB ÷ 70.3 KB ≈ the ~6 M-token pool (e.g. 128 × 50 K ctx) | — |
| P6 | PD-disagg vs co-located | **TTFT p95 ≥2× better at matched throughput** (or goodput ≥1.3× under SLO) | co-located prefill bursts evict decode from the batch (R4.2/R4.6 at scale); measured on the two-load trace — shallow for TTFT, saturated for throughput (R3b lesson) | — |
| P7 | MTP spec-decode (stretch) | acceptance **60–80%**, decode **×1.5–2** | R1 ships an MTP head; acceptance is domain-dependent — register the band, measure | — |

---

## 12. DoD — the six headline numbers + the postmortem (all `[FACT]`-ledgered BEFORE release)

The node is not released until these are in `bench/RESULTS.md`:

- [ ] P1–P6 measured or explicitly killed with the observed blocker (P7 stretch).
- [ ] **The six headline numbers:**
  1. **busbw large / small** — GB/s at ≥256 MB and μs floor at 1 MB (P1)
  2. **B=1 tok/s + its bound** — the number and whether it's latency/bandwidth-bound (P2)
  3. **aggregate peak + knee-B** — peak tok/s and the concurrency where the curve bends (P3)
  4. **EP:TP ratio + balance histogram** — the speedup and the max/mean expert-load ratio (P4)
  5. **KV/token + pool size** — engine-reported KB/token and max-token pool (P5)
  6. **disagg delta** — TTFT p95 (and/or goodput) improvement, disagg vs co-located (P6)
- [ ] **Postmortem note** (1 page, peer-review quality) at **`performance/notes/A6_serving_day.md`**:
      what the node taught that sm_120 could not (node-scale latency floor, MoE amortization bend,
      EP wire economics, MLA KV at scale, PD-disagg goodput).

---

## 13. KILL CRITERIA (decide fast; do not debug on node-time)

| Trigger | Action | Rationale |
|---|---|---|
| **Bring-up > 2 h** (server won't come up, or §3.4 correctness garbage persists) | **Swap to `Qwen/Qwen3-235B-A22B-FP8`** (§0 `$FALLBACK_MODEL`): 235 GB fits with 365 GB headroom, battle-tested; relaunch §3.3 with `--model-path` = the fallback. The day's physics (busbw, B=1 latency, EP/TP, KV, disagg) survives the model swap — only the R1 flag is lost. | ADR-0012 fallback; PERF_PLAN kill criteria. |
| **Topo shows PIX/PHB** (§2.1) **or weight-pull ETA > 90 min** (§2.3) | **Kill the listing inside hour 1** — destroy the instance (sunk ≤ $30). Re-rent a verified NVLink listing. | Not the node we paid for / pull will eat the day. spec §2 rule 6. |
| **An engine bug blocks EP (§6) or disagg (§8)** | Do **NOT** debug the engine on node-time. **Measure the TP-8 surface completely** (P1/P2/P3/P5 + P6 co-located baseline), file the gap in the postmortem, move on. | Node-time is for measuring physics, not patching engines. PERF_PLAN kill criteria. |
| **12 node-hrs elapsed** (§0 alarm) | Stop and justify continuing or release. | Hard budget alarm. |

**Cost envelope:** 6–10 node-hrs × $20–32/node-hr ≈ **$150–320** [UNCERTAIN: market] (Vast-class 2026;
8×H200 ~$20–32/node-hr — re-check at session time, the tier is the decision, price only moves the total).

---

### Appendix — reaching the server from OUTSIDE the node (if driving load off-box)

The whole plan drives load ON the node (`--host 127.0.0.1` → bypasses Caddy, no token). To reach the
SGLang port from your laptop instead, either SSH-forward (most secure, no open port):

```bash
ssh -p $VAST_TCP_PORT_22 -L 30000:127.0.0.1:${PORT} root@$PUBLIC_IPADDR   # then http://localhost:30000 locally
```

or expose ${PORT} behind the Caddy auth edge (AGENTS.md §7) and call it with the token
(`Authorization: Bearer $OPEN_BUTTON_TOKEN`). On-node localhost is simplest and is what §3–§9 assume.
