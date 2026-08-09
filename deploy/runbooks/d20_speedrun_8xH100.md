# The $100 d20 — 8×H100 press-play runbook

**What this is.** The provision→stage→launch→eval→teardown script for the headline artifact: a
**nanochat-grade d20 (480.4M params measured at vocab 32768, D = ratio-20 ≈ 9.6B tokens, C ≈ 2.77e19)**
pretrained data-parallel on **8×H100 SXM**, then scored on the DCLM CORE 22-task suite. Optimizer is
**Muon+AdamW — ADOPTED, not raced** (`bench/RESULTS.md` §*Decision — Muon+AdamW ADOPTED*: external
evidence stack, FOP-7 reuse > re-measure). Pre-registered **CORE band 0.19–0.22** vs the
original-nanochat-d20 anchor **0.2219** — presented as a *CORE-vs-FLOPs point on nanochat's published
curve* (we buy 73% of the anchor's compute at 86% of its N), **never** a depth-matched headline.

> ## ⛔ THIS IS THE PLAN, NOT AN AUTO-RUN — two hard gates before any `vastai create`
>
> 1. **GATED on P5 — the $10–15 d12 dress rehearsal on 1×H100 must pass first.** P5 (TASKSPEC d20
>    gate) validates the recipe (bf16/compile-on-sm90), checkpoint kill/resume, the CORE harness
>    against a public checkpoint, the loss curve vs nanochat's published d12, **and — new here — the
>    four unbuilt entrypoints §0.5 lists** (SDPA-in-training, the torchrun launcher shim, resumable
>    distributed checkpoint, the 10B streamer). A d12 that trips its loss-band tripwire catches a
>    Muon-implementation bug for a fraction of a diverged-at-hour-2 d20 (~$50).
> 2. **GATED on the user's explicit $100 authorization.** No paid 8×H100 instance is created without
>    the Navigator saying "go". This document is executed by a human who has read it, not launched by
>    an agent.
>
> Everything below is `[FACT]` (verified in code/docs at HEAD) / `[MEASURED]` (a number off a box) /
> `[INFERENCE]` (a pre-registered or scaled prediction), labeled honestly per FOP-4.

---

## §0.5 — The honest gap: what is shipped vs what P5 must land (READ THIS FIRST)

The A7 optimizer-embedded ZeRO-2 math is **shipped + gloo-verified** (`utils/dist_train.py`,
`DistMuonAdamW`; 12/12 ACCEPT incl. single-proc byte-identity + 3-rank oracle). But the *press-play
launch command in §3 does not work against today's HEAD* — four load-bearing entrypoints are unbuilt.
Each is CPU- or 1×H100-buildable and **must land + go green inside P5**. Naming them is the runbook's
job (FOP-4); pretending §3 runs today would be the dishonest move.

| # | Gap `[FACT, verified at HEAD]` | Why the d20 can't press-play without it | Lands in |
|---|---|---|---|
| G1 | **SDPA is OFF in the training path.** `ModelConfig.use_sdpa` defaults **False** (`model.py:79`) and `model_config_for_depth` (`speedrun.py:112`) never sets it → speedrun builds the d20 with **eager attention** that materializes fp32 `(B,H,S,S)` scores. | P1: at device-batch 32 × ctx 2048 × 20 layers × 10 heads the retained scores are **~107 GB > 80 GB HBM** — the run **cannot physically execute**. Needs `use_sdpa=True` **and** forcing `SDPBackend.CUDNN_ATTENTION`/FA3 (PyTorch SDPA does *not* auto-select cuDNN on H100). | P5 (add a `--attention sdpa` knob to speedrun / default `use_sdpa=True` for `depth≥N`; additive, byte-identical off) |
| G2 | **No torchrun launcher shim.** `grep -rn init_process_group src/` = **zero hits**; the process group is initialized only in `tests/`. `train()` gates the ZeRO-2 path on `dist.is_initialized()` (`train.py:255`) — which is **False** under a bare `torchrun … python -m scratch_llm.speedrun`. | The launch in §3 would spawn **8 independent single-GPU replicas** (silent 8× data loss, no `reduce_scatter`/`all_gather`), and all 8 would hammer `cuda:0`. Needs `init_process_group` from torchrun env + `torch.cuda.set_device(LOCAL_RANK)` + per-rank `device=cuda:{LOCAL_RANK}` + `destroy_process_group`. | P5 (the shim + a **2-GPU NCCL smoke** asserting ranks draw *different* batches — the A7 DoD) |
| G3 | ~~Consolidated checkpoint is BUILT; speedrun launcher does not yet wire it~~ **WIRED 2026-07-28.** `DistMuonAdamW.consolidated_state_dict()`/`load_consolidated()` + `save/load_consolidated_checkpoint()` full-gather the optimizer moments to rank 0 and reshard on resume (bitwise round-trip, incl. cross-topology W2→W1; `tests/test_dist_train.py`). The launcher gap is closed: `speedrun.stage_pretrain` now threads `--checkpoint-every` → `work_dir/pretrain_ckpt.pt` (optimizer-state; consolidated full-gather under torch.distributed via `train()`'s existing path), distinct from the optimizer-free stage-boundary `pretrain.pt` (`tests/test_speedrun.py`). | Optimizer-state resume works via the speedrun CLI (`--checkpoint-every N --work-dir runs/d20`) or `train()` directly; the model weights round-trip regardless (all-gathered to identical replicas every step). | done — launch with `--checkpoint-every` set (~30 min cadence) |
| G4 | **No 10B-token streamer.** `load_dataset_tokens` (`data/shards.py:170`) does `np.concatenate([load_shard(p) … ])` — **all shards into host RAM**; its own docstring says *"multiple shards … concatenated in RAM — fine at nano scale, … exactly the boundary where the §D d20 streamer takes over."* | ~20 GB uint16 + a **~40 GB transient** during concat. An 8×H100 SXM node's host RAM (≥1 TB typical) *physically fits* it, so this is **accept-with-verification**, not a hard blocker — but it is not the memory-bounded shuffled streamer §D specifies. | P5 (verify host RAM ≥ ~64 GB free, OR land the §D shuffled multi-shard sampler) |

**Everything that IS shipped and press-play-ready:** the `DistMuonAdamW` ZeRO-2 math (G2's *interior*),
per-rank data sharding logic (`train.py:269`, numpy seed offset by rank), the **collective NaN guard**
(`train.py:363-380`, `all_reduce(MAX)` at `log_every` — the divergence kill-switch), the parquet 10B
shard *builder* (P2, proven at 0.7B/sub-hour), the CORE 22-task suite + CLI (`bench/core_eval.py`,
byte-identical to nanochat `core.yaml`), and the model at exactly **480,431,360 params** `[FACT,
instantiated at HEAD]`.

---

## The pre-registered scorecard (from `bench/RESULTS.md` §*$100 d20 run*; do not edit — the falsifier)

| axis | predicted | KILL / abort | label |
|---|---|---|---|
| N (params, measured) | **480,431,360** (d_model 1280 · 20 layers · 10 heads · head_dim 128, untied) | — | `[FACT]` |
| D (tokens, ratio-20) | **9.6B** ⇒ C = 6ND ≈ **2.77e19** (27% below anchor's 3.77e19) | — | `[FACT]` |
| global batch | **524,288 tok** = 8 GPU × 32 seq × 2048 ctx ⇒ **18,311 steps** | — | `[FACT]` |
| per-step FLOPs | 6·480.4M·524,288 = **1.51e18** | — | `[FACT]` |
| pretrain MFU (bf16, run-level, 6ND w/ embeddings) | **33–40%** (anchor nanochat d20 = 34.4%) | **<28% sustained** after G1 lands | `[INFERENCE]` |
| step time @ global batch | **0.48–0.58 s** | **>0.70 s** | `[INFERENCE]` |
| DP comm overhead (0.96 GB bf16 grads, RS+AG overlapped) | **<1.5% wall** | **>3%** | `[INFERENCE]` |
| 8-GPU scaling efficiency | **≥97%** | **<93%** | `[INFERENCE]` |
| pretrain wall / cost (@ $24/h node) | **2.4–3.3 h / $58–79** | **abort at $90 cumulative → downsize d16** | `[INFERENCE]` |
| **CORE** (22-task DCLM, decontaminated) vs anchor 0.2219 @ 3.77e19 | **0.19–0.22** (buy 73% of C, 86% of N) | **<0.15** (stack bug, not sizing) | `[INFERENCE]` |
| val bits-per-byte (vocab-independent honesty metric) | within band of nanochat d20 interpolated to C≈2.77e19 | — | `[INFERENCE]` |

**Not in the paid run** `[FACT]`: midtrain (`speedrun.py` raises `NotImplementedError`), FP8, any
TP/PP/FSDP (roofline: DP has ~51× headroom over machine balance at B_gpu=65,536 — see A7 budget).

---

## Step 0 — provision + node-quality gate (P6) (~20–40 min, meter starts here)

Uses the `vastai` skill conventions (always `--raw`; register the SSH key before create; poll status;
`-y` on destroy). One-time auth if the account is fresh:

```bash
vastai set api-key <YOUR_KEY>                          # https://console.vast.ai/manage-keys/
vastai show user                                       # verify auth + credit balance ≥ $100
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"   # BEFORE create; skip if already registered
```

Search + create an **8×H100 SXM** node (SXM, *not* PCIe — the d20's ≥97% scaling target needs NVLink;
big disk for venv + torch + ~20 GB shards + checkpoints):

```bash
vastai search offers 'num_gpus=8 gpu_name=H100_SXM reliability>0.98 \
  inet_down>2000 disk_space>250 cuda_vers>=12.4' -o 'dph+' --raw | head
OFFER=<id from above>
vastai create instance "$OFFER" --image vastai/pytorch:@vastai-automatic-tag \
  --disk 300 --ssh --direct --raw
# poll until running, then connect
while :; do st=$(vastai show instance "$INSTANCE" --raw | jq -r '.actual_status'); echo "$st"; \
  [ "$st" = running ] && break; \
  case "$st" in exited|offline|unknown) echo "BAD status; destroy+retry"; break;; esac; sleep 5; done
vastai ssh-url "$INSTANCE"                              # → ssh://root@HOST:PORT
```

**On the box — the P6 node-quality gate (run BEFORE committing to the run):**

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader   # expect 8× H100 80GB
nvidia-smi topo -m                                                       # every pair must read NV# (NVLink), not SYS/PHB
# nccl-tests all_reduce bus bandwidth (build once; the canonical node-health probe):
git clone https://github.com/NVIDIA/nccl-tests && cd nccl-tests && make -j MPI=0 && cd ..
./nccl-tests/build/all_reduce_perf -b 8 -e 512M -f 2 -g 8            # read the busbw (GB/s) column
```

- **Predicted `[INFERENCE]`:** healthy H100 SXM NVLink busbw **~370–480 GB/s** on the large-message
  all-reduce (A100 SXM measures ~250–350; H100 is higher).
- **GATE (P6) `[FACT, pre-registered]`:** **busbw ≥ 350 GB/s** on the largest size. A node below the
  floor is a **bad node → `vastai destroy "$INSTANCE" -y` and re-rent** — never train on it (a slow
  interconnect silently blows the ≥97% scaling target and the <1.5% comm budget). Also abort if any
  GPU shows ECC errors or a topo pair that isn't NVLink.
- **Standing $90 abort `[FACT, pre-registered]`:** track cumulative `vastai` spend; at **$90
  cumulative** (node + any re-rents + storage) stop and **downsize to d16** — do not chase the d20
  past the cap.
- **Wall:** provision poll ~5–15 min; nccl-tests build+run ~10–20 min.

Bootstrap the repo on the pod (see `docs/VASTAI_BOOTSTRAP.md` / `scripts/bootstrap-pod.sh`):

```bash
git clone <repo> ~/scratch_llm && cd ~/scratch_llm && uv venv && source .venv/bin/activate
uv pip install -e ".[dev,data]"                        # data extra = pyarrow for the parquet path
python -c "import torch;print(torch.__version__, torch.cuda.device_count(), torch.version.cuda)"
```

---

## Step 1 — stage the ~10B-token corpus (P2) (hours, **OFF the GPU clock**)

The tokenize is pure-Python and CPU-bound — **it must not run against the $24/h meter**. Two honest
options: (a) build it on a **cheap CPU box or a fast persistent volume the night before** and
`vastai copy` the shards onto the node, or (b) if building on the node, do it *before* you would
otherwise start burning GPU — the H100s sit idle either way, so pre-stage.

**Dry-run the download plan first** (prints file list + GiB + est. tokens; downloads nothing):

```bash
python -m scratch_llm.data.shards --out data/fineweb_edu_10b \
  --fineweb-parquet --target-tokens 1e10 --dry-run
```

Then the real build (the decontaminated 10B corpus):

```bash
python -m scratch_llm.data.shards --out data/fineweb_edu_10b --vocab-size 32768 \
  --fineweb-parquet --target-tokens 1e10 --num-workers 16 --decontaminate \
  --eval-file eval_sets/arc.txt --eval-file eval_sets/mmlu.txt \
  --eval-file eval_sets/gsm8k_test.txt --eval-file eval_sets/hellaswag.txt   # …the CORE sets, 1 item/line
```

- **Predicted `[INFERENCE, scaled from the 0.7B build]`:** the F1-scale build measured **4.54 B/tok ·
  42 shards @ 16.8M tok · 4.01 GiB parquet** for 0.7B `[MEASURED, 2026-07-17]`. The 10B variant is
  **the same code, `--target-tokens 1e10`** — expect **~20 GB uint16 across ~595 shards**, **~50–60
  GiB parquet** download (~14× the 0.7B build), BPE trained on a 16 MiB capped sample.
- **`--decontaminate` (P3/A0) is mandatory here** — the 13-gram gate strips train docs overlapping
  the CORE sets *before* BPE training, so the scored run isn't contaminated. `--eval-file` per real
  test split (else they leak). The final accounting line logs the overlap rate.
- **KILL `[FACT]`:** >20% of docs filtered by `--decontaminate` ⇒ an eval set leaked into the corpus
  slice — investigate before training (not a "just proceed").
- **Epochs sanity (do it by hand — speedrun has no built-in guard; that lives in the race driver):**
  `epochs = D / corpus = 9.6B / ~10B ≈ **0.96** ≤ 1.0` — a healthy **single pass**, no repetition-memorization.
- **KILL `[FACT]`:** any token id ≥ 2¹⁶ ⇒ the writer switches to uint32 (the pre-registered
  kill-switch); at vocab 32768 this never fires, but the shard sidecar records the realized dtype.
- **Wall `[INFERENCE]`:** download minutes; BPE training tens of minutes; the pure-Python encode of
  ~30 GB text dominates — **hours** even at 16 workers. **This is why it is off-clock.**
- **G4 note `[FACT]`:** at train time `load_dataset_tokens` concatenates these ~595 shards into host
  RAM (~20 GB, ~40 GB transient). Confirm `free -g` shows ≥ ~64 GB free on the node before launch, or
  land the §D streamer (see §0.5 G4).

---

## Step 2 — recipe re-validation on Hopper / sm90 (P5 tie-in) (~10 min)

The **bf16 + torch.compile NaN is an sm120 / torch-2.12-inductor FACT** (`bench/RESULTS.md` F4;
reproduces with plain AdamW — it is a codegen bug, not our logic). It **may or may not reproduce on
sm90 — do NOT assume either way.** Re-test it on *this* H100 arch **before** the paid run (ideally
already covered by the P5 d12 rehearsal; re-confirm on the actual rented node):

```bash
# tiny 2-step smoke on 1 GPU: does bf16 + compile stay finite on THIS node's torch/CUDA?
python -m scratch_llm.speedrun --nano --depth 4 --bf16 --compile --device cuda --steps 20
```

- **PASS:** loss stays finite (no `RuntimeError: non-finite loss …`) → the H100 compile path is
  usable; keep `--bf16 --compile` for the ~cheap-MFU win.
- **If it NaNs `[INFERENCE — unknown until measured on sm90]`:** **fall back to eager** — drop
  `--compile`, run **bf16-eager**. The run just gets slower (no inductor fusion); correctness is
  unaffected. The `train.py:371-380` guard fails loud either way, so a NaN here costs 20 steps, not 2 hours.
- **Attention `[FACT, P1/G1]`:** SDPA/FA3 in the *training* path is non-negotiable (see §0.5 G1) —
  the launch in §3 assumes the P5 `--attention sdpa` knob (or `use_sdpa=True`) has landed and forces
  `SDPBackend.CUDNN_ATTENTION`. Verify with a one-layer forward that peak memory is far below the
  ~107 GB eager blowup before committing the full run.

---

## Step 2.5 — P5.5: the target-scale LR probe gate (ADR-0020) (~30–45 min, ≈$9–12)

**MANDATORY before Step 3's full-budget launch.** The d20's LR was swept at d12 / batch 16,384
(η\* = 0.0021) and transferred by the composite rule's √B term — an assumption its own author
flags as unstudied. Karpathy's 320-sweep lesson applies: *validate at target scale*. The probe
runs 3 arms at the EXACT d20 config, ~5% of the token budget each, and picks the LR the full run
commits at. Pre-registered predictions + falsifiers: `bench/RESULTS.md` §*Pre-registration: the
P5.5 d20 probe gate (2026-08-09)*.

```bash
python scripts/d20_probe.py plan          # arms + these exact commands; runs nothing
for m in 0.7 1.0 1.4; do                  # each arm is idempotent — safe to re-run after preemption
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nproc_per_node=8 scripts/d20_probe.py run --mult $m \
    --data-dir <D20_CORPUS> --out-root artifacts/d20_probe --bf16 --device cuda
done
python scripts/d20_probe.py select --out-root artifacts/d20_probe   # prints + saves the winner lr
```

- **Mechanics `[FACT]`:** each arm = 916 steps (480M tokens) at global batch 524,288, muon_adamw,
  cosine over the probe horizon, warmup steps/20 — the P5 sweep's protocol at the d20's shape.
  The probe carries its own torchrun shim (`utils/dist_launch.py` — **G2 closed for this path**;
  speedrun's `main` still needs the same wiring for Step 3, a P5 deliverable). Rank 0 writes
  `artifacts/d20_probe/lr<mult>/results.json`; a finished arm is never re-paid (idempotent skip).
- **Decision rule `[FACT, pre-registered]`:** min val_bpb wins; ties within **0.003 bpb** break to
  the **lower LR**; `select` REFUSES a partial bracket (re-run the missing arm — never select
  around a crash).
- **Tripwires that fire HERE instead of at hour 2 `[pre-registered]`:** per-arm wall >12 min
  (step-time re-price), any `non-finite loss`, step-0 CE ≠ ~10.40, or **lr1.4 winning outright**
  (√B under-transfers at 32× batch — re-derive the transfer before committing; do not just take
  the win). Probe total >$20 ⇒ stop and report.
- **Data `[FACT]`:** `--data-dir` is the staged d20 corpus = **ClimbMix** (F12 operator override,
  FINAL 2026-08-02) — the same shards Step 3 trains on, so the probe's val slice is the run's
  val slice. (This runbook's Step 1 text predates the override; the corpus choice of record is
  ClimbMix.)
- **The winner's lr is Step 3's `<P5_SWEPT_MUON_LR>`** — replacing the composite-rule center if
  the probe disagrees with it. That is the point of the gate.

---

## Step 3 — launch the distributed d20 pretrain (~2.4–3.3 h `[INFERENCE]`)

**Assumes §0.5 G1+G2 have landed in P5.** The ZeRO-2 path activates *inside* `train()` via
`dist.is_initialized()` — so the launcher shim (G2) must init the process group from torchrun's env
and set the per-rank device. Once shipped, this is the command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=8 -m scratch_llm.speedrun \
  --depth 20 --vocab 32768 --context 2048 --batch 32 \
  --steps 18311 --optimizer muon_adamw --lr <P5_SWEPT_MUON_LR> \
  --attention sdpa --bf16 --device cuda \
  --data-dir data/fineweb_edu_10b --work-dir runs/d20
```

- **The single global Muon LR is `<P5_SWEPT_MUON_LR>` from the P5 d12 sweep** — **NOT** nanochat's
  per-group constants (emb 0.3 / unemb 0.008 / matrix 0.02 / scalar 0.5). `[FACT]` Our vanilla-Muon +
  Moonlight RMS-match (`0.2·√max(A,B)`) single-global-LR is a *different variant family*; their
  constants do **not** transfer 1:1 (`bench/RESULTS.md` §Decision). SpeedrunConfig's `3e-3` default is
  a placeholder — replace it with the d12-swept winner.
- **What activates when `--nproc_per_node=8` + the shim initializes the group `[FACT, train.py]`:**
  per-rank data sharding (`np.random.seed(seed + 1000·rank + 1)` — each rank draws a *different* batch
  stream, tested), `DistMuonAdamW` optimizer-embedded ZeRO-2 (reduce_scatter grads → owner-rank whole-
  matrix update → all_gather params), the grad-clip global norm computed by distributed reduction, the
  collective NaN guard, and rank-0-only logging/eval/checkpoint.
- **global batch = 8 × 32 × 2048 = 524,288 tok**; **18,311 steps × 524,288 = 9.6B tokens** (D) `[FACT]`.
- **Predicted `[INFERENCE, pre-registered]`:** **MFU 33–40%**, **step time 0.48–0.58 s**, **comm
  <1.5% wall**, **8-GPU scaling ≥97%**, **wall ≈ 2.4–3.3 h**, **pretrain cost $58–79** @ $24/h.
  Wall arithmetic: 18,311 steps × 0.48–0.58 s = **8,789–10,620 s = 2.4–3.0 h** (the 3.3 h band top
  absorbs eval/checkpoint/warmup overhead). Anchored to the sm120-d8 calibration
  (baseline **0.283 s/step**, B=16, 18.2 GiB, 98% util `[MEASURED, 2026-07-17]`) — but that is a
  *different arch and model*; the real step time is the **first number the P5 d12 rehearsal must
  return**, and the go/no-go for this launch.
- **Loss-at-init oracle (watch the first log line) `[FACT]`:** step-0 CE ≈ **log(32768) = 10.40**;
  off ⇒ a masking/embedding bug, kill immediately (don't pay for a broken init). On the F1 run this
  read **10.378 @ step 10** `[MEASURED]`.
- **KILL `[FACT, pre-registered]`:** MFU <28% sustained (after G1) · step time >0.70 s · comm >3% ·
  scaling <93% · any `non-finite loss` raise (the collective guard) · **cumulative spend hits $90**
  → stop, and downsize to d16 if the cause is sizing not a bug.

---

## Step 4 — guardrails (standing, throughout Step 3)

1. **Preemption-resumable checkpoints (A8 — ckpt code landed 2026-07-19) `[FACT]`.** A consolidated,
   full-gather checkpoint (`save_consolidated_checkpoint`, all optimizer moments gathered to rank 0
   and resharded on resume — bitwise round-trip incl. cross-topology, `tests/test_dist_train.py`)
   every **~30 min**, `vastai copy`'d off-box immediately (spot preemption loses the node with no
   warning). **Full optimizer-state resume works from the speedrun CLI** (wired 2026-07-28):
   launch with `--checkpoint-every N --work-dir runs/d20` — `stage_pretrain` writes
   `runs/d20/pretrain_ckpt.pt` (optimizer-state, consolidated under torch.distributed) every N
   steps alongside the stage-boundary `pretrain.pt`. Off-box sync:
   ```bash
   vastai copy "$INSTANCE":~/scratch_llm/runs/d20/ local:./d20_ckpts/   # or rclone to a bucket
   ```
2. **Loss-divergence kill-switch `[FACT, shipped]`.** `train.py:363-380` all-reduces a non-finite
   flag across ranks every `log_every` and raises loud on any rank's NaN/inf — a divergence one rank
   sees stops all eight (no silent-NaN burn). Add a **slope check** as a human tripwire: if val CE
   (rank-0 `eval_every`) stops decreasing over ~1k steps, kill — don't pay out a plateaued run.
3. **$/token projection vs the $100 cap `[INFERENCE]`.** Continuously: `cost_so_far =
   hours_elapsed × $24`; `projected = cost_so_far × (18,311 / step_now)`. If `projected > $90`,
   invoke the abort → d16. Budget: pretrain $58–79 leaves ~$20–40 headroom under $100 for CORE +
   re-rents.
4. **Reproducibility manifest** (write to `runs/d20/manifest.json` at launch): **git SHA**
   (`git rev-parse HEAD`), **all seeds** (`--seed`; note weight-init seed must be set before model
   construction — `train()`'s reseed covers data sampling only, `train.py:235`), **shard hashes**
   (`sha256sum data/fineweb_edu_10b/*.bin`), **torch/CUDA/driver versions**
   (`torch.__version__`, `torch.version.cuda`, `nvidia-smi --query-gpu=driver_version`), the
   **node fingerprint** (`nvidia-smi -L`, `nvidia-smi topo -m`, the measured nccl busbw), and the
   **exact launch command + `<P5_SWEPT_MUON_LR>`**. This is what makes the CORE number defensible.

---

## Step 5 — evaluate: CORE + val_bpb (~20–40 min)

On the trained checkpoint (`runs/d20/pretrain.pt` — config-carrying; `build_model_from_checkpoint`
reconstructs the exact `ModelConfig`). Score the full DCLM CORE 22-task suite, **decontaminated**:

```bash
python bench/core_eval.py --ckpt runs/d20/pretrain.pt --data-dir data/fineweb_edu_10b \
  --device cuda --batch-size 32 --out runs/d20/core_eval.json
```

- **Predicted `[INFERENCE, pre-registered]`:** **CORE 0.19–0.22**. The CLI prints per-task accuracy +
  centered accuracy + the CORE mean, and a paste-ready ledger row (`_ledger_row`, anchors baked in:
  nanochat d20 **0.2219** · GPT-2 XL 0.2565).
- **How to report it `[FACT]`:** as a **CORE-vs-FLOPs point on nanochat's published curve** — we spent
  **C ≈ 2.77e19 = 73% of the anchor's 3.77e19** (86% of its N), so a number *below* 0.2219 at *lower
  compute* is a **successful, honest replication**, not a miss. **Never** headline it as
  depth-matched, and **never** compare per-token loss across vocabs — use **val bits-per-byte** (the
  vocab-independent metric) for the loss comparison.
- **KILL `[FACT, pre-registered]`:** CORE **<0.15** ⇒ a *stack bug*, not a sizing shortfall
  (0.15–0.19 would be a sizing/recipe discussion; <0.15 means something is broken — investigate the
  tokenizer/BOS handling, the checkpoint config round-trip, or the decontamination before believing it).
- **BOS handling `[FACT]`:** `core_eval` prepends the `<|eot|>` doc-separator as the BOS analog if the
  tokenizer carries it (`core_eval.py:73-77`) — nanochat prepends `<|bos|>`; the delta is documented
  in `core_suite.py`. The P5 validation-against-a-public-checkpoint pins that this is scored right.
- Also record **val_bpb** (the report card's vocab-independent honesty metric) for the CORE-vs-FLOPs
  point.

---

## Step 6 — teardown + release surface (A9 hand-off) (~15 min)

1. **Consolidate the model** — ensure `runs/d20/pretrain.pt` is the final config-carrying checkpoint
   (model + `ModelConfig`; optimizer-free is fine for release).
2. **Sync everything off-box BEFORE destroy** (GPU billing runs until `destroy`; storage bills from
   `create`):
   ```bash
   vastai copy "$INSTANCE":~/scratch_llm/runs/d20/ local:./d20_final/   # ckpt, core_eval.json, manifest, loss log
   vastai copy "$INSTANCE":~/scratch_llm/data/fineweb_edu_10b/tokenizer.json local:./d20_final/
   ```
3. **Destroy the instance** (stops the meter):
   ```bash
   vastai destroy instance "$INSTANCE" -y
   vastai show instances --raw | jq -r '.[].id'      # confirm it's gone
   ```
4. **A9 release surface (the portfolio artifact):** assemble a **model card** (config, data provenance
   + the decontamination statement + overlap rate, the CORE-vs-FLOPs-point table vs the 0.2219 anchor,
   val_bpb, intended use, license) + a reproducible bundle (weights + tokenizer.json + config + the
   manifest + a sample transcript). Do **not** publish it as an impersonation of nanochat — it is
   *our* independent replication point.
5. **Advance the ledger:** fill the `measured` column of the `bench/RESULTS.md` §*$100 d20* scorecard
   (predicted-vs-measured, honest either way — a KILL is a result), append the CORE row the CLI
   printed, advance the TASKSPEC `Next-node:` marker and STATUS, and update the auto-memory. `git add -p`
   the ledger/docs (never `-A` — shared checkout), commit, push, confirm remote CI green.

---

## Time + cost

| Step | Wall | On the meter? | Notes |
|---|---|---|---|
| 0 provision + P6 gate | ~20–40 min | yes | nccl busbw ≥ 350 GB/s or destroy+re-rent |
| 1 stage 10B corpus | **hours `[INFERENCE]`** | **NO — off-clock** | pre-stage the night before / cheap CPU box; `vastai copy` on |
| 2 recipe re-validate (sm90) | ~10 min | yes | bf16+compile NaN check; fall back to bf16-eager if it NaNs |
| 2.5 P5.5 probe gate (ADR-0020) | ~30–45 min | yes | 3 LR arms × 916 steps; winner lr = Step 3's `--lr`; ≈$9–12 |
| 3 distributed pretrain | **2.4–3.3 h `[INFERENCE]`** | yes | **$58–79**; the P5 d12 step-time is the go/no-go |
| 4 guardrails | (continuous) | yes | ckpt sync, NaN guard, $/token projection, manifest |
| 5 CORE + val_bpb | ~20–40 min | yes | the headline number |
| 6 teardown + release | ~15 min | partial | sync off-box BEFORE destroy |
| **total (paid)** | **~3.5–4.5 h `[INFERENCE]`** | | **~$58–79 pretrain + eval/overhead; hard cap $100; abort at $90 cumulative → d16** |

**Provenance.** Numbers from `bench/RESULTS.md` §*Pre-registration: the $100 d20 run (2026-07-16)* +
§*Decision — Muon+AdamW ADOPTED*, `docs/FRONTIER_2026_TASKSPEC.md` §A7/A8 + the d20 gate P1–P6,
`docs/adr/ADR-0018` §C. Param count, `use_sdpa` default, the missing `init_process_group`, the
rank-0-shard checkpoint, and `load_dataset_tokens`' in-RAM concat were **verified in code at HEAD**
while writing this runbook (the §0.5 `[FACT]` gaps). The four gaps are the P5 deliverables that turn
§3 from a plan into a press-play command.
