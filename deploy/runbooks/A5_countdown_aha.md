# Runbook — A5 R1-Zero "aha" on Countdown (Qwen2.5-1.5B via `grpo_train_loop`)

> ## 🖥️ RENTAL-DEFERRED · exec-spec node **W10**
> Self-contained. A fresh agent on a freshly-rented Vast.ai box executes this top-to-bottom with
> no other context. It rents one GPU, boots the repo, drives a **real** Qwen2.5-1.5B through the
> **from-scratch** `scratch_llm.algos.grpo.grpo_train_loop` on the Countdown env, and looks for the
> R1-Zero "aha" (the policy learns to *reason longer* and *solve* a verifiable puzzle). Budget
> **~$30–100**. Do the **free CPU pre-check** (below) first — it costs nothing and catches every
> wiring bug before you rent.

---

## 0. Purpose — what this proves, and why it is the capstone

This is the **engine-validation** run for the whole A5 RL stack. The `grpo_train_loop` is already
proven to *learn* on a toy single-token env in CI (`tests/test_grpo_algos.py`, CPU, deterministic).
What CI **cannot** prove is that the same loop drives a *real* language model to acquire a *real*
reasoning skill. That is what this run establishes:

- **The claim under test:** `scratch_llm`'s own GRPO/Dr.GRPO optimizer + verifiable grader + rollout
  seam, wired to a stock HF Qwen2.5-1.5B, reproduces the well-known **R1-Zero Countdown result**
  (Jiayi Pan's *TinyZero*, DeepSeek-R1-Zero's mechanism): with *no SFT*, purely from a rule-based
  verifiable reward, the base model learns the output contract, then learns to **search / reason /
  self-correct** in the `<think>` block, and its Countdown solve-rate climbs from ≈0.
- **The discriminator (why 1.5B, not 0.5B):** the "aha" is an **emergent capability of scale**. At
  **Qwen2.5-1.5B** it emerges; at **Qwen2.5-0.5B** it does **not** (format is learnable, but the
  answer-reward stays near 0 — the small model never develops the reasoning that Countdown needs).
  Running 0.5B as a **negative control** is the cleanest proof the *loop* is correct and the
  *capability* is what's scale-gated — not a bug in our code. This is the interview-grade framing:
  "a green loop that learns a toy is necessary but not sufficient; the artifact is a real model
  acquiring a real skill, and the negative control rules out the trivial explanation."
- **Not a leaderboard.** We are not chasing SOTA Countdown accuracy or a throughput number. Success
  = a **clear upward inflection in answer-reward** on 1.5B with a **flat** negative control on 0.5B,
  with the full discipline-#4 RL logs to interpret it.

## 1. Which `scratch_llm` code this exercises (real paths)

Every one of these is *your* from-scratch implementation — the run is their integration test on a
real model:

| Path | Role in this run |
|---|---|
| `src/scratch_llm/algos/grpo.py` | **The loop under test.** `grpo_train_loop` (Alg. 3: rollout → grade → group-normalize advantage → microbatch update → mandatory logging); `compute_group_normalized_rewards`, `grpo_microbatch_train_step`, the Dr.GRPO toggles. |
| `src/scratch_llm/envs/countdown.py` | **The task + grader.** `CountdownEnv` (seeded, solvable-by-construction pool), the safe `ast`-based expression evaluator, `grade_countdown_response` (format-gate × answer). Accepts a `codec` — we plug the **Qwen tokenizer** in via the documented `TextCodec` seam. |
| `src/scratch_llm/rewards/r1_zero.py` | **The verifiable reward.** The r1-zero prompt template (`render_r1_zero_prompt`, ends in `Assistant: <think>`) and the strict `</think> <answer>…</answer>` format gate the env composes. |
| `src/scratch_llm/algos/sft.py` | `get_response_log_probs` (the grad-bearing policy log-probs — its docstring explicitly supports **HF causal LMs** that return `.logits`), `masked_mean` / `masked_normalize` (GRPO vs Dr.GRPO aggregation), `compute_entropy`. |
| `src/scratch_llm/utils/monitors.py` | **Discipline #4 logging.** `build_snapshot` / `mean_kl` / `importance_ratios` → entropy, `KL(cur‖ref)`, `KL(cur‖old)`, IS-ratio mean+ESS, reward + length-by-correctness stats. |
| `src/scratch_llm/rollout/types.py` | The `Rollout` contract our custom HF sampler emits (prompt/response ids + per-token log π + stop reason). |
| `src/scratch_llm/optim.py` | `AdamW` (decoupled decay, β₂=0.95 LM default) — the policy optimizer, exercising A1 code too. |

**One honest engineering note you must understand before running** (this is the "next unmodeled
constraint" the plan asks you to name): the committed `grpo_train_loop` was written for the
**CPU** from-scratch model. It builds `input_ids` / `response_mask` / `advantages` **on CPU** in
`_collate_rollouts` and **never moves them to a device**, and its logging path calls `.numpy()` on
the log-prob rows. So you **cannot** hand it a GPU model directly (device-mismatch, and `.numpy()`
fails on CUDA tensors). The bridge — a **thin CPU-boundary adapter** that runs the heavy forward on
GPU and returns logits to CPU (the `.to('cpu')` copy is differentiable, so grads still flow to the
GPU params) — is in the driver script in §7. **We do not edit committed source.** The two
consequences of this bridge (a full-vocab logits round-trip to host each forward; three model copies
live from the loop's `deepcopy` of `π_ref`/`π_old`) are the dominant costs and set the GPU tier and
the predicted throughput. They are called out again where they bite.

---

## 2. FREE pre-check (do this BEFORE you rent — $0)

The loop's composition and learning are provable on CPU in seconds. Run these on **any** box
(including this one) and only rent if they are green:

```bash
cd /workspace/scratch_llm && source .venv/bin/activate
# The toy end-to-end: grpo_train_loop makes mean reward strictly RISE on a learnable env,
# entropy falls, and the mandatory RL logs are finite every step (~12 s, deterministic):
pytest tests/test_grpo_algos.py -q
# The two that matter most here:
pytest "tests/test_grpo_algos.py::test_grpo_train_loop_reward_strictly_rises" \
       "tests/test_grpo_algos.py::test_grpo_train_loop_runs_on_real_countdown_env" -q
```

- `test_grpo_train_loop_reward_strictly_rises` — proves the loop **learns** (E[r] 0.186 → 0.666 on
  the toy, seed 0) with the exact Dr.GRPO config this run uses. If this is red, **do not rent** —
  the optimizer is broken, no GPU will fix it.
- `test_grpo_train_loop_runs_on_real_countdown_env` — proves the loop **composes with the real
  byte-level `CountdownEnv`** (rollout → grade → advantage → update → log) and that entropy at init
  ≈ `log(vocab)` (loss-at-init sanity). This is the exact wiring the GPU run scales up — only the
  model, the codec (byte → Qwen tokenizer), and the device change.

If both are green, the only things the rental adds are a real tokenizer, a real model, and a GPU.
**That is the whole point of doing the pre-check: you are paying only for scale, not for debugging.**

---

## 3. Rent the GPU (tier + why)

**Tier: 1× H100 80 GB (SXM preferred), or 1× A100 80 GB.** Why 80 GB and why one GPU:

- **Why one GPU:** the loop is single-process; there is no data/model parallel here. More GPUs buy
  nothing.
- **Why 80 GB (not 40/24):** peak VRAM `[INFERENCE]` ≈ **25–40 GB** — Qwen2.5-1.5B in bf16 (~3.1 GB)
  **× three copies** (policy + the loop's `deepcopy`'d frozen `π_ref` and per-step `π_old`) ≈ 9.4 GB,
  + AdamW moments (~6–12 GB) + grads (~3 GB) + the group-batched `generate` KV-cache + forward
  activations. 40 GB is *feasible* with `--group-size 4 --microbatch 1 --max-new 200`; 80 GB gives
  the headroom to keep `group_size=8` (which you want for reward variance). **24 GB will OOM** — do
  not attempt the 1.5B run there (the standing dev box is 25 GB Blackwell; this is exactly the
  "rent for what this card can't do" case).
- The full-vocab log-softmax in the logging path (`_log_step`) materializes `(M, V)` fp32 rows with
  Qwen's **V≈151 936** vocab; at our small `n_tasks × group_size × max_new` it is a few GB on the
  **host** (we return logits to CPU), not VRAM — keep `n_tasks`/`group_size`/`max_new` modest so the
  host copy and its transfer stay cheap.

```bash
# 0) one-time auth on the box (skip if `vastai show user` already works)
vastai set api-key <YOUR_VAST_API_KEY>          # https://console.vast.ai/manage-keys/
vastai show user                                 # verify auth + credit balance
vastai create ssh-key ~/.ssh/id_ed25519.pub      # register your pubkey BEFORE create

# 1) query the LIVE marketplace — never hardcode an offer id; pick cheapest that fits.
#    cuda_max_good>=12.8 so the sm90/sm100 wheels + any Blackwell host work; verified+rentable only.
vastai search offers --raw -o 'dph_total' \
  'gpu_name=H100_SXM num_gpus=1 gpu_ram>=80 cuda_max_good>=12.8 \
   reliability>0.97 disk_space>=120 inet_down>=300 verified=true rentable=true dph_total<3.0' \
  | jq -r '.[0] | "offer \(.id)  \(.gpu_name)  $\(.dph_total)/hr  \(.geolocation)  rel=\(.reliability)"'
# If none: swap gpu_name=A100_SXM4 (also 80 GB), or raise dph_total<, or drop to gpu_name=A100_PCIE.

OFFER=$(vastai search offers --raw -o 'dph_total' \
  'gpu_name=H100_SXM num_gpus=1 gpu_ram>=80 cuda_max_good>=12.8 reliability>0.97 \
   disk_space>=120 inet_down>=300 verified=true rentable=true dph_total<3.0' | jq -r '.[0].id')

# 2) create it. The vastai/pytorch image gives CUDA+torch; we rebuild our exact env in §5 anyway.
vastai create instance "$OFFER" \
  --image vastai/pytorch:@vastai-automatic-tag \
  --disk 120 --ssh --direct --label countdown-aha

# 3) poll to running (add a timeout branch: if actual_status becomes exited/unknown/offline it will
#    NEVER reach running — destroy and retry another offer, per the vastai skill).
INST=<new_contract_id_from_create>
until [ "$(vastai show instance $INST --raw | jq -r .actual_status)" = "running" ]; do
  st=$(vastai show instance $INST --raw | jq -r .actual_status)
  echo "status=$st"; case "$st" in exited|unknown|offline) echo "DEAD — destroy+retry"; exit 1;; esac
  sleep 10
done
vastai ssh-url $INST                              # connect string
```

> **Cheaper (optional):** add `--type bid` to the search and pass `--bid_price <floor>` to
> `create` for interruptible pricing (often ~40–60% off). If outbid, the instance **stops** (disk
> preserved) — your `metrics.jsonl` + last checkpoint survive; resume with `vastai start instance`.
> Only worth it if you checkpoint (the driver does, every `--chunk` steps).

---

## 4. Bootstrap the pod

SSH in, clone the repo, and run the one-command bootstrap (it builds the pinned env, re-links the
green-CI hook, restores auto-memory, and prints the current node):

```bash
ssh <paste vastai ssh-url output>
cd /workspace 2>/dev/null || cd ~
git clone https://github.com/<owner>/scratch_llm.git   # or `gh repo clone` if gh is authed
cd scratch_llm
bash scripts/bootstrap-pod.sh
```

`scripts/bootstrap-pod.sh` (read it — it is short) does: `uv venv .venv` + `uv pip install -e
".[dev]"` + the **sm120/cu130 torch** wheel this repo was measured on. **On this rented H100/A100
(sm90), that specific `--index-url .../cu130` line is for Blackwell — you want the default CUDA
wheel instead.** After bootstrap, correct the torch build for the rented GPU:

```bash
source .venv/bin/activate
# Replace the sm120/cu130 wheel with the default build for this H100/A100 (let uv pick current CUDA):
uv pip install --python .venv/bin/python --reinstall torch
python -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0))"
```

## 5. Env + dependency setup (model runtime)

Core deps come from `".[dev]"`. This run additionally needs `transformers` + `accelerate` to load
Qwen (the standing dev box already has `transformers` 5.x; a fresh pod may not):

```bash
source /workspace/scratch_llm/.venv/bin/activate
uv pip install --python .venv/bin/python transformers accelerate
python -c "import transformers, torch; print('transformers', transformers.__version__)"
```

> **Do NOT `pip install` concurrently with any other agent on a shared box** (races corrupt the
> env). If `transformers`/`accelerate` are already importable, skip this step and record the
> versions you found. No `hf_transfer`/`flash-attn` needed — we use eager attention and stock
> `generate`.

## 6. Model + data acquisition

- **Data = generated in-process, no download.** `CountdownEnv` builds a **seeded, solvable-by-
  construction** task pool at init (`generate_countdown_tasks`); every task carries a witness
  solution and the same seed reproduces the pool bit-for-bit. Nothing to fetch, nothing to clean.
- **Model = one HF download** (Qwen2.5-1.5B **base**, not Instruct — R1-Zero starts from the base
  model; the point is that RL *alone* induces reasoning):

```bash
export HF_HOME=${WORKSPACE:-/workspace}/.hf_home        # cache on the big disk
# (Qwen2.5 is ungated — no HF token needed. Set HF_TOKEN only if you mirror a gated copy.)
python - <<'PY'
from huggingface_hub import snapshot_download
for m in ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B"):   # 0.5B = the negative control
    p = snapshot_download(m); print("cached", m, "->", p)
PY
```

---

## 7. The driver script (write it on the pod — not into `src/`)

The committed loop is CPU-collation-bound (§1). Bridge it to the GPU model with this driver. Write
it to a **scratch path** (never into `src/`, tests, or anything committed):

```bash
mkdir -p /workspace/countdown_aha && cat > /workspace/countdown_aha/driver.py <<'PY'
#!/usr/bin/env python
"""Countdown R1-Zero 'aha' — Qwen2.5 through the from-scratch grpo_train_loop (engine-validation).

Bridges the committed CPU-collation loop to a GPU HF model via a differentiable CPU-boundary
adapter, plugs the Qwen tokenizer into CountdownEnv as its TextCodec, and drives grpo_train_loop
in chunks (checkpoint + stream metrics between calls). Writes metrics.jsonl + ckpt-* under --out.
Run with the repo's .venv active, cwd = /workspace/scratch_llm."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from scratch_llm.envs.countdown import CountdownEnv
from scratch_llm.algos.grpo import grpo_train_loop
from scratch_llm.optim import AdamW           # from-scratch; torch.optim.AdamW is a drop-in fallback
from scratch_llm.rollout.types import Rollout

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")   # 0.5B for the negative control
p.add_argument("--steps", type=int, default=400)
p.add_argument("--n-tasks", type=int, default=16)        # prompts per GRPO step (= env pool size)
p.add_argument("--group-size", type=int, default=8)      # rollouts per prompt → reward variance
p.add_argument("--n-numbers", type=int, default=4)       # Countdown difficulty knob
p.add_argument("--max-value", type=int, default=12)
p.add_argument("--max-new", type=int, default=256)       # completion budget
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--lr", type=float, default=1e-6)         # RL LRs are tiny; 1e-6 is a safe start
p.add_argument("--microbatch", type=int, default=2)      # bounds forward/backward activation memory
p.add_argument("--chunk", type=int, default=20)          # steps per outer call (checkpoint cadence)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="/workspace/countdown_aha/run")
args = p.parse_args()

DEV = "cuda"
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

# --- Qwen tokenizer AS the env codec (the documented "swap in a real tokenizer" seam) ------------
tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
class QwenCodec:  # structural TextCodec: encode(text)->list[int], decode(ids)->str
    def encode(self, text): return tok.encode(text, add_special_tokens=False)
    def decode(self, ids):  return tok.decode(list(ids), skip_special_tokens=True)
env = CountdownEnv(n_tasks=args.n_tasks, seed=args.seed,
                   n_numbers=args.n_numbers, max_value=args.max_value, codec=QwenCodec())

# --- model + differentiable CPU-boundary adapter (the loop collates on CPU; §1) -------------------
hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(DEV)
hf.config.use_cache = False
class CPUBoundaryPolicy(nn.Module):
    """Heavy forward on GPU; return logits to CPU so the loop's CPU-side loss/KL math composes.
    The .to('cpu') copy is differentiable → grads flow back to the GPU params the optimizer holds."""
    def __init__(self, m): super().__init__(); self.model = m
    def forward(self, input_ids):
        return self.model(input_ids.to(DEV)).logits.float().cpu()
policy = CPUBoundaryPolicy(hf)
optimizer = AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

# --- HF batched-group sampler: SampleFn (task, group_size) -> list[Rollout] -----------------------
EOS = tok.eos_token_id
@torch.no_grad()
def sample_fn(task, group_size):
    hf.eval(); hf.config.use_cache = True
    prompt = torch.tensor([list(task.prompt_ids)], device=DEV).repeat(group_size, 1)
    gen = hf.generate(prompt, do_sample=True, temperature=args.temperature, top_p=1.0,
                      max_new_tokens=args.max_new, pad_token_id=tok.pad_token_id, eos_token_id=EOS,
                      return_dict_in_generate=True, output_scores=True)
    hf.config.use_cache = False; hf.train()
    new = gen.sequences[:, prompt.shape[1]:]                    # (G, T_new)
    logp = torch.log_softmax(torch.stack(gen.scores, 1).float() / args.temperature, dim=-1)
    rolls = []
    for g in range(group_size):
        ids = new[g].tolist()
        if EOS in ids: ids = ids[: ids.index(EOS) + 1]          # trim at first EOS (inclusive)
        n = len(ids)
        lp = logp[g, torch.arange(n), torch.tensor(ids)].tolist()
        stop = "stop" if ids and ids[-1] == EOS else "length"
        rolls.append(Rollout(tuple(task.prompt_ids), tuple(ids), tuple(lp), stop))
    return rolls

# --- drive in chunks: checkpoint + stream metrics between calls -----------------------------------
mpath = out / "metrics.jsonl"; done = 0; t0 = time.time()
print(f"model={args.model}  steps={args.steps}  n_tasks={args.n_tasks}  G={args.group_size}  "
      f"n_numbers={args.n_numbers}  lr={args.lr}  microbatch={args.microbatch}", flush=True)
with mpath.open("a") as f:
    while done < args.steps:
        k = min(args.chunk, args.steps - done)
        hist = grpo_train_loop(
            policy, env, sample_fn, optimizer,
            n_grpo_steps=k, group_size=args.group_size,
            normalize_by_std=False, length_normalization="constant",
            normalize_constant=float(args.max_new),          # Dr.GRPO: fixed-length normalizer
            loss_type="reinforce_with_baseline", epochs_per_rollout_batch=1,  # on-policy
            microbatch_size=args.microbatch, max_grad_norm=1.0,
            pad_id=tok.pad_token_id, seed=args.seed + done,
        )
        for h in hist:
            row = dict(step=done + h.step, wall_s=round(time.time() - t0, 1),
                       reward=h.mean_reward, format=h.mean_format_reward, answer=h.mean_answer_reward,
                       loss=h.mean_loss, entropy=h.entropy, kl_ref=h.kl_current_ref,
                       kl_old=h.kl_current_old, is_mean=h.is_ratio_mean, is_ess=h.is_ratio_ess,
                       len_correct=h.length_correct_mean, len_incorrect=h.length_incorrect_mean,
                       len_p90=h.snapshot.length_p90, len_max=h.snapshot.length_max)
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"step {row['step']:4d}  R={row['reward']:.3f}  fmt={row['format']:.3f}  "
                  f"ans={row['answer']:.3f}  ent={row['entropy']:.2f}  klref={row['kl_ref']:.3f}  "
                  f"len_ok={row['len_correct']}", flush=True)
        done += k
        hf.save_pretrained(out / f"ckpt-{done}"); tok.save_pretrained(out / f"ckpt-{done}")
print(f"DONE {done} steps in {(time.time() - t0) / 3600:.2f} h", flush=True)
PY
```

> **Caveat on chunked driving (know this before reading the logs):** `grpo_train_loop` re-snapshots
> its frozen `π_ref` (via `deepcopy`) at the **start of each call**. Driving in `--chunk 20` steps
> therefore **resets `π_ref` every 20 steps**, so `kl_ref` measures drift *within* a chunk, not from
> the original base model. For checkpoint safety on a multi-hour run this is the right trade. If you
> want a **true fixed-`π_ref`** KL curve for the writeup, do a single call: pass `--chunk` equal to
> `--steps` (you then get one checkpoint at the end — run a short chunked pass first to de-risk).

## 8. Launch

```bash
cd /workspace/scratch_llm && source .venv/bin/activate
export HF_HOME=${WORKSPACE:-/workspace}/.hf_home
# Run detached under tmux/nohup so an SSH drop doesn't kill it; watch the log live.
nohup python /workspace/countdown_aha/driver.py \
  --model Qwen/Qwen2.5-1.5B --steps 400 --n-tasks 16 --group-size 8 \
  --n-numbers 4 --max-value 12 --max-new 256 --lr 1e-6 --microbatch 2 --chunk 20 \
  --out /workspace/countdown_aha/qwen1p5b > /workspace/countdown_aha/qwen1p5b.log 2>&1 &
watch -n 30 "tail -n 20 /workspace/countdown_aha/qwen1p5b.log; echo; nvidia-smi | sed -n '9,12p'"
```

**Smoke first (≈$1–3, ~10–20 min):** before committing to 400 steps, run **20 steps** to confirm
memory fits, format-reward starts moving, and the per-step wall-time so you can price the full run:

```bash
python /workspace/countdown_aha/driver.py --model Qwen/Qwen2.5-1.5B --steps 20 --chunk 20 \
  --out /workspace/countdown_aha/smoke
```

**Negative control (run after the 1.5B shows the inflection, or in parallel if budget allows):**

```bash
python /workspace/countdown_aha/driver.py --model Qwen/Qwen2.5-0.5B --steps 400 --n-tasks 16 \
  --group-size 8 --n-numbers 4 --max-value 12 --lr 1e-6 --microbatch 2 --chunk 20 \
  --out /workspace/countdown_aha/qwen0p5b
```

---

## 9. Predict-before-run (pre-registration) + kill criteria

State these **out loud in the writeup before the full run** — they are the debugging anchor and the
graded skill. All `[INFERENCE]` (from the R1-Zero / TinyZero literature + the toy-loop behavior);
mark them measured only after the run.

| # | Metric (`GRPOStepMetrics` field) | Predicted trajectory (Qwen2.5-1.5B) | Negative control (0.5B) |
|---|---|---|---|
| P1 | `mean_format_reward` | **Fast:** ≥0.5 by ~step 20, ≥0.9 by ~step 50. The `</think> <answer>…</answer>` contract is easy to learn. | Also ≥0.9 (format is not scale-gated) |
| P2 | `mean_answer_reward` (Countdown solve-rate) | **The aha:** ≈0 through the format-learning phase, then a **clear upward inflection**, reaching **~0.3–0.6 by ~step 300–500**. | **Stays ≈0** (never develops the reasoning) |
| P3 | `mean_reward` (= format×answer) | Tracks P2 (0 → ~0.3–0.6) once format is solved. | Near 0 |
| P4 | `length_correct_mean` vs `_incorrect` | Completion length of *correct* rollouts **grows** as the model learns to "think longer" — the reasoning-emergence tell. | No sustained growth |
| P5 | `entropy` | Falls **modestly** (exploration must persist); **not** a collapse to ~0. A hard collapse early = mode-collapse, not learning. | May fall without any answer gain |
| P6 | `kl_current_ref` | Grows steadily (policy moves from base). Bounded — a spike = instability. | Grows but with no answer payoff |
| P7 | `is_ratio_ess`, `is_ratio_mean` | ESS ≈ 1.0, mean ≈ 1.0 (on-policy, `epochs_per_rollout_batch=1` → `π_old`≈`π_cur`). A drift means the sampler and scorer disagree — a bug. | same |

**Throughput / cost prediction `[INFERENCE]`:** the rollout is stock HF `generate` (no vLLM/paged
attention) + a full-vocab logits round-trip to host per forward — both deliberately un-optimized
(this is engine-validation, not a throughput run). Expect **~1–4 min/step** on one H100 at
`n_tasks=16 × group_size=8 × max_new=256`. 400 steps ⇒ **~7–27 h ⇒ ~$15–75** at ~$2/hr. Measure the
real per-step time from the smoke run and reprice **before** committing.

### Kill criteria (stop and diagnose — don't burn budget)

- **K1 — format never moves:** `mean_format_reward` < 0.5 by step 50 ⇒ tokenization/prompt/format-
  gate wiring bug (check the r1-zero prompt renders, the Qwen codec round-trips, the `</think>
  <answer>` separator matches). **Kill immediately** — no GPU fixes this.
- **K2 — answer flat with format solved:** `mean_answer_reward` still ≈0 at **step ~300** while
  `format`≈1 ⇒ either too-small model (expected on 0.5B — that's the *control*, not a failure), LR
  wrong, or the task is too hard/easy. **Kill by ~$40 spend**; try `--lr 3e-6`, or `--n-numbers 3`
  to lower difficulty, before re-committing.
- **K3 — variance collapse:** reward std ≈0 across groups (every group all-right or all-wrong) ⇒
  zero advantage ⇒ no gradient (the group-variance limitation GRPO can't escape). Fix **difficulty**
  (`--n-numbers` / `--max-value`) or raise `--temperature` for exploration.
- **K4 — instability:** `entropy` collapses to ~0 early, or `kl_current_ref` spikes, or `is_ratio_ess`
  falls well below 1 ⇒ LR too high / off-policy drift. Lower `--lr`, keep `epochs_per_rollout_batch=1`.
- **K5 — OOM:** cut in order `--group-size 4` → `--microbatch 1` → `--max-new 200` → `--n-tasks 8`.
- **K6 — budget cap:** if `mean_answer_reward` shows **no** upward inflection by the spend ceiling
  you set (default **$100**), stop. The negative result (loop correct, capability absent at this
  scale/config) is itself a recordable finding — log it honestly.

**Success = P2 inflects up on 1.5B while the 0.5B control stays flat, with P1/P4/P5/P6 consistent.**

## 10. Logging checklist (discipline #4 — an RL run without these is uninterpretable)

The loop wires all of these into every `GRPOStepMetrics` + its `MonitorSnapshot`; the driver streams
them to `metrics.jsonl`. Confirm each column is present and finite as you watch:

- [ ] `step`, `wall_s`
- [ ] `reward` (`mean_reward`), `format` (`mean_format_reward`), `answer` (`mean_answer_reward`)
- [ ] `loss` (`mean_loss`)
- [ ] `entropy` — response-token entropy (exploration / collapse signal)
- [ ] `kl_ref` (`KL(current‖ref)`) **and** `kl_old` (`KL(current‖old)`) — **logged separately** (never
      summed); at step 0 of a chunk they coincide (π_old = π_ref = start-of-chunk policy)
- [ ] `is_mean`, `is_ess` — importance-ratio mean + normalized ESS (off-policy drift tripwire)
- [ ] `len_correct` (`length_correct_mean`) **vs** `len_incorrect` — length **by correctness** (the
      verbosity-reward-hacking / "think longer" tell), plus `len_p90`, `len_max` from the snapshot
- [ ] reward distribution — `reward_mean`/std/min/max live on `h.snapshot`

Plot `answer` and `len_correct` vs `step` from `metrics.jsonl` — the "aha" is the moment `answer`
lifts off 0 *and* `len_correct` starts climbing. Also eyeball a few full completions from a late
checkpoint (load `ckpt-400` and generate) to *see* the reasoning in the `<think>` block — that
qualitative read is what makes the artifact interview-ready.

## 11. Where results go

- **Raw:** keep `metrics.jsonl` + the final `ckpt-*` and `qwen1p5b.log` (and the 0.5B control). Copy
  them off the pod **before teardown** — the container filesystem is wiped on destroy:
  ```bash
  # from your laptop / the standing box:
  vastai copy <INST>:/workspace/countdown_aha/qwen1p5b/metrics.jsonl local:./countdown_aha/
  vastai copy <INST>:/workspace/countdown_aha/qwen1p5b.log            local:./countdown_aha/
  ```
- **Ledger (the durable record):** append a **W10 · A5 Countdown "aha"** entry to
  `bench/RESULTS.md` under the existing **`## Main track (CS336 A2→A5)`** section (next to the
  `### W8c · A5 — GRPO / Dr.GRPO engine` entry). Use the same table shape already in that file:
  `| date | node / artifact | metric | predicted (pre-registered) | measured | note |`, one row per
  P1–P7, plus the 0.5B negative-control row. State the predicted-vs-measured verdict and label every
  number `[FACT]` (measured) vs `[INFERENCE]`. This edit is **additive** — pull-rebase before commit
  (the perf front shares this file), never `git add -A`.
- **Status:** note the run outcome in `docs/STATUS.md` A5 section (W11 closeout folds it in).

## 12. Teardown (stop the billing)

```bash
# after results are copied OFF the box:
vastai destroy instance <INST> -y      # deletes instance + disk; stops ALL billing
vastai show instances                  # confirm it's gone
```

> `stop` (not `destroy`) preserves the disk (and checkpoints) but **keeps charging storage** — only
> use it if you intend to resume within hours. For a finished run, **`destroy`**.

---

### Appendix — rough $ cost

| Item | Estimate |
|---|---|
| H100 80 GB on-demand | ~$1.8–2.8/hr (A100 80 GB often ~$1.2–1.8/hr; spot ~40–60% less) |
| Smoke (20 steps) | ~10–20 min ⇒ **~$1–3** |
| Full 1.5B run (400 steps) | ~7–27 h ⇒ **~$15–75** (dominated by un-optimized HF `generate` rollouts) |
| 0.5B negative control (400 steps) | ~half the 1.5B wall-time ⇒ **~$8–35** |
| **Total (1.5B + control + smoke)** | **~$30–100** |

Keep it near the low end by: pricing the full run off the **measured** smoke per-step time, using
`--type bid` with checkpointing, running the control only after 1.5B shows the inflection, and
**killing on K1–K6** instead of letting a dead run bleed budget.
