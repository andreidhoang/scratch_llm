# Runbook — A5 reasoning-RL on Qwen2.5-Math-1.5B (MATH: baseline → SFT → EI → GRPO ≥25%)

> **Exec-spec node:** W10 (`docs/EXECUTION_SPEC_CS336_FINISH.md` → "Rental-deferred pack").
> **Status:** RENTAL-DEFERRED. The A5 *algorithms* are code-complete and CPU/toy-tested on the
> standing 24 GB sm120 box (W8a–c). The graded run needs a real model (Qwen2.5-Math-1.5B) at
> rollout-batch scale, which does not fit the full sweep on 24 GB — so it is written now, run on a
> rented H100. A **24 GB-local fallback** (§10) is included for a first end-to-end smoke on the
> standing card at $0 marginal cost.
>
> **Self-contained:** a fresh agent on a freshly-rented pod can execute this top to bottom. Every
> repo path below is real; read the module docstrings if a step is unclear.

---

## 0. What you are doing (and why it is graded)

Reproduce the CS336 A5 *main* deliverable on **Qwen2.5-Math-1.5B**:

1. **Zero-shot MATH baseline** — eval the base model with the r1-zero prompt, categorize by
   (format × answer) reward. This is the number every later stage must beat.
2. **SFT sweep** — supervised fine-tune on MATH solutions, data-size sweep `{128,256,512,1024,full}`.
3. **Expert Iteration (STaR) sweep** — sample `G`, keep verifiably-correct, SFT, repeat
   `n_ei_steps=5`; sweep `G∈{4,8,16}`.
4. **GRPO / Dr.GRPO** — the policy-gradient loop to **≥25% MATH** (the graded bar), then the
   **LR sweep** and the **baseline-vs-no-baseline ablation**.

The grade is a *reward curve that clears 25% MATH* plus the two ablations, all with the mandatory
RL logging (§8) present. "Implemented" ≠ "measured" — only a logged run counts (FOP-4).

### Which scratch_llm code this exercises (real paths)

| Stage | Repo module(s) exercised | CPU test that already pins it |
|---|---|---|
| Prompt + grader | `src/scratch_llm/rewards/r1_zero.py` (`render_r1_zero_prompt`, `r1_zero_reward_fn`) | `tests/test_rewards.py` |
| Env seam | `src/scratch_llm/envs/protocol.py` (`Task`/`Graded`/`VerifiableEnv`), `envs/gsm_math.py` (`GSMMathEnv`, codec-parametrized) | `tests/test_envs.py`, `tests/test_env_protocol.py` |
| SFT primitives | `src/scratch_llm/algos/sft.py` (`tokenize_prompt_and_output`, `get_response_log_probs`, `masked_mean`/`masked_normalize`, `sft_microbatch_train_step`) | `tests/test_sft_algos.py` |
| Expert Iteration | `src/scratch_llm/algos/expert_iteration.py` (`expert_iteration`, `collate_prompt_response_ids`) | `tests/test_expert_iteration.py` |
| GRPO / Dr.GRPO | `src/scratch_llm/algos/grpo.py` (`compute_group_normalized_rewards`, `compute_policy_gradient_loss`, `grpo_microbatch_train_step`, `grpo_train_loop`) | `tests/test_grpo_algos.py` |
| Mandatory logging | `src/scratch_llm/utils/monitors.py` (`build_snapshot`, `mean_kl`, `importance_ratios`, `normalized_ess`) | `tests/test_monitors.py` |

The **oracle grader** for real MATH answers is the official
`/workspace/lectures/assignment5-alignment/cs336_alignment/drgrpo_grader.py`
(`r1_zero_reward_fn`, `extract_boxed_answer`, `last_boxed_only_string`) — `math_verify`/sympy-backed,
which the repo's simpler `r1_zero.py` grader is **not** (see §6, this matters for MATH).

---

## 1. The honest gap — what is shipped vs what you wire on the pod

The repo ships the graded **primitives + a reference loop**, tested on CPU with a *from-scratch*
`TransformerLM` and a *byte-level* codec. To drive them on **Qwen (an HF model)** you wire three
small pieces of **throwaway harness** on the pod (NOT committed repo source — do not edit
`src/scratch_llm/`):

1. **HF rollout `SampleFn`** — `src/scratch_llm/rollout/local.py::LocalBackend` and
   `grpo.py::make_rollout_sampler` are hard-wired to the from-scratch model
   (`from scratch_llm.model import TransformerLM`, `KVCache`, `model.cfg.context_length`) and will
   **not** accept a Qwen `AutoModelForCausalLM`. You provide an HF-backed sampler that returns the
   same `Rollout(prompt_ids, response_ids, logprobs, stop_reason)` (`rollout/types.py`). The RL
   loops take `sample_fn` as a plug (Protocol/Callable), so this is the only generation glue needed.
2. **HF codec** — `GSMMathEnv(items=..., codec=...)` accepts any `encode/decode` object
   (`envs/countdown.py::TextCodec`). Pass a Qwen-tokenizer codec so `Task.prompt_ids` and `decode`
   use real tokens, not bytes.
3. **Official-grader env** — `GSMMathEnv.grade` calls the *repo* grader (string/numeric match).
   `expert_iteration` grades via `env.grade` (no `reward_fn` override), so for MATH you supply a
   tiny `VerifiableEnv` whose `grade` calls the **official** `r1_zero_reward_fn`. `grpo_train_loop`
   *does* take a `reward_fn=` override, so GRPO can use the official grader without a new env.

Why the `get_response_log_probs` / SFT / GRPO *update* code needs **no** change: it duck-types on
`out.logits if hasattr(out, "logits") else out` (`algos/sft.py:150`) — HF causal LMs return
`.logits`, the from-scratch model returns raw logits. The masking, advantage, loss, and monitor
paths are model-agnostic by construction. Only *generation* and *grading* need HF glue.

> The full glue (~120 lines) is given verbatim in §7. It lives on the pod under `~/a5_run/`, never
> in git.

---

## 2. GPU tier + why

| Tier | GPU | Why | Use |
|---|---|---|---|
| **Primary** | **1× H100 SXM/PCIe 80 GB** | Full sweep fits: 1.5B bf16 policy + AdamW fp32 state + `grpo_train_loop`'s 3× model-resident (ref + old + policy, §9) + full-vocab KL rows all live in 80 GB. sm90 runs the pinned `torch 2.12.1+cu130` wheel. | baseline · SFT sweep · EI sweep · GRPO ≥25% · ablations |
| Cheaper | 1× A100 80 GB | Same 80 GB headroom, ~30% slower, often cheaper. Fine if H100 offers are pricey. | same as H100 |
| Do-not | 1× 40–48 GB (A6000 / A100-40 / L40S) | Tight: 3× policy (§9-A) + AdamW leaves little for activations + the 151 k-vocab KL rows (§9-B). Possible only with a *tiny* rollout batch — no real speed win over the local fallback. | skip; use §10 locally instead |
| Local fallback | standing **RTX PRO 4000 Blackwell sm120, 24 GB** | Free (already rented for the perf front). Fits SFT-small + a GRPO *smoke*, not the full sweep. | §10 first-attempt only |

**Single GPU is correct here** — A5 is not a multi-GPU assignment. The bottleneck is rollout
generation throughput, not model-parallel training. (Faster rollouts = a vLLM/SGLang serving engine,
which is *not installed*; §7 uses plain HF `.generate`. Wiring vLLM is an optional 2–4× speedup, not
required, and would need `uv pip install vllm` — a deliberate, non-concurrent install.)

---

## 3. Rent the box (vastai)

Uses the `vastai` skill conventions (always `--raw`; register SSH key before create; poll status;
`-y` on destroy). One-time auth if the pod is fresh:

```bash
vastai set api-key <YOUR_KEY>                        # https://console.vast.ai/manage-keys/
vastai show user                                     # verify auth + credit balance
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)" # BEFORE create; skip if already registered
```

Search + pick the cheapest matching offer (never hardcode an offer id — the market moves):

```bash
# H100 80GB, single GPU, verified, direct SSH, enough disk for model+data+checkpoints.
vastai search offers \
  'gpu_name=H100_SXM num_gpus=1 gpu_ram>=80 reliability>0.98 cuda_max_good>=12.0 \
   direct_port_count>=1 disk_space>=120 inet_down>=300 dph_total<3.0 verified=true rentable=true' \
  -o 'dph_total' --raw | jq -r '.[0] | "\(.id)  $\(.dph_total)/hr  \(.gpu_name)  rel=\(.reliability)"'

OFFER_ID=<id-from-above>
vastai create instance "$OFFER_ID" \
  --image vastai/pytorch:@vastai-automatic-tag \
  --disk 120 --ssh --direct --label a5-qwen-math-rl --raw   # → {"new_contract": <INSTANCE_ID>}

INSTANCE_ID=<new_contract>
# Poll until running (add a timeout — exited/offline never reach running; storage bills from create):
for i in $(seq 1 60); do
  st=$(vastai show instance "$INSTANCE_ID" --raw | jq -r '.actual_status')
  echo "status=$st"; [ "$st" = running ] && break
  case "$st" in exited|offline|unknown) echo "BAD status; destroy+retry"; break;; esac
  sleep 10
done
vastai ssh-url "$INSTANCE_ID"                        # → ssh://root@HOST:PORT
```

**Cheaper option:** append `--type bid` to `search offers` to see `min_bid`, then create with
`--bid_price <floor>` (spot ~40% off; interruptible — checkpoint often, §8). Without `--bid_price`
after a bid search you are billed on-demand anyway (skill caveat).

Destroy the moment the run is done (stops all billing; disk is wiped):

```bash
vastai destroy instance "$INSTANCE_ID" -y
```

---

## 4. Bootstrap the pod

SSH in (`ssh -p PORT root@HOST`), then reconstitute the repo + env. The repo is **private**
(`andreidhoang/scratch_llm`); clone with a GitHub token.

```bash
# --- clone the repo (token used once, not persisted) ---
export GITHUB_TOKEN=ghp_...            # a token with read access, or run `gh auth login` first
git clone "https://${GITHUB_TOKEN}@github.com/andreidhoang/scratch_llm.git" /workspace/scratch_llm
cd /workspace/scratch_llm
git remote set-url origin https://github.com/andreidhoang/scratch_llm.git   # strip token from config

# --- one-command env (scripts/bootstrap-pod.sh — real, idempotent) ---
bash scripts/bootstrap-pod.sh
#   builds .venv (py3.11) · installs -e ".[dev]" · torch 2.12.1+cu130 (works on H100 sm90 too)
#   · relinks the green-CI hook · restores auto-memory · prints torch+CUDA + the green gate.
```

`bootstrap-pod.sh` installs only `[dev]`. The A5 run needs the `[gpu]` extra + the MATH grader deps
(NOT in `pyproject.toml`):

```bash
source /workspace/scratch_llm/.venv/bin/activate
uv pip install -e ".[gpu]"                    # transformers, datasets (declared in pyproject [gpu])
uv pip install math-verify accelerate         # official grader backend (sympy/antlr) + device_map
# The official grader also needs the scaffold importable:
uv pip install -e /workspace/lectures/assignment5-alignment      # provides cs336_alignment.*
python -c "import torch,transformers,datasets,math_verify; \
  from cs336_alignment.drgrpo_grader import r1_zero_reward_fn; \
  print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0)); \
  print('grader ok:', r1_zero_reward_fn('x </think> <answer> 4 </answer>', '4'))"
```

Expected last line: `grader ok: {'format_reward': 1.0, 'answer_reward': 1.0, 'reward': 1.0}`.

> If `/workspace/lectures/` is absent (SKIP_LECTURES pods), clone the A5 scaffold alone:
> `git clone --depth 1 https://github.com/stanford-cs336/assignment5-alignment.git /workspace/lectures/assignment5-alignment`.
> Verify the `.venv` torch sees the GPU **before** downloading anything — no CUDA ⇒ wrong wheel,
> stop and fix (do not run on CPU).

---

## 5. Acquire the model + the MATH data

**Model** — public, no token needed (cache lands under `HF_HOME` = `${WORKSPACE}/.hf_home`):

```bash
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B", torch_dtype=torch.bfloat16)
t = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
print("params(B):", sum(p.numel() for p in m.parameters())/1e9, "| vocab:", m.config.vocab_size)
PY
# Expect ~1.54B params, vocab 151936. (These two numbers drive the memory budget in §9.)
```

**MATH data** — the scaffold ships GSM8K but **not** MATH (large, gated upstream). Pull via
`datasets`. The MATH schema is `{"problem": str, "solution": str}` where the gold answer is inside
`\boxed{...}` in `solution`. Extract it with the official `extract_boxed_answer`.

```bash
python - <<'PY'
from datasets import load_dataset
from cs336_alignment.drgrpo_grader import extract_boxed_answer
import json, os, pathlib

# Primary mirror (7.5k train / 5k test, subject configs merged under 'all').
# If this repo id 404s, fall back to any equivalent mirror: "lighteval/MATH",
# "qwedsacf/competition_math", or the 500-problem "HuggingFaceH4/MATH-500" for a quick baseline.
ds = load_dataset("EleutherAI/hendrycks_math", "all")   # keys: 'train', 'test'
out = pathlib.Path.home() / "a5_run" / "data"; out.mkdir(parents=True, exist_ok=True)
for split in ("train", "test"):
    rows = []
    for ex in ds[split]:
        gt = extract_boxed_answer(ex["solution"])       # gold final answer, normalized-boxed
        if gt is None:                                   # skip the handful with no \boxed gold
            continue
        rows.append({"problem": ex["problem"], "answer": gt, "solution": ex["solution"]})
    (out / f"math_{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print(split, len(rows))
PY
# Expect ~7.5k train, ~5k test. The SFT sweep uses train solutions; baseline+eval use test.
```

For SFT you need `(prompt, response)` where the response is the r1-zero-formatted chain — build it
from `solution` (reasoning) + the boxed answer:
`"<reasoning...> </think> <answer> {answer} </answer>"`. The CS336 course ships a curated `sft.jsonl`;
lacking it, the boxed solutions above are a legitimate substitute (note this in the writeup).

---

## 6. Grader choice (do not get this wrong)

Two graders exist and they are **not** interchangeable for MATH:

- **`scratch_llm.rewards.r1_zero.r1_zero_reward_fn`** — snapshot-pinned to the official *format*
  gate + a *numeric/casefold* answer match. Correct for the unit tests, the toy Countdown/GSM envs,
  and integer-answer problems. **Too weak for MATH** (fractions, `\frac`, surds, `\pi`, sets — it
  will mark `\frac{1}{2}` ≠ `0.5` and score correct answers as wrong).
- **`cs336_alignment.drgrpo_grader.r1_zero_reward_fn(response, ground_truth, fast=True)`** —
  `math_verify`/sympy-backed, handles `\boxed`, LaTeX-equality, symbolic equality. **Use this for
  every MATH reward** (baseline, EI grading, GRPO reward). Same return dict
  `{"reward","format_reward","answer_reward"}` — a drop-in `RewardFn`.

Both share the identical format gate (`"</think> <answer>"` + `"</answer>"`) and the identical
r1-zero prompt (`rewards/r1_zero.py::R1_ZERO_PROMPT_TEMPLATE` is byte-for-byte the scaffold's
`prompts/r1_zero.prompt`), so the format-reward numbers are comparable across both.

---

## 7. The pod harness (throwaway — `~/a5_run/harness.py`, not committed)

The glue from §1. Marked clearly as run-harness; do **not** copy into `src/scratch_llm/`.

```python
# ~/a5_run/harness.py — HF glue for the A5 Qwen run. NOT repo source.
import torch
from scratch_llm.rollout.types import Rollout
from scratch_llm.envs.protocol import Task, Graded
from scratch_llm.rewards.r1_zero import render_r1_zero_prompt
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn as math_reward_fn

# ---- 1. HF tokenizer codec (plugs into GSMMathEnv / builds Task.prompt_ids) ----
class HFCodec:
    def __init__(self, tok): self.tok = tok
    def encode(self, text):  return self.tok.encode(text, add_special_tokens=False)
    def decode(self, ids):   return self.tok.decode(list(ids), skip_special_tokens=True)

# ---- 2. Official-grader MATH env (EI grades via env.grade, so grader lives here) ----
class MathEnv:
    def __init__(self, rows, codec):        # rows: list[{"problem","answer"}]
        self.codec = codec
        self._tasks = [Task(task_id=f"math-{i}",
                            prompt_ids=tuple(codec.encode(render_r1_zero_prompt(r["problem"]))),
                            ground_truth=r["answer"],
                            metadata={"question": r["problem"]})
                       for i, r in enumerate(rows)]
    def decode(self):        return self.codec.decode
    def tasks(self):         return list(self._tasks)
    def grade(self, task, rollout):
        text = self.codec.decode(rollout.response_ids)
        rd = math_reward_fn(text, task.ground_truth)     # <-- math_verify-backed
        return Graded(rd["reward"], rd["format_reward"], rd["answer_reward"], text)

# ---- 3. HF rollout SampleFn (returns Rollout with temp-1 re-scored logprobs) ----
def make_hf_sampler(model, tok, device, max_new_tokens=512, temperature=1.0, top_p=1.0):
    eos = tok.eos_token_id
    @torch.no_grad()
    def sample_fn(task, group_size):
        prompt = torch.tensor([task.prompt_ids], device=device)
        prompt = prompt.repeat(group_size, 1)            # G copies of the prompt
        model.eval()
        gen = model.generate(prompt, do_sample=True, temperature=temperature, top_p=top_p,
                             max_new_tokens=max_new_tokens, pad_token_id=eos,
                             eos_token_id=eos, num_return_sequences=1)
        out = []
        plen = len(task.prompt_ids)
        for row in gen:
            resp = row[plen:].tolist()
            if eos in resp:                              # trim at first eos
                resp = resp[:resp.index(eos) + 1]
            # temp-1 re-score of the TAKEN tokens (repo convention: raw log-softmax, not temp-scaled)
            full = torch.tensor([list(task.prompt_ids) + resp], device=device)
            logits = model(full).logits.float()[0]       # (L, V)
            lp = torch.log_softmax(logits[plen-1:plen-1+len(resp)], dim=-1)
            taken = torch.tensor(resp, device=device)
            logprobs = lp.gather(-1, taken.unsqueeze(-1)).squeeze(-1).tolist()
            out.append(Rollout(tuple(task.prompt_ids), tuple(resp), tuple(logprobs),
                               "stop" if (resp and resp[-1] == eos) else "length"))
        return out
    return sample_fn
```

> `num_return_sequences=1` + an explicit `repeat` keeps each rollout's logprobs cheap to re-score;
> `generate(num_return_sequences=G)` also works and is faster, but re-scoring must still run per
> sequence to honor the repo's temp-1 logprob convention (used by the IS-ratio monitor and, for
> `epochs_per_rollout_batch>1`, by the clip's `π_old`).

---

## 8. Launch — the four stages

All stages import `~/a5_run/harness.py`. Run from `/workspace/scratch_llm` with the venv active.
Save each stage's metrics + a checkpoint to `~/a5_run/` (survives nothing — `vastai copy` results
off the box before destroy, §11).

### 8.1 Zero-shot MATH baseline

```python
# baseline.py — categorize test set by (format × answer). No training.
import json, torch, pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from scratch_llm.rewards.r1_zero import render_r1_zero_prompt
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B",
            torch_dtype=torch.bfloat16).to(dev).eval()
rows = [json.loads(l) for l in (pathlib.Path.home()/"a5_run/data/math_test.jsonl").read_text().splitlines()]
rows = rows[:500]                                   # 500-problem eval slice (fast, standard)
n_fmt = n_both = 0
for r in rows:
    ids = tok(render_r1_zero_prompt(r["problem"]), return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(**ids, do_sample=False, max_new_tokens=512, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    rd = r1_zero_reward_fn(text, r["answer"])
    n_fmt += rd["format_reward"]; n_both += rd["reward"]
print(f"format={n_fmt/len(rows):.3f}  MATH(format&answer)={n_both/len(rows):.3f}")
```

### 8.2 SFT sweep `{128, 256, 512, 1024, full}`

Full fine-tune with the repo's `sft_microbatch_train_step` (the graded SFT step). For each data size:
load Qwen fresh, build `(prompt, response)` batches through
`scratch_llm.algos.sft.tokenize_prompt_and_output(prompts, responses, tok)` (HF tokenizer satisfies
the `TokenizerLike` duck-type), then per microbatch:

```python
from scratch_llm.algos.sft import tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step
b = tokenize_prompt_and_output(prompts, responses, tok)                       # input_ids/labels/response_mask
b = {k: v.to(dev) for k, v in b.items()}
lp = get_response_log_probs(model, b["input_ids"], b["labels"])["log_probs"] # grad-bearing
loss, meta = sft_microbatch_train_step(lp, b["response_mask"],
              gradient_accumulation_steps=GA, normalize_constant=1.0)        # calls backward()
# every GA microbatches: optimizer.step(); optimizer.zero_grad()
```

Sweep sizes, eval each on the 500-problem test slice (§8.1), plot val-MATH vs data size. Also run
the **filtered-correct** variant (SFT only on solutions the base model already gets right) — it is
the EI seed.

### 8.3 Expert Iteration sweep

`expert_iteration` grades via `env.grade`, so use the `MathEnv` (official grader) + the HF sampler:

```python
from scratch_llm.algos.expert_iteration import expert_iteration
from scratch_llm.optim import AdamW   # or torch.optim.AdamW
env  = MathEnv(train_rows_subset, HFCodec(tok))
samp = make_hf_sampler(model, tok, dev, max_new_tokens=512, temperature=1.0)
hist = expert_iteration(model, AdamW(model.parameters(), lr=1e-5), env, samp,
                        n_ei_steps=5, group_size=G, sft_batch_size=8, keep_threshold=1.0)
for m in hist:   # EIStepMetrics
    print(m.step, m.kept_fraction, m.n_kept, m.mean_reward, m.mean_sft_loss)
```

Sweep `G∈{4,8,16}` (and batch `∈{512,1024,2048}` train prompts/step). Watch `kept_fraction` rise
then plateau (the no-credit-assignment ceiling — the motivation for GRPO, per the module docstring).

### 8.4 GRPO / Dr.GRPO to ≥25% MATH

`grpo_train_loop` takes a `reward_fn=` override, so pass the official grader directly (no MathEnv
needed for the reward; still pass a `MathEnv` for `tasks()`/`decode()`):

```python
from scratch_llm.algos.grpo import grpo_train_loop
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
env  = MathEnv(train_rows_subset, HFCodec(tok))
samp = make_hf_sampler(model, tok, dev, max_new_tokens=512, temperature=1.0)
hist = grpo_train_loop(
    policy=model, env=env, sample_fn=samp,
    optimizer=torch.optim.AdamW(model.parameters(), lr=3e-6),
    reward_fn=r1_zero_reward_fn, decode_fn=HFCodec(tok).decode,
    n_grpo_steps=200, group_size=8,
    normalize_by_std=False, length_normalization="constant",     # Dr.GRPO defaults (ADR-0017)
    loss_type="reinforce_with_baseline",                          # on-policy; epochs=1
    epochs_per_rollout_batch=1, microbatch_size=8, max_grad_norm=1.0, seed=0,
)
for m in hist:   # GRPOStepMetrics — log ALL of these (§8 checklist below)
    print(m.step, m.mean_reward, m.entropy, m.kl_current_ref, m.kl_current_old,
          m.is_ratio_ess, m.length_correct_mean, m.length_incorrect_mean)
```

**Critical scale note (read §9 first):** `MathEnv(train_rows_subset, ...)` is the *per-step* prompt
set — `grpo_train_loop` samples `group_size` rollouts for **every** task in `env.tasks()` each step.
With Qwen's 151 k vocab, keep `len(tasks) × group_size` small (≤ ~32) or the logging OOMs (§9-B).
Feed a small rotating slice of prompts per step; do not pass all 7.5 k MATH prompts as one env.

Then the two graded ablations:
- **LR sweep:** rerun 8.4 with `lr ∈ {1e-6, 3e-6, 1e-5}`. Overlay the reward curves.
- **Baseline vs no-baseline:** `loss_type="no_baseline"` (pass `raw_rewards`, REINFORCE, no group
  baseline) vs `loss_type="reinforce_with_baseline"` (group advantage). Same LR, same seed.

---

## 8-bis. Mandatory RL logging checklist (discipline #4 — an unlogged RL run is worthless)

`grpo_train_loop` already wires these through `utils/monitors.py` into each `GRPOStepMetrics`
(+ its `.snapshot: MonitorSnapshot`). **Before you trust any reward number, confirm every channel is
present and sane:**

- [ ] **Entropy** (`m.entropy`) — token entropy of the policy. Should start high (~`log V` region on a
      fresh head, lower for Qwen) and *fall gradually*. A cliff to ~0 = entropy collapse (LR too high).
- [ ] **`KL(current‖ref)`** (`m.kl_current_ref`) — drift from the frozen initial policy. Rises
      monotonically; a blow-up (> ~0.5 and climbing) = policy running away → lower LR / add KL penalty.
- [ ] **`KL(current‖old)`** (`m.kl_current_old`) — per-step step size. Near-0 for on-policy
      (`epochs=1`); grows with `epochs_per_rollout_batch>1` (that is what the clip bounds).
- [ ] **IS-ratio mean + ESS** (`m.is_ratio_mean` ≈ 1.0; `m.is_ratio_ess` = normalized ESS in [0,1]).
      ESS < ~0.1 = the off-policy correction blew up → shrink epochs or LR.
- [ ] **`kl_train_infer`** (`m.snapshot.kl_train_infer`) — train-engine vs serving-engine drift.
      **0.0 by construction here** (single HF engine does both roles), but the channel must exist;
      **HALT at > 0.10** (`monitors.KL_TRAIN_INFER_HALT`) the moment a separate serving engine (vLLM)
      is wired in. `m.snapshot.halt` is the tripwire.
- [ ] **Reward distribution** (`m.mean_reward`, `snapshot.reward_{mean,std,min,max}`) — the learning
      curve. `mean_reward` is measured on rollouts sampled *before* the step's update.
- [ ] **Length by correctness** (`m.length_correct_mean` vs `m.length_incorrect_mean`) — the
      verbosity reward-hacking tell. If incorrect answers get systematically longer, the model is
      padding to game the format reward.

Persist all of the above per step to `~/a5_run/grpo_log.jsonl`. This is the graded artifact as much
as the final accuracy.

---

## 9. Predict-before-run + kill criteria (FOP-2/3)

Write these numbers down **before** launching; they are the debugging anchors. Bands are
`[INFERENCE]` (from published Qwen2.5-Math-1.5B RL results + the CS336 handout) until measured.

### Predicted numbers

| Stage | Predicted `[INFERENCE]` | The falsifier / what it tells you |
|---|---|---|
| Baseline format-reward | 0.5 – 0.9 | < 0.1 ⇒ prompt/tokenizer wiring broken (model never emits `</think> <answer>`) — **stop, fix before spending on RL** |
| Baseline MATH (format&answer) | ~10 – 20% | this is the bar every later stage must beat |
| SFT full | ~20 – 25% (monotone rise with data size) | flat across sizes ⇒ mask/label misalignment (re-check `response_mask`, `test_sft_algos.py`) |
| EI, `kept_fraction` | rises steps 0→3, plateaus by 4–5 | never rises ⇒ sampler temperature too low (no diversity) or grader rejecting valid answers |
| GRPO final MATH | **≥ 25%** (the graded bar), curve above baseline | flat 30 steps w/ healthy entropy ⇒ group-variance collapse (all-same-reward groups → A=0); raise `group_size`/temperature |
| GRPO entropy | falls gradually | cliff to ~0 ⇒ entropy collapse (LR too high) |
| LR sweep | 3e-6 stable; 1e-5 may collapse; 1e-6 slow | too-high LR ⇒ reward crash + KL(cur‖ref) blow-up |
| Baseline vs no-baseline | baseline lower-variance, faster, higher plateau | if no_baseline matches baseline ⇒ group advantage not wired (check `compute_group_normalized_rewards`) |

### Kill criteria (stop the run — do not burn GPU-$ on a broken loop)

1. **Baseline format < 0.1** → generation/prompt glue is wrong. Fix §7 before any training.
2. **GRPO `mean_reward` flat for 30 steps** → decision tree: entropy collapsing → LR too high (kill,
   drop LR 3×); entropy healthy → group variance collapsed (kill, raise `group_size`/temperature).
3. **`kl_current_ref` > 0.5 and climbing** → policy diverging (kill, lower LR / shorten steps).
4. **`is_ratio_ess` < 0.1** (only with `epochs>1`) → off-policy blow-up (kill, set `epochs=1`).
5. **`snapshot.halt` True** (kl_train_infer > 0.10) → only fires with a separate serving engine;
   if it fires, the gradient is against a policy you are not serving — **HALT**.
6. **OOM** → see the two constraints below (this is the expected failure, not a bug).

### The two memory constraints a coding agent misses (name them, size for them)

Both are in `grpo_train_loop` / `_log_step` (`algos/grpo.py`), fine at CPU/toy scale, load-bearing
at Qwen scale:

- **9-A · Triple-resident model.** The loop keeps `ref_model = deepcopy(policy)` (line 464,
  persistent) **and** `old_model = deepcopy(policy)` (line 499, rebuilt every step). So **3× the 1.5B
  policy is resident** (~3.0 GB bf16 each ≈ **9 GB**) *before* AdamW fp32 state (m+v = 2·4·1.5e9 ≈
  **12 GB**), grads (~3 GB), and activations. ≈ **24 GB floor before activations** → OOMs a 24 GB
  card; comfortable on 80 GB. *This is why the tier is H100 and the local fallback is a smoke only.*
- **9-B · Full-vocab KL rows.** `_log_step` calls `_response_rows` for policy **+ ref + old**, each
  materializing `log_softmax` rows `(M, V)` then `.numpy()` (lines 559–564), with **V = 151,936**.
  Memory ≈ `3 × M × 151936 × 4 bytes`, where `M` = total response tokens in the step's rollout batch.
  At `rollout_batch=256 × ~256 tokens` (M ≈ 65 k) that is **~120 GB → OOM even on H100.** *Mitigation
  (required):* keep the per-step rollout batch small — choose `len(env.tasks()) × group_size` and
  `max_new_tokens` so `3 · M · V · 4 < 0.5 · free_VRAM`. Concretely on H100: `len(tasks) × group_size
  ≤ ~32`, `max_new_tokens ≤ 512` ⇒ `M ≲ 16k` ⇒ ~30 GB for the KL rows, safe. Rotate a fresh 4-task
  slice through 200 steps rather than one giant env. (A committed fix would chunk `_log_step` over
  the vocab / subsample KL positions — **out of scope here, do not edit `src/`**; size around it.)

---

## 10. 24 GB local fallback recipe (standing sm120 box — first attempt, $0)

Run this on the standing RTX PRO 4000 Blackwell (24 GB) **before** renting, to shake out the glue at
$0. It is a *smoke*, not the graded sweep — it proves the loop learns on a real model, nothing more.

Deltas from the H100 recipe:
- **Model:** same `Qwen/Qwen2.5-Math-1.5B`, `torch_dtype=torch.bfloat16` (do **not** try fp32).
- **Avoid the triple-resident OOM (9-A):** full-FT 1.5B + AdamW + 3× model does not fit 24 GB. For
  the smoke, either (a) freeze all but the top ~4 transformer blocks + the LM head (cut optimizer
  state ~5×), or (b) install `bitsandbytes` + use 8-bit AdamW, or (c) `peft` LoRA (rank 16) — each is
  a *deliberate, non-concurrent* `uv pip install` (do not race the perf agents). Cheapest to start:
  option (a), pure-torch, no new dep.
- **Avoid the KL-rows OOM (9-B):** tiny rollout batch. `MathEnv(train_rows[:2], codec)` (2 tasks) ×
  `group_size=4` = 8 rollouts, `max_new_tokens=256` ⇒ `M ≲ 2k` ⇒ KL rows ~3.5 GB. Safe.
- **Grad-accum:** `microbatch_size=2`, let `grpo_train_loop` chunk (grad-accum is automatic —
  `grad_accum = len(chunks)`).
- **Steps:** `n_grpo_steps=20`, `lr=3e-6`. Expect `mean_reward` to *move up* off the baseline and
  entropy to fall — same shape as the toy W8c ledger row (`bench/RESULTS.md`, E[r] 0.19→0.67), now on
  a real model. If it moves, the glue is correct → rent the H100 for the real sweep.

```python
# local smoke — 2 tasks × G=4, top-blocks-only, 20 steps
for p in model.parameters(): p.requires_grad_(False)
for blk in model.model.layers[-4:]:                    # unfreeze top 4 blocks
    for p in blk.parameters(): p.requires_grad_(True)
for p in model.lm_head.parameters(): p.requires_grad_(True)
opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-6)
hist = grpo_train_loop(model, MathEnv(train_rows[:2], HFCodec(tok)),
                       make_hf_sampler(model, tok, "cuda", max_new_tokens=256, temperature=1.2),
                       opt, reward_fn=r1_zero_reward_fn, decode_fn=HFCodec(tok).decode,
                       n_grpo_steps=20, group_size=4, microbatch_size=2,
                       normalize_by_std=False, length_normalization="constant", seed=0)
```

Kill line for the smoke: `mean_reward` flat for 20 steps **and** entropy not moving ⇒ the sampler
isn't producing diverse/correct rollouts — debug the glue (§7), do **not** rent yet.

---

## 11. Where the results go

- **`bench/RESULTS.md` → "Main track (CS336 A2→A5)" section** — append (never edit) a dated block,
  same shape as the existing `W8c · A5 — GRPO toy engine` rows: `| date | node/artifact | metric |
  predicted (pre-registered) | measured | note |`. Log: baseline MATH%, SFT-sweep curve, EI
  `kept_fraction` trajectory, **GRPO final MATH% (the ≥25% gate)**, the LR-sweep verdict, and the
  baseline-vs-no-baseline delta. Mark each `[FACT]` (measured, seed-fixed) vs `[INFERENCE]`.
- **`docs/STATUS.md`** — flip the A5 run-tier ledger entry from "code-complete, rental-deferred" to
  "measured (H100, <date>)" with the headline number.
- **Raw artifacts off the box** — before `vastai destroy`, pull logs + curves back:
  ```bash
  vastai copy "$INSTANCE_ID":/root/a5_run/ local:./a5_results/   # grpo_log.jsonl, checkpoints, plots
  ```
  Do **not** commit checkpoints (`.gitignore`); commit only the RESULTS.md/STATUS.md numbers +
  any small plot PNG under `bench/`.

> The harness (`~/a5_run/harness.py`, §7) is throwaway — it stays on the pod, never in git. Only the
> *numbers* and a plot are durable artifacts.

---

## 12. Rough cost estimate

H100 SXM on Vast ≈ **$1.8–3.0/hr** on-demand (~40% less on `--type bid` spot).

| Stage | ~wall time (1× H100, HF-generate rollouts) |
|---|---|
| Baseline (500-problem eval) | ~0.3 h |
| SFT sweep (5 sizes + filtered variant) | ~2 h |
| EI sweep (G∈{4,8,16}, 5 steps each) | ~4–6 h |
| GRPO 200 steps (small rollout batch, §9-B) | ~4–8 h |
| LR sweep (×3) + baseline ablation (×1) | ~3–5 h |
| **Total** | **~14–22 wall-hours** |

**≈ $40–90 on-demand** (buffer for restarts/OOM-retries), **≈ $25–55 on spot**. A vLLM/SGLang
rollout engine would cut EI+GRPO 2–4× (optional, needs a deliberate `uv pip install vllm`). The
**24 GB-local smoke (§10) is $0** and should always run first. **Destroy the instance the moment the
last `vastai copy` completes** — GPU billing runs until `destroy`, and storage bills from `create`.

---

## 13. One-paragraph orientation for the next agent

The A5 RL *algorithms* are done and CPU-tested (`algos/{sft,expert_iteration,grpo}.py`,
`rewards/r1_zero.py`, `envs/`, `utils/monitors.py`; green in `tests/test_*`). This runbook is the
*measurement* half: rent one H100, bootstrap the repo (`scripts/bootstrap-pod.sh` + `[gpu]` +
`math-verify` + the scaffold for `cs336_alignment.drgrpo_grader`), download Qwen2.5-Math-1.5B + MATH,
write the ~120-line HF glue (§7, throwaway), then run baseline → SFT sweep → EI sweep → GRPO to
≥25% MATH + the LR/baseline ablations, logging every §8-bis channel. The only two things that will
bite you are the two memory constraints in §9 (triple-resident model, 151 k-vocab KL rows) — size the
rollout batch for them and it is a clean run. Numbers land in `bench/RESULTS.md` main-track; the box
gets destroyed the moment results are copied off.
```
