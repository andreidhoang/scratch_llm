# Runbook — A3 `scaling_laws` leaderboard (Stanford training API)

> ## 🚫 BLOCKED-EXTERNAL
> The training API (`http://hyperturing.stanford.edu:8000`) is reachable **only from the
> Stanford network** (campus or Stanford VPN) and requires an enrolled student's 8-digit
> API key. Nothing here can run from this box. The method is code-complete and tested
> offline (`src/scratch_llm/scaling/planner.py` + `scaling/isoflop.py`,
> `tests/test_planner.py::test_end_to_end_planner_recovers_planted_optimum` simulates the
> whole flow against a planted loss surface). This runbook is the exact procedure for
> whoever has network access.

## What you are doing

Spend a **12 B200-hour (43 200 s) wall-clock query budget** against a live training API to
fit `N_opt ∝ C^a`, then submit the predicted compute-optimal config + predicted loss for a
**48 B200-hour** run (`POST /final_submission`, 50 pts). The budget accounting is
adversarial — the planner exists to make it impossible to lose money to it:

| Event | Budget effect |
|---|---|
| `POST /submit` accepted (queued/running) | **reserves the FULL `max_runtime_seconds`** |
| run completes | refunded down to actual runtime, **floored at 1 s** |
| run times out / fails | **charged the full `max_runtime_seconds`**, no loss returned |
| duplicate `training_config` | HTTP **409**, nothing charged |
| `max_runtime_seconds` > remaining budget | HTTP **400**, nothing charged |

`QueryPlanner` mirrors every row of that table offline (`scaling/planner.py`); the server's
implementation is `cs336_scaling/budget.py` + `api/public.py` in the scaffold if you need
to re-verify a rule.

## Prerequisites

1. Stanford VPN up (Cisco AnyConnect, `su-vpn.stanford.edu`) or on-campus network.
2. `export A3_API_KEY=<8-digit student id>` (e.g. `06123456`).
3. The official scaffold: `/workspace/lectures/assignment3-scaling/` — client, schemas,
   and the notebook `examples/client_example.ipynb`. Set up its env:
   ```bash
   cd /workspace/lectures/assignment3-scaling && uv sync
   ```
4. This repo importable in the same env (planner + fitter):
   ```bash
   uv pip install -e /workspace/scratch_llm
   ```
5. Sanity: `uv run python -c "from cs336_scaling import client; print(client.get_budget())"`
   → `used_seconds=0, remaining_seconds=43200, total_budget_seconds=43200` on a fresh key.

## Step 1 — calibrate `max_runtime_seconds` from two cheap runs

Never guess the reservation. Submit **two tiny configs** (≈2–5 M params, ≈10⁸ tokens) with a
generous-but-small cap (~600 s each; worst case burns 1200 s = 2.8 % of budget), poll to
completion, and read `status.used_runtime_seconds`:

```python
secs_per_flop = mean(used_runtime_seconds_i / (6 * N_i * D_i) for the two runs)
max_runtime(config) = max(60.0, 2.0 * 6 * N * D * secs_per_flop)   # 2x headroom
```

Too loose over-reserves the cap (queued runs pin the full reservation, serializing your
grid); too tight risks a timeout charged at full price with **no loss returned** — the only
true loss of money in the game. 2× headroom is the tested default.

## Step 2 — predeclare the grid (the discipline)

Decide the whole grid **before** the first grid submission, then never improvise. Shape
(mirrors the offline E2E test):

- 5–6 compute budgets `C_i`, geometric, e.g. `1e17 … 3e19` FLOPs.
- Per budget: 5–8 architectures off a fixed `(n_layer, d_model)` ladder (head_dim 64,
  `num_heads = d_model/64`, `num_key_value_heads == num_heads` — the API rejects GQA),
  keeping tokens-per-param `D/N` inside `[1, 1000]`; `D = C_i / (6N)` rounded to a multiple
  of `512 * train_batch_size`.
- Hold non-N hyperparameters at sane defaults (AdamW, peak LR ~3e-4, batch 32). Fitting a
  1-D `N_opt(C)` law is the budget-affordable choice — record the decision as
  `ADR: A3 scaling-law query budget allocation`.

```python
from scratch_llm.scaling.planner import PlannedRun, QueryPlanner, RunConfig

planner = QueryPlanner()                 # 43_200 s cap
# replay the two calibration runs into the ledger first (submit + complete), then:
worst_case = planner.declare_grid(grid)  # raises if all-timeout cost exceeds remaining
```

`declare_grid` refuses a grid whose **all-timeout worst case** exceeds the remaining
budget, rejects internal duplicates, and locks the plan: later submissions outside the grid
raise `UndeclaredConfigError`. Aim for worst case ≤ ~80 % of remaining, keeping a reserve
for one re-run of a failed experiment.

## Step 3 — submit / poll / settle, planner first

The planner is the gate; the HTTP call only happens after the planner accepts. Config
validation (`hidden_size == heads * head_dim`, `heads % kv_heads == 0`,
`total_train_tokens % (512 * batch) == 0`, positivity) runs locally in
`RunConfig.validate()` before any budget is risked.

```python
import time
from cs336_scaling import client

def to_training_config(c: RunConfig, max_runtime_seconds: float) -> dict:
    return {
        "architecture_config": {
            "num_hidden_layers": c.num_hidden_layers,
            "num_attention_heads": c.num_attention_heads,
            "num_key_value_heads": c.num_key_value_heads,
            "head_dim": c.head_dim,
            "hidden_size": c.hidden_size,
            "intermediate_size": 4 * c.hidden_size,
            "attention_bias": False, "rms_norm_eps": 1e-5, "rope_theta": 10000,
            "tie_word_embeddings": True, "dtype": "bfloat16", "vocab_size": 32000,
        },
        "optimizer_config": {
            "lr_scheduler": {"peak_value": c.learning_rate},
            "weight_decay": 1e-2, "beta1": 0.9, "beta2": 0.95,
            "eps": 1e-8, "eps_root": 1e-8, "grad_clip_norm": 1.0,
        },
        "train_batch_size": c.train_batch_size, "val_batch_size": 32,
        "n_evals": 16, "total_train_tokens": c.total_train_tokens,
        "max_runtime_seconds": max_runtime_seconds, "model_seed": 0,
    }

for run in grid:
    rid = planner.submit(run.config, run.max_runtime_seconds)   # local 409/400 gate
    resp = client.submit_experiment(to_training_config(run.config, run.max_runtime_seconds))
    while True:
        exp = client.get_experiment(resp.experiment_id)
        status = exp.status
        if status.status_type == "completed":
            planner.complete(rid, status.used_runtime_seconds, status.val_losses[-1])
            break
        if status.status_type == "failed":
            planner.fail_timeout(rid)      # timeout/failure: charged full reservation
            break
        time.sleep(30)
    remote = client.get_budget()
    assert abs(remote.remaining_seconds - planner.remaining_seconds) < 2.0, "ledger drift — STOP"
```

Notes:
- `n_evals` must divide `total_optimizer_steps` (= `total_train_tokens / (512 * batch)`) —
  keep step counts multiples of 16 or lower `n_evals`.
- The API's duplicate check hashes the **whole** `TrainingConfig` (including
  `max_runtime_seconds`); the planner is deliberately stricter and keys on the config
  alone — re-buying a config under a different cap is wasted budget.
- Submitting several runs concurrently is fine (the planner already reserved for all of
  them the moment you called `submit`); the ledger-drift assert is the tripwire.

## Step 4 — fit, predict, final submission

```python
from scratch_llm.scaling.isoflop import fit_powerlaw, isoflop_min, tokens_from_compute_params

rows = [
    {"compute_budget": budget_of[cfg], "parameters": cfg.nonembed_params, "final_loss": loss}
    for cfg, loss in planner.results()
]
pairs = isoflop_min(rows)                                  # (C_i, N_opt_i) per budget
n_law = fit_powerlaw([c for c, _ in pairs], [n for _, n in pairs])

C_TARGET = 48 * 3600 * B200_FLOPS_PER_SEC                  # read the PDF's stated conversion;
n_opt = n_law.predict(C_TARGET)                            # else 2.25e15 * MFU as fallback
d_opt = tokens_from_compute_params(C_TARGET, n_opt)
# map N -> architecture via N ≈ 12 * n_layer * d_model**2 (nearest ladder point, log space),
# predict the loss by evaluating a loss-vs-C power-law fit at C_TARGET,
client.save_final_submission(to_training_config(final_cfg, 48 * 3600.0), predicted_loss)
```

- **Predict before you submit**: state `n_opt`, `d_opt`, the predicted loss, and the
  extrapolation factor `C_TARGET / max(C_i)` (should be ≤ ~30×) out loud in the writeup
  *before* the POST — that is the graded skill.
- Sanity gates from the offline test: `n_law.exponent ∈ (0.35, 0.55)` and
  `exponent_N + exponent_D ≈ 1`. If either fails, the grid data is bad — do not submit.
- `final_submission` is resubmittable; the latest one counts. `GET /final_submission`
  confirms what is on file.

## Offline dry-run (no VPN needed)

The entire flow above, minus HTTP, is executed by
`pytest tests/test_planner.py::test_end_to_end_planner_recovers_planted_optimum -q`
in this repo: calibration → `declare_grid` → reserve/refund ledger → `isoflop_min` →
`fit_powerlaw` → recovery of the planted 48-B200-hr optimum. Run it once before the live
session; if you change the grid recipe, change it there first.
