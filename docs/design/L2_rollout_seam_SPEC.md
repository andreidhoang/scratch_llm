# Design spec — L2 rollout seam (the train↔infer comparison harness)

> **Status:** spec / building now. The CPU-buildable step after the KV-cache (see
> [`L2_kv_cache_SPEC.md`](L2_kv_cache_SPEC.md) §6).
> **Layer:** L2 Systems (A2.3) — the rollout/inference engine over which the `kl_train_infer`
> measurement compares training-engine and serving-engine logits.
> **Files touched:** new `src/scratch_llm/rollout/types.py`, `src/scratch_llm/rollout/local.py`,
> `src/scratch_llm/rollout/__init__.py` (exports); `src/scratch_llm/sampling.py` (add a
> logprob-returning decode variant, `generate` unchanged); new `tests/test_rollout.py`.

## 1. Why (the problem this solves)

A2.3 (`A2_systems_BUILD_GUIDE.md` §4) defines the rollout engine's purpose: it is the point where
**training-engine logits and serving-engine logits are compared** — and that comparison *is*
`kl_train_infer = KL(train‖infer)` (HALT@0.10), already implemented in `utils/monitors.py`. L5's RL
spine also needs rollouts to develop against: a stable contract that yields, per generated token,
the **policy log-probability** that feeds advantages and IS-ratios.

So we need a **backend-agnostic seam**: one `RolloutClient` contract, a CPU `LocalBackend` now, and
the GPU `SGLangBackend` later behind the *same* interface (so swapping engines is a one-line change
and `kl_train_infer` measures real engine drift, not an API mismatch).

`LocalBackend` plays a double role on CPU: it is **both** the train-engine logprob source **and**,
until SGLang lands, the **stand-in infer engine**. Re-scoring one rollout with two `LocalBackend`s
(identical model) gives `kl_train_infer ≈ 0` — proving the A2.3 plumbing end-to-end before any GPU
spend; the number becomes meaningful the moment the serving backend's kernels/precision differ.

## 2. The contract

```python
@dataclass(frozen=True)
class Rollout:
    prompt_ids:   tuple[int, ...]
    response_ids: tuple[int, ...]
    logprobs:     tuple[float, ...]   # per response token: policy log π(a_t|s_t), temperature 1
    stop_reason:  Literal["stop", "length"]   # "stop" = a stop_id was emitted; else "length"

class RolloutClient(Protocol):
    def generate(self, prompt_ids: Sequence[int], params: SamplingParams) -> Rollout: ...
    def generate_batch(self, prompts: Sequence[Sequence[int]], params: SamplingParams) -> list[Rollout]: ...
```

`LocalBackend(model, device="cpu")` implements `RolloutClient` over the from-scratch model +
`sampling.generate`, plus two re-scoring methods used by the harness and the contract test:

- `score(prompt_ids, response_ids) -> list[float]` — **teacher-forced** per-token policy log π of
  the *taken* tokens. Feeds `monitors.importance_ratios`; the contract-test oracle.
- `distribution_logprobs(prompt_ids, response_ids) -> Tensor` — full per-position `log_softmax`
  rows (`(T_resp, vocab)`) at the response positions. Feeds `monitors.mean_kl` → the three KLs.

## 3. Decisions & trade-offs

- **Logprob convention = raw policy log π(a|s) at temperature 1** (`log_softmax(logits)[a]`),
  *not* the post-temperature/top-p sampling distribution. Temperature/top-p stay pure exploration
  knobs; advantages/IS-ratios use the true policy logprob. This is the PPO/GRPO convention and
  exactly the input `monitors.importance_ratios(logp_current, logp_old)` expects. → [ADR-0006](../adr/ADR-0006-policy-logprob-convention.md).
- **Capture via a shared decode core, not a duplicated loop.** `sampling.py` factors a private
  `_decode(...)` returning `(ids, logprobs)`; `generate()` keeps its `list[int]` return (existing
  sampling + KV-cache tests unchanged), and `generate_with_logprobs()` exposes both. Single source
  of decode truth; preserves ADR-0003 (sampler parity — config unchanged, only a richer return).
- **`score()` re-derives logprobs by teacher forcing**, independent of the decode path, so the
  contract test (inline == re-scored) is a real cross-check, not a tautology. Equality holds because
  the cached decode path is logit-identical to recompute (the KV-cache invariant).
- **Position alignment:** response token at sequence index `p+i` is predicted by the logit row at
  position `p+i-1`; so `distribution_logprobs` returns rows `[p-1 : p-1+T]`. Requires a non-empty
  prompt (already enforced by `generate`).

## 4. Correctness invariants (`tests/test_rollout.py`)

1. **Logprob consistency (the contract):** `LocalBackend.generate(...)` inline `logprobs` ==
   `score(prompt_ids, response_ids)` of the same ids (`assert_close`).
2. **Seed reproducibility:** same `SamplingParams.seed` ⇒ identical `Rollout` (ids + logprobs).
3. **Batch parity:** `generate_batch([p1, p2], params)` == `[generate(p1), generate(p2)]`.
4. **stop_reason:** emitting a `stop_ids` token ⇒ `"stop"`; exhausting `max_tokens` ⇒ `"length"`.
5. **kl_train_infer scaffold:** re-score one rollout with a second identical `LocalBackend`;
   `monitors.mean_kl(rows_A, rows_B) ≈ 0`. Proves the A2.3 comparison harness on CPU.

## 5. Scope / deferred

- **In:** the `Rollout`/`RolloutClient` contract, `LocalBackend` (generate/batch/score/
  distribution_logprobs), the shared logprob decode, CPU-tested.
- **Deferred to a GPU box (resourced, not skipped — `CLAUDE.md`):** `SGLangBackend` behind the same
  Protocol, measuring real `kl_train_infer` drift; FA2 Triton kernel + roofline; DDP-overlap +
  ZeRO-1 + the 100B memory one-pager. SGLang-vs-vLLM is an ADR trigger (A2 guide §8 #2).

## 6. Build sequence

1. `sampling.py`: factor `_decode`, add `generate_with_logprobs` (keep `generate` byte-identical).
2. `rollout/types.py`: `Rollout` + `RolloutClient`.
3. `rollout/local.py`: `LocalBackend` (generate/generate_batch/score/distribution_logprobs).
4. `rollout/__init__.py`: exports.
5. `tests/test_rollout.py`: the §4 invariants — write failing first, then pass.
6. Green-CI; update `docs/STATUS.md`; commit.
