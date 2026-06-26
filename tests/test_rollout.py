"""Tests for the L2 rollout seam: the contract (inline logprobs == teacher-forced re-score),
seed reproducibility, batch parity, stop_reason, and the kl_train_infer comparison scaffold."""

import torch

from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.rollout import LocalBackend, Rollout, RolloutClient
from scratch_llm.sampling import SamplingParams
from scratch_llm.utils.monitors import mean_kl


def _tiny_model(seed: int = 0) -> TransformerLM:
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, context_length=16)
    return TransformerLM(cfg)


def test_logprobs_match_teacher_forced_score() -> None:
    """The contract: the logprobs captured inline during decode equal an INDEPENDENT
    teacher-forced re-score of the same ids (so the seam's logprobs are trustworthy)."""
    backend = LocalBackend(_tiny_model())
    roll = backend.generate(
        [1, 2, 3], SamplingParams(temperature=1.0, top_p=0.9, max_tokens=6, seed=7)
    )
    scored = backend.score(roll.prompt_ids, roll.response_ids)
    torch.testing.assert_close(
        torch.tensor(roll.logprobs), torch.tensor(scored), atol=1e-4, rtol=1e-4
    )


def test_local_backend_satisfies_rollout_client() -> None:
    # Structural-typing sanity (pyright checks the assignment; runtime checks the shape).
    backend: RolloutClient = LocalBackend(_tiny_model())
    roll = backend.generate([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=3))
    assert isinstance(roll, Rollout)
    assert len(roll.response_ids) == len(roll.logprobs)


def test_seed_reproducible() -> None:
    backend = LocalBackend(_tiny_model())
    params = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=8, seed=123)
    assert backend.generate([1, 2, 3], params) == backend.generate([1, 2, 3], params)


def test_batch_parity() -> None:
    backend = LocalBackend(_tiny_model())
    params = SamplingParams(temperature=0.0, max_tokens=4)  # greedy → deterministic
    prompts = [[1, 2, 3], [5, 6]]
    assert backend.generate_batch(prompts, params) == [backend.generate(p, params) for p in prompts]


def test_stop_reason_stop_vs_length() -> None:
    backend = LocalBackend(_tiny_model())
    prompt = [1, 2, 3]
    r_len = backend.generate(prompt, SamplingParams(temperature=0.0, max_tokens=4))
    assert r_len.stop_reason == "length"
    assert len(r_len.response_ids) == 4

    first = r_len.response_ids[0]  # make the first greedy token a stop id
    r_stop = backend.generate(
        prompt, SamplingParams(temperature=0.0, max_tokens=4, stop_ids=(first,))
    )
    assert r_stop.stop_reason == "stop"
    assert r_stop.response_ids == (first,)


def test_kl_train_infer_scaffold_zero_for_identical_engines() -> None:
    """The A2.3 plumbing end-to-end on CPU: re-score one rollout with a second (identical)
    engine and feed the rows to monitors.mean_kl. Identical engines ⇒ KL ≈ 0; the number
    becomes meaningful — and HALT@0.10 fires — only when a real serving engine drifts."""
    model = _tiny_model()
    train, infer = LocalBackend(model), LocalBackend(model)
    roll = train.generate([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=5))
    rows_train = train.distribution_logprobs(roll.prompt_ids, roll.response_ids).numpy()
    rows_infer = infer.distribution_logprobs(roll.prompt_ids, roll.response_ids).numpy()
    assert abs(mean_kl(rows_train, rows_infer)) < 1e-6


def test_kl_train_infer_scaffold_detects_drift() -> None:
    """Positive control: two engines with DIFFERENT weights ⇒ KL > 0. Proves the harness
    actually detects drift — without this, the zero-drift test above would pass even if the
    rollout→distribution_logprobs→mean_kl composition silently produced zero."""
    train, infer = LocalBackend(_tiny_model(seed=0)), LocalBackend(_tiny_model(seed=1))
    roll = train.generate([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=5))
    rows_train = train.distribution_logprobs(roll.prompt_ids, roll.response_ids).numpy()
    rows_infer = infer.distribution_logprobs(roll.prompt_ids, roll.response_ids).numpy()
    assert mean_kl(rows_train, rows_infer) > 0.0
