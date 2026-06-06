"""Serve-engine backend — SGLang offline `Engine`, behind the same `RolloutClient` Protocol.

L2 Systems (A2.3) — the serving side of the `kl_train_infer` bridge and the "real" rollout engine
the CPU `LocalBackend` stands in for. It generates rollouts (returning the serve engine's per-token
logprobs of the sampled tokens) and scores given sequences. Diffed against `HFReferenceBackend`
(the train engine) this yields `kl_train_infer` — the divergence HALT@0.10 guards.

GPU-only (imports `sglang`, serves a model). Not imported by `rollout/__init__.py` (CPU/CI safe);
the measurement gates on `pytest.importorskip("sglang")` + the gpu marker.

Honest limitation (the caveat `utils/monitors.py` flags): a serving engine returns only the
*taken-token* logprobs, not the full-vocab distribution — so the full-vocab `monitors.mean_kl` is
unavailable here; `kl_train_infer` is the sampled estimator over those token logprobs (see
bench/kl_train_infer.py). Assumes self-attention causal decoding (the standard LM rollout).
"""

from __future__ import annotations

from collections.abc import Sequence

from reasoning_llm.rollout.types import Rollout, StopReason
from reasoning_llm.sampling import SamplingParams


class SGLangBackend:
    """A :class:`RolloutClient` over an SGLang offline `Engine`."""

    def __init__(
        self, model_path: str, *, dtype: str = "bfloat16", **engine_kwargs: object
    ) -> None:
        import sglang as sgl

        self.engine = sgl.Engine(model_path=model_path, dtype=dtype, **engine_kwargs)

    def _sampling(self, params: SamplingParams) -> dict:
        return {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_new_tokens": params.max_tokens,
            "stop_token_ids": list(params.stop_ids),
        }

    def generate(self, prompt_ids: Sequence[int], params: SamplingParams) -> Rollout:
        out = self.engine.generate(
            input_ids=[list(prompt_ids)],
            sampling_params=self._sampling(params),
            return_logprob=True,
        )[0]
        # meta_info["output_token_logprobs"] is a list of (logprob, token_id, text) per sampled token.
        otl = out["meta_info"]["output_token_logprobs"]
        response_ids = tuple(int(tok) for _, tok, *_ in otl)
        logprobs = tuple(float(lp) for lp, *_ in otl)
        stop_reason: StopReason = (
            "stop" if response_ids and response_ids[-1] in params.stop_ids else "length"
        )
        return Rollout(
            prompt_ids=tuple(prompt_ids),
            response_ids=response_ids,
            logprobs=logprobs,
            stop_reason=stop_reason,
        )

    def generate_batch(
        self, prompts: Sequence[Sequence[int]], params: SamplingParams
    ) -> list[Rollout]:
        return [self.generate(p, params) for p in prompts]

    def score(self, prompt_ids: Sequence[int], response_ids: Sequence[int]) -> list[float]:
        """Serve-engine teacher-forced per-token logprobs of the response tokens (max_new_tokens=0,
        return input logprobs from the first response position)."""
        seq = list(prompt_ids) + list(response_ids)
        out = self.engine.generate(
            input_ids=[seq],
            sampling_params={"max_new_tokens": 0, "temperature": 0.0},
            return_logprob=True,
            logprob_start_len=len(prompt_ids),
        )[0]
        itl = out["meta_info"]["input_token_logprobs"]  # (logprob, token_id, text) per input token
        return [float(lp) for lp, *_ in itl]

    def shutdown(self) -> None:
        self.engine.shutdown()
