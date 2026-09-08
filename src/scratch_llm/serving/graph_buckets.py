"""S1/S-R3 — batch-bucketed CUDA-graph decode: one capture per bucket, padded replay, eager above.

``serving/cudagraph.py`` captures ONE decode step at ONE batch size: the graph records a fixed
``(n_slots, 1)`` input and a fixed ``(B, H)`` grid, so a capture taken at B=32 is a graph *about*
32 rows and nothing else. A server whose batch changes every step therefore cannot use it as
written — which is why vLLM captures a *set* of sizes and pads up to the nearest one
(``vllm/config/vllm.py:2121-2141``, ``vllm/v1/cudagraph_dispatcher.py:82-92``).

This module is that dispatch layer, and only that. It owns:

  * the bucket list and the rule for choosing one (:func:`select_bucket`);
  * a fixed-address staging row-vector per bucket, so a replay never re-allocates its input;
  * the per-bucket runner registry and the capture ORDER (largest bucket first: the caching
    allocator then serves every smaller capture out of the pool the largest one already claimed —
    ``gpu_model_runner.py:6988`` gives the same reason);
  * the counters that make padding waste visible.

It does NOT re-derive capture. The runner for a bucket is injected as a factory, so the
side-stream warm-up / length-snapshot dance stays in ``CudaGraphDecoder`` where it is already
written and already token-exact, and so the whole dispatch path is exercisable on a CPU box with
fake runners — which is where the bug below is actually caught.

**The bug this module exists to make impossible.** Replaying a graph captured at bucket ``b`` with
a batch of ``n > b`` rows is not an error in torch: the staging buffer is a real ``(b,)`` tensor,
``staging[:n] = last_tokens`` raises, but ``staging.copy_(last_tokens[:b])`` does not, and
``out[:n]`` on a ``(b,)`` result silently returns ``b`` rows. Whatever the caller then zips those
rows against — request ids, sequence slots — is off the end, so request ``b`` receives the token
the graph computed for request ``b-1``'s slot on the previous replay. No exception, no NaN, no
wrong-looking logits: just another request's text. Correct behaviour above the largest bucket is
to fall back to eager, and :func:`select_bucket` returns ``None`` there rather than clamping to
the largest bucket. ``tests/serving/test_s1_s_r3.py`` pins exactly that.

Spec: experiments/S1/S-R3/spec.md   ·   Map: experiments/S1/S-R3/map.md
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Sequence

import torch
from torch import Tensor

#: A runner for one bucket: takes a padded ``(bucket,)`` long tensor of last tokens, returns the
#: ``(bucket,)`` long tensor of next tokens. A captured CUDA graph's replay satisfies this; so does
#: an eager step; so does a fake in a CPU test. The decoder holds only this contract.
GraphRunner = Callable[[Tensor], Tensor]


def default_decode_buckets(max_batch: int) -> tuple[int, ...]:
    """``1, 2, 4`` then multiples of 8 up to ``max_batch`` — vLLM's small-batch ladder.

    Mirrors ``vllm/config/vllm.py:2129-2136`` for the range a from-scratch engine plausibly serves,
    and stops there: vLLM's second tier (stride 16 above 256) is for batches this engine's KV pool
    cannot hold, and inventing buckets we never measure would put capture cost and memory into the
    number for no return. ``max_batch`` is always itself a bucket, on- or off-stride — the largest
    batch the scheduler admits is exactly the one that must not fall to eager.
    """
    if max_batch < 1:
        raise ValueError(f"max_batch must be >= 1, got {max_batch}")
    sizes = [b for b in (1, 2, 4) if b <= max_batch]
    sizes += list(range(8, max_batch + 1, 8))
    if max_batch not in sizes:
        sizes.append(max_batch)
    return tuple(sorted(set(sizes)))


def validate_buckets(buckets: Sequence[int]) -> tuple[int, ...]:
    """Ascending, unique, positive — the preconditions :func:`select_bucket`'s bisect assumes."""
    if not buckets:
        raise ValueError("buckets must be non-empty")
    if any(b < 1 for b in buckets):
        raise ValueError(f"bucket sizes must be >= 1, got {tuple(buckets)}")
    if list(buckets) != sorted(set(buckets)):
        raise ValueError(f"buckets must be sorted ascending and unique, got {tuple(buckets)}")
    return tuple(buckets)


def select_bucket(batch: int, buckets: Sequence[int]) -> int | None:
    """The smallest bucket ``>= batch``; ``None`` when ``batch`` exceeds the largest bucket.

    ``None`` means "run eager", and it is the only safe answer above the top bucket. Returning the
    largest bucket instead would be a shape lie — see this module's docstring for what the caller
    then hands back to its users. vLLM answers the same way, at
    ``vllm/v1/cudagraph_dispatcher.py:278`` (``num_tokens > max_size`` -> ``CUDAGraphMode.NONE``).
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    ordered = validate_buckets(buckets)
    idx = bisect_left(ordered, batch)
    return ordered[idx] if idx < len(ordered) else None


class BucketedGraphDecoder:
    """Dispatches a decode step to a per-bucket captured graph, or to eager above the top bucket.

    ``capture(bucket)`` is called at most once per bucket — eagerly by :meth:`capture_all`, or
    lazily on the first step that lands in it — and must return a :data:`GraphRunner` for exactly
    that row count. ``eager`` handles any batch, and is what runs above the largest bucket.

    Padding is real work: a batch of 5 dispatched to bucket 8 runs the model on 8 rows and throws
    3 away. ``padded_rows`` accumulates that waste so the bucket ladder can be judged against it
    rather than assumed — a denser ladder trades capture memory for padded FLOPs, and this counter
    is the only thing that makes the trade visible in a result line.
    """

    def __init__(
        self,
        buckets: Sequence[int],
        capture: Callable[[int], GraphRunner],
        eager: GraphRunner,
        *,
        pad_token: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.buckets = validate_buckets(buckets)
        self._capture = capture
        self._eager = eager
        self.pad_token = pad_token
        self.device = torch.device(device)
        self._runners: dict[int, GraphRunner] = {}
        self._staging: dict[int, Tensor] = {}
        self.graph_steps = 0
        self.eager_steps = 0
        self.padded_rows = 0

    # -- capture ---------------------------------------------------------------------------------

    def capture_all(self) -> None:
        """Capture every bucket, largest first.

        Order matters and is not cosmetic: the caching allocator hands each capture a private
        memory pool, and taking the largest first lets the smaller graphs be carved out of blocks
        it already owns instead of growing the reservation bucket by bucket. vLLM captures
        largest-first for this reason (``gpu_model_runner.py:6987-6989``).
        """
        for bucket in sorted(self.buckets, reverse=True):
            self._runner_for(bucket)

    def _runner_for(self, bucket: int) -> GraphRunner:
        runner = self._runners.get(bucket)
        if runner is None:
            runner = self._capture(bucket)
            self._runners[bucket] = runner
            self._staging[bucket] = torch.full(
                (bucket,), self.pad_token, dtype=torch.long, device=self.device
            )
        return runner

    # -- replay ----------------------------------------------------------------------------------

    def bucket_for(self, batch: int) -> int | None:
        """The bucket this batch would replay, or ``None`` if it would run eager."""
        return select_bucket(batch, self.buckets)

    def step(self, last_tokens: Tensor) -> Tensor:
        """One decode step for ``last_tokens`` ``(batch,)`` long; returns ``(batch,)`` next tokens.

        Above the largest bucket this is the eager step, unchanged and un-padded. Inside the
        ladder it pads into the bucket's fixed-address staging vector, replays, and slices back —
        with the returned width checked, because a runner that hands back fewer rows than it was
        given is the silent cross-request bug and must be an exception rather than a slice.
        """
        if last_tokens.ndim != 1:
            raise ValueError(
                f"last_tokens must be 1-D (batch,), got shape {tuple(last_tokens.shape)}"
            )
        n = int(last_tokens.shape[0])
        if n < 1:
            raise ValueError("last_tokens must hold at least one row")

        bucket = self.bucket_for(n)
        if bucket is None:
            self.eager_steps += 1
            return self._eager(last_tokens)

        runner = self._runner_for(bucket)
        staging = self._staging[bucket]
        staging.fill_(self.pad_token)
        staging[:n] = last_tokens.to(device=staging.device, dtype=staging.dtype)

        out = runner(staging)
        if out.ndim != 1 or int(out.shape[0]) != bucket:
            raise RuntimeError(
                f"bucket {bucket} runner returned shape {tuple(out.shape)}, expected ({bucket},) — "
                "a replay whose width does not match its capture returns another request's tokens "
                "silently; refusing to slice it"
            )
        self.graph_steps += 1
        self.padded_rows += bucket - n
        return out[:n]

    def stats(self) -> dict[str, int]:
        """Dispatch counters, for the result line: graph steps, eager steps, padded rows."""
        return {
            "graph_steps": self.graph_steps,
            "eager_steps": self.eager_steps,
            "padded_rows": self.padded_rows,
            "captured_buckets": len(self._runners),
        }


def paged_capture_factory(
    model: object,
    make_cache: Callable[[int], tuple[object, Tensor]],
) -> Callable[[int], GraphRunner]:
    """Bind ``CudaGraphDecoder`` as the per-bucket capture, for the GPU driver.

    ``make_cache(bucket)`` returns a prefilled ``PagedKVCache`` sized to exactly ``bucket`` slots
    plus its first-token vector; one cache per bucket is unavoidable, because the captured grid is
    ``(B, H)`` and B is baked in at capture time. The import is deferred so that a CPU box can
    import this module — and run the whole dispatch test suite — without the paged Triton kernel.
    """
    from scratch_llm.serving.cudagraph import CudaGraphDecoder

    def capture(bucket: int) -> GraphRunner:
        cache, _first = make_cache(bucket)
        decoder = CudaGraphDecoder(model, cache)  # type: ignore[arg-type]
        decoder.capture()
        return decoder.step

    return capture
