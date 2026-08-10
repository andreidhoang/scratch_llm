"""A1 Rung 4.4 — CUDA-graph decode over the fixed-address paged pool.

R1 measured decode as *launch/overhead-bound*, not memory-bound: ~955 kernel launches per token on
the eager path (51 tok/s), rising to 173 tok/s only when `torch.compile` fuses the pointwise ops.
The residual gap to the 0.55 TB/s memory wall is the per-launch CPU dispatch. A CUDA graph captures
the whole decode step ONCE and replays it as a single submission — the CPU issues one `cudaGraphLaunch`
instead of hundreds of launches, so the GPU stops waiting on the host between kernels.

Why the paged Triton kernel is the capturable substrate (and the dense/cat paths are not):
- R1's `torch.cat` KV cache grows to fresh addresses each step ⇒ capture is illegal (the graph
  records fixed pointers). The R4.1 block **pool is fixed-address** — writes are in-place.
- The dense `BatchedKVCache` decode reads a `[:, :, :view_len]` slice whose SHAPE grows each step ⇒
  a graph (fixed shapes) can't cover it. The paged kernel's launch is a **fixed `(B, H)` grid**; the
  per-row key count is a runtime loop bound read from the device `lengths` tensor — so one capture
  serves every decode length. `torch.compile` reduce-overhead REFUSES this path anyway (the in-place
  `lengths += active` in `advance` is a "mutated input"); manual capture handles the mutation because
  we own it.

Invariant (tests/test_cudagraph_decode.py, gpu): graph-replayed greedy tokens are IDENTICAL to the
eager paged-kernel decode — capture changes only HOW the step is launched, never WHAT it computes.
"""

from __future__ import annotations

import torch
from torch import Tensor

from scratch_llm.model import PagedKVCache, TransformerLM


class CudaGraphDecoder:
    """Captures one greedy decode step over a prefilled :class:`PagedKVCache` (``use_kernel=True``)
    and replays it per token. The token input is copied into a fixed-address staging tensor before
    each replay; block allocation (``pre_decode_reserve``) and the python length mirror
    (``mirror_advance``) run on the host around the replay, mutating the pool's fixed-address block
    table in place so the captured kernel sees the update.
    """

    def __init__(self, model: TransformerLM, cache: PagedKVCache) -> None:
        if not isinstance(cache, PagedKVCache) or not cache.use_kernel:
            raise ValueError("CudaGraphDecoder needs a PagedKVCache with use_kernel=True")
        self.model = model
        self.cache = cache
        self.device = cache.lengths.device
        self._static_in = torch.zeros((cache.n_slots, 1), dtype=torch.long, device=self.device)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_logits: Tensor | None = None

    @torch.no_grad()
    def capture(self, warmup_steps: int = 3) -> None:
        """Warm up the step on a side stream (primes Triton autotune + the caching allocator), then
        capture it. Warmup + capture advance the cache; we snapshot and restore the lengths so the
        prefilled state is intact for the first real replay (the garbage K/V written past the
        restored length is masked and overwritten by real decodes)."""
        self.model.eval()
        saved = self.cache.lengths.clone()
        saved_py = list(self.cache.py_lengths)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):  # pyright: ignore[reportArgumentType]
            for _ in range(warmup_steps):
                self.model(self._static_in, self.cache)
        torch.cuda.current_stream().wait_stream(stream)
        self.cache.lengths.copy_(saved)  # undo warmup advances before capture

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_logits = self.model(self._static_in, self.cache)

        self.cache.lengths.copy_(saved)  # undo the capture's own advance
        self.cache.py_lengths = list(saved_py)
        self.cache._recompute_view_len()

    @torch.no_grad()
    def step(self, last_tokens: Tensor) -> Tensor:
        """Decode one token for every row: replay the captured step, return the greedy next tokens.
        ``last_tokens``: ``(n_slots,)`` long. Mirrors the eager loop's host work around the replay."""
        assert self._graph is not None and self._static_logits is not None, "call capture() first"
        self.cache.pre_decode_reserve()  # host: allocate a block if a row crosses a 16-boundary
        self._static_in.copy_(last_tokens.view(-1, 1))
        self._graph.replay()
        nxt = self._static_logits[:, -1].argmax(dim=-1).clone()
        self.cache.mirror_advance()
        return nxt

    @torch.no_grad()
    def decode(self, first_tokens: Tensor, n_steps: int) -> Tensor:
        """Greedy-decode ``n_steps`` tokens for all rows from ``first_tokens`` ``(n_slots,)``.
        Returns ``(n_steps, n_slots)`` — token-identical to the eager paged decode."""
        last = first_tokens
        out: list[Tensor] = []
        for _ in range(n_steps):
            last = self.step(last)
            out.append(last)
        return torch.stack(out)
