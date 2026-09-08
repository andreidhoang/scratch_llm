"""S1/S-R4 — the fused MoE layer: route → permute → grouped GEMM → activation → GEMM → un-permute.

The pipeline, and the fp32 oracle it is judged against. The grouped GEMM is injected rather than
imported so the *whole* pipeline is exercised on a laptop: pass the torch reference and every
index the permutation computes, every offset the groups carry, and the un-permute that has to
invert them exactly are executed and compared against :func:`reference_moe` — which routes the
same tokens with no permutation at all. That is the only way to be sure the un-permute is right
before the kernel exists, and un-permute bugs are the ones that do not crash.

Weight layout is vLLM's, so one set of tensors feeds both this rung and its floor:

* ``w1: (E, 2I, H)`` — expert ``e``'s gate rows first, then its up rows. The activation is
  ``silu(gate) * up`` (``fused_moe.py:1798`` applies it to ``intermediate_cache1.view(-1, N)``).
* ``w2: (E, H, I)`` — the down projection.

For Qwen3-30B-A3B: ``E = 128``, ``I = 768``, ``H = 2048``, ``top_k = 8``, no shared expert.

Spec: ``experiments/S1/S-R4/spec.md``   ·   Map: ``experiments/S1/S-R4/map.md``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from scratch_llm.kernels.moe.grouped_gemm import (
    GroupedGEMMConfig,
    default_config,
    exact_m_tiles,
    max_m_tiles,
    reference_grouped_gemm,
)
from scratch_llm.kernels.moe.routing import (
    RUNG,
    ExpertLoad,
    Permutation,
    Routing,
    build_permutation,
    combine,
    expert_load,
    gather_tokens,
)


class GroupedGEMM(Protocol):
    """What the pipeline needs from a grouped GEMM. Two implementations satisfy it: the torch
    reference (CPU, fp32, obviously correct) and the Triton kernel (GPU, the rung)."""

    def __call__(
        self,
        a: Tensor,
        b: Tensor,
        group_offsets: Tensor,
        *,
        config: GroupedGEMMConfig | None = ...,
    ) -> Tensor: ...


def torch_grouped_gemm(
    a: Tensor,
    b: Tensor,
    group_offsets: Tensor,
    *,
    config: GroupedGEMMConfig | None = None,
) -> Tensor:
    """The oracle, adapted to the :class:`GroupedGEMM` signature. ``config`` is ignored — a Python
    loop has no tiles — and accepted so the two implementations are substitutable at the call site."""
    del config
    return reference_grouped_gemm(a, b, group_offsets)


def resolve_gemm(device: torch.device | str) -> GroupedGEMM:
    """The Triton kernel on CUDA, the torch reference everywhere else.

    Imported lazily: ``grouped_gemm_triton`` imports ``triton`` at module scope (the house pattern
    for a Triton kernel file), and this repo's CPU test suite runs on boxes with no Triton.
    """
    if torch.device(device).type == "cuda":
        from scratch_llm.kernels.moe.grouped_gemm_triton import grouped_gemm_triton  # noqa: PLC0415

        return grouped_gemm_triton
    return torch_grouped_gemm


def silu_and_mul(h: Tensor) -> Tensor:
    """``silu(h[:, :I]) * h[:, I:]`` — the SwiGLU activation between the two expert GEMMs.

    vLLM fuses this into a CUDA op (``apply_moe_activation``, ``fused_moe.py:1798``); here it is
    two torch ops on a ``(n_rows, 2I)`` tensor. At prefill 4k that tensor is 32768x1536 — 96 MiB in
    bf16, read once and written half back, which is HBM traffic the fused op would not spend. It
    is a named divergence, not an oversight: fusing it changes nothing about the grouped GEMM the
    rung is measuring, and un-fusing makes the intermediate inspectable in a test.
    """
    if h.ndim != 2 or h.shape[1] % 2:
        raise ValueError(f"{RUNG}: silu_and_mul expects (rows, 2I), got {tuple(h.shape)}")
    gate, up = h.chunk(2, dim=-1)
    return F.silu(gate) * up


def _check_weights(x: Tensor, w1: Tensor, w2: Tensor) -> tuple[int, int, int]:
    """``(n_experts, intermediate, hidden)`` after checking the layout the whole file assumes."""
    if x.ndim != 2:
        raise ValueError(f"{RUNG}: x must be (n_tokens, hidden), got {tuple(x.shape)}")
    if w1.ndim != 3 or w2.ndim != 3:
        raise ValueError(
            f"{RUNG}: w1 must be (E, 2I, H) and w2 (E, H, I), got {tuple(w1.shape)} and {tuple(w2.shape)}"
        )
    n_experts, two_i, hidden = (int(v) for v in w1.shape)
    if two_i % 2:
        raise ValueError(f"{RUNG}: w1's second axis is {two_i}, must be 2*intermediate")
    inter = two_i // 2
    if tuple(int(v) for v in w2.shape) != (n_experts, hidden, inter):
        raise ValueError(f"{RUNG}: w2 is {tuple(w2.shape)}, expected {(n_experts, hidden, inter)}")
    if int(x.shape[1]) != hidden:
        raise ValueError(f"{RUNG}: x's hidden {int(x.shape[1])} != w1's {hidden}")
    return n_experts, inter, hidden


def s_r4_fused_moe(
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    routing: Routing,
    *,
    gemm: GroupedGEMM | None = None,
    config: GroupedGEMMConfig | None = None,
) -> Tensor:
    """The rung. ``(n_tokens, hidden)`` in, the MoE FFN delta out — no residual, no shared expert.

    ``routing`` is taken already computed rather than derived from logits inside, for one reason:
    the floor must be handed the identical ``topk_ids``/``topk_weights``, so that a difference in
    output is a difference in arithmetic and never a difference in which experts were chosen.
    """
    n_experts, inter, hidden = _check_weights(x, w1, w2)
    if routing.n_experts != n_experts:
        raise ValueError(
            f"{RUNG}: routing has {routing.n_experts} experts, weights have {n_experts}"
        )
    if routing.n_tokens != int(x.shape[0]):
        raise ValueError(
            f"{RUNG}: routing is for {routing.n_tokens} tokens, x has {int(x.shape[0])}"
        )
    run = gemm or resolve_gemm(x.device)
    # ONE config for both GEMMs, keyed on the TOKEN count — not on the row count the GEMM actually
    # sees, which is top_k times larger. That is vLLM's rule (`config = get_config_func(M)` with
    # `M = num_tokens`, fused_moe.py:1737, used for both dispatches), and matching it is what makes
    # the comparison tile-for-tile. Keying on rows instead is a defensible different choice and is
    # one `config=` away — but it must be a choice, not a default that quietly differs.
    cfg = config or default_config(
        n_tokens=routing.n_tokens, n_experts=n_experts, n_out=2 * inter, k_dim=hidden
    )

    p = build_permutation(routing.topk_ids, n_experts)
    rows = gather_tokens(x, p)  # (n_rows, H) — the physical permutation vLLM does not do
    h = run(rows, w1, p.group_offsets, config=cfg).to(x.dtype)  # (n_rows, 2I)
    act = silu_and_mul(h)  # (n_rows, I)
    y_rows = run(act, w2, p.group_offsets, config=cfg).to(x.dtype)  # (n_rows, H)
    return combine(y_rows, p, routing.topk_weights)


def reference_moe(x: Tensor, w1: Tensor, w2: Tensor, routing: Routing) -> Tensor:
    """THE ORACLE. fp32, one expert at a time, gathered by mask — and no permutation anywhere.

    Structurally different from the thing it checks, which is the point: it never builds a
    permutation, never computes a group offset, and never tiles. It reproduces the pattern of
    ``src/scratch_llm/moe.py``'s CPU loop (gather → expert → ``index_add`` scatter), which has been
    in the tree since A1.1 and is the shape of every textbook MoE forward.
    """
    n_experts, _inter, hidden = _check_weights(x, w1, w2)
    if routing.n_experts != n_experts:
        raise ValueError(
            f"{RUNG}: routing has {routing.n_experts} experts, weights have {n_experts}"
        )
    out = torch.zeros((int(x.shape[0]), hidden), dtype=torch.float32, device=x.device)
    ids = routing.topk_ids.to(torch.int64)
    for e in range(n_experts):
        hit = (ids == e).nonzero(as_tuple=False)  # (m, 2) rows of (token, slot)
        if hit.numel() == 0:
            continue  # an expert nobody chose. Normal at decode; the loop must not stumble.
        tok, slot = hit[:, 0], hit[:, 1]
        xe = x.index_select(0, tok).float()
        gate_up = xe @ w1[e].float().transpose(0, 1)
        gate, up = gate_up.chunk(2, dim=-1)
        ye = (F.silu(gate) * up) @ w2[e].float().transpose(0, 1)
        out.index_add_(0, tok, ye * routing.topk_weights[tok, slot].float().unsqueeze(-1))
    return out.to(x.dtype)


@dataclass(frozen=True)
class MoETrace:
    """What the layer did, in numbers a profile can be checked against — no timing in here.

    ``as_row()`` is what the bench writes into its JSON row: the per-expert load the plan asks for,
    the tile counts (how much of the grid is padding), and the byte and FLOP models that say which
    wall each point should be sitting against.
    """

    load: ExpertLoad
    permutation: Permutation
    config: GroupedGEMMConfig
    hidden: int
    intermediate: int
    elem_bytes: int

    @property
    def n_rows(self) -> int:
        return self.permutation.n_rows

    @property
    def flops(self) -> float:
        """``6 * n_rows * H * I`` — 2·(n_rows·H·2I) for the gate/up GEMM, 2·(n_rows·I·H) for down."""
        return 6.0 * self.n_rows * self.hidden * self.intermediate

    @property
    def weight_bytes(self) -> float:
        """Expert weights that must cross HBM: 3·I·H per expert that got at least one row.

        The decode wall. At B=64, top_k 8 and 128 experts, essentially every expert is touched, so
        this is ~1.2 GB per layer regardless of how few tokens each one serves — the GEMM is
        streaming weights, not doing arithmetic.
        """
        touched = self.load.n_experts - self.load.n_empty
        return 3.0 * self.intermediate * self.hidden * self.elem_bytes * touched

    @property
    def permute_bytes(self) -> float:
        """HBM the permutation and its inverse spend: gather (read+write) plus combine (read+write)."""
        rows = self.n_rows * self.hidden * self.elem_bytes
        return 2.0 * rows + rows + self.load.n_tokens * self.hidden * self.elem_bytes

    @property
    def tile_padding_ratio(self) -> float:
        """Grid tiles launched over tiles that carry a row — 1.0 is no waste. The decode tax."""
        exact = exact_m_tiles(self.permutation.group_offsets, self.config.block_m)
        bound = max_m_tiles(self.n_rows, self.load.n_experts, self.config.block_m)
        return bound / max(exact, 1)

    @property
    def row_padding_ratio(self) -> float:
        """Rows in the m-tiles over rows that exist. At decode a group of 4 rows still costs a
        whole ``block_m`` tile, so this is the number that says why decode is not compute-bound."""
        exact = exact_m_tiles(self.permutation.group_offsets, self.config.block_m)
        return exact * self.config.block_m / max(self.n_rows, 1)

    def as_row(self) -> dict[str, object]:
        return {
            "n_rows": self.n_rows,
            "flops": self.flops,
            "weight_bytes": self.weight_bytes,
            "permute_bytes": self.permute_bytes,
            "exact_m_tiles": exact_m_tiles(self.permutation.group_offsets, self.config.block_m),
            "max_m_tiles": max_m_tiles(self.n_rows, self.load.n_experts, self.config.block_m),
            "tile_padding_ratio": self.tile_padding_ratio,
            "row_padding_ratio": self.row_padding_ratio,
            "config": self.config.as_row(),
            "load": self.load.as_row(),
        }


def trace_moe(
    routing: Routing,
    *,
    hidden: int,
    intermediate: int,
    elem_bytes: int = 2,
    config: GroupedGEMMConfig | None = None,
) -> MoETrace:
    """Build the trace for one routing decision. Pure index arithmetic — no GPU, no timing."""
    load = expert_load(routing.topk_ids, routing.n_experts)
    problems = load.validate()
    if problems:
        raise ValueError(f"{RUNG}: load accounting is inconsistent — " + "; ".join(problems))
    cfg = config or default_config(
        n_tokens=routing.n_tokens, n_experts=routing.n_experts, n_out=2 * intermediate, k_dim=hidden
    )
    return MoETrace(
        load=load,
        permutation=build_permutation(routing.topk_ids, routing.n_experts),
        config=cfg,
        hidden=hidden,
        intermediate=intermediate,
        elem_bytes=elem_bytes,
    )


__all__ = [
    "GroupedGEMM",
    "MoETrace",
    "reference_moe",
    "resolve_gemm",
    "s_r4_fused_moe",
    "silu_and_mul",
    "torch_grouped_gemm",
    "trace_moe",
]
