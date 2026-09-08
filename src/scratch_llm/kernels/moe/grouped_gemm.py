"""S1/S-R4 — the grouped GEMM's host side: the tile plan, the offsets contract, and the oracle.

A grouped GEMM is ``E`` independent matrix multiplies whose left operands are contiguous slices of
one tall matrix::

    C[group_offsets[e] : group_offsets[e+1], :]  =  A[group_offsets[e] : group_offsets[e+1], :] @ B[e].T

with ``A: (M, K)``, ``B: (E, N, K)``, ``C: (M, N)``. The ``(E, N, K)`` weight layout is vLLM's
(``fused_moe.py:1707`` reads ``E, N, _ = w1.size()``), so one set of weight tensors feeds both this
rung and its floor and no transpose can creep in between the two measurements.

This module holds every decision that does not need silicon:

* :func:`validate_group_offsets` — the contract the kernel is entitled to assume;
* :func:`max_m_tiles` / :func:`exact_m_tiles` — the launch grid, and the sync it does or does not
  cost. The distinction is the whole reason this is a separate function and not an inline
  ``cdiv``: the exact count needs the offsets on the host, and a host sync inside a decode step is
  a measurable latency, so the launcher uses the bound and the kernel skips the empty tiles;
* :func:`default_config` — the tile shape, mirroring the branch of vLLM's ``get_default_config``
  (``fused_moe.py:1374-1414``) that its bf16 path takes, so the comparison is tile-for-tile at
  both of the rung's points and any gap is the kernel, not the tiling;
* :func:`reference_grouped_gemm` — the fp32 oracle. A Python loop over experts. Slow, obviously
  correct, and the thing the Triton kernel is judged against.

The Triton kernel itself is ``grouped_gemm_triton.py`` and its mainloop is Huy's.

Spec: ``experiments/S1/S-R4/spec.md``   ·   Map: ``experiments/S1/S-R4/map.md``
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from scratch_llm.kernels.moe.routing import RUNG


def cdiv(a: int, b: int) -> int:
    """Ceiling division, spelled out once so no call site improvises ``(a + b - 1) // b``."""
    if b <= 0:
        raise ValueError(f"{RUNG}: cdiv by {b}")
    return -(-a // b)


@dataclass(frozen=True)
class GroupedGEMMConfig:
    """One launch's tile shape. Huy owns the final numbers; this is a defensible starting point.

    Threaded through every entry point as an argument rather than read from a module global, so a
    sweep can vary it without editing a source file and the bench can record which one produced a
    row.
    """

    block_m: int = 64
    block_n: int = 64
    block_k: int = 64
    num_warps: int = 4
    num_stages: int = 3

    def validate(self) -> list[str]:
        bad: list[str] = []
        for name, v in (
            ("block_m", self.block_m),
            ("block_n", self.block_n),
            ("block_k", self.block_k),
        ):
            if v < 16 or v & (v - 1):
                bad.append(f"{name}={v} must be a power of two >= 16 (Triton tile constraint)")
        if self.num_warps not in (1, 2, 4, 8, 16):
            bad.append(f"num_warps={self.num_warps} is not a legal Triton warp count")
        if self.num_stages < 1:
            bad.append(f"num_stages={self.num_stages} must be >= 1")
        return bad

    def as_row(self) -> dict[str, int]:
        return {
            "block_m": self.block_m,
            "block_n": self.block_n,
            "block_k": self.block_k,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }


def default_config(*, n_tokens: int, n_experts: int, n_out: int, k_dim: int) -> GroupedGEMMConfig:
    """The tile shape vLLM's untuned bf16 path would pick at this shape, restated.

    Restated rather than imported: this must be readable next to the kernel it launches, and vLLM
    reaches it through ``try_get_optimal_moe_config`` → ``get_default_config``
    (``fused_moe.py:1422-1451`` → ``:1299``), keyed on ``M = num_tokens`` (NOT ``num_tokens *
    top_k`` — the row count the GEMM actually sees is ``top_k`` times larger than the number this
    branch is chosen by, which is worth knowing before reading a profile).

    There is no ``E=128,N=768,device_name=NVIDIA_H100_80GB_HBM3.json`` in vLLM's config directory,
    so at Qwen3-30B-A3B's shape on an H100 the floor takes this default branch too — the
    comparison is untuned-vs-untuned, which is the honest one for a first version.
    """
    if n_tokens <= 32:
        block_m = 16
    elif n_tokens <= 96:
        block_m = 32
    elif n_tokens <= 512:
        block_m = 64
    else:
        block_m = 128
    block_n = 64 if n_tokens <= 64 else 128
    block_k = 128 if n_tokens <= 64 else 64
    num_warps = 4 if n_tokens <= 128 else 8
    num_stages = 4 if n_tokens <= 32 else 3
    # A tile wider than the problem is pure waste; clamp so a small N (Qwen3's 768) does not run
    # half-empty N tiles. vLLM does not clamp — it masks — and at N=768 with block_n=128 the
    # question never arises; it does arise for the second GEMM's K=768 tail.
    block_n = min(block_n, max(16, 1 << (max(n_out, 16) - 1).bit_length()))
    block_k = min(block_k, max(16, 1 << (max(k_dim, 16) - 1).bit_length()))
    # n_experts is in the signature and unused on purpose: vLLM's chooser ignores it too, and the
    # obvious next tiling — one that shrinks BLOCK_M when the expected rows per expert are few —
    # needs it. Keeping it here means that change does not move every call site.
    del n_experts
    return GroupedGEMMConfig(block_m, block_n, block_k, num_warps, num_stages)


def validate_group_offsets(group_offsets: Tensor, *, n_rows: int, n_experts: int) -> list[str]:
    """Everything the kernel is allowed to assume about the offsets, checked. Empty list ⇒ valid.

    Stated as a list of sentences rather than assertions so a test can print all of them at once,
    and so the launcher can refuse with a message that names the actual violation.
    """
    bad: list[str] = []
    if group_offsets.ndim != 1 or int(group_offsets.numel()) != n_experts + 1:
        bad.append(
            f"group_offsets must be 1-D of length n_experts+1 = {n_experts + 1}, got {tuple(group_offsets.shape)}"
        )
        return bad
    if int(group_offsets[0]) != 0:
        bad.append(f"group_offsets[0] = {int(group_offsets[0])}, must be 0")
    if int(group_offsets[-1]) != n_rows:
        bad.append(f"group_offsets[-1] = {int(group_offsets[-1])}, must equal n_rows = {n_rows}")
    if bool((group_offsets[1:] < group_offsets[:-1]).any()):
        first = int(torch.nonzero(group_offsets[1:] < group_offsets[:-1])[0])
        bad.append(
            f"group_offsets is not monotone: entry {first + 1} is smaller than entry {first}"
        )
    return bad


def max_m_tiles(n_rows: int, n_experts: int, block_m: int) -> int:
    """Upper bound on m-tiles across all groups, computable WITHOUT looking at the offsets.

    Each group is tiled independently, so the worst case is every group wasting ``block_m - 1``
    rows of its last tile: ``sum_e cdiv(c_e, BM) <= cdiv(sum_e c_e + E*(BM-1), BM)``. That is the
    same bound vLLM sizes its padded id array with (``moe_align_block_size.py:74``:
    ``topk_ids.numel() + num_experts * (block_size - 1)``).

    Using the bound instead of :func:`exact_m_tiles` costs some empty CTAs that return immediately
    and buys a launch with no device→host copy. At decode the bound is loose — 512 rows over 128
    experts at block_m 32 gives 140 tiles for the ~126 that carry a row — and being loose is still
    cheaper than a sync in a 64-token step. Which way this trade goes is measurable and is one of
    the things the rung's profile settles.
    """
    if n_rows < 0 or n_experts < 1:
        raise ValueError(f"{RUNG}: max_m_tiles(n_rows={n_rows}, n_experts={n_experts})")
    return cdiv(n_rows + n_experts * (block_m - 1), block_m)


def exact_m_tiles(group_offsets: Tensor, block_m: int) -> int:
    """The true tile count: ``sum_e cdiv(count_e, block_m)``. Reads the offsets ⇒ syncs on GPU.

    An empty group contributes ZERO tiles, not one — ``cdiv(0, bm) == 0``. Getting that wrong in
    the other direction (a tile per expert, always) is harmless for correctness and pure waste at
    decode; getting it wrong here, by rounding a group of 1 row down to 0 tiles, silently drops a
    token. The test asserts both directions.
    """
    counts = (group_offsets[1:] - group_offsets[:-1]).to(torch.int64)
    return int(((counts + (block_m - 1)) // block_m).sum())


def reference_grouped_gemm(
    a: Tensor,
    b: Tensor,
    group_offsets: Tensor,
    *,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    """THE ORACLE. ``C[rows of e] = A[rows of e] @ B[e].T``, accumulated in fp32, one group at a time.

    Deliberately the slowest possible correct thing: a Python loop, one ``torch.matmul`` per
    expert, fp32 throughout, no fusion, no masking, no tiles. Every failure mode of the real
    kernel — a tile that reads past its group, an empty group that shifts the ones after it, a
    tail row left unwritten — produces a different answer than this, and none of them produce a
    different answer than each other, which is why the comparison has to be against something with
    no tiling in it at all.
    """
    if a.ndim != 2 or b.ndim != 3:
        raise ValueError(
            f"{RUNG}: expected a (M, K) and b (E, N, K), got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    n_experts, n_out, k_dim = (int(x) for x in b.shape)
    if int(a.shape[1]) != k_dim:
        raise ValueError(f"{RUNG}: a's K = {int(a.shape[1])} != b's K = {k_dim}")
    problems = validate_group_offsets(group_offsets, n_rows=int(a.shape[0]), n_experts=n_experts)
    if problems:
        raise ValueError(f"{RUNG}: bad group_offsets — " + "; ".join(problems))
    out = torch.zeros((int(a.shape[0]), n_out), dtype=torch.float32, device=a.device)
    offs = group_offsets.tolist()
    for e in range(n_experts):
        start, stop = int(offs[e]), int(offs[e + 1])
        if stop == start:
            continue  # an expert with no tokens. Legal, common at decode, and not a special case.
        out[start:stop] = a[start:stop].float() @ b[e].float().transpose(0, 1)
    return out if out_dtype is None else out.to(out_dtype)


__all__ = [
    "GroupedGEMMConfig",
    "cdiv",
    "default_config",
    "exact_m_tiles",
    "max_m_tiles",
    "reference_grouped_gemm",
    "validate_group_offsets",
]
