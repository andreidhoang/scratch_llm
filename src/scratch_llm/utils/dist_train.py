"""Optimizer-embedded ZeRO-2 for the MuonAdamW hybrid — nanochat's verified distributed shape (A7).

The d20 pretrain (480.4M on 8×H100) is pure data parallelism, but there is deliberately NO DDP
wrapper: local gradients are ``reduce_scatter``-ed straight into the optimizer (each rank
receives the cluster-AVERAGED grads for the shards it owns), the owner runs the update with
rank-local optimizer state, and an ``all_gather`` broadcasts the updated parameters back — ZeRO-2
traffic (1× grads + 1× params per step) instead of DDP-all-reduce + ZeRO-1-broadcast's ~1.5×.

The load-bearing constraint is Muon: Newton–Schulz orthogonalizes a WHOLE matrix, so the Muon
shard granularity is the *matrix*, never the element (classic element-wise ZeRO is incompatible).
Same-shape matrices are stacked to ``(K, A, B)``, zero-padded to a multiple of ``world_size``,
and reduce-scattered so every rank's chunk holds whole matrices. AdamW's update IS elementwise,
so its large tensors shard along dim 0 (zero-padded rows) and its small ones
(< ``SMALL_PARAM_NUMEL`` elements) stay replicated behind a plain ``all_reduce(AVG)`` — shard
bookkeeping is not worth 32 floats of a norm gain.

Gradient clipping lives INSIDE the distributed step: the global ℓ₂ norm is a property of the
*averaged* gradient, so clipping unreduced local grads would clip the wrong norm. Each rank
squares its owned averaged shards (zero pads contribute exactly 0; the replicated small grads
are counted once, on rank 0), an ``all_reduce(SUM)`` yields the true global norm, and the owned
shards are scaled before the update — the same norm-and-scale rule as
:func:`scratch_llm.optim.gradient_clipping`.

Falsifiable invariants (tests/test_dist_train.py, 2-rank gloo on CPU): one distributed step from
identical replicas, given per-rank grads whose average is known, equals one single-process
:class:`~scratch_llm.optim.CombinedOptimizer` step on the averaged grads; after N steps on
different per-rank batches the replicas are BITWISE identical across ranks (every sharded
element's value is computed by exactly one owner and broadcast, and the replicated small-param
updates are deterministic on identical inputs).

Interview question this answers: "Why can't Muon be ZeRO-sharded element-wise like Adam, and
what does embedding ZeRO-2 in the optimizer buy over a DDP wrapper plus a sharded optimizer?"
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn

from scratch_llm.optim import (
    _zeropower_via_newtonschulz5,
    adamw_update_,
    muon_apply_,
    muon_momentum_,
)

# Below this many elements an AdamW tensor stays replicated (one all_reduce, full m/v on every
# rank) instead of row-sharded: the second collective + slice bookkeeping cost more than the
# few KB of redundant state (nanochat draws the same line).
SMALL_PARAM_NUMEL = 1024


def _ceil_to(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


@dataclass
class _MuonShard:
    """One same-shape group of Muon matrices, sharded matrix-granularly across ranks."""

    params: list[Tensor]  # all K matrices, registration order (identical on every rank)
    k_pad: int  # K rounded up to a multiple of world_size (zero-pad slots at the tail)
    chunk: int  # matrices per rank = k_pad // world_size
    owned: list[int]  # global indices this rank owns — real matrices only (< K), contiguous
    momentum: list[Tensor] | None = None  # lazy; one buffer per OWNED matrix (the ZeRO shard)


@dataclass
class _RowShard:
    """One large AdamW tensor, sharded along dim 0 (zero-padded rows at the tail)."""

    param: Tensor
    rows_pad: int  # dim 0 rounded up to a multiple of world_size
    chunk_rows: int  # rows per rank = rows_pad // world_size
    row0: int  # first owned row (global index) = rank * chunk_rows
    n_rows: int  # owned REAL rows (may be 0 on tail ranks when dim0 < world_size)
    m: Tensor | None = None  # lazy; moments exist only for the owned slice
    v: Tensor | None = None


class DistMuonAdamW(torch.optim.Optimizer):
    """Data-parallel MuonAdamW with the ZeRO-2 reduction embedded in ``step()``.

    Reuses the EXACT single-process math — :func:`~scratch_llm.optim.muon_momentum_` →
    Newton–Schulz → :func:`~scratch_llm.optim.muon_apply_` for the 2-D block matrices,
    :func:`~scratch_llm.optim.adamw_update_` for everything else — on reduce-scattered shards,
    then all-gathers the updated parameters so every rank ends the step with identical replicas.

    ``param_groups`` (one "muon" group, one "adamw" group, tagged by an ``algo`` key) are live
    dicts, so ``train()``'s per-step ``group["lr"] = lr`` write reaches both algorithms — the
    same contract as :class:`~scratch_llm.optim.CombinedOptimizer`. ``max_l2_norm > 0`` enables
    the collective gradient clip inside ``step()`` (the caller must then SKIP its local clip);
    the pre-clip global norm lands in ``last_grad_norm`` for logging.

    Without an initialized process group this degenerates to the plain single-process
    optimizer (world size 1: reductions are copies), which is what the equivalence tests pin.

    Assumes every parameter receives a gradient on every step (``train()`` guarantees it): the
    step count for AdamW bias correction is global, and a param whose grad is None on ALL ranks
    still contributes a zero grad to its shard's update rather than being skipped.

    AdamW bias correction uses the shared step count ``self._t`` (identical across ranks by
    construction — every rank calls ``step()`` the same number of times).
    """

    def __init__(
        self,
        muon_params: Iterable[Tensor],
        adamw_params: Iterable[Tensor],
        *,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        muon_momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        rms_scale: float = 0.2,
        max_l2_norm: float = 0.0,
    ) -> None:
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size, self.rank = 1, 0

        muon_list = list(muon_params)
        adamw_list = list(adamw_params)
        for p in muon_list:
            if p.ndim != 2:
                raise ValueError(
                    f"Muon shards are matrix-granular; got a {p.ndim}-D tensor of shape "
                    f"{tuple(p.shape)} in muon_params — route non-2-D params to adamw_params "
                    "(see split_muon_adamw_params)"
                )
        groups: list[dict[str, Any]] = []
        if muon_list:
            groups.append(
                {
                    "params": muon_list,
                    "algo": "muon",
                    "lr": lr,
                    "momentum": muon_momentum,
                    "nesterov": nesterov,
                    "ns_steps": ns_steps,
                    "weight_decay": weight_decay,
                    "rms_scale": rms_scale,
                }
            )
        if adamw_list:
            groups.append(
                {
                    "params": adamw_list,
                    "algo": "adamw",
                    "lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay,
                }
            )
        super().__init__(groups, defaults={})
        self.max_l2_norm = max_l2_norm
        self.last_grad_norm: float | None = None  # pre-clip global norm (set when clipping)
        self._t = 0  # shared AdamW bias-correction step count (see class docstring)

        # The sharding plan is derived from parameter ORDER and SHAPES only, both identical on
        # every rank (same model, same construction) — so all ranks agree on ownership without
        # communicating (the same determinism argument as zero1.partition_by_numel).
        by_shape: dict[tuple[int, int], list[Tensor]] = {}
        for p in muon_list:
            by_shape.setdefault((p.shape[0], p.shape[1]), []).append(p)
        self._muon_shards: list[_MuonShard] = []
        for mats in by_shape.values():
            k_pad = _ceil_to(len(mats), self.world_size)
            chunk = k_pad // self.world_size
            owned = [g for g in range(self.rank * chunk, (self.rank + 1) * chunk) if g < len(mats)]
            self._muon_shards.append(_MuonShard(mats, k_pad, chunk, owned))

        self._adamw_small: list[Tensor] = []
        self._row_shards: list[_RowShard] = []
        for p in adamw_list:
            if p.numel() < SMALL_PARAM_NUMEL:
                self._adamw_small.append(p)
                continue
            rows_pad = _ceil_to(p.shape[0], self.world_size)
            chunk_rows = rows_pad // self.world_size
            row0 = self.rank * chunk_rows
            n_rows = max(0, min(p.shape[0] - row0, chunk_rows))
            self._row_shards.append(_RowShard(p, rows_pad, chunk_rows, row0, n_rows))
        self._small_m: list[Tensor | None] = [None] * len(self._adamw_small)
        self._small_v: list[Tensor | None] = [None] * len(self._adamw_small)

    # ---------------------------------------------------------------------- comm primitives

    def _reduce_scatter_avg(self, out: Tensor, stack: Tensor) -> None:
        """Rank r's chunk of mean-over-ranks(``stack``) lands in ``out`` — ZeRO-2's gradient
        reduction. gloo (the CPU test backend) and NCCL (the 8×H100 path) both implement the
        fused op + AVG natively on torch ≥ 2.12."""
        if self.world_size == 1:
            out.copy_(stack)
            return
        dist.reduce_scatter_tensor(out, stack, op=dist.ReduceOp.AVG)

    def _all_gather(self, full: Tensor, mine: Tensor) -> None:
        """Gather each rank's ``mine`` chunk into ``full`` (rank-major) — ZeRO-2's param
        broadcast; every replica ends with the same bytes."""
        if self.world_size == 1:
            full.copy_(mine)
            return
        dist.all_gather_into_tensor(full, mine)

    # ------------------------------------------------------------------------------- step

    def _hyper(self, algo: str) -> dict[str, Any] | None:
        for group in self.param_groups:
            if group["algo"] == algo:
                return group
        return None

    def _clip_(self, muon_red: list[Tensor], row_red: list[Tensor]) -> None:
        """Global-ℓ₂ clip of the AVERAGED grads, in place on the reduced shards + the
        replicated small grads. Same norm and scale rule as ``optim.gradient_clipping``, moved
        after the reduction because the clip norm is a property of the averaged gradient.
        Pad slots are exact zeros (zero-padded stacks, AVG of zeros) so summing whole chunks is
        safe; small grads are identical on every rank, so rank 0 counts them once and every
        rank scales its own copy by the same factor."""
        anchor = (muon_red or row_red or self._adamw_small)[0]
        sq = torch.zeros(1, dtype=anchor.dtype, device=anchor.device)
        for reduced in muon_red:
            sq += reduced.pow(2).sum()
        for reduced in row_red:
            sq += reduced.pow(2).sum()
        if self.rank == 0:
            for p in self._adamw_small:
                if p.grad is not None:
                    sq += p.grad.pow(2).sum()
        if self.world_size > 1:
            dist.all_reduce(sq, op=dist.ReduceOp.SUM)
        total_norm = sq.sqrt()
        self.last_grad_norm = float(total_norm.item())
        if self.last_grad_norm > self.max_l2_norm:
            scale = self.max_l2_norm / (total_norm + 1e-6)  # gradient_clipping's eps
            for reduced in muon_red:
                reduced.mul_(scale)
            for reduced in row_red:
                reduced.mul_(scale)
            for p in self._adamw_small:
                if p.grad is not None:
                    p.grad.mul_(scale)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # ---- phase 1: reduce. Zero-padded stacks so every reduce_scatter chunk holds whole
        # matrices (Muon) / whole rows (AdamW); pad slots land as exact zeros on their owner.
        # (Sync per tensor for now — overlapping the three phases nanochat-style is a profiled
        # A8/perf refinement, not a correctness need.)
        muon_red: list[Tensor] = []
        for shard in self._muon_shards:
            ref = shard.params[0]
            stack = torch.zeros((shard.k_pad, *ref.shape), dtype=ref.dtype, device=ref.device)
            for i, p in enumerate(shard.params):
                if p.grad is not None:
                    stack[i].copy_(p.grad)
            out = torch.empty((shard.chunk, *ref.shape), dtype=ref.dtype, device=ref.device)
            self._reduce_scatter_avg(out, stack)
            muon_red.append(out)
        row_red: list[Tensor] = []
        for rshard in self._row_shards:
            p = rshard.param
            stack = torch.zeros((rshard.rows_pad, *p.shape[1:]), dtype=p.dtype, device=p.device)
            if p.grad is not None:
                stack[: p.shape[0]].copy_(p.grad)
            out = torch.empty((rshard.chunk_rows, *p.shape[1:]), dtype=p.dtype, device=p.device)
            self._reduce_scatter_avg(out, stack)
            row_red.append(out)
        if self.world_size > 1:
            for p in self._adamw_small:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        # ---- phase 2: collective grad clip on the averaged grads (see _clip_).
        if self.max_l2_norm > 0:
            self._clip_(muon_red, row_red)

        # ---- phase 3: owner updates, rank-local state — the single-process math, verbatim.
        self._t += 1
        muon_hp = self._hyper("muon")
        if muon_hp is not None:
            for shard, red in zip(self._muon_shards, muon_red, strict=True):
                if shard.momentum is None:
                    shard.momentum = [torch.zeros_like(shard.params[g]) for g in shard.owned]
                for buf, g in zip(shard.momentum, shard.owned, strict=True):
                    local = g - self.rank * shard.chunk
                    g_eff = muon_momentum_(
                        red[local], buf, muon_hp["momentum"], muon_hp["nesterov"]
                    )
                    ortho = _zeropower_via_newtonschulz5(g_eff, muon_hp["ns_steps"])
                    muon_apply_(
                        shard.params[g],
                        ortho,
                        lr=muon_hp["lr"],
                        weight_decay=muon_hp["weight_decay"],
                        rms_scale=muon_hp["rms_scale"],
                    )
        adamw_hp = self._hyper("adamw")
        if adamw_hp is not None:
            beta1, beta2 = adamw_hp["betas"]
            for rshard, red in zip(self._row_shards, row_red, strict=True):
                if rshard.n_rows == 0:
                    continue  # dim0 < world_size: this rank owns only pad rows
                if rshard.m is None or rshard.v is None:
                    rshard.m = torch.zeros_like(red[: rshard.n_rows])
                    rshard.v = torch.zeros_like(red[: rshard.n_rows])
                p_slice = rshard.param[rshard.row0 : rshard.row0 + rshard.n_rows]  # a view
                adamw_update_(
                    p_slice,
                    red[: rshard.n_rows],
                    rshard.m,
                    rshard.v,
                    self._t,
                    lr=adamw_hp["lr"],
                    beta1=beta1,
                    beta2=beta2,
                    eps=adamw_hp["eps"],
                    weight_decay=adamw_hp["weight_decay"],
                )
            for i, p in enumerate(self._adamw_small):
                # Replicated update: identical averaged grads + identical state on every rank
                # keep the replicas bitwise in sync without any gather.
                if p.grad is None:
                    continue
                if self._small_m[i] is None:
                    self._small_m[i] = torch.zeros_like(p)
                    self._small_v[i] = torch.zeros_like(p)
                m, v = self._small_m[i], self._small_v[i]
                assert m is not None and v is not None
                adamw_update_(
                    p,
                    p.grad,
                    m,
                    v,
                    self._t,
                    lr=adamw_hp["lr"],
                    beta1=beta1,
                    beta2=beta2,
                    eps=adamw_hp["eps"],
                    weight_decay=adamw_hp["weight_decay"],
                )

        if self.world_size == 1:
            return loss

        # ---- phase 4: all_gather the updated shards back to every replica. Each sharded
        # element's post-step bytes come from exactly one owner's computation, so replicas are
        # bitwise identical after the copy-back (pad slots — global index ≥ K / row ≥ dim0 —
        # are simply never copied back).
        for shard in self._muon_shards:
            ref = shard.params[0]
            mine = torch.zeros((shard.chunk, *ref.shape), dtype=ref.dtype, device=ref.device)
            for g in shard.owned:
                mine[g - self.rank * shard.chunk].copy_(shard.params[g])
            full = torch.empty((shard.k_pad, *ref.shape), dtype=ref.dtype, device=ref.device)
            self._all_gather(full, mine)
            for g, p in enumerate(shard.params):
                p.copy_(full[g])
        for rshard in self._row_shards:
            p = rshard.param
            mine = torch.zeros((rshard.chunk_rows, *p.shape[1:]), dtype=p.dtype, device=p.device)
            if rshard.n_rows:
                mine[: rshard.n_rows].copy_(p[rshard.row0 : rshard.row0 + rshard.n_rows])
            full = torch.empty((rshard.rows_pad, *p.shape[1:]), dtype=p.dtype, device=p.device)
            self._all_gather(full, mine)
            p.copy_(full[: p.shape[0]])
        return loss

    # ------------------------------------------------------------------------ checkpointing

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        """This RANK's shard state only: Muon momentum for owned matrices, AdamW moments for
        owned rows, the replicated small-param moments, and the shared step count.

        Distributed optimizer state is rank-local by design — a resume needs the same
        (world_size, rank) layout, validated in :meth:`load_state_dict`. For a single-file,
        topology-INDEPENDENT save (resume on any world size, or extract the model for serving)
        use :meth:`consolidated_state_dict` / :meth:`load_consolidated` (A8)."""
        return {
            "t": self._t,
            "world_size": self.world_size,
            "rank": self.rank,
            "muon_momentum": [
                None if s.momentum is None else [b.clone() for b in s.momentum]
                for s in self._muon_shards
            ],
            "adamw_rows": [
                {"m": s.m.clone(), "v": s.v.clone()}
                if s.m is not None and s.v is not None
                else None
                for s in self._row_shards
            ],
            "adamw_small": [
                {"m": m.clone(), "v": v.clone()} if m is not None and v is not None else None
                for m, v in zip(self._small_m, self._small_v, strict=True)
            ],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict["world_size"] != self.world_size or state_dict["rank"] != self.rank:
            raise ValueError(
                "DistMuonAdamW state is rank-local: restore it on the same (world_size, rank) "
                f"it was saved from — saved ({state_dict['world_size']}, {state_dict['rank']}), "
                f"this process ({self.world_size}, {self.rank}). To resume on a different world "
                "size, save with consolidated_state_dict and load with load_consolidated."
            )
        self._t = int(state_dict["t"])
        for shard, bufs in zip(self._muon_shards, state_dict["muon_momentum"], strict=True):
            shard.momentum = None if bufs is None else [b.clone() for b in bufs]
        for rshard, mv in zip(self._row_shards, state_dict["adamw_rows"], strict=True):
            rshard.m = None if mv is None else mv["m"].clone()
            rshard.v = None if mv is None else mv["v"].clone()
        small = state_dict["adamw_small"]
        self._small_m = [None if mv is None else mv["m"].clone() for mv in small]
        self._small_v = [None if mv is None else mv["v"].clone() for mv in small]

    # ------------------------------------------------------ consolidated checkpoint (A8)

    @torch.no_grad()
    def consolidated_state_dict(self) -> dict[str, Any] | None:
        """The FULL, un-sharded, topology-INDEPENDENT optimizer state — the format you ship a
        model in or resume on a different world size (the rank-local :meth:`state_dict` above is
        only good for a same-topology resume). Every rank runs the SAME ``all_gather``s in the
        SAME order (collective-correct — no rank-0-only collective); rank 0 assembles and returns
        the dict, every other rank returns ``None`` after participating in the gathers.

        Layout (all state grouped exactly like ``self._muon_shards`` / ``self._row_shards`` /
        ``self._adamw_small`` — those groupings are derived from parameter ORDER + SHAPE only, so
        they are identical on every world size, which is what makes the format topology-free):

        - ``muon_momentum[j]`` — ``None`` (uninitialised) or a list of ``K_j`` full ``(A, B)``
          SGD-momentum buffers, one per WHOLE matrix of shape-group ``j``, in registration order.
          The ``all_gather`` reunites the matrix-granular ZeRO-2 shards and the tail zero-pad
          slots (global index ≥ ``K_j``) are dropped — the un-pad.
        - ``adamw_rows[i]`` — ``None`` or ``{"m", "v"}`` holding the i-th large param's FULL
          ``(dim0, …)`` moments; the dim-0 shards are gathered and the zero-pad rows
          (global row ≥ ``dim0``) dropped.
        - ``adamw_small[i]`` — ``None`` or ``{"m", "v"}``; small params are replicated (identical
          on every rank), so rank 0's local copy is authoritative — no collective needed.
        - ``t`` — the shared AdamW bias-correction step count.

        Mapping onto a single-process ``CombinedOptimizer([Muon(muon_params), AdamW(adamw_params)])``
        (documented so the format is not opaque; the round-trip is *proven* via a world-size-1
        :meth:`load_consolidated` in the tests): group ``muon_params`` by shape in first-occurrence
        order (exactly ``split_muon_adamw_params`` → ``by_shape`` order) — the k-th param of group j
        takes ``Muon.state[p]["momentum_buffer"] = muon_momentum[j][k]``; the large adamw params (in
        order) take ``adamw_rows[i]["m"/"v"]`` as their ``m``/``v`` and the small ones
        ``adamw_small[i]``; every adamw param's ``state["step"] = t``.
        """
        # ``self._t`` is identical across ranks (every rank steps together), so gating the gathers
        # on it keeps every rank's collective schedule in lockstep. Before the first step nothing
        # is initialised (momentum/moments are lazy) → an all-``None`` consolidated dict.
        has_state = self._t > 0
        is0 = self.rank == 0

        muon_full: list[list[Tensor] | None] = []
        for shard in self._muon_shards:
            if not has_state:
                muon_full.append(None)
                continue
            ref = shard.params[0]
            mine = torch.zeros((shard.chunk, *ref.shape), dtype=ref.dtype, device=ref.device)
            if shard.momentum:  # empty list on ranks that own only pad slots → contribute zeros
                for buf, g in zip(shard.momentum, shard.owned, strict=True):
                    mine[g - self.rank * shard.chunk].copy_(buf)
            full = torch.empty((shard.k_pad, *ref.shape), dtype=ref.dtype, device=ref.device)
            self._all_gather(full, mine)  # collective — every rank
            muon_full.append([full[g].clone() for g in range(len(shard.params))] if is0 else None)

        rows_full: list[dict[str, Tensor] | None] = []
        for rshard in self._row_shards:
            if not has_state:
                rows_full.append(None)
                continue
            p = rshard.param
            tail = p.shape[1:]
            m_mine = torch.zeros((rshard.chunk_rows, *tail), dtype=p.dtype, device=p.device)
            v_mine = torch.zeros((rshard.chunk_rows, *tail), dtype=p.dtype, device=p.device)
            if rshard.n_rows and rshard.m is not None and rshard.v is not None:
                m_mine[: rshard.n_rows].copy_(rshard.m)
                v_mine[: rshard.n_rows].copy_(rshard.v)
            m_full = torch.empty((rshard.rows_pad, *tail), dtype=p.dtype, device=p.device)
            v_full = torch.empty((rshard.rows_pad, *tail), dtype=p.dtype, device=p.device)
            self._all_gather(m_full, m_mine)  # collective — every rank
            self._all_gather(v_full, v_mine)
            rows_full.append(
                {"m": m_full[: p.shape[0]].clone(), "v": v_full[: p.shape[0]].clone()}
                if is0
                else None
            )

        if not is0:
            return None
        return {
            "format": "consolidated",
            "t": self._t,
            "world_size": self.world_size,  # the SAVING topology — informational only
            "muon_momentum": muon_full,
            "adamw_rows": rows_full,
            "adamw_small": [
                {"m": m.clone(), "v": v.clone()} if m is not None and v is not None else None
                for m, v in zip(self._small_m, self._small_v, strict=True)
            ],
        }

    def load_consolidated(
        self,
        state: dict[str, Any],
        world_size: int | None = None,
        rank: int | None = None,
    ) -> None:
        """Re-shard a :meth:`consolidated_state_dict` back onto THIS process's topology — pure
        local slicing, NO collectives (each rank keeps only the matrices / rows it owns). A d20
        saved on 8 ranks resumes on 8; the same consolidated file resumes at world size 1 (owned =
        everything) so the full model params can be extracted for serving. ``world_size``/``rank``,
        if given, are asserted to match this optimizer's own topology (a guard against loading a
        shard plan built for a different layout — the reshard target is always ``self``).

        Inverse of the save's un-pad: the consolidated tensors carry exactly ``K`` matrices /
        ``dim0`` rows, and this rank re-selects its owned slice (Muon: ``shard.owned`` matrices;
        AdamW: rows ``[row0, row0 + n_rows)``); pad-only ranks get ``None`` state, exactly as a
        fresh :meth:`step` would leave them.
        """
        if world_size is not None and world_size != self.world_size:
            raise ValueError(
                f"load_consolidated reshards onto THIS optimizer's topology (world_size "
                f"{self.world_size}); got world_size={world_size}"
            )
        if rank is not None and rank != self.rank:
            raise ValueError(
                f"load_consolidated reshards onto THIS optimizer's rank ({self.rank}); "
                f"got rank={rank}"
            )
        if state.get("format") != "consolidated":
            raise ValueError(
                "load_consolidated expects a consolidated_state_dict (format='consolidated'); "
                "for a rank-local shard resume use load_state_dict"
            )
        self._t = int(state["t"])
        for shard, group_mom in zip(self._muon_shards, state["muon_momentum"], strict=True):
            if group_mom is None:
                shard.momentum = None
            else:
                shard.momentum = [group_mom[g].clone() for g in shard.owned]
        for rshard, mv in zip(self._row_shards, state["adamw_rows"], strict=True):
            if mv is None or rshard.n_rows == 0:
                rshard.m = None
                rshard.v = None
            else:
                r0, r1 = rshard.row0, rshard.row0 + rshard.n_rows
                rshard.m = mv["m"][r0:r1].clone()
                rshard.v = mv["v"][r0:r1].clone()
        small = state["adamw_small"]
        self._small_m = [None if mv is None else mv["m"].clone() for mv in small]
        self._small_v = [None if mv is None else mv["v"].clone() for mv in small]


@torch.no_grad()
def assert_model_replicas_identical(model: nn.Module) -> None:
    """Collectively verify every rank holds byte-identical model weights — the guard to run
    BEFORE a consolidated save, whose model payload is rank 0's ``state_dict`` alone. Each
    parameter is ``all_gather``-ed and rank r compared to rank 0; a mismatch raises the SAME
    error on EVERY rank (all ranks call the same ``all_gather``s and see the same gathered
    tensors, so they raise together at the same parameter — no lone raiser, no hang). A no-op
    without an initialised process group (single-process replicas are trivially identical).

    Catches a silent replica desync (a broken ``all_gather`` in :meth:`DistMuonAdamW.step`, a
    stray per-rank weight write) at save time, rather than shipping a subtly wrong model."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    for name, p in model.named_parameters():
        gathered = [torch.empty_like(p) for _ in range(world_size)]
        dist.all_gather(gathered, p.data.contiguous())
        for r in range(1, world_size):
            if not torch.equal(gathered[0], gathered[r]):
                raise RuntimeError(
                    f"model replicas diverged at parameter {name!r}: rank {r} disagrees with "
                    "rank 0 — refusing a consolidated save whose model state is rank 0's alone "
                    "(a desync bug; check the all_gather copy-back in DistMuonAdamW.step)"
                )


__all__ = ["SMALL_PARAM_NUMEL", "DistMuonAdamW", "assert_model_replicas_identical"]
