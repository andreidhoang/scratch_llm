"""k3.model — Kimi K3 / mini-K3 assembly (DELEGATED track, HANDCRAFTED.md Class 3).

This module is the *assembly* — the agent-owned plumbing that composes the hand-built
``core/`` mechanisms (``situ.py``, ``kda.py``, ``gated_mla.py``, ``latent_moe.py``,
``attn_res.py``) into the 93-layer (full) or 12-layer (mini) K3 architecture. Agents own
this file; the human reviews. The math of every silent-bug-prone mechanism lives in
``core/`` — hand-built, line by line, by the human. A wrong residual-add is a *loud* bug
(shape crash) and is assembly-owned; a wrong KDA recurrence is a *silent* bug (loss still
drops, model subtly forgets) and is core-owned. That split *is* the two-citizenship rule
(``../HANDCRAFTED.md``).

Architecture of record (FACTS A2, A11; config.json):
  93 layers = 23 blocks × (3 KDA + 1 Gated-MLA) + 1 terminal MLA  ⇒  69 KDA : 24 MLA.
  Layer 1 keeps a **dense** MLP (``first_k_dense_replace=1``); layers 2..93 are LatentMoE.
  Every layer still has KDA or MLA attention — "dense" is an FFN property, not an attention
  one. Block AttnRes (block size 12) merges block outputs back into the residual stream.
  Fully **NoPE** (FACTS A11): positions are carried data-dependently by the KDA recurrence
  (Kimi Linear §6.1 — the gated delta rule is a multiplicative positional encoding), so MLA
  needs no RoPE; ``qk_rope_head_dim`` in config.json is vestigial.

The assembly owns:
  - the residual stream and the pre-norm wiring (RMSNorm at hidden width);
  - layer dispatch (dense-FFN vs MoE-FFN, KDA vs MLA attention) by 1-indexed layer id;
  - the hybrid recurrent-state / latent-KV cache contract (K6 train; K9 serve);
  - AttnRes source bookkeeping (embedding + preceding block representations);
  - final norm + (optionally tied) LM head;
  - the loss-at-init and overfit-one-batch gates' surface.

The assembly does NOT own (``core/`` hand-builds):
  SiTU-GLU numerics, KDA recurrence + chunkwise form, MLA weight-absorption + gate,
  LatentMoE routing + Quantile Balancing, AttnRes online-softmax merge.

Invariants (gates wired in ``tests/test_k3_model.py`` once ``core/`` lands):
  - **Loss at init ≈ log(vocab):** a fresh ``K3Model`` on uniform targets hits
    log(163840)≈12.01 (full) / log(32768)≈10.40 (mini). Off by > 0.1 ⇒ head/embed/norm/init
    bug (the ``initializer_range=0.02`` contract, FACTS A13).
  - **NoPE autoregressive contract:** perturbing a future token must not change logits at
    earlier positions (causality), bounded by KDA recurrence order, not by an attention mask.
  - **Hybrid-cache equivalence:** prefill (``state=None``) logits ≡ incremental-decode logits,
    within dtype tolerance, on both the KDA and MLA layers.
  - **Param closure:** ``K3Model(k3_full())`` reproduces 2,779,931,837,184 params (FACTS A18,
    gate ``test_k3_param_count.py``); ``mini_k3_d12()`` reproduces 370,303,424.

Interview questions:
  - Why is the *terminal* layer MLA, not KDA? — the final representation must aggregate over
    arbitrarily long history; MLA's softmax is the only path that does exact long-range
    retrieval, where KDA's recurrence has decayed toward noise.
  - Why does layer 1 stay dense-FFN? — the router is unstable on raw embeddings; MoE needs a
    non-trivial representation to route on (DeepSeek-V3 heritage; FACTS A2).
  - Why is NoPE safe here? — see the KDA recurrence: input-modulated decay is a data-dependent
    positional encoding (Kimi Linear §6.1); by the time a token reaches an MLA layer, the KDA
    layers have already written position into the residual stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from scratch_llm.k3.config import K3Config, k3_full, mini_k3_d12

# Assembly-level primitives reused from the A1 substrate (tested, CPU-pure, green). The
# K3-specific norms inside core/ (e.g. the pre-up RMSNorm in LatentMoE at latent width) are
# constructed by those modules; the assembly only norms at the residual-stream width.
from scratch_llm.model import Embedding, Linear, RMSNorm, cross_entropy

if TYPE_CHECKING:
    # Hand-built core/ mechanisms — type-only imports. The runtime imports are deferred to
    # each submodule's __init__ so this scaffold loads before the human authors any core
    # module. As each core module lands (HANDCRAFTED.md build order: situ → kda → gated_mla →
    # latent_moe → attn_res), un-comment its import and swap the _PendingCore in the block.
    # from scratch_llm.k3.core.situ import SiTUGLU, DenseSiTUMLP              # K5 prep
    # from scratch_llm.k3.core.kda import KDALayer, KDAState                  # K2 (critical path)
    # from scratch_llm.k3.core.gated_mla import GatedMLA, MLALatentKV         # K3
    # from scratch_llm.k3.core.latent_moe import LatentMoE, QBStats           # K5
    # from scratch_llm.k3.core.attn_res import BlockAttnRes                   # K4
    pass

__all__ = [
    "K3Model",
    "K3Block",
    "HybridState",
    "KDARecurrenceState",
    "MLALatentKV",
    "build_k3",
    "build_mini_k3",
]


# ---------------------------------------------------------------------------
# Pending-core placeholder — construction works, forward fails LOUDLY.
# ---------------------------------------------------------------------------


class _PendingCore(nn.Module):
    """Placeholder for a hand-built ``core/`` submodule not yet authored.

    Constructs cleanly so the assembly's structure is introspectable (every block, every
    submodule slot visible; pyright/import pass; eventual param introspection works). Raises
    loudly on ``forward`` so no ``nn.Identity`` pass-through ever masquerades as a working
    model — a silent scaffold is exactly the failure mode this repo exists to prevent.

    When the human's ``core/`` module clears its HANDCRAFTED.md rung (delete-test + adversarial
    red-team green), delete the ``_PendingCore`` site and construct the real class there. The
    construction line below each site shows the exact swap.
    """

    def __init__(self, name: str, rung: str) -> None:
        super().__init__()
        self._name = name
        self._rung = rung

    def forward(self, *args: object, **kwargs: object) -> Tensor:
        raise NotImplementedError(
            f"{self._name} is hand-built in core/ (HANDCRAFTED.md {self._rung}). The "
            f"assembly does not substitute for it — author the module, pass its delete-test, "
            f"then replace this _PendingCore with the real class."
        )


# ---------------------------------------------------------------------------
# Hybrid state — the assembly's only novel data structure.
# ---------------------------------------------------------------------------


@dataclass
class KDARecurrenceState:
    """Per-layer per-batch recurrent state for one KDA layer.

    Two pieces: ``S_t`` (the delta-rule matrix state, shape (B, H, d_k, d_v)) and the
    short-conv ring buffer (shape (B, H, conv_kernel, d_k)). Both are **constant-size in
    sequence length** — the K3 1M-context claim rests on this (FACTS A12; ROADMAP K8). The
    ``S_t`` update itself is hand-built in ``core/kda.py``; the assembly only carries the
    state across decode steps.

    Left as a dataclass shell here: the factory that allocates zero-state on the right
    device/dtype ships with ``core/kda.py`` (the human owns the state's numerics — wrong
    zero-init is a silent bug).
    """

    s_t: Tensor | None = None
    conv_state: Tensor | None = None


@dataclass
class MLALatentKV:
    """Per-layer per-batch latent KV cache for one Gated-MLA layer.

    Caches the compressed ``kv_lora_rank`` latent only — the weight-absorption identity
    (``mla.py``) means the per-head K/V never need to be materialized at inference. Grows
    linearly with sequence length; at 1M tokens this is 24 MLA layers × 512 × 2 B per token
    per layer (FACTS A11; ROADMAP K8 reproduces this table from config)."""

    k_latent: Tensor | None = None  # (B, kv_lora_rank, T) — cached compressed key latent
    v_latent: Tensor | None = None  # (B, kv_lora_rank, T) — cached compressed value latent


@dataclass
class HybridState:
    """The K3 hybrid decode cache.

    A constant-size KDA recurrent state per KDA layer, plus a growing latent-KV cache per
    MLA layer. This is the data structure ``serve.py`` paginates (K9) and that K8's
    1M-context memory table counts. Carried as lists indexed by ``layer_id - 1`` so the
    layer loop can address each slot in O(1).

    The assembly owns the *container*; each ``core/`` attention module owns the read/write of
    its own slot's numerics (KDA recurrence update, MLA latent append). Wrong slot bookkeeping
    is a loud bug (index error); wrong recurrence update is silent (core-owned).
    """

    kda_states: list[KDARecurrenceState | None]  # entry per KDA layer; None on MLA layers
    mla_caches: list[MLALatentKV | None]  # entry per MLA layer; None on KDA layers
    lengths: Tensor  # (B,) per-sequence position cursor — bumped once per forward, post-loop

    @classmethod
    def empty(
        cls, cfg: K3Config, batch: int, device: torch.device, dtype: torch.dtype
    ) -> HybridState:
        """Allocate zero hybrid state for a fresh decode stream.

        Raises until ``core/kda.py`` and ``core/gated_mla.py`` land their zero-state
        factories — the human owns the zero-init numerics (a wrong zero-state still "trains"
        but forgets the first token silently)."""
        # TODO(core/kda.py, core/gated_mla.py): once both land, build the per-layer slots:
        #   kda = [KDARecurrenceState.zeros(cfg.kda, batch, device, dtype) if (i+1) in cfg.kda_layers else None for i in range(cfg.num_layers)]
        #   mla = [MLALatentKV.empty(cfg.mla, batch, device, dtype) if (i+1) in cfg.mla_layers else None for i in range(cfg.num_layers)]
        #   lengths = torch.zeros(batch, dtype=torch.long, device=device)
        raise NotImplementedError(
            "HybridState.empty needs core/kda.py + core/gated_mla.py zero-state factories "
            "(HANDCRAFTED.md K2 + K3 rungs). The container is scaffolded; the numerics are not."
        )


# ---------------------------------------------------------------------------
# Block + model assembly.
# ---------------------------------------------------------------------------


class K3Block(nn.Module):
    """One K3 block — standard pre-norm residual, with layer-type dispatch by 1-indexed id.

      x ← x + attn( norm(x) )          # KDA or MLA, by layer id
      x ← x + ffn( norm(x) )           # dense SiTU-MLP on layer ≤ first_k_dense, else LatentMoE

    Attention type and FFN type are **independent** dispatches (FACTS A2): layer 1 is
    KDA-attention + *dense*-MLP; layers 2..93 are KDA/MLA-attention + LatentMoE. The
    residual adds are assembly-owned (loud: a wrong shape crashes); the attn/ffn math is
    ``core/``-owned (silent: wrong recurrence still trains).
    """

    def __init__(self, cfg: K3Config, layer_id: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id  # 1-indexed, matches config.json
        # Attention dispatch — exactly one of KDA / MLA.
        self.is_kda = layer_id in cfg.kda_layers
        self.is_mla = layer_id in cfg.mla_layers
        if self.is_kda == self.is_mla:  # pragma: no cover — config bug
            raise ValueError(
                f"layer {layer_id} must be exactly one of KDA/MLA "
                f"(in kda={self.is_kda}, in mla={self.is_mla}); check build_layer_pattern"
            )
        # FFN dispatch — dense on the leading layer(s), LatentMoE after.
        self.is_dense_ffn = layer_id <= cfg.first_k_dense

        # Pre-norm before attention / FFN — residual-stream width, assembly-owned.
        self.attn_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        # Attention submodule — hand-built in core/. _PendingCore until that rung is PROVEN.
        if self.is_kda:
            # Swap-in line (HANDCRAFTED.md K2 — critical path):
            #   from scratch_llm.k3.core.kda import KDALayer
            #   self.attn = KDALayer(cfg.kda, cfg.hidden_size)
            self.attn: nn.Module = _PendingCore("core/kda.py :: KDALayer", "K2 rung")
        else:  # self.is_mla
            # Swap-in line (HANDCRAFTED.md K3):
            #   from scratch_llm.k3.core.gated_mla import GatedMLA
            #   self.attn = GatedMLA(cfg.mla, cfg.hidden_size)
            self.attn = _PendingCore("core/gated_mla.py :: GatedMLA", "K3 rung")

        # FFN submodule — hand-built in core/.
        if self.is_dense_ffn:
            # Swap-in line (HANDCRAFTED.md K5 prep):
            #   from scratch_llm.k3.core.situ import DenseSiTUMLP
            #   self.ffn = DenseSiTUMLP(cfg.hidden_size, cfg.moe.dense_intermediate, cfg)
            self.ffn: nn.Module = _PendingCore("core/situ.py :: DenseSiTUMLP", "K5 prep")
        else:
            # Swap-in line (HANDCRAFTED.md K5):
            #   from scratch_llm.k3.core.latent_moe import LatentMoE
            #   self.ffn = LatentMoE(cfg.moe, cfg.hidden_size, cfg)
            self.ffn = _PendingCore("core/latent_moe.py :: LatentMoE", "K5 rung")

    def forward(
        self,
        x: Tensor,
        state: HybridState | None,
        block_reps: list[Tensor] | None = None,
    ) -> tuple[Tensor, None]:
        """Pre-norm attention then pre-norm FFN; both residual-add. Returns ``(x, stats)``.

        ``stats`` is ``None`` today; once ``core/latent_moe.py`` lands it carries the
        Quantile-Balancing bias-update signal (aux-loss-free, FACTS A14) up to the trainer.

        ``block_reps`` collects this block's output for AttnRes sources (embedding + each
        block output, FACTS A9) when the caller is wiring K4; ``None`` skips the append.
        The attention submodule reads/writes its own slot in ``state`` by ``self.layer_id``.
        """
        a = (
            self.attn(self.attn_norm(x), state, self.layer_id)
            if state is not None
            else self.attn(self.attn_norm(x))
        )
        x = x + a
        f = self.ffn(self.ffn_norm(x))
        x = x + f
        if block_reps is not None:
            block_reps.append(x)  # AttnRes source list — assembly bookkeeping; merge is core/K4
        return x, None


class K3Model(nn.Module):
    """Full K3 / mini-K3: embed → blocks (+ AttnRes merge) → final norm → LM head.

    ``forward(token_ids: (B, S) long, state=None) → logits (B, S, vocab_size)``.

    With ``state=None`` this is the training (prefill) forward — full sequence, no cache.
    With a ``HybridState`` this is incremental decode — one token per row, each attention
    layer reads and updates its own slot. The two paths must agree to dtype tolerance (the
    cache-equivalence gate, ``test_k3_model.py``).

    NoPE means no ``positions`` tensor is threaded (contrast ``scratch_llm.model``'s RoPE
    path): position is implicit in the KDA recurrence length, tracked by
    ``HybridState.lengths``.
    """

    def __init__(self, cfg: K3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList(K3Block(cfg, layer_id=i) for i in range(1, cfg.num_layers + 1))
        # Block AttnRes — hand-built in core/ (K4 rung). Sits across the block stack: learned
        # pseudo-queries attend over RMSNorm-ed block representations (embedding + each block
        # output) and merge back into the residual stream via online-softmax (FACTS A9).
        # Swap-in line:
        #   from scratch_llm.k3.core.attn_res import BlockAttnRes
        #   self.attn_res = BlockAttnRes(cfg.hidden_size, cfg.attn_res_block_size, cfg.num_layers)
        self.attn_res: nn.Module | None = None  # TODO(core/attn_res.py K4 rung)
        self.final_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = Linear(cfg.hidden_size, cfg.vocab_size)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.token_emb.weight  # weight sharing (off by default in K3)

    def forward(
        self,
        token_ids: Tensor,
        state: HybridState | None = None,
        return_aux: bool = False,
    ) -> Tensor:
        """``token_ids`` (B, S) → logits (B, S, vocab_size).

        ``return_aux`` is the hook for the aux-loss-free Quantile-Balancing stats once
        ``core/latent_moe.py`` lands; today the path returns logits only (the dense-overfit
        and loss-at-init gates don't need aux).

        AttnRes wiring (K4): the block loop appends each output to ``block_reps``; after the
        loop, ``self.attn_res.merge(x, block_reps)`` folds the retrieved residuals back in.
        Until K4 lands, ``attn_res`` is None and the path is the plain residual stack —
        the K6 ablation arm "AttnRes off" is exactly this path with ``attn_res`` left None.
        """
        x = self.token_emb(token_ids)

        collect = self.attn_res is not None  # only build the source list if K4 will consume it
        block_reps: list[Tensor] = [x] if collect else []
        for block in self.blocks:
            x, _stats = block(x, state, block_reps)

        if self.attn_res is not None:
            # TODO(core/attn_res.py K4): x = self.attn_res.merge(x, block_reps)
            #   — online-softmax merge of inter/intra-block sources; FACTS A9.
            raise NotImplementedError(
                "attn_res is set but core/attn_res.py :: BlockAttnRes.merge is not wired "
                "(HANDCRAFTED.md K4 rung)."
            )

        x = self.final_norm(x)
        logits = self.lm_head(x)

        # Bump the position cursor once, after all layers — each layer saw the same start.
        if state is not None:
            state.lengths = state.lengths + token_ids.shape[1]

        return logits

    # -- gates (assembly-owned surfaces) -------------------------------------

    def loss_at_init(self, token_ids: Tensor, targets: Tensor) -> float:
        """The K0/K6 gate: a fresh model's cross-entropy on its OWN next-token targets must
        be ≈ log(vocab). Pure measurement — call once at construction, before any optimizer
        step. mini-K3: log(32768) ≈ 10.397; full K3: log(163840) ≈ 12.009. Off by > 0.1 ⇒ a
        bug in head/embed/norm/init scale (the ``initializer_range=0.02`` contract, FACTS A13)."""
        with torch.no_grad():
            logits = self.forward(token_ids)
            loss = cross_entropy(logits, targets)
        return loss.item()


# ---------------------------------------------------------------------------
# Builders.
# ---------------------------------------------------------------------------


def build_k3() -> K3Model:
    """The released 2.78T-checkpoint architecture (``k3_full()``).

    Constructs today (structure is introspectable) but every block's attention and FFN are
    ``_PendingCore``; ``forward`` raises until the human authors the ``core/`` modules. This
    IS the honest state of the scaffold: the hand-built territory is what stands between this
    file and a runnable K3.
    """
    return K3Model(k3_full())


def build_mini_k3() -> K3Model:
    """The d12 miniature (``mini_k3_d12()``, ROADMAP K6). Same ``_PendingCore`` surface as
    :func:`build_k3`; the K6 training run begins the moment ``core/`` clears K5."""
    return K3Model(mini_k3_d12())


if __name__ == "__main__":
    # The scaffold's only runtime behavior is to state its dependencies honestly. Once core/
    # lands, this becomes the loss-at-init self-check (log(vocab) gate) for the mini.
    raise SystemExit(
        "k3.model is scaffold-only (DELEGATED assembly; HANDCRAFTED.md Class 3). The math\n"
        "lives in core/ — hand-built by the human in this serial order: situ → kda →\n"
        "gated_mla → latent_moe → attn_res. Each block's _PendingCore points at the exact\n"
        "module + rung to author next. Until then, construct inspects; forward raises."
    )
