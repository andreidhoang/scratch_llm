"""Data loading, checkpointing, and the training loop — the plumbing that turns the
model + optimizer into a trained policy, and (crucially) the same plumbing an RL run
reuses to warm-start from a checkpoint.

L1 substrate (A1). ``get_batch`` reads (input, next-token) windows from a flat token
array (``np.memmap`` so a corpus larger than RAM never loads fully); ``save/load_checkpoint``
round-trip model + optimizer + step — a STATE round-trip, not an exact resume: no RNG
state and no start-step are restored, so a restarted ``train()`` always begins at step 0
with a fresh warmup and replays the same seeded batch stream (true resume-at-step is
A2/A8 §D scope).

Correctness invariants (tested in tests/test_train.py):
- **Next-token alignment:** ``targets`` is ``inputs`` shifted by one position.
- **Checkpoint round-trip:** save → load restores the step and every parameter.
- **Reproducibility:** two runs with the same seed produce an identical loss history.
- **It learns:** on a structured corpus the loss drops well below log(vocab_size).

The same loop trains K3 (``k3.model.K3Model``); see :func:`train` for the three K3 branches and
:func:`build_k3_from_checkpoint` for its checkpoints (gates: tests/test_k3_train.py).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from scratch_llm.k3 import config as k3_config
from scratch_llm.k3.core.gated_mla import GatedMLA
from scratch_llm.k3.model import K3Model
from scratch_llm.k3.muon import apply_k3_qk_clip, build_k3_optimizer
from scratch_llm.model import ModelConfig, TransformerLM, cross_entropy
from scratch_llm.moe import MoEConfig
from scratch_llm.optim import (
    CombinedOptimizer,
    apply_qk_clip,
    build_optimizer,
    cosine_lr,
    gradient_clipping,
    split_muon_adamw_params,
)
from scratch_llm.utils.dist_train import (
    DistMuonAdamW,
    assert_model_replicas_identical,
)
from scratch_llm.utils.seeding import seed_everything


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Sample ``batch_size`` random (input, next-token) windows of length ``context_length``.

    Returns ``(inputs, targets)`` as long tensors of shape (batch_size, context_length),
    where ``targets[b, t] == inputs[b, t+1]`` in the underlying stream.
    """
    max_start = len(data) - context_length - 1
    if max_start < 1:
        raise ValueError(
            f"corpus of {len(data)} tokens too short for context_length={context_length}"
        )
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    inputs = np.stack([data[s : s + context_length] for s in starts])
    targets = np.stack([data[s + 1 : s + 1 + context_length] for s in starts])
    return (
        torch.from_numpy(inputs).long().to(device),
        torch.from_numpy(targets).long().to(device),
    )


#: ``model_config["family"]`` of a K3 checkpoint. A ModelConfig payload has no "family" key, so
#: TransformerLM checkpoints are byte-identical to the pre-K3 format and a payload without the key
#: is a TransformerLM.
_K3_FAMILY = "k3"


def _config_payload(model: torch.nn.Module) -> dict[str, Any] | None:
    """The checkpoint's ``model_config``: ``asdict(model.cfg)`` for a ``ModelConfig``, the same
    tagged ``family=_K3_FAMILY`` for a ``K3Config``, None for anything else. ``asdict`` recurses
    into the nested dataclasses and keeps tuples as tuples, so the payload is plain dicts, tuples
    and scalars: ``torch.load(weights_only=True)`` reads it."""
    cfg = getattr(model, "cfg", None)
    if isinstance(cfg, ModelConfig):
        return asdict(cfg)
    if isinstance(cfg, k3_config.K3Config):
        return {"family": _K3_FAMILY, **asdict(cfg)}
    return None


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | CombinedOptimizer | None,
    step: int,
    out: str | Path,
) -> None:
    """Persist model + optimizer + step, and — when the module carries a ``ModelConfig`` or a
    ``K3Config`` at ``.cfg`` (TransformerLM and K3Model do) — the config itself, so
    ``build_model_from_checkpoint`` / ``build_k3_from_checkpoint`` can rebuild the model from the
    file alone (A2 checkpoint chaining).

    ``optimizer=None`` writes a **stage-boundary** snapshot (model+config+step only): the
    pinned stage-transition policy — each speedrun stage starts a FRESH optimizer (pretrain:
    cosine LR with warmup; SFT: constant lr), because resuming Adam/Muon moments across an
    ``adamw↔muon_adamw`` switch is undefined. Intra-run snapshots (same stage, same optimizer) keep optimizer state via
    the ``checkpoint_every`` path in ``train()`` — a warm-start, not an exact resume (no
    RNG/start-step restore; see the module docstring).
    """
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "model_config": _config_payload(model),
        },
        out,
    )


def load_checkpoint(
    src: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | CombinedOptimizer | None = None,
    map_location: str = "cpu",
) -> int:
    """Restore model (and optionally optimizer) state; return the saved step."""
    # weights_only=False: we load our own trusted checkpoints, which include the
    # optimizer's param-group metadata (not just tensors).
    ckpt = torch.load(src, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        if ckpt.get("optim") is None:
            raise ValueError(
                f"{src} is a stage-boundary checkpoint (no optimizer state): load it with "
                "optimizer=None and build a fresh optimizer — the stage-transition policy."
            )
        optimizer.load_state_dict(ckpt["optim"])
    return int(ckpt["step"])


def _read_model_config(src: str | Path, map_location: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a checkpoint and a copy of its ``model_config`` payload (the caller pops fields);
    a checkpoint without one fails loud."""
    ckpt = torch.load(src, map_location=map_location, weights_only=False)
    cfg_dict = ckpt.get("model_config")
    if cfg_dict is None:
        raise ValueError(
            f"{src} carries no 'model_config' (pre-A2 format): construct the ModelConfig "
            "yourself and use load_checkpoint, or re-save with save_checkpoint."
        )
    return ckpt, dict(cfg_dict)  # don't mutate the loaded dict


def build_model_from_checkpoint(
    src: str | Path,
    map_location: str = "cpu",
) -> tuple[TransformerLM, int]:
    """Rebuild the exact model from a config-carrying checkpoint alone — no external
    ``ModelConfig`` needed. Returns ``(model, step)``.

    This is the rental safety-net and the joint the stage spine (A4 midtrain / A5 SFT /
    A6 chat) chains from. The strict ``load_state_dict`` is the kill-switch: a
    reconstructed config that shape-mismatches the weights fails loud here, never truncates.
    A K3 checkpoint raises: it rebuilds with :func:`build_k3_from_checkpoint`.
    """
    ckpt, cfg_dict = _read_model_config(src, map_location)
    if cfg_dict.get("family") == _K3_FAMILY:
        raise ValueError(f"{src} holds a K3Model: rebuild it with build_k3_from_checkpoint")
    moe = cfg_dict.pop("moe", None)
    cfg = ModelConfig(**cfg_dict, moe=MoEConfig(**moe) if moe is not None else None)
    model = TransformerLM(cfg)
    model.load_state_dict(ckpt["model"])
    return model, int(ckpt["step"])


def build_k3_from_checkpoint(
    src: str | Path,
    map_location: str = "cpu",
) -> tuple[K3Model, int]:
    """:func:`build_model_from_checkpoint` for a ``K3Model`` checkpoint; returns ``(model,
    step)``. A separate entry point so that each returns one concrete type: the TransformerLM
    callers (chat, vibe evals, serving) keep theirs.

    The nested ``KDAConfig``, ``MLAConfig``, K3 ``MoEConfig`` and ``VisionConfig`` (None in the
    mini presets) are rebuilt from their dicts. The tuple fields (``kda_layers``, ``mla_layers``,
    ``vision.merge_kernel``) need nothing: ``asdict`` and pickle both keep tuples, so the rebuilt
    frozen config equals the saved one field for field. The strict ``load_state_dict`` then
    overwrites the whole init, the Quantile-Balancing bias included (a parameter, so it is in the
    state dict).
    """
    ckpt, fields = _read_model_config(src, map_location)
    if fields.pop("family", None) != _K3_FAMILY:
        raise ValueError(
            f"{src} holds a TransformerLM: rebuild it with build_model_from_checkpoint"
        )
    kda, mla, moe, vision = (fields.pop(name) for name in ("kda", "mla", "moe", "vision"))
    cfg = k3_config.K3Config(
        **fields,
        kda=k3_config.KDAConfig(**kda),
        mla=k3_config.MLAConfig(**mla),
        moe=k3_config.MoEConfig(**moe),
        vision=k3_config.VisionConfig(**vision) if vision is not None else None,
    )
    model = K3Model(cfg)
    model.load_state_dict(ckpt["model"])
    return model, int(ckpt["step"])


def save_consolidated_checkpoint(
    model: torch.nn.Module,
    optimizer: DistMuonAdamW,
    step: int,
    out: str | Path,
) -> None:
    """A8 distributed checkpoint — the topology-INDEPENDENT counterpart of
    :func:`save_checkpoint`. Unlike the rank-local shards :meth:`DistMuonAdamW.state_dict` writes
    (same-topology resume only), this consolidates the optimizer's ZeRO-2 shards into one
    un-sharded state that resumes on ANY world size and from which the model can be extracted for
    serving.

    **Collective — every rank must call it.** :func:`assert_model_replicas_identical` and
    :meth:`DistMuonAdamW.consolidated_state_dict` both run ``all_gather``s that all ranks
    participate in; only rank 0 then writes the file (model = rank 0's replica, verified identical;
    optimizer = the consolidated dict; plus ``step``, ``model_config``, and the saving
    ``world_size``). Resume with :func:`load_consolidated_checkpoint`.
    """
    assert_model_replicas_identical(model)  # collective; fails loud on a replica desync
    consolidated = optimizer.consolidated_state_dict()  # collective; rank 0 gets the dict
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    torch.save(
        {
            "model": model.state_dict(),
            "optim_consolidated": consolidated,
            "step": step,
            "model_config": _config_payload(model),
            "world_size": optimizer.world_size,
            "format": "consolidated",
        },
        out,
    )


def load_consolidated_checkpoint(
    src: str | Path,
    model: torch.nn.Module,
    optimizer: DistMuonAdamW | None = None,
    map_location: str = "cpu",
) -> int:
    """Resume from a :func:`save_consolidated_checkpoint` file: restore the model and re-shard the
    consolidated optimizer state onto ``optimizer``'s CURRENT topology (works at any world size —
    the format is topology-independent). Returns the saved step. Pure per-rank slicing, no
    collectives (see :meth:`DistMuonAdamW.load_consolidated`)."""
    ckpt = torch.load(src, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        if ckpt.get("optim_consolidated") is None:
            raise ValueError(
                f"{src} carries no consolidated optimizer state: load the model with "
                "optimizer=None (serving), or point at a save_consolidated_checkpoint file."
            )
        optimizer.load_consolidated(ckpt["optim_consolidated"])
    return int(ckpt["step"])


@dataclass
class TrainConfig:
    max_steps: int
    batch_size: int
    context_length: int
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 0
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)
    log_every: int = 10
    checkpoint_every: int = 0  # 0 = never
    checkpoint_path: str | None = None
    device: str = "cpu"
    seed: int = 0
    verbose: bool = False
    # F1/F4 — close-the-loop knobs (defaults reproduce the A1 AdamW/fp32 path exactly).
    optimizer: str = "adamw"  # "adamw" | "muon_adamw" (Muon on block matrices + AdamW on the rest)
    # | "k3_muon" (K3Model only: k3.muon.build_k3_optimizer, Per-Head Muon + AdamW, one lr).
    muon_momentum: float = 0.95
    amp_dtype: str | None = None  # None ⇒ fp32; "bf16" ⇒ bf16 autocast (no GradScaler needed)
    compile: bool = False  # torch.compile the forward (the cheap-MFU win on the GPU box)
    # F1-run — val-eval hook + NS instrument (defaults are byte-identical no-ops).
    eval_every: int = 0  # 0 = never; >0 ⇒ val CE every N steps AND at the final step
    eval_batches: int = 8  # fixed sequential val windows per eval (no RNG — see _val_loss)
    muon_profile_ns: bool = False  # time Newton–Schulz inside Muon (perturbs; measurement-only)
    # F9 — QK-Clip guard (Kimi-K2, arXiv:2507.20534): post-step per-head W_q/W_k rescale of any
    # head whose max pre-softmax logit exceeded τ this step. Requires the model built with
    # ModelConfig.track_attn_logits=True (the observer supplies S_max). Default OFF = untouched loop.
    qk_clip: bool = False
    qk_clip_tau: float = 100.0
    # F2a — MTP aux head (DeepSeek-V3): weight λ on the one-token-further-ahead CE. Active only
    # when the model was built with ModelConfig.mtp_depth >= 1; ignored otherwise (loss unchanged).
    mtp_loss_weight: float = 0.3


def _val_loss(
    model: TransformerLM | K3Model,
    val_data: np.ndarray,
    context_length: int,
    eval_batches: int,
    device: str,
) -> float:
    """Mean cross-entropy over FIXED sequential windows of ``val_data`` — deliberately no RNG.

    Determinism is the contract: the same checkpoint always scores the same number, every arm of
    an A/B race scores on the same windows, and — critically — evaluating consumes no random
    state, so switching eval ON cannot perturb the training batch stream (tested in
    tests/test_optimizer_race.py). Runs in fp32 (no autocast): val numbers must be comparable
    across arms that train under different precision regimes.
    """
    n_windows = min(eval_batches, (len(val_data) - 1) // context_length)
    if n_windows < 1:
        raise ValueError(
            f"val corpus of {len(val_data)} tokens too short for context_length={context_length}"
        )
    starts = [i * context_length for i in range(n_windows)]
    inputs = np.stack([val_data[s : s + context_length] for s in starts])
    targets = np.stack([val_data[s + 1 : s + 1 + context_length] for s in starts])
    was_training = model.training
    model.eval()
    with torch.no_grad():
        loss = cross_entropy(
            model(torch.from_numpy(inputs).long().to(device)),
            torch.from_numpy(targets).long().to(device),
        )
    if was_training:
        model.train()
    return float(loss.item())


def train(
    cfg: TrainConfig,
    train_data: np.ndarray,
    model: TransformerLM | K3Model,
    val_data: np.ndarray | None = None,
    eval_hook: Callable[[int, float], None] | None = None,
    optimizer_out: list[torch.optim.Optimizer | CombinedOptimizer] | None = None,
    batch_fn: Callable[[int], tuple[Tensor, Tensor]] | None = None,
) -> list[tuple[int, float]]:
    """Run the training loop. Returns the loss history as (step, loss) pairs.

    F1-run additions (all default-off, byte-identical to the A1 path): when
    ``cfg.eval_every > 0`` and ``val_data`` is given, ``eval_hook(step, val_ce)`` fires every
    ``eval_every`` steps and at the final step (the iso-FLOP endpoint); ``optimizer_out``, if a
    list, receives the built optimizer so callers can read instruments (Muon NS counters) after
    the run — the loop's return type stays unchanged.

    ``batch_fn`` (T1 A/B seam, optional): when given, it OWNS the data stream — ``batch_fn(step)``
    replaces ``get_batch`` and with it the global-numpy draw, so the batch order becomes a function
    of the caller's seed alone and cannot be shifted by anything else an arm consumes randomness
    for (see ``scratch_llm.training.run_matrix.batch_schedule``). It also owns per-rank sharding:
    the ``cfg.seed + 1000*rank`` offset below no longer reaches the data. ``None`` is byte-identical
    to the pre-seam behaviour.

    Same plumbing an RL fine-tune warm-starts from: cosine LR per step, global-ℓ₂ grad clip,
    periodic checkpointing (state snapshots only — a reload starts again at step 0 with a
    fresh warmup; see the module docstring).

    ``cfg.seed`` reseeds the data-sampling RNG here so batches are deterministic. For
    *end-to-end* reproducibility, seed before constructing ``model`` too — weight init
    happens before this call and is not covered by the reseed.

    A7 — distributed mode (activated ONLY by an initialized ``torch.distributed`` process
    group; single-process runs are byte-identical to before): params broadcast once from
    rank 0, each rank samples a DIFFERENT batch stream (numpy RNG offset by rank), the
    optimizer becomes :class:`~scratch_llm.utils.dist_train.DistMuonAdamW` (optimizer-embedded
    ZeRO-2 — the grad clip moves inside its step), the NaN guard is collective (all ranks
    raise together), and logging/eval/checkpointing are rank-0-only.

    K3 (a ``K3Model``) takes the same loop, with three branches keyed on the model type:

    - ``optimizer="k3_muon"`` builds ``k3.muon.build_k3_optimizer`` (Per-Head Muon on the
      matrices, AdamW on embeddings, head, norms, convs and AttnRes queries) with ONE learning
      rate for both halves: the schedule below writes the same lr into every group. That is the
      Kimi K2 / Moonlight recipe as torchtitan runs it (``kimi_k2_7/config_registry.py``:334-348
      passes the same ``lr`` to Muon, with ``match_rms_adamw``, and to AdamW). Per unit lr, Muon's
      update has element RMS 0.2 (0.2·√max(rows, cols) times an orthogonal factor, ``k3.muon``),
      the low end of AdamW's typical 0.2-0.4 (Moonlight §2.2), so one lr moves both halves at a
      matched per-element rate.
      ``"muon_adamw"`` is refused: its split keys on TransformerLM names and would put
      ``embed_tokens`` and the AttnRes pseudo-queries on Muon. ``"adamw"`` is the plain baseline.
    - ``qk_clip`` switches on every ``GatedMLA``'s max-logit observer (a module flag, not a config
      field; it stays on after ``train()`` returns) and applies ``k3.muon.apply_k3_qk_clip`` after
      the step. KDA is never clipped: its q and k are L2-normalized.
    - The MoE branch runs as it does for a TransformerLM MoE: ``aux.total`` is a zero (Quantile
      Balancing has no auxiliary loss) and ``moe_update_biases()`` then assigns every router bias
      from the step's routing statistics. The order is weights, clip, bias. Only weights → clip
      matters (the clip rescales the stepped weights, as in K2); the bias update commutes with
      both, since it reads only the forward's statistics and writes only the bias, which neither
      the optimizer (``requires_grad=False``) nor the clip touches. torchtitan runs the bias as a
      pre-step hook and the clip as a post-step one (``components/optimizer/optimizer.py``:523,
      ``kimi_k2_7/qk_clip.py``:198).

    K3 has no MTP head, and distributed K3 is refused by the MoE guard (the QB statistics would
    need an all-reduce first; ``k3.core.latent_moe.QBStats``).
    """
    is_k3 = isinstance(model, K3Model)
    if cfg.amp_dtype not in (None, "bf16"):
        raise ValueError(
            f"amp_dtype must be None or 'bf16' (fp16 needs a GradScaler); got {cfg.amp_dtype!r}"
        )
    if cfg.qk_clip and not is_k3 and not model.cfg.track_attn_logits:
        raise ValueError(
            "qk_clip=True requires ModelConfig.track_attn_logits=True — the clip reads the "
            "per-head max-logit observer; without it every step would be a silent no-op"
        )
    if cfg.optimizer == "k3_muon" and not is_k3:
        raise ValueError(
            "optimizer='k3_muon' partitions parameters by K3 module names "
            "(k3.muon.k3_param_groups): it needs a K3Model"
        )
    if is_k3 and cfg.optimizer == "muon_adamw":
        raise ValueError(
            "a K3Model trains with optimizer='k3_muon' or 'adamw': 'muon_adamw' splits on "
            "TransformerLM names and would put embed_tokens and the AttnRes queries on Muon"
        )
    if cfg.optimizer == "k3_muon" and cfg.muon_profile_ns:
        raise ValueError("muon_profile_ns times optim.Muon; PerHeadMuon has no Newton–Schulz timer")
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    if distributed and model.cfg.moe is not None:
        raise ValueError(
            "distributed train() does not support MoE yet: moe_update_biases() nudges router "
            "biases from per-rank token counts, which would silently desync the replicas"
        )
    if distributed and cfg.qk_clip:
        raise ValueError(
            "distributed train() does not support qk_clip yet: per-rank S_max observations "
            "would rescale head weights differently on each rank, silently desyncing replicas"
        )
    seed_everything(cfg.seed)
    if distributed:
        # Per-rank data sharding: seed_everything just synced numpy's RNG across ranks, so
        # without this offset every rank would draw IDENTICAL batches — a silent world_size×
        # data loss. Torch RNG stays synced (identical init); only the batch sampler diverges.
        np.random.seed(cfg.seed + 1000 * rank + 1)
    model.to(cfg.device)
    if cfg.qk_clip and is_k3:
        for module in model.modules():
            if isinstance(module, GatedMLA):
                module.track_max_logits = True
    if distributed:
        # Belt-and-suspenders identical init: seeding before construction already makes the
        # replicas identical, but a caller that seeded ranks differently would silently
        # diverge — rank 0's weights are the canonical start.
        handles = [dist.broadcast(p.data, src=0, async_op=True) for p in model.parameters()]
        for handle in handles:
            assert handle is not None  # async_op=True always yields a Work handle
            handle.wait()
    optimizer: torch.optim.Optimizer | CombinedOptimizer
    if distributed:
        # nanochat's verified shape: NO DDP wrapper — grads reduce-scatter INTO the optimizer,
        # owner ranks update, params all_gather back (utils/dist_train.py). The grad clip moves
        # inside its step (clipping unreduced local grads would clip the wrong norm); Muon's
        # profile_ns instrument is single-process-only and is not wired here.
        if cfg.optimizer == "muon_adamw":
            muon_params, adamw_params = split_muon_adamw_params(model)
        elif cfg.optimizer == "adamw":
            blocks, rest = split_muon_adamw_params(model)
            muon_params, adamw_params = [], blocks + rest  # everything on the AdamW shard path
        else:
            raise ValueError(
                f"unknown optimizer kind {cfg.optimizer!r} (expected 'adamw' or 'muon_adamw')"
            )
        optimizer = DistMuonAdamW(
            muon_params,
            adamw_params,
            lr=cfg.max_lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
            muon_momentum=cfg.muon_momentum,
            max_l2_norm=cfg.grad_clip,
        )
    elif cfg.optimizer == "k3_muon":
        optimizer = build_k3_optimizer(
            model,
            lr=cfg.max_lr,
            adamw_lr=cfg.max_lr,  # one lr for both halves (docstring); the schedule rewrites it
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            momentum=cfg.muon_momentum,
        )
    else:
        optimizer = build_optimizer(
            model,
            kind=cfg.optimizer,
            lr=cfg.max_lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
            muon_momentum=cfg.muon_momentum,
            muon_profile_ns=cfg.muon_profile_ns,
        )
    if optimizer_out is not None:
        optimizer_out.append(optimizer)
    # torch.compile wraps the module; keep the ORIGINAL for .cfg / .moe_update_biases / checkpoint
    # (the compiled module's state_dict carries an `_orig_mod.` prefix, and the split for Muon must
    # see the real submodules).
    forward_model = torch.compile(model) if cfg.compile else model
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else None
    device_type = "cuda" if "cuda" in str(cfg.device) else "cpu"

    history: list[tuple[int, float]] = []
    for step in range(cfg.max_steps):
        lr = cosine_lr(step, cfg.max_lr, cfg.min_lr, cfg.warmup_steps, cfg.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = (
            batch_fn(step)
            if batch_fn is not None
            else get_batch(train_data, cfg.batch_size, cfg.context_length, cfg.device)
        )
        optimizer.zero_grad()
        amp_ctx = (
            torch.autocast(device_type=device_type, dtype=amp_dtype)
            if amp_dtype is not None
            else nullcontext()
        )
        with amp_ctx:
            if not is_k3 and model.cfg.mtp_depth > 0:
                # F2a MTP: main CE (+ MoE aux if present) + λ·CE one token further ahead.
                # Reads model.cfg (not forward_model) — the same wrapper-safe pattern as the
                # MoE branch below (torch.compile / DDP wrappers forward the call but the
                # ORIGINAL module carries the config). forward_train too: torch.compile compiles
                # forward only, and its wrapper resolves forward_train to this same bound method.
                logits, aux, mtp_logits = model.forward_train(inputs, targets)
                loss = cross_entropy(logits, targets) + cfg.mtp_loss_weight * cross_entropy(
                    mtp_logits[:, :-1], targets[:, 1:]
                )
                if aux is not None:
                    loss = loss + aux.total
            elif model.cfg.moe is not None:
                # MoE: add the sparse-regularization terms (seq-wise balance + router z-loss) to CE.
                # K3 is aux-loss-free: its aux.total is a zero.
                logits, aux = forward_model(inputs, return_aux=True)
                loss = cross_entropy(logits, targets) + aux.total
            else:
                loss = cross_entropy(forward_model(inputs), targets)
        loss.backward()
        if not distributed:
            # Distributed mode: the clip lives INSIDE DistMuonAdamW.step — the global norm is a
            # property of the REDUCED (averaged) grads, so clipping local grads here is wrong.
            gradient_clipping(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if cfg.qk_clip:
            # F9 QK-Clip: rescale any head whose observed max logit exceeded τ this step. The
            # observer recorded S_max on this step's forward (pre-update weights); the clip lands
            # post-step, in weight space — exactly the Kimi-K2 MuonClip ordering.
            if is_k3:
                apply_k3_qk_clip(model, cfg.qk_clip_tau)
            else:
                apply_qk_clip(model, cfg.qk_clip_tau)
        if model.cfg.moe is not None:
            # Aux-loss-free load balancing: nudge the router biases after the weight update.
            model.moe_update_biases()

        if cfg.log_every and step % cfg.log_every == 0:
            loss_value = loss.item()  # the one host sync per log interval (R1 lesson)
            nonfinite = not math.isfinite(loss_value)
            if distributed:
                # Collective guard: a divergence one rank sees must stop ALL ranks together —
                # a lone raiser would leave the others blocked in the next collective forever.
                # Every rank hits this same log step, so the all_reduce is symmetric.
                flag_device = cfg.device if dist.get_backend() == "nccl" else "cpu"
                flag = torch.tensor([1.0 if nonfinite else 0.0], device=flag_device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                nonfinite = bool(flag.item() > 0)
            if nonfinite:
                # Fail LOUD on divergence — never burn compute on a silently-NaN run (FRONTIER
                # 'silent divergence' triage). Known trigger on this box: bf16 autocast + torch.compile
                # together NaN on sm120 / torch-2.12 inductor (reproduces with plain AdamW); use
                # bf16-eager or fp32-compile here — bf16+compile is the H100-rental path.
                where = loss_value if not math.isfinite(loss_value) else "on another rank"
                raise RuntimeError(
                    f"non-finite loss ({where}) at step {step}: training diverged. Check the LR, "
                    "or avoid bf16 + torch.compile together on sm120 (an inductor codegen bug)."
                )
            history.append((step, loss_value))
            if cfg.verbose and rank == 0:
                print(f"step {step:6d} | lr {lr:.2e} | loss {loss_value:.4f}")
        if (
            rank == 0  # val windows are deterministic — one rank's eval is authoritative, and
            # _val_loss runs no collectives, so the guard cannot desync the ranks
            and cfg.eval_every
            and val_data is not None
            and eval_hook is not None
            and (step % cfg.eval_every == 0 or step == cfg.max_steps - 1)
        ):
            eval_hook(
                step,
                _val_loss(model, val_data, cfg.context_length, cfg.eval_batches, cfg.device),
            )
        checkpoint_now = bool(
            cfg.checkpoint_every
            and cfg.checkpoint_path
            and step > 0
            and step % cfg.checkpoint_every == 0
        )
        # ``checkpoint_now`` is a function of cfg + step only, so every rank agrees — the
        # distributed branch's consolidation collectives stay in lockstep.
        if checkpoint_now and distributed:
            # A8: ALL ranks participate in the replica-identity check + the ZeRO-2 consolidation
            # gathers; rank 0 writes ONE topology-independent file (resumable on any world size).
            assert isinstance(optimizer, DistMuonAdamW)
            assert cfg.checkpoint_path is not None
            save_consolidated_checkpoint(model, optimizer, step, cfg.checkpoint_path)
        elif checkpoint_now and rank == 0:
            # Single-process path — byte-identical to before A8 (replicas are trivially identical;
            # rank 0's save is the run's save).
            assert cfg.checkpoint_path is not None
            save_checkpoint(model, optimizer, step, cfg.checkpoint_path)

    return history


__all__ = [
    "TrainConfig",
    "build_k3_from_checkpoint",
    "build_model_from_checkpoint",
    "get_batch",
    "load_checkpoint",
    "load_consolidated_checkpoint",
    "save_checkpoint",
    "save_consolidated_checkpoint",
    "train",
]
