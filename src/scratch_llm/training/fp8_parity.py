"""T1/T-R2 — FP8 vs bf16 at a fixed token budget: two arms, two seeds, one variable.

    python -m scratch_llm.training.fp8_parity --data-dir DIR --tokens 268435456   the 2x2 matrix
    python -m scratch_llm.training.fp8_parity --data-dir DIR --floor bf16_tok_s   the tok/s floor
    python -m scratch_llm.training.fp8_parity --data-dir DIR --floor bf16_bpb     the bpb floor
    python -m scratch_llm.training.fp8_parity --self-test                         CPU plumbing
    python -m scratch_llm.training.fp8_parity --dry-run                           the plan, no run

The last line of stdout is always one bare number: the speedup for a matrix run, the floor's value
for a floor run. ``experiments/T1/T-R2/{run,floor}.sh`` read exactly that line.

**The two floors come out of one bf16 run.** ``--floor bf16_tok_s`` and ``--floor bf16_bpb`` are
the same command with a different last line, and both are printed either way, because a tok/s from
Monday's run and a bpb from Tuesday's are not a floor for the same thing.

**Why a speedup without the bpb gate is not a result:** casting the linears to fp8 is not a
transformation that preserves the loss. It is a cheaper, noisier arithmetic that the run may or may
not absorb, and whether it absorbed it is only visible in the final bpb. A tok/s number on its own
is a measurement of a different model — you can always go faster by training something worse.

The one variable is :attr:`ArmConfig.precision`. Everything else — init, data order, token budget,
eval stream, optimizer, LR schedule, autocast dtype, even the ``nn.Linear`` surgery below — is
identical in both arms, and :func:`~scratch_llm.training.run_matrix.assert_one_variable` checks it
before either arm runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scratch_llm.model import Linear as ScratchLinear
from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.tokenizer import Tokenizer
from scratch_llm.train import TrainConfig, train
from scratch_llm.training import bpb as bpb_mod
from scratch_llm.training.run_matrix import (
    ArmResult,
    StepTimer,
    assert_one_variable,
    batch_schedule,
    batches_from_schedule,
    compare,
    digest,
    eval_bpb,
    eval_stream,
    steps_for_tokens,
)
from scratch_llm.utils.seeding import seed_everything


@dataclass(frozen=True)
class Preset:
    """A model shape both arms are built from. ``vocab_size`` is NOT here: it comes from the
    tokenizer that produced the shards, because a guessed vocab is a different model."""

    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int

    def model_config(self, *, vocab_size: int, context_length: int) -> ModelConfig:
        return ModelConfig(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            d_ff=self.d_ff,
        )

    @property
    def linears_per_block(self) -> int:
        """4 attention projections + 3 SwiGLU matrices. The count ``apply_float8`` must swap."""
        return 7


#: Llama-3.2-1B's shape — what plan section 05's T-R0/T-R1 measure torchtitan at.
PRESETS: dict[str, Preset] = {
    "llama3_1b": Preset(d_model=2048, n_layers=16, n_heads=32, n_kv_heads=8, d_ff=8192),
    "debug": Preset(d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=256),
}


@dataclass(frozen=True)
class ArmConfig:
    """One arm. ``precision`` is the ONLY field the two arms may disagree on.

    ``recipe`` and ``filter_fqns`` mirror torchtitan's llama3-405B float8 row verbatim
    (``models/llama3/config_registry.py:359-361``): rowwise dynamic scaling, lm_head left in high
    precision. They are fields rather than constants so a recipe sweep is a new matrix, not an
    edit — but a sweep changes them in BOTH arms, or the bf16 arm stops being a control.
    """

    precision: str = "bf16"
    amp_dtype: str | None = "bf16"
    recipe: str = "rowwise"
    filter_fqns: tuple[str, ...] = ("lm_head",)
    compile: bool = False
    optimizer: str = "adamw"
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    grad_clip: float = 1.0


@dataclass(frozen=True)
class RunSpec:
    """Everything both arms share."""

    tokens: int
    batch_size: int
    context_length: int
    preset: str = "llama3_1b"
    world_size: int = 1
    rank: int = 0
    eval_windows: int = 64
    device: str = "cpu"
    timing_warmup: int = 20
    timing_min_steps: int = 50

    @property
    def steps(self) -> int:
        return steps_for_tokens(
            self.tokens,
            batch_size=self.batch_size,
            context_length=self.context_length,
            world_size=self.world_size,
        )

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.context_length * self.world_size


# ---------------------------------------------------------------------------------------------
# model surgery — applied to BOTH arms
# ---------------------------------------------------------------------------------------------


def to_nn_linear(module: nn.Module) -> int:
    """Replace every ``scratch_llm.model.Linear`` with an ``nn.Linear(bias=False)`` that SHARES
    its weight Parameter. Returns the number replaced.

    Necessary because ``scratch_llm.model.Linear`` is an ``nn.Module`` computing ``x @ W.T``, not
    an ``nn.Linear`` — and torchao's converter (like torchtitan's, which swaps
    ``torchtitan.models.common.linear.Linear``, an ``nn.Linear`` subclass) matches on the class.
    Left alone, ``convert_to_float8_training`` would swap zero modules and the "fp8" arm would be
    a bf16 arm with a different label; :func:`apply_float8` refuses a zero swap for that reason.

    Applied to BOTH arms so the shim's own numerics (``F.linear`` vs ``x @ W.T`` dispatch to
    different BLAS paths and are not bit-identical) cancel in the delta instead of landing on fp8.
    Parameter identity is preserved, so weight init and optimizer state are untouched.
    """
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, ScratchLinear):
            out_features, in_features = child.weight.shape
            shim = nn.Linear(in_features, out_features, bias=False)
            shim.weight = child.weight  # same Parameter object, not a copy
            setattr(module, name, shim)
            replaced += 1
        else:
            replaced += to_nn_linear(child)
    return replaced


def apply_float8(model: nn.Module, *, recipe: str, filter_fqns: tuple[str, ...]) -> int:
    """Swap eligible ``nn.Linear`` for torchao's ``Float8Linear``; return how many were swapped.

    The filter reproduces ``torchtitan/components/quantization/utils.py:16-30``: both dims a
    multiple of 16 (an fp8 tensorcore requirement, not a heuristic), and no fqn containing a
    filtered name. Raises on a zero swap — the failure mode this rung cannot survive is an fp8 arm
    that never ran an fp8 GEMM and reports a speedup of 1.00 and a Delta bpb of 0.
    """
    try:
        from torchao.float8 import (  # pyright: ignore[reportMissingImports]
            Float8LinearConfig,
            convert_to_float8_training,
        )
    except ImportError as exc:  # pragma: no cover - GPU box only
        raise ImportError(
            "the fp8 arm needs torchao: USE_CPP=0 pip install git+https://github.com/pytorch/ao.git"
        ) from exc

    config = Float8LinearConfig.from_recipe_name(recipe)

    def keep(mod: nn.Module, fqn: str) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if mod.in_features % 16 or mod.out_features % 16:
            return False
        return not any(f in fqn for f in filter_fqns)

    convert_to_float8_training(model, config=config, module_filter_fn=keep)
    swapped = sum(1 for m in model.modules() if type(m).__name__ == "Float8Linear")
    if swapped == 0:
        raise RuntimeError(
            f"float8 conversion swapped 0 modules (recipe={recipe!r}, filter_fqns={filter_fqns}) — "
            "an fp8 arm that is silently bf16 is the one result this rung must never produce"
        )
    return swapped


# ---------------------------------------------------------------------------------------------
# one cell of the matrix
# ---------------------------------------------------------------------------------------------


def build_model(spec: RunSpec, vocab_size: int, seed: int) -> tuple[TransformerLM, ModelConfig]:
    """Seed, construct, shim. Called identically by both arms, so their initial weights are
    bitwise identical for a given seed (``tests/training/test_t1_t_r2.py`` asserts it)."""
    seed_everything(seed)
    cfg = PRESETS[spec.preset].model_config(
        vocab_size=vocab_size, context_length=spec.context_length
    )
    model = TransformerLM(cfg)
    to_nn_linear(model)
    return model, cfg


def run_arm(
    spec: RunSpec,
    arm: ArmConfig,
    seed: int,
    *,
    train_data: np.ndarray,
    val_data: np.ndarray,
    lengths: np.ndarray,
    zero_byte_ids: tuple[int, ...] = (),
) -> ArmResult:
    """Train one arm at one seed for ``spec.tokens`` tokens and score it."""
    if arm.precision not in ("bf16", "fp8"):
        raise ValueError(f"unknown precision {arm.precision!r}")
    model, model_cfg = build_model(spec, int(lengths.size), seed)
    converted = (
        apply_float8(model, recipe=arm.recipe, filter_fqns=arm.filter_fqns)
        if arm.precision == "fp8"
        else 0
    )

    schedule = batch_schedule(
        seed,
        steps=spec.steps,  # exactly the budget: an extra step would train on more than `tokens`
        batch_size=spec.batch_size,
        context_length=spec.context_length,
        corpus_len=len(train_data),
        world_size=spec.world_size,
        rank=spec.rank,
    )
    timer = StepTimer(
        batches_from_schedule(train_data, schedule, spec.context_length, spec.device),
        device=spec.device,
    )
    train(
        TrainConfig(
            max_steps=spec.steps,
            batch_size=spec.batch_size,
            context_length=spec.context_length,
            max_lr=arm.max_lr,
            min_lr=arm.min_lr,
            warmup_steps=arm.warmup_steps,
            grad_clip=arm.grad_clip,
            device=spec.device,
            seed=seed,
            optimizer=arm.optimizer,
            amp_dtype=arm.amp_dtype,
            compile=arm.compile,
            # TrainConfig's default (10). NOT 0: `train()` gates its non-finite-loss guard on
            # log_every, and a diverged fp8 arm that runs the full budget before `bits_per_byte`
            # notices has burned the rung's silicon. Identical in both arms, and StepTimer already
            # syncs per step, so the extra host sync does not tilt the timing.
        ),
        train_data,
        model,
        batch_fn=timer,
    )
    # steps-1 intervals: the first call starts the clock, the last step is never closed. Padding
    # the loop to recover it would train past the budget, and the budget is the control.
    timing = timer.summary(
        tokens_per_step=spec.tokens_per_step,
        warmup=spec.timing_warmup,
        min_timed=spec.timing_min_steps,
    )

    stream = eval_stream(val_data, spec.context_length, spec.eval_windows)
    scorer = eval_model(model, model_cfg, spec)
    score, mean_ce, acc = eval_bpb(
        scorer, stream, lengths, device=spec.device, zero_byte_ids=zero_byte_ids
    )
    return ArmResult(
        arm=arm.precision,
        seed=seed,
        tokens=spec.tokens,
        steps=spec.steps,
        bpb=score,
        mean_ce=mean_ce,
        tok_s=timing["tok_s"],
        tok_s_p25=timing["tok_s_p25"],
        tok_s_p75=timing["tok_s_p75"],
        data_digest=digest(schedule),
        eval_digest=stream.digest,
        accounting=acc,
        converted_modules=converted,
    )


def eval_model(trained: nn.Module, model_cfg: ModelConfig, spec: RunSpec) -> nn.Module:
    """A plain high-precision copy of ``trained``, for scoring.

    The fp8 arm's own forward still quantizes on the fly, so scoring it in place would fold an
    fp8 INFERENCE effect into a number the rung reports as an fp8 TRAINING effect, and no
    later analysis could separate the two. Both arms are therefore scored through this same
    un-quantized module, loaded from the trained weights (which are high precision in both arms —
    torchao's Float8Linear keeps ``weight`` in high precision and casts per step).
    """
    scorer = TransformerLM(model_cfg)
    to_nn_linear(scorer)
    scorer.load_state_dict(trained.state_dict(), strict=True)
    return scorer.to(spec.device)


# ---------------------------------------------------------------------------------------------
# corpus + entry point
# ---------------------------------------------------------------------------------------------


def load_corpus(
    data_dir: Path, val_fraction: float = 0.02
) -> tuple[np.ndarray, np.ndarray, Tokenizer]:
    """Shards + the tokenizer staged beside them; a deterministic tail slice is the val set."""
    from scratch_llm.data.shards import load_dataset_tokens

    tokens = np.asarray(load_dataset_tokens(data_dir))
    tok = Tokenizer.load(data_dir / "tokenizer.json")
    cut = int(len(tokens) * (1.0 - val_fraction))
    return tokens[:cut], tokens[cut:], tok


def special_ids(tok: Tokenizer) -> tuple[int, ...]:
    """The tokenizer's special-token ids, via the public ``encode`` path (each special encodes to
    exactly one id). Their byte length under ``decode`` semantics is the literal text — 13 bytes
    for ``<|endoftext|>`` — which inflates the bpb denominator; see :mod:`scratch_llm.training.bpb`.
    """
    ids: list[int] = []
    for token in tok.special_tokens:
        encoded = tok.encode(token)
        if len(encoded) != 1:
            raise ValueError(f"special {token!r} encoded to {encoded}, expected one id")
        ids.append(encoded[0])
    return tuple(sorted(set(ids)))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--data-dir", type=Path, help="token shards + tokenizer.json")
    p.add_argument("--tokens", type=int, default=1 << 28, help="fixed token budget per arm")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--context-length", type=int, default=4096)
    p.add_argument("--preset", default="llama3_1b", choices=sorted(PRESETS))
    p.add_argument("--eval-windows", type=int, default=64)
    p.add_argument(
        "--zero-byte-specials",
        action="store_true",
        help="score special tokens at 0 bytes instead of their literal decode bytes",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--floor", choices=["bf16_tok_s", "bf16_bpb"], help="bf16 arm only")
    p.add_argument("--json", type=Path, help="write the full row here")
    p.add_argument("--self-test", action="store_true", help="tiny CPU run of the plumbing")
    p.add_argument("--dry-run", action="store_true", help="print the plan, measure nothing")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv)
    seeds = [int(s) for s in ns.seeds.split(",") if s.strip()]
    spec = RunSpec(
        tokens=ns.tokens,
        batch_size=ns.batch_size,
        context_length=ns.context_length,
        preset=ns.preset,
        eval_windows=ns.eval_windows,
        device=ns.device,
    )
    baseline, treatment = ArmConfig(precision="bf16"), ArmConfig(precision="fp8")
    assert_one_variable(baseline, treatment, "precision")
    arms = [baseline] if ns.floor else [baseline, treatment]

    if ns.self_test:
        spec = replace(spec, preset="debug", batch_size=2, context_length=32, eval_windows=2)
        spec = replace(spec, tokens=80 * 2 * 32, timing_warmup=2, timing_min_steps=4)
        arms = [baseline]

    if ns.dry_run:
        print(
            f"T1/T-R2 [dry-run] preset={spec.preset} tokens={spec.tokens} steps={spec.steps} "
            f"batch={spec.batch_size} ctx={spec.context_length} seeds={seeds} "
            f"arms={[a.precision for a in arms]} device={spec.device}"
        )
        print(f"  recipe={treatment.recipe} filter_fqns={treatment.filter_fqns}")
        return 0

    if ns.self_test:
        rng = np.random.Generator(np.random.PCG64(0))
        corpus = rng.integers(0, 256, size=1 << 15, dtype=np.int64)
        train_data, val_data = corpus[: -1 << 12], corpus[-1 << 12 :]
        lengths = np.ones(256, dtype=np.int64)
        zero_ids: tuple[int, ...] = ()
    else:
        if ns.data_dir is None:
            raise SystemExit("--data-dir is required (token shards + tokenizer.json)")
        train_data, val_data, tok = load_corpus(ns.data_dir)
        zero_ids = special_ids(tok) if ns.zero_byte_specials else ()
        lengths = bpb_mod.byte_lengths(tok.vocab, zero_byte_ids=zero_ids)

    results = [
        run_arm(
            spec,
            arm,
            seed,
            train_data=train_data,
            val_data=val_data,
            lengths=lengths,
            zero_byte_ids=zero_ids,
        )
        for arm in arms
        for seed in seeds
    ]
    row: dict[str, object] = {"rung": "T1/T-R2", "results": [r.as_dict() for r in results]}
    for r in results:
        print(
            f"{r.arm:>5} seed {r.seed}: {r.tok_s:,.0f} tok/s "
            f"[{r.tok_s_p25:,.0f}-{r.tok_s_p75:,.0f}] · bpb {r.bpb:.5f} · "
            f"{r.accounting.bytes_per_token:.3f} bytes/token · data {r.data_digest}"
        )

    if len(arms) == 2:
        cmp_ = compare(results, baseline="bf16", treatment="fp8")
        row["comparison"] = cmp_.as_dict()
        lo, hi = cmp_.baseline_sigma.ci()
        print(
            f"bf16 sigma_hat {cmp_.baseline_sigma.sigma:.5f} bpb "
            f"(dof {cmp_.baseline_sigma.dof}, 95% CI {lo:.5f}-{hi:.5f}) · "
            f"Delta bpb paired {cmp_.delta_bpb_paired:+.5f} · means {cmp_.delta_bpb_means:+.5f}"
        )
        print("gate: pending — bpb_gate_tolerance is unfilled (experiments/T1/T-R2/spec.md)")
        bare = cmp_.speedup
        print(
            f"speedup {bare:.4f}x · floors bf16_tok_s {cmp_.baseline_tok_s:,.1f} "
            f"bf16_bpb {cmp_.baseline_sigma.mean:.5f}"
        )
    else:
        tok_s = float(np.median([r.tok_s for r in results]))
        mean_bpb = float(np.mean([r.bpb for r in results]))
        row["floors"] = {"bf16_tok_s": tok_s, "bf16_bpb": mean_bpb}
        print(f'  make floor L=T1 R=T-R2 M=bf16_tok_s V={tok_s:.1f} DEV="<device>"')
        print(f'  make floor L=T1 R=T-R2 M=bf16_bpb  V={mean_bpb:.5f} DEV="<device>"')
        bare = mean_bpb if ns.floor == "bf16_bpb" else tok_s

    if ns.json:
        ns.json.parent.mkdir(parents=True, exist_ok=True)
        ns.json.write_text(json.dumps(row, indent=1), encoding="utf-8")
    print(f"{bare:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
