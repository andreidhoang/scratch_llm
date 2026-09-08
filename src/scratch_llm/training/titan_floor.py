"""T1/T-R0 — the torchtitan floor: one pinned config, one measurement protocol, one number.

"torchtitan Llama-3 1B" is not a number, it is a family. tok/s moves by more than 2x across the
family's free parameters (sequence length, local batch, TP degree, AC policy, compile on/off), so
T-R1's ">= 85% of torchtitan tok/s" is meaningless until one member is named. This module names it,
twice — once as a torchtitan config function (the thing torchtitan actually runs) and once as a
launcher that runs it, reads its log, and reduces 100 steps to a median with an IQR.

Two roles in one file, on purpose: the config and the harness that measures it must not be able to
drift apart. torchtitan's ``ConfigManager`` takes ``--module <importable> --config <function>``
(`oss/torchtitan/torchtitan/config/manager.py:105-159`), so this module *is* the config registry:

    torchrun ... -m torchtitan.train --module scratch_llm.training.titan_floor --config llama3_1b_t_r0

The torchtitan imports live inside the config functions, not at module scope, so this file imports
on a laptop with no torchtitan installed — ``--dry-run`` and the log parser are testable on CPU.

Protocol (workspace invariant 4, translated from kernels to a training loop):
  * clock-locked by ``infra/bench.sh`` (``nvidia-smi -lgc``) around the whole job;
  * warm-up = the first ``WARMUP_STEPS`` logged steps are discarded, not averaged in. Step 1
    carries torch.compile, FSDP2's first all-gather, and the allocator's first growth; including
    it would understate the floor by several percent;
  * >= 50 iterations: ``STEPS - WARMUP_STEPS`` = 100 logged steps at ``log_freq=1``, so each log
    line is exactly one step (`components/metrics.py:482-487` averages over the interval since the
    last log — at log_freq=10 you get 10-step averages and no IQR worth the name);
  * median + IQR, never the mean: a GC pause (``training.gc_freq=50``) or a straggler rank shows
    up as a tail, and the mean quietly eats it;
  * no L2 flush knob — a training step's working set is orders of magnitude past L2;
  * seed fixed (``debug.seed=0``); provenance captured by ``infra/bench.sh``.

The cross-check that makes the number trustworthy: torchtitan prints both ``tps`` and ``tflops``,
and ``tflops = num_flops_per_token * tps / 1e12`` (`components/metrics.py:493`). Dividing recovers
*torchtitan's own* FLOPs-per-token, which this harness compares against
``scratch_llm.training.flops`` computed from the shape. If the two disagree by more than
``FLOPS_CHECK_TOL``, the run is refused: it would mean the config actually executed is not the one
this module thinks it pinned (wrong flavor, wrong seq_len, silent quantization), and every derived
MFU would be a comparison against a different quantity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from scratch_llm.training.flops import (
    AC_SELECTIVE_OP,
    LLAMA3_1B,
    TORCHTITAN_PEAK_BF16_DENSE,
    flop_breakdown,
    step_report,
)

# --- the pin ------------------------------------------------------------------------------------
TITAN_PIN = "d263ca0a1b569ed198b9943b6e8c2117a61d8843"  # oss/torchtitan, the sha infra/bootstrap.sh installs
MODEL_FLAVOR = "1B"  # torchtitan/models/llama3/__init__.py:379
SEQ_LEN = 8192
LOCAL_BATCH = 2  # sequences per DP rank per forward
TOKENS_PER_MICROBATCH_PER_DP_RANK = LOCAL_BATCH * SEQ_LEN  # 16384
# Global tokens per optimizer step, pinned EXPLICITLY. Left at torchtitan's default (-1) it
# would be `microbatch × dp_degree` (`trainer.py:462-464`), which TP=2 halves — the two presets
# would then differ in global batch as well as in parallelism, and the comparison T-R1 rests on
# would be between two different optimizations. Pinning it makes the TP=2 arm run
# `gradient_accumulation_steps = 2` (`trainer.py:472-474`) at the same microbatch shape.
GLOBAL_TOKENS_PER_STEP = 8 * TOKENS_PER_MICROBATCH_PER_DP_RANK  # 131072 = 16384 tok/GPU/step
STEPS = 120
WARMUP_STEPS = 20  # discarded; STEPS - WARMUP_STEPS >= 50 is the measurement window
LOG_FREQ = 1
NPROC_PER_NODE = 8
SEED = 0
LEARNING_RATE = 3e-4  # llama3_8b's value (config_registry.py:254); throughput is LR-independent
FLOPS_CHECK_TOL = 0.01  # relative; the log's rounding contributes ~1e-4


@dataclass(frozen=True)
class Preset:
    """One member of the family, its ledger metric name, and the config function that builds it."""

    name: str
    config_fn: str
    tensor_parallel_degree: int
    metric: str
    note: str


PRESETS: dict[str, Preset] = {
    # The plan's T-R0 row: "torchtitan Llama-3 1B, FSDP2 + compile + AC".
    "fsdp8": Preset(
        name="fsdp8",
        config_fn="llama3_1b_t_r0",
        tensor_parallel_degree=1,
        metric="titan_llama3_1b_fsdp8_tps",
        note="FSDP2 over all 8 ranks, TP=1 — the plan's T-R0 row",
    ),
    # The plan's T1 exit criterion reads "≥ 85% torchtitan tok/s (FSDP2+TP=2)", which is a
    # DIFFERENT member: TP=2 halves the DP degree and adds two collectives per block. Measure both
    # or the 85% is against an unnamed quantity.
    "fsdp4_tp2": Preset(
        name="fsdp4_tp2",
        config_fn="llama3_1b_t_r0_tp2",
        tensor_parallel_degree=2,
        metric="titan_llama3_1b_fsdp4_tp2_tps",
        note="FSDP2 x 4, TP=2 — what T-R1's 85% target is written against",
    ),
}


# --- the torchtitan config (this module is the --module target) ---------------------------------
def llama3_1b_t_r0(seq_len: int | None = None):
    """The pinned floor config. Every value here is a choice T-R1 will be judged against.

    Pinned, and why:
      * ``model_registry("1B")`` — dim 2048, 16 layers, 32 q-heads / 8 kv-heads, head_dim 64,
        ffn_hidden 8192, vocab 128256, **tied embeddings**
        (`torchtitan/models/llama3/__init__.py:151-195`). There is no ``llama3_1b`` Trainer.Config
        upstream at this pin — the flavor exists, the recipe does not — which is why this function
        exists at all rather than a one-line ``--config`` reference.
      * ``max_context_length = 8192``. This is the load-bearing choice: attention FLOPs are 30% of
        model FLOPs here and 9% at 2048, so the same kernel scores very differently under the same
        formula. It is also the seq_len torchtitan feeds its own FLOP model (`trainer.py:431` passes
        ``config.training.max_context_length``).
      * ``num_tokens_per_microbatch_per_dp_rank = 16384`` = local batch 2 — the shape torchtitan's
        own published H100 table uses (`benchmarks/llama3_h100_202412_torchtitan.md`, Table 1) — and
        ``num_tokens_per_train_step = 131072`` pinned on top of it, so the global batch is the same
        number in both presets and only the parallelism differs (see ``GLOBAL_TOKENS_PER_STEP``).
      * ``compile.enable = True``, components ``["model", "loss"]`` (the default set): per-block
        ``torch.compile`` (`distributed/compile.py:69-70`) plus the chunked-loss region.
      * ``SelectiveAC`` — per-op AC, save the compute-intensive ops and recompute every second mm
        (`distributed/activation_checkpoint.py:186,265-279`). This is torchtitan's default and what
        its 8B/debug recipes use.
      * bf16 params / fp32 reduce (`TrainingConfig` defaults, configs.py:96-108) — FSDP2's
        MixedPrecisionPolicy, applied at `distributed/fsdp.py:223-227`.
      * ``c4_test`` + the in-repo test tokenizer: a 4.75 MB local JSON, repeated
        (`components/data/loader.py:65`), so the floor has no network in it and no gated download.
        Token *ids* do not change FLOPs or shapes; the lm_head GEMM is 128256-wide either way. The
        ``time_metrics/data_loading(%)`` column keeps that claim checkable.
      * defaults left alone deliberately: profiler off, checkpointing off, validation off,
        ``gc_freq=50``, ``debug.deterministic=False``, ``debug.enable_structured_logging=True``. A
        floor is the shipped thing, not a tuned thing; each of these is a one-flag experiment if
        the number looks wrong.

    Imports are function-local so this module stays importable without torchtitan (see module
    docstring); torchtitan calls this at `config/manager.py:158`.
    """
    from torchtitan.components.checkpointer import CheckpointManager
    from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
    from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
    from torchtitan.components.metrics import MetricsProcessor
    from torchtitan.components.optimizer import LRSchedulersContainer, default_adamw
    from torchtitan.config import CompileConfig, DebugConfig, ParallelismConfig, TrainingConfig
    from torchtitan.distributed.activation_checkpoint import SelectiveAC
    from torchtitan.hf_datasets.text_datasets import DATASETS
    from torchtitan.models.common.config_utils import decoder_vocab_size
    from torchtitan.models.llama3 import model_registry
    from torchtitan.trainer import Trainer

    model_spec = model_registry(MODEL_FLAVOR, seq_len=seq_len or SEQ_LEN)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model_spec)),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_spec,
        optimizer=default_adamw(lr=LEARNING_RATE),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=TOKENS_PER_MICROBATCH_PER_DP_RANK,
            num_tokens_per_train_step=GLOBAL_TOKENS_PER_STEP,
            max_context_length=model_spec.max_context_length,
            steps=STEPS,
        ),
        # torchtitan's default warmup is 200 steps, sized for a 10k-step run; at 120 steps its
        # build() clips it to the whole run with a warning (`lr_scheduler.py:107-112`). Pinned
        # short so the schedule is stated rather than clipped. Throughput does not depend on it.
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        # log_freq=1 => one log line per step => a real per-step distribution, not 10-step means.
        # disable_color_printing=True => no ANSI escapes for the parser to trip over.
        metrics=MetricsProcessor.Config(log_freq=LOG_FREQ, disable_color_printing=True),
        parallelism=ParallelismConfig(),  # dp_shard = -1 = all leftover ranks; TP = PP = CP = 1
        compile=CompileConfig(enable=True),
        activation_checkpoint=SelectiveAC.Config(),
        checkpoint=CheckpointManager.Config(enable=False),
        debug=DebugConfig(seed=SEED, print_config=True),
    )


def llama3_1b_t_r0_tp2(seq_len: int | None = None):
    """The same model, microbatch and **global batch**, sharded FSDP2 x 4 with TP=2 — T-R1's
    stated comparison point.

    Only the parallelism changes. TP=2 halves the DP degree to 4, so holding
    ``num_tokens_per_train_step`` at 131072 costs 2 gradient-accumulation steps
    (`trainer.py:472-474`) rather than halving the batch. Note what does *not* change: ``tps`` stays
    tokens/sec/GPU because `components/metrics.py:485-487` divides the rank-local token count by
    ``cp*tp*pp``, and grad accumulation scales tokens and time together — so the two presets'
    numbers are directly comparable.
    """
    config = llama3_1b_t_r0(seq_len=seq_len)
    config.parallelism.tensor_parallel_degree = 2
    return config


# --- log parsing ---------------------------------------------------------------------------------
# `[titan] <ts> - root - INFO - step: 12  loss: 8.12345  grad_norm: 1.2345  memory: 12.34GiB(9.9%)
#  tps: 12,345  tflops: 123.45  mfu: 12.34%`   (components/metrics.py:532-541, colors disabled)
_STEP_RE = re.compile(
    r"\bstep:\s*(?P<step>\d+)\b.*?"
    r"\bloss:\s*(?P<loss>[-\d.naN]+).*?"
    r"\btps:\s*(?P<tps>[\d,]+).*?"
    r"\btflops:\s*(?P<tflops>[\d,.]+).*?"
    r"\bmfu:\s*(?P<mfu>N/A|[\d.]+%)"
)
_PEAK_RE = re.compile(r"Peak FLOPS used for computing MFU:\s*(?P<peak>[\d.eE+-]+)")
_PARAMS_RE = re.compile(r"size:\s*(?P<params>[\d,]+) total parameters")


@dataclass(frozen=True)
class StepRow:
    step: int
    loss: float
    tps: float
    tflops: float
    mfu_pct: float | None

    @property
    def implied_flops_per_token(self) -> float:
        """torchtitan's own ``num_flops_per_token``, recovered from the two columns it prints."""
        return self.tflops * 1e12 / self.tps


@dataclass
class ParsedLog:
    steps: list[StepRow] = field(default_factory=list)
    peak_flops: float | None = None
    total_params: int | None = None


def parse_log(lines: list[str]) -> ParsedLog:
    """Pull the per-step throughput rows and the two provenance lines out of a torchtitan log."""
    out = ParsedLog()
    for line in lines:
        if out.peak_flops is None:
            peak = _PEAK_RE.search(line)
            if peak:
                out.peak_flops = float(peak.group("peak"))
        if out.total_params is None:
            params = _PARAMS_RE.search(line)
            if params:
                out.total_params = int(params.group("params").replace(",", ""))
        m = _STEP_RE.search(line)
        if m:
            mfu = m.group("mfu")
            out.steps.append(
                StepRow(
                    step=int(m.group("step")),
                    loss=float(m.group("loss")),
                    tps=float(m.group("tps").replace(",", "")),
                    tflops=float(m.group("tflops").replace(",", "")),
                    mfu_pct=None if mfu == "N/A" else float(mfu.rstrip("%")),
                )
            )
    return out


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile; explicit so the IQR does not depend on a numpy version."""
    if not sorted_values:
        raise ValueError("no values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def summarize(values: list[float]) -> dict[str, float]:
    """median, q25, q75, and the IQR as a percentage of the median (the trust column)."""
    s = sorted(values)
    med = statistics.median(s)
    q25, q75 = _percentile(s, 0.25), _percentile(s, 0.75)
    return {
        "median": med,
        "q25": q25,
        "q75": q75,
        "iqr_pct": 100.0 * (q75 - q25) / med if med else float("nan"),
        "min": s[0],
        "max": s[-1],
        "n": float(len(s)),
    }


# --- the run --------------------------------------------------------------------------------------
def titan_root(explicit: str | None = None) -> Path:
    """Absolute path to the pinned checkout. Absolute on purpose: it becomes a subprocess ``cwd``,
    and torchtitan's dataset/tokenizer paths are relative to it (``./tests/assets/...``)."""
    if explicit:
        return Path(explicit).resolve()
    from scratch_llm._workspace import workspace_root

    return (workspace_root() / "oss" / "torchtitan").resolve()


def torchrun_command(preset: Preset, *, nproc: int, steps: int | None) -> list[str]:
    """The exact argv, mirroring `oss/torchtitan/run_train.sh:43-45` with our --module/--config."""
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint=localhost:0",
        "--local-ranks-filter",
        "0",
        "--role",
        "rank",
        "--tee",
        "3",
        "-m",
        "torchtitan.train",
        "--module",
        "scratch_llm.training.titan_floor",
        "--config",
        preset.config_fn,
    ]
    if steps is not None:
        cmd += ["--training.steps", str(steps)]
    return cmd


def run_env() -> dict[str, str]:
    """torchtitan's own alloc setting (run_train.sh:41) plus src/ on PYTHONPATH for --module."""
    env = dict(os.environ)
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    src = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = f"{src}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src
    return env


def measure(
    preset: Preset,
    *,
    root: Path,
    nproc: int,
    steps: int,
    warmup_steps: int,
    log_path: Path | None = None,
) -> dict:
    """Run the job, parse its log, reduce to median+IQR, and cross-check the FLOP accounting."""
    cmd = torchrun_command(preset, nproc=nproc, steps=None if steps == STEPS else steps)
    # Streamed, not captured: this job runs for ten-plus minutes and a silent pipe is a job you
    # cannot tell from a hang. torchtitan's lines go to STDERR (so stdout keeps the harness
    # convention that its last line is the one bare number) and to --log as they arrive, so a
    # crash at step 80 still leaves 80 steps of evidence on disk.
    lines: list[str] = []
    sink = log_path.open("w", encoding="utf-8") if log_path is not None else None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            env=run_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip("\n"))
            sys.stderr.write(line)
            if sink is not None:
                sink.write(line)
        returncode = proc.wait()
    finally:
        if sink is not None:
            sink.close()
    if returncode != 0:
        raise SystemExit(f"titan_floor: torchrun exited {returncode} — see {log_path}")
    return reduce_log(lines, preset=preset, warmup_steps=warmup_steps, nproc=nproc)


def reduce_log(lines: list[str], *, preset: Preset, warmup_steps: int, nproc: int) -> dict:
    """Log lines -> the rung's number, its spread, and every MFU convention beside it."""
    parsed = parse_log(lines)
    if not parsed.steps:
        raise SystemExit(
            "titan_floor: no step lines parsed. Was metrics.disable_color_printing set, and did "
            "the job reach step 1? (torchtitan logs step 1 then every log_freq steps.)"
        )
    window = [row for row in parsed.steps if row.step > warmup_steps]
    if len(window) < 50:
        raise SystemExit(
            f"titan_floor: {len(window)} steps after a {warmup_steps}-step warm-up; the workspace "
            f"asks for >= 50. Raise --steps (default {STEPS})."
        )

    tps = summarize([row.tps for row in window])
    titan_mfu = [row.mfu_pct for row in window if row.mfu_pct is not None]
    peak = parsed.peak_flops or TORCHTITAN_PEAK_BF16_DENSE["H100 SXM"]

    breakdown = flop_breakdown(LLAMA3_1B, SEQ_LEN, ac_policy=AC_SELECTIVE_OP)
    implied = statistics.median([row.implied_flops_per_token for row in window])
    ratio = implied / breakdown.model_per_token
    if abs(ratio - 1.0) > FLOPS_CHECK_TOL:
        raise SystemExit(
            "titan_floor: REFUSED — torchtitan's FLOPs/token "
            f"({implied:.6g}, recovered from its tflops/tps columns) disagrees with this module's "
            f"model ({breakdown.model_per_token:.6g}) by {100 * (ratio - 1):.2f}%. The config that "
            "ran is not the config this module pinned; the number would be against a different "
            "quantity. Compare seq_len, flavor, and quantization before recording anything."
        )

    report = step_report(
        LLAMA3_1B,
        seq_len=SEQ_LEN,
        tokens_per_s_per_device=tps["median"],
        peak_flops_per_device=peak,
        ac_policy=AC_SELECTIVE_OP,
    )
    return {
        "rung": "T1/T-R0",
        "preset": preset.name,
        "metric": preset.metric,
        "note": preset.note,
        "torchtitan_pin": TITAN_PIN,
        "model_flavor": MODEL_FLAVOR,
        "seq_len": SEQ_LEN,
        "local_batch": LOCAL_BATCH,
        "tensor_parallel_degree": preset.tensor_parallel_degree,
        "nproc_per_node": nproc,
        "warmup_steps": warmup_steps,
        "measured_steps": len(window),
        "tps_per_device": tps,
        "titan_reported_mfu_pct": summarize(titan_mfu) if titan_mfu else None,
        # Recorded, not gated: a NaN or a rising loss means the floor diverged and the tok/s is a
        # number for a run nobody would ship. Reading it is Huy's call, not a constant of mine.
        "loss": summarize([row.loss for row in window]),
        "loss_first_last": [window[0].loss, window[-1].loss],
        "titan_total_params": parsed.total_params,
        "titan_peak_flops": peak,
        "titan_implied_flops_per_token": implied,
        "flops_per_token_agreement": ratio,
        "accounting": report.as_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="T1/T-R0 — measure the torchtitan Llama-3 1B floor (tok/s per GPU)."
    )
    ap.add_argument("--preset", choices=sorted(PRESETS), default="fsdp8")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    ap.add_argument("--nproc", type=int, default=NPROC_PER_NODE)
    ap.add_argument("--titan-root", default=None, help="default: <workspace>/oss/torchtitan")
    ap.add_argument("--json", default=None, help="write the full result dict here")
    ap.add_argument("--log", default=None, help="write torchtitan's raw log here")
    ap.add_argument("--dry-run", action="store_true", help="print the command, measure nothing")
    ap.add_argument("--parse", default=None, help="reduce an existing log file instead of running")
    args = ap.parse_args(argv)

    preset = PRESETS[args.preset]
    root = titan_root(args.titan_root)
    if args.dry_run:
        # exactly what measure() would run: steps ride in the config unless overridden
        cmd = torchrun_command(
            preset, nproc=args.nproc, steps=None if args.steps == STEPS else args.steps
        )
        print(f"# cwd={root}")
        print(f"# metric={preset.metric}  ({preset.note})")
        print("# PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=<scratch_llm/src>")
        print(" ".join(cmd))
        return 0

    if args.parse:
        result = reduce_log(
            Path(args.parse).read_text(encoding="utf-8").splitlines(),
            preset=preset,
            warmup_steps=args.warmup_steps,
            nproc=args.nproc,
        )
    else:
        result = measure(
            preset,
            root=root,
            nproc=args.nproc,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            log_path=Path(args.log) if args.log else None,
        )

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=1), encoding="utf-8")

    acc = result["accounting"]
    tps = result["tps_per_device"]
    sys.stderr.write(
        f"{preset.metric}: median {tps['median']:.1f} tok/s/GPU over {result['measured_steps']} "
        f"steps (IQR {tps['iqr_pct']:.2f}%)\n"
        f"  MFU 6ND only          {100 * acc['mfu_6nd_only']:.2f}%\n"
        f"  MFU model (torchtitan) {100 * acc['mfu_model_torchtitan']:.2f}%   "
        f"[attention = {100 * acc['attention_share_of_model']:.1f}% of it]\n"
        f"  MFU model, causal 0.5  {100 * acc['mfu_model_causal_half']:.2f}%\n"
        f"  HFU executed (AC x{acc['ac_multiplier']:.4f}) {100 * acc['hfu_executed']:.2f}%\n"
    )
    # The harness convention: the last line of stdout is the one bare number.
    print(f"{tps['median']:.1f}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
