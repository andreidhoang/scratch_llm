"""The S1 serving ladder's measurement entry point — S-R3's two numbers and their two floors.

S-R3 makes two claims that share a rung and share nothing else: CUDA graphs make the decode step
cheaper (a host-side submission claim, judged against EAGER decode at a fixed batch), and
speculation makes tokens cheaper (a target-forward-count claim, judged against NON-SPECULATIVE
decode of the same target). They are measured by the same driver so the methodology is identical —
same clock lock, same warm-ups, same median+IQR — and reported as two metrics so that a prediction
error on one is never hidden by the other.

Three things this driver has to get right that a kernel bench does not:

  1. **A decode "iteration" is a step, not a kernel.** The unit is ms/decode-step at a stated batch,
     with the aggregate tok/s beside it. Both arms run the SAME paged Triton kernel, so the delta
     isolates graph replay against eager launch, exactly as ``bench/cudagraph_decode.py`` set up.
  2. **No L2 flush.** Both arms issue an identical device-side kernel sequence over an identical
     fixed-address pool; the difference is host submission. A flush would add the same device-side
     constant to both arms and pull the ratio toward 1 — a real effect, which is why the absolute
     ms/step of both arms is printed next to the ratio and the ratio is never quoted alone. The
     memory-bound side of decode is S-R1's number, not this one.
  3. **Acceptance is per position.** The spec arm reports the conditional profile, the cumulative
     profile (vLLM's), and the flat average, in that order of usefulness. The analytic model that
     turns the profile into a predicted speedup is S-R3's `# HUY:` hole
     (``scratch_llm.serving.acceptance.spec_round_model``); until it is filled this driver prints
     the measured numbers and says the model is unwritten, rather than substituting an average.

    python bench/serving/s1_ladder.py --mode graphs            the graph number (default)
    python bench/serving/s1_ladder.py --mode spec              the speculation number
    python bench/serving/s1_ladder.py --floor eager            the eager-decode floor alone
    python bench/serving/s1_ladder.py --floor nospec           the plain-decode floor alone
    python bench/serving/s1_ladder.py --mode graphs --dry-run  print the plan, run nothing

The last line of stdout is always a single bare number — the mode's metric, or the floor's value.
``experiments/S1/S-R3/{run,floor}.sh`` read exactly that line. Recorded measurements go through
``infra/bench.sh``, which locks clocks, records provenance, and refuses to run without a prediction
already in the ledger.

Spec: experiments/S1/S-R3/spec.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _bootstrap_path() -> None:
    """Put ``src/`` and ``bench/`` on ``sys.path``.

    Called from the import helpers rather than at module import, so ``--dry-run`` works on a box
    with no torch (this Mac) and so the module stays free of import-not-at-top.
    """
    for rel in ("src", "bench"):
        p = str(_REPO_ROOT / rel)
        if p not in sys.path:
            sys.path.insert(0, p)


# =================================================================================================
# The shapes. Fixed by plan §05's S1 row: the exit is stated at B=32, so that is the graph shape.
# =================================================================================================


@dataclass(frozen=True)
class GraphShape:
    """The decode shape the graph claim is measured at. ``batch`` is the number of live rows."""

    batch: int = 32
    prompt: int = 32
    warmup_steps: int = 20
    iters: int = 50

    @property
    def label(self) -> str:
        return f"B={self.batch} prompt={self.prompt}"


@dataclass(frozen=True)
class SpecShape:
    """The speculation shape. Batch 1 because ``speculative_generate`` is single-sequence.

    That is a limitation of the substrate and not a choice: batched speculation needs per-row
    rollback of a paged cache, which this engine's ``KVCache.truncate`` does not express. It is
    also why this number cannot share a ledger row with the graph number — see spec.md.
    """

    max_new_tokens: int = 128
    draft_len: int = 4
    prompt_period: int = 16
    prompt_len: int = 64
    warmup_iters: int = 20
    iters: int = 50


@dataclass
class Result:
    """One measured row. ``value`` is what lands on the last line of stdout."""

    metric: str
    value: float
    unit: str
    detail: dict = field(default_factory=dict)


# =================================================================================================
# Timing — one methodology for both arms
# =================================================================================================


def _time_ms(fn, n_warmup: int, n_iters: int) -> tuple[float, float, float]:
    """Median, p20, p80 milliseconds per call — ``bench/_harness.wallclock_ms``, not a private loop.

    ``_harness`` is the single source of measurement truth for this repo, and it already draws the
    line this rung sits on: :func:`bench_ms` is the KERNEL methodology (CUDA events, L2 flush
    between reps, microseconds) and :func:`wallclock_ms` is the ENGINE one (host clock, device
    synchronized inside the window, hundreds of kernels behind a Python loop). A CUDA-event window
    would be the wrong instrument here twice over — it measures the stream, and this rung's whole
    claim is about the host-side submission the stream does not see; and it would make S-R3's step
    time incommensurable with S-R1's, which is the number it has to be readable against.

    The import is deferred because ``_harness`` imports triton at module scope, and ``--dry-run``
    has to work on a box with neither triton nor a GPU.
    """
    _bootstrap_path()
    from _harness import wallclock_ms

    return wallclock_ms(fn, warmup=n_warmup, iters=n_iters)


# =================================================================================================
# The graph arm
# =================================================================================================


def _build_model(device: str, ckpt: str = ""):  # noqa: ANN202 — torch types not importable here
    """The target. ``ckpt`` loads a TRAINED one; without it the weights are random.

    Random weights are fine for the graph arm — capture is about how a step is launched, and an
    untrained model launches the same kernels. They are useless for the spec arm:
    ``bench/f3_deconfound_acceptance.py:3-5`` records that an untrained target's greedy
    continuation has no structure for a drafter to track, so acceptance measures ~0 on every
    prompt and the "speedup" is pure drafter overhead. ``measure_spec`` refuses to emit a number
    in that state rather than letting it reach the ledger.
    """
    _bootstrap_path()
    import torch

    from decode_roofline import RUNG1_CONFIG

    torch.manual_seed(0)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if ckpt:
        from scratch_llm.train import build_model_from_checkpoint

        model, _step = build_model_from_checkpoint(ckpt, map_location=device)
        model = model.to(device=device, dtype=dtype).eval()
        return model, model.cfg

    from scratch_llm.model import TransformerLM

    return TransformerLM(RUNG1_CONFIG).to(device=device, dtype=dtype).eval(), RUNG1_CONFIG


def _prefilled(model, cfg, batch: int, prompt: int, device: str):  # noqa: ANN202
    """A paged cache prefilled with ``batch`` random prompts, plus each row's first token."""
    import random

    import torch

    from scratch_llm.model import PagedKVCache, PrefillView

    block = PagedKVCache.BLOCK
    max_blocks = (prompt + 256 + block - 1) // block
    cache = PagedKVCache(
        n_layers=cfg.n_layers,
        n_slots=batch,
        n_kv_heads=cfg.kv_heads,
        max_ctx=cfg.context_length,
        head_dim=cfg.head_dim,
        n_blocks=batch * max_blocks + 1,
        device=device,
        dtype=torch.bfloat16,
    )
    cache.use_kernel = True
    rng = random.Random(1)
    ids = [[rng.randrange(cfg.vocab_size) for _ in range(prompt)] for _ in range(batch)]
    x = torch.tensor(ids, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x, PrefillView(cache, list(range(batch)), [prompt] * batch))
    cache.mirror_admit(list(range(batch)), [prompt] * batch)
    first = logits[torch.arange(batch, device=device), prompt - 1].argmax(dim=-1)
    return cache, first


def _eager_step_fn(model, cache):  # noqa: ANN202
    """The eager decode step, as a closure over its own last-token state — the floor arm."""
    import torch

    state = {"last": None}

    def step(last=None):  # noqa: ANN001, ANN202
        with torch.no_grad():
            tokens = state["last"] if last is None else last
            cache.pre_decode_reserve()
            logits = model(tokens.unsqueeze(1), cache)
            nxt = logits[:, -1].argmax(dim=-1)
            cache.mirror_advance()
            state["last"] = nxt
            return nxt

    return step, state


def measure_graphs(shape: GraphShape, device: str, buckets: tuple[int, ...] | None) -> Result:
    """Eager ms/step vs bucketed-graph ms/step at ``shape.batch``; metric = the ratio."""
    _bootstrap_path()
    from scratch_llm.serving.graph_buckets import (
        BucketedGraphDecoder,
        default_decode_buckets,
        paged_capture_factory,
    )

    model, cfg = _build_model(device)
    ladder = buckets or default_decode_buckets(shape.batch)

    eager_cache, eager_first = _prefilled(model, cfg, shape.batch, shape.prompt, device)
    eager_step, eager_state = _eager_step_fn(model, eager_cache)
    eager_state["last"] = eager_first
    eager_ms, eager_lo, eager_hi = _time_ms(eager_step, shape.warmup_steps, shape.iters)

    dec = BucketedGraphDecoder(
        ladder,
        paged_capture_factory(model, lambda b: _prefilled(model, cfg, b, shape.prompt, device)),
        eager_step,
        device=device,
    )
    dec.capture_all()
    graph_state = {"last": eager_first.clone()}

    def graph_step() -> None:
        graph_state["last"] = dec.step(graph_state["last"])

    graph_ms, graph_lo, graph_hi = _time_ms(graph_step, shape.warmup_steps, shape.iters)

    speedup = eager_ms / graph_ms
    print(
        f"  {shape.label} | eager {eager_ms:6.3f} ms/step (p20-p80 {eager_lo:.3f}-{eager_hi:.3f}) | "
        f"graph {graph_ms:6.3f} ms/step (p20-p80 {graph_lo:.3f}-{graph_hi:.3f}) | "
        f"x{speedup:.3f} | agg {shape.batch * 1e3 / graph_ms:.0f} tok/s "
        f"(eager {shape.batch * 1e3 / eager_ms:.0f})"
    )
    print(f"  buckets {ladder} | dispatch {dec.stats()}")
    return Result(
        metric="graph_speedup_vs_eager",
        value=speedup,
        unit="x",
        detail={
            "eager_ms_per_step": eager_ms,
            "eager_spread_p20_p80_ms": [eager_lo, eager_hi],
            "graph_ms_per_step": graph_ms,
            "graph_spread_p20_p80_ms": [graph_lo, graph_hi],
            "buckets": list(ladder),
            "dispatch": dec.stats(),
            **asdict(shape),
        },
    )


def measure_eager_floor(shape: GraphShape, device: str) -> Result:
    """The eager-decode floor alone: ms/step at ``shape.batch``, same kernel, same pool."""
    _bootstrap_path()
    model, cfg = _build_model(device)
    cache, first = _prefilled(model, cfg, shape.batch, shape.prompt, device)
    step, state = _eager_step_fn(model, cache)
    state["last"] = first
    ms, lo, hi = _time_ms(step, shape.warmup_steps, shape.iters)
    print(f"  eager decode | {shape.label} | {ms:6.3f} ms/step (p20-p80 {lo:.3f}-{hi:.3f})")
    return Result(
        metric="eager_decode_ms_per_step_b32",
        value=ms,
        unit="ms/step",
        detail={"spread_p20_p80_ms": [lo, hi], **asdict(shape)},
    )


# =================================================================================================
# The speculation arm
# =================================================================================================


def _repetitive_prompt(cfg, period: int, length: int) -> list[int]:  # noqa: ANN001
    """A prompt with an exact period — the regime where prompt-lookup speculation has anything to
    accept at all. An unstructured prompt gives ~0 acceptance and measures the drafter's cost only;
    both are run so the spec claim is quoted with the regime it holds in."""
    base = [(i * 7 + 3) % cfg.vocab_size for i in range(period)]
    return (base * ((length // period) + 1))[:length]


def _tok_s(fn, shape: SpecShape) -> tuple[float, float, float]:
    """Median tok/s and its p20-p80 band, from the same :func:`_time_ms` window as the graph arm.

    A whole ``max_new_tokens`` generation is one iteration: speculation's unit of work is a ROUND,
    rounds commit a variable number of tokens, and a per-step time would therefore be a time per
    variable-sized thing. Tokens per second over a fixed token budget is the only rate that means
    the same thing in both arms. The band is inverted with the median (fast ms = high tok/s), so
    ``lo`` below is the slow tail.
    """
    med, lo, hi = _time_ms(fn, shape.warmup_iters, shape.iters)
    n = shape.max_new_tokens
    return n * 1e3 / med, n * 1e3 / hi, n * 1e3 / lo


def _spec_arms(shape: SpecShape, device: str, ckpt: str = "", draft_ckpt: str = ""):  # noqa: ANN202
    """Build the plain-decode and speculative closures over one shared model and prompt.

    ``draft_ckpt`` swaps the training-free n-gram drafter for a smaller model
    (:class:`ModelDrafter`) — the substrate's stand-in for the plan's Qwen3-0.6B draft, which has
    no loader here. Both satisfy the same ``Drafter`` protocol, so nothing else changes.
    """
    _bootstrap_path()
    from scratch_llm.sampling import SamplingParams, generate
    from scratch_llm.serving.acceptance import PositionAcceptance
    from scratch_llm.serving.speculative import ModelDrafter, NGramDrafter, speculative_generate

    model, cfg = _build_model(device, ckpt)
    prompt = _repetitive_prompt(cfg, shape.prompt_period, shape.prompt_len)
    if draft_ckpt:
        draft_model, _ = _build_model(device, draft_ckpt)
        drafter = ModelDrafter(model=draft_model, device=device)
    else:
        drafter = NGramDrafter(n=3)
    acc = PositionAcceptance(draft_len=shape.draft_len)

    def plain() -> None:
        generate(
            model, prompt, SamplingParams(temperature=0.0, max_tokens=shape.max_new_tokens), device
        )

    def spec() -> None:
        speculative_generate(
            model,
            drafter,
            prompt,
            shape.max_new_tokens,
            device,
            k=shape.draft_len,
            on_round=acc.observe_round,
        )

    return plain, spec, acc, drafter, prompt


class NothingWasDrafted(RuntimeError):
    """Raised when the run produced no draft tokens at all, so the ratio measures nothing."""


def measure_spec(
    shape: SpecShape, device: str, ckpt: str = "", draft_ckpt: str = "", cost_ratio: float = -1.0
) -> Result:
    """Speculative tok/s against non-speculative tok/s for the same target; metric = the ratio."""
    plain, spec, acc, drafter, prompt = _spec_arms(shape, device, ckpt, draft_ckpt)
    base, base_lo, base_hi = _tok_s(plain, shape)
    spec_rate, spec_lo, spec_hi = _tok_s(spec, shape)
    speedup = spec_rate / base

    if sum(acc.offered) == 0:
        raise NothingWasDrafted(
            f"the drafter proposed nothing in {acc.n_rounds} rounds, so x{speedup:.3f} is the cost "
            "of asking, not the value of speculating, and there is no acceptance profile to report. "
            "Almost always an UNTRAINED target (bench/f3_deconfound_acceptance.py:3-5): pass "
            "--ckpt <trained checkpoint>. This is refused rather than printed because the bare last "
            "line is what run.sh hands to `make measure`."
        )

    # The cost ratio the model wants: one draft step over one target decode forward. The
    # denominator is exact and already measured — plain greedy decode is one target forward per
    # token, so it is 1e3 / base. The numerator is one `propose` call divided by the k tokens it
    # returns. Measured, not assumed, and printed so it can be argued with.
    draft_ms = _time_ms(
        lambda: drafter.propose(prompt, shape.draft_len), shape.warmup_iters, shape.iters
    )[0]
    target_ms_per_forward = 1e3 / base
    measured_ratio = (draft_ms / shape.draft_len) / target_ms_per_forward
    effective_ratio = measured_ratio if cost_ratio < 0 else cost_ratio

    cond = acc.conditional()
    cond_s = ", ".join("--" if a is None else f"{a:.3f}" for a in cond)
    cum_s = ", ".join(f"{a:.3f}" for a in acc.cumulative())
    print(
        f"  plain {base:8.1f} tok/s (p20-p80 {base_lo:.1f}-{base_hi:.1f}) | "
        f"spec {spec_rate:8.1f} tok/s (p20-p80 {spec_lo:.1f}-{spec_hi:.1f}) | x{speedup:.3f}"
    )
    print(f"  alpha conditional (per position, the model's input): [{cond_s}]")
    print(f"  alpha cumulative  (vLLM's per-pos log line):         [{cum_s}]")
    print(
        f"  flat average {acc.flat_acceptance_rate:.3f} · offered {acc.offered} · "
        f"reached {acc.reached} · accepted {acc.accepted} · rounds {acc.n_rounds}"
    )
    print(f"  measured E[accepted]/round = {acc.mean_accepted_per_round:.4f}")
    print(
        f"  draft {draft_ms / shape.draft_len:.3f} ms/token vs target "
        f"{target_ms_per_forward:.3f} ms/forward -> cost_ratio {measured_ratio:.4f} "
        f"(model fed {effective_ratio:.4f})"
    )
    print(_analytic_line(cond, effective_ratio))
    return Result(
        metric="spec_speedup_vs_nonspec",
        value=speedup,
        unit="x",
        detail={
            "nonspec_tok_s": base,
            "nonspec_spread_p20_p80_tok_s": [base_lo, base_hi],
            "spec_tok_s": spec_rate,
            "spec_spread_p20_p80_tok_s": [spec_lo, spec_hi],
            "alpha_conditional": cond,
            "alpha_cumulative": acc.cumulative(),
            "flat_acceptance_rate": acc.flat_acceptance_rate,
            "offered": acc.offered,
            "reached": acc.reached,
            "accepted": acc.accepted,
            "n_rounds": acc.n_rounds,
            "mean_accepted_per_round": acc.mean_accepted_per_round,
            "draft_ms_per_token": draft_ms / shape.draft_len,
            "target_ms_per_forward": target_ms_per_forward,
            "cost_ratio_measured": measured_ratio,
            "cost_ratio_used": effective_ratio,
            **asdict(shape),
        },
    )


def _analytic_line(cond: list, cost_ratio: float) -> str:
    """The measured profile put through the analytic model, or a statement that it is unwritten.

    ``cost_ratio`` is the measured draft/target ratio by default and ``--cost-ratio`` when the
    caller overrides it — the second of the model's two inputs, and a property of the drafter/target
    PAIR rather than of either alone. With the n-gram drafter it is near zero (there is no model
    forward to pay for); with ``--draft-ckpt`` it is a real number, and the one that decides
    break-even K.
    """
    _bootstrap_path()
    from scratch_llm.serving.acceptance import spec_round_model

    if any(a is None for a in cond):
        return "  analytic: some positions were never reached — no conditional profile to model"
    try:
        r = spec_round_model([float(a) for a in cond], cost_ratio)
    except NotImplementedError:
        return (
            "  analytic: spec_round_model is S-R3's HUY hole — measured profile above is its "
            "input; `make holes` lists it"
        )
    return f"  analytic: E[accepted]={r.expected_accepted:.4f} speedup=x{r.speedup:.3f}"


def measure_nospec_floor(shape: SpecShape, device: str, ckpt: str = "") -> Result:
    """The non-speculative floor alone: greedy ``sampling.generate`` tok/s on the same prompt."""
    plain, _spec, _acc, _drafter, _prompt = _spec_arms(shape, device, ckpt)
    rate, lo, hi = _tok_s(plain, shape)
    print(
        f"  plain greedy decode | max_new={shape.max_new_tokens} | {rate:.1f} tok/s "
        f"(p20-p80 {lo:.1f}-{hi:.1f})"
    )
    return Result(
        metric="nonspec_decode_tok_s",
        value=rate,
        unit="tok/s",
        detail={"spread_p20_p80_tok_s": [lo, hi], **asdict(shape)},
    )


# =================================================================================================
# CLI
# =================================================================================================


def _plan(args: argparse.Namespace) -> str:
    what = f"--floor {args.floor}" if args.floor else f"--mode {args.mode}"
    return (
        f"s1_ladder S-R3 [dry-run] {what} · device {args.device}\n"
        f"  graph shape: {GraphShape(batch=args.batch)}\n"
        f"  spec shape:  {SpecShape(draft_len=args.draft_len)}\n"
        f"  target: {args.ckpt or 'RANDOM WEIGHTS — spec arm will refuse; pass --ckpt'}\n"
        f"  drafter: {args.draft_ckpt or 'n-gram(n=3), training-free'}\n"
        f"  floors: eager decode ms/step (graph claim) · plain greedy tok/s (spec claim)\n"
        f"  last stdout line is the bare metric value"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S1 serving ladder — S-R3 graphs + speculation")
    ap.add_argument("--mode", choices=("graphs", "spec"), default="graphs")
    ap.add_argument("--floor", choices=("eager", "nospec"), default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32, help="decode batch for the graph claim")
    ap.add_argument("--draft-len", type=int, default=4, help="K, draft tokens per round")
    ap.add_argument(
        "--buckets", default="", help="comma-separated capture sizes (default: derived)"
    )
    ap.add_argument("--ckpt", default="", help="trained target checkpoint — REQUIRED for spec")
    ap.add_argument("--draft-ckpt", default="", help="smaller draft model (else the n-gram one)")
    ap.add_argument(
        "--cost-ratio",
        type=float,
        default=-1.0,
        help="draft step / target forward; <0 (default) measures it",
    )
    ap.add_argument("--json", default="", help="write the full result dict here")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.dry_run:
        print(_plan(args))
        return 0

    gshape = GraphShape(batch=args.batch)
    sshape = SpecShape(draft_len=args.draft_len)
    buckets = tuple(int(x) for x in args.buckets.split(",")) if args.buckets else None

    try:
        if args.floor == "eager":
            res = measure_eager_floor(gshape, args.device)
        elif args.floor == "nospec":
            res = measure_nospec_floor(sshape, args.device, args.ckpt)
        elif args.mode == "graphs":
            res = measure_graphs(gshape, args.device, buckets)
        else:
            res = measure_spec(sshape, args.device, args.ckpt, args.draft_ckpt, args.cost_ratio)
    except NothingWasDrafted as exc:
        print(f"s1_ladder: REFUSED — {exc}")
        return 2  # no bare last line: there is no number to record

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(asdict(res), indent=2), encoding="utf-8")

    print(f"{res.metric} = {res.value:.6g} {res.unit}")
    print(f"{res.value:.6g}")  # the bare number run.sh / floor.sh read
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
