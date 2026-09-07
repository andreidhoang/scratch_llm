"""K3/PR-2 — the KDA chunked-scan memory model, and the startup check that spends it.

vLLM issue #54775. A KDA (Kimi Delta Attention) *prefill* does not stream. It materialises the
whole chunk-recurrent state for the batch before the output kernel reads a single element of it,
and two of those buffers are linear in the number of tokens in the batch. Nothing in vLLM's
startup accounting knows they exist, so the first real prefill after a clean profile run can take
the process straight to OOM at a batch size the engine itself chose.

Where the bytes are, on ``oss/vllm`` @ ``fix/kda-chunk-buffers-memory-profiling-54775``::

    vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py:320  chunk_gated_delta_rule_fwd_h
      :345   NT = triton.cdiv(T, BT)          # one packed sequence, cu_seqlens is None
      :347   NT = len(chunk_indices)          # varlen: the batch's TOTAL chunk count
      :350   assert K <= 256                  # the kernel's own head-dim ceiling
      :352   h           = k.new_empty(B, NT, H, V, K)              # k.dtype; B == 1 in varlen
      :354   final_state = k.new_empty(N, H, V, K, torch.float32)   # iff output_final_state
      :357   v_new       = torch.empty_like(u)                      # (B, T, H, V), k.dtype

    called from vllm/models/kimi_k3/nvidia/ops/third_party/kda/chunk.py:627 — and that caller is
    still holding ``w``, ``u``, ``kg`` (chunk.py:209-211, each ``empty_like`` of k or v) and
    ``Aqk`` (chunk_intra.py:488, ``(B, T, HV, BT)``) live *across* the call. It frees them at
    chunk.py:651, after the output kernel. So one layer's peak is strictly larger than
    ``fwd_h``'s own footprint, and a model that stops at ``h + v_new`` will under-predict.

    Same code upstream, for the fla-side half of K3: oss/fla/fla/ops/common/chunk_delta_h.py:687
    (``h`` at :715/:718, ``v_new`` at :721).

``NT`` is the whole game and it is not ``T/64``. In varlen mode it is the batch's total chunk
count, ``sum_i ceil(T_i / BT)`` — so for a fixed token budget it depends on how the tokens are
split across sequences, and the split that maximises it is not the one a single-sequence dummy
batch produces. That gap is exactly why the profiling fix (direction 1) insists on a
multi-sequence profile batch, and why the check below takes ``num_seqs`` as a shape parameter
rather than assuming one packed sequence.

Layers do not stack: chunk.py:651 releases the buffers before the next KDA layer runs, so the
caching allocator hands the same blocks back and the engine-wide peak is one layer's live set,
not ``num_layers`` of them. Whether the allocator actually achieves that under fragmentation is a
modelling question, which is to say it is Huy's.

Two things live in this module and only one of them is written here:

* :func:`kda_chunk_scan_peak_bytes` — the model. Open ``# HUY:`` hole. CLAUDE.md keeps the memory
  model with Huy for the same reason it keeps the roofline: a memory model you did not derive is
  a number you cannot defend, and this one decides whether the engine refuses to start.
* everything else — the shape, the validation, the budget arithmetic, the refusal and its message.
  Plumbing. Written now, tested on CPU against a stub model, so that filling the model is the
  whole of the remaining work.

Spec, floor and kill rule: ``experiments/K3/PR-2/spec.md``.
Measured curve the model is falsified against: ``experiments/K3/PR-2/measure_peak_alloc.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

#: The rung identity, in one place, so module / test / spec / PR body cannot drift apart.
RUNG = "K3/PR-2"

#: Upstream's chunk length for the gated delta rule, mirrored from
#: ``vllm/third_party/flash_linear_attention/ops/utils.py:31`` (``FLA_CHUNK_SIZE = 64``). It is a
#: *default*, not a constant: the kernel takes ``chunk_size`` and the model must be a function of
#: it, because halving BT doubles NT and therefore doubles ``h``.
FLA_CHUNK_SIZE = 64

#: The kernel's own head-dimension ceiling — ``assert K <= 256`` at chunk_delta_h.py:350. Mirrored
#: so a bad config is rejected here, at startup, with a sentence, instead of inside a Triton launch.
MAX_HEAD_K = 256

# ---------------------------------------------------------------------------------------------
# The safety margin. HUY SETS THIS, with its argument, before the check is armed.
#
# The check has two ways to be wrong and they are not symmetric. Refusing a config that would in
# fact have run costs a user a working server and gets the check reverted. Admitting a config that
# then OOMs costs a crash mid-serve, which is what #54775 already does today. The margin is where
# that asymmetry gets priced, and it has to be priced against a *measured* prediction error — the
# `pred err %` from experiments/K3/PR-2/results/, not a round number chosen because it looks safe.
#
# Leaving it None is deliberate, and None is not "no check": it means the comparison is exact
# (predicted > available refuses), which is the honest behaviour of a model whose error bar has
# not been measured yet.
# ---------------------------------------------------------------------------------------------
SAFETY_MARGIN: float | None = None


class KdaScanWouldNotFit(RuntimeError):
    """The predicted KDA chunked-scan peak does not fit in the memory left for it.

    Raised at startup by :func:`check_kda_chunk_scan_headroom`, which is the entire point: a
    ``RuntimeError`` naming the four knobs beats a CUDA OOM inside the first prefill, where the
    traceback points at a Triton launch and says nothing about ``--max-num-batched-tokens``.
    """


@dataclass(frozen=True)
class KdaScanShape:
    """One KDA layer's chunked scan, as the *worst* batch the engine can schedule presents it.

    Field names follow the kernel's, not the model's: ``num_v_heads`` is ``H`` at
    chunk_delta_h.py:338 (``u.shape[-2]``, the value-head count, which for KDA may exceed the
    query-head count ``Hg``), ``head_k`` is ``K``, ``head_v`` is ``V``. Keeping the kernel's names
    is what lets a reader check the model against the allocation lines without a translation step.

    Frozen because a shape that can be mutated after the budget was computed is a budget about a
    batch that no longer exists.
    """

    #: Number of sequences in the batch. ``N`` at chunk_delta_h.py:347; also the leading dim of
    #: ``final_state`` at :354. Never zero — vLLM's dummy batch builder guarantees at least one
    #: (gpu_model_runner.py:6014, ``num_reqs = min(num_tokens, max_num_reqs)``).
    num_seqs: int
    #: Total scheduled tokens across those sequences — ``max_num_batched_tokens`` at the ceiling.
    num_tokens: int
    #: ``H`` — value heads per layer, after tensor-parallel sharding.
    num_v_heads: int
    #: ``K`` — key/query head dimension.
    head_k: int
    #: ``V`` — value head dimension.
    head_v: int
    #: ``BT``. A parameter, not a constant: NT scales as 1/BT and ``h`` scales with NT.
    chunk_size: int = FLA_CHUNK_SIZE
    #: Bytes per element of ``k``/``u``, and so of ``h`` and ``v_new``. 2 for bf16/fp16.
    elem_bytes: int = 2
    #: Whether the caller asks for the carried state back. True on vLLM's prefill path —
    #: kimi_gdn_linear_attn.py:589 passes ``output_final_state=True`` — which is why the fp32
    #: ``final_state`` at chunk_delta_h.py:354 is part of the live set and not an optional extra.
    output_final_state: bool = True
    #: ``save_new_value`` at chunk_delta_h.py:329/357. True on the vLLM path: ``v_new`` feeds the
    #: output kernel at chunk.py:641.
    save_new_value: bool = True

    def __post_init__(self) -> None:
        for name in ("num_seqs", "num_tokens", "num_v_heads", "head_k", "head_v", "chunk_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.elem_bytes not in (1, 2, 4):
            raise ValueError(
                f"elem_bytes must be 1, 2 or 4 (fp8 / bf16 / fp32), got {self.elem_bytes}"
            )
        if self.num_tokens < self.num_seqs:
            # Every scheduled sequence carries at least one token, so a batch with fewer tokens
            # than sequences cannot exist. Catching it here matters because the chunk-count term
            # of the model is bounded using exactly this fact.
            raise ValueError(
                f"num_tokens ({self.num_tokens}) < num_seqs ({self.num_seqs}): every scheduled "
                "sequence contributes at least one token, so this batch cannot be built"
            )
        if self.head_k > MAX_HEAD_K:
            raise ValueError(
                f"head_k ({self.head_k}) exceeds the kernel's ceiling of {MAX_HEAD_K} — "
                "chunk_delta_h.py:350 asserts K <= 256, so this config cannot run at all"
            )


@dataclass(frozen=True)
class KdaScanBudget:
    """What the model predicted, what was left for it, and whether that is a refusal.

    Returned by :func:`plan_kda_chunk_scan` whether or not it fits, so a caller that wants to log
    the number without refusing (a warning-only rollout, say) does not have to catch an exception
    to read it.
    """

    shape: KdaScanShape
    #: The model's answer for this shape, in bytes.
    predicted_bytes: int
    #: Bytes actually available to the scan: free device memory after weights, KV cache and
    #: activations, as the caller measured it.
    available_bytes: int
    #: The margin in force when this budget was computed. ``None`` means an exact comparison.
    margin: float | None

    @property
    def budget_bytes(self) -> int:
        """Available bytes after the margin is withheld. ``None`` margin withholds nothing."""
        if self.margin is None:
            return self.available_bytes
        return int(self.available_bytes * (1.0 - self.margin))

    @property
    def fits(self) -> bool:
        return self.predicted_bytes <= self.budget_bytes

    @property
    def headroom_bytes(self) -> int:
        """Signed: positive is slack, negative is the shortfall the refusal message quotes."""
        return self.budget_bytes - self.predicted_bytes

    def report(self) -> str:
        """One line for a log, in GiB, with the shape that produced it.

        Reported even when it fits: the whole reason #54775 was hard to see is that nothing ever
        printed this number, so an operator had no way to know the buffer existed until it did not.
        """
        gib = 1024**3
        s = self.shape
        margin = "exact" if self.margin is None else f"margin {self.margin:.1%}"
        return (
            f"KDA chunked scan: predicted peak {self.predicted_bytes / gib:.3f} GiB vs "
            f"{self.budget_bytes / gib:.3f} GiB available ({margin}) at "
            f"num_seqs={s.num_seqs} num_tokens={s.num_tokens} H={s.num_v_heads} "
            f"K={s.head_k} V={s.head_v} BT={s.chunk_size} elem={s.elem_bytes}B"
        )


# =============================================================================================
# The hole — the model itself
# =============================================================================================


def kda_chunk_scan_peak_bytes(shape: KdaScanShape) -> int:
    # HUY: the KDA chunked-scan memory model — peak bytes(B, T, H, K, V, chunk) at one layer's high-water mark — spec: experiments/K3/PR-2/spec.md — fill before PR-2
    #
    # One line, deliberately: infra/holes.py matches `HUY:` and `spec:` within a single line, so a
    # marker wrapped for width drops out of `make holes` and the morning inventory loses this rung.
    #
    # What the model has to account for, in the order the bytes appear (citations in the module
    # docstring, all on oss/vllm @ fix/kda-chunk-buffers-memory-profiling-54775):
    #   h            (B, NT, H, V, K) x elem_bytes          chunk_delta_h.py:352
    #   v_new        (B, T,  H, V)    x elem_bytes          chunk_delta_h.py:357
    #   final_state  (N, H, V, K)     x 4                   chunk_delta_h.py:354
    #   still live across the call: w, u, kg (chunk.py:209-211), Aqk (chunk_intra.py:488)
    #
    # And the one genuinely modelled quantity: NT. In varlen mode it is sum_i ceil(T_i / BT)
    # (chunk_delta_h.py:347), not ceil(T / BT) — so it is a function of how the scheduler split
    # the token budget across `num_seqs` sequences, and the check needs the worst legal split,
    # not the typical one. `num_tokens >= num_seqs` is enforced in KdaScanShape.__post_init__, so
    # the bound may assume every sequence carries at least one token.
    #
    # Deriving that bound, deciding whether the cross-call live set belongs in the peak, and
    # deciding what the caching allocator actually returns between layers IS the rung. It is the
    # "serving memory model" line in plan/SIXTY_DAYS_SIX_LADDERS.md §02, and it is the number the
    # PR body has to defend to a vLLM maintainer.
    #
    # Falsify it before shipping it: experiments/K3/PR-2/measure_peak_alloc.py prints predicted
    # against torch.cuda.max_memory_allocated over a T grid. `pred err %` from that table is what
    # sets SAFETY_MARGIN above.
    # The sentinel tests/conftest.py greps for is the contiguous `NotImplementedError("HUY:` —
    # so the message opens on this line and continues by implicit concatenation. Breaking after
    # the paren would hide the hole from the guard and from `make holes` alike.
    raise NotImplementedError(
        "HUY: KDA chunked-scan peak-bytes model is unwritten. Read the "
        "allocation sites in this module's docstring, derive bytes(B, T, H, K, V, chunk) "
        "including the NT bound over sequence splits, then check it against "
        "experiments/K3/PR-2/measure_peak_alloc.py before arming SAFETY_MARGIN. "
        "Spec: experiments/K3/PR-2/spec.md"
    )


# =============================================================================================
# The plumbing — everything that spends the model's answer
# =============================================================================================


def scan_shape_for_profile_run(
    *,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    num_v_heads: int,
    head_k: int,
    head_v: int,
    chunk_size: int = FLA_CHUNK_SIZE,
    elem_bytes: int = 2,
) -> KdaScanShape:
    """The worst batch the engine can schedule, expressed as a :class:`KdaScanShape`.

    ``num_seqs = min(max_num_seqs, max_num_batched_tokens)`` is not a guess: it is the arithmetic
    vLLM's own dummy-batch builder does at gpu_model_runner.py:6014, and it is the ceiling the
    scheduler enforces at run time too. Taking the engine's numbers rather than a model config is
    what makes this a *startup* check — everything it needs is known before a single weight loads.
    """
    return KdaScanShape(
        num_seqs=min(max_num_seqs, max_num_batched_tokens),
        num_tokens=max_num_batched_tokens,
        num_v_heads=num_v_heads,
        head_k=head_k,
        head_v=head_v,
        chunk_size=chunk_size,
        elem_bytes=elem_bytes,
    )


def plan_kda_chunk_scan(
    shape: KdaScanShape,
    available_bytes: int,
    *,
    model: Callable[[KdaScanShape], int] = kda_chunk_scan_peak_bytes,
    margin: float | None = SAFETY_MARGIN,
) -> KdaScanBudget:
    """Run the model against the headroom and return the verdict without acting on it.

    ``model`` is injected rather than called by name so the surrounding arithmetic can be tested
    on a laptop against a stub while the real one is still a hole — and so a future direction-2
    fix (measure the buffers inside the profiler) can substitute a measured curve for the analytic
    one without touching a line of this.
    """
    if not isinstance(available_bytes, int) or isinstance(available_bytes, bool):
        raise TypeError(f"available_bytes must be an int, got {available_bytes!r}")
    if available_bytes < 0:
        raise ValueError(f"available_bytes must be non-negative, got {available_bytes}")
    if margin is not None and not 0.0 <= margin < 1.0:
        raise ValueError(f"margin must lie in [0, 1), got {margin}")

    predicted = model(shape)
    if not isinstance(predicted, int) or isinstance(predicted, bool) or predicted < 0:
        raise TypeError(
            f"the memory model must return a non-negative int of bytes, got {predicted!r}"
        )
    return KdaScanBudget(
        shape=shape,
        predicted_bytes=predicted,
        available_bytes=available_bytes,
        margin=margin,
    )


def check_kda_chunk_scan_headroom(
    shape: KdaScanShape,
    available_bytes: int,
    *,
    model: Callable[[KdaScanShape], int] = kda_chunk_scan_peak_bytes,
    margin: float | None = SAFETY_MARGIN,
) -> KdaScanBudget:
    """Refuse, at startup, a config whose KDA chunked scan will not fit. This is direction 3.

    Returns the budget when it fits so the caller can log :meth:`KdaScanBudget.report`; raises
    :class:`KdaScanWouldNotFit` when it does not. The message names the knobs because the failure
    it replaces — a CUDA OOM inside ``chunk_gated_delta_rule_fwd_h`` — names none of them, and an
    operator who cannot see which number to turn will turn ``--gpu-memory-utilization`` down until
    the KV cache is useless, which fixes the symptom by destroying the throughput.
    """
    budget = plan_kda_chunk_scan(shape, available_bytes, model=model, margin=margin)
    if budget.fits:
        return budget

    gib = 1024**3
    s = shape
    raise KdaScanWouldNotFit(
        f"{budget.report()}\n"
        f"Short by {-budget.headroom_bytes / gib:.3f} GiB. The KDA prefill materialises its whole "
        f"chunk-recurrent state before the output kernel reads it, and two of those buffers are "
        f"linear in the batch's token count, so this is a hard requirement of the batch size — "
        f"not a transient the allocator can retry around.\n"
        f"Turn one of:\n"
        f"  --max-num-batched-tokens (now {s.num_tokens}) — both linear terms scale with it;\n"
        f"  --max-num-seqs (now {s.num_seqs}) — more sequences means more partial chunks;\n"
        f"  --gpu-memory-utilization — buys headroom by shrinking the KV cache;\n"
        f"  the KDA chunk size (now {s.chunk_size}) — the state buffer scales as 1/BT.\n"
        f"Model and derivation: scratch_llm/src/scratch_llm/kernels/linear_attn/kda_memory.py; "
        f"upstream issue: vllm-project/vllm#54775."
    )
