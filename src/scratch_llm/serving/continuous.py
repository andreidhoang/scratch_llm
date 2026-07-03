"""A1 Rung 3b — iteration-level (continuous, Orca-style) batching on the static slot buffer.

Static request-level batching runs a wave until its LONGEST row finishes: short rows idle their
slots, so aggregate throughput is capped at ``utilization = mean_len / max_len`` of capacity.
Iteration-level scheduling re-decides *every step* — evict finished rows, admit queued requests
into the freed slots — so the batch stays saturated:
``speedup ≈ max_len / (mean_len + admit_tax)`` on a saturated queue (the corrected analytic model,
``performance/notes/A1_R3_continuous_batching.md`` §2).

Both policies — ``"continuous"`` and the ``"wave"`` control — run through this ONE engine (same
cache, same forward, same metrics clocking); the measured Δ is pure scheduling, nothing else.
Scope (R3b): greedy-only, budget-only (``max_new_tokens``; no stop tokens) — the oracle contract is
against ``sampling.generate(temperature=0)``.

Invariant (tests/test_continuous.py): each request's output is token-identical to its standalone
single-stream greedy decode — scheduling may change *when* a token is produced, never *which*.

Interview question this answers: why does continuous batching beat static batching, by exactly how
much, and what does it cost (ITL spikes at admission, prefill-in-the-decode-stream)?
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from scratch_llm.model import BatchedKVCache, PrefillView, TransformerLM
from scratch_llm.serving.metrics import RequestRecord


@dataclass(frozen=True)
class Request:
    """One decode job: greedy-continue ``prompt_ids`` for exactly ``max_new_tokens`` tokens."""

    request_id: int
    prompt_ids: tuple[int, ...]
    max_new_tokens: int


@dataclass(frozen=True)
class Completed:
    """A finished request: its tokens plus the timing trace for the metrics harness.

    ``admitted_at_step`` is the number of lockstep decode steps that ran before this request's
    prefill — the queue wait measured in *steps*, a hardware-independent scheduling fact (wall-time
    TTFT is the metrics harness's job)."""

    request: Request
    token_ids: tuple[int, ...]
    record: RequestRecord
    admitted_at_step: int


@dataclass(frozen=True)
class ServeResult:
    """One serve run. ``utilization[i]`` is the active-slot fraction at decode step ``i`` — the
    quantity that *explains* any continuous-vs-wave throughput gap (and the kill-criterion
    diagnostic: low utilization = the scheduler isn't refilling; high utilization but low speedup =
    admission-prefill stalls). ``prefill_s``/``decode_s`` split the wall time so the admit tax of
    the analytic model is measured, not assumed."""

    completed: tuple[Completed, ...]  # sorted by request_id
    utilization: tuple[float, ...]
    n_decode_steps: int
    n_prefill_forwards: int
    prefill_s: float
    decode_s: float

    @property
    def mean_utilization(self) -> float:
        return sum(self.utilization) / len(self.utilization) if self.utilization else 0.0


@torch.no_grad()
def serve(
    model: TransformerLM,
    requests: Sequence[Request],
    n_slots: int,
    device: str = "cpu",
    *,
    policy: Literal["continuous", "wave"],
    clock: Callable[[], float] = time.perf_counter,
    prefill_model: TransformerLM | None = None,
) -> ServeResult:
    """Run ``requests`` to completion over ``n_slots`` static KV slots under one of two policies.

    - ``"continuous"`` (Orca iteration-level): admit into freed slots **every step**.
    - ``"wave"`` (static request-level control): admit only when **all** slots are free.

    Loop: evict budget-hit rows → admit (ONE right-padded prefill forward for all admitted rows —
    per-request pause-prefill would stall ≈ ``n_slots/mean_len`` of the run) → one lockstep decode
    step over all ``n_slots`` rows (static shapes: inactive rows compute masked, discarded work).
    All requests are treated as arriving at t₀, so TTFT includes queue wait — the number that shows
    what wave scheduling does to a queued request.

    ``prefill_model`` (default: ``model``) runs the admission prefills. Pass the UNCOMPILED module
    here when ``model`` is ``torch.compile``d: at steady state admissions arrive in dribbles
    (n_admit = 1, 2, 3, …) and every new admission width is a new shape — compiling the prefill
    path floods the dynamo cache with per-width graphs (measured 2026-07-03: 47 graphs, ~35 s of
    compile inside the run, plus a per-call linear guard scan taxing every step). Prefill is ~1% of
    wall; the decode loop is the hot path that should own the compile budget.
    """
    if n_slots < 1:
        raise ValueError("n_slots must be ≥ 1")
    if len(requests) == 0:
        raise ValueError("requests must be non-empty")
    ids = [r.request_id for r in requests]
    if len(set(ids)) != len(ids):
        raise ValueError("request_id values must be unique")
    max_ctx = model.cfg.context_length
    for r in requests:
        if len(r.prompt_ids) == 0:
            raise ValueError(f"request {r.request_id}: prompt must be non-empty")
        if r.max_new_tokens < 1:
            raise ValueError(f"request {r.request_id}: max_new_tokens must be ≥ 1")
        if len(r.prompt_ids) + r.max_new_tokens > max_ctx:
            raise ValueError(
                f"request {r.request_id}: prompt {len(r.prompt_ids)} + max_new "
                f"{r.max_new_tokens} exceeds context_length {max_ctx} (a slot cannot overflow)"
            )

    model.eval()
    dev = torch.device(device)
    dtype = next(model.parameters()).dtype
    cache = BatchedKVCache(
        n_layers=model.cfg.n_layers,
        n_slots=n_slots,
        n_kv_heads=model.cfg.kv_heads,
        max_ctx=max_ctx,
        head_dim=model.cfg.head_dim,
        device=dev,
        dtype=dtype,
    )

    def now() -> float:
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return clock()

    queue: deque[Request] = deque(requests)
    slot_req: list[Request | None] = [None] * n_slots
    slot_tokens: list[list[int]] = [[] for _ in range(n_slots)]
    slot_times: list[list[float]] = [[] for _ in range(n_slots)]
    slot_admit_step: list[int] = [0] * n_slots
    last_ids = torch.zeros(n_slots, dtype=torch.long, device=dev)
    completed: list[Completed] = []
    utilization: list[float] = []
    n_decode_steps = 0
    n_prefill_forwards = 0
    prefill_s = 0.0
    decode_s = 0.0

    t0 = now()  # every request "arrives" here: TTFT includes time spent queued

    def finish(slot: int) -> None:
        req = slot_req[slot]
        assert req is not None
        completed.append(
            Completed(
                request=req,
                token_ids=tuple(slot_tokens[slot]),
                record=RequestRecord(
                    prompt_len=len(req.prompt_ids),
                    start_s=t0,
                    token_times_s=tuple(slot_times[slot]),
                ),
                admitted_at_step=slot_admit_step[slot],
            )
        )
        slot_req[slot] = None
        slot_tokens[slot] = []
        slot_times[slot] = []
        cache.free_slot(slot)

    while queue or any(r is not None for r in slot_req):
        # 1) evict rows that hit their budget — frees slots for THIS step's admission
        for b, req in enumerate(slot_req):
            if req is not None and len(slot_tokens[b]) >= req.max_new_tokens:
                finish(b)

        # 2) admit — continuous: whenever a slot is free; wave: only into an all-free batch
        free = [b for b, r in enumerate(slot_req) if r is None]
        if queue and free and (policy == "continuous" or len(free) == n_slots):
            admits = [queue.popleft() for _ in range(min(len(free), len(queue)))]
            slots = free[: len(admits)]
            lens = [len(r.prompt_ids) for r in admits]
            t_pre = now()
            first = _prefill(prefill_model if prefill_model is not None else model,
                             cache, admits, slots, lens, dev)
            cache.mirror_admit(slots, lens)  # python half of the admission (outside the graph)
            t_post = now()
            prefill_s += t_post - t_pre
            n_prefill_forwards += 1
            first_list: list[int] = first.tolist()
            for i, (b, req) in enumerate(zip(slots, admits, strict=True)):
                slot_req[b] = req
                slot_tokens[b] = [first_list[i]]  # the prefill emits token 1 (that's TTFT)
                slot_times[b] = [t_post]
                slot_admit_step[b] = n_decode_steps  # queue wait, in decode steps
            last_ids[torch.tensor(slots, dtype=torch.long, device=dev)] = first
            # a request admitted at its full budget (max_new_tokens == 1) is already complete —
            # finish it before the decode step or it would over-generate
            for b in slots:
                req = slot_req[b]
                if req is not None and len(slot_tokens[b]) >= req.max_new_tokens:
                    finish(b)

        # 3) one lockstep decode step over ALL slots (static shapes; inactive rows are masked)
        if any(r is not None for r in slot_req):
            utilization.append(sum(r is not None for r in slot_req) / n_slots)
            t_step = now()
            logits = model(last_ids.unsqueeze(1), cache)
            assert isinstance(logits, Tensor)
            next_ids = logits[:, -1].argmax(dim=-1)
            last_ids = next_ids
            cache.mirror_advance()  # python half of advance() (outside the graph)
            t_done = now()
            decode_s += t_done - t_step
            n_decode_steps += 1
            vals: list[int] = next_ids.tolist()
            for b, req in enumerate(slot_req):
                if req is not None:
                    slot_tokens[b].append(vals[b])
                    slot_times[b].append(t_done)

    return ServeResult(
        completed=tuple(sorted(completed, key=lambda c: c.request.request_id)),
        utilization=tuple(utilization),
        n_decode_steps=n_decode_steps,
        n_prefill_forwards=n_prefill_forwards,
        prefill_s=prefill_s,
        decode_s=decode_s,
    )


def serve_continuous(
    model: TransformerLM,
    requests: Sequence[Request],
    n_slots: int,
    device: str = "cpu",
    *,
    clock: Callable[[], float] = time.perf_counter,
    prefill_model: TransformerLM | None = None,
) -> ServeResult:
    """Iteration-level (Orca) scheduling: freed slots are refilled every decode step."""
    return serve(
        model, requests, n_slots, device,
        policy="continuous", clock=clock, prefill_model=prefill_model,
    )


def serve_static_wave(
    model: TransformerLM,
    requests: Sequence[Request],
    n_slots: int,
    device: str = "cpu",
    *,
    clock: Callable[[], float] = time.perf_counter,
    prefill_model: TransformerLM | None = None,
) -> ServeResult:
    """Request-level (static-wave) control: a wave runs until its longest row finishes."""
    return serve(
        model, requests, n_slots, device,
        policy="wave", clock=clock, prefill_model=prefill_model,
    )


@torch.no_grad()
def _prefill(
    model: TransformerLM,
    cache: BatchedKVCache,
    admits: Sequence[Request],
    slots: Sequence[int],
    lens: Sequence[int],
    dev: torch.device,
) -> Tensor:
    """One right-padded prefill forward for all admitted rows; returns each row's first greedy
    token ``(n,)``. Fresh slots start at offset 0, so this is the ordinary uniform-causal forward
    routed through :class:`PrefillView`; each row's first token comes from its LAST VALID position
    (right-padding puts garbage logits after it). The caller mirrors the admission
    (``cache.mirror_admit``) after this returns."""
    x = torch.zeros((len(admits), max(lens)), dtype=torch.long, device=dev)
    for i, r in enumerate(admits):
        x[i, : lens[i]] = torch.tensor(r.prompt_ids, dtype=torch.long, device=dev)
    view = PrefillView(cache, list(slots), lens)
    logits = model(x, view)
    assert isinstance(logits, Tensor)
    rows = torch.arange(len(admits), device=dev)
    last = logits[rows, torch.tensor(lens, dtype=torch.long, device=dev) - 1]
    return last.argmax(dim=-1)
