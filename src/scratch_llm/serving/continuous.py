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

from scratch_llm.model import (
    BatchedKVCache,
    ChunkPrefillView,
    PagedKVCache,
    PrefillView,
    SlotKVCache,
    TransformerLM,
)
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
    # paged storage only (P4.1.1 accounting; None for the dense slab):
    paged_frag_mean: float | None = None  # mean internal fragmentation across decode steps
    paged_alloc_peak_tokens: int | None = None  # peak allocated block-tokens

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
    cache_kind: Literal["dense", "paged"] = "dense",
    paged_n_blocks: int | None = None,
    paged_use_kernel: bool = False,
    prefill_chunk_size: int | None = None,
    arrivals_s: Sequence[float] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ServeResult:
    """Run ``requests`` to completion over ``n_slots`` static KV slots under one of two policies.

    - ``"continuous"`` (Orca iteration-level): admit into freed slots **every step**.
    - ``"wave"`` (static request-level control): admit only when **all** slots are free.

    Loop: evict budget-hit rows → admit (ONE right-padded prefill forward for all admitted rows —
    per-request pause-prefill would stall ≈ ``n_slots/mean_len`` of the run) → one lockstep decode
    step over all ``n_slots`` rows (static shapes: inactive rows compute masked, discarded work).
    All requests are treated as arriving at t₀, so TTFT includes queue wait — the number that shows
    what wave scheduling does to a queued request. ``arrivals_s`` (S1 S-R1) replaces that closed
    batch with an **open loop**: offsets in seconds from t₀ at which each request is *submitted*.
    This is the client boundary, not a scheduling policy — an un-arrived request is simply not in
    the queue yet, and FCFS admission below is untouched — and it makes ``RequestRecord.start_s``
    the request's own arrival, so TTFT is queue-inclusive per request rather than per run.
    ``sleep`` is the idle wait used when nothing is admissible (injectable for a fake clock).

    ``prefill_model`` (default: ``model``) runs the admission prefills. Pass the UNCOMPILED module
    here when ``model`` is ``torch.compile``d: at steady state admissions arrive in dribbles
    (n_admit = 1, 2, 3, …) and every new admission width is a new shape — compiling the prefill
    path floods the dynamo cache with per-width graphs (measured 2026-07-03: 47 graphs, ~35 s of
    compile inside the run, plus a per-call linear guard scan taxing every step). Prefill is ~1% of
    wall; the decode loop is the hot path that should own the compile budget.

    ``cache_kind="paged"`` swaps the dense slab for :class:`PagedKVCache` (A1 R4.1): same engine,
    same policies — the measured Δ isolates storage. Admission then runs a **committed-blocks
    guard**: a request is admitted only if the pool can hold *every* active request run to its
    full budget (Σ ⌈(prompt+max_new)/16⌉ ≤ usable blocks) — with budget-only requests this makes
    pool exhaustion impossible without preemption machinery. FCFS is preserved: a blocked head
    stalls admission (no skip-ahead). ``paged_n_blocks`` defaults to the trivially safe
    ``n_slots × max_request_blocks + 1``.

    ``prefill_chunk_size`` (A1 R4.2): when set, a prompt of length ``L`` is prefilled in
    ``⌈L/C⌉`` chunks of ≤ ``C`` tokens, **one chunk per scheduler iteration interleaved with a
    decode step**, so a long prefill no longer head-of-line-blocks the decode stream (the measured
    R3b/R4.1 ITL p99 admission spike). A prefilling request holds a reserved-but-inactive slot
    (masked out of the concurrent decode batch) until its final chunk emits its first token
    (= TTFT), then it joins the decode batch. Token-exact to the one-shot path (``ChunkPrefillView``
    writes bit-identical KV — RoPE rotates each token at its absolute position regardless of chunk
    boundaries). Requires ``policy="continuous"`` and ``cache_kind="dense"`` (paged+chunk deferred);
    ``None`` (default) = one-shot admission, unchanged.
    """
    if n_slots < 1:
        raise ValueError("n_slots must be ≥ 1")
    if len(requests) == 0:
        raise ValueError("requests must be non-empty")
    if prefill_chunk_size is not None:
        if prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be ≥ 1")
        if policy != "continuous":
            raise ValueError("chunked prefill (prefill_chunk_size) requires policy='continuous'")
        if cache_kind != "dense":
            raise ValueError(
                "chunked prefill is implemented for cache_kind='dense' (paged+chunk deferred, R4.2)"
            )
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

    def _blocks_needed(r: Request) -> int:
        total = len(r.prompt_ids) + r.max_new_tokens
        return (total + PagedKVCache.BLOCK - 1) // PagedKVCache.BLOCK

    cache: SlotKVCache
    if cache_kind == "paged":
        n_blocks = paged_n_blocks or n_slots * max(_blocks_needed(r) for r in requests) + 1
        cache = PagedKVCache(
            n_layers=model.cfg.n_layers,
            n_slots=n_slots,
            n_kv_heads=model.cfg.kv_heads,
            max_ctx=max_ctx,
            head_dim=model.cfg.head_dim,
            n_blocks=n_blocks,
            device=dev,
            dtype=dtype,
        )
        cache.use_kernel = paged_use_kernel  # R4.1b: fused Triton decode instead of gather+SDPA
    else:
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

    arrival_of: dict[int, float] = dict.fromkeys(ids, 0.0)
    if arrivals_s is not None:
        if len(arrivals_s) != len(requests):
            raise ValueError("arrivals_s must carry one offset per request")
        if any(a < 0 for a in arrivals_s):
            raise ValueError("arrivals_s offsets must be ≥ 0 (they are offsets from t₀)")
        arrival_of = {r.request_id: float(a) for r, a in zip(requests, arrivals_s, strict=True)}
    order = sorted(requests, key=lambda r: (arrival_of[r.request_id], r.request_id))
    pending: deque[Request] = deque(order if arrivals_s is not None else ())
    queue: deque[Request] = deque(() if arrivals_s is not None else requests)
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
    committed_blocks = 0  # paged admission guard: worst-case blocks promised to active slots
    frag_samples: list[float] = []
    alloc_peak = 0
    # R4.2 chunked prefill: slots mid-prefill, FIFO; entry = [slot, request, next_offset].
    prefilling: deque[list[object]] = deque()

    t0 = now()  # every request "arrives" here: TTFT includes time spent queued

    def finish(slot: int) -> None:
        nonlocal committed_blocks
        req = slot_req[slot]
        assert req is not None
        if isinstance(cache, PagedKVCache):
            committed_blocks -= _blocks_needed(req)
        completed.append(
            Completed(
                request=req,
                token_ids=tuple(slot_tokens[slot]),
                record=RequestRecord(
                    prompt_len=len(req.prompt_ids),
                    start_s=t0 + arrival_of[req.request_id],
                    token_times_s=tuple(slot_times[slot]),
                ),
                admitted_at_step=slot_admit_step[slot],
            )
        )
        slot_req[slot] = None
        slot_tokens[slot] = []
        slot_times[slot] = []
        cache.free_slot(slot)

    while pending or queue or any(r is not None for r in slot_req):
        # 0) open-loop arrivals (S-R1): submit whatever has arrived, then idle if nothing can run.
        # `clock()`, not `now()` — an arrival is a client-side fact and must not buy a device sync
        # per scheduler iteration.
        while pending and clock() - t0 >= arrival_of[pending[0].request_id]:
            queue.append(pending.popleft())
        if pending and not queue and all(r is None for r in slot_req):
            sleep(max(0.0, t0 + arrival_of[pending[0].request_id] - clock()))
            continue

        # 1) evict rows that hit their budget — frees slots for THIS step's admission
        for b, req in enumerate(slot_req):
            if req is not None and len(slot_tokens[b]) >= req.max_new_tokens:
                finish(b)

        # 2) admit — continuous: whenever a slot is free; wave: only into an all-free batch.
        # Paged: the committed-blocks guard admits a request only if the pool can hold every
        # active request run to its FULL budget (no preemption needed, FCFS preserved).
        if prefill_chunk_size is None:
            free = [b for b, r in enumerate(slot_req) if r is None]
            admits: list[Request] = []
            if queue and free and (policy == "continuous" or len(free) == n_slots):
                for _ in range(min(len(free), len(queue))):
                    if isinstance(cache, PagedKVCache):
                        need = _blocks_needed(queue[0])
                        if committed_blocks + need > cache.n_blocks - 1:  # block 0 is trash
                            break  # head blocked ⇒ admission stalls (no skip-ahead)
                        committed_blocks += need
                    admits.append(queue.popleft())
            if admits:
                slots = free[: len(admits)]
                lens = [len(r.prompt_ids) for r in admits]
                t_pre = now()
                first = _prefill(
                    prefill_model if prefill_model is not None else model,
                    cache,
                    admits,
                    slots,
                    lens,
                    dev,
                )
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
                # a request admitted at full budget (max_new_tokens == 1) is already complete —
                # finish it before the decode step or it would over-generate
                for b in slots:
                    req = slot_req[b]
                    if req is not None and len(slot_tokens[b]) >= req.max_new_tokens:
                        finish(b)
        else:
            # R4.2 chunked prefill: reserve free slots into the prefilling FIFO, then advance ONE
            # C-token chunk of the oldest prefilling slot (the interleave that bounds per-gap
            # prefill work). The chunk's KV is bit-identical to a one-shot prefill.
            free = [b for b, r in enumerate(slot_req) if r is None]
            for b in free:
                if not queue:
                    break
                req = queue.popleft()
                slot_req[b] = req  # reserved-but-inactive: masked out of the decode batch
                prefilling.append([b, req, 0])
            if prefilling:
                entry = prefilling[0]
                slot = int(entry[0])  # type: ignore[arg-type]
                creq = entry[1]
                assert isinstance(creq, Request)
                offset = int(entry[2])  # type: ignore[arg-type]
                length = len(creq.prompt_ids)
                s = min(prefill_chunk_size, length - offset)
                chunk_ids = creq.prompt_ids[offset : offset + s]
                t_pre = now()
                chunk_logits = _prefill_chunk(
                    prefill_model if prefill_model is not None else model,
                    cache,
                    slot,
                    offset,
                    chunk_ids,
                    dev,
                )
                t_post = now()
                prefill_s += t_post - t_pre
                n_prefill_forwards += 1
                new_offset = offset + s
                if new_offset >= length:  # final chunk → emit first token, activate the slot
                    first_id = int(chunk_logits[s - 1].argmax())  # token at absolute position L
                    cache.active[slot] = True  # device half of the admission
                    cache.mirror_admit([slot], [length])  # python half
                    slot_tokens[slot] = [first_id]
                    slot_times[slot] = [t_post]
                    slot_admit_step[slot] = n_decode_steps
                    last_ids[slot] = first_id
                    prefilling.popleft()
                    if len(slot_tokens[slot]) >= creq.max_new_tokens:  # budget==1 already done
                        finish(slot)
                else:
                    entry[2] = new_offset

        # 3) one lockstep decode step over active slots (static shapes; inactive rows are masked).
        # Guard/util on py_active (decoding rows) — identical to slot_req for the one-shot path,
        # and correct for chunked prefill where reserved-but-prefilling slots are not yet active.
        if any(cache.py_active):
            utilization.append(sum(cache.py_active) / n_slots)
            cache.pre_decode_reserve()  # paged: boundary rows get a private block (outside graph)
            if isinstance(cache, PagedKVCache):
                alloc_tok, _, frag = cache.waste_stats()
                frag_samples.append(frag)
                alloc_peak = max(alloc_peak, alloc_tok)
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
            # Record the token only for slots that actually DECODED this step — the active rows.
            # (One-shot path: occupied == active, so this equals the old `slot_req is not None`.
            # Chunked prefill: a reserved-but-prefilling slot is occupied yet inactive and must NOT
            # accrue decode tokens, or it would hit its budget and be evicted mid-prefill.)
            for b in range(n_slots):
                if cache.py_active[b]:
                    slot_tokens[b].append(vals[b])
                    slot_times[b].append(t_done)

    return ServeResult(
        completed=tuple(sorted(completed, key=lambda c: c.request.request_id)),
        utilization=tuple(utilization),
        n_decode_steps=n_decode_steps,
        n_prefill_forwards=n_prefill_forwards,
        prefill_s=prefill_s,
        decode_s=decode_s,
        paged_frag_mean=(sum(frag_samples) / len(frag_samples)) if frag_samples else None,
        paged_alloc_peak_tokens=alloc_peak if frag_samples else None,
    )


def serve_continuous(
    model: TransformerLM,
    requests: Sequence[Request],
    n_slots: int,
    device: str = "cpu",
    *,
    clock: Callable[[], float] = time.perf_counter,
    prefill_model: TransformerLM | None = None,
    cache_kind: Literal["dense", "paged"] = "dense",
    paged_n_blocks: int | None = None,
    paged_use_kernel: bool = False,
    prefill_chunk_size: int | None = None,
    arrivals_s: Sequence[float] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ServeResult:
    """Iteration-level (Orca) scheduling: freed slots are refilled every decode step.

    ``prefill_chunk_size`` (A1 R4.2) interleaves ⌈L/C⌉-token prefill chunks with decode steps so a
    long prompt does not head-of-line-block the decode stream; ``None`` = one-shot admission.
    ``arrivals_s`` (S1 S-R1) turns the closed request list into an open-loop arrival stream."""
    return serve(
        model,
        requests,
        n_slots,
        device,
        policy="continuous",
        clock=clock,
        prefill_model=prefill_model,
        cache_kind=cache_kind,
        paged_n_blocks=paged_n_blocks,
        paged_use_kernel=paged_use_kernel,
        prefill_chunk_size=prefill_chunk_size,
        arrivals_s=arrivals_s,
        sleep=sleep,
    )


def serve_static_wave(
    model: TransformerLM,
    requests: Sequence[Request],
    n_slots: int,
    device: str = "cpu",
    *,
    clock: Callable[[], float] = time.perf_counter,
    prefill_model: TransformerLM | None = None,
    cache_kind: Literal["dense", "paged"] = "dense",
    paged_n_blocks: int | None = None,
    paged_use_kernel: bool = False,
) -> ServeResult:
    """Request-level (static-wave) control: a wave runs until its longest row finishes."""
    return serve(
        model,
        requests,
        n_slots,
        device,
        policy="wave",
        clock=clock,
        prefill_model=prefill_model,
        cache_kind=cache_kind,
        paged_n_blocks=paged_n_blocks,
        paged_use_kernel=paged_use_kernel,
    )


@torch.no_grad()
def _prefill(
    model: TransformerLM,
    cache: SlotKVCache,
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


@torch.no_grad()
def _prefill_chunk(
    model: TransformerLM,
    cache: SlotKVCache,
    slot: int,
    offset: int,
    chunk_ids: Sequence[int],
    dev: torch.device,
) -> Tensor:
    """One chunked-prefill forward — A1 R4.2. Writes ``s`` tokens of a single slot at absolute
    positions ``[offset, offset+s)`` into the slot's KV via :class:`ChunkPrefillView` and returns
    the chunk's logits ``(s, vocab)``; the caller reads the last row (position ``offset+s−1``) for
    the request's first generated token when this is the FINAL chunk. The KV written is bit-identical
    to a one-shot prefill of the same prompt (RoPE rotates each token at its absolute position), so
    the emitted token stream is token-exact to the non-chunked path."""
    x = torch.tensor([list(chunk_ids)], dtype=torch.long, device=dev)  # (1, s)
    view = ChunkPrefillView(cache, slot, offset)
    logits = model(x, view)
    assert isinstance(logits, Tensor)
    return logits[0]  # (s, vocab)
