"""S1/S-R1 — one load generator, two engines, one set of definitions.

The rung is not a new engine. It is the *instrument* that puts the engine already in this package
(:mod:`~scratch_llm.serving.continuous`) next to vLLM under a single load generator, so that a
throughput or latency gap can be attributed instead of asserted. Nothing here schedules, batches, or
decodes; it only decides *when* a request is submitted and *from which timestamps* TTFT and ITL are
computed. Both of those decisions are copied from ``vllm bench serve`` on purpose — the map at
``experiments/S1/S-R1/map.md`` cites the upstream line for every one of them, including the four
places where an exact copy is impossible and the residual difference is named.

Three parts:

1. **The stream** (:func:`build_stream`). A frozen list of ``(request_id, prompt_ids,
   max_new_tokens, arrival_s)``. Arrivals come from :func:`poisson_arrivals`, which reproduces
   ``vllm/benchmarks/serve.py::get_request`` exactly: Gamma(shape=burstiness,
   scale=1/(rate·burstiness)) inter-arrivals — Exponential, i.e. a Poisson process, at
   burstiness=1 — cumulatively summed and then **rescaled so the last arrival lands at n/rate**
   (serve.py:480-489). That rescale is not part of a Poisson process and is the single largest
   reason a hand-rolled "Poisson driver" does not reproduce vLLM's numbers.

2. **The metrics** (:class:`RequestOutcome`, :func:`summarize`). TTFT is measured from the moment
   the request is *submitted* — so it contains server-side queueing, exactly as vLLM's
   ``st = time.perf_counter()`` immediately before the HTTP POST does
   (``lib/endpoint_request_func.py:200``). ITL is the gap between *successive* output chunks and
   therefore never contains queueing or prefill (:243). Percentiles are ``numpy``-linear, not
   nearest-rank: ``vllm bench serve`` reports ``np.percentile`` (serve.py:747,759), while
   :func:`scratch_llm.serving.metrics.percentiles` is nearest-rank. Both are defensible; only one
   is comparable, so this module carries its own.

3. **The adapters** (:class:`ScratchEngineAdapter`, :class:`OpenAIServerAdapter`). Each takes the
   *same* :class:`RequestStream` object and is responsible only for realizing it in its engine's
   idiom. :meth:`RequestStream.digest` is the receipt: two arms that report different digests did
   not see the same load, and their ratio is fiction.

The model is Qwen3-8B's **geometry**, not its weights (:func:`qwen3_8b_shape`) — this repo has no
safetensors loader. With ``--load-format dummy`` on the vLLM side, ``ignore_eos`` on both, and a
fixed input/output length, the two arms perform identical FLOPs and move identical bytes per token;
only the token *values* differ, and no timing metric here reads a token value.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from scratch_llm.model import ModelConfig

__all__ = [
    "ArmReport",
    "ArmRun",
    "EngineAdapter",
    "OpenAIServerAdapter",
    "RequestOutcome",
    "RequestSpec",
    "RequestStream",
    "ScratchEngineAdapter",
    "build_stream",
    "percentile",
    "poisson_arrivals",
    "qwen3_8b_shape",
    "summarize",
]

#: ``vllm bench serve --dataset-name random`` defaults (datasets/datasets.py:576-577). The rung
#: uses them unchanged so the floor can be re-run with no flags beyond rate and seed.
DEFAULT_INPUT_LEN = 1024
DEFAULT_OUTPUT_LEN = 128
#: Qwen3-8B's tokenizer vocabulary — the prompt-id alphabet, and the LM head's output width.
QWEN3_VOCAB_SIZE = 151936


def qwen3_8b_shape(context_length: int = 2048) -> ModelConfig:
    """Qwen3-8B's architecture, with **random** weights.

    36 layers · d_model 4096 · 32 query heads over 8 KV heads (GQA 4:1) · head_dim 128 · SwiGLU
    d_ff 12288 · QK-norm · RoPE θ=1e6 · vocab 151936. Those numbers fix the FLOPs per token, the
    bytes per token of weight traffic, and the bytes per token of KV — which is everything a TTFT /
    ITL / throughput comparison reads. They do not fix the token values, and nothing in this module
    reads a token value.

    ``context_length`` sizes the dense KV slab (``n_layers × n_slots × context_length × kv_heads ×
    head_dim × 2 × dtype``), so it is a memory-model knob, not a fidelity knob; set it to just
    above ``input_len + output_len`` for the workload under test.
    """
    return ModelConfig(
        vocab_size=QWEN3_VOCAB_SIZE,
        d_model=4096,
        n_layers=36,
        n_heads=32,
        n_kv_heads=8,
        d_ff=12288,
        context_length=context_length,
        rope_theta=1_000_000.0,
        qk_norm=True,
    )


# --------------------------------------------------------------------------------------------
# 1. the stream
# --------------------------------------------------------------------------------------------


def poisson_arrivals(
    n: int,
    request_rate: float,
    seed: int,
    *,
    burstiness: float = 1.0,
    normalize: bool = True,
) -> tuple[float, ...]:
    """Arrival offsets (seconds from the stream epoch), reproducing ``vllm bench serve``.

    ``vllm/benchmarks/serve.py::get_request`` draws each inter-arrival from
    ``np.random.gamma(shape=burstiness, scale=1/(request_rate·burstiness))`` (serve.py:466-470).
    At ``burstiness == 1`` a Gamma(1, θ) *is* an Exponential(1/θ), so the arrivals are a homogeneous
    Poisson process of rate ``request_rate`` — that is the sense in which "Poisson at r rps" is
    true here, and :func:`inter_arrivals` is the thing to test it on.

    ``normalize`` reproduces serve.py:480-489: after the cumulative sum, every offset is multiplied
    by ``(n/request_rate) / offsets[-1]`` so the last arrival lands exactly at ``n/request_rate``.
    vLLM's stated reason is that the raw sum misses the target window by 1-2%, which moves the
    reported throughput between seeds. The consequence is that the *normalized* sequence is no
    longer a Poisson process — it is exponential increments conditioned on their sum — and its
    empirical rate is exactly ``request_rate`` by construction rather than in expectation. Keep it
    on for any comparison against vLLM; turn it off to test the underlying process.

    Uses the legacy ``RandomState`` for the same reason vLLM does (``np.random.seed`` at
    serve.py:2018 seeds exactly this global generator), so a given ``seed`` reproduces vLLM's own
    arrival sequence element-for-element **provided nothing else has drawn from the global
    ``np.random`` first**. Verify, do not assume: the bench prints the first three inter-arrivals of
    both arms.
    """
    if n < 1:
        raise ValueError("n must be ≥ 1")
    if not (request_rate > 0.0) or math.isinf(request_rate):
        raise ValueError("request_rate must be finite and > 0 (rate=inf is the closed-loop case)")
    if not burstiness > 0.0:
        raise ValueError(f"A positive burstiness factor is expected, but given {burstiness}.")
    deltas = inter_arrivals(n, request_rate, seed, burstiness=burstiness)
    offsets: list[float] = []
    running = 0.0
    for d in deltas:
        running += d
        offsets.append(running)
    if normalize and offsets[-1] != 0.0:
        factor = (n / request_rate) / offsets[-1]
        offsets = [o * factor for o in offsets]
    return tuple(offsets)


def inter_arrivals(
    n: int, request_rate: float, seed: int, *, burstiness: float = 1.0
) -> tuple[float, ...]:
    """The ``n`` raw inter-arrival gaps, before the cumulative sum and before vLLM's rescale.

    This is the object with a distribution: Gamma(burstiness, 1/(rate·burstiness)), which at
    ``burstiness=1`` is Exponential with mean ``1/rate`` and CV 1. The cumulative, rescaled
    sequence :func:`poisson_arrivals` returns is *not* — see its docstring.
    """
    rng = np.random.RandomState(seed)
    theta = 1.0 / (request_rate * burstiness)
    return tuple(float(rng.gamma(shape=burstiness, scale=theta)) for _ in range(n))


@dataclass(frozen=True)
class RequestSpec:
    """One request as the load generator sees it: what to send, and when to send it.

    ``arrival_s`` is an offset from the stream epoch, not a wall-clock time — a stream is a
    hardware-independent object that both arms replay against their own epoch.
    """

    request_id: int
    prompt_ids: tuple[int, ...]
    max_new_tokens: int
    arrival_s: float

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)


@dataclass(frozen=True)
class RequestStream:
    """The load, frozen. One object, handed to every adapter.

    :meth:`digest` is what makes "the two engines saw the same load" checkable rather than
    asserted: it hashes every field that can change an engine's work, and the bench prints it once
    per arm. Two arms with different digests are not comparable, whatever their numbers say.
    """

    specs: tuple[RequestSpec, ...]
    request_rate: float
    seed: int
    burstiness: float = 1.0
    normalized: bool = True

    def __len__(self) -> int:
        return len(self.specs)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(s.prompt_len for s in self.specs)

    @property
    def total_max_new_tokens(self) -> int:
        return sum(s.max_new_tokens for s in self.specs)

    def digest(self) -> str:
        """SHA-256 over the whole stream, 16 hex chars. Order-sensitive, content-sensitive,
        arrival-sensitive; quantizes arrivals to 1 µs so a float repr cannot fork the hash."""
        h = hashlib.sha256()
        h.update(
            f"{self.request_rate!r}|{self.seed}|{self.burstiness!r}|{self.normalized}".encode()
        )
        for s in self.specs:
            h.update(f"{s.request_id}|{s.max_new_tokens}|{round(s.arrival_s * 1e6)}|".encode())
            h.update(np.asarray(s.prompt_ids, dtype=np.int64).tobytes())
        return h.hexdigest()[:16]


def build_stream(
    n_requests: int,
    request_rate: float,
    *,
    seed: int,
    input_len: int = DEFAULT_INPUT_LEN,
    output_len: int = DEFAULT_OUTPUT_LEN,
    vocab_size: int = QWEN3_VOCAB_SIZE,
    burstiness: float = 1.0,
    normalize: bool = True,
) -> RequestStream:
    """Build the one stream both arms replay.

    Prompt ids follow ``vllm bench serve``'s RandomDataset recipe —
    ``allowed_tokens[(offset + index + arange(input_len)) % len(allowed_tokens)]``
    (datasets/datasets.py:755-756) — drawn from an
    **isolated** ``default_rng(seed)`` exactly as upstream does (:584), so it cannot perturb the
    global generator the arrival draw reads. With the upstream default ``--random-range-ratio 0.0``
    every request is exactly ``input_len`` in and ``output_len`` out, which is what makes the two
    arms' work identical rather than merely similar.
    """
    if n_requests < 1:
        raise ValueError("n_requests must be ≥ 1")
    if input_len < 1 or output_len < 1:
        raise ValueError("input_len and output_len must be ≥ 1")
    arrivals = poisson_arrivals(
        n_requests, request_rate, seed, burstiness=burstiness, normalize=normalize
    )
    rng = np.random.default_rng(seed)
    offsets = rng.integers(0, vocab_size, size=n_requests)
    base = np.arange(input_len, dtype=np.int64)
    specs = tuple(
        RequestSpec(
            request_id=i,
            prompt_ids=tuple(int(t) for t in (int(offsets[i]) + i + base) % vocab_size),
            max_new_tokens=output_len,
            arrival_s=arrivals[i],
        )
        for i in range(n_requests)
    )
    return RequestStream(
        specs=specs,
        request_rate=request_rate,
        seed=seed,
        burstiness=burstiness,
        normalized=normalize,
    )


# --------------------------------------------------------------------------------------------
# 2. the metrics
# --------------------------------------------------------------------------------------------


def percentile(values: Sequence[float], p: float) -> float:
    """``np.percentile(values, p)`` with ``method="linear"``, in pure Python.

    This is the percentile ``vllm bench serve`` reports (serve.py:747 TTFT, :759 ITL), and it is
    **not** the nearest-rank percentile of :func:`scratch_llm.serving.metrics.percentiles`
    (``idx = round(p·(n−1))``). On n=10 at p=99 the two disagree by most of a sample gap. Carrying
    both, with the comparable one here, is cheaper than silently reporting a p99 that no floor
    shares.
    """
    if not values:
        raise ValueError("percentile of an empty sequence is undefined")
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"p must be in [0, 100], got {p}")
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


@dataclass(frozen=True)
class RequestOutcome:
    """What one request did, in the timestamps the metrics are defined on.

    ``sent_s`` is vLLM's ``st`` (``lib/endpoint_request_func.py:200``): the instant the client
    hands the request over, taken *after* the arrival wait. TTFT therefore contains everything the
    server does — queueing, scheduling, prefill — and nothing the client does waiting to send.

    ``chunk_times_s`` is one timestamp per streamed output chunk, ``[0]`` being the first token.
    vLLM's ITL is a gap between *chunks*, not between tokens (:243); with one token per chunk —
    which is what vLLM's streaming decode produces — the two coincide, and ``output_tokens``
    records the authoritative count so a bundled chunk cannot silently deflate TPOT.
    """

    request_id: int
    prompt_len: int
    sent_s: float
    chunk_times_s: tuple[float, ...]
    output_tokens: int
    success: bool = True
    error: str = ""

    @property
    def ttft_s(self) -> float:
        """``chunk_times_s[0] − sent_s`` — includes queueing (endpoint_request_func.py:238)."""
        return self.chunk_times_s[0] - self.sent_s

    @property
    def itls_s(self) -> tuple[float, ...]:
        """Successive chunk gaps *after* the first — never contains queueing or prefill (:243)."""
        return tuple(
            self.chunk_times_s[i] - self.chunk_times_s[i - 1]
            for i in range(1, len(self.chunk_times_s))
        )

    @property
    def latency_s(self) -> float:
        """``chunk_times_s[-1] − sent_s`` — vLLM's ``most_recent_timestamp − st`` (:260)."""
        return self.chunk_times_s[-1] - self.sent_s

    @property
    def tpot_s(self) -> float:
        """``(latency − ttft) / (output_tokens − 1)``, 0 for a one-token reply (serve.py:614-620).
        Divides by the *token* count, not the chunk count — that is upstream's choice, and it is
        why mean ITL and TPOT are not the same number when chunks bundle."""
        if self.output_tokens <= 1:
            return 0.0
        return (self.latency_s - self.ttft_s) / (self.output_tokens - 1)


@dataclass(frozen=True)
class ArmReport:
    """One engine's verdict on one stream. Field names mirror ``vllm bench serve``'s result JSON
    (serve.py:1281-1302) so the two can be diffed key by key."""

    arm: str
    stream_digest: str
    request_rate: float
    completed: int
    failed: int
    duration_s: float
    total_input_tokens: int
    total_output_tokens: int
    request_throughput: float
    output_throughput: float
    mean_concurrency: float
    ttft_ms: dict[str, float]
    itl_ms: dict[str, float]
    tpot_ms: dict[str, float]
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "stream_digest": self.stream_digest,
            "request_rate": self.request_rate,
            "completed": self.completed,
            "failed": self.failed,
            "duration": self.duration_s,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "request_throughput": self.request_throughput,
            "output_throughput": self.output_throughput,
            "mean_concurrency": self.mean_concurrency,
            "ttft_ms": self.ttft_ms,
            "itl_ms": self.itl_ms,
            "tpot_ms": self.tpot_ms,
            "extra": self.extra,
        }


_PCTS = (50.0, 99.0)


def _dist_ms(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p99": 0.0}
    out = {"mean": 1000.0 * sum(values) / len(values)}
    for p in _PCTS:
        out[f"p{p:g}"] = 1000.0 * percentile(values, p)
    return out


def summarize(
    arm: str,
    stream: RequestStream,
    outcomes: Sequence[RequestOutcome],
    *,
    epoch_s: float,
    end_s: float,
    extra: dict[str, float] | None = None,
) -> ArmReport:
    """Aggregate one arm's outcomes exactly as ``calculate_metrics`` does (serve.py:564-770).

    ``duration_s = end_s − epoch_s`` mirrors ``benchmark_duration`` (serve.py:1012, :1091): the
    clock starts *before* the first arrival, not at the first arrival, so the arrival window is
    inside the denominator for both arms. ``output_throughput = Σ output_tokens / duration_s``
    (:740). ``mean_concurrency`` is Little's law over the same window, ``Σ latency / duration_s``:
    at a fixed open-loop rate the batch size is an *outcome*, not a setting, and this is the number
    that says what "B=32" actually was.
    """
    ok = [o for o in outcomes if o.success and o.chunk_times_s]
    duration_s = end_s - epoch_s
    if duration_s <= 0.0:
        raise ValueError(f"duration must be > 0, got {duration_s}")
    total_out = sum(o.output_tokens for o in ok)
    total_in = sum(o.prompt_len for o in ok)
    itls: list[float] = []
    for o in ok:
        itls.extend(o.itls_s)
    return ArmReport(
        arm=arm,
        stream_digest=stream.digest(),
        request_rate=stream.request_rate,
        completed=len(ok),
        failed=len(outcomes) - len(ok),
        duration_s=duration_s,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        request_throughput=len(ok) / duration_s,
        output_throughput=total_out / duration_s,
        mean_concurrency=sum(o.latency_s for o in ok) / duration_s,
        ttft_ms=_dist_ms([o.ttft_s for o in ok]),
        itl_ms=_dist_ms(itls),
        tpot_ms=_dist_ms([o.tpot_s for o in ok if o.output_tokens > 1]),
        extra=dict(extra or {}),
    )


# --------------------------------------------------------------------------------------------
# 3. the adapters
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmRun:
    """Raw output of one arm: the outcomes plus the two clock readings ``summarize`` needs.

    ``epoch_s`` is the arm's zero for ``RequestSpec.arrival_s`` — vLLM's ``benchmark_start_time``
    (serve.py:1012). ``end_s`` is taken after the last request is collected (:1091)."""

    outcomes: tuple[RequestOutcome, ...]
    epoch_s: float
    end_s: float
    extra: dict[str, float] = field(default_factory=dict)


class EngineAdapter(Protocol):
    """The whole contract. An adapter converts a stream into outcomes and nothing else — it does
    not choose the load, the seed, the arrival process, or the metric definitions, so no arm can
    be accused of having been given an easier stream than the other."""

    name: str

    def run(self, stream: RequestStream) -> ArmRun: ...


class ScratchEngineAdapter:
    """Drives ``scratch_llm.serving.continuous.serve_continuous`` at the stream's arrival times.

    The engine's ``arrivals_s`` argument is the client boundary, not a scheduling policy: a request
    is not *submitted* until its offset elapses, and FCFS admission downstream is untouched. The
    engine stamps ``RequestRecord.start_s = t0 + arrival_s``, i.e. the **nominal** arrival, so TTFT
    also carries the scheduler-loop granularity between the nominal arrival and the engine noticing
    it. That is the conservative direction (it can only make our arm look worse) and it is a real
    difference from vLLM, whose ``st`` is the actual send; map.md states it.
    """

    def __init__(
        self,
        model: Any,
        *,
        n_slots: int,
        device: str = "cpu",
        cache_kind: str = "dense",
        prefill_chunk_size: int | None = None,
        prefill_model: Any = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        name: str = "scratch_llm",
    ) -> None:
        self.name = name
        self._model = model
        self._n_slots = n_slots
        self._device = device
        self._cache_kind = cache_kind
        self._prefill_chunk_size = prefill_chunk_size
        self._prefill_model = prefill_model
        self._clock = clock
        self._sleep = sleep

    def run(self, stream: RequestStream) -> ArmRun:
        from scratch_llm.serving.continuous import Request, serve_continuous

        requests = [
            Request(
                request_id=s.request_id,
                prompt_ids=s.prompt_ids,
                max_new_tokens=s.max_new_tokens,
            )
            for s in stream.specs
        ]
        arrivals = [s.arrival_s for s in stream.specs]
        epoch_s = self._clock()
        result = serve_continuous(
            self._model,
            requests,
            self._n_slots,
            self._device,
            clock=self._clock,
            prefill_model=self._prefill_model,
            cache_kind=self._cache_kind,  # type: ignore[arg-type]
            prefill_chunk_size=self._prefill_chunk_size,
            arrivals_s=arrivals,
            sleep=self._sleep,
        )
        end_s = self._clock()
        outcomes = tuple(
            RequestOutcome(
                request_id=c.request.request_id,
                prompt_len=c.record.prompt_len,
                sent_s=c.record.start_s,
                chunk_times_s=c.record.token_times_s,
                output_tokens=len(c.record.token_times_s),
            )
            for c in result.completed
        )
        return ArmRun(
            outcomes=outcomes,
            epoch_s=epoch_s,
            end_s=end_s,
            extra={
                "mean_slot_utilization": result.mean_utilization,
                "n_slots": float(self._n_slots),
                "n_decode_steps": float(result.n_decode_steps),
                "n_prefill_forwards": float(result.n_prefill_forwards),
                "prefill_s": result.prefill_s,
                "decode_s": result.decode_s,
            },
        )


class OpenAIServerAdapter:
    """Drives any OpenAI-compatible ``/v1/completions`` server — here, ``vllm serve``.

    Deliberately mirrors ``async_request_openai_completions``
    (``lib/endpoint_request_func.py:161-266``) rather than importing it: the payload fields, the
    ``st`` placement, the first-chunk branch and the ITL append are copied line for line, so that
    "vLLM under our generator" and "vLLM under ``vllm bench serve``" are the same measurement to
    within the differences map.md enumerates. Uses ``http.client`` + one thread per in-flight
    request because this repo pins no async HTTP client (pyproject.toml:16-21) and adding one to
    take a measurement is a dependency the measurement does not need; the arrival pacing loop is
    the structural twin of vLLM's ``async for … asyncio.create_task`` (serve.py:1026-1081).

    Known risk, and its detector: 100+ Python threads parsing SSE JSON contend on the GIL in a way
    vLLM's single-threaded asyncio client does not, which can inflate this arm's ITL p99. The
    ``vllm`` arm and the ``floor`` arm (``vllm bench serve`` itself) are run against the same server
    precisely so that disagreement between them exposes it — read that check before any ratio.

    Prompts go over the wire as **token ids**, not text. ``vllm bench serve`` sends decoded text and
    lets the server re-tokenize (datasets/datasets.py:569 decodes then re-encodes to force the
    count); sending ids removes the tokenizer from the loop and makes ``prompt_len`` exact instead
    of approximately right. It is a divergence from the floor and map.md says so.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        ignore_eos: bool = True,
        timeout_s: float = 600.0,
        name: str = "vllm",
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._ignore_eos = ignore_eos
        self._timeout_s = timeout_s

    def _one(self, spec: RequestSpec, out: list[RequestOutcome], lock: threading.Lock) -> None:
        import http.client
        from urllib.parse import urlsplit

        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": list(spec.prompt_ids),
            "repetition_penalty": 1.0,
            "max_tokens": spec.max_new_tokens,
            "logprobs": None,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self._ignore_eos:
            payload["ignore_eos"] = True
        parts = urlsplit(self._base_url)
        chunk_times: list[float] = []
        output_tokens = 0
        error = ""
        conn = http.client.HTTPConnection(
            parts.hostname or "127.0.0.1", parts.port or 80, timeout=self._timeout_s
        )
        st = time.perf_counter()  # endpoint_request_func.py:200 — after the arrival wait
        try:
            conn.request(
                "POST",
                f"{parts.path}/v1/completions",
                body=json.dumps(payload),
                headers={"Content-Type": "application/json", "x-request-id": str(spec.request_id)},
            )
            resp = conn.getresponse()
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} {resp.reason}: {resp.read()[:200]!r}")
            for raw in resp:
                line = raw.strip()
                if not line or line.startswith(b":"):
                    continue
                chunk = line.removeprefix(b"data: ")
                if chunk == b"[DONE]":
                    continue
                data = json.loads(chunk)
                if data.get("choices"):
                    chunk_times.append(time.perf_counter())
                elif data.get("usage"):
                    output_tokens = int(data["usage"].get("completion_tokens") or 0)
        except Exception as exc:  # a failed request is data, not a crash
            error = f"{type(exc).__name__}: {exc}"
        finally:
            conn.close()
        outcome = RequestOutcome(
            request_id=spec.request_id,
            prompt_len=spec.prompt_len,
            sent_s=st,
            chunk_times_s=tuple(chunk_times),
            output_tokens=output_tokens or len(chunk_times),
            success=bool(chunk_times) and not error,
            error=error,
        )
        with lock:
            out.append(outcome)

    def run(self, stream: RequestStream) -> ArmRun:
        outcomes: list[RequestOutcome] = []
        lock = threading.Lock()
        threads: list[threading.Thread] = []
        epoch_s = time.perf_counter()
        for spec in stream.specs:
            delay = epoch_s + spec.arrival_s - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            t = threading.Thread(target=self._one, args=(spec, outcomes, lock), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        end_s = time.perf_counter()
        outcomes.sort(key=lambda o: o.request_id)
        return ArmRun(outcomes=tuple(outcomes), epoch_s=epoch_s, end_s=end_s)
