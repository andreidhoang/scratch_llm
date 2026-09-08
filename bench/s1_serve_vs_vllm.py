"""S1/S-R1 — our engine vs vLLM, one load generator, 1×H100, Qwen3-8B bf16.

Plan §05 S-R1: "Your engine vs vLLM on 1×H100, Qwen3-8B bf16, `vllm bench serve` at 4/8/16 rps:
TTFT p50/p99, ITL, throughput; nsys both; name every gap" — target ≤ 2× at B=32.

Four arms, one stream generator (:mod:`scratch_llm.serving.loadgen`):

  --arm scratch   our continuous-batching engine, open loop at the stream's arrival times
  --arm vllm      vLLM's OpenAI server driven by the SAME stream object (adapter #2)
  --arm both      vllm then scratch, sequentially, one process; prints pct_of_vllm_tok_s
  --arm floor     `vllm bench serve` itself, unmodified — the ladder's named floor

Why three vLLM-shaped numbers and not one. `vllm bench serve` is the floor the plan names, so the
floor must be *it*, run with its own dataset, its own client, its own arrival draw. But its arrival
sequence cannot be handed to our engine (it lives inside an asyncio generator over the global numpy
RNG, serve.py:466-489), so a head-to-head against it alone would compare two different loads. The
`vllm` arm closes that: same `RequestStream`, same digest, same definitions. If `vllm` and `floor`
disagree by more than a few percent the generator is wrong and the head-to-head is void — that
check is the first thing to read in the JSON, before any ratio.

Both arms cannot hold the GPU at once (vLLM takes 90% of HBM by default), so `--arm both` runs them
sequentially with vLLM FIRST: clocks are locked by infra/bench.sh, so the residual is thermal, and
running our engine on the hotter card biases the comparison against our own claim.

    experiments/S1/S-R1/run.sh                     # the rung's number, through infra/bench.sh
    experiments/S1/S-R1/floor.sh                   # the floor, `vllm bench serve`
    python bench/s1_serve_vs_vllm.py --arm both --rps 4,8,16 --json out.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scratch_llm.serving.loadgen import (  # noqa: E402
    ArmReport,
    OpenAIServerAdapter,
    RequestStream,
    ScratchEngineAdapter,
    build_stream,
    qwen3_8b_shape,
    summarize,
)

DEFAULT_MODEL = "Qwen/Qwen3-8B"


@dataclass(frozen=True)
class Workload:
    """Everything that fixes the work, so both arms and the floor can be given it verbatim."""

    model: str
    input_len: int
    output_len: int
    num_prompts: int
    seed: int
    rates: tuple[float, ...]
    headline_rps: float
    n_slots: int
    warmup_requests: int
    device: str

    @property
    def max_model_len(self) -> int:
        return self.input_len + self.output_len + 8  # +8 slack for special tokens


# ---------------------------------------------------------------------------------------------
# vLLM server lifecycle (arm `vllm` and arm `floor` both need a live server)
# ---------------------------------------------------------------------------------------------


class VllmServer:
    """`vllm serve` as a context manager, with `--load-format dummy`.

    Dummy weights are the point, not a shortcut: this repo has no safetensors→TransformerLM loader,
    so our arm serves Qwen3-8B's *geometry* with random weights. Giving vLLM random weights too
    makes the two arms' FLOPs, bytes, and KV footprint identical; `--ignore-eos` makes the output
    length identical; nothing in the metric set reads a token value. The HF config and tokenizer are
    still fetched (they carry the geometry), so the box needs network or a warm HF cache once.
    """

    def __init__(self, wl: Workload, port: int, extra: list[str] | None = None) -> None:
        self._wl = wl
        self._port = port
        self._extra = extra or []
        self._proc: subprocess.Popen[bytes] | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    def __enter__(self) -> VllmServer:
        cmd = [
            "vllm",
            "serve",
            self._wl.model,
            "--load-format",
            "dummy",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(self._wl.max_model_len),
            "--port",
            str(self._port),
            "--seed",
            str(self._wl.seed),
            # NOT --disable-log-requests: that flag does not exist at oss/vllm@dedcfa4483 —
            # request logging is off by default (arg_utils.py:2916 enable_log_requests=False),
            # and passing the removed flag makes `vllm serve` exit before it binds a port.
            *self._extra,
        ]
        print(f"# starting: {' '.join(cmd)}", flush=True)
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        deadline = time.time() + 900.0
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited with {self._proc.returncode} during startup")
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as r:
                    if r.status == 200:
                        print("# vllm server ready", flush=True)
                        return self
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(2.0)
        raise TimeoutError("vllm serve did not become healthy within 900 s")

    def __exit__(self, *exc: object) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=60)


# ---------------------------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------------------------


def _stream(
    wl: Workload, rate: float, *, seed_offset: int = 0, n: int | None = None
) -> RequestStream:
    return build_stream(
        n if n is not None else wl.num_prompts,
        rate,
        seed=wl.seed + seed_offset,
        input_len=wl.input_len,
        output_len=wl.output_len,
    )


def run_vllm_arm(wl: Workload, base_url: str) -> dict[str, ArmReport]:
    """vLLM's server, driven by OUR generator. Same stream object our engine gets."""
    adapter = OpenAIServerAdapter(base_url, wl.model)
    if wl.warmup_requests:
        adapter.run(_stream(wl, max(wl.rates), seed_offset=9973, n=wl.warmup_requests))
    out: dict[str, ArmReport] = {}
    for rate in wl.rates:
        stream = _stream(wl, rate)
        run = adapter.run(stream)
        out[f"{rate:g}"] = summarize(
            "vllm", stream, run.outcomes, epoch_s=run.epoch_s, end_s=run.end_s, extra=run.extra
        )
    return out


def run_scratch_arm(wl: Workload) -> dict[str, ArmReport]:
    """Our engine. Qwen3-8B geometry, random bf16 weights, dense KV slab, greedy decode."""
    import torch

    from scratch_llm.model import TransformerLM

    cfg = qwen3_8b_shape(wl.max_model_len)
    model = TransformerLM(cfg).to(device=wl.device, dtype=torch.bfloat16).eval()
    adapter = ScratchEngineAdapter(model, n_slots=wl.n_slots, device=wl.device)
    if wl.warmup_requests:
        adapter.run(_stream(wl, max(wl.rates), seed_offset=9973, n=wl.warmup_requests))
    out: dict[str, ArmReport] = {}
    for rate in wl.rates:
        stream = _stream(wl, rate)
        run = adapter.run(stream)
        out[f"{rate:g}"] = summarize(
            "scratch_llm",
            stream,
            run.outcomes,
            epoch_s=run.epoch_s,
            end_s=run.end_s,
            extra=run.extra,
        )
    return out


def run_floor(wl: Workload, base_url: str, rate: float, result_path: Path) -> dict[str, Any]:
    """`vllm bench serve`, unmodified — the ladder's floor, run with its own client and dataset."""
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        base_url,
        "--model",
        wl.model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(wl.input_len),
        "--random-output-len",
        str(wl.output_len),
        "--random-range-ratio",
        "0.0",
        "--num-prompts",
        str(wl.num_prompts),
        "--num-warmups",
        str(wl.warmup_requests),
        "--request-rate",
        str(rate),
        "--burstiness",
        "1.0",
        "--ignore-eos",
        "--seed",
        str(wl.seed),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,99",
        "--save-result",
        "--result-dir",
        str(result_path.parent),
        "--result-filename",
        result_path.name,
    ]
    print(f"# floor: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    with result_path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("scratch", "vllm", "both", "floor"), default="both")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--rps", default="4,8,16", help="comma-separated request rates")
    p.add_argument("--headline-rps", type=float, default=16.0, help="rate the bare number reports")
    p.add_argument("--input-len", type=int, default=1024)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--num-prompts", type=int, default=500)
    p.add_argument("--warmup-requests", type=int, default=32)
    p.add_argument("--n-slots", type=int, default=64, help="our engine's KV slot count")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--port", type=int, default=8117)
    p.add_argument("--json", type=Path, default=None, help="write the full report here")
    args = p.parse_args(argv)

    rates = tuple(float(x) for x in args.rps.split(","))
    wl = Workload(
        model=args.model,
        input_len=args.input_len,
        output_len=args.output_len,
        num_prompts=args.num_prompts,
        seed=args.seed,
        rates=rates,
        headline_rps=args.headline_rps,
        n_slots=args.n_slots,
        warmup_requests=args.warmup_requests,
        device=args.device,
    )
    key = f"{wl.headline_rps:g}"
    report: dict[str, Any] = {
        "rung": "S1/S-R1",
        "workload": {
            "model": wl.model,
            "dtype": "bfloat16",
            "weights": "dummy/random",
            "input_len": wl.input_len,
            "output_len": wl.output_len,
            "num_prompts": wl.num_prompts,
            "seed": wl.seed,
            "rates": list(rates),
            "headline_rps": wl.headline_rps,
            "n_slots": wl.n_slots,
            "device": wl.device,
        },
        "arms": {},
    }

    bare: float
    if args.arm == "floor":
        with VllmServer(wl, args.port) as srv:
            out = run_floor(wl, srv.base_url, wl.headline_rps, Path(args.json or "floor.json"))
        report["arms"]["vllm_bench_serve"] = out
        bare = float(out["output_throughput"])
    elif args.arm == "vllm":
        with VllmServer(wl, args.port) as srv:
            arm = run_vllm_arm(wl, srv.base_url)
        report["arms"]["vllm"] = {k: v.to_dict() for k, v in arm.items()}
        bare = arm[key].output_throughput
    elif args.arm == "scratch":
        arm = run_scratch_arm(wl)
        report["arms"]["scratch_llm"] = {k: v.to_dict() for k, v in arm.items()}
        bare = arm[key].output_throughput
    else:  # both — vLLM first (see module docstring: heat-soak biases against our own claim)
        with VllmServer(wl, args.port) as srv:
            vllm_arm = run_vllm_arm(wl, srv.base_url)
        scratch_arm = run_scratch_arm(wl)
        report["arms"]["vllm"] = {k: v.to_dict() for k, v in vllm_arm.items()}
        report["arms"]["scratch_llm"] = {k: v.to_dict() for k, v in scratch_arm.items()}
        report["stream_digests_match"] = {
            k: vllm_arm[k].stream_digest == scratch_arm[k].stream_digest for k in vllm_arm
        }
        report["pct_of_vllm_tok_s"] = {
            k: 100.0 * scratch_arm[k].output_throughput / vllm_arm[k].output_throughput
            for k in vllm_arm
        }
        if not all(report["stream_digests_match"].values()):
            raise SystemExit("REFUSED: the two arms did not see the same stream — ratio is fiction")
        bare = report["pct_of_vllm_tok_s"][key]

    if args.json is not None and args.arm != "floor":
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"{bare:.4f}")  # the harness convention: one bare number, last line
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
