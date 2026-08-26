"""Sweep driver: how far apart are the two paths, and what predicts it.

Three hypotheses are pre-registered here. The sweep is designed to separate
them, and the sweep is worthless if you have not written your prediction down
first. That is not ceremony -- "sizing the gap between actual performance and
theoretical rooflines" is the literal job description, and a gap has two sides.

  H1  TRIANGULAR CONDITIONING.  The WY form needs (I + tril(B K K^T,-1))^{-1}.
      When the gate -> 1 the transition product is undamped and that system is
      ill-conditioned.  PREDICTS: divergence tracks cond(T), rises as
      log_alpha -> 0, and SURVIVES IN FLOAT64.

  H2  GATE RESCALING.  The substitution puts beta_t / a_t in the numerator, and
      1/a_t grows geometrically inside a chunk as the gate decays.
      PREDICTS: divergence tracks max(1/a) and rises as log_alpha -> -inf,
      i.e. the OPPOSITE direction from H1.

  H3  PRECISION ONLY.  Nothing structural; the difference is just bf16 operands
      and tensor-core accumulation order.
      PREDICTS: float64 divergence is ~1e-14 at EVERY gate value, and the whole
      effect disappears when you raise precision.

H3 is the control and it is the one that matters. If fp64 chunked-vs-recurrent
agreement is machine-eps flat across the gate sweep, then H1 is dead, the FLA
#389 report is a pure precision artifact, and the paper you are writing is a
different paper. Find that out on day one, on a CPU, for zero dollars -- not in
week nine on a rented H100.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import torch

from .paths import chunked_wy


@dataclass
class Point:
    log_gate: float
    dtype: str
    chunk_size: int
    seq_len: int
    d_head: int
    rel_err_o: float  # ||O_chunked - O_ref|| / ||O_ref||,  vs fp64 reference
    max_abs_err_o: float
    rel_err_state: float
    cond_T_max: float  # H1 diagnostic
    max_inv_a: float  # H2 diagnostic
    ref_norm: float


@dataclass
class SweepResult:
    points: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(
                {"meta": self.meta, "points": [asdict(p) for p in self.points]},
                f,
                indent=2,
            )
        return path


DTYPES = {"float64": torch.float64, "float32": torch.float32, "bfloat16": torch.bfloat16}


def make_inputs(T, dk, dv, log_gate, seed, device="cpu", beta_scale=1.0):
    """Fixed synthetic inputs. Keys are L2-normalised -- that is what real models
    produce after their key norm, and it is what makes the delta rule's eraser
    a projection rather than an arbitrary rank-1 perturbation."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(T, dk, generator=g, dtype=torch.float64)
    k = torch.randn(T, dk, generator=g, dtype=torch.float64)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(T, dv, generator=g, dtype=torch.float64)
    # beta in (0,1): sigmoid of a standard normal, scaled. beta -> 1 = full overwrite.
    beta = torch.sigmoid(torch.randn(T, generator=g, dtype=torch.float64)) * beta_scale
    log_alpha = torch.full((T,), float(log_gate), dtype=torch.float64)
    return [t.to(device) for t in (q, k, v, log_alpha, beta)]


def _rel(a, b):
    """Relative error against b, computed in fp64 so the metric is not itself lossy."""
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    n = b64.norm()
    return ((a64 - b64).norm() / n).item() if n > 0 else float("nan")


def sweep(
    reference_fn,
    log_gates=(0.0, -1e-4, -1e-3, -1e-2, -0.05, -0.1, -0.25, -0.5, -1.0),
    dtypes=("float64", "float32", "bfloat16"),
    chunk_sizes=(64,),
    seq_lens=(512,),
    d_heads=(64,),
    seed=0,
    device="cpu",
    beta_scale=1.0,
    solve_dtype=None,
):
    """Ground truth is ALWAYS reference_fn in float64. Every reported number is a
    deviation from that single oracle, so numbers from different rows are
    comparable -- which is the whole reason to fix one oracle."""
    res = SweepResult(
        meta={
            "oracle": f"{reference_fn.__module__}.{reference_fn.__name__} @ float64",
            "device": device,
            "seed": seed,
            "beta_scale": beta_scale,
            "solve_dtype": str(solve_dtype) if solve_dtype else "auto (fp32 for bf16/fp16)",
            "torch": torch.__version__,
            "hypotheses": ["H1 cond(T)", "H2 max(1/a)", "H3 precision-only"],
        }
    )
    for T in seq_lens:
        for dh in d_heads:
            for lg in log_gates:
                q, k, v, la, b = make_inputs(T, dh, dh, lg, seed, device, beta_scale)
                O_ref, S_ref = reference_fn(q, k, v, la, b)  # fp64 oracle
                for C in chunk_sizes:
                    for dn in dtypes:
                        dt = DTYPES[dn]
                        O_c, S_c, diag = chunked_wy(
                            q.to(dt),
                            k.to(dt),
                            v.to(dt),
                            la.to(dt),
                            b.to(dt),
                            chunk_size=C,
                            return_diagnostics=True,
                            solve_dtype=DTYPES.get(solve_dtype)
                            if isinstance(solve_dtype, str)
                            else solve_dtype,
                        )
                        res.points.append(
                            Point(
                                log_gate=lg,
                                dtype=dn,
                                chunk_size=C,
                                seq_len=T,
                                d_head=dh,
                                rel_err_o=_rel(O_c, O_ref),
                                max_abs_err_o=(O_c.to(torch.float64) - O_ref.to(torch.float64))
                                .abs()
                                .max()
                                .item(),
                                rel_err_state=_rel(S_c, S_ref),
                                cond_T_max=max(diag["cond_T"]) if diag["cond_T"] else float("nan"),
                                max_inv_a=max(diag["max_inv_a"])
                                if diag["max_inv_a"]
                                else float("nan"),
                                ref_norm=O_ref.to(torch.float64).norm().item(),
                            )
                        )
    return res


def verdict(res: SweepResult) -> dict:
    """Mechanically score the three hypotheses. No interpretation, no vibes."""
    f64 = [p for p in res.points if p.dtype == "float64"]
    bf16 = [p for p in res.points if p.dtype == "bfloat16"]
    out = {}
    if f64:
        out["fp64_max_rel_err"] = max(p.rel_err_o for p in f64)
        out["fp64_at_gate_1"] = next((p.rel_err_o for p in f64 if p.log_gate == 0.0), None)
        # H3 dies if fp64 divergence is well above machine eps anywhere.
        out["H3_precision_only"] = "SUPPORTED" if out["fp64_max_rel_err"] < 1e-11 else "REFUTED"
    if bf16:
        by_gate = sorted(bf16, key=lambda p: -p.log_gate)  # 0.0 first, most negative last
        out["bf16_rel_err_by_gate"] = [(p.log_gate, p.rel_err_o) for p in by_gate]
        if len(by_gate) > 1:
            hi, lo = by_gate[0].rel_err_o, by_gate[-1].rel_err_o
            out["H1_gate_to_1_worse"] = "SUPPORTED" if hi > 2 * lo else "REFUTED"
            out["H2_gate_to_0_worse"] = "SUPPORTED" if lo > 2 * hi else "REFUTED"
        cs = [p.cond_T_max for p in bf16 if math.isfinite(p.cond_T_max)]
        es = [p.rel_err_o for p in bf16 if math.isfinite(p.cond_T_max)]
        if len(cs) > 2:
            out["corr_err_vs_condT"] = _pearson(
                [math.log10(max(c, 1e-30)) for c in cs], [math.log10(max(e, 1e-30)) for e in es]
            )
            out["corr_err_vs_inv_a"] = _pearson(
                [math.log10(max(p.max_inv_a, 1e-30)) for p in bf16],
                [math.log10(max(p.rel_err_o, 1e-30)) for p in bf16],
            )
    return out


def _pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    # strict=True: a length mismatch here would silently truncate to the shorter series and
    # return a perfectly plausible correlation computed over the wrong set of points. In the
    # file that adjudicates H1/H2/H3, a wrong-but-believable number is the worst failure mode.
    num = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")
