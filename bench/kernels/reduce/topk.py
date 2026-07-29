"""A2 R4 — top-k roofline: measure the failure, then measure the fusion win.

Two things this bench proves, both honestly:

1. **Top-k is a poor GPU fit.** A row-wise top-k moves little data (read the row once, write k) but
   is a *selection* — k serial, dependent max-reductions with ~zero arithmetic intensity. So its
   effective HBM bandwidth (bytes / time) lands FAR below the 0.55 TB/s peak: it is
   latency/occupancy-bound, not bandwidth-bound. The low %-of-peak is the CORRECT result. We report
   the Triton kernel next to ``torch.topk`` (radix-select) so the failure is not blamed on our code.

2. **Fusing softmax+top-k saves a HBM round trip.** A two-pass ``softmax`` (read N, write N) then
   ``topk`` (read N, write k) moves ~3N elements/row; the fused kernel reads N once and writes 2k.
   We measure the traffic + wall-time win.

ncu is BLOCKED on this box (ERR_NVGPUCTRPERM) → the bound is established by achieved-vs-measured-peak
% (this harness); the ncu metric we WOULD inspect is registered as debt below.

Run on the GPU box:  python bench/topk.py   (--help for shape overrides)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # bench/ on sys.path for _harness

from _harness import Roofs, bench_ms, provenance_line, spread_pct  # noqa: E402

from scratch_llm.kernels.topk_triton import fused_softmax_topk, topk_last_dim  # noqa: E402

# The ncu metric this rung would inspect on the H100 day (counters blocked here).
NCU_DEBT = "launch__waves_per_multiprocessor + warp state 'No Eligible' (dependent-reduction stall)"


def _gbps(bytes_: float, ms: float) -> float:
    return bytes_ / (ms * 1e-3)


def run(m: int, n: int, k: int, dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("bench/topk.py requires CUDA.")
    elem = torch.tensor([], dtype=dtype).element_size()
    roofs = Roofs.measure(dtype)
    peak = roofs.bw_bytes_s

    print(provenance_line("A2 R4 — top-k roofline"))
    print(f"# shape M={m} N={n} k={k} dtype={dtype}  | HBM peak {peak / 1e12:.3f} TB/s")

    x = torch.randn(m, n, device="cuda", dtype=dtype)

    # ---- correctness gate: never publish timings for a wrong kernel -------------------------------
    xf = x.float() + torch.arange(n, device="cuda") * 1e-6  # de-tie for an exact index oracle
    v, idx = topk_last_dim(xf, k)
    rv, ri = torch.topk(xf, k, dim=-1, sorted=True)
    assert torch.equal(idx, ri) and torch.equal(v, rv), "top-k kernel disagrees with torch.topk"
    del xf, v, idx, rv, ri
    torch.cuda.empty_cache()

    # ---- Ladder stage 1+2: the top-k "failure" — Triton vs torch.topk, both far below peak -------
    # Ideal traffic: read the row once + write k values + k indices (idx int64).
    topk_bytes = m * (n * elem + k * elem + k * 8)

    tri_ms, lo, hi = bench_ms(lambda: topk_last_dim(x, k))
    torch_ms = bench_ms(lambda: torch.topk(x, k, dim=-1, sorted=True))[0]

    tri_gbps, torch_gbps = _gbps(topk_bytes, tri_ms), _gbps(topk_bytes, torch_ms)

    print("\n# --- top-k (selection; expect LOW %-of-peak — this is the rung's lesson) ---")
    print(f"{'stage':<26}{'ms':>9}{'spread%':>9}{'GB/s':>9}{'%peak':>8}")
    print(
        f"{'1. torch.topk (radix)':<26}{torch_ms:>9.4f}{'':>9}{torch_gbps / 1e9:>9.1f}"
        f"{100 * torch_gbps / peak:>7.1f}%"
    )
    print(
        f"{'2. triton iter-max':<26}{tri_ms:>9.4f}{spread_pct(tri_ms, lo, hi):>9.1f}"
        f"{tri_gbps / 1e9:>9.1f}{100 * tri_gbps / peak:>7.1f}%"
    )

    # ---- Ladder stage 3: fused softmax+top-k vs the two-pass path — the traffic win --------------
    # Two-pass: softmax reads N + writes N, then topk reads N + writes k  →  ~3N + k moved / row.
    twopass_bytes = m * (3 * n * elem + k * elem + k * 8)
    fused_bytes = m * (n * elem + k * 4 + k * 8)  # read N once; write k fp32 probs + k int64 idx

    def _two_pass():
        p = torch.softmax(x, dim=-1)
        return torch.topk(p, k, dim=-1, sorted=True)

    # fused correctness vs the two-pass reference (indices exact; probs to fp32 tol)
    fp, fi = fused_softmax_topk(x, k)
    rp, rri = torch.topk(torch.softmax(x, dim=-1), k, dim=-1, sorted=True)
    assert torch.equal(fi, rri), "fused indices diverged"
    torch.testing.assert_close(fp, rp.float(), atol=1e-5, rtol=1e-3)
    del fp, fi, rp, rri
    torch.cuda.empty_cache()

    two_ms = bench_ms(_two_pass)[0]
    fused_ms, flo, fhi = bench_ms(lambda: fused_softmax_topk(x, k))

    print("\n# --- softmax + top-k: fusion saves the softmax write + re-read round trip ---")
    print(f"{'stage':<26}{'ms':>9}{'spread%':>9}{'GB/s':>9}{'traffic':>10}")
    print(
        f"{'A. softmax|topk (2-pass)':<26}{two_ms:>9.4f}{'':>9}"
        f"{_gbps(twopass_bytes, two_ms) / 1e9:>9.1f}{twopass_bytes / 1e6:>9.1f}MB"
    )
    print(
        f"{'B. fused softmax+topk':<26}{fused_ms:>9.4f}{spread_pct(fused_ms, flo, fhi):>9.1f}"
        f"{_gbps(fused_bytes, fused_ms) / 1e9:>9.1f}{fused_bytes / 1e6:>9.1f}MB"
    )
    print(
        f"\n# fusion: {two_ms / fused_ms:.2f}x faster wall, "
        f"{twopass_bytes / fused_bytes:.2f}x less ideal HBM traffic "
        f"({twopass_bytes / 1e6:.1f}MB -> {fused_bytes / 1e6:.1f}MB)"
    )
    print(f"# ncu-debt (blocked here; H100 day): {NCU_DEBT}")


def _parse_dtype(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--n", type=int, default=32768)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--dtype", type=_parse_dtype, default=torch.float32, help="bf16|fp16|fp32")
    a = ap.parse_args()
    run(a.m, a.n, a.k, a.dtype)
