"""E001 -- Does the chunked training path agree with the recurrent decode path,
and if not, what predicts the gap?

Runs on a CPU. Costs nothing. Takes about a minute. This is the whole week-2
result in precursor form: if the effect is real it shows up here in pure
PyTorch, which means it is a property of the ALGEBRA and not of anybody's
kernel -- a much stronger claim than "FLA's Triton kernel disagrees with
itself", and one you can then go confirm against the real kernels on silicon.

    python -m experiments.e001_gate_sweep --self-test   # is reference.py right?
    python -m experiments.e001_gate_sweep --smoke       # is the driver right?
    python -m experiments.e001_gate_sweep --run         # the experiment

BEFORE --run, write these three numbers in LEDGER.md:
    1. float64 chunked-vs-recurrent relative error at log_gate = 0.  (order of magnitude)
    2. bfloat16 relative error at log_gate = 0.
    3. Which hypothesis you expect to survive: H1, H2, or H3. And the mechanism.
No prediction, no run. You cannot size a gap with one side of it.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from scratch_llm.mastery.divergence import sweep, verdict, make_inputs, DTYPES, _rel
from scratch_llm.mastery.paths import chunked_wy, mock_linear_attention


def self_test():
    """Is reference.py correct? fp64, benign gate, small tensors. Pass/fail only --
    this deliberately reports no science."""
    from scratch_llm.mastery.reference import recurrent_reference
    ok = True
    for lg in (-0.05, -0.2):
        for T, dh in ((128, 16), (257, 32)):
            q, k, v, la, b = make_inputs(T, dh, dh, lg, seed=3)
            try:
                O_r, S_r = recurrent_reference(q, k, v, la, b)
            except NotImplementedError as e:
                print(f"\n{e}\n")
                return False
            O_c, S_c = chunked_wy(q, k, v, la, b, chunk_size=64)
            eo, es = _rel(O_c, O_r), _rel(S_c, S_r)
            good = eo < 1e-10 and es < 1e-10
            ok &= good
            print(f"  log_gate={lg:<6} T={T:<4} d={dh:<3} "
                  f"out {'PASS' if eo < 1e-10 else 'FAIL'} ({eo:.2e})   "
                  f"state {'PASS' if es < 1e-10 else 'FAIL'} ({es:.2e})")
    if not ok:
        print("\n  Your recurrence disagrees with the chunked path in float64.\n"
              "  In fp64 at a benign gate they MUST agree to ~1e-13. Something is\n"
              "  structurally wrong, not numerically. Two usual suspects:\n"
              "    - o_t computed from S_{t-1} instead of S_t (off-by-one in time)\n"
              "    - the eraser applied after the write instead of before\n")
    else:
        print("\n  reference.py is correct. Now write your three predictions, then --run.")
    return ok


def smoke():
    """Does the driver work end to end? Uses a deliberately WRONG oracle (plain
    gated linear attention, no delta term) so nothing about the real question
    leaks. We only assert that the machinery runs and produces finite numbers."""
    res = sweep(mock_linear_attention, log_gates=(0.0, -0.1),
                dtypes=("float64", "bfloat16"), seq_lens=(128,), d_heads=(16,))
    assert len(res.points) == 4, len(res.points)
    for p in res.points:
        assert p.rel_err_o == p.rel_err_o, "NaN in driver"
        assert p.cond_T_max > 0 and p.max_inv_a > 0, "diagnostics not populated"
    print(f"  driver OK: {len(res.points)} points, diagnostics populated, no NaN")
    print("  (rel_err is meaningless here by construction -- wrong oracle on purpose)")
    return True


def run(args):
    from scratch_llm.mastery.reference import recurrent_reference
    res = sweep(
        recurrent_reference,
        log_gates=tuple(float(x) for x in args.gates.split(",")),
        dtypes=("float64", "float32", "bfloat16"),
        chunk_sizes=tuple(int(x) for x in args.chunks.split(",")),
        seq_lens=(args.seq_len,),
        d_heads=(args.d_head,),
        seed=args.seed,
        beta_scale=args.beta_scale,
        solve_dtype=args.solve_dtype,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    path = res.to_json(args.out)

    print(f"\n  oracle: {res.meta['oracle']}   seq_len={args.seq_len} d_head={args.d_head}")
    print(f"  solve_dtype: {res.meta['solve_dtype']}\n")
    hdr = f"  {'log_gate':>10} {'dtype':>9} {'C':>4} {'rel_err_o':>12} {'cond(T)':>11} {'max 1/a':>11}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for p in res.points:
        print(f"  {p.log_gate:>10.5g} {p.dtype:>9} {p.chunk_size:>4} "
              f"{p.rel_err_o:>12.3e} {p.cond_T_max:>11.3e} {p.max_inv_a:>11.3e}")

    print("\n  VERDICT")
    for kk, vv in verdict(res).items():
        if kk == "bf16_rel_err_by_gate":
            continue
        print(f"    {kk:<24} {vv}")
    print(f"\n  written: {path}")
    print("  Now: LEDGER.md row -- predicted vs measured, prediction error %, and the MECHANISM.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--gates", default="0.0,-1e-4,-1e-3,-1e-2,-0.05,-0.1,-0.25,-0.5,-1.0")
    ap.add_argument("--chunks", default="64")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--d-head", type=int, default=64)
    ap.add_argument("--beta-scale", type=float, default=1.0)
    ap.add_argument("--solve-dtype", default=None,
                    choices=[None, "float64", "float32", "bfloat16"],
                    help="precision of the CxC triangular solve, held separate from the "
                         "matmul dtype. None = auto (fp32 for bf16/fp16), which is what FLA "
                         "ships. Forcing bfloat16 prices the cost side of the divergence-cost curve.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/e001_gate_sweep.json")
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    if a.smoke:
        sys.exit(0 if smoke() else 1)
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if a.run:
        run(a)
    else:
        ap.print_help()
