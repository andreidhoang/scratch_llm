"""Hopper FP8 GEMM (sm_90a) roofline — STUB bench.

The kernel is unimplemented; this bench prints the pre-registered target and
exits without running (so the runner reports the rung honestly). The moment you
implement ``csrc/gemm/fp8_gemm_sm90.cu``, this bench runs end-to-end with no
edits — the loader + dispatch + harness are all in place.

Run on the rental box (H100):
  PYTHONPATH=src python -m bench.kernels.gemm.cuda_fp8
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from scratch_llm.kernels.common.arch import compute_capability, require_cc  # noqa: E402
from scratch_llm.kernels.gemm.cuda.fp8 import fp8_gemm  # noqa: E402


def main() -> None:
    require_cc(9, 0, fn_name="cuda_fp8 bench")
    print(f"# FP8 GEMM bench · cc={compute_capability()}")
    print("# STATUS: STUB — kernel not implemented (csrc/gemm/fp8_gemm_sm90.cu)")
    print("# PRE-REGISTERED TARGET: ~1200-1400 TF/s on H100 @4096^3 fp8 (60-70% of 1979 dense)")
    print("# Once implemented, this bench measures %-of-cuBLAS + %-of-H100-fp8-peak.")
    print("# Triggering the loader to confirm the NotImplementedError surfaces cleanly:")
    try:
        import torch  # noqa: PLC0415

        fp8_gemm(
            torch.zeros(64, 64, device="cuda", dtype=torch.float8_e4m3fn),
            torch.zeros(64, 64, device="cuda", dtype=torch.float8_e4m3fn),
        )
    except NotImplementedError as e:
        print(f"# NotImplementedError (expected): {e.args[0][:80]}...")


if __name__ == "__main__":
    main()
