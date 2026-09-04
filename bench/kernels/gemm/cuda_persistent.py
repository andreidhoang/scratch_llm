"""Hopper persistent GEMV (sm_90a) roofline — STUB bench.

The kernel is unimplemented; this bench prints the pre-registered target and
exits. Once ``csrc/persistent/persistent_gemv_sm90.cu`` is implemented, this
bench measures the persistent + TMA decode path vs the Triton gemv_split.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from scratch_llm.kernels.common.arch import compute_capability, require_arch  # noqa: E402
from scratch_llm.kernels.gemm.cuda.persistent import persistent_gemv  # noqa: E402


def main() -> None:
    require_arch((9, 0), fn_name="cuda_persistent bench")
    print(f"# persistent GEMV bench · cc={compute_capability()}")
    print("# STATUS: STUB — kernel not implemented (csrc/persistent/persistent_gemv_sm90.cu)")
    print("# PRE-REGISTERED: ~95% of H100 HBM bandwidth (3.35 TB/s) on M=1, K=N=4096 decode")
    try:
        import torch  # noqa: PLC0415

        persistent_gemv(
            torch.zeros(4096, device="cuda", dtype=torch.float16),
            torch.zeros(4096, 4096, device="cuda", dtype=torch.float16),
        )
    except NotImplementedError as e:
        print(f"# NotImplementedError (expected): {e.args[0][:80]}...")


if __name__ == "__main__":
    main()
