"""Hopper stream-K GEMM (sm_90a) roofline — STUB bench.

The kernel is unimplemented; this bench prints the pre-registered target and
exits. Once ``csrc/gemm/stream_k_sm90.cu`` is implemented, this bench measures
the stream-K wave-quantization win over plain wave-parallel wgmma.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from scratch_llm.kernels.common.arch import compute_capability, require_cc  # noqa: E402
from scratch_llm.kernels.gemm.cuda.stream_k import stream_k_gemm  # noqa: E402


def main() -> None:
    require_cc(9, 0, fn_name="cuda_stream_k bench")
    print(f"# stream-K GEMM bench · cc={compute_capability()}")
    print("# STATUS: STUB — kernel not implemented (csrc/gemm/stream_k_sm90.cu)")
    print("# PRE-REGISTERED: closes the tail-wave gap on wgmma (90% → 97% of cuBLAS @4096^3)")
    try:
        import torch  # noqa: PLC0415

        stream_k_gemm(
            torch.zeros(64, 64, device="cuda", dtype=torch.float16),
            torch.zeros(64, 64, device="cuda", dtype=torch.float16),
        )
    except NotImplementedError as e:
        print(f"# NotImplementedError (expected): {e.args[0][:80]}...")


if __name__ == "__main__":
    main()
