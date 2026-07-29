"""Hopper FlashAttention-3 (sm_90a) vs the CPU oracle — oracle-first correctness (GPU, H100/H200).

Mirrors ``test_flash_attention_triton.py``: the oracle is
``flash_attention_forward`` (the CPU FA2 recurrence — pure PyTorch, the load-
bearing ground truth). The FA3 kernel is the warp-specialized WGMMA+TMA path;
correctness is checked against the oracle in FP32 (tight tolerance) and in
fp16 (loose — the kernel's accumulator is fp32 but inputs/outputs are fp16).

ARCH GATE: skips on any non-Hopper device. Runs on the H100/H200 rental day.
"""

import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("flash_attention_fa3_forward requires CUDA", allow_module_level=True)

from scratch_llm.kernels.common.arch import compute_capability  # noqa: E402

if compute_capability() != (9, 0):
    pytest.skip(
        f"flash_attention_fa3_forward requires Hopper (sm_90a); current device is "
        f"{compute_capability()}",
        allow_module_level=True,
    )

from scratch_llm.kernels import flash_attention_forward  # noqa: E402
from scratch_llm.kernels.attention.prefill.fa3 import (  # noqa: E402
    flash_attention_fa3_forward,
)


@pytest.mark.parametrize("is_causal", [True, False], ids=["causal", "noncausal"])
def test_matches_oracle(is_causal: bool) -> None:
    """FA3 fwd matches the CPU oracle (D=64, the WGMMA atom width; one head, short seq)."""
    torch.manual_seed(0)
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    o_fa3 = flash_attention_fa3_forward(q, k, v, is_causal=is_causal)
    # Oracle takes (S, D, S, D) self-attention shape; collapse B,H into the seq axis.
    q3 = q.reshape(S, D)
    k3 = k.reshape(S, D)
    v3 = v.reshape(S, D)
    o_ref, _lse = flash_attention_forward(q3, k3, v3, is_causal=is_causal, q_tile=64, k_tile=64)
    o_ref = o_ref.reshape(B, H, S, D).cuda()
    rel = (o_fa3.float() - o_ref.float()).abs().max().item() / (
        o_ref.float().abs().max().item() + 1e-6
    )
    assert rel <= 5e-2, f"FA3 vs oracle rel err {rel:.3e} > 5e-2"
