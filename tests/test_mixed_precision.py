"""The two mixed-precision numerics lessons, as CPU invariants.

(1) Sequential fp16 accumulation drifts far more than fp32 and *under*-counts (small addends vanish).
(2) autocast runs matmuls in bf16 but keeps LayerNorm/softmax/reductions in fp32."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scratch_llm.utils.mixed_precision import naive_accumulate


def test_fp16_accumulation_loses_small_addends() -> None:
    value, count = 0.01, 4096
    exact = value * count  # 40.96
    s32 = naive_accumulate(value, count, torch.float32).item()
    s16 = naive_accumulate(value, count, torch.float16).item()

    # fp32 is essentially exact; fp16 drifts by orders of magnitude more...
    assert abs(s32 - exact) < 1e-2
    assert abs(s16 - exact) > 1.0
    # ...and it under-counts (the running sum stalls once increments fall below half an ulp).
    assert s16 < s32


def test_fp16_accumulation_stalls() -> None:
    # The vivid failure: once the partial sum hits a magnitude where the increment < ulp/2, adding
    # more never moves it. Summing far more terms past the stall point changes nothing.
    s_short = naive_accumulate(0.01, 4096, torch.float16).item()
    s_long = naive_accumulate(0.01, 40000, torch.float16).item()
    assert s_short == s_long  # frozen


def test_autocast_dtype_rule() -> None:
    x = torch.randn(8, 16)
    w = torch.randn(16, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mm = x @ w
        ln = F.layer_norm(x, (16,))
        sm = torch.softmax(x, dim=-1)

    assert mm.dtype == torch.bfloat16  # matmul → low precision (throughput)
    assert ln.dtype == torch.float32  # LayerNorm → fp32 (stability)
    assert sm.dtype == torch.float32  # softmax/reduction → fp32 (stability)
