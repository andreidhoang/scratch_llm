"""Comms-algebra invariants: the ring identity, the W→∞ limit, the exact DP/ZeRO-1/FSDP wire
ratios, crossover monotonicity in bandwidth and batch, and the xl worked example that generates
every number in docs/design/A2_COMMS_ALGEBRA.md — locked here as a regression."""

from __future__ import annotations

import pytest
import torch

from scratch_llm.utils.comms_calc import (
    all_gather_bytes,
    alternate_ring_allreduce_bytes,
    alternate_ring_allreduce_time,
    comms_bound_world_size,
    compute_step_time,
    crossover_link_bandwidth,
    ddp_step_bytes,
    dp_max_world,
    ffn_bwd_flops,
    ffn_fwd_flops,
    ffn_weight_bytes,
    fsdp_max_world,
    fsdp_step_bytes,
    fsdp_tp_max_world,
    naive_allreduce_bytes,
    reduce_scatter_bytes,
    ring_allreduce_bytes,
    ring_allreduce_time,
    tp_max_world_bwd,
    tp_max_world_fwd,
    tp_step_bytes,
    transformer_nonembed_params,
    zero1_step_bytes,
)

GPU_FLOPS = 1e15  # ~1 PFLOP/s bf16 accelerator
NVLINK_BW = 400e9  # ~NVLink-class egress, bytes/s
XL_D_MODEL, XL_D_FF, XL_LAYERS = 2560, 10240, 32  # official A2 Table 1 "xl"


def test_ring_identity_and_naive_ratio() -> None:
    for size in (1.0, 2**20, 6 * XL_D_MODEL * XL_D_FF):
        for world in (2, 3, 8, 64):
            ring = ring_allreduce_bytes(size, world)
            assert ring == reduce_scatter_bytes(size, world) + all_gather_bytes(size, world)
            naive = naive_allreduce_bytes(size, world)
            assert ring / naive == pytest.approx(2 / world, rel=1e-12)
    assert ring_allreduce_bytes(1024, 2) == naive_allreduce_bytes(1024, 2) == 1024
    assert ring_allreduce_bytes(1024, 4) < naive_allreduce_bytes(1024, 4)


def test_ring_limit_is_two_s() -> None:
    size = 1e9
    prev = 0.0
    for world in (2, 4, 16, 256, 65536):
        cur = ring_allreduce_bytes(size, world)
        assert prev < cur < 2 * size
        prev = cur
    assert ring_allreduce_bytes(size, 10**9) == pytest.approx(2 * size, rel=1e-6)


def test_single_device_moves_no_bytes() -> None:
    assert ring_allreduce_bytes(1e9, 1) == 0.0
    assert ddp_step_bytes(10**9, torch.float16, 1) == 0.0
    with pytest.raises(ValueError):
        ring_allreduce_bytes(1e9, 0)
    with pytest.raises(ValueError):
        ring_allreduce_time(1e9, 2, link_bw=0.0)


def test_ring_time_alpha_beta_terms() -> None:
    size, world, bw, lat = 1e9, 8, 100e9, 5e-6
    t = ring_allreduce_time(size, world, bw, latency=lat, hops=2)
    assert t == pytest.approx(2 * (world - 1) * 2 * lat + ring_allreduce_bytes(size, world) / bw)
    assert ring_allreduce_time(size, world, bw) == pytest.approx(
        ring_allreduce_bytes(size, world) / bw
    )


def test_alternate_ring_is_w_over_two_slower() -> None:
    size, bw = 1e8, 50e9
    for world in (2, 4, 16):
        alt = alternate_ring_allreduce_time(size, world, bw)
        assert alt == pytest.approx((world - 1) * size / bw)
        assert alt / ring_allreduce_time(size, world, bw) == pytest.approx(world / 2)
        assert alternate_ring_allreduce_bytes(size, world) == (world - 1) * size


def test_scheme_byte_ratios_match_closed_forms() -> None:
    n_params = transformer_nonembed_params(XL_LAYERS, XL_D_MODEL, XL_D_FF)
    for world in (2, 8, 128):
        ddp = ddp_step_bytes(n_params, torch.bfloat16, world)
        assert zero1_step_bytes(n_params, torch.bfloat16, world) == ddp
        assert fsdp_step_bytes(n_params, torch.bfloat16, world) == pytest.approx(
            1.5 * ddp, rel=1e-12
        )
        assert ddp == pytest.approx(
            2 * (world - 1) / world * n_params * torch.bfloat16.itemsize, rel=1e-12
        )


def test_ffn_flops_and_weight_bytes() -> None:
    tokens = 4096
    assert ffn_fwd_flops(tokens, XL_D_MODEL, XL_D_FF) == 6 * tokens * XL_D_MODEL * XL_D_FF
    assert ffn_bwd_flops(tokens, XL_D_MODEL, XL_D_FF) == 2 * ffn_fwd_flops(
        tokens, XL_D_MODEL, XL_D_FF
    )
    assert ffn_weight_bytes(XL_D_MODEL, XL_D_FF, torch.float16) == 6 * XL_D_MODEL * XL_D_FF
    assert compute_step_time(1e15, GPU_FLOPS) == 1.0


def _dp_crossover(tokens: int, link_bw: float) -> int | None:
    weight_bytes = ffn_weight_bytes(XL_D_MODEL, XL_D_FF, torch.float16)
    return comms_bound_world_size(
        step_flops=ffn_bwd_flops(tokens, XL_D_MODEL, XL_D_FF),
        per_device_step_bytes=lambda w: ring_allreduce_bytes(weight_bytes, w),
        gpu_flops=GPU_FLOPS,
        link_bw=link_bw,
    )


def test_crossover_solver_matches_bruteforce() -> None:
    for tokens, bw in ((1024, 50e9), (65536, 50e9), (65536, 400e9)):
        weight_bytes = ffn_weight_bytes(XL_D_MODEL, XL_D_FF, torch.float16)
        flops = ffn_bwd_flops(tokens, XL_D_MODEL, XL_D_FF)
        brute = next(
            w
            for w in range(2, 500)
            if ring_allreduce_bytes(weight_bytes, w) / bw > flops / w / GPU_FLOPS
        )
        assert _dp_crossover(tokens, bw) == brute


def _crossovers(configs: list[tuple[int, float]]) -> list[int]:
    out: list[int] = []
    for tokens, bw in configs:
        w = _dp_crossover(tokens, bw)
        assert w is not None
        out.append(w)
    return out


def test_crossover_monotone_in_bandwidth_and_batch() -> None:
    by_bw = _crossovers([(65536, bw) for bw in (25e9, 50e9, 100e9, 200e9, 400e9)])
    assert by_bw == sorted(by_bw) and len(set(by_bw)) > 1
    by_tokens = _crossovers([(tokens, NVLINK_BW) for tokens in (8192, 16384, 32768, 65536)])
    assert by_tokens == sorted(by_tokens) and len(set(by_tokens)) > 1


def test_crossover_none_when_never_bound() -> None:
    assert (
        comms_bound_world_size(
            step_flops=1e30,
            per_device_step_bytes=lambda w: ring_allreduce_bytes(1.0, w),
            gpu_flops=GPU_FLOPS,
            link_bw=NVLINK_BW,
            max_world=1 << 16,
        )
        is None
    )


def test_crossover_link_bandwidth_self_consistent() -> None:
    tokens, world = 65536, 8
    flops = ffn_bwd_flops(tokens, XL_D_MODEL, XL_D_FF)
    bytes_at_world = ring_allreduce_bytes(
        ffn_weight_bytes(XL_D_MODEL, XL_D_FF, torch.float16), world
    )
    bw_min = crossover_link_bandwidth(flops, bytes_at_world, GPU_FLOPS, world)
    assert bytes_at_world / bw_min == pytest.approx(compute_step_time(flops / world, GPU_FLOPS))
    assert crossover_link_bandwidth(flops, bytes_at_world, GPU_FLOPS, 2 * world) > bw_min


def test_closed_form_bound_relationships() -> None:
    tokens = 65536
    assert fsdp_max_world(tokens, NVLINK_BW, GPU_FLOPS) == dp_max_world(
        tokens, NVLINK_BW, GPU_FLOPS
    )
    assert tp_max_world_bwd(XL_D_FF, NVLINK_BW, GPU_FLOPS) == pytest.approx(
        2 * tp_max_world_fwd(XL_D_FF, NVLINK_BW, GPU_FLOPS)
    )
    overlapped = fsdp_tp_max_world(tokens, XL_D_FF, NVLINK_BW, GPU_FLOPS, overlapped=True)
    sequential = fsdp_tp_max_world(tokens, XL_D_FF, NVLINK_BW, GPU_FLOPS, overlapped=False)
    assert overlapped / sequential == pytest.approx(4.0)
    assert overlapped == pytest.approx(
        tp_max_world_fwd(XL_D_FF, NVLINK_BW, GPU_FLOPS) * dp_max_world(tokens, NVLINK_BW, GPU_FLOPS)
    )
    assert dp_max_world(2 * tokens, NVLINK_BW, GPU_FLOPS) == pytest.approx(
        2 * dp_max_world(tokens, NVLINK_BW, GPU_FLOPS)
    )


def test_xl_worked_example_regression() -> None:
    """The exact numbers published in docs/design/A2_COMMS_ALGEBRA.md — any drift breaks the doc."""
    tokens = 65536  # batch 128 × seq 512
    assert dp_max_world(tokens, NVLINK_BW, GPU_FLOPS) == pytest.approx(26.2144, rel=1e-12)
    assert dp_max_world(tokens, 50e9, GPU_FLOPS) == pytest.approx(3.2768, rel=1e-12)
    assert tp_max_world_fwd(XL_D_FF, NVLINK_BW, GPU_FLOPS) == pytest.approx(6.144, rel=1e-12)
    assert tp_max_world_bwd(XL_D_FF, NVLINK_BW, GPU_FLOPS) == pytest.approx(12.288, rel=1e-12)
    assert fsdp_tp_max_world(tokens, XL_D_FF, NVLINK_BW, GPU_FLOPS) == pytest.approx(
        161.0612736, rel=1e-9
    )
    assert fsdp_tp_max_world(
        tokens, XL_D_FF, NVLINK_BW, GPU_FLOPS, overlapped=False
    ) == pytest.approx(40.2653184, rel=1e-9)
    assert _dp_crossover(tokens, NVLINK_BW) == 28  # exact solver: first W with comm > compute

    n_params = transformer_nonembed_params(XL_LAYERS, XL_D_MODEL, XL_D_FF)
    assert n_params == 3_355_443_200
    assert ddp_step_bytes(n_params, torch.bfloat16, 8) == 11_744_051_200.0
    assert zero1_step_bytes(n_params, torch.bfloat16, 8) == 11_744_051_200.0
    assert fsdp_step_bytes(n_params, torch.bfloat16, 8) == 17_616_076_800.0
    assert tp_step_bytes(8, 512, XL_D_MODEL, XL_LAYERS, torch.bfloat16, 8) == 4_697_620_480.0
