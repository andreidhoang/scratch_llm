"""Discipline tests for the Gated-DeltaNet linear-attention block (F10.1).

The whole point of the rung is the three-path contract: the chunkwise-parallel training
path, the recurrent-scan decode path, and the naive float64 reference loop must compute
the SAME function. If any pair diverges, the train/decode seam is broken and every
downstream ablation (F10.2) or kernel (DELTA GDN-2) built on it measures a lie.

DoD coverage (docs/FRONTIER_2026_TASKSPEC.md §B · F10, first slice):
- three-way equivalence at rel err ≤ 1e-5 (float64, B=2, T=96 — ragged tail vs chunk 64)
- chunk_size independence (16 / 64 / T+1 — state crosses boundaries correctly)
- causality (perturb t=T/2, prefix unchanged)
- decode contract: token-by-token ``step`` == chunkwise, token-exact in float64
- state bytes constant in T + < GQA-8 KV cache at T=4096 (the memory-wall claim)
- gate limits: α≡1 ⇒ plain delta rule; α≡0 ⇒ memoryless
- gradients flow through the chunkwise path to every parameter
"""

import torch

from scratch_llm.linear_attn import (
    GatedDeltaNet,
    LinearAttnConfig,
    LinearAttnState,
    gated_delta_rule_chunkwise,
    gated_delta_rule_reference,
    gated_delta_rule_step,
    linear_attn_state_bytes,
)

# DoD-mandated sizes: T=96 is deliberately NOT a multiple of chunk_size=64 (ragged tail).
B, T, H, D_K = 2, 96, 2, 16
CFG = LinearAttnConfig(d_head=D_K, n_heads=H)  # expand_v=2 ⇒ d_v=32; chunk_size=64
D_V = CFG.d_v


def _random_inputs(
    seed: int = 0, dtype: torch.dtype = torch.float64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random (q, k, v, alpha, beta) in the [B, H, T, *] convention, gates in (0, 1)."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, D_K, generator=gen, dtype=dtype)
    k = torch.randn(B, H, T, D_K, generator=gen, dtype=dtype)
    v = torch.randn(B, H, T, D_V, generator=gen, dtype=dtype)
    alpha = torch.rand(B, H, T, generator=gen, dtype=dtype)
    beta = torch.rand(B, H, T, generator=gen, dtype=dtype)
    return q, k, v, alpha, beta


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def _scan_with_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recurrent-scan path: fold gated_delta_rule_step over time (the decode contract)."""
    bsz, n_heads, seq_len, d_k = q.shape
    d_v = v.shape[-1]
    state = torch.zeros(bsz, n_heads, d_k, d_v, dtype=q.dtype)
    ys = []
    for t in range(seq_len):
        y_t, state = gated_delta_rule_step(
            state, q[:, :, t], k[:, :, t], v[:, :, t], alpha[:, :, t], beta[:, :, t]
        )
        ys.append(y_t)
    return torch.stack(ys, dim=2), state


# ---------------------------------------------------------------------------
# Three-way equivalence: chunkwise == recurrent-scan == float64 reference
# ---------------------------------------------------------------------------


def test_three_way_equivalence() -> None:
    q, k, v, alpha, beta = _random_inputs()
    y_ref, s_ref = gated_delta_rule_reference(q, k, v, alpha, beta)
    y_chunk, s_chunk = gated_delta_rule_chunkwise(q, k, v, alpha, beta, chunk_size=CFG.chunk_size)
    y_rec, s_rec = _scan_with_step(q, k, v, alpha, beta)

    assert _rel_err(y_chunk, y_ref) <= 1e-5
    assert _rel_err(y_rec, y_ref) <= 1e-5
    assert _rel_err(y_rec, y_chunk) <= 1e-5
    assert _rel_err(s_chunk, s_ref) <= 1e-5
    assert _rel_err(s_rec, s_ref) <= 1e-5


def test_chunk_size_independence() -> None:
    """State must pass correctly across chunk boundaries: any chunking, same function.

    chunk_size = T + 1 exercises the single-ragged-chunk edge; 16 and 64 exercise
    even and ragged-tail chunkings of T=96.
    """
    q, k, v, alpha, beta = _random_inputs(seed=1)
    outputs = [
        gated_delta_rule_chunkwise(q, k, v, alpha, beta, chunk_size=cs)
        for cs in (16, CFG.chunk_size, T + 1)
    ]
    y0, s0 = outputs[0]
    for y_i, s_i in outputs[1:]:
        # Same op sequence regardless of chunking on CPU float64 ⇒ bitwise identical.
        assert torch.equal(y_i, y0)
        assert torch.equal(s_i, s0)


# ---------------------------------------------------------------------------
# Causality (block level: projections are pointwise-in-time, recurrence is causal)
# ---------------------------------------------------------------------------


def test_causality_no_future_leak() -> None:
    torch.manual_seed(0)
    block = GatedDeltaNet(CFG).double()
    x = torch.randn(B, T, CFG.d_model, dtype=torch.float64)
    x_perturbed = x.clone()
    x_perturbed[:, T // 2] += 1.0
    with torch.no_grad():
        y = block(x)
        y_perturbed = block(x_perturbed)
    assert torch.equal(y[:, : T // 2], y_perturbed[:, : T // 2])
    assert not torch.equal(y[:, T // 2 :], y_perturbed[:, T // 2 :])


# ---------------------------------------------------------------------------
# Decode contract: teacher-forced step-by-step == chunkwise, token-exact
# ---------------------------------------------------------------------------


def test_decode_contract_token_exact() -> None:
    """The decode path the DELTA kernel later accelerates must reproduce training
    outputs token-exactly (float64) — otherwise train/serve drift is baked in."""
    torch.manual_seed(0)
    block = GatedDeltaNet(CFG).double()
    x = torch.randn(B, T, CFG.d_model, dtype=torch.float64)
    with torch.no_grad():
        q, k, v, alpha, beta = block.project(x)
        y_chunk, s_final = gated_delta_rule_chunkwise(
            q, k, v, alpha, beta, chunk_size=CFG.chunk_size
        )
        state = block.init_state(B, dtype=torch.float64)
        for t in range(T):
            y_t, state = block.step(
                state, q[:, :, t], k[:, :, t], v[:, :, t], alpha[:, :, t], beta[:, :, t]
            )
            assert torch.equal(y_t, y_chunk[:, :, t]), f"decode drift at t={t}"
    assert torch.equal(state.S, s_final)


# ---------------------------------------------------------------------------
# State memory: constant in T, and beats the KV cache at long context
# ---------------------------------------------------------------------------


def test_state_bytes_constant_in_t_and_beats_kv_cache() -> None:
    dtype = torch.float64
    sizes = []
    for seq_len in (32, 256):
        gen = torch.Generator().manual_seed(2)
        q = torch.randn(1, H, seq_len, D_K, generator=gen, dtype=dtype)
        k = torch.randn(1, H, seq_len, D_K, generator=gen, dtype=dtype)
        v = torch.randn(1, H, seq_len, D_V, generator=gen, dtype=dtype)
        alpha = torch.rand(1, H, seq_len, generator=gen, dtype=dtype)
        beta = torch.rand(1, H, seq_len, generator=gen, dtype=dtype)
        _, s_final = gated_delta_rule_chunkwise(q, k, v, alpha, beta, chunk_size=CFG.chunk_size)
        sizes.append(LinearAttnState(S=s_final).bytes())
    # Constant in T, and matches the analytic per-sequence size.
    assert sizes[0] == sizes[1] == linear_attn_state_bytes(CFG, dtype)

    # vs a GQA-8 KV cache at T=4096, same itemsize (bf16-class, 2 bytes):
    # KV bytes = 2 (K and V) · n_kv_heads · d_head · T · itemsize — GROWS with T;
    # the delta-rule state is n_heads · d_head · d_v · itemsize — CONSTANT.
    itemsize = torch.bfloat16.itemsize
    state_bytes = linear_attn_state_bytes(CFG, torch.bfloat16)
    kv_bytes = 2 * 8 * CFG.d_head * 4096 * itemsize
    print(
        f"\nlinear-attn state = {state_bytes} B vs GQA-8 KV @ T=4096 = {kv_bytes} B "
        f"(ratio {kv_bytes / state_bytes:.0f}x smaller)"
    )
    assert state_bytes < kv_bytes


# ---------------------------------------------------------------------------
# Gate limits: α ≡ 1 ⇒ plain delta rule; α ≡ 0 ⇒ memoryless
# ---------------------------------------------------------------------------


def test_alpha_one_reduces_to_plain_delta_rule() -> None:
    """With no decay the recurrence is the ungated delta rule
    S_t = S_{t-1} + β_t k_t (v_t − S_{t-1}ᵀ k_t)ᵀ — re-derived independently here."""
    q, k, v, _, beta = _random_inputs(seed=3)
    alpha = torch.ones(B, H, T, dtype=torch.float64)
    y_chunk, _ = gated_delta_rule_chunkwise(q, k, v, alpha, beta, chunk_size=CFG.chunk_size)

    s = torch.zeros(B, H, D_K, D_V, dtype=torch.float64)
    ys = []
    for t in range(T):
        k_t, v_t, q_t = k[:, :, t], v[:, :, t], q[:, :, t]
        b_t = beta[:, :, t].unsqueeze(-1)
        pred = torch.einsum("bhkv,bhk->bhv", s, k_t)
        s = s + torch.einsum("bhk,bhv->bhkv", k_t, b_t * (v_t - pred))
        ys.append(torch.einsum("bhkv,bhk->bhv", s, q_t))
    y_plain = torch.stack(ys, dim=2)
    torch.testing.assert_close(y_chunk, y_plain, rtol=1e-10, atol=1e-10)


def test_alpha_zero_is_memoryless() -> None:
    """α ≡ 0 wipes the state each step: S_t = β_t k_t v_tᵀ ⇒ y_t = β_t (q_t·k_t) v_t."""
    q, k, v, _, beta = _random_inputs(seed=4)
    alpha = torch.zeros(B, H, T, dtype=torch.float64)
    y_chunk, _ = gated_delta_rule_chunkwise(q, k, v, alpha, beta, chunk_size=CFG.chunk_size)
    y_expected = (beta * (q * k).sum(-1)).unsqueeze(-1) * v
    torch.testing.assert_close(y_chunk, y_expected, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Gradients flow through the chunkwise (training) path
# ---------------------------------------------------------------------------


def test_gradients_flow_to_all_parameters() -> None:
    torch.manual_seed(0)
    block = GatedDeltaNet(CFG).double()
    x = torch.randn(B, 32, CFG.d_model, dtype=torch.float64)
    block(x).sum().backward()
    for name, param in block.named_parameters():
        assert param.grad is not None, f"no grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"


def test_block_forward_shape() -> None:
    torch.manual_seed(0)
    block = GatedDeltaNet(CFG)
    x = torch.randn(B, 16, CFG.d_model)
    y = block(x)
    assert y.shape == (B, 16, CFG.d_model)
