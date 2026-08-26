"""The two computational paths for the gated delta rule.

CONVENTION (fixed once, everything downstream assumes it)
---------------------------------------------------------
State is a MATRIX  S_t in R^{dk x dv}  -- an associative memory, not a vector.

    S_t = alpha_t * (I - beta_t k_t k_t^T) S_{t-1}  +  beta_t k_t v_t^T
    o_t = q_t^T S_t                    <-- state AFTER absorbing token t

    alpha_t in (0,1]   the gate.  log_alpha_t = 0  <=>  alpha_t = 1  <=>  "forget nothing"
    beta_t             write strength
    (I - beta k k^T)   the delta rule: a rank-1 eraser that removes whatever was
                       already stored at key k before writing the new value

Shapes throughout:  q, k -> (T, dk)   v -> (T, dv)   log_alpha, beta -> (T,)

`recurrent_reference` (in reference.py) computes this literally, one token at a time.
`chunked_wy` (here) computes the SAME thing with matmuls, the way a training kernel
must, because a sequential loop over 32k tokens has no parallelism and no tensor cores.

They are algebraically identical and numerically different. Measuring that difference
is the point of this repo.
"""

from __future__ import annotations

from typing import Literal, overload

import torch


# ---------------------------------------------------------------------------
# THE THING UNDER TEST.  This mirrors what FLA's `chunk_gated_delta_rule` does
# structurally: reduce the chunk to matmuls via the WY representation, which
# requires inverting a CxC triangular matrix.  Written here in pure PyTorch so
# the MECHANISM is separable from any particular kernel implementation.
# ---------------------------------------------------------------------------
@overload
def chunked_wy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = ...,
    S0: torch.Tensor | None = ...,
    *,
    return_diagnostics: Literal[True],
    solve_dtype: torch.dtype | None = ...,
) -> tuple[torch.Tensor, torch.Tensor, dict]: ...


@overload
def chunked_wy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = ...,
    S0: torch.Tensor | None = ...,
    return_diagnostics: Literal[False] = ...,
    solve_dtype: torch.dtype | None = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...


# The two @overload stubs above are TYPE-ONLY: they tell the checker that the return arity is
# decided by `return_diagnostics`, which it cannot infer from the conditional return on its own.
# The implementation signature and body below are unchanged.
def chunked_wy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    S0: torch.Tensor | None = None,
    return_diagnostics: bool = False,
    solve_dtype: torch.dtype | None = None,
):
    """Chunked (WY-form) evaluation of the gated delta rule.

    Derivation, so every line is defensible:

    1.  alpha_t is a SCALAR, so it factors straight out of the transition product:
            prod_t alpha_t (I - beta_t k_t k_t^T) = (prod_t alpha_t) prod_t (I - beta_t k_t k_t^T)

    2.  Substituting S_t = a_t * Shat_t, where a_t = prod_{i<=t} alpha_i, kills the
        gate from the homogeneous part and pushes it onto the write strength:
            Shat_t = (I - beta_t k_t k_t^T) Shat_{t-1} + (beta_t / a_t) k_t v_t^T
        ^ note beta_t / a_t.  This term is hypothesis H2's hazard: as the gate decays,
          1/a_t grows geometrically inside the chunk.

    3.  Writing Shat_t = Shat_0 + sum_{j<=t} k_j u_j^T and solving for U gives
            (I + tril(B K K^T, -1)) U = Bhat V - B K Shat_0
        i.e. a CxC TRIANGULAR SOLVE.  Its conditioning is hypothesis H1's hazard.

    4.  Outputs come out as two matmuls:
            O = Qhat Shat_0 + tril(Qhat K^T, 0) U          (Qhat_t = a_t q_t)

    `solve_dtype` -- the precision of step 3's triangular solve, held SEPARATE from
    the matmul precision on purpose. Production kernels (FLA included) run the WY
    transform in fp32 while feeding bf16 to the tensor cores, so the honest default
    for a bf16/fp16 sweep is fp32. Sweeping this parameter is itself an experiment:
    if forcing the solve to fp32 collapses the divergence, you have just found the
    fix, and the cost of that fix is the determinism tax you will publish.
    """
    T, dk = k.shape
    dv = v.shape[1]
    dtype, device = q.dtype, q.device
    if solve_dtype is None:
        solve_dtype = torch.float32 if dtype in (torch.bfloat16, torch.float16) else dtype

    S = torch.zeros(dk, dv, dtype=dtype, device=device) if S0 is None else S0.clone()
    O = torch.empty(T, dv, dtype=dtype, device=device)
    diag = {"cond_T": [], "max_inv_a": [], "chunks": 0}

    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        C = e - s
        qc, kc, vc = q[s:e], k[s:e], v[s:e]
        la, bc = log_alpha[s:e], beta[s:e]

        # a_t = prod_{i<=t} alpha_i, computed in log space (as every real kernel does)
        a = torch.exp(torch.cumsum(la, dim=0))  # (C,)
        b_hat = bc / a  # step 2: the rescaled write strength

        B = bc.unsqueeze(1)  # (C,1)
        # Step 3: the CxC triangular system.  strictly-lower so it is unit-triangular.
        A = torch.tril(B * (kc @ kc.transpose(0, 1)), diagonal=-1)
        I = torch.eye(C, dtype=solve_dtype, device=device)
        rhs = b_hat.unsqueeze(1) * vc - B * (kc @ S)  # (C, dv)
        U = torch.linalg.solve_triangular(
            I + A.to(solve_dtype),
            rhs.to(solve_dtype),
            upper=False,
            unitriangular=True,
        ).to(dtype)

        # Step 4: outputs, two matmuls.  diagonal=0 -> o_t sees the update at t.
        q_hat = a.unsqueeze(1) * qc
        O[s:e] = q_hat @ S + torch.tril(q_hat @ kc.transpose(0, 1), diagonal=0) @ U

        # carry: S_C = a_C * Shat_C
        S = a[-1] * (S + kc.transpose(0, 1) @ U)

        if return_diagnostics:
            M = (I + A).to(torch.float64)
            diag["cond_T"].append(torch.linalg.cond(M).item())
            diag["max_inv_a"].append((1.0 / a.to(torch.float64)).max().item())
            diag["chunks"] += 1

    return (O, S, diag) if return_diagnostics else (O, S)


# ---------------------------------------------------------------------------
# A DELIBERATELY DIFFERENT recurrence -- NOT the delta rule.  Used only to smoke
# test the sweep driver end to end before reference.py is filled in.  Plain
# gated linear attention: no eraser term, so no triangular solve, so it should
# show ~machine-eps agreement between paths at every gate value.  If the driver
# reports anything else for THIS, the driver is broken, not the math.
# ---------------------------------------------------------------------------
def mock_linear_attention(q, k, v, log_alpha, beta, chunk_size=64, S0=None):
    T, dk = k.shape
    dv = v.shape[1]
    S = torch.zeros(dk, dv, dtype=q.dtype, device=q.device) if S0 is None else S0.clone()
    O = torch.empty(T, dv, dtype=q.dtype, device=q.device)
    for t in range(T):
        S = torch.exp(log_alpha[t]) * S + beta[t] * torch.outer(k[t], v[t])
        O[t] = S.transpose(0, 1) @ q[t]
    return O, S
