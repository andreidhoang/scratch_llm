"""Validates the WY algebra in paths.chunked_wy -- NOT a reference for reference.py.

|  SEAL NOTE.  The helper below materialises the full (dk, dk) transition matrix
|  and multiplies it out.  That is the ANTI-PATTERN reference.py's docstring
|  explicitly tells you not to write: it is O(dk^2) per token, it throws away the
|  rank-1 structure, and it is unusable as a decode oracle.  It exists here only
|  to check a linear-algebra identity.  Reading it will not hand you the answer
|  to reference.py -- it will hand you the wrong shape of the answer.
"""

import torch

from scratch_llm.mastery.paths import chunked_wy


def _explicit_transition_product(q, k, v, log_alpha, beta, S0=None):
    """Deliberately naive: build (I - beta k k^T) as a dense matrix, apply it."""
    T, dk = k.shape
    dv = v.shape[1]
    S = torch.zeros(dk, dv, dtype=q.dtype) if S0 is None else S0.clone()
    O = torch.empty(T, dv, dtype=q.dtype)
    I = torch.eye(dk, dtype=q.dtype)
    for t in range(T):
        M = I - beta[t] * torch.outer(k[t], k[t])  # dense, dk x dk
        S = torch.exp(log_alpha[t]) * (M @ S) + beta[t] * torch.outer(k[t], v[t])
        O[t] = S.transpose(0, 1) @ q[t]
    return O, S


def _inputs(T=256, dk=32, dv=32, log_gate=-0.05, seed=7):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(T, dk, generator=g, dtype=torch.float64)
    k = torch.randn(T, dk, generator=g, dtype=torch.float64)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(T, dv, generator=g, dtype=torch.float64)
    beta = torch.sigmoid(torch.randn(T, generator=g, dtype=torch.float64))
    la = torch.full((T,), float(log_gate), dtype=torch.float64)
    return q, k, v, la, beta


def test_wy_matches_explicit_product_fp64():
    """In fp64 the two formulations must agree to near machine precision.
    If this fails the WY derivation is wrong, and every divergence number the
    sweep produces is measuring a bug instead of a mechanism."""
    for lg in (0.0, -0.01, -0.1, -0.5):
        for C in (16, 64, 128):
            q, k, v, la, b = _inputs(log_gate=lg)
            O_wy, S_wy = chunked_wy(q, k, v, la, b, chunk_size=C)
            O_ex, S_ex = _explicit_transition_product(q, k, v, la, b)
            rel = (O_wy - O_ex).norm() / O_ex.norm()
            assert rel < 1e-10, f"log_gate={lg} C={C}: WY identity broken, rel={rel:.3e}"
            rs = (S_wy - S_ex).norm() / S_ex.norm()
            assert rs < 1e-10, f"log_gate={lg} C={C}: state mismatch, rel={rs:.3e}"


def test_chunk_size_invariance_fp64():
    """Chunk size is an implementation detail. In fp64 it must not move the answer."""
    q, k, v, la, b = _inputs(log_gate=-0.02)
    ref, _ = chunked_wy(q, k, v, la, b, chunk_size=256)
    for C in (8, 32, 64, 128):
        O, _ = chunked_wy(q, k, v, la, b, chunk_size=C)
        rel = (O - ref).norm() / ref.norm()
        assert rel < 1e-10, f"C={C} changed the fp64 answer, rel={rel:.3e}"


if __name__ == "__main__":
    test_wy_matches_explicit_product_fp64()
    test_chunk_size_invariance_fp64()
    print("WY identity: OK")
