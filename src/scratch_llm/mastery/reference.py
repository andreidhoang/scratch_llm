"""THE ORACLE.  This file is yours -- do not let an agent fill it in.

Everything in this repo is measured as a deviation from what this function
returns in float64.  You do not own a reading from an instrument you did not
build, and this is the instrument.

It is about five lines.  The point is not the typing; it is that you can write
the recurrence cold, from the equation, without looking anything up -- because
in ten weeks you will be arguing with vLLM maintainers about exactly this
recurrence and there is no version of that conversation where you check a file
first.

--------------------------------------------------------------------------
The equation (this is all you get, and it is all you need):

    S_t = alpha_t * (I - beta_t k_t k_t^T) S_{t-1}  +  beta_t k_t v_t^T
    o_t = q_t^T S_t

    S_0 = 0 (or the passed-in S0)
    alpha_t = exp(log_alpha_t)
    S is (dk, dv);  q, k are (T, dk);  v is (T, dv);  log_alpha, beta are (T,)
    o_t uses the state AFTER absorbing token t.

Two things worth deriving on paper before you type, because they are the
whole reason the chunked path exists:

  1. What is the arithmetic intensity of this loop at T=1 (decode)?
     FLOPs per token vs bytes per token.  Which wall are you under?
  2. Why can this loop not be used for training?  Answer in terms of the
     dependency chain, not "it is slow".

Write your predicted answers in LEDGER.md before you run anything.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import torch


def recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    beta: torch.Tensor,
    S0: torch.Tensor | None = None,
):
    """Literal sequential evaluation of the gated delta rule. YOUR IMPLEMENTATION.

    Returns
    -------
    O : (T, dv)   outputs
    S : (dk, dv)  final state

    Notes
    -----
    - Do NOT build the (dk, dk) matrix (I - beta k k^T) explicitly. Apply it.
      The whole point of the delta rule is that it is a rank-1 update; if you
      materialise the identity you have thrown away the only reason it is cheap,
      and the arithmetic-intensity answer you wrote down will be wrong.
    - Run in whatever dtype the inputs arrive in. The driver casts, not you.
    """
    # Written in pairing on 2026-09-03 to end a 22-day stall on five lines. The rep is still
    # available: delete this body and retype it cold from the equation above (10 minutes).
    # Rank-1 form, never materialising (I - beta k k^T):
    #   (I - b k k^T)(a S) = a S - b k (k^T (a S))
    T, dk = q.shape
    dv = v.shape[-1]
    S = torch.zeros(dk, dv, dtype=q.dtype, device=q.device) if S0 is None else S0.clone()
    O = torch.empty(T, dv, dtype=q.dtype, device=q.device)
    for t in range(T):
        k_t = k[t]
        S = torch.exp(log_alpha[t]) * S  # decay
        S = S - beta[t] * torch.outer(k_t, k_t @ S)  # erase (rank-1 apply)
        S = S + beta[t] * torch.outer(k_t, v[t])  # write
        O[t] = q[t] @ S  # read AFTER absorbing token t
    return O, S
