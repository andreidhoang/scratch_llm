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
    raise NotImplementedError(
        "reference.recurrent_reference is yours to write.\n"
        "Five lines. The equation is in the docstring above.\n"
        "Predict the two answers in LEDGER.md first, then implement, then run:\n"
        "    python -m experiments.e001_gate_sweep --self-test"
    )
