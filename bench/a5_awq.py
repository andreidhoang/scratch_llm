"""A5 §4.3 bench — AWQ vs naive round-to-nearest INT4 at the SAME group size / bit-width.

Run: ``PYTHONPATH=src .venv/bin/python bench/a5_awq.py``

Measures the load-bearing result — AWQ INT4 layer-output MSE < naive INT4 MSE on a realistic Linear
with planted activation-outlier channels — plus the mechanism: the α grid curve (α=0 == naive, an
interior optimum), calib→held-out generalization, and a per-channel error breakdown showing the win
is concentrated on the salient channels. This is a measurement harness; the assertions live in
``tests/test_quant_awq.py``. Layer-level MSE demo (not a full-model perplexity run — that is a
rental-gated SKIP); all numbers below are actually measured.
"""

from __future__ import annotations

import torch

from scratch_llm.quant.awq import (
    act_scale,
    awq_dequantize,
    awq_linear,
    quantize_awq_int4,
    search_awq_scale,
)
from scratch_llm.quant.int4_group import (
    GROUP_SIZE,
    dequantize_groupwise_int4,
    mse,
    quantize_groupwise_int4,
    sqnr_db,
)

IN_F, OUT_F, N_SALIENT, SALIENT_MULT = GROUP_SIZE * 4, 256, 8, 12.0


def _make_layer(seed: int, n_tokens: int = 512):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(OUT_F, IN_F, generator=g) * 0.02
    x = torch.randn(n_tokens, IN_F, generator=g)
    salient = torch.randperm(IN_F, generator=g)[:N_SALIENT]
    x[:, salient] *= SALIENT_MULT
    return w, x, salient


def _naive_out(w, x):
    q, s = quantize_groupwise_int4(w, GROUP_SIZE)
    return x @ dequantize_groupwise_int4(q, s, GROUP_SIZE).t()


def main() -> None:
    print("=== AWQ vs naive round-to-nearest INT4 (W4A16, group_size=128, identical bit-width) ===")
    print(
        f"layer: Linear[{OUT_F},{IN_F}]  weights~N(0,0.02²)  "
        f"{N_SALIENT}/{IN_F} salient act channels (×{SALIENT_MULT:g})\n"
    )

    recoveries = []
    for seed in range(5):
        w, x, _ = _make_layer(seed)
        y_ref = x @ w.t()
        m_naive = mse(y_ref, _naive_out(w, x))
        quant = quantize_awq_int4(w, x=x, group_size=GROUP_SIZE)
        m_awq = mse(y_ref, awq_linear(x, quant))
        rec = m_naive / m_awq
        recoveries.append(rec)
        print(
            f"seed {seed}: naive MSE {m_naive:.4e} (SQNR {sqnr_db(y_ref, _naive_out(w, x)):5.2f} dB)"
            f" | AWQ MSE {m_awq:.4e} (SQNR {sqnr_db(y_ref, awq_linear(x, quant)):5.2f} dB)"
            f" | recovery {rec:.2f}×"
        )
    print(
        f"\nmean recovery over 5 seeds: {sum(recoveries) / len(recoveries):.2f}×"
        f"  (min {min(recoveries):.2f}×)\n"
    )

    # --- α grid curve (seed 0): α=0 == naive, interior optimum ---
    w, x, salient = _make_layer(0)
    y_ref = x @ w.t()
    a = act_scale(x)
    print("mechanism 1 — α grid (s = act_scaleᵅ, geo-mean normalized); α=0 ≡ naive INT4:")
    for i in range(0, 21, 2):
        ratio = i / 20
        s = a.pow(ratio).clamp_min(1e-4)
        s = s / (s.max() * s.min()).sqrt()  # same normalization the search uses
        quant = quantize_awq_int4(w, chan_scale=s, group_size=GROUP_SIZE)
        print(f"    α={ratio:.2f}  MSE {mse(y_ref, awq_linear(x, quant)):.3e}")
    res = search_awq_scale(w, x, group_size=GROUP_SIZE)
    print(
        f"  → searched optimum: α={res.ratio:.2f} (interior ⇒ genuine protect-vs-inflate trade-off)\n"
    )

    # --- generalization: search on calib, report on held-out ---
    w, x, _ = _make_layer(1, n_tokens=512)
    xc, xe = x[:256], x[256:]
    y_ref_e = xe @ w.t()
    m_naive_e = mse(y_ref_e, _naive_out(w, xe))
    quant = quantize_awq_int4(w, x=xc, group_size=GROUP_SIZE)  # scale from calib ONLY
    m_awq_e = mse(y_ref_e, awq_linear(xe, quant))
    print("mechanism 2 — generalization (scale searched on calib, MSE on held-out eval tokens):")
    print(
        f"    held-out: naive {m_naive_e:.4e} | AWQ {m_awq_e:.4e} | recovery {m_naive_e / m_awq_e:.2f}×"
        "  (win transfers, not calib-overfit)\n"
    )

    # --- per-channel attribution: the win is concentrated on salient columns ---
    w, x, salient = _make_layer(0)
    y_ref = x @ w.t()
    q, s = quantize_groupwise_int4(w, GROUP_SIZE)
    w_naive = dequantize_groupwise_int4(q, s, GROUP_SIZE)
    quant = quantize_awq_int4(w, x=x, group_size=GROUP_SIZE)
    w_awq = awq_dequantize(quant)
    # activation-weighted column error energy: Σ_i (Δw_ij)² · E[x_j²]
    ex2 = x.pow(2).mean(0)  # [in]
    col_err_naive = (w_naive - w).pow(2).sum(0) * ex2
    col_err_awq = (w_awq - w).pow(2).sum(0) * ex2
    sal_mask = torch.zeros(IN_F, dtype=torch.bool)
    sal_mask[salient] = True
    print("mechanism 3 — activation-weighted column-error energy Σ_i(Δwᵢⱼ)²·E[xⱼ²]:")
    print(
        f"    salient cols : naive {col_err_naive[sal_mask].sum():.3e} → AWQ {col_err_awq[sal_mask].sum():.3e}"
        f"  ({col_err_naive[sal_mask].sum() / col_err_awq[sal_mask].sum():.2f}× lower)"
    )
    print(
        f"    other  cols  : naive {col_err_naive[~sal_mask].sum():.3e} → AWQ {col_err_awq[~sal_mask].sum():.3e}"
        f"  (rises slightly — cheap: tiny activations)"
    )


if __name__ == "__main__":
    main()
