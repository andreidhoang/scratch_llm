"""k3.core — HAND-BUILT territory (see ../HANDCRAFTED.md).

Every module in this package is authored by the human, line by line, from primary sources.
Agents are READ-ONLY here: adversarial tests and written proposals live outside this
package; nothing agent-authored lands in `core/`. Mastery bar per module: the delete test —
`rm` the file, rewrite it from its own derivation docstring.

Build order (serial — do not parallelize hand-builds):
1. `situ.py`        — SiTU-GLU warmup (β1=4 gate / β2=25 up, |f| ≤ 100)      [K5 prep]
2. `kda.py`         — Kimi Delta Attention: the critical path (K2 rung)
3. `gated_mla.py`   — MLA + full-rank output gate, NoPE                      [K3 rung]
4. `latent_moe.py`  — sigmoid router + Quantile Balancing at 0.5× latent     [K5 rung]
5. `attn_res.py`    — Block AttnRes, learned pseudo-queries                  [K4 rung]
"""
