"""k3.core — the K3 mechanisms. Live status and next steps: ``scratch_llm/k3/__init__.py``.

No file-ownership boundary (removed 2026-09-22, ``../AGENTS.md``): anyone may implement any
module here. Build order (later modules depend on earlier ones):
1. situ.py        — SiTU-GLU (β1=4 gate / β2=25 up, |f| ≤ 100)                      ✅ built
2. kda.py         — Kimi Delta Attention — critical path                             🟠 FLA tol open
3. gated_mla.py   — MLA, NoPE with live unrotated rope dims, sigmoid gate pre-o_proj ✅ built
4. latent_moe.py  — sigmoid router at full width + Quantile Balancing, 0.5× latent   ✅ built
5. attn_res.py    — Block AttnRes before every sublayer, learned pseudo-queries      ✅ built
   norm.py        — the one K3 RMSNorm (HF KimiRMSNorm op order), used by all of the above

Gate per module is its test file ``tests/test_k3_<module>.py`` (parity vs an in-file transcription
of Moonshot's HF reference, fp64 path equivalence, planted-bug mutations must fail) — not who
wrote it.
"""
