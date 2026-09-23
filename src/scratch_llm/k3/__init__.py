"""scratch_llm.k3 — build & host Kimi K3 from scratch.

This docstring is the live state ledger of the rebuild: update it whenever a module changes status.
Background spec: ``docs/k3/FACTS.md`` and ``docs/k3/ROADMAP.md`` (historical — where they disagree
with the oracles below, the oracles win; known errors are listed under DOC DEBT). No file-ownership
boundary on ``core/`` (removed 2026-09-22, ``AGENTS.md``): the gates are evidence, not authorship.

ORACLES (primary first)
-----------------------
  1. Moonshot's HF reference for the released checkpoint — ``../ladders/oss/kimi_k3_hf/``
     (``modeling_kimi_linear.py``, ``config.json``; fetched 2026-09-22). Tests carry in-file
     transcriptions of it and never import from there.
  2. vLLM @ dedcfa4483 ``vllm/models/kimi_k3/`` — serves the real checkpoint.
  3. FLA @ 5f68edb — ``fla/ops/kda``, ``fla/ops/attnres/naive.py``, ``fla/models/kda``.
  4. ``param_count.py`` — exact 2,779,931,837,184 closure; the shape oracle for every module.

STATUS (2026-09-22; counts from ``pytest tests/test_k3_<name>.py``)
------------------------------------------------------------------
  Rung  Module                Status      Evidence
  ────  ────────────────────  ──────────  ────────────────────────────────────────────────────────
  K0    config.py             ✅ done      param_count closure; + routed_scaling_factor=1.0
  K0    param_count.py        ✅ done      9/9 exact-equality tests
  K5p   core/situ.py          ✅ built     16/16: exact |f|≤100 bound, bitwise ≡ HF SituAndMul
  K2    core/kda.py           🟠 built     13/13 local + FLA parity 5/4: the 4 skips wait for
                                          HUY_V0_REL_TOL (measured 0.9-7e-7 vs FLA)
  K3    core/gated_mla.py     ✅ built     22/22: ≡ HF KimiMLAAttention fp64 1e-10 (rope 0 and >0),
                                          absorbed ≡ naive (outputs + grads), prefill ≡ decode,
                                          QK-Clip observer = running max over training forwards
  K5    core/latent_moe.py    ✅ built     26/26: ≡ HF gate + sparse block fp64, QB → load k/n,
                                          one-step delay, histogram ≡ exact quantile
  K4    core/attn_res.py      ✅ built     45/45: ≡ HF _apply_attn_res and FLA naive_attnres,
                                          bank/prefix ≡ naive re-derivation ≡ HF loop (+ grads)
        core/norm.py          ✅ built     the one K3 RMSNorm (HF KimiRMSNorm op order, fp64 kept)
  K6    muon.py               ✅ built     20/20: head_rows=None ≡ optim.Muon, per-block NS,
                                          MuonClip math (MLA only, τ=100)
  K6    model.py              ✅ built     18/18: whole model ≡ HF KimiLinearModel transcription
                                          (fp64 1e-10, logits + every grad, KDA leg independent),
                                          prefill ≡ decode ≡ split, causality, loss-at-init band,
                                          overfit < 1e-2, params == count_params − A_log tail
                                          (mini 370,794,512; full 2,779,484,476,000 on meta)
        train.py / speedrun   ✅ wired     test_k3_train 8/8: train(optimizer="k3_muon") with QK-Clip
                                          + QB after step, K3 checkpoints, model_family="k3_mini"
        checkpoint.py         ✅ built     20/20 + network gate 1/1 (run by hand): HF names →
                                          K3, MXFP4 expert dequant, strict LoadReport
        end to end            ✅ wired     test_k3_e2e 23/23: run_speedrun(model_family="k3_mini")
                                          tokenizer → pretrain → SFT → eval → sample → chat;
                                          cached decode (HybridState) ≡ uncached fp64 1e-10, exact
                                          logprobs, QB once per SFT step, bitwise resume
        vision (MoonViT-V2)   ⬜ reading   params accounted exactly (447,358,976 = shard headers);
                                          loader skips the 168 tensors — see NEXT 2
  Every row was adversarially reviewed (planted-bug mutations must fail a test) and is ruff +
  whole-repo pyright clean; CI selection green. Real mini-K3 smoke (fp32 CPU): builds in 6 s,
  (1,12) prefill 1.6 s, decode ≡ prefill to 6e-6 (fp32 noise), CE at init 10.67 vs 10.60 predicted.
  "Built" ≠ trained: no training run of any size has produced a quality claim. K3 chat through
  ``chat_cli.main``/``batch_reply`` is still dense-only (checkpoint family probe + batched decode).

NEXT (in order)
---------------
  1. Huy: set ``HUY_V0_REL_TOL`` (closes K2 at 9/9) — the one decision reserved to him (below).
  2. Vision, if Huy opens the scope (SIXTY_DAYS:52/180 keep it text-only until he asks): a
     checkpoint-parity, image-only, inference MoonViT-V2 + PatchMergerV2 + processor. It is the one
     part of K3 checkable against the REAL weights on this Mac (447M). Traps settled 2026-09-22:
     RMSNorm eps=None is fp32 eps on torch 2.12 (measured bitwise), ViT GELU-tanh vs projector
     GELU-erf (config fields unread), 2D RoPE x/y interleave (HF docstring contradicts its code),
     merge order TL/TR/BL/BR, pad-then-normalize (pad = −1), one placeholder → (H/28)·(W/28) rows.
  3. GPU host (A2): hybrid mini-K3 vs a matched dense control at equal parameter budget — the only
     route to a quality claim. Known CPU costs that do not transfer: depthwise conv ≈ 80% of the
     CPU forward; bf16 Newton–Schulz on CPU makes a PerHeadMuon step at mini scale take minutes.
  4. Distributed K3: the QBStats all-reduce (documented on ``latent_moe.QBStats``) and a ZeRO-2
     PerHeadMuon are not wired; ``train()`` refuses distributed K3.
  5. Loader at scale: no lazy per-shard path or safetensors reader in the library (callers pass
     ``(name, tensor)`` pairs); a full K3 load is a reference path, not a Mac workload.
  Out of scope so far: MTP layer, vision tower (params accounted only), MXFP4 QAT.

OPEN — HUY ONLY
---------------
  ``HUY_V0_REL_TOL`` (tests/test_kda_parity_fla.py) — the campaign's A1a exit is "a tolerance Huy
  argues in one line", so it stays his. Measured input for that line: FLA naive computes in fp32;
  the longest rounding chain is T·K ≤ 128·32 = 4096 ops, random-walk error sqrt(4096)·2^-24 ≈
  3.8e-6; measured v0-vs-FLA 0.9-7e-7; the three plausible algebra slips (read before absorb,
  erase after write, gate after write) measure ≥ 0.44 relative. 1e-5 would separate them by 4.6
  decades.

DECISIONS (2026-09-22, lead agent on Huy's explicit delegation — each is one line to revert)
-------------------------------------------------------------------------------------------
  1. mini ``qk_rope_head_dim`` 0 → 32: K3's head-shared unrotated key part is live (64 of 192
     score dims), so the miniature keeps it at the full-scale ratio nope:rope = 2:1. Mini grows by
     491,520 params to 370,794,944 (test_k3_param_count pins it); no A2 run had used the old preset.
  2. AttnRes pseudo-query init: HF normal(0, 0.02) (Moonshot's code + census: layer-0 query ≈
     4e-6, which zero-init + weight decay could not produce); paper/FLA zero init stays a kwarg.
  3. Quantile Balancing α_i from ``s + b`` (the cutoff lives in routing-score space; convergence
     to k/n is gated); 512 histogram bins.
  4. Per-head Muon: KDA q/k/v partitioned into head_dim-row blocks (report: "partitioned Q/K/V"),
     block-shape RMS scale; one lr for Muon and AdamW under ``k3_muon`` (torchtitan K2.7:
     RMS-matched Muon). The mini lr 3e-3 is untuned — an A2 sweep item, not a decision.
  5. Loss-at-init gate = derived log V + σ²/2 − (e^{σ²}−1)/(2V) with a derived band.
  6. MLA latent-norm eps: HF 1e-6 (the checkpoint's reference) over vLLM's 1e-5.
  7. QK-Clip observer: ``last_max_logits`` is a running max over training-mode forwards since the
     last clip (torchtitan's max over microbatches); eval forwards never record. Update order in
     ``train()``: weights → QK-Clip → QB bias.
  8. Checkpoint policy: E8M0 code 0 → 2^-127 (OCP MX); a dequantized value the target dtype cannot
     hold raises; F32 checkpoint families must land in fp32 params; MTP layers skipped; HF's
     embedding ``padding_idx`` not modeled (our vocab has no pad id).
  Side finding, no action: dt_bias census −4.63 ± 0.05 ≈ the mean of FLA's init distribution —
  FACTS A19(d)'s "learned retention" reading may just be the init.

DOC DEBT (historical docs, left unedited on purpose)
---------------------------------------------------
  FACTS A11 / ROADMAP call the 64 rope dims vestigial (live, unrotated); ROADMAP K3 counts MLA KV
  bytes as 512+0 (it is 512+64); ROADMAP K4 says param overhead L·d (it is 4L·d + 2d) and AttnRes
  scale 1/√d (none); the old scaffolds merged AttnRes once after the block loop (it runs before
  every sublayer) and put the MLA gate after W_o with width H→H (before o_proj, H→heads·v); FLA's
  attnres_block_size counts sublayers, K3's counts layers.

PUBLIC API
----------
    from scratch_llm.k3 import k3_full, mini_k3_d12                  # config presets
    from scratch_llm.k3.model import build_k3, build_mini_k3, HybridState
    from scratch_llm.k3.param_count import count_params              # 2.78T closure
    from scratch_llm.k3.muon import build_k3_optimizer, apply_k3_qk_clip
    from scratch_llm.k3.checkpoint import load_hf_checkpoint         # HF names + MXFP4 → K3Model
    from scratch_llm.train import train, build_k3_from_checkpoint    # TrainConfig(optimizer="k3_muon")
"""

from scratch_llm.k3.config import K3Config, k3_full, mini_k3_d12

__all__ = ["K3Config", "k3_full", "mini_k3_d12"]
