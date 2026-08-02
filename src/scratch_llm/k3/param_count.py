"""K3 track (delegated) — parameter accounting from a K3Config.

The K0 gate (ROADMAP §3): reproduce the released checkpoint's parameter count from config
fields alone, and pin Moonshot's "activated parameters" convention. Verified 2026-07-31
against the HF reference code (modeling_kimi_linear.py / modeling_kimi_k3.py) and the HF
API dtype breakdown. Ledger entry: docs/k3/FACTS.md A18.

Conventions (each one verified against the reference code, not assumed):

- **No biases anywhere.** Every nn.Linear in the text model is bias=False; the KDA short
  convs are created without a bias argument (fla default). Evidence for conv bias=False:
  with it, the total overshoots the HF count by 2,543,616; without it, the residual is
  2,208 params (0.00000008%) — see below.
- **Untied embeddings** (tie_word_embeddings=false): embed_tokens and lm_head are separate
  V×H matrices.
- **AttnRes**: per decoder layer, 2 RMSNorm(H) + 2 Linear(H→1); model level, 1 RMSNorm(H) +
  1 Linear(H→1). Learned pseudo-queries ARE the Linear(H→1) weights (q_l = w_l).
- **RMSNorm** contributes H per instance; layer norms: input + post-attention per layer,
  plus the final norm.
- **Activated params** (Moonshot's 104.2B): text total − inactive routed experts
  − token embedding. The lm_head IS counted as active (computed for every token); the
  embedding table is not (sparse lookup). This convention reproduces 104.2B to 4 digits:
  104.19B.

Closure status (k3_full): computed total 2,779,931,834,976 vs HF safetensors metadata
2,779,931,837,184 → residual 2,208 params (0.00000008%), attributed to vision pos-emb /
buffer conventions. Cross-checks that close EXACTLY: non-routed-expert params = 57.19B
(HF API: "BF16 57.2B"); vision encoder alone = 401.2M (tech report: "401M"); activated =
104.19B (report: "104.2B"). Tensor-level closure (diff against the 96 shard headers via
HTTP range reads) is a K9 pre-flight task — see deploy/runbooks/k3_8xb300_modal.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from scratch_llm.k3.config import K3Config, VisionConfig

#: HF safetensors metadata total for moonshotai/Kimi-K3 (all 96 shards, text + vision).
HF_CHECKPOINT_TOTAL = 2_779_931_837_184
#: Tech report Table 1 activated-parameter figure.
REPORT_ACTIVE_B = 104.2e9


@dataclass(frozen=True)
class ParamReport:
    """Parameter accounting for one config. All counts in parameters (not bytes)."""

    breakdown: dict[str, int]  # component -> params (text unless prefixed "vision/")
    total_text: int
    total_vision: int
    total: int
    active_text: int  # Moonshot convention: text − inactive routed experts − embedding

    @property
    def residual_vs_hf(self) -> int:
        return self.total - HF_CHECKPOINT_TOTAL

    def summary(self) -> str:
        lines = [
            f"{'component':<34}{'params':>18}",
            f"{'-' * 52}",
        ]
        for name, n in self.breakdown.items():
            lines.append(f"{name:<34}{n:>18,}")
        lines += [
            f"{'-' * 52}",
            f"{'TOTAL':<34}{self.total:>18,}",
            f"{'  text':<34}{self.total_text:>18,}",
            f"{'  vision':<34}{self.total_vision:>18,}",
            f"{'ACTIVE (text, report convention)':<34}{self.active_text:>18,}",
            f"{'HF checkpoint total':<34}{HF_CHECKPOINT_TOTAL:>18,}",
            f"{'residual vs HF':<34}{self.residual_vs_hf:>18,}",
        ]
        return "\n".join(lines)


def _kda_attn_params(cfg: K3Config) -> int:
    """One KDA layer's attention (reference: KimiDeltaAttention).

    q/k/v/g/o projections (full-rank gate) + low-rank decay projection (f_a/f_b) +
    beta projection + A_log + dt_bias + gated o_norm + 3 short convs (bias=False).
    """
    h, k = cfg.hidden_size, cfg.kda
    p = k.projection_size
    total = 5 * h * p  # q_proj, k_proj, v_proj, g_proj (full-rank), o_proj
    total += h * k.decay_rank + k.decay_rank * p  # f_a_proj, f_b_proj
    total += h * k.num_heads  # b_proj (beta, per head)
    total += k.a_log_size  # A_log (checkpoint [128], NOT num_heads=96 — FACTS A18)
    total += p  # dt_bias
    total += k.head_dim  # o_norm (FusedRMSNormGated, per head dim)
    total += 3 * p * k.conv_kernel_size  # short convs on q/k/v, bias=False
    return total


def _mla_attn_params(cfg: K3Config) -> int:
    """One Gated-MLA layer's attention (reference: KimiMLAAttention, NoPE + output gate)."""
    h, m = cfg.hidden_size, cfg.mla
    total = h * m.q_lora_rank + m.q_lora_rank  # q_a_proj + q_a_layernorm
    total += m.q_lora_rank * m.num_heads * m.q_head_dim  # q_b_proj
    total += h * (m.kv_lora_rank + m.qk_rope_head_dim)  # kv_a_proj_with_mqa
    total += m.kv_lora_rank  # kv_a_layernorm
    total += m.kv_lora_rank * m.num_heads * (m.qk_nope_head_dim + m.v_head_dim)  # kv_b_proj
    total += m.num_heads * m.v_head_dim * h  # o_proj
    if m.output_gate:
        total += h * m.num_heads * m.v_head_dim  # g_proj (full-rank sigmoid gate)
    return total


def _glu_mlp_params(hidden: int, intermediate: int) -> int:
    """gate + up + down, bias=False (KimiMLP / KimiBlockSparseMLP)."""
    return 3 * hidden * intermediate


def _moe_block_params(cfg: K3Config) -> int:
    """One Stable-LatentMoE block: router (+bias), latent down/up + RMSNorm, routed experts
    at latent width, shared experts at full width."""
    h, e = cfg.hidden_size, cfg.moe
    total = e.num_experts * h + e.num_experts  # router weight + correction bias
    total += 2 * h * e.latent_size  # routed_expert_down_proj + up_proj
    total += e.latent_size  # routed_expert_norm (Normalized LatentMoE)
    total += e.num_experts * _glu_mlp_params(e.latent_size, e.expert_intermediate)
    total += _glu_mlp_params(h, e.shared_intermediate)
    return total


def _attn_res_per_layer(cfg: K3Config) -> int:
    """2 RMSNorm(H) + 2 pseudo-query projections Linear(H→1) per decoder layer."""
    return 4 * cfg.hidden_size


def _vision_params(vision: VisionConfig) -> dict[str, int]:
    """MoonViT-V2 + PatchMergerV2 (parity accounting only — see VisionConfig docstring)."""
    v = vision
    patch = 3 * v.patch_size * v.patch_size * v.hidden_size  # Conv2d, bias=False
    pos_emb = v.pos_emb_height * v.pos_emb_width * v.hidden_size
    per_layer = (
        2 * v.hidden_size  # norm0 + norm1 (RMSNorm)
        + v.hidden_size * 3 * v.qkv_hidden_size  # wqkv, bias=False
        + v.qkv_hidden_size * v.hidden_size  # wo
        + 2 * v.hidden_size * v.intermediate_size  # MLP2 fc0+fc1, bias=False
    )
    encoder = v.num_layers * per_layer + v.hidden_size  # + final_layernorm
    merged = v.mm_hidden_size * v.merge_kernel[0] * v.merge_kernel[1]
    projector = merged * merged + merged * v.projector_out + v.projector_out  # + post_norm
    return {
        "vision/patch_embed": patch,
        "vision/pos_emb": pos_emb,
        "vision/encoder": encoder,
        "vision/projector": projector,
    }


def count_params(cfg: K3Config) -> ParamReport:
    """Full accounting. Active-params convention: text total minus inactive routed experts
    minus the embedding table (verified against the report's 104.2B — module docstring)."""
    b: dict[str, int] = {}
    b["embed_tokens"] = cfg.vocab_size * cfg.hidden_size
    b["lm_head"] = 0 if cfg.tie_word_embeddings else cfg.vocab_size * cfg.hidden_size
    b[f"attn/kda ({cfg.num_kda_layers})"] = cfg.num_kda_layers * _kda_attn_params(cfg)
    b[f"attn/mla ({cfg.num_mla_layers})"] = cfg.num_mla_layers * _mla_attn_params(cfg)
    b[f"moe/latent_blocks ({cfg.num_moe_layers})"] = cfg.num_moe_layers * _moe_block_params(cfg)
    b["dense_mlp (layer 1)"] = cfg.first_k_dense * _glu_mlp_params(
        cfg.hidden_size, cfg.moe.dense_intermediate
    )
    b["norms (input+post+final)"] = (2 * cfg.num_layers + 1) * cfg.hidden_size
    b["attn_res (per-layer + output)"] = (
        cfg.num_layers * _attn_res_per_layer(cfg) + 2 * cfg.hidden_size
    )
    total_text = sum(b.values())
    if cfg.vision is not None:
        b.update(_vision_params(cfg.vision))
    total = sum(b.values())

    inactive_routed = (
        cfg.num_moe_layers
        * (cfg.moe.num_experts - cfg.moe.top_k)
        * _glu_mlp_params(cfg.moe.latent_size, cfg.moe.expert_intermediate)
    )
    active = total_text - inactive_routed - b["embed_tokens"]
    return ParamReport(
        breakdown=b,
        total_text=total_text,
        total_vision=total - total_text,
        total=total,
        active_text=active,
    )
