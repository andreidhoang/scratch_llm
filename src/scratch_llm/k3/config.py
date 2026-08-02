"""K3 track (delegated) — Kimi K3 configuration of record + mini-K3 presets.

Every full-scale value is lifted verbatim from the released checkpoint's ``config.json``
(huggingface.co/moonshotai/Kimi-K3, 2026-07-27) and cross-checked against the tech report
(arXiv:2607.24653) Table 1 — see ``docs/k3/FACTS.md`` A1–A17. Nothing here is invented;
where the checkpoint carries a vestigial field (``qk_rope_head_dim=64`` under full NoPE) we
keep it and mark it, because parity with the checkpoint beats elegance.

This is a DELEGATED-track module (see ``src/scratch_llm/k3/HANDCRAFTED.md``): agent-authored,
human-reviewed. Pure stdlib so it imports in the CPU core env (no torch).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KDAConfig:
    """Kimi Delta Attention (linear-attention workhorse, 69 of 93 layers at full scale)."""

    num_heads: int
    head_dim: int  # d_k == d_v per head (reference code: head_k_dim = head_dim)
    conv_kernel_size: int = 4  # short causal conv on q/k/v, SiLU activation
    gate_lower_bound: float = -5.0  # scaled-sigmoid log-decay floor (K3 change vs Kimi Linear)
    full_rank_gate: bool = True  # K3: full-rank sigmoid output gate (Kimi Linear was low-rank)
    decay_rank: int = 128  # low-rank decay projection width (f_a: H->128, f_b: 128->proj)
    # A_log size per KDA layer. CHECKPOINT OF RECORD: [128] (= head_dim) in all 69 layers.
    # The HF reference code constructs A_log as [num_heads=96] — a genuine code-vs-checkpoint
    # mismatch discovered by the R0 census (the −2,208 in the pre-census accounting).
    # Semantics for our hand-build (per-head vs per-dim scale) resolved in core/kda.py.
    a_log_size: int = 128

    @property
    def projection_size(self) -> int:
        return self.num_heads * self.head_dim


@dataclass(frozen=True)
class MLAConfig:
    """Gated Multi-head Latent Attention (24 of 93 layers). Fully NoPE in K3."""

    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int  # cached latent dim
    qk_nope_head_dim: int  # content head dim
    qk_rope_head_dim: int  # VESTIGIAL under NoPE (params still exist in the checkpoint)
    v_head_dim: int
    use_nope: bool = True  # K3 is NoPE everywhere (tech report §2.1.2); False = K2 heritage
    output_gate: bool = True  # full-rank sigmoid gate on attention output (arXiv:2505.06708)

    @property
    def q_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


@dataclass(frozen=True)
class MoEConfig:
    """Stable LatentMoE: sigmoid router + Quantile Balancing, latent-width routed experts."""

    num_experts: int
    top_k: int
    num_shared_experts: int
    expert_intermediate: int  # routed-expert FFN width (at latent dim)
    latent_size: int  # routed latent width ℓ (dispatch happens here; 0.5× hidden at full scale)
    dense_intermediate: int  # layer-1 dense MLP width (first_k_dense_replace=1)
    renormalize: bool = True  # top-k weights renormalized (bias excluded from weights)

    @property
    def shared_intermediate(self) -> int:
        # reference code: shared KimiMLP intermediate = moe_intermediate_size * num_shared_experts
        return self.expert_intermediate * self.num_shared_experts


@dataclass(frozen=True)
class VisionConfig:
    """MoonViT-V2 (401M-param vision encoder + PatchMergerV2 projector). Parity-only:

    the K3 track builds the text model by hand; vision is accounted here so the parameter
    closure against the released checkpoint is exact."""

    hidden_size: int = 1024
    num_layers: int = 27
    num_heads: int = 12
    qkv_hidden_size: int = 1536
    intermediate_size: int = 4096
    patch_size: int = 14
    pos_emb_height: int = 64
    pos_emb_width: int = 64
    mm_hidden_size: int = 1024
    merge_kernel: tuple[int, int] = (2, 2)
    projector_out: int = 7168  # text hidden size


@dataclass(frozen=True)
class K3Config:
    """Full model configuration (text + optional vision tower)."""

    hidden_size: int
    num_layers: int  # 1-indexed layer ids 1..num_layers
    kda_layers: tuple[int, ...]
    mla_layers: tuple[int, ...]
    vocab_size: int
    kda: KDAConfig
    mla: MLAConfig
    moe: MoEConfig
    attn_res_block_size: int  # Block AttnRes block size (12 at full scale)
    max_position_embeddings: int
    first_k_dense: int = 1  # leading layers with dense MLP instead of MoE
    tie_word_embeddings: bool = False
    vision: VisionConfig | None = None
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    situ_beta_gate: float = 4.0  # SiTU-GLU β1 (softcap on gate branch)
    situ_beta_up: float = 25.0  # SiTU-GLU β2 (softcap on up branch), bound |f| ≤ β1·β2 = 100
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def num_kda_layers(self) -> int:
        return len(self.kda_layers)

    @property
    def num_mla_layers(self) -> int:
        return len(self.mla_layers)

    @property
    def moe_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(1, self.num_layers + 1) if i > self.first_k_dense)

    @property
    def num_moe_layers(self) -> int:
        return len(self.moe_layers)


def build_layer_pattern(
    num_layers: int, period: int = 4, terminal_mla: bool = True
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """K3's block pattern: (period−1) KDA layers then 1 Gated-MLA layer per block, plus one
    terminal MLA layer at the end of the backbone (tech report §2.1).

    Full scale: 93 layers → MLA {4,8,…,92,93} (24), KDA = the remaining 69 (69:24 = 2.875:1,
    not exactly 3:1 — FACTS A2). Layer ids are 1-indexed, matching config.json.
    """
    mla = [i for i in range(1, num_layers + 1) if i % period == 0]
    if terminal_mla and num_layers not in mla:
        mla.append(num_layers)
    kda = [i for i in range(1, num_layers + 1) if i not in set(mla)]
    return tuple(kda), tuple(sorted(mla))


def k3_full() -> K3Config:
    """The released Kimi K3 checkpoint (text + MoonViT-V2), verbatim from config.json."""
    kda_layers, mla_layers = build_layer_pattern(93)
    assert len(kda_layers) == 69 and len(mla_layers) == 24  # config.json linear_attn_config
    return K3Config(
        hidden_size=7168,
        num_layers=93,
        kda_layers=kda_layers,
        mla_layers=mla_layers,
        vocab_size=163840,
        kda=KDAConfig(num_heads=96, head_dim=128),
        mla=MLAConfig(
            num_heads=96,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,  # vestigial: mla_use_nope=true, rotary_emb=None
            v_head_dim=128,
        ),
        moe=MoEConfig(
            num_experts=896,
            top_k=16,
            num_shared_experts=2,
            expert_intermediate=3072,
            latent_size=3584,
            dense_intermediate=33792,
        ),
        attn_res_block_size=12,
        max_position_embeddings=1048576,
        vision=VisionConfig(),
    )


def mini_k3_d12() -> K3Config:
    """mini-K3 d12 — the trainable miniature (ROADMAP §3 K6). Ratios kept from full scale:
    3:1+terminal layer pattern, 0.5× latent MoE, 2 shared experts, same gates/decay/SiTU.
    Deliberate substitutions: vocab 32768 (our BPE, not K3's tiktoken 160K), qk_rope_head_dim=0
    (clean NoPE — the full checkpoint's 64 is vestigial anyway), context 8K.
    """
    kda_layers, mla_layers = build_layer_pattern(12)
    assert kda_layers == (1, 2, 3, 5, 6, 7, 9, 10, 11) and mla_layers == (4, 8, 12)
    return K3Config(
        hidden_size=1024,
        num_layers=12,
        kda_layers=kda_layers,
        mla_layers=mla_layers,
        vocab_size=32768,
        kda=KDAConfig(num_heads=16, head_dim=64, decay_rank=64, a_log_size=64),
        mla=MLAConfig(
            num_heads=16,
            q_lora_rank=256,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=0,  # clean NoPE
            v_head_dim=64,
        ),
        moe=MoEConfig(
            num_experts=64,
            top_k=4,
            num_shared_experts=2,
            expert_intermediate=192,
            latent_size=512,
            dense_intermediate=4096,
        ),
        attn_res_block_size=4,
        max_position_embeddings=8192,
        vision=None,
    )
