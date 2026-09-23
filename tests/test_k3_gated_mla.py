"""K3/gated_mla gates (plan §3). Oracle: HF KimiMLAAttention (modeling_kimi_linear.py :335-474),
transcribed below.

1. HF transcription ≡ ours, fp64, with rope = 0 (as mini) and rope > 0 (as full): outputs and
   the gradients of x and all 8 parameters.
2. Absorbed ≡ naive, outputs AND gradients, fp64 (training may take either path): the two methods
   directly, and through forward (stateless vs stateful from an empty cache).
3. Hybrid cache: prefill ≡ token-by-token decode ≡ split prefill ≡ one-shot stateful prefill,
   fp64; the caches agree, hold the normed latent and the raw k_rope, and pass no gradient
   back (no cross-call BPTT). A stateful forward never runs kv_b_proj as a module (decode is
   absorbed; the cache is never expanded to per-head K/V).
4. Causality on both paths: perturbing future tokens leaves past outputs unchanged.
5. Gate position: zero g_proj ⇒ exactly ½ × the ungated output; g_proj reads x itself and
   o_proj reads attn ⊙ σ(g_proj x) (the gate sits before o_proj, width h·v).
6. Parameters: the checkpoint's names and shapes; count == param_count._mla_attn_params.
7. last_max_logits ≡ a brute-force per-pair max on both paths, each on its own input (so a stale
   value cannot pass); None when tracking is off.
8. Rope dims are live but unrotated: perturbing k_rope's rows of kv_a_proj_with_mqa changes the
   output, and permuting a token's strict past leaves its output unchanged (no positions).
9. bf16 runs (prefill + decode, bf16 module or fp32 under autocast) and stays finite; attention
   math is fp32 on both paths (pre-gate output and recorded max); a bf16 layer's cache is bf16.
10. Stale caches raise ValueError; a forward that raises leaves the cache untouched.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from scratch_llm.k3.config import MLAConfig, k3_full, mini_k3_d12
from scratch_llm.k3.core.gated_mla import GatedMLA, MLALatentKV
from scratch_llm.k3.model import HybridState
from scratch_llm.k3.param_count import _mla_attn_params

FP64_STRUCTURAL_REL = 1e-10  # same bar as test_kda_parity_fla.py (two identical algebras)

_ROPE = MLAConfig(
    num_heads=3,
    q_lora_rank=12,
    kv_lora_rank=10,
    qk_nope_head_dim=8,
    qk_rope_head_dim=4,
    v_head_dim=6,
)
_NOPE = MLAConfig(
    num_heads=3,
    q_lora_rank=12,
    kv_lora_rank=10,
    qk_nope_head_dim=8,
    qk_rope_head_dim=0,
    v_head_dim=6,
)
_CONFIGS = pytest.mark.parametrize("cfg", [_NOPE, _ROPE], ids=["rope0", "rope4"])
_HIDDEN = 16  # != heads * v = 18, so a post-o_proj (H -> H) gate could not take g_proj's shape

_CHECKPOINT_NAMES = {
    "q_a_proj.weight",
    "q_a_layernorm.weight",
    "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
    "o_proj.weight",
    "g_proj.weight",
}


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


# ---------------------------------------------------------------------------
# HF reference, transcribed (never imported from ../ladders).
# ---------------------------------------------------------------------------


def _hf_float(dtype: torch.dtype) -> torch.dtype:
    """HF casts to fp32 (``.float()`` :234, ``softmax(dtype=torch.float32)`` :328). Transcribed
    as promote(dtype, fp32): identical for fp32/bf16 inputs, and fp64 runs stay fp64."""
    return torch.promote_types(dtype, torch.float32)


class _HFKimiRMSNorm(nn.Module):
    """modeling_kimi_linear.KimiRMSNorm (:226-236)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        x = hidden_states.to(_hf_float(dtype))
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


class _HFKimiMLAAttention(nn.Module):
    """modeling_kimi_linear.KimiMLAAttention (:335-474) + eager_attention_forward (:311-332),
    K3 branch: q-LoRA, NoPE, output gate, num_key_value_heads == num_attention_heads (repeat_kv
    is the identity), no cache, no dropout."""

    def __init__(self, cfg: MLAConfig, hidden_size: int) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.q_lora_rank = cfg.q_lora_rank
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.kv_lora_rank = cfg.kv_lora_rank
        self.v_head_dim = cfg.v_head_dim
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.q_head_dim ** (-0.5)
        self.q_a_proj = nn.Linear(hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = _HFKimiRMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = _HFKimiRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden_size, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.num_heads * self.v_head_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.q_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        # eager_attention_forward
        scores = torch.einsum("bhqd,bhkd->bhqk", query_states, key_states) * self.scaling
        scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
        probs = F.softmax(scores, dim=-1, dtype=_hf_float(scores.dtype)).to(query_states.dtype)
        attn_output = torch.einsum("bhqk,bhkd->bhqd", probs, value_states).transpose(1, 2)

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        g = self.g_proj(hidden_states).sigmoid()
        attn_output = attn_output * g
        return self.o_proj(attn_output)


def _hf_causal_mask(T: int, dtype: torch.dtype) -> torch.Tensor:
    """Additive [1, 1, T, T] mask as transformers' create_causal_mask: finfo.min above the diagonal."""
    future = torch.ones(T, T, dtype=torch.bool).triu(1)
    return torch.zeros(T, T, dtype=dtype).masked_fill(future, torch.finfo(dtype).min)[None, None]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _layer(cfg: MLAConfig, seed: int = 0) -> GatedMLA:
    """fp64 layer moved off-init: norm weights spread around 1, and q_b / kv_b scaled by 3 so the
    logits have std ~3 (softmax peaked but not one-hot; ~0.3 at nn.Linear's default init)."""
    torch.manual_seed(seed)
    layer = GatedMLA(cfg, _HIDDEN).double()
    with torch.no_grad():
        layer.q_a_layernorm.weight.uniform_(0.5, 1.5)
        layer.kv_a_layernorm.weight.uniform_(0.5, 1.5)
        layer.q_b_proj.weight.mul_(3.0)
        layer.kv_b_proj.weight.mul_(3.0)
    return layer


def _x(B: int, T: int, seed: int = 1, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.randn(B, T, _HIDDEN, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def _fresh_state(batch: int) -> HybridState:
    return HybridState(kda_states=[None], mla_caches=[None], lengths=torch.zeros(batch))


def _cache_rows(state: HybridState) -> torch.Tensor:
    """[c_kv | k_rope] of slot 0 — never empty, unlike k_rope alone when rope = 0."""
    cache = state.mla_caches[0]
    assert cache is not None
    return torch.cat((cache.c_kv, cache.k_rope), dim=-1)


def _out_and_grads(
    module: nn.Module, run: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """out = run(x), and the gradients of <out, probe> (a fixed random probe) w.r.t. x and every
    parameter of ``module``, keyed by name ("x" for the input). A missing name is a parameter the
    graph never reached: a detach or a no_grad cut leaves forward values intact but drops it here."""
    module.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_()
    out = run(x)
    probe = torch.randn(out.shape, generator=torch.Generator().manual_seed(7), dtype=out.dtype)
    (out * probe).sum().backward()
    grads = {n: p.grad for n, p in module.named_parameters() if p.grad is not None}
    if x.grad is not None:
        grads["x"] = x.grad
    return out.detach(), grads


def _assert_same_outputs_and_grads(
    a: tuple[torch.Tensor, dict[str, torch.Tensor]],
    b: tuple[torch.Tensor, dict[str, torch.Tensor]],
    reached: set[str],
) -> None:
    """Two :func:`_out_and_grads` results agree in fp64, and both reach exactly ``reached`` + x."""
    (out_a, grads_a), (out_b, grads_b) = a, b
    assert _rel(out_b, out_a) < FP64_STRUCTURAL_REL
    assert grads_a.keys() == grads_b.keys() == reached | {"x"}
    for name, g in grads_a.items():
        assert _rel(grads_b[name], g) < FP64_STRUCTURAL_REL, name


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


@_CONFIGS
def test_matches_hf_transcription_fp64(cfg: MLAConfig) -> None:
    """The training forward: outputs, and the gradients of x and all 8 parameters."""
    ours = _layer(cfg)
    ref = _HFKimiMLAAttention(cfg, _HIDDEN).double()
    ref.load_state_dict(ours.state_dict())
    x = _x(2, 9)
    mask = _hf_causal_mask(9, x.dtype)
    _assert_same_outputs_and_grads(
        _out_and_grads(ref, lambda x_in: ref(x_in, mask), x),
        _out_and_grads(ours, ours, x),
        _CHECKPOINT_NAMES,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_latent_norm_bitwise_equals_hf_kimi_rmsnorm(dtype: torch.dtype) -> None:
    """The fp64 gates cannot see rounding order; this pins HF's cast-then-weight order in bf16."""
    layer = _layer(_ROPE).to(dtype)
    ref = _HFKimiRMSNorm(_ROPE.kv_lora_rank).to(dtype)
    ref.load_state_dict(layer.kv_a_layernorm.state_dict())
    z = (torch.randn(4, 7, _ROPE.kv_lora_rank) * 3).to(dtype)
    assert torch.equal(layer.kv_a_layernorm(z), ref(z))


@_CONFIGS
def test_absorbed_equals_naive_outputs_and_grads(cfg: MLAConfig) -> None:
    """The two methods directly (pre-gate output), then through forward: stateless takes the
    naive path, stateful from an empty cache the absorbed one. Nothing is detached there (only
    the committed cache copy is), so both forwards train x and all 8 parameters identically."""
    layer = _layer(cfg)
    x = _x(2, 9)
    _assert_same_outputs_and_grads(
        _out_and_grads(layer, lambda x_in: layer.attend_naive(*layer._project(x_in)), x),
        _out_and_grads(layer, lambda x_in: layer.attend_absorbed(*layer._project(x_in)), x),
        _CHECKPOINT_NAMES - {"o_proj.weight", "g_proj.weight"},
    )
    _assert_same_outputs_and_grads(
        _out_and_grads(layer, layer, x),
        _out_and_grads(layer, lambda x_in: layer(x_in, _fresh_state(2), 1), x),
        _CHECKPOINT_NAMES,
    )


@_CONFIGS
def test_prefill_equals_decode_equals_split_prefill(cfg: MLAConfig) -> None:
    layer = _layer(cfg)
    x = _x(2, 11)
    full = layer(x)  # naive path

    decode_state = _fresh_state(2)
    decoded = torch.cat([layer(x[:, t : t + 1], decode_state, 1) for t in range(11)], dim=1)

    split_state = _fresh_state(2)
    split = torch.cat([layer(x[:, :5], split_state, 1), layer(x[:, 5:], split_state, 1)], dim=1)

    oneshot_state = _fresh_state(2)
    oneshot = layer(x, oneshot_state, 1)  # absorbed path from an empty cache

    for out in (decoded, split, oneshot):
        assert _rel(out, full) < FP64_STRUCTURAL_REL
    # Not bitwise: the latents come from different-shaped matmuls (1 vs 5 vs 11 tokens).
    reference = _cache_rows(oneshot_state)
    assert reference.shape == (2, 11, cfg.kv_lora_rank + cfg.qk_rope_head_dim)
    for state in (decode_state, split_state):
        assert _rel(_cache_rows(state), reference) < FP64_STRUCTURAL_REL

    # The cache is the normed latent and the raw k_rope: no rotation, no norm on k_rope.
    cache = oneshot_state.mla_caches[0]
    assert cache is not None and cache.length == 11
    with torch.no_grad():
        c_kv, k_rope = layer.kv_a_proj_with_mqa(x).split(
            (cfg.kv_lora_rank, cfg.qk_rope_head_dim), -1
        )
        assert torch.equal(cache.c_kv, layer.kv_a_layernorm(c_kv))
        assert torch.equal(cache.k_rope, k_rope)
    assert not cache.c_kv.requires_grad and not cache.k_rope.requires_grad

    # No cross-call BPTT, even through a cache that carries grad (one built by hand, say).
    past = MLALatentKV(cache.c_kv.clone().requires_grad_(), cache.k_rope.clone().requires_grad_())
    layer(x[:, :1], HybridState([None], [past], torch.zeros(2)), 1).sum().backward()
    assert past.c_kv.grad is None and past.k_rope.grad is None


def test_stateful_forward_never_expands_the_cache() -> None:
    """Decode takes the absorbed path. Only the naive path runs the kv_b_proj module (per-head
    K/V from the latent); the absorbed path reads its weight. The paths agree numerically, so
    counting the module's calls is what tells them apart."""
    layer = _layer(_ROPE)
    calls: list[torch.Size] = []
    handle = layer.kv_b_proj.register_forward_pre_hook(
        lambda _module, args: calls.append(args[0].shape)
    )
    x = _x(2, 7)
    layer.attend_naive(*layer._project(x))
    assert len(calls) == 1  # the probe fires on the naive path
    state = _fresh_state(2)
    layer(x[:, :4], state, 1)  # prefill into an empty cache
    layer(x[:, 4:5], state, 1)  # single-token decode
    layer(x[:, 5:], state, 1)  # multi-token extend
    handle.remove()
    assert len(calls) == 1


@pytest.mark.parametrize("stateful", [False, True], ids=["naive", "absorbed"])
def test_causal(stateful: bool) -> None:
    layer = _layer(_ROPE, seed=2)
    x = _x(1, 13)
    t0 = 6
    x2 = x.clone()
    x2[:, t0 + 1 :] = torch.randn_like(x2[:, t0 + 1 :])
    if stateful:
        out, out2 = layer(x, _fresh_state(1), 1), layer(x2, _fresh_state(1), 1)
    else:
        out, out2 = layer(x), layer(x2)
    torch.testing.assert_close(out2[:, : t0 + 1], out[:, : t0 + 1], rtol=0, atol=1e-12)
    assert not torch.allclose(out2[:, t0 + 1 :], out[:, t0 + 1 :])


def test_zero_gate_halves_the_ungated_output() -> None:
    layer = _layer(_ROPE)
    x = _x(2, 7)
    with torch.no_grad():
        layer.g_proj.weight.zero_()
        ungated = layer.o_proj(layer.attend_naive(*layer._project(x)))
        gated = layer(x)
    # sigmoid(0) = 1/2 exactly, and scaling by a power of two commutes with every rounding.
    assert torch.equal(gated, 0.5 * ungated)


def test_gate_reads_x_and_sits_before_o_proj() -> None:
    layer = _layer(_ROPE)
    h_v = _ROPE.num_heads * _ROPE.v_head_dim
    assert layer.g_proj.weight.shape == (h_v, _HIDDEN)  # H -> h·v, not a post-o_proj H -> H
    assert layer.o_proj.weight.shape == (_HIDDEN, h_v)
    seen: dict[str, torch.Tensor] = {}

    def record(name: str):
        def hook(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            seen[name] = args[0]

        return hook

    handles = [
        layer.g_proj.register_forward_pre_hook(record("g_proj")),
        layer.o_proj.register_forward_pre_hook(record("o_proj")),
    ]
    x = _x(2, 7)
    with torch.no_grad():
        layer(x)
        attn = layer.attend_naive(*layer._project(x))
        expected_o_in = attn * torch.sigmoid(layer.g_proj(x))
    for handle in handles:
        handle.remove()
    assert seen["g_proj"] is x  # the layer input itself, not the attention output
    assert torch.equal(seen["o_proj"], expected_o_in)


def test_checkpoint_names_shapes_and_param_count() -> None:
    for cfg in (mini_k3_d12(), k3_full()):
        with torch.device("meta"):
            layer = GatedMLA(cfg.mla, cfg.hidden_size)
        assert {n for n, _ in layer.named_parameters()} == _CHECKPOINT_NAMES
        assert sum(p.numel() for p in layer.parameters()) == _mla_attn_params(cfg)

    # Full scale, per config.json: h=96, q_lora=1536, kv_lora=512, nope=128, rope=64, v=128.
    with torch.device("meta"):
        full = GatedMLA(k3_full().mla, 7168)
    assert {n: tuple(p.shape) for n, p in full.named_parameters()} == {
        "q_a_proj.weight": (1536, 7168),
        "q_a_layernorm.weight": (1536,),
        "q_b_proj.weight": (96 * 192, 1536),
        "kv_a_proj_with_mqa.weight": (512 + 64, 7168),
        "kv_a_layernorm.weight": (512,),
        "kv_b_proj.weight": (96 * 256, 512),
        "o_proj.weight": (7168, 96 * 128),
        "g_proj.weight": (96 * 128, 7168),
    }
    assert full.scale == 1.0 / math.sqrt(192)


def _brute_max_logits(layer: GatedMLA, x: torch.Tensor) -> torch.Tensor:
    """max over (b, t, s <= t) of scale * <[q_nope | q_rope], [k_nope | k_rope]>, one pair at a time."""
    cfg = layer.cfg
    B, T, _ = x.shape
    nope = cfg.qk_nope_head_dim
    with torch.no_grad():
        q, kv = layer._project(x)
        k_nope = layer.kv_b_proj(kv.c_kv).view(B, T, cfg.num_heads, -1)[..., :nope]
        best = torch.full((cfg.num_heads,), -math.inf, dtype=x.dtype)
        for b, h, t in itertools.product(range(B), range(cfg.num_heads), range(T)):
            for s in range(t + 1):
                key = torch.cat((k_nope[b, s, h], kv.k_rope[b, s]))
                best[h] = max(best[h], layer.scale * torch.dot(q[b, t, h], key))
    return best


@_CONFIGS
def test_last_max_logits_matches_brute_force(cfg: MLAConfig) -> None:
    """Each path gets its own input and starts from a consumed (None) record, so a path that fails
    to record cannot pass on a value an earlier call left behind."""
    layer = _layer(cfg)
    layer.track_max_logits = True
    for x, state in ((_x(2, 6, seed=1), None), (_x(2, 6, seed=2), _fresh_state(2))):
        x = x.requires_grad_()  # the recorded max must still come out detached
        layer.last_max_logits = None  # what apply_k3_qk_clip leaves behind
        layer(x, state, 1)  # state None: naive; a fresh state: absorbed
        got = layer.last_max_logits
        assert got is not None and got.shape == (cfg.num_heads,) and not got.requires_grad
        assert _rel(got, _brute_max_logits(layer, x)) < FP64_STRUCTURAL_REL

    layer.track_max_logits = False
    layer(_x(2, 6))
    assert layer.last_max_logits is None


def test_last_max_logits_is_a_running_max_over_training_forwards_only() -> None:
    """Two training microbatches before one clip: the record is the elementwise max of both
    (torchtitan qk_clip.py:46/:158). An eval forward in between must not touch it."""
    layer = _layer(_ROPE)
    layer.track_max_logits = True
    x1, x2, x_eval = _x(2, 6, seed=1), _x(2, 6, seed=2), 50 * _x(2, 6, seed=3)
    layer(x1)
    layer.eval()
    layer(x_eval)  # far larger logits; recording them would inflate the clip
    layer.train()
    layer(x2)
    got = layer.last_max_logits
    want = torch.maximum(_brute_max_logits(layer, x1), _brute_max_logits(layer, x2))
    assert got is not None and _rel(got, want) < FP64_STRUCTURAL_REL
    b1, b2 = _brute_max_logits(layer, x1), _brute_max_logits(layer, x2)
    assert bool((b1 > b2).any()) and bool((b2 > b1).any())  # both microbatches contribute


def test_rope_dims_are_live() -> None:
    layer = _layer(_ROPE)
    x = _x(2, 7)
    L = _ROPE.kv_lora_rank
    with torch.no_grad():
        before = layer(x)
        layer.kv_a_proj_with_mqa.weight[L:] += torch.randn_like(layer.kv_a_proj_with_mqa.weight[L:])
        after = layer(x)
    # Token 0 attends to itself alone (softmax weight 1), so only its logit can move, not its output.
    torch.testing.assert_close(after[:, 0], before[:, 0], rtol=0, atol=1e-12)
    assert (after[:, 1:] - before[:, 1:]).abs().amax().item() > 1e-3


def test_no_positional_encoding() -> None:
    """NoPE: out[t] is a set function of x[:t] (plus x[t]), so permuting the strict past of the
    last token leaves its output unchanged. Any rotation of k_rope or q_rope would break this."""
    layer = _layer(_ROPE, seed=3)
    x = _x(2, 8)
    perm = torch.tensor([4, 0, 6, 2, 1, 5, 3, 7])
    out, permuted = layer(x), layer(x[:, perm])
    assert _rel(permuted[:, -1], out[:, -1]) < FP64_STRUCTURAL_REL
    assert _rel(permuted[:, 3], out[:, 3]) > 1e-3  # earlier tokens do see a different past


@pytest.mark.parametrize("autocast", [False, True], ids=["bf16-module", "fp32-autocast"])
def test_bf16_runs_finite(autocast: bool) -> None:
    """Projections in bf16; scores, softmax and weighted sums in fp32 on both paths, seen as the
    fp32 pre-gate output and the fp32 recorded max; bf16 out, finite."""
    torch.manual_seed(0)
    layer = GatedMLA(_ROPE, _HIDDEN)
    layer.track_max_logits = True
    x = _x(2, 6, dtype=torch.float32)
    if not autocast:
        layer, x = layer.bfloat16(), x.bfloat16()
    state = _fresh_state(2)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        q, kv = layer._project(x)
        assert q.dtype == torch.bfloat16  # so the fp32 below is the math dtype, not the input's
        assert layer.attend_naive(q, kv).dtype == torch.float32
        assert layer.attend_absorbed(q, kv).dtype == torch.float32
        for chunk, chunk_state in ((x, None), (x[:, :4], state), (x[:, 4:5], state)):
            layer.last_max_logits = None
            out = layer(chunk, chunk_state, 1)
            assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()
            got = layer.last_max_logits
            assert got is not None and got.dtype == torch.float32
    if not autocast:  # a bf16 layer keeps a bf16 cache (2 bytes per latent value), not fp32
        cache = state.mla_caches[0]
        assert cache is not None and cache.c_kv.dtype == cache.k_rope.dtype == torch.bfloat16


def test_stale_cache_raises_and_failed_forward_commits_nothing() -> None:
    layer = _layer(_ROPE)
    L, R = _ROPE.kv_lora_rank, _ROPE.qk_rope_head_dim
    state = _fresh_state(2)
    layer(_x(2, 4), state, 1)
    with pytest.raises(ValueError, match="does not match"):
        layer(_x(3, 1), state, 1)  # wrong batch
    with pytest.raises(ValueError, match="1-based"):
        layer(_x(2, 1), state, 0)
    for bad in (
        MLALatentKV(torch.zeros(2, 3, L + 1), torch.zeros(2, 3, R)),  # wrong latent width
        MLALatentKV(torch.zeros(2, 3, L), torch.zeros(2, 3, R + 1)),  # wrong rope width
        MLALatentKV(torch.zeros(2, 3, L), torch.zeros(2, 4, R)),  # lengths disagree
    ):
        with pytest.raises(ValueError, match="does not match"):
            layer(_x(2, 1), HybridState([None], [bad], torch.zeros(2)), 1)

    def boom(_module: nn.Module, _args: tuple[torch.Tensor, ...]) -> None:
        raise RuntimeError("boom")

    cache = state.mla_caches[0]
    assert cache is not None
    snapshot = (cache.c_kv.clone(), cache.k_rope.clone())
    handle = layer.o_proj.register_forward_pre_hook(boom)
    with pytest.raises(RuntimeError, match="boom"):
        layer(_x(2, 1), state, 1)
    handle.remove()
    assert state.mla_caches[0] is cache and cache.length == 4
    assert torch.equal(cache.c_kv, snapshot[0]) and torch.equal(cache.k_rope, snapshot[1])
