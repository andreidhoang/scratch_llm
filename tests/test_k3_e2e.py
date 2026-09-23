"""K3 end to end: decoding a K3Model through ``sampling`` and the whole ``run_speedrun`` chain.

(a) Cached decode (prefill the prompt into ``HybridState``, then one token per forward) ≡ the
    uncached oracle (a full recompute every step): identical greedy tokens, logprobs at the
    fp64 bar; in fp32 the same tokens and logprobs within a derived fp32 bound. The cache is
    allocated in the embedding's dtype. The model is off-init and the greedy decode certified
    tie-free, so an agreement is not an accident of flat logits. Because a recompute every step
    would match the oracle too, the forwards themselves are pinned: one prefill of the prompt,
    then one token per forward against the same state, whose length ramps by one, and every
    decode starts from an empty state.
(b) ``generate_with_logprobs`` returns log π at temperature 1 read from the logits each token was
    sampled from: equal to an independent teacher-forced re-score (``LocalBackend.score``, one
    forward over prompt + generation) at the fp64 bar, for a top-p sampled decode on both paths.
(c) ``stop_ids`` end the decode with the stop token included, ``max_tokens`` is exact, a seed
    reproduces a sampled decode (and the two paths draw the same tokens under it).
(d) ``run_speedrun(model_family="k3_mini", k3_config=<tiny>)`` runs every stage on CPU in seconds:
    the stage list, a finite report card, a non-empty sample and chat reply without leaked
    specials, the model built from ``k3_config`` with the run's vocab and context, SFT on a fresh
    ``build_k3_optimizer`` (the pretrain kind), Quantile Balancing after every pretrain and SFT
    step, and router biases that moved in both stages. ``stage_sft`` alone, at two batches of
    different lengths per epoch: the optimizer is the pretrain's kind for each of None,
    ``k3_muon`` and ``adamw``, and every step runs forward, optimizer step, then one Quantile
    Balancing update that consumes exactly that step's batch (not an epoch's, not only the first).
(e) ``resume=True`` rebuilds the pretrain and SFT models from their checkpoints (training is
    patched to raise) and the resumed model is the first run's: every parameter and the logits
    bitwise, the report card, sample and chat reply equal.
(f) The dense family is gated by the existing speedrun/sampling/chat tests, unchanged. Here only
    the dispatch: a ``torch.compile``'d TransformerLM, which is not a TransformerLM instance,
    still gets a KVCache and decodes exactly as the bare model.
(g) A cached K3 decode whose prompt plus budget exceeds ``max_position_embeddings`` is refused
    before any forward; the uncached oracle crops to the last ``max_position_embeddings`` tokens.
    A refused ``ChatSession.reply`` leaves the history as it was.
(h) The CLI threads ``--model-family`` and ``--optimizer`` into ``SpeedrunConfig`` and refuses
    ``--nano`` with ``k3_mini``.

Tiny config: 4 layers at period 4 (KDA 1-3, MLA 4 with rope 4, live), layer 1 dense and 2-4
LatentMoE, B = 2, window 24 = prompt 6 + budget 18, so the decode gates run exactly at the edge
that (g) moves past.
"""

from __future__ import annotations

import copy
import math
import sys
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pytest
import torch
from torch import Tensor, nn

import scratch_llm.algos.chat_sft as chat_sft
import scratch_llm.optim as optim
import scratch_llm.speedrun as speedrun
from scratch_llm.chat import CHAT_SPECIAL_TOKENS, EOT
from scratch_llm.chat_cli import ChatSession
from scratch_llm.eval import ReportCard
from scratch_llm.eval.metrics import bits_per_byte
from scratch_llm.k3.config import K3Config, KDAConfig, MLAConfig, MoEConfig, build_layer_pattern
from scratch_llm.k3.core.gated_mla import MLALatentKV
from scratch_llm.k3.core.kda import KDAState
from scratch_llm.k3.core.latent_moe import QBStats
from scratch_llm.k3.model import HybridState, K3Model
from scratch_llm.model import KVCache, ModelConfig, TransformerLM
from scratch_llm.optim import CombinedOptimizer
from scratch_llm.rollout import LocalBackend
from scratch_llm.sampling import (
    CausalLM,
    SamplingParams,
    _decode_cache,
    generate,
    generate_with_logprobs,
)
from scratch_llm.speedrun import SpeedrunConfig, SpeedrunResult, run_speedrun
from scratch_llm.tokenizer import Tokenizer

VOCAB, WINDOW, PROMPT_LEN = 32, 24, 6
BUDGET = WINDOW - PROMPT_LEN  # prompt + budget == window: the largest cached decode allowed
# Chunkwise prefill + recurrent decode vs a chunkwise recompute: two algebras, one function. The
# bar is test_k3_model.py's prefill ≡ decode bar; measured 7e-16 to 2.7e-15 on seeds 0-4.
FP64_REL = 1e-10
# fp32 u = 2^-24 ≈ 6e-8; measured max |Δ logprob| 3.5e-6 to 1.1e-5 on seeds 0-4, and each path
# alone is off the fp64 decode by ≤ 2e-5 (≈ 80 u·|z|max, |z| ≤ 4: rounding compounded through
# 4 layers). 1e-4 = 5× the worst single-path error, 3 decades under the O(0.1) that one mis-fed
# token or a stale state produces.
FP32_LOGPROB_ATOL = 1e-4
# Tie-free certificate: top-1 minus top-2 logit at every greedy step (measured ≥ 1.0e-2). Above
# 1e-3 no path error under 5e-4 per logit (fp32's ≤ 2e-5 included) can flip the argmax.
TIE_FREE_MARGIN = 1e-3


def _tiny_cfg() -> K3Config:
    kda_layers, mla_layers = build_layer_pattern(4, period=4)
    assert kda_layers == (1, 2, 3) and mla_layers == (4,)
    return K3Config(
        hidden_size=32,
        num_layers=4,
        kda_layers=kda_layers,
        mla_layers=mla_layers,
        vocab_size=VOCAB,
        kda=KDAConfig(num_heads=2, head_dim=8, decay_rank=4, a_log_size=4),
        mla=MLAConfig(
            num_heads=4,
            q_lora_rank=12,
            kv_lora_rank=10,
            qk_nope_head_dim=8,
            qk_rope_head_dim=4,
            v_head_dim=6,
        ),
        moe=MoEConfig(
            num_experts=8,
            top_k=2,
            num_shared_experts=1,
            expert_intermediate=12,
            latent_size=16,
            dense_intermediate=48,
        ),
        attn_res_block_size=2,
        max_position_embeddings=WINDOW,
        rms_norm_eps=2e-5,
    )


@torch.no_grad()
def _off_init(model: K3Model, seed: int) -> K3Model:
    """tests/test_k3_model.py's off-init, in place: 2-D weights N(0, 1/fan_in) (unit-scale
    activations, so the logits spread over O(1) and greedy has clear winners), AttnRes
    pseudo-queries N(0, 4/H), norm gains U(0.5, 1.5), KDA A_log N(0, 0.25) and dt_bias N(0, 4)
    (spread decays, so the recurrent state matters), QB bias N(0, 0.05²). The router and the
    short convs keep their draws."""
    gen = torch.Generator().manual_seed(seed)
    H = model.cfg.hidden_size

    def normal(p: Tensor, std: float) -> None:
        p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * std)

    for name, p in model.named_parameters():
        if name.endswith("norm.weight"):
            p.copy_(0.5 + torch.rand(p.shape, generator=gen, dtype=p.dtype))
        elif name.endswith("res_proj.weight"):
            normal(p, 2 / math.sqrt(H))
        elif name.endswith("A_log"):
            normal(p, 0.5)
        elif name.endswith("dt_bias"):
            normal(p, 2.0)
        elif name.endswith("e_score_correction_bias"):
            normal(p, 0.05)
        elif p.ndim == 2 and not name.endswith("gate.weight"):
            normal(p, 1 / math.sqrt(p.shape[1]))
    return model


def _model(seed: int) -> K3Model:
    torch.manual_seed(0)
    return _off_init(K3Model(_tiny_cfg()).double(), seed).eval()


def _prompt(seed: int, length: int = PROMPT_LEN) -> list[int]:
    gen = torch.Generator().manual_seed(100 + seed)
    return torch.randint(0, VOCAB, (length,), generator=gen).tolist()


def _rel(a: list[float], b: list[float]) -> float:
    x, y = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    return ((x - y).norm() / y.norm()).item()


@torch.no_grad()
def _top2_margins(model: K3Model, prompt: list[int], generated: list[int]) -> Tensor:
    """Top-1 minus top-2 logit at each position that chose a generated token (one forward)."""
    logits = model(torch.tensor([prompt + generated]))[0]
    rows = logits[len(prompt) - 1 : len(prompt) - 1 + len(generated)]
    top2 = rows.topk(2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


# ---------------------------------------------------------------------------
# (a) Cached ≡ uncached.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_cached_decode_equals_the_uncached_recompute(seed: int) -> None:
    model, prompt = _model(seed), _prompt(seed)
    greedy = SamplingParams(temperature=0.0, max_tokens=BUDGET)

    tokens, lp_cached = generate_with_logprobs(model, prompt, greedy, use_cache=True)
    oracle, lp_oracle = generate_with_logprobs(model, prompt, greedy, use_cache=False)
    assert len(tokens) == BUDGET and tokens == oracle
    assert _rel(lp_cached, lp_oracle) < FP64_REL
    assert _top2_margins(model, prompt, tokens).min().item() > TIE_FREE_MARGIN

    fp32 = copy.deepcopy(model).float()
    tokens32, lp32_cached = generate_with_logprobs(fp32, prompt, greedy, use_cache=True)
    oracle32, lp32_oracle = generate_with_logprobs(fp32, prompt, greedy, use_cache=False)
    assert tokens32 == oracle32 == tokens
    worst = max(abs(a - b) for a, b in zip(lp32_cached, lp32_oracle, strict=True))
    assert worst <= FP32_LOGPROB_ATOL, worst


def test_cached_decode_prefills_once_then_feeds_one_token_per_forward() -> None:
    """What the cached path costs, which its output cannot show: re-prefilling the whole
    sequence every step, or never passing the state, decodes the same tokens as the oracle.

    A pre-hook records each forward's input shape, its state object and that state's lengths
    before the forward bumps them. The loop feeds the last generated token back too, so BUDGET
    tokens take 1 + BUDGET forwards. The second decode on the same model must start from an
    empty state again."""
    model, prompt = _model(seed=0), _prompt(seed=0)
    calls: list[tuple[tuple[int, ...], HybridState | None, list[int]]] = []

    def record(_: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        token_ids = args[0]
        state = args[1] if len(args) > 1 else kwargs.get("state")
        lengths = state.lengths.tolist() if isinstance(state, HybridState) else []
        calls.append((tuple(token_ids.shape), state, lengths))

    model.register_forward_pre_hook(record, with_kwargs=True)
    greedy = SamplingParams(temperature=0.0, max_tokens=BUDGET)
    for _ in range(2):
        calls.clear()
        assert len(generate(model, prompt, greedy)) == BUDGET
        assert [shape for shape, _, _ in calls] == [(1, PROMPT_LEN)] + [(1, 1)] * BUDGET
        state = calls[0][1]
        assert isinstance(state, HybridState) and all(s is state for _, s, _ in calls)
        assert [n for _, _, n in calls] == [[0]] + [[PROMPT_LEN + i] for i in range(BUDGET)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
def test_the_decode_cache_is_a_hybrid_state_in_the_embedding_dtype(dtype: torch.dtype) -> None:
    """Pinned directly: both attention modules accept a cache slot of any dtype and write it back
    in their own, so a wrong allocation dtype shows in no logit (tests/test_k3_model.py)."""
    model = K3Model(_tiny_cfg()).to(dtype)
    state = _decode_cache(model, PROMPT_LEN, BUDGET, "cpu")
    assert isinstance(state, HybridState) and not state.lengths.any()
    for kda, mla in zip(state.kda_states, state.mla_caches, strict=True):
        if kda is not None:
            assert isinstance(kda, KDAState) and kda.conv.dtype == dtype
        else:
            assert isinstance(mla, MLALatentKV) and mla.c_kv.dtype == dtype


# ---------------------------------------------------------------------------
# (b) Inline logprobs ≡ a teacher-forced re-score.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_cache", [True, False], ids=["cached", "uncached"])
def test_inline_logprobs_equal_a_teacher_forced_rescore(use_cache: bool) -> None:
    """Sampled at temperature 0.8 under top-p 0.9, so the returned numbers must be the raw
    temperature-1 log π, not the log of the distribution the token was drawn from."""
    model, prompt = _model(seed=2), _prompt(seed=2)
    params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=12, seed=3)
    tokens, logprobs = generate_with_logprobs(model, prompt, params, use_cache=use_cache)
    assert len(tokens) == 12
    assert _rel(logprobs, LocalBackend(model).score(prompt, tokens)) < FP64_REL


# ---------------------------------------------------------------------------
# (c) Stop ids, budget, seed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_cache", [True, False], ids=["cached", "uncached"])
def test_stop_ids_budget_and_seed(use_cache: bool) -> None:
    model, prompt = _model(seed=0), _prompt(seed=0)

    def run(params: SamplingParams, cached: bool = use_cache) -> list[int]:
        return generate(model, prompt, params, use_cache=cached)

    greedy = SamplingParams(temperature=0.0, max_tokens=BUDGET)
    full = run(greedy)
    assert run(replace(greedy, max_tokens=7)) == full[:7]  # the budget, exactly
    stop = full[3]
    first = full.index(stop)  # the stop may already appear before position 3
    stopped = run(replace(greedy, stop_ids=(stop,)))
    assert stopped == full[: first + 1] and stopped[-1] == stop  # stop token included

    sampled = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=10, seed=7)
    draw = run(sampled)
    assert draw == run(sampled) == run(sampled, cached=not use_cache)
    assert draw != run(replace(sampled, seed=8))  # positive control: the seed reproduces it


# ---------------------------------------------------------------------------
# (g) The window.
# ---------------------------------------------------------------------------


def test_cached_decode_refuses_past_the_window_and_the_oracle_crops() -> None:
    model, prompt = _model(seed=0), _prompt(seed=0)
    forwards: list[int] = []
    model.register_forward_pre_hook(lambda *_: forwards.append(1))

    edge = SamplingParams(temperature=0.0, max_tokens=BUDGET + 1)  # one token past (a)'s decode
    with pytest.raises(ValueError, match="max_position_embeddings"):
        generate(model, prompt, edge, use_cache=True)
    with pytest.raises(ValueError, match="max_position_embeddings"):  # the prompt alone
        generate(model, _prompt(seed=0, length=WINDOW + 1), replace(edge, max_tokens=1))
    assert not forwards  # refused before any forward

    # The oracle decodes past the window by cropping: its last token is chosen from the last
    # WINDOW tokens alone, and reading all of them would have scored it differently.
    over = replace(edge, max_tokens=BUDGET + 4)
    tokens, logprobs = generate_with_logprobs(model, prompt, over, use_cache=False)
    context = (prompt + tokens)[:-1]
    assert len(context) == WINDOW + 3
    with torch.no_grad():
        cropped = torch.log_softmax(model(torch.tensor([context[-WINDOW:]]))[0, -1], dim=-1)
        whole = torch.log_softmax(model(torch.tensor([context]))[0, -1], dim=-1)
    assert tokens[-1] == int(cropped.argmax())
    assert abs(logprobs[-1] - cropped[tokens[-1]].item()) <= FP64_REL * abs(logprobs[-1])
    assert abs(logprobs[-1] - whole[tokens[-1]].item()) > 1e-3  # the crop is not a no-op


CONTEXT = 32  # the chat and speedrun window: "say hi" renders to 10 tokens, + a 12-token budget


def _chat_cfg(**overrides: Any) -> SpeedrunConfig:
    """A tiny k3_mini run config with the chat specials (BPE vocab 300 on the built-in corpus)."""
    return replace(
        SpeedrunConfig(
            model_family="k3_mini",
            k3_config=_tiny_cfg(),
            vocab_size=300,
            context_length=CONTEXT,
            chat=True,
            sample_tokens=12,
            seed=0,
        ),
        **overrides,
    )


def _chat_k3(tokenizer: Tokenizer) -> K3Model:
    torch.manual_seed(0)
    cfg = replace(_tiny_cfg(), vocab_size=len(tokenizer.vocab), max_position_embeddings=CONTEXT)
    return K3Model(cfg)


@pytest.fixture(scope="module")
def chat_tokenizer() -> Tokenizer:
    return speedrun.stage_tokenizer(_chat_cfg())[0]


def test_a_refused_chat_reply_leaves_the_history_unchanged(chat_tokenizer: Tokenizer) -> None:
    """Every reply renders the whole history, so a user turn kept from a refused decode would be
    rendered into every later reply. A budget of the whole window refuses any prompt; the reply
    after it, at a budget that fits, equals a fresh session's."""
    model = _chat_k3(chat_tokenizer)
    session = ChatSession(model, chat_tokenizer, max_tokens=CONTEXT)
    with pytest.raises(ValueError, match="max_position_embeddings"):
        session.reply("say hi")
    assert session.history == []

    session.max_tokens = 12
    fresh = ChatSession(model, chat_tokenizer, max_tokens=12)
    assert session.reply("say hi") == fresh.reply("say hi")
    assert session.history == fresh.history and len(session.history) == 2


# ---------------------------------------------------------------------------
# (d) + (e) run_speedrun on K3, and its resume.
# ---------------------------------------------------------------------------

TRAIN_STEPS, SFT_STEPS = 30, 10


def test_k3_init_is_seeded_from_the_run_seed() -> None:
    """stage_pretrain builds the K3 from cfg.seed, whatever the global RNG did before (0 steps:
    the returned model is the init)."""
    cfg = SpeedrunConfig(model_family="k3_mini", k3_config=_tiny_cfg(), train_steps=0, seed=3)
    corpus = np.arange(64, dtype=np.int64) % VOCAB

    def build(run_cfg: SpeedrunConfig) -> dict[str, Tensor]:
        torch.randn(7)  # move the global RNG between builds
        return speedrun.stage_pretrain(run_cfg, corpus, VOCAB)[0].state_dict()

    first, again, other = build(cfg), build(cfg), build(replace(cfg, seed=4))
    assert all(torch.equal(t, again[n]) for n, t in first.items())
    assert not torch.equal(first["embed_tokens.weight"], other["embed_tokens.weight"])


@dataclass
class _Run:
    """One run_speedrun with the observers the gates read: every model handed to stage_eval (the
    final model), every build_k3_optimizer call speedrun makes (SFT's; train() builds pretrain's
    through its own import), and the number of Quantile-Balancing updates."""

    result: SpeedrunResult
    eval_models: list[CausalLM]
    sft_optimizer_calls: list[dict[str, Any]]
    qb_updates: int


def _observed_run(cfg: SpeedrunConfig, *, forbid_training: bool = False) -> _Run:
    """``run_speedrun(cfg)`` with the observers patched in by name (run_speedrun resolves its
    stages, and speedrun its optimizer builder, through module globals). ``forbid_training``
    makes ``train`` and ``chat_sft_epoch`` raise, so a resumed run must load every stage."""
    eval_models: list[CausalLM] = []
    optimizer_calls: list[dict[str, Any]] = []
    qb_updates = 0
    real_eval, real_build = speedrun.stage_eval, speedrun.build_k3_optimizer
    real_update = K3Model.moe_update_biases

    def spy_eval(
        run_cfg: SpeedrunConfig, model: CausalLM, tokenizer: Tokenizer, tokens: np.ndarray
    ) -> ReportCard:
        eval_models.append(model)
        return real_eval(run_cfg, model, tokenizer, tokens)

    def spy_build(model: nn.Module, **kw: Any) -> CombinedOptimizer:
        optimizer_calls.append(kw)
        return real_build(model, **kw)

    def counted_update(self: K3Model) -> dict[str, QBStats]:
        nonlocal qb_updates
        qb_updates += 1
        return real_update(self)

    def refuse(*_: object, **__: object) -> NoReturn:
        raise AssertionError("a resumed stage retrained instead of loading its checkpoint")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(speedrun, "stage_eval", spy_eval)
        mp.setattr(speedrun, "build_k3_optimizer", spy_build)
        mp.setattr(K3Model, "moe_update_biases", counted_update)
        if forbid_training:
            mp.setattr(speedrun, "train", refuse)
            mp.setattr(chat_sft, "chat_sft_epoch", refuse)
        result = run_speedrun(cfg)
    return _Run(result, eval_models, optimizer_calls, qb_updates)


@pytest.fixture(scope="module")
def first_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[SpeedrunConfig, _Run]]:
    """The tiny K3 through every stage (~4 s on CPU): ``_chat_cfg`` with the built-in corpus and
    chat set, at batch 8, so SFT takes the 4 conversations in one batch."""
    cfg = _chat_cfg(
        train_steps=TRAIN_STEPS,
        batch_size=8,
        sft_steps=SFT_STEPS,
        work_dir=str(tmp_path_factory.mktemp("k3_speedrun")),
    )
    yield cfg, _observed_run(cfg)


def _biases(state: dict[str, Tensor]) -> dict[str, Tensor]:
    return {n: t for n, t in state.items() if n.endswith("e_score_correction_bias")}


def test_run_speedrun_k3_mini_composes_every_stage(first_run: tuple[SpeedrunConfig, _Run]) -> None:
    cfg, run = first_run
    res = run.result
    assert res.stages == ["tokenizer", "pretrain", "sft", "eval", "sample", "chat"]
    card = res.report_card
    assert card.val_bpb is not None and math.isfinite(card.val_bpb) and card.val_bpb > 0
    assert card.nats_per_token is not None and math.isfinite(card.nats_per_token)
    assert res.sample and res.chat_reply
    assert not any(special in res.chat_reply for special in CHAT_SPECIAL_TOKENS)
    assert res.summary().startswith("speedrun family=k3_mini")

    (model,) = run.eval_models
    assert isinstance(model, K3Model) and cfg.work_dir is not None
    vocab = len(Tokenizer.load(Path(cfg.work_dir) / "tokenizer.json").vocab)
    assert model.cfg == replace(_tiny_cfg(), vocab_size=vocab, max_position_embeddings=CONTEXT)
    assert res.n_params == sum(p.numel() for p in model.parameters())

    # SFT: one fresh k3_muon optimizer at the run's lr for both halves, no weight decay.
    assert run.sft_optimizer_calls == [{"lr": cfg.lr, "adamw_lr": cfg.lr, "weight_decay": 0.0}]
    # Quantile Balancing once per optimizer step: train()'s pretrain steps, then every SFT step.
    # Here one SFT step is one epoch, so the count cannot tell a step from an epoch; the stage_sft
    # test below runs two batches per epoch and can.
    assert run.qb_updates == TRAIN_STEPS + SFT_STEPS

    # The router biases start at 0 and only Quantile Balancing writes them (no gradient), so a
    # nonzero pretrain bias moved in pretrain and a changed SFT bias moved in SFT.
    work = Path(cfg.work_dir)
    pretrained = _biases(torch.load(work / "pretrain.pt", weights_only=True)["model"])
    tuned = _biases(torch.load(work / "sft.pt", weights_only=True)["model"])
    final = _biases(model.state_dict())
    assert pretrained.keys() == tuned.keys() == final.keys() and len(final) == 3  # layers 2-4
    for name, bias in final.items():
        assert not model.get_parameter(name).requires_grad
        assert pretrained[name].any(), name
        assert not torch.equal(tuned[name], pretrained[name]), name
        assert torch.equal(bias, tuned[name]), name


def test_resumed_run_rebuilds_the_same_k3(first_run: tuple[SpeedrunConfig, _Run]) -> None:
    cfg, first = first_run
    resumed = _observed_run(replace(cfg, resume=True), forbid_training=True)
    res, ref = resumed.result, first.result
    assert res.stages == [
        "tokenizer[resumed]",
        "pretrain[resumed]",
        "sft",
        "eval",
        "sample",
        "chat",
    ]
    assert res.report_card == ref.report_card
    assert (res.sample, res.chat_reply) == (ref.sample, ref.chat_reply)
    assert resumed.qb_updates == 0 and not resumed.sft_optimizer_calls

    (model,), (original,) = resumed.eval_models, first.eval_models
    assert isinstance(model, K3Model) and model is not original
    for (name, p), q in zip(model.named_parameters(), original.parameters(), strict=True):
        assert torch.equal(p, q), name
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(0, model.cfg.vocab_size, (2, 9), generator=gen)
    with torch.no_grad():
        assert torch.equal(model(ids), original(ids))


@pytest.mark.parametrize("optimizer", [None, "k3_muon", "adamw"])
def test_k3_sft_uses_the_pretrain_kind_and_balances_after_every_step(
    optimizer: str | None, chat_tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``stage_sft`` at batch 2: the 4 built-in conversations make two batches of different
    lengths per epoch, so 4 steps span two epochs. Quantile Balancing once per epoch would run
    twice on the summed counts, and on the first batch only would run twice; both show here.

    The fresh optimizer is the pretrain's kind (``stage_sft`` docstring): ``build_k3_optimizer``
    at the run's lr for both halves for None and ``k3_muon``, AdamW for ``adamw``; weight decay 0
    either way."""
    cfg = _chat_cfg(batch_size=2, sft_steps=4, optimizer=optimizer)
    model = _chat_k3(chat_tokenizer)
    events: list[str] = []
    built: list[tuple[str, dict[str, Any]]] = []
    updates: list[dict[str, QBStats]] = []
    real_k3, real_adamw = speedrun.build_k3_optimizer, optim.build_optimizer
    real_update = K3Model.moe_update_biases

    def logged(opt: Any) -> Any:
        """The optimizer, with each ``step()`` logged after it runs."""
        real_step = opt.step

        def step() -> None:
            real_step()
            events.append("step")

        monkeypatch.setattr(opt, "step", step)
        return opt

    def spy_k3(m: nn.Module, **kw: Any) -> CombinedOptimizer:
        built.append(("k3_muon", kw))
        return logged(real_k3(m, **kw))

    def spy_adamw(m: nn.Module, **kw: Any) -> torch.optim.Optimizer | CombinedOptimizer:
        built.append(("adamw", kw))
        return logged(real_adamw(m, **kw))

    def spy_update(self: K3Model) -> dict[str, QBStats]:
        events.append("qb")
        updates.append(real_update(self))
        return updates[-1]

    monkeypatch.setattr(speedrun, "build_k3_optimizer", spy_k3)
    monkeypatch.setattr(optim, "build_optimizer", spy_adamw)  # stage_sft imports it at call time
    monkeypatch.setattr(K3Model, "moe_update_biases", spy_update)
    model.register_forward_pre_hook(lambda *_: events.append("fwd"))

    speedrun.stage_sft(cfg, model, chat_tokenizer)

    if optimizer == "adamw":
        assert built == [("adamw", {"kind": "adamw", "lr": cfg.lr, "weight_decay": 0.0})]
    else:
        assert built == [("k3_muon", {"lr": cfg.lr, "adamw_lr": cfg.lr, "weight_decay": 0.0})]
    assert events == ["fwd", "step", "qb"] * 4

    conversations = speedrun._load_chat_set(cfg)
    pad = chat_tokenizer.encode(EOT)[0]
    sizes = [
        chat_sft.collate_chat_batch(conversations[s : s + 2], chat_tokenizer, pad_token_id=pad)[
            "input_ids"
        ].numel()
        for s in (0, 2)
    ]
    assert sizes[0] != sizes[1]  # so an update over both batches cannot pass for either
    for step, stats in enumerate(updates):
        assert len(stats) == 3, step  # LatentMoE layers 2-4
        assert {s.num_tokens.item() for s in stats.values()} == {sizes[step % 2]}, step


# ---------------------------------------------------------------------------
# (f) The dense dispatch.
# ---------------------------------------------------------------------------


def test_a_compiled_transformer_lm_still_decodes_on_the_dense_path() -> None:
    """``sampling`` tests for K3Model and decodes every other model as dense. ``torch.compile``
    returns an OptimizedModule, which forwards ``cfg``, ``blocks`` and the cache argument to the
    TransformerLM it wraps but is not one, so a test for TransformerLM would send it down the K3
    branch. The eager backend runs the same ops, so both paths equal the bare model's exactly."""
    torch.manual_seed(0)
    dense = TransformerLM(
        ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2, context_length=WINDOW)
    ).eval()
    # Typed as a plain callable by torch; at runtime the OptimizedModule sampling receives.
    compiled: Any = torch.compile(dense, backend="eager")
    assert not isinstance(compiled, TransformerLM)
    assert isinstance(_decode_cache(compiled, PROMPT_LEN, 8, "cpu"), KVCache)

    greedy, prompt = SamplingParams(temperature=0.0, max_tokens=8), _prompt(seed=0)
    for use_cache in (True, False):
        expected = generate_with_logprobs(dense, prompt, greedy, use_cache=use_cache)
        assert generate_with_logprobs(compiled, prompt, greedy, use_cache=use_cache) == expected


# ---------------------------------------------------------------------------
# (h) The CLI.
# ---------------------------------------------------------------------------


class _Parsed(Exception):
    """Raised by the patched ``run_speedrun`` once ``main`` has built its config."""


@pytest.mark.parametrize(
    ("argv", "family", "optimizer"),
    [
        ([], "dense", None),
        (["--model-family", "k3_mini"], "k3_mini", None),
        (["--model-family", "k3_mini", "--optimizer", "adamw"], "k3_mini", "adamw"),
    ],
)
def test_cli_threads_the_model_family_and_optimizer(
    argv: list[str], family: str, optimizer: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the threading, ``--model-family k3_mini`` would silently train the dense d20."""
    configs: list[SpeedrunConfig] = []

    def capture(cfg: SpeedrunConfig) -> NoReturn:
        configs.append(cfg)
        raise _Parsed

    monkeypatch.setattr(speedrun, "run_speedrun", capture)
    monkeypatch.setattr(sys, "argv", ["speedrun", *argv])
    with pytest.raises(_Parsed):
        speedrun.main()
    (cfg,) = configs
    assert (cfg.model_family, cfg.optimizer) == (family, optimizer)


def test_cli_refuses_nano_with_k3_mini(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(cfg: SpeedrunConfig) -> NoReturn:
        raise AssertionError(f"main ran {cfg.model_family} instead of refusing")

    monkeypatch.setattr(speedrun, "run_speedrun", refuse)
    monkeypatch.setattr(sys, "argv", ["speedrun", "--nano", "--model-family", "k3_mini"])
    with pytest.raises(SystemExit) as exit_info:
        speedrun.main()
    assert exit_info.value.code == 2 and "--nano is the dense pre-flight" in capsys.readouterr().err


def test_bits_per_byte_defaults_to_the_k3_window() -> None:
    """K3 has no cfg.context_length (no positional encoding); the bpb scorer's default window is
    its max_position_embeddings, identical to passing that window explicitly."""
    torch.manual_seed(0)
    model = K3Model(_tiny_cfg()).double()
    ids = torch.randint(0, VOCAB, (3 * WINDOW + 5,))
    default = bits_per_byte(model, ids, num_bytes=len(ids))
    explicit = bits_per_byte(model, ids, num_bytes=len(ids), context_length=WINDOW)
    assert default == explicit and math.isfinite(default.bits_per_byte)
