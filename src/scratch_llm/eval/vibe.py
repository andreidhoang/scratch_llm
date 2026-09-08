"""Vibe-evals — the fixed-prompt, fixed-decode qualitative probe run on every checkpoint (T1/T-R4).

A "vibe check" is the oldest evaluation in the book: talk to the model and see whether it is
there yet. It is also the easiest one to fool yourself with, and there is exactly one reason:
**a vibe-eval that resamples per checkpoint compares two things that differ by noise as much as
by training.** Change the prompts between the pretrain card and the SFT card, or leave the
sampler free-running, and the difference you read off is a draw from the decoding distribution,
not a difference between the two models.

So this module pins the three things that must not move:

1. **The prompt set** — :data:`VIBE_PROMPTS`, a frozen tuple with a set id (:data:`VIBE_SET_ID`).
   Editing it is allowed; editing it *silently* is not, so the id is versioned and every run
   carries a hash of the prompts it actually used.
2. **The decoding config** — :class:`VibeDecode` (template, temperature, top-p, budget, seed).
   Greedy by default, because a comparison you can re-run bit-for-bit is worth more here than a
   sample that looks livelier. At temperature > 0 the seed is derived **per prompt**
   (:func:`prompt_seed`) rather than taken from a single stream, so a run is invariant to prompt
   order and to which prompts were skipped — the same prompt gets the same noise every time.
3. **The outputs** — stored as *token ids* (not just text), so "identical" is decided before the
   decoder's byte-level round-trip can hide a difference.

Two runs are comparable only if all three match; :func:`diff_runs` raises :class:`VibeMismatch`
naming the fields that differ rather than returning a diff that means nothing. The chat template
(``template="chat"``) renders through :func:`scratch_llm.chat.render_for_completion` and stops at
``<|eot|>``, so the run measures the thing SFT is supposed to teach — *stopping*. The raw template
(``template="raw"``) feeds the prompt text straight in and always runs to the budget; a base
checkpoint has no turn structure to terminate, and pretending otherwise would score the pretrain
card on a behaviour it was never trained for. The two are not comparable, and the fingerprint says so.

Interview question this answers: "how do you know the SFT checkpoint is better than the base one
and not just luckier?" — identical prompts, identical decode, identical seeds, token-level diff.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

from scratch_llm.chat import CHAT_SPECIAL_TOKENS, EOT, Message, render_for_completion
from scratch_llm.model import TransformerLM
from scratch_llm.sampling import SamplingParams, generate
from scratch_llm.tokenizer import Tokenizer

Template = Literal["raw", "chat"]

# Bump this id whenever VIBE_PROMPTS changes. Cards carry it, and a card evaluated on "vibe-v1"
# will refuse to be compared with one evaluated on "vibe-v2" — which is the point: a prompt-set
# edit is a new experiment, not a continuation of the old one.
VIBE_SET_ID = "vibe-v1"


class VibeMismatch(ValueError):
    """Raised when two vibe runs are not comparable (different prompts, template, or decode)."""


@dataclass(frozen=True)
class VibePrompt:
    """One fixed probe. ``id`` is the stable key: seeds, diffs and stored outputs join on it."""

    id: str
    text: str
    category: str


# The fixed set. Short, ASCII, and drawn from behaviours a d20-scale chat model can plausibly
# show: greeting, self-description, a fact, one-digit arithmetic, list-following, definition,
# plain continuation, code, summarisation, an unanswerable question, an instruction with a count
# in it, and a one-sentence explanation. Each is a *probe*, not a benchmark — the graded numbers
# live on the report card (CORE, val_bpb); what these measure is whether the model talks, and
# whether it stops.
VIBE_PROMPTS: tuple[VibePrompt, ...] = (
    VibePrompt("greet", "say hi", "social"),
    VibePrompt("identity", "who are you?", "social"),
    VibePrompt("fact_capital", "what is the capital of France?", "factual"),
    VibePrompt("arith_small", "what is 2 + 3?", "arithmetic"),
    VibePrompt("list_three", "list three colors.", "instruction"),
    VibePrompt("define_lm", "what is a language model?", "definition"),
    VibePrompt("continue_fox", "the quick brown fox", "continuation"),
    VibePrompt("code_add", "write a python function that adds two numbers.", "code"),
    VibePrompt(
        "summarize", "summarize: the transformer reads the whole context at once.", "summary"
    ),
    VibePrompt("unknowable", "what did i eat for breakfast?", "honesty"),
    VibePrompt("repeat_count", "repeat the word banana three times.", "instruction"),
    VibePrompt("explain_attn", "explain attention in one sentence.", "explanation"),
)


@dataclass(frozen=True)
class VibeDecode:
    """The decoding contract. Frozen, hashed into every run, and compared before any diff.

    ``template`` chooses how a prompt becomes ids: ``"chat"`` renders the chat template and stops
    at ``<|eot|>`` (needs a tokenizer trained with :data:`~scratch_llm.chat.CHAT_SPECIAL_TOKENS`);
    ``"raw"`` encodes the prompt text and decodes to the budget.
    """

    template: Template = "chat"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 96
    seed: int = 0

    @property
    def id(self) -> str:
        return (
            f"{self.template}-t{self.temperature:g}-p{self.top_p:g}-n{self.max_tokens}-s{self.seed}"
        )


def prompt_seed(base_seed: int, prompt_id: str) -> int:
    """A per-prompt seed derived from ``(base_seed, prompt_id)``.

    Deriving instead of sharing one RNG stream is what makes a run order-invariant: prompt #7's
    tokens do not depend on how many tokens prompts #0..#6 consumed, so re-running a subset, or
    the same set shuffled, reproduces every output exactly. blake2b (not :func:`hash`) because
    Python's string hash is salted per process.
    """
    digest = hashlib.blake2b(f"{base_seed}:{prompt_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def prompts_fingerprint(prompts: Sequence[VibePrompt]) -> str:
    """sha256 over (id, text) of the prompts actually used — catches a silent edit of the set."""
    payload = json.dumps([[p.id, p.text] for p in prompts], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class VibeOutput:
    """One prompt's stored result. ``token_ids`` is the comparison key; ``text`` is for humans."""

    prompt_id: str
    token_ids: tuple[int, ...]
    text: str
    eot_terminated: bool
    n_new_tokens: int

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_id": self.prompt_id,
            "token_ids": list(self.token_ids),
            "text": self.text,
            "eot_terminated": self.eot_terminated,
            "n_new_tokens": self.n_new_tokens,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> VibeOutput:
        ids = cast(Sequence[int], d["token_ids"])
        return cls(
            prompt_id=str(d["prompt_id"]),
            token_ids=tuple(int(t) for t in ids),
            text=str(d["text"]),
            eot_terminated=bool(d["eot_terminated"]),
            n_new_tokens=int(cast(int, d["n_new_tokens"])),
        )


@dataclass(frozen=True)
class VibeRun:
    """Every stored output of one checkpoint on one (prompt set × decode config)."""

    set_id: str
    prompts_hash: str
    decode: VibeDecode
    outputs: tuple[VibeOutput, ...]

    @property
    def fingerprint(self) -> str:
        """The identity of the *evaluation* — what must match before two runs may be diffed."""
        return f"{self.set_id}/{self.prompts_hash}/{self.decode.id}"

    def by_prompt(self) -> dict[str, VibeOutput]:
        return {o.prompt_id: o for o in self.outputs}

    def to_dict(self) -> dict[str, object]:
        return {
            "set_id": self.set_id,
            "prompts_hash": self.prompts_hash,
            "decode": {
                "template": self.decode.template,
                "temperature": self.decode.temperature,
                "top_p": self.decode.top_p,
                "max_tokens": self.decode.max_tokens,
                "seed": self.decode.seed,
            },
            "fingerprint": self.fingerprint,
            "outputs": [o.to_dict() for o in self.outputs],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> VibeRun:
        dec = cast(Mapping[str, object], d["decode"])
        template = str(dec["template"])
        if template not in ("raw", "chat"):
            raise VibeMismatch(f"unknown vibe template {template!r} (expected 'raw' or 'chat')")
        outputs = cast(Sequence[Mapping[str, object]], d["outputs"])
        return cls(
            set_id=str(d["set_id"]),
            prompts_hash=str(d["prompts_hash"]),
            decode=VibeDecode(
                template=cast(Template, template),
                temperature=float(cast(float, dec["temperature"])),
                top_p=float(cast(float, dec["top_p"])),
                max_tokens=int(cast(int, dec["max_tokens"])),
                seed=int(cast(int, dec["seed"])),
            ),
            outputs=tuple(VibeOutput.from_dict(o) for o in outputs),
        )


def _special_ids(tokenizer: Tokenizer) -> set[int]:
    """Ids of the chat specials, when the tokenizer carries them; empty set otherwise."""
    ids: set[int] = set()
    for special in CHAT_SPECIAL_TOKENS:
        encoded = tokenizer.encode(special)
        if len(encoded) == 1:
            ids.add(encoded[0])
    return ids


def _clean(gen_ids: Sequence[int], tokenizer: Tokenizer, specials: set[int]) -> str:
    """Human-readable text: drop the scaffolding ids, decode the rest. Never used for equality."""
    return tokenizer.decode([t for t in gen_ids if t not in specials])


def run_vibe_evals(
    model: TransformerLM,
    tokenizer: Tokenizer,
    *,
    decode: VibeDecode | None = None,
    prompts: Sequence[VibePrompt] = VIBE_PROMPTS,
    device: str = "cpu",
) -> VibeRun:
    """Run the fixed prompt set through ``model`` under ``decode`` and store the outputs.

    Deterministic by construction: ``model.eval()``, one fresh conversation per prompt (no history
    carried between probes), and a per-prompt seed from :func:`prompt_seed`. Calling this twice on
    the same checkpoint returns runs whose ``token_ids`` are equal, at any temperature.
    """
    if not prompts:
        raise ValueError("prompts must be non-empty")
    dec = decode or VibeDecode()
    model.eval()

    specials = _special_ids(tokenizer)
    eot_ids = tokenizer.encode(EOT)
    if dec.template == "chat":
        if len(eot_ids) != 1:
            raise VibeMismatch(
                f"template='chat' needs a tokenizer trained with CHAT_SPECIAL_TOKENS — {EOT!r} "
                f"encodes to {len(eot_ids)} ids (train it with SpeedrunConfig.chat=True), or use "
                "template='raw' for a base checkpoint."
            )
        eot_id = eot_ids[0]
        stop_ids: tuple[int, ...] = (eot_id,)
    else:
        eot_id = eot_ids[0] if len(eot_ids) == 1 else -1
        stop_ids = ()

    outputs: list[VibeOutput] = []
    for prompt in prompts:
        if dec.template == "chat":
            prompt_ids = render_for_completion([Message("user", prompt.text)], tokenizer)
        else:
            prompt_ids = tokenizer.encode(prompt.text)
        params = SamplingParams(
            temperature=dec.temperature,
            top_p=dec.top_p,
            max_tokens=dec.max_tokens,
            stop_ids=stop_ids,
            seed=prompt_seed(dec.seed, prompt.id),
        )
        gen = generate(model, prompt_ids, params, device=device)
        terminated = bool(gen) and gen[-1] == eot_id
        outputs.append(
            VibeOutput(
                prompt_id=prompt.id,
                token_ids=tuple(int(t) for t in gen),
                text=_clean(gen, tokenizer, specials),
                eot_terminated=terminated,
                n_new_tokens=len(gen),
            )
        )
    return VibeRun(
        set_id=VIBE_SET_ID,
        prompts_hash=prompts_fingerprint(prompts),
        decode=dec,
        outputs=tuple(outputs),
    )


def vibe_metrics(run: VibeRun) -> dict[str, float]:
    """The scalar summary that lands on a report card.

    ``vibe_eot_rate`` exists only for the chat template: the fraction of probes the model ended
    itself, which is precisely what the SFT mask teaches (assistant content **plus** the closing
    eot). Under the raw template there is no turn to end, so the key is absent rather than 0.0 —
    an absent metric is an error at card construction, a 0.0 would be a lie that reads as a
    measurement.
    """
    n = len(run.outputs)
    if n == 0:
        raise ValueError("vibe run has no outputs")
    metrics = {
        "vibe_mean_new_tokens": sum(o.n_new_tokens for o in run.outputs) / n,
        "vibe_truncated_rate": sum(
            1 for o in run.outputs if o.n_new_tokens >= run.decode.max_tokens
        )
        / n,
        "vibe_empty_text_rate": sum(1 for o in run.outputs if not o.text.strip()) / n,
    }
    if run.decode.template == "chat":
        metrics["vibe_eot_rate"] = sum(1 for o in run.outputs if o.eot_terminated) / n
    return metrics


def diff_runs(a: VibeRun, b: VibeRun) -> tuple[str, ...]:
    """Prompt ids whose generated token ids differ. Raises if the two runs are not comparable.

    The refusal is the feature. Two runs on different prompt sets, templates or decode configs
    have no meaningful diff, and returning one would let a prompt-set edit read as a training
    effect.
    """
    if a.set_id != b.set_id or a.prompts_hash != b.prompts_hash:
        raise VibeMismatch(
            f"prompt set differs: {a.set_id}/{a.prompts_hash} vs {b.set_id}/{b.prompts_hash} — "
            "re-run both checkpoints on one set before comparing them."
        )
    if a.decode.id != b.decode.id:
        raise VibeMismatch(
            f"decode config differs: {a.decode.id} vs {b.decode.id} — a difference measured "
            "across two samplers is a difference between samplers."
        )
    left, right = a.by_prompt(), b.by_prompt()
    if left.keys() != right.keys():
        raise VibeMismatch(
            f"prompt ids differ: {sorted(left.keys() ^ right.keys())} appear in one run only."
        )
    return tuple(pid for pid in left if left[pid].token_ids != right[pid].token_ids)


def runs_identical(a: VibeRun, b: VibeRun) -> bool:
    """True iff every prompt produced the same token ids. Raises on incomparable runs."""
    return not diff_runs(a, b)


def with_seed(decode: VibeDecode, seed: int) -> VibeDecode:
    """A copy of ``decode`` at a new base seed — the only sanctioned way to move the sampler."""
    return replace(decode, seed=seed)


__all__ = [
    "VIBE_PROMPTS",
    "VIBE_SET_ID",
    "Template",
    "VibeDecode",
    "VibeMismatch",
    "VibeOutput",
    "VibePrompt",
    "VibeRun",
    "diff_runs",
    "prompt_seed",
    "prompts_fingerprint",
    "run_vibe_evals",
    "runs_identical",
    "vibe_metrics",
    "with_seed",
]
