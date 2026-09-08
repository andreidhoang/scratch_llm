"""S1/S-R2 — the FP8 quality gate's two arms, on HF-transformers Qwen3-8B.

S-R2's claim has two halves: FP8 weights make our engine faster (a speedup against
``bench/s1_fp8_serving.py``'s bf16 arm), and they do not cost quality (Δppl plus a paired
GSM8K accuracy drop, judged by ``quant/s_r2_gate``). **This module serves only the second
half**, and is deliberately incapable of serving the first.

Why it is separate from the engine that produces the speedup
------------------------------------------------------------
``s1_fp8_serving.resolve_engine_factory`` refuses to substitute anything for the bf16 floor,
and it is right to: a speedup measured against a different engine is a comparison to nothing
(workspace invariant 2). This module would be exactly such a substitute if it were handed to
``--engine``, so :meth:`QualityArm.decode_batch` raises rather than returning a token
count. There is no configuration of this file that yields a tok/s.

The quality question, unlike the throughput question, is a property of the *quantization
scheme* — which tensors receive E4M3 codes and what scale each output channel carries — and
not of the engine that multiplies them. ``quant.fp8_weights.convert_linears_to_fp8`` is that
scheme, and it is the identical object in both places. Running it over HF's Qwen3-8B answers
"does this scheme cost accuracy" without blocking on a weight loader for our own stack.

What a PASS here does NOT establish
-----------------------------------
Our engine's FP8 GEMM path. ``convert_linears_to_fp8`` dispatches on
``isinstance(m, nn.Linear)``; HF's Qwen3 uses real ``nn.Linear``, so the conversion lands
here. Our own ``model.Linear`` is **not** ``nn.Linear`` (``model.py:125``), so the same call
is a silent no-op on our stack — the trap recorded in CLAUDE.md. A numerical test on the fp8
GEMM itself is a separate obligation and is not discharged by a verdict from this file.
Both facts belong in the rung's write-up; neither is inferable from the number.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from scratch_llm.quant.fp8_weights import DEFAULT_FAST_ACCUM, convert_linears_to_fp8

__all__ = [
    "CORPORA",
    "GSM8K_SLICES",
    "QualityArm",
    "build_quality_arms",
    "extract_final_number",
    "gsm8k_gold",
]

#: Qwen3-8B on the Hub. Named once; both arms must load the same id or the pairing is void.
DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"

#: Corpus ids the spec is allowed to name, resolved to an exact `datasets` triple. A corpus is
#: part of the measurement's identity ("Δppl" without the stream is not reproducible), so an
#: unknown id raises instead of falling back to something similar.
CORPORA: dict[str, tuple[str, str | None, str]] = {
    "wikitext2-test": ("wikitext", "wikitext-2-raw-v1", "test"),
}

#: Likewise for the GSM8K slice. The slice string carries the item count, and the gate pairs
#: items positionally, so "gsm8k-test" without a bound is not a slice and is not accepted.
GSM8K_SLICES: dict[str, tuple[str, str, str]] = {
    "gsm8k-test[:200]": ("gsm8k", "main", "test[:200]"),
}

#: Teacher-forced scoring window. Fixed rather than derived from the config so the transition
#: count is a property of (corpus, tokenizer, this constant) and cannot drift with a model's
#: max_position_embeddings.
NLL_WINDOW = 2048

#: Greedy decode budget per GSM8K item. Greedy, so both arms are deterministic and any
#: difference in the flags is attributable to the weights rather than to sampling.
GSM8K_MAX_NEW_TOKENS = 256

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_final_number(text: str) -> str | None:
    """Last number in ``text``, commas stripped, or None when there is none.

    GSM8K answers are graded on the final numeric token the model emits. Taking the *last*
    match rather than the first is what makes chain-of-thought gradeable: the reasoning is
    full of intermediate numbers and only the final one is the answer.
    """
    matches = _NUMBER.findall(text)
    if not matches:
        return None
    cleaned = matches[-1].replace(",", "").rstrip(".")
    return cleaned or None


def gsm8k_gold(answer: str) -> str:
    """The reference answer, which GSM8K places after a ``####`` marker."""
    _, _, tail = answer.rpartition("####")
    gold = extract_final_number(tail if tail else answer)
    if gold is None:
        raise ValueError(f"no gold number in GSM8K answer: {answer[:120]!r}")
    return gold


@dataclass
class _Item:
    question: str
    gold: str


class QualityArm:
    """One arm of the quality comparison: an HF causal LM, optionally FP8-quantized.

    Satisfies the ``score_nll`` / ``gsm8k_correct`` half of ``s1_fp8_serving.ServingEngine``
    so ``collect_evidence`` can be reused verbatim — the pairing and token-count checks that
    make the two arms comparable live there, and duplicating them here would be a second
    place for them to be wrong. ``decode_batch`` is present and raises, so the same object
    can never be mistaken for a throughput engine.
    """

    def __init__(self, arm: str, model, tokenizer, *, converted: Sequence[str] = ()) -> None:
        self.arm = arm
        self._model = model
        self._tok = tokenizer
        #: Which layers the FP8 conversion actually replaced. Empty on the bf16 arm; on the
        #: fp8 arm an empty list means the conversion silently no-opped, which is the
        #: `model.Linear` trap and is checked at construction by `build_quality_arms`.
        self.converted = tuple(converted)

    def decode_batch(self, shape: object) -> int:
        raise NotImplementedError(
            "QualityArm measures quality only and has no throughput number. S-R2's "
            "speedup is against our own bf16 engine at matched model and shape "
            "(s1_fp8_serving.resolve_engine_factory); a tok/s from HF-transformers would be "
            "a comparison to a different engine, which is a comparison to nothing."
        )

    @torch.no_grad()
    def score_nll(self, corpus: str) -> tuple[float, int]:
        """Summed teacher-forced NLL in nats and the number of transitions scored.

        Non-overlapping windows of ``NLL_WINDOW`` tokens over the concatenated corpus. Each
        window contributes ``len(window) - 1`` transitions, and the trailing partial window
        is dropped so both arms see an identical stream regardless of tokenizer edge cases.
        """
        ids = self._corpus_ids(corpus)
        device = next(self._model.parameters()).device
        total_nats = 0.0
        total_transitions = 0
        n_windows = ids.numel() // NLL_WINDOW
        if n_windows == 0:
            raise RuntimeError(
                f"corpus {corpus!r} tokenizes to {ids.numel()} tokens, shorter than one "
                f"{NLL_WINDOW}-token window — nothing to score"
            )
        for w in range(n_windows):
            window = ids[w * NLL_WINDOW : (w + 1) * NLL_WINDOW].unsqueeze(0).to(device)
            logits = self._model(window).logits.float()
            # Predict token t+1 from position t: drop the last logit and the first target.
            loss = nn.functional.cross_entropy(logits[0, :-1, :], window[0, 1:], reduction="sum")
            total_nats += float(loss)
            total_transitions += NLL_WINDOW - 1
        return total_nats, total_transitions

    @torch.no_grad()
    def gsm8k_correct(self, slice_id: str) -> Sequence[bool]:
        """One flag per item of the named slice, **in slice order**, greedy-decoded."""
        items = self._gsm8k_items(slice_id)
        device = next(self._model.parameters()).device
        flags: list[bool] = []
        for item in items:
            prompt = f"Question: {item.question}\nAnswer:"
            enc = self._tok(prompt, return_tensors="pt").to(device)
            out = self._model.generate(
                **enc,
                max_new_tokens=GSM8K_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
            )
            completion = self._tok.decode(
                out[0, enc["input_ids"].shape[-1] :], skip_special_tokens=True
            )
            flags.append(extract_final_number(completion) == item.gold)
        return tuple(flags)

    # ── data resolution ─────────────────────────────────────────────────────────────────
    def _corpus_ids(self, corpus: str) -> torch.Tensor:
        if corpus not in CORPORA:
            raise KeyError(
                f"unknown corpus {corpus!r}; known: {sorted(CORPORA)}. A corpus is part of "
                "this measurement's identity — substituting a similar one silently would "
                "make the two arms' Δppl unreproducible."
            )
        name, config, split = CORPORA[corpus]
        rows = _load_dataset(name, config, split)
        text = "\n\n".join(r["text"] for r in rows if r["text"].strip())
        return self._tok(text, return_tensors="pt")["input_ids"][0]

    def _gsm8k_items(self, slice_id: str) -> list[_Item]:
        if slice_id not in GSM8K_SLICES:
            raise KeyError(
                f"unknown GSM8K slice {slice_id!r}; known: {sorted(GSM8K_SLICES)}. The gate "
                "pairs items positionally, so the slice bound is part of the comparison."
            )
        name, config, split = GSM8K_SLICES[slice_id]
        rows = _load_dataset(name, config, split)
        return [_Item(question=r["question"], gold=gsm8k_gold(r["answer"])) for r in rows]


def _load_dataset(name: str, config: str | None, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised on the box, not the laptop
        raise RuntimeError(
            "the `datasets` package is required for S-R2's quality gate (declared in "
            "scratch_llm/pyproject.toml). infra/bootstrap.sh installs it on the box."
        ) from exc
    return load_dataset(name, config, split=split)


def build_quality_arms(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
    use_fast_accum: bool = DEFAULT_FAST_ACCUM,
) -> tuple[QualityArm, QualityArm]:
    """Load Qwen3-8B twice — bf16 and FP8 — and return ``(bf16_arm, fp8_arm)``.

    Two separate loads rather than one model quantized in place: the gate compares the arms
    on the same items in the same process, so both must exist simultaneously. The FP8 arm is
    built by converting a freshly loaded bf16 copy, so the only difference between the arms
    is the conversion — not a load order, not a dtype cast applied twice.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised on the box, not the laptop
        raise RuntimeError(
            "the `transformers` package is required for S-R2's quality gate (declared in "
            "scratch_llm/pyproject.toml). infra/bootstrap.sh installs it on the box."
        ) from exc

    tok = AutoTokenizer.from_pretrained(model_id)

    def _load():
        m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        return m.to(device).eval()

    bf16 = QualityArm("bf16", _load(), tok)
    fp8_model = _load()
    converted = convert_linears_to_fp8(fp8_model, use_fast_accum=use_fast_accum)
    if not converted:
        raise RuntimeError(
            f"convert_linears_to_fp8 replaced nothing in {model_id}. It dispatches on "
            "isinstance(m, nn.Linear); a model whose projections are a custom class is a "
            "silent no-op, and the fp8 arm would be bf16 wearing an fp8 label. See "
            "CLAUDE.md's `model.Linear is not nn.Linear` trap."
        )
    return bf16, QualityArm("fp8", fp8_model, tok, converted=converted)
