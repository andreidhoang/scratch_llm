"""A6 chat REPL — talk to the SFT'd model (the close-the-loop payoff: a model you converse with).

:class:`ChatSession` holds a growing list of :class:`~scratch_llm.chat.Message` turns and, on each
``reply``, renders the whole history with :func:`~scratch_llm.chat.render_for_completion` (which
ends on the ``<|assistant|>`` id), greedily decodes with a ``stop_ids=(<|eot|>,)`` budget, and
surfaces the assistant's text with the eot and any special ids stripped. The turn terminator is
the crux: an SFT'd model that learned the mask emits ``<|eot|>`` and the session stops *before*
``max_tokens``; a model that never emits it runs to the budget — exactly the A6 kill criterion.

:func:`batch_reply` is the optional throughput path: it routes equal-length single-turn prompts
through the perf-front's public :func:`~scratch_llm.serving.batched.batched_greedy_decode` (R3a
static batch — one weight read serves all rows), whose contract is *row b == single-stream greedy
of prompt b*, so batched replies equal per-turn greedy. This consumes ``serving/`` through its
public function only (zone rule); nothing here edits the serving layer.

Interview question this answers: "you've SFT'd a chat model — how does the serving loop know when
to stop a turn, and why must the stop token be a single trained id rather than a decoded string?"
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field

from scratch_llm.chat import CHAT_SPECIAL_TOKENS, EOT, Message, render_for_completion
from scratch_llm.model import TransformerLM
from scratch_llm.sampling import SamplingParams, generate
from scratch_llm.serving.batched import batched_greedy_decode
from scratch_llm.tokenizer import Tokenizer


def _special_ids(tokenizer: Tokenizer) -> set[int]:
    """The set of chat-special ids — stripped from surfaced text so no scaffolding leaks out."""
    return {tokenizer.encode(s)[0] for s in CHAT_SPECIAL_TOKENS}


def _clean_answer(
    gen_ids: Sequence[int], tokenizer: Tokenizer, eot_id: int, special_ids: set[int]
) -> str:
    """Truncate the generation at the first eot, drop any special ids, and decode the rest.

    Both the single-stream and batched paths funnel through here, so they surface identical text
    for identical greedy token streams (the batched path may run past eot to its budget; the
    truncation makes that invisible)."""
    answer: list[int] = []
    for tid in gen_ids:
        if tid == eot_id:
            break
        if tid in special_ids:
            continue
        answer.append(tid)
    return tokenizer.decode(answer)


@dataclass
class ChatSession:
    """A stateful multi-turn chat over a trained model. ``reply`` appends the user turn, generates,
    appends the assistant turn, and returns the assistant text."""

    model: TransformerLM
    tokenizer: Tokenizer
    max_tokens: int = 256
    temperature: float = 0.0
    device: str = "cpu"
    history: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._eot_id = self.tokenizer.encode(EOT)[0]
        self._special_ids = _special_ids(self.tokenizer)

    def reply(self, user_text: str) -> str:
        """Add a user turn, greedily generate the assistant turn (stopping at eot), return its text."""
        self.history.append(Message("user", user_text))
        prompt = render_for_completion(self.history, self.tokenizer)
        gen = generate(
            self.model,
            prompt,
            SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop_ids=(self._eot_id,),
            ),
            device=self.device,
        )
        text = _clean_answer(gen, self.tokenizer, self._eot_id, self._special_ids)
        self.history.append(Message("assistant", text))
        return text

    def reset(self) -> None:
        """Clear the conversation history (start a fresh dialogue with the same model)."""
        self.history.clear()


def batch_reply(
    model: TransformerLM,
    tokenizer: Tokenizer,
    user_texts: Sequence[str],
    *,
    max_tokens: int = 256,
    device: str = "cpu",
) -> list[str]:
    """Greedy-reply to independent single-turn user messages via the serving batched path.

    Renders each ``user_text`` to a completion prompt and routes them through
    :func:`~scratch_llm.serving.batched.batched_greedy_decode` (which requires equal-length
    prompts — an R3a static batch — and raises otherwise). Each surfaced reply equals the
    corresponding single-stream :meth:`ChatSession.reply` because the underlying decode is greedy
    and per-row identical to single-stream greedy."""
    prompts = [render_for_completion([Message("user", t)], tokenizer) for t in user_texts]
    gens = batched_greedy_decode(model, prompts, max_tokens=max_tokens, device=device)
    eot_id = tokenizer.encode(EOT)[0]
    special_ids = _special_ids(tokenizer)
    return [_clean_answer(g, tokenizer, eot_id, special_ids) for g in gens]


def repl(
    model: TransformerLM, tokenizer: Tokenizer, *, max_tokens: int = 256, device: str = "cpu"
) -> None:  # pragma: no cover - interactive
    """Minimal read-eval-print loop: type a line, get a reply; Ctrl-D / empty EOF to exit."""
    session = ChatSession(model, tokenizer, max_tokens=max_tokens, device=device)
    print("chat — Ctrl-D to exit")
    while True:
        try:
            user = input("you> ")
        except EOFError:
            print()
            break
        print("bot>", session.reply(user))


def main() -> None:  # pragma: no cover - CLI entry
    from scratch_llm.data.shards import load_tokenizer
    from scratch_llm.train import build_model_from_checkpoint

    ap = argparse.ArgumentParser(description="Chat with an SFT'd checkpoint.")
    ap.add_argument(
        "--ckpt", required=True, help="config-carrying checkpoint (train.save_checkpoint)"
    )
    ap.add_argument("--data-dir", required=True, help="dir with the staged tokenizer.json")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, _ = build_model_from_checkpoint(args.ckpt, map_location=args.device)
    model.to(args.device)
    tokenizer = load_tokenizer(args.data_dir)
    repl(model, tokenizer, max_tokens=args.max_tokens, device=args.device)


__all__ = ["ChatSession", "batch_reply", "repl"]


if __name__ == "__main__":  # pragma: no cover
    main()
