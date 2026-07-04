"""Shared structural types for the eval harness — keeps `eval/` decoupled from the concrete
`Tokenizer`/model classes so a stub satisfies the same contract in tests."""

from __future__ import annotations

from typing import Protocol


class TextTokenizer(Protocol):
    """The tokenizer surface the report card needs: text ↔ token ids. `scratch_llm.tokenizer.
    Tokenizer` satisfies it structurally, and so does any stub with the same two methods."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...
