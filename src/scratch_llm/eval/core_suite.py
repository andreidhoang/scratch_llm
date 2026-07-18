"""DCLM CORE task suite — the official 22-task benchmark behind nanochat's headline metric (P3).

Clean-room re-implementation of the CORE evaluation recipe (DCLM, arXiv:2406.11794) against our
``TransformerLM`` / ``Tokenizer`` APIs, so ``core_style_score`` (report_card.py) is finally fed the
*official* suite and the d20's headline number is comparable to nanochat's published anchors
(original d20 CORE 0.2219 · GPT-2 XL 0.2565).

Invariant: for every task the scoring rule, prompt template, few-shot sampling, subsampling
order, and centering formula match nanochat's implementation (verified against
``github.com/karpathy/nanochat`` @ HEAD, fetched 2026-07-17):

- **Prompt templates** — ``nanochat/core_eval.py:17-83`` (``render_prompts_{mc,schema,lm}``; the
  Jinja templates reduce to the f-strings used here; LM ``context`` is ``strip()``-ed and the
  without-continuation prompt is additionally ``strip()``-ed, core_eval.py:82).
- **Continuation region** — common token *prefix* across MC choices / common token *suffix*
  across schema contexts (``find_common_length`` + ``batch_sequences_*``, core_eval.py:86-141);
  LM region = tokens of the with-continuation prompt past the without-prompt prefix.
- **Scoring** — MC/schema: argmin of mean cross-entropy over the region == argmax of
  length-normalized log-prob (core_eval.py:232-237); LM: correct iff the teacher-forced argmax
  matches *every* region token (core_eval.py:224-231) — equivalent to greedy-decode exact match.
- **Few-shot sampling** — ``random.Random(1234 + idx)`` over the other examples
  (core_eval.py:176-181); **subsampling** — shuffle with ``random.Random(1337)`` then truncate
  (scripts/base_eval.py:103-107); **truncation** — keep the *last* ``max_seq_len`` tokens,
  shifting the region (core_eval.py:196-213); a BOS id is prepended to every sequence
  (core_eval.py:115/125/135).
- **Centering / aggregate** — ``(acc − b) / (1 − b)`` with per-task random baselines and CORE =
  unweighted mean (scripts/base_eval.py:112,117) — computed here by *reusing*
  :func:`scratch_llm.eval.report_card.core_style_score`, not re-deriving it.
- **Manifest** — the 22 tasks, dataset files, few-shot counts, task types, and continuation
  delimiters are transcribed from ``eval_bundle/core.yaml``; random baselines (percent) from
  ``eval_bundle/eval_meta_data.csv`` (bundle @ ``EVAL_BUNDLE_URL``, Last-Modified 2025-10-09).

Data source: the nanochat eval bundle itself — a plain public S3 zip (no auth, no new
dependency), which is *bit-identical* to the data nanochat scores, the strongest compatibility
choice per task. Downloaded lazily at RUNTIME by the task loaders (``ensure_eval_bundle``);
importing this module and constructing :class:`TaskSpec` lists touches no network.

Known deltas from nanochat (documented, not silent):
- **Tokenizer**: ours (byte-level BPE, ``<|eot|>`` doc-separator as the BOS analog) vs nanochat's
  GPT-4-style BPE with ``<|bos|>``. The recipe is identical but CORE is tokenizer-sensitive
  (especially the exact-match LM tasks), so our absolute numbers are comparable in *recipe*, not
  guaranteed equal per-task. The measured-vs-nanochat validation run happens at P5 on a real
  checkpoint.
- Scoring runs in fp32 (``log_softmax(logits.float())``); nanochat scores in the model's autocast
  dtype. Negligible, but not bit-identical.
- A degenerate empty region (one option a strict token-prefix of another) scores ``-inf`` /
  incorrect here; nanochat computes ``nan`` and its ``min()`` is order-dependent
  (core_eval.py:234-236). Identical behavior whenever regions are non-empty (the practical case).
- Few-shot k is capped at ``len(data) − 1``; nanochat raises when a subsample is smaller than the
  shot count. Only reachable under ``limit`` smoke runs.
- Inherited caveat: nanochat's own header notes its squad accuracy trails the DCLM reference
  (31% vs 37%, core_eval.py:6) — matching nanochat means inheriting that gap.
"""

from __future__ import annotations

import json
import random
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from scratch_llm.eval.protocols import TextTokenizer
from scratch_llm.eval.report_card import core_style_score

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

# nanochat's seeds — load-bearing for prompt-level reproducibility (see module docstring).
_SHUFFLE_SEED = 1337  # scripts/base_eval.py:104
_FEWSHOT_SEED = 1234  # nanochat/core_eval.py:178

TaskType = Literal["multiple_choice", "schema", "language_modeling"]
_TASK_TYPES: frozenset[str] = frozenset(("multiple_choice", "schema", "language_modeling"))

Example = Mapping[str, Any]


# ---------------------------------------------------------------------------------------------
# TaskSpec — one CORE task: where its data comes from and how it is prompted/scored/centered
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    """One evaluation task. ``loader`` is called lazily at evaluation time (never at import),
    so a spec for a network-backed task costs nothing to construct."""

    name: str
    task_type: TaskType
    loader: Callable[[], Sequence[Example]]
    random_baseline: float  # fraction in [0, 1) — the centering point
    num_fewshot: int = 0
    continuation_delimiter: str = " "  # base_eval.py:95 default

    def __post_init__(self) -> None:
        if self.task_type not in _TASK_TYPES:
            raise ValueError(f"unknown task_type {self.task_type!r} (must be one of {_TASK_TYPES})")
        if not 0.0 <= self.random_baseline < 1.0:
            raise ValueError(f"random_baseline must be in [0, 1), got {self.random_baseline}")
        if self.num_fewshot < 0:
            raise ValueError(f"num_fewshot must be >= 0, got {self.num_fewshot}")


# ---------------------------------------------------------------------------------------------
# Prompt builders — nanochat's Jinja templates reduced to f-strings (core_eval.py:17-83)
# ---------------------------------------------------------------------------------------------


def render_prompts_mc(item: Example, delimiter: str, fewshot: Sequence[Example] = ()) -> list[str]:
    """One prompt per choice: few-shot blocks ``{query}{delim}{gold choice}\\n\\n`` then
    ``{query}{delim}{choice}`` (core_eval.py:17-33)."""
    prefix = "".join(f"{ex['query']}{delimiter}{ex['choices'][ex['gold']]}\n\n" for ex in fewshot)
    return [f"{prefix}{item['query']}{delimiter}{choice}" for choice in item["choices"]]


def render_prompts_schema(
    item: Example, delimiter: str, fewshot: Sequence[Example] = ()
) -> list[str]:
    """One prompt per context option: few-shot blocks ``{gold context}{delim}{continuation}\\n\\n``
    then ``{context_option}{delim}{continuation}`` (core_eval.py:36-53)."""
    prefix = "".join(
        f"{ex['context_options'][ex['gold']]}{delimiter}{ex['continuation']}\n\n" for ex in fewshot
    )
    return [
        f"{prefix}{option}{delimiter}{item['continuation']}" for option in item["context_options"]
    ]


def render_prompts_lm(
    item: Example, delimiter: str, fewshot: Sequence[Example] = ()
) -> tuple[str, str]:
    """The ``(without, with)`` continuation pair. Contexts are ``strip()``-ed (the Jinja ``trim``)
    and the without-prompt is additionally ``strip()``-ed so no trailing delimiter whitespace
    breaks the token-prefix property (core_eval.py:56-83)."""
    prefix = "".join(
        f"{ex['context'].strip()}{delimiter}{ex['continuation']}\n\n" for ex in fewshot
    )
    stem = f"{prefix}{item['context'].strip()}{delimiter}"
    return stem.strip(), f"{stem}{item['continuation']}"


def common_prefix_len(sequences: Sequence[Sequence[int]]) -> int:
    """Length of the shared token prefix, bounded by the shortest sequence (core_eval.py:86-101)."""
    min_len = min(len(seq) for seq in sequences)
    first = sequences[0]
    for i in range(min_len):
        if not all(seq[i] == first[i] for seq in sequences):
            return i
    return min_len


def common_suffix_len(sequences: Sequence[Sequence[int]]) -> int:
    """Length of the shared token suffix, bounded by the shortest sequence (core_eval.py:86-101)."""
    min_len = min(len(seq) for seq in sequences)
    first = sequences[0]
    for i in range(min_len):
        if not all(seq[-1 - i] == first[-1 - i] for seq in sequences):
            return i
    return min_len


def fewshot_indices(idx: int, n_data: int, num_fewshot: int) -> list[int]:
    """The few-shot example indices for item ``idx``: sampled with ``Random(1234 + idx)`` from
    every other index (core_eval.py:176-181). k caps at ``n_data − 1`` (documented delta)."""
    if num_fewshot <= 0:
        return []
    rng = random.Random(_FEWSHOT_SEED + idx)
    available = [i for i in range(n_data) if i != idx]
    return rng.sample(available, min(num_fewshot, len(available)))


# ---------------------------------------------------------------------------------------------
# Row assembly — (token sequence, region start, region end) per forward-pass row
# ---------------------------------------------------------------------------------------------

# One scoring row: token ids plus the [start, end) region whose tokens are scored.
_Row = tuple[list[int], int, int]


@dataclass(frozen=True)
class _ExampleRows:
    rows: list[_Row]
    gold: int | None  # option index for MC/schema; None ⇒ language modeling (exact match)


def _truncate_row(row: _Row, max_seq_len: int, name: str) -> _Row:
    """Keep the LAST ``max_seq_len`` tokens, shifting the region (core_eval.py:196-213). Raises
    when the region itself would be cropped (nanochat asserts the same condition)."""
    seq, start, end = row
    if len(seq) <= max_seq_len:
        return row
    crop = len(seq) - max_seq_len
    if start - crop < 1:  # scoring reads logits[start-1]; the region must survive the crop
        raise ValueError(
            f"{name}: sequence of {len(seq)} tokens leaves no room for the continuation region "
            f"under max_seq_len={max_seq_len} (region starts at {start})"
        )
    return seq[crop:], start - crop, end - crop


def _rows_for_example(
    item: Example,
    idx: int,
    data: Sequence[Example],
    spec: TaskSpec,
    tokenizer: TextTokenizer,
    *,
    bos_id: int | None,
    max_seq_len: int | None,
) -> _ExampleRows:
    """Render, tokenize, and locate the continuation region for one example."""
    fewshot = [data[i] for i in fewshot_indices(idx, len(data), spec.num_fewshot)]
    delim = spec.continuation_delimiter
    bos = [] if bos_id is None else [bos_id]

    if spec.task_type == "multiple_choice":
        prompts = render_prompts_mc(item, delim, fewshot)
        tokens = [bos + tokenizer.encode(p) for p in prompts]
        start = common_prefix_len(tokens)  # batch_sequences_mc, core_eval.py:113-120
        rows = [(seq, start, len(seq)) for seq in tokens]
        gold = int(item["gold"])
    elif spec.task_type == "schema":
        prompts = render_prompts_schema(item, delim, fewshot)
        tokens = [bos + tokenizer.encode(p) for p in prompts]
        suffix = common_suffix_len(tokens)  # batch_sequences_schema, core_eval.py:123-130
        rows = [(seq, len(seq) - suffix, len(seq)) for seq in tokens]
        gold = int(item["gold"])
    else:  # language_modeling
        without, with_ = render_prompts_lm(item, delim, fewshot)
        tokens_without = bos + tokenizer.encode(without)
        tokens_with = bos + tokenizer.encode(with_)
        start, end = len(tokens_without), len(tokens_with)
        # The prefix property nanochat asserts (core_eval.py:138-139): if the tokenizer merges
        # across the without/with boundary, the region is undefined — fail loud.
        if not (start < end and tokens_with[:start] == tokens_without):
            raise ValueError(
                f"{spec.name}: without-prompt is not a token prefix of with-prompt "
                f"(lens {start} vs {end}) — tokenizer merged across the continuation boundary"
            )
        rows = [(tokens_with, start, end)]
        gold = None

    # A non-empty region must start at token >= 1 (scoring reads logits[start-1]; index -1 would
    # silently wrap). nanochat guarantees this structurally by always prepending BOS.
    for _seq, start, end in rows:
        if start < 1 and end > start:
            raise ValueError(
                f"{spec.name}: continuation region starts at token 0 — pass bos_id (nanochat "
                "always prepends BOS) or ensure prompts share at least one leading token"
            )
    if max_seq_len is not None:
        rows = [_truncate_row(row, max_seq_len, spec.name) for row in rows]
    return _ExampleRows(rows=rows, gold=gold)


# ---------------------------------------------------------------------------------------------
# Batched scoring through one forward — mean region log-prob + region argmax exact-match
# ---------------------------------------------------------------------------------------------


@torch.no_grad()
def _score_rows(
    model: nn.Module, rows: Sequence[_Row], *, device: str, batch_size: int
) -> list[tuple[float, bool]]:
    """Per row: ``(mean log p over [start, end), all-argmax-match over [start, end))``.

    Rows are padded right and batched through ONE forward per chunk; causal attention makes the
    pad id irrelevant (scored positions never attend to it). Mean log-prob is the negative of
    nanochat's mean cross-entropy (core_eval.py:232-237); the argmax match is its LM rule
    (core_eval.py:224-231). An empty region scores ``(-inf, False)`` (documented delta).
    """
    out: list[tuple[float, bool]] = []
    for off in range(0, len(rows), batch_size):
        chunk = rows[off : off + batch_size]
        max_len = max(len(seq) for seq, _, _ in chunk)
        input_ids = torch.zeros((len(chunk), max_len), dtype=torch.long)
        for i, (seq, _, _) in enumerate(chunk):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        logits = model(input_ids.to(device))  # (B, T, V)
        logp = torch.log_softmax(logits.float(), dim=-1)
        preds = logits.argmax(dim=-1)
        for i, (seq, start, end) in enumerate(chunk):
            if end <= start:
                out.append((float("-inf"), False))
                continue
            targets = torch.tensor(seq[start:end], dtype=torch.long, device=logp.device)
            # logits at position j predict token j+1 ⇒ region tokens [start, end) are scored by
            # positions [start-1, end-1).
            region_logp = logp[i, start - 1 : end - 1].gather(-1, targets.unsqueeze(-1))
            match = bool(torch.equal(preds[i, start - 1 : end - 1], targets))
            out.append((float(region_logp.mean()), match))
    return out


# ---------------------------------------------------------------------------------------------
# Task / suite evaluation
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CoreTaskResult:
    name: str
    accuracy: float
    centered: float  # (acc − b) / (1 − b), via report_card.core_style_score
    n: int
    random_baseline: float


@dataclass(frozen=True)
class CoreSuiteResult:
    """Per-task results plus the CORE aggregate (mean of centered accuracies)."""

    tasks: tuple[CoreTaskResult, ...]

    @property
    def core(self) -> float:
        return core_style_score(
            {t.name: t.accuracy for t in self.tasks},
            {t.name: t.random_baseline for t in self.tasks},
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "results": {t.name: t.accuracy for t in self.tasks},
            "centered_results": {t.name: t.centered for t in self.tasks},
            "n_examples": {t.name: t.n for t in self.tasks},
            "core": self.core,
        }


def evaluate_core_task(
    model: nn.Module,
    tokenizer: TextTokenizer,
    spec: TaskSpec,
    *,
    device: str = "cpu",
    limit: int | None = None,
    batch_size: int = 16,
    bos_id: int | None = None,
    max_seq_len: int | None = None,
) -> CoreTaskResult:
    """Evaluate one task with the CORE recipe.

    ``limit`` subsamples after the seed-1337 shuffle (matching ``--max-per-task``,
    base_eval.py:103-107); few-shot examples are drawn from the same subsample, as nanochat does.
    ``bos_id`` is prepended to every sequence (their BOS; our ``<|eot|>`` doc separator).
    ``max_seq_len`` truncates to the model's context (pass ``model.cfg.context_length``).
    """
    data = list(spec.loader())
    if not data:
        raise ValueError(f"{spec.name}: loader returned no examples")
    random.Random(_SHUFFLE_SEED).shuffle(data)
    if limit is not None and limit > 0:
        data = data[:limit]

    all_rows: list[_Row] = []
    spans: list[tuple[int, int]] = []  # (offset, n_rows) per example
    golds: list[int | None] = []
    for idx, item in enumerate(data):
        ex = _rows_for_example(
            item, idx, data, spec, tokenizer, bos_id=bos_id, max_seq_len=max_seq_len
        )
        spans.append((len(all_rows), len(ex.rows)))
        golds.append(ex.gold)
        all_rows.extend(ex.rows)

    scores = _score_rows(model, all_rows, device=device, batch_size=batch_size)

    correct = 0
    for (off, k), gold in zip(spans, golds, strict=True):
        chunk = scores[off : off + k]
        if gold is None:  # language modeling: every region token argmax-matched
            correct += int(chunk[0][1])
        else:  # MC/schema: highest mean log-prob, first index on ties (== nanochat's min())
            pred = max(range(k), key=lambda j: chunk[j][0])
            correct += int(pred == gold)

    accuracy = correct / len(data)
    centered = core_style_score(
        {spec.name: accuracy}, {spec.name: spec.random_baseline}
    )  # single-task mean == (acc − b)/(1 − b)
    return CoreTaskResult(spec.name, accuracy, centered, len(data), spec.random_baseline)


def evaluate_core_suite(
    model: nn.Module,
    tokenizer: TextTokenizer,
    specs: Sequence[TaskSpec],
    *,
    device: str = "cpu",
    limit: int | None = None,
    batch_size: int = 16,
    bos_id: int | None = None,
    max_seq_len: int | None = None,
    progress: Callable[[CoreTaskResult], None] | None = None,
) -> CoreSuiteResult:
    """Evaluate a list of tasks and aggregate; ``progress`` fires after each task (CLI printing)."""
    if not specs:
        raise ValueError("specs must be non-empty")
    results: list[CoreTaskResult] = []
    for spec in specs:
        res = evaluate_core_task(
            model,
            tokenizer,
            spec,
            device=device,
            limit=limit,
            batch_size=batch_size,
            bos_id=bos_id,
            max_seq_len=max_seq_len,
        )
        results.append(res)
        if progress is not None:
            progress(res)
    return CoreSuiteResult(tasks=tuple(results))


# ---------------------------------------------------------------------------------------------
# The official 22-task manifest — transcribed from eval_bundle/core.yaml + eval_meta_data.csv
# ---------------------------------------------------------------------------------------------

# (name, dataset_uri, task_type, num_fewshot, continuation_delimiter, random_baseline_percent)
# Order follows core.yaml. Baselines are the CSV's percent values (converted to fractions below).
_CORE_MANIFEST: tuple[tuple[str, str, TaskType, int, str, float], ...] = (
    (
        "hellaswag_zeroshot",
        "language_understanding/hellaswag.jsonl",
        "multiple_choice",
        0,
        " ",
        25.0,
    ),
    ("jeopardy", "world_knowledge/jeopardy_all.jsonl", "language_modeling", 10, "\nAnswer: ", 0.0),
    (
        "bigbench_qa_wikidata",
        "world_knowledge/bigbench_qa_wikidata.jsonl",
        "language_modeling",
        10,
        " ",
        0.0,
    ),
    ("arc_easy", "world_knowledge/arc_easy.jsonl", "multiple_choice", 10, "\nAnswer: ", 25.0),
    (
        "arc_challenge",
        "world_knowledge/arc_challenge.jsonl",
        "multiple_choice",
        10,
        "\nAnswer: ",
        25.0,
    ),
    ("copa", "commonsense_reasoning/copa.jsonl", "multiple_choice", 0, " ", 50.0),
    (
        "commonsense_qa",
        "commonsense_reasoning/commonsense_qa.jsonl",
        "multiple_choice",
        10,
        " ",
        20.0,
    ),
    ("piqa", "commonsense_reasoning/piqa.jsonl", "multiple_choice", 10, "\nAnswer: ", 50.0),
    ("openbook_qa", "commonsense_reasoning/openbook_qa.jsonl", "multiple_choice", 0, " ", 25.0),
    (
        "lambada_openai",
        "language_understanding/lambada_openai.jsonl",
        "language_modeling",
        0,
        " ",
        0.0,
    ),
    ("hellaswag", "language_understanding/hellaswag.jsonl", "multiple_choice", 10, " ", 25.0),
    ("winograd", "language_understanding/winograd_wsc.jsonl", "schema", 0, " ", 50.0),
    ("winogrande", "language_understanding/winogrande.jsonl", "schema", 0, " ", 50.0),
    (
        "bigbench_dyck_languages",
        "symbolic_problem_solving/bigbench_dyck_languages.jsonl",
        "language_modeling",
        10,
        " ",
        0.0,
    ),
    (
        "agi_eval_lsat_ar",
        "symbolic_problem_solving/agi_eval_lsat_ar.jsonl",
        "multiple_choice",
        3,
        " ",
        20.0,
    ),
    (
        "bigbench_cs_algorithms",
        "symbolic_problem_solving/bigbench_cs_algorithms.jsonl",
        "language_modeling",
        10,
        " ",
        0.0,
    ),
    (
        "bigbench_operators",
        "symbolic_problem_solving/bigbench_operators.jsonl",
        "language_modeling",
        10,
        " ",
        0.0,
    ),
    (
        "bigbench_repeat_copy_logic",
        "symbolic_problem_solving/bigbench_repeat_copy_logic.jsonl",
        "language_modeling",
        10,
        " ",
        0.0,
    ),
    ("squad", "reading_comprehension/squad.jsonl", "language_modeling", 10, " ", 0.0),
    ("coqa", "reading_comprehension/coqa.jsonl", "language_modeling", 0, " ", 0.0),
    ("boolq", "reading_comprehension/boolq.jsonl", "multiple_choice", 10, "\nAnswer: ", 62.0),
    (
        "bigbench_language_identification",
        "language_understanding/bigbench_language_identification.jsonl",
        "multiple_choice",
        10,
        " ",
        9.1,
    ),
)

CORE_TASK_NAMES: tuple[str, ...] = tuple(row[0] for row in _CORE_MANIFEST)


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "scratch_llm"


def ensure_eval_bundle(cache_dir: str | Path | None = None) -> Path:
    """Download + extract the nanochat eval bundle (idempotent); return the ``eval_bundle`` dir.

    This is the ONLY function that touches the network, and only on a cache miss. The bundle is
    the exact data nanochat evaluates (scripts/base_eval.py:42), so task data is bit-identical.
    """
    root = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    bundle_dir = root / "eval_bundle"
    if bundle_dir.is_dir():
        return bundle_dir
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "eval_bundle.zip"
    if not zip_path.exists():
        tmp_path = zip_path.with_suffix(".zip.part")
        urllib.request.urlretrieve(EVAL_BUNDLE_URL, tmp_path)  # noqa: S310 — pinned https URL
        tmp_path.rename(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.startswith("eval_bundle/") and ".." not in m]
        zf.extractall(root, members=members)
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"eval bundle extracted but {bundle_dir} is missing")
    return bundle_dir


def _load_jsonl(path: Path) -> list[Example]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def bundle_task_loader(
    dataset_uri: str, cache_dir: str | Path | None = None
) -> Callable[[], list[Example]]:
    """A lazy loader for one bundle jsonl — network (if any) happens at call time, never before."""

    def load() -> list[Example]:
        bundle_dir = ensure_eval_bundle(cache_dir)
        return _load_jsonl(bundle_dir / "eval_data" / dataset_uri)

    return load


def core_task_specs(
    cache_dir: str | Path | None = None, names: Sequence[str] | None = None
) -> tuple[TaskSpec, ...]:
    """The official 22 CORE :class:`TaskSpec`\\s (or the ``names`` subset, in manifest order).

    Constructing the specs performs no I/O; each task lazily loads its jsonl from the eval
    bundle on first evaluation.
    """
    if names is not None:
        unknown = sorted(set(names) - set(CORE_TASK_NAMES))
        if unknown:
            raise ValueError(f"unknown CORE task(s) {unknown}; valid: {list(CORE_TASK_NAMES)}")
    selected = set(CORE_TASK_NAMES if names is None else names)
    return tuple(
        TaskSpec(
            name=name,
            task_type=task_type,
            loader=bundle_task_loader(uri, cache_dir),
            random_baseline=baseline_pct / 100.0,  # CSV stores percent; base_eval.py:112
            num_fewshot=num_fewshot,
            continuation_delimiter=delim,
        )
        for name, uri, task_type, num_fewshot, delim, baseline_pct in _CORE_MANIFEST
        if name in selected
    )


__all__ = [
    "CORE_TASK_NAMES",
    "EVAL_BUNDLE_URL",
    "CoreSuiteResult",
    "CoreTaskResult",
    "TaskSpec",
    "bundle_task_loader",
    "common_prefix_len",
    "common_suffix_len",
    "core_task_specs",
    "ensure_eval_bundle",
    "evaluate_core_suite",
    "evaluate_core_task",
    "fewshot_indices",
    "render_prompts_lm",
    "render_prompts_mc",
    "render_prompts_schema",
]
