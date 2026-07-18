"""Tests for the DCLM CORE task suite (P3) — tiny synthetic fixtures, NO network.

Coverage per the P3 DoD: scoring math per task TYPE on rigged toy models (MC/schema
length-normalized region log-prob, LM greedy exact-match), prompt-builder goldens for
representative tasks, the centered-accuracy formula + aggregation reuse of
``report_card.core_style_score``, the verified 22-task manifest constants, and the CLI
end-to-end on a nano ``TransformerLM`` with fixture tasks registered via the same
:class:`TaskSpec` framework.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from scratch_llm.data.shards import DOC_SEPARATOR
from scratch_llm.eval.core_suite import (
    CORE_TASK_NAMES,
    CoreSuiteResult,
    CoreTaskResult,
    TaskSpec,
    common_prefix_len,
    common_suffix_len,
    core_task_specs,
    evaluate_core_suite,
    evaluate_core_task,
    fewshot_indices,
    render_prompts_lm,
    render_prompts_mc,
    render_prompts_schema,
)
from scratch_llm.eval.report_card import core_style_score

# ---------------------------------------------------------------------------------------------
# Fixtures: a bigram-table toy model (full control of every next-token distribution)
# ---------------------------------------------------------------------------------------------

_V = 128  # covers all CharTokenizer (ord) ids used below


class TableLM(nn.Module):
    """logits at each position = ``table[token_at_position]`` — a rigged bigram model."""

    def __init__(self, table: Tensor) -> None:
        super().__init__()
        self.table = table

    def forward(self, ids: Tensor) -> Tensor:
        return self.table[ids]


def favor_table(favored: str, boost: float = 12.0) -> Tensor:
    """Position/context-independent: one column boosted (a FavorLM)."""
    t = torch.zeros(_V, _V)
    t[:, ord(favored)] = boost
    return t


def bigram_table(transitions: dict[str, str], boost: float = 10.0) -> Tensor:
    """After char p, char n is the argmax (and much likelier than the uniform rest)."""
    t = torch.zeros(_V, _V)
    for p, n in transitions.items():
        t[ord(p), ord(n)] = boost
    return t


class CharTokenizer:
    """id = ord(char) — satisfies the TextTokenizer protocol with zero merge ambiguity."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def mc_spec(examples: list[dict], **kw) -> TaskSpec:
    defaults = dict(
        name="toy_mc",
        task_type="multiple_choice",
        loader=lambda: examples,
        random_baseline=0.5,
        num_fewshot=0,
        continuation_delimiter=" ",
    )
    defaults.update(kw)
    return TaskSpec(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# Prompt-builder goldens (representative tasks: arc-style MC, winograd-style schema, jeopardy LM)
# ---------------------------------------------------------------------------------------------


def test_render_prompts_mc_golden_arc_style() -> None:
    """10-shot arc_easy uses delimiter '\\nAnswer: '; few-shot blocks show the GOLD choice."""
    fewshot = [{"query": "Q1", "choices": ["a", "b"], "gold": 0}]
    item = {"query": "Q2", "choices": ["x", "y"], "gold": 1}
    prompts = render_prompts_mc(item, "\nAnswer: ", fewshot)
    assert prompts == [
        "Q1\nAnswer: a\n\nQ2\nAnswer: x",
        "Q1\nAnswer: a\n\nQ2\nAnswer: y",
    ]


def test_render_prompts_mc_zeroshot_space_delim_hellaswag_style() -> None:
    item = {"query": "A man sits.", "choices": ["He reads", "He flies"], "gold": 0}
    assert render_prompts_mc(item, " ") == ["A man sits. He reads", "A man sits. He flies"]


def test_render_prompts_schema_golden_winograd_style() -> None:
    """Contexts vary, continuation fixed; few-shot blocks use the GOLD context option."""
    fewshot = [{"context_options": ["f1", "f2"], "continuation": "fc", "gold": 1}]
    item = {"context_options": ["c1", "c2"], "continuation": "end", "gold": 0}
    assert render_prompts_schema(item, " ", fewshot) == [
        "f2 fc\n\nc1 end",
        "f2 fc\n\nc2 end",
    ]


def test_render_prompts_lm_golden_jeopardy_style() -> None:
    """The without-prompt strips the delimiter's trailing whitespace (nanochat core_eval.py:82) so
    the with-prompt extends it cleanly in token space."""
    fewshot = [{"context": "FC", "continuation": "FA"}]
    item = {"context": " C ", "continuation": "A"}  # context is trimmed
    without, with_ = render_prompts_lm(item, "\nAnswer: ", fewshot)
    assert without == "FC\nAnswer: FA\n\nC\nAnswer:"
    assert with_ == "FC\nAnswer: FA\n\nC\nAnswer: A"


def test_render_prompts_lm_space_delim_strips_trailing() -> None:
    without, with_ = render_prompts_lm({"context": "ctx", "continuation": "cont"}, " ")
    assert (without, with_) == ("ctx", "ctx cont")


# ---------------------------------------------------------------------------------------------
# Token-region finders + few-shot sampling
# ---------------------------------------------------------------------------------------------


def test_common_prefix_and_suffix_len() -> None:
    assert common_prefix_len([[1, 2, 3], [1, 2, 4]]) == 2
    assert common_prefix_len([[1, 2], [1, 2, 9]]) == 2  # bounded by the shortest
    assert common_suffix_len([[9, 2, 3], [8, 2, 3]]) == 2
    assert common_suffix_len([[1, 2], [3, 4]]) == 0


def test_fewshot_indices_deterministic_and_excludes_self() -> None:
    a = fewshot_indices(3, 20, 5)
    assert a == fewshot_indices(3, 20, 5)  # Random(1234 + idx) — reproducible
    assert len(a) == 5 and 3 not in a
    assert fewshot_indices(0, 2, 10) == [1]  # k caps at n-1 (documented delta)
    assert fewshot_indices(0, 5, 0) == []


# ---------------------------------------------------------------------------------------------
# Scoring math per task type
# ---------------------------------------------------------------------------------------------


def test_mc_scoring_length_normalized_region_logprob() -> None:
    """Choices 'b a' vs 'bc' share the token prefix 'q b'; regions are [' ','a'] vs ['c'].
    With 'a' favored, MEAN log-prob picks choice 0 while SUMMED log-prob would pick choice 1
    (−12.0016 < −12.0008) — this pins both the common-prefix region and the normalization."""
    model = TableLM(favor_table("a"))
    examples = [{"query": "q", "choices": ["b a", "bc"], "gold": 0}]
    res = evaluate_core_task(model, CharTokenizer(), mc_spec(examples))
    assert res.accuracy == 1.0
    wrong = [{"query": "q", "choices": ["b a", "bc"], "gold": 1}]
    assert evaluate_core_task(model, CharTokenizer(), mc_spec(wrong)).accuracy == 0.0


def test_mc_strict_prefix_option_scores_neg_inf_never_wins() -> None:
    """A choice that is a strict token-prefix of another has an EMPTY region → -inf (documented
    delta from nanochat's nan): the longer option wins deterministically."""
    model = TableLM(favor_table("a"))
    examples = [{"query": "q", "choices": ["b", "ba"], "gold": 1}]
    assert evaluate_core_task(model, CharTokenizer(), mc_spec(examples)).accuracy == 1.0


def test_schema_scoring_context_options_discriminated_by_bigram() -> None:
    """Common suffix ' a' is scored under each context; T['a'][' '] boosted ⇒ option ending in
    'a' yields the higher region log-prob."""
    model = TableLM(bigram_table({"a": " ", " ": "a"}))
    tok = CharTokenizer()
    spec = lambda gold: TaskSpec(  # noqa: E731
        name="toy_schema",
        task_type="schema",
        loader=lambda: [{"context_options": ["xa", "xb"], "continuation": "a", "gold": gold}],
        random_baseline=0.5,
    )
    assert evaluate_core_task(model, tok, spec(0)).accuracy == 1.0
    assert evaluate_core_task(model, tok, spec(1)).accuracy == 0.0


def test_lm_scoring_greedy_exact_match_all_tokens() -> None:
    """LM is correct iff EVERY continuation token is the teacher-forced argmax: the chain
    a→' '→b→c matches 'bc' and fails 'bd' (one wrong token kills the example)."""
    model = TableLM(bigram_table({"a": " ", " ": "b", "b": "c"}))
    spec = TaskSpec(
        name="toy_lm",
        task_type="language_modeling",
        loader=lambda: [
            {"context": "a", "continuation": "bc"},
            {"context": "a", "continuation": "bd"},
        ],
        random_baseline=0.0,
    )
    res = evaluate_core_task(model, CharTokenizer(), spec)
    assert res.accuracy == 0.5
    assert res.centered == res.accuracy  # baseline 0 ⇒ centered == raw


def test_lm_tokenizer_merge_across_boundary_raises() -> None:
    """If the tokenizer merges across the without/with boundary the region is undefined —
    the nanochat prefix assert (core_eval.py:138-139) becomes a loud ValueError here."""

    class MergingTok:
        def encode(self, text: str) -> list[int]:
            return {"a": [1, 2], "a b": [1, 3, 4]}[text]

        def decode(self, ids: list[int]) -> str:
            return ""

    spec = TaskSpec(
        name="toy_lm",
        task_type="language_modeling",
        loader=lambda: [{"context": "a", "continuation": "b"}],
        random_baseline=0.0,
    )
    with pytest.raises(ValueError, match="token prefix"):
        evaluate_core_task(TableLM(favor_table("a")), MergingTok(), spec)


def test_region_at_token_zero_without_bos_raises() -> None:
    """Divergence at token 0 with no BOS would read logits[-1] — refused loudly."""
    model = TableLM(favor_table("a"))
    examples = [{"query": "", "choices": ["a", "b"], "gold": 0}]
    with pytest.raises(ValueError, match="bos_id"):
        evaluate_core_task(model, CharTokenizer(), mc_spec(examples, continuation_delimiter=""))


def test_bos_prepend_allows_token_zero_divergence() -> None:
    examples = [{"query": "", "choices": ["a", "b"], "gold": 0}]
    res = evaluate_core_task(
        TableLM(favor_table("a")),
        CharTokenizer(),
        mc_spec(examples, continuation_delimiter=""),
        bos_id=0,
    )
    assert res.accuracy == 1.0


def test_truncation_keeps_tail_and_preserves_result() -> None:
    """max_seq_len crops from the LEFT and shifts the region (core_eval.py:196-213); a bigram
    model is window-independent so the verdict must not change."""
    model = TableLM(favor_table("a"))
    tok = CharTokenizer()
    examples = [{"query": "qqqqqq", "choices": ["b a", "bc"], "gold": 0}]
    full = evaluate_core_task(model, tok, mc_spec(examples))
    cropped = evaluate_core_task(model, tok, mc_spec(examples), max_seq_len=4)
    assert full.accuracy == cropped.accuracy == 1.0
    with pytest.raises(ValueError, match="no room"):
        evaluate_core_task(model, tok, mc_spec(examples), max_seq_len=2)


def test_batch_size_invariance() -> None:
    """Padding + chunking must not change any verdict (causal scoring reads no pad position)."""
    model = TableLM(favor_table("a"))
    tok = CharTokenizer()
    examples = [
        {"query": "q", "choices": ["b a", "bc"], "gold": 0},
        {"query": "quite long query", "choices": ["b a", "bc"], "gold": 1},
        {"query": "zz", "choices": ["na", "nb"], "gold": 0},
    ]
    r1 = evaluate_core_task(model, tok, mc_spec(examples), batch_size=1)
    r16 = evaluate_core_task(model, tok, mc_spec(examples), batch_size=16)
    assert r1.accuracy == r16.accuracy


def test_limit_subsamples_after_seeded_shuffle() -> None:
    model = TableLM(favor_table("a"))
    examples = [{"query": f"q{i}", "choices": ["b a", "bc"], "gold": 0} for i in range(5)]
    res = evaluate_core_task(model, CharTokenizer(), mc_spec(examples), limit=2)
    assert res.n == 2
    # Deterministic: Random(1337) shuffle ⇒ the same subsample (and accuracy) every run.
    again = evaluate_core_task(model, CharTokenizer(), mc_spec(examples), limit=2)
    assert (res.accuracy, res.n) == (again.accuracy, again.n)


# ---------------------------------------------------------------------------------------------
# Centering + aggregation (must be report_card.core_style_score, not a re-derivation)
# ---------------------------------------------------------------------------------------------


def test_centered_accuracy_formula() -> None:
    """centered = (acc − b) / (1 − b): acc 0.5 at baseline 0.25 → 1/3."""
    model = TableLM(favor_table("a"))
    examples = [
        {"query": "q", "choices": ["b a", "bc"], "gold": 0},  # model picks 0: correct
        {"query": "r", "choices": ["b a", "bc"], "gold": 1},  # model picks 0: wrong
    ]
    res = evaluate_core_task(model, CharTokenizer(), mc_spec(examples, random_baseline=0.25))
    assert res.accuracy == 0.5
    assert math.isclose(res.centered, (0.5 - 0.25) / 0.75, rel_tol=1e-9)


def test_suite_core_equals_report_card_aggregation() -> None:
    tasks = (
        CoreTaskResult("t1", accuracy=0.5, centered=(0.5 - 0.25) / 0.75, n=4, random_baseline=0.25),
        CoreTaskResult("t2", accuracy=1.0, centered=1.0, n=4, random_baseline=0.5),
    )
    suite = CoreSuiteResult(tasks=tasks)
    expected = core_style_score({"t1": 0.5, "t2": 1.0}, {"t1": 0.25, "t2": 0.5})
    assert math.isclose(suite.core, expected, rel_tol=1e-12)
    assert suite.to_dict()["core"] == suite.core


def test_evaluate_core_suite_runs_all_tasks_and_reports_progress() -> None:
    model = TableLM(bigram_table({"a": " ", " ": "b", "b": "c"}))
    specs = [
        mc_spec([{"query": "q", "choices": ["b a", "bc"], "gold": 0}]),
        TaskSpec(
            name="toy_lm",
            task_type="language_modeling",
            loader=lambda: [{"context": "a", "continuation": "bc"}],
            random_baseline=0.0,
        ),
    ]
    seen: list[str] = []
    suite = evaluate_core_suite(
        model, CharTokenizer(), specs, progress=lambda r: seen.append(r.name)
    )
    assert seen == ["toy_mc", "toy_lm"]
    assert set(suite.to_dict()["results"]) == {"toy_mc", "toy_lm"}  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# The verified 22-task manifest (spec-pinning: any drift from core.yaml fails here)
# ---------------------------------------------------------------------------------------------

_VERIFIED_NAMES = (
    "hellaswag_zeroshot",
    "jeopardy",
    "bigbench_qa_wikidata",
    "arc_easy",
    "arc_challenge",
    "copa",
    "commonsense_qa",
    "piqa",
    "openbook_qa",
    "lambada_openai",
    "hellaswag",
    "winograd",
    "winogrande",
    "bigbench_dyck_languages",
    "agi_eval_lsat_ar",
    "bigbench_cs_algorithms",
    "bigbench_operators",
    "bigbench_repeat_copy_logic",
    "squad",
    "coqa",
    "boolq",
    "bigbench_language_identification",
)


def test_manifest_is_the_verified_22_tasks() -> None:
    assert len(CORE_TASK_NAMES) == 22
    assert CORE_TASK_NAMES == _VERIFIED_NAMES


def test_core_task_specs_no_io_and_verified_fields(tmp_path: Path) -> None:
    specs = core_task_specs(cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []  # constructing specs touches no disk/network
    by_name = {s.name: s for s in specs}
    assert len(specs) == 22

    boolq = by_name["boolq"]
    assert (boolq.task_type, boolq.num_fewshot, boolq.continuation_delimiter) == (
        "multiple_choice",
        10,
        "\nAnswer: ",
    )
    assert math.isclose(boolq.random_baseline, 0.62)
    assert math.isclose(by_name["bigbench_language_identification"].random_baseline, 0.091)
    assert (by_name["winogrande"].task_type, by_name["winogrande"].num_fewshot) == ("schema", 0)
    jeopardy = by_name["jeopardy"]
    assert (jeopardy.task_type, jeopardy.continuation_delimiter) == (
        "language_modeling",
        "\nAnswer: ",
    )
    assert by_name["agi_eval_lsat_ar"].num_fewshot == 3
    assert by_name["lambada_openai"].random_baseline == 0.0
    # hellaswag appears twice (0-shot + 10-shot) over the same dataset file — 22 tasks, 21 files.
    assert by_name["hellaswag_zeroshot"].num_fewshot == 0
    assert by_name["hellaswag"].num_fewshot == 10


def test_core_task_specs_subset_and_unknown_name() -> None:
    subset = core_task_specs(names=["boolq", "arc_easy"])
    assert [s.name for s in subset] == ["arc_easy", "boolq"]  # manifest order
    with pytest.raises(ValueError, match="unknown CORE task"):
        core_task_specs(names=["arc_easy", "nope"])


def test_task_spec_validation() -> None:
    with pytest.raises(ValueError, match="task_type"):
        TaskSpec("x", "generative", lambda: [], 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="random_baseline"):
        TaskSpec("x", "schema", lambda: [], 1.0)
    with pytest.raises(ValueError, match="num_fewshot"):
        TaskSpec("x", "schema", lambda: [], 0.5, num_fewshot=-1)


# ---------------------------------------------------------------------------------------------
# CLI end-to-end: nano TransformerLM + fixture tasks through the same TaskSpec framework
# ---------------------------------------------------------------------------------------------


@pytest.fixture()
def cli_env(tmp_path: Path) -> tuple[str, str]:
    """A staged tokenizer dir + a config-carrying nano checkpoint."""
    from scratch_llm.model import ModelConfig, TransformerLM
    from scratch_llm.tokenizer import Tokenizer, train_bpe
    from scratch_llm.train import save_checkpoint

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the cat sat on the mat. a dog ran fast. yes no maybe.\n" * 20, "utf-8")
    vocab, merges = train_bpe(corpus, vocab_size=300, special_tokens=[DOC_SEPARATOR])
    tokenizer = Tokenizer(vocab, merges, special_tokens=[DOC_SEPARATOR])
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tokenizer.save(data_dir / "tokenizer.json")

    model = TransformerLM(
        ModelConfig(vocab_size=len(vocab), d_model=32, n_layers=2, n_heads=4, context_length=64)
    )
    ckpt = tmp_path / "model.pt"
    save_checkpoint(model, None, step=7, out=ckpt)
    return str(ckpt), str(data_dir)


def _fixture_specs() -> list[TaskSpec]:
    mc_examples = [
        {"query": "the cat", "choices": ["sat", "ran"], "gold": 0},
        {"query": "a dog", "choices": ["sat", "ran"], "gold": 1},
        {"query": "yes", "choices": ["no", "maybe"], "gold": 0},
        {"query": "the mat", "choices": ["cat", "dog"], "gold": 0},
        {"query": "fast", "choices": ["dog", "cat"], "gold": 1},
    ]
    lm_examples = [{"context": "the cat", "continuation": "sat"} for _ in range(3)]
    return [
        TaskSpec(
            name="toy_mc",
            task_type="multiple_choice",
            loader=lambda: mc_examples,
            random_baseline=0.5,
            num_fewshot=1,
        ),
        TaskSpec(
            name="toy_lm",
            task_type="language_modeling",
            loader=lambda: lm_examples,
            random_baseline=0.0,
        ),
    ]


def test_cli_end_to_end_on_nano_model(
    cli_env: tuple[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
    try:
        from core_eval import main as core_eval_main
    finally:
        sys.path.pop(0)

    ckpt, data_dir = cli_env
    out = tmp_path / "core.json"
    core_eval_main(
        ["--ckpt", ckpt, "--data-dir", data_dir, "--out", str(out)],
        specs=_fixture_specs(),
    )
    payload = json.loads(out.read_text("utf-8"))
    assert set(payload["results"]) == {"toy_mc", "toy_lm"}
    assert payload["n_examples"]["toy_mc"] == 5
    assert payload["ckpt_step"] == 7
    assert payload["bos_id"] is not None  # DOC_SEPARATOR special found ⇒ BOS prepend active
    assert -2.0 <= payload["core"] <= 1.0
    for name, acc in payload["results"].items():
        b = 0.5 if name == "toy_mc" else 0.0
        assert math.isclose(payload["centered_results"][name], (acc - b) / (1 - b), abs_tol=1e-9)
    stdout = capsys.readouterr().out
    assert "CORE metric" in stdout
    assert payload["ledger_row"].startswith("| CORE (DCLM 22-task) |")
    assert payload["ledger_row"] in stdout

    # --tasks subset + --limit smoke path on the same fixture registry.
    out2 = tmp_path / "core2.json"
    core_eval_main(
        [
            "--ckpt",
            ckpt,
            "--data-dir",
            data_dir,
            "--tasks",
            "toy_mc",
            "--limit",
            "3",
            "--out",
            str(out2),
        ],
        specs=_fixture_specs(),
    )
    payload2 = json.loads(out2.read_text("utf-8"))
    assert set(payload2["results"]) == {"toy_mc"}
    assert payload2["n_examples"]["toy_mc"] == 3

    with pytest.raises(SystemExit):
        core_eval_main(
            ["--ckpt", ckpt, "--data-dir", data_dir, "--tasks", "nope", "--out", str(out2)],
            specs=_fixture_specs(),
        )
