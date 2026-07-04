"""W7a tests — exact line dedup + MinHash/LSH document dedup (`scratch_llm.data.dedup`).

Covers the load-bearing claims in order of trust: (1) the P[row match] = Jaccard identity on
hand-built sets with known Jaccard (binomial tolerance stated inline), (2) the LSH S-curve's
recall monotone in bands at fixed k, (3) transitive cluster-and-drop keeps exactly one, (4) the
official A4 fixture semantics (read-only from the course scaffold; skipped if absent). All
hermetic tests are seeded and CPU-only.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from scratch_llm.data.dedup import (
    exact_line_dedup,
    jaccard,
    lsh_candidates,
    lsh_collision_probability,
    minhash_dedup,
    minhash_signature,
    ngram_set,
    normalize_text,
)

FIXTURES = Path("/workspace/lectures/assignment4-data/tests/fixtures")
needs_official_fixtures = pytest.mark.skipif(
    not FIXTURES.is_dir(), reason="official A4 fixtures not present on this box"
)


# ---------------------------------------------------------------------------
# Exact line dedup
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_exact_line_dedup_collapses_to_corpus_unique_lines(tmp_path: Path) -> None:
    """Known-duplicate corpus → survivors are exactly the corpus-unique lines, order preserved."""
    src = tmp_path / "src"
    src.mkdir()
    a = _write(src / "a.txt", "shared banner\nunique to A\nshared footer\n")
    b = _write(src / "b.txt", "shared banner\nrepeat me\nrepeat me\nunique to B\n")
    c = _write(src / "c.txt", "shared footer\nunique C one\nunique C two\n")
    d = _write(src / "d.txt", "shared banner")  # no trailing newline: same line identity

    out = tmp_path / "out"
    written = exact_line_dedup([a, b, c, d], out)

    assert [p.name for p in written] == ["a.txt", "b.txt", "c.txt", "d.txt"]
    assert (out / "a.txt").read_text() == "unique to A\n"
    assert (out / "b.txt").read_text() == "unique to B\n"  # within-file dup drops BOTH copies
    assert (out / "c.txt").read_text() == "unique C one\nunique C two\n"
    assert (out / "d.txt").read_text() == ""  # its only line appears 3x corpus-wide

    # The collapse is exact: kept lines == corpus-unique lines (each exactly once).
    all_lines = [ln for p in (a, b, c, d) for ln in p.read_text().splitlines()]
    unique = {ln for ln in all_lines if all_lines.count(ln) == 1}
    kept = [ln for p in written for ln in p.read_text().splitlines()]
    assert sorted(kept) == sorted(unique)


def test_exact_line_dedup_basename_collision_raises(tmp_path: Path) -> None:
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    p1 = _write(d1 / "same.txt", "x\n")
    p2 = _write(d2 / "same.txt", "y\n")
    with pytest.raises(ValueError, match="collide"):
        exact_line_dedup([p1, p2], tmp_path / "out")


@needs_official_fixtures
def test_exact_line_dedup_official_fixture_semantics(tmp_path: Path) -> None:
    """Read-only cross-check against the course fixtures: outputs match the expected
    deduplicated documents as a multiset (mirrors official test_exact_line_deduplication)."""
    inputs = sorted((FIXTURES / "documents_with_line_duplicates").glob("doc*.txt"))
    expected = sorted(
        p.read_text() for p in (FIXTURES / "documents_line_deduplicated").glob("doc*.txt")
    )
    written = exact_line_dedup(inputs, tmp_path)
    assert len(written) == 5
    assert sorted(p.read_text() for p in written) == expected


# ---------------------------------------------------------------------------
# MinHash primitives: normalization, n-grams, Jaccard
# ---------------------------------------------------------------------------


def test_normalize_text_nfd_case_punct_whitespace() -> None:
    # Accents (via NFD), case, punctuation, and whitespace runs all collapse.
    assert normalize_text("Héllo,  WORLD!\n\tfoo—bar") == "hello world foobar"
    assert normalize_text("Hello world") == normalize_text("  héllo,\tWORLD?? ")


def test_ngram_set_window_and_short_doc() -> None:
    grams = ngram_set("a b c d", 2)
    assert grams == {("a", "b"), ("b", "c"), ("c", "d")}
    assert ngram_set("only four words here", 5) == set()  # shorter than n → no evidence


def test_jaccard_hand_values() -> None:
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert jaccard(set(), set()) == 1.0
    assert jaccard({1}, {2}) == 0.0


# ---------------------------------------------------------------------------
# P[row match] = Jaccard (the identity everything rests on)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_shared", "n_own", "expected_j"),
    [(20, 10, 0.5), (10, 20, 0.2)],  # J = shared / (shared + 2*own)
)
def test_row_match_probability_estimates_jaccard(
    n_shared: int, n_own: int, expected_j: float
) -> None:
    """Fraction of agreeing signature rows ≈ true Jaccard, within a 5σ binomial band.

    Row matches are k i.i.d. Bernoulli(J) trials, so the match fraction has
    σ = sqrt(J(1-J)/k); with k = 2000 and a 5σ tolerance this is deterministic for the
    fixed seed and would fail on any bias larger than ~0.056 (J=0.5) / ~0.045 (J=0.2).
    """
    k = 2000
    shared = [f"shared{i}" for i in range(n_shared)]
    set_a = set(shared) | {f"a{i}" for i in range(n_own)}
    set_b = set(shared) | {f"b{i}" for i in range(n_own)}
    assert jaccard(set_a, set_b) == pytest.approx(expected_j)

    sig_a = minhash_signature(set_a, k, seed=0)
    sig_b = minhash_signature(set_b, k, seed=0)
    match_frac = sum(x == y for x, y in zip(sig_a, sig_b, strict=True)) / k

    tolerance = 5 * math.sqrt(expected_j * (1 - expected_j) / k)
    assert abs(match_frac - expected_j) < tolerance


def test_minhash_signature_deterministic_and_seed_sensitive() -> None:
    items = {("a", "b"), ("c", "d"), ("e", "f")}
    assert minhash_signature(items, 32, seed=0) == minhash_signature(items, 32, seed=0)
    assert minhash_signature(items, 32, seed=0) != minhash_signature(items, 32, seed=1)


def test_minhash_signature_empty_set_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        minhash_signature(set(), 16)


# ---------------------------------------------------------------------------
# LSH banding: S-curve recall monotone in bands at fixed k
# ---------------------------------------------------------------------------


def test_s_curve_recall_monotone_in_bands_at_fixed_k() -> None:
    """At fixed k = 48, empirical candidate recall for J = 0.7 pairs is non-decreasing in the
    band count, spanning ~0 (b=2, r=24) to ~1 (b=24, r=2) — the S-curve knee sweeping left."""
    k = 48
    n_pairs = 150
    # Exact J = 0.7 by construction: 14 shared + 3 own each → 14 / 20.
    sig_pairs: list[tuple[list[int], list[int]]] = []
    for p in range(n_pairs):
        shared = {f"p{p}s{i}" for i in range(14)}
        set_a = shared | {f"p{p}a{i}" for i in range(3)}
        set_b = shared | {f"p{p}b{i}" for i in range(3)}
        sig_pairs.append((minhash_signature(set_a, k, seed=0), minhash_signature(set_b, k, seed=0)))

    recalls: list[float] = []
    for bands in (2, 6, 12, 24):
        hits = sum((0, 1) in lsh_candidates([sig_a, sig_b], bands) for sig_a, sig_b in sig_pairs)
        recalls.append(hits / n_pairs)

    assert all(lo <= hi for lo, hi in zip(recalls, recalls[1:], strict=False)), recalls
    assert recalls[0] < 0.1, recalls  # r=24: 0.7^24 ≈ 2e-4 per band
    assert recalls[-1] > 0.9, recalls  # r=2: P ≈ 1 − 0.51^24 ≈ 1.0


def test_s_curve_analytic_monotone_and_knee() -> None:
    k = 120
    for s in (0.3, 0.5, 0.7, 0.9):
        probs = [lsh_collision_probability(s, b, k // b) for b in (2, 4, 6, 12, 24, 60)]
        assert all(lo <= hi for lo, hi in zip(probs, probs[1:], strict=False)), (s, probs)
    # ADR-0015 default (b=10, r=10): knee at (1/b)^(1/r) ≈ 0.794 sits at the 0.8 threshold.
    assert pytest.approx(0.794, abs=5e-3) == (1 / 10) ** (1 / 10)
    assert lsh_collision_probability(0.9, 10, 10) > 0.98
    assert lsh_collision_probability(0.5, 10, 10) < 0.01


def test_lsh_candidates_validation() -> None:
    with pytest.raises(ValueError, match="divide"):
        lsh_candidates([[1, 2, 3], [4, 5, 6]], 2)
    with pytest.raises(ValueError, match="same length"):
        lsh_candidates([[1, 2], [1, 2, 3]], 1)
    assert lsh_candidates([], 4) == set()


def test_lsh_identical_signatures_always_candidates() -> None:
    sig = minhash_signature({("x", "y"), ("y", "z")}, 40, seed=0)
    assert (0, 1) in lsh_candidates([sig, list(sig)], 8)


# ---------------------------------------------------------------------------
# minhash_dedup driver: clustering, determinism, edge cases
# ---------------------------------------------------------------------------


def test_transitive_closure_keeps_exactly_one(tmp_path: Path) -> None:
    """A~B and B~C above threshold but A~C below: the cluster is {A,B,C} by transitivity and
    exactly one (the first in input order) survives; an unrelated doc D is untouched."""
    words = [f"tok{i}" for i in range(30)]
    a = _write(tmp_path / "a.txt", " ".join(words[0:20]))  # J(a,b) = 15/25 = 0.6
    b = _write(tmp_path / "b.txt", " ".join(words[5:25]))  # J(b,c) = 15/25 = 0.6
    c = _write(tmp_path / "c.txt", " ".join(words[10:30]))  # J(a,c) = 10/30 ≈ 0.33 < 0.5
    d = _write(tmp_path / "d.txt", " ".join(f"zeta{i}" for i in range(20)))

    # Sanity: the intended true Jaccards hold for unigram sets.
    ga, gb, gc = (ngram_set(p.read_text(), 1) for p in (a, b, c))
    assert jaccard(ga, gb) == pytest.approx(0.6)
    assert jaccard(gb, gc) == pytest.approx(0.6)
    assert jaccard(ga, gc) == pytest.approx(1 / 3)

    out = tmp_path / "out"
    survivors = minhash_dedup(
        [a, b, c, d], num_hashes=200, num_bands=50, ngrams=1, jaccard_threshold=0.5, out_dir=out
    )
    assert [p.name for p in survivors] == ["a.txt", "d.txt"]
    assert (out / "a.txt").read_text() == a.read_text()  # survivor copied byte-identically
    assert not (out / "b.txt").exists() and not (out / "c.txt").exists()


def test_true_jaccard_confirm_rejects_lsh_candidate(tmp_path: Path) -> None:
    """LSH proposes, Jaccard disposes: an aggressive band split (b=50, r=2) proposes a J=0.6
    pair with P = 1 − (1 − 0.6²)^50 ≈ 1 − 1e−10, but the confirm at threshold 0.8 must reject
    it — both docs survive. Kills a confirm step that unions every LSH candidate."""
    words = [f"tok{i}" for i in range(25)]
    a = _write(tmp_path / "a.txt", " ".join(words[0:20]))
    b = _write(tmp_path / "b.txt", " ".join(words[5:25]))  # 15 shared / 25 union = 0.6

    ga, gb = ngram_set(a.read_text(), 1), ngram_set(b.read_text(), 1)
    assert jaccard(ga, gb) == pytest.approx(0.6)  # well below the 0.8 confirm threshold

    # Teeth check: LSH must actually propose the pair, else the survival assertion is vacuous.
    # Deterministic at seed 0 (the driver's default); analytic miss probability ≈ 1e-10.
    sigs = [minhash_signature(g, 100, seed=0) for g in (ga, gb)]
    assert (0, 1) in lsh_candidates(sigs, 50)

    survivors = minhash_dedup(
        [a, b],
        num_hashes=100,
        num_bands=50,
        ngrams=1,
        jaccard_threshold=0.8,
        out_dir=tmp_path / "out",
    )
    assert [p.name for p in survivors] == ["a.txt", "b.txt"]


def test_minhash_dedup_exact_duplicates_and_distinct_doc(tmp_path: Path) -> None:
    text = "the quick brown fox jumps over the lazy dog again and again"
    a = _write(tmp_path / "a.txt", text)
    b = _write(tmp_path / "b.txt", text)  # exact duplicate of a
    c = _write(tmp_path / "c.txt", "completely different content about minhash lsh banding here")
    survivors = minhash_dedup(
        [a, b, c],
        num_hashes=100,
        num_bands=10,
        ngrams=3,
        jaccard_threshold=0.8,
        out_dir=tmp_path / "out",
    )
    assert [p.name for p in survivors] == ["a.txt", "c.txt"]


def test_minhash_dedup_short_docs_have_no_evidence_and_survive(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.txt", "hi there")  # < ngrams words → empty n-gram set
    b = _write(tmp_path / "b.txt", "hi there")
    survivors = minhash_dedup(
        [a, b],
        num_hashes=20,
        num_bands=5,
        ngrams=5,
        jaccard_threshold=0.8,
        out_dir=tmp_path / "out",
    )
    assert [p.name for p in survivors] == ["a.txt", "b.txt"]


def test_empty_evidence_doc_before_duplicate_pair_still_dedups(tmp_path: Path) -> None:
    """A short/empty-evidence doc placed BEFORE a genuine duplicate pair must not desync the
    LSH-local -> global index remap (dedup.py:319 `i, j = hashable[local_i], hashable[local_j]`).

    ``solo`` has < ngrams words so its n-gram set is empty and it is EXCLUDED from the LSH index,
    making ``hashable == [1, 2] != range(3)``. The only candidate pair is therefore local (0, 1)
    over docs [1, 2] and dedups iff it remaps to the global pair (1, 2). Dropping the remap
    (``i, j = local_i, local_j``) confirms the wrong pair — global (0, 1), i.e. empty ``solo`` vs
    ``dup_a`` at Jaccard 0 — so the true duplicate is never merged and all three docs survive
    (silent under-dedup)."""
    solo = _write(tmp_path / "a_solo.txt", "solo")  # 1 word, ngrams=2 -> empty n-gram set
    dup_text = "the quick brown fox jumps over the lazy dog"
    dup_a = _write(tmp_path / "b_dup.txt", dup_text)
    dup_b = _write(tmp_path / "c_dup.txt", dup_text)  # exact duplicate of dup_a

    # Preconditions that give the remap teeth: solo carries no evidence (empty set, excluded),
    # so the sole LSH candidate is local (0, 1) and MUST remap to global (1, 2).
    assert ngram_set(solo.read_text(), 2) == set()
    assert ngram_set(dup_a.read_text(), 2) == ngram_set(dup_b.read_text(), 2)

    survivors = minhash_dedup(
        [solo, dup_a, dup_b],
        num_hashes=20,
        num_bands=5,
        ngrams=2,
        jaccard_threshold=0.8,
        out_dir=tmp_path / "out",
    )
    # Real impl: dup_b merges into dup_a's cluster -> two survivors. Dropping the remap leaves
    # all three (c_dup.txt would also appear), so this pins the local->global remap exactly.
    assert [p.name for p in survivors] == ["a_solo.txt", "b_dup.txt"]


def test_confirm_boundary_merges_at_exact_threshold(tmp_path: Path) -> None:
    """The confirm boundary is ``>= jaccard_threshold`` (dedup.py:320): a pair whose TRUE Jaccard
    equals the threshold exactly must still merge (one survivor). Kills a ``>``-for-``>=`` mutant.

    Construction pins Jaccard to exactly 0.5: 10 shared + 5 own words each -> 10 / 20 unigrams."""
    shared = [f"s{i}" for i in range(10)]
    a = _write(tmp_path / "a.txt", " ".join(shared + [f"a{i}" for i in range(5)]))
    b = _write(tmp_path / "b.txt", " ".join(shared + [f"b{i}" for i in range(5)]))

    ga, gb = ngram_set(a.read_text(), 1), ngram_set(b.read_text(), 1)
    assert jaccard(ga, gb) == pytest.approx(0.5)  # exactly the confirm threshold

    # Teeth check: LSH must actually propose the pair (deterministic at seed 0), else vacuous.
    sigs = [minhash_signature(g, 100, seed=0) for g in (ga, gb)]
    assert (0, 1) in lsh_candidates(sigs, 50)

    survivors = minhash_dedup(
        [a, b],
        num_hashes=100,
        num_bands=50,
        ngrams=1,
        jaccard_threshold=0.5,
        out_dir=tmp_path / "out",
    )
    assert [p.name for p in survivors] == ["a.txt"]  # merged at exact Jaccard == threshold


def test_minhash_dedup_band_split_validation(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.txt", "some words here")
    with pytest.raises(ValueError, match="divide"):
        minhash_dedup(
            [a],
            num_hashes=100,
            num_bands=7,
            ngrams=2,
            jaccard_threshold=0.8,
            out_dir=tmp_path / "out",
        )


# ---------------------------------------------------------------------------
# Official fixture semantics (read-only cross-check; skipped when absent)
# ---------------------------------------------------------------------------


@needs_official_fixtures
def test_minhash_official_exact_duplicate_fixture(tmp_path: Path) -> None:
    """Mirrors official test_minhash_deduplication_exact_duplicates: doc1 == doc2 exactly, so
    4 of 5 docs survive and the surviving contents match the expected multiset."""
    inputs = sorted((FIXTURES / "documents_with_line_duplicates").glob("doc*.txt"))
    expected = sorted(p.read_text() for p in inputs if p.name != "doc2.txt")
    survivors = minhash_dedup(
        inputs, num_hashes=100, num_bands=10, ngrams=5, jaccard_threshold=0.8, out_dir=tmp_path
    )
    assert len(survivors) == 4
    assert sorted(p.read_text() for p in survivors) == expected


@needs_official_fixtures
def test_minhash_official_fuzzy_duplicate_fixture(tmp_path: Path) -> None:
    """Mirrors official test_minhash_deduplication_fuzzy_duplicates: rails/react MIT licenses
    are fuzzy duplicates (one survives), the pytorch license is distinct (always survives)."""
    fuzzy_dir = FIXTURES / "documents_with_fuzzy_duplicates"
    inputs = sorted(fuzzy_dir.glob("*.txt"))
    survivors = minhash_dedup(
        inputs, num_hashes=500, num_bands=50, ngrams=5, jaccard_threshold=0.8, out_dir=tmp_path
    )
    names = {p.name for p in survivors}
    assert len(survivors) == 2
    assert "pytorch_license.txt" in names
    assert len(names & {"rails_mit_license.txt", "react_mit_license.txt"}) == 1
