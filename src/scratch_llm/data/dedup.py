"""A4 dedup machinery — exact line dedup + MinHash/LSH fuzzy document dedup.

Two primitives (CS336 A4 §3.1–3.2), both LOAD-BEARING:

- **`exact_line_dedup`** — two-pass hash-count line dedup. Pass 1 streams every file and counts
  16-byte BLAKE2b digests of lines (bounded memory: hashes, never raw lines); pass 2 rewrites each
  file keeping only lines whose *corpus-wide* count is 1. Invariant: a line surviving anywhere
  appears exactly once in the whole corpus.

- **`minhash_dedup`** — fuzzy document dedup: normalize (NFD → lowercase → strip accents/punct →
  collapse whitespace) → word n-gram sets → k seeded hash functions → MinHash signature → LSH
  banding (k = b·r) → candidate pairs from band-bucket collisions → **true-Jaccard confirm**
  ≥ threshold → union-find transitive clusters → keep one doc per cluster (first in input order).
  Key identity: ``P[minhash_i(A) = minhash_i(B)] = Jaccard(A, B)``, so the fraction of agreeing
  signature rows is an unbiased Jaccard estimate in O(k) space. LSH collision probability for a
  pair at similarity s is the S-curve ``1 − (1 − s^r)^b`` with knee near ``(1/b)^(1/r)`` —
  LSH proposes, Jaccard disposes, union-find drops.

Invariants: deterministic for a fixed seed + input order (no salted `hash()` anywhere; survivors
are chosen by input position, not dict order); duplicate clusters keep *exactly one* member
(connected components, never pairwise drop-both); LSH never changes *who* is a duplicate — only
which pairs get checked (the confirm step uses the true Jaccard on the full n-gram sets).

Interview question this module answers: "walk me through MinHash + LSH dedup and how you'd set
the band count" — signature = per-hash-fn minimum over the n-gram set; the (b, r) split of k
hashes is the precision/recall dial (more bands → knee moves left → higher recall, more false
candidates to confirm); pick (b, r) so the knee sits at your Jaccard threshold and quote
``1 − (1 − s^r)^b`` at the similarities you must catch.
"""

from __future__ import annotations

import hashlib
import os
import random
import shutil
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import MappingProxyType

# 2^61 − 1 (Mersenne prime): universal-hash modulus. Python ints don't overflow, so
# (a·x + b) mod p is exact — no numpy uint64 wraparound traps.
_MERSENNE_61 = (1 << 61) - 1

#: Seed for the k hash functions. Fixed so signatures are reproducible run-to-run.
DEFAULT_SEED = 0

#: Recommended pipeline setting (ADR-0015): knee (1/10)^(1/10) ≈ 0.794 sits at the 0.8
#: confirm threshold; recall ≈ 0.99 for near-exact duplicates (s ≥ 0.9).
DEFAULT_MINHASH_PARAMS = MappingProxyType(
    {"num_hashes": 100, "num_bands": 10, "ngrams": 5, "jaccard_threshold": 0.8}
)

_PathLike = str | os.PathLike[str]


# ---------------------------------------------------------------------------
# Exact line dedup
# ---------------------------------------------------------------------------


def _line_key(line: str) -> bytes:
    """Stable 16-byte digest of a line (sans terminator) — the bounded-memory hash key."""
    return hashlib.blake2b(line.encode("utf-8"), digest_size=16).digest()


def _check_unique_basenames(paths: Sequence[Path]) -> None:
    names = [p.name for p in paths]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"input basenames collide in the output directory: {dupes}")


def exact_line_dedup(input_paths: Sequence[_PathLike], out_dir: _PathLike) -> list[Path]:
    """Rewrite each input file into ``out_dir`` keeping only corpus-unique lines.

    Two passes, both streaming: (1) count BLAKE2b line digests across *all* files; (2) rewrite
    each file, keeping a line iff its corpus-wide count is 1. Memory is O(#distinct lines) in
    16-byte digests — never raw text. Line identity ignores the ``\\r\\n``/``\\n`` terminator
    (and its absence at EOF); kept lines are written back byte-identically, in original order.

    Matches the official ``run_exact_line_deduplication(input_files, output_directory)`` adapter.
    Returns the written output paths (one per input, same basename — possibly empty files).
    """
    paths = [Path(p) for p in input_paths]
    _check_unique_basenames(paths)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    counts: Counter[bytes] = Counter()
    for path in paths:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            for line in f:
                counts[_line_key(line.rstrip("\r\n"))] += 1

    written: list[Path] = []
    for path in paths:
        out_path = out / path.name
        with (
            open(path, encoding="utf-8", errors="replace", newline="") as src,
            open(out_path, "w", encoding="utf-8", newline="") as dst,
        ):
            for line in src:
                if counts[_line_key(line.rstrip("\r\n"))] == 1:
                    dst.write(line)
        written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# MinHash / LSH primitives (pure, exposed for tests)
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """MinHash canonical form: NFD → lowercase → drop combining marks (accents) and punctuation
    → collapse all whitespace runs to single spaces.

    This *defines* what counts as a near-duplicate: two texts differing only in case, accents,
    punctuation, or whitespace normalize identically (Jaccard 1.0).
    """
    decomposed = unicodedata.normalize("NFD", text).lower()
    kept: list[str] = []
    for ch in decomposed:
        cat = unicodedata.category(ch)
        if cat.startswith(("M", "P")):  # Mn/Mc/Me combining marks; P* punctuation
            continue
        kept.append(ch)
    return " ".join("".join(kept).split())


def ngram_set(text: str, n: int) -> set[tuple[str, ...]]:
    """Word n-grams of the *normalized* text, as a set of word tuples.

    A doc with fewer than ``n`` normalized words yields the empty set — such docs carry no
    MinHash evidence and are always kept by ``minhash_dedup``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    words = normalize_text(text).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _stable_hash64(item: tuple[str, ...] | str | bytes) -> int:
    """Deterministic 64-bit base hash of an n-gram (never Python's salted ``hash()``)."""
    if isinstance(item, tuple):
        payload = "\x1f".join(item).encode("utf-8")
    elif isinstance(item, str):
        payload = item.encode("utf-8")
    else:
        payload = item
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _hash_family(num_hashes: int, seed: int) -> list[tuple[int, int]]:
    """k universal-hash coefficient pairs ``(a, b)``: ``h_i(x) = (a·x + b) mod (2^61 − 1)``."""
    rng = random.Random(seed)
    return [
        (rng.randrange(1, _MERSENNE_61), rng.randrange(_MERSENNE_61)) for _ in range(num_hashes)
    ]


def minhash_signature(
    items: Iterable[tuple[str, ...] | str | bytes],
    num_hashes: int,
    seed: int = DEFAULT_SEED,
) -> list[int]:
    """MinHash signature: ``sig[i] = min over items of h_i(item)`` for k seeded hash functions.

    The one identity everything rests on: for two sets, ``P[sig_A[i] == sig_B[i]] =
    Jaccard(A, B)`` (each h_i induces a near-uniform random ordering; the sets' minima agree iff
    the union's minimum lies in the intersection). Raises on an empty item set — an empty set has
    no minimum and a sentinel would silently match other empty sets.
    """
    base = [_stable_hash64(item) for item in set(items)]
    if not base:
        raise ValueError("cannot MinHash an empty set (no minimum exists)")
    return [
        min((a * x + b) % _MERSENNE_61 for x in base) for a, b in _hash_family(num_hashes, seed)
    ]


def lsh_candidates(signatures: Sequence[Sequence[int]], num_bands: int) -> set[tuple[int, int]]:
    """Candidate pairs ``(i, j)``, i < j, whose signatures collide in ≥ 1 of ``num_bands`` bands.

    Splits each k-row signature into b bands of r = k/b rows; docs sharing any band's exact
    r-tuple land in one bucket, and every within-bucket pair becomes a candidate. For a pair at
    Jaccard s the collision probability is ``1 − (1 − s^r)^b`` — candidates still need the
    true-Jaccard confirm (LSH trades the O(N²) all-pairs scan for bucketed near-linear work,
    it does not decide duplication).
    """
    if not signatures:
        return set()
    k = len(signatures[0])
    if any(len(sig) != k for sig in signatures):
        raise ValueError("all signatures must have the same length")
    if num_bands < 1 or k % num_bands != 0:
        raise ValueError(f"num_bands={num_bands} must divide the signature length k={k}")
    rows = k // num_bands

    buckets: defaultdict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for doc_idx, sig in enumerate(signatures):
        for band in range(num_bands):
            key = (band, tuple(sig[band * rows : (band + 1) * rows]))
            buckets[key].append(doc_idx)

    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i_pos in range(len(members)):
            for j_pos in range(i_pos + 1, len(members)):
                pairs.add((members[i_pos], members[j_pos]))
    return pairs


def jaccard(a: set, b: set) -> float:
    """``|A ∩ B| / |A ∪ B|``; two empty sets are identical (1.0) by convention."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def lsh_collision_probability(similarity: float, num_bands: int, rows_per_band: int) -> float:
    """The LSH S-curve ``P[candidate] = 1 − (1 − s^r)^b`` — the (b, r) precision/recall dial.

    Monotone increasing in b at fixed k = b·r; knee near ``s ≈ (1/b)^(1/r)``. Used to *justify*
    a band split (ADR-0015), not in the dedup hot path.
    """
    return 1.0 - (1.0 - similarity**rows_per_band) ** num_bands


# ---------------------------------------------------------------------------
# Union-find (transitive duplicate clusters)
# ---------------------------------------------------------------------------


class _UnionFind:
    """Connected components over doc indices: A~B and B~C ⇒ {A,B,C} is one cluster.

    Dropping both members of every confirmed pair would over-delete; clustering keeps exactly
    one representative per component.
    """

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, x: int, y: int) -> None:
        # Attach the larger root under the smaller so every root is its cluster's min index —
        # the deterministic "keep first in input order" choice falls out of find() directly.
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if ry < rx:
            rx, ry = ry, rx
        self._parent[ry] = rx


# ---------------------------------------------------------------------------
# MinHash dedup driver
# ---------------------------------------------------------------------------


def minhash_dedup(
    input_paths: Sequence[_PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    out_dir: _PathLike,
    *,
    seed: int = DEFAULT_SEED,
) -> list[Path]:
    """Fuzzy-dedup documents; copy one survivor per duplicate cluster into ``out_dir``.

    Pipeline: n-gram sets (normalized, ``ngrams`` words) → k = ``num_hashes`` MinHash rows →
    LSH over ``num_bands`` bands → candidate pairs → true-Jaccard confirm ≥
    ``jaccard_threshold`` → union-find transitive closure → survivor = lowest input index per
    cluster. Docs with fewer than ``ngrams`` normalized words have no evidence and always
    survive. Survivor files are byte-identical copies (same basename).

    Matches the official ``run_minhash_deduplication(input_files, num_hashes, num_bands,
    ngrams, jaccard_threshold, output_directory)`` adapter. Returns surviving output paths in
    input order.
    """
    if num_hashes < 1 or num_bands < 1 or num_hashes % num_bands != 0:
        raise ValueError(
            f"num_bands={num_bands} must divide num_hashes={num_hashes} (k = bands x rows)"
        )
    if not 0.0 < jaccard_threshold <= 1.0:
        raise ValueError(f"jaccard_threshold must be in (0, 1], got {jaccard_threshold}")

    paths = [Path(p) for p in input_paths]
    _check_unique_basenames(paths)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    doc_ngrams: list[set[tuple[str, ...]]] = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            doc_ngrams.append(ngram_set(f.read(), ngrams))

    # Only docs with evidence (non-empty n-gram sets) enter the LSH index.
    hashable = [i for i, grams in enumerate(doc_ngrams) if grams]
    signatures = [minhash_signature(doc_ngrams[i], num_hashes, seed=seed) for i in hashable]

    uf = _UnionFind(len(paths))
    for local_i, local_j in lsh_candidates(signatures, num_bands):
        i, j = hashable[local_i], hashable[local_j]
        if jaccard(doc_ngrams[i], doc_ngrams[j]) >= jaccard_threshold:
            uf.union(i, j)

    survivors: list[Path] = []
    for idx, path in enumerate(paths):
        if uf.find(idx) != idx:  # cluster root == min input index (see _UnionFind.union)
            continue
        out_path = out / path.name
        shutil.copyfile(path, out_path)
        survivors.append(out_path)
    return survivors


# ---------------------------------------------------------------------------
# Official A4 adapter names (W9 wiring is `from scratch_llm.data.dedup import ...`)
# ---------------------------------------------------------------------------


def run_exact_line_deduplication(
    input_files: Sequence[_PathLike], output_directory: _PathLike
) -> list[Path]:
    """Official-adapter alias for :func:`exact_line_dedup`."""
    return exact_line_dedup(input_files, output_directory)


def run_minhash_deduplication(
    input_files: Sequence[_PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: _PathLike,
) -> list[Path]:
    """Official-adapter alias for :func:`minhash_dedup`."""
    return minhash_dedup(
        input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory
    )
