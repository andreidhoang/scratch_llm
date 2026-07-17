"""Real-corpus token shards — the pretraining data path (A1, close-the-loop front).

Turns a document stream (FineWeb-EDU or any list of texts) into durable, memmap-able token
shards that ``train.py::get_batch`` consumes unchanged — replacing the in-RAM toy corpus as
the substrate every downstream rung (A2 checkpoint chaining, F1-run, midtrain/SFT, the d20)
builds on.

Format (nanoGPT/nanochat convention, headerless): raw native-little-endian token ids, one
``<|eot|>`` id appended after every document, dtype uint16 while every id < 2¹⁶ — with the
pre-registered kill-switch to uint32 the moment any id crosses it (no silent wraparound).
Metadata lives in a ``<stem>.meta.json`` sidecar so the ``.bin`` stays exactly
``itemsize·n_tokens`` bytes. The tokenizer that produced the shards is staged alongside them
(``tokenizer.json``) — shards are meaningless without the vocab that indexed them.

Invariants (tested in tests/test_shards.py, pre-registered in bench/RESULTS.md):
- **Round-trip:** shard bytes → ids == ``encode(d0)+[eot]+encode(d1)+[eot]``.
- **Size:** file bytes == ``itemsize·n_tokens``, zero header/padding.
- **Alignment:** ``get_batch`` on the loaded memmap is next-token-aligned.

Scope note (§D): the base path (``build_dataset``) is nano-scale — documents are buffered in
RAM per shard. The F1-scale bulk path also lives here: hub parquet discovery → plan →
download (``list_hub_parquet_files`` / ``plan_parquet_download`` / ``download_parquet_files``)
feeding ``build_dataset_streaming``, where doc batches flow parquet → tokenizer (optionally a
``multiprocessing.Pool``) → shard writer without ever holding the corpus in RAM, stopping
within one doc-batch of ``target_tokens``. The ~11B-token *shuffled* multi-shard streamer for
the d20 remains a deferred extension.

Interview question this answers: "why memmap uint16 shards with an EOT separator, and what
breaks if you train across document boundaries without one?"
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import pickle
import tempfile
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from scratch_llm.tokenizer import Tokenizer, train_bpe

# The document separator. Matches the `<|eot|>` special the A3 chat template will formalize,
# so shards built today stay valid when chat specials land.
DOC_SEPARATOR = "<|eot|>"

_UINT16_MAX = 2**16 - 1


class TokenEncoder(Protocol):
    """The minimal tokenizer surface a shard builder needs."""

    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class ShardMeta:
    """Sidecar metadata for one shard (`<stem>.meta.json`). The .bin itself is headerless."""

    n_tokens: int
    n_docs: int
    dtype: str  # "uint16" | "uint32" — the pre-registered kill-switch axis
    eot_id: int
    max_token_id: int
    # Docs rejected by the A0 ``doc_filter`` seam (decontamination). Defaults to 0 so pre-A0
    # sidecars (missing the key) keep loading unchanged.
    n_docs_filtered: int = 0


def _meta_path(shard_path: Path) -> Path:
    return shard_path.with_suffix(".meta.json")


def _write_shard(
    ids: list[int], n_docs: int, n_docs_filtered: int, eot_id: int, out_path: Path
) -> ShardMeta:
    """Write one headerless shard + sidecar from already-encoded ids (EOTs included).

    The single write path for both the buffered (:func:`tokenize_to_shard`) and streaming
    (:func:`stream_tokenize_to_shards`) builders — byte-identical output by construction.
    ``ids`` must be non-empty and already carry one ``eot_id`` after every document.
    """
    max_token_id = max(max(ids), eot_id)
    # Kill-switch (pre-registered): any id ≥ 2¹⁶ silently wrapping in uint16 would corrupt the
    # corpus — switch the whole shard to uint32 instead.
    dtype = np.uint16 if max_token_id <= _UINT16_MAX else np.uint32
    arr = np.asarray(ids, dtype=np.int64)
    arr.astype(dtype).tofile(out_path)

    meta = ShardMeta(
        n_tokens=len(ids),
        n_docs=n_docs,
        dtype=np.dtype(dtype).name,
        eot_id=eot_id,
        max_token_id=int(max_token_id),
        n_docs_filtered=n_docs_filtered,
    )
    _meta_path(out_path).write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    return meta


def tokenize_to_shard(
    docs: Iterable[str],
    tokenizer: TokenEncoder,
    eot_id: int,
    out_path: str | Path,
    *,
    doc_filter: Callable[[str], bool] | None = None,
) -> ShardMeta:
    """Encode ``docs``, appending ``eot_id`` after every document, and write the shard.

    The EOT after *every* doc (not between) makes concatenated shards seamless and gives the
    model an unambiguous document boundary — without it, windows sampled across two unrelated
    documents teach spurious long-range dependencies.

    ``doc_filter`` (A0 seam, optional): a keep-predicate (True = keep) applied to each doc
    BEFORE tokenization — e.g. :func:`scratch_llm.data.decontaminate.decontam_doc_filter`.
    ``None`` is byte-identical to the pre-seam behavior; rejected docs are counted in
    ``ShardMeta.n_docs_filtered``.
    """
    out_path = Path(out_path)
    ids: list[int] = []
    n_docs = 0
    n_docs_filtered = 0
    for doc in docs:
        if doc_filter is not None and not doc_filter(doc):
            n_docs_filtered += 1
            continue
        ids.extend(tokenizer.encode(doc))
        ids.append(eot_id)
        n_docs += 1
    if not ids:
        if n_docs_filtered:
            raise ValueError(
                f"tokenize_to_shard: doc_filter rejected all {n_docs_filtered} document(s)"
            )
        raise ValueError("tokenize_to_shard needs at least one document")

    return _write_shard(ids, n_docs, n_docs_filtered, eot_id, out_path)


def load_shard(path: str | Path) -> tuple[np.ndarray, ShardMeta]:
    """Load one shard as a read-only memmap plus its sidecar metadata.

    Validates the headerless-size invariant (file bytes == itemsize·n_tokens) so a truncated
    or foreign file fails loudly instead of training on garbage.
    """
    path = Path(path)
    meta_raw = json.loads(_meta_path(path).read_text(encoding="utf-8"))
    meta = ShardMeta(**meta_raw)
    itemsize = np.dtype(meta.dtype).itemsize
    actual = path.stat().st_size
    if actual != itemsize * meta.n_tokens:
        raise ValueError(
            f"{path} size mismatch: {actual} bytes != {itemsize}·{meta.n_tokens} "
            "(truncated shard or wrong sidecar?)"
        )
    tokens = np.memmap(path, dtype=np.dtype(meta.dtype), mode="r")
    return tokens, meta


def load_dataset_tokens(data_dir: str | Path, shard_glob: str = "*.bin") -> np.ndarray:
    """Load every shard in ``data_dir`` (sorted by name) as one flat token array.

    A single shard stays a zero-copy memmap; multiple shards are concatenated in RAM — fine at
    nano scale, and exactly the boundary where the §D d20 streamer takes over.
    """
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob(shard_glob))
    if not paths:
        raise FileNotFoundError(f"no shards matching {shard_glob!r} in {data_dir}")
    if len(paths) == 1:
        return load_shard(paths[0])[0]
    return np.concatenate([load_shard(p)[0] for p in paths])


# ---------------------------------------------------------------------------------------------
# Tokenizer staging — shards are only meaningful with the vocab that produced them.
# ---------------------------------------------------------------------------------------------

_TOKENIZER_FILE = "tokenizer.json"


def save_tokenizer_files(tokenizer: Tokenizer, out_dir: str | Path) -> Path:
    """Stage the tokenizer next to the shards as one JSON file.

    Since A2, a thin delegate to :meth:`Tokenizer.save` (which canonicalized this module's
    JSON format — same payload, so pre-A2 ``tokenizer.json`` files keep loading). The
    two-file ``Tokenizer.from_files`` format stays unused here: its ``A B``-per-line merges
    file is ambiguous when the left symbol itself contains a space.
    """
    return tokenizer.save(Path(out_dir) / _TOKENIZER_FILE)


def load_tokenizer(data_dir: str | Path) -> Tokenizer:
    """Rebuild the exact tokenizer staged by :func:`save_tokenizer_files`."""
    return Tokenizer.load(Path(data_dir) / _TOKENIZER_FILE)


# ---------------------------------------------------------------------------------------------
# One-call dataset build: docs → tokenizer.json + shard_%05d.bin (+ sidecars)
# ---------------------------------------------------------------------------------------------


def build_dataset(
    docs: Sequence[str],
    out_dir: str | Path,
    vocab_size: int,
    docs_per_shard: int | None = None,
    *,
    doc_filter: Callable[[str], bool] | None = None,
) -> list[ShardMeta]:
    """Train a byte-level BPE on ``docs`` (with the EOT special), stage it, and shard the corpus.

    ``docs_per_shard=None`` writes one shard; otherwise documents are chunked in order. The
    tokenizer is trained on the same documents it shards — so ``doc_filter`` (A0's
    decontamination gate, True = keep) is applied ONCE, up front, before BPE training: eval
    text must not shape the merges either. ``None`` is byte-identical to the pre-seam
    behavior. For dropped-doc accounting, run
    :func:`scratch_llm.data.decontaminate.decontaminate_docs` first and pass ``result.kept``.
    """
    if not docs:
        raise ValueError("build_dataset needs at least one document")
    if doc_filter is not None:
        n_before = len(docs)
        docs = [doc for doc in docs if doc_filter(doc)]
        if not docs:
            raise ValueError(f"build_dataset: doc_filter rejected all {n_before} document(s)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # train_bpe reads a file; stage the corpus, separators included (the trainer splits them
    # out pre-tokenization, so they only mark boundaries — they are never merged across).
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write(DOC_SEPARATOR.join(docs))
        staged = Path(fh.name)
    try:
        vocab, merges = train_bpe(staged, vocab_size, special_tokens=[DOC_SEPARATOR])
    finally:
        staged.unlink(missing_ok=True)
    tokenizer = Tokenizer(vocab, merges, special_tokens=[DOC_SEPARATOR])
    save_tokenizer_files(tokenizer, out_dir)

    eot_ids = tokenizer.encode(DOC_SEPARATOR)
    if len(eot_ids) != 1:
        raise RuntimeError(f"{DOC_SEPARATOR!r} must encode to exactly one id, got {eot_ids}")

    chunk = len(docs) if docs_per_shard is None else docs_per_shard
    metas: list[ShardMeta] = []
    for shard_idx, start in enumerate(range(0, len(docs), chunk)):
        metas.append(
            tokenize_to_shard(
                docs[start : start + chunk],
                tokenizer,
                eot_ids[0],
                out_dir / f"shard_{shard_idx:05d}.bin",
            )
        )
    return metas


# ---------------------------------------------------------------------------------------------
# FineWeb-EDU slice — network path (stdlib only; tests self-skip without the opt-in env var)
# ---------------------------------------------------------------------------------------------


def download_fineweb_slice(
    n_docs: int = 64,
    *,
    dataset: str = "HuggingFaceFW/fineweb-edu",
    config: str = "sample-10BT",
    split: str = "train",
    offset: int = 0,
    timeout: float = 30.0,
) -> list[str]:
    """Fetch ``n_docs`` document texts via the HF datasets-server rows API (no extra deps).

    Nano-scale on purpose: the API pages 100 rows at a time (measured ~800 tokens/doc, so
    even thousands of docs is only ~millions of tokens) — F1-scale corpora come from the
    parquet bulk path below (``list_hub_parquet_files`` → ``build_dataset_streaming``),
    never this endpoint.
    """
    docs: list[str] = []
    while len(docs) < n_docs:
        length = min(100, n_docs - len(docs))
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={urllib.parse.quote(dataset, safe='')}"
            f"&config={config}&split={split}&offset={offset + len(docs)}&length={length}"
        )
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (https, fixed host)
            payload = json.load(resp)
        rows = payload.get("rows", [])
        if not rows:
            raise RuntimeError(f"datasets-server returned no rows at offset {offset + len(docs)}")
        docs.extend(str(r["row"]["text"]) for r in rows)
    return docs[:n_docs]


# ---------------------------------------------------------------------------------------------
# FineWeb-EDU parquet bulk path (F1 scale) — hub discovery / plan / download (stdlib only)
# ---------------------------------------------------------------------------------------------

_HF_HUB = "https://huggingface.co"
FINEWEB_EDU_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_SAMPLE_10BT = "sample/10BT"

# Compressed parquet bytes per token, used ONLY to size the download plan. [FACT, hub API
# 2026-07-16]: sample-10BT = 14 files, 28,518,193,415 B for the card's ~9.67B GPT-2 tokens
# ⇒ ≈ 2.95 B/token. [INFERENCE] our 32k byte-BPE yields ≥ tokens/byte than GPT-2's 50k vocab,
# so this estimate errs toward over-supply — and the exact stop is ``target_tokens`` in the
# streaming builder anyway (``StreamStats.bytes_per_token`` measures the realized ratio);
# ``plan_parquet_download``'s safety margin absorbs the residual error.
PARQUET_BYTES_PER_TOKEN = 2.95


@dataclass(frozen=True)
class ParquetFileInfo:
    """One hub-hosted parquet file: repo-relative path, byte size, resolved download URL."""

    path: str
    size: int
    url: str


def hub_tree_parquet_infos(
    entries: Iterable[dict], *, dataset: str, revision: str = "main"
) -> list[ParquetFileInfo]:
    """Pure: hub tree-API JSON entries → sorted parquet file infos (files only, ``*.parquet``).

    Split from :func:`list_hub_parquet_files` so the parse/URL contract is testable without
    the network. Download URLs use the hub's ``/resolve/`` route (redirects to the CDN).
    """
    infos = [
        ParquetFileInfo(
            path=str(e["path"]),
            size=int(e["size"]),
            url=(
                f"{_HF_HUB}/datasets/{dataset}/resolve/{revision}/"
                f"{urllib.parse.quote(str(e['path']), safe='/')}"
            ),
        )
        for e in entries
        if e.get("type") == "file" and str(e.get("path", "")).endswith(".parquet")
    ]
    return sorted(infos, key=lambda f: f.path)


def _next_link(link_header: str | None) -> str | None:
    """Extract the ``rel="next"`` URL from an RFC-5988 Link header (hub API pagination)."""
    if not link_header:
        return None
    for part in link_header.split(","):
        url_part, _, params = part.partition(";")
        if 'rel="next"' in params:
            return url_part.strip().strip("<>")
    return None


def list_hub_parquet_files(
    dataset: str = FINEWEB_EDU_DATASET,
    subdir: str = FINEWEB_EDU_SAMPLE_10BT,
    *,
    revision: str = "main",
    timeout: float = 30.0,
) -> list[ParquetFileInfo]:
    """Discover the parquet files under ``dataset/subdir`` via the hub tree API (paged).

    Discovery, never a hardcoded naming guess: the hub is the source of truth for file names,
    and the byte sizes it reports drive :func:`plan_parquet_download`.
    """
    url: str | None = (
        f"{_HF_HUB}/api/datasets/{urllib.parse.quote(dataset, safe='/')}"
        f"/tree/{revision}/{urllib.parse.quote(subdir, safe='/')}"
    )
    entries: list[dict] = []
    while url:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (https, fixed host)
            entries.extend(json.load(resp))
            url = _next_link(resp.headers.get("Link"))
    infos = hub_tree_parquet_infos(entries, dataset=dataset, revision=revision)
    if not infos:
        raise RuntimeError(f"no parquet files under {dataset}/{subdir}@{revision}")
    return infos


@dataclass(frozen=True)
class DownloadPlan:
    """A dry-runnable download plan: which parquet files cover ``target_tokens``."""

    files: tuple[ParquetFileInfo, ...]
    target_tokens: int
    bytes_per_token: float

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def est_tokens(self) -> int:
        """Estimated tokens in the planned bytes — an [INFERENCE] estimate, not a measurement."""
        return int(self.total_bytes / self.bytes_per_token)

    def describe(self) -> str:
        """The dry-run report: file list + sizes + the token estimate. Nothing is downloaded."""
        lines = [
            f"download plan: {len(self.files)} file(s), {self.total_bytes / 2**30:.2f} GiB "
            f"≈ {self.est_tokens:.3g} tokens (target {self.target_tokens:.3g} "
            f"@ {self.bytes_per_token} B/token est.)"
        ]
        lines += [f"  {f.path}  {f.size / 2**20:,.1f} MiB" for f in self.files]
        return "\n".join(lines)


def plan_parquet_download(
    files: Sequence[ParquetFileInfo],
    target_tokens: int,
    *,
    bytes_per_token: float = PARQUET_BYTES_PER_TOKEN,
    safety: float = 1.25,
) -> DownloadPlan:
    """Pick the file prefix whose bytes cover ``target_tokens`` (× ``safety`` headroom).

    ``safety`` absorbs the [INFERENCE] bytes-per-token estimate error; the exact stop happens
    later, in :func:`stream_tokenize_to_shards` via ``target_tokens``. If every file together
    still falls short, the plan is simply all of them (``describe()`` then shows
    ``est_tokens`` < target — an under-supplied corpus fails loudly at the accounting stage,
    not silently).
    """
    if not files:
        raise ValueError("plan_parquet_download needs at least one file")
    need_bytes = target_tokens * bytes_per_token * safety
    picked: list[ParquetFileInfo] = []
    got = 0
    for f in files:
        picked.append(f)
        got += f.size
        if got >= need_bytes:
            break
    return DownloadPlan(
        files=tuple(picked), target_tokens=target_tokens, bytes_per_token=bytes_per_token
    )


def download_parquet_files(
    files: Sequence[ParquetFileInfo],
    dest_dir: str | Path,
    *,
    timeout: float = 60.0,
    progress: Callable[[str], None] | None = None,
) -> list[Path]:
    """Download ``files`` into ``dest_dir`` (flat, by basename), returning local paths in order.

    Resumable at file granularity: a file already present with the exact hub-reported size is
    skipped; in-flight bytes land on a ``.part`` name and are renamed only when complete, so a
    killed run never leaves a truncated ``.parquet`` masquerading as a good one.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for info in files:
        dest = dest_dir / Path(info.path).name
        if dest.exists() and dest.stat().st_size == info.size:
            if progress is not None:
                progress(f"cached  {dest.name} ({info.size / 2**20:,.1f} MiB)")
            paths.append(dest)
            continue
        part = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(info.url, timeout=timeout) as resp, part.open("wb") as out:  # noqa: S310 (https, hub-resolved)
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        part.replace(dest)
        if progress is not None:
            progress(f"fetched {dest.name} ({dest.stat().st_size / 2**20:,.1f} MiB)")
        paths.append(dest)
    return paths


def _require_pyarrow_parquet():
    """Guarded import: only the parquet ingest path needs the optional ``data`` extra."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for the parquet ingest path — install the data extra: "
            'uv pip install -e ".[data]" (or: pip install pyarrow)'
        ) from exc
    return pq


def iter_parquet_doc_batches(
    paths: Sequence[str | Path],
    *,
    text_column: str = "text",
    batch_size: int = 1024,
) -> Iterator[list[str]]:
    """Stream document texts batch-wise from parquet files — bounded RAM, order-stable.

    Reads ``text_column`` only (columnar projection: other columns never leave disk) via
    ``ParquetFile.iter_batches``, so peak memory is one record batch, never a whole file.
    Null/empty texts are dropped — they carry no tokens and would emit bare-EOT documents.
    """
    pq = _require_pyarrow_parquet()
    for path in paths:
        with pq.ParquetFile(path) as pf:
            for batch in pf.iter_batches(batch_size=batch_size, columns=[text_column]):
                docs = [str(t) for t in batch.column(text_column).to_pylist() if t]
                if docs:
                    yield docs


# ---------------------------------------------------------------------------------------------
# Streaming build: doc batches → (multiprocess) tokenizer → shard writer, bounded RAM
# ---------------------------------------------------------------------------------------------

DEFAULT_TOKENS_PER_SHARD = 1 << 24  # ≈16.8M tokens ⇒ 32 MiB/shard at uint16 (~42 shards @ 7e8)

# Byte cap on the BPE *training* sample (doc-granular; see build_dataset_streaming). The
# pure-Python trainer, not the corpus, is the bound: MEASURED 2 MiB natural text → 3.4 s at
# vocab 2048 / 15.0 s at vocab 8192 (M-series dev box, 2026-07-16). Pretokenization is linear
# in bytes and the merge loop is O(merges · distinct pairs), so 16 MiB @ vocab 32768 lands in
# the tens-of-minutes range — bounded; GB-scale is not viable. Merge statistics saturate far
# below the cap anyway; raise per-run via --max-train-bytes if vocab coverage needs it.
DEFAULT_BPE_TRAIN_BYTES = 16 << 20

# Set once per worker process by the Pool initializer — pickling the tokenizer per batch
# would swamp the encode work itself.
_WORKER_ENCODER: TokenEncoder | None = None


def _init_worker_encoder(tokenizer: TokenEncoder) -> None:
    global _WORKER_ENCODER
    _WORKER_ENCODER = tokenizer


def _encode_doc_batch(docs: Sequence[str]) -> list[list[int]]:
    """Worker task: encode one doc batch. EOTs are appended by the parent (single writer)."""
    assert _WORKER_ENCODER is not None, "pool worker used before _init_worker_encoder"
    return [_WORKER_ENCODER.encode(d) for d in docs]


def _iter_encoded_batches(
    batches: Iterable[Sequence[str]], tokenizer: TokenEncoder, num_workers: int
) -> Iterator[list[list[int]]]:
    """Encode batches order-stably: ``Pool.imap`` when ``num_workers > 1``, else in-process.

    Falls back to single-process when the tokenizer cannot cross a process boundary (pickle
    pre-flight) — identical output either way, only wall-clock differs. Early termination by
    the consumer tears the pool down via the ``with`` block.
    """
    if num_workers > 1:
        try:
            pickle.dumps(tokenizer)
        except Exception:
            num_workers = 1  # process-local encoder: fall back rather than crash in a worker
    if num_workers > 1:
        with multiprocessing.Pool(
            num_workers, initializer=_init_worker_encoder, initargs=(tokenizer,)
        ) as pool:
            yield from pool.imap(_encode_doc_batch, batches)
    else:
        for batch in batches:
            yield [tokenizer.encode(d) for d in batch]


@dataclass(frozen=True)
class StreamStats:
    """Accounting for one streaming build — docs, tokens, text bytes, shards, drops.

    ``bytes_per_token`` is measured over the *kept* docs' utf-8 text bytes: the number the
    download planner's [INFERENCE] estimate gets reconciled against on a real run.
    """

    n_docs: int
    n_tokens: int
    n_text_bytes: int
    n_shards: int
    n_docs_filtered: int

    @property
    def bytes_per_token(self) -> float:
        return self.n_text_bytes / self.n_tokens if self.n_tokens else 0.0


def stream_tokenize_to_shards(
    doc_batches: Iterable[Sequence[str]],
    tokenizer: TokenEncoder,
    eot_id: int,
    out_dir: str | Path,
    *,
    tokens_per_shard: int = DEFAULT_TOKENS_PER_SHARD,
    target_tokens: int | None = None,
    num_workers: int = 0,
    doc_filter: Callable[[str], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[ShardMeta], StreamStats]:
    """Streaming counterpart of :func:`tokenize_to_shard`: doc batches → shards, bounded RAM.

    Peak memory is one shard buffer (``tokens_per_shard`` ids, + at most one doc of overshoot)
    plus the in-flight batches — never the corpus. Shards split at *document* boundaries (a
    flush happens once the buffer reaches ``tokens_per_shard``), so every shard is a valid
    ``encode(d)+[eot]…`` stream, and output is byte-identical to the buffered path (both
    funnel through :func:`_write_shard`).

    ``target_tokens`` stops the build within ONE doc-batch of the budget: the batch that
    crosses it is kept whole, so token accounting stays exact and the stop is batch-granular.
    ``num_workers > 1`` encodes batches in a ``multiprocessing.Pool`` (order-stable ``imap``;
    single-process fallback if the tokenizer does not pickle). ``doc_filter`` (A0 seam) runs
    in the parent so multi- and single-process output are identical; drops are counted in
    ``StreamStats.n_docs_filtered`` — per-shard sidecars keep the format's default 0, because
    a shard boundary has no principled attribution of dropped docs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-batch side ledger (n_filtered, kept_text_bytes): imap yields one result per input,
    # in order, so a FIFO popleft realigns accounting with results. Entries for batches that
    # were prefetched but never consumed (early target_tokens stop) are simply dropped.
    ledger: deque[tuple[int, int]] = deque()

    def kept_batches() -> Iterator[list[str]]:
        for batch in doc_batches:
            kept = list(batch) if doc_filter is None else [d for d in batch if doc_filter(d)]
            ledger.append((len(batch) - len(kept), sum(len(d.encode("utf-8")) for d in kept)))
            yield kept  # empty batches are still dispatched: keeps the ledger aligned

    buf: list[int] = []
    buf_docs = 0
    metas: list[ShardMeta] = []
    n_docs = n_tokens = n_text_bytes = n_docs_filtered = 0

    def flush() -> None:
        nonlocal buf, buf_docs
        meta = _write_shard(buf, buf_docs, 0, eot_id, out_dir / f"shard_{len(metas):05d}.bin")
        metas.append(meta)
        if progress is not None:
            budget = f" / {target_tokens:,}" if target_tokens is not None else ""
            progress(
                f"shard_{len(metas) - 1:05d}: {meta.n_tokens:,} tokens, {meta.n_docs:,} docs "
                f"(total {n_tokens:,}{budget})"
            )
        buf = []
        buf_docs = 0

    for encoded in _iter_encoded_batches(kept_batches(), tokenizer, num_workers):
        batch_filtered, batch_text_bytes = ledger.popleft()
        n_docs_filtered += batch_filtered
        n_text_bytes += batch_text_bytes
        for ids in encoded:
            buf.extend(ids)
            buf.append(eot_id)
            buf_docs += 1
            n_tokens += len(ids) + 1
            if len(buf) >= tokens_per_shard:
                flush()
        n_docs += len(encoded)
        if target_tokens is not None and n_tokens >= target_tokens:
            break
    if buf:
        flush()

    if not metas:
        if n_docs_filtered:
            raise ValueError(
                f"stream_tokenize_to_shards: doc_filter rejected all {n_docs_filtered} document(s)"
            )
        raise ValueError("stream_tokenize_to_shards needs at least one document")

    stats = StreamStats(
        n_docs=n_docs,
        n_tokens=n_tokens,
        n_text_bytes=n_text_bytes,
        n_shards=len(metas),
        n_docs_filtered=n_docs_filtered,
    )
    return metas, stats


def build_dataset_streaming(
    doc_batches: Callable[[], Iterable[Sequence[str]]],
    out_dir: str | Path,
    vocab_size: int,
    *,
    max_train_bytes: int = DEFAULT_BPE_TRAIN_BYTES,
    tokens_per_shard: int = DEFAULT_TOKENS_PER_SHARD,
    target_tokens: int | None = None,
    num_workers: int = 0,
    doc_filter: Callable[[str], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[ShardMeta], StreamStats]:
    """Streaming counterpart of :func:`build_dataset`: BPE on a capped sample, then shard.

    ``doc_batches`` is a ZERO-ARG FACTORY (e.g. ``lambda: iter_parquet_doc_batches(paths)``)
    because the stream is consumed twice — once truncated for BPE training, once fully for
    sharding — and a plain iterator would arrive exhausted at the second pass.

    BPE-training cap: the sample is staged doc-by-doc to a temp file until ``max_train_bytes``
    utf-8 text bytes (doc-granular — the doc that crosses the cap is kept whole, then staging
    stops). Rationale at :data:`DEFAULT_BPE_TRAIN_BYTES`: the pure-Python trainer is the
    bound, and merge statistics saturate far below F1 corpus size. ``doc_filter`` applies to
    the sample too — eval text must not shape the merges (same rule as
    :func:`build_dataset`). When uncapped (cap ≥ corpus), the staged bytes — and therefore the
    tokenizer and the shards — are identical to :func:`build_dataset` on the same docs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    staged_bytes = 0
    staged_docs = 0
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        staged = Path(fh.name)
        for batch in doc_batches():
            for doc in batch:
                if doc_filter is not None and not doc_filter(doc):
                    continue
                if staged_docs:
                    fh.write(DOC_SEPARATOR)
                fh.write(doc)
                staged_docs += 1
                staged_bytes += len(doc.encode("utf-8"))
                if staged_bytes >= max_train_bytes:
                    break
            if staged_bytes >= max_train_bytes:
                break
    try:
        if not staged_docs:
            raise ValueError("build_dataset_streaming: no documents survived for BPE training")
        if progress is not None:
            progress(
                f"BPE sample: {staged_docs:,} docs, {staged_bytes / 2**20:.1f} MiB "
                f"(cap {max_train_bytes / 2**20:.0f} MiB) → train_bpe(vocab_size={vocab_size})"
            )
        vocab, merges = train_bpe(staged, vocab_size, special_tokens=[DOC_SEPARATOR])
    finally:
        staged.unlink(missing_ok=True)
    tokenizer = Tokenizer(vocab, merges, special_tokens=[DOC_SEPARATOR])
    save_tokenizer_files(tokenizer, out_dir)

    eot_ids = tokenizer.encode(DOC_SEPARATOR)
    if len(eot_ids) != 1:
        raise RuntimeError(f"{DOC_SEPARATOR!r} must encode to exactly one id, got {eot_ids}")

    return stream_tokenize_to_shards(
        doc_batches(),
        tokenizer,
        eot_ids[0],
        out_dir,
        tokens_per_shard=tokens_per_shard,
        target_tokens=target_tokens,
        num_workers=num_workers,
        doc_filter=doc_filter,
        progress=progress,
    )


def _decontam_filter(eval_files: list[str]) -> Callable[[str], bool]:
    """Build the A0 decontamination keep-predicate (lazy import: decontaminate stays optional)."""
    from scratch_llm.data.decontaminate import (
        build_eval_ngrams,
        decontam_doc_filter,
        default_eval_texts,
    )

    return decontam_doc_filter(build_eval_ngrams(default_eval_texts(eval_files)))


def main() -> None:
    p = argparse.ArgumentParser(description="Build token shards (A1): FineWeb/corpus → memmap.")
    p.add_argument("--out", required=True, help="Output dataset directory.")
    p.add_argument(
        "--vocab-size",
        type=int,
        default=32768,
        help="Default 32768 = the repo-wide 2^15 convention (matches SpeedrunConfig.vocab_size "
        "and the F1 runbook); ids stay well inside uint16.",
    )
    p.add_argument("--n-docs", type=int, default=64, help="FineWeb-EDU docs to download.")
    p.add_argument("--corpus", default=None, help="Local text file (blank-line-separated docs).")
    p.add_argument("--docs-per-shard", type=int, default=None)
    p.add_argument(
        "--decontaminate",
        action="store_true",
        help="A0 gate: strip train docs that 13-gram-overlap the eval sets before sharding.",
    )
    p.add_argument(
        "--eval-file",
        action="append",
        default=[],
        help="Extra eval-text file (one item per line) to guard against; repeatable.",
    )
    p.add_argument(
        "--fineweb-parquet",
        action="store_true",
        help="F1-scale bulk path: download FineWeb-EDU sample-10BT parquet from the HF hub and "
        "stream-tokenize to shards (bounded RAM; needs the pyarrow data extra).",
    )
    p.add_argument(
        "--target-tokens",
        type=float,
        default=7e8,
        help="Token budget for --fineweb-parquet; the build stops within one doc-batch of it.",
    )
    p.add_argument(
        "--parquet-cache",
        default=None,
        help="Directory for the downloaded parquet files (default: <out>/parquet).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the --fineweb-parquet download plan (files + sizes + est. tokens) and exit "
        "without downloading anything.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Tokenizer processes for the streaming path (0/1 = single-process).",
    )
    p.add_argument("--tokens-per-shard", type=int, default=DEFAULT_TOKENS_PER_SHARD)
    p.add_argument(
        "--max-train-bytes",
        type=int,
        default=DEFAULT_BPE_TRAIN_BYTES,
        help="Byte cap on the staged BPE-training sample (pure-Python trainer is the bound; "
        "see build_dataset_streaming).",
    )
    args = p.parse_args()

    if args.dry_run and not args.fineweb_parquet:
        p.error("--dry-run only applies to --fineweb-parquet")

    if args.fineweb_parquet:
        target = int(args.target_tokens)
        plan = plan_parquet_download(list_hub_parquet_files(), target)
        print(plan.describe())
        if args.dry_run:
            return
        cache = Path(args.parquet_cache) if args.parquet_cache else Path(args.out) / "parquet"
        paths = download_parquet_files(plan.files, cache, progress=print)
        metas, stats = build_dataset_streaming(
            lambda: iter_parquet_doc_batches(paths),
            args.out,
            args.vocab_size,
            max_train_bytes=args.max_train_bytes,
            tokens_per_shard=args.tokens_per_shard,
            target_tokens=target,
            num_workers=args.num_workers,
            doc_filter=_decontam_filter(args.eval_file) if args.decontaminate else None,
            progress=print,
        )
        print(
            f"{stats.n_shards} shard(s), {stats.n_tokens:,} tokens ({metas[0].dtype}), "
            f"{stats.n_docs:,} docs, {stats.bytes_per_token:.2f} bytes/token, "
            f"{stats.n_docs_filtered} filtered → {args.out}"
        )
        return

    if args.corpus is not None:
        text = Path(args.corpus).read_text(encoding="utf-8")
        docs = [d for d in text.split("\n\n") if d.strip()]
    else:
        docs = download_fineweb_slice(n_docs=args.n_docs)

    doc_filter = None
    if args.decontaminate:
        doc_filter = _decontam_filter(args.eval_file)
        kept = [d for d in docs if doc_filter(d)]
        print(
            f"A0 decontamination: kept {len(kept)}/{len(docs)} docs ({len(docs) - len(kept)} dropped)"
        )

    metas = build_dataset(
        docs, args.out, args.vocab_size, args.docs_per_shard, doc_filter=doc_filter
    )
    total = sum(m.n_tokens for m in metas)
    print(
        f"{len(metas)} shard(s), {total:,} tokens ({metas[0].dtype}), {len(docs)} docs → {args.out}"
    )


if __name__ == "__main__":
    main()
