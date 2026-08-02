#!/usr/bin/env python3
"""K3 track — R0 "Anatomy of Kimi K3" tensor fetcher (docs/k3/ABLATIONS.md §R0).

Fetches SMALL tensors from the released 2.8T checkpoint via HTTP range reads — no shard
downloads, no GPU. safetensors layout: 8-byte LE header length N, then N bytes of JSON
(name -> {dtype, shape, data_offsets}), then raw tensor bytes at 8+N+offset. The index JSON
maps every tensor name to its shard, so we fetch only (a) each needed shard's header and
(b) the exact byte ranges of the target tensors.

Anatomy sets (pre-registered in ABLATIONS.md):
  qb        — block_sparse_moe.gate.e_score_correction_bias  (92 × 896: QB's learned equilibrium)
  attnres   — {self_attention,mlp,output_attn}_res_proj.weight (187 × 7168: depth-reuse map)
  decay     — self_attn.A_log (69 × 96) + self_attn.dt_bias (69 × 12288): learned timescales

Also writes census.json (every tensor: shard/dtype/shape/offsets) — which completes the
FACTS A18 tensor-level closure (census param count vs k3/param_count.py).

Stdlib + numpy only. Polite: sequential requests, retries, no parallelism. Public repo, no auth.

Usage:
  python scripts/k3_fetch_tensors.py --smoke            # 2 shards, verify pipeline
  python scripts/k3_fetch_tensors.py                    # full anatomy census + fetch
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = "moonshotai/Kimi-K3"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
INDEX_NAME = "model.safetensors.index.json"

SETS: dict[str, str] = {
    "qb": r"block_sparse_moe\.gate\.e_score_correction_bias$",
    "attnres": r"(self_attention_res_proj|mlp_res_proj|output_attn_res_proj)\.weight$",
    "decay": r"self_attn\.(A_log|dt_bias)$",
}

DTYPE_NP = {"F32": np.float32, "F16": np.float16, "U8": np.uint8, "I64": np.int64}
DTYPE_SIZE = {"F32": 4, "BF16": 2, "F16": 2, "U8": 1, "I64": 8}


def http_get_range(url: str, start: int, end: int, retries: int = 4) -> bytes:
    """GET bytes [start, end] inclusive via curl (follows redirects, Range preserved)."""
    for attempt in range(retries):
        p = subprocess.run(
            ["curl", "-sSL", "--fail", "--max-time", "300", "-r", f"{start}-{end}", url],
            capture_output=True,
        )
        if p.returncode == 0:
            return p.stdout
        wait = 2**attempt
        print(f"    retry {attempt + 1}/{retries} after curl exit {p.returncode} ({wait}s)", file=sys.stderr)
        time.sleep(wait)
    raise RuntimeError(f"range GET failed {retries}x: {url} [{start}-{end}]")


def load_index(cache: Path) -> dict[str, str]:
    if not cache.exists():
        print(f"fetching index -> {cache}")
        data = http_get_range(f"{BASE}/{INDEX_NAME}", 0, 2**62 - 1)
        cache.write_bytes(data)
    return json.loads(cache.read_text())["weight_map"]


def shard_header(shard: str) -> dict[str, dict]:
    url = f"{BASE}/{shard}"
    n = struct.unpack("<Q", http_get_range(url, 0, 7))[0]
    header = json.loads(http_get_range(url, 8, 8 + n - 1))
    return {"__len__": n, "tensors": header}


def decode(raw: bytes, dtype: str, shape: list[int]) -> np.ndarray:
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
    return np.frombuffer(raw, dtype=DTYPE_NP[dtype]).reshape(shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/k3_anatomy")
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    ap.add_argument("--smoke", action="store_true", help="only the first 2 needed shards")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "tensors").mkdir(parents=True, exist_ok=True)
    weight_map = load_index(out / "index.json")

    wanted = {
        name: shard
        for name, shard in weight_map.items()
        if any(re.search(SETS[s], name) for s in args.sets)
    }
    shards = sorted(set(wanted.values()))
    if args.smoke:
        shards = shards[:2]
        wanted = {n: s for n, s in wanted.items() if s in shards}
    print(f"targets: {len(wanted)} tensors across {len(shards)} shards")

    census: dict[str, dict] = {}
    fetched: dict[str, np.ndarray] = {}
    for i, shard in enumerate(shards, 1):
        h = shard_header(shard)
        base = 8 + h["__len__"]
        entries = h["tensors"]
        for name, meta in entries.items():
            if name == "__metadata__":
                continue
            census[name] = {"shard": shard, "dtype": meta["dtype"], "shape": meta["shape"]}
        todo = {n: e for n, e in entries.items() if n in wanted}
        for name, meta in todo.items():
            b, e = meta["data_offsets"]
            raw = http_get_range(f"{BASE}/{shard}", base + b, base + e - 1)
            arr = decode(raw, meta["dtype"], meta["shape"])
            fetched[name] = arr
            np.save(out / "tensors" / (name.replace("/", "__") + ".npy"), arr)
        print(f"[{i}/{len(shards)}] {shard}: header {h['__len__'] / 1e6:.1f} MB, "
              f"{len(entries)} tensors, fetched {len(todo)}")

    (out / "census.json").write_text(json.dumps(census))

    # A18 closure: exact param count from census shapes vs HF metadata + our accounting.
    total = sum(int(np.prod(m["shape"])) for m in census.values())
    by_dtype: dict[str, int] = {}
    for m in census.values():
        n = int(np.prod(m["shape"]))
        by_dtype[m["dtype"]] = by_dtype.get(m["dtype"], 0) + n
    print(f"\ncensus tensors: {len(census)}  params: {total:,}")
    print(f"by dtype: {by_dtype}")
    print("HF metadata total: 2,779,931,837,184  residual:", f"{total - 2_779_931_837_184:,}")

    stats: dict[str, dict] = {}
    for name, arr in fetched.items():
        stats[name] = {
            "shape": list(arr.shape),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    (out / "anatomy_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"wrote {out}/census.json, {len(fetched)} tensors, anatomy_stats.json")


if __name__ == "__main__":
    main()
