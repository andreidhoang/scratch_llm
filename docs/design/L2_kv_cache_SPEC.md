# Design spec — L2 KV-cache (incremental decoding)

> **Status:** spec / not yet built. The next implementation after L1 substrate + `utils/monitors.py`.
> **Layer:** L2 Systems (A2) — the CPU-buildable half (FA2/SGLang/DDP need a GPU box).
> **Files touched:** `src/scratch_llm/model.py` (cache-aware attention + forward), `src/scratch_llm/sampling.py`
> (cache-driven `generate`), new `tests/test_kv_cache.py`.

## 1. Why (the problem this solves)

Naive autoregressive decoding recomputes K, V for the *entire* prefix at every step: generating
T tokens from a prompt of length P costs Σᵢ O((P+i)²·d) — quadratic per step. A KV-cache stores
each past token's K, V projections so a new step computes K, V for **only the new token** and
attends against the cache: per-step cost drops to O((P+i)·d). This is the inference-economics core
(KV-cache is 60–85% of wall-clock past long context) and the prerequisite for cheap rollouts —
L5's GRPO samples G completions × a batch per prompt; without a cache that is unrunnably slow.

We built RoPE to slice by **actual `positions`** (not `range(S)`) precisely so a cache feeding one
token at absolute position `t` rotates it correctly. This spec cashes in that decision.

## 2. The mechanics

Per layer, cache K and V of shape `(B, n_kv_heads, T_cached, head_dim)` (GQA-native: store the
`n_kv_heads`, repeat to `n_heads` at attention time — the cache is `n_heads/n_kv_heads`× smaller).

Two phases:
- **Prefill** — run the whole prompt (S>1) once, populate the cache, return last-position logits.
- **Decode** — feed one new token (S=1); compute its Q, K, V; append K, V to the cache; attend Q
  over `[cached_K ; new_K]`; return next-token logits.

**Positions:** with a cache of length `L`, the new tokens are at absolute positions `arange(L, L+S)`.
RoPE rotates Q, K at those absolute positions — already supported.

**Masking:**
- Decode (S=1): the single query is at position `L` and attends keys `0..L` — all in the past, so
  **no mask** is needed.
- Prefill (S>1) from cache length `L`: query row `i` (absolute `L+i`) may attend key `j` iff
  `j ≤ L+i`. From an empty cache (`L=0`) this is the standard causal triangle.

## 3. API (minimal, training path untouched)

```python
class KVCache:
    """Per-layer (K, V) buffers; length = positions cached so far (same across layers)."""
    def __init__(self, n_layers: int) -> None: ...
    @property
    def length(self) -> int: ...
    def get(self, layer: int) -> tuple[Tensor, Tensor] | None: ...
    def append(self, layer: int, k_new: Tensor, v_new: Tensor) -> None: ...
    def advance(self, n: int) -> None: ...   # bump length once per forward, after all layers
```

`MultiHeadSelfAttention.forward(x, positions, cache=None, layer_idx=None)` — when `cache` is given,
append this layer's new K, V and attend over the concatenation; mask logic per §2.

`TransformerLM.forward(token_ids, cache=None)` — `cache is None` ⇒ today's full forward (training,
unchanged). With a cache: `positions = arange(cache.length, cache.length + S)`, thread `cache`
through each block, `cache.advance(S)` after the last block.

`sampling.generate(...)` — internally: build a `KVCache`, **prefill** the prompt, then loop **decode**
one token at a time. Output must be byte-identical to today's recompute path.

## 4. Correctness invariant (the make-or-break test)

> **Cached incremental decode must equal full recompute, token-for-token and logit-for-logit.**

`tests/test_kv_cache.py`:
1. **Logit parity (prefill):** `model(prompt, cache)` last-position logits ≈ `model(prompt)[:, -1]`.
2. **Logit parity (decode):** after prefill, feeding token `t` via the cache yields the same logits
   as a full recompute of `[prompt; …; t]` — `torch.testing.assert_close`, several steps.
3. **End-to-end:** greedy `generate(..., use_cache=True)` == greedy `generate(..., use_cache=False)`
   (identical id list), for both MHA (`n_kv_heads==n_heads`) and GQA (`n_kv_heads<n_heads`).
4. **Position correctness:** a prompt whose continuation depends on absolute position decodes the
   same cached vs. recompute (guards a RoPE-position-in-cache bug).
5. **Batch > 1** parity.

If (3) ever diverges, the cache is wrong — no other signal needed.

## 5. Decisions & trade-offs

- **Dynamic append (`torch.cat`) for the reference impl**, not a pre-allocated ring buffer. Simpler,
  CPU-correct, and the correctness test is identical. *Trade-off:* repeated realloc is slower and
  not paged — fine on CPU; the pre-allocated `(B, H, max_len, D)` buffer + write-index (the path to
  paged attention) is the **GPU-phase optimization**, deferred. → candidate ADR.
- **GQA-native cache** (store `n_kv_heads`): smaller cache, matches ADR-0002; repeat happens at attn.
- **`use_cache` flag on `generate`**, default `True`, with the no-cache path kept as the test oracle.
- **Sampler parity preserved** (ADR-0003): the cache changes *how* logits are computed, never the
  decode config — same `SamplingParams`, so `kl_train_infer` semantics are untouched.

## 6. Scope / deferred

- **In:** KV-cache + cache-aware attention/forward + cached `generate`, CPU-tested.
- **Next (after this):** the `rollout/` client **seam** — an interface + a local backend wrapping
  `generate`, so L5 can develop rollouts against a contract; the SGLang backend slots in on GPU.
- **Deferred to a GPU box:** Triton FA2, pre-allocated/paged KV, DDP/ZeRO, real SGLang serving.

## 7. Build sequence

1. `KVCache` class.
2. Make `MultiHeadSelfAttention.forward` cache-aware (append + concat + mask per §2).
3. Make `TransformerLM.forward` accept `cache` (positions from `cache.length`, `advance` after).
4. Write `tests/test_kv_cache.py` (the §4 parity tests) — these should fail first, then pass.
5. Wire `generate(use_cache=True)` (prefill + decode loop); keep `use_cache=False` as the oracle.
6. Green-CI; update `docs/STATUS.md`.
