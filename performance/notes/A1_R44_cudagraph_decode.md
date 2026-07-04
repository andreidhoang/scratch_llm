# A1 Rung 4.4 — CUDA-graph decode · engineering spec + result

> **Spec-with-falsifiers (FOP-2).** Parent: `performance/A1_transformer_inference.md` §4.4. Oracle =
> R4.1 paged-kernel eager decode (token-exact). Gate = outputs identical + step-time reduction with
> nsys CPU-gap evidence. **Pre-registered** (from the R1 close-out forward-pointer, `bench/RESULTS.md`):
> B=1 step-time reduction **~20–28%** (vLLM-V1's number), bound = launch overhead. This rung closes
> the R1 gap: eager 51 → compiled 173 tok/s (53% of the 327 tok/s wall) — the residual is per-token
> launch dispatch that only a static graph removes; R1's `torch.cat` cache broke capture, so the
> fixed-address paged pool was built precisely to enable this.

## 1. Mechanism — one submission instead of hundreds of launches

Decode is launch-bound on this consumer GPU: even the fused paged path issues ~54–68 `cudaLaunchKernel`
per token (measured, nsys), each paying CPU dispatch + a host↔device gap, so a 0.84B model that should
decode in ~1.5 ms (memory wall) takes **15.4 ms/step eager**. A CUDA graph records the whole step once
and replays it as a single `cudaGraphLaunch` — the CPU stops feeding the GPU kernel-by-kernel.

Why the paged Triton kernel is the only capturable substrate here:
- **Fixed addresses:** R1's `cat`-cache grows to new pointers each step (illegal to capture); the R4.1
  block pool is written in place.
- **Fixed shapes:** the dense decode reads a `[:, :, :view_len]` slice whose shape grows each step; the
  paged kernel launches a fixed `(B, H)` grid and reads the per-row key count from the device `lengths`
  tensor at runtime — so ONE capture serves every decode length.
- `torch.compile(mode="reduce-overhead")` REFUSES this path (it flags the in-place `lengths += active`
  in `advance` as a "mutated input" and silently skips cudagraphs — measured). **Manual**
  `torch.cuda.CUDAGraph` capture works because we own the mutation: `pre_decode_reserve` (block alloc)
  and `mirror_advance` run on the host around the replay, mutating the fixed-address block table in
  place; the captured `advance` auto-increments `lengths` each replay.

Build: `serving/cudagraph.py::CudaGraphDecoder` — `capture()` (side-stream warmup to prime Triton
autotune + the allocator, snapshot/restore lengths so the prefilled state survives capture) + `step()`
/`decode()` (copy the token into a fixed staging tensor, `replay()`, argmax).

## 2. Measured (2026-07-04, `bench/cudagraph_decode.py`, sm120 0.84B bf16, paged kernel BOTH arms)

| B | token-exact | eager ms/step | graph ms/step | reduction | agg tok/s (eager → graph) |
|---|---|---|---|---|---|
| 1 | ✅ | 15.38 | **3.96** | **−74.3%** | 65 → **253** |
| 8 | ✅ | 15.95 | 4.57 | −71.3% | 502 → 1749 |
| 32 | ✅ | 16.06 | 5.07 | −68.4% | 1993 → 6307 |

**B=1 graph decode = 253 tok/s = 77% of the 327 tok/s memory wall** (0.55 TB/s ÷ 1.68 GB weights) —
the highest fraction yet, past compiled's 53%. nsys `[FACT]`: eager ≈54–68 `cudaLaunchKernel`/step →
graph **1 `cudaGraphLaunch`/step**.

**Verdict — SHIPPED; prediction FALSIFIED in the good direction (−68–74% ≫ −20–28%).** The R1 launch-
overhead thesis is confirmed and the wall gap is closed: the ONLY thing that changed between arms is the
launch mechanism (same paged kernel, same math, token-exact), and step time fell ~4× at B=1 where
compute is negligible — so the eager time was almost entirely CPU launch dispatch. The pre-registered
20–28% was vLLM-V1's number on an already-optimized H100 path; **our eager baseline is far more
overhead-bound** (consumer GPU, ~54–68 host launches/step at ~250 µs each), so the graph win is much
larger. Honest caveat: clocks unlocked (`nvidia-smi -lgc` blocked) — the ~4× ratio dwarfs the ±15%
clock drift, and the token-exactness + launch-count collapse make the mechanism unambiguous.

## 3. DoD

- [x] Outputs identical to eager R4.1 paged decode (token-exact, gpu test, B∈{1,8,32}).
- [x] Step-time reduction measured (−68–74%) with nsys launch-count evidence (54–68 → 1 per step).
- [x] Closes the R1 wall gap: B=1 reaches 77% of the memory ceiling (vs eager 20%, compiled 53%).
- Scope note: single fixed-batch capture (the demonstrator). Continuous-batching + cudagraph (ragged
  admission churn re-captures / a graph pool per batch size) is the production extension — the fixed
  `(B,H)` grid makes it mechanical, deferred with the R4.2b piggyback work.
