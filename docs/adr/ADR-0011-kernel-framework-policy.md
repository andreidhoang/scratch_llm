# ADR-0011 — Kernel framework policy: Triton-primary, CUDA/CUTLASS second-tier

- **Status:** Accepted (2026-06-29)
- **Layer:** L2 Systems (A2 kernels) + Capstone DELTA — `src/scratch_llm/kernels/`
- **Decides:** which language/framework each kernel is written in across the from-scratch build, so the choice is never re-litigated per kernel

> **Note (2026-07-01 · perf-curriculum reset).** The concrete example files cited below
> (`kernels/gemv_triton.py`, `kernels/csrc/gemm/gemv.cu`, `kernels/csrc/norm/rmsnorm.cu`,
> `bindings.cpp`) were **removed** to rebuild the `performance/` A1–A7 track from scratch; they are
> preserved at git tag `pre-perf-kernel-reset`. This **decision is unchanged** — Triton-primary still
> governs the rebuild. The kept CS336 A2 `kernels/flash_attention_triton.py` remains the reference
> Triton kernel. When the rebuilt suite lands, refresh the file paths below.

## Context

The build writes ~9 compute kernels from scratch (the `GPU_FROM_ZERO.md` ladder R2→R7) plus the DELTA
GDN-2 decode kernel, all under the Mode-3 boundary (the human writes the bodies;
`kernel-write-guard.sh` enforces it). Every kernel needs a language/framework chosen. The candidates
and what production serving at frontier labs actually ships in `[FACT]`:

- **Triton** (Python-embedded, tile-based, autotuned): `torch.compile` emits it; many vLLM/SGLang fused
  kernels, Liger-Kernel, and **`fla` / flash-linear-attention (the DELTA reference)** are Triton.
- **CUDA C++ + CUTLASS/CuTe**: FlashAttention 2/3, FlashInfer, TensorRT-LLM, cuBLAS/cuDNN, DeepSeek
  DeepGEMM/DeepEP/FlashMLA — the hottest GEMM/attention hot paths that serve at scale.
- **CuTeDSL** (Python→CUDA via Cutlass): FlashAttention-4 on Blackwell — the emerging successor.
- **Pallas/JAX (Mosaic)**: Google/GDM kernels on the TPU substrate.
- ThunderKittens, Mojo, TileLang: emerging, not production-dominant.

The governing first principle is the **roofline**: a kernel's regime dictates the framework.
**Memory-bound** kernels (decode attention, norms, elementwise, KV gather, MoE routing) are capped by
HBM bandwidth — Triton reaches the bandwidth ceiling and CUDA buys ~nothing (no compute headroom to
exploit). **Compute-bound** kernels (prefill attention, large GEMMs) need WGMMA/TMA/warp-specialization
to reach tensor-core peak (≥94% with WGMMA vs ~63% without) — territory where Triton historically caps
lower (cf. this repo's honest 53%-of-SDPA result). Decisively for this build: **the serving P&L is
decode, decode is memory-bound, and the DELTA spike (GDN-2, `AI<1`) is memory-bound** — exactly the
regime where Triton is both competitive and sufficient. The repo already half-encodes this split
(`kernels/flash_attention_triton.py`, `kernels/gemv_triton.py` alongside `kernels/csrc/gemm/gemv.cu`,
`kernels/csrc/norm/rmsnorm.cu`), and the DELTA plan already says "Triton-first."

## Decision

1. **Triton is the primary framework** for the from-scratch ladder (R2 vector-add → R3 fused norm/softmax
   → R5 FlashAttention-2 → R6 decode-attention + paged-KV → R7 FP8 GEMM) and for the **DELTA GDN-2 decode
   kernel**. Rationale: highest learning velocity, matches the repo, `fla` is Triton, GPU MODE / kernel
   leaderboards accept Triton, and it reaches the bandwidth ceiling for the memory-bound kernels that
   dominate serving wall-clock.
2. **CUDA C++ → CUTLASS/CuTe is the deliberate second tier**, invested at two specific points: (a) the
   **tiled GEMM (R4) is done in *both* Triton and CUDA C++** to feel the compute-bound regime and why
   CUTLASS exists (WGMMA/TMA); (b) as the **literacy required to read FlashAttention/FlashInfer and to
   land the DELTA PR into FlashInfer/SGLang core** (their hot paths are CUDA/CUTLASS). `kernels/csrc/`
   is the reference for this tier.
3. **CuTeDSL and Pallas/JAX are know-it** (awareness, do not build): CuTeDSL is the FA4/Blackwell-era
   successor to whiteboard; Pallas/JAX matters only if targeting Google/GDM's TPU substrate.
4. **The roofline regime, not preference, picks the framework for any *new* kernel** not on the ladder:
   memory-bound → Triton; compute-bound and chasing tensor-core peak → CUDA/CUTLASS/CuTe.

## Consequences

- (+) No per-kernel framework debate: the ladder and DELTA are Triton; the two CUDA touchpoints (R4,
  the FlashInfer PR) are named in advance.
- (+) Aligns the build with where the serving P&L actually is (memory-bound decode) and keeps the
  highest-EV skill — predicting + hitting a roofline — language-agnostic and front-loaded.
- (+) The existing `kernels/csrc/` (CUDA gemv/rmsnorm + `bindings.cpp`) and Triton kernels are both on
  the sanctioned path — neither is orphaned.
- (−) Triton's compute-bound ceiling is real: a Triton prefill-attention/GEMM will trail hand-tuned
  CUTLASS at peak. We accept this — we *report* the gap honestly (the 53% model) rather than sink time
  into a CUTLASS rewrite of a kernel whose regime doesn't reward it.
- (−) Landing a kernel into FlashInfer/SGLang *core* may require CUDA/CUTLASS even if the artifact
  itself is Triton; budget the R4 CUDA reps as the on-ramp.
- **ADR trigger to revisit:** if the rented GPU is Hopper/Blackwell *and* a target kernel is
  compute-bound at peak (a prefill-attention or training-GEMM differentiator), promote that one kernel
  to CUTLASS/CuTe(DSL) and record it; the policy above stays the default.
- **Pairs with** ADR-0008 (Hopper-gated serve engine) and the A2 guide §7 FA2-vs-FA3/FA4 hardware notes.
- **Amendment (2026-08-02):** K10.2 of the K3 roadmap re-aims the DELTA capstone at the **KDA decode
  kernel** (per-channel decay) — the DELTA GDN-2 kernel is its port base, and its correctness contract
  inherits from `kda.py`'s three-path equivalence ([`../k3/ROADMAP.md`](../k3/ROADMAP.md) §K10.2).
