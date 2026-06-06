# ADR-0008 — SGLang serve engine is Hopper+ only; HF engine-pair stand-in on Ada

- **Status:** Accepted (2026-06-06)
- **Layer:** L2 Systems (A2.3) `rollout/` — the `kl_train_infer` bridge
- **Decides:** how A2.3 measures real train↔infer drift given the rented hardware

## Context

The A2.3 deliverable measures the **real** `kl_train_infer` between a training engine and a serving
engine on the same model (the repo's discipline pillar). The named serve engine is SGLang
(`rollout/sglang_client.py`). On the rented **RTX 4090 (Ada, sm89)**, installing `sglang[all]`
pulled `sglang==0.5.12.post1` + `torch==2.11.0+cu130`, and `sgl_kernel` ships compiled kernels for
**sm90 (Hopper) and sm100 (Blackwell) only — no sm89**. `import sgl_kernel` fails to load
`common_ops` on the 4090. Current SGLang has dropped pre-Hopper kernel support; the cu130 build also
mismatches the container's CUDA 12.4 (NVRTC/`libnuma` runtime gaps). This is a hardware/build
incompatibility, not a fixable config — running real SGLang needs a Hopper+ box or a carefully
pinned older sglang+torch+cu124 stack.

## Decision

1. **Retain `SGLangBackend`** (`rollout/sglang_client.py`) as the serve-engine implementation behind
   the `RolloutClient` Protocol — it is correct code, runnable on a Hopper+ box unchanged.
2. **Measure `kl_train_infer` on Ada via an HF engine pair** (`HFReferenceBackend`) that differs on
   the two axes that actually cause train↔infer drift: **attention kernel** and **precision** —
   *serve = sdpa + bf16*, *train = eager + {bf16, fp32}*. Same model, same tokenizer, same sampled
   tokens, so the only variables are kernel and precision: exactly what the metric must isolate.
3. **Defer real SGLang serving** to a Hopper block (or a pinned older sglang) — a scheduled GPU
   step, not abandoned (CLAUDE.md "Follow the plan — never skip").

## Consequences

- (+) The A2.3 `kl_train_infer` measurement + HALT@0.10 mechanism is delivered reliably on Ada,
  without sinking the session into sgl_kernel/cu130 dependency hell.
- (+) "Current SGLang is Hopper-only" is itself a recorded systems finding (cf. A2 guide §1.4:
  FA3 Hopper-only → a falsifiable hardware fact), and `SGLangBackend` is ready for a Hopper box.
- (−) On Ada we do not exercise SGLang's actual fused kernels / quantized KV, so the measured drift
  is HF-sdpa-vs-eager, a *floor* on real serve-engine drift (SGLang's drift would be ≥ this).
- **Pairs with** the A2 §8 #2 ADR trigger (SGLang vs vLLM): vLLM also ships limited pre-Hopper
  wheels — confirm arch support before the next serve-engine rental.
