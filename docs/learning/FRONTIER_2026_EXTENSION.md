# FRONTIER 2026 EXTENSION — Beyond Text LLMs
## The 4 Missing Frontiers (added 2026-07-26)

> **What this is.** The 89-Bài curriculum (M1-M10, S1-S6) builds the text-LLM foundation correctly.
> But the 2026 frontier is broader: multimodal reasoning, diffusion, agentic systems, and physical AI.
> This file maps the 4 missing frontiers, their interview gates, and where they slot in.
>
> **Key principle:** These are NOT replacements for the existing curriculum. They are EXTENSIONS that
> connect the foundation to the full 2026 frontier. Most are "know-it-discuss" (⚪) — you own the
> tradeoff framing, not the from-scratch build. A few are "build" (🔨) because they're direct
> differentiators.

---

## The 2026 Frontier Bar (verified from primary sources)

[FACT] From dataford.io, CallSphere, NVIDIA JDs, interviewing.io, arXiv (Jul 2026):

| What was "advanced" in 2024 | 2026 status | What's now the differentiator |
|---|---|---|
| Transformer architecture | **Table stakes** | Multimodal architecture depth (VLM, DiT, video) |
| RLHF concept | **Table stakes** | Reasoning model training (GRPO, RLVR, test-time compute) |
| KV cache / PagedAttention | **Table stakes** | Agentic systems design (multi-turn, sandbox, eval) |
| MoE | **Table stakes** at senior | Physical AI / world models (NVIDIA focus) |
| RAG pipeline | **Table stakes** | Evaluation engineering (trajectory matching, LLM-as-judge) |

---

## Frontier A: Multimodal Architecture (VLM + Diffusion + Video + Audio)

### What the frontier looks like (Jul 2026, verified from primary sources)

**The convergent 2026 architecture (every frontier model):**
| Component | Universal choice | Evidence |
|---|---|---|
| Backbone | **Sparse MoE transformer** | Gemini 3, DeepSeek V4, Qwen3.5, Llama 4, GPT-5 — all MoE |
| Multimodality | **Native early-fusion** (not bolted-on encoder) | Gemini 3, Llama 4, Qwen3.5, Cosmos 3, GPT-5 |
| Context | **1M tokens** (Llama 4 Scout: 10M) | Universal floor |
| Reasoning | **RL-trained adaptive thinking** w/ effort controls | OpenAI, Anthropic, Gemini Deep Think |
| Post-training | **RLVR + multi-turn agentic RL** | Universal |
| Attention | **Hybrid**: sparse + linear + dense interleaved | DeepSeek V4 (CSA+HCA), Qwen3.5 (Gated DeltaNet+softmax) |
| Optimizer | **Muon** (Newton-Schulz) for weights, AdamW for emb | DeepSeek V4, Moonlight, Kimi-K2 — you have Muon in M3 3.4! |
| Precision | **FP4 QAT** (quantization-aware training) | DeepSeek V4 experts + indexer |

**NVIDIA Cosmos 3 (May 2026, GTC Taipei) — the "any-to-any" omni-model:**
- **Mixture-of-Transformers (MoT):** A reasoning transformer + a generation transformer in one architecture.
- Understands object interactions, motion, spatiotemporal relationships → then generates video, text, sound, AND action trajectories in one forward pass.
- Replaces the previous fragmented Cosmos family (Predict/Transfer/Reason/Policy) with one unified model.

**GR00T 1.7 (Jul 2026) — VLA model for humanoid robots:**
- Backbone: Cosmos-Reason2-2B (Qwen3-VL based) VLM
- Action head: **Diffusion Transformer (DiT)** cross-attending to VLM embeddings
- Loss: flow matching + **FLARE** (Future LAtent Representation Alignment) — aligns with future embeddings, not pixel-space generation
- Result: 93.3% language-following rate on real GR-1 robot (up from 46.6% in N1)

**DeepSeek V4 (Apr 2026) — the efficiency frontier:**
- Hybrid attention: CSA (4× compressed, FP4-indexed top-k block selection) + HCA (128× compressed global view) + 128-token sliding window
- mHC (Manifold-Constrained Hyper-Connections): replaces residual stream with doubly-stochastic 4-channel pathway — bounds spectral norm at 1 across 61 layers
- FP4 QAT for experts, 1M context at ~10% of V3.2's KV cache cost

**Diffusion models:**
- **DiT (Diffusion Transformer)** replaced U-Net — every frontier image/video model since mid-2024.
- **Rectified flow** is the default training objective (SD3, FLUX, Veo, Sora).
- **Latent diffusion:** compress to latent space (VAE), diffuse there, decode. All production models.
- Kernel optimization: SageAttention (FP8/FP4 attention), NVFP4/MXFP4 quantization for DiT.

**Video:**
- **Sora 2** (Sep 2025): DiT on spacetime patches, native audio, **models failure** (physics). Discontinued Apr 2026 but set the standard.
- **Veo 3** (May 2025): Unified audio+video latent diffusion, 4K/24fps, 48kHz audio.

**Audio:**
- **Moshi** (Kyutai): Full-duplex speech-text model, ~200ms latency, fully open.
- **Qwen2.5-Omni:** Thinker-Talker architecture, TMRoPE for audio-video sync.
- **Gemini 3:** Native audio (no ASR→LLM→TTS cascade).

### Interview gates this frontier earns

| Gate | Where in curriculum | Tier |
|---|---|---|
| "Design a VLM — how do you handle different modalities?" | Extension after M2 (you own the Transformer; VLM adds a vision encoder + connector) | ★ differentiator (2026) |
| "How does image generation work? Explain diffusion." | Extension — new concept, not in 89-Bài | ★ differentiator (2026) |
| "Your diffusion model takes too long. Speed it up." | Connects to S5 (quantization) + S3 (roofline) | ★ kernel lane |
| "Design a video understanding pipeline" | Extension — temporal attention, tokenization | know-it-discuss |

### How it connects to what you ALREADY own

```
M2 (Transformer)     ──→ VLM = ViT encoder + connector + YOUR Transformer
                          Cosmos 3 MoT = TWO Transformers (reason + generate) = your M2 × 2
M3 3.4 (Muon)        ──→ DeepSeek V4 uses Muon for weights! Your Muon derivation = frontier practice
S3/S4 (kernels)      ──→ DiT uses THE SAME attention/matmul ops you built
S5 (quantization)    ──→ NVFP4/MXFP4 for DiT + FP4 QAT for DeepSeek V4 = your quant knowledge
M8 (GRPO/RLVR)       ──→ Vision-R1 = GRPO for multimodal reasoning (SAME algorithm!)
M9 (MoE)             ──→ Every 2026 frontier model = sparse MoE (you build this)
M9 (MLA/hybrid attn) ──→ DeepSeek V4 CSA/HCA = next-gen of the MLA you study
S1/S2 (serving)      ──→ Video serving = long-context KV + temporal batching
                          Test-time compute = your serving layer + reasoning budget
reasoningLLM         ──→ Multi-turn agentic RL = your dial-env + rollout infra
```

**The insight is even deeper than I first thought:** Your curriculum doesn't just "connect" to the 2026 frontier — it builds the **exact components** the frontier uses:
- **Muon optimizer** (your M3 3.4) is what DeepSeek V4 uses in production
- **MoE** (your M9) is the backbone of every 2026 frontier model
- **GRPO/RLVR** (your M8) is how Vision-R1 trains visual reasoning
- **Hybrid attention** (your M9 MLA + S6 Ring Attention) is the DeepSeek V4 / Qwen3.5 pattern
- **FP4 quantization** (your S5) is the DeepSeek V4 training precision
- **Triton kernels** (your S3) are the exact ops DiT and Cosmos 3 need

### What to add (when you reach the right stage)

| Add as | What | Type | When |
|---|---|---|---|
| **M11** (new série) | VLM: ViT patch embedding, LLaVA connector, cross-attention vs early fusion, Janus decoupling | 🔨 Build ViT + connector (connects to your M2 Transformer) | After M9 (frontier arch) |
| **S7** (new série) | Diffusion: DDPM math → DDIM → latent diffusion → DiT → rectified flow → consistency/distillation | ⚪ Know-it-discuss (build toy DDPM if time) | After S5 (quant — connects NVFP4 for DiT) |
| **M11.5** | Video: spatiotemporal tokenization, temporal attention, frame sampling, long-video context | ⚪ Know-it-discuss | After M11 (VLM) |
| **M11.6** | Audio: streaming speech (Moshi pattern), Whisper architecture, audio tokenization | ⚪ Know-it-discuss | Optional |

---

## Frontier B: Reasoning AI & Test-Time Compute Scaling

### What the frontier looks like

**Test-time compute = scaling inference, not training:**
- **Parallel sampling:** Generate N candidates, pick best (Best-of-N, majority vote).
- **Sequential reasoning:** Chain-of-thought, extended thinking (Claude), "thinking tokens" (o3/o4).
- **Search at inference:** MCTS, beam search, tree-of-thought over the reasoning space.
- **Speculative decoding for reasoning:** Draft model proposes reasoning step, verifier checks.

**The RL that produces reasoning:**
- **RLVR (RL with Verifiable Rewards):** Skip reward model → verifier (code test, math check) directly. DeepSeek-R1 recipe.
- **Cold-start SFT → RL:** Seed with high-quality reasoning traces, then RL to exceed them.
- **The "aha moment":** RL training emergently produces self-correction behavior.
- **Vision-R1:** GRPO + hard formatting reward → trains visual reasoning. Same algorithm you derived in M8.

**The inference economics:**
- Reasoning models are 5-10× slower → must reason about latency/cost.
- Token budget allocation: how many "thinking tokens" before committing?
- Parallel vs sequential: when does Best-of-N beat CoT?

### Interview gates

| Gate | Where in curriculum | Tier |
|---|---|---|
| "Design a test-time compute system" | Extension after M8 (you own GRPO; TTC extends it to inference) | ★ differentiator |
| "How would you train a reasoning model with RL?" | M8 8.3-8.4 already covers GRPO + RLVR | Already built |
| "RLVR vs RLHF — what's the difference?" | M8 covers this | Already built |
| "Explain the 'aha moment' in RLVR" | M8 F7 prediction | Already built |
| "How do you handle reward hacking in reasoning traces?" | reasoningLLM flagship = direct hit | Already built |

### The insight

**Your curriculum already covers reasoning RL.** M8 (GRPO, RLVR, DPO) + reasoningLLM (verifier exploitability) = the exact skills this frontier demands. The ONLY gap is **test-time compute scaling** — the inference-side of reasoning (parallel sampling, search, token budget). That's an inference/serving extension, not a new RL concept.

### What to add

| Add as | What | Type | When |
|---|---|---|---|
| **S2.10** (extend serving) | Test-time compute: Best-of-N, parallel sampling, search-at-inference, token budget allocation | ⚪ Know-it-discuss (connects to S2 speculative decoding) | After S2 (serving engines) |

---

## Frontier C: Agentic Systems

### What the frontier looks like

**This is Anthropic's CORE business.** Claude Code, Cowork, Managed Agents = the flagship product.

- **Multi-turn tool use:** Model calls tools, gets results, decides next action. ReAct / Plan-and-Execute / Multi-Agent patterns.
- **Sandboxed execution:** gVisor/Firecracker containers, network allowlist, filesystem isolation.
- **Agent memory:** Working (in-context), short-term (conversation), long-term (vector DB), episodic (task-indexed).
- **Safety stack:** Action classification by risk, human-in-the-loop for irreversible actions, budget limits (token/action/time/cost).
- **Evaluation:** Trajectory matching (score tool-call sequences), task completion, tool selection accuracy, cost-per-task.

**MCP (Model Context Protocol)** is now the standard tool integration protocol — Anthropic's contribution.

### Interview gates

| Gate | Where in curriculum | Tier |
|---|---|---|
| "Design an agentic coding system safely" | a-systems round (already in mock rotation) | ★ differentiator |
| "Your agent is stuck in a loop. Detect and break." | Extension — failure modes | know-it-discuss |
| "How do you evaluate an agent?" | Extension — eval engineering | know-it-discuss |
| "Design the safety stack for a long-running agent" | a-values + a-systems | know-it-discuss |

### The insight

**Your reasoningLLM IS an agentic system.** The dial-env, rollout seam, verifiers, async Trio — that's the infrastructure for agentic RL. The gap is the SYSTEMS DESIGN framing (sandboxing, safety stack, evaluation) that the a-systems round tests. This is a mock-round extension, not a curriculum extension.

### What to add

| Add as | What | Type | When |
|---|---|---|---|
| **a-agentic** (new mock round) | Agent system design: multi-turn orchestration, sandboxing, eval, safety | Mock round (not curriculum node) | Add to mock rotation immediately |
| **Reading** | Anthropic engineering blog: "Demystifying evals for AI agents", "AI-resistant technical evaluations", "Scaling Managed Agents" | Gap-driven reading | Before a-systems / a-agentic mocks |

---

## Frontier D: Physical AI / Embodied AI (NVIDIA Lane)

### What the frontier looks like

**GR00T (NVIDIA's open VLA model):**
- Vision-Language-Action model: image + language + robot state → VLM backbone → diffusion transformer → humanoid actions.
- Dual-system: System 1 (fast reactive) + System 2 (slow planning).
- Data: 32K hours real demos + egocentric video + 8K hours simulated rollouts.
- Deployment: ONNX/TensorRT on Jetson Thor.

**World models:**
- **V-JEPA 2** (Meta): 1.2B world model, self-supervised on 1M+ hours video. Zero-shot pick-and-place at 65-80%.
- **NVIDIA Cosmos:** 3 families — Predict (generate world states), Transfer (structured inputs → photoreal video), Reason (chain-of-thought spatiotemporal reasoning).
- **Genie 3** (DeepMind): Text → navigable 3D world at 24fps/720p.

**Sim-to-real pipeline:**
- Omniverse → Isaac Sim → physical robot.
- GR00T-Dreams: Cosmos world models → synthetic trajectories → action token extraction.

### Interview gates (NVIDIA-specific)

| Gate | Where in curriculum | Tier |
|---|---|---|
| "What does a DevTech engineer do for GR00T?" | NVIDIA DevTech JD — kernel optimization for robotics workloads | ★ differentiator for NVIDIA |
| "Explain sim-to-real transfer" | Extension — world models | know-it-discuss |
| "How would you optimize a VLA model for real-time inference?" | Connects to S1/S2 (serving) + S5 (quant for Jetson) | know-it-discuss |

### The insight

**Physical AI is where your kernel skills meet embodied AI.** The VLA model is a Transformer (you built it) + diffusion (Frontier A) + serving on edge hardware (S1/S5 extended). The DevTech work is: profile the VLA inference → find the bottleneck (attention? diffusion step? action head?) → optimize the kernel. That's YOUR k_live + k-forensics skill applied to a new workload.

Your FaithBench (FROZEN) sits exactly on this surface — reasoning AVs = physical AI.

### What to add

| Add as | What | Type | When |
|---|---|---|---|
| **k-physical** (new mock round) | VLA inference optimization, sim-to-real pipeline, Jetson Thor deployment | Mock round (NVIDIA-specific) | Optional — add when targeting NVIDIA DevTech |
| **Reading** | GR00T 1.7 model card, Cosmos paper, V-JEPA 2 paper | Gap-driven reading | Before k-physical mock |

---

## Summary: The Extended Curriculum Map

```
EXISTING (89 Bài — text-LLM foundation):
M1→M2→M3→M4  →  [PPPM→S3→S4]  →  M5+S6  →  M6+M7  →  M8  →  M9  →  S1+S2+S5  →  M10
 ✅ Foundation   ✅ Kernels       ✅ Distributed ✅ Science ✅ RL ✅ Arch ✅ Serving  ✅ Capstone

EXTENSIONS (4 frontiers — connect to foundation):
                                                    ↓
                                              M11 (VLM + ViT + Diffusion + Video)
                                                    ↓
                                              S2.10 (Test-time compute scaling)
                                                    ↓
                                              [a-agentic] (Agent systems design — mock round)
                                                    ↓
                                              [k-physical] (VLA/GR00T — NVIDIA mock round)
```

### The governing rule (unchanged)

**Curriculum order is sacred.** You're at M3 3.3. The extensions are Stage 9+ — you reach them
AFTER finishing the 89-Bài path. The mock rounds test them diagnostically NOW (you'll fail
multimodal/diffusion questions — that's expected, it's a diagnostic). The reading fills the gap
the mock names. You don't jump forward.

### What changes in the daily system

1. **Mock rotation expands:** Add `a-agentic` (agentic systems) and `a-multimodal` (VLM/diffusion)
   to the Lane A rotation. Add `k-physical` (optional, NVIDIA-specific) to Lane K.
2. **Reading queue expands:** Add frontier papers (Veo 3 tech report, Vision-R1, V-JEPA 2, GR00T)
   as gap-driven reading — only when a mock names the gap.
3. **FRONTIER_HIRING_MAP adds new scarce buckets:** Multimodal architecture, reasoning/test-time
   compute, agentic systems, physical AI.
4. **The 89-Bài path is unchanged.** It's still the foundation. The extensions are additive,
   not substitutive.

---

*Authored from 3 research agents (multimodal landscape, interview probes, diffusion/world models)
+ codebase audit + existing curriculum. The text-LLM foundation is correct; the 2026 frontier
extends it into multimodal, reasoning, agentic, and physical AI.*
