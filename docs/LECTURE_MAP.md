# CS336 Lecture → Assignment → Source Map

> **What this is.** The 17 official CS336 lectures (`../../lectures/lecture_*`) mapped onto the five
> assignments and the modules you build in `src/scratch_llm/`. Use it to answer one question:
> *"I'm about to engineer assignment AX — which lectures do I read first, in what order, and where
> does each one land in the code?"*
>
> **How to use it.** Read the assignment's two-to-four lectures **before** writing the module, in the
> order given below; build from scratch in `src/scratch_llm/`; verify against the official scaffold's
> `tests/adapters.py`. The build *how-to* (load-bearing 20%, equations, gates) is in the matching
> [`assignment_guides/A*_BUILD_GUIDE.md`](assignment_guides/INDEX.md) — this doc only tells you which
> slides feed which build.
>
> **Two format notes that change how you consume a lecture:**
> - **`.py` = runnable trace lecture** — execute it next to your build to *see* the concept
>   (lectures 1, 2, 6, 7, 10, 12, 13, 14, 17). Higher signal than slides: you can step through it.
> - **`.pdf` = slides** — read-only (lectures 3, 4, 5, 8, 9, 11, 15, 16).

---

## Master table — all 17 lectures

| # | File | Fmt | Title / core idea (title-level) | Backs | Lands in `src/scratch_llm/` |
|---|---|---|---|---|---|
| 1 | `lecture_01.py` | py | **Overview + tokenization** — "understanding via building"; byte-level BPE | **A1** | `tokenizer.py` |
| 2 | `lecture_02.py` | py | **Resource accounting** — FLOPs, memory, the efficiency frontier | **A1** (·A2) | `model.py`/`optim.py` accounting → A2 100B memory one-pager |
| 3 | `lecture_03.pdf` | pdf | **LM architecture & hyperparameters** — RMSNorm, RoPE, SwiGLU, norms | **A1** | `model.py` |
| 4 | `lecture_04.pdf` | pdf | **Attention alternatives & Mixture of Experts** — GQA, MoE routing | **A1** | `model.py` (GQA), `moe.py` (opt-in) |
| 5 | `lecture_05.pdf` | pdf | **GPUs** — execution model, memory hierarchy, the roofline | **A2** | `kernels/` (roofline) |
| 6 | `lecture_06.py` | py | **Kernels** — benchmarking/profiling + writing Triton kernels | **A2** | `kernels/` (FlashAttention-2) |
| 7 | `lecture_07.py` | py | **Parallelism across GPUs** — collectives, avoiding data-transfer bottlenecks | **A2** | DDP / `rollout/` |
| 8 | `lecture_08.pdf` | pdf | **Parallelism basics** — DP / TP / PP, the comms-vs-compute algebra | **A2** | DDP → ZeRO-1 → FSDP |
| 9 | `lecture_09.pdf` | pdf | **Scaling laws — basics** — the IsoFLOP / power-law method | **A3** | `scaling/` |
| 10 | `lecture_10.py` | py | **Inference** — KV-cache, reducing inference complexity | **A2** + **A5** | `rollout/` (KV-cache), `sampling.py` |
| 11 | `lecture_11.pdf` | pdf | **Scaling — case study & details** — `C=6ND`, Chinchilla extrapolation | **A3** | `scaling/` |
| 12 | `lecture_12.py` | py | **Evaluation** — benchmarks, what "good" means, eval pitfalls | **A3·A4·A5** (cross-cutting) | (discipline, not one module) |
| 13 | `lecture_13.py` | py | **Data 1** — what data to train on; what open models disclose | **A4** | `data/` |
| 14 | `lecture_14.py` | py | **Data 2** — HTML→text, extraction, the curation pipeline | **A4** | `data/curate` |
| 15 | `lecture_15.pdf` | pdf | **"After pretraining" (mid/post-training)** — SFT | **A5** | `algos/` (SFT primitives) |
| 16 | `lecture_16.pdf` | pdf | **Post-training 2 — RL from Verifiable Rewards** — GRPO | **A5** | `algos/` (GRPO/Dr.GRPO), `rewards/`, `envs/` |
| 17 | `lecture_17.py` | py | **Multimodal / omni models** — beyond text | *frontier enrichment* | — (no assignment) |

---

## Read-first order, per assignment

Open these **before** you start building the assignment; run the `.py` ones alongside the code.

### A1 · Basics — the substrate ✅
**`L1 → L2 → L3 → L4`**
- **L1** (py) tokenization → build `tokenizer.py` (byte-level BPE, the deterministic merge rule).
- **L2** (py) resource accounting → the `*_accounting` memory/FLOP math (recurs as the A2 100B answer).
- **L3** (pdf) architecture → the Transformer chain in `model.py` (RMSNorm · RoPE · SwiGLU · SDPA · MHA).
- **L4** (pdf) attention alternatives + MoE → GQA in `model.py`; the opt-in `moe.py`.

### A2 · Systems — fast one GPU, coherent many GPUs 🟡
**`L5 → L6 → L7 → L8`** *(then `L10` for inference; revisit `L2` for the memory math)*
- **L5** (pdf) GPUs → the roofline (arithmetic intensity, BW- vs compute-bound) under `kernels/`.
- **L6** (py) kernels → benchmark/profile + the Triton FlashAttention-2 forward in `kernels/`.
- **L7** (py) multi-GPU parallelism → collectives → DDP (naive → flat-bucket → overlap).
- **L8** (pdf) parallelism basics → the DP/TP/PP comms algebra → ZeRO-1 → FSDP + the 100B one-pager.
- **L10** (py) inference → the KV-cache incremental-decode path (`rollout/`, `sampling.py`).

### A3 · Scaling — fit a curve, extrapolate under a FLOP budget ⬜
**`L9 → L11`** *(then `L12` for eval discipline)*
- **L9** (pdf) scaling-law basics → the log-log power-law fitter in `scaling/`.
- **L11** (pdf) scaling case study → the `C=6ND` bridge, the `a+b≈1` check, honest extrapolation.

### A4 · Data — CommonCrawl → filter → dedup → train ⬜
**`L13 → L14`** *(then `L12` for eval / contamination)*
- **L13** (py) data 1 → what-data choices; the trusted-positive signal for the quality classifier.
- **L14** (py) data 2 → HTML→text extraction + the pipeline order that `data/curate` implements.

### A5 · Alignment — the RL post-training crown ⬜
**`L15 → L16`** *(then `L10` for rollout/inference, `L12` for eval)*
- **L15** (pdf) mid/post-training → the SFT masking primitives in `algos/` (`response_mask` is everything).
- **L16** (pdf) RL from verifiable rewards → GRPO / Dr.GRPO in `algos/`, the grader in `rewards/`, the env in `envs/`.

### Frontier enrichment (no assignment)
- **L17** (py) multimodal / omni models — read for breadth and interview awareness; nothing to build here.

---

## Cross-cutting lectures (deliberately not single-homed)

| Lecture | Primary | Also feeds | Why it travels |
|---|---|---|---|
| **L2** resource accounting | A1 | A2 | the FLOP/memory math you do in A1 *is* the input to A2's "train a 100B model" memory one-pager |
| **L10** inference | A2 | A5 | KV-cache is an A2 systems deliverable; the same serving path is what makes A5's rollouts cheap |
| **L12** evaluation | A3 | A4, A5 | scaling needs a loss surface, data work needs contamination/quality eval, RL needs reward/quality eval |

---

## Pointers

- The build *how-to* (load-bearing 20%, equations, gates): [`assignment_guides/INDEX.md`](assignment_guides/INDEX.md).
- The build spine (order, briefs, definition of done): [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).
- The source lectures + the 5 scaffolds + PDFs: `../../lectures/`.
