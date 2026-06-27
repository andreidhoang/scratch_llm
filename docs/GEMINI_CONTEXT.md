# Gemini Context Engineering & Harness Reference Manual

This manual is the onboarding reference for Gemini when working on the `scratch_llm` project, particularly on a fresh deployment (e.g., a rented GPU instance on Vast.ai).

---

## 1. Project Topology & Core Mission
The mission of this repository is **mastering CS336 (Stanford: Language Modeling from Scratch) from scratch to production**. We own every layer of the language model from the byte up to the RL post-training update.

*   **Repository Root (Vast.ai clone)**: `scratch_llm/`
*   **Source Code**: `src/scratch_llm/`
    *   `tokenizer.py`: Byte-level BPE tokenizer.
    *   `model.py`: Decoder LM (RMSNorm, RoPE, SwiGLU, MHA, GQA, QK-norm).
    *   `moe.py`: DeepSeek-style MoE FFN (sigmoid gate, aux-loss-free balancer).
    *   `optim.py`: AdamW with decoupled weight decay + cosine schedule.
    *   `train.py`: Checkpoint-aware training loop on memmapped inputs.
    *   `sampling.py`: Temperature + nucleus (top-p) decoding.
    *   `kernels/`: GPU kernels (e.g., Triton FlashAttention-2 fwd/bwd).
    *   `algos/`, `rewards/`, `envs/`: SFT, Expert Iteration, GRPO/Dr.GRPO.
    *   `utils/`: Checkpointing, mixed-precision, monitors (entropy, KLs, reward stats).
*   **Assignment Specs**: Located in `docs/assignment_guides/` (mapped to assignment PDFs).
*   **Build Status & Roadmap**: `docs/STATUS.md` and `docs/IMPLEMENTATION_PLAN.md`.

---

## 2. Gemini Context Engineering Rules
To keep development efficient and prevent **context rot** (performance decay due to overfilled context windows):

1.  **Context Hygiene**: Reset/clear the conversation between unrelated tasks or at the start of a new work session. Rely on durable project state (`docs/STATUS.md`, `docs/IMPLEMENTATION_PLAN.md`, or code files) to rebuild context instead of keeping long chat histories.
2.  **Lean Always-On**: Do not auto-load long documentation. Refer to `docs/` files only when relevant to the task (Lever 2: progressive disclosure).
3.  **Durable State**: Always write conclusions/code to files before executing long-running validation commands. If a session is lost/interrupted, the code remains.
4.  **No Placeholders**: Never use placeholder implementations or mock code. If code is generated, it must be production-ready and fully written.

---

## 3. Custom Harness & Invariants
The repository has automated safety barriers and strict learning protocols:

### A. The Green-CI Gate
We enforce a green-only policy. Before any code is committed, the following tests and linters must pass clean:
```bash
ruff check src tests
ruff format --check src tests
pyright
pytest -m "not gpu"
```
*Note: A git pre-commit hook (`.git/hooks/pre-commit` pointing to `.claude/hooks/green-ci-gate.sh`) enforces this. Do not bypass it.*

### B. The Meat-Boundary Backstop (Kernel Write Guard)
*   **The Invariant**: Gemini **MUST NOT** edit or write core GPU kernel implementations directly.
*   **Affected files**: `src/scratch_llm/kernels/matmul.py`, `src/scratch_llm/kernels/*_triton.py`, or any `*_kernel.py`.
*   **Your Role**: Gemini can tutor the user, explain mathematical derivations, analyze benchmark rooflines, and write test/oracle wrappers, but the user must write the actual kernel logic. If asked to edit a kernel implementation, refuse and explain the mechanism conceptually instead.

### C. First-Principles Mastery & Visualizations
When implementing load-bearing layers, follow the **Socratic learning cycle**:
1.  **Derive the math** first.
2.  **Visualize** through three lenses: **Tensor shapes**, **ASCII system-flow**, and a **tiny worked numeric example**.
3.  **Predict-before-run**: Write the expected tensor shape/value before running the test.
4.  **Green-CI**: Fix failures until clean.

---

## 4. Running on Vast.ai (Rented GPU Box)
When launching on a Vast.ai instance:

### A. Clone and Setup
```bash
git clone https://github.com/andreidhoang/scratch_llm.git && cd scratch_llm
uv sync
uv pip install torch triton
```

### B. Sanity Checks
Ensure the GPU is active and the kernel test harness works:
```bash
nvidia-smi
# Run a baseline roofline bench
PYTHONPATH=src uv run python -c "from scratch_llm.kernels.bench import matmul_roofline; \
  from scratch_llm.kernels.matmul import matmul_naive; matmul_roofline(matmul_naive, 4096, 4096, 4096)"
# Run the GPU-specific tests
uv run pytest -m gpu tests/test_matmul.py
```

### C. General Testing
*   **CPU tests**: `pytest -m "not gpu"`
*   **GPU tests**: `pytest -m gpu` (e.g., FlashAttention kernels, Triton benchmarks)
*   **Run linter**: `ruff check src tests && ruff format --check src tests`
*   **Type check**: `pyright`
