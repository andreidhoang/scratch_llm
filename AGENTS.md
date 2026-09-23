# AGENTS.md — active workspace bridge

Reviewed 2026-09-21. `scratch_llm` supplies the kernels, numerical references, serving and
training substrate for the GPU systems + coding-RL research campaign.

## Authority and orientation

Read `../ladders/CLAUDE.md`, `../ladders/plan/SIXTY_DAYS_SIX_LADDERS.md` v6 and
`../ladders/experiments/KW/map.md` first, then this file and `CLAUDE.md` for repository facts.
The workspace plan controls priorities, units, exit
criteria and the kill list; there is no calendar. `../ladders/experiments/CURRENT`, the active
rung contract and raw results identify the current experiment; a pointer is not readiness.
The planned next unit is CPU `K3/kda-v0`, then `KW/R01-roofline` on an accepted NVIDIA host. This Mac can validate KDA parity but cannot close an NVIDIA roofline or counter gate; `experiments/CURRENT` remains Huy-owned.

**VI:** Chạy đúng phép kiểm tra KDA trên Mac là bằng chứng về tính đúng, không phải số đo tốc độ GPU. Không viết instrument CUDA theo phỏng đoán rồi gọi đó là tiến độ R01.

`PLAN.md`, `MASTERY_LEDGER.md`, `docs/KERNEL_MASTERY_SPEC.md`, `docs/k3/`, the assignment guides,
`performance/*.md`, `deploy/runbooks/` and `docs/CONTEXT_ENGINEERING.md` are historical references,
not competing plans. Preserve them; consult a relevant scientific mechanism without reviving its
old schedule, ownership mode or shipping rule. **KW is the mechanism lane; KDA / the Kimi Linear
lineage is v6's production spine** (A1 upstream, A2 miniature). `docs/k3/ROADMAP.md §0` is still the
argument for building it here, its sequencing is not. R1/R-R5 kernel-coding RL is planned in A3; its
integrated implementation is pending.

Interview practice follows `../ladders/plan/SIXTY_DAYS_SIX_LADDERS.md` §2.4 and the dated
`../ladders/worklogs/2026-09-21_interview_engineering.md` §§3, 6. The Anthropic MLE
Deep-ML track is a third-party prompt index mapped to existing KW/A1–A3 evidence, never an
authority for experiment order, current Anthropic questions, or hiring readiness. Actual
assessment instructions control AI and document use.

**VI:** Bài luyện chỉ chỉ đường tới unit đang có; nguồn Anthropic chính thức quyết định quy tắc phỏng vấn, còn plan v6 quyết định gate kỹ thuật. Không biến điểm mock thành kết quả GPU hay model.

For mentoring pace, follow plan §2.3's 2026-09-21 update: lead with the substantive mechanism and real code/evidence, not serial elementary quizzes. A skipped toy explanation never skips independent correctness, benchmark sanity checks, Huy's prediction or an active gate.

**VI:** Bỏ bài hỏi đáp quá nhỏ, không bỏ kiểm chứng. Sau khi hiểu cơ chế, chuyển thẳng sang việc thật và đặt câu hỏi tại ranh giới quyết định.

On a remote host, locate the matching ladders checkout and recorded revision before choosing
work. Export `LADDERS_PROVIDER`, `LADDERS_REGION`, `LADDERS_INSTANCE_TYPE` and `LADDERS_IMAGE`; use
the ladders bootstrap/benchmark workflow so raw artifacts return to the active rung. Missing
workspace context does not promote the archived `PLAN.md` to authority.

## Ownership

Huy owns derivations, predictions, tolerance design, causal diagnosis and technical defence.
Agents may write harnesses, independent reference oracles, adapters, adversarial tests, maps,
reproductions, and implement fixes after Huy names the diagnosis. Independence requires a
separate derivation or trusted reference, not copying the candidate implementation. No additional
planning Markdown is needed.

**`src/scratch_llm/k3/core/` file-ownership boundary removed 2026-09-22** (formerly documented in
`src/scratch_llm/k3/HANDCRAFTED.md`, now deleted): any agent may create, edit or implement modules
there. The gate that used to be enforced by file ownership is enforced by evidence instead —
`tests/test_kda_parity_fla.py` (9/9), the three-path equivalence (chunkwise ≡ recurrent ≡ float64
reference), each module's `tests/test_k3_<module>.py` (in-file transcription of Moonshot's HF
reference, fp64 path equivalence, planted-bug mutations that must fail), and a matched dense
control at equal parameter budget before any A2 claim. The live K3 state — what is built, what is
next, which decisions are Huy's — is the docstring of `src/scratch_llm/k3/__init__.py`; update it
whenever a module changes status.

## Verification and evidence

Use `CLAUDE.md` for actual CI commands and installed hook behavior. Correctness precedes timing;
match the baseline's semantics, dtype, shape and execution protocol. Record campaign measurements
under `../ladders/experiments/<L>/<R>/results/<timestamp>/`; the ladders ledger indexes claims while
`bench/RESULTS.md` retains local historical measurements and links.
Label claims measured, reported, reproduced or unverified with their exact scope. Compile success,
fixture tests, historical `[FACT]` labels and a green process exit do not certify GPU performance.
Document null results and unmet gates without inventing a speedup or capability improvement.

The old oracle/kernel write-guard hooks and `.claude/execution-mode` are absent in this checkout.
Ownership is policy; existing lint/CI hooks are separate mechanisms and stay installed. Review
commands do not automatically authorize a commit, push, rental or publication.
