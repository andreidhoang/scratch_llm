# AGENTS.md — active workspace bridge

Reviewed 2026-09-12. `scratch_llm` supplies the kernels, numerical references, serving and
training substrate for the GPU systems + coding-RL research campaign.

## Authority and orientation

Read `../ladders/CLAUDE.md` and `../ladders/plan/SIXTY_DAYS_SIX_LADDERS.md` v5 first, then this
file and `CLAUDE.md` for repository facts. The workspace plan controls priorities, units, exit
criteria and the kill list; there is no calendar. `../ladders/experiments/CURRENT`, the active
rung contract and raw results identify the current experiment; a pointer is not readiness.

`PLAN.md`, `MASTERY_LEDGER.md`, `docs/KERNEL_MASTERY_SPEC.md`, `docs/k3/`, the assignment guides,
`performance/*.md`, `deploy/runbooks/` and `docs/CONTEXT_ENGINEERING.md` are historical references,
not competing plans. Preserve them; consult a relevant scientific mechanism without reviving its
old schedule, ownership mode or shipping rule. **KDA / the Kimi Linear lineage is v5's spine**
(A1 upstream, A2 miniature); `docs/k3/ROADMAP.md §0` is still the argument for building it here,
its sequencing is not. R1/R-R5 kernel-coding RL is planned in the workspace (A3); its
integrated implementation is pending.

On a remote host, locate the matching ladders checkout and recorded revision before choosing
work. The legacy provisioning scripts do not install that workspace automatically. Missing
workspace context does not promote the archived `PLAN.md` to authority.

## Ownership

Huy owns derivations, predictions, the first kernel core / loss-math / memory-model implementation,
tolerance design, causal diagnosis and technical defence. Agents may write harnesses, independent
reference oracles, adapters, adversarial tests, maps and reproductions, and implement fixes after
Huy names the diagnosis. Independence requires a separate derivation or trusted reference, not
copying the candidate implementation. No additional planning Markdown is needed.

**Stricter boundary: agents never create, edit, move or delete files under
`src/scratch_llm/k3/core/`.** Read `src/scratch_llm/k3/HANDCRAFTED.md` before work touching the K3
surface. For core modules, only adversarial tests after human authorship and prose proposals for
the human are allowed; no implementation diffs. Its dated scaffolding waiver is not blanket
permission to write the core. Preserve the paired ownership of K3 math outside `core/` as well.

## Verification and evidence

Use `CLAUDE.md` for actual CI commands and installed hook behavior. Correctness precedes timing;
match the baseline's semantics, dtype, shape and execution protocol. Record campaign measurements
and raw artifacts in ladders; `bench/RESULTS.md` retains local historical measurements and links.
Label claims measured, reported, reproduced or unverified with their exact scope. Compile success,
fixture tests, historical `[FACT]` labels and a green process exit do not certify GPU performance.
Document null results and unmet gates without inventing a speedup or capability improvement.

The old oracle/kernel write-guard hooks and `.claude/execution-mode` are absent in this checkout.
Ownership is policy; existing lint/CI hooks are separate mechanisms and stay installed. Review
commands do not automatically authorize a commit, push, rental or publication.
