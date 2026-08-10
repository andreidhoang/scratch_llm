# HANDCRAFTED.md — the two-citizenship rule for the K3 track

> The K3 track optimizes **defensible understanding per week**, not lines written
> (docs/k3/ROADMAP.md §0). The sharp criterion: *if a bug would be silent — code runs, loss
> still drops, model is subtly worse — the human owns the file. If a bug would be loud —
> shape error, crash, failed test — it may be delegated.* This file is the enforcement.
> It is referenced by the root `AGENTS.md`, which binds every agent session.

> **BOUNDARY WAIVER — logged 2026-08-04.** The user explicitly waived the core/ read-only
> rule to obtain scaffolding. The 5 core modules now exist as 🟡 **agent-authored templates**
> (docstring + signatures + `raise NotImplementedError` math bodies — see status table below).
> The delete-test still applies: when a hand-build passes its mastery bar, `rm` the template
> and rewrite from the docstring, then mark ✅ PROVEN. **Templates are scaffolds to build on,
> NOT verified math** — never assume the parameter layout or signatures are correct; they are
> a starting point the human re-derives.

## Class 1 — `core/` : HAND-BUILT (human authors every line)

**Agent rule: read-only.** Agents never create, edit, move, or delete files under
`src/scratch_llm/k3/core/`. For core modules agents may only:
(a) write **adversarial tests** in `tests/` *after* the human authors a module (red team:
hostile shapes, extreme decay values, train/decode drift probes), and
(b) write **proposals** as markdown (never diffs to apply) — the human retypes what they keep.
Retyping is the mechanism, not overhead.

| Module | Mechanism (why it's silent-bug-prone) | Mastery bar | Status |
|---|---|---|---|
| `situ.py` | SiTU-GLU softcaps β1=4 / β2=25 — bf16 saturation → activation outliers | delete-test + bound proof \|f\| ≤ 100 | 🟡 SCAFFOLD (template, 2026-08-04) |
| `kda.py` | delta rule + per-channel Diag(α) + scaled-sigmoid g_min=−5 — wrong decay floor still trains, forgets badly at long context; A_log semantics trap: the real checkpoint carries [128] per-dim decay tensors in every KDA layer and the released code's per-head framing does not explain them (FACTS A18) | delete-test + chunkwise≡recurrent≡f64 + FLA parity explained + loads the real checkpoint's [128] A_log with verified semantics (pre-flight: resolve the released-code mapping) | 🟡 SCAFFOLD (template, 2026-08-04) — **CRITICAL PATH** |
| `gated_mla.py` | weight-absorption identity + NoPE — wrong fold looks correct, wastes cache; gate placement: the sigmoid gate is taken pre-W_o from the layer input x_t — moving it post-W_o still trains, silently (K2_PROPOSAL_KDA §3; K3 report Eqs. 6–7) | delete-test + absorbed≡naive proof + gate-source probe (gate must read x_t, pre-W_o) | 🟡 SCAFFOLD (template, 2026-08-04) |
| `latent_moe.py` | sigmoid router + Quantile Balancing — wrong bias sign/delay → slow expert collapse | delete-test + load convergence q=mk/n | 🟡 SCAFFOLD (template, 2026-08-04) |
| `attn_res.py` | learned pseudo-queries + online-softmax merge — merge bookkeeping (running max/normalizer) wrong but still trains; a wrong source list is silent: FACTS A19(a) measured single-source pseudo-queries receiving zero gradient and decaying to ~0 (drop the embedding source and nothing complains) — block≡full catches neither | delete-test + block≡full equivalence + merge-bookkeeping probe + pseudo-query gradient-flow probe | 🟡 SCAFFOLD (template, 2026-08-04) |

**Delete test:** the human can `rm` the file and rewrite it from its own derivation docstring
(repo convention: intent + invariant + interview question, like `linear_attn.py`). Only then
is the module marked ✅ PROVEN (date recorded here).

## Class 2 — PAIRED (human writes the math, agents write plumbing + tests)

| Module | Human owns | Agents own | Status |
|---|---|---|---|
| `muon.py` | per-head NS partitioning on Q/K/V momentum blocks + K2 weight clipping | optimizer plumbing, F1-race harness wiring | 🟡 SCAFFOLD (template, 2026-08-04) |
| `qat.py` | MXFP4 quantizer + STE derivation (paper-first) | training-loop integration, QAT arms | 🟡 SCAFFOLD (template, 2026-08-04) |
| `param_count.py` spec | the accounting conventions (already landed, FACTS A18) | the script + gate test ✅ DONE | ✅ DONE |

## Class 3 — DELEGATED (agents own, human reviews)

`config.py` ✅ DONE · `param_count.py` ✅ DONE · `model.py` assembly (✅ scaffold, 2026-08-04) ·
`serve.py` (🟡 scaffold, 2026-08-04) · non-core tests · runbooks
(`deploy/runbooks/k3_8xb300_modal.md`) · `docs/k3/FACTS.md` maintenance · Aug-3 book
cross-read (log disagreements in FACTS.md) · Modal ops in K9 (human designs the measurements
and reads every number) · docs formatting.

## Protocol per core module

1. Human reads the primary source (FACTS.md §2 anchors), hand-builds the module + derivation docstring.
2. Human asks an agent to red-team it: adversarial tests only, no fixes.
3. Human fixes what the red team finds — by hand.
4. Delete test → module marked PROVEN with the date in the table above.
