# Context Engineering for Claude Code — a Lead Harness Engineer's Field Manual

> Written 2026-06-05 as the teaching artifact for the `reasoningLLM` clean-room harness.
> The harness under `.claude/` *is* the worked example for everything here. Read this once
> end-to-end; thereafter it is reference.

---

## 0. The one idea everything reduces to

A single turn of an agent is a pure function:

```
next_action = LLM( context_window )
```

You do not get to edit the weights. The **only** lever you have over the quality of `next_action`
is **what tokens are in `context_window` at the moment of decision**. That is the whole game.

> **Context engineering = curating the smallest set of high-signal tokens that maximize the
> probability of the correct next action, treating the context window as a finite, decaying resource.**

Three claims are doing work in that sentence; each is a first principle.

### 0.1 Context is *finite*
The window has a hard token budget. Everything competes for it: the system prompt, every CLAUDE.md
that loaded, every tool result, every file you read, the whole conversation so far. When it fills,
the harness **compacts** (summarizes older turns) — lossy by definition. So tokens are not free;
each one you spend on low-value material is a token of capacity, and of *attention*, taken from the
material that decides the action.

### 0.2 Context *decays* — "context rot"
Even below the hard limit, quality degrades as the window grows. Long contexts dilute attention;
the model is likelier to miss a fine detail buried in 100k tokens than the same detail in 5k. More
context is **not** monotonically better. This is why "just paste everything in" is an anti-pattern,
and why a 200-line CLAUDE.md outperforms a 1000-line one even though both "fit."

### 0.3 Signal, not volume
The objective is not "maximal information." It is the **smallest high-signal set**. A precise fact
("API handlers live in `src/api/handlers/`") earns its tokens; a paragraph of generic advice
("write clean, maintainable code") does not — the base model already knows it, so it is pure cost.

### 0.4 Why this beats "prompt engineering"
Prompt engineering optimizes one string you write. Context engineering optimizes the **entire token
stream across a long, multi-turn, tool-using session** — what auto-loads, what the model pulls in
just-in-time, what tool outputs return, what survives compaction. For an agent that runs for hours
over a 14-day sprint, the prompt is a rounding error; the *context strategy* is the system.

---

## 1. The four levers

Every mechanism Claude Code gives you is one of four levers, distinguished by **when its tokens
enter the window** and therefore **what they cost**. Internalize this table; it is the design
language for the rest of the manual.

| Lever | Mechanism | Loads | Token cost | Use it for |
|---|---|---|---|---|
| **1. Always-on** | `CLAUDE.md` (root + nested), `MEMORY.md` index | every turn | **highest** — paid on every decision | small, stable, load-bearing facts the model must never forget |
| **2. On-demand** | skills, slash commands, `@file` refs, memory topic files | only when invoked / matched | ~zero until used | depth, procedures, references — pulled *just-in-time* |
| **3. Produced** | subagents, `/clear`, compaction | from tool output | managed *down* | doing big reads/searches without flooding the main window |
| **4. Structural** | hooks, `settings.json` permissions, plan mode | deterministic, external | ~zero (runs outside the model) | things that must *always* happen, regardless of model discretion |

### The master rule
> **Enforce with structure (4). Reference on-demand (2). Keep always-on (1) lean. Isolate big work in produced context (3).**

Two corollaries you will use constantly:

- **A CLAUDE.md line is a *suggestion*; a hook is *enforcement*.** "Never commit red code" written in
  prose is something the model *may* follow. The same rule as a `PreToolUse` hook that exits 2 on a
  red build is something it *cannot* violate. Put guarantees on lever 4, not lever 1.
- **Progressive disclosure beats pre-loading.** A skill/command/agent shows the model only its
  *description* until it is relevant, then loads its body. You can have fifty of them for almost no
  always-on cost. Pre-loading the same content into CLAUDE.md would pay for all fifty on every turn.

---

## 2. Decision log — every file in this harness, and the lever that justifies it

This is the heart of the manual: for each file we authored, *why it exists*, *which lever*, and
*what the wrong lever would have cost*. Reading this is how you learn to make these calls yourself.

### `CLAUDE.md` (repo root) — Lever 1
- **What:** the constitution — the through-line metric, frozen v0.1.0 scope, L1–L5↔assignment↔file
  map, the engineering disciplines (loss-at-init, overfit-one-batch, seeds, the 3-KL logging), the
  green-CI rule, build commands, and a pointer table to on-demand docs.
- **Why lever 1:** these are the facts the model must apply on *every* decision in this repo —
  forgetting the metric or the frozen scope mid-task is catastrophic. They are small and stable, so
  they earn always-on residence.
- **Wrong-lever cost:** as a skill (lever 2) the model might not load it before acting and would
  drift on scope. As prose-enforcement of green-CI it would be ignorable — so that *one* rule was
  pushed down to lever 4 (the hook), and CLAUDE.md only *describes* it.
- **Discipline:** kept ≤ ~150 lines. Depth — this manual, the [master plan](../../UNIFIED_FRONTIER_PROJECT_SPEC.md),
  the [per-layer build guides](assignment_guides/INDEX.md) — is *referenced*, not inlined (see the
  `docs/` knowledge-base entry below for why that's Lever 2, not Lever 1).

### `.claude/settings.json` — Lever 4 (permissions) + wiring for Lever 4 (hooks)
- **What:** pre-approves the safe, frequent sprint commands (`uv`, `pytest`, `ruff`, `pyright`,
  read-only `git`) and registers the three hooks.
- **Why lever 4:** permissions shape the *action space* deterministically — you stop being prompted
  50×/day for `pytest` (which would be pure friction) without granting anything destructive.
- **The teaching moment:** my first draft added `git add`/`git commit` to the allow-list. The
  auto-mode classifier **blocked the write**, citing your `RULES.md` ("never auto-commit"). That is
  lever 4 defending you against *me*. The fix was to keep commits prompt-gated. Lesson: least
  privilege on the allow-list; never auto-approve irreversible/outward actions.

### `.claude/hooks/green-ci-gate.sh` — Lever 4 (the flagship enforcement)
- **What:** a `PreToolUse` hook on `git commit` that runs ruff + ruff-format + pyright + pytest and
  **exits 2 (blocks the commit)** if any fail. Graceful: no git → skip; no tests → pass (pytest 5).
- **Why lever 4, not lever 1:** "no red commits" is a *guarantee*, and guarantees cannot live in
  prose the model may rationalize around. The hook makes a red commit *issued through Claude Code*
  mechanically impossible. **Caveat (know the boundary of your enforcement):** a `PreToolUse` hook
  only sees commits made via Claude Code's Bash tool — a commit from an external terminal bypasses
  it. The robust fix is DRY: the same script exits non-zero on failure, so symlinking it as a
  git-native hook (`ln -s ../../.claude/hooks/green-ci-gate.sh .git/hooks/pre-commit` at `git init`)
  enforces on *every* commit path. Always know which paths your structural lever actually covers.
- **Independence you must understand:** hooks and permissions are *orthogonal* levers. `git commit`
  is **not** on the permission allow-list (so it still prompts you — your `RULES.md`), yet the hook
  still fires on it. Permission = "may the tool run?"; hook = "what happens around the run?"

### `.claude/hooks/lint-on-edit.sh` — Lever 4 (convenience, non-blocking)
- **What:** a `PostToolUse` hook on `Edit|Write` that ruff-fixes+formats the just-edited `.py` file.
  Always exits 0 — style is never allowed to block work.
- **Why:** takes formatting entirely off your (and the model's) cognitive budget — a recurring
  low-value concern automated into structure. Note it parses the tool JSON on stdin with `python3`,
  not `jq`: **fewer dependencies = fewer failure modes** is itself a context-engineering value (a
  missing `jq` would silently break a `jq`-based hook on a fresh machine).

### `.claude/hooks/session-start.sh` — Lever 4 feeding Lever 1
- **What:** a `SessionStart` hook whose stdout is injected into context: current branch, uncommitted
  file count, venv presence, and the cockpit pointer.
- **Why this is subtle and good:** CLAUDE.md is *static* — it cannot know your branch right now. The
  hook supplies the small slice of *dynamic* state worth always-on residence. Keep it ~2 lines:
  every line is paid once per session, so it must out-earn its tokens. (Don't echo what CLAUDE.md
  already says — that's double-paying.)

### `.claude/agents/*.md` — Levers 2 + 3
Two specialists. The frontmatter is deliberately **minimal and verified** (`name`, `description`,
`tools`, `model`) — I discarded several fields a docs-summary hallucinated, because an invented field
is silently ignored. The `description` is load-bearing: it is the *only* part the model sees until
delegation, so it is written to trigger on exactly the right request.

- `ship-reviewer` (Lever 3) — gates a diff on correctness + design/scope. Isolates the diff-reading;
  returns ACCEPT/REJECT.
- `rl-run-auditor` (Lever 3) — scans run logs for the mandatory guardrails (the three KLs, IS-ratio
  histograms, reward/length stats, `kl_train_infer` HALT@0.10). A mechanical checklist that gates
  trust in any RL number.

### `.claude/commands/*.md` — Lever 2
You-triggered macros, **zero context cost until typed**, that expand into precise prompts delegating
to the agents. `/ship` (reviews the diff, runs green-CI, then hands *you* the commit command — never
auto-commits) and `/audit-rl` (routes RL run logs to the auditor). **Command vs skill vs agent:**
command = you always invoke it manually; skill = model auto-loads when relevant; agent = isolated
context + restricted tools. We use commands for deliberate, author-driven gates.

### Project memory (`~/.claude/projects/.../memory/`) — Lever 1 index + Lever 2 bodies
`MEMORY.md` (always-on, one line each) + topic files (recalled on relevance). Seeded with the
*non-obvious, hard-won* facts — above all the three-repo topology I had to discover — and explicitly
*not* with what the repo already encodes (code structure, git history). **Why this matters:** memory
is the only context that survives `/clear` between sprint days. A fresh session reads `MEMORY.md` and
instantly knows not to confuse the clean-room with the ship repo.

### `cs336/CLAUDE.md` (umbrella) — Lever 1, thin
A ~30-line bridge so that launching from the umbrella root still orients you across the three repos.
Thin on purpose: the real constitutions live in the sub-repos; duplicating them here would create
*conflicting* always-on instructions (a real failure mode — when two CLAUDE.md files disagree, the
model picks one arbitrarily).

### `docs/` — the on-demand knowledge base (Lever 2)
The `.claude/` harness decides *how* the agent acts; the `docs/` tree supplies *what* to build and
*why each piece is load-bearing*. It is deliberately **Lever 2** — referenced just-in-time, never
auto-loaded — so all of its depth costs ~zero on a normal turn.

- **[`../../UNIFIED_FRONTIER_PROJECT_SPEC.md`](../../UNIFIED_FRONTIER_PROJECT_SPEC.md)** — the master plan:
  §1 thesis, §2 five-layer stack, §3 per-layer briefs (core + senior add-ons, each with its falsifiable
  prediction + kill criterion), §5 evidence ledger / role map, frozen scope. This is the *source of
  truth for the project's content*; the root `CLAUDE.md` (Lever 1) is only its lean, always-on summary —
  the L1–L5↔assignment↔file map there is the spec's §2 table compressed to fit always-on residence.
- **[`assignment_guides/`](assignment_guides/INDEX.md)** — five senior reviews of the CS336 PDFs, each
  mapping the load-bearing 20% of an assignment onto `src/reasoning_llm/`, tagged
  LOAD-BEARING / COURSE-ROTE / SKIP, with the A2-first build order and per-day schedule. Enter at
  `INDEX.md`; pull the one guide for the layer you're building, when you build it.
- **Why Lever 2, not Lever 1:** spec + guides are thousands of lines. Inlining them into `CLAUDE.md`
  would pay for *all* of it on *every* turn (§0.1) and trigger context rot (§0.2) — to surface a single
  per-layer brief the model needs only while working that layer. Progressive disclosure (§1) is exactly
  the right tool: the model reads the pointer always-on, the depth on demand.
- **The discipline (bidirectional pointers, no orphans):** `CLAUDE.md` and `MEMORY.md` hold only the
  pointers *into* `docs/`; the spec and guides hold the depth and point *back* (the guides' §"Source-of-truth
  pointers" cite the spec and `CLAUDE.md`). So whichever file you enter from, you can reach the rest —
  and no copy of the L1–L5 map is the authoritative one twice (which would re-create the conflicting-
  duplicate failure mode above). One source of truth per fact; everything else links to it.

---

## 3. The Lead-Harness-Engineer operating manual (how to *drive* this across the sprint)

The files above are the static harness. This section is the dynamic discipline — how you run Claude
Code day to day so context stays high-signal.

### 3.1 Launch from the right directory
The clean-room harness (settings, hooks, CLAUDE.md) only loads when cwd is inside
`reasoningLLM_scratch/`, because configuration is discovered by walking **up** from cwd. `cd` into
the repo before `claude`. (This is also what makes `$CLAUDE_PROJECT_DIR` in the hook commands
resolve to the repo root.)

### 3.2 Context hygiene: `/clear` vs compaction
- **`/clear`** wipes the conversation. Use it **between tasks/days** — the moment the current thread's
  content stops paying rent. Your external memory (tracker, `MEMORY.md`, the repo) carries state, so
  clearing is cheap and *good*: it resets you to a clean, high-signal window. Frozen decision #8 in
  the daily contract says exactly this: "External memory > chat memory; `/clear` between days is fine."
- **Compaction** is the harness auto-summarizing when the window fills. It is lossy. Prefer to
  `/clear` deliberately *before* you'd hit compaction, so you control what survives rather than
  letting a summarizer guess.

### 3.3 Delegate big reads to subagents — keep conclusions, discard dumps
When a task means "search many files / read long logs / explore an unknown area," send a **subagent**
(the built-in `Explore`, or one of ours). It does the heavy reading in its *own* window and returns
the conclusion. Your main context never sees the 10k lines it sifted. Rule of thumb: **if the raw
material won't be referenced again after the question is answered, isolate it.** This is lever 3, and
it is the single biggest defense against context rot on a long project.

### 3.4 Use plan mode for anything non-trivial
Plan mode (read-only until you approve) forces exploration and a written plan *before* edits. It is
context engineering as *process*: you front-load understanding into a durable plan artifact, then
execute against it — instead of discovering scope halfway through a pile of edits. Use it for any
new file, schema change, or multi-step build. (This very harness was built that way.)

### 3.5 The advisor cadence
Call the `advisor` *before* committing to an approach and *before* declaring done — not after every
step. Its highest value is catching a wrong premise before you build on it. (In building this harness,
the advisor caught that a docs-summary had *fabricated* config schemas — preventing a harness that
would silently not load. That one call paid for itself many times over.)

### 3.6 Verify mechanisms *fire*, never just that they read well
A harness file that looks perfect but doesn't load is worse than none — it gives false confidence.
After authoring, **introspect**: `/memory` (what CLAUDE.md/memory loaded), `/agents` (are the
specialists discovered?), `/hooks` (are the hooks registered?), and actually *run* a hook/command
once. We did this here: we executed all three hooks and watched their output before trusting them.

### 3.7 Write conclusions to durable state before long operations
Before a long advisor call, a big agent run, or anything that might be interrupted, make the
deliverable durable (write the file, commit). If the session ends mid-operation, a durable result
persists; an in-flight one is lost. This is why every file in this build was written, not described.

---

## 4. §F — Audit of the global `~/.claude/` framework (measured, with a recommendation)

You asked for a measurement of what your global "SuperClaude" framework actually costs on every turn,
and a recommendation. Here it is — *recommendation only; I changed nothing global*.

### What auto-loads on every turn (measured 2026-06-05)
`~/.claude/CLAUDE.md` `@import`s exactly two files. The always-on global cost is:

| File | Lines | Words | Loads every turn? |
|---|---|---|---|
| `~/.claude/CLAUDE.md` | 223 | 1048 | yes (user-level CLAUDE.md) |
| `@PRINCIPLES.md` | 160 | 1175 | yes (imported) |
| `@RULES.md` | 72 | 386 | yes (imported) |
| **Total always-on** | **455** | **~2609** | **≈ 3,300–3,900 tokens, every turn** |

### What does *not* load (the important finding)
`ORCHESTRATOR.md` (533), `PERSONAS.md` (467), `FLAGS.md` (220), `MODES.md` (309), `MCP.md` (225),
`COMMANDS.md` (159) — **1,913 lines** — are referenced only in *prose* (e.g. RULES.md line 54
"see ORCHESTRATOR.md"), **not** via `@import`. So they are **dormant**: they cost nothing on a normal
turn, but they also do nothing — they are documentation no mechanism actually loads.

### Recommendation (your call; I did not apply it)
1. **Trim the always-on 455 lines toward signal.** Much of `PRINCIPLES.md`/`RULES.md` is generic
   senior-dev advice the base model already knows (SOLID, DRY, "fail fast") — high token cost, low
   marginal signal (§0.3). Keep the *specific, non-default* rules (e.g. "never auto-commit", the
   auto-approved command list); move the generic philosophy to an on-demand skill you load when you
   actually want a review lens. Plausible saving: ~1,500–2,500 tokens **per turn**, which over a long
   session is large.
2. **Convert the 1,913 dormant lines to real mechanisms or delete them.** As inert `.md` files they
   are neither loaded nor enforced. If `PERSONAS`/`FLAGS`/`MODES` describe behaviors you want, they
   belong as **skills** (load on-demand) or **hooks** (enforced), not as prose no turn ever sees.
   Otherwise they are maintenance weight pretending to be a framework.
3. **Resolve global↔project conflicts.** Your global `RULES.md` auto-approves `pytest`/`ruff`; this
   repo's `settings.json` does too — consistent, good. But watch for the general hazard: when global
   and project CLAUDE.md disagree, the model picks arbitrarily (§2, umbrella note). Periodically read
   both and remove contradictions.

The principle behind all three: **always-on residence is the most expensive real estate in the
system — make every line there earn it, and push everything else down a lever.**

---

## 5. Verified mechanism reference (so you can extend this yourself)

Authoritative schemas, confirmed against on-disk files + `code.claude.com` docs (2026-06). Author
**only fields you can verify** — an invented frontmatter field is silently ignored.

- **CLAUDE.md:** root `./CLAUDE.md` + nested `subdir/CLAUDE.md` (nested loads on-demand when the model
  works in that subtree). `@path` imports resolve relative to the importing file (≤4 hops). User
  `~/.claude/CLAUDE.md` loads before project; project wins on conflict (additive, most-specific-last).
- **Skills:** `.claude/skills/<name>/SKILL.md`, frontmatter `name` + `description` (+ optional
  `metadata`). Description loads at startup; body on invocation/relevance.
- **Subagents:** `.claude/agents/<name>.md`, frontmatter `name`, `description`, `tools` (list),
  `model` (`opus`/`sonnet`/`haiku` or full id); body = the agent's system prompt. Description drives
  auto-delegation; the agent runs in an isolated window and returns only its final message.
- **Slash commands:** `.claude/commands/<name>.md`, frontmatter `description` (+ `argument-hint`);
  body is the expansion. `$ARGUMENTS` interpolates; `@file` injects a file; a leading `!` runs bash.
- **Hooks (`settings.json`):** real events include `SessionStart`, `UserPromptSubmit`, `PreToolUse`,
  `PostToolUse`, `PreCompact`, `SubagentStop`, `Stop`, `SessionEnd` (and more). Shape:
  `hooks → Event → [ { matcher, if, hooks:[ {type:"command", command, timeout} ] } ]`.
  **Exit 0** = proceed (stdout of `SessionStart`/`UserPromptSubmit` is *added to context*);
  **exit 2** = block (on `PreToolUse` blocks the call; stderr is fed back to the model);
  any other code = non-blocking error. `PostToolUse` receives the tool-call JSON on stdin
  (`tool_input.file_path` for `Edit`/`Write`).
- **settings precedence:** managed > CLI flags > `.claude/settings.local.json` > `.claude/settings.json`
  > `~/.claude/settings.json`. Permission **deny** rules merge across all scopes (most restrictive wins).

### How to verify *this* harness loaded (do this once now)
```
/memory     # expect: reasoningLLM_scratch/CLAUDE.md + the MEMORY.md entries
/agents     # expect: ship-reviewer, rl-run-auditor
/hooks      # expect: SessionStart, PostToolUse(Edit|Write), PreToolUse(git commit)
```
(Run Claude from inside `reasoningLLM_scratch/`, or these won't be in scope.)

---

## 6. Self-check (does the harness design hold up?)

Sanity questions on the context-engineering choices above — if any answer is shaky, re-read that section:

1. **Lever placement.** "No red commits" is on lever 4 (a hook), but "the through-line metric is
   `true_quality_gap`" is on lever 1 (CLAUDE.md). Why is each on its lever and not the other's?
2. **Modify-and-predict.** If we moved the entire 1,913 lines of dormant framework docs into
   `@import`s in `~/.claude/CLAUDE.md`, predict the effect on (a) per-turn token cost, (b) the
   model's adherence to your *specific* rules, (c) context rot on a long session.
3. **Subagent economics.** You need to find every place `kl_train_infer` is computed across both
   reasoningLLM repos. Why send a subagent rather than grep-and-read in the main thread? What exactly
   does your main window gain?
4. **The independence point.** `git commit` is *not* on the permission allow-list, yet the
   green-CI hook still fires on it. Explain why permissions and hooks are orthogonal levers.
5. **Why `/clear` is a feature, not a loss.** Given external memory, argue why clearing between days
   *improves* the next day's output rather than throwing away useful context.

If any answer is shaky, that's the section to re-read — then we discuss it.
