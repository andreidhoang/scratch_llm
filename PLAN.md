# PLAN — the only plan (2026-08-26 · production-first)

One page. If any document contradicts this file, this file wins. If this file goes stale,
fix THIS file — never write a second one. **Superseded material is DELETED, not archived**
(cleanup 31/08): `git log --diff-filter=D --name-only` is the archive, and `git show <sha>:<path>`
restores any line. A repo-side archive folder is just stale files with a nicer name.

> **REDIRECT 2026-08-26 evening (operator, verbatim): "actually engineering and building in
> real project for real production from now on … follow the right building and contribution
> path not just for the sake of learning."** Two consequences, recorded not argued:
> **(1) The fork below is RESOLVED → (X).** The production-contribution path *is* branch X
> — E1 public, applications fire, the vLLM lane opens. (Y) parked the only artifacts with
> external consumers; the directive un-parks them. Operator may veto by reverting this line.
> **(2) The inversion, named:** `src/scratch_llm/mastery/` is not a learning sandbox — it is
> the evidence base for a live vLLM review **with a clock** (PR #45819 under active review,
> rebased 08-24, zero approvals, contested numerics). The no-external-consumer work was the
> K3 rebuild's outer rungs (K3–K10), which move to the backlog; **K2 is kept, re-scoped as
> the reference implementation that adjudicates production kernels.** Learning continues
> in-flight on real work (T-loop/PRR/seal unchanged); it is the byproduct measured by
> shrinking prediction error, never the goal. **Every task names its external consumer; a
> task that cannot is cut or backlogged.**

> **AMENDMENT 2026-08-31 (operator, verbatim): "k3 and performance + kernel engineering
> plan is real production plan not just learning plan if we master these we master
> everything."** Recorded, not argued — this is the veto the 26/08 header invited.
> **(1) K3–K10 and the M0→M6 kernel ladder LEAVE the backlog** and return as **the mastery
> lane**: production depth, not study. **(2) The lane is delivered as SESSIONS on the L0–L5
> altitudes** — spec **§12.5** owns the table (altitude · deliverable · consumer · OWNER);
> a session's output is code or a measured number, never notes. **(3) The 29/08 binding rule
> is satisfied, not waived** — every session names its consumer; the loop itself is one
> (§11 gap 7: the recorded reject reason was *"optimized by trial and error rather than by
> rigorous measurement"*). **(4) Ordering is UNCHANGED where a clock exists.** #45819's
> review window is still the only dated external event in this file; the mastery lane runs
> the rest of the day. **The cost, stated once:** depth spends hours the dated node would
> otherwise take — **rule 4 (one public change, pushed, before 21:00) is the circuit breaker**
> that stops this becoming Jul–Aug again (~87 planning files, 0 kernels).

> **AMENDMENT 2026-08-31 evening (operator, verbatim): "start with learning mastery first with
> kernels engineering curriculum we built on and then build kimik3 and while doing those
> execution we would implement main plan … I don't want to do directly without mastering from
> first principles scratch and fundamentals."** **This supersedes consequence (4) above.** The
> tracks INVERT: the **ladder session owns the day**, and the dated node (`oracle → E1 →
> review → E3`) is what rungs L0.1 → L1.3 → L2.1 → L2.3 → L2.4 → L2.5 *produce* — byproduct by
> the operator's explicit choice, not by drift. **Rule 4 keeps its teeth by MOVING, not by
> relaxing:** it is now scored on the session's **pushed** commit — a DoD unpushed at 21:00
> scores zero, three zeros still trip the breaker. Cost, stated once and accepted: the review
> lands ~9–12 sessions in rather than ~3.

## WHAT we are building — production artifacts, each with a named external consumer

1. **E1 — the instrument + divergence map** (CPU, $0). Sequential-vs-chunked divergence of
   the gated-delta-rule recurrence (`src/scratch_llm/mastery/`): fp64 oracle (5 lines,
   mine) → pre-registered predictions → sweep gate × dtype × chunk → public JSON + ledger
   rows. *Consumer: the review in item 2 — every claim in it cites this map.* **Claim
   wording (novelty kill-check 27/08, PARTIAL-survives): "first public MAP — gate × dtype
   × chunk vs fp64 oracle, pre-registered — of seq-vs-chunked GDR divergence"; cite
   neighbors proactively: CARVE arXiv:2606.27229 (chunk sweep of its own method, no
   dtype/fp64/gate axes) · arXiv:2606.06034 ((I+T)⁻¹ precision) · FLA #104/#389 ·
   #45819's in-thread probes.**
2. **The adjudicating review on vLLM PR #45819 vs PR #49827** (batch invariance for
   GDN_ATTN, issue #42960). The two open PRs encode competing hypotheses — reduction-order
   sensitivity (#45819's bs≈60–62 finding) vs 64-token chunk alignment (#49827) — and
   reviewers demanded determinism-suite evidence neither thread has. The map adjudicates.
   *Consumer: vLLM maintainers. The review window is open NOW — the only dated external
   event in this file.*
3. **E2 — RL-loop divergence on a production checkpoint, shipped as a `vllm-project/
   recipes` PR**: the existing `moonshotai/Kimi-Linear.md` recipe is BARE (zero measured
   numbers — an acknowledged gap rendering at recipes.vllm.ai). Ship the H200 TTFT/ITL/
   tok-s/$ profile + a determinism appendix (solo-vs-batched divergence, rollout-vs-HF
   logprob delta, IS-clip fraction). $16 smoke test decides shape; full sweep $120–200.
   *Consumers: the recipes repo (merged PR), the RL ecosystem (#41733, vllm-omni #4864 —
   note 26/08 correction: #48613 itself asks for the validation harness, not logprob
   metrics), applications.* Detail: spec §9.5.
4. **E3 — a PR to vLLM**: the regression harness in the official determinism suite (the
   #41292 diff-vs-reference methodology) and/or the residual fix the review reveals
   (cross-chunk state carry, fixed reduction widths). *Floor = open PR with maintainer
   engagement; target = merged.*
5. **Verifier**: `tolerances.yaml` derived only from measured envelopes — the
   anti-KernelBench artifact. *Consumer: our CI + upstream suites where accepted.*
6. **K2 · KDA reference (re-scoped 26/08 evening).** `core/kda.py`, Huy's hand under
   `k3/HANDCRAFTED.md` — kept because it **adjudicates production kernels** (three-path
   equivalence ≡ fp64 with Diag(α) · `fla.ops.kda` parity fwd+bwd = evidence with a
   consumer: the FLA/vLLM threads). Same five lines as E1's oracle, α → Diag(α) — one
   dependency chain, not two lanes. **K3–K10 RETURNED from backlog 31/08** (gated-MLA,
   AttnRes, SiTU/LatentMoE, mini-K3, 1M-ctx, self-hosting) — they now run as mastery-lane
   sessions (item 8), each carrying its own consumer; the 26/08 demotion is reverted, and
   the reversion's cost is the one named in the 31/08 header.
7. **FLA contributions (new lane, 26/08 evening).** The gate-at-0 soft spot (#104/#389 —
   closed unrooted; true magnitude 0.13 abs / 0.63% rel) characterized + regression test
   upstreamed to fla; KDA parity tests vs `fla/ops/kda/{naive,fused_recurrent,chunk}`.
   *Consumer: the kernel library Qwen/Kimi-class production models ship on.*
8. **The mastery lane — K3 + perf/kernel engineering (promoted to a named artifact
   31/08).** Previously scattered across the NVIDIA overlay N1–N6 and the rental table with
   no lane of its own, which is why it read as "learning". It is not: its outputs are a
   public NCU profile (gap 2), a CuTe-DSL re-expression (gaps 3+6), an NVFP4 divergence rung
   (gap 4), Blackwell wgmma→tcgen05 numbers (gap 5), and the K3 reference implementations
   that adjudicate the kernels above. *Consumers: NVIDIA JR2018988/JR2021962 · the FLA/vLLM
   threads · the interview loop (§11 gap 7).* **Delivered as L0–L5 sessions — spec §12.5.**

## WHY

- **After it ships (the value that exists in the world):** the first public divergence map +
  regression harness of the gated delta rule inside vLLM's determinism suite — protecting RL
  training of Qwen/Kimi-class hybrids for everyone — plus a measured serving recipe people
  copy, a measured-envelope verifier, and one engineer who owns that subsystem. (Chain + why-
  depth: CLAUDE.md North Star · the VN wall-map artifact, 5 min/morning.)
- **Science:** the GDN recurrence ships in production hybrids (Qwen3-Next/3.5/3.6, Kimi
  Linear); its two computational paths disagree on record — **0.13 abs / 0.63% rel at the
  pathological gate (FLA #389, closed without root cause; the "13%" previously here was a
  misreading of `o diff: 0.130267`, corrected 26/08)**; GRPO divides their logprobs; nobody
  has published the map; MiniMax reverted a whole architecture over exactly this trust gap.
- **Market:** root-cause + measurement = requirement #1 (81% of the 26/08 live N=21 scan;
  38/45 in the broader 08-14 scan); **correctness/numerics is now explicit JD language**
  (Anthropic Inference Systems: "correctness as an engineering discipline: numerics"); a
  measured vLLM contribution is the strongest no-PhD credential. **Spearheads: Anthropic
  (Perf-RL / Perf-GPU, $350–850k, visa sponsored) and NVIDIA (AI Inference Perf new-grad,
  $124–241k, no hard degree gate).** Behind them: TML, Together SG, Cohere, xAI.
- **Personal:** frontier seat ASAP. Quantified: Anthropic-tier comp ≈ 1–1.5 years to the
  family goal; VN salary ≈ never. This window is the only ASAP-consistent path.

## HOW — the schedule (production lanes, dependency order)

| Phase | Deliverable | Owner | Gate |
|---|---|---|---|
| **▶ 0 · the oracle** | `recurrent_reference` — 5 lines, fp64, from the equation | **Huy's hand** | `--self-test` PASS. Serves E1 and K2 both; clears the last pyright error → CI green here and nowhere earlier |
| ⑂ *(resolved → X, 26/08 evening — see header)* | `--run` + publish the map (+4 h) → E1 public, applications fire, vLLM lane opens, K3 gains a numerics CI | Huy | map public = E1's 8 atoms |
| **1 · review + claim** | the adjudicating review posted on #45819 (cross-ref #49827), every claim citing the public map; **same push: a claim-comment on #48613 with the E001-informed harness design** (it is open/unassigned and asks for exactly this; competitors are circling — a flashinfer-GDN-prefill commit is already in vLLM CI) | Huy writes; agents draft structure | maintainer engagement. **Clock: days, not weeks.** Evidence shape = spec §9.3. **Audience fact (27/08): the #45819 reviewer = yewentao256, who ALSO leads the official determinism suite (#27433) — E3's consumer is the same person; he is on record against AI-generated prose and vacuous passes, so the review = short, numbers-only, full-suite-aware** |
| **1a · the free half of the review (NEW 29/08 · claim CORRECTED 30/08 after the precondition ran)** | **The coverage-gap comment.** ⚠️ **The 29/08 wording was REFUTED by its own precondition check — do not post it.** Falsified as stated: "every green run tops out at bs=16 (`VLLM_NEEDLE_BATCH_SIZE=8` … `max_batch_size=8`)". First-party at vllm HEAD `cacc429` (28/08), `tests/v1/determinism/test_batch_invariance.py`: **L79 `max_batch_size = int(os.getenv("VLLM_NEEDLE_BATCH_SIZE", "128"))` — the default is 128, not 8**; other tests run `max_num_seqs=` 128 (L203), 32 (L484, L703), 1 (L417). The bs=8/16 figures were one contributor's *reported run*, not the suite default — conflating the two is exactly the error `yewentao256` is on record for rejecting. **SURVIVING, SHARPER CLAIM (verified, arithmetic):** L126 samples `batch_size = random.randint(max_batch_size // 2, max_batch_size)`, so at the default the needle test draws **only from [64, 128]** — its sampling *floor* steps over the one reported failure band (**bs≈60–62**, bfoing, H100/FP8, 26/06) rather than its ceiling falling short. The band is reachable only by setting `VLLM_NEEDLE_BATCH_SIZE` such that `[n//2, n]` covers it (e.g. `120` → `[60,120]`, or `62` → `[31,62]`); nothing in the suite or CI does. **Post the corrected version only.** Second free observation: the 28/08 serving benchmark shows an unremarked determinism tax — output throughput **1260.64 → 301.37 tok/s (4.18×)**, TPOT **3.14 → 545.39 ms (174×)** — posted with no attribution between `--enforce-eager`, the per-sequence loop, and the invariant ops. **Precondition before posting: read `tests/v1/determinism/utils.py` at HEAD and confirm the default batch bounds first-party** — the claim is arithmetic, so it must be exact. | Huy posts | a maintainer replies. This is the day's public atom on any day the oracle does not finish |
| **2 · E3** | regression harness → official determinism suite, and/or the residual fix | paired (sealed math his) | upstream CI green; floor = open PR, target = merged |
| **3 · K2 · KDA** | `core/kda.py` — Diag(α) · scaled sigmoid g_min=−5 · conv k=4 + Swish · L2Norm q/k · full-rank output gate · chunk-64 WY/UT | **Huy's hand** | chunkwise ≡ recurrent ≡ fp64-ref with per-channel α · `fla.ops.kda` parity fwd+bwd · A_log [128] semantics resolved BEFORE allocation |
| **4 · FLA lane** | gate-at-0 characterization + regression test PR; KDA parity tests | paired | merged test or maintainer-acknowledged issue |
| **5 · E2** | H200 day on Kimi-Linear-48B: logprob delta + IS-clip + TTFT/ITL/tok-s/$ | paired | numbers public + cited in a thread |
| **6 · verifier** | `tolerances.yaml` from measured envelopes only | Huy (sealed) | consumed by our CI; offered upstream |

**Hardware law (prices + ncu re-verified live 27/08):** rent, never buy. 5090 $0.33–0.35
(iterate; **ncu needs Vast KVM/vms_enabled — $0.326 — not default docker pods**) · H100
$2.50–3.90 ncu-capable (Hyperstack PCIe $2.50 floor; cheaper Vast containers are
ncu-blocked) · H200 $3.99 Hyperstack — **also ncu-capable, best single vendor for E2** ·
**B200 $4.6–6.1 (ncu path: Verda $6.11 / Lambda $6.99 VMs; a full FA4+DeepGEMM+KDA
profiling day ≈ $40–60)** · GB300 1× VM exists (Verda $8.62). ncu tiers: Crusoe
documented · Hyperstack/Verda(ex-DataCrunch)/Lambda/Nebius/TensorDock = VM-capable ·
**RunPod pods + Modal + Vast default docker = BLOCKED (ERR_NVGPUCTRPERM)**. **In-window
cap $1,500; warchest ≥$3,500 (sum = the $5,000 family envelope — split unchanged; the warchest is also reasoningLLM's SOLE GPU source per its ADR-0020: science band ≤$3.0K, conditional on measured $/step — GO / de-scope / defer-to-income; and the Blackwell day
fits inside the cap). Every rental names its measurement + consumer.** **Workspace zones (spec §9.6):** ours = this repo (`mastery/` · `experiments/`
· `results/` · `k3/core/` · `performance/rental/` · `deploy/`) · upstream forks =
`~/Desktop/oss/<repo>` (never nested here) · pods = ephemeral, bootstrap → measure →
artifacts return to `results/` + ledger → destroy.

**Arch-routing rule (added 30/08 — this is the law the ncu-debt misfile violated).** **Route every
profile/kernel rung by the KERNEL'S TARGET ARCH, never by the largest card you could rent.** A bigger
card is not a superset: a `-arch=sm_120` cubin/PTX **does not load or JIT on sm_90** (PTX compat is
forward-only), and a Triton kernel run on a different arch **recompiles** to different SASS and
different autotune configs — a different kernel instance, whose counters do not discharge a claim
made about the original. Consequence, measured 30/08: **4 of the 5 registered `ncu`-debt metrics in
`bench/` were filed against "the H100 day" and sat 57 days, when H100 could never have discharged
them.** They are sm_120 and cost **~$0.33/hr** on a `vms_enabled` KVM 5090 — available now. Only the
WGMMA tensor-pipe metric is genuinely Hopper; §4.1–4.6 (sm_90a) and tcgen05 (sm_100a) stay correctly
routed. Corollary — **same compute capability ≠ same card**: the ledger's sm120 rows are an RTX PRO
4000 Blackwell (**70 SMs**, 0.551 TB/s / 72.1 TF/s); a 5090 is ~170 SMs / ~1.8 TB/s. Counter claims
and the %-of-peak method transfer; **absolute rows do not — re-run R0 first to re-anchor peaks**, and
never write "same card". **First rental gate (the `vms_enabled ⇒ ncu` claim is still [INFERENCE]):**
minutes 0–10, profile a trivial kernel under `ncu`; on `ERR_NVGPUCTRPERM` you have root in a KVM →
`options nvidia NVreg_RestrictProfilingToAdminUsers=0` in modprobe.d + driver reload; if that fails,
destroy the instance and switch provider rather than spend the hour.

## MEASUREMENT — what a stranger can verify on Oct 16

① E1 + E2 public (JSON + ledger, predictions vs measured) · ② PR open (URL) · ③ the
adjudicating review live on #45819 (URL) · ④ ≥24 applications, ≥1 interview loop ·
⑤ ≥$1k income (Mercor/PI) · ⑥ every claim in the CV traces to a number in the repo.
*(All six reachable under (X); the fork's (Y)-cost note is discharged with the fork.)*

## RULES — everything that survived the simplification

1. **Evidence before claims.** Probe git/URLs; never trust memory or prior sessions.
2. **Predictions before measurements.** Ink first, then run. A result without a mechanism
   sentence doesn't count.
3. **The seal** (agents locked out, hook-enforced — unchanged by the redirect; it is exactly
   what makes the review defensible): `reference.py`'s body · kernel bodies under active
   study · RL loss math · verifier tolerances. Everything else — agents execute freely.
4. **One public change per day, pushed before 21:00.** The day scores Y/N on that alone.
5. **Applications fire the moment E1 is public.** No readiness debates. *(Live — fork
   resolved (X).)*
6. **This file only shrinks or updates in place.** New idea → one line in the backlog, or
   it dies. No new plan documents, no new trackers, no new boards.
7. **Ordering — production-first (amended 26/08 evening, operator directive quoted in the
   header; supersedes the same-day morning amendment's K3-outward ordering).** From-scratch
   happens at the layer under study **in service of a shipping artifact**: the oracle
   tonight (my hand) → the #45819/#49827 adjudicating review → the upstream harness/fix →
   K2's reference alongside → `chunked_wy` rebuilt from blank when its lane needs it. The
   morning amendment's dependency insight is retained (E1 and K2 share the five lines); its
   ordering is not. Full-stack rebuild-FIRST remains the named failure mode (Jul–Aug: ~87
   planning files, 0 kernels). Measurement is not what follows understanding — it is how
   understanding is tested into existence; the only falsifiable mastery metric is
   prediction error shrinking.

## THE DAY — one ladder (set 31/08 evening; inverts the two-track split of the same morning)

**The session owns the day.** One L-session from spec §12.5, run through an existing door
(`/master` deep-dive · `/rebuild` blank-slate · `/kernel-day` on a rental · `/feynman` as the
exit gate · `/op` wrapping) — never a new ceremony. **DoD is code or a measured number,
committed AND pushed before 21:00.**

**Rule 4 is scored on that push.** A pushed session commit IS a public change; a session that
ends in notes, or in a commit still sitting local, scores zero. Three zeros still trips the
circuit breaker. This one line is what keeps mastery-first from becoming Jul–Aug again.

**The dated node is the ladder's OUTPUT, not a parallel track.** #45819's window stays the only
dated external event in this file; if it closes before the ladder reaches L2.4, the map
retargets to #49827 or the successor thread — **the measurement does not expire, only the
venue does.**

**The rental queue is the lane's savings account.** CPU sessions *accumulate* measurements
that need silicon; a rented KVM hour discharges the whole queue from a pre-written script
(L3.3). The 57-day ncu-debt (4 of 5 metrics misrouted) is what happens without this discipline.

## TODAY (Mon 31/08)

**Oracle STUB · 3 PREDICTED unwritten · 0 result files · 4 commits AHEAD of `origin/main`,
unpushed → 3 days scoring zero under rule 4.** CI red is **exactly one error, measured 31/08**:
`pyright` → `test_e001_regression.py:71 — "NoReturn" is not iterable`. That is the stub, and
the five lines clear it. Hardware truth of 29/08 stands (no GPU, no CUDA toolchain, arm64;
`vastai` works, 5090 KVM `vms_enabled` ncu-capable $0.326/hr in stock) — CLAUDE.md carries it.

**Done 29/08, $0, no GPU — the L4 dispatch read** (spec §12.2 named it as its own test;
`~/Desktop/oss/{fla,vllm}` cloned per the §9.6 zone, fla @ `c3db408` HEAD):

- **F1 — our own claim refuted as stated and replaced by a sharper one.** No autotune key on
  the state/WY path contains batch or T (6 kernels tabulated in spec §12.2); `T` is
  de-specialized. Batch size reaches numerics through **one door**: the generic gate cumsum
  (`cumsum.py:29`, `key=['B',…]`), taken by default at `chunk.py:63` when
  `use_gate_in_kernel=False` → `B` changes the key → can change `num_warps` → changes the
  warp-shuffle **summation order of the log-gates** → changes `a_t`, hence `β_t/a_t` in the
  WY solve.
- **F2 — an unnamed precision knob on our exact H1 hazard.** `chunk_fwd.py:20-23` picks
  `tf32` vs `ieee` for the fused KKᵀ+`solve_tril` dot **at import time, by compute
  capability** (`_device.py:152`). 10-bit vs 24-bit mantissa on the most ill-conditioned
  step, invisible at the call site. Explains #45819's sm_120/sm_86/sm_90 split.
- **F3** vendor changes the num_warps search space (`wy_fast.py:26`).
- **F4** default `FLA_CACHE_MODE=DISABLED` → **live** autotune, and `fla/configs/` is not
  shipped → config chosen by runtime benchmark → run-to-run divergence with **no shape change
  at all**. Cheapest thing on this list to demonstrate.

Nothing here is quantified. Do not present F1–F4 as measured.

**The trunk (rule 7 unchanged; medium changed 29/08 — predictions now live in code, not on
paper):**

```
1 [Huy]  5 lines of recurrent_reference  +  3 PREDICTED values   ← ONE commit
         (tests/test_e001_regression.py::PREDICTED — git's timestamp IS the
          pre-registration; MASTERY_LEDGER's 1e-__ table is retired)
2 [agt]  python -m experiments.e001_gate_sweep --self-test
3 [agt]  commit + push → CI GREEN (the stub was the only error) → public change
4 [agt]  --run → results/e001_gate_sweep.json
5 [Huy]  BASELINE ×3 + TOL_RATIO + the one-line tolerance argument
6 [agt]  push → regression gate live = E1 PUBLIC → rule 5 fires
7 [Huy]  mechanism + what-breaks-at-10× → the #45819 review, now citing F1–F4
```

Calibration for step 1: the literature's worst-gate anchor is 0.63% rel on THEIR shapes;
yours are T=512, d=64.

**Kernel-engineering curriculum — live state (spec §12.3 owns the 1:1 map; this is its
status line).** Binding rule stands: *a unit with no production node attached does not run.*

| $0, runnable now | Node | State |
|---|---|---|
| L4 dispatch read | spec §12.2 test | **✅ done today (F1–F4)** |
| L1 roofline predict-then-measure | Row 002 / E002 | needs a GPU to have a ridge point — moved out of E001 29/08 |
| DD7 reverse-engineering loop | FA4/DeepGEMM source read | 🎯 open, $0, no GPU |
| DD4 land inside vLLM | the review | **▶ blocked only on step 7** |

| Rental-gated (was wrongly marked "$0 on the standing card") | Needs |
|---|---|
| N2 CuTe-DSL · N3 NVFP4 · Proton lab · GEMM ≥90% · self-sabotage drill | any ncu-capable card (5090 KVM $0.326/hr) |
| F2 tf32-vs-ieee on silicon · F4 live-autotune run-to-run | sm_80+ · any GPU |
| FA3/TMA/wgmma · FA4/tcgen05 | H100 half-day · B200 day $40–60 |

**The rental now has a named measurement (law: "every rental names its measurement +
consumer"): F2 and F4, consumer = the #45819 review.** It is authorized by operator word,
not by this line. F2 is *also* priceable on CPU for $0 — `paths.py` already exposes
`solve_dtype`; emulating tf32 is truncating the fp32 mantissa to 10 bits (~6 lines, sealed
harness, Huy's hand).

**Fallback public atom, $0, no oracle:** phase **1a** above. Wall: 21:00.

```
git add -A && git commit -m "docs: cut noise from the constitution; L4 dispatch read (F1-F4); predictions move to code"
git push origin main        # ← 15 days no public change; CI red since 03/08
```

## NVIDIA TARGET — fastest path

Depth + provenance: `docs/KERNEL_MASTERY_SPEC.md` **§11** (27/08 · 10 live JDs pulled full-text
from NVIDIA's eightfold API · 37 CONFIRMED / 5 CORRECTED / 0 REFUTED). **This section is ORDERING
ONLY — it adds no work.** Every item below is an existing lane's output, relabelled for one
consumer. All ten live reqs carry **"(or equivalent experience)"**: there is no degree gate.

| Req | Role | Gate | Why this one, given what is actually being built |
|---|---|---|---|
| **JR2020181** | Sr DL Performance Architect | 4+ yrs | analytical perf modeling + profiling — **predict-the-number IS the job description**; lowest senior bar of the ten |
| **JR2018988** | Sr SWE, CUTLASS Kernels (Math Libs) | **3+ yrs** | lowest gate anywhere on the board; wants Tensor-Core kernels in CUTLASS C++ **and Python DSL** for Blackwell/Rubin + "open-source contributions to math kernel libraries" |
| **JR2021962** | Sr Inference Eng, GPU Kernel Optimization | 6+ yrs | the bullseye: agentic kernel optimization at the assembly layer + silicon-measured benchmarking + vLLM + FlashInfer/Triton/CUTLASS contributions = this repo's architecture, named. **Apply over-gate.** |

**Missing vs those JDs right now** (axes + frequencies = spec §11.2):

1. **Zero merged OSS contributions** — named as the stand-out in 6/10 posts, which name our exact
   repos (FlashInfer · Triton · CUTLASS · vLLM · TRT-LLM · SGLang). **The only blocking gap, and
   lanes 1–4 above already close it.** Nothing new is required to fix this.
2. **No public NCU/NSYS profile** — profiling is the #1 axis (10/10). E1 is CPU-only. **Re-dated
   30/08: this no longer waits for E2/N5.** The 4 sm_120 ncu-debt metrics registered in `bench/`
   since 04/07 discharge on a **$0.33/hr KVM 5090** (arch-routing rule above) — so the first public
   ncu profile is a **~$1, one-session item**, and it is the cheapest gap on this list to close.
3. **No CUTLASS/CuTe-DSL artifact** (5/10 want DSL kernel authoring) → **N2**, on the 5090 KVM
   (~$0.33/hr — the earlier "$0" assumed the standing card that does not exist).
4. **No low-precision-numerics artifact** (4/10; NVFP4 is NVIDIA's strategic bet) → **N3**, same card.
5. **No Blackwell silicon numbers** — the wgmma→tcgen05 chasm is the 2026 differentiator → **N5**.
6. **PTX/SASS reading fluency** (3/10, the kernel-core three) — nothing owned; rides N2.
7. **Interview reps unrehearsed.** The loop is take-homes + "how did you know it was
   bandwidth-bound?" probes, never LeetCode; the documented rejection is *"optimized by trial and
   error rather than by rigorous measurement."* The ledger discipline is the counter-training —
   but it has to be narratable cold.

**30 / 60 / 90 — existing lanes, labelled for this consumer; ordering unchanged (rule 7)**

- **0–30d · close the one blocking gap.** Oracle (5 lines) → E1 map public → the adjudicating
  review on #45819 + the #48613 claim-comment. **Rule 5 fires the day E1 is public: apply to all
  three reqs above** — the review URL is the cover letter. Over-gate on years, never on evidence.
- **30–60d · convert the review into a merged artifact + the two free rungs.** E3 (regression
  harness into the official determinism suite) — a merged vLLM PR closes gap 1 outright. In
  parallel, both **rental-gated** (there is no standing card — see TODAY): **N2** CuTe-DSL re-expression of one owned kernel
  (gaps 3 + 6) and **N3** NVFP4 block-scaling divergence (gap 4). E2's H200 day supplies the
  first real profile (gap 2).
- **60–90d · Blackwell + the public scoreboard.** **N5** B200 evidence day (gap 5) → narrate
  wgmma→tcgen05 with measured numbers on both sides. **N4** MLSys 2026 FlashInfer contest,
  NVIDIA Track — agents generate, human verifies on silicon, scored in public: that is
  JR2021962's job spec, entered rather than asserted. Batch-2 (new-grad/intern DevTech) fires
  on E002 per the backlog. Interview reps run in the evening lane: GEMM ladder cold, coalescing
  narrated to an Nsight trace, one 48 h take-home rehearsed end to end.

**Kill criterion.** If E1 + the review are public and 30 days pass with zero recruiter contact,
the bottleneck is the résumé surface, not the artifact: spend ONE day on the public write-up
(blog + a FlashInfer-visible thread) — not on a fourth application batch.

## BACKLOG (one-liners; undated; no ceremony)

- ~~K3–K10 rebuild rungs~~ — **left the backlog 31/08** (operator amendment); they are the
  mastery lane now, item 8, sequenced in spec §12.5. M0→M6 rides with them.
- E002 cross-arch · rebuild `chunked_wy` from blank (my hand, when its lane needs it)
- **Blackwell evidence day (B200 1×, Verda VM $6.11 ncu-capable, ≈$40–60/day):** FA4 +
  DeepGEMM-SM100 + fla-KDA + tcgen05/TMEM numbers — all four verified runnable on one
  rented B200 (CUDA 12.9+); dead on sm_120 anyway (no TMEM) — and there is no sm_120 card
  here either (TODAY, 29/08). Hiring verdict
  27/08: Blackwell artifacts = the 2026 differentiator, Hopper = the assumed substrate;
  the winning portfolio narrates the wgmma→tcgen05 delta with numbers on BOTH. **Fires
  after E1 + the review land — operator word; no external clock.** Rubin = watch-item
  only (sm_107 needs R615 driver, unrentable 2026; read CUTLASS 4.8 examples, $0)
- **Rental-day consolidation (perf track × this plan — one purchase, two DoDs, 27/08; re-routed
  30/08):** **the ncu-debt no longer rides any of these days** — it is sm_120 and discharges on a
  $0.33/hr KVM 5090 (arch-routing rule), which is also the card E001's GPU cells and the F1/F4
  discharges want: **one ~$1 session serves the review AND the ledger debt.** What genuinely remains
  bundled: the perf curriculum's Hopper/Blackwell days — the H100 wgmma block
  (~$25–30, half day: A2 §4.1–4.6 · A3 R3–4 · A4 R4/FA3) rides as PRELUDE to N5's B200 day
  (the wgmma→tcgen05 narration needs measured numbers on BOTH); A-series H200-runnable debt
  banks into E2's H200 day; **A6's 8×H200 serving day stays PARKED** (≈$250+, weak
  post-redirect consumer). No new spend authorized by this line.
- MI300X ROCm day if slack (Crusoe $3.45; ~$50)
- AI-Infra hackathon: automated gate-check fires 04/09; ignore otherwise
- **WAFER — new lane, verified 29/08 (highest EV-per-hour item found this month).** YC S25,
  $4M seed (Fifty Years; angels **Jeff Dean**, **Woj Zaremba**), ~6 people, SF. Product =
  *autonomous agents that profile, diagnose and optimize GPU inference, kernels → production
  pipelines* — **their company thesis IS this repo's architecture** (agent writes, human
  verifies on silicon, verifier is the moat). Two surfaces, both live: **(a) `wafer-ai/
  gpu-perf-engineering-resources`** — 2.2k★, 240 forks, MIT, CONTRIBUTING.md, PRs accepted,
  maintained by **emilio@wafer.ai (the founder)**, "last verified 2026-08-23"; its stated
  source policy is *our* §9.3 standard verbatim ("performance claims require hardware
  details, workload specs, precision, baselines, and correctness methods"), and its 8-section
  taxonomy has **no determinism/batch-invariance section** — a named, fillable gap. A PR
  adding that section (TML post · vLLM #27433 · SGLang deterministic inference ·
  `batch_invariant_ops` · FLA #104/#389 · our map once public) is **the cheapest first merged
  OSS contribution available to this profile: $0, days, read by the founder.** **(b)** MTS +
  intern roles on the YC board (job page 404'd to unauthenticated fetch 29/08 — **terms,
  comp, remote policy and visa stance UNVERIFIED; confirm before applying**). Fires with the
  E1-public batch; the resources PR can fire earlier since it needs no measurement.
- **Application adds (27/08, primary-source verified):** Inferact remote MTS (vLLM's
  creators, $150M seed; "contributions to vLLM" = preferred qualification — the most
  aligned consumer of the E1→review→PR chain) fires with the E1-public batch · NVIDIA
  Vietnam R&D center = in-country channel, NO visa, same batch · Anthropic Fellows =
  VN-INELIGIBLE (US/UK/CA work-auth wall, no fellow visas) — do not spend an application ·
  OpenAI Residency 2026 closed. Detail: memory `income-channels-verified-2026-08-27`
- Batch-2 applications (NVIDIA new-grad/intern DevTech, DeepMind, Fireworks, Baseten…) when
  **E002** is public — distinct from the three senior NVIDIA reqs, which fire earlier, at **E1**
  public, per `## NVIDIA TARGET` above (bar reverse-engineered 27/08; detail in spec §11)
- Vietnamese wall-map — SHIPPED 27/08 as artifact "Từ Năm Dòng Code"
  (https://claude.ai/code/artifact/e3fc28db-aacd-4c84-99eb-39f6c131c756); refresh after E1/E2 land
- **sm_120 NVFP4 lane** (flashinfer #2577 silent-zeros repro on the local card, $0; vLLM
  #31085/#47749) — parallel deep lane once the harness claim is in; spec §9.5
- GPU MODE KernelBot (free compute; no live cash purse 27/08) · **MLSys 2026 FlashInfer
  contest ENDED 24/04, awards 22/05 — stale "live window" fixed 27/08**; GDN-track winner
  repo public (romitjain/kachua-mlsys) = study target · tinygrad bounties reopened
  ~mid-08, 4 open ($200–2k class, "the only path to a job here") · Mercor CUDA $300/task,
  VN-payable (entry = résumé + assessment, NOT merged PRs — corrected 27/08) · Prime
  Intellect env bounties $100–500 open-access / $1k–5k application tier
- **NVIDIA overlay rungs N2/N3 ($0, sm_120):** CuTe-DSL/cuTile re-expression of one owned kernel +
  NVFP4 block-scaling divergence rung — earns the CUTLASS-JD "Python DSL"/"cuTile" axis; spec §11.8
- Fresh adjacent numerics surfaces when the map is public: vLLM #51562 (GDN first-chunk
  metadata, 0 comments) · #49918 (cudagraph skips GDN state write) · sglang #31720
  (cross-ENGINE divergence, same ckpt clean on vLLM) · #35150 · unowned batch-invariant
  FlashInfer GDN prefill (spans 3 repos) — full map: spec §9.5
- **Curriculum binding (29/08, operator-set):** the Vizuara ladder is the **delivery order for
  lessons**, never for work — spec §12 maps its 6 parts × 8 lectures × 8 deep-dives onto the
  lanes above, and every unit either (a) fires on a live production node, (b) is already
  receipted, or (c) is fenced by hardware. It adds **no rung and no date** to this file. The
  binding rule: *a curriculum unit with no production node attached does not run.*
  **Amended 31/08:** the rule stands, and K3 + perf/kernel now *satisfy* it as item 8 —
  §12.5 attaches a consumer and an owner to every session, so the ladder runs rather than
  waits. Vizuara still supplies no ordering and no dates.
- Kernel-path map + production operating system: `docs/KERNEL_MASTERY_SPEC.md` **v2+§9/§10/§11**
  (26/08 evening — Vizuara ×5, skip verdict ×5, live N=21 market scan, 2026 stack ledger,
  corrections §4, curriculum spine §6, lesson ladder §7, **§9 = daily loop as practiced ·
  variance-control regimen · landed-PR evidence standard · hired-from-OSS pattern · live
  surface map · sequencing**; **§11 = the NVIDIA lane, 27/08 — 10 live JDs full-text ·
  axes · IC-ladder/comp · interview loop · PhD escape hatch · OSS pipeline · overlay N1–N6**)
