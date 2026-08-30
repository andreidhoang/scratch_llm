# LEDGER

One row per experiment. The row is raw material for a public artifact — the #45819 review, the
PR description, the regression gate. A row without a **mechanism** is incomplete and does not
count. `TRACED` (I followed it) is not `DEFENDED` (I can rebuild it from blank, predict its
number, defend every choice against "why not X", and say what breaks at 10×).

`pred err %` = `|predicted − measured| / measured`. Track it as a **time series**. If it is not
shrinking, mastery is not accruing no matter how much shipped — the single falsifiable mastery
metric here, and a diagnostic, never a target.

**Where predictions live (changed 2026-08-29).** Not in this file. A prediction is
pre-registered by being **committed before the run**, in the code that consumes it:
`tests/test_e001_regression.py` → `PREDICTED` (yours, before `--run`) and `BASELINE` (yours,
from the run). Git's timestamp is the pre-registration. This file records the **mechanism** —
the part no dict can hold.

---

## Row 001 — E001 gate sweep (CPU, pure PyTorch)

**Predictions:** `tests/test_e001_regression.py::PREDICTED`, committed before `--run`.
**Measured:** `results/e001_gate_sweep.json` + `::BASELINE`.
**pred err %:** computed from those two — not transcribed here.

*(after the run — this is the review draft, write it as if a vLLM reviewer will read it)*

- Verdict: `________`
- **Mechanism** — why, at the level of the arithmetic: `________`
- What surprised me: `________`
- What breaks at 10× (seq len? head dim? chunk size?): `________`
- Tolerance argument — smallest drift CI should shout about, and absolute band vs ratio, in one
  line (this IS the verifier-design skill; it is what `TOL_RATIO` encodes): `________`
- Provenance: torch `____`, device `____`, seed `____`, solve_dtype `____`, commit `____`
- Next experiment this implies: `________`

---

## Row 002 — E002 real kernels on silicon

*Only after E001. The CPU result says what to expect; the GPU result says whether FLA's actual
kernels behave like the algebra does. Divergence from the pure-PyTorch prediction is a finding.*

**Roofline (moved here 2026-08-29 — it is a GPU question; E001 is CPU and does not ask it).**
Recurrent path at decode, T=1, per token per head; state is `(dk, dv)`.

- Byte-boundary chosen (state only / +weights / +activations) and why: `________`
- FLOPs/token · bytes/token (bf16) · arithmetic intensity: `________`
- Ridge point of the target GPU → which wall: `________`
- Mechanism (one sentence): `________`

**Prediction before renting anything:** `________`
