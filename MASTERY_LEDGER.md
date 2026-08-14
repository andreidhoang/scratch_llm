# LEDGER

One row per experiment. Four numbers plus a **mechanism**. A row without a mechanism is
incomplete and does not count. `TRACED` (I followed it) is not `DEFENDED` (I can rebuild it
from blank, predict its number, defend every choice against "why not X", and say what breaks
at 10×).

`pred err %` = `|predicted − measured| / measured`. Track it as a **time series**. If it is
not shrinking, mastery is not accruing no matter how much shipped — that is the single
falsifiable mastery metric here, and it is a diagnostic, never a target.

---

## Row 000 — derivations, before any code

*Fill these on paper before writing `reference.py`. Editor closed.*

**D1. Arithmetic intensity of the recurrent path at decode (T=1), per token per head.**
State is `(dk, dv)`. Count FLOPs and count bytes moved. Which wall are you under?

- FLOPs/token: `________`
- Bytes/token (bf16 state): `________`
- Arithmetic intensity (FLOP/byte): `________`
- Ridge point of the target GPU: `________`  → **which wall:** `________`
- Mechanism (one sentence, why): `________`

**D2. Why can the recurrent loop not be used for training?**
Answer in terms of the dependency chain, not "it is slow."

- `________`

**D3. What does the chunked form buy, and what does it cost?**
Name the operation that appears in the chunked form and appears nowhere in the recurrent
form. That operation is the entire subject of this repo.

- Buys: `________`
- Costs: `________`

---

## Row 001 — E001 gate sweep (CPU, pure PyTorch)

**Predictions — written BEFORE `--run`. No prediction, no run.**

| | Predicted | Measured | pred err % |
|---|---|---|---|
| float64 rel err at `log_gate = 0` | `1e-__` | | |
| bfloat16 rel err at `log_gate = 0` | `1e-__` | | |
| bfloat16 rel err at `log_gate = -1.0` | `1e-__` | | |

**Hypothesis I expect to survive:** `H_` — because: `________`

**Falsifier I accept in advance:** if `________` then my mechanism is wrong.

*(after the run)*

- Measured verdict: `________`
- **Mechanism** — why, at the level of the arithmetic: `________`
- What surprised me: `________`
- What breaks at 10× (seq len? head dim? chunk size?): `________`
- Delegation level: `L_` · delegated: `________` · refused: `________`
- Provenance: torch `____`, device `____`, seed `____`, solve_dtype `____`, commit `____`
- Next experiment this implies: `________`

---

## Row 002 — E002 real kernels on silicon

*Only after E001. The CPU result tells you what to expect; the GPU result tells you whether
FLA's actual kernels behave like the algebra says they should. If they diverge from the
pure-PyTorch prediction, the difference is the kernel's own choices — and that difference is
a finding.*

**Prediction before renting anything:** `________`
