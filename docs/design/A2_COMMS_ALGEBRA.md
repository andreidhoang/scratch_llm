# A2 §8 — Communication algebra: the parallelism-scaling written answers

Answers to the official A2 problems `alternate_ring_all_reduce`, `data_parallel_calcs`,
`fsdp_calcs`, `tp_calcs`, and `fsdp_tp_calcs`, in the problems' own shape and notation.
**Every number in this document is produced by `src/scratch_llm/utils/comms_calc.py`** (each
formula's derivation lives in its function docstring) **and locked as a regression by
`tests/test_comms_calc.py::test_xl_worked_example_regression`** — none is typed by hand.

**Setting (official §8):** $N$ devices, each with egress bandwidth $W$ bytes/s and accelerator
speed $C$ FLOP/s; compute and communication overlap, so a step is *communication-bound* when comm
time exceeds compute time. All weights/activations FP16 (2 bytes). The layer under analysis is the
gated FFN: $x\,(B,D)$; $W_1, W_2\,(D, D_{ff})$; $W_3\,(D_{ff}, D)$ — $3DD_{ff}$ parameters,
forward $6BDD_{ff}$ FLOPs (three matmuls), backward $12BDD_{ff}$ FLOPs (six matmuls).

Ring collectives (the primitives everything reduces to, `ring_allreduce_bytes` /
`all_gather_bytes` / `reduce_scatter_bytes`):

| collective | per-device bytes | time (bandwidth term) |
|---|---|---|
| all-gather | $\frac{N-1}{N}S$ | $\frac{N-1}{N}\frac{S}{W}$ |
| reduce-scatter | $\frac{N-1}{N}S$ | $\frac{N-1}{N}\frac{S}{W}$ |
| all-reduce (= RS + AG) | $2\frac{N-1}{N}S$ | $2\frac{N-1}{N}\frac{S}{W}$ |

As $N\to\infty$ the all-reduce cost saturates at $2S/W$ — per-device wire volume is bounded no
matter the world size (tested: `test_ring_limit_is_two_s`). Below we use the large-$N$
approximation $\frac{N-1}{N}\approx 1$, as the official solutions do.

---

## Problem (alternate_ring_all_reduce)

**Answer: $(N-1)\frac{S}{W}$ seconds** (`alternate_ring_allreduce_time`).

Justification: the alternate algorithm circulates *whole tensors* — each of the $N-1$ steps sends
the full $S$ bytes at egress bandwidth $W$ — instead of $S/N$-byte chunks, so it is a factor
$\frac{(N-1)S}{2\frac{N-1}{N}S} = \frac{N}{2}$ slower than the standard reduce-scatter +
all-gather ring (tested: `test_alternate_ring_is_w_over_two_slower`).

---

## Problem (data_parallel_calcs)

**(a) Backward FLOPs with $N_{DP}$:** $\dfrac{12BDD_{ff}}{N_{DP}}$ per device
(`ffn_bwd_flops(tokens=B/N_DP, …)`).

Justification: the backward is six matmuls of $2 \cdot \frac{B}{N_{DP}} \cdot D \cdot D_{ff}$
FLOPs each ($dz$, the two $dx$ contributions, and the three weight gradients), on the
batch-sharded $B/N_{DP}$ rows.

**(b) Backward communication time:** one ring all-reduce of the three weight gradients,
$S = 2\cdot 3DD_{ff} = 6DD_{ff}$ bytes, so
$2\frac{N_{DP}-1}{N_{DP}}\frac{6DD_{ff}}{W} \approx \dfrac{12DD_{ff}}{W}$.

Justification: weight-gradient size is independent of the batch shard, so DP's wire volume does
not shrink as devices are added — only compute does.

**(c) Scaling bound:** comm ≤ compute $\iff \frac{12DD_{ff}}{W} \le \frac{12BDD_{ff}}{N_{DP}C}$

$$\boxed{N_{DP} \le \frac{BW}{C}}\qquad(\texttt{dp\_max\_world})$$

Justification: $D$ and $D_{ff}$ cancel — the DP ceiling is set purely by tokens-per-step times
the interconnect-to-compute ratio of the hardware.

---

## Problem (fsdp_calcs)

**(a) FLOPs:** backward $\dfrac{12BDD_{ff}}{N_{FSDP}}$, forward $\dfrac{6BDD_{ff}}{N_{FSDP}}$ —
identical to DP.

Justification: sharding the *weights* changes where bytes live, not how many FLOPs the
batch-sharded forward/backward performs.

**(b) Communication time:** backward = 3 all-gathers (re-materialize $W_{1..3}$) + 3
reduce-scatters (gradients to shard owners) $= 2\frac{N-1}{N}\frac{6DD_{ff}}{W} \approx
\dfrac{12DD_{ff}}{W}$; forward = 3 all-gathers $\approx \dfrac{6DD_{ff}}{W}$.

Justification: an all-gather and a reduce-scatter each move $\frac{N-1}{N}S$ per device — the
backward's AG+RS pair costs exactly what DP's single all-reduce costs, and the forward adds one
more AG that DP doesn't have (per-step wire ratio FSDP:DP = 3:2 exactly, at any $N$ — tested:
`test_scheme_byte_ratios_match_closed_forms`).

**(c) Scaling bounds:** backward $\frac{12DD_{ff}}{W} \le \frac{12BDD_{ff}}{N C}$ and forward
$\frac{6DD_{ff}}{W} \le \frac{6BDD_{ff}}{N C}$ give the same bound:

$$\boxed{N_{FSDP} \le \frac{BW}{C}}\qquad(\texttt{fsdp\_max\_world})$$

Justification: each pass's extra communication is matched by that pass's FLOPs, so FSDP buys a
$N\times$ memory reduction at 1.5× wire bytes with **the same scaling ceiling as DP** — memory is
why you choose it, not throughput.

---

## Problem (tp_calcs)

**(a) Backward pass** (column-parallel $W_1^{(i)}, W_2^{(i)}\,(D, \frac{D_{ff}}{N_{TP}})$,
row-parallel $W_3^{(i)}\,(\frac{D_{ff}}{N_{TP}}, D)$; every device already holds the full $dy$
because $y$ was all-reduced in the forward):

$$
\begin{aligned}
dz^{(i)} &= dy\, W_3^{(i)\top} &&(B, \tfrac{D_{ff}}{N_{TP}})\\
dx_2^{(i)} &= dz^{(i)} * f(x_1^{(i)})\\
dx_1^{(i)} &= dz^{(i)} * f'(x_1^{(i)}) * x_2^{(i)}\\
dW_3^{(i)} &= z^{(i)\top} dy\\
dW_2^{(i)} &= x^\top dx_2^{(i)}\\
dW_1^{(i)} &= x^\top dx_1^{(i)}\\
dx &= \text{all-reduce}\big(\{\,dx_1^{(i)} W_1^{(i)\top} + dx_2^{(i)} W_2^{(i)\top}\}_{i=0}^{N_{TP}-1}\big)
\end{aligned}
$$

One collective total: the all-reduce of the partial $dx$ contributions (each device's slice of
$D_{ff}$ contributes a partial sum to the full $(B, D)$ input gradient).

**(b) FLOPs:** forward $\dfrac{6BDD_{ff}}{N_{TP}}$, backward $\dfrac{12BDD_{ff}}{N_{TP}}$.

Justification: every matmul has its $D_{ff}$ dimension sharded $N_{TP}$ ways, so per-device FLOPs
divide evenly while total FLOPs are unchanged.

**(c) Communication time:** forward = one all-reduce of $y\,(B,D)$ in FP16, $S = 2BD$, so
$2\frac{N-1}{N}\frac{2BD}{W} \approx \dfrac{4BD}{W}$; backward = one all-reduce of the $dx$
partials, the same $\approx \dfrac{4BD}{W}$.

Justification: TP communicates **activations, not weights** — wire bytes scale with $B$ and $D$
but not with $D_{ff}$ or $N_{TP}$.

**(d) Scaling bounds:** backward $\frac{4BD}{W} \le \frac{12BDD_{ff}}{N_{TP}C} \iff N_{TP} \le
3\frac{D_{ff}W}{C}$; forward (the binding one, half the FLOPs against the same bytes):

$$\boxed{N_{TP} \le \frac{3}{2}\frac{D_{ff}W}{C}}\qquad
(\texttt{tp\_max\_world\_fwd}, \texttt{tp\_max\_world\_bwd})$$

Justification: $B$ cancels — TP's ceiling is set by model *width*, which is why TP stays at ~8
inside the NVLink island regardless of job size, and why it is on the critical path (its
all-reduce cannot hide behind another shard's compute the way DP's gradient all-reduce hides
behind the backward).

---

## Problem (fsdp_tp_calcs)

**(a) Forward FLOPs:** $\dfrac{6BDD_{ff}}{N_{FSDP}N_{TP}} = \dfrac{6BDD_{ff}}{N}$.

Justification: FSDP shards the batch $N_{FSDP}$ ways and TP shards $D_{ff}$ $N_{TP}$ ways, so the
three matmuls' FLOPs divide by the full grid size $N$.

**(b) Forward communication time (axes overlapped):** FSDP axis all-gathers this TP rank's weight
slice ($\frac{6DD_{ff}}{N_{TP}}$ bytes in FP16) and the TP axis all-reduces the batch-sharded
output ($2\frac{B}{N_{FSDP}}D$ bytes):

$$\max\left(\frac{6DD_{ff}}{N_{TP}W},\ \frac{4BD}{N_{FSDP}W}\right)$$

Justification: each axis's collective operates on tensors already sharded by the *other* axis
(weights by TP, activations by FSDP), so growing either axis shrinks the other axis's traffic —
that coupling is the whole point of 2D.

**(c) Scaling bound, overlapped:** each branch of the max gives an independent bound —
$\frac{6DD_{ff}}{N_{TP}W} \le \frac{6BDD_{ff}}{NC} \Rightarrow N_{FSDP} \le \frac{BW}{C}$ and
$\frac{4BD}{N_{FSDP}W} \le \frac{6BDD_{ff}}{NC} \Rightarrow N_{TP} \le \frac{3}{2}\frac{D_{ff}W}{C}$
— and they multiply:

$$\boxed{N \le \frac{3}{2}\,B\,D_{ff}\left(\frac{W}{C}\right)^{2}}\qquad
(\texttt{fsdp\_tp\_max\_world(overlapped=True)})$$

Justification: the two 1D ceilings are attained simultaneously at
$N_{FSDP}^\* = BW/C$, $N_{TP}^\* = \frac{3}{2}D_{ff}W/C$ (which also equalizes the two branches of
the max), so 2D parallelism scales to the *product* of what either strategy reaches alone
(tested: `test_closed_form_bound_relationships`).

**(d) Scaling bound, not overlapped:** comm = the *sum* $\frac{6DD_{ff}}{N_{TP}W} +
\frac{4BD}{N_{FSDP}W}$. For fixed $N$ the sum is minimized at
$N_{TP}^\* = \sqrt{3D_{ff}N/2B}$ (the two terms equalize), giving comm
$= \frac{8BD}{W}\sqrt{\frac{3D_{ff}}{2BN}}$; setting that $\le \frac{6BDD_{ff}}{NC}$ and solving:

$$\boxed{N \le \frac{3}{8}\,B\,D_{ff}\left(\frac{W}{C}\right)^{2}}$$

Justification: serializing the two axes costs exactly a factor 4 in the ceiling versus (c) — the
sum doubles the effective comm time and the bound is quadratic in $W/C$.

---

## Worked example — official "xl" (Table 1: $D=2560$, $D_{ff}=10240$, $L=32$)

Hardware model: $C = 10^{15}$ FLOP/s (~1 PFLOP/s dense BF16), NVLink-class egress
$W = 400\,$GB/s vs. IB-class $W = 50\,$GB/s. Global batch $B = 65{,}536$ tokens/step
(128 seqs × 512). All values below computed by the named functions.

**Maximum world size before communication-bound (per-FFN-layer algebra):**

| bound | formula | $W=400$ GB/s | $W=50$ GB/s |
|---|---|---|---|
| DP / ZeRO-1 (`dp_max_world`) | $BW/C$ | **26.2** | 3.3 |
| FSDP (`fsdp_max_world`) | $BW/C$ | **26.2** | 3.3 |
| TP forward (`tp_max_world_fwd`) | $\frac{3}{2}D_{ff}W/C$ | **6.1** | 0.77 |
| TP backward (`tp_max_world_bwd`) | $3D_{ff}W/C$ | 12.3 | 1.5 |
| 2D overlapped (`fsdp_tp_max_world`) | $\frac{3}{2}BD_{ff}(W/C)^2$ | **161.1** | 2.5 |
| 2D sequential | $\frac{3}{8}BD_{ff}(W/C)^2$ | 40.3 | — |

The exact solver (`comms_bound_world_size`, no $\frac{N-1}{N}\to 1$ approximation) puts the first
communication-bound DP world size at **28** for the 400 GB/s column (closed form 26.2 + the
$\frac{N-1}{N}$ slack), and `crossover_link_bandwidth` says holding 64 DP ranks compute-bound at
this batch would need **~961 GB/s** egress — beyond any current single-NIC fabric, which is the
same fact as the DP ceiling.

**Per-device wire bytes per step, whole xl model** (non-embedding params
`transformer_nonembed_params(32, 2560, 10240)` = 3.36 B; BF16 grads/params; world = 8):

| scheme | function | bytes/step/device | vs DDP |
|---|---|---|---|
| DDP (all-reduce grads) | `ddp_step_bytes` | 11.74 GB (10.94 GiB) | 1× |
| ZeRO-1 (RS grads + AG params) | `zero1_step_bytes` | 11.74 GB | **1× — memory saving is comms-free** |
| FSDP (AG fwd + AG bwd + RS grads) | `fsdp_step_bytes` | 17.62 GB (16.41 GiB) | 1.5× |
| TP activations (4 AR/block, batch 8 × seq 512) | `tp_step_bytes` | 4.70 GB (4.38 GiB) | — (scales with $B{\cdot}s{\cdot}D$, not params) |

---

## When does scaling stop?

Scaling stops when the network, not the accelerator, sets the step time — and each strategy hits
that wall on a different axis. Data parallelism (and ZeRO-1, and FSDP, which moves 1.5× the bytes
but earns no lower ceiling) dies at $N \approx BW/C$: its wire volume is a fixed ~2–3 copies of
the gradients per step, so once the per-device token share $B/N$ gets small enough, there isn't
enough compute left to hide them — the batch size is the budget, and the critical batch size caps
it. Tensor parallelism dies at $N_{TP} \approx \frac{3}{2}D_{ff}W/C$: it ships activations on the
critical path, its ceiling is set by model width alone, and at realistic $W/C$ (~4×10⁻⁴ for the
xl example: 6-way) that means TP stays inside the NVLink island. Combining them multiplies the
two ceilings — $N \le \frac{3}{2}BD_{ff}(W/C)^2$ with overlapped axes, a quarter of that without
overlap — which is why 2D (and 3D, with pipeline) parallelism exists at all. But the closed form
also says what *cannot* be engineered around: with $B$ pinned by the critical batch size and
$D_{ff}$ pinned by scaling laws, the only remaining lever is the hardware ratio $W/C$, and it
enters *squared*. When the fleet outgrows $\frac{3}{2}BD_{ff}(W/C)^2$, additional devices spend
their step waiting on collectives — at that point you buy interconnect, not accelerators.
