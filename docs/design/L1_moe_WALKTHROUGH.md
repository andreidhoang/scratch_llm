# MoE, traced end-to-end with real tensors

> A line-by-line walkthrough of `src/reasoning_llm/moe.py` — every tensor shown with **actual
> values from real runs**, then the full training loop (forward → loss → backward → bias update →
> multi-step balancing). Companion to [ADR-0007](../adr/ADR-0007-moe-pulled-forward.md). Read it
> with `moe.py` open beside you.
>
> All numbers below are reproducible: the forward trace uses `torch.manual_seed(7)`, the training
> traces use `torch.manual_seed(0/1/3)`. Nothing here is hand-waved.

---

## 0. The mental model (read this first)

A **dense** FFN sends every token through one big `SwiGLU`. A **MoE** FFN keeps a *committee* of
small experts and, per token, runs **only the few** the router picks — same compute, far more
capacity. Three jobs have to happen for every token:

1. **Route** — decide *which* experts handle this token (the router).
2. **Mix** — run those experts and *blend* their outputs with weights that sum to 1 (the gates).
3. **Balance** — make sure traffic spreads across experts instead of collapsing onto a few
   (the aux-loss-free bias).

The hard part isn't (1) or (2) — it's (3). A MoE that routes 90% of tokens to 2 experts has
wasted the other 254. DeepSeek-V3's trick is to balance **without** a gradient penalty that fights
the loss: a tiny per-expert *bias* that the trainer nudges by hand. We'll watch it work in §5.

### The config we trace

```
d_model = 4 · n_routed_experts = 4 · n_experts_per_tok (top-k) = 2 · n_shared_experts = 1
batch = 1 · seq = 3   →   N = 3 tokens
```

Tiny on purpose — 3 tokens × 4 experts fits on a page. The shapes generalize directly to V3's
`256 experts, top-8`.

---

## 1. The forward pass, line by line

`MoEFeedForward.forward(self, x)` in `moe.py`. Our input — 3 tokens, each a 4-vector (imagine the
hidden state after attention + RMSNorm):

```python
x.shape == (1, 3, 4)        # (B, S, d_model)
x = [[[ 2.024, -1.251, -1.395,  0.668],     # token 0
      [-0.476,  0.761, -0.972,  0.418],     # token 1
      [ 1.387, -1.042, -0.238, -0.417]]]    # token 2
```

### Line: `b, s, d = x.shape` and `xf = x.reshape(-1, d)`

```python
xf.shape == (3, 4)          # (N, d) — the (B, S) grid is flattened to a flat list of N tokens
xf = [[ 2.024, -1.251, -1.395,  0.668],
      [-0.476,  0.761, -0.972,  0.418],
      [ 1.387, -1.042, -0.238, -0.417]]
```

**Why this single line earns two of your tests.** From here on the router sees a *flat bag of
tokens*; it never knows token 1 came after token 0. Routing is therefore **per-token and
position-independent**, which is exactly why:

| Invariant | Why flattening guarantees it |
|---|---|
| **decode/cache parity** (`test_decode_cache_parity_with_moe`) | a token routes identically whether it's 1-of-12 in a full forward or the lone token during incremental decode |
| **causal no-leak** (`test_causal_no_leak_with_moe`) | routing depends only on a token's own vector — a future token can't change an earlier token's route |

This is the property the whole repo cares about: train and infer route the same → `kl_train_infer`
stays clean. The day they diverge is the MoE×RL collapse that A5.3 hunts.

### Line: `logits = self.router(xf)`

`self.router.gate` is a `Linear(4, 4)`. Its weight holds **4 expert centroids** `e_0..e_3`
(one 4-vector per expert, stored as rows). The logit is a **dot product**, `logit[t,i] = xf[t]·e_i`:

```python
logits.shape == (3, 4)      # (N, n_routed)  — "token t's alignment with expert i"
logits = [[ 0.465,  0.060, -1.689, -0.081],     # token 0:   most aligned w/ expert 0
          [-0.456,  0.566, -0.009, -0.186],     # token 1:   most aligned w/ expert 1
          [ 0.367, -0.378, -0.931,  0.128]]     # token 2:   most aligned w/ expert 0
          grad_fn=<MmBackward0>                  # ← carries gradient: centroids are learned
```

The centroids are ordinary learnable weights — gradient descent shapes them so each expert
"claims" a region of token-space (we'll see their gradient in §4).

### Line: `affinity = torch.sigmoid(logits)`  — **the most important design choice**

```python
affinity.shape == (3, 4)    # each value in (0, 1), per-expert INDEPENDENT
affinity = [[0.614, 0.515, 0.156, 0.480],       # token 0  (row does NOT sum to 1)
            [0.388, 0.638, 0.498, 0.454],       # token 1
            [0.591, 0.407, 0.283, 0.532]]       # token 2
```

Each expert scores the token **on its own merit** — "am I a match? yes/no in (0,1)" — with no
reference to the others. Rows don't sum to 1 (token 0: `1.765`). That independence is deliberate:

> **V2 → V3.** DeepSeek-V2 used `softmax` here, so experts *competed* on a simplex (scores sum to
> 1) — raising one expert's score mechanically lowers another's. V3 switched to **sigmoid** so the
> scores are decoupled. That decoupling is what lets the balancing bias (next line) push one
> expert's selection up or down **without** perturbing every other expert's score. Sigmoid is the
> enabler of aux-loss-free balancing. *(V3 Eq. 15.)*

### Line: `sel_scores = affinity + self.router.bias`  — the balancing lever

```python
self.router.bias == [0., 0., 0., 0.]     # all zero at init
sel_scores == affinity                    # so right now, identical
```

`bias` is a **non-grad buffer** `b_i` — one scalar per expert. At birth it's zero. But this is the
*entire* load-balancing mechanism: after each optimizer step the trainer raises `b_i` for starved
experts and lowers it for overloaded ones (§5). It enters **only here, in the selection key**.
*(V3 Eq. 16.)*

### Line: `topk_idx = sel_scores.topk(self.k, dim=-1).indices`  — sparsity is born

```python
topk_idx.shape == (3, 2)    # (N, k) — the 2 chosen expert ids per token
topk_idx = [[0, 1],         # token 0 → experts 0 and 1   (scores 0.614, 0.515 were highest)
            [1, 2],         # token 1 → experts 1 and 2
            [0, 3]]         # token 2 → experts 0 and 3
```

Each token keeps its **2 highest** experts and discards the other 2. **This is where the compute
saving lives:** a token runs `k / n_routed = 2/4 = 50%` of experts here; at V3 scale it's
`8/256 ≈ 3%`. The discarded experts cost nothing for this token.

> **See the bias work, concretely.** Token 1's scores are `[0.388, 0.638, 0.498, 0.454]` → it
> picks experts **1, 2**. If the balancer had starved expert 0 and set `b_0 = +0.3`, the scores
> become `[0.688, 0.638, 0.498, 0.454]` → top-2 flips to experts **0, 1**. The bias *rerouted a
> token* with no gradient — just by editing this one selection key.

### Line: `gate_sel = affinity.gather(-1, topk_idx)`  — the bias's *other half*

```python
gate_sel.shape == (3, 2)    # RAW affinity at the selected ids — gathered from `affinity`, NOT sel_scores
gate_sel = [[0.614, 0.515],     # token 0: raw s for experts 0,1
            [0.638, 0.498],     # token 1: raw s for experts 1,2
            [0.591, 0.532]]     # token 2: raw s for experts 0,3
```

We gather the **raw `affinity`**, not `sel_scores`. This is the subtle, load-bearing split:

- **`sel_scores = s + b`** decided *who* is in (the route).
- **`gate_sel = s`** decides *how much each one counts* (the weight).

So the balancer can **force an expert into the committee without inflating its influence**. In the
what-if above, expert 0 would be *selected* (score 0.688) but weighted by its raw `0.388`. Your
`test_bias_steers_selection_not_value` pins exactly this. *(V3 Eq. 13 uses `s`; Eq. 16 uses `s+b`
for the TopK — the same separation.)*

### Line: `gate_norm = gate_sel / gate_sel.sum(-1, keepdim=True)`

```python
gate_norm.shape == (3, 2)    # each row sums to 1 — a convex blend
gate_norm = [[0.544, 0.456],     # 0.614/(0.614+0.515), 0.515/(...)
             [0.562, 0.438],
             [0.526, 0.474]]      # rowsums = [1.000, 1.000, 1.000]
```

The output will be `g_0·expert_0(x) + g_1·expert_1(x)`. Normalizing pins the **total mixing weight
to 1** regardless of how confident the router happened to be — without it, the FFN's contribution
to the residual stream would swell or shrink with raw sigmoid magnitudes, destabilizing training.

> **This is the dense-equivalence anchor.** With `n_routed=1, k=1`, the one selected expert gets
> `gate_norm = s/s = 1` exactly → the MoE *is* a plain SwiGLU. That's `test_dense_equivalence_single_expert`,
> and it's why this normalization must be exact (no epsilon — sigmoid is strictly positive, so the
> denominator can't be zero).

### Lines: `gates = scatter(...)` and `selected = scatter(...)`, `counts = selected.sum(0)`

We expand the compact `(N, k)` form back to a dense `(N, n_routed)` matrix — weight in the chosen
slots, **0 elsewhere**:

```python
gates    = [[0.544, 0.456, 0.000, 0.000],     selected = [[1, 1, 0, 0],
            [0.000, 0.562, 0.438, 0.000],                 [0, 1, 1, 0],
            [0.526, 0.000, 0.000, 0.474]]                 [1, 0, 0, 1]]

counts = selected.sum(0) == [2., 2., 1., 1.]   # tokens routed to each expert
```

The **zeros are the sparsity**. `counts` says experts 0 and 1 each drew 2 tokens, experts 2 and 3
drew 1 — this vector feeds both the balancer (§5) and the diagnostics (§2).

### The expert loop — `gather → run → ×gate → index_add`

```python
y = torch.zeros_like(xf)                                  # (3, 4) accumulator
for e in range(self.n_routed):
    idx = (selected[:, e] > 0).nonzero(as_tuple=True)[0]  # which tokens chose expert e
    if idx.numel() == 0:
        continue
    out = self.routed_experts[e](xf[idx])                 # run ONLY those rows through SwiGLU e
    weighted = gates[idx, e].unsqueeze(-1) * out          # scale each by its gate
    y = y.index_add(0, idx, weighted)                     # scatter-add back into y
```

Read it **column-wise off the gate matrix**:

```
expert 0  (column [0.544, 0, 0.526]) → tokens [0, 2], gates [0.544, 0.526]
expert 1  (column [0.456, 0.562, 0]) → tokens [0, 1], gates [0.456, 0.562]
expert 2  (column [0, 0.438, 0])     → token  [1],    gate  [0.438]
expert 3  (column [0, 0, 0.474])     → token  [2],    gate  [0.474]

routed sum y = [[-0.053,  0.045,  0.087,  0.035],
                [-0.014,  0.007, -0.016, -0.006],
                [-0.063, -0.038, -0.009, -0.026]]   grad_fn=<IndexAddBackward0>
```

Token 0 appears in **both** expert 0's and expert 1's lists — `index_add` accumulates both into
row 0, giving the mixture `y[0] = 0.544·expert0(x0) + 0.456·expert1(x0)`.

> **Three things to notice:**
> - **Why `index_add` (out-of-place), not `y[idx] += ...`?** In-place indexed writes on a tensor
>   that's part of the autograd graph can corrupt the backward pass. `y = y.index_add(...)` returns
>   a fresh tensor each iteration → gradients flow cleanly to every expert *and* back through the
>   gates to the router.
> - **Why is `y` so tiny?** At init the SwiGLU experts have small truncated-normal weights, so
>   their output ≈ 0. That's what keeps **loss-at-init ≈ log(vocab)** — the FFN barely perturbs the
>   residual stream at birth (§6).
> - **The CPU trade-off.** This is a Python loop over experts — clear and correct, O(n_routed)
>   iterations. At 256 experts on GPU you replace it with one batched grouped-matmul. That's the
>   noted perf follow-up; the math is identical.

### Lines: scaling + shared experts → the **delta**

```python
y = y * self.cfg.routed_scaling_factor      # = 1.0 here (a no-op); V3 uses ~2.5
for shared in self.shared_experts:
    y = y + shared(xf)                       # shared expert runs on ALL tokens, ungated

final delta = [[-0.201, -0.132,  0.140, -0.109],     # routed sum + shared(x), per token
               [-0.031,  0.067, -0.053, -0.015],
               [-0.123, -0.074, -0.026, -0.121]]
```

The **shared expert is always-on and ungated** — it absorbs the "common knowledge" every token
needs, so the routed experts don't waste capacity relearning it (DeepSeekMoE *shared-expert
isolation*). Compare the routed-only `y` to this: every row shifted by `shared(x)`.

### Line: `return y.reshape(b, s, d), stats`  — the residual contract

```python
y.reshape(1, 3, 4)   # back to (B, S, d) — and this is the DELTA, not x + delta
```

`TransformerBlock.forward` does `x = x + self.ffn(...)`. The MoE returns the **FFN contribution
only** — identical contract to `SwiGLU`. If it returned `x + delta`, the block would add `x` twice
(the **double-residual bug** `test_dense_equivalence_single_expert` guards). The verify line from
the trace confirms our manual `y` equals the module output exactly.

---

## 2. The stats — what gets measured (`_stats`)

Three numbers come out alongside the delta. With our `counts = [2,2,1,1]` over `N=3`:

```python
# seq-wise balance loss (V3 Eq. 17-20)
f_i = (n_routed / (k·N)) · counts = (4/(2·3))·[2,2,1,1] = [1.333, 1.333, 0.667, 0.667]
P_i = mean over tokens of normalized affinity s'_{i,t}   = [0.290, 0.280, 0.165, 0.265]
aux_loss = α · Σ(f_i · P_i) = 1e-4 · 1.046 = 0.000105

# router z-loss (ST-MoE stabilizer, on raw logits)
z_loss  = c_z · mean_t( logsumexp_i(logits) )²

# diagnostics
entropy = -Σ p_i log p_i  where p = counts/Σcounts = [0.333,0.333,0.167,0.167]
        = 1.330   (max = log 4 = 1.386)  → router is well-spread
```

- **`aux_loss`** multiplies the **hard** load `f_i` (detached — a count, no gradient) by the
  **soft** preference `P_i` (differentiable — gradient flows to the router). It's large only when
  an expert is *both* overloaded *and* over-preferred, gently nudging the router off that expert.
  It's tiny (`α=1e-4`) on purpose — the **bias** does the real balancing; this is just a backstop.
- **`entropy`** is the collapse alarm. `1.330` vs max `1.386` → near-uniform. If routing collapsed
  to one expert, entropy → 0. This is the falsifiable A1.1 prediction (`> 0.9·log N_r`).
- **`counts`** also get added to `router.load_count` (training mode only) — the accumulator the
  balancer reads in §5.

---

## 3. How it plugs into the model (`model.py`)

The FFN returns `(delta, stats)`; the block and LM thread the stats out **explicitly** (no
side-channels), and the dense path is untouched.

```python
# TransformerBlock.forward
if self.is_moe:
    delta, stats = self.ffn(self.ffn_norm(x))     # MoEFeedForward → (Tensor, MoEStats)
else:
    delta, stats = self.ffn(self.ffn_norm(x)), None   # SwiGLU → Tensor, no stats
x = x + delta                                      # ← identical residual either way
return x, stats

# TransformerLM.forward(..., return_aux=False)
layer_stats = []
for layer_idx, block in enumerate(self.blocks):
    x, stats = block(x, positions, cache, layer_idx)
    if stats is not None:
        layer_stats.append(stats)
...
if return_aux:
    return logits, self._aggregate_aux(layer_stats)   # sums aux/z across MoE layers
return logits                                          # default & decode path: just logits
```

`return_aux=False` is the default **and** the decode path → behavior is byte-for-byte the dense
build. `return_aux=True` (training) hands back `AuxOutput` whose `.total = aux_loss + z_loss`.

---

## 4. One training step, traced (`train.py`)

Now the end-to-end loop. Real run: `vocab=32, d_model=8, 1 layer, 4 experts, top-2, 1 shared`,
`seed=0`, a `(2, 6)` batch.

```python
# --- the loop body in train.py, MoE branch ---
logits, aux = model(inputs, return_aux=True)
loss = cross_entropy(logits, targets) + aux.total
loss.backward()
gradient_clipping(model.parameters(), cfg.grad_clip)
optimizer.step()
model.moe_update_biases()          # ← the aux-loss-free balancing step
```

### 4a. The loss

```
ce    = 3.6323         # cross-entropy
aux   = 0.000110       # seq-wise balance loss  (α·Σ f·P)
z     = 0.003884       # router z-loss
total = 3.6363         # ce + aux + z

log(32) = 3.4657   →   Δ = 0.167   ✓ loss-at-init in the ±0.3 band
```

The regularizers are a rounding error next to CE — they shape *routing*, not *prediction*. The
loss at init sits right at `log(vocab)` (uniform-guess entropy): the MoE FFN, with its near-zero
expert outputs, hasn't disturbed the head.

### 4b. The backward — which tensors carry gradient?

```
router.gate.weight.grad   norm = 0.00931      ← centroids ARE learned (by SGD)
routed_experts[0].w1.grad norm = 0.04574      ← experts ARE learned
router.bias.grad               = None         ← bias is NOT learned by SGD
router.bias.requires_grad      = False        ← it's a buffer, not a parameter
```

This is the **two-speed system** that defines DeepSeek-style MoE:

```
            learned by gradient descent          updated by hand-rule (no gradient)
            ───────────────────────────          ──────────────────────────────────
   router centroids e_i   (slow, smooth)   |   per-expert bias b_i   (fast, ±γ steps)
   expert weights         (slow, smooth)   |
            ▲                                            ▲
   minimizes prediction loss                  equalizes load — orthogonal objective
```

Keeping balance **off** the gradient is the whole point: a balance *penalty* (the old way) fights
the prediction loss and degrades quality. The bias balances on a separate channel, so quality and
balance don't trade off.

### 4c. The bias update (`moe_update_biases` → `Router.update_bias`)

The accumulated load this step was `load_count = [9, 2, 8, 5]` (mean = 6):

```python
violation = load_count.mean() - load_count = 6 - [9, 2, 8, 5] = [-3, +4, -2, +1]
bias += speed · sign(violation)
      = [0,0,0,0] + 0.001 · [-1, +1, -1, +1]
      = [-0.001, +0.001, -0.001, +0.001]
load_count.zero_()                         # reset for the next accumulation window
```

Experts 0 and 2 were **overloaded** (9, 8 > 6) → biases **down**; experts 1 and 3 **underloaded**
→ biases **up**. Next step, the down-biased experts win slightly fewer top-k slots and the
up-biased ones win slightly more. The `sign` (not magnitude) keeps each step a fixed `±γ` nudge —
robust to outliers, V3's exact rule.

---

## 5. Balancing dynamics over many steps (the payoff)

Watch the bias actually rescue a skewed router. Real run: 4 experts, top-2, 32 tokens, expert 0's
centroid over-strong so it's over-selected at the start; `bias_update_speed = 0.03`.

```
step | load_fraction e0 e1 e2 e3     |   H   (max=1.386) | bias e0..e3
   0 | [0.281, 0.250, 0.219, 0.250]  | 1.382             | [ 0.00,  0.00,  0.00,  0.00]
   2 | [0.266, 0.234, 0.234, 0.266]  | 1.384             | [-0.03, -0.03,  0.06,  0.00]
   5 | [0.250, 0.250, 0.250, 0.250]  | 1.386             | [-0.06, -0.03,  0.12, -0.03]
  10 | [0.250, 0.250, 0.250, 0.250]  | 1.386             | [-0.06, -0.03,  0.12, -0.03]
  60 | [0.250, 0.250, 0.250, 0.250]  | 1.386             | [-0.06, -0.03,  0.12, -0.03]
```

Read the story across the rows:

- **step 0** — expert 0 hogs `0.281` of the load (it had the strong centroid); expert 2 is starved
  at `0.219`. Entropy `1.382`, just under the max.
- **steps 1–5** — the bias drifts: e0 (overloaded) goes **negative**, e2 (most starved) climbs to
  the **highest** bias `+0.12`. Traffic equalizes.
- **step 5 onward** — perfect uniform `[0.25, 0.25, 0.25, 0.25]`, entropy pinned at `log 4 = 1.386`,
  and the **bias stops moving** — equilibrium. Once balanced, the `±γ` nudges cancel out
  (equal over/under counts), so the system *holds*.

> This is the discriminating behavior `test_balancer_overcomes_any_routing_preference` proves
> deterministically: because sigmoid affinities live in `(0,1)`, a bias gap `> 1` can flip **any**
> routing preference — so the balancer can always break a collapse. With `bias_update_speed = 0`
> the bias column would stay all-zeros and the imbalance would persist.

> **Caveat worth knowing (bang-bang).** With **top-1** routing and near-identical tokens, the load
> is one-hot every step, so the bias can't *spread* within a step — it just rotates the single
> winner (e0→e1→e2…), entropy stuck at 0. Balancing needs either `k ≥ 2` or token diversity so the
> load *can* distribute. Real models have both; it's a thing to recognize in a toy.

---

## 6. The discipline checks, visible in the numbers

**Loss-at-init ≈ log(vocab)** (§4a): `ce = 3.63` vs `log 32 = 3.47`, Δ`0.167`. The predicted band
held — proof there's no double-residual, the scaling is 1.0, and the shared expert isn't blowing up
the residual. *Predict the number first; it's the debugging anchor.*

**Overfit one batch** — can the sparse FFN + router memorize a single batch? (regularizers off):

```
step   0 | loss 3.5666
step  30 | loss 1.6012
step  60 | loss 0.6602
step  90 | loss 0.2576
step 120 | loss 0.1395
step 150 | loss 0.1062     → CE driven toward 0 ✓
```

If this *couldn't* reach ~0, the bug would be in the optimizer/data/loss wiring — not the data.
It does → the gather/run/`index_add`/gate plumbing and the gradient path through the router are all
correct.

---

## Appendix A — shapes at a glance

| Tensor | Shape | Meaning |
|---|---|---|
| `x` / `xf` | `(B,S,d)` / `(N,d)` | input; flattened to a bag of N tokens |
| `logits` | `(N, n_routed)` | token·centroid alignment (pre-sigmoid) |
| `affinity` | `(N, n_routed)` | `sigmoid(logits)`, each in (0,1), independent |
| `sel_scores` | `(N, n_routed)` | `affinity + bias` — the selection key |
| `topk_idx` | `(N, k)` | chosen expert ids per token |
| `gate_sel` / `gate_norm` | `(N, k)` | raw / normalized gate weights (rows sum to 1) |
| `gates` / `selected` | `(N, n_routed)` | dense gate matrix / 0-1 mask (sparse) |
| `counts` | `(n_routed,)` | tokens routed to each expert |
| `y` (delta) | `(N,d)` → `(B,S,d)` | routed mix + shared; the **FFN contribution only** |
| `bias` / `load_count` | `(n_routed,)` | balancing state (buffers, non-grad) |

## Appendix B — gradient flow map

```
loss = cross_entropy(logits, targets) + aux_loss + z_loss
                       │                   │          │
        ┌──────────────┘                   │          └── router centroids (logit magnitude)
        ▼                                  ▼
   lm_head → blocks → MoE delta       router centroids (via P_i, the soft preference)
        │                  │
        │       ┌──────────┴───────────┐
        ▼       ▼                      ▼
   attention  routed_experts[e]   shared_experts   ← all learned by SGD
                      ▲
                  gates (∂ through normalized affinity → router centroids)

   NOT in the graph:  router.bias  ·  load_count  ·  counts/f_i (detached)
                      └── updated by Router.update_bias(±γ·sign(load violation)) instead
```

## Appendix C — glossary → DeepSeek-V3 equations

| Code | Paper (arXiv:2412.19437) | One line |
|---|---|---|
| `affinity = sigmoid(logits)` | Eq. 15 | per-expert score, sigmoid not softmax |
| `topk(affinity + bias)` | Eq. 16 | bias enters **selection** only |
| `gate = affinity / Σ selected` | Eq. 13 | value uses **raw** affinity, normalized |
| `shared + scaling·Σ gate·routed` | Eq. 12 | shared (always-on) + routed mixture |
| `bias += γ·sign(mean−load)` | §Aux-loss-free | hand-rule balancer, no gradient |
| `α·Σ f_i·P_i` | Eq. 17-20 | tiny seq-wise balance backstop |
| z-loss on raw logits | *(ST-MoE, not V3)* | repo add-on; logit-magnitude stabilizer |
