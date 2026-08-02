# PROPOSAL — `core/situ.py` (SiTU-GLU), for hand-retyping

> **Boundary notice (HANDCRAFTED.md Class 1):** agent-written *proposal*, not a diff — the
> human retypes what they keep. Built FIRST in the core serial order (ROADMAP layout: warmup
> for the KDA rung); it lands in the model at K5 (Stable LatentMoE). Every equation traced to
> K3 tech report arXiv:2607.24653 (§2.3.2 Eq. 12, App. B Eqs. 18–19, Table 1) and the released
> HF reference (`modeling_kimi_linear.py` @ moonshotai/Kimi-K3). Extracted 2026-08-02.

## 1. The equation (§2.3.2, Eq. 12)

One tanh softcap `softcap(x, β) = β·tanh(x/β)` per GLU branch, different β per branch:

```
g, u = W_g x, W_u x                              (the two GLU pre-activations)
SiTU-GLU(x) = [ β1·tanh(g/β1) ⊙ σ(g) ] ⊙ [ β2·tanh(u/β2) ]
out = W_down ( SiTU-GLU(x) )
```

- **β1 = 4** (gate branch), **β2 = 25** (up branch). K3 config: `hidden_act: "situ"`,
  `activation_situ_beta: 4.0`, `activation_situ_linear_beta: 25.0`.
- **The trap:** the sigmoid factor takes the **uncapped** pre-activation `g` — NOT
  `σ(β1·tanh(g/β1))`. β1 caps only the *linear factor* of the Swish gate. This asymmetry is
  the whole design: it "retains the approximately linear positive regime of Swish" while
  bounding it, and keeps the negative tail (sigmoid → 0) intact (App. B).
- Caps apply **per-branch, before the elementwise product** — never to the product itself.
  The down projection sees a product already bounded by β1·β2.
- It is NOT softsign, NOT `σ(β2·x)`, NOT a Gemma-style two-stage cap.

## 2. Why it exists (§2.3 motivation, verbatim)

"Both multiplicative factors in SwiGLU are unbounded, so coincident large coordinates can
produce activation outliers and increase overflow risk in low-precision arithmetic. The
sigmoid gate of the original GLU avoids unbounded gate growth, but it does not retain the
approximately linear positive regime of Swish." — SiTU-GLU is one of the three Stable
LatentMoE components (with pre-up RMSNorm and Quantile Balancing), targeting the exploding
routed branch at 896-expert sparsity. This is also why our Tier-2 P7 (SiTU × fake-quant
MXFP8) is the designed-in interaction arm.

## 3. Usage sites (for K5/K6 wiring — one code path everywhere)

Routed experts (w1/w3 packed) · shared experts · the dense layer-1 MLP. NOT used in: KDA
internals (Swish after ShortConv — different activation), router (sigmoid), vision tower.

## 4. Reference implementation notes (HF code, verbatim behavior)

```python
gate = x[..., :d].to(torch.float32); up = x[..., d:].to(torch.float32)
situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)      # sigmoid on UNCAPPED gate
up     = linear_beta * torch.tanh(up / linear_beta)                # if linear_beta set
return (situ_a * up).to(x.dtype)
```

- HF packs the two projections (`cat([w1(x), w3(x)], −1)` → split half). Our repo convention
  is separate `gate_proj`/`up_proj` — equivalent; the proposal is the math, not the packing.
- **Compute in fp32, cast back** to the activation dtype (K3's fp32-sensitive numerics
  posture; matters for the bit-level tests below).

## 5. Properties with proofs (App. B — the test suite falls out of these)

- **Bound:** |tanh| < 1 and 0 < σ < 1 ⇒ |SiTU-GLU| ≤ β1·β2 = **100** per coordinate (Eq. 19).
  The activation-bound test ("no |x| > 100 post-SiTU") is exact, not statistical.
- **Near-origin identity:** β·tanh(z/β) = z + O(z³/β²) ⇒ SiTU-GLU matches SwiGLU to first
  order around 0 (Eq. 18); recovers SwiGLU pointwise as β1, β2 → ∞.
- **Smooth-cap gradients:** nonzero away from saturation (unlike hard clamping) — "we find
  [this] to give better training behavior."
- **fp32 saturation:** tanh computed in fp32 saturates to exactly ±1.0 for |z/β| ≳ 8.7 —
  so `β·tanh(z/β) == ±β` exactly in fp32 for large |z| (the fp32 tanh-saturation test).

## 6. Gates & red-team menu (mastery bar: delete-test + bound proof |f| ≤ 100)

1. **Bound proof test:** random fp32 sweeps over extreme magnitudes (±1e4) — no output
   coordinate exceeds 100; the gate branch alone never exceeds β1 = 4, the capped up branch
   never exceeds β2 = 25.
2. **Near-origin test:** |SiTU-GLU − SwiGLU| = O(|z|³) for |z| ≪ 1 — verifies the uncapped-
   sigmoid wiring (a capped-sigmoid variant fails this at second order, not first — so also
   test a mid-range point, e.g. g = 6: sigmoid(6)·(4·tanh(1.5)) vs sigmoid(4·tanh(1.5))·…).
3. **fp32 saturation test:** g = ±1e3 ⇒ β1·tanh(g/β1) == ±β1 exactly (fp32); product finite.
4. **Negative-tail test:** g → −∞ ⇒ gate branch → 0 (sigmoid kills the capped linear factor);
   output → 0 regardless of u.
5. **β→∞ limit:** β1 = β2 = 1e9 reproduces SwiGLU to fp32 tolerance.
6. **Derivative spot-check** against autograd on a hand product-rule implementation
   (paper gives no explicit derivative — derive it; the human derives, I check).

## 7. mini-K3 instantiation

Same β1 = 4 / β2 = 25 at every site (ROADMAP K6 table: "SiTU 4/25 — exact"). Applied at:
dense layer-1 MLP, LatentMoE routed experts (at latent width), 2 shared experts. Sits
downstream of the K5 pre-up RMSNorm in the routed branch (order at K5: latent down → router
→ experts@latent → RMSNorm → up; the SiTU-GLU is INSIDE each expert's gate/up/down).
