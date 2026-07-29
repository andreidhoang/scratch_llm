# Chapter 1 — The loss at initialization

**Walked module:** `src/scratch_llm/model.py` (516 LOC) · **Series:** codewalk, fundamentals lane
**Date:** 2026-07-17 · **Verdict:** `__` (DEFENDS-COLD when you can re-derive §3 on a blank page)

> **Karpathy's rule, and this repo's FOP #3:** *predict the number before the run.* A model you can't
> predict is a model you can't debug. We're going to predict one number — the cross-entropy loss of a
> freshly initialized `TransformerLM`, before a single gradient step — to four significant figures,
> from the config alone. Then we measure it.
>
> Everything you need is in `model.py`'s own docstring, line 8:
> *"a fresh LM's cross-entropy on random data is ≈ log(vocab_size). Off by more than a small band ⇒
> head/embedding/norm bug."*
>
> That sentence is **almost right**, and this chapter is about the "almost". The gap between `≈ log(V)`
> and the exact answer is where the understanding lives.

---

## 1. First principles — what is this number even measuring?

Strip away the transformer. A language model is a claim about a probability distribution over sequences:

$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$$

That's not a modeling choice — it's the chain rule of probability, exactly true for **any** distribution.
The only modeling choice is *the factors are computed by a neural net with shared weights*. Autoregressive
LMs are the chain rule plus weight sharing. Nothing else.

We fit it by maximum likelihood. Maximize the log-likelihood of the data ⇔ minimize its negative:

$$\mathcal{L} = -\frac{1}{N}\sum_{t=1}^{N} \log P_\theta(x_t \mid x_{<t})$$

Now here is the move that makes the number mean something. Write that as an expectation over the data
distribution $q$, and expand:

$$\mathcal{L} = \mathbb{E}_{x \sim q}\big[-\log p_\theta(x)\big] = \underbrace{H(q)}_{\text{data's own entropy}} + \underbrace{D_{\mathrm{KL}}(q \,\|\, p_\theta)}_{\text{your ignorance}}$$

Three consequences, all load-bearing:

1. **The loss is measured in nats.** Divide by $\ln 2$ → bits. Cross-entropy *is* the expected code length
   for compressing the data with your model as the codebook. Training an LM **is** learning to compress.
   (Hutter Prize, `bits-per-byte` in every eval harness — same quantity.)
2. **The loss has a floor you cannot cross**: $H(q)$. $D_{\mathrm{KL}} \geq 0$ always. A loss of 0 would mean the
   text is deterministic. Irreducible entropy is why loss curves flatten, not a bug.
3. **The loss has a ceiling that means "I know nothing."** If $p_\theta$ is uniform over $V$ tokens,
   $-\log p = -\log(1/V) = \ln V$ for every token, regardless of the data.

That third one is our target. **$\ln V$ is the loss of maximal ignorance** — and a freshly initialized net
is supposed to be maximally ignorant. So $\mathcal{L}_0 \approx \ln V$.

Now — is it *exactly* $\ln V$? Only if the initial logits are exactly uniform. Let's go find out whether
they are, by tracing an actual tensor through the actual code.

---

## 2. Trace one tensor — token ids → logits

Config: `V=50257, d_model=512, n_layers=4, n_heads=8`, batch `B=4`, seq `S=64`. Measured, not guessed
(`torch.manual_seed(1337)`):

```
token_emb(ids)                (4, 64, 512)   RMS = 0.9898
attn_norm(x)                  (4, 64, 512)   RMS = 1.0000     ← RMSNorm pins this
  q/k/v after view+transpose  (4, 8, 64, 64)                  ← head becomes a batch dim
  after RoPE                  |q| = 356.9106 → 356.9100       ← norm preserved (it's a rotation)
  scores QK^T/sqrt(d)         (4, 8, 64, 64) std = 0.9697     ← the 1/sqrt(d) is doing its job
  attn out (o_proj)           (4, 64, 512)
x + attn  (residual)          (4, 64, 512)   RMS = 1.0458     ← residual stream grows
x + ffn   (residual)          (4, 64, 512)   RMS = 1.1019     ← ... and grows
final_norm(x)                 (4, 64, 512)   RMS = 0.999998   ← ... and gets pinned again
                                             ||row||² = 512.00 (= d_model, exactly)
lm_head(x) → logits           (4, 64, 50257) std = 0.14013  mean = 0.00009
```

Read that last line again. **The logits are not zero.** They have std ≈ 0.14. They are not uniform. So the
loss is not exactly $\ln V$, and we can predict by exactly how much.

Four observations from the trace, each a line of code:

**(a) `RMSNorm` pins the norm exactly** — `model.py:154-158`
```python
x32 = x.float()
rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
return (x32 * rms * self.weight.float()).to(dtype)
```
`weight` is initialized to `ones`. So after the final norm, `mean(x²) = 1` per row exactly, therefore
$\|x\|^2 = d_{\text{model}} = 512$. **Exactly.** Not approximately. That measured `512.00` isn't luck —
it's an algebraic identity, and it's what makes the whole derivation clean. Note also: `.float()` then
`.to(dtype)` — the norm is computed in fp32 even in a bf16 model. That's not paranoia; `mean(x²)` over
4096 channels in bf16 loses ~3 bits of the mantissa to accumulation error.

**(b) RoPE is a rotation, so it can't change the norm** — `model.py:214-216`
```python
rot_even = x_even * cos - x_odd * sin
rot_odd  = x_even * sin + x_odd * cos
```
That's a 2×2 rotation matrix applied to each coordinate pair $(x_{2k}, x_{2k+1})$. Rotations are orthogonal
⇒ $\|Rx\| = \|x\|$. The trace confirms: `356.9106 → 356.9100` (the 1e-6 drift is fp32 rounding, not math).
This is *why* RoPE is safe to bolt onto a trained model in a way that additive position embeddings are not:
it never changes the scale of anything, only the angle.

**(c) The $1/\sqrt{d_k}$ in attention is a variance normalizer** — `model.py:174`
```python
scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
```
$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$. If $q_i, k_i$ are roughly unit-variance and independent, the sum of
$d_k$ products has variance $d_k$, so std $\sqrt{d_k}$. Dividing by $\sqrt{d_k}$ returns it to ~1.
Measured: `std = 0.9697`. Predicted: 1.0. **That is why the scale factor exists** — not tradition. Without
it, scores would have std 8 (for `head_dim=64`), softmax would saturate at init, gradients through the
softmax would be ~0, and your model would start as a hard argmax over random keys.

**(d) The residual stream grows, monotonically** — `model.py:426-431`
```python
x = x + self.attn(self.attn_norm(x), positions, cache, layer_idx)
...
x = x + delta
```
RMS 0.99 → 1.05 → 1.10 across *one* block. Each sub-layer adds a roughly-independent vector, so variances
add: $\sigma^2_{\text{out}} \approx \sigma^2_{\text{in}} + \sigma^2_{\text{delta}}$ ⇒ RMS grows like
$\sqrt{\text{depth}}$. This is the pre-norm signature and it's *the reason pre-norm trains without warmup*:
each block reads a **normalized** copy (`attn_norm(x)`), so it never sees the growing scale, while the
residual highway carries an un-normalized identity path straight to the loss. In post-norm the highway
itself gets normalized, the identity path is broken, and you need warmup + careful init to survive.

---

## 3. The derivation — predict the number

We want $\mathcal{L}_0 = \mathbb{E}[\log \sum_j e^{z_j} - z_y]$ where $z$ are the initial logits and $y$ is the
target. This is exactly what the code computes — `model.py:507-516`:

```python
logits = logits.float()
log_z = torch.logsumexp(logits, dim=-1)
chosen = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
return (log_z - chosen).mean()
```
(No `log(softmax(...))` — `logsumexp` cancels the $\log \circ \exp$ analytically. Never materialize a
probability you're about to take the log of.)

### Step 1 — what distribution are the logits?

`lm_head` is `Linear(d_model, vocab_size)` — `model.py:121-129`:
```python
def forward(self, x):
    return x @ self.weight.T          # weight is (V, d_model)
```
so $z_i = x \cdot W_i$, where $W_i$ is row $i$. Init — `model.py:113-118`:
```python
std = math.sqrt(2.0 / (in_features + out_features))
nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3*std, b=3*std)
```
Two facts collide here, and this is the crux of the chapter:

- From **(a)**: $\|x\|^2 = d$ **exactly** (final RMSNorm, gain = 1).
- From init: $W_{ij}$ iid, mean 0, variance $\sigma_W^2$.

Conditioned on $x$, each logit is a fixed linear functional of iid zero-mean weights:

$$\mathbb{E}[z_i] = 0, \qquad \operatorname{Var}(z_i) = \sum_j x_j^2 \operatorname{Var}(W_{ij}) = \|x\|^2 \sigma_W^2 = d\,\sigma_W^2$$

And the $z_i$ are **independent across $i$** (different rows of $W$, no shared randomness). This is
untied embeddings earning their keep — `tie_embeddings=False` (ADR-0004). If the head were tied to the
embedding table, $W$ and $x$ would be correlated and this step would be a lie. The docstring says
*"untied by default — keeps the loss-at-init derivation clean."* This is that sentence, cashed out.

### Step 2 — the truncation factor (the detail everyone drops)

`trunc_normal_` at $\pm 3\sigma$ does **not** have variance $\sigma^2$. For $X \sim \mathcal{TN}(0,\sigma^2,[-\alpha\sigma, \alpha\sigma])$:

$$\operatorname{Var}(X) = \sigma^2\left[1 - \frac{2\alpha\,\phi(\alpha)}{2\Phi(\alpha)-1}\right]$$

At $\alpha=3$: $\phi(3) = 0.0044318$, $2\Phi(3)-1 = 0.9973002$:

$$\kappa = 1 - \frac{6 \times 0.0044318}{0.9973002} = 1 - 0.026663 = \boxed{0.97334}$$

Measured on the actual weight tensor: `W.var()/σ_nominal² = 0.9737`. ✓ (3 decimal places)

So $\sigma_W^2 = \kappa \cdot \dfrac{2}{d+V}$ and

$$\sigma_z^2 = d\,\sigma_W^2 = \frac{2\kappa d}{d+V}$$

Check: $d=512, V=50257$ ⇒ $\sigma_z^2 = 2(0.97334)(512)/50769 = 0.01963$ ⇒ std $= 0.1401$.
**Measured: `std = 0.14013`.** Four significant figures, from the config alone, zero fitting.

### Step 3 — the expected logsumexp

$\mathbb{E}[z_y] = 0$, so $\mathcal{L}_0 = \mathbb{E}[\log S]$ where $S = \sum_{j=1}^{V} e^{z_j}$.

The temptation is $\mathbb{E}[\log S] = \log \mathbb{E}[S]$. That's **wrong in general** (Jensen: $\log$ is
concave, so $\mathbb{E}\log S \le \log \mathbb{E}S$). It's *nearly* right here, and the reason is
concentration — we have to earn it, not assume it.

$e^{z_j}$ is lognormal. The MGF of a normal at $t=1$ gives $\mathbb{E}[e^{z_j}] = e^{\sigma_z^2/2}$, so

$$\mathbb{E}[S] = V e^{\sigma_z^2/2}, \qquad \operatorname{Var}(S) = V\big(e^{2\sigma_z^2} - e^{\sigma_z^2}\big)$$

Relative fluctuation:

$$\frac{\operatorname{std}(S)}{\mathbb{E}[S]} = \sqrt{\frac{e^{\sigma_z^2}-1}{V}}$$

For $\sigma_z^2 = 0.0196$, $V = 50257$: $\sqrt{0.0198/50257} = 6.3 \times 10^{-4}$. $S$ is a sum of 50257 iid
terms — the law of large numbers crushes it against its mean. So $\log S \to \log \mathbb{E}[S]$ and

$$\boxed{\;\mathcal{L}_0 \;=\; \ln V + \frac{\sigma_z^2}{2} \;=\; \ln V + \frac{\kappa\, d}{d + V}\;}$$

(the Jensen correction is $-\operatorname{Var}(S)/2\mathbb{E}[S]^2 \approx -2\times10^{-7}$ — below fp32 noise, drop it.)

**A closed-form law with no free parameters, in two variables.** Read it:

- $V \gg d$ (GPT-2: 50257 vs 512) ⇒ correction $\to \kappa d/V \to 0$. The folk wisdom $\mathcal{L}_0 \approx \ln V$ holds **because** vocab ≫ width, not because logits are zero.
- $V \sim d$ (small char-level model, or a wide model with a small vocab) ⇒ correction $\to \kappa/2 \approx 0.49$. **The folk wisdom breaks and your "loss ≈ ln V" test fails at the wrong band.**
- The correction is $\kappa \cdot \sigma(\text{logit})^2/2$: it is *exactly* the price of the logits not being flat.

---

## 4. Measure it

`B=16, S=128` (2048 tokens), `n_layers=2`:

| V | d | ln(V) | + κd/(d+V) | = predicted | **measured** | err (naive ln V) | err (ours) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 128 | 5.5452 | 0.3244 | 5.8696 | **5.9047** | 0.3595 | **0.0351** |
| 1024 | 256 | 6.9315 | 0.1947 | 7.1261 | **7.1293** | 0.1978 | **0.0031** |
| 8192 | 512 | 9.0109 | 0.0573 | 9.0682 | **9.0654** | 0.0545 | **0.0028** |
| 50257 | 512 | 10.8249 | 0.0098 | 10.8347 | **10.8433** | 0.0184 | **0.0086** |
| 50257 | 1024 | 10.8249 | 0.0194 | 10.8443 | **10.8402** | 0.0153 | **0.0042** |

Row 1 is the money row. `V=256, d=128` — a char-level model, exactly the toy you'd reach for first.
`ln V` is off by **0.36 nats**. If your test asserts `abs(loss - log(V)) < 0.1` it fails on *correct code*.
The law says the answer is 5.8696 and the answer is 5.90.

**Is the residual error real, or noise?** Karpathy's discipline: never accept a gap you haven't explained.
Per-token loss has std $\approx \sigma_z = 0.14$, so with $N$ tokens the standard error is $\sigma_z/\sqrt{N}$:

| N tokens | predicted | measured | error | 1 SE | **error / SE** |
|---:|---:|---:|---:|---:|---:|
| 64 | 10.83472 | 10.83845 | +0.00373 | 0.01750 | **+0.21** |
| 512 | 10.83472 | 10.83933 | +0.00461 | 0.00619 | **+0.74** |
| 4096 | 10.83472 | 10.83218 | −0.00254 | 0.00219 | **−1.16** |
| 6144 | 10.83472 | 10.83544 | +0.00072 | 0.00179 | **+0.40** |

Every residual is within ±1.2 SE, and the error doesn't shrink toward a nonzero constant — it stays
centred on 0 while the SE shrinks. **The law is exact; what's left is sampling noise.** That's the
difference between "my formula roughly matches" and "my formula is right."

---

## 5. Now break it — the diagnostic that makes this worth knowing

The reason to own this number is that $\mathcal{L}_0$ is a **one-forward-pass X-ray of your init**. Corrupt only
`lm_head`'s init std and watch (V=50257, d=512, so $\sigma_z^2 = 512 \cdot \text{std}^2$):

| lm_head std | σ_z² | small-σ: ln V + σ²/2 | Gumbel: E[max] | **measured** |
|---:|---:|---:|---:|---:|
| 0.00628 (correct) | 0.02 | **10.835** | 0.661 | **10.836** |
| 0.02 | 0.205 | **10.927** | 2.106 | **10.931** |
| 0.1 | 5.12 | **13.385** | 10.528 | **13.383** |
| 1.0 | 512.0 | 266.825 | **96.145** | **95.978** |

Two different physics regimes, and the crossover is at $\sigma_z^2 \sim 2\ln V \approx 21.6$:

**Small σ** ($\sigma_z^2 \ll 2\ln V$): the sum $S$ concentrates, $\mathcal{L}_0 = \ln V + \sigma_z^2/2$. Nails it
through $\sigma_z^2 = 5.12$ (predicted 13.385, measured 13.383).

**Large σ** ($\sigma_z^2 \gg 2\ln V$): concentration fails. One term dominates the sum, so
$\log \sum_j e^{z_j} \to \max_j z_j$, and the max of $V$ iid Gaussians is a **Gumbel** problem:

$$\mathbb{E}[\max_j z_j] = \underbrace{\sigma_z\left(r - \frac{\ln\ln V + \ln 4\pi}{2r}\right)}_{a_V} + \gamma \cdot \underbrace{\frac{\sigma_z}{r}}_{b_V}, \qquad r = \sqrt{2\ln V}$$

($\gamma = 0.5772$, Euler–Mascheroni.) At std=1.0: **predicts 96.145, measured 95.978** — 0.17% error, in
the regime where the naive small-σ formula says 267 and the crude $\sigma\sqrt{2\ln V}$ says 105.

Verified across the crossover on pure sampled logits:

| σ_z² | crude σ√(2lnV) | Gumbel E[max] | measured logsumexp |
|---:|---:|---:|---:|
| 5.12 | 10.528 | 9.615 | **13.383** ← small-σ regime, use ln V + σ²/2 |
| 46.08 | 31.585 | 28.844 | **29.469** ← crossover, neither is exact |
| 512.0 | 105.284 | 96.145 | **95.978** ← Gumbel regime ✓ |
| 4608.0 | 315.852 | 288.436 | **286.895** ← Gumbel regime ✓ |

### The debugging table this buys you

You run a fresh model, print the loss. What does the number mean?

| Observed L₀ | Diagnosis |
|---|---|
| $\approx \ln V + \kappa d/(d{+}V)$ | ✅ init, norm, head, and embedding are all correct |
| $\approx \ln V$ but you expected the correction | your $V \gg d$; the correction is genuinely ~0. Fine |
| slightly **above** $\ln V + \kappa d/(d{+}V)$ | logit std too big: head init std, or a final-norm gain ≠ 1, or a missing final norm |
| **far** above (tens–hundreds) | you're in the Gumbel regime — head init is off by ~100×. Solve $\mathbb{E}[\max] = \mathcal{L}_0$ for $\sigma_z$ to recover *how far* off |
| **below** $\ln V$ | 🚨 **impossible unless you're leaking.** Uniform is the max-entropy answer; nothing untrained beats it. Look for target/input misalignment (off-by-one shift), label leakage, or a broken causal mask |
| exactly $\ln V$ to 6 decimals | suspicious — your logits are *identically* zero. Head weights are zeroed, or the forward isn't reaching the head |
| `nan` | in fp16 check the softmax; here `softmax` subtracts the max (`model.py:163`) so the usual suspect is your data, not the model |

That "below ln V is impossible" row is the highest-value one in the table. It converts a vague "loss looks
low, nice" into **"I have a leak"** on step 0, before you've burned a single GPU-hour.

And this is exactly why the docstring's *causal no-leak* invariant sits next to the loss invariant:
`model.py:331-341` builds the mask
```python
q_pos = torch.arange(past_len, total, device=x.device).unsqueeze(1)   # (s, 1)
k_pos = torch.arange(total, device=x.device).unsqueeze(0)             # (1, total)
mask = k_pos <= q_pos                                                 # True = attend
```
Break that `<=` into `<` or forget the mask entirely, and the model can see token $t$ while predicting
token $t$. Loss at init drops below $\ln V$ instantly. **One number catches it.**

---

## 6. 🇻🇳 Feynman — giải thích lại như cho một đứa trẻ thông minh

**Câu hỏi:** một mô hình ngôn ngữ *chưa học gì cả* thì "sai" bao nhiêu? Và tại sao ta phải quan tâm?

**Trò chơi đoán chữ.** Tưởng tượng một trò chơi: tôi che một chữ trong câu, bạn phải đoán. Cuốn từ điển
có $V$ = 50257 chữ. Bạn **hoàn toàn không biết gì** — chưa đọc một câu tiếng Việt nào trong đời.

Bạn sẽ đoán thế nào? Cách khôn ngoan nhất của một người không biết gì: **nói mỗi chữ đều có khả năng như
nhau**, tức xác suất mỗi chữ là $1/50257$. Đừng giả vờ biết.

Bây giờ ta chấm điểm. Luật chơi: điểm phạt = $-\ln(\text{xác suất bạn đặt vào chữ đúng})$.
Bạn đặt $1/V$ vào mọi chữ, nên dù chữ đúng là gì, điểm phạt luôn là $-\ln(1/V) = \ln V = \ln 50257 = 10.82$.

**Đây chính là con số 10.82.** Nó không phải một con số ngẫu nhiên nào đó — nó là **giá của sự vô tri**.
Nó là điểm phạt của một người thành thật nói "tôi không biết gì cả". Và vì mô hình lúc mới khởi tạo
đúng là không biết gì, loss của nó phải đúng bằng con số đó.

**Cách hiểu thứ hai — nén file.** Con số này chính là *số bit cần để nén văn bản* (chia cho $\ln 2$ ra bit).
Bạn không biết gì về tiếng Việt → phải dùng $\log_2 50257 = 15.6$ bit cho mỗi chữ. Học tiếng Việt giỏi hơn
→ đoán tốt hơn → nén chặt hơn → ít bit hơn. **Huấn luyện một LLM chính xác là dạy nó nén văn bản.** Đó
không phải phép ẩn dụ; hai bài toán là một, cùng một công thức.

**Nhưng — và đây là phần thú vị.** Mô hình mới khởi tạo *không* thành thật hoàn hảo. Trọng số của nó là
số ngẫu nhiên nhỏ, không phải số 0. Nên nó không nói "mọi chữ đều 1/50257". Nó **lẩm bẩm**: "ừm... chữ này
có vẻ nhỉnh hơn tí xíu..." — hoàn toàn vô nghĩa, chỉ là tiếng ồn từ số ngẫu nhiên.

Và tiếng lẩm bẩm vô nghĩa đó **phải trả giá**. Nó luôn làm bạn tệ đi, không bao giờ tốt lên. Tại sao?
Vì phân bố đều đã là câu trả lời **tốt nhất có thể** cho người không biết gì. Mọi sai lệch khỏi nó đều là
đoán bừa, và đoán bừa thì trung bình luôn lỗ.

Giá của tiếng lẩm bẩm đó, chính xác, là:

$$\frac{\kappa \, d}{d + V}$$

Với $V=50257, d=512$: giá = 0.0098. Rất nhỏ — vì từ điển 50257 chữ **quá lớn** so với 512 chiều, tiếng ồn
bị pha loãng. Nên "loss ≈ ln V" *đúng*.

Với $V=256, d=128$ (mô hình ký tự): giá = 0.32. **Không hề nhỏ!** Ở đây "loss ≈ ln V" *sai* rõ rệt, và nếu
bạn viết test `assert abs(loss - log(V)) < 0.1` thì test **đỏ dù code hoàn toàn đúng**. Bạn sẽ ngồi debug
một con bug không tồn tại.

**Bài học một câu:** người ta thuộc lòng "loss lúc init ≈ ln V". Người hiểu biết còn biết **tại sao nó gần
đúng** (từ điển ≫ chiều rộng), **sai bao nhiêu** ($\kappa d/(d+V)$), và **khi nào nó sập** (khi $V \sim d$).
Khoảng cách giữa hai người đó chính là khoảng cách bạn đang lấp trong chương này.

**Và mẹo giá trị nhất:** nếu loss lúc init **thấp hơn** $\ln V$ → **không thể nào**. Không có gì chưa học mà
lại giỏi hơn "tôi không biết". Nếu bạn thấy nó thấp hơn, bạn đang **rò rỉ đáp án** — mask nhân quả hỏng,
hoặc target lệch một vị trí. Một con số, một forward pass, bắt được con bug đắt nhất trong pretraining.

---

## 7. Gap cards → next rep

Written **before** looking anything up. These become tomorrow's `/derive` or 06:30 rep.

1. **`Embedding` init is `N(0,1)`, not `N(0, 2/(V+d))`** (`model.py:138` vs `model.py:116`). Why does the
   embedding table get unit variance while every `Linear` gets fan-in/fan-out scaling? Trace: what would
   `attn_norm(x)` do differently if the embedding had std 0.02 like GPT-2's? (Hint: it's downstream of an
   RMSNorm — does the embedding scale even survive? What *is* it doing then?)
2. **The residual grows as √depth. At `n_layers=64` the residual RMS is ~8× the block delta.** Does that
   mean deep blocks contribute proportionally less? Is this the argument for `1/√(2N)` output-proj scaling
   (GPT-2) — and why does `model.py` **not** do it? Is `qk_norm` + RMSNorm enough to cover it?
3. **`tie_embeddings=True` breaks my Step-1 independence claim.** What is $\mathcal{L}_0$ for a tied model?
   $z_i = x \cdot E_i$ where $x$ is itself built from $E$ — so $z_{x_t}$ correlates with $x$. Predict the
   sign of the shift and then measure it. (I predict $\mathcal{L}_0$ comes out **below** $\ln V$ — a
   "leak" that isn't a leak. If true, the debug table above needs a caveat row.)
4. **The crossover $\sigma_z^2 \sim 2\ln V$.** Derive it properly: at what $\sigma_z$ does $\text{std}(S)/\mathbb{E}[S] = \sqrt{(e^{\sigma^2}-1)/V}$
   reach $O(1)$? That gives $\sigma_z^2 \approx \ln V$, not $2\ln V$. Which is right, and why did the
   measured crossover (σ²=46 fitting neither) land between them?

### Verdict
- `model.py` walk 🚶 — **done 2026-07-17**
- Feynman 🗣 — **done** (§6)
- Derive 📐 — **done** (§3), blank-page re-derivation pending
- **DEFENDS-COLD: not yet.** Gate: re-derive $\mathcal{L}_0 = \ln V + \kappa d/(d+V)$ on a blank page,
  including the concentration argument, without notes.

### Next chapter
**Ch. 2 — Attention: why softmax(QKᵀ/√d)V and not something else.** We have the shapes
`(4, 8, 64, 64)` and the fact that `std = 0.9697`. Next: why *this* function, what the causal mask costs in
FLOPs and memory, and why `use_sdpa` (`model.py:353-358`) exists at all — the (B,H,S,S) score tensor is
`4 × 8 × 64 × 64 × 4 bytes` here, but at S=4096 it's **2 GB per layer, retained for backward**. That number
is the entire reason FlashAttention exists, and it sets up `kernels/flash_attention_triton.py`.

---

## Provenance

Every number above is measured against this repo's `src/scratch_llm/model.py` on `torch 2.13.0+cpu`,
`torch.manual_seed(1337)` (§2, §5) / `(0)` (§4), config `V=50257, d_model=512, n_heads=8`.
No fitted constants: $\kappa = 0.97334$ is analytic (truncated-normal variance at $\alpha=3$),
$\gamma = 0.5772157$ is Euler–Mascheroni.

**Measured tensor facts** referenced in the derivation:

| quantity | value | where |
|---|---|---|
| residual RMS, `token_emb` → `L3 +ffn` | 0.9898 → 1.4911 | grows $\sqrt{\text{depth}}$; §2(d) |
| residual RMS after every `*_norm` | 1.0000 exactly | RMSNorm pins it; §2(a) |
| $\|x\|^2$ after `final_norm` | 512.00 = $d$ | the identity §3 Step 1 rests on |
| RoPE norm preservation | 356.9106 → 356.9100 | rotation, §2(b) |
| `QKᵀ/√d` std / max / min | 0.9697 / +4.696 / −4.389 | the $1/\sqrt{d_k}$ working; §2(c) |
| logit std / mean (12.9 M logits) | 0.14013 / 0.000095 | vs predicted std 0.14011; §3 Step 2 |
| softmax prob max / min at init | 3.59e−5 / 1.01e−5 | uniform = 1.99e−5 → **3.56× spread** |
| per-token loss mean / std | 10.83779 / 0.13833 | std ≈ $\sigma_z$ = 0.14013, as derived |
