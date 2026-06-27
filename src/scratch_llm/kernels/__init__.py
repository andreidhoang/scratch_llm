"""L2 Systems (A2) — attention kernels and the FlashAttention-2 roofline artifact.

This is the systems-layer benchmark, **not** the dense v0.1.0 policy path (the model uses the
plain `scaled_dot_product_attention` in `model.py`). It exists to produce the A2.1 roofline
(FA2-Triton vs `F.scaled_dot_product_attention`, % of peak) and to teach *why* the train and serve
engines compute different logits (the mechanistic root of `kl_train_infer`). See
docs/design/L2_flash_attention_SPEC.md.

Only the pure-PyTorch reference is exported here so importing this package never requires Triton
(CPU/CI safe). The Triton kernel lives in `flash_attention_triton.py` and is imported on demand on
a GPU box.
"""

from scratch_llm.kernels.flash_attention import (
    FlashAttentionPyTorch,
    flash_attention_forward,
)

__all__ = ["FlashAttentionPyTorch", "flash_attention_forward"]
