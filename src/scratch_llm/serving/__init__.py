"""Serving layer — turn the forward pass into an inference *engine*.

A1 (performance curriculum). The from-scratch decoder + KV cache + GQA already live in
``model.py`` / ``sampling.py`` and are token-exact, so A1 is **not** a decoder rebuild — it is a
*measurement & serving-systems* assignment (FOP-4: implemented ≠ measured). This package holds the
apparatus that (a) measures where inference time and memory go (:mod:`~scratch_llm.serving.metrics`)
and (b), rung by rung, adds the serving levers — continuous batching, paged KV, speculative decoding,
CUDA-graph decode — that attack the memory-bandwidth wall proven in A1 Rung 1.

Orientation: ``performance/A1_transformer_inference.md`` (spec) ·
``docs/KERNEL_MASTERY_SPEC.md`` §12.5 (the session ladder — the current node).
"""
