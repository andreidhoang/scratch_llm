"""T1 training-ladder instruments — the parts a *comparison* of two training runs needs.

:mod:`scratch_llm.train` runs one loop. This package is what turns two loops into an experiment:

* :mod:`~scratch_llm.training.run_matrix` — a batch schedule that is a function of the seed and
  nothing else (so two arms at one seed see the same data in the same order), bits-per-byte scored
  on a fixed eval stream, and an arm x seed matrix that refuses to compare arms that were not run
  at the same token budget.
* :mod:`~scratch_llm.training.bpb` — bits per byte, against the tokenizer's BYTE count.
* :mod:`~scratch_llm.training.seed_noise` — the seed-to-seed sigma of bpb, and its degrees of
  freedom, which at two seeds is 1.
* :mod:`~scratch_llm.training.fp8_parity` — T1/T-R2's two arms (bf16, fp8) and its entry point.

Nothing outside ``fp8_parity`` is FP8-specific: the same matrix compares an optimizer change, a
parallelism change, or a data change, and the reason it can is that the arms are described by a
config whose fields are diffed before the run (``run_matrix.differing_fields``).

Sibling T1 rungs put their own instruments in this package; import submodules by path rather than
relying on names re-exported here.
"""

from __future__ import annotations
