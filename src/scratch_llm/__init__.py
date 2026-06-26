"""scratch_llm — a from-scratch implementation of the CS336 language-model stack.

Build a language model end to end, to production engineering standard, owning every layer
from the byte to the RL update. The subpackages map to the five CS336 assignments:

    (A1) substrate   tokenizer.py · model.py · optim.py · train.py · sampling.py · moe.py
    (A2) systems     kernels/ (FlashAttention-2) · rollout/ · utils/monitors.py
    (A3) scaling     scaling/  (IsoFLOP / Chinchilla fits)
    (A4) data        data/     (filtering · dedup · quality classification)
    (A5) alignment   algos/ (SFT · Expert Iteration · GRPO/Dr.GRPO · DPO) · rewards/ · envs/

The engineering disciplines are baked into the tests: loss-at-init ≈ log(vocab),
overfit-one-batch, fixed-seed reproducibility, and (for RL) the three-KL logging convention.
"""

__version__ = "0.1.0"
