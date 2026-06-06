"""reasoning_llm — clean-room RLVR engine.

The one number this package exists to measure and protect::

    true_quality_gap = reward - true_quality        (+ hack_rate, kl_train_infer)

Subpackages map to the five layers (CLAUDE.md):
    algos    L5  GRPO / Dr.GRPO advantage + off-policy correction
    rewards  L5  reward shaping + hack detection
    envs     L5  verifier-exploitability dial + true-quality oracle
    rollout  L2  inference/serving client (the kl_train_infer source)
    scaling  L3  IsoFLOP machinery refit to hack_rate vs compute
    data     L4  curation + train<->eval contamination check
    utils    L2  monitors (three-KL logging, HALT@0.10)
"""

__version__ = "0.1.0"
