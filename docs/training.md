# Training Data and Method

This document summarizes the training-scale facts reported for *Frontis-MA1: Towards Recursive Self-Improvement in Machine Learning Engineering*. Operational instructions live in the component guides for [SFT](../OpenMLE-ERL/SFT/README.md) and [RL](../OpenMLE-ERL/RL/README.md).

## OpenMLE-Gym

OpenMLE-Gym contains 5,758 quality-gated executable tasks:

| Source | Tasks |
| --- | ---: |
| Curated Anchors | 156 |
| Kaggle Dataset tasks | 3,362 |
| Kaggle Competition tasks | 2,240 |
| **Total** | **5,758** |

## Supervised corpus

The released SFT corpus contains 26,259 examples after exact deduplication and a 32,768-token full-message length filter:

| View | Category | Examples | Share |
| --- | --- | ---: | ---: |
| Supervision type | Full responses | 17,245 | 65.7% |
| Supervision type | Trajectory steps | 9,014 | 34.3% |
| Operator | Draft | 19,436 | 74.0% |
| Operator | Improve | 1,741 | 6.6% |
| Operator | Crossover | 742 | 2.8% |
| Operator | Debug | 4,340 | 16.5% |

## Reinforcement learning

The paper's RL configuration samples Draft/Improve/Debug/Crossover with probabilities `0.50/0.17/0.17/0.16`, uses 16 prompts × 16 samples per rollout, allows responses up to 24,576 tokens, and optimizes with GSPO plus execution-grounded reward post-processing.
