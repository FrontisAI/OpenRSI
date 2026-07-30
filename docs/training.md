# Training Data and Method

This document summarizes the training-scale facts reported for *Frontis-MA1: Towards Recursive Self-Improvement in Machine Learning Engineering*. Operational instructions live in the component guides for [SFT](../OpenMLE-ERL/SFT/README.md) and [RL](../OpenMLE-ERL/RL/README.md).

## OpenMLE-Gym

OpenMLE-Gym contains 5,835 executable tasks:

| Source | Tasks |
| --- | ---: |
| Curated Anchors | 168 |
| Kaggle Dataset tasks | 3,375 |
| Kaggle Competition tasks | 2,292 |
| **Total** | **5,835** |

Of these, 5,375 tasks are used for SFT data collection. Another 460 are reserved for the RL prompt pool, leaving 457 RL prompt tasks after filtering.

## Supervised corpus

The final SFT corpus contains 26,574 examples after exact deduplication and a 32,768-token full-message length filter:

| View | Category | Examples | Share |
| --- | --- | ---: | ---: |
| Supervision type | Full responses | 17,344 | 65.3% |
| Supervision type | Trajectory steps | 9,230 | 34.7% |
| Operator | Draft | 19,567 | 73.6% |
| Operator | Improve | 1,787 | 6.7% |
| Operator | Crossover | 755 | 2.8% |
| Operator | Debug | 4,465 | 16.8% |

## Reinforcement learning

The paper's RL configuration samples Draft/Improve/Debug/Crossover with probabilities `0.50/0.17/0.17/0.16`, uses 16 prompts × 16 samples per rollout, allows responses up to 24,576 tokens, and optimizes with GSPO plus execution-grounded reward post-processing.
