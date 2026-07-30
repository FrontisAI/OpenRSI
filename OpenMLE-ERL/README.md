# OpenMLE-ERL

OpenMLE-ERL trains the shared Draft, Improve, Debug, and Crossover operator space in two stages: supervised fine-tuning followed by execution-grounded reinforcement learning.

## Training Stages

| Stage | Role | Entry |
| --- | --- | --- |
| **SFT** | Collect and filter executable trajectories, then warm-start the policy model | [`SFT/`](SFT/) |
| **RL** | Optimize operator behavior from sandbox execution rewards | [`RL/`](RL/) |

## Workflow

```text
OpenMLE-Gym tasks
        │
        ▼
SFT rollout collection and data selection
        │
        ▼
Supervised policy warm start
        │
        ▼
Execution-grounded reinforcement learning
        │
        ▼
Frontis-MA1 used by OpenMLE-Evo
```

SFT and RL have separate dependency environments, launchers, external inputs, and validation boundaries. Follow each module's own README rather than carrying configuration from one stage into the other.

Data, model weights, checkpoints, sandbox services, and credentials are external inputs and are not bundled in this repository.

Original OpenMLE material is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Dependencies retain their upstream terms.
