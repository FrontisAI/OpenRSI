# OpenMLE-ERL: Reinforcement Learning

This module contains the execution-grounded reinforcement-learning implementation used in *Frontis-MA1: Towards Recursive Self-Improvement in Machine Learning Engineering*.

## Scope

OpenMLE-ERL trains the same Draft, Improve, Debug, and Crossover operators used by OpenMLE-Evo at inference time. The release includes four launch profiles, editable configuration templates, OpenMLE-specific training logic, Program Database support, reward post-processing, and evaluation logging.

It does not contain training data, model weights, historical checkpoints, leaderboard assets, sandbox infrastructure, or credentials.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `configs/` | One editable configuration per launch mode |
| `scripts/` | Single-node/multi-node and sync/async launchers |
| `model_configs/` | Released model-parallel settings |
| `train_with_patch.py` | Synchronous SLIME entry |
| `train_async_with_patch.py` | Asynchronous SLIME entry |
| `fully_async_rollout.py` | Asynchronous rollout implementation |
| `program_database.py` | Executed-program storage and parent sampling |
| `docs/` | Complete configuration, launch, semantics, and validation guide |

The Python modules intentionally remain at this directory root because SLIME imports custom functions by module path. Moving them into another package would change the runtime contract.

## Quick Start

```bash
git clone https://github.com/THUDM/slime.git
cd slime
git checkout 680824dd5e01a2e83750bf87fc366ec6fa98766c
cp -a /path/to/OpenMLE/OpenMLE-ERL/RL examples/openmle_rl
python -m pip install -r examples/openmle_rl/requirements.txt

cp examples/openmle_rl/configs/async_single_node.env.example \
  /path/to/my_async_single_node.env
PRECHECK_ONLY=1 \
  bash examples/openmle_rl/scripts/run_openmle_rl_async_single_node.sh \
  /path/to/my_async_single_node.env
```

Read [`docs/usage.md`](docs/usage.md) before configuring or launching training. It records the pinned SLIME dependency, derived batch-size rule, `SAVE_INTERVAL=10` contract, four launch profiles, required external inputs, and validation boundary.

## Validation Status

The released single-node asynchronous profile completed real checkpoint restore, two rollout-and-optimizer steps, post-step weight synchronization to SGLang, and evaluation on eight H200 GPUs. This validates the released path end to end; it is not a claim of exact paper reproduction.

Original OpenMLE material follows the repository-level [CC BY-NC 4.0 license](../../LICENSE). Dependencies retain their upstream terms.
