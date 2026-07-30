# OpenMLE-ERL: Reinforcement Learning — Usage Guide

OpenMLE-ERL is the execution-grounded reinforcement-learning implementation used in *Frontis-MA1: Towards Recursive Self-Improvement in Machine Learning Engineering*. This directory contains the OpenMLE-specific training logic, four launch profiles, and editable configuration templates.

It does **not** contain training data, model weights, historical checkpoints, leaderboard assets, sandbox infrastructure, or credentials.

> **Validated release path.** The single-node asynchronous profile completed real checkpoint restore, two OpenMLE-ERL/Evo rollout-and-optimizer steps, post-step weight synchronization to SGLang, and evaluation on 8 H200 GPUs. This demonstrates that the released training path runs end to end; it does not claim exact paper reproduction.

[Repository overview](../../../README.md) · [Frontis-MA1-30B](https://huggingface.co/FrontisAI/Frontis-MA1-30B) · [Frontis-MA1-35B](https://huggingface.co/FrontisAI/Frontis-MA1-35B)

## Connection to the paper

OpenMLE-ERL trains the same four program-evolution operators used by OpenMLE-Evo at inference time:

| Operator | Training role | Paper probability |
| --- | --- | ---: |
| Draft | Generate a complete solution without a parent | 0.50 |
| Improve | Refine a positively scored parent | 0.17 |
| Debug | Repair an invalid or non-positive-scoring parent | 0.17 |
| Crossover | Recombine two positively scored parents | 0.16 |

The paper configuration uses 16 prompts × 16 samples, up to 24,576 response tokens, and GSPO with execution-grounded reward post-processing. The launchers derive the global batch size from the rollout shape:

```text
GLOBAL_BATCH_SIZE =
  ROLLOUT_BATCH_SIZE × N_SAMPLES_PER_PROMPT ÷ NUM_STEPS_PER_ROLLOUT
```

Every launcher enforces `SAVE_INTERVAL=10`.

## Repository layout

```text
ERL/
├── configs/                 # One complete editable configuration per launch mode
├── scripts/                 # Four user-facing launchers
├── model_configs/           # Model-parallel settings for the released architectures
├── train_with_patch.py      # Synchronous SLIME entry
├── train_async_with_patch.py
├── fully_async_rollout.py
├── generate_mle.py
├── reward_func_utils.py
├── program_database.py
├── patch_eval_logger.py
└── requirements.txt
```

The core Python modules intentionally remain at the directory root because SLIME imports custom functions by module path. Moving them into additional packages would change the runtime import contract.

There is no wrapper/parent split. Each file under `scripts/` is the actual launcher:

```text
configs/<mode>.env
  -> scripts/run_openmle_rl_<mode>.sh
  -> train_with_patch.py or train_async_with_patch.py
  -> SLIME train.py or train_async.py
```

## Runtime prerequisites

- [THUDM/slime](https://github.com/THUDM/slime), commit `680824dd5e01a2e83750bf87fc366ec6fa98766c`
- Linux/amd64 with supported NVIDIA GPUs
- A compatible SLIME image providing PyTorch/CUDA, Ray, SGLang, Megatron-LM, and Transformer Engine
- OpenMLE direct dependencies from [`requirements.txt`](../requirements.txt)

The verified smoke used:

```text
slimerl/slime:qwen35-qwen36-nightly-dev-20260629a-20260701
sha256:9ce561a9b952ba3f3c2102f1158389c5d2e6c6f870b01f54cd80d645405a2097
```

Install the release inside the pinned SLIME checkout:

```bash
git clone https://github.com/THUDM/slime.git
cd slime
git checkout 680824dd5e01a2e83750bf87fc366ec6fa98766c
cp -a /path/to/repository/OpenMLE-ERL/RL examples/openmle_rl
python -m pip install -r examples/openmle_rl/requirements.txt
```

## Choose a launch profile

| Mode | Default model architecture | Topology | Configuration |
| --- | --- | --- | --- |
| Single-node sync | Qwen3-30B-A3B-Thinking-2507 | 1 × 8 GPUs, colocated | [`sync_single_node.env.example`](../configs/sync_single_node.env.example) |
| Single-node async | Qwen3-30B-A3B-Thinking-2507 | 4 training + 4 rollout GPUs | [`async_single_node.env.example`](../configs/async_single_node.env.example) |
| Multi-node sync | Qwen3.6-35B-A3B | 2 × 8 GPUs, colocated | [`sync_multi_node.env.example`](../configs/sync_multi_node.env.example) |
| Multi-node async | Qwen3.6-35B-A3B | 8 training + 8 rollout GPUs | [`async_multi_node.env.example`](../configs/async_multi_node.env.example) |

The single-node asynchronous path has real runtime validation. The other profiles are provided from the same launch chain but should be treated as unvalidated until they complete equivalent real runs.

## Configure

Copy one template outside the repository:

```bash
cp examples/openmle_rl/configs/async_single_node.env.example \
  /path/to/my_async_single_node.env
```

Edit that single file. Users do not need to modify the launcher or Python source.

| Group | Settings to provide |
| --- | --- |
| Data | `PROMPT_DATA`, `EVAL_PROMPT_DATA`, `LEADERBOARD_ROOTS` |
| Model | `MODEL_CONFIG`, `HF_CHECKPOINT`, `REF_LOAD` |
| Execution service | `SANDBOX_BASE_URL`, `SANDBOX_API_KEY`, `HF_ENDPOINT` |
| Hack checking | `GPT_BASE_URL`, `GPT_API_KEY` |
| Outputs | `OUTPUT_ROOT`, `EXPERIMENT_ID` |
| Tracking | `WANDB_KEY`, `WANDB_PROJECT`, `WANDB_MODE` |
| Network | `SOCKET_IFNAME` |
| Resume, if used | `RESUME_MODEL_PATH`, `RESUME_CKPT_STEP`, `LOAD_OPTIMIZER`, `LOAD_RNG`, `MANUAL_DB_PATH` |
| Multi-node only | `MASTER_ADDR`, `HOSTFILE`, `RAY_WORKER_CONTAINER`, `RAY_WORKER_SSH_USER` |

The four templates also expose rollout size, generation length, optimizer settings, reward mapping, evaluation frequency, search policy, Ray topology, and diagnostic controls with explicit defaults.

Fill the three API keys directly in the configuration file. Do not commit a configuration file containing real keys.

## Precheck and launch

Run the non-mutating precheck first:

```bash
PRECHECK_ONLY=1 \
  bash examples/openmle_rl/scripts/run_openmle_rl_async_single_node.sh \
  /path/to/my_async_single_node.env
```

The precheck validates configuration values, local paths, the network interface, dataset metadata, leaderboard roots, checkpoint directories, and the training entry. It does not start Ray or training.

Launch with the same configuration:

```bash
bash examples/openmle_rl/scripts/run_openmle_rl_async_single_node.sh \
  /path/to/my_async_single_node.env
```

Use the matching script and template for the other three modes. Launchers do not kill existing Ray or SGLang processes by default. Set `CLEANUP_EXISTING_RAY=1` only in a dedicated environment after confirming those processes belong to the intended run.

Resume is disabled by default. Set `RESUME_MODEL_PATH` and `RESUME_CKPT_STEP` together. Set `MANUAL_DB_PATH` when continuing a Program Database. Keep `LOAD_OPTIMIZER=0` and `LOAD_RNG=0` for SFT-to-RL starts or cross-topology weight-only loading; use `1/1` only when the checkpoint and current parallel topology are compatible.

## Execution semantics

For each sample:

1. Program Database selects Draft, Improve, Debug, or Crossover state and builds the prompt.
2. SGLang generates code and the hack-check service validates it.
3. Accepted code is submitted to the configured execution service.
4. Execution score, leaderboard context, and reward shaping produce the training reward.
5. Results are persisted to Program Database and training metrics.

The synchronous path completes generation and reward collection before the optimizer step. The asynchronous path uses `fully_async_rollout.generate_rollout_fully_async` and consumes complete groups from the data buffer.

## Real validation

| Item | Validated value |
| --- | --- |
| Hardware | 1 node, 8 × H200 |
| Profile | Single-node asynchronous |
| Rollout shape | 4 prompts × 4 samples, global batch 16 |
| Generation cap | 128 response tokens for a fast execution smoke |
| Checkpoint | Model and optimizer restored from step 669 |
| Result | Evo rollout and optimizer steps 670 and 671 completed |
| Post-step path | Step 670 weights synchronized to SGLang and evaluation completed |

Validation boundaries:

- The checkpoint used TP=4 while the smoke used TP=2, so RNG state was intentionally not restored. This is a runnable weight-and-optimizer continuation check, not a bitwise-identical resume claim.
- The 128-token cap was selected only to validate the training control path quickly; it is not a quality or benchmark configuration.
- Multi-node profiles, checkpoint save/reload, and exact paper-result reproduction are not claimed by this smoke.

## License

Original OpenMLE material is licensed under [CC BY-NC 4.0](../../../LICENSE) for attribution-required, non-commercial use. SLIME and other third-party dependencies retain their upstream terms; see [`NOTICE`](../../../NOTICE).
