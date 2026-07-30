# Running Experiments

This guide explains how to run experiments with the RAD framework, including configuring, executing, and analyzing results.

## Experiment Types

RAD supports several types of experiments:

1. **Single Run**: Running one solver on one task
2. **Parallel Runner**: Running multiple solver-task-seed pairs in parallel

## Understanding the Configuration System

RAD uses a hierarchical configuration system powered by [Hydra](https://hydra.cc/):

1. **Default Configs**: Located in their respective directories (`src/dojo/configs/solver/`, etc.)
2. **Experiment Configs**: Override defaults with specific settings (`src/dojo/configs/_exp/`)
3. **Command-line Overrides**: Specific parameters can be overridden at runtime

The full configuration flow:

```
Base Defaults → Experiment Overrides → Command-line Overrides → Runtime Config
```

### Configuration Example

For a typical experiment:

```yaml
# From src/dojo/configs/_exp/run_example.yaml
# @package _global_
defaults:
  - override /solver/client: litellm_4o
  - override /interpreter: jupyter
  - override /solver: mlebench/greedy
  - override /task: mlebench/aerial-cactus-identification

metadata:
  git_issue_id: example # Ideally, this should be a number fetched from github issue when running an actual experiment.

solver:
  step_limit: 5
```

Here is a breakdown:
1. This config would typically be called with src/dojo/main_run.py which uses src/dojo/config/default_run.yaml as the base configuration.
2. `override /solver/client` overrides the client with litellm_4o (calls GPT-4o using litellm as the backend for operator calls)
3. `override /interpreter` overrides the interpreter with the jupyter interpreter
4. `override /solver` overrides the solver with the greedy solver (with operators tailored for MLE-bench tasks, see [`src/dojo/configs/solver/mlebench/greedy`](../src/dojo/configs/solver/mlebench/greedy.yaml))
5. `override /task` overrides the task with the aerial-cactus-identification task from MLE-bench, see [`src/dojo/configs/task/mlebench/aerial-cactus-identification`](../src/dojo/configs/task/mlebench/aerial-cactus-identification.yaml)
6. `metadata.git_issue_id` overrides the metadata for the experiment (used to name the folder where the experiment results are stored)
7. `solver.step_limit` overrides the solver's step limit to 5 (i.e. the solver will generate at most 5 nodes in the search tree)

## Running a Single Experiment

To run a single experiment (one solver on one task), use the `dojo.main_run` module:

```bash
python -m dojo.main_run +_exp=<experiment_config> [optional_overrides]
```

This will be referring to the _exp config files present within the run/* folder.
### Example

```bash
# Run Greedy with AIRA operators on an MLEBench task (aerial cactus identification)
python -m dojo.main_run +_exp=run_example logger.use_wandb=False
```

This command:
- Uses the configuration in [`src/dojo/configs/_exp/runner_example`](../src/dojo/configs/_exp/runner_example.yaml)
- Disables Weights & Biases logging by overriding the `use_wandb` field of the logger.

## Running Multiple Experiments in Parallel

To run multiple experiments in parallel on a SLURM cluster, use the `dojo.main_runner_job_array` module:

```bash
python -m dojo.main_runner_job_array +_exp=<runner_config> [optional_overrides]
```

### Example

```bash
# Run a set of experiments in parallel
python -m dojo.main_runner_job_array +_exp=runner_example logger.use_wandb=False launcher.debug=True
```
This command submits multiple jobs to run in parallel using slurm, with each job running a different experiment from the configuration. We set `launcher.debug=True` so the jobs won't actually be submitted

## Creating an Experiment Configuration

To create a new experiment configuration, create a YAML file in `src/dojo/configs/_exp/`:

We usually load in the default configs from the tasks, interpreter, etc to override everything necessary for a specific experiment. Look at the following example:

```yaml
# @package _global_
defaults:
  - override /interpreter: python
  - override /task: mlebench/spaceship-titanic
  - override /solver: mlebench/evo

interpreter:
  working_dir: ${output_dir}/workspace_agent/

```

This configuration file specifies the settings for running a task using a Python interpreter. The configuration uses a Python interpreter, which serves as a simple debugging environment for Python code. The task being executed is spaceship-titanic from the mlebench suite. The solver used is EVO from the mlebench collection. Note that the configuration for EVO is located in the mlebench folder. These overrides can alternatively be placed in this folder if needed. The working directory for the interpreter is set to ${output_dir}/workspace_agent/. For a deeper understanding of the configuration and its components, it is recommended to review the individual configuration files.

## Analyzing Results

Experiment results are logged in several ways:

### 1. Local Logs

Logs are stored by default directory specified by `LOGGING_DIR` your `.env` file. And your log's structure will look like this:

```
{LOGGING_DIR}/aira-dojo/
├── user_{USER}_issue_{GIT_ISSUE_ID}/
│   └── user_{USER}_issue_{GIT_ISSUE_ID}_seed_{SEED}_{run_hash}/
│       ├── checkpoint/      # Checkpoint files
│       ├── dojo_config      # Dojo configuration file
│       ├── env_variables.json # Environment variables used
│       ├── json/        # JSON files with experiment results
```
where `USER` is your username, `GIT_ISSUE_ID` is defined in your experiment at `metadata.git_issue_id`, `SEED` is the random seed used for the experiment and `run_hash` is a unique identifier for the run.
### 2. Weights & Biases

If enabled, results are also logged to Weights & Biases, which provides visualization tools for analyzing experiments.

To enable W&B logging:

```bash
python -m dojo.main_run +_exp=my_experiment logger.use_wandb=True logger.project="my-project"
```

We personally do not use wandb due to rate limiting issues but if you are only running one or two experiments it can be useful to debug.

### 3. Visualization Tools

RAD includes basic utilities for analyzing results (see the [notebooks](../notebooks/)) and visualizing results (see the [UI's README](../src/dojo/ui/README.md) directory). You can use these tools to load experiment results, visualize performance, and compare different runs.

## Common Configurations

### Changing Solver Parameters

To override solver parameters:

```bash
python -m dojo.main_run +_exp=runner_example solver.step_limit=10 solver.debug_prob=0.5
```

### Changing Task Parameters

To override task parameters:

```bash
python -m dojo.main_run +_exp=runner_example task.name=denoising-dirty-documents
```