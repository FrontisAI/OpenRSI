# Project Structure

This document provides an overview of the RAD (Research Agent Dojo) project structure to help you navigate and understand the codebase.

## Directory Overview

```
aira-dojo/
├── src/
|   |──dojo/               # Main package directory
│     ├── core/              # Core framework components
│     ├── solvers/           # Agent solver implementations
│     ├── tasks/             # Task implementations
│     ├── utils/             # Utility functions
│     ├── configs/           # Configuration files (YAML)
│     ├── config_dataclasses/  # Configuration dataclasse, the yaml configs are loaded into these
│     ├── analysis_utils/    # Analysis, plotting and visualization tools
│     ├── ui/                # User interface components AKA Dashboard
│     ├── main_run.py        # Entry point for running single experiments
│     ├── main_runner.py     # Entry point for running multiple experiments
│     └── main_runner_job_array.py  # Entry point for distributed experiments
|   |──aira_core/            # Minimal package containing the base dataclass for config
├── notebooks/             # Jupyter notebooks for analysis, demos, and playing with raw experiment data
├── requirements.txt       # Dependency requirement of aira-dojo
└── .streamlit/            # Streamlit configuration
...
```

## Key Components

### Core (src/dojo/core/)

The `core` directory contains the fundamental building blocks of the RAD framework:

- **tasks/**: Base classes and interfaces for defining tasks
- **solvers/**: Base classes and interfaces for solvers
- **interpreters/**: Components that execute solver actions in the environment
- **runners/**: Infrastructure for running experiments

### Solvers (src/dojo/solvers/)

This directory contains different solver implementations:

- **greedy/**: Implementation of the greedy solver
- **evo/**: Evolutionary search Solver
- **mcts/**: Monte Carlo Tree Search Solver

### Tasks (src/dojo/tasks/)

Task implementations for different domains:
- **mlebench/**: MLE-bench

### Configs (src/dojo/configs/)

Configuration files for different experiment setups:

- **_exp/**: Experiment-specific configurations (this combines the configuration files below to create a full experiment)
- **benchmarks/**: Benchmark configurations (aggregates tasks configurations for benchmarking)
- **interpreter/**: Interpreter configurations
- **solver/**: Solver configurations (includes configurations of search strategies, operators, llm client etc.)
- **launcher/**: Launcher configurations for running experiments
- **logger/**: Logger configurations
- **metadata/**: Metadata configurations for experiments
- **task/**: Task configurations
- **default_run.yaml**: Default configuration file for single runs (default for src/dojo/main_run.py)
- **default_runner.yaml**: Default configuration file for running multiple experiments in parrallel (default for src/dojo/main_runner_job_array.py)

### Configs (src/dojo/config_dataclasses/)
Dataclasses that deserialize YAML configurations from `src/dojo/configs/`.

- **client/**: LLM clients dataclass configurations. Used to configure the backend
- **interpreter/**: Interpreter dataclass configurations 
- **launcher/**: Launcher dataclass configurations
- **llm/**: LLM dataclass configurations (prompts, temperature etc.)
- **omegaconf/**: OmegaConf custom resolvers
- **operators/**: Operator dataclass configurations
- **solver/**: Solver dataclass configurations
- **task/**: Task dataclass configurations
- **benchmark.py**: Benchmark dataclass configuration
- **logger.py**: Logger dataclass configuration
- **run.py**: Dataclass configuration of a single experiment (for src/dojo/main_run.py)
- **runner.py**: Dataclass configuration for running multiple experiments in parrallel (for src/dojo/main_runner_job_array.py)
- **utils.py**: Utils for dataclasses

### Utilities (src/dojo/utils/)

Helper functions and utilities used throughout the codebase e.g. the logger and text parsing functions.

## Entry Points

The framework has several entry points for running experiments:

1. **src/dojo/main_run.py**: Used for running a single experiment with one solver on one task
2. **src/dojo/main_runner_job_array.py**: Used for running multiple experiments in parallel where each is a separate job on slurm.

## Configuration System

RAD uses [Hydra](https://hydra.cc/) for configuration management. Configuration files are located in the `src/dojo/configs/` directory and are organized by experiment type.

Key configuration patterns:
- Use `+_exp=<path>` to select experiment configuration
- Use `logger.use_wandb=<bool>` to toggle Weights & Biases logging
- Use `launcher.debug=<bool>` to toggle debug mode

## Adding New Components

### Adding a New Solver

New solvers should be added under `src/dojo/solvers/` in their own directory. See the [Solver Development Guide](./SOLVER_DEVELOPMENT.md) for details.

### Adding a New Task

New tasks should be added under `src/dojo/tasks/` in their own directory. See the [Task Development Guide](./TASK_DEVELOPMENT.md) for details.

## Key Files and Their Purposes

### Core Framework

- `src/dojo/core/tasks/base.py`: Abstract base classes for tasks
- `src/dojo/core/solvers/base.py`: Abstract base classes for solvers
- `src/dojo/core/interpreters/base.py`: Abstract base classes for interpreters
- `src/dojo/core/tasks/constants.py`: Constants used for task-solver communication
- `src/dojo/core/solvers/utils/journal.py`: Solution tracking and history utilities
- `src/dojo/core/solvers/operators/`: Operator functions used by solvers

### Utilities

- `src/dojo/utils/logger.py`: Logging utilities for structured logging
- `src/dojo/utils/code_parsing.py`: Utilities for parsing code from text

### Interpreters

- `src/dojo/core/interpreters/python.py`: Simple Python code execution
- `src/dojo/core/interpreters/jupyter/jupyter_interpreter.py`: Jupyter notebook execution

## Execution Flow

Understanding how components interact:

1. **Runner (`src/dojo/main_run.py`)**: 
   - Parses configuration
   - Instantiates task, solver, and interpreter
   - Manages execution lifecycle

2. **Task Preparation**:
   - `task.prepare()` sets up the environment
   - Returns initial state and task information

3. **Solver Execution**:
   - Solver receives task information
   - Iteratively generates solutions via `solver(task, state)`

4. **Evaluation**:
   - Each solution is evaluated using `task.step_task()`
   - Final performance measured with `task.evaluate_fitness()`

5. **Results Collection**:
   - Logging and saving results
   - Analysis of solver performance
