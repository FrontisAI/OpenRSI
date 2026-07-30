# Solver Development Guide

This guide provides detailed instructions for developing new solvers for RAD. Solvers are agents that attempt to solve tasks through strategic interaction and exploration.

## Core Concepts

A solver is responsible for:

1. **Generating solutions** for a given task (e.g., writing code, choosing actions)
2. **Interacting with the task** through the `step_task` interface
3. **Evaluating results** and deciding on the next actions
4. **Maintaining state** across solving attempts
5. **Implementing a search strategy** to explore the solution space efficiently

## Operators: The Building Blocks of Solvers

Operators are the fundamental building blocks of RAD solvers. They are functions that transform or generate solutions, enabling modular, composable search strategies.

### What are Operators?

In practical terms, operators in RAD are functions that:
1. Take inputs like task descriptions, existing solutions, or execution results
2. Process these inputs (often using LLMs) to generate new or improved solutions
3. Return structured outputs that can be used by the solver

The core operators of aira-dojo are:
1. [**Draft**](../src/dojo/core/solvers/operators/draft.py): Generates a new solution from scratch

2. [**Improve**](../src/dojo/core/solvers/operators/improve.py): Refines an existing solution

3. [**Debug**](../src/dojo/core/solvers/operators/debug.py): Fixes errors in a broken solution

4. [**Analyze**](../src/dojo/core/solvers/operators/analyze.py): Evaluates a solution and extracts metrics

5. [**Crossover**](../src/dojo/core/solvers/operators/crossover.py): Combines elements from multiple solutions (used in evolutionary solvers)


### Operator Selection Strategies

Different search algorithms use different strategies for selecting which operators to apply and when:

| Search Strategy | Operator Set | Node Selection Strategy |
|-----------------|-------------|-------------------------|
| Greedy | {draft, debug, improve, analyze} | ε-greedy |
| Monte Carlo Tree Search (MCTS) | {draft, debug, improve, analyze} | UCT (Upper Confidence bounds applied to Trees) |
| Evolutionary | {draft, improve, analyze, crossover} | Score-based Sampling |

The operator selection strategy should balance:
- **Exploration**: Discovering diverse solutions (often using draft)
- **Exploitation**: Refining promising solutions (often using improve)


### Using Operators in Solvers

Operators are typically set up in the solver's initialization and then used within its search loop (see `src/dojo/solvers/` for examples):

```python
...

class MyOperatorBasedSolver(Solver):
    def __init__(self, cfg: SolverConfig, task_info: Dict[str, Any]):
        super().__init__(task_info=task_info, **cfg)
        self.setup_operators()
    
    def __call__(self, task, state):
        # ~~~ Main solving loop (e.g., call self.step N times)~~~

    def setup_operators(self):
        """Initialize the operators needed by this solver."""
        # ~~~ Logic to instantiate operators (see src/dojo/solvers/ for inspiration)~~

    def step(self, task, state):
        """Execute one step of the search process."""
        # Decide whether to draft a new solution or improve an existing one
        # ~~~ 1. Add logic to choose which operator to call here (see src/dojo/solvers/ for inspiration) ~~~

        # ~~~ 2. Evaluate the solution using the task's step_task method ~~~

        # ~~~ 3. Log metrics and results using the logger ~~~
...
```

See [src/dojo/solvers/greey/greedy.py](../src/dojo/solvers/greedy/greedy.py) for a complete example of a solver using operators.

### Creating Custom Operators

You can create custom operators for your solver's specific needs:

1. Define the operator function in `src/dojo/core/solvers/operators/`

2. Set up the operator in your solver in `src/dojo/core/solvers/`

3. Prepare your operator configuration files in `src/dojo/configs/solver/operators/` and it's dataclass in `src/dojo/config_dataclasses/operators/` (if needed)

4. Add your operator in the configs. See "Solver Configuration" subsection below for details.

### Solver Configuration

Operators in RAD are configured through a combination of YAML files:

1. Define your dataclass `MySolverConfig` config, inheriting from [MySolverConfig](../src/dojo/config_dataclasses/solver/base.py). Add any additional field your solver needs.

2. The main solver configuration file references operator configuration files:
    ```yaml
    # src/dojo/configs/solver/mlebench/my_solver.yaml
    defaults:
        # reference your operator configurations here as shown below
        - /solver/operators@operators:
            - mlebench/aira_operators/draft    # Enables draft mode for generating initial outputs (tailored for MLE-bench tasks)
            - mlebench/aira_operators/improve  # Enables improvement mode for refining outputs (tailored for MLE-bench tasks)
            - mlebench/aide_operators/analyze  # Enables evaluation mode for assessing outputs  (tailored for MLE-bench tasks)
            - mlebench/aira_operators/debug    # Enables debugging mode for fixing broken solutions (tailored for MLE-bench tasks)

    _target_: dojo.config_dataclasses.solver.my_solver.MySolverConfig

    #~~~ Additional solver-specific configuration here ~~~
    ```

## Tracking Metrics and Logging

Properly track metrics and log progress using the logger provided by the base Solver class:

```python
# Log a message
self.logger.info(f"Generated solution: {solution[:50]}...")

# Log structured data for analysis
self.logger.log(
    {
        "step": self.current_step,
        "solution_length": len(solution),
        "score": validation_fitness,
        "is_valid": is_valid
    },
    "METRICS",  # Tag for filtering logs
    step=self.current_step
)
```

If you want to use all the existing analysis utilities, we recommend looking at how the logs are generated in the existing solvers. Essentially, we always log each node's data to a file called JOURNAL.jsonl. A node is simply a possible solution.


### Viewing Results

Logs are stored by default directory specified by `LOGGING_DIR` your `.env` file.
You can analyze logs in:
```
{LOGGING_DIR}/aira-dojo/user_{USER}_issue_{GIT_ISSUE_ID}/
```
where `LOGGING_DIR` is defined in your `.env` file, `USER` is your username, and `GIT_ISSUE_ID` is defined in your experiment at `metadata.git_issue_id`.

And visualize results using the notebooks in `notebooks/` or with the [UI](../src/dojo/ui/README.md).

## Real Examples

For real-world examples, examine:

- `src/dojo/solvers/greedy/greedy.py`: Implements the Greedy solver (LLM-based with exploration)
- `src/dojo/solvers/mcts/mcts.py`: Implements Monte Carlo Tree Search for solution discovery
- `src/dojo/solvers/evo/evo.py`: Implements an evolutionary search solver