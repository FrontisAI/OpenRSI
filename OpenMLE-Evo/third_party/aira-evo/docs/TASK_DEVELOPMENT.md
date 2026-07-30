# Task Development Guide

This guide explains how to develop new tasks for the RAD framework. Tasks define the problems that solvers attempt to solve, including the environment, interactions, and evaluation criteria.

## Core Concepts

- **Task**: A specific problem or challenge that the AI agent (solver) is designed to solve. Each task has a defined execution environment, solver action space, and evaluation function.
- **Solver**: An AI agent that attempts to solve a given task. A solver is composed of:
    - **Operators**: Functions that are used to generate new solutions (e.g., a call to an LLM with a specific prompt and some context).
    - **Search Strategy**: The method used to explore the solution space and orchestrate the execution of operators (e.g., greedy search, evolutionary search, Monte Carlo Tree Search)
- **Interpreter**: The component that executes the solver's actions within a defined environment (e.g., Bash, Python Kernel) and returns observations/results. The Task interacts heavily with the Interpreter.

## Task Class Structure

Your task class should inherit from `dojo.core.tasks.base.Task` and implement the following abstract methods (see ~):

```python
from dojo.core.tasks.base import Task
from dojo.config_dataclasses.task.base import TaskConfig
from dojo.core.interpreters.base import Interpreter
from typing import Dict, Any, Tuple, Optional

class MyNewTask(Task):
    def __init__(self, **cfg: TaskConfig):
        """
        Initialize task resources based on configuration (self.cfg).
        Load instructions, data paths, evaluation parameters, etc.
        """
        super().__init__(**cfg)
        # ~~ Additional initialization logic here, e.g., loading data, setting up paths ~~

    def prepare(self, **task_args: Dict) -> Tuple[Dict, Dict]:
        """
        Set up the environment before the solver starts.
        Receives initial arguments like the solver_interpreter instance.
        Should return:
          - state (Dict): Initial state dictionary, often including the interpreter.
          - task_info (Dict): Information for the solver (e.g., task description, evaluation goal).
        """
        
        # ~~ Any additional preparation logic here, e.g., necessary info for solver (to be passed in task_info) ~~
        # ~~ or info related to state to be passed to the solver ~~
        task_info = {
            "TASK_DESCRIPTION": self.task_description, 
            "lower_is_better": False,  # Important for solvers like Greedy
        }
        state = {}
        return state, task_info

    def step_task(self, state: Dict, action: Any) -> Tuple[Dict, Dict]:
        """
        Process a single action from the solver.
        This is the core interaction loop.
        Receives current state and the solver's action.
        Should return:
          - state (Dict): Updated state (can be the same if stateless).
          - outcome (Dict): Result of the action (e.g., execution output, fitness metrics).
        """
        # example of accessing the interpreter from state
        interpreter = state['interpreter']
        
        # ~~ All logic related to evaluating a submission here. You could alss call evaluate_fitness here~~
        
        # Prepare outcome to return back to solver
        outcome = {
            "EXECUTION_OUTPUT": exec_result,  # Required by all solvers
            # The following fields are optional depending on your task/solver:
            "VALIDATION_FITNESS": score,      # Some solvers can calculate their own validation if not provided
            "VALID_SOLUTION": True,           # Indicates if the solution meets validity criteria
            "AUX_EVAL_INFO": {}               # Additional metrics/information for analysis
        }
        return state, outcome

    def evaluate_fitness(
        self,
        solution: Optional[Any] = None,
        state: Optional[Dict] = None,
        interpreter: Optional[Interpreter] = None,
        aux_info: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the final performance. Can be called by the framework or internally by step_task.
        Receives optional solution, state, interpreter, and auxiliary info.
        Should return the final fitness score(s) or evaluation results.
        """
        # ~~ logic to evaluate a solution ~~

    def close(self, state: Dict) -> None:
        """
        Clean up resources (e.g., close interpreter session).
        """
        # ~~ Clean up ressources if needed here (e.g., close interpreter if needed). Called at the end of a run ~~
```

## Interaction with the Interpreter

The `Interpreter` instance, typically accessed via `state['interpreter']`, is essential for task execution. Key methods include:
- **`interpreter.run(code_or_command: str, file_name: Optional[str] = None, persist_file: bool = False, ...)`**:
  - `code_or_command`: Executes Python code or shell commands (prefixed with `!`).
  - `file_name`: Saves the code to this file if provided, useful for multi-line scripts.
  - `persist_file`: Keeps the file after execution if `True`.
  - Returns an `ExecutionResult` with `term_out`, `exit_code`, etc.
- **`interpreter.fetch_file(remote_path: str)`**:
  - Copies a file from the interpreter's environment to the local filesystem.
  - Returns the local `Path` object if successful, `None` otherwise.
- **`interpreter.working_dir`**: Path to the working directory within the interpreter.
- **`interpreter.close()` / `interpreter.cleanup_session()`**: Cleans up interpreter resources.


## How Solvers Use Task Results (Greedy Example)

Understanding how solvers like Greedy use task results can help design better tasks:

1.  **Greedy's flow**:
    -   Generates a solution (drafting, improving, or debugging)
    -   Calls `task.step_task(state, code)` with the solution code
    -   Analyzes the returned `eval_result` dictionary to:

2.  **Task's responsibility**:
    -   Provide clear validation feedback so solvers can improve
    -   Return consistent fitness metrics that solvers can optimize
    -   Signal buggy or invalid solutions clearly

## Configuration (`self.cfg`)

Tasks are configured via Hydra YAML files (`scr/dojo/configs`) and their dataclass (`src/dojo/config_dataclasses`). Access configuration parameters within your task class using `self.cfg.parameter_name`. Define necessary parameters like data paths, evaluation script paths, specific task settings, etc.

## Debugging Task Issues

Common issues when developing tasks include:

1. **Interpreter Path Issues**: Use `Path(interpreter.working_dir).resolve()` to get absolute paths
2. **File Permission Problems**: Ensure interpreter has permissions to read/write files
3. **Execution Timeouts**: Adjust `execution_timeout` in your the solver configuration
4. **Missing Dependencies**: Ensure required packages are available in the interpreter
5. **Parse Errors**: Include robust parsing of output files and execution results

For debugging:
- Inspect the `EXECUTION_OUTPUT` to see exact stdout/stderr from the interpreter

## Examples

For examples of task implementations, see:

- `src/dojo/tasks/mlebench/`: Machine Learning Engineering benchmark tasks (local evaluation pattern)

These examples demonstrate different patterns for task implementation. 