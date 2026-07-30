# Logging Flow

1. **Initialization**: 
   - Journal object created during solver initialization

2. **Step Execution**:
   - Each search step records detailed information about:
     - Solution generation (draft/debug/improve)
     - Code execution results
     - Analysis of solutions
     - Performance metrics

3. **Node Data Structure**:
   - Each solution iteration is stored as a Node object
   - Nodes track parent-child relationships between solutions
   - Execution results, metrics, prompts, etc are all attached to nodes

4. **Journal Entry Creation**:
   - After each step, node data is manually logged to the journal
   - Current best solution is tracked across iterations
   - Entry is timestamped and assigned a unique ID

5. **Example of calling Logger**:
   ```python
   self.logger.log(
       self.journal.get_node_data(self.current_step) | {"current_best_node": best_node_step},
       "JOURNAL",
       step=self.current_step,
   )
   ```

## Core Data Structures

### Node
Each solution iteration is represented as a Node object containing:
- **Code**: The actual solution implementation
- **Plan**: The strategy or approach for this solution
- **Execution data**: Terminal output, execution time, and exit code
- **Evaluation**: Analysis results and performance metrics
- **Relationships**: Parent-child connections to other solutions
- **Metadata**: Timestamps, unique IDs, and operator information

### Journal
The Journal maintains the collection of all solution nodes and provides:
- Methods for tracking solution lineage (parent-child relationships)
- Filters for draft, buggy, and successful nodes
- Metrics tracking across iterations
- Logic for identifying the best solution

## JSONL to Journal Conversion
The journal entries are initially stored as JSONL (one JSON object per line), then processed into the structured journal.json file with all fields documented below.

## Visualization
The exported journal supports tree visualization to show the evolution of solutions and relationships between different solution attempts. So as long as you log in this way, you get provided benefits. You can use the [UI](../src/dojo/ui/README.md) to visualize the journal.

# Journal Fields Documentation

## Core Fields
- `step`: Integer indicating the current step number in the process. i.e. node number.
- `id`: String UUID identifying the specific log entry/node.
- `plan`: String describing the planned approach or strategy for the current step
- `code`: String containing the actual code implementation
- `metric`: Numerical value representing the performance/validation metric (can be null)
- `metric_maximize`: Boolean indicating if the metric should be maximized (can be null)
- `is_buggy`: Boolean indicating if the implementation contains bugs
- `analysis`: String containing analysis of the execution results

## Execution Information
- `term_out`: String containing the raw terminal output
- `_term_out`: Array of strings containing parsed terminal output lines
- `operators_used`: Array of strings listing the operators used (e.g., "debug", "analysis"). i.e. which operators were used on that node.
- `exec_time`: Float indicating execution time in seconds
- `exit_code`: Integer indicating the process exit code (0 for success)
- `current_best_node`: Integer indicating the best performing node/step thus far.

## Operator Metrics
- `operators_metrics`: Array of objects containing:
  - `usage`: Object containing:
    - `completion_tokens`: Integer count of completion tokens used
    - `prompt_tokens`: Integer count of prompt tokens used
    - `total_tokens`: Integer total token count
    - `latency`: Float indicating response latency in seconds
    - `success`: Boolean indicating if the operation was successful
    - `cost`: Float indicating operation cost
    - `cumulative_num_llm_calls`: Integer count of cumulative LLM calls

## Metric Information found in aux metric info - these fields are specific to mlebench.
- `metric_info/competition_id`: String identifying the competition
- `metric_info/score`: Float indicating the current test score
- `metric_info/gold_threshold`: Float threshold for gold medal
- `metric_info/silver_threshold`: Float threshold for silver medal
- `metric_info/bronze_threshold`: Float threshold for bronze medal
- `metric_info/median_threshold`: Float threshold for median performance
- `metric_info/any_medal`: Boolean indicating if any medal was achieved
- `metric_info/gold_medal`: Boolean indicating if gold medal was achieved
- `metric_info/silver_medal`: Boolean indicating if silver medal was achieved
- `metric_info/bronze_medal`: Boolean indicating if bronze medal was achieved
- `metric_info/above_median`: Boolean indicating if performance is above median
- `metric_info/submission_exists`: Boolean indicating if submission file exists
- `metric_info/valid_submission`: Boolean indicating if submission is valid
- `metric_info/is_lower_better`: Boolean indicating if lower scores are better
- `metric_info/created_at`: ISO timestamp of metric info creation
- `metric_info/submission_path`: String path to submission file

## Temporal Information
- `timestamp`: ISO timestamp indicating when the log entry was created



