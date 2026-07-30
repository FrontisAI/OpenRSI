# Three-Task Concurrent Real Run

This directory preserves display-ready outputs from a real end-to-end run of
the README workflow with three competitions processed concurrently:

- `titanic`
- `spaceship-titanic`
- `house-prices-advanced-regression-techniques`

The run completed with three successful builds, three passing metric checks,
three metadata rows, and three completed quality evaluations. All three tasks
were `recommended`; their overall scores were 4.4, 4.4, and 4.6,
respectively.

## Commands used

The run used one ID for the complete pipeline. The builder's internal batch
name was the fixed directory name `build`, not a second user-facing run name.

```sh
RUN_ID=display-real-3
RUN_ROOT="artifacts/runs/$RUN_ID"
BATCH_ROOT="$RUN_ROOT/build"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/metadata" "$RUN_ROOT/evaluation"

uv run --no-editable openmle-task build \
  --slugs-file examples/slugs.txt \
  --output-root "$RUN_ROOT" \
  --info-csv builder_core/info.csv \
  --batch-name build \
  --max-concurrency 3 \
  --retry 2 \
  --execute \
  > "$RUN_ROOT/logs/build.log" 2>&1

find "$BATCH_ROOT/data" \
  -mindepth 1 \
  -maxdepth 1 \
  -type d \
  -exec basename {} \; \
  | sort > "$BATCH_ROOT/tasks.txt"

metric_failures=0
while IFS= read -r task; do
  if uv run --no-editable openmle-task metric-check \
    --task-dir "$BATCH_ROOT/data/$task" \
    > "$RUN_ROOT/logs/metric-$task.log" 2>&1; then
    echo "metric-check passed: $task"
  else
    echo "metric-check failed: $task" >&2
    metric_failures=$((metric_failures + 1))
  fi
done < "$BATCH_ROOT/tasks.txt"
echo "metric-check failures: $metric_failures"

uv run --no-editable openmle-task overview \
  --tasks-root "$BATCH_ROOT/data" \
  --output-csv "$RUN_ROOT/metadata/overview.csv" \
  --workers 3 \
  > "$RUN_ROOT/logs/overview.log" 2>&1

uv run --no-editable openmle-task evaluate \
  --root-dir "$BATCH_ROOT/data" \
  --task-list "$BATCH_ROOT/tasks.txt" \
  --overview-csv "$RUN_ROOT/metadata/overview.csv" \
  --output-dir "$RUN_ROOT/evaluation" \
  --workers 3 \
  > "$RUN_ROOT/logs/evaluate.log" 2>&1
```

The optional leaderboard branch was not needed for this run. Provider logs
and credentials are not included in the preserved example.

## Preserved outputs

- `task_package/`: generated raw and processed task packages.
- `builder_forge/`: construction descriptions, file inventories, generated
  preparation code, and attempt records.
- `metadata/overview.csv`: the three-row metadata aggregate.
- `evaluation/`: per-task evaluations and the combined JSON/CSV aggregates.
- `tasks.txt`: stable task order used by the batch evaluator.

Machine-specific paths in the copied metadata were normalized to paths within
this example. No evaluation scores or findings were edited.

## Offline verification

These checks require neither Kaggle nor an LLM:

```sh
metric_failures=0
while IFS= read -r task; do
  if uv run --no-editable openmle-task metric-check \
    --task-dir "examples/real-run-3-concurrency/task_package/$task"; then
    echo "metric-check passed: $task"
  else
    echo "metric-check failed: $task" >&2
    metric_failures=$((metric_failures + 1))
  fi
done < examples/real-run-3-concurrency/tasks.txt
echo "metric-check failures: $metric_failures"

mkdir -p artifacts/runs/example-smoke/metadata

uv run --no-editable openmle-task overview \
  --tasks-root examples/real-run-3-concurrency/task_package \
  --output-csv artifacts/runs/example-smoke/metadata/overview.csv \
  --workers 3 \
  --skip-llm

uv run --no-editable openmle-task evaluate \
  --root-dir examples/real-run-3-concurrency/task_package \
  --task-list examples/real-run-3-concurrency/tasks.txt \
  --overview-csv artifacts/runs/example-smoke/metadata/overview.csv \
  --output-dir artifacts/runs/example-smoke/evaluation \
  --workers 3 \
  --local-only
```

Offline evaluation verifies structure and metric execution. It does not
reproduce or replace the preserved online quality scores.
