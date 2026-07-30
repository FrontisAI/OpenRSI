# OpenMLE-Gym — Usage Guide

OpenMLE Gym turns Kaggle competitions into standardized machine-learning task packages, then extracts task metadata and runs quality checks over the generated packages.

The release workflow is:

1. Download public leaderboard CSV files.
2. Build standardized task packages from Kaggle competition slugs.
3. Generate metadata with the bundled metadata pipeline.
4. Evaluate task package quality.

External actions are opt-in. Commands that mutate a Kaggle account, download data, or run the task builder require `--execute`. Metadata and evaluation use LLM calls by default; pass `--skip-llm` or `--local-only` when you need an offline check. `build --max-concurrency`, `overview --workers`, and `evaluate --workers` expose the main concurrency controls.

Batch commands continue processing remaining items when one item fails. By default, `build` and `leaderboard` return a non-zero exit code after printing results if any item failed. Pass `--allow-partial-success` when you want partial failures recorded in JSON but still want process exit code `0`.

Task-building, overview, metric-check, and evaluation work runs in one disposable child process per task. `--max-concurrency` and `--workers` limit how many task processes run at once; a task crash or timeout is recorded for that task without cancelling siblings. Use `--task-timeout` to change the per-task limit.

`--execution-mode process` is the compatibility default. `--execution-mode isolated` additionally requires Docker or Podman and an `OPENMLE_GYM_ISOLATED_IMAGE` containing the locked OpenMLE-Gym package and dependencies. Generated prepare code and task metric code then run without network access, as a non-root user, with only the current task inputs mounted read-only and its explicit outputs writable. Download and LLM stages keep their existing behavior in the task process. If the runtime or image is unavailable, that task fails rather than falling back to in-process execution.

## Repository Layout

- `openmle_gym/`: maintained Python package and `openmle-task` CLI.
- `builder_core/`: LangGraph task-builder core used by `openmle-task build`.
- `metadata_pipeline/`: metadata extraction pipeline used by `openmle-task overview` and `openmle-task evaluate`.
- `examples/`: reusable smoke-test inputs, slug lists, and a three-task real run example generated with `--max-concurrency 3`.
- `scripts/`: optional command wrappers for leaderboard download, metadata generation, and evaluation.
- `pyproject.toml` and `uv.lock`: Python package metadata and locked dependency environment.
- `.env.example`: credential template. Copy it to `.env` locally; never commit real credentials.

Runtime outputs are intentionally not stored in the repository. The documented commands write them under `artifacts/`, which is ignored by Git.

## Installation

Install the full environment:

```bash
uv sync --no-editable --extra all
```

For local smoke tests that do not contact Kaggle or an LLM:

```bash
uv sync --no-editable
```

## Credentials

Copy `.env.example` to `.env`, then fill only the credentials required by the stages you plan to run.

```bash
cp .env.example .env
```

Required variables by stage:

- Kaggle downloads and leaderboards: `KAGGLE_USERNAME` and `KAGGLE_KEY`, or `KAGGLE_CONFIG_DIR` pointing to a directory containing `kaggle.json`.
- Task building: `OPENMLE_BUILD_LLM_API_KEY`, `OPENMLE_BUILD_LLM_BASE_URL`, and `OPENMLE_BUILD_LLM_MODEL`.
- Metadata and quality evaluation: `OPENMLE_EVAL_LLM_API_KEY`, `OPENMLE_EVAL_LLM_BASE_URL`, and `OPENMLE_EVAL_LLM_MODEL`.
- The two stage configurations may use different OpenAI-compatible gateways or models. Legacy `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `MODEL` values remain fallbacks for both stages.
- Existing Anthropic quality-judge configurations remain supported when no `OPENMLE_EVAL_LLM_*` variable is set: `ANTHROPIC_API_KEY` or `ANTHROPIC_COMPAT_API_KEY`, plus optional `ANTHROPIC_BASE_URL` and `ANTHROPIC_MODEL`.

The CLI loads `.env` automatically when `python-dotenv` is installed.

## Inputs

Slug files are plain text, one Kaggle competition slug or URL per line. Blank lines and lines starting with `#` are ignored.

```text
titanic
https://www.kaggle.com/competitions/spaceship-titanic
```

Included examples:

- `examples/leaderboard-slugs.txt`
- `examples/slugs.txt`
- `examples/real-run-3-concurrency/`: display-ready artifacts from a real
  three-task concurrent end-to-end run, including generated task packages,
  construction records, metadata, and quality-judge outputs. Provider logs,
  credentials, and the optional leaderboard branch are not preserved.

## Artifact Layout

The documented workflow gives every invocation a unique run directory:

```text
artifacts/runs/{run-id}/
├── build/
│   ├── data/
│   ├── forge/
│   └── tasks.txt
├── logs/
├── metadata/
├── evaluation/
└── leaderboards/
```

This keeps repeated runs independent and prevents logs, metadata, evaluation
aggregates, and per-task JSON files from overwriting or mixing with earlier
runs. The internal builder batch is always named `build`, so no additional
user-facing batch name is needed.

`artifacts/` is ignored by Git.

## Full Workflow

Choose one unique run ID and reuse these variables for every stage:

```bash
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="artifacts/runs/$RUN_ID"
BATCH="$RUN_ROOT/build"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/metadata" "$RUN_ROOT/evaluation"
```

### 1. Download Leaderboards

Dry-run:

```bash
uv run --no-editable openmle-task leaderboard \
  --slugs-file examples/leaderboard-slugs.txt \
  --out-dir "$RUN_ROOT/leaderboards"
```

Real download:

```bash
uv run --no-editable openmle-task leaderboard \
  --slugs-file examples/leaderboard-slugs.txt \
  --out-dir "$RUN_ROOT/leaderboards" \
  --execute
```

Each leaderboard is saved as
`artifacts/runs/{run-id}/leaderboards/{slug}.csv`.

### 2. Build Task Packages

Dry-run:

```bash
uv run --no-editable openmle-task build \
  --slugs-file examples/slugs.txt \
  --output-root "$RUN_ROOT" \
  --batch-name build \
  --info-csv builder_core/info.csv \
  --max-concurrency 3
```

Real build:

```bash
uv run --no-editable openmle-task build \
  --slugs-file examples/slugs.txt \
  --output-root "$RUN_ROOT" \
  --batch-name build \
  --info-csv builder_core/info.csv \
  --max-concurrency 3 \
  --retry 2 \
  --execute
```

The builder writes task packages to
`artifacts/runs/{run-id}/build/data/{competition}/`.

By default, the builder keeps each task package's `raw/` directory. To remove raw files after a successful build while retaining a lightweight inventory, use `--delete-raw`:

```bash
uv run --no-editable openmle-task build \
  --slugs-file examples/slugs.txt \
  --output-root "$RUN_ROOT" \
  --batch-name build \
  --info-csv builder_core/info.csv \
  --max-concurrency 3 \
  --delete-raw \
  --execute
```

When `--delete-raw` is used, the builder copies
`"$BATCH/forge/{competition}/fileinfo.txt"` to
`"$BATCH/data/{competition}/raw.txt"`, then deletes the task's `raw/`
directory.

After a build, create a task list for the batch:

```bash
find "$BATCH/data" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort > "$BATCH/tasks.txt"
```

### 3. Generate Metadata

Default metadata generation calls the bundled metadata pipeline category, CPU/GPU, and metric-validation steps. `--workers` controls chunked concurrency for local package inspection, task category classification, CPU/GPU classification, and sample-submission metric validation. Output rows stay sorted by task name, so changing `--workers` does not change the CSV schema or row order.

```bash
uv run --no-editable openmle-task overview \
  --tasks-root "$BATCH/data" \
  --output-csv "$RUN_ROOT/metadata/overview.csv" \
  --workers 3
```

Offline metadata check:

```bash
uv run --no-editable openmle-task overview \
  --tasks-root "$BATCH/data" \
  --output-csv "$RUN_ROOT/metadata/overview-local.csv" \
  --workers 3 \
  --skip-llm
```

The CSV includes task name, modality, task type, raw size, final data size, CPU/GPU suitability, metric validation output, and package path.

### 4. Evaluate Task Quality

Default evaluation applies metadata pipeline output, executes the sample
submission metric, and uses the OpenAI-compatible `OPENMLE_EVAL_LLM_*`
configuration when present. If those variables are absent, it retains the
existing Anthropic quality-judge compatibility path. Install the evaluator
dependencies with `uv sync --no-editable --extra evaluator` or
`uv sync --no-editable --extra all` before running the default judge.
`--workers` controls chunked concurrency for task inspection and quality
judging. Final `all_results.json` and `evaluation_summary.csv` follow the input
`task-list` order, so worker count changes should not change task ordering.

Before semantic judging, deterministic validation parses complete logical CSV
records and checks package structure, target exposure, and finite metric
execution. Submission-to-answer schema and identifier semantics belong to the
task-specific metric's `validate_submission()` and `evaluate()` methods rather
than a generic column-equality rule. The random sample submission is a schema
and execution fixture; its score is not treated as model quality.
Raw-data usage is based on actual raw CSV evidence and labeled-row
conservation across processed train/test outputs; private answers are not
double-counted, and original unlabeled test rows are recorded separately.
Provider failures and malformed judge responses are recorded as evaluation
failures instead of being presented as task-quality rejections.

```bash
uv run --no-editable openmle-task evaluate \
  --root-dir "$BATCH/data" \
  --task-list "$BATCH/tasks.txt" \
  --overview-csv "$RUN_ROOT/metadata/overview.csv" \
  --workers 3 \
  --output-dir "$RUN_ROOT/evaluation"
```

Offline structural evaluation:

```bash
uv run --no-editable openmle-task evaluate \
  --root-dir "$BATCH/data" \
  --task-list "$BATCH/tasks.txt" \
  --overview-csv "$RUN_ROOT/metadata/overview-local.csv" \
  --workers 3 \
  --output-dir "$RUN_ROOT/evaluation-local" \
  --local-only
```

Offline mode performs deterministic validation and the metric smoke test only.
It does not run an alternative scoring heuristic: quality-score and
recommendation fields remain blank, while `Evaluation Status` is `skipped`.
If deterministic validation fails, online evaluation applies the hard
`not_recommended` gate; offline evaluation reports validation failure without
inventing quality scores.

Outputs:

- `artifacts/runs/{run-id}/evaluation/all_results.json`
- `artifacts/runs/{run-id}/evaluation/evaluation_summary.csv`
- `artifacts/runs/{run-id}/evaluation/{task}.json`

`evaluation_summary.csv` uses the validation and quality-status table format:

```text
Task Name, Overall Score, Recommendation, Task Validity, Data Sufficiency,
Raw Data Usage, Task Complexity, Data Quality, Train Rows, Test Rows,
Major Issues, Validation Status, Evaluation Status, __source_file,
__source_group
```

## Local Smoke Test

These checks require no Kaggle account and no LLM:

```bash
uv run --no-editable openmle-task build --slugs-file examples/slugs.txt --output-root artifacts/runs/smoke --info-csv builder_core/info.csv --batch-name build --max-concurrency 3
while read -r task; do
  uv run --no-editable openmle-task metric-check --task-dir "examples/real-run-3-concurrency/task_package/$task"
done < examples/real-run-3-concurrency/tasks.txt
mkdir -p artifacts/runs/smoke/metadata
uv run --no-editable openmle-task overview --tasks-root examples/real-run-3-concurrency/task_package --output-csv artifacts/runs/smoke/metadata/overview.csv --workers 3 --skip-llm
uv run --no-editable openmle-task evaluate --root-dir examples/real-run-3-concurrency/task_package --task-list examples/real-run-3-concurrency/tasks.txt --overview-csv artifacts/runs/smoke/metadata/overview.csv --workers 3 --output-dir artifacts/runs/smoke/evaluation --local-only
uv run --no-editable openmle-task leaderboard --slugs-file examples/leaderboard-slugs.txt --out-dir artifacts/runs/smoke/leaderboards
```

## Script Wrappers

The canonical interface is `openmle-task`. The `scripts/` directory also provides thin Python wrappers for users who prefer file-based commands:

- `scripts/download_leaderboard.py` runs `openmle-task leaderboard`.
- `scripts/generate_overview.py` runs `openmle-task overview`.
- `scripts/evaluate.py` runs `openmle-task evaluate`.

Use `openmle-task` for new workflows.
