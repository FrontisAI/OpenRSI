# OpenMLE-Gym

OpenMLE-Gym converts Kaggle competitions into standardized machine-learning task packages, extracts task metadata, validates generated metrics, and evaluates package quality.

The maintained interface is the `openmle-task` CLI. Runtime outputs are written under the ignored `artifacts/` directory.

## What the workflow does

The main workflow is:

1. Build task packages from Kaggle competition slugs.
2. Validate each generated metric against its sample submission.
3. Generate task metadata.
4. Evaluate task-package quality.
5. Optionally download public leaderboard CSV files.

Build, overview, metric-check, and evaluation run each task in a disposable child process. A task crash, timeout, or invalid result is recorded for that task without cancelling sibling tasks. `--max-concurrency` and `--workers` control concurrency.

The default `--execution-mode process` preserves compatibility while isolating task failures. `--execution-mode isolated` additionally requires Docker or Podman and an `OPENMLE_GYM_ISOLATED_IMAGE`; it runs generated prepare and metric code without network access and with resource and filesystem restrictions.

## Requirements and installation

The project requires Python 3.10 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/).

One official way to install `uv` on a POSIX-compatible system is:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the complete environment and verify the CLI:

```sh
cd OpenMLE-Gym
uv sync --no-editable --extra all
uv run --no-editable openmle-task --help
```

Use `uv sync --no-editable` without extras only for offline structural smoke checks. Real builds require the `build` dependencies, and default AI quality evaluation requires the `evaluator` dependencies; `--extra all` installs both.

## Credentials

Create a local environment file:

```sh
cp .env.example .env
```

Fill only the variables needed by the stages you will run:

```dotenv
# Kaggle downloads and optional leaderboard access
KAGGLE_CONFIG_DIR="/absolute/path/to/directory-containing-kaggle-json"
# Alternatively:
# KAGGLE_USERNAME=""
# KAGGLE_KEY=""

# OpenAI-compatible model used by Build
OPENMLE_BUILD_LLM_API_KEY=""
OPENMLE_BUILD_LLM_BASE_URL="https://build-provider.example/v1"
OPENMLE_BUILD_LLM_MODEL=""

# Independently configurable OpenAI-compatible model used by Overview
# and Evaluate, including the quality judge
OPENMLE_EVAL_LLM_API_KEY=""
OPENMLE_EVAL_LLM_BASE_URL="https://eval-provider.example/v1"
OPENMLE_EVAL_LLM_MODEL=""
```

`KAGGLE_CONFIG_DIR` must name a directory containing a file named `kaggle.json`, not the JSON file itself. Keep `.env` and `kaggle.json` private; both are ignored by Git.

The two LLM configurations are independent: they may use different keys,
gateways, and models. Build never reads the evaluation-prefixed settings;
Overview and Evaluate never read the build-prefixed settings.

For compatibility, the unprefixed `OPENAI_API_KEY`, `OPENAI_API_BASE`, and
`MODEL` variables remain fallbacks for either stage. When no
`OPENMLE_EVAL_LLM_*` variable is set, the existing Anthropic quality-judge
variables remain supported: `ANTHROPIC_API_KEY` or
`ANTHROPIC_COMPAT_API_KEY`, plus optional `ANTHROPIC_BASE_URL` and
`ANTHROPIC_MODEL`.

The CLI loads `.env` automatically without overriding variables explicitly
provided by the calling environment. Before a long run, verify the configured
services independently:

```sh
uv run --no-editable kaggle competitions files titanic --page-size 1

uv run --no-editable python -c \
  'from dotenv import load_dotenv; load_dotenv(); from openmle_gym.llm_config import build_llm_config; from langchain_openai import ChatOpenAI; c=build_llm_config().require("Build"); print(ChatOpenAI(model=c.model, api_key=c.api_key, base_url=c.base_url).invoke("Reply exactly OK").content)'

uv run --no-editable python -c \
  'from dotenv import load_dotenv; load_dotenv(); from openmle_gym.llm_config import eval_llm_config; from langchain_openai import ChatOpenAI; c=eval_llm_config().require("Evaluation"); print(ChatOpenAI(model=c.model, api_key=c.api_key, base_url=c.base_url).invoke("Reply exactly OK").content)'
```

The preferred evaluation path uses the OpenAI-compatible
`OPENMLE_EVAL_LLM_*` configuration. Anthropic is retained as a compatibility
path for existing installations.

## Inputs

Slug files contain one Kaggle competition slug or competition URL per line. Blank lines and comments beginning with `#` are ignored.

```text
titanic
spaceship-titanic
house-prices-advanced-regression-techniques
```

The repository includes this three-task list at `examples/slugs.txt`.

## Complete real workflow

Every invocation uses a unique run directory so repeated workflows preserve
their own build, logs, metadata, and evaluation outputs. Set `RUN_ID`
explicitly to a meaningful unique value, or use the timestamp default below.
The builder's internal batch name is fixed to `build`; users do not need to
manage a second name.

```sh
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="artifacts/runs/$RUN_ID"
BATCH_ROOT="$RUN_ROOT/build"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/metadata" "$RUN_ROOT/evaluation"
```

### 1. Build three task packages concurrently

```sh
uv run --no-editable openmle-task build \
  --slugs-file examples/slugs.txt \
  --output-root "$RUN_ROOT" \
  --info-csv builder_core/info.csv \
  --batch-name build \
  --max-concurrency 3 \
  --retry 2 \
  --execute \
  > "$RUN_ROOT/logs/build.log" 2>&1
```

The default exit code is nonzero if any task fails, but all sibling task results and successful artifacts are retained. Add `--allow-partial-success` only when a batch containing recorded task failures should still return exit code `0`.

Successful task packages are written to:

```text
artifacts/runs/{run-id}/build/data/{competition}/
```

Create the exact task list for the successful batch:

```sh
find "$BATCH_ROOT/data" \
  -mindepth 1 \
  -maxdepth 1 \
  -type d \
  -exec basename {} \; \
  | sort > "$BATCH_ROOT/tasks.txt"
```

### 2. Validate every generated metric

```sh
metric_failures=0
while IFS= read -r task; do
  if uv run --no-editable openmle-task metric-check \
    --task-dir "$BATCH_ROOT/data/$task" \
    > "$RUN_ROOT/logs/metric-$task.log" 2>&1; then
    echo "metric-check passed: $task"
  else
    echo "metric-check failed: $task (see $RUN_ROOT/logs/metric-$task.log)" >&2
    metric_failures=$((metric_failures + 1))
  fi
done < "$BATCH_ROOT/tasks.txt"
echo "metric-check failures: $metric_failures"
```

Every task is checked even if an earlier metric fails. Each successful result
contains a metric class and a finite score; task-specific submission semantics
remain the responsibility of that metric. Failed checks remain in their
per-task logs and are evaluated independently by the later batch stages.

### 3. Generate metadata concurrently

```sh
uv run --no-editable openmle-task overview \
  --tasks-root "$BATCH_ROOT/data" \
  --output-csv "$RUN_ROOT/metadata/overview.csv" \
  --workers 3 \
  > "$RUN_ROOT/logs/overview.log" 2>&1
```

For an offline metadata check, add `--skip-llm` and use a different output CSV.

### 4. Evaluate task quality concurrently

Reuse the overview generated above:

```sh
uv run --no-editable openmle-task evaluate \
  --root-dir "$BATCH_ROOT/data" \
  --task-list "$BATCH_ROOT/tasks.txt" \
  --overview-csv "$RUN_ROOT/metadata/overview.csv" \
  --output-dir "$RUN_ROOT/evaluation" \
  --workers 3 \
  > "$RUN_ROOT/logs/evaluate.log" 2>&1
```

Default evaluation uses the OpenAI-compatible `OPENMLE_EVAL_LLM_*`
configuration when present; otherwise, it retains the existing Anthropic
quality-judge compatibility path. Deterministic checks parse complete CSV
records, verify package alignment, and treat the metric result only as a
finite-value smoke test. Raw-data usage compares actual raw/train.csv rows
with the combined processed train/test rows; the mirrored private answer is
not double-counted, and an original unlabeled raw test set is tracked as an
intentional exclusion. A provider or malformed-response failure is reported
as an evaluation failure and is never converted into `not_recommended`.

For an offline structural evaluation, add `--local-only` and write to a
separate output directory. Offline mode does not produce quality scores or a
recommendation; it records deterministic validation and marks quality
evaluation as `skipped`.

Expected evaluation outputs are:

```text
artifacts/runs/{run-id}/evaluation/all_results.json
artifacts/runs/{run-id}/evaluation/evaluation_summary.csv
artifacts/runs/{run-id}/evaluation/{task}.json
```

The reused overview remains at
`artifacts/runs/{run-id}/metadata/overview.csv`. If
`--overview-csv` is omitted, evaluation generates
`artifacts/runs/{run-id}/evaluation/overview.csv` itself.

### 5. Optionally download public leaderboards

Leaderboard download is a read-only branch independent of task construction and evaluation. It is not a prerequisite for the main workflow and does not change Kaggle account state.

Dry-run:

```sh
uv run --no-editable openmle-task leaderboard \
  --slugs-file examples/leaderboard-slugs.txt \
  --out-dir "$RUN_ROOT/leaderboards"
```

Real download:

```sh
uv run --no-editable openmle-task leaderboard \
  --slugs-file examples/leaderboard-slugs.txt \
  --out-dir "$RUN_ROOT/leaderboards" \
  --execute \
  > "$RUN_ROOT/logs/leaderboard.log" 2>&1
```

## Completion checks

A complete successful three-task run should satisfy all of the following:

- the build command exits with code `0`;
- `"$BATCH_ROOT/tasks.txt"` contains three unique task names;
- every task directory contains `data/public`, `data/private`, and `utils/metric.py`;
- every metric-check command exits with code `0` and reports a finite score;
- `"$RUN_ROOT/metadata/overview.csv"` contains one row per task;
- `"$RUN_ROOT/evaluation/all_results.json"` and `evaluation_summary.csv` contain one result per task;
- online evaluation rows have `Validation Status=passed` and
  `Evaluation Status=completed`; offline rows have
  `Evaluation Status=skipped` and blank quality-score fields;
- all command logs exist under `"$RUN_ROOT/logs/"`;
- optional leaderboard execution creates one CSV per requested slug.

## Offline smoke test

The preserved three-task example can be checked without Kaggle or LLM access:

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

mkdir -p artifacts/runs/smoke/metadata

uv run --no-editable openmle-task overview \
  --tasks-root examples/real-run-3-concurrency/task_package \
  --output-csv artifacts/runs/smoke/metadata/overview.csv \
  --workers 3 \
  --skip-llm

uv run --no-editable openmle-task evaluate \
  --root-dir examples/real-run-3-concurrency/task_package \
  --task-list examples/real-run-3-concurrency/tasks.txt \
  --overview-csv artifacts/runs/smoke/metadata/overview.csv \
  --output-dir artifacts/runs/smoke/evaluation \
  --workers 3 \
  --local-only
```

## Repository contents

| Path | Purpose |
| --- | --- |
| `openmle_gym/` | Maintained package and `openmle-task` CLI |
| `builder_core/` | LangGraph task-builder core |
| `metadata_pipeline/` | Metadata extraction and quality pipeline |
| `examples/` | Slug inputs and preserved three-task example |
| `scripts/` | Optional thin wrappers around maintained CLI commands |
| `docs/usage.md` | Detailed command and isolation reference |
| `docs/source-manifest.md` | Imported source identity and release boundary |

Original OpenMLE material follows the repository-level [CC BY-NC 4.0 license](../LICENSE). Imported and third-party material retains the terms recorded in [`NOTICE`](../NOTICE).
