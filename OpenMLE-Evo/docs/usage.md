# OpenMLE-Evo: MLE-Bench Run Guide

NatureBench Lite-v2 uses the same runtime but has its own data, eval service, and Docker/SCM configuration; see [`../benchmarks/naturebench_lite_v2/RUNNING.md`](../benchmarks/naturebench_lite_v2/RUNNING.md).

## 1. Scope of This Runtime

This directory provides the search and evaluation orchestration code for OpenMLE-Evo. It does not launch or serve models, prepare MLE-Bench data, or ship sandbox images. Before running, you must already have:

1. Python 3.11 or 3.12;
2. an OpenAI-compatible model service;
3. a GPU/CPU sandbox compatible with the `/api/v1/jobs` protocol;
4. the evaluation parquet, prepared task data, and leaderboard metadata;
5. the corresponding sandbox API keys.

The standard and multi-GPU profiles share the same code. The standard profile runs a synchronous generation loop; the multi-GPU profile runs multiple async steady-state workers in a single task process, with a sandbox router assigning the actual GPU workers.

## 2. Installation

```bash
cd /path/to/repository/OpenMLE-Evo
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs the current `tts_search` and the vendored `third_party/aira-evo` in editable mode. The launch scripts also prepend the current directory to `PYTHONPATH`, so a reused virtualenv cannot accidentally load same-named packages from another worktree.

## 3. Environment Configuration

```bash
cp .env.example .env
```

The following fields must be edited:

| Variable | Purpose |
| --- | --- |
| `OPENMLE_EVAL_DATA` | Absolute path to the evaluation parquet |
| `OPENMLE_LEADERBOARD_DIR` | Leaderboard metadata directory |
| `OPENMLE_SUBMIT_DATA_DIR_ROOT` | Root of the prepared task data used for final submission scoring |
| `SGLANG_BASE_URL` | OpenAI-compatible model API; must end with `/v1` |
| `OPENMLE_MODEL_ID` | Model name returned by `/v1/models` or accepted by the service |
| `PRIMARY_KEY` | Model API key; set to `EMPTY` for unauthenticated local services |
| `SANDBOX_URL` | Direct sandbox endpoint for standard mode |
| `SANDBOX_ROUTER_URL` | Sandbox router for multi-GPU mode |
| `SANDBOX_CPU_API_KEY` | CPU sandbox key |
| `SANDBOX_GPU_API_KEY` | GPU sandbox/router key |

The default configuration targets `Qwen/Qwen3-30B-A3B-Thinking-2507` and carries the `extra_body` required for Qwen thinking. When switching to an incompatible model, add or modify the configs under `tts_search/configs/litellm/`.

Secure defaults:

- `sandbox.verify_tls=true`; HTTPS sandboxes verify certificates;
- `sandbox.trust_model_validation_score=false`; self-valid stdout scores are only
  recorded in `raw_scores`, and search selection uses the validation/score
  returned by the sandbox;
- set `sandbox.trust_model_validation_score=true` explicitly only when a
  reproduction experiment specifically requires the legacy self-valid semantics;
- the resolved runner config writes API key/token/password/secret fields as
  `null`; the actual model key is passed only through the subprocess environment.

The evaluation parquet requires at least:

- a `prompt` column: the system/user message sequence;
- a `metadata` column: containing `task_name`, `uuid`, `task`, `cpu_gpu`, `data_dir`, `higher_is_better`, and the score-range fields.

## 4. Service Health Checks

Model:

```bash
curl -fsS "${SGLANG_BASE_URL}/models"
```

Sandbox router:

```bash
curl -fsS "${SANDBOX_ROUTER_URL}/health"
curl -fsS \
  -H "X-API-Key: ${SANDBOX_GPU_API_KEY}" \
  "${SANDBOX_ROUTER_URL}/api/v1/workers/status"
```

## 5. Standard Mode

```bash
./scripts/run_standard.sh
```

Run only specific tasks:

```bash
./scripts/run_standard.sh \
  'search.runner.task_list=[spooky-author-identification]'
```

Standard mode pins:

```yaml
execution_mode: generation
async_workers: 1
```

## 6. Multi-GPU Mode

```bash
AIRAEVO_WORKERS=8 ./scripts/run_multi_gpu.sh
```

This mode explicitly sets:

```yaml
execution_mode: async_steady_state
async_workers: ${AIRAEVO_WORKERS}
async_sandbox_urls:
  - ${SANDBOX_ROUTER_URL}
```

Multiple workers generate and submit sandbox jobs concurrently, but the Journal, SolutionsDatabase, strategy board, and checkpoints are still updated through a single-writer commit path. When completions arrive out of order, each step uses the node's own `attempt_id`, `worker_id`, `gpu_index`, and `sandbox_url` rather than guessing from the global state at commit time.

When a single async attempt hits a transient exception, it retries with exponential backoff instead of immediately allocating a new attempt id. This is tunable via `async_worker_max_retries` and `async_worker_retry_backoff_secs`. Time spent waiting on GPU/resource pools does not count against the effective search budget; `max_wall_time_secs` remains the hard cap on total wall-clock time.

## 7. Minimal Smoke Test

Standard profile:

```bash
OPENMLE_CONFIG_NAME=experiment/openmle_evo_smoke \
./scripts/run_standard.sh \
  'search.runner.task_list=[spooky-author-identification]'
```

Two-worker multi-GPU:

```bash
OPENMLE_CONFIG_NAME=experiment/openmle_evo_smoke \
AIRAEVO_WORKERS=2 \
./scripts/run_multi_gpu.sh \
  'search.runner.task_list=[spooky-author-identification]'
```

Success criteria:

1. `runner_manifest.json` shows the correct profile and requested/resolved worker counts;
2. `status_count.success` in `stat.json` is greater than 0;
3. every successful step has `status_code: 200` and non-empty token statistics;
4. multi-GPU steps include distinct `worker_id` values and router jobs reach `completed`;
5. `submit_score` is non-empty and `submission.csv` passes the scorer;
6. the process exits with code 0.

## 8. Full Evaluation and Common Overrides

Default full-evaluation config:

```text
experiment/openmle_evo
```

Key default settings:

- `max_steps=800`
- `time_budget=43200`
- `model_plus_sandbox_time_budget=64800`
- `n_samples_per_task=3`
- `evaluation_protocol=self_valid`
- `execution_timeout=7200`
- experience memory, score/delta/novelty parent selection, and sibling ranking are enabled by default

Hydra override example:

```bash
./scripts/run_multi_gpu.sh \
  output_dir=/absolute/path/to/output \
  max_steps=100 \
  time_budget=14400 \
  model_plus_sandbox_time_budget=21600 \
  n_samples_per_task=1 \
  llm_concurrency=8 \
  sandbox.concurrency=8 \
  'search.runner.task_list=[task-a,task-b]'
```

## 9. Resuming

Use the same `output_dir` and enable strict resume:

```bash
./scripts/run_multi_gpu.sh \
  output_dir=/absolute/path/to/existing-output \
  search.runner.strict_resume=true
```

Async steady-state uses at-most-once attempt semantics. Checkpoints record the attempt IDs that have been allocated; if the process crashes, in-flight attempts that were never committed may leave ID gaps, but the committed Journal stays consistent with the step artifacts.

## 10. Output Layout

```text
outputs/<experiment>/<date>/<time>/
├── runner_manifest.json
├── runner_resolved.yaml
├── summary.csv
├── .hydra/
└── program_ep_<n>/<task>/
    ├── stat.json
    ├── valid_code_final.py
    ├── submit_code.py
    ├── checkpoint/
    └── step_<n>/
        ├── response.md
        ├── stat.json
        ├── raw_run_log.txt
        └── clear_run_log.txt
```

## 11. Tests

Quick tests:

```bash
python -m pytest -q tests
```

AIRA-Evo tests including the async scheduler:

```bash
python -m pytest -q third_party/aira-evo/tests/test_async_steady_state.py
```

## 12. Troubleshooting

### Imports resolve to another worktree

Always use `run_standard.sh` or `run_multi_gpu.sh` from this directory. They explicitly set `PYTHONPATH` to the current release directory. When launching manually, set it yourself:

```bash
export PYTHONPATH="$PWD:$PWD/third_party/aira-evo/src${PYTHONPATH:+:$PYTHONPATH}"
```

### Analysis function-call returns HTTP 400

Some SGLang deployments do not support OpenAI function calling. AIRA-Evo falls back to a plain-text JSON review; as long as the node is eventually written to the Journal with status success, this warning does not affect search results.

### Tree HTML visualization warnings

The search JSON, checkpoints, code, and statistics are the authoritative artifacts. A failed HTML tree render does not change nodes, scores, or the final submission status.

### Only one Python process visible in multi-GPU mode

This is by design: the current implementation is a single-process async multi-worker. The actual GPU work is scheduled by the model service and the sandbox router, not by one local CUDA process per worker.
