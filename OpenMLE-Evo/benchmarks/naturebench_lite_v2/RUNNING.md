# NatureBench Lite-v2 Run Guide

## 1. External Dependencies

This directory contains only the evaluation adapter and the Lite-v2 configuration; it does not include NatureBench data, hidden labels, the eval service, or container images. Before a full run you need:

1. an accessible NatureBench root directory and the ten task packages;
2. a NatureBench eval service implementing the `register`, `start_timer`, and `evaluate` endpoints;
3. a `cnsbench-base:v3` or compatible Docker image containing the task dependencies;
4. an OpenAI-compatible model service;
5. for SCM mode, a passwordless-SSH host plus the remote task root and workspace directories.

## 2. Environment Variables

First copy and edit the root `.env.example`:

```bash
cp .env.example .env
```

Key variables:

| Variable | Purpose |
| --- | --- |
| `NATUREBENCH_ROOT` | Local NatureBench root; used to read the task set or local tasks |
| `NATUREBENCH_TASKS_ROOT` | Local task directory; defaults to the relative path `naturebench` |
| `NATUREBENCH_EVAL_SERVICE_URL` | Eval service used in local Docker mode |
| `NATUREBENCH_SCM_HOST` | SCM execution host; must support BatchMode SSH |
| `NATUREBENCH_SCM_WORKSPACE_ROOT` | Remote workspace for candidate code |
| `NATUREBENCH_SCM_TASK_ROOT` | Remote task-package root |
| `NATUREBENCH_SCM_EVAL_SERVICE_URL` | Eval service address as reachable from the SCM host |
| `NATUREBENCH_CONTAINER_EVAL_SERVICE_URL` | Eval service address as reachable from inside containers |
| `NATUREBENCH_DOCKER_IMAGE` | NatureBench/CNSBench runtime image |

`NATUREBENCH_SCM_GPU_DEVICES` is comma-separated, e.g. `0,1,2,3`. The default public configuration maps all official resource lines to the same SCM host; multi-host deployments should copy `tts_search/configs/data/naturebench_scm_all.yaml` and set `scm_resource_lines.<name>.scm_host` and the GPU pools individually.

## 3. Local Docker Smoke

For development on a trusted workstation without Docker, use the local-process
quick path in [`../naturebench_local_quick/README.md`](../naturebench_local_quick/README.md).
That path uses the same NatureBench adapter and evaluator protocol, but Conda is
dependency isolation rather than a security sandbox and is not suitable for
formal benchmark submissions.

Pick one task that exists locally:

```bash
NATUREBENCH_CONFIG_NAME=experiment/naturebench_smoke \
./scripts/run_naturebench.sh \
  'data.task_list=[TASK_ID]' \
  data.task_set_path=null
```

In this mode the task execution backend is `docker`; the AIRA-Evo search scheduler defaults to synchronous `generation`.

## 4. SCM Smoke

```bash
NATUREBENCH_CONFIG_NAME=experiment/naturebench_scm_smoke \
./scripts/run_naturebench.sh \
  'data.task_list=[TASK_ID]' \
  data.task_set_path=null
```

Verify first:

```bash
ssh -o BatchMode=yes "${NATUREBENCH_SCM_HOST}" \
  "test -d '${NATUREBENCH_SCM_TASK_ROOT}' && docker version"
ssh -o BatchMode=yes "${NATUREBENCH_SCM_HOST}" \
  "curl -fsS '${NATUREBENCH_SCM_EVAL_SERVICE_URL}/health'"
```

If the eval service has no `/health` endpoint, substitute whatever health check its deployment provides.

## 5. Full Lite-v2 Run

The ten tasks are pinned in `experiment/naturebench_scm_lite_v2` and [`tasks.txt`](tasks.txt):

```bash
NATUREBENCH_CONFIG_NAME=experiment/naturebench_scm_lite_v2 \
./scripts/run_naturebench.sh
```

Default settings:

- 1 sample per task;
- 80 generations with 2 candidates per generation;
- task concurrency 10, LLM concurrency 66;
- per-task, model-plus-execution, and eval timeouts all configured at the 4-hour scale;
- experience memory, parent selection, and score sanitization enabled;
- final submission disabled by default; the best node is selected by the eval-service scores of successful nodes observed during search.

To compare against the original AIRA-Evo TTS behavior:

```bash
NATUREBENCH_CONFIG_NAME=experiment/naturebench_scm_lite_v2_original_airaevo \
./scripts/run_naturebench.sh
```

## 6. Search Scheduling Switch

`NATUREBENCH_SEARCH_PROFILE` controls the AIRA-Evo search scheduler; it does not change the task's own `docker`/`scm_docker` execution backend:

```bash
# synchronous generation
NATUREBENCH_SEARCH_PROFILE=standard ./scripts/run_naturebench.sh

# async steady-state with multiple candidate workers
NATUREBENCH_SEARCH_PROFILE=multi_gpu \
AIRAEVO_WORKERS=8 \
./scripts/run_naturebench.sh
```

NatureBench async workers call the NatureBench task adapter directly; the actual GPUs are assigned from the exclusive/shared GPU pools in `scm_resource_lines`, and `SANDBOX_ROUTER_URL` is not used.

## 7. Outputs and Success Criteria

Outputs live under `outputs/<experiment>/<date>/<time>/`. Every successful task should have at least:

- `benchmark: naturebench` in `stat.json`;
- `score_protocol: naturebench`;
- at least one node with `status_code: 200`;
- a non-empty `aggregate_improvement`;
- `valid_code_final.py` and `submit_code.py`;
- a `summary.csv` in the run root.

`aggregate_improvement=0` means the baseline was matched; `>0.1` means the Surpass-SOTA threshold adopted by the current adapter was reached. Formal reports should also retain per-instance scores, raw eval-service responses, the config snapshot, and the source manifest.

## 8. Tests

```bash
python -m pytest -q tests/test_naturebench_integration.py
python -m pytest -q
```

To validate config composition only, without running tasks:

```bash
python scripts/evaluate_naturebench.py \
  --cfg job \
  --resolve \
  --config-name experiment/naturebench_scm_lite_v2
```
