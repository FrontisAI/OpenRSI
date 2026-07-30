# OpenMLE-Evo

This directory contains the runnable OpenMLE-Evo test-time search and evaluation release. MLE-Bench and NatureBench Lite-v2 are parallel benchmark adapters over one shared AIRA-Evo runtime. Standard and asynchronous multi-GPU search use the same journal, checkpoint, memory, parent-selection, and output implementation.

## Benchmark adapters

| Adapter | Entrypoint | Formal config | Runbook |
| --- | --- | --- | --- |
| MLE-Bench | `scripts/evaluate_airaevo.py` | `experiment/openmle_evo` | [`benchmarks/mle_bench/`](../benchmarks/mle_bench/) |
| NatureBench Lite-v2 | `scripts/evaluate_naturebench.py` | `experiment/naturebench_scm_lite_v2` | [`benchmarks/naturebench_lite_v2/`](../benchmarks/naturebench_lite_v2/) |

## Included

- `scripts/run_standard.sh` and `scripts/run_multi_gpu.sh`: MLE-Bench launchers.
- `scripts/run_naturebench.sh`: NatureBench launcher with a `standard|multi_gpu` search-profile switch.
- `tts_search/`: concurrency, sandbox, scoring, prompt, guidance, and Hydra configuration code, including the final experience-memory policy and NatureBench Lite-v2 visible-data analyses.
- `third_party/aira-evo/`: the vendored AIRA-Dojo runtime required by both adapters, with its original license and third-party notices.
- `tests/`: public configuration and guidance tests.

Datasets, NatureBench task packages and eval service, leaderboard assets, model weights, model servers, sandbox services, container images, credentials, and paper-result outputs are external and are not included.

## Execution profiles

| Benchmark | Profile | Hydra override | Search scheduler | Worker execution |
| --- | --- | --- | --- | --- |
| MLE-Bench | Standard | `execution=standard` | synchronous generation | `SANDBOX_URL` |
| MLE-Bench | Multi-GPU | `execution=multi_gpu` | async steady-state | `SANDBOX_ROUTER_URL` |
| NatureBench | Standard | `execution=standard` | synchronous generation | Docker or SCM adapter |
| NatureBench | Multi-GPU | `execution=naturebench_multi_gpu` | async steady-state | SCM resource/GPU pools |

`gpu_index` in async metadata identifies a worker slot. Actual GPU allocation is owned by the configured sandbox router for MLE-Bench and by NatureBench SCM resource lines for NatureBench.

## Quick start

```bash
cd OpenMLE-Evo
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cp .env.example .env
# Edit .env and replace every path, endpoint, and sandbox key.
```

Standard run:

```bash
./scripts/run_standard.sh
```

MLE-Bench multi-GPU run:

```bash
AIRAEVO_WORKERS=8 ./scripts/run_multi_gpu.sh
```

NatureBench Lite-v2 run:

```bash
NATUREBENCH_CONFIG_NAME=experiment/naturebench_scm_lite_v2 \
./scripts/run_naturebench.sh
```

NatureBench async multi-worker run:

```bash
NATUREBENCH_SEARCH_PROFILE=multi_gpu \
AIRAEVO_WORKERS=8 \
./scripts/run_naturebench.sh
```

MLE-Bench two-worker public smoke:

```bash
OPENMLE_CONFIG_NAME=experiment/openmle_evo_smoke \
AIRAEVO_WORKERS=2 \
./scripts/run_multi_gpu.sh \
  'search.runner.task_list=[spooky-author-identification]'
```

Outputs are written below `outputs/<experiment>/<date>/<time>/`. A completed task contains `stat.json`, `valid_code_final.py`, `submit_code.py`, step artifacts, checkpoints, `summary.csv`, and `runner_manifest.json`.

## Documentation

| Document | Contents |
| --- | --- |
| [`usage.md`](usage.md) | MLE-Bench installation, configuration, launch, resume, outputs, and troubleshooting |
| [`../benchmarks/naturebench_lite_v2/RUNNING.md`](../benchmarks/naturebench_lite_v2/RUNNING.md) | NatureBench Lite-v2 operations |
| [`validation.md`](validation.md) | Source-level and runtime validation boundary |
| [`mlebench_validation_split_instructions_22.md`](mlebench_validation_split_instructions_22.md) | Fixed validation instructions for the 22-task MLE-Bench Lite split |
| [`source-manifest.md`](source-manifest.md) | Included runtime scope and release adjustments |

## Reproducibility boundary

The default `openmle_evo` experiment encodes the MLE-Bench paper-scale search
budgets. Model-reported self-validation scores are retained for diagnostics but
are not trusted for parent selection unless
`sandbox.trust_model_validation_score=true` is explicitly enabled for legacy
reproduction. The NatureBench Lite-v2 config fixes the ten-task manifest and its
four-hour-scale search/evaluation budgets. Reproducing reported numbers
additionally requires the exact external datasets, task packages, evaluator
revisions, model checkpoint/server, and execution images.

## Provenance and license

The repository commit or release tag containing this directory is its public source identity. The included runtime scope and excluded deployment artifacts are listed in [`source-manifest.md`](source-manifest.md).

Original OpenMLE material follows the repository-level CC BY-NC 4.0 license. Vendored AIRA-Dojo material remains under [its own CC BY-NC 4.0 license](../third_party/aira-evo/LICENSE), with additional terms recorded in [THIRD_PARTY_LICENSES.md](../third_party/aira-evo/THIRD_PARTY_LICENSES.md).
