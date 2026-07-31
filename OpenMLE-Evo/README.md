# OpenMLE-Evo

OpenMLE-Evo is the test-time search and evaluation system used in *Frontis-MA1: Training an AI4AI Model towards Recursive Self-Improvement in Machine Learning Engineering*.

MLE-Bench and NatureBench Lite-v2 are parallel benchmark adapters over one shared AIRA-Evo runtime. Standard and asynchronous multi-GPU search use the same journal, checkpoint, memory, parent-selection, and output implementation.

## Benchmark Adapters

| Adapter | Entrypoint | Runbook |
| --- | --- | --- |
| MLE-Bench | `scripts/evaluate_airaevo.py` | [`benchmarks/mle_bench/`](benchmarks/mle_bench/) |
| NatureBench Lite-v2 | `scripts/evaluate_naturebench.py` | [`benchmarks/naturebench_lite_v2/`](benchmarks/naturebench_lite_v2/) |

## Repository Contents

| Path | Purpose |
| --- | --- |
| `scripts/` | Standard, multi-GPU, and NatureBench launchers |
| `tts_search/` | Search, sandbox, scoring, prompting, guidance, and Hydra configuration |
| `benchmarks/` | MLE-Bench and NatureBench Lite-v2 adapters |
| `third_party/aira-evo/` | Vendored AIRA-Dojo runtime |
| `tests/` | Existing public configuration and scheduler checks |
| `docs/` | Architecture, operations, validation, and source scope |

## Quick Start

```bash
cd OpenMLE-Evo
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

After configuring `.env`, use `./scripts/run_standard.sh` for the standard MLE-Bench profile or follow the benchmark-specific runbook.

## Documentation

| Document | Contents |
| --- | --- |
| [`docs/overview.md`](docs/overview.md) | Runtime architecture, benchmark adapters, execution profiles, and command overview |
| [`docs/usage.md`](docs/usage.md) | MLE-Bench installation, configuration, launch, resume, outputs, and troubleshooting |
| [`benchmarks/naturebench_lite_v2/RUNNING.md`](benchmarks/naturebench_lite_v2/RUNNING.md) | NatureBench Lite-v2 operations |
| [`docs/validation.md`](docs/validation.md) | Source-level and runtime validation boundary |
| [`docs/mlebench_validation_split_instructions_22.md`](docs/mlebench_validation_split_instructions_22.md) | Fixed validation instructions for the 22-task MLE-Bench Lite split |
| [`docs/source-manifest.md`](docs/source-manifest.md) | Included runtime scope and release adjustments |

Datasets, task packages, evaluator services, model weights, model servers, sandbox services, container images, credentials, and paper-result outputs are external.

Original OpenMLE material follows the repository-level [CC BY-NC 4.0 license](../LICENSE). Vendored AIRA-Dojo material retains [its own license](third_party/aira-evo/LICENSE) and [third-party notices](third_party/aira-evo/THIRD_PARTY_LICENSES.md).
