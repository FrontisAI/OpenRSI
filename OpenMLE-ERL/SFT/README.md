# OpenMLE-ERL: Supervised Fine-Tuning

This module contains the rollout, data-selection, and full-parameter SFT code used to train the OpenMLE models.

## Scope

The released pipeline combines:

| Source | Examples |
| --- | ---: |
| Parallel full responses | 17,344 |
| Evolutionary trajectory steps | 9,230 |
| **Total** | **26,574** |

It is a source release adapted from the production rollout and SLIME paths. Cluster paths, credentials, generated samples, checkpoints, logs, W&B state, and the exact final sample list are not bundled.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `scripts/` | Rollout, evaluation, and data-selection entry points |
| `tts_search/` | Parallel and evolutionary rollout runtime |
| `slime_scripts/` | 30B/35B full-parameter training launchers |
| `slime/` | Vendored SLIME source required by the launchers |
| `third_party/aira-evo/` | Evolutionary-search runtime and MLE bridge |
| `docs/` | Complete usage and training guide |

## Quick Start

Use the locked rollout environment for source-level operation:

```bash
cd OpenMLE-ERL/SFT
UV_PROJECT_ENVIRONMENT=.venv-rollout uv sync --locked --extra rollout
source .venv-rollout/bin/activate
```

Full rollout generation, data selection, final gates, and 30B/35B training commands are documented in [`docs/usage.md`](docs/usage.md).

## Documentation

| Document | Contents |
| --- | --- |
| [`docs/usage.md`](docs/usage.md) | Prerequisites, inputs, rollout generation, selection rules, full-parameter training, outputs, and resume behavior |
| [`third_party/aira-evo/THIRD_PARTY_LICENSES.md`](third_party/aira-evo/THIRD_PARTY_LICENSES.md) | Vendored dependency notices |

Original OpenMLE material follows the repository-level [CC BY-NC 4.0 license](../../LICENSE). Dependencies retain their upstream terms.
