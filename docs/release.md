# Release Artifacts and Boundaries

This repository contains the OpenMLE-Gym task-construction/evaluation tools, the supervised fine-tuning and OpenMLE-ERL training implementations, and the OpenMLE-Evo test-time search/evaluation implementation. OpenMLE Tasks is a separate task release. Training data, model weights, task artifacts, and deployment services are distributed separately from this code repository.

## Artifact map

| Artifact | Canonical target | What it contains |
| --- | --- | --- |
| Frontis-MA1-30B | [BF16 model](https://huggingface.co/FrontisAI/Frontis-MA1-30B) | Canonical 30B Transformers weights and model card |
| Frontis-MA1-30B GGUF | [GGUF model](https://huggingface.co/FrontisAI/Frontis-MA1-30B-GGUF) | Local-deployment derivative of the 30B model |
| Frontis-MA1-35B | [BF16 model](https://huggingface.co/FrontisAI/Frontis-MA1-35B) | Canonical 35B Transformers weights and model card |
| Frontis-MA1-35B GGUF | [GGUF model](https://huggingface.co/FrontisAI/Frontis-MA1-35B-GGUF) | GGUF language model plus F16 multimodal projector for the 35B model |
| OpenMLE-SFT-Traces | [SFT dataset](https://huggingface.co/datasets/FrontisAI/OpenMLE-SFT-Traces) | Supervised fine-tuning trajectories used by Frontis-MA1 |
| OpenMLE Tasks | [Task release](https://huggingface.co/datasets/FrontisAI/OpenMLE-Tasks) | Audited task inventory and category-specific task release artifacts |
| OpenMLE-Gym code | [`OpenMLE-Gym/`](../OpenMLE-Gym/) | Task construction, metadata extraction, local evaluation, and three public smoke task packages |
| OpenMLE SFT code | [`OpenMLE-ERL/SFT/`](../OpenMLE-ERL/SFT/) | Rollout collection, data selection, and full-parameter SFT launchers |
| OpenMLE-ERL code | [`OpenMLE-ERL/RL/`](../OpenMLE-ERL/RL/) | Four launch modes and the source-faithful RL execution chain |
| OpenMLE-Evo code | [`OpenMLE-Evo/`](../OpenMLE-Evo/) | Shared standard/async runtime plus parallel MLE-Bench and NatureBench Lite-v2 adapters, configs, and runbooks |

## OpenMLE Tasks

`OpenMLE-Tasks` is the task-environment companion release to OpenMLE. The technical report accounts for all 5,758 OpenMLE-Gym environments: full task-package data is released for 1,415 tasks, while `prepare.py` and `metric.py` reconstruction scripts are released for the remaining 4,343 tasks because source-data licensing and copyright constraints prevent redistribution.

The release uses two artifact categories:

- Category 2: rebuild recipes and processing scripts, without redistributing upstream task data.
- Category 3: complete built task packages when upstream terms permit redistribution.

## Public source scope

- OpenMLE-Gym tooling, SFT/RL training code, and test-time search/evaluation implementations are present in this code repository.
- OpenMLE Tasks is distributed as a separate task artifact.
- The reported MLE-Bench and NatureBench numbers require external task environments, sandbox services, datasets, and evaluation assets.
- BF16 model quickstarts must be paired with a tested serving/runtime matrix; each model repository records its GGUF artifact smoke under `gguf/`.
- The 35B repository retains upstream vision and MTP components, but OpenMLE post-training and reported evaluations cover text/code behavior.

## Excluded runtime material

Training corpora, model weights, checkpoints, service credentials, private infrastructure configuration, generated outputs, and deployment-specific endpoints are deliberately kept outside this code repository.

Component-specific source manifests and third-party notices remain authoritative for imported or vendored code:

- [`OpenMLE-Gym/docs/source-manifest.md`](../OpenMLE-Gym/docs/source-manifest.md)
- [`OpenMLE-Evo/docs/source-manifest.md`](../OpenMLE-Evo/docs/source-manifest.md)
- [`NOTICE`](../NOTICE)
