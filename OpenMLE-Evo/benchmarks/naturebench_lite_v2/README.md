# NatureBench Lite-v2 evaluation

This adapter runs the ten-task NatureBench Lite-v2 evaluation through the same AIRA-Evo search runtime used by MLE-Bench. Benchmark-specific task building, Docker/SCM execution, prompts, visible-data analyses, and `aggregate_improvement` scoring remain isolated behind the NatureBench adapter.

- Entrypoint: `scripts/evaluate_naturebench.py`
- Launcher: `scripts/run_naturebench.sh`
- Formal config: `experiment/naturebench_scm_lite_v2`
- Original-AIRA-Evo ablation: `experiment/naturebench_scm_lite_v2_original_airaevo`
- Smoke configs: `experiment/naturebench_smoke` and `experiment/naturebench_scm_smoke`
- Task manifest: [`tasks.txt`](tasks.txt)
- Detailed operations: [`RUNNING.md`](RUNNING.md)

NatureBench task packages, the evaluation service, Docker images, and hidden evaluation data are external and are not redistributed here.
