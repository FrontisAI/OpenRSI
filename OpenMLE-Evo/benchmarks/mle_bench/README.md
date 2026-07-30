# MLE-Bench evaluation

This is the OpenMLE-Evo MLE-Bench adapter. It consumes an evaluation parquet, prepared task data, leaderboard metadata, an OpenAI-compatible model service, and the OpenMLE sandbox protocol.

- Entrypoint: `scripts/evaluate_airaevo.py`
- Standard launcher: `scripts/run_standard.sh`
- Async multi-GPU launcher: `scripts/run_multi_gpu.sh`
- Formal config: `experiment/openmle_evo`
- Smoke config: `experiment/openmle_evo_smoke`

See [RUNNING.md](RUNNING.md) for benchmark-specific commands and [`docs/usage.md`](../../docs/usage.md) for the complete operations guide.
