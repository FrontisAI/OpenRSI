# Running MLE-Bench

After setting up `OPENMLE_EVAL_DATA`, `OPENMLE_LEADERBOARD_DIR`, `OPENMLE_SUBMIT_DATA_DIR_ROOT`, the model service, and the sandbox configuration in `.env`, run:

```bash
# single worker
./scripts/run_standard.sh

# async steady-state with multiple sandbox/GPU workers
AIRAEVO_WORKERS=8 ./scripts/run_multi_gpu.sh
```

Minimal smoke:

```bash
OPENMLE_CONFIG_NAME=experiment/openmle_evo_smoke \
./scripts/run_standard.sh \
  'search.runner.task_list=[spooky-author-identification]'
```

See [`../../docs/usage.md`](../../docs/usage.md) for the full environment variables, resuming, output layout, and success criteria.
