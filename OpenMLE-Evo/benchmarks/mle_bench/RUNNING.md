# MLE-Bench 运行

准备 `.env` 中的 `OPENMLE_EVAL_DATA`、`OPENMLE_LEADERBOARD_DIR`、 `OPENMLE_SUBMIT_DATA_DIR_ROOT`、模型服务和 sandbox 配置后运行：

```bash
# 单 worker
./scripts/run_standard.sh

# async steady-state，多 sandbox/GPU worker
AIRAEVO_WORKERS=8 ./scripts/run_multi_gpu.sh
```

最小 smoke：

```bash
OPENMLE_CONFIG_NAME=experiment/openmle_evo_smoke \
./scripts/run_standard.sh \
  'search.runner.task_list=[spooky-author-identification]'
```

完整环境变量、续跑、输出结构和成功判据见 [`../../docs/usage.md`](../../docs/usage.md)。
