# OpenMLE-Evo：MLE-Bench 运行说明

NatureBench Lite-v2 使用同一运行时但有独立的数据、eval service 和 Docker/SCM 配置；请见 [`../benchmarks/naturebench_lite_v2/RUNNING.md`](../benchmarks/naturebench_lite_v2/RUNNING.md)。

## 1. 运行边界

本目录提供 OpenMLE-Evo 的搜索与评测编排代码，不负责启动或分发模型、 准备 MLE-Bench 数据，也不包含 sandbox 镜像。运行前必须已有：

1. Python 3.11 或 3.12；
2. OpenAI-compatible 模型服务；
3. 与 `/api/v1/jobs` 协议兼容的 GPU/CPU sandbox；
4. evaluation parquet、prepared task 数据和 leaderboard 元数据；
5. 对应 sandbox API key。

标准版与多 GPU 版共用同一套代码。标准版运行同步 generation loop；多 GPU 版在单个任务进程中运行多个 async steady-state worker，并由 sandbox router 分配实际 GPU worker。

## 2. 安装

```bash
cd /path/to/repository/OpenMLE-Evo
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 会以 editable 模式安装当前 `tts_search` 和 vendored `third_party/aira-evo`。启动脚本还会把当前目录置于 `PYTHONPATH` 首位， 避免复用旧虚拟环境时误加载其他工作树中的同名包。

## 3. 配置环境

```bash
cp .env.example .env
```

必须修改以下字段：

| 变量 | 用途 |
| --- | --- |
| `OPENMLE_EVAL_DATA` | evaluation parquet 的绝对路径 |
| `OPENMLE_LEADERBOARD_DIR` | leaderboard 元数据目录 |
| `OPENMLE_SUBMIT_DATA_DIR_ROOT` | 最终提交评测使用的 prepared task 根目录 |
| `SGLANG_BASE_URL` | OpenAI-compatible 模型 API，必须以 `/v1` 结尾 |
| `OPENMLE_MODEL_ID` | `/v1/models` 返回或服务接受的模型名 |
| `PRIMARY_KEY` | 模型 API key；无鉴权本地服务可设为 `EMPTY` |
| `SANDBOX_URL` | 标准模式的直接 sandbox endpoint |
| `SANDBOX_ROUTER_URL` | 多 GPU 模式的 sandbox router |
| `SANDBOX_CPU_API_KEY` | CPU sandbox key |
| `SANDBOX_GPU_API_KEY` | GPU sandbox/router key |

默认配置针对 `Qwen/Qwen3-30B-A3B-Thinking-2507`，并携带 Qwen thinking 所需的 `extra_body`。更换不兼容的模型时，应新增或修改 `tts_search/configs/litellm/` 下的配置。

安全默认值：

- `sandbox.verify_tls=true`，HTTPS sandbox 会校验证书；
- `sandbox.trust_model_validation_score=false`，self-valid stdout 分数仅记录在
  `raw_scores`，搜索选择使用 sandbox 返回的 validation/score；
- 只有在复现实验明确需要旧 self-valid 语义时，才显式设置
  `sandbox.trust_model_validation_score=true`；
- 解析后的 runner 配置会将 API key/token/password/secret 字段写成 `null`，
  实际模型 key 仅通过子进程环境传递。

evaluation parquet 至少需要：

- `prompt` 列：system/user 消息序列；
- `metadata` 列：包含 `task_name`、`uuid`、`task`、`cpu_gpu`、 `data_dir`、`higher_is_better` 及评分范围字段。

## 4. 服务健康检查

模型：

```bash
curl -fsS "${SGLANG_BASE_URL}/models"
```

Sandbox router：

```bash
curl -fsS "${SANDBOX_ROUTER_URL}/health"
curl -fsS \
  -H "X-API-Key: ${SANDBOX_GPU_API_KEY}" \
  "${SANDBOX_ROUTER_URL}/api/v1/workers/status"
```

## 5. 标准模式

```bash
./scripts/run_standard.sh
```

只运行指定任务：

```bash
./scripts/run_standard.sh \
  'search.runner.task_list=[spooky-author-identification]'
```

标准模式固定：

```yaml
execution_mode: generation
async_workers: 1
```

## 6. 多 GPU 模式

```bash
AIRAEVO_WORKERS=8 ./scripts/run_multi_gpu.sh
```

该模式显式设置：

```yaml
execution_mode: async_steady_state
async_workers: ${AIRAEVO_WORKERS}
async_sandbox_urls:
  - ${SANDBOX_ROUTER_URL}
```

多个 worker 并发生成和提交 sandbox 作业，但 Journal、SolutionsDatabase、 strategy board 和 checkpoint 仍由单写者提交路径更新。乱序完成时，每个 step 使用节点自身的 `attempt_id`、`worker_id`、`gpu_index` 和 `sandbox_url`，不会用提交时的全局状态猜测。

单个 async attempt 遇到瞬时异常时会使用指数退避重试，不会立即分配新的
attempt id。可通过 `async_worker_max_retries` 和
`async_worker_retry_backoff_secs` 调整。GPU/资源池等待时间不计入有效搜索预算，
`max_wall_time_secs` 仍作为总墙钟时间的硬上限。

## 7. 最小 smoke test

标准版：

```bash
OPENMLE_CONFIG_NAME=experiment/openmle_evo_smoke \
./scripts/run_standard.sh \
  'search.runner.task_list=[spooky-author-identification]'
```

两 worker 多 GPU：

```bash
OPENMLE_CONFIG_NAME=experiment/openmle_evo_smoke \
AIRAEVO_WORKERS=2 \
./scripts/run_multi_gpu.sh \
  'search.runner.task_list=[spooky-author-identification]'
```

成功条件：

1. `runner_manifest.json` 中 profile、requested/resolved worker 数正确；
2. `stat.json` 的 `status_count.success` 大于 0；
3. 每个成功 step 均有 `status_code: 200` 和非空 token 统计；
4. 多 GPU step 包含不同 `worker_id`，router 作业进入 `completed`；
5. `submit_score` 非空且 `submission.csv` 通过 scorer；
6. 进程退出码为 0。

## 8. 正式评测与常用覆盖

默认正式配置：

```text
experiment/openmle_evo
```

关键默认口径：

- `max_steps=800`
- `time_budget=43200`
- `model_plus_sandbox_time_budget=64800`
- `n_samples_per_task=3`
- `evaluation_protocol=self_valid`
- `execution_timeout=7200`
- experience memory、score/delta/novelty parent selection 与 sibling ranking 默认开启

Hydra 覆盖示例：

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

## 9. 续跑

使用相同 `output_dir` 并打开严格续跑：

```bash
./scripts/run_multi_gpu.sh \
  output_dir=/absolute/path/to/existing-output \
  search.runner.strict_resume=true
```

Async steady-state 使用 at-most-once attempt 语义。checkpoint 记录已分配的 attempt ID；进程崩溃时，尚未提交的 in-flight attempt 可能留下 ID 空洞， 但已提交 Journal 与 step 产物保持一致。

## 10. 输出结构

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

## 11. 测试

快速测试：

```bash
python -m pytest -q tests
```

包含 async scheduler 的 AIRA-Evo 测试：

```bash
python -m pytest -q third_party/aira-evo/tests/test_async_steady_state.py
```

## 12. 常见问题

### Import 指向其他工作树

始终使用本目录的 `run_standard.sh` 或 `run_multi_gpu.sh`。它们会显式设置当前发布目录的 `PYTHONPATH`。手工启动时需自行设置：

```bash
export PYTHONPATH="$PWD:$PWD/third_party/aira-evo/src${PYTHONPATH:+:$PYTHONPATH}"
```

### Analysis function-call 返回 HTTP 400

部分 SGLang 部署不支持 OpenAI function calling。当前 AIRA-Evo 会退回纯文本 JSON review；只要节点最终写入 Journal 且状态为 success，该告警不影响搜索结果。

### Tree HTML 可视化告警

搜索 JSON、checkpoint、代码和统计是权威产物。HTML tree 生成失败不会改变节点、评分或最终提交状态。

### 多 GPU 只看到一个 Python 进程

这是设计行为：当前实现是单进程 async multi-worker。实际 GPU 任务由模型服务和 sandbox router 调度，不是每个 worker 一个本地 CUDA 进程。
