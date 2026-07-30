# NatureBench Lite-v2 运行说明

## 1. 外部依赖

本目录只包含评测适配器和 Lite-v2 配置，不包含 NatureBench 数据、隐藏标签、 eval service 或容器镜像。正式运行前需要：

1. 可访问的 NatureBench 根目录和十个任务包；
2. 实现 `register`、`start_timer`、`evaluate` 接口的 NatureBench eval service；
3. 包含任务依赖的 `cnsbench-base:v3` 或兼容 Docker 镜像；
4. OpenAI-compatible 模型服务；
5. SCM 模式下可用的免交互 SSH 主机、远端任务根目录和工作目录。

## 2. 环境变量

先复制并编辑根目录 `.env.example`：

```bash
cp .env.example .env
```

关键变量：

| 变量 | 用途 |
| --- | --- |
| `NATUREBENCH_ROOT` | 本机 NatureBench 根目录；用于读取 task-set 或本地任务 |
| `NATUREBENCH_TASKS_ROOT` | 本地任务目录；默认相对路径 `naturebench` |
| `NATUREBENCH_EVAL_SERVICE_URL` | 本机 Docker 模式使用的 eval service |
| `NATUREBENCH_SCM_HOST` | SCM 执行主机，需支持 BatchMode SSH |
| `NATUREBENCH_SCM_WORKSPACE_ROOT` | 远端候选代码工作目录 |
| `NATUREBENCH_SCM_TASK_ROOT` | 远端任务包根目录 |
| `NATUREBENCH_SCM_EVAL_SERVICE_URL` | 从 SCM 主机访问 eval service 的地址 |
| `NATUREBENCH_CONTAINER_EVAL_SERVICE_URL` | 从容器访问 eval service 的地址 |
| `NATUREBENCH_DOCKER_IMAGE` | NatureBench/CNSBench 运行镜像 |

`NATUREBENCH_SCM_GPU_DEVICES` 使用逗号分隔，例如 `0,1,2,3`。默认公开配置把所有官方 resource line 映射到同一 SCM 主机；多主机部署应复制 `tts_search/configs/data/naturebench_scm_all.yaml`，分别设置 `scm_resource_lines.<name>.scm_host` 和 GPU 池。

## 3. 本机 Docker smoke

选择本机存在的一个任务：

```bash
NATUREBENCH_CONFIG_NAME=experiment/naturebench_smoke \
./scripts/run_naturebench.sh \
  'data.task_list=[TASK_ID]' \
  data.task_set_path=null
```

该模式的任务执行方式是 `docker`；AIRA-Evo 搜索调度默认是同步 `generation`。

## 4. SCM smoke

```bash
NATUREBENCH_CONFIG_NAME=experiment/naturebench_scm_smoke \
./scripts/run_naturebench.sh \
  'data.task_list=[TASK_ID]' \
  data.task_set_path=null
```

先确认：

```bash
ssh -o BatchMode=yes "${NATUREBENCH_SCM_HOST}" \
  "test -d '${NATUREBENCH_SCM_TASK_ROOT}' && docker version"
ssh -o BatchMode=yes "${NATUREBENCH_SCM_HOST}" \
  "curl -fsS '${NATUREBENCH_SCM_EVAL_SERVICE_URL}/health'"
```

如果 eval service 没有 `/health`，可改用其部署提供的健康检查命令。

## 5. Lite-v2 正式运行

十个任务已固定在 `experiment/naturebench_scm_lite_v2` 和 [`tasks.txt`](tasks.txt)：

```bash
NATUREBENCH_CONFIG_NAME=experiment/naturebench_scm_lite_v2 \
./scripts/run_naturebench.sh
```

默认口径：

- 每任务 1 个样本；
- 80 generations、每 generation 2 个候选；
- 任务并发 10，LLM 并发 66；
- 单次任务、模型加执行和 eval timeout 均按 4 小时级配置；
- experience memory、parent selection 和 score sanitization 开启；
- 最终提交默认关闭，以搜索过程中 eval service 返回的成功节点分数选优。

对照原始 AIRA-Evo TTS 行为：

```bash
NATUREBENCH_CONFIG_NAME=experiment/naturebench_scm_lite_v2_original_airaevo \
./scripts/run_naturebench.sh
```

## 6. 搜索调度开关

`NATUREBENCH_SEARCH_PROFILE` 控制 AIRA-Evo 搜索调度，不改变任务本身的 `docker`/`scm_docker` 执行方式：

```bash
# 同步 generation
NATUREBENCH_SEARCH_PROFILE=standard ./scripts/run_naturebench.sh

# async steady-state，多候选 worker
NATUREBENCH_SEARCH_PROFILE=multi_gpu \
AIRAEVO_WORKERS=8 \
./scripts/run_naturebench.sh
```

NatureBench 的 async worker 直接调用 NatureBench task adapter，实际 GPU 由 `scm_resource_lines` 的 exclusive/shared GPU 池分配，不使用 `SANDBOX_ROUTER_URL`。

## 7. 输出与成功判据

输出位于 `outputs/<experiment>/<date>/<time>/`。每个成功任务至少应有：

- `stat.json` 中 `benchmark: naturebench`；
- `score_protocol: naturebench`；
- 至少一个 `status_code: 200` 的节点；
- 非空 `aggregate_improvement`；
- `valid_code_final.py` 和 `submit_code.py`；
- 根目录 `summary.csv`。

`aggregate_improvement=0` 表示达到基线；`>0.1` 表示达到当前适配器采用的 Surpass-SOTA 判定阈值。正式报告应同时保留每实例分数、原始 eval service 响应、配置快照和 source manifest。

## 8. 测试

```bash
python -m pytest -q tests/test_naturebench_integration.py
python -m pytest -q
```

只验证配置合成、不运行任务：

```bash
python scripts/evaluate_naturebench.py \
  --cfg job \
  --resolve \
  --config-name experiment/naturebench_scm_lite_v2
```
