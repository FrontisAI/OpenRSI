# OpenMLE Sandbox Router

独立的 Sandbox API 聚合入口。它不直接管理 worker，也不修改现有 controller，只把客户端请求路由到已经部署好的后端服务。

## 目标

- 客户端只修改 `BASE_URL`。
- 启动时通过 YAML 选择接入任意 controller 组合。
- 运行中自动热加载 YAML；`enabled: false` 后只停止向该后端提交新 job。
- `POST /api/v1/jobs` 后持久化 `job_id -> backend_id`，后续 status、logs、cancel 回到同一后端。
- 聚合 `/api/v1/workers/status`，返回所有已启用 controller 的 worker 总量。

## 启动

普通部署可以使用下面的公开 Docker 镜像，也可以在服务器已有的 Python
3.10+ 环境中直接运行源码。该公开镜像面向 `linux/amd64`。默认从腾讯云
TCR 匿名拉取；如果 TCR 在当前网络不可用或速度较慢，可在运行下面命令前执行
`export OPENMLE_SANDBOX_IMAGE_PREFIX="ghcr.io/lifeissosolong"`，切换到公开的
个人 GHCR 备用镜像。

### Docker（推荐）

```bash
cd OpenMLE-Gym/openmle-sandbox/node_router
mkdir -p router-state
export SANDBOX_GATEWAY_ADMIN_TOKEN="$(openssl rand -hex 32)"
OPENMLE_SANDBOX_IMAGE_PREFIX="${OPENMLE_SANDBOX_IMAGE_PREFIX:-ccr.ccs.tencentyun.com/frontisai-openmle}"
ROUTER_IMAGE="$OPENMLE_SANDBOX_IMAGE_PREFIX/openmle-sandbox-router:1.0.0"

docker pull "$ROUTER_IMAGE"

docker run --rm \
  --name openmle-sandbox-router \
  -p 6591:6591 \
  -e SANDBOX_GATEWAY_CONFIG=/app/backends.yaml \
  -e SANDBOX_GATEWAY_DB=/router-state/gateway_routes.sqlite3 \
  -e SANDBOX_GATEWAY_ADMIN_TOKEN="$SANDBOX_GATEWAY_ADMIN_TOKEN" \
  -v "$PWD/backends.yaml:/app/backends.yaml:ro" \
  -v "$PWD/router-state:/router-state" \
  "$ROUTER_IMAGE"
```

### Python 进程（源码方式）

准备一个可以通过 `pip` 安装 `requirements.txt` 的 Python 3.10+ 环境，并将
`PYTHON_BIN` 指向对应解释器。下面以 `python3.10` 为默认值：

```bash
cd OpenMLE-Gym/openmle-sandbox/node_router
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -m pip install -r requirements.txt

export SANDBOX_GATEWAY_CONFIG="$PWD/backends.yaml"
export SANDBOX_GATEWAY_ADMIN_TOKEN="$(openssl rand -hex 32)"
"$PYTHON_BIN" -m uvicorn gateway:app --host 0.0.0.0 --port 6591
```

默认配置文件是本目录的 `backends.yaml`。管理 Token 只用于 `/admin/*`，普通 job 请求继续使用各 controller 接受的 `X-API-Key`。

## 公网入口与 Base URL

上面的两种启动方式都会让 Router 监听宿主机全部网卡：原生方式使用
`--host 0.0.0.0 --port 6591`，Docker 方式的 `-p 6591:6591` 等价于将
宿主机 `0.0.0.0:6591` 发布到容器。

- 如果服务器网卡直接拥有公网 IP，通常不需要额外端口映射，只需确认
  防火墙或安全组允许客户端访问 TCP `6591`。
- 如果服务器位于 NAT、共享公网 IP、负载均衡或云端口转发之后，需要把
  一个公网端口映射到 Router 主机的 `6591`；公网端口可以与 `6591` 不同。
- 如果只在私网或 VPN 内使用，不必配置公网入口，客户端直接使用可达的
  私网地址即可。

客户端的 Base URL 应填写最终可达的入口，例如直接公网部署使用
`http://router.example.com:6591`，端口映射部署则使用
`http://router.example.com:<公网端口>`。`backends.yaml` 中各 Controller
地址通常继续填写 Router 可达的集群内网地址，不需要绕到公网。

完成部署后，应从 Router 所在网络之外的真实客户端执行：

```bash
ROUTER_URL="http://router.example.com:6591"
SANDBOX_API_KEY="replace-with-a-key-accepted-by-every-controller"

curl -fsS "$ROUTER_URL/health"
curl -fsS \
  -H "X-API-Key: $SANDBOX_API_KEY" \
  "$ROUTER_URL/api/v1/workers/status"
```

本机 `127.0.0.1`、内网地址或 SSH 隧道可以验证 Router 业务功能，但不能
替代公网入口验收。只有最终公网 URL 的 health、鉴权 worker status 和真实
任务都从外部客户端成功，才能认为公网 Base URL 可用。公网部署前还应更换
默认 API Key 和 Router 管理 Token，并尽量使用来源地址限制与 HTTPS 反向代理。

## 配置

`backends.yaml` 示例：

```yaml
backends:
  - id: controller-a
    base_url: http://203.0.113.10:6580
    enabled: true
    weight: 1
  - id: controller-b
    base_url: http://203.0.113.11:6580
    enabled: true
    weight: 1
  - id: controller-c
    base_url: http://203.0.113.12:6580
    enabled: false
    weight: 1

routing:
  strategy: idle_gpu_first
  fallback: round_robin
  worker_status_ttl_seconds: 2.0

storage:
  route_store_path: gateway_routes.sqlite3
```

`203.0.113.0/24` 是文档保留地址，不能直接用于部署。请将示例 URL 替换为自己的 controller 地址。`enabled: true/false` 控制 Router 接入哪些服务；对外总容量等于所有已启用 controller 的 worker 容量之和。

Router 会在处理请求前检查 `backends.yaml` 的修改时间，文件变化后自动重新加载。也可以手动触发：

```bash
curl -X POST \
  -H "X-Gateway-Admin-Token: replace-with-a-private-admin-token" \
  http://127.0.0.1:6591/admin/reload
```

禁用后端只影响新提交的 job。已经写入 SQLite 的旧 job 仍然通过 `job_id -> backend_id` 回到原后端查询 status、logs 或 cancel。为了避免旧 job 断链，不要删除仍可能承载旧 job 的后端配置项；需要停用时只把 `enabled` 改成 `false`。

## 已实现接口

- `GET /health`
- `POST /api/v1/jobs`
- `GET /api/v1/jobs/{job_id}`
- `GET /api/v1/jobs/{job_id}/logs`
- `DELETE /api/v1/jobs/{job_id}`
- `GET /api/v1/jobs`
- `GET /api/v1/workers/status`
- `GET /admin/config`
- `POST /admin/reload`

## 暂不实现

- `POST /api/v1/jobs/submit_and_wait`
- `POST /api/v1/uploads`

当前客户端使用 `POST /api/v1/jobs` 加轮询 `GET /api/v1/jobs/{job_id}`，不依赖这两个接口。`submit_and_wait` 返回 501，避免长连接和 Router 重启时的语义风险；`uploads` 返回 501，避免 upload 后续 job 必须绑定同一后端的兼容风险。

## 持久化

Python 方式默认的 SQLite 文件：

```text
OpenMLE-Gym/openmle-sandbox/node_router/gateway_routes.sqlite3
```

Docker 方式通过 `SANDBOX_GATEWAY_DB` 将同一状态文件持久化到宿主机的
`node_router/router-state/gateway_routes.sqlite3`。

SQLite 只存 Router 自己的路由表，不连接、不修改各 controller 的 PostgreSQL 或 Redis。单 Router 进程使用即可；如果未来需要多实例高可用，应改用 Redis 或 PostgreSQL 作为 route store。

源目录中的额外环境专用 backend 配置没有进入公开目录；公开仓库只保留已去除真实地址和环境名称的通用 `backends.yaml`。
