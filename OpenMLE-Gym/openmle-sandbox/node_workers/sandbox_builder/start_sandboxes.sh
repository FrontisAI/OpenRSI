#!/bin/bash
set -euo pipefail

OPENMLE_SANDBOX_IMAGE_PREFIX="${OPENMLE_SANDBOX_IMAGE_PREFIX:-ccr.ccs.tencentyun.com/frontisai-openmle}"
SANDBOX_IMAGE="${SANDBOX_IMAGE:-${OPENMLE_SANDBOX_IMAGE_PREFIX}/openmle-sandbox-worker:1.0.0}"
CONTAINER_WORKSPACE="/mnt/pubdatasets2"
LOCAL_SCRATCH_HOST_ROOT="${LOCAL_SCRATCH_HOST_ROOT:-/local_nvme/sandbox_workdir}"
LOCAL_SCRATCH_CONTAINER_ROOT="${LOCAL_SCRATCH_CONTAINER_ROOT:-/mnt/local_sandbox_workdir}"
NFS_MOUNT="${NFS_MOUNT:-/nfs2/users/pubdatasets2}"

CPU_LIMIT="${CPU_LIMIT:-6.0}"
MEM_LIMIT="${MEM_LIMIT:-200g}"
SHM_SIZE="${SHM_SIZE:-16g}"

detect_host_ip() {
  local ip=""
  if command -v hostname >/dev/null 2>&1; then
    ip=$(hostname -I 2>/dev/null \
      | tr ' ' '\n' \
      | grep -E '^[0-9]{1,3}(\.[0-9]{1,3}){3}$' \
      | grep -Ev '^(127\.|169\.254\.)' \
      | head -n1 || true)
  fi
  if [ -z "$ip" ] && command -v ip >/dev/null 2>&1; then
    ip=$(ip -o -4 addr show scope global 2>/dev/null \
      | awk 'NR == 1 {split($4, parts, "/"); print parts[1]}' || true)
  fi
  echo "$ip"
}

read_int() {
  local prompt="$1"
  local value
  # read -p 的提示语默认是输出到 stderr 的，所以这里不需要改，但下面的 echo 必须改
  while true; do
    read -rp "$prompt" value
    if [[ "$value" =~ ^[0-9]+$ ]]; then
      echo "$value"
      return
    fi
    echo "请输入数字。" >&2
  done
}

read_int_default() {
  local prompt="$1"
  local default_value="$2"
  local value
  while true; do
    read -rp "$prompt" value
    if [[ -z "$value" ]]; then
      echo "$default_value"
      return
    fi
    if [[ "$value" =~ ^[0-9]+$ ]]; then
      echo "$value"
      return
    fi
    echo "请输入数字。" >&2
  done
}

confirm_ip() {
  local ip="$1"
  if [ -z "$ip" ]; then
    echo "未自动检测到内网 IP。" >&2
    read -rp "请输入 Controller 可访问的内网 IP (如 10.0.0.21): " ip
  fi
  while true; do
    echo "检测到内网 IP: $ip" >&2
    read -rp "是否正确? (y/n): " confirm
    case "$confirm" in
      y|Y)
        # 只有这一行输出到 stdout，作为函数的返回值
        echo "$ip"
        return
        ;;
      n|N)
        read -rp "请输入 Controller 可访问的内网 IP (如 10.0.0.21): " ip
        ;;
      *)
        echo "请输入 y 或 n。" >&2
        ;;
    esac
  done
}

echo "NFS_MOUNT: $NFS_MOUNT"

mkdir -p "$LOCAL_SCRATCH_HOST_ROOT" "$LOCAL_SCRATCH_HOST_ROOT/tmp"
chmod 777 "$LOCAL_SCRATCH_HOST_ROOT" "$LOCAL_SCRATCH_HOST_ROOT/tmp" || true

HOST_IP=$(detect_host_ip)
# 修正调用逻辑：confirm_ip 的提示语现在去 stderr 了，HOST_IP 只会拿到纯净的 IP
HOST_IP=$(confirm_ip "$HOST_IP")

echo "如果是在调度服务器本机上部署sandbox容器，端口号只能是10000 ~ 10500之间，否则需要额外开放端口"
GPU_DEVICE_COUNT=$(read_int_default "请输入机器GPU数量(默认8): " 8)
GPU_COUNT=$(read_int "请输入 需要部署的GPU sandbox 数量，例如8: ")
BASE_PORT_GPU=$(read_int "请输入 GPU sandbox 起始端口,例如10150: ")
CPU_COUNT=$(read_int "请输入 CPU sandbox 数量，例如8: ")
BASE_PORT_CPU=$(read_int "请输入 CPU sandbox 起始端口，例如10160: ")

if [ "$GPU_COUNT" -gt 0 ] && [ "$GPU_DEVICE_COUNT" -le 0 ]; then
  echo "GPU 数量为 0，但要求部署 GPU sandbox，已退出。" >&2
  exit 1
fi

if [ "$GPU_COUNT" -gt "$GPU_DEVICE_COUNT" ]; then
  echo "提示: 需要部署的 GPU sandbox 数量($GPU_COUNT) > 物理 GPU 数量($GPU_DEVICE_COUNT)。" >&2
  echo "将从第 ${GPU_DEVICE_COUNT} 张卡开始循环复用 GPU (例如第9个容器会从 GPU0 开始)。" >&2
fi

# 下面是展示信息，直接输出到屏幕（stdout）没问题，因为这里没有变量捕获
cat <<EOF

配置确认:
  NFS_MOUNT: $NFS_MOUNT
  HOST_IP: $HOST_IP
  GPU_DEVICE_COUNT: $GPU_DEVICE_COUNT
  GPU_COUNT: $GPU_COUNT
  GPU_BASE_PORT: $BASE_PORT_GPU
  CPU_COUNT: $CPU_COUNT
  CPU_BASE_PORT: $BASE_PORT_CPU
  IMAGE: $SANDBOX_IMAGE
  CPU_LIMIT: $CPU_LIMIT
  MEM_LIMIT: $MEM_LIMIT
  SHM_SIZE: $SHM_SIZE
  LOCAL_SCRATCH_HOST_ROOT: $LOCAL_SCRATCH_HOST_ROOT
  LOCAL_SCRATCH_CONTAINER_ROOT: $LOCAL_SCRATCH_CONTAINER_ROOT
EOF

read -rp "确认开始启动? (y/n): " proceed
if [[ ! "$proceed" =~ ^[Yy]$ ]]; then
  echo "已取消。"
  exit 0
fi

if [ "$GPU_COUNT" -gt 0 ]; then
  for ((i=0; i<GPU_COUNT; i++)); do
    HOST_PORT=$((BASE_PORT_GPU + i))
    CONTAINER_NAME="ml-sandbox-gpu-$i"
    GPU_DEVICE=$((i % GPU_DEVICE_COUNT))
    echo "正在启动 GPU Sandbox $CONTAINER_NAME on GPU:$GPU_DEVICE, Port:$HOST_PORT ..."
    docker run -d \
      --security-opt seccomp=unconfined \
      --name "$CONTAINER_NAME" \
      -p "$HOST_IP:$HOST_PORT:8080" \
      --gpus "device=$GPU_DEVICE" \
      -v "$NFS_MOUNT:$CONTAINER_WORKSPACE:ro" \
      -v "$NFS_MOUNT/mlsandbox:$CONTAINER_WORKSPACE/mlsandbox:rw" \
      -v "$NFS_MOUNT/mlsandbox_dev:$CONTAINER_WORKSPACE/mlsandbox_dev:rw" \
      -v "$NFS_MOUNT/mlsandbox_3:$CONTAINER_WORKSPACE/mlsandbox_3:rw" \
      -v "$LOCAL_SCRATCH_HOST_ROOT:$LOCAL_SCRATCH_CONTAINER_ROOT:rw" \
      --cpus "$CPU_LIMIT" \
      --memory "$MEM_LIMIT" \
      --shm-size "$SHM_SIZE" \
      -e HF_ENDPOINT=https://hf-mirror.com \
      -e HF_HOME=$CONTAINER_WORKSPACE/mlsandbox/hf_home \
      -e TORCH_HOME=$CONTAINER_WORKSPACE/mlsandbox/torch_home \
      -e TMPDIR=$LOCAL_SCRATCH_CONTAINER_ROOT/tmp \
      "$SANDBOX_IMAGE"
    echo "$CONTAINER_NAME 已启动。"
  done
  echo "所有 GPU Sandbox 实例已启动。端口号: $BASE_PORT_GPU ~ $((BASE_PORT_GPU + GPU_COUNT - 1))"
fi

if [ "$CPU_COUNT" -gt 0 ]; then
  for ((i=0; i<CPU_COUNT; i++)); do
    HOST_PORT=$((BASE_PORT_CPU + i))
    CONTAINER_NAME="ml-sandbox-cpu-$i"
    echo "正在启动 CPU Sandbox $CONTAINER_NAME, Port:$HOST_PORT ..."
    docker run -d \
      --security-opt seccomp=unconfined \
      --name "$CONTAINER_NAME" \
      -p "$HOST_IP:$HOST_PORT:8080" \
      -v "$NFS_MOUNT:$CONTAINER_WORKSPACE:ro" \
      -v "$NFS_MOUNT/mlsandbox:$CONTAINER_WORKSPACE/mlsandbox:rw" \
      -v "$NFS_MOUNT/mlsandbox_dev:$CONTAINER_WORKSPACE/mlsandbox_dev:rw" \
      -v "$NFS_MOUNT/mlsandbox_3:$CONTAINER_WORKSPACE/mlsandbox_3:rw" \
      -v "$LOCAL_SCRATCH_HOST_ROOT:$LOCAL_SCRATCH_CONTAINER_ROOT:rw" \
      --cpus "$CPU_LIMIT" \
      --memory "$MEM_LIMIT" \
      --shm-size "$SHM_SIZE" \
      -e HF_ENDPOINT=https://hf-mirror.com \
      -e HF_HOME=$CONTAINER_WORKSPACE/mlsandbox/hf_home \
      -e TORCH_HOME=$CONTAINER_WORKSPACE/mlsandbox/torch_home \
      -e TMPDIR=$LOCAL_SCRATCH_CONTAINER_ROOT/tmp \
      "$SANDBOX_IMAGE"
    echo "$CONTAINER_NAME 已启动。"
  done
  echo "所有 CPU Sandbox 实例已启动。端口号: $BASE_PORT_CPU ~ $((BASE_PORT_CPU + CPU_COUNT - 1))"
fi

CONFIG_FILE="sandbox_config_${HOST_IP}.json"
GPU_END=$((BASE_PORT_GPU + GPU_COUNT - 1))
CPU_END=$((BASE_PORT_CPU + CPU_COUNT - 1))
GPU_RANGE=""
CPU_RANGE=""
# 注意：JSON 格式里如果列表为空，不需要内容，所以这里的逻辑基本是 ok 的
if [ "$GPU_COUNT" -gt 0 ]; then
  GPU_RANGE="      {\"host\": \"${HOST_IP}\", \"start\": ${BASE_PORT_GPU}, \"end\": ${GPU_END}}"
fi
if [ "$CPU_COUNT" -gt 0 ]; then
  CPU_RANGE="      {\"host\": \"${HOST_IP}\", \"start\": ${BASE_PORT_CPU}, \"end\": ${CPU_END}}"
fi

cat <<EOF > "$CONFIG_FILE"
{
  "gpu": {
    "ranges": [
${GPU_RANGE}
    ]
  },
  "cpu": {
    "ranges": [
${CPU_RANGE}
    ]
  }
}
EOF

echo "已生成配置文件: ${CONFIG_FILE}"
