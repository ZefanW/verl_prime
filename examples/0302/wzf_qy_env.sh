# 设置全局变量，运行ray start以前运行
export TRAIN_DATASET=/home/wangzefan/dataset/dataset/prime-rl-math-wo-prompt/train.parquet
export TEST_DATASET=/home/wangzefan/dataset/dataset/prime-rl-math-wo-prompt/validation.parquet
export SFT_MODEL_PATH=/home/wangzefan/huggingface/Qwen2.5-Math-7B
export PARALLEL_SIZE=1
export CKPT_PATH=/home/wangzefan/data/verl_prime/checkpoints
export WANDB_MODE=offline
export OUTLINES_CACHE_DIR=~/.cache/outlines/$(uuidgen)
export NCCL_DEBUG=WARN
export WANDB_API_KEY='194ba8b74c01d7f88fbf18db8f53206e24b2d46a'
#export VLLM_ATTENTION_BACKEND=XFORMERS
unset VLLM_ATTENTION_BACKEND
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PYTORCH_CUDA_ALLOC_CONF
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=offline
export WANDB_DIR=/home/wangzefan/data/verl_prime/
export SWANLAB_API_KEY='vP17PxpkO7x33BK1MWTK6'
declare -g -n SWANLAB_LOG_DIR=WANDB_DIR

# for eval
export EVAL_BASE=/home/wangzefan/data/PRIME/eval
export EVAL_RESULT_BASE=/home/wangzefan/data/verl_prime/eval_results
export CONDA_SH=/home/wangzefan/anaconda3/etc/profile.d/conda.sh

# ray init nodes
export WORKING_DIR=/home/wangzefan/data/verl_prime
export PYTHONPATH=/home/wangzefan/data/verl_prime

# network
#export HYDRA_FULL_ERROR=1
#export NCCL_DEBUG=INFO
#export CUDA_DEVICE_MAX_CONNECTIONS=1
#export UCX_NET_DEVICES=ens22f0
#export GLOO_SOCKET_IFNAME=ens22f0
#export NCCL_SOCKET_IFNAME=ens22f0
#export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_2,mlx5_3"

# vpn
#export http_proxy=http://127.0.0.1:7890
#export HTTP_PROXY=http://127.0.0.1:7890
#export https_proxy=http://127.0.0.1:7890
#export HTTPS_PROXY=http://127.0.0.1:7890
unset http_proxy
unset HTTP_PROXY
unset https_proxy
unset HTTPS_PROXY

# network for swanlab, set ssh proxy to the login machine

TMUX_SESSION="ssh_proxy"
TMUX_WINDOW="proxy"

REMOTE_USER="wangzefan"

# SSH 要连到的目标（跳板）IP（或域名）和端口
REMOTE_HOST="10.1.5.231"
REMOTE_SSH_PORT=41198

# 本地用来监听 SOCKS5 的端口
LOCAL_SOCKS_PORT=17890

HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == "dmx-login01" || "$HOSTNAME" == "dmx-login02" || "$HOSTNAME" == "dmx-login03" ]]; then
  echo "[INFO] 当前主机 ($HOSTNAME) 已经是跳板机，跳过代理隧道建立。"
  # 如果脚本是被 source 的，使用 return；否则 exit
  return 0 2>/dev/null || exit 0
fi

if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "[INFO] tmux session '$TMUX_SESSION' 不存在，准备后台创建并启动自动重连隧道…"
  tmux new-session -d -s "$TMUX_SESSION" -n "$TMUX_WINDOW" bash -lc "\
while true; do
  echo \"[\$(date '+%F %T')] (tmux: $TMUX_SESSION/$TMUX_WINDOW) 正在建立 SSH 隧道…\"
  sshpass -p \"$PASSWORD\" ssh -o StrictHostKeyChecking=no \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      -N -D $LOCAL_SOCKS_PORT -p $REMOTE_SSH_PORT $REMOTE_USER@$REMOTE_HOST
  echo \"[\$(date '+%F %T')] SSH 隧道断开，5 秒后重连…\"
  sleep 5
done"
  echo "[INFO] 已创建 tmux session '$TMUX_SESSION'（窗口 '$TMUX_WINDOW'）并在后台启动自动重连。"
elif ! tmux list-windows -t "$TMUX_SESSION" 2>/dev/null | grep -q "^$TMUX_WINDOW"; then
  echo "[INFO] tmux session '$TMUX_SESSION' 存在，但没有窗口 '$TMUX_WINDOW'，准备新建…"
  tmux new-window -d -t "$TMUX_SESSION" -n "$TMUX_WINDOW" bash -lc "\
while true; do
  echo \"[\$(date '+%F %T')] (tmux: $TMUX_SESSION/$TMUX_WINDOW) 正在建立 SSH 隧道…\"
  sshpass -p \"$PASSWORD\" ssh -o StrictHostKeyChecking=no \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      -N -D $LOCAL_SOCKS_PORT -p $REMOTE_SSH_PORT $REMOTE_USER@$REMOTE_HOST
  echo \"[\$(date '+%F %T')] SSH 隧道断开，5 秒后重连…\"
  sleep 5
done"
  echo "[INFO] 已在 tmux session '$TMUX_SESSION' 中新增 '$TMUX_WINDOW' 窗口并启动自动重连。"
else
  echo "[INFO] tmux session '$TMUX_SESSION' 与窗口 '$TMUX_WINDOW' 都已存在，假定隧道在运行。"
fi

# 导出环境变量，让当前 shell 走本地 socks5 代理
export HTTP_PROXY="socks5h://127.0.0.1:${LOCAL_SOCKS_PORT}"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"

echo "[INFO] 已导出代理环境变量："
echo "       HTTP_PROXY=$HTTP_PROXY"
echo "       HTTPS_PROXY=$HTTPS_PROXY"
echo "       http_proxy=$http_proxy"
echo "       https_proxy=$https_proxy"
echo
echo "[TIP] 如要查看隧道日志/状态，请执行： tmux attach -t $TMUX_SESSION"
echo "[TEST] curl http://ifconfig.me  # 应该返回跳板机的公网 IP"