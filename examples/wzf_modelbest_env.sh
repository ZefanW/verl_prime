# 设置全局变量，运行ray start以前运行
# 用来在notebook环境下运行
export PARALLEL_SIZE=1
export CKPT_PATH=/user/wangzefan/checkpoints_prime/
export WANDB_MODE=offline
export WANDB_DIR=/user/wangzefan
#export WANDB_BASE_URL=https://zefan.top/https://api.wandb.ai
unset WADB_BASE_URL

export SWANLAB_API_KEY='vP17PxpkO7x33BK1MWTK6'
export WANDB_API_KEY='194ba8b74c01d7f88fbf18db8f53206e24b2d46a'
export OUTLINES_CACHE_DIR=~/.cache/outlines/$(uuidgen)
export NCCL_DEBUG=WARN
unset VLLM_ATTENTION_BACKEND

unset PYTORCH_CUDA_ALLOC_CONF
export TOKENIZERS_PARALLELISM=true

# install packages with 7891 proxy
export http_proxy=http://whitelist-proxy.cybertron.svc.cluster.local:7891
export HTTP_PROXY=http://whitelist-proxy.cybertron.svc.cluster.local:7891
export https_proxy=http://whitelist-proxy.cybertron.svc.cluster.local:7891
export HTTPS_PROXY=http://whitelist-proxy.cybertron.svc.cluster.local:7891
export HF_ENDPOINT=https://hf-mirror.com
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install vllm==0.8.3
pip install transformers==4.52.4
pip install liger-kernel==0.5.10
#pip install docker-pycreds
pip install wandb==0.20
pip install antlr4-python3-runtime==4.9.3
pip install omegaconf
pip install codetiming
pip install torchdata
#pip install hydra
# network for swanlab and wandb, should start a clash host locally

TMUX_SESSION="ssh_proxy"
TMUX_WINDOW="proxy"

# 本地用来监听 SOCKS5 的端口，写死的
LOCAL_SOCKS_PORT=7891

tmux new-session -d -s "$TMUX_SESSION" -n "$TMUX_WINDOW" bash -lc "\
while true; do
  cd /user/wangzefan/verl_prime/clash-for-linux-without-sudo-main
  chmod +x bin/clash-linux-amd64
  ./bin/clash-linux-amd64 -d conf
  echo \"[\$(date '+%F %T')] SSH 隧道断开，5 秒后重连…\"
  sleep 5
done"
unset http_proxy
unset HTTP_PROXY
unset https_proxy
unset HTTPS_PROXY
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
