# 设置全局变量，运行ray start以前运行
export TRAIN_DATASET=/data3/wzf/datasets/prime-rl-math-wo-prompt/train.parquet
export TEST_DATASET=/data3/wzf/datasets/prime-rl-math-wo-prompt/validation.parquet
export SFT_MODEL_PATH=/data3/wzf/huggingface/Qwen2.5-Math-7B
export PARALLEL_SIZE=2
export CKPT_PATH=/data1/verl_prime/checkpoints
export WANDB_MODE=offline
export OUTLINES_CACHE_DIR=~/.cache/outlines/$(uuidgen)
export NCCL_DEBUG=WARN
export WANDB_API_KEY='194ba8b74c01d7f88fbf18db8f53206e24b2d46a'
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=online
export WANDB_DIR=/data1/verl_prime/


# for eval
export EVAL_BASE=/home/wangzefan/data/PRIME/eval
export EVAL_RESULT_BASE=/home/wangzefan/data/verl_prime/eval_results
export CONDA_SH=/home/wangzefan/anaconda3/etc/profile.d/conda.sh

# ray init nodes
export WORKING_DIR=/data3/wzf/verl_prime
export PYTHONPATH=/data3/wzf/verl_prime

# network
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=INFO
export CUDA_DEVICE_MAX_CONNECTIONS=1
export UCX_NET_DEVICES=bond0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA="mlx5_0,mlx5_2,mlx5_5,mlx5_8"
