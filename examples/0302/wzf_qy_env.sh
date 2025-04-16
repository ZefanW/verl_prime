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