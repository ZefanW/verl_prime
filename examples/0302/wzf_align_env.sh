# 设置全局变量，运行ray start以前运行
export TRAIN_DATASET=/home/test/test05/wzf/verl_prime/datasets/prime-rl-math-wo-prompt/train.parquet
export TEST_DATASET=/home/test/test05/wzf/verl_prime/datasets/prime-rl-math-wo-prompt/validation.parquet
export SFT_MODEL_PATH=/home/test/test05/wzf/huggingface/Qwen2.5-Math-7B
export PARALLEL_SIZE=1
export CKPT_PATH=/home/test/test05/wzf/verl_prime/checkpoints
export WANDB_MODE=offline
export OUTLINES_CACHE_DIR=~/.cache/outlines/$(uuidgen)
export NCCL_DEBUG=WARN
export WANDB_API_KEY='194ba8b74c01d7f88fbf18db8f53206e24b2d46a'
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=offline
export WANDB_DIR=/home/test/test05/wzf/verl_prime/wandb2/

# for eval
export EVAL_BASE=/home/test/test05/wzf/PRIME/eval
export EVAL_RESULT_BASE=/home/test/test05/wzf/verl_prime/eval_results
export CONDA_SH=/home/test/test05/anaconda3/etc/profile.d/conda.sh