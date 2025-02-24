set -x
# 用于测试main exp用到的所有功能

export OUTLINES_CACHE_DIR=~/.cache/outlines/$(uuidgen)
export NCCL_DEBUG=WARN
export WANDB_API_KEY='194ba8b74c01d7f88fbf18db8f53206e24b2d46a'
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=offline
export WANDB_DIR=/data3/wzf/verl_prime/

BASE_DIR=/data1/verl_prime
HF_DIR=/data3/wzf/huggingface
SOLVABLE_NUMINA_PATH=/data3/wzf/datasets/numina_solvable
CODE_PATH=/data3/wzf/datasets/code_1113_short
PROJECT_NAME='o1_rm'
EXPERIMENT_NAME='s28-prime-fastrm-nogt-orm'

PARALLEL_SIZE=2

python3 -m verl.trainer.main_ppo \
    data.train_files=["$SOLVABLE_NUMINA_PATH/train.parquet","$CODE_PATH/train.parquet"] \
    data.val_files=["$SOLVABLE_NUMINA_PATH/test.parquet","$CODE_PATH/test.parquet"] \
    data.train_batch_size=256 \
    data.val_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    actor_rollout_ref.model.path=$HF_DIR/Eurus-2-7B-SFT\
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((256*PARALLEL_SIZE)) \
    actor_rollout_ref.actor.ppo_micro_batch_size=$((8*PARALLEL_SIZE)) \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$((64*PARALLEL_SIZE)) \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$((64*PARALLEL_SIZE)) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$BASE_DIR/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=16 \
    trainer.test_freq=16 \
    trainer.total_epochs=1 \
    data.n_samples=4 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.2 \
    data.accuracy_upper_bound=0.8 \
    algorithm.adv_estimator=rloo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    reward_model.rm_type=prime \
    reward_model.rm_coef=5 \
    reward_model.prime_model.path=$HF_DIR/Eurus-2-7B-SFT \
    reward_model.prime_model.ref_path=$HF_DIR/Eurus-2-7B-SFT \
    reward_model.model.input_tokenizer=null \
    reward_model.prime_granularity=whole \
    reward_model.micro_batch_size=$((8*PARALLEL_SIZE)) \
    reward_model.prime_model.update=before \
    reward_model.prime_model.beta_train=0.05 \
    reward_model.prime_model.optim.lr=1e-6 \
    reward_model.prime_model.optim.grad_clip=10.0 \
    reward_model.prime_model.input_tokenizer=null \
    reward_model.prime_lambda=0.5 \
    reward_model.mini_batch_size=$((64*PARALLEL_SIZE)) \
    trainer.default_local_dir=$BASE_DIR/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.model.use_remove_padding=True \
    verifier.reward_coef=0 \
