set -x
# 用于测试main exp用到的所有功能

export OUTLINES_CACHE_DIR=~/.cache/outlines/$(uuidgen)
export NCCL_DEBUG=WARN
export WANDB_API_KEY='194ba8b74c01d7f88fbf18db8f53206e24b2d46a'
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=offline
export WANDB_DIR=/home/test/test05/wzf/verl_prime/

BASE_DIR=/home/test/test05/wzf/verl
HF_DIR=/home/test/test05/wzf/huggingface
SOLVABLE_NUMINA_PATH=/home/test/test05/cgq/data/numina_solvable
CODE_PATH=$BASE_DIR/datasets/code_1113_short
COMBINE_PATH=$BASE_DIR/datasets/combine1203
CODE_PATH=$BASE_DIR/datasets/code_1113_short
SOLVABLE_NUMINA_PATH=/home/test/test05/cgq/data/numina_solvable
PROJECT_NAME='o1_pr'
EXPERIMENT_NAME='gt-prm-online-before-solvable-0.2-0.8-ppo'

python3 -m verl.trainer.main_ppo \
    data.train_files=["$SOLVABLE_NUMINA_PATH/train.parquet","$CODE_PATH/train.parquet"] \
    data.val_files=["$SOLVABLE_NUMINA_PATH/test.parquet","$CODE_PATH/test.parquet"] \
    data.train_batch_size=256 \
    data.val_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    actor_rollout_ref.model.path=/home/test/test04/yuanjiarui/o1-sft/saves/qwen_all_abla_numina_oly_orca/full/qwen_all_abla_numina_oly_orca \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    critic.model.path=home/test/test04/yuanjiarui/o1-sft/saves/qwen_all_abla_numina_oly_orca/full/qwen_all_abla_numina_oly_orca \
    critic.ppo_micro_batch_ssize=8 \
    critic.optim.lr=1e-6 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$BASE_DIR/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=16 \
    trainer.test_freq=16 \
    trainer.total_epochs=1 \
    data.n_samples=4 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.2 \
    data.accuracy_upper_bound=0.8 \
    algorithm.adv_estimator=gae \
    algorithm.adv_params.verifier_gamma=1.0 \
    reward_model.rm_coef=0 \
    trainer.default_local_dir=$BASE_DIR/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME \

