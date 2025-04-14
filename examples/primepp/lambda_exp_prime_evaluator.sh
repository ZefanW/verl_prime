# PRIME当成value model来用，最终更新公式用MC return。需要一个特殊的adv estimator来完成这件事
# 比较reward model性质需要用最科学的算法，需要decouple DPO loss，reverse mode
set -x

if [[ -v POLICY ]]; then
  REF_TYPE='policy+freeze'
else
  REF_TYPE=freeze
fi

echo "running exp on prime lam ${LAM}"

PROJECT_NAME='prime-lambda-exp'
EXPERIMENT_NAME="prime-${LAM}-strict-dpo-tll-${REF_TYPE}-noavgpar"
SFT_MODEL_PATH=/home/wangzefan/huggingface/Qwen2.5-Math-1.5B
export WANDB_DIR=$WANDB_DIR/wandb_exp/$PROJECT_NAME
mkdir -p $WANDB_DIR/wandb
#    data.train_files="/home/wangzefan/dataset/dataset/prime/train.parquet" \
#    data.val_files="/home/wangzefan/dataset/dataset/prime/validation.parquet" \

python3 -m recipe.prime.main_prime \
    data.train_files="/home/wangzefan/dataset/dataset/prime-rl-math-wo-prompt/math7500.parquet" \
    data.val_files="/home/wangzefan/dataset/dataset/prime-rl-math-wo-prompt/math500.parquet" \
    data.train_batch_size=64 \
    data.val_batch_size=6312 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_accuracy=True \
    data.filter_truncate=False \
    data.resample=True \
    data.accuracy_lower_bound=0.2 \
    data.accuracy_upper_bound=0.8 \
    data.oversample_factor=1 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    actor_rollout_ref.actor.use_token_level_loss=True \
    algorithm.adv_estimator=prime \
    algorithm.kl_ctrl.kl_coef=0. \
    algorithm.lam=$LAM \
    reward_model.enable=True \
    reward_model.reward_manager=prime \
    algorithm.reward_gt_coef=5 \
    algorithm.reward_dpo_coef=5 \
    reward_model.model.path=$SFT_MODEL_PATH \
    reward_model.micro_batch_size_per_gpu=1 \
    reward_model.model.update=after \
    reward_model.model.beta_train=0.05 \
    reward_model.model.optim.lr=1e-6 \
    reward_model.model.optim.grad_clip=10.0 \
    reward_model.model.input_tokenizer=null \
    reward_model.mini_batch_size=256 \
    reward_model.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    reward_model.prime_norm=none \
    reward_model.model.loss_type=dpo \
    reward_model.model.update=reverse \
    reward_model.model.ref_type=$REF_TYPE \
    critic.optim.lr=1e-6 \
    critic.model.path=$SFT_MODEL_PATH \
    critic.model.enable_gradient_checkpointing=False \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.model.use_remove_padding=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.balance_batch=False \
    trainer.save_freq=16 \
    trainer.test_freq=16 \
    trainer.total_epochs=10 \
    trainer.default_local_dir="$CKPT_PATH"/"$PROJECT_NAME"/"$EXPERIMENT_NAME"