# 1.5B qwen math在eurus-math上训练，在MATH上评测（直接Top1)。训练尽量轻量，能看出基本的性能差别就行。还是要做数据筛选，这个步骤依然是必须的。
set -x

if [[ -v DECOUPLE ]]; then
  LAMC=1.0
else
  LAMC=$LAM
fi

#if [[ -v CE ]]; then
#  LOSS_TYPE=ce
#else
#  LOSS_TYPE=td
#fi
# LOSS_TYPE可以设置ce, td, sigtd

echo "running exp on ppo lam ${LAMC} ${LAM} with ${LOSS_TYPE} loss"

PROJECT_NAME='prime-lambda-exp'
EXPERIMENT_NAME="ppo-${LAMC}-${LAM}-${LOSS_TYPE}-fastrm"
SFT_MODEL_PATH=/home/wangzefan/huggingface/Qwen2.5-Math-1.5B
export WANDB_DIR=$WANDB_DIR/wandb_exp/$PROJECT_NAME
mkdir -p $WANDB_DIR/wandb
#    data.train_files="/home/wangzefan/dataset/dataset/prime/train.parquet" \
#    data.val_files="/home/wangzefan/dataset/dataset/prime/validation.parquet" \

python3 -m verl.trainer.main_ppo \
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
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0. \
    algorithm.lam_critic=$LAMC \
    algorithm.lam=$LAM \
    reward_model.enable=False \
    reward_model.reward_manager=prime \
    critic.optim.lr=1e-6 \
    critic.model.path=$SFT_MODEL_PATH \
    critic.model.enable_gradient_checkpointing=False \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.model.use_remove_padding=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.ppo_mini_batch_size=64 \
    critic.critic_loss=$LOSS_TYPE \
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