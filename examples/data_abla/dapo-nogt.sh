set -x

# baseline for zero experiment
# main changes:
# n_samples=8
# filter_truncate=True
# accuracy_lower_bound = 0.1
# accuracy_upper_bound=0.9
# max_response_length=3072 math base model的上限
# oversample=4
# still without warmup
# gradient_checkpointing=True
# remove_padding=True
# rm/critic bs不一定，但基本上还是256和64两种选项，但训练方式一定是before
# reward model clip range 仍为10， 因为本来rm的grad norm就可以保持在10左右
# 需要在文件外设置几个全局变量

PROJECT_NAME='data-abla-n4'
EXPERIMENT_NAME='dapo-nogt'
SFT_MODEL_PATH=/home/wangzefan/huggingface/Qwen2.5-Math-7B
DAPO=/home/wangzefan/dataset/dataset/prime-rl-math-wo-prompt/dapo.parquet
AIME=/home/wangzefan/dataset/dataset/prime-rl-math-wo-prompt/aime2024_32.parquet
AMC=/home/wangzefan/dataset/dataset/prime-rl-math-wo-prompt/amc_32.parquet

export WANDB_DIR=$WANDB_DIR/wandb_exp/$PROJECT_NAME
mkdir -p $WANDB_DIR/wandb

python3 -m recipe.prime.main_prime \
    data.train_files="$DAPO" \
    data.val_files="$AMC" \
    data.train_batch_size=64 \
    data.val_batch_size=6312 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_accuracy=True \
    data.filter_truncate=False \
    data.resample=True \
    data.accuracy_lower_bound=0.05 \
    data.accuracy_upper_bound=0.95 \
    data.oversample_factor=1 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    algorithm.adv_estimator=rloo \
    algorithm.reward_gt_coef=0 \
    algorithm.reward_dpo_coef=5 \
    reward_model.model.path=$SFT_MODEL_PATH \
    reward_model.micro_batch_size_per_gpu=1 \
    reward_model.model.update=before \
    reward_model.model.beta_train=0.05 \
    reward_model.model.optim.lr=1e-6 \
    reward_model.model.optim.grad_clip=10.0 \
    reward_model.model.input_tokenizer=null \
    reward_model.mini_batch_size=64 \
    reward_model.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    trainer.val_before_train=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=32 \
    trainer.test_freq=16 \
    trainer.total_epochs=10 \
    trainer.default_local_dir="$CKPT_PATH"/"$PROJECT_NAME"/"$EXPERIMENT_NAME"

