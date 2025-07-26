# 和vineppo设置对齐

set -x


PROJECT_NAME='prime_rebuttal-fixn'
EXPERIMENT_NAME='tiny-prime-rho-prime-low-weight-nobox'
SFT_MODEL_PATH=/home/wangzefan/huggingface/rho-1b-sft-MATH
export WANDB_DIR=$WANDB_DIR/wandb_exp/$PROJECT_NAME
mkdir -p $WANDB_DIR/wandb

python3 -m recipe.prime.main_prime \
    data.train_files="/home/wangzefan/dataset/dataset/prime-rl-math-wo-prompt/math7500_nobox.parquet" \
    data.val_files="/home/wangzefan/dataset/dataset/prime-rl-math-wo-prompt/math500_nobox.parquet" \
    data.train_batch_size=64 \
    data.val_batch_size=6312 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_accuracy=True \
    data.filter_truncate=False \
    data.resample=True \
    data.accuracy_lower_bound=0.2 \
    data.accuracy_upper_bound=0.8 \
    data.oversample_factor=1 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    algorithm.adv_estimator=rloo \
    algorithm.reward_gt_coef=5 \
    algorithm.reward_dpo_coef=1 \
    reward_model.model.path=$SFT_MODEL_PATH \
    reward_model.micro_batch_size_per_gpu=4 \
    reward_model.model.update=after \
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
    trainer.save_freq=16 \
    trainer.test_freq=16 \
    trainer.total_training_steps=256 \
    trainer.val_before_train=True \
    trainer.default_local_dir="$CKPT_PATH"/"$PROJECT_NAME"/"$EXPERIMENT_NAME"

#    actor_rollout_ref.rollout.stop='["\n\nQuestion:","\n\nProblem:","[MATH_TASK]","\n\n"]' \

