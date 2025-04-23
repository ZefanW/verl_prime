# 基本的32B baseline，用after来降低开销
# 4 machine, tp=2


PROJECT_NAME='PRIMER'
EXPERIMENT_NAME='baseline-dapo-608cont-64cont-80cont-224cont'
SFT_MODEL_PATH=/home/wangzefan/huggingface/Qwen2.5-32B
DAPO=/home/wangzefan/dataset/dataset/prime-rl-math-wo-prompt/dapo.parquet
AIME=/home/wangzefan/dataset/dataset/prime-rl-math-wo-prompt/aime2024_32.parquet
AIME24_TTRL=/home/wangzefan/dataset/dataset/prime-rl-math-wo-prompt/aime2024_128_ttrl.parquet
AIME25_TTRL=/home/wangzefan/dataset/dataset/prime-rl-math-wo-prompt/aime2025_128_ttrl.parquet
# dynamic batch size不太重要
PARALLEL_SIZE=1

export WANDB_DIR=$WANDB_DIR/wandb_exp/$PROJECT_NAME
mkdir -p $WANDB_DIR/wandb

# 注意actor_rollout_ref.model.ref_path需要和reward_model.ref_path一致，否则reward model的ref model可能会出错
# PRIMER训练曲线里的kl loss暴涨就是这个问题导致的，ref model不会load ckpt，而是永远和actor_rollout.model.ref_path相同，默认值为model.path

python3 -m recipe.prime.main_prime \
    data.train_files="[$DAPO]" \
    data.val_files="$AIME" \
    data.train_batch_size=64 \
    data.val_batch_size=6312 \
    data.max_prompt_length=1024 \
    data.max_response_length=7168 \
    data.filter_accuracy=True \
    data.filter_truncate=False \
    data.resample=True \
    data.accuracy_lower_bound=0.05 \
    data.accuracy_upper_bound=0.95 \
    data.oversample_factor=1 \
    actor_rollout_ref.model.path=/home/wangzefan/data/verl_prime/checkpoints/PRIMER/baseline-dapo-608cont-64cont-80cont/global_step_224/actor/huggingface \
    actor_rollout_ref.model.ref_path=$SFT_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.clip_high=0.28 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.max_num_batched_tokens=40000 \
    actor_rollout_ref.rollout.max_seq_len_to_capture=8192 \
    actor_rollout_ref.actor.use_token_level_loss=True \
    algorithm.adv_estimator=rloo \
    algorithm.lam=0. \
    algorithm.reward_gt_coef=5 \
    algorithm.reward_dpo_coef=5 \
    reward_model.model.path=/home/wangzefan/data/verl_prime/checkpoints/PRIMER/baseline-dapo-608cont-64cont-80cont/global_step_224/reward/huggingface \
    reward_model.model.ref_path=$SFT_MODEL_PATH \
    reward_model.micro_batch_size_per_gpu=1 \
    reward_model.model.update=after \
    reward_model.prime_norm=batch_norm \
    reward_model.model.loss_type=ce \
    reward_model.model.ref_type=freeze \
    reward_model.model.beta_train=0.05 \
    reward_model.model.optim.lr=1e-6 \
    reward_model.model.optim.grad_clip=10.0 \
    reward_model.model.input_tokenizer=null \
    reward_model.mini_batch_size=64 \
    reward_model.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=8 \
    trainer.save_freq=16 \
    trainer.test_freq=16 \
    trainer.total_epochs=10 \
    trainer.val_generations_to_log_to_wandb=64 \
    trainer.default_local_dir="$CKPT_PATH"/"$PROJECT_NAME"/"$EXPERIMENT_NAME" \
    trainer.validate_sample=True