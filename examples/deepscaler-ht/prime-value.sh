# prime在保持advantage function不变的情况下转value model，允许直接设置lambda
set -x

PROJECT_NAME='deepscaler1.5b-t1.0'
#EXPERIMENT_NAME="primepp-stable-milde-${LAM}-q0+-${REF_TYPE}"
EXPERIMENT_NAME="primev-cetruncate-nobox-nogt-${LAM}-middlece-linear"
SFT_MODEL_PATH=/home/wangzefan/huggingface/DeepSeek-R1-Distill-Qwen-1.5B
DEEPSCALER=/home/wangzefan/dataset/dataset/prime-rl-math-wo-prompt/deepscaler_nobox.parquet
AIME=/home/wangzefan/dataset/dataset/prime-rl-math-wo-prompt/aime2024_32.parquet
AMC=/home/wangzefan/dataset/dataset/prime-rl-math-wo-prompt/amc_32.parquet
DAPO=/home/wangzefan/dataset/dataset/prime-rl-math-wo-prompt/dapo.parquet

export WANDB_DIR=$WANDB_DIR/wandb_exp/$PROJECT_NAME
mkdir -p $WANDB_DIR/wandb

python3 -m recipe.prime.main_prime \
    data.train_files="$DEEPSCALER" \
    data.val_files="$AIME" \
    data.train_batch_size=128 \
    data.val_batch_size=6312 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_accuracy=True \
    data.filter_truncate=False \
    data.resample=True \
    data.accuracy_lower_bound=0.05 \
    data.accuracy_upper_bound=0.95 \
    data.oversample_factor=1 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.ref_path=$SFT_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0.2 \
    actor_rollout_ref.actor.entropy_type=Adaptive \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_seq_len_to_capture=9216 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    algorithm.adv_estimator=prime_middle_ce_linear \
    algorithm.reward_gt_coef=0 \
    algorithm.reward_dpo_coef=10 \
    algorithm.lam=$LAM \
    reward_model.model.path=$SFT_MODEL_PATH \
    reward_model.model.ref_path=$SFT_MODEL_PATH \
    reward_model.micro_batch_size_per_gpu=1 \
    reward_model.model.update=before \
    reward_model.model.beta_train=0.05 \
    reward_model.model.beta_test=0.05 \
    reward_model.model.optim.lr=1e-6 \
    reward_model.model.optim.grad_clip=10.0 \
    reward_model.model.input_tokenizer=null \
    reward_model.mini_batch_size=64 \
    reward_model.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    reward_model.model.loss_type=middle_ce \
    reward_model.model.truncate=True \
    reward_model.prime_norm=none \
    reward_model.model.ref_type=freeze \
    trainer.val_before_train=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=16 \
    trainer.test_freq=16 \
    trainer.total_epochs=100 \
    trainer.default_local_dir="$CKPT_PATH"/"$PROJECT_NAME"/"$EXPERIMENT_NAME" \
    trainer.validate_sample=True \
    trainer.val_before_train=False \
#    actor_rollout_ref.actor.ppo_epochs=5e-5 \
#    actor_rollout_ref.actor.clip_high=adaptive_bound \

