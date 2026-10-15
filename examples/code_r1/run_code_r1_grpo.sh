#!/usr/bin/env bash
# Code-R1 GRPO（R1-Zero for code）—— 官方 main_grpo.sh 在 verl 0.10 上的写法。
# 配置里最需要对照的迁移点（完整对照表见 examples/code_r1/README.md 第三节）：
#   reward_model.reward_manager=prime                        -> CodeR1RewardManager（自定义，见下）
#   actor_rollout_ref.rollout.log_prob_micro_batch_size       -> ..._per_gpu
#   （新增）                                                  -> reward.num_workers
# 奖励管理器用 examples/code_r1/code_reward_manager.py 的 CodeR1RewardManager 换掉 naive：
# 它给单个 reward worker 内的沙箱并发上了信号量。naive 下 8 worker × 32 样本/worker 会开出
# 上百个 firejail（超配核数 2~12 倍），集体超时就是错误奖励。理由见该文件头部注释。
# 需要：单机 8 GPU + vLLM + 一个沙箱后端（Linux 用 firejail，见 README 第五节）。
set -xeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# ---------------- 可调参数 ----------------
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct-1M}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/code-r1-12k}

MAX_EPOCHS=${MAX_EPOCHS:-8}
ROLLOUT_N=${ROLLOUT_N:-16}                # 每个 prompt 采样多少条
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}  # 每步多少个 prompt
MICRO_BATCH_PER_GPU=${MICRO_BATCH_PER_GPU:-8}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
LR=${LR:-5e-7}

# 奖励并行度 = reward loop worker 数；沙箱真正吃 CPU（父进程只是在等），别盲目调大
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-8}
export CODER1_EXEC=${CODER1_EXEC:-firejail}   # firejail | local

NUM_CPUS=${NUM_CPUS:-$(nproc)}
# 每个 reward worker 内**并发执行**的沙箱数上限（CodeR1RewardManager 的信号量）。
# 总并发 ≈ CODE_REWARD_CONCURRENCY × REWARD_NUM_WORKERS，这里显式按核数算，不用
# code_reward_manager.py 里 cpu_count()//num_workers 的默认值（它取整机核数，忽略 cgroup）。
CODE_REWARD_CONCURRENCY=${CODE_REWARD_CONCURRENCY:-$((NUM_CPUS / REWARD_NUM_WORKERS))}
(( CODE_REWARD_CONCURRENCY > 0 )) || CODE_REWARD_CONCURRENCY=1   # 0 会让信号量永久阻塞
CODE_REWARD_RETRIES=${CODE_REWARD_RETRIES:-1}                    # 重试无退避，别调大
export CODE_REWARD_CONCURRENCY CODE_REWARD_RETRIES
export CODER1_TIMEOUT=${CODER1_TIMEOUT:-20}   # 短超时 + 重试，优于长超时（默认 30）

PROJECT_NAME=${PROJECT_NAME:-code-r1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-code-r1-12k-grpo}

# ---------------- 派生配置 ----------------
TOTAL_SAMPLES=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
GLOBAL_BATCH_SIZE=$((GPUS_PER_NODE * MICRO_BATCH_PER_GPU * GRAD_ACC_STEPS))
if (( TOTAL_SAMPLES % GLOBAL_BATCH_SIZE != 0 )); then
    echo "TRAIN_BATCH_SIZE * ROLLOUT_N ($TOTAL_SAMPLES) 必须能被" >&2
    echo "GPUS_PER_NODE * MICRO_BATCH_PER_GPU * GRAD_ACC_STEPS ($GLOBAL_BATCH_SIZE) 整除" >&2
    exit 1
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$GLOBAL_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=256 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=256 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    reward.num_workers=$REWARD_NUM_WORKERS \
    reward.reward_manager.source=importlib \
    reward.reward_manager.name=CodeR1RewardManager \
    reward.reward_manager.module.path=$REPO_ROOT/examples/code_r1/code_reward_manager.py \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.default_local_dir=./checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.save_freq=64 \
    trainer.test_freq=16 \
    trainer.val_before_train=False \
    trainer.total_epochs=$MAX_EPOCHS \
    "$@"
