#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:4
#SBATCH --output=sft.txt
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00

#set -x
source ~/.bashrc
conda activate verl-raft

echo "Clearing caches..."
rm -rf ~/.triton/cache 2>/dev/null || true
rm -rf ~/.cache/triton 2>/dev/null || true
rm -rf ~/.torch 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

set -x
export VLLM_ATTENTION_BACKEND=XFORMERS

export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo

export RAY_TMPDIR=/dss/dssfs05/pn39qo/pn39qo-dss-0001/tmp
mkdir -p $RAY_TMPDIR

data=numina_math
project_name=raft++
algorithm=raft
model=Qwen2.5-Math-1.5B
model_name_or_path=Qwen/$model
policy_loss=vanilla # vanilla, plusplus (importance sample + clipping)
n=4
experiment_name=${model}-${algorithm}-${policy_loss}-${data}-n${n}
my_world_size=1

math_train_path=./data/$data/train.parquet
math_test_path=./data/math500/test.parquet 

train_files="['$math_train_path']"
test_files="['$math_test_path']"

mkdir -p logs/${project_name}

#ray start --head --port=6379 --object-manager-port=8076 --node-manager-port=8077 \
#	--dashboard-host=0.0.0.0 --dashboard-port=8265 --redis-password="" \
#	--temp-dir=$RAY_TMPDIR --disable-usage-stats
#
#sleep 10

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export MPICC=mpicc

export TRITON_CACHE_DIR=${RAY_TMPDIR}/triton_cache_${SLURM_JOB_ID}
mkdir -p $TRITON_CACHE_DIR
rm -rf ~/.triton/cache 2>/dev/null || true

export RAY_DISABLE_RUNTIME_METRICS=1
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DISABLE_MPI=1
export RAY_BACKEND_CONTEXT_MANAGER=0

echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$model_name_or_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.policy_loss=$policy_loss \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=$my_world_size \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.default_local_dir=/dss/dssfs05/pn39qo/pn39qo-dss-0001/ebony/checkpoints/${project_name}/${experiment_name} \
    trainer.test_freq=5 \
    trainer.total_epochs=1 2>&1 | tee -a /dss/dssfs05/pn39qo/pn39qo-dss-0001/ebony/logs/${project_name}/${experiment_name}.log

#rm -rf $RAY_TMPDIR
