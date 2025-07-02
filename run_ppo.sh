#!/bin/bash

# cd /root/paddlejob/workspace/env_run/gpu
# bash kill_gpu.sh
# bash stop.sh
# cd /root/paddlejob/workspace/env_run/verl

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

nq_train_path=/root/paddlejob/workspace/env_run/verl_z/data/nq/train.parquet
nq_test_path=/root/paddlejob/workspace/env_run/verl_z/data/nq/test.parquet

hotpot_qa_train_path=/root/paddlejob/workspace/env_run/verl_z/data/hotpot_qa/train.parquet
hotpot_qa_test_path=/root/paddlejob/workspace/env_run/verl_z/data/hotpot_qa/test.parquet

# train_files="['$gsm8k_train_path', '$math_train_path']"
# test_files="['$gsm8k_test_path', '$math_test_path']"

train_files="['$nq_train_path']"
test_files="['$nq_test_path']"

# train_files="['$hotpot_qa_train_path']"
# test_files="['$hotpot_qa_test_path']"

CUDA_VISIBLE_DEVICES=5,6 python3 -m verl.trainer.main_ppo_agentic_rag --config-path=./config --config-name='ppo_trainer'\
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=4 \
    data.max_prompt_length=1024 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    actor_rollout_ref.model.path=/root/paddlejob/workspace/env_run/verl/models_fund/Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    critic.optim.lr=1e-5 \
    critic.model.path=/root/paddlejob/workspace/env_run/verl/models_fund/Qwen/Qwen2.5-7B-Instruct \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.use_kl_in_reward=True \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.wandb_proxy=http://agent.baidu.com:8891 \
    trainer.project_name='verl_ppo_nq_debug' \
    trainer.experiment_name='qwen2.5_7b_debug' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100000 \
    trainer.test_freq=100000 \
    trainer.total_epochs=1 $@


# cd /root/paddlejob/workspace/env_run/gpu
# bash gpu.sh