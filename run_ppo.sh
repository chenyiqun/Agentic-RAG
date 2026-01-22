#!/bin/bash

cd /root/workspace/env_run/gpu
bash kill_gpu.sh
bash stop.sh
cd /root/workspace/env_run/verl

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

nq_train_path=/root/workspace/env_run/agentic_rag_baseline/data/nq/train.parquet
nq_test_path=/root/workspace/env_run/agentic_rag_baseline/data/nq/test.parquet
nq_test_path_996=/root/workspace/env_run/agentic_rag_baseline/data/nq/test_996.parquet

hotpot_qa_train_path=/root/workspace/env_run/agentic_rag_baseline/data/hotpot_qa/train.parquet
hotpot_qa_test_path=/root/workspace/env_run/agentic_rag_baseline/data/hotpot_qa/test.parquet
hotpot_qa_test_path_996=/root/workspace/env_run/agentic_rag_baseline/data/hotpot_qa/test_996.parquet

pop_qa_test_path_996=/root/workspace/env_run/agentic_rag_baseline/data/pop_qa/test_996.parquet
ambig_qa_test_path_996=/root/workspace/env_run/agentic_rag_baseline/data/ambig_qa/test_996.parquet
2wikimultihop_qa_test_path_996=/root/workspace/env_run/agentic_rag_baseline/data/2wikimultihop_qa/test_996.parquet
musique_test_path_996=/root/workspace/env_run/agentic_rag_baseline/data/musique/test_996.parquet
bamboogle_test_path=/root/workspace/env_run/agentic_rag_baseline/data/bamboogle/test.parquet

train_files="['$nq_train_path']"
# test_files="['$nq_test_path']"
test_files="['$nq_test_path_996']"

# train_files="['$hotpot_qa_train_path']"
# # test_files="['$hotpot_qa_test_path']"
# test_files="['$hotpot_qa_test_path_996']"

# nq_distill_path=/root/workspace/env_run/agentic_rag_baseline/data/nq/train_distill.parquet
# train_files="['$nq_train_path']"
# # test_files="['$nq_test_path']"
# test_files="['$nq_distill_path']"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 -m verl.trainer.main_ppo_agentic_rag --config-path=./config --config-name='ppo_trainer'\
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=96 \
    data.val_batch_size=96 \
    data.max_prompt_length=1024 \
    data.max_response_length=16 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    actor_rollout_ref.model.path=/root/workspace/env_run/verl/checkpoints/verl_sft_nq/qwen2.5_1.5b_distillation/global_step_11 \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=96 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.optim.lr=1e-5 \
    critic.model.path=/root/workspace/env_run/verl/checkpoints/verl_sft_nq/qwen2.5_1.5b_distillation/global_step_11 \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.use_kl_in_reward=True \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.wandb_proxy=http://xxx \
    trainer.project_name='verl_ppo_nq' \
    trainer.experiment_name='qwen2.5_7b_gpt-3.5_nq' \
    trainer.n_gpus_per_node=6 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.total_epochs=1 $@


cd /root/workspace/env_run/gpu
bash gpu.sh
