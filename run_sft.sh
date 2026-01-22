#!/bin/bash

cd /root/workspace/env_run/gpu
bash kill_gpu.sh
bash stop.sh
cd /root/workspace/env_run/verl

hotpot_qa_distill_train_path=/root/workspace/env_run/agentic_rag_baseline/data/hotpot_qa/train_distillation.parquet
hotpot_qa_distill_test_path=/root/workspace/env_run/agentic_rag_baseline/data/hotpot_qa/test_distillation.parquet

nq_distill_train_path=/root/workspace/env_run/agentic_rag_baseline/data/nq/train_distillation.parquet
nq_distill_test_path=/root/workspace/env_run/agentic_rag_baseline/data/nq/test_distillation.parquet

train_files="$nq_distill_train_path"
test_files="$nq_distill_test_path"

set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

torchrun --nproc_per_node=6 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12345 -m verl.trainer.fsdp_sft_trainer --config-path=./config --config-name='sft_trainer' \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=question \
    data.response_key=answer \
    data.train_batch_size=96
    data.micro_batch_size_per_gpu=8 \
    data.prompt_key=question \
    data.response_key=answer \
    data.is_sub_key=is_sub \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    model.partial_pretrain=/root/workspace/env_run/verl/models_fund/Qwen/Qwen2.5-0.5B-Instruct \
    trainer.total_epochs=3 \
    trainer.logger=['console']
    trainer.project_name='verl_sft_nq' \
    trainer.experiment_name='qwen2.5_0.5b_distillation_nq' \
    trainer.wandb_proxy=http://xxx \
    save_freq=1000000 \
    test_freq=1000000 \
    nnodes=1 \
    n_gpus_per_node=6 \

cd /root/workspace/env_run/gpu
bash gpu.sh
