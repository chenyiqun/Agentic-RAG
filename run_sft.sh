#!/bin/bash

cd /root/paddlejob/workspace/env_run/gpu
bash kill_gpu.sh
bash stop.sh
cd /root/paddlejob/workspace/env_run/verl

# Limit the visible GPUs to only GPU 7
export CUDA_VISIBLE_DEVICES=7

# 设置脚本路径
SCRIPT_PATH="bash /root/paddlejob/workspace/env_run/verl/examples/sft/gsm8k/run_qwen_05_peft.sh"

# 设置参数
NPROC_PER_NODE=1  # Since you're using only one GPU, set this to 1
SAVE_PATH=/root/paddlejob/workspace/env_run/verl/results

# 运行脚本
$SCRIPT_PATH $NPROC_PER_NODE $SAVE_PATH

cd /root/paddlejob/workspace/env_run/gpu
bash gpu.sh