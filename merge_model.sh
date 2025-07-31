python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /root/paddlejob/workspace/env_run/verl/checkpoints/verl_ppo_nq/qwen2.5_7b_alpha0.25/global_step_25/actor \
    --target_dir /root/paddlejob/workspace/env_run/verl/merged_models/verl_ppo_nq/qwen2.5_7b_alpha0.25/global_step_25/actor