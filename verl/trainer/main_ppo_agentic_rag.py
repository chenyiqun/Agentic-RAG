# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""


import hydra
import ray
import torch

from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.ray_trainer_agentic_rag_2 import RayPPOTrainer
from verl import DataProto
from typing import Dict, List, Any

from collections import Counter


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none

        # vllm early verify
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer_agentic_rag_2 import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Load the reward manager for training and validation.
        # 后续要删掉
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        reward_manager = RewardManager(tokenizer)

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_manager=reward_manager,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


import random
import re
import string

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def remove_trailing_marker(text):
    # Check if the text ends with the marker and remove it
    marker = "<|im_end|>"
    if text.endswith(marker):
        return text[:-len(marker)]
    return text


class RewardManager():
    """The reward manager."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_format_penalty(self, data: DataProto, is_legal_list: List[bool]):
        # format penalty
        format_penalty_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        for i in range(len(data)):
            # get position
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

            # format penalty
            if is_legal_list[i]:
                format_penalty_tensor[i, valid_response_length - 1] = 0
            else:
                format_penalty_tensor[i, valid_response_length - 1] = -1.0

        return format_penalty_tensor

    def assign_rewards(self, data, metrics, context_list, all_metrics_dict_list, turn_id):
        # f1 score reward
        f1_rewards_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # batch是一个turn的, i 是遍历一个turn中batch size条数据
        for i in range(len(context_list)): 
            # get position
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

            begin_step = context_list[i]['begin_step']
            end_step = context_list[i]['end_step']
            f1_score = all_metrics_dict_list[i]['f1']
            
            # 只有当begin_step到end_step才有f1，其余的为0.
            if turn_id >= begin_step and turn_id <= end_step:
                f1_rewards_tensor[i, valid_response_length - 1] = f1_score
            else:
                f1_rewards_tensor[i, valid_response_length - 1] = 0

        return f1_rewards_tensor

    def assign_token_retrieval_cost(self, data, metrics, context_list, token_cost_list, retrieval_api_cost_list, turn_id):
         # token and retrieval cost
        token_retrieval_cost_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # batch是一个turn的, i 是遍历一个turn中batch size条数据
        for i in range(len(context_list)): 
            # get position
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

            begin_step = context_list[i]['begin_step']
            end_step = context_list[i]['end_step']

            token_cost = token_cost_list[i]
            retrieval_api_cost = retrieval_api_cost_list[i]
            # print('token_cost', token_cost)
            # print('retrieval_api_cost', retrieval_api_cost)
            if context_list[i]['mode'] == "serial" and turn_id == begin_step:
                turn_latency_cost = 0.25 * len(context_list[i]['sub_query'])
            elif context_list[i]['mode'] == "parallel" and turn_id == begin_step:
                turn_latency_cost = 0.25 * 1
            else:
                turn_latency_cost = 0.0
            # print('turn_latency_cost', turn_latency_cost)

            coeff_token, coeff_retrieval, coeff_turn = -1.0, -0.25, -0.5
            final_cost = coeff_token * token_cost + coeff_retrieval * retrieval_api_cost + coeff_turn * turn_latency_cost

            coeff_cost = 0.0
            final_cost = final_cost * coeff_cost
            # print('final_cost', final_cost)
            # print('\n')
            
            # 只有当begin_step到end_step才有f1，其余的为0.
            if turn_id >= begin_step and turn_id <= end_step:
                token_retrieval_cost_tensor[i, valid_response_length - 1] = final_cost
            else:
                token_retrieval_cost_tensor[i, valid_response_length - 1] = 0

        return token_retrieval_cost_tensor

    def get_rewards(self, data: DataProto, is_legal_list: List[bool]):
        # rewards
        metric_tensor_acc = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        metric_tensor_em = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        metric_tensor_f1 = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        metric_tensor_pre = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        metric_tensor_rec = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # format penalty
        format_penalty_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        predicted_answers_list = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            # valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            # valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            # sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences = valid_response_ids
            sequences_str = self.tokenizer.decode(sequences)
            sequences_str = remove_trailing_marker(sequences_str)
            predicted_answers_list.append(sequences_str)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            metrics = self.compute_scores(predict_answers=[sequences_str], golden_answers=[ground_truth])

            # score = metrics['f1']
            # reward_tensor[i, valid_response_length - 1] = score

            metric_tensor_acc[i, valid_response_length - 1] = metrics['acc']
            metric_tensor_em[i, valid_response_length - 1] = metrics['em']
            metric_tensor_f1[i, valid_response_length - 1] = metrics['f1']
            metric_tensor_pre[i, valid_response_length - 1] = metrics['precision']
            metric_tensor_rec[i, valid_response_length - 1] = metrics['recall']

            # format penalty
            if is_legal_list[i]:
                format_penalty_tensor[i, valid_response_length - 1] = 0
            else:
                format_penalty_tensor[i, valid_response_length - 1] = -1.0

            # final reward = reward + format penalty
            final_reward_tensor = metric_tensor_f1 + format_penalty_tensor

            # print('\n')
            # print('metric_tensor_f1', metric_tensor_f1)
            # print('format_penalty_tensor', format_penalty_tensor)
            # print('final_reward_tensor', final_reward_tensor)
            # print('\n')

        metrics_tensor_all = [metric_tensor_acc, metric_tensor_em, metric_tensor_f1, metric_tensor_pre, metric_tensor_rec]

        return final_reward_tensor, metrics_tensor_all


    def compute_scores(self, predict_answers, golden_answers):
        # print('\n')
        # print('*****************************')
        # print('predict_answers', predict_answers)
        # print('golden_answers', golden_answers)
        # print('*****************************')
        # print('\n')
        assert len(predict_answers) == len(golden_answers), "预测答案和标准答案的长度不相等"
        final_metric = {"acc": 0, "em": 0, "f1": 0, "precision": 0, "recall": 0}
        total = len(predict_answers)

        for prediction, ground_truths in zip(predict_answers, golden_answers):
            if isinstance(ground_truths, str):
                ground_truths = [ground_truths]
            elif isinstance(ground_truths, list):
                pass
            else:
                raise ValueError("The answer must be str or list format.")

            temp_metric_dict = {"acc": 0, "em": 0, "f1": 0, "precision": 0, "recall": 0}
            normalized_prediction = normalize_answer(prediction)
            for ground_truth in ground_truths:
                normalized_ground_truth = normalize_answer(ground_truth)

                if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                    continue
                if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                    continue
                
                if normalized_ground_truth in normalized_prediction:# or normalized_prediction in normalized_ground_truth:
                    temp_metric_dict["acc"] = 1.0

                if normalized_prediction == normalized_ground_truth:
                    temp_metric_dict["em"] = 1.0

                prediction_tokens = normalized_prediction.split()
                ground_truth_tokens = normalized_ground_truth.split()
                common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
                num_same = sum(common.values())
                if num_same == 0:
                    continue

                precision = 1.0 * num_same / len(prediction_tokens)
                recall = 1.0 * num_same / len(ground_truth_tokens)
                f1 = (2 * precision * recall) / (precision + recall)

                temp_metric_dict["precision"] = max(precision, temp_metric_dict["precision"])
                temp_metric_dict["recall"] =  max(recall, temp_metric_dict["recall"])
                temp_metric_dict["f1"] = max(f1, temp_metric_dict["f1"])
            
            for k in ['acc', 'em', 'f1', 'precision', 'recall']:
                final_metric[k] += temp_metric_dict[k]

        for k in ['acc', 'em', 'f1', 'precision', 'recall']:
            final_metric[k] /= total

        return final_metric


if __name__ == "__main__":
    main()
