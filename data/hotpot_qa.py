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
Preprocess the hotpot qa dataset to parquet format
"""

import argparse
import os
import json

from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs
from typing import List


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./hotpot_qa")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "hotpotqa/hotpot_qa"

    # dataset = datasets.load_dataset(data_source, 'fullwiki', trust_remote_code=True)

    # train_dataset = dataset['train']
    # test_dataset = dataset['validation']

    train_dir = "./hotpot_qa/hotpotqa_train_questions_and_answers.json"
    test_dir = "./hotpot_qa/hotpotqa_test_questions_and_answers.json"

    with open(train_dir, 'r', encoding='utf-8') as f:
        train_dataset = json.load(f)
        train_dataset = Dataset.from_list(train_dataset)

    with open(test_dir, 'r', encoding='utf-8') as f:
        test_dataset = json.load(f)
        test_dataset = Dataset.from_list(test_dataset)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw

            solution = example.pop("answer")
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "QA",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_json(os.path.join(local_dir, "train.jsonl"))
    test_dataset.to_json(os.path.join(local_dir, "test.jsonl"))

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
