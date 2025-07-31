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
from typing import Dict, List, Any, Tuple, Optional

from verl.utils.hdfs_io import copy, makedirs


def create_planning_messages(question: str, is_sub: bool = False) -> List[Dict[str, str]]:
    if not is_sub:
        messages = [
            {"role": "system", "content": "You are a helpful assistant to plan a Workflow for the given information using the give tool/agent."},
            {"role": "assistant", "content": "Ok, please show me you requirement information and the available tool/agent"},
            {"role": "user", "content": """The descriptions of the available tool/agent are as follows: 

    Query Rewriter (QR): The input of QR is a question, and the output of QR is a rewritten question which is more concise, clearer, and more accurate.
    Query Decomposition Parallel (QDP): The input of QDP is a question, and the output of QDP contains multiple independent sub-questions. And QDP is suitable for decomposing those multi-hop questions that can be decomposed into multiple searchable and independent sub-questions in parallel.
    Query Decomposition Serial (QDS): The input of QDS is a question. The output of QDS is multiple related sub-questions, and later sub-questions may need answers to previous sub-questions before they become searchable. And QDS is suitable for decomposing those multi-hop questions with dependencies into multiple sub-questions.
    Retrieval (R): The input of R is a question, and the output of R is multiple relevant candidate documents to the given question.
    Document Selector (DS): The input of DS is a given question and multiple relevant candidate documents from search engine, and the output of DS is the documents which are helpful to answer the given question.
    Answer Generator (AG): The input of G may contains the given question, the some related candidate documents from search engine. The output of AG is the answer to the given question.

    With above tools/agents you can build a Workflow to answer a given question.


    The followings are some examples:

    For simple questions that may require external knowledge to answer:
    Question: "Which team is the 2025 NBA champion?"
    Workflow: R,AG
    We can just retrieve the external information about the given question through Retrieval Agent (RA), then the Answer Generator (AG) can output the answer to the given question.

    For multi-hop questions like this:
    Question: "Which city will have the highest GDP in 2024: Singapore, Houston, Beijing?"
    Workflow: QDP
    This kind of problem requires first breaking the initial question into multiple independent searchable sub-questions. For such a question, only the QDP needs to be given, and no other tool/agent needs to be given.

    For multi-hop questions like this:
    Question: "What is the most played song on Michael Jackson's third best-selling album?"
    Workflow: QDS
    This kind of problem requires breaking down the original problem into multiple subproblems that depend on each other. Convenient for subsequent processing. For such a question, only the QDS needs to be given, and no other tool/agent needs to be given.

    For multi-hop questions like this:
    Question: When was the director of Terminator 2 born?
    Workflow: QDS
    This kind of problem requires breaking down the original problem into multiple subproblems that depend on each other. Convenient for subsequent processing. For such a question, only the QDS needs to be given, and no other tool/agent needs to be given.



    Next I'll give you the Question that need to be answered, just output the appropriate Workflow in the format above, don't output anything else.
    """},
            {"role": "assistant", "content": "Ok, you can provide the Question and I will give the appropriate Workflow to answer the Question."},
            {"role": "user", "content": """Please give the appropriate Workflow in the format requested above.
    Qustion: "{}"
    Workflow: """.format(question)},
            ]

    else:
            # current_step = context["current_step"]
            # question = context['sub_query'][current_step]
            messages = [
                {"role": "system", "content": "You are a helpful assistant to plan a Workflow for the given information using the give tool/agent."},
                {"role": "assistant", "content": "Ok, please show me you requirement information and the available tool/agent"},
                {"role": "user", "content": """The descriptions of the available tool/agent are as follows: 

    Retrieval (R): The input of R is a question, and the output of R is multiple relevant candidate documents to the given question.
    Document Selector (DS): The input of DS is a given question and multiple relevant candidate documents from search engine, and the output of DS is the documents which are helpful to answer the given question.
    Answer Generator (AG): The input of G may contains the given question, the some related candidate documents from search engine. The output of AG is the answer to the given question.

    With above tools/agents you can build a Workflow to answer a given question.


    The followings are some examples:

    For the easiest question, the Answer Generator (AG) can answer it directly without any query reformulation and external information:
    Question: "Which city is the capital of Australia."
    Workflow: AG
    Then, we can use Answer Generator (AG) to get the answer directly.

    For simple questions that may require external knowledge to answer:
    Question: "Which team is the 2025 NBA champion?"
    Workflow: R,AG
    We can just retrieve the external information about the given question through Retrieval Agent (RA), then the Answer Generator (AG) can output the answer to the given question.


    Next I'll give you the Question that need to be answered, just output the appropriate Workflow in the format above, don't output anything else.
    """},
                {"role": "assistant", "content": "Ok, you can provide the Question and I will give the appropriate Workflow to answer the Question."},
                {"role": "user", "content": """Please give the appropriate Workflow in the format requested above.
    Qustion: "{}"
    Workflow: """.format(question)},
            ]

    return messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./nq")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "nq/distillation"

    train_dir = "/root/paddlejob/workspace/env_run/verl/distillation_data_nq.jsonl"
    test_dir = "/root/paddlejob/workspace/env_run/verl/distillation_data_nq.jsonl"

    def load_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    # 读取训练数据集
    train_dataset = load_jsonl(train_dir)
    train_dataset = Dataset.from_list(train_dataset)

    # 读取测试数据集，提取最后 996 条记录
    test_dataset = load_jsonl(test_dir)
    test_dataset = Dataset.from_list(test_dataset)

    # with open(train_dir, 'r', encoding='utf-8') as f:
    #     train_dataset = json.load(f)
    #     train_dataset = Dataset.from_list(train_dataset)

    # with open(test_dir, 'r', encoding='utf-8') as f:
    #     test_dataset = json.load(f)[-996:]
    #     test_dataset = Dataset.from_list(test_dataset)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw

            solution = example.pop("workflow")
            is_sub = example.pop("is_sub")
            data = {
                "data_source": data_source,
                "question": question,
                "answer": solution,
                "is_sub": is_sub,
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

    train_dataset.to_json(os.path.join(local_dir, "train_distillation.jsonl"))
    test_dataset.to_json(os.path.join(local_dir, "test_distillation.jsonl"))

    train_dataset.to_parquet(os.path.join(local_dir, "train_distillation.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test_distillation.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
