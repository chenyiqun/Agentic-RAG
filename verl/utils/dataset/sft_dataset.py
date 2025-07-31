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
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask

from typing import Dict, List, Any, Tuple, Optional


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


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        use_shm = config.get('use_shm', False)

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

        # is sub
        is_sub_key = config.get("is_sub_key", "is_sub")
        self.is_sub_key = is_sub_key if isinstance(is_sub_key, (tuple, list)) else [is_sub_key]

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]

        # # 假设 self.prompts 在某个上下文中已被定义
        # if isinstance(self.prompts, pd.Series):
        #     print("self.prompts 是一个 pandas Series 对象")
        # else:
        #     print("self.prompts 不是一个 pandas Series 对象")
        # if isinstance(self.prompts, pd.DataFrame):
        #     print("self.prompts 是一个 pandas DataFrame 对象")
        # else:
        #     print("self.prompts 不是一个 pandas DataFrame 对象")

        # for key in self.prompt_dict_keys:
        #     # type(x): pandas.core.series.Series
        #     # type(x[0]): numpy.ndarray
        #     # type(x[0][0]): dict
        #     try:
        #         self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
        #         # def debug_apply(x):
        #         #     item = series_to_item(x)
        #         #     print(f"x: {x}, item: {item}")
        #         #     return item[key]
        #         # self.prompts = self.prompts.apply(debug_apply, axis=1)
        #     except Exception:
        #         print(f"self.prompts={self.prompts}")
        #         raise
        
        if isinstance(self.prompts, pd.DataFrame):
            self.prompts = self.prompts.squeeze()
        self.prompts = self.prompts.tolist()

        # responses
        self.responses = self.dataframe[self.response_key]
        # for key in self.response_dict_keys:
        #     try:
        #         self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
        #     except Exception:
        #         print(f"self.responses={self.responses}")
        #         raise
        if isinstance(self.responses, pd.DataFrame):
            self.responses = self.responses.squeeze()
        self.responses = self.responses.tolist()

        self.is_sub_key = ['is_sub']

        # is sub
        self.is_sub_key = ['is_sub']
        self.is_subs = self.dataframe[self.is_sub_key]
        if isinstance(self.is_subs, pd.DataFrame):
            self.is_subs = self.is_subs.squeeze()
        self.is_subs = self.is_subs.tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]
        # is sub
        is_sub = self.is_subs[item]

        # apply chat template
        # prompt_chat = [{"role": "user", "content": prompt}]
        prompt_chat = create_planning_messages(prompt, is_sub)
        
        # string
        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        response_chat_str = response + tokenizer.eos_token

        # tokenize
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
