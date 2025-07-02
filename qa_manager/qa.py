import numpy as np
from typing import Dict, List, Any
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl import DataProto

from qa_manager.BaseAgent import *
from qa_manager.PlanningAgent import PlanningAgent


def remove_trailing_marker(text):
    # Check if the text ends with the marker and remove it
    marker = "<|im_end|>"
    if text.endswith(marker):
        return text[:-len(marker)]
    return text


class AgentWorkflow:
    def __init__(self, planning_agent) -> None:
        self.agent_pool = AgentPool()
        self.planning_agent = planning_agent

    def run(self, context: Dict, workflow: List[str]):

        # temp_context = {
        #         "original_query": questions[temp_i],
        #         "query": questions[temp_i],
        #         "parallel_sub": False,
        #         "is_sub": False,
        #         "current_step": -1,
        #         "answer": ""
        #     }

        if 'sub_query' not in context:
            if 'QueryDecompositionAgentParallel' in workflow:
                workflow = ['QueryDecompositionAgentParallel']
            elif 'QueryDecompositionAgentSerial' in workflow:
                workflow = ['QueryDecompositionAgentSerial']

            for agent_name in workflow:
                agent = self.agent_pool.get(agent_name)
                logger.info(f"\n\t==> Running Agent: {agent_name}")
                agent.run(context)
        else:
            pass

        # if context["answer"]:
        #     logger.info("\n\t\t--> Program terminated.")
        #     break

        # if "QueryDecompositionAgentParallel" in workflow:
        #     for sub_q in context["sub_query"]:
        #         # print('************ sub q number: {}'.format(len(context['sub_query'])))
        #         sub_context = AgentWorkflow(self.planning_agent).run(sub_q, is_sub=True, parallel_sub=True)
        #         context.setdefault("sub_answer", []).append(sub_context["answer"])

        #     agent = self.agent_pool.get("AnswerSummarizationAgent")
        #     logger.info("\n\t==> Running Agent: AnswerSummarizationAgent")
        #     agent.run(context)
        #     logger.info("\n\t\t--> Program terminated.")
        #     break

        context["current_step"] += 1
        if context["current_step"] != context["steps"]:
            logger.info(f"\n\t\t--> Serial steps remaining: {context['steps'] - context['current_step']}. Injecting new workflow.")
            if context["current_step"] >= 1:
                agent = self.agent_pool.get("QueryRewriteAgent")
                logger.info("\n\t==> Running Agent: QueryRewriteAgent")
                agent.run(context)
        # else:
        #     agent = self.agent_pool.get("AnswerSummarizationAgent")
        #     logger.info("\n\t==> Running Agent: AnswerSummarizationAgent")
        #     agent.run(context)
        #     logger.info("\n\t\t--> Serial steps execution finished.")
        #     break
        
        # print('************ answer: {}'.format(context['answer']))
        # print('\n')

        return context


class BatchAgentWorkflow:
    def __init__(self, planning_agent, max_workers: int = 4) -> None:
        self.max_workers = max_workers
        self.planning_agent = planning_agent

    def run_batch(self, questions: List[str]) -> List[str]:
        results = [None] * len(questions)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.run_single, question): idx
                for idx, question in enumerate(questions)
            }

            # for future in as_completed(future_to_index):
            #     idx = future_to_index[future]
            #     results[idx] = future.result()

            # Wrap the as_completed iterator with tqdm for progress display
            for future in tqdm(as_completed(future_to_index), total=len(questions), desc='Processing'):
                idx = future_to_index[future]
                results[idx] = future.result()

        return results

    def run_single(self, question: str) -> str:
        try:
            wf = AgentWorkflow(self.planning_agent)
            context = wf.run(question)

            return context["answer"]

        except Exception as e:
            logger.error(f"Error processing questionquestion '{question}': {e}")
            return f"question: {question}, error: {str(e)}"


class Agentic_RAG_Manager:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.planning_agent = PlanningAgent()
        self.agent_pool = AgentPool()

        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", True)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "right")

        self.api_url = 'http://localhost:8000/search'
    
    def trans_rawprompt_to_ids(self, batch_dict, is_sub_list):
        questions = [item['question'] for item in batch_dict['extra_info']]

        messages_list = [self.planning_agent.create_planning_messages(questions[i], is_sub=is_sub_list[i]) for i in range(len(questions))]

        # "input_ids", "attention_mask", "position_ids"
        update_dict_list = [self.get_single_ids(messages) for messages in messages_list]
        for i in range(len(update_dict_list)):
            update_dict = update_dict_list[i]

            for key in ["input_ids", "attention_mask", "position_ids"]:
                # print('*************')
                # print('key', key)
                # print('update_dict[key]', update_dict[key], len(update_dict[key]))
                # print('batch_dict[key][i]', batch_dict[key][i], len(batch_dict[key][i]))
                # print('*************')
                # print('\n')
                batch_dict[key][i] = update_dict[key]
        
        # "raw_prompt"
        new_raw_prompt_list = [update_dict["raw_prompt"] for update_dict in update_dict_list]
        batch_dict["raw_prompt"] = np.array(new_raw_prompt_list, dtype=object)

        # "raw_prompt_ids"
        new_raw_prompt_ids_list = [update_dict["raw_prompt_ids"] for update_dict in update_dict_list]
        batch_dict["raw_prompt_ids"] = np.array(new_raw_prompt_ids_list, dtype=object)

        return batch_dict
    
    def get_single_ids(self, messages):

        update_dict = {}

        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        update_dict["input_ids"] = input_ids[0]
        update_dict["attention_mask"] = attention_mask[0]
        update_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        update_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            update_dict["raw_prompt"] = messages

        return update_dict

    def get_answers_subs_list(self, batch: DataProto):
        workflows_1turn_text = self.get_answers_text(batch)  # batch size长度的list
        workflows_1turn = [self.parse_workflow(item) for item in workflows_1turn_text]  # batch size长度的list
        is_legal_list = [self.is_valid_list(item) for item in workflows_1turn]

        return workflows_1turn, is_legal_list

    def get_answers_text(self, data: DataProto):
        
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

        return predicted_answers_list

    def parse_workflow(self, workflow_string):
        # # Remove the 'Workflow: ' prefix and strip any leading/trailing whitespace
        # modules_string = workflow_string.replace('Workflow: ', '').strip()
        # # Split the modules by ', ' to get a list of individual modules
        # modules = modules_string.split(', ')

        mapping_dict = {
            'QR': 'QueryRewriteAgent',
            'QDP': 'QueryDecompositionAgentParallel',
            'QDS': 'QueryDecompositionAgentSerial',
            'R': 'RetrievalAgent',
            'DS': 'DocumentSelectionAgent',
            'AG': 'AnswerGenerationAgent',
            'AS': 'AnswerSummarizationAgent',
        }

        workflow_list = [mapping_dict[module] for module in workflow_string if module in mapping_dict]
        
        return workflow_list

    def is_valid_list(self, input_list, context):
        # 定义映射字典
        mapping_dict = {
            'QR': 'QueryRewriteAgent',
            'QDP': 'QueryDecompositionAgentParallel',
            'QDS': 'QueryDecompositionAgentSerial',
            'R': 'RetrievalAgent',
            'DS': 'DocumentSelectionAgent',
            'AG': 'AnswerGenerationAgent',
            'AS': 'AnswerSummarizationAgent'
        }
        
        # 如果列表为空，返回False
        if not input_list:
            return False
        
        # 检查列表中是否有重复元素
        if len(input_list) != len(set(input_list)):
            return False
        
        # 检查列表中的每个元素是否都在mapping_dict的键中
        for item in input_list:
            if item not in mapping_dict:
                # print('********* {} is not in agent pool. **********'.format(item))
                # print('input_list:', input_list)
                return False
        
        # # 在进行多sub-query改写的情况下，判断QDP QDS是否在开头
        # if ('QDP' in input_list) and (input_list[0]!='QDP'):
        #     return False
        # if ('QDS' in input_list) and (input_list[0]!='QDS'):
        #     return False

        # 在进行多sub-query改写的情况下，判断QDP QDS是否在开头
        if ('QDP' in input_list) and (len(input_list)!=1):
            return False
        if ('QDS' in input_list) and (len(input_list)!=1):
            return False

        # 在不进行多sub-query改写的情况下，判断结尾是否为AG
        if ('QDP' not in input_list) and ('QDS' not in input_list):
            if 'AG' not in input_list:
                return False
            if (input_list[-1] != 'AG'):
                return False

        # 如果之前已经进行过planning，说明本次不是第一次plan了。那么如果再出现串并行的query改写就是不对的。
        if context['begin_step'] >= 0:
            if 'QDP' in input_list:
                return False
            if 'QDS' in input_list:
                return False

        # 如果没有发现不合法的元素，返回True
        return True


class QA_Manager:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        # initial agent pool
        self.pool = AgentPool()
        self.pool.register([
            QueryRewriteAgent(AGENT_CONFIG['QueryRewriteAgent']),
            QueryDecompositionAgentParallel(AGENT_CONFIG['QueryDecompositionAgentParallel']),
            QueryDecompositionAgentSerial(AGENT_CONFIG['QueryDecompositionAgentSerial']),
            RetrievalAgent(AGENT_CONFIG['RetrievalAgent']),
            DocumentSelectionAgent(AGENT_CONFIG['DocumentSelectionAgent']),
            AnswerGenerationAgent(AGENT_CONFIG['AnswerGenerationAgent']),
            # IterativeWorkflowAgent(AGENT_CONFIG['IterativeWorkflowAgent'])
        ])
        self.Workflow = ["RetrievalAgent", "AnswerGenerationAgent"]
        # self.Workflow = ["AnswerGenerationAgent"]
        self.AnswerGenerationAgent = self.pool.get("AnswerGenerationAgent")

        self.max_prompt_length = config.get("max_prompt_length", 2048)
        self.return_raw_chat = config.get("return_raw_chat", True)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "right")

        self.api_url = 'http://localhost:8000/search'
    
    def trans_rawprompt_to_ids(self, batch_dict):
        questions = [item['question'] for item in batch_dict['extra_info']]
        answers = [item['answer'] for item in batch_dict['extra_info']]

        top_k_docs_list = []
        batch_size = 15  # Process questions in batches of 15
        for i in range(0, len(questions), batch_size):
            questions_batch = questions[i:i + batch_size]
            retrieved_docs_batch = self.retreiver(questions_batch)
            top_k_docs_list.extend(retrieved_docs_batch)

        messages_list = [self.AnswerGenerationAgent.build_message(questions[i], top_k_docs_list[i]) for i in range(len(questions))]

        # "input_ids", "attention_mask", "position_ids"
        update_dict_list = [self.get_single_ids(messages) for messages in messages_list]
        for i in range(len(update_dict_list)):
            update_dict = update_dict_list[i]

            for key in ["input_ids", "attention_mask", "position_ids"]:
                # print('*************')
                # print('key', key)
                # print('update_dict[key]', update_dict[key], len(update_dict[key]))
                # print('batch_dict[key][i]', batch_dict[key][i], len(batch_dict[key][i]))
                # print('*************')
                # print('\n')
                batch_dict[key][i] = update_dict[key]
        
        # "raw_prompt"
        new_raw_prompt_list = [update_dict["raw_prompt"] for update_dict in update_dict_list]
        batch_dict["raw_prompt"] = np.array(new_raw_prompt_list, dtype=object)

        # "raw_prompt_ids"
        new_raw_prompt_ids_list = [update_dict["raw_prompt_ids"] for update_dict in update_dict_list]
        batch_dict["raw_prompt_ids"] = np.array(new_raw_prompt_ids_list, dtype=object)

        return batch_dict
    
    def get_single_ids(self, messages):

        update_dict = {}

        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        update_dict["input_ids"] = input_ids[0]
        update_dict["attention_mask"] = attention_mask[0]
        update_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        update_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            update_dict["raw_prompt"] = messages

        return update_dict

    def retreiver(self, questions: List[str]) -> Dict[str, List[str]]:
        headers = {'Content-Type': 'application/json'}
        payload = {
            'questions': questions,
            'N': 5
        }
        response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an error for bad responses

        top_k_docs_list = []
        for question in response.json():
            temp_docs_list = []
            for doc_id, doc in enumerate(question['top_k_docs']):
                temp_docs_list.append(doc)
            top_k_docs_list.append(temp_docs_list)

        return top_k_docs_list


# # 使用示例
# model_name = "bert-base-uncased"  # 你可以选择其他模型
# qa_instance = QA_Manager(model_name)

# prompt = "What is the capital of France?"
# input_ids = qa_instance.trans_rawprompt_to_ids(prompt)
# attention_mask = qa_instance.get_attention_mask(prompt)
# position_ids = qa_instance.get_position_ids(prompt)

# print("Input IDs:", input_ids)
# print("Attention Mask:", attention_mask)
# print("Position IDs:", position_ids)