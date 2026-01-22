import time
from abc import ABC, abstractmethod
from openai import OpenAI
from typing import Dict, List, Any, Tuple, Optional
from qa_manager.config import AGENT_CONFIG, EXAMPLE_PROMPT
import requests
import json
from qa_manager.tools import TokenUsageTracker, setup_logger, setup_logger_no_print
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from threading import Lock
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from qa_manager.PlanningAgent import PlanningAgent



class BaseAgent(ABC):
    def __init__(self, config: Dict):
        self.name = config.get('name', '')

    def run(self, context: Dict[str, Any]) -> None:
        self.log(context)

    def log(self, context: Dict[str, Any]) -> None:
        # logger.info(f"\n\t\t-----------------------------------------")
        # logger.info(f"\t\t--             --CONTEXT--               ")
        # for key, value in context.items():
        #     if key in ["sub_query", "sub_answer", "results"]:
        #         logger.info(f"\t\t-- {key.upper()} --")
        #         for index, result in enumerate(value):
        #             result_str = ', '.join([r.split('\n')[0] for r in result]) if key == "results" else result
        #             logger.info(f"\t\t\t- {index} - \t {result_str}")
        #     else:
        #         logger.info(f"\t\t-- {key.upper()} -- \t {value}")
        # logger.info(f"\t\t-----------------------------------------")
        pass


class ApiEnabledAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model = config.get('model', 'gpt-4o-mini')
        self.api_key = config.get('api_key', '')
        self.api_base = config.get('api_base', 'https://api.openai.com/v1')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2000)
        self.timeout = config.get('timeout', 30)
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    def get_response(self, message: List[Dict]) -> str:
        """
        通用的响应获取方法，调用OpenAI API。
        """
        # for m in message:
        #     logger.info(f"\t\t{m['role'].upper()} - {m['content']}")

        response = ''
        max_waiting_time = 16
        current_sleep_time = 1
        while response == '':
            try:
                response = self.client.chat.completions.create(
                    messages=message,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )

            except Exception as e:
                logger.error(f'{e}')
                time.sleep(current_sleep_time)
                if current_sleep_time < max_waiting_time:
                    current_sleep_time *= 2

        usage = {
            'agent_name': self.name,
            'model': response.model,
            'completion_tokens': response.usage.completion_tokens,
            'prompt_tokens': response.usage.prompt_tokens,
            'total_tokens': response.usage.total_tokens,
        }
        tracker = TokenUsageTracker()
        tracker.record(usage)

        # logger.info(f"\t\tRESPONSE - {response.choices[0].message.content}")
        return response.choices[0].message.content

    @abstractmethod
    def build_message(self, *args, **kwargs) -> List[Dict]:
        """
        构建输入消息的方法。不同的agent可以有不同的实现。
        """
        pass

    @abstractmethod
    def post_process(self, *args, **kwargs):
        """
        后处理API响应的结果。
        """
        pass

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        封装调用流程：构造 prompt -> 获取响应 -> 后处理
        """
        message = self.build_message(context)
        raw_response = self.get_response(message)
        self.post_process(context, raw_response)
        super().log(context)

        return context


class QueryRewriteAgent(ApiEnabledAgent):
    def set_query(self, context: Dict[str, Any], query: str) -> None:
        if context["mode"] in ["serial", "parallel"]:
            context["sub_query"][context["current_step"]] = query
        else:
            context["query"] = query

    def get_query(self, context: Dict[str, Any]) -> str:
        if context["mode"] in ["serial", "parallel"]:
            return context["sub_query"][context["current_step"]]
        else:
            return context["query"]

    def build_message(self, context: Dict[str, Any]) -> List[Dict]:
        if context["mode"] == "serial":
            query = context["query"]
            query_draft = self.get_query(context)

            observation = ''
            if "sub_answer" in context:
                for index, (q, a) in enumerate(zip(context["sub_query"], context["sub_answer"])):
                    observation += f'Sub question {index}: {q}\n'
                    observation += f'Sub Answer: {a}\n\n'
            observation = observation if observation else "None"

            return [
                {'role': 'system', 'content': (
                    'You are a professional assistant skilled at decomposing complex questions into a sequence of logically dependent, '
                    'independently searchable sub-questions. Each sub-question must:\n'
                    '- Be self-contained and specific\n'
                    '- Be suitable for direct information retrieval from search engines or structured databases\n'
                    'You are now at an intermediate step in the decomposition chain. Your task is to revise the current sub-question draft using the context from the previous result, so that the revised question becomes self-contained and suitable for direct search.'
                )},
                {'role': 'user', 'content': (
                    f'Original question is: {query}\n'
                    f'Observation from previous query: {observation}\n'
                    f'Current sub-question draft: {query_draft}\n'
                    'Now revise the current sub-question draft using the context from the observation, so that it can be independently searched and yields a factual, retrievable answer. Return only the revised sub-question without any extra explanation or formatting.'
                )}
            ]
        else:
            query = self.get_query(context)

            return [
                {'role': 'system', 'content': 'You are a professional assistant skilled at rewriting overly detailed or redundant questions into a single, concise, and searchable query. Your goal is to keep only the essential part of the question that is needed to find the answer efficiently.'},
                {'role': 'assistant', 'content': 'Okay, I will return a concise rewritten query.'},
                {'role': 'user', 'content': f'Original question is {query}. Now rewrite the original question into a single, clear query that focuses only on the essential information needed to find the answer. Avoid unnecessary context, vague references, and maintain specificity. Output only the rewritten query without any extra explanation or formatting.'}
            ]

    def post_process(self, context: Dict[str, Any], response: str) -> None:
        self.set_query(context, response)
        if context["mode"] == "normal":
            context["results"] = [[]]
            context["sub_query"] = []
            context["sub_answer"] = []



class QueryDecompositionAgentParallel(ApiEnabledAgent):
    def build_message(self, context: Dict[str, Any]) -> List[Dict]:
        query = context.get("query")
        return [
            {'role': 'system',
             'content': 'You are a professional assistant skilled at decomposing complex multi-entity or multi-location questions into multiple independent and searchable sub-questions. Each sub-question should be specific, logically complete, and not repeat others.'},
            {'role': 'assistant', 'content': 'Okay, I will return the parallel sub-questions.'},
            {'role': 'user',
             'content': f"Original question is '{query}'. Break down this question into the minimum number of specific, logically complete, and independently searchable sub-questions needed to fully understand and answer the original question. Do not generate more than 4 sub-questions. Each sub-question should be on a separate line, avoid vague demonstratives or repetition, and ensure that each question is self-contained."}
        ]

    def post_process(self, context: Dict[str, Any], response: str) -> None:
        context["sub_query"] = [q.strip() for q in response.split('\n') if q.strip()][:4]
        context["steps"] = len(context.get("sub_query"))
        context["mode"] = "parallel"
        context["results"] = [[] for _ in range(context["steps"])]
        context["sub_answer"] = ["" for _ in range(context["steps"])]


class QueryDecompositionAgentSerial(ApiEnabledAgent):
    def build_message(self, context: Dict[str, Any]) -> List[Dict]:
        query = context.get("query")
        return [
            {'role': 'system', 'content': (
                'You are a professional assistant skilled at decomposing complex questions into a minimal sequence of logically dependent, independently searchable sub-questions. Each sub-question must:\n'
                '- Be self-contained and specific\n'
                '- Be suitable for direct information retrieval from search engines or structured databases\n'
                '- Be strictly necessary to answer the original question\n'
                'You must keep the number of sub-questions as low as possible, and never exceed 4 in total. Avoid redundancy and do not include trivial or overly granular sub-questions.'
            )},
            {'role': 'assistant',
            'content': 'Understood. I will return only factual, retrievable sub-questions, one per line.'},
            {'role': 'user',
            'content': (
                f'Original question is: {query}\n'
                'Now decompose the original question into a logically ordered list of sub-questions. '
                'Do not number the sub-questions, write one sub-question per line.'
            )}
        ]

    def post_process(self, context: Dict[str, Any], response: str) -> None:
        context["sub_query"] = [q.strip() for q in response.split('\n') if q.strip()][:4]
        context["steps"] = len(context.get("sub_query"))
        context["mode"] = "serial"
        context["results"] = [[] for _ in range(context["steps"])]
        context["sub_answer"] = ["" for _ in range(context["steps"])]


class DocumentSelectionAgent(ApiEnabledAgent):
    def get_query(self, context: Dict[str, Any]) -> str:
        if context["mode"] in ["serial", "parallel"]:
            return context["sub_query"][context["current_step"]]
        else:
            return context["query"]

    def get_result(self, context: Dict[str, Any]) -> List[str]:
        if context["mode"] in ["serial", "parallel"]:
            return context["results"][context["current_step"]] if "results" in context else []
        else:
            return context["results"][0] if "results" in context else []

    def set_result(self, context: Dict[str, Any], result: List[str]) -> None:
        if context["mode"] in ["serial", "parallel"]:
            context["results"][context["current_step"]] = result
        else:
            context["results"][0] = result

    def build_message(self, context: Dict[str, Any]) -> List[Dict]:
        query = self.get_query(context)
        result = self.get_result(context)

        doc_content = ''.join([f"Document {doc_id}: {doc}\n\n" for doc_id, doc in enumerate(result)])

        return [
            {'role': 'system', 'content': f'You are a helpful, respectful and honest assistant. Your task is to output the ID of the candidate Documents (0, 1, 2,..., {len(result)-1}) which are helpful in answering the Question.'},
            {'role': 'assistant', 'content': 'Okay, I will provide the ID of candidate Documents which are helpful in answering the Question.'},
            {'role': 'user', 'content': f'Question is: {query}\n\n{doc_content}'},
            {'role': 'assistant', 'content': "OK, I received the Question and the candidate Documents."},
            {'role': 'user', 'content': "Now, output the ID of the candidate Documents (0,1,2,...,{len(result)-1}) which are helpful in answering the Question: {question}, for example, in the following format: Document0,Document4,Document6,Document7."}
        ]

    def post_process(self, context: Dict[str, Any], response: str) -> None:
        result = self.get_result(context)

        # doc_ids = [int(p.replace("Document", "")) for p in response.split(",")]
        try:
            doc_ids = [int(p.replace("Document", "")) for p in response.split(",")]
        except ValueError as e:
            print(f"Error converting document IDs: {e}")
            doc_ids = list(range(len(result)))

        self.set_result(context, [result[i] for i in doc_ids if i < len(result)])

    def run(self, context: Dict[str, Any]) -> None:
        if not context.get('results', []):
            return

        super().run(context)


class AnswerGenerationAgent(ApiEnabledAgent):
    def get_query(self, context: Dict[str, Any]) -> str:
        if context["mode"] in ["serial", "parallel"]:
            query = context["sub_query"][context["current_step"]]
        else:
            query = context["query"]
        return query
    
    def get_result(self, context: Dict[str, Any]) -> List[str]:
        if context["mode"] in ["serial", "parallel"]:
            result = context["results"][context["current_step"]] if "results" in context else []
        else:
            result = context["results"][0] if "results" in context else []
        return result

    def set_answer(self, context: Dict[str, Any], response: str) -> None:
        if context["mode"] in ["serial", "parallel"]:
            context["sub_answer"][context["current_step"]] = response
        else:
            context['answer'] = response


    def build_message(self, context: Dict[str, Any], signal: int=-2) -> List[Dict]:
        query = self.get_query(context)
        result = self.get_result(context)
        doc_content = ''.join([f"Document {doc_id}: {doc}\n\n" for doc_id, doc in enumerate(result)])

        prefix_prompt = ''
        if context["mode"] == "serial":
            prefix_prompt = f'Original Question is: {context["query"]}\n'
            if "sub_answer" in context:
                for index, (q, a) in enumerate(zip(context["sub_query"], context["sub_answer"])):
                    prefix_prompt += f'Sub Question {index}: {q}\n'
                    prefix_prompt += f'Sub Answer: {a}\n\n'

        example_str = EXAMPLE_PROMPT if context["mode"] == "normal" else ""

        if doc_content:
            return [
                {'role': 'system', 'content': fr'''You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question based on the given documents. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible.{example_str}'''},
                {'role': 'assistant', 'content': 'Okay, I will provide the answer to the question based on the corresponding documents. Please provide the question and the corresponding documents.'},
                {'role': 'user', 'content': f'{prefix_prompt}Question is: {query}\n\n{doc_content}Now, answer the Question: {query}, based on the above Documents'},
                {'role': 'assistant', 'content': "OK, I received the Question and the corresponding Documents."},
                {'role': 'user', 'content': "Given the Question and the corresponding Documents, predict the answer to the Question as briefly and accurately as possible based on the Documents. Only give the brief and accurate answer with the form of **answer**."}
            ]
            
        else:
            return [
                {'role': 'system', 'content': f'''You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible.{example_str}'''},
                {'role': 'assistant', 'content': 'Okay, I will provide the answer to the question. Please provide the question.'},
                {'role': 'user', 'content': f'{prefix_prompt}Question is: {query}\n\nNow, answer the Question: {query}.'},
                {'role': 'assistant', 'content': "OK, I received the Question."},
                {'role': 'user', 'content': "Given the Question, predict the answer to the Question as briefly and accurately as possible. Only give the brief and accurate answer with the form of **answer**."}
            ]

    def post_process(self, context: Dict[str, Any], response: str) -> None:
        # response = re.sub(r"\*\*answer\*\*: ", "", response, count=1, flags=re.IGNORECASE)
        # response = response.replace("*", "")
        # self.set_answer(context, response)
        try:
            # 检查 response 是否为字符串或字节对象
            if isinstance(response, (str, bytes)):
                # 如果是字符串或字节对象，执行正则表达式替换和字符串替换
                response = re.sub(r"\*\*answer\*\*: ", "", response, count=1, flags=re.IGNORECASE)
                response = response.replace("*", "")
            else:
                # 如果不是字符串或字节对象，直接赋值为错误信息
                response = "Error in response AG"
                print(response)
            # 将结果存储在 context["answer"] 中
            self.set_answer(context, response)
        except Exception as e:
            # 捕获任何异常并处理
            print(f"Caught an exception: {e} AG")
            response = "Error in response AG"
            self.set_answer(context, response)


class AnswerSummarizationAgent(ApiEnabledAgent):
    def build_message(self, context: Dict[str, Any]) -> List[Dict]:
        query = context["query"]

        observation = ''

        if "sub_query" not in context:
            context["sub_query"] = []
        if "sub_answer" not in context:
            context["sub_answer"] = []

        for index, (q, a) in enumerate(zip(context["sub_query"], context["sub_answer"])):
            observation += f'Sub-question {index}: {q}\n'
            observation += f'Answer: {a}\n\n'

        return [
            {'role': 'system', 'content': f'''You are a helpful, respectful and honest assistant. Your task is to predict the final answer to the original question based on the answers to its decomposed sub-questions. If you are not sure about the final answer, do not make up information. Give the most accurate and concise answer possible based on the sub-question answers.{EXAMPLE_PROMPT}'''},
            {'role': 'assistant', 'content': 'Okay, I will provide the final answer to the original question based on the sub-questions and their corresponding answers. Please provide the original question, the sub-questions, and their answers.'},
            {'role': 'user', 'content': f'Original Question: {query}\n\n{observation}Now, based on the above sub-questions and their answers, answer the Original Question: {query}'},
            {'role': 'assistant', 'content': 'OK, I received the Original Question, its Sub-questions, and their Answers.'},
            {'role': 'user', 'content': 'Given the Original Question, the Sub-questions and their Answers, predict the final answer to the Original Question as briefly and accurately as possible. Only give the brief and accurate answer in the form of **answer**.'}]

    def post_process(self, context: Dict[str, Any], response: str) -> None:
        # response = re.sub(r"\*\*answer\*\*: ", "", response, count=1, flags=re.IGNORECASE)
        # response = response.replace("*", "")
        # context["answer"] = response
        try:
            # 检查 response 是否为字符串或字节对象
            if isinstance(response, (str, bytes)):
                # 如果是字符串或字节对象，执行正则表达式替换和字符串替换
                response = re.sub(r"\*\*answer\*\*: ", "", response, count=1, flags=re.IGNORECASE)
                response = response.replace("*", "")
            else:
                # 如果不是字符串或字节对象，直接赋值为错误信息
                response = "Error in response AS"
                print(response)
            # 将结果存储在 context["answer"] 中
            context["answer"] = response
        except Exception as e:
            # 捕获任何异常并处理
            print(f"Caught an exception: {e} AG")
            context["answer"] = "Error in response"


class RetrievalAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.api_url = config.get('api_url', '')
        self.num_results = config.get('num_results', '')

    def get_query(self, context: Dict[str, Any]) -> str:
        if context["mode"] in ["serial", "parallel"]:
            query = context["sub_query"][context["current_step"]]
        else:
            query = context["query"]
        return query

    def set_result(self, context: Dict[str, Any], result: List[str]) -> None:
        if context["mode"] in ["serial", "parallel"]:
            context["results"][context["current_step"]] = result
        else:
            context["results"][0] = result


    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        questions = [self.get_query(context)]

        headers = {'Content-Type': 'application/json'}
        payload = {
            'questions': questions,
            'N': self.num_results
        }

        # logger.info(f"\t\t-- QUESTION --\t{questions[0]}")

        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise an error for bad responses
            result = response.json()[0]["top_k_docs"]

            self.set_result(context, result)

        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred: {e}")
            self.set_result(context, [])

        finally:
            super().log(context)

        return context

class AgentPool:
    _instance = None
    _lock = Lock()
    _initialized = False

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(AgentPool, cls).__new__(cls)
                    cls._instance.agents = {}
                    cls._instance._initialize_agents()
        return cls._instance

    def _initialize_agents(self):
        if self._initialized:
            return

        self.register([
            QueryRewriteAgent(AGENT_CONFIG['QueryRewriteAgent']),
            QueryDecompositionAgentParallel(AGENT_CONFIG['QueryDecompositionAgentParallel']),
            QueryDecompositionAgentSerial(AGENT_CONFIG['QueryDecompositionAgentSerial']),
            RetrievalAgent(AGENT_CONFIG['RetrievalAgent']),
            DocumentSelectionAgent(AGENT_CONFIG['DocumentSelectionAgent']),
            AnswerGenerationAgent(AGENT_CONFIG['AnswerGenerationAgent']),
            AnswerSummarizationAgent(AGENT_CONFIG['AnswerSummarizationAgent']),
        ])
        self._initialized = True

    def register(self, agents: List[BaseAgent]) -> None:
        for agent in agents:
            self.agents[agent.name] = agent

    def get(self, name: str) -> BaseAgent:
        return self.agents[name]


class AgentWorkflow:
    def __init__(self, planning_agent) -> None:
        self.agent_pool = AgentPool()
        self.planning_agent = planning_agent

    def run(self, query: str, is_sub: bool = False) -> Dict[str, Any]:

        context = {
            "original_query": query,
            "query": query,
            "mode": "normal",
            "is_sub": is_sub,
            "current_step": -1,
            "answer": ""
        }

        # print('************ query: {}'.format(query))

        while True:
            if context["mode"] == "normal":
                # planning agent 
                workflow = self.run_planning_agent(query, is_sub)
                if 'QueryDecompositionAgentParallel' in workflow:
                    workflow = ['QueryDecompositionAgentParallel']
                elif 'QueryDecompositionAgentSerial' in workflow:
                    workflow = ['QueryDecompositionAgentSerial']
                # print('************ workflow:', workflow)
            else:
                # planning agent 
                sub_q = context["sub_query"][context["current_step"]]
                workflow = self.run_planning_agent(sub_q, is_sub=True)
                # print('************ serial workflow:', workflow)


            for agent_name in workflow:
                agent = self.agent_pool.get(agent_name)
                # logger.info(f"\n\t==> Running Agent: {agent_name}")
                agent.run(context)

            if context["answer"]:
                # logger.info("\n\t\t--> Program terminated.")
                break

            context["current_step"] += 1
            
            if context["current_step"] != context["steps"]:
                # logger.info(f"\n\t\t--> Serial steps remaining: {context['steps'] - context['current_step']}. Injecting new workflow.")
                if context["current_step"] >= 1 and context["mode"] == "serial":
                    agent = self.agent_pool.get("QueryRewriteAgent")
                    # logger.info("\n\t==> Running Agent: QueryRewriteAgent")
                    agent.run(context)
                    
            else:
                agent = self.agent_pool.get("AnswerSummarizationAgent")
                # logger.info("\n\t==> Running Agent: AnswerSummarizationAgent")
                agent.run(context)
                # logger.info("\n\t\t--> Serial steps execution finished.")
                break
        
        # print('************ answer: {}'.format(context['answer']))
        # print('\n')

        return context

    def run_planning_agent(self, query: str, is_sub: bool = False) -> List[str]:
        messages = self.planning_agent.create_planning_messages(query, is_sub)

        text = self.planning_agent.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.planning_agent.tokenizer([text], return_tensors="pt").to(self.planning_agent.model.device)

        generated_ids = self.planning_agent.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.planning_agent.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        workflow = self.parse_workflow(response)

        return workflow
    
    def parse_workflow(self, workflow_string):
        # Remove the 'Workflow: ' prefix and strip any leading/trailing whitespace
        modules_string = workflow_string.replace('Workflow: ', '').strip()
        # Split the modules by ', ' to get a list of individual modules
        modules = modules_string.split(', ')

        mapping_dict = {
            'QR': 'QueryRewriteAgent',
            'QDP': 'QueryDecompositionAgentParallel',
            'QDS': 'QueryDecompositionAgentSerial',
            'RA': 'RetrievalAgent',
            'DS': 'DocumentSelectionAgent',
            'AG': 'AnswerGenerationAgent',
            'AS': 'AnswerSummarizationAgent',
        }

        workflow_list = [mapping_dict[module] for module in modules if module in mapping_dict]
        
        return workflow_list


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


logger = setup_logger_no_print(log_file='log/log_whole.log')

if __name__ == "__main__":

    model_name = "/root/workspace/env_run/verl/models_fund/Qwen/Qwen2.5-7B-Instruct"
    # 获取可用的GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    planning_agent = PlanningAgent(model, tokenizer)

    # load dataset

    # test questions
    questions = [
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
        "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?",
        "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
        "The director of the romantic comedy \"Big Stone Gap\" is based in what New York city?",
        "2014 S/S is the debut album of a South Korean boy group that was formed by who?",
        "Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?",
        "The arena where the Lewiston Maineiacs played their home games can seat how many people?",
        "Who is older, Annie Morton or Terry Richardson?",
        "Are Local H and For Against both from the United States?",
        "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?",
        "What screenwriter with credits for \"Evolution\" co-wrote a film starring Nicolas Cage and T\u00e9a Leoni?",
        "What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger as a former New York Police detective?",

        # "Were Scott Derrickson and Ed Wood of the same nationality?",
        # "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
        # "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?",
        # "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
        # "The director of the romantic comedy \"Big Stone Gap\" is based in what New York city?",
        # "2014 S/S is the debut album of a South Korean boy group that was formed by who?",
        # "Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?",
        # "The arena where the Lewiston Maineiacs played their home games can seat how many people?",
        # "Who is older, Annie Morton or Terry Richardson?",
        # "Are Local H and For Against both from the United States?",
        # "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?",
        # "What screenwriter with credits for \"Evolution\" co-wrote a film starring Nicolas Cage and T\u00e9a Leoni?",
        # "What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger as a former New York Police detective?"
    ]
    # questions = [
    #     # 简单问题（单跳）
    #     "What is the capital of France?",
    #     "Who wrote 'Romeo and Juliet'?",
        
    #     # 简单问题（多跳）
    #     "What is the capital of the country where the Great Wall is located?",
    #     "Who is the author of the book that inspired the movie 'Jurassic Park'?",
        
    #     # 复杂问题（多跳）
    #     "How did technological advancements during World War II lead to changes in international relations?",
    #     "What are the environmental and economic impacts of deforestation in the Amazon Rainforest?"
    # ]
    '''
    for query in questions:
        
        # # workflow logs
        # logger.info(f"====== Workflow ======")
        # for agent_name in workflow:
        #     logger.info(f"\t==> {agent_name}")

        # # query logs
        # logger.info(f"\n====== Query ======")
        # logger.info(f"\t==> {query}\n")

        logger.info(f"====== Starting Process... ======")
        wf = AgentWorkflow(planning_agent)

        final_context = wf.run(query)

        logger.info(f"\n====== Token Usage ======")
        tracker = TokenUsageTracker()
        logger.info(f'Token Usage: {tracker.get_usage()}')

    '''

    batch_size=30
    batch_runner = BatchAgentWorkflow(batch_size)
    final_answers = batch_runner.run_batch(questions)

    logger.info(f"\n====== Token Usage Summary ======")
    tracker = TokenUsageTracker()
    logger.info(f'Token Usage: {tracker.get_usage()}')

    for answer in final_answers:
        print(answer)
