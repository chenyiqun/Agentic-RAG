import time
from abc import ABC, abstractmethod
from openai import OpenAI
from typing import Dict, List, Any
from config import AGENT_CONFIG
import requests
import json
from tools import TokenUsageTracker, setup_logger, setup_logger_no_print
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed


class BaseAgent(ABC):
    def __init__(self, config: Dict):
        self.name = config.get('name', '')

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> None:
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
        for m in message:
            logger.info(f"\t\t{m['role'].upper()} - {m['content']}")

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

        logger.info(f"\t\tRESPONSE - {response.choices[0].message.content}")
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

    def run(self, context: Dict[str, Any]) -> None:
        """
        封装调用流程：构造 prompt -> 获取响应 -> 后处理
        """
        message = self.build_message(context)
        raw_response = self.get_response(message)
        self.post_process(context, raw_response)


class QueryRewriteAgent(ApiEnabledAgent):
    def build_message(self, context: Dict[str, Any]) -> List[Dict]:
        query = context.get("query")
        return [
            {'role': 'system',
             'content': 'You are a professional assistant skilled at rewriting overly detailed or redundant questions into a single, concise, and searchable query. Your goal is to keep only the essential part of the question that is needed to find the answer efficiently.'},
            {'role': 'assistant', 'content': 'Okay, I will return a concise rewritten query.'},
            {'role': 'user',
             'content': f'Original question is {query}. Now rewrite the original question into a single, clear query that focuses only on the essential information needed to find the answer. Avoid unnecessary context, vague references, and maintain specificity. Output only the rewritten query without any extra explanation or formatting.'}
        ]

    def post_process(self, context: Dict[str, Any], response: str) -> None:
        context["rewritten_query"] = [response]


class QueryDecompositionAgentParallel(ApiEnabledAgent):
    def build_message(self, context: Dict[str, Any]) -> List[Dict]:
        query = context.get("query")
        return [
            {'role': 'system',
             'content': 'You are a professional assistant skilled at decomposing complex multi-entity or multi-location questions into multiple independent and searchable sub-questions. Each sub-question should be specific, logically complete, and not repeat others.'},
            {'role': 'assistant', 'content': 'Okay, I will return the parallel sub-questions.'},
            {'role': 'user',
             'content': f"Original question is '{query}'. Now decompose the question into multiple specific sub-questions that can be independently searched. Each sub-question should be on a separate line, avoid vague demonstratives or repetition, and ensure that each question is self-contained."}
        ]

    def post_process(self, context: Dict[str, Any], response: str) -> None:
        context["rewritten_query"] = [q.strip() for q in response.split('\n') if q.strip()]


class QueryDecompositionAgentSerial(ApiEnabledAgent):
    def build_message(self, context: Dict[str, Any]) -> List[Dict]:
        if context.get('current_step') == 0:
            query = context.get("query")
            return [
                {'role': 'system', 'content': (
                    'You are a professional assistant skilled at decomposing complex questions into a sequence of logically dependent, '
                    'independently searchable sub-questions. Each sub-question must:\n'
                    '- Be self-contained and specific\n'
                    '- Be suitable for direct information retrieval from search engines or structured databases\n'
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
        else:
            query = context.get("query")
            current_step = context.get("current_step")
            query_draft = context.get("rewritten_query")[current_step]

            observation = ''
            for index, c in enumerate(context.get("sub_answers", [])):
                observation += f'Sub-question {index}: {c["question"]}\n'
                observation += f'Answer: {c["answer"]}\n\n'

            return [
                {'role': 'system', 'content': (
                    'You are a professional assistant skilled at decomposing complex questions into a sequence of logically dependent, '
                    'independently searchable sub-questions. Each sub-question must:\n'
                    '- Be self-contained and specific\n'
                    '- Be suitable for direct information retrieval from search engines or structured databases\n'
                    'You are now at an intermediate step in the decomposition chain. Your task is to revise the current sub-question draft using the context from the previous result, so that the revised question becomes self-contained and suitable for direct search.'
                )},
                {'role': 'user',
                'content': (
                    f'Original question is: {query}\n'
                    f'Observation from previous query: {observation}\n'
                    f'Current sub-question draft: {query_draft}\n'
                    'Now revise the current sub-question draft using the context from the observation, so that it can be independently searched and yields a factual, retrievable answer. Return only the revised sub-question without any extra explanation or formatting.'
                )}
            ]

    def post_process(self, context: Dict[str, Any], response: str) -> None:
        if context.get('current_step') == 0:
            context["rewritten_query"] = [q.strip() for q in response.split('\n') if q.strip()]
            context["steps"] = len(context.get("rewritten_query"))
        else:
            current_step = context.get("current_step")
            context["rewritten_query"][current_step] = response


class DocumentSelectionAgent(ApiEnabledAgent):
    def build_message(self, query: str, top_k_docs: List) -> List[Dict]:
        doc_content = ''.join([f"Document {doc_id}: {doc}\n\n" for doc_id, doc in enumerate(top_k_docs)])

        return [
            {'role': 'system', 'content': f'You are a helpful, respectful and honest assistant. Your task is to output the ID of the candidate Documents (0, 1, 2,..., {len(top_k_docs)-1}) which are helpful in answering the Question.'},
            {'role': 'assistant', 'content': 'Okay, I will provide the ID of candidate Documents which are helpful in answering the Question.'},
            {'role': 'user', 'content': f'Question is: {query}\n\n{doc_content}'},
            {'role': 'assistant', 'content': "OK, I received the Question and the candidate Documents."},
            {'role': 'user', 'content': "Now, output the ID of the candidate Documents (0,1,2,...,{len(top_k_docs)-1}) which are helpful in answering the Question: {question}, for example, in the following format: Document0,Document4,Document6,Document7."}
        ]

    def post_process(self, response: str) -> List:
        return [int(p.replace("Document", "")) for p in response.split(",")]

    def _run(self, query: str, results: List) -> List:
        message = self.build_message(query, results)
        raw_response = self.get_response(message)
        doc_ids = self.post_process(raw_response)
        return [results[i] for i in doc_ids]

    def run(self, context: Dict[str, Any], merge: bool = True) -> None:
        # 没有retrieval，跳过
        if not context.get('results', []):
            return

        # 串行，只选择当前step的sub-query
        if context.get("serial", False):
            current_step = context.get("current_step")
            questions = context.get("rewritten_query")[current_step]

            results = context.get('results')[current_step]['top_k_docs']

            context['results'][current_step] = {
                'question': questions,
                'top_k_docs': self._run(questions, results)
            }

        elif merge:
            query = context.get('query')
            results = context.get('results', [])

            all_results = [doc for item in results for doc in item['top_k_docs']]
            filtered_results = self._run(query, all_results)

            result_set = set(filtered_results)
            new_results = []
            for item in results:
                new_results.append({
                    'question': item['question'],
                    'top_k_docs': [r for r in item['top_k_docs'] if r in result_set]
                })
            context['results'] = new_results
        else:
            query = context.get('query')
            results = context.get('results', [])
            context['results'] = [
                {
                    'question': item['question'],
                    'top_k_docs': self._run(item['question'], item['top_k_docs'])
                }
                for item in results
            ]

        logger.info("\t\t=== Selected Documents ===")
        for query in context['results']:
            logger.info(f"\t\t\tQuery: {query['question']}")
            for doc_id, doc in enumerate(query['top_k_docs']):
                doc = doc.replace('\n', ' ')[:50]
                logger.info(f"\t\t\t\tDocument {doc_id}: {doc}...")


class AnswerGenerationAgent(ApiEnabledAgent):
    def build_message(self, query: str, results: List, summary: bool = False, example: bool = False) -> List[Dict]:
        example_str = '''
- Example:
Question: When did the simpsons first air on television?
Answer: December 17, 1989

Question: When did the lightning thief book come out?
Answer: 2005

Question: Who said i'm late i'm late for a very important date?
Answer: The White Rabbit

Question: Where does the short happy life of francis macomber take place?
Answer: Africa

Question: What was the fourth expansion pack for sims 2?
Answer: Pets

Question: Voice of the snake in the jungle book?
Answer: The Jungle Book (2016 film)

Question: How many seasons are there of star wars the clone wars?
Answer: 6

Question: Which us president appears as a character in the play annie?
Answer: Franklin D. Roosevelt

Question: Are Calochone and Adlumia both plants?
Answer: yes

Question: Yukio Mishima and Roberto Bola\u00f1o, are Chilean?
Answer: no
''' if example else ''

        if summary:
            observation = ''
            for index, c in enumerate(results):
                observation += f'Sub-question {index}: {c["question"]}\n'
                observation += f'Answer: {c["answer"]}\n\n'

            return [
                {'role': 'system', 'content': f'''You are a helpful, respectful and honest assistant. Your task is to predict the final answer to the original question based on the answers to its decomposed sub-questions. If you are not sure about the final answer, do not make up information. Give the most accurate and concise answer possible based on the sub-question answers.{example_str}'''},
                {'role': 'assistant', 'content': 'Okay, I will provide the final answer to the original question based on the sub-questions and their corresponding answers. Please provide the original question, the sub-questions, and their answers.'},
                {'role': 'user', 'content': f'Original Question: {query}\n\n{observation}... \n\nNow, based on the above sub-questions and their answers, answer the Original Question: {query}'},
                {'role': 'assistant', 'content': 'OK, I received the Original Question, its Sub-questions, and their Answers.'},
                {'role': 'user', 'content': 'Given the Original Question, the Sub-questions and their Answers, predict the final answer to the Original Question as briefly and accurately as possible. Only give the brief and accurate answer in the form of **answer**.'}]

        if results:
            doc_content = ''.join([f"Document {doc_id}: {doc}\n\n" for doc_id, doc in enumerate(results)])

            return [
                {'role': 'system', 'content': fr'''You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question based on the given documents. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible.{example_str}'''},
                {'role': 'assistant', 'content': 'Okay, I will provide the answer to the question based on the corresponding documents. Please provide the question and the corresponding documents.'},
                {'role': 'user', 'content': f'Question is: {query}\n\n{doc_content}Now, answer the Question: {query}, based on the above Documents'},
                {'role': 'assistant', 'content': "OK, I received the Question and the corresponding Documents."},
                {'role': 'user', 'content': "Given the Question and the corresponding Documents, predict the answer to the Question as briefly and accurately as possible based on the Documents. Only give the brief and accurate answer with the form of **answer**."}
            ]
        else:
            return [
                {'role': 'system', 'content': f'''You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible.{example_str}'''},
                {'role': 'assistant', 'content': 'Okay, I will provide the answer to the question. Please provide the question.'},
                {'role': 'user', 'content': f'Question is: {query}\n\nNow, answer the Question: {query}.'},
                {'role': 'assistant', 'content': "OK, I received the Question."},
                {'role': 'user', 'content': "Given the Question, predict the answer to the Question as briefly and accurately as possible. Only give the brief and accurate answer with the form of **answer**."}
            ]

    def post_process(self, response: str) -> str:
        # return re.sub(r"\*\*answer\*\*: ", "", response, count=1, flags=re.IGNORECASE)
        modified_response = re.sub(r"\*\*answer\*\*: ", "", response, count=1, flags=re.IGNORECASE)
        modified_response = modified_response.replace("*", "")
        return modified_response

    def _run(self, query: str, results: List, summary: bool = False, example: bool = False) -> str:
        message = self.build_message(query, results, summary, example)
        raw_response = self.get_response(message)
        return self.post_process(raw_response)

    def run(self, context: Dict[str, Any], merge: bool = False) -> None:
        # 串行，只回答当前step的sub-query
        if context.get("serial", False):
            current_step = context.get("current_step")
            questions = context.get("rewritten_query")[current_step]

            results = context.get('results')[current_step]['top_k_docs'] if 'results' in context else []

            context.setdefault('sub_answers', []).append({
                'question': questions,
                'answer': self._run(questions, results)
            })
            if context["current_step"] + 1 == context["steps"]:
                context['answer'] = self._run(context.get('query'), context['sub_answers'], summary=True, example=True)

        # 合并生成模式/没有参考文档/查询只有一个的时候，只调用一次generator
        elif merge or 'results' not in context or len(context['results']) == 1:
            query = context.get('query')
            results = context.get('results', [])
            all_results = [doc for item in results for doc in item['top_k_docs']] if results else None
            context['answer'] = self._run(query, all_results, example=True)
        else:
            query = context.get('query')
            results = context.get('results', [])
            context['sub_answers'] = [
                {
                    'question': item['question'],
                    'answer': self._run(item['question'], item['top_k_docs'])
                }
                for item in results
            ]
            context['answer'] = self._run(query, context['sub_answers'], summary=True, example=True)


class RetrievalAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        self.api_url = config.get('api_url', '')
        self.num_results = config.get('num_results', '')

    def run(self, context: Dict[str, Any]) -> None:
        if context.get("serial", False):
            current_step = context.get("current_step")
            questions = [context.get("rewritten_query")[current_step]]
        else:
            questions = context.get("rewritten_query", [context.get("query")])

        headers = {'Content-Type': 'application/json'}
        payload = {
            'questions': questions,
            'N': self.num_results
        }

        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise an error for bad responses

            if context.get("serial", False):
                context.setdefault("results", []).extend(response.json())
            else:
                context["results"] = response.json()

            logger.info("\t\t=== Documents ===")
            for query in response.json():
                logger.info(f"\t\t\tQuery: {query['question']}")
                for doc_id, doc in enumerate(query['top_k_docs']):
                    doc = doc.replace('\n', ' ')[:50]
                    logger.info(f"\t\t\t\tDocument {doc_id}: {doc}...")

        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred: {e}")


class IterativeWorkflowAgent(BaseAgent):
    def run(self, context: Dict[str, Any]) -> None:
        if context.get("serial", False):
            context["current_step"] += 1
            if context["current_step"] != context["steps"]:
                logger.info(f"\n\t\t--> Serial steps remaining: {context['steps'] - context['current_step']}. Injecting new workflow.")
                context['workflow_queue'].extend(context['initial_workflow'])
            else:
                context["answer"] = context["sub_answers"][-1]["answer"]
                logger.info("\n\t\t--> Serial steps execution finished.")
        else:
            logger.info("\n\t\t--> No serial steps to execute. Program terminated.")


class AgentPool:
    def __init__(self) -> None:
        self.agents = {}

    def register(self, agents: List[BaseAgent]) -> None:
        for agent in agents:
            self.agents[agent.name] = agent

    def get(self, name: str) -> BaseAgent:
        return self.agents[name]


class AgentWorkflow:
    def __init__(self, agent_pool: AgentPool) -> None:
        self.agent_pool = agent_pool

    def run(self, query: str, initial_workflow: List[str]) -> Dict[str, Any]:
        initial_workflow = initial_workflow + ['IterativeWorkflowAgent']

        context = {
            "query": query,
            "initial_workflow": initial_workflow,
            "workflow_queue": deque(initial_workflow),
            "serial": True if "QueryDecompositionAgentSerial" in initial_workflow else False,
            "current_step": 0
        }

        while context.get("workflow_queue"):
            agent_name = context.get("workflow_queue").popleft()
            agent = self.agent_pool.get(agent_name)
            logger.info(f"\n\t==> Running Agent: {agent_name}")
            agent.run(context)

        return context


class BatchAgentWorkflow:
    def __init__(self, agent_pool: AgentPool, max_workers: int = 2) -> None:
        self.agent_pool = agent_pool
        self.max_workers = max_workers

    def run_batch(self, queries_with_workflows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_query = {
                executor.submit(self.run_single, pair['query'], pair['workflow']): pair
                for pair in queries_with_workflows
            }

            for future in as_completed(future_to_query):
                result = future.result()
                results.append(result)

        return results

    def run_single(self, query: str, workflow: List[str]) -> Dict[str, Any]:
        try:
            # workflow logs
            logger.info(f"====== Workflow ======")
            for agent_name in workflow:
                logger.info(f"\t==> {agent_name}")

            # query logs
            logger.info(f"\n====== Query ======")
            logger.info(f"\t==> {query}\n")

            logger.info(f"====== Starting Process... ======")
            wf = AgentWorkflow(self.agent_pool)
            context = wf.run(query, workflow)

            logger.info(f"Completed query: {query}")
            return context

        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            return {"query": query, "error": str(e)}


logger = setup_logger_no_print(log_file='log/log_6.log')

if __name__ == "__main__":
    # 1. RetrievalAgent判断是否retrieval
    # 2. QueryDecompositionAgentSerial最后加上summary
    # 3. example只在final answer的时候加上

    pool = AgentPool()
    pool.register([
        QueryRewriteAgent(AGENT_CONFIG['QueryRewriteAgent']),
        QueryDecompositionAgentParallel(AGENT_CONFIG['QueryDecompositionAgentParallel']),
        QueryDecompositionAgentSerial(AGENT_CONFIG['QueryDecompositionAgentSerial']),
        RetrievalAgent(AGENT_CONFIG['RetrievalAgent']),
        DocumentSelectionAgent(AGENT_CONFIG['DocumentSelectionAgent']),
        AnswerGenerationAgent(AGENT_CONFIG['AnswerGenerationAgent']),
        IterativeWorkflowAgent(AGENT_CONFIG['IterativeWorkflowAgent'])
    ])

    # workflow = [
    #     "QueryDecompositionAgentSerial",
    #     "RetrievalAgent",
    #     "AnswerGenerationAgent"
    # ]

    # query = "What nationality was James Henry Miller's wife?"

    # # workflow logs
    # logger.info(f"====== Workflow ======")
    # for agent_name in workflow:
    #     logger.info(f"\t==> {agent_name}")

    # # query logs
    # logger.info(f"\n====== Query ======")
    # logger.info(f"\t==> {query}\n")

    # logger.info(f"====== Starting Process... ======")
    # wf = AgentWorkflow(pool)
    # final_context = wf.run(query, workflow)

    # logger.info(f"\n====== Token Usage ======")
    # tracker = TokenUsageTracker()
    # logger.info(f'Token Usage: {tracker.get_usage()}')


    query_workflow_pairs = [
        # {
        #     "query": "What nationality was James Henry Miller's wife?",
        #     "workflow": [
        #         "QueryRewriteAgent",
        #         "RetrievalAgent",
        #         "AnswerGenerationAgent"
        #     ]
        # },
        {
            "query": "What nationality was James Henry Miller's wife?",
            "workflow": [
                "RetrievalAgent",
                "AnswerGenerationAgent"
            ]
        },
        {
            "query": "What nationality was James Henry Miller's wife?",
            "workflow": [
                "RetrievalAgent",
                "AnswerGenerationAgent"
            ]
        },
        
    ]

    batch_runner = BatchAgentWorkflow(pool, max_workers=8)
    final_contexts = batch_runner.run_batch(query_workflow_pairs)

    logger.info(f"\n====== Token Usage Summary ======")
    tracker = TokenUsageTracker()
    logger.info(f'Token Usage: {tracker.get_usage()}')

    for context in final_contexts:
        print(f"\nQuery: {context.get('query', 'N/A')}")
        if 'error' in context:
            print(f"Error: {context['error']}")
        else:
            print(f"Final Context: {context}")
