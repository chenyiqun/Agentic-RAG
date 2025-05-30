import time
from abc import ABC, abstractmethod
from openai import OpenAI
from typing import Dict, List, Optional, Any
from config import AGENT_CONFIG
import requests
import json
from tools import TokenUsageTracker, setup_logger
import re


class BaseAgent(ABC):
    def __init__(self, config: Dict):
        self.name = config.get('name', '')

    @abstractmethod
    def run(self, context: Dict[str, Any]):
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
        tracker.record(self.name, usage)
        tracker.record("all", usage)

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
        query = context.get("query")
        return [
            {'role': 'system', 'content': (
                'You are a professional assistant skilled at decomposing complex questions into a sequence of logically dependent, '
                'independently searchable sub-questions. Each sub-question must:\n'
                '- Be self-contained and specific\n'
                '- Be suitable for direct information retrieval from search engines or structured databases\n'
                '- Avoid logical inference, reasoning, or computation (e.g., "how to calculate age" or "what is the current date")\n'
                '- Depend on the previous answer only for context, not for logic or calculation\n'
                'Only include sub-questions that yield factual, retrievable answers. Do NOT include questions that require reasoning, math, date comparison, or context-specific interpretation.'
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
        context["rewritten_query"] = [q.strip() for q in response.split('\n') if q.strip()]


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
        query = context.get('query')
        results = context.get('results', [])
        if merge:
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
    def build_message(self, query: str, results: Optional[List]) -> List[Dict]:
        if results:
            doc_content = ''.join([f"Document {doc_id}: {doc}\n\n" for doc_id, doc in enumerate(results)])

            return [
                {'role': 'system', 'content': fr"You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question based on the given documents. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible."},
                {'role': 'assistant', 'content': 'Okay, I will provide the answer to the question based on the corresponding documents. Please provide the question and the corresponding documents.'},
                {'role': 'user', 'content': f'Question is: {query}\n\n{doc_content}Now, answer the Question: {query}, based on the above Documents'},
                {'role': 'assistant', 'content': "OK, I received the Question and the corresponding Documents."},
                {'role': 'user', 'content': "Given the Question and the corresponding Documents, predict the answer to the Question as briefly and accurately as possible based on the Documents. Only give the brief and accurate answer with the form of **answer**."}
            ]
        else:
            return [
                {'role': 'system', 'content': "You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible."},
                {'role': 'assistant', 'content': 'Okay, I will provide the answer to the question. Please provide the question.'},
                {'role': 'user', 'content': f'Question is: {query}\n\nNow, answer the Question: {query}.'},
                {'role': 'assistant', 'content': "OK, I received the Question."},
                {'role': 'user', 'content': "Given the Question, predict the answer to the Question as briefly and accurately as possible. Only give the brief and accurate answer with the form of **answer**."}
            ]

    def post_process(self, response: str) -> str:
        return re.sub(r"\*\*answer\*\*: ", "", response, count=1, flags=re.IGNORECASE)

    def _run(self, query: str, results: Optional[List]) -> str:
        message = self.build_message(query, results)
        raw_response = self.get_response(message)
        return self.post_process(raw_response)

    def run(self, context: Dict[str, Any], merge: bool = True) -> None:
        query = context.get('query')
        results = context.get('results', [])
        # 合并生成模式/没有参考文档/查询只有一个的时候，只调用一次generator
        if merge or not results or len(results) == 1:
            all_results = [doc for item in results for doc in item['top_k_docs']] if results else None
            context['answer'] = self._run(query, all_results)
        else:
            context['sub_answers'] = [
                {
                    'question': item['question'],
                    'answer': self._run(item['question'], item['top_k_docs'])
                }
                for item in results
            ]


class RetrievalAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.api_url = config.get('api_url', '')
        self.num_results = config.get('num_results', '')

    def run(self, context: Dict[str, Any]) -> None:
        questions = context.get("rewritten_query", [context.get("query")])
        headers = {'Content-Type': 'application/json'}
        payload = {
            'questions': questions,
            'N': self.num_results
        }

        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise an error for bad responses
            context["results"] = response.json()

            logger.info("\t\t=== Documents ===")
            for query in response.json():
                logger.info(f"\t\t\tQuery: {query['question']}")
                for doc_id, doc in enumerate(query['top_k_docs']):
                    doc = doc.replace('\n', ' ')[:50]
                    logger.info(f"\t\t\t\tDocument {doc_id}: {doc}...")

        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred: {e}")


class AgentPool:
    def __init__(self) -> None:
        self.agents = {}

    def register(self, agents: List[BaseAgent]) -> None:
        for agent in agents:
            self.agents[agent.name] = agent

    def get(self, name: str) -> BaseAgent:
        return self.agents[name]


class AgentWorkflow:
    def __init__(self, agent_pool: AgentPool, workflow_steps: List[str]) -> None:
        self.agent_pool = agent_pool
        self.workflow_steps = workflow_steps

    def run(self, query: str) -> Dict[str, Any]:
        context = {
            "query": query
        }

        for agent_name in self.workflow_steps:
            agent = self.agent_pool.get(agent_name)
            logger.info(f"\n\t==> Running Agent: {agent_name}")
            agent.run(context)
        return context


logger = setup_logger(log_file='log_2.log')

if __name__ == "__main__":
    pool = AgentPool()
    pool.register([
        QueryRewriteAgent(AGENT_CONFIG['QueryRewriteAgent']),
        QueryDecompositionAgentParallel(AGENT_CONFIG['QueryDecompositionAgentParallel']),
        QueryDecompositionAgentSerial(AGENT_CONFIG['QueryDecompositionAgentSerial']),
        RetrievalAgent(AGENT_CONFIG['RetrievalAgent']),
        DocumentSelectionAgent(AGENT_CONFIG['DocumentSelectionAgent']),
        AnswerGenerationAgent(AGENT_CONFIG['AnswerGenerationAgent']),
    ])

    workflow = [
        "QueryRewriteAgent",
        "RetrievalAgent",
        "DocumentSelectionAgent",
        "AnswerGenerationAgent"
    ]

    query = "What is the impact of climate change on polar bears?"

    # workflow logs
    logger.info(f"====== Workflow ======")
    for agent_name in workflow:
        logger.info(f"\t==> {agent_name}")

    # query logs
    logger.info(f"\n====== Query ======")
    logger.info(f"\t==> {query}\n")

    logger.info(f"====== Starting Process... ======")
    wf = AgentWorkflow(pool, workflow)
    final_context = wf.run(query)

    logger.info(f"\n====== Token Usage ======")
    tracker = TokenUsageTracker()
    logger.info(f'QueryRewriteAgent: {tracker.get_usage("QueryRewriteAgent")}')
    logger.info(f'QueryDecompositionAgentParallel: {tracker.get_usage("QueryDecompositionAgentParallel")}')
    logger.info(f'QueryDecompositionAgentSerial: {tracker.get_usage("QueryDecompositionAgentSerial")}')
    logger.info(f'RetrievalAgent: {tracker.get_usage("RetrievalAgent")}')
    logger.info(f'DocumentSelectionAgent: {tracker.get_usage("DocumentSelectionAgent")}')
    logger.info(f'AnswerGenerationAgent: {tracker.get_usage("AnswerGenerationAgent")}')

    logger.info(f'All: {tracker.get_usage("all")}')

