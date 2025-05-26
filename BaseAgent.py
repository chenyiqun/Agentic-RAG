import time
from abc import ABC, abstractmethod
from openai import OpenAI
from typing import Dict, List
from config import *


class BaseAgent(ABC):
    def __init__(self, config):
        self.name = config.get('name', '')
        self.model = config.get('model', 'gpt-4o-mini')
        self.api_key = config.get('api_key', '')
        self.api_base = config.get('api_base', 'https://api.openai.com/v1')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2000)
        self.timeout = config.get('timeout', 30)
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    @abstractmethod
    def build_message(self, input_data):
        """
        构建输入消息的方法。不同的agent可以有不同的实现。
        """
        pass

    def get_response(self, message: List[Dict]):
        """
        通用的响应获取方法，调用OpenAI API。
        """
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
                print(e)
                time.sleep(current_sleep_time)
                if current_sleep_time < max_waiting_time:
                    current_sleep_time *= 2

        return response.choices[0].message.content

    @abstractmethod
    def post_process(self, response):
        """
        后处理API响应的结果。
        """
        pass

    def run(self, input_data):
        """
        封装调用流程：构造 prompt -> 获取响应 -> 后处理
        """
        message = self.build_message(input_data)
        raw_response = self.get_response(message)
        return self.post_process(raw_response)



class QueryRewriteAgent(BaseAgent):
    """
    # --- 简单 Query Rewrite ---
    """

    def build_message(self, query: str):
        return [
            {'role': 'system',
             'content': 'You are a professional assistant skilled at rewriting overly detailed or redundant questions into a single, concise, and searchable query. Your goal is to keep only the essential part of the question that is needed to find the answer efficiently.'},
            {'role': 'assistant', 'content': 'Okay, I will return a concise rewritten query.'},
            {'role': 'user',
             'content': f'Original question is {query}. Now rewrite the original question into a single, clear query that focuses only on the essential information needed to find the answer. Avoid unnecessary context, vague references, and maintain specificity. Output only the rewritten query without any extra explanation or formatting.'}
        ]

    def post_process(self, response) -> List:
        return [response]


class QueryDecompositionAgentParallel(BaseAgent):
    def build_message(self, query: str):
        return [
            {'role': 'system',
             'content': 'You are a professional assistant skilled at decomposing complex multi-entity or multi-location questions into multiple independent and searchable sub-questions. Each sub-question should be specific, logically complete, and not repeat others.'},
            {'role': 'assistant', 'content': 'Okay, I will return the parallel sub-questions.'},
            {'role': 'user',
             'content': f"Original question is '{query}'. Now decompose the question into multiple specific sub-questions that can be independently searched. Each sub-question should be on a separate line, avoid vague demonstratives or repetition, and ensure that each question is self-contained."}
        ]

    def post_process(self, response) -> List:
        return [q.strip() for q in response.split('\n') if q.strip()]


class QueryDecompositionAgentSerial(BaseAgent):
    def build_message(self, query: str):
        return [
            {'role': 'system', 'content': (
                'You are a professional assistant skilled at decomposing complex questions into a sequence of logically dependent, '
                'independently searchable sub-questions. Each sub-question must:\n'
                '- Be self-contained and specific\n'
                '- Be suitable for direct information retrieval from search engines or structured databases\n'
                '- Avoid logical inference, reasoning, or computation (e.g., "how to calculate age" or "what is the current date")\n'
                '- Depend on the previous answer only for context, not for logic or calculation\n'
                'Only include sub-questions that yield factual, retrievable answers. Do NOT include questions that require reasoning, '
                'math, date comparison, or context-specific interpretation.'
            )},
            {'role': 'assistant',
             'content': 'Understood. I will return only factual, retrievable sub-questions, one per line.'},
            {'role': 'user',
             'content': (
                 f'Original question is: {query}\n'
                 'Now decompose the original question into a logically ordered list of sub-questions. '
                 'Write one sub-question per line.'
             )}
        ]

    def post_process(self, response) -> List:
        return [q.strip() for q in response.split('\n') if q.strip()]


class DocumentSelectionAgent(BaseAgent):
    def build_message(self, input_data):
        # 实现特定于文档选择的message构建逻辑
        message = f"Select documents relevant to: {input_data}"
        return message

    def post_process(self, response):
        # 实现获取文档选择响应的逻辑
        response = f"Documents selected based on: {response}"
        return response


class AnswerGenerationAgent(BaseAgent):
    def build_message(self, input_data):
        # 实现特定于答案生成的message构建逻辑
        message = f"Generate answer for: {input_data}"
        return message

    def post_process(self, response):
        # 实现获取答案生成响应的逻辑
        response = f"Generated answer for: {response}"
        return response


if __name__ == "__main__":
    queries = [
        ['Lin Zhixuan, at 58 years old, performed "Wukong" in Singer 2025 and ranked second to last. Who is the composer of this song?'],
        ['What were the GDP rankings of Beijing, Houston, and Singapore in 2024?'],
        ['How old is the singer who performed Wukong in the first episode of Singer 2025?',
         'I was impressed by the talk given by the academic director of ADL Episode 158. What are some well-known open-source projects in AI search developed by their research team?']
    ]

    agents = [
        QueryRewriteAgent(AGENT_CONFIG['QueryRewriteAgent']),
        QueryDecompositionAgentParallel(AGENT_CONFIG['QueryDecompositionAgentParallel']),
        QueryDecompositionAgentSerial(AGENT_CONFIG['QueryDecompositionAgentSerial'])
    ]

    for query, agent in zip(queries, agents):
        print(f"\n--- {agent.name} ---")
        for qq in query:
            print(f"\n--- {qq} ---")
            result = agent.run(qq)
            print(result)
