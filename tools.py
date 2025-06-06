from collections import defaultdict
from threading import Lock
import logging
from typing import Dict, List


# per 1K tokens, https://platform.closeai-asia.com/pricing/
MODEL_PRICING = {
    'gpt-4o-2024-11-20': {'prompt': 0.00375, 'completion': 0.015},
    'gpt-4o-mini-2024-07-18': {'prompt': 0.000225,'completion': 0.0009},
}

class TokenUsageTracker:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TokenUsageTracker, cls).__new__(cls)
                cls._instance.logs: List[Dict] = []
            return cls._instance

    def record(self, usage: Dict) -> None:
        model = usage.get('model', '')
        completion_pricing = MODEL_PRICING[model]['completion']
        prompt_pricing = MODEL_PRICING[model]['prompt']

        self.logs.append({
            'model': model,
            'agent_name': usage.get('agent_name', ''),
            'completion_tokens': usage.get('completion_tokens', 0),
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0),
            'completion_price': completion_pricing * usage.get('completion_tokens', 0) / 1000,
            'prompt_price': prompt_pricing * usage.get('prompt_tokens', 0) / 1000,
            'total_price': completion_pricing * usage.get('completion_tokens', 0) / 1000 + prompt_pricing * usage.get('prompt_tokens', 0) / 1000,
        })

    def get_usage(self) -> List[Dict]:
        return [
            {   
                'calls': sum([1 for l in self.logs]),
                'completion_tokens': sum([l['completion_tokens'] for l in self.logs]),
                'prompt_tokens': sum([l['prompt_tokens'] for l in self.logs]),
                'total_tokens': sum([l['total_tokens'] for l in self.logs]),
                'completion_price': sum([l['completion_price'] for l in self.logs]),
                'prompt_price': sum([l['prompt_price'] for l in self.logs]),
                'total_price': sum([l['total_price'] for l in self.logs]),
            }
        ]


def setup_logger(name='logger', log_file='log.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        formatter = logging.Formatter(
            '%(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def setup_logger_no_print(name='logger', log_file='log.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        formatter = logging.Formatter(
            '%(message)s'
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger
