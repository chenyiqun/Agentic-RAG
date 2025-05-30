from collections import defaultdict
from threading import Lock
import logging


class TokenUsageTracker:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TokenUsageTracker, cls).__new__(cls)
                cls._instance.data = defaultdict(lambda: {
                    'completion_tokens': 0,
                    'prompt_tokens': 0,
                    'total_tokens': 0,
                    'calls': 0
                })
                cls._instance.logs = []
            return cls._instance

    def record(self, agent_name, usage):
        """
        记录 token 使用情况。
        usage 是一个 dict，包含：completion_tokens, prompt_tokens, total_tokens。
        """

        if agent_name != 'all' and 'model' not in self.data[agent_name]:
            self.data[agent_name]['model'] = usage.get('model', '')

        self.data[agent_name]['completion_tokens'] += usage.get('completion_tokens', 0)
        self.data[agent_name]['prompt_tokens'] += usage.get('prompt_tokens', 0)
        self.data[agent_name]['total_tokens'] += usage.get('total_tokens', 0)
        self.data[agent_name]['calls'] += 1

    def get_usage(self, agent_name=None):
        if agent_name:
            return self.data.get(agent_name, {})
        return dict(self.data)

    def reset(self):
        self.data = defaultdict(lambda: {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0,
            'calls': 0
        })


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
