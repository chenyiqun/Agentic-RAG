API_KEY = ""
API_BASE = "https://api.openai-proxy.org/v1"

COMMON_CONFIG = {
    'api_key': API_KEY,
    'api_base': API_BASE,
}

# gpt-4o-mini gpt-4o gpt-3.5-turbo
AGENT_CONFIG = {
    "QueryRewriteAgent": {
        **COMMON_CONFIG,
        'name': 'QueryRewriteAgent',
        'model': 'gpt-4o-mini',
        'temperature': 0,
        'max_tokens': 10000,
        'timeout': 300,
    },
    "QueryDecompositionAgentParallel": {
        **COMMON_CONFIG,
        'name': 'QueryDecompositionAgentParallel',
        'model': 'gpt-4o-mini',
        'temperature': 0,
        'max_tokens': 10000,
        'timeout': 300,
    },
    "QueryDecompositionAgentSerial": {
        **COMMON_CONFIG,
        'name': 'QueryDecompositionAgentSerial',
        'model': 'gpt-4o',  # 4o-mini 不行！！！！！！
        'temperature': 0,
        'max_tokens': 10000,
        'timeout': 300,
    },
    "DocumentSelectionAgent": {
        **COMMON_CONFIG,
        'name': 'DocumentSelectionAgent',
        'model': 'gpt-4o-mini',
        'temperature': 0,
        'max_tokens': 10000,
        'timeout': 300,
    },
    "AnswerGenerationAgent": {
        **COMMON_CONFIG,
        'name': 'AnswerGenerationAgent',
        'model': 'gpt-4o-mini',
        'temperature': 0,
        'max_tokens': 10000,
        'timeout': 300,
    },
    "RetrievalAgent": {
        'name': 'RetrievalAgent',
        'api_url': 'http://localhost:8000/search',
        'num_results': 5
    },
    "IterativeWorkflowAgent": {
        'name': 'IterativeWorkflowAgent',
    }
}

