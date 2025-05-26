API_KEY = ""
API_BASE = "https://api.openai-proxy.org/v1"

COMMON_CONFIG = {
    'api_key': API_KEY,
    'api_base': API_BASE,
}

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
        'model': 'gpt-4o',  # 4o-mini 不行
        'temperature': 0,
        'max_tokens': 10000,
        'timeout': 300,
    },
}