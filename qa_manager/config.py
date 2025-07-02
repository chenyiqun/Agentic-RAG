API_KEY = "sk-PVRw74ZXeE1Wl3pXejUTCmM16g46QiMYbIU9bGTyh1YDxGwV"
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
        'model': 'gpt-4o-mini',  # gpt-4o-mini
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
        **COMMON_CONFIG,
        'name': 'RetrievalAgent',
        'model': 'gpt-4o-mini',
        'temperature': 0,
        'max_tokens': 10000,
        'timeout': 300,
        'api_url': 'http://localhost:8000/search',
        'num_results': 5
    },
    "AnswerSummarizationAgent": {
        **COMMON_CONFIG,
        'name': 'AnswerSummarizationAgent',
        'model': 'gpt-4o-mini',
        'temperature': 0,
        'max_tokens': 10000,
        'timeout': 300,
        'api_url': 'http://localhost:8000/search',
        'num_results': 5
    }
}

EXAMPLE_PROMPT = '''
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
'''

