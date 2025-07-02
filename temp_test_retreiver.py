import time
from abc import ABC, abstractmethod
from openai import OpenAI
from typing import Dict, List, Any
import requests
import json
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed


def retreiver(questions_batch):
    headers = {'Content-Type': 'application/json'}
    payload = {
        'questions': questions_batch,
        'N': 3
    }
    response = requests.post('http://localhost:8000/search', headers=headers, data=json.dumps(payload))
    # print("***********")
    # print(len(response.json()))
    # print("***********")
    response.raise_for_status()  # Raise an error for bad responses

    top_k_docs_batch_list = []
    for question in response.json():
        temp_docs_list = []
        for doc_id, doc in enumerate(question['top_k_docs']):
            temp_docs_list.append(doc)
        top_k_docs_batch_list.append(temp_docs_list)
    
    return top_k_docs_batch_list

questions = ['Beijing']*33

top_k_docs_list = []
batch_size = 15  # Process questions in batches of 15
for i in range(0, len(questions), batch_size):
    questions_batch = questions[i:i + batch_size]
    retrieved_docs_batch = retreiver(questions_batch)
    print('len(retrieved_docs_batch), len(retrieved_docs_batch[0])', len(retrieved_docs_batch), len(retrieved_docs_batch[0]))
    top_k_docs_list.extend(retrieved_docs_batch)

print(len(top_k_docs_list), len(top_k_docs_list[0]))
print(top_k_docs_list[0])

