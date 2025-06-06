
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import get_conversation_template
import time
import json

from flask import Flask, request, jsonify
import faiss
import torch
import sys
import os

from threading import Lock  # 引入Lock类
# 创建一个全局锁对象
request_lock = Lock()

# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]

    return sentence_embeddings

def get_embeddings(sentences):
    # Apply tokenizer
    sentences_input = retriever_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    if torch.cuda.is_available():
        sentences_input = sentences_input.to(device)
        with torch.no_grad():  # 如果不需要梯度，使用torch.no_grad()可以减少内存消耗  
            sentences_output = retriever_model(**sentences_input)

        # Compute token embeddings
        sentences_embedding = mean_pooling(sentences_output[0], sentences_input['attention_mask'])
        
        sentences_embedding = sentences_embedding.cpu().numpy()

        # Delete variables and empty cache
        del sentences_input, sentences_output
        torch.cuda.empty_cache()

    else:
        sentences_output = retriever_model(**sentences_input) 
        sentences_embedding = mean_pooling(sentences_output[0], sentences_input['attention_mask'])
        sentences_embedding = sentences_embedding.numpy()

    return sentences_embedding


# 创建 Flask 应用实例
app = Flask(__name__)
# 定义 /search 路由和处理函数
@app.route('/search', methods=['POST'])
def get_top_k_docs():
    # 获取锁，确保一次只处理一个请求
    with request_lock:
        # 从请求体中获取 JSON 数据
        data = request.get_json()
        questions = data.get('questions', [])
        N = data.get('N', 10)

        questions_embeddings = get_embeddings(questions)
        batch_D, batch_I = index.search(questions_embeddings, N)

        # 结果列表，用于存储每个问题及其对应的文档
        results = [] 
        # 遍历每个问题，生成对应的 top_k_docs
        for q in range(len(questions)):
            question = questions[q]
            top_k_docs = [str(df.loc[doc_id, 'title'])+'\n' + str(df.loc[doc_id, 'text']) for doc_id in batch_I[q]]
            # 构造成字典并添加到结果列表
            result = {
                'question': question,
                'top_k_docs': top_k_docs
            }
            results.append(result)
        
        # 返回 JSON 响应
        return jsonify(results)


if __name__ == '__main__':

    port = 8000
    # if len(sys.argv) > 1:
    #     port = int(sys.argv[1])
    # cuda_id = port - 8000
    # print('cuda_id: {}'.format(cuda_id))

    # # 设置环境变量
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)

    # print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    # print("Using device cuda:{}".format(cuda_id))

    # ************************************ load indexing and retrieval model ************************************
    pre_path = '/root/paddlejob/workspace/env_run/rag_reranker/corpus/'
    docs_path = pre_path+'psgs_w100.tsv'
    retrieval_model_name = 'e5'

    if retrieval_model_name == 'contriever':
        retriever_model_path = '/root/paddlejob/workspace/env_run/rag_reranker/models_fund/facebook/contriever'
        index_path = pre_path+'wikipedia.contriever'
    elif retrieval_model_name == 'bge':
        retriever_model_path = '/root/paddlejob/workspace/env_run/rag_reranker/models_fund/BAAI/bge-base-en-v1.5'
        index_path = pre_path+'wikipedia.bge'
    elif retrieval_model_name == 'e5':
        retriever_model_path = '/root/paddlejob/workspace/env_run/rag_reranker/models_fund/intfloat/e5-base-v2'
        index_path = pre_path+'wikipedia.e5'

    # loading retriever model
    print('*'*20)
    print('loading retriever model')
    # 检查GPU是否可用  
    start_time = time.time()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # 设置PyTorch设备
    device = torch.device("cuda:0")  # 这里的0是指CUDA_VISIBLE_DEVICES中的第一个
    retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_path)
    retriever_model = AutoModel.from_pretrained(retriever_model_path)
    retriever_model = retriever_model.to(device)
    end_time = time.time()
    print('time consuming: {} seconds'.format(end_time - start_time))

    # loading index
    print('*'*20)
    print('loading index')
    start_time = time.time()

    # # cpu faiss
    # print('using cpu index')
    # index = faiss.read_index(index_path)
    # gpu faiss
    print('put index on gpu')
    res = faiss.StandardGpuResources()  # 创建一个Faiss资源对象
    read_index = faiss.read_index(index_path)  # 读取索引
    index = faiss.index_cpu_to_gpu(res, 0, read_index)  # 把索引放到第cuda_id个GPU上

    end_time = time.time()
    print('time consuming: {} seconds'.format(end_time - start_time))

    # loading docs data
    print('*'*20)
    print('loading data')
    start_time = time.time()
    df = pd.read_csv(docs_path, sep='\t')
    end_time = time.time()
    print('time consuming: {} seconds'.format(end_time - start_time))

    # ************************************ run the server ************************************
    app.run(host='0.0.0.0', port=port)
