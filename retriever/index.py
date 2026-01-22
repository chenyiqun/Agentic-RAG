import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]

    return sentence_embeddings

def get_embeddings(sentences):
    # Apply tokenizer
    sentences_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    if torch.cuda.is_available():
        sentences_input = sentences_input.to(device)
        with torch.no_grad():  # 如果不需要梯度，使用torch.no_grad()可以减少内存消耗  
            sentences_output = model(**sentences_input)

        # Compute token embeddings
        sentences_embedding = mean_pooling(sentences_output[0], sentences_input['attention_mask'])
        
        sentences_embedding = sentences_embedding.cpu().numpy()

        # Delete variables and empty cache
        del sentences_input, sentences_output
        torch.cuda.empty_cache()

    else:
        sentences_output = model(**sentences_input) 
        sentences_embedding = mean_pooling(sentences_output[0], sentences_input['attention_mask'])
        sentences_embedding = sentences_embedding.numpy()

    return sentences_embedding


if __name__ == '__main__':

    pre_path = '/root/workspace/env_run/verl/retriever/'

    retrieval_model_name = 'e5'  # contriever e5 bge
    if retrieval_model_name == 'contriever':
        retriever_model_path = '/root/workspace/env_run/rag_reranker/models_fund/facebook/contriever'
        save_path = pre_path+'wikipedia.contriever'
    elif retrieval_model_name == 'bge':
        retriever_model_path = '/root/workspace/env_run/rag_reranker/models_fund/BAAI/bge-base-en-v1.5'
        save_path = pre_path+'wikipedia.bge'
    elif retrieval_model_name == 'e5':
        retriever_model_path = '/root/workspace/env_run/verl/retriever/intfloat/e5-base-v2'
        save_path = pre_path+'wikipedia.e5'

    # loading data
    print('*'*20)
    print('loading data')
    df = pd.read_csv(pre_path+'psgs_w100.tsv', sep='\t')

    # spliting data
    print('*'*20)
    print('spliting data')
    batch_size = 3000
    part_num = len(df)//batch_size+1
    sentences_list = []
    for k in range(part_num):
        sentences = []
        for i in range(batch_size*k, min(batch_size*(k+1), len(df))):
            sentences.append(str(df.loc[i, 'title']) + '\t' + str(df.loc[i, 'text']))
        sentences_list.append(sentences)

    if len(sentences_list[-1]) == 0:
        sentences_list = sentences_list[:-1]

    # loading retriever model
    print('*'*20)
    print('loading retriever model')
    import torch
    from transformers import AutoTokenizer, AutoModel
    # 检查GPU是否可用  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    tokenizer = AutoTokenizer.from_pretrained(retriever_model_path)
    model = AutoModel.from_pretrained(retriever_model_path)
    model = model.to(device)

    # indexing
    print('*'*20)
    print('getting embeddings')

    # construct index
    embeddings_list = []
    for x in tqdm(range(len(sentences_list))):
        sentences = sentences_list[x]
        embeddings = get_embeddings(sentences)
        embeddings_list.append(embeddings)
    #     # add index

    # construct index in batch way
    # embeddings_list = []
    # batch_size = 1000
    # Process sentences in batches
    # for i in tqdm(range(0, len(sentences_list), batch_size)):
    #     batch = sentences[i:i + batch_size]
    #     embeddings = get_embeddings(batch)
    #     embeddings_list.append(embeddings)
    # for i in tqdm(range(0, len(sentences_list), batch_size)):
    #     batch = sentences[i:i + batch_size]
    #     embeddings = get_embeddings(batch)
    #     embeddings_list.append(embeddings)

    # concat all embeddings
    concat_embeddings = np.concatenate(embeddings_list, axis=0)
    print('concat_embeddings', len(concat_embeddings), concat_embeddings.shape)

    # constructing index
    print('*'*20)
    print('constructing index')
    dim = 768
    index = faiss.IndexFlatL2(dim)
    index.add(concat_embeddings)

    # saving index
    print('*'*20)
    print('saving index')
    faiss.write_index(index, save_path)


    print('Fininsed!')
