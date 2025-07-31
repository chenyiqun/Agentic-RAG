import requests
import json


def call_search_api(questions, num_results=10):
    url = 'http://localhost:8000/search'  # API的URL
    headers = {'Content-Type': 'application/json'}  # 设置请求头

    # 构建请求体
    payload = {
        'questions': questions,
        'N': num_results
    }

    # 发送POST请求
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # 检查请求是否成功
    if response.status_code == 200:
        # 解析响应的JSON数据
        results = response.json()
        return results
    else:
        print(f"Request failed with status code {response.status_code}")
        return None


class RetrieverClient:
    def __init__(self, api_url='http://localhost:8000/search'):
        self.api_url = api_url

    def search(self, questions, num_results=10):
        headers = {'Content-Type': 'application/json'}
        payload = {
            'questions': questions,
            'N': num_results
        }

        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise an error for bad responses

            results = response.json()
            return results

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None


if __name__ == "__main__":

    # 使用示例
    retriever_client = RetrieverClient(api_url='http://localhost:8000/search')
    questions = ["What is the capital of France?", "Who wrote '1984'?"]
    results = retriever_client.search(questions)

    print('results', results)

    # 打印结果
    if results:
        for result in results:
            print(f"Question: {result['question']}")
            for doc_id in range(len(result['top_k_docs'])):
                doc = result['top_k_docs'][doc_id]
                print(f"Document {doc_id}: {doc}")
            print("\n")

    # print(results[0].keys())