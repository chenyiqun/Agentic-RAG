from BaseAgent import *
from normalize_answers import *
from tqdm import tqdm
from collections import Counter


def normalize_answer_final(answer):
    pre_answer = answer.split('\n\n')[-1].split('Answer: ')[-1].split('The answer is: ')[-1]
    final_answer = normalize_answer(pre_answer)
    return final_answer

def answer_post_refine(answer):
    return answer.split("Answer: ")[-1]

def compute_scores(predict_answers, golden_answers):
    assert len(predict_answers) == len(golden_answers), "预测答案和标准答案的长度不相等"
    final_metric = {"acc": 0, "em": 0, "f1": 0, "precision": 0, "recall": 0}
    total = len(predict_answers)

    for prediction, ground_truth in zip(predict_answers, golden_answers):
        normalized_prediction = normalize_answer_final(prediction)
        normalized_ground_truth = normalize_answer_final(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        
        if normalized_ground_truth in normalized_prediction:# or normalized_prediction in normalized_ground_truth:
            final_metric["acc"] += 1.0

        if normalized_prediction == normalized_ground_truth:
            final_metric["em"] += 1.0

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        final_metric["f1"] += f1
        final_metric["precision"] += precision
        final_metric["recall"] += recall

    for k in ['acc', 'em', 'f1', 'precision', 'recall']:
        final_metric[k] /= total

    return final_metric


logger = setup_logger(log_file='log/log_test.log')

if __name__ == "__main__":

    dataset_name = 'hotpotqa'
    print('Testing on {}'.format(dataset_name))

    # 准备数据
    project_path = '/root/paddlejob/workspace/env_run/agentic_rag'
    data_train, data_test = [], []
    if dataset_name == 'ambigqa':
        with open('/root/paddlejob/workspace/env_run/rag_reranker/data/ambigqa/train_data.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data_train.append(json.loads(line.strip()))
        with open('/root/paddlejob/workspace/env_run/rag_reranker/data/ambigqa/test_data.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data_test.append(json.loads(line.strip()))
        print('len(data_train): {}, len(data_test): {}'.format(len(data_train), len(data_test)))

    elif dataset_name == 'hotpotqa':
        with open('/root/paddlejob/workspace/env_run/rag_reranker/data/hotpotqa/hotpotqa_train_questions_and_answers.json', 'r', encoding='utf-8') as file:
            data_train = json.load(file)
        with open('/root/paddlejob/workspace/env_run/rag_reranker/data/hotpotqa/hotpotqa_test_questions_and_answers.json', 'r', encoding='utf-8') as file:
            data_test = json.load(file)
        print('len(data_train): {}, len(data_test): {}'.format(len(data_train), len(data_test)))

    elif dataset_name == '2wikimultihopqa':
        with open('/root/paddlejob/workspace/env_run/rag_reranker/data/2wikimultihopqa/train.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data_train.append(json.loads(line.strip()))
        with open('/root/paddlejob/workspace/env_run/rag_reranker/data/2wikimultihopqa/dev.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data_test.append(json.loads(line.strip()))
        print('len(data_train): {}, len(data_test): {}'.format(len(data_train), len(data_test)))
    elif dataset_name == 'musique':
        data_train, data_test = [], []
        with open('/root/paddlejob/workspace/env_run/rag/data/musique/musique_ans_v1.0_train.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data_train.append(json.loads(line.strip()))
        with open('/root/paddlejob/workspace/env_run/rag/data/musique/musique_ans_v1.0_dev.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data_test.append(json.loads(line.strip()))
        print('len(data_train): {}, len(data_test): {}'.format(len(data_train), len(data_test)))

    # questions and answers
    data_test = data_test[:10]
    questions = [item['question'] for item in data_test]
    if dataset_name == 'ambigqa':
        golden_answers = [concatenate_strings(item['nq_answer']) for item in data_test]
    else:
        golden_answers = [item['answer'] for item in data_test]
    # print('len(questions): {}'.format(len(questions)))

    # initial agent pool
    pool = AgentPool()
    pool.register([
        QueryRewriteAgent(AGENT_CONFIG['QueryRewriteAgent']),
        QueryDecompositionAgentParallel(AGENT_CONFIG['QueryDecompositionAgentParallel']),
        QueryDecompositionAgentSerial(AGENT_CONFIG['QueryDecompositionAgentSerial']),
        RetrievalAgent(AGENT_CONFIG['RetrievalAgent']),
        DocumentSelectionAgent(AGENT_CONFIG['DocumentSelectionAgent']),
        AnswerGenerationAgent(AGENT_CONFIG['AnswerGenerationAgent']),
        IterativeWorkflowAgent(AGENT_CONFIG['IterativeWorkflowAgent'])
    ])

    Batch_size = 1
    Workflow = ["RetrievalAgent", "AnswerGenerationAgent"]
    predict_answers = []
    query_workflow_pairs_list = []
    for i in range(0, len(questions), Batch_size):
        batch_questions = questions[i: min(i+Batch_size, len(questions))]
        query_workflow_pairs = [{'query': question, 'workflow': Workflow} for question in batch_questions]
        query_workflow_pairs_list.append(query_workflow_pairs)
        
    for query_workflow_pairs in tqdm(query_workflow_pairs_list):
        batch_runner = BatchAgentWorkflow(pool, max_workers=len(query_workflow_pairs))
        final_contexts = batch_runner.run_batch(query_workflow_pairs)
        batch_answers = []
        for context in final_contexts:
            batch_answers.append(context['answer'])
        predict_answers.extend(batch_answers)

    print('predict_answers: ', predict_answers)

    logger.info(f"\n====== Token Usage Summary ======")
    tracker = TokenUsageTracker()
    logger.info(f'Token Usage: {tracker.get_usage()}')
    print(f'Token Usage: {tracker.get_usage()}')

    print('Final metrics: {}'.format(compute_scores(predict_answers, golden_answers)))
    