from BaseAgent3 import *
from PlanningAgent import PlanningAgent
from normalize_answers import *
from tqdm import tqdm
from collections import Counter

import os


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

def save_lists_to_files(batch_questions, batch_predict_answers, batch_golden_answers, new_dir_path):
    
    # 定义保存文件的路径
    questions_file = os.path.join(new_dir_path, 'questions.txt')
    predict_answers_file = os.path.join(new_dir_path, 'predict_answers.txt')
    golden_answers_file = os.path.join(new_dir_path, 'golden_answers.txt')

    # 将列表写入文件
    with open(questions_file, 'a', encoding='utf-8') as f:
        for question in batch_questions:
            f.write(f"{question}\n")
    
    with open(predict_answers_file, 'a', encoding='utf-8') as f:
        for answer in batch_predict_answers:
            f.write(f"{answer}\n")
    
    with open(golden_answers_file, 'a', encoding='utf-8') as f:
        for answer in batch_golden_answers:
            f.write(f"{answer}\n")


logger = setup_logger(log_file='log/log_test.log')

if __name__ == "__main__":

    dataset_name = 'hotpotqa'
    print('Testing on {}'.format(dataset_name))

    # 准备数据
    project_path = '/root/workspace/env_run/agentic_rag'
    data_train, data_test = [], []
    if dataset_name == 'ambigqa':
        with open('/root/workspace/env_run/rag_reranker/data/ambigqa/train_data.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data_train.append(json.loads(line.strip()))
        with open('/root/workspace/env_run/rag_reranker/data/ambigqa/test_data.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data_test.append(json.loads(line.strip()))
        print('len(data_train): {}, len(data_test): {}'.format(len(data_train), len(data_test)))

    elif dataset_name == 'hotpotqa':
        with open('/root/workspace/env_run/rag_reranker/data/hotpotqa/hotpotqa_train_questions_and_answers.json', 'r', encoding='utf-8') as file:
            data_train = json.load(file)
        with open('/root/workspace/env_run/rag_reranker/data/hotpotqa/hotpotqa_test_questions_and_answers.json', 'r', encoding='utf-8') as file:
            data_test = json.load(file)
        print('len(data_train): {}, len(data_test): {}'.format(len(data_train), len(data_test)))

    elif dataset_name == '2wikimultihopqa':
        with open('/root/workspace/env_run/rag_reranker/data/2wikimultihopqa/train.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data_train.append(json.loads(line.strip()))
        with open('/root/workspace/env_run/rag_reranker/data/2wikimultihopqa/dev.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data_test.append(json.loads(line.strip()))
        print('len(data_train): {}, len(data_test): {}'.format(len(data_train), len(data_test)))
    elif dataset_name == 'musique':
        data_train, data_test = [], []
        with open('/root/workspace/env_run/rag/data/musique/musique_ans_v1.0_train.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data_train.append(json.loads(line.strip()))
        with open('/root/workspace/env_run/rag/data/musique/musique_ans_v1.0_dev.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data_test.append(json.loads(line.strip()))
        print('len(data_train): {}, len(data_test): {}'.format(len(data_train), len(data_test)))

    # questions and answers
    # data_test = data_test[:10]
    questions = [item['question'] for item in data_test]
    if dataset_name == 'ambigqa':
        golden_answers = [concatenate_strings(item['nq_answer']) for item in data_test]
    else:
        golden_answers = [item['answer'] for item in data_test]
    # print('len(questions): {}'.format(len(questions)))


    model_name = "/root/workspace/env_run/verl/models_fund/Qwen/Qwen2.5-7B-Instruct"
    # 获取可用的GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    planning_agent = PlanningAgent(model, tokenizer)

    batch_size=100
    batch_runner = BatchAgentWorkflow(planning_agent, batch_size)
    predict_answers = batch_runner.run_batch(questions)

    for answer in predict_answers:
        print(answer)

    logger.info(f"\n====== Token Usage Summary ======")
    tracker = TokenUsageTracker()
    logger.info(f'Token Usage: {tracker.get_usage()}')
    print(f'All Token Usage: {tracker.get_usage()}')

    print('Final metrics: {}'.format(compute_scores(predict_answers, golden_answers)))
    
