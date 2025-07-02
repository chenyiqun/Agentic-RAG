import os

def save_lists_to_files(questions, predict_answers, golden_answers):
    # 确保'results'文件夹存在
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 获取'results'文件夹下的所有文件夹并找到下一个编号
    existing_dirs = [int(d) for d in os.listdir('results') if os.path.isdir(os.path.join('results', d)) and d.isdigit()]
    if existing_dirs:
        next_dir_number = max(existing_dirs) + 1
    else:
        next_dir_number = 0
    
    # 创建新的文件夹
    new_dir_path = os.path.join('results', str(next_dir_number))
    os.makedirs(new_dir_path)
    
    # 定义保存文件的路径
    questions_file = os.path.join(new_dir_path, 'questions.txt')
    predict_answers_file = os.path.join(new_dir_path, 'predict_answers.txt')
    golden_answers_file = os.path.join(new_dir_path, 'golden_answers.txt')

    # 将列表写入文件
    with open(questions_file, 'w', encoding='utf-8') as f:
        for question in questions:
            f.write(f"{question}\n")
    
    with open(predict_answers_file, 'w', encoding='utf-8') as f:
        for answer in predict_answers:
            f.write(f"{answer}\n")
    
    with open(golden_answers_file, 'w', encoding='utf-8') as f:
        for answer in golden_answers:
            f.write(f"{answer}\n")

    print(f"Results saved in {new_dir_path}")

# 示例数据
questions = ["What is the capital of France?", "What is 2+2?"]
predict_answers = ["Paris", "4"]
golden_answers = ["Paris", "4"]

save_lists_to_files(questions, predict_answers, golden_answers)