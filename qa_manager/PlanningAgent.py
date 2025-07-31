from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any, Tuple, Optional
import torch


class PlanningAgent:
    def __init__(self):
        pass

    def create_planning_messages(self, question: str, is_sub: bool = False) -> List[Dict[str, str]]:
        # * planning agent功能描述
        # * QR S G等agent的功能描述
        # * 输出样例example
        # * memory内容
        # * 重述planning agent的功能
        if not is_sub:
            # question = context['query']
            messages = [
                {"role": "system", "content": "You are a helpful assistant to plan a Workflow for the given information using the give tool/agent."},
                {"role": "assistant", "content": "Ok, please show me you requirement information and the available tool/agent"},
                {"role": "user", "content": """The descriptions of the available tool/agent are as follows: 

    Query Rewriter (QR): The input of QR is a question, and the output of QR is a rewritten question which is more concise, clearer, and more accurate.
    Query Decomposition Parallel (QDP): The input of QDP is a question, and the output of QDP contains multiple independent sub-questions. And QDP is suitable for decomposing those multi-hop questions that can be decomposed into multiple searchable and independent sub-questions in parallel.
    Query Decomposition Serial (QDS): The input of QDS is a question. The output of QDS is multiple related sub-questions, and later sub-questions may need answers to previous sub-questions before they become searchable. And QDS is suitable for decomposing those multi-hop questions with dependencies into multiple sub-questions.
    Retrieval (R): The input of R is a question, and the output of R is multiple relevant candidate documents to the given question.
    Document Selector (DS): The input of DS is a given question and multiple relevant candidate documents from search engine, and the output of DS is the documents which are helpful to answer the given question.
    Answer Generator (AG): The input of G may contains the given question, the some related candidate documents from search engine. The output of AG is the answer to the given question.

    With above tools/agents you can build a Workflow to answer a given question.


    The followings are some examples:

    For simple questions that may require external knowledge to answer:
    Question: "Which team is the 2025 NBA champion?"
    Workflow: QR,R,AG
    We can just retrieve the external information about the given question through Retrieval Agent (RA), then the Answer Generator (AG) can output the answer to the given question.

    For multi-hop questions like this:
    Question: "Which city will have the highest GDP in 2024: Singapore, Houston, Beijing?"
    Workflow: QDP
    This kind of problem requires first breaking the initial question into multiple independent searchable sub-questions. For such a question, only the QDP needs to be given, and no other tool/agent needs to be given.

    For multi-hop questions like this:
    Question: "What is the most played song on Michael Jackson's third best-selling album?"
    Workflow: QDS
    This kind of problem requires breaking down the original problem into multiple subproblems that depend on each other. Convenient for subsequent processing. For such a question, only the QDS needs to be given, and no other tool/agent needs to be given.

    For multi-hop questions like this:
    Question: When was the director of Terminator 2 born?
    Workflow: QDS
    This kind of problem requires breaking down the original problem into multiple subproblems that depend on each other. Convenient for subsequent processing. For such a question, only the QDS needs to be given, and no other tool/agent needs to be given.



    Next I'll give you the Question that need to be answered, just output the appropriate Workflow in the format above, don't output anything else.
    """},
                {"role": "assistant", "content": "Ok, you can provide the Question and I will give the appropriate Workflow to answer the Question."},
                {"role": "user", "content": """Please give the appropriate Workflow in the format requested above.
    Qustion: "{}"
    Workflow: """.format(question)},
            ]

        else:
            # current_step = context["current_step"]
            # question = context['sub_query'][current_step]
            messages = [
                {"role": "system", "content": "You are a helpful assistant to plan a Workflow for the given information using the give tool/agent."},
                {"role": "assistant", "content": "Ok, please show me you requirement information and the available tool/agent"},
                {"role": "user", "content": """The descriptions of the available tool/agent are as follows: 

    Retrieval (R): The input of R is a question, and the output of R is multiple relevant candidate documents to the given question.
    Document Selector (DS): The input of DS is a given question and multiple relevant candidate documents from search engine, and the output of DS is the documents which are helpful to answer the given question.
    Answer Generator (AG): The input of G may contains the given question, the some related candidate documents from search engine. The output of AG is the answer to the given question.

    With above tools/agents you can build a Workflow to answer a given question.


    The followings are some examples:

    For the easiest question, the Answer Generator (AG) can answer it directly without any query reformulation and external information:
    Question: "Which city is the capital of Australia."
    Workflow: AG
    Then, we can use Answer Generator (AG) to get the answer directly.

    For simple questions that may require external knowledge to answer:
    Question: "Which team is the 2025 NBA champion?"
    Workflow: R,AG
    We can just retrieve the external information about the given question through Retrieval Agent (RA), then the Answer Generator (AG) can output the answer to the given question.


    Next I'll give you the Question that need to be answered, just output the appropriate Workflow in the format above, don't output anything else.
    """},
                {"role": "assistant", "content": "Ok, you can provide the Question and I will give the appropriate Workflow to answer the Question."},
                {"role": "user", "content": """Please give the appropriate Workflow in the format requested above.
    Qustion: "{}"
    Workflow: """.format(question)},
            ]

        return messages

if __name__ == "__main__":

    model_name = "/root/paddlejob/workspace/env_run/verl/models_fund/Qwen/Qwen2.5-7B-Instruct"

    # 获取可用的GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    planning_agent = PlanningAgent(model, tokenizer)

    # question = '"A Japanese manga series based on a 16 year old high school student Ichitaka Seto, is written and illustrated by someone born in what year?"'
    questions = [
        # 简单问题（单跳）
        "What is the capital of France?",
        "Who wrote 'Romeo and Juliet'?",
        
        # 简单问题（多跳）
        "What is the capital of the country where the Great Wall is located?",
        "Who is the author of the book that inspired the movie 'Jurassic Park'?",
        
        # 复杂问题（多跳）
        "How did technological advancements during World War II lead to changes in international relations?",
        "What are the environmental and economic impacts of deforestation in the Amazon Rainforest?"
    ]

    for question in questions:
        context = {
            "original_query": question,
            "query": question,
            # "serial": True if "QueryDecompositionAgentSerial" in workflow else False,
            # "parallel_sub": parallel_sub,
            "current_step": -1,
            "answer": ""
        }
        messages = planning_agent.create_planning_messages(context)

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print('***********************************')
        print('question: {}'.format(question))
        print(response)
        # print('Workflow: {}'.format(response))
        print('\n\n')