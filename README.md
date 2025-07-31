# MAO-ARAG: Multi-Agent Orchestration for Adaptive Retrieval-Augmented Generation

**MAO-ARAG** is a multi-agent orchestration framework for adaptive Retrieval-Augmented Generation (RAG) in question-answering systems. It dynamically selects and integrates different RAG modules based on query complexity, balancing answer quality, cost, and latency.

---

## 🧠 Overview

Traditional RAG systems struggle to serve all types of queries efficiently, as they target either low-complexity or high-complexity questions. **MAO-ARAG** introduces a multi-turn, agent-based architecture equipped with a planner agent and multiple executor agents (e.g., reformulators, retrievers, generators). The planner learns to construct optimal workflows per query via reinforcement learning, maximizing answer quality while minimizing cost.

---

## 📋 Table of Contents

1. [Computational Resource Requirements](#computational-resource-requirements)
2. [Download and Process Data](#download-and-process-data)
3. [Deploy Retriever](#deploy-retriever)
4. [Training](#training)

---

## ⚙️ Computational Resource Requirements

We used 6\*A800 for training of MAO-ARAG and deployed retreiver with the help of 1\*A800 to accelerate retrieval.

---

## 📦 Download and Process Data

MAO-ARAG supports multiple QA datasets from the Hugging Face Hub. All datasets can be loaded using the `datasets` library and processed into a unified format for downstream tasks.

##### Supported Datasets

The following datasets are used in our framework:

| Dataset Name | Hugging Face Identifier            |
| ------------ | ---------------------------------- |
| NQ           | `google-research-datasets/nq_open` |
| PopQA        | `akariasai/PopQA`                  |
| AmbigQA      | `sewon/ambig_qa`                   |
| HotpotQA     | `hotpotqa/hotpot_qa`               |
| 2Wiki        | `voidful/2WikiMultihopQA`          |
| Musique      | `bdsaglam/musique`                 |
| Bamboogle    | `chiayewken/bamboogle`             |

To download a dataset, use the following code snippet:

```python
from datasets import load_dataset

data_source = "<dataset_identifier>"  # e.g., "google-research-datasets/nq_open"
dataset = load_dataset(data_source)
```

##### Extract Questions and Answers

After downloading the raw dataset, process it into a list of dictionaries. Each dictionary should have the format:

```json
{
    "question": "<question text>",
    "answer": "<answer text>"
}
```

Save this list to the following path:

```python
data/{dataset_name}/{dataset_name}__train_questions_and_answers.json
data/{dataset_name}/{dataset_name}__test_questions_and_answers.json
```

##### Generate Parquet-formatted Data

Once the JSON files are created, run the corresponding dataset processing script to generate the final `.parquet` files:

```bash
python data/{dataset_name}.py
```

---

## 🔍 Deploy Retriever

Firstly, you should getting index for the corpus. You should have a corpus, a dense retrieval model, and run index.py in ./retriever:

```bash
CUDA_VISIBLE_DEVICES=0 python index.py
```

Then, run the run_server.sh in ./qa_manager to deploy the retreiver:

```bash
bash run_server.sh
```


---

## 🏋️ Training

Run the run_ppo.sh to start the train loop of MAO-ARAG.

```bash
bash run_ppo.sh
```
