import torch
import json
import random
import re
import pandas as pd
import os

from tqdm import tqdm
from datasets import load_dataset

random.seed(42)
# #Math Loader
# def load_math_data():
#     file_path = "./datasets/Math/AnswerableMath_test.csv"
#     df = pd.read_csv(file_path)
#     return df.to_dict(orient="records")

# def math_fn(example):
#     label = eval(example['answer'])[0]
#     prompt = f"Q:{example['question']},A:''"
#     return label, prompt


#Triviaqa loader
def load_triviaqa_data(split="train"):
    if split == "train":
        file_path = os.path.expanduser(f'~/ShenxuC/LLaDA-hallu/datasets/TriviaQA/qa/wikipedia-{split}.json')
    else:
        file_path = os.path.expanduser(f'~/ShenxuC/LLaDA-hallu/datasets/TriviaQA/qa/wikipedia-dev.json')  

    with open(file_path, "r") as f:
        dataset = json.load(f)['Data']

    
    if split == "train":
        train_index = random.sample(range(len(dataset)), 2000)
        dataset = [dataset[i] for i in train_index]
    else:
        dev_index = random.sample(range(len(dataset)), 400)
        random.shuffle(dev_index)
        if split == "val":
            dataset = [dataset[i] for i in dev_index[:200]]
        else:
            dataset = [dataset[i] for i in dev_index[200:]]

    return dataset

def triviaqa_fn(example):
    label = example["Answer"]["Aliases"]
    prompt = f"Answer the question concisely. Question:{example['Question']} \n And please put your final answer in <answer> </answer>"
    return label, prompt


#hotpotqa loader
def load_hotpotqa_data(split="train"):
    dataset = load_dataset('hotpot_qa', 'distractor', cache_dir='~/ShenxuC/LLaDA-hallu/datasets/hotpotqa')

    if split == "train":
        train_data = dataset["train"]

        num_samples = 1300
        total_samples = len(train_data)
        random_indices = random.sample(range(total_samples), num_samples)
        dataset = train_data.select(random_indices)
    else:
        dev_data = dataset["validation"]
        dev_index = random.sample(range(len(dev_data)), 400)
        random.shuffle(dev_index)
        if split == "val":
            dataset = dev_data.select(dev_index[:200])
        else:
            dataset = dev_data.select(dev_index[200:])
    return dataset

def hotpotqa_fn(example):
    label = example["answer"]
    
    titles = example["context"]["title"][:1]
    all_sentences = example["context"]["sentences"][:1]
    
    context_list = []
    for t, s_list in zip(titles, all_sentences):
        context_list.append(f"[{t}]: {' '.join(s_list)}")
    
    context = "\n".join(context_list)
    prompt = (
        f"You are given the following context\n{context}\n\n"
        f"Question: {example['question']}\n"
        "Answer the question based on the context only. "
        "Please put your final answer in <answer> </answer>"
    )
    
    return label, prompt

# #halluqa loader
# def load_halluqa_data():
#     file_path = "./datasets/halluqa/HalluQA.json"
#     with open(file_path, "r") as f:
#         return json.load(f)

# def halluqa_fn(example):
#     label = example["answer"]
#     prompt = f"Answer the question concisely. Question:{example['Question']} \n and please put your answer in <answer> </answer>"
#     return label, prompt



#commonsenseqa_loader
def load_commonsenseqa_data(split="train"):
    dataset = load_dataset('commonsense_qa', cache_dir='./datasets/commonsenseqa')

    if split == "train":
        train_data = dataset["train"]

        num_samples = 2000
        total_samples = len(train_data)
        random_indices = random.sample(range(total_samples), num_samples)
        dataset = train_data.select(random_indices)
    else:
        dev_data = dataset["validation"]
        dev_index = random.sample(range(len(dev_data)), 400)
        random.shuffle(dev_index)
        if split == "val":
            dataset = dev_data.select(dev_index[:200])
        else:
            dataset = dev_data.select(dev_index[200:])

    return dataset

def commonsenseqa_fn(example):
    label_key = example["answerKey"]
    question = example["question"]
    options = example["choices"]["text"]
    labels = example["choices"]["label"]  
    answer_index = labels.index(label_key)  

    answer_text = options[answer_index]
    label = [label_key, answer_text]
    labeled_options = [f"{l}. {t}" for l, t in zip(labels, options)]
    options_text = "\n".join(labeled_options)

    prompt = (
        f"Question: {question}\n"
        f"Options:\n{options_text}\n\n"
        "Instruction:\n"
        "- Select exactly ONE correct option (A, B, C, D, E).\n"
        "- DO NOT generate explanations.\n"
        "- Output format MUST be: <answer>X</answer>, where X ∈ {A,B,C,D,E}.\n"
        "- Any other output will be considered invalid.\n\n"
        "Your output:"
    )

    return label, prompt



#Medqa loader
def load_medqa_data(split="dev"):
    file_path = os.path.expanduser(f"~/ShenxuC/LLaDA-hallu/datasets/MedQA/{split}.jsonl")
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f] 
    return data
    
def medqa_fn(example):
    question = example["question"]
    label = example["answer"]
    answer_index = example["answer_index"]
    choices = example["options"]
    label = [answer_index, label]
    options_str = "\n".join(f"{choice}" for i, choice in enumerate(choices))

    prompt = (
            "You are a strict answer machine for multiple-choice questions."
        f"Question: {question}\n"
        f"Options:\n{options_str}\n\n"
        "Select the correct option and respond with your answer inside <answer></answer> tags.\n"
        "Do not add explanations."
    )
    return label, prompt

# --- Local Project Data Loaders ---

def load_local_json(path):
    with open(path, "r") as f:
        return json.load(f)

def local_math_fn(example):
    # Format: {"question": "...", "label": "0/1", "answer": "..."}
    # In Arshia datasets, 'answer' is the Ground Truth string.
    prompt = f"Question: {example['question']}\nAnswer: "
    return example['answer'], prompt

def local_csqa_fn(example):
    # Format: {"question": "...", "label": "0/1", "answer": "..."}
    # Arshia CSQA includes options in the question string usually or we can format simply.
    prompt = f"Question: {example['question']}\nAnswer: "
    return example['answer'], prompt

def local_hotpotqa_fn(example):
    prompt = f"Question: {example['question']}\nAnswer: "
    return example['answer'], prompt

def local_triviaqa_fn(example):
    prompt = f"Question: {example['question']}\nAnswer: "
    return example['answer'], prompt

def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return text

#ELI5 loader
def load_eli5_data(split="train"):
    dataset = load_dataset('hotpot_qa', 'distractor', cache_dir='~/ShenxuC/LLaDA-hallu/datasets/hotpotqa')

    total_samples = len(dataset["train"])
    total_samples = 2000
    random_indices = random.sample(range(total_samples), total_samples)
    random.shuffle(random_indices)
    
    if split == "train":
        dataset = dataset.select(random_indices[1400:])  
    elif split == "val":
        dataset = dataset.select(random_indices[1400:1700])  
    else:
        dataset = dataset.select(random_indices[1700:])  

    return dataset

def elis_fn(example):
    label = example["answer"]
    prompt = (
        "Explain the following question in simple terms, "
        "as if you were talking to a five-year-old. "
        "Try to answer in three to four sentences.\n\n"
        f"Question: {example['question']}"
    )
    return label, prompt




PROCESS_FN = {
    'triviaqa': triviaqa_fn,
    'hotpotqa':hotpotqa_fn,
    'sciqa': sciqa_fn,
    'medqa': medqa_fn,
    'commonsenseqa': commonsenseqa_fn,

}

LOADER_FN = {
    'triviaqa': load_triviaqa_data,
    "hotpotqa": load_hotpotqa_data,
    'sciqa': load_sciqa_data,
    'medqa': load_medqa_data,
    'commonsenseqa': load_commonsenseqa_data,
}
