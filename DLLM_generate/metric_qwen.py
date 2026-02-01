from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import json
import os
import re
from tqdm import tqdm

def load_qwen(model_path = '/scratch/local/ssd/yujunchi/qwen8b'):
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', trust_remote_code=True, cache_dir='/scratch/local/ssd/yujunchi/qwen8b')
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen3-8B',
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        cache_dir='/scratch/local/ssd/yujunchi/qwen8b'
    ).eval()
    return tokenizer, model

def get_prompt(sample):
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': ''}
    ]
    messages[-1]['content'] = '''Your task is to determine whether the output contains hallucination. Follow these guidelines strictly:

Fluency Check: If the output is not fluent natural language (e.g., it contains garbled or unreadable text), it should be considered hallucinated.

Relevance Check: If the output contains many correct facts but does not directly answer the question, it should be considered hallucinated.

Support Check: If the output cannot be inferred from any of the reference answers, or contains information inconsistent with the reference answers, it should be considered hallucinated.

Exact Match Rule: If the output is supported by any one of the reference correct answers, it should be considered not hallucinated.

Semantic Match Rule: If the output is not directly supported by any reference answer, but is semantically similar (i.e., expresses the same meaning), it should be considered not hallucinated.

Unknown Answer Rule: If the reference answers include phrases like "This question cannot be answered", then an output like "I don't know" or "Cannot answer this question" should be considered not hallucinated.'''

    messages.append({'role': 'assistant',
                     'content': 'I understand. Please provide the question and the bot\'s answer.'})
    messages.append({'role': 'user', 'content': ''})


    user_input_for_judging = 'Question:{}\n\n'.format(sample['question'].strip())
    user_input_for_judging += 'The correct anwer example is as follow:\n'
    if isinstance(sample['label'], str):
        user_input_for_judging += '{}\n'.format(sample['label'].strip())
    else:
        for example_answer in sample['label']:
            if isinstance(example_answer, str):
                user_input_for_judging += '{}\n'.format(example_answer.strip())
            elif isinstance(example_answer, list):
                user_input_for_judging += ', '.join([example_answer[0].strip()]) + '\n'
            else:
                raise ValueError("Unsupported answer format: {}".format(type(example_answer)))


    user_input_for_judging += '\nThe bot replied as follow:\n'
    user_input_for_judging += '{}\n\n'.format(sample['answer'].strip())
    user_input_for_judging += 'Now please judge whether the bot\'s answer is hallucinated or not. If it is hallucinated, please answer "yes", otherwise answer "no".  Dont show thinking and put your answer in <answer> </answer>.\n'

    messages[-1]['content'] = user_input_for_judging

    return messages


def compute_correctness_truthfulqa(answer_path, model, tokenizer):

    with open(answer_path, "r") as f:
        results = json.load(f)

    correctness = []
    for index, sample in enumerate(results):
        messages = get_prompt(sample)
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        if not messages or not isinstance(messages, list):
            print(f"Skipping index {index}: invalid message format.")
            correctness.append(0)
            continue

        with torch.no_grad():
            output_ids = model.generate(**inputs, do_sample=False)
        output_text = extract_answer(tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower())
        if output_text == "no":
            sample['is_hallucination'] = output_text
        else:
            sample['is_hallucination'] = "yes"
        results[index] = sample
        
        if "yes" in output_text:
            correctness.append(0)  # hallucinated
        elif "no" in output_text:
            correctness.append(1)  # not hallucinated
        else:
            correctness.append(0)  # treat unclear answers as hallucinated

        torch.cuda.empty_cache()

    
    eval_dir = os.path.abspath(os.path.join(os.path.dirname(answer_path), "..", "eval"))
    os.makedirs(eval_dir, exist_ok=True)  

    eval_filename = os.path.basename(answer_path)
    eval_path = os.path.join(eval_dir, eval_filename)

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return correctness

def compute_correctness_sciqa(answer_path):
    with open(answer_path, "r") as f:
        results = json.load(f)

    correctness = []
    for index, sample in enumerate(results):
        label_list = sample['label']
        answer = sample['answer']

        num_label = str(label_list[0])

        if num_label in answer or label_list[1].lower() in answer.lower():
            sample['is_hallucination'] = "no"
            correctness.append(1)
        else:
            sample['is_hallucination'] = "yes"
            correctness.append(0)

        results[index] = sample

    eval_dir = os.path.abspath(os.path.join(os.path.dirname(answer_path), "..", "eval"))
    os.makedirs(eval_dir, exist_ok=True)

    eval_filename = os.path.basename(answer_path)
    eval_path = os.path.join(eval_dir, eval_filename)

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return correctness

def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return text

if __name__ == "__main__":
    model_path = "/scratch/local/ssd/yujunchi/qwen8b"
    tokenizer, model = load_qwen(model_path)

    input_dir = "./results"
    dataset = "sciqa"
    generate_task = "emb"
    for filename in tqdm(os.listdir(input_dir)):
        answer_path = os.path.join(input_dir, filename)
        output_file = filename.replace("_resultlist.json", "_labellist.pt")
        output_path = f"./DLM_generate/process_data/triviaqa_Instruct_64_entropy/"
        
        print(f"Evaluating {filename}...")
        
        if "sciqa" in filename or "commonsenseqa" in filename:
            correctness = compute_correctness_sciqa(answer_path)
        else:
            correctness = compute_correctness_truthfulqa(answer_path, model, tokenizer)
            
        correctness = torch.tensor(correctness)
        
        torch.save(correctness, output_path + output_file)
