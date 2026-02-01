import torch
import numpy as np
import torch.nn.functional as F
import re
import json
import argparse
import gc
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import sys
import os
import pdb

# Add the parent directory to the path to import data_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DLM_generate.data_loader import LOADER_FN, PROCESS_FN


def parse_args():
    parser = argparse.ArgumentParser(description="A script for evaluating hallucination for Dream-7B")
    parser.add_argument('--model', default='Dream-org/Dream-v0-Instruct-7B', 
                       choices=['Dream-org/Dream-v0-Instruct-7B', 'Dream-org/Dream-v0-Base-7B'])
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--steps", type=int, default=64, help="Diffusion steps")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum new tokens to generate")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name")
    parser.add_argument('--device', type=int, default=0, help="GPU ID to use")
    parser.add_argument("--task", type=str, default="test", help="Task split")
    parser.add_argument("--generate_task", type=str, default="step_output", help="Generation task")
    parser.add_argument("--alg", type=str, default="maskgit_plus", 
                       choices=["origin", "maskgit_plus", "topk_margin", "entropy"],
                       help="Remasking algorithm")
    parser.add_argument("--alg_temp", type=float, default=0., help="Algorithm temperature")
    parser.add_argument("--output_dir", type=str, default="step_output", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    return args


def extract_answer(text):
    """Extract answer from text enclosed in <answer></answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return text


class LogitsHook:
    """Hook to capture intermediate logits during diffusion generation."""
    
    def __init__(self, prompt_length):
        self.step_entropies = []
        self.step_outputs = []
        self.step_logits = []
        self.prompt_length = prompt_length
    
    def reset(self):
        """Reset captured data for new generation."""
        self.step_entropies = []
        self.step_outputs = []
        self.step_logits = []
    
    def __call__(self, step, tokens, logits):
        """
        Hook function called during generation.
        
        Args:
            logits: Logits tensor [batch_size, seq_len, vocab_size]
            step: Current diffusion step
            tokens: Current token sequence
        """
        # Calculate entropy for the generation part (excluding prompt)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        entropy = -(probs * log_probs).sum(dim=-1)  # [batch_size, seq_len]
        
        # Store entropy for this step, excluding prompt indices (following generate.py pattern)
        self.step_entropies.append(entropy[:, self.prompt_length:].detach().cpu())
        # Store current output tokens for this step
        self.step_outputs.append(tokens.detach().cpu())
        self.step_logits.append(logits.detach().cpu())
        return logits  # Return unchanged logits


@torch.no_grad()
def generate_with_entropy(model, tokenizer, prompt, args):
    """
    Generate text using Dream model while capturing entropy at each step.
    
    Args:
        model: Dream model
        tokenizer: Dream tokenizer
        prompt: Input prompt string
        args: Command line arguments
        
    Returns:
        raw_answer: Generated tokens
        step_outputs: List of token sequences at each step
        step_entropies: List of entropy tensors at each step
    """
    
    # Tokenize input
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = inputs.input_ids.to(device=f"cuda:{args.device}")
    attention_mask = inputs.attention_mask.to(device=f"cuda:{args.device}")
    
    # Create logits hook with prompt length
    prompt_length = input_ids.shape[1]
    logits_hook = LogitsHook(prompt_length)
    
    # Generate with diffusion
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        output_history=True,
        return_dict_in_generate=True,
        steps=args.steps,
        temperature=args.temperature,
        alg=args.alg,
        alg_temp=args.alg_temp,
        generation_logits_hook_func=logits_hook
    )
    
    # Extract the generated part (excluding input prompt)
    generated_tokens = output.sequences[0][len(input_ids[0]):]
    
    return generated_tokens, logits_hook.step_outputs, logits_hook.step_entropies, logits_hook.step_logits, output.history, prompt_length


def main():
    print(">>> dream_entropy.py started")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    args = parse_args()
    if args.model == "Dream-org/Dream-v0-Instruct-7B":
        model_name = "Dream-Instruct"
    else:
        model_name = "Dream-Base"

    
    # Set random seeds for reproducibility
    print(f"Setting random seed to: {args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    device = f"cuda:{args.device}"
    
    # Load Dream model and tokenizer
    print(f"Loading model: {args.model}")
    model = AutoModel.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, 
        trust_remote_code=True
    )

    dataset_name = args.dataset
    task = args.task
    model_name = args.model.split('/')[-1]  # Extract model name from path

    # Load data
    print(f"Loading dataset: {dataset_name}, task: {task}")
    data = LOADER_FN[dataset_name](task)
        
    result_list = []
    step_outputs_list = []
    entropy_list = []
    history_list = []
    mask_="<|mask|>"
    mask_replace_ = "[.]"
    eof_ = "<|endoftext|>"
    eof_replace_ = "[eof]"
    mask_token_id = 151666

    print(f"Processing {len(data)} examples...")
    for example in tqdm(data):
        # Extract question
        if 'Question' in example:
            question = example['Question']
        elif 'question' in example:
            question = example['question']
        else:
            raise KeyError("Missing 'question' field in example")
            
        # Process example to get label and prompt
        label, prompt = PROCESS_FN[dataset_name](example)
        
        # Create result dictionary
        result = {
            "question": question,
            "label": label,
            "answer": None,
            "is_hallucination": "Unknown"
        }
        
        
        # Generate with entropy tracking
        raw_answer, step_outputs, step_entropies, step_logits, historys, prompt_length = generate_with_entropy(
            model, tokenizer, prompt, args
        )
        
        # Decode the answer
        answer_text = tokenizer.decode(raw_answer.tolist(), skip_special_tokens=True)
        result['answer'] = extract_answer(answer_text)
        
        # Store step outputs and entropies
        step_outputs_list.append(torch.stack(step_outputs) if step_outputs else torch.empty(0))
        entropy_list.append(torch.stack(step_entropies) if step_entropies else torch.empty(0))
        if args.generate_task == "step_output":
            #step_outputs_decode_list = [tokenizer.decode(step_output.squeeze()[prompt_length:], skip_special_tokens=False) for step_output in step_outputs]
            step_logits_list = step_logits
            #softmax and decode the step_logits
            step_logits_decode_list = [tokenizer.decode(torch.argmax(step_logit, dim=-1).squeeze()[prompt_length:], skip_special_tokens=True) for step_logit in step_logits_list]
            step_unmask_decode_list = [tokenizer.decode(history.squeeze()[prompt_length:], skip_special_tokens=False) for history in historys]
            step_unmask_decode_list = [step_unmask_decode_list[i].replace(mask_, mask_replace_) for i in range(len(step_unmask_decode_list))]
            step_unmask_decode_list = [step_unmask_decode_list[i].replace(eof_, eof_replace_) for i in range(len(step_unmask_decode_list))]
            result['step_output'] = step_logits_decode_list
            #result['step_unmask'] = step_unmask_decode_list
                

        
        result_list.append(result)

    # Create output directory
    output_dir = f"./DLM_generate/{args.output_dir}/{dataset_name}_{model_name}_{args.generate_task}_{args.steps}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results based on generate_task
    if args.generate_task == "entropy":
        torch.save(entropy_list, f"{output_dir}/{dataset_name}_{task}.pt")
    elif args.generate_task == "step_outputs":
        pass
    else:
        torch.save(step_outputs_list, f"{output_dir}/{dataset_name}_{task}.pt")

    # Save result list
    with open(f"{output_dir}/{dataset_name}_{task}_resultlist.json", "w", encoding="utf-8") as f:
        json.dump(result_list, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_dir}")
    print(">>> dream_entropy.py completed")


if __name__ == '__main__': 
    main()
