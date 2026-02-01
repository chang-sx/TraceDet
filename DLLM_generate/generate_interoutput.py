import torch
import numpy as np
import torch.nn.functional as F
import re

from DLM_generate.data_loader import LOADER_FN, PROCESS_FN
import argparse
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import gc
import json




def pars_args():
    parser = argparse.ArgumentParser(description="A scrip for evaluating hallucination for LLada-8B")
    parser.add_argument('--model', default='Instruct', choices=['Instruct','Base'])
    parser.add_argument("--temperature", type=float, default=0., help="Sampling temperature for generation")
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"], help="Remasking strategy")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=128)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--cfg_scale', type=float, default=1.0, help="Classifier-free guidance scale")
    parser.add_argument('--device', type=int, default=0, help="GPU ID to use")
    parser.add_argument("--task", type=str, default = "train")
    parser.add_argument("--generate_task", type=str, default="emb", help="Generation task")
    # parser.add_argument("--output_index", type=int, default=128, help="Index of the output to return during generation")


    args = parser.parse_args()
    return args



def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise # This is not the exact gumble noise formula


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step
    this numbers are almost equally sliced.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, tokenizer, prompt, result_dict, gen_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
    '''
    block_length = gen_length
    steps = gen_length
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()  # replace the first prompt_length index to be prompt

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    outputlist = []
    outputentropy = []

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                # makes the model more consistent with prompt conditixons
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            intermediate_answer = tokenizer.batch_decode(x0[:, prompt.shape[1]:], skip_special_tokens=True)[0]
            result_dict['step_output'].append(intermediate_answer)

            probs = F.softmax(logits, dim=-1)  # [B, L, V]
            log_probs = torch.log(probs + 1e-9)

            entropy = -(probs * log_probs).sum(dim=-1)  # [B, L]


            outputentropy.append(entropy[:, prompt.shape[1]:])


            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                # assign the softmax confidence of each prediction
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            # setting the operated blocks and prompt to be -np.inf

            x0 = torch.where(mask_index, x0, x)
            outputlist.append(x0[:, prompt.shape[1]:])
            # outputlogits.append(max_logits[:, prompt.shape[1]:])
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    final_output = torch.cat(outputlist, dim=0)
    # outputlogits = torch.cat(outputlogits, dim=0)
    outputentropy = torch.cat(outputentropy, dim = 0)
    return x, final_output, outputentropy


def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return text
    

def main():
    print(">>> entropy.py started")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    args = pars_args()
    device = f"cuda:{args.device}"
    model = AutoModel.from_pretrained(f'GSAI-ML/LLaDA-8B-{args.model}', trust_remote_code=True,
                                               torch_dtype=torch.bfloat16, cache_dir=f'/scratch/local/ssd/yujunchi/llada_model_{args.model}').to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(f'GSAI-ML/LLaDA-8B-{args.model}', trust_remote_code=True, cache_dir=f'/scratch/local/ssd/yujunchi/llada_model_{args.model}')

    remasking = args.remasking
    gen_length = args.gen_length
    steps = args.steps
    cfg_scale = args.cfg_scale
    dataset_name = args.dataset
    task = args.task
    model_name = args.model

    data = LOADER_FN[dataset_name](task)
        
    result_list = []
    emb_list = []
    entropy_list = []


    for example in tqdm(data):
        if 'Question' in example:
            question = example['Question']
        elif 'question' in example:
            question = example['question']
        else:
            raise KeyError("Missing 'question' field in example")
        label, prompt = PROCESS_FN[dataset_name](example)
        result = {
                "question": question,
                "label": label,
                "answer": None,
                "is_hallucination": "Unknown",
                "step_output":[]
            }
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)

        raw_answer, step_list, step_max_logits = generate(model, tokenizer, input_ids, result, temperature=args.temperature, remasking=remasking, gen_length=gen_length, cfg_scale=cfg_scale)
        result['answer'] = extract_answer(tokenizer.batch_decode(raw_answer[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
        emb_list.append(step_list.unsqueeze(0))
        entropy_list.append(step_max_logits.unsqueeze(0))

        result_list.append(result)
    emb_list = torch.cat(emb_list, dim=0)
    entropy_list = torch.cat(entropy_list, dim=0)
    if args.generate_task == "entropy":
        torch.save(entropy_list, f"./DLM_generate/process_data/{dataset_name}_{model_name}_{args.gen_length}_entropy/{dataset_name}_{task}.pt")
    else:
        torch.save(emb_list, f"./DLM_generate/process_data/{dataset_name}_{model_name}_{args.gen_length}_emb/{dataset_name}_{task}.pt")

    with open(f"./DLM_generate/process_data/{dataset_name}_{model_name}_{args.gen_length}_{args.generate_task}/{dataset_name}_{task}_resultlist.json", "w", encoding="utf-8") as f:
        json.dump(result_list, f, indent=2, ensure_ascii=False)
        
        


if __name__ == '__main__': 
    main()

    





