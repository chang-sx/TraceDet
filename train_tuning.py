import argparse
import itertools
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from utils.predictors.loss import Poly1CrossEntropyLoss
from trainers.train_TimeHalu import train_TimeHalu
from models.TimeHalu import TimeHalu, MaskGenerator
from models.encoders.transformer_simple import TransformerMVTS
from utils.data import process_Synth
from utils.predictors import eval_mvts_transformer
from sklearn.metrics import roc_auc_score
from utils.predictors import eval_mvts_transformer
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_repeats', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./tuning')
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--task", type=str, default="entropy")
    parser.add_argument("--model", type=str, default="Instruct")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument('--step_length', type=int, default=2)
    parser.add_argument('--rem_strat', type=int, default=10)

    return parser.parse_args()


def train_one_run(params, device, args):

    

    D = process_Synth(split_no = 1, device=device, base_path=f'./datasets/{args.dataset}_{args.model}_{args.gen_length}_{args.task}/')
    train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size=params['batch_size'], shuffle=True)
    val, test = D['val'], D['test']
    if args.gen_length == 64:
        feed_dim =8
    else:
        feed_dim =16
    
    clf = TransformerMVTS(
        d_inp=val[0].shape[-1],
        max_len=val[0].shape[0],
        n_classes=2,
        nlayers=params['nlayers'],
        trans_dim_feedforward=feed_dim,
        trans_dropout=params['dropout_rate'],
        d_pe=feed_dim,
        aggreg='last_token',
    )
    extractor = MaskGenerator(d_z=(val[0].shape[-1]+feed_dim), d_pe=feed_dim, max_len=val[0].shape[0], tau=1.0, use_ste=True)
    model = TimeHalu(clf, extractor,
                      loss_weight_dict={'gsat':params['gsat'], 'connect':0.05},
                      task=args.task, gsat_r=params['gsat_r'],
                      d_inp=val[0].shape[-1],
                      d_pe= feed_dim).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=0.1)
    
    trained_model = train_TimeHalu(
        model=model,
        train_loader=train_loader,
        val_tuple=val,
        test_tuple=test,
        num_epochs=100,
        save_path=f"./{args.dataset}_{args.model}_{args.gen_length}_{args.task}_temp.pt",
        optimizer=optimizer,
        early_stopping=True,
        use_scheduler=True,
        patience=20
    )
    
    #---test---
    model.load_state_dict(torch.load(f"./{args.dataset}_{args.model}_{args.gen_length}_{args.task}_temp.pt"))
    model.eval()
    with torch.no_grad():
        X, times, y = test
        out = model(X, times, y, captum_input=False)
        y_true = y.cpu().numpy()
        y_prob = out['y_pred_masked'].detach().cpu().numpy()
        test_auc = roc_auc_score(y_true, y_prob[:,1])
    torch.cuda.empty_cache()
    return test_auc





if __name__ == "__main__":
    args = parse_args()
    print("function starts")
    print(f'./datasets/{args.dataset}_{args.model}_{args.gen_length}_{args.task}/')
    
    
    param_space = {
        "lr": np.logspace(-5, -3, 8).tolist(),          
        "batch_size": [8, 64],               
        "dropout_rate": np.linspace(0, 0.4, 5).tolist(), 
        "nlayers": [2, 3, 4],                        
        "gsat": np.linspace(0.0, 2, 6).tolist(),     
        "gsat_r": np.linspace(0.1, 0.2, 2).tolist()  
    }
    keys, values = zip(*param_space.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    Path(args.save_dir).mkdir(exist_ok=True)
    results_file = Path(args.save_dir) / f"{args.dataset}_{args.model}_{args.gen_length}_results.json"
    
    results = []
    for i, params in tqdm(enumerate(all_combinations), total=len(all_combinations)):

        print(f"=== Hyperparam Set {i+1}/{len(all_combinations)} ===")
        metrics = []
        for r in range(args.num_repeats):

            auc = train_one_run(params, device="cuda", args=args)
            metrics.append(auc)
            print(f"  Repeat {r+1}: AUC = {auc:.4f}")

        
        if metrics:
            mean_auc = float(np.mean(metrics))
            std_auc = float(np.std(metrics))
        else:
            mean_auc, std_auc = None, None
        
        result_entry = {
            "params": params,
            "metrics": metrics,
            "mean_auc": mean_auc,
            "std_auc": std_auc
        }
        results.append(result_entry)
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
