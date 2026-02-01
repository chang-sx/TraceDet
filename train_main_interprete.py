import torch
import json
# import parser
from utils.predictors.loss import Poly1CrossEntropyLoss
from trainers.train_TimeHalu import train_TimeHalu
from models.TimeHalu import TimeHalu, MaskGenerator
from models.encoders.transformer_simple import TransformerMVTS
from utils.data import process_Synth
from utils.predictors import eval_mvts_transformer
from sklearn.metrics import roc_auc_score
from utils.predictors import eval_mvts_transformer
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Training Arguments for TimeHalu Model')
    
    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    # Model architecture parameters
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--nlayers', type=int, default=3, help='Number of model layers')
    
    # GSAT-related parameters
    parser.add_argument('--gsat', type=float, default=0.1, help='GSAT main parameter')
    parser.add_argument('--gsat_r', type=float, default=0.1, help='GSAT r parameter')
    parser.add_argument('--connect', type=float, default=0.0, help='Connection parameter')
    
    # Dataset and task parameters
    parser.add_argument('--dataset', type=str, default='triviaqa', 
                        choices=['triviaqa', 'truthfulqa', 'sciqa'], help='Dataset name')
    parser.add_argument('--task', type=str, default='entropy', 
                        choices=['entropy', 'emb'], help='Task type')
    
    # Save path
    parser.add_argument('--save_path', type=str, default='./timehalu_best.pt', 
                        help='Path to save the best model')
    
    # Additional useful arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', 
                        choices=['cuda', 'cpu'], help='Device to use for training')
    parser.add_argument('--log_interval', type=int, default=10, 
                        help='How many batches to wait before logging training status')
    
    return parser.parse_args()

def main():
    num_epochs = 100
    batch_size = 8
    lr = 1e-5
    gsat_r = 0.1
    save_path = "./timehalu_best.pt"
    num_classes = 2
    patience = 20
    gsat = 0.4
    connect = 0.0
    nlayers = 4
    dataset = "triviaqa"
    task = "entropy"
    rem_strategy = "entropy"
    model = "Instruct"
    gen_length = 64
    dropout_rate = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    D = process_Synth(split_no = 1, device = device, base_path = f'./datasets/{dataset}_{model}_{gen_length}_{task}/')
    train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = batch_size, shuffle = True)

    val, test = D['val'], D['test']
    mu = D['train_loader'].X.mean(dim=1)
    std = D['train_loader'].X.std(unbiased = True, dim = 1)

    clf = TransformerMVTS(
                d_inp = val[0].shape[-1],
                max_len = val[0].shape[0], 
                n_classes = num_classes, 
                nlayers = nlayers,
                trans_dim_feedforward = 8,
                trans_dropout = dropout_rate,
                d_pe = 8,
                aggreg = 'last_token',
            )
    extractor = MaskGenerator(d_z = (val[0].shape[-1] + 8), d_pe = 8, max_len = val[0].shape[0], tau = 1.0, use_ste = True)

    loss_weight_dict = {
                    'gsat': gsat,
                    'connect': connect
                }

    model = TimeHalu(clf, 
                    extractor, 
                    loss_weight_dict, 
                    task = task,
                    gsat_r = gsat_r, 
                    masktoken_stats = (mu, std),
                    d_inp = val[0].shape[-1],
                    d_pe = 8,
                ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 0.1)

    trained_model = train_TimeHalu(
        model=model,
        train_loader=train_loader,
        val_tuple=val,
        test_tuple=test,
        num_epochs=num_epochs,
        save_path=save_path,
        optimizer=optimizer,
        early_stopping=True,
        use_scheduler=True,
        patience=patience
    )


    model.load_state_dict(torch.load(save_path))
    model.to(device)
    model.eval()
    with torch.no_grad():
        with open("./DLM_generate/process_data/triviaqa_Instruct_64_entropy/triviaqa_test_resultlist.json") as f:
            results = json.load(f)
        X, times, y = test
        out = trained_model(X, times, y, captum_input=False)
        pred = out['y_pred_masked']
        # mask = out["mask_logits"]
        mask = out["mask_reparam"]
        for i in range(mask.shape[0]):
            results[i]['mask'] = mask[i].cpu().numpy().tolist()
        for i in range(len(results)):
            results[i]['remained'] = []
            for j in range(len(results[i]['mask'])):
                if results[i]['mask'][j] == 1:
                    results[i]['remained'].append(results[i]["step_output"][j])
        y_true = y.cpu().numpy()
        y_prob = pred.detach().cpu().numpy()
        test_auc = roc_auc_score(y_true, y_prob[:,1])
        with open(f"triviaqa_test_resultlist_mask.json", 'w') as f:
            json.dump(results, f, indent=2)
    print('Test AUROC: {:.4f}'.format(test_auc))
    
            
if __name__ == "__main__":
    main()

