import torch
from utils.data import process_Synth
from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = 'commonsenseqa'
model_type = 'Instruct'
gen_length = 64

D = process_Synth(split_no = 1, device = device, base_path = './datasets/commonsenseqa_Dream-v0-Instruct-7B_64_entropy/')

val = D['val']

X, times, y = val
Entropy_mean = X.mean(dim=(0, 2))

Entropy_mean_np = Entropy_mean.detach().cpu().numpy()
y_np = (1 - y).detach().cpu().numpy()

auc = roc_auc_score(y_np, Entropy_mean_np)
print('Val AUROC: {:.4f}'.format(auc))