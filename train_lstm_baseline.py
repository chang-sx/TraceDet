import torch

from utils.predictors.loss import Poly1CrossEntropyLoss
from trainers.train_transformer import train
from models.encoders.simple import LSTM
from utils.data import process_Synth
from utils.predictors import eval_mvts_transformer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 2,
    epsilon = 1.0,
    weight = None,
    reduction = 'mean'
)

D = process_Synth(split_no = 1, device = device, base_path = './datasets/truthfulqa/')
train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 64, shuffle = True)

val, test = D['val'], D['test']

model = LSTM(
    d_inp = val[0].shape[-1],
    n_classes = 2,
)

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 0.1)

spath = 'models/Scomb_transformer_split=1.pt'

model, loss, auc = train(
    model,
    train_loader,
    val_tuple = val, 
    n_classes = 2,
    num_epochs = 100,
    save_path = spath,
    optimizer = optimizer,
    show_sizes = False,
    use_scheduler = False,
)

model_sdict_cpu = {k:v.cpu() for k, v in  model.state_dict().items()}
torch.save(model_sdict_cpu, spath)

f1,auc = eval_mvts_transformer(test, model, auroc=True)
print('Test AUROC: {:.4f}'.format(auc))
print('Test F1: {:.4f}'.format(f1))
