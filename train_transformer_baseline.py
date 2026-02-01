import torch

from utils.predictors.loss import Poly1CrossEntropyLoss
from trainers.train_transformer import train
from models.encoders.transformer_simple import TransformerMVTS
from utils.data import process_Synth
from utils.predictors import eval_mvts_transformer
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()

D = process_Synth(split_no = 1, device = device, base_path = './datasets/commonsenseqa_Instruct_64_entropy')
train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 128, shuffle = True)


val, test = D['val'], D['test']

model = TransformerMVTS(
    d_inp = val[0].shape[-1],
    max_len = val[0].shape[0],
    n_classes = 2,
    nlayers = 5,
    trans_dim_feedforward = 16,
    trans_dropout = 0.0,
    d_pe =16,
    aggreg = 'mean',
    # norm_embedding = True
)


model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-4, weight_decay = 0.01)

spath = 'models/Scomb_transformer_split=1.pt'

model, loss, auc = train(
    model,
    train_loader,
    val_tuple = val, 
    n_classes = 2,
    num_epochs = 50,
    save_path = spath,
    optimizer = optimizer,
    show_sizes = False,
    use_scheduler = False,
)


f1,auc = eval_mvts_transformer(test, model, auroc=True)
print('Test AUROC: {:.4f}'.format(auc))
print('Test F1: {:.4f}'.format(f1))
