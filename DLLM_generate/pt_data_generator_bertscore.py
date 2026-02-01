import torch
import numpy as np
import random
import json
from types import SimpleNamespace
from utils.data.preprocess import RWDataset

set_seed = 42
random.seed(set_seed)

dataset_name = 'triviaqa'
task = "emb"
data_path = f'./DLM_generate/process_data/{dataset_name}_{task}/'

#----train_data------
X_train = torch.load(data_path + f'{dataset_name}_train.pt')
print(f"Data shape: {X_train.shape}")
X_train = X_train.transpose(0, 1)  # [time_length, num_data, feature_dim]
gts = np.load(data_path + 'ml_triviaqa_train_bleurt_score.npy') 
Y_train = torch.tensor(gts > 0.5, dtype=torch.long)
print(f"Label length: {len(Y_train)}")

assert X_train.shape[1] == len(Y_train)


time_step = X_train.shape[0]
num_sample = X_train.shape[1]
T_train = torch.arange(1, time_step + 1).unsqueeze(1).repeat(1, num_sample)
print(f"Time shape: {T_train.shape}")


#----val data------
X_val = torch.load(data_path + f'{dataset_name}_val.pt')
print(f"Data shape: {X_val.shape}")
X_val = X_val.transpose(0, 1)  # [time_length, num_data, feature_dim]
gts = np.load(data_path + 'ml_triviaqa_val_bleurt_score.npy') 
print(len(gts))
Y_val = torch.tensor(gts > 0.5, dtype=torch.long)
print(f"Label length: {len(Y_val)}")

assert X_val.shape[1] == len(Y_val)


time_step = X_val.shape[0]
num_sample = X_val.shape[1]
T_val = torch.arange(1, time_step + 1).unsqueeze(1).repeat(1, num_sample)
print(f"Time shape: {T_val.shape}")

#----test data------
X_test = torch.load(data_path + f'{dataset_name}_test.pt')
print(f"Data shape: {X_test.shape}")
X_test = X_test.transpose(0, 1)  # [time_length, num_data, feature_dim]
gts = np.load(data_path + 'ml_triviaqa_test_bleurt_score.npy') 
Y_test = torch.tensor(gts > 0.5, dtype=torch.long)
print(f"Label length: {len(Y_test)}")

assert X_test.shape[1] == len(Y_test)


time_step = X_test.shape[0]
num_sample = X_test.shape[1]
T_test = torch.arange(1, time_step + 1).unsqueeze(1).repeat(1, num_sample)
print(f"Time shape: {T_val.shape}")




print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")



D = {
    'train_loader': RWDataset(X_train, T_train, Y_train),
    'val': (X_val, T_val, Y_val),
    'test': (X_test, T_test, Y_test),
    'gt_exps': None  
}

torch.save(D, data_path + f'{dataset_name}_split.pt')