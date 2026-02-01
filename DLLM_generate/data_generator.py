import torch
import random
import json
from types import SimpleNamespace
from utils.data.preprocess import RWDataset
import os

set_seed = 42
random.seed(set_seed)

dataset_name = ["commonsenseqa"]
task_name = 'entropy'
model_name = "Dream-v0-Instruct-7B"
rem_strategy = ["random"]
gen_lengths = [16,32]

for dataset_name in dataset_name:
     for rem in rem_strategy:

        data_path = f'./DLM_generate/process_data/{dataset_name}_{model_name}_{rem}_{task_name}/'

        if not os.path.exists(f'./datasets/{dataset_name}_{model_name}_{rem}_{task_name}/'):
                    os.mkdir(f'./datasets/{dataset_name}_{model_name}_{rem}_{task_name}/')

        #----train----
        X_train = torch.load(data_path + f'{dataset_name}_train.pt')
        if isinstance(X_train, list):
            X_train = torch.stack([torch.tensor(x) for x in X_train]).squeeze()
        print(f"Data shape: {X_train.shape}")
        X_train = X_train.transpose(0, 1)  # [time_length, num_data, feature_dim]
        Y_train = torch.load(data_path + f'{dataset_name}_train_labellist.pt')
        if isinstance(Y_train, list):
            Y_train = torch.tensor(Y_train)



        time_step = X_train.shape[0]
        num_sample = X_train.shape[1]
        T_train = torch.arange(1, time_step + 1).unsqueeze(1).repeat(1, num_sample)
        print(f"Time shape: {T_train.shape}")

        #----val-----
        X_val = torch.load(data_path + f'{dataset_name}_val.pt')
        if isinstance(X_val, list):
            X_val = torch.stack([torch.tensor(x) for x in X_val]).squeeze()
        print(f"Data shape: {X_train.shape}")
        X_val = X_val.transpose(0, 1)  # [time_length, num_data, feature_dim]
        Y_val = torch.load(data_path + f'{dataset_name}_val_labellist.pt')
        if isinstance(Y_val, list):
            Y_val = torch.tensor(Y_val)
        print(f"Label length: {len(Y_val)}")

        assert X_val.shape[1] == len(Y_val)


        time_step = X_val.shape[0]
        num_sample = X_val.shape[1]
        T_val = torch.arange(1, time_step + 1).unsqueeze(1).repeat(1, num_sample)
        print(f"Time shape: {T_val.shape}")

        #----test----

        X_test = torch.load(data_path + f'{dataset_name}_test.pt')
        if isinstance(X_test, list):
            X_test = torch.stack([torch.tensor(x) for x in X_test]).squeeze()
        print(f"Data shape: {X_test.shape}")
        X_test = X_test.transpose(0, 1)  # [time_length, num_data, feature_dim]
        Y_test = torch.load(data_path + f'{dataset_name}_test_labellist.pt')
        if isinstance(Y_test, list):
            Y_test = torch.tensor(Y_test)
        print(f"Label length: {len(Y_test)}")

        assert X_test.shape[1] == len(Y_test)


        time_step = X_test.shape[0]
        num_sample = X_test.shape[1]
        T_test = torch.arange(1, time_step + 1).unsqueeze(1).repeat(1, num_sample)
        print(f"Time shape: {T_test.shape}")


        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")



        # train_flat = X_train.reshape(-1, X_train.shape[2]).float()  
        # mean = train_flat.mean(dim=0, keepdim=True)        
        # std  = train_flat.std(dim=0, keepdim=True) + 1e-6  


        # X_train = (X_train.float() - mean) / std
        # X_val   = (X_val.float()   - mean) / std
        # X_test  = (X_test.float()  - mean) / std


        D = {
            'train_loader': RWDataset(X_train, T_train, Y_train),
            'val': (X_val, T_val, Y_val),
            'test': (X_test, T_test, Y_test),
            'gt_exps': None  
        }

        torch.save(D, data_path + f'{dataset_name}_ent_labellist_split.pt')
        torch.save(D, f'./datasets/{dataset_name}_{model_name}_{rem}_{task_name}/split=1.pt')

