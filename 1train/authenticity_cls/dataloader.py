import os
import pandas as pd
import torch
import pdb

from torch.utils.data import Dataset, DataLoader

def load_dataframe(args, config):
    # read dataframe
    if config['mode'] == 'test':
        if config['val_dataset'] == 'MG1655_RegulonDB':
            df_val = pd.read_csv(os.path.join(config['root_project'], 'MG1655-RegulonDB', 'MG1655-RegulonDB_promoters_catATG.csv'), delimiter=',')
        
        return df_val
    elif config['mode'] == 'predict':
        df_predict = pd.read_csv(config['predict_filepath'], delimiter=',', header=None)
        return df_predict

def make_loader(args, config, df_valid, tokenizer):
    valid_dataset = PromoterDataset(df=df_valid, tokenizer=tokenizer)
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config['valid_batch_size'], 
        shuffle=False, 
        drop_last=False,
        num_workers=args.num_workers, 
        pin_memory=True
    )
        
    data_loader = {
        'val': valid_loader
    }
    return data_loader

def make_loader_predict(args, config, df_predict, tokenizer):
    predict_dataset = PredictDataset(df=df_predict, tokenizer=tokenizer)
    predict_loader = DataLoader(
        predict_dataset, 
        batch_size=config['valid_batch_size'], 
        shuffle=False, 
        drop_last=False,
        num_workers=args.num_workers, 
        pin_memory=True
        )
    data_loader = {
        'predict': predict_loader
    }
    return data_loader

class PredictDataset(Dataset):
    def __init__(self, df, tokenizer):
        super(PredictDataset, self).__init__()
        self.df = df
        
        file = df.to_numpy()
        sequence = file[:, 0].tolist()

        self.seq_output = tokenizer(
            text=sequence, 
            return_tensors="pt", 
            max_length=100, 
            padding=True,
            truncation=True,
        )
        self.seq_input_ids = self.seq_output['input_ids']
        self.seq_attention_mask = self.seq_output['attention_mask']
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.seq_input_ids[idx], self.seq_attention_mask[idx]

class PromoterDataset(Dataset):
    def __init__(self, df, tokenizer):
        super(PromoterDataset, self).__init__()
        self.df = df
        
        file = df.to_numpy()
        sequence = file[:, 0].tolist()
        self.labels = file[:, 1]
        
        self.seq_output = tokenizer(
            text=sequence, 
            return_tensors="pt", 
            max_length=100, 
            padding=True,
            truncation=True,
        )
        self.seq_input_ids = self.seq_output['input_ids']
        self.seq_attention_mask = self.seq_output['attention_mask']
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.float32).float()
                
        return self.seq_input_ids[idx], self.seq_attention_mask[idx], label
