import os
import os.path as osp
import pandas as pd
import numpy as np
import torch
import pdb

from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from collections import Counter

def get_word_freq(tokens, tokenizer):
    counter = Counter(np.array(tokens).reshape([-1]))
    word_freq = torch.zeros((tokenizer.vocab_size,), dtype=torch.int64)
    for i in range(tokenizer.vocab_size):
        word_freq[i] += counter[i]
    word_freq = word_freq_preprocess_fn(word_freq)
    word_freq[tokenizer.pad_token_id] = 0.  # stable training
    word_freq[tokenizer.sep_token_id] = 0.
    word_freq[tokenizer.cls_token_id] = 0.
    word_freq_matrix = torch.stack([process_fn_in_getitem(word_freq.gather(0, d)) for d in tokens], dim=0)
    return word_freq, word_freq_matrix

def word_freq_preprocess_fn(wf):
    wf = wf + 1
    wf = wf.log()
    wf = wf / wf.max()

    # range: 0 - 1
    return wf

def process_fn_in_getitem(wf):
    return wf - wf.mean()

def load_dataframe(args, config):
    # read dataframe
    datapath = os.path.join(config['root_data'], 'data_gene_tpm')
    
    if 'prokaryotes' in config['dataset']:
        dataname = 'tpm_gene_' + config['dataset'][6:] + '_filtered'
    else:            
        dataname = 'tpm_gene_' + config['dataset'][6:]
    
    if 'newtoken' in config['pretrained_path']:
        dataname += '_newtoken'
    elif 'mergetoken' in config['pretrained_path']:
        dataname += '_mergetoken'
        
    if config['catATG'] and ('prokaryotes' not in config['dataset']):
        dataname += '_catATG'
    
    datapath = osp.join(datapath, dataname+'.npz')
    df_raw = np.load(datapath)
    
    sequences = np.array(df_raw['sequences'])
    masks = np.array(df_raw['masks'])
    intensity = np.array(df_raw['intensity'])
    
    if config['select_active']:
        index = np.where(intensity>=10)[0]
        df = {
            'sequences': torch.tensor(sequences[index], dtype=torch.int64),
            'masks': torch.tensor(masks[index], dtype=torch.int64)
        }
    else:
        df = {
            'sequences': torch.tensor(sequences, dtype=torch.int64),
            # 'labels': torch.tensor(labels, dtype=torch.float32),
            'masks': torch.tensor(masks, dtype=torch.int64)
        }

    # KFold
    df['fold'] =  torch.tensor([-1 for _ in range(len(df['sequences']))], dtype=torch.int64)
    
    kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=args.seed)
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df['sequences'])):
        df['fold'][val_idx] = fold_idx

    return df

def make_loader(args, config, df_train, df_valid, tokenizer):
    tokens = torch.concat([df_train['sequences'], df_valid['sequences']], dim=0)[:, 1:-1]
    masks = torch.concat([df_train['masks'], df_valid['masks']], dim=0)[:, 1:-1]
    word_freq, word_freq_matrix = get_word_freq(tokens, tokenizer)  # must combine train&valid data to compute word_freq
    
    valid_dataset = NpyDataset(
        df=df_valid, 
        tokenizer=tokenizer,
        tokens=tokens[-len(df_valid['sequences']):],
        masks=masks[-len(df_valid['sequences']):],
        word_freq=word_freq,
        word_freq_matrix=word_freq_matrix[-len(df_valid['sequences']):],
    )

    dataset = {
        'val': valid_dataset
    }
    return dataset

class PromoterDataset(Dataset):
    def __init__(self, df, tokenizer, tokens, masks, word_freq, word_freq_matrix):
        super(PromoterDataset, self).__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.seq_input_ids = tokens
        self.seq_attention_mask = masks
        self.word_freq = word_freq
        self.word_freq_matrix = word_freq_matrix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.seq_input_ids[idx], self.seq_attention_mask[idx], self.word_freq_matrix[idx]

class NpyDataset(Dataset):
    def __init__(self, df, tokenizer, tokens, masks, word_freq, word_freq_matrix):
        super(NpyDataset, self).__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.seq_input_ids = tokens
        self.seq_attention_mask = masks
        self.word_freq = word_freq
        self.word_freq_matrix = word_freq_matrix

    def __len__(self):
        return len(self.seq_input_ids)

    def __getitem__(self, idx):
        return self.seq_input_ids[idx], self.seq_attention_mask[idx], self.word_freq_matrix[idx]
