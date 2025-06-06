import os
import os.path as osp
import pandas as pd
import numpy as np
import torch
import pdb

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold


def load_dataframe(args, config):
    # read dataframe
    if config['dataset'] == 'components':
        # filename = 'components_cut'
        filename = 'components_cut_filtered'
        
        df_raw = np.load(osp.join(config['root_data'], filename+'.npz'))
        
        sequences = df_raw['sequences']
        masks = df_raw['masks']
        masks_blank = df_raw['masks_blank']
        masks_up = df_raw['masks_up']
        masks_core = df_raw['masks_core']
        masks_down = df_raw['masks_down']
        masks_gene = df_raw['masks_gene']
        intensity = df_raw['intensity']
        
        intensity = np.log1p(intensity)
        intensity = (intensity - intensity.mean()) / intensity.std()  # mean - 4.813026468891894, std - 1.6166693708525242
        
        # mean = 4.813026468891894
        # std = 1.6166693708525242
        # y_fix = intensity * std + mean
        # y_fix2 = np.expm1(y_fix)
        # pdb.set_trace()
        
        df = {
            'sequences': torch.tensor(sequences, dtype=torch.int64),
            'intensity': torch.tensor(intensity, dtype=torch.float32),
            'masks': torch.tensor(masks, dtype=torch.int64),
            'masks_blank': torch.tensor(masks_blank, dtype=torch.int64),
            'masks_up' : torch.tensor(masks_up, dtype=torch.int64),
            'masks_core': torch.tensor(masks_core, dtype=torch.int64),
            'masks_down': torch.tensor(masks_down, dtype=torch.int64),
            'masks_gene': torch.tensor(masks_gene, dtype=torch.int64),
        }

    if config['mode'] == 'test':
        if config['dataset'] == 'components':
            df['fold'] = torch.tensor([-1 for _ in range(len(df['sequences']))], dtype=torch.int64)
            
            kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=args.seed)
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df['sequences'])):
                df['fold'][val_idx] = fold_idx

        # pdb.set_trace()
        return df

def make_loader(args, config, df_valid):
    valid_dataset = NpyComponentDataset(df=df_valid)
    
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

class NpyComponentDataset(Dataset):
    def __init__(self, df):
        super(NpyComponentDataset, self).__init__()
        self.df = df

        self.intensity = df['intensity']
        self.seq_input_ids = df['sequences']
        self.masks = df['masks']
        self.masks_blank = df['masks_blank']
        self.masks_up = df['masks_up']
        self.masks_core = df['masks_core']
        self.masks_down = df['masks_down']
        self.masks_gene = df['masks_gene']
        
    def __len__(self):
        return len(self.df['intensity'])

    def __getitem__(self, idx):
        return self.seq_input_ids[idx], self.masks[idx], self.masks_blank[idx], self.masks_up[idx], self.masks_core[idx], self.masks_down[idx], self.masks_gene[idx], self.intensity[idx]
