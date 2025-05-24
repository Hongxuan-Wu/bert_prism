import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
import torch
import time
import pandas as pd
import pdb

import nltk
from nltk.translate.bleu_score import SmoothingFunction as SF
from umap import UMAP
from transformers import AutoTokenizer
from model import PromoterGenerationDiffusion
from dataloader import load_dataframe, make_loader
from utils import to_gpu


class Trainer(object):
    def __init__(self, args, config):
        self.writer = args.writer
        self.logger = args.logger
        
        # load data
        if config['mode'] == 'generate':
            self.df_all = load_dataframe(args, config)
        else:
            exit()
            
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['pretrained_path'],
            use_fast=True,
            trust_remote_code=True,
        )
        
        # set module
        self.model = PromoterGenerationDiffusion(config, self.tokenizer, args.device).to(args.device)
        self.model = to_gpu(args, self.logger, self.model)
        
        if config['dtype'] == 'fp16':
            self.dtype = torch.float16
        elif config['dtype'] == 'bf16':
            self.dtype = torch.bfloat16
        elif config['dtype'] == 'fp32':
            self.dtype = torch.float32
        
        self.args = args
        self.config = config

    def generate(self, fold=0):
        # split data & dataloader
        fold_train = torch.where(self.df_all['fold']!=fold)[0] 
        fold_valid = torch.where(self.df_all['fold']==fold)[0]
        
        df_train = {
            'sequences': torch.index_select(self.df_all['sequences'], dim=0, index=fold_train),
            'masks': torch.index_select(self.df_all['masks'], dim=0, index=fold_train)
        }
        df_valid = {
            'sequences': torch.index_select(self.df_all['sequences'], dim=0, index=fold_valid),
            'masks': torch.index_select(self.df_all['masks'], dim=0, index=fold_valid)
        }
        dataset = make_loader(self.args, self.config, df_train, df_valid, self.tokenizer)

        check_point = torch.load(os.path.join(self.config['checkpoint_dir'], "last0.pth"), map_location=self.model.device)
        self.model.load_state_dict(check_point)
        
        # generate & save
        df_tokens = pd.DataFrame()
        df_seqs = pd.DataFrame()
        for i in range(self.config['predict_num']//self.config['predict_batch']):
            start_time = time.time()
            res_dict = self.model.generate(dataset['val'].word_freq.to(self.args.device))  # word_freq is same in train_dataset & valid_dataset
            elapsed = time.time() - start_time
            
            sentences = self.tokenizer.batch_decode(res_dict['final_state'])
            seqs = [sentence.replace(' ', '') for sentence in sentences]
            self_bleu = self.self_bleu(sentences)
            diversity = self.calculate_diversity(res_dict['final_state'])
            
            self.logger.info("Generation {}  Time {}s || Self-Bleu {}    Diversity {}".format(
                i,
                round(elapsed, 3), 
                round(self_bleu, 4),
                round(diversity, 4)
            ))
            
            df_tokens_tmp = pd.DataFrame(res_dict['final_state'].cpu())
            df_tokens = pd.concat([df_tokens, df_tokens_tmp])
            df_seqs_tmp = pd.DataFrame(seqs)
            df_seqs = pd.concat([df_seqs, df_seqs_tmp])
        
        df_tokens.to_csv(osp.join(self.args.log_dir, 'gen_tokens.csv'), sep=',', index=False, header=False)
        df_seqs.to_csv(osp.join(self.args.log_dir, 'gen_seqs.csv'), sep=',', index=False, header=False)

        if not os.path.exists(self.config['predict_dir']):
            os.makedirs(self.config['predict_dir'])
        
        # df_tokens.to_csv(osp.join(self.config['predict_dir'], 'gen_tokens.csv'), sep=',', index=False, header=False)
        df_seqs.to_csv(osp.join(self.config['predict_dir'], 'gen_seqs.csv'), sep=',', index=False, header=False)
    
    def calculate_diversity(self, recovers):
        num_chars = recovers.shape[1]
        diversity = 0.0
        for recover in recovers:
            num_unique_chars = torch.unique(recover).shape[0]
            diversity += num_unique_chars / num_chars
        diversity /= recovers.shape[0]
        return diversity

    def bleu(self, referencesList, recoversList):
        referencesList_split = [[reference.split()] for reference in referencesList]
        recoversList_split = [recover.split() for recover in recoversList]
        
        bleu = nltk.translate.bleu_score.corpus_bleu(
            referencesList_split, 
            recoversList_split, 
            smoothing_function=SF().method4
        )
        return bleu

    def self_bleu(self, recoversList):
        """
        This function is a canonical implementation of self-BLEU.
        The deviation from the above one is that the references are ALL THE REST sentences and this one uses CORPUS bleu.
        """
        recoversList = [recover.split() for recover in recoversList]
        
        res = 0.
        for i in range(len(recoversList)):
            res += nltk.translate.bleu_score.corpus_bleu(
                [recoversList[:i] + recoversList[i + 1:]], 
                [recoversList[i]], 
                smoothing_function=SF().method4
            )
        return res / len(recoversList)
