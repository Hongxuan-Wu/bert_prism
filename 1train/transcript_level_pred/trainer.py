import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
import scipy.stats
import sklearn.metrics
import torch
import torch.nn as nn
import sklearn
import scipy
import pandas as pd
import pdb

from torch import autocast
from transformers import AutoTokenizer

from model import TranscriptLevelPred
from dataloader import load_dataframe, make_loader, make_loader_predict
from utils import AverageMeter, to_gpu, kl_divergence_score


class Trainer(object):
    def __init__(self, args, config):
        self.writer = args.writer
        self.logger = args.logger
        
        # load data
        if config['mode'] == 'test':
            self.df_val = load_dataframe(args, config)
        elif config['mode'] == 'predict':
            self.df_predict = load_dataframe(args, config)
        else:
            exit()
            
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['pretrained_path'],
            use_fast=True,
            trust_remote_code=True,
        )
        
        # set module
        self.model = TranscriptLevelPred(config).to(args.device)
        self.model = to_gpu(args, self.logger, self.model)
        
        # set loss
        self.criterion = nn.MSELoss(reduction='mean')
        
        self.args = args
        self.config = config

    def test(self):
        data_loader = make_loader(self.args, self.config, self.df_val, self.tokenizer)
        check_point = torch.load(os.path.join(self.config['checkpoint_dir'], "last.pth"), map_location=self.args.device)
        
        self.model.load_state_dict(check_point)
        self.logger.info("Loaded model.")
        
        losses = AverageMeter()
        total_len = len(data_loader)
        predictions = []
        Y = []
        features = []
        
        for step, (X, mask, y) in enumerate(data_loader['val']):
            X = X.to(self.args.device)
            mask = mask.to(self.args.device)
            y = y.to(self.args.device)
            batch_size = y.size(0)

            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.float16):
                    preds, feature = self.model(X, mask)
                    loss = self.criterion(preds, y)
            losses.update(loss.item(), batch_size)
            predictions.append(preds)
            Y.append(y)
            features.append(feature)

            if self.args.print_step:
                if step % self.args.display_step == 0 or step == (total_len - 1):
                    self.logger.info(f"Eval[{step}/{total_len}]  "
                        f"Loss: {losses.val:.5f} ({losses.avg:.5f})  "
                        )
        predictions = torch.cat(predictions)
        Y = torch.cat(Y)
        features = torch.cat(features)

        y_true = Y.cpu().numpy()
        y_pred = predictions.detach().cpu().numpy()
        features = features.detach().cpu().numpy()
        
        metrics = self.compute_metrics(y_true, y_pred)
        loss_avg = losses.avg
        
        return [loss_avg, metrics]

    def predict(self):
        data_loader = make_loader_predict(self.args, self.config, self.df_predict, self.tokenizer)
        
        check_point = torch.load(os.path.join(self.config['checkpoint_dir'], "last.pth"), map_location=self.args.device)
        self.model.load_state_dict(check_point)
        self.logger.info("Loaded model.")

        predictions = []
        
        for step, (X, mask) in enumerate(data_loader['predict']):
            X = X.to(self.args.device)
            mask = mask.to(self.args.device)

            with torch.no_grad():
                # with autocast(device_type='cuda', dtype=torch.float16):
                #     preds = self.model(X, mask)
                preds, feature = self.model(X, mask)
            predictions.append(preds)
        predictions = torch.cat(predictions)
        y_pred = predictions.detach().cpu().numpy()

        df = pd.DataFrame(y_pred)
        df.to_csv(osp.join(self.args.metrics_dir, 'transcript_level_pred.csv'), sep=',', index=False, header=False)

        if not os.path.exists(self.config['predict_dir']):
            os.makedirs(self.config['predict_dir'])
        
        df.to_csv(osp.join(self.config['predict_dir'], 'transcript_level_pred.csv'), sep=',', index=False, header=False)
    
    def compute_metrics(self, y_true, y_pred):
        r2 = sklearn.metrics.r2_score(y_true, y_pred)
        mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
        rmse = sklearn.metrics.mean_squared_error(y_true, y_pred, squared=False)
        mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
        pearsonr_corr_coefficient, pearsonr_p = scipy.stats.pearsonr(y_true, y_pred)
        spearmanr_corr_coefficient, spearmanr_p = scipy.stats.spearmanr(y_true, y_pred)
        kl_divergence = kl_divergence_score(y_true, y_pred)
        
        return {
            'r2': r2, 
            'pearsonr_corr_coefficient': pearsonr_corr_coefficient,
            'pearsonr_p': pearsonr_p,
            'spearmanr_corr_coefficient': spearmanr_corr_coefficient,
            'spearmanr_p': spearmanr_p,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'kl_divergence': kl_divergence,
        }
