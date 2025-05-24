import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
import sklearn.metrics
import torch
import torch.nn as nn
import sklearn
import numpy as np
import pandas as pd
import pdb

import matplotlib.pyplot as plt
from umap import UMAP
from torch import autocast
from transformers import AutoTokenizer

from model import AuthenticityCls
from dataloader import load_dataframe, make_loader, make_loader_predict
from utils import AverageMeter, to_gpu


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
        self.model = AuthenticityCls(config).to(args.device)
        self.model = to_gpu(args, self.logger, self.model)
        
        # set loss
        self.criterion = nn.BCEWithLogitsLoss()

        self.reducer = UMAP(n_components=2, random_state=args.seed)
        
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
        
        if self.config['save_roc_auc']:
            df_roc_auc = pd.DataFrame([metrics['fpr'], metrics['tpr']]).T
            df_roc_auc.to_csv(osp.join(self.args.metrics_dir, 'roc_auc.csv'), sep=',', index=False, header=['FPR', 'TPR'])
        
        if self.config['save_y']:
            df_y = pd.DataFrame([y_true, y_pred]).T
            df_y.to_csv(osp.join(self.args.metrics_dir, 'y.csv'), sep=',', index=False, header=['y_trues', 'y_pred'])
        
        if self.config['save_scatter']:
            features_reduced = self.reducer.fit_transform(features)
            df = pd.DataFrame(features_reduced)
            df['label'] = y_true

            df.to_csv(osp.join(self.args.metrics_dir, 'scatter.csv'), sep=',', index=False, header=True)
        
            colors = ['#179b73' if y==1 else '#d48aaf' for y in y_true]
            plt.scatter(features_reduced[:,0], features_reduced[:,1], s=5, c=colors, alpha=1)
            plt.savefig(osp.join(self.args.metrics_dir, 'scatter.png'))

        if self.config['save_metrics']:
            df_metrics = pd.DataFrame([metrics]).T
            df_metrics.columns = ['metrics']
            df_metrics.to_csv(osp.join(self.args.metrics_dir, 'metrics.csv'), sep=',', index=True, header=True)
        
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
                    # preds, feature = self.model(X, mask)
                preds, feature = self.model(X, mask)
            predictions.append(preds)
        predictions = torch.cat(predictions)
        y_pred = predictions.detach().cpu().numpy()
        df = pd.DataFrame(y_pred)
        df.to_csv(osp.join(self.args.metrics_dir, 'authenticity_cls.csv'), sep=',', index=False, header=False)
        
        if not os.path.exists(self.config['predict_dir']):
            os.makedirs(self.config['predict_dir'])
        
        df.to_csv(osp.join(self.config['predict_dir'], 'authenticity_cls.csv'), sep=',', index=False, header=False)

    def compute_metrics(self, y_true, y_pred):
        y_pred = 1 / (1 + np.exp(-y_pred)) # sigmoid
        
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred)
        auc = sklearn.metrics.auc(fpr, tpr)

        y_pred_class = np.round(y_pred)
        accuracy = sklearn.metrics.accuracy_score(y_true, y_pred_class)
        precision = sklearn.metrics.precision_score(y_true, y_pred_class, zero_division=0)
        recall = sklearn.metrics.recall_score(y_true, y_pred_class, zero_division=0)
        f1 = sklearn.metrics.f1_score(y_true, y_pred_class, zero_division=0)
        mcc = sklearn.metrics.matthews_corrcoef(y_true, y_pred_class)
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc, 
            'accuracy': accuracy, 
            'precision': precision, 
            'recall': recall,
            'f1': f1, 
            'mcc': mcc, 
        }
