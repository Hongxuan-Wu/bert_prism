import os
import numpy as np
import pandas as pd
import random
import argparse
import logging
import time
import pdb

# https://docs.rapids.ai/install/
# conda create -n rapids -c rapidsai -c conda-forge -c nvidia cudf=25.04 cuml=25.04 python=3.12 'cuda-version>=12.0,<=12.8'
# pip install scikit-learn xgboost
# pip install optuna

import optuna
import cuml
import cupy
import sklearn

from datetime import datetime
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold
from cuml.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from cuml.svm import SVC
from cuml.ensemble import RandomForestClassifier


config = OrderedDict()
config['root'] = '/data/whx/'

# ------------------------------------------ Data ------------------------------------------
config['root_project'] = config['root'] + 'projects/PRISM/1train/authenticity_cls/data/'
config['root_genes_tokens'] = config['root'] + 'prism/datasets/authenticity/'

# ------------------------------------------ Model ------------------------------------------
config['convert_type'] = 'CA_D'  # CA_D / AP_D / 
# config['convert_type'] = 'AP_D' 

# config['model'] = 'XGBoost' 
# config['model'] = 'RandomForest' 
# config['model'] = 'LogisticRegression' 
config['model'] = 'SVM'

# ---------------------------------------- Training ----------------------------------------
############################
# config['kfold'] = True
config['kfold'] = False
############################

if config['kfold']:
    config['n_folds'] = 10  # 5/10
    
    if config['n_folds'] == 5:
            # config['fold'] = [0]
            config['fold'] = [0,1,2,3,4]
    elif config['n_folds'] == 10:
        config['fold'] = [0,1,2,3,4,5,6,7,8,9]
    
    config['dataset'] = 'genes_escherichia'
    
    if 'genes_' in config['dataset']:
        config['datasize'] = 58498
else:
    config['train_dataset'] = 'genes_escherichia'

    if 'genes_' in config['train_dataset']:
        config['train_datasize'] = 58498
    
    config['val_dataset'] = 'MG1655_RegulonDB'  # 7916


#############################################################################################################################
#############################################################################################################################
def convert_to_values(sequences, convert_type='CA_D'):
    # Convert the sequences to Energy Values
    energy_value_map= {}
    if convert_type == 'CA_D':
        energy_value_map = {
            'AA': 0.703, 'AT': 0.854, 'TA': 0.615, 'AG': 0.78, 
            'GA': 1.23, 'TT': 0.703, 'AC': 1.323, 'CA': 0.79, 
            'TG': 0.79, 'GT': 1.323, 'TC': 1.23, 'CT': 0.78, 
            'CC': 0.984, 'CG': 1.124, 'GC': 1.792, 'GG': 0.984,
        }
    elif convert_type == 'AP_D':
        energy_value_map = {
            'AA': -17.5, 'AT': -16.7, 'TA': -17, 'AG': -15.8,
            'GA': -14.7, 'TT': -17.5, 'AC': -18.1, 'CA': -19.5,
            'TG': -19.5, 'GT': -18.1, 'TC': -14.7, 'CT': -15.8,
            'CC': -14.9, 'CG': -19.2, 'GC': -14.7, 'GG': -14.9,
        }

    list_values = []
    for i in range(len(sequences)):
        seq = sequences.iloc[i]
        values = []
        for pos in range(len(seq) - 1):
            nucleotide_pair = seq.upper()[pos:pos+2]
            val = energy_value_map.get(nucleotide_pair, None)
            if val is not None:
                values.append(val)
        list_values.append(values)

    df_values = pd.DataFrame(list_values)
    df_values.fillna(0, inplace=True)  # 空值填 0 
    
    return df_values

def load_dataframe(args, config):
    if config['kfold']:
        if 'genes_' in config['dataset']:
            prokaryotes_type = config['dataset'][6:]
            
            df = pd.read_csv(os.path.join(config['root_genes_tokens'], prokaryotes_type + '/genes_' + prokaryotes_type + '_catATG.csv'), delimiter=',')

        if 'genes_' in config['dataset']:
            length = len(df['sequence'])
            real_ids = random.sample(range(0,length//2), 58498//2)  # ESM - 58498
            # real_ids = [random.randint(0, length//2-1) for _ in range(58498//2)]
            fake_ids = [x + length//2 for x in real_ids]
            ids = real_ids + fake_ids
            
            sequences = df['sequence'][ids]
            labels = df['label'][ids].reset_index(drop=True)
        else:
            sequences = df['sequence']
            labels = df['label']
        
        values = convert_to_values(sequences, convert_type=config['convert_type'])
        df_new = pd.concat([values,labels],axis=1)
        
        # KFold
        df_new["fold"] = -1
        kfold = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=args.seed)
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df_new, y=df_new['label'])):
            df_new.loc[val_idx, "fold"] = fold_idx
        
        # pdb.set_trace()
        return df_new
    else:
        if 'genes_' in config['train_dataset']:
            prokaryotes_type = config['train_dataset'][6:]
            train_dataset = pd.read_csv(os.path.join(config['root_genes_tokens'], prokaryotes_type + '/genes_' + prokaryotes_type + '_catATG.csv'), delimiter=',')

        if 'genes_' in config['train_dataset']:
            length = len(train_dataset['sequence'])
            
            if config['train_datasize'] < length:
                real_ids = random.sample(range(0,length//2), config['train_datasize']//2)  # ESM - 58498
                # real_ids = [random.randint(0, length//2-1) for _ in range(58498//2)]
                fake_ids = [x + length//2 for x in real_ids]
                ids = real_ids + fake_ids
                
                sequences_train = train_dataset['sequence'][ids]
                labels_train = train_dataset['label'][ids].reset_index(drop=True)
            else:
                sequences_train = train_dataset['sequence']
                labels_train = train_dataset['label']
        else:
            sequences_train = train_dataset['sequence']
            labels_train = train_dataset['label']

        values_train = convert_to_values(sequences_train, convert_type=config['convert_type'])
        train_dataset = pd.concat([values_train, labels_train], axis=1)

        if config['val_dataset'] == 'MG1655_RegulonDB':
            val_dataset = pd.read_csv(os.path.join(config['root_project'], 'MG1655-RegulonDB', 'MG1655-RegulonDB_promoters_catATG.csv'), delimiter=',')

        sequences_val = val_dataset['sequence']
        labels_val = val_dataset['label']
        values_val = convert_to_values(sequences_val, convert_type=config['convert_type'])
        val_dataset = pd.concat([values_val, labels_val], axis=1)
        
        # pdb.set_trace()
        return train_dataset, val_dataset
    
class Trainer(object):
    def __init__(self, args, config):
        self.logger = args.logger
        
        # load data
        if config['kfold']:
            self.df_all = load_dataframe(args, config)
        else:
            self.train_dataset, self.val_dataset = load_dataframe(args, config)
        
        self.args = args
        self.config = config

    def run(self, fold=0):
        if self.config['kfold']:
            # split data & dataloader
            df_train = self.df_all[self.df_all.fold != fold].reset_index(drop=True).drop('fold', axis=1)
            df_valid = self.df_all[self.df_all.fold == fold].reset_index(drop=True).drop('fold', axis=1)
            x_train = df_train.drop('label', axis=1).iloc[:, :400].to_numpy()
            y_train = df_train['label'].to_numpy()
            x_valid = df_valid.drop('label', axis=1).iloc[:, :400].to_numpy()
            y_valid = df_valid['label'].to_numpy()
        else:
            x_train = self.train_dataset.drop('label', axis=1).iloc[:, :400].to_numpy()
            y_train = self.train_dataset['label'].to_numpy()
            x_valid = self.val_dataset.drop('label', axis=1).iloc[:, :400].to_numpy()
            y_valid = self.val_dataset['label'].to_numpy()
        
        if config['model'] == 'XGBoost':
            x_train = cupy.asarray(x_train)
            y_train = cupy.asarray(y_train)
            x_valid = cupy.asarray(x_valid)
            y_valid = cupy.asarray(y_valid)
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                    
                    'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0),
                    
                    'gamma': trial.suggest_float('gamma', 0, 5),
                }
                
                model = XGBClassifier(
                    **params, 
                    objective='binary:logistic',
                    eval_metric='logloss',
                    booster='gbtree',
                    early_stopping_rounds=50, 
                    device=self.args.device, 
                    sampling_method='gradient_based',
                    random_state=self.args.seed,
                )
                model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=0)
                return cuml.metrics.log_loss(y_valid, model.predict_proba(x_valid))
    
            study = optuna.create_study(
                direction='minimize', 
                sampler=optuna.samplers.TPESampler(seed=self.args.seed)
            )
            study.optimize(objective, n_trials=1000, n_jobs=4)
            self.args.logger.info(f"Best Params: {study.best_params}")
            self.args.logger.info(f"Best Loss: {study.best_value}")
        elif config['model'] == 'LogisticRegression':
            def objective(trial):
                params = {
                    'C': trial.suggest_float('C', 1e-5, 1000, log=True),
                    'max_iter': trial.suggest_int('max_iter', 50, 2000),
                    'tol': trial.suggest_float('tol', 1e-6, 1e-3, log=True)
                }
                    
                model = LogisticRegression(
                    **params, 
                    penalty='l2', 
                    random_state=self.args.seed, 
                    solver='lbfgs',
                    # n_jobs=1, 
                    verbose=0
                )
                model.fit(x_train, y_train)
                return sklearn.metrics.log_loss(y_valid, model.predict_proba(x_valid))
            study = optuna.create_study(
                direction='minimize', 
                sampler=optuna.samplers.TPESampler(seed=self.args.seed)
            )
            study.optimize(objective, n_trials=100, n_jobs=4)
            self.args.logger.info(f"Best Params: {study.best_params}")
            self.args.logger.info(f"Best Loss: {study.best_value}")
        elif config['model'] == 'SVM':
            scaler = StandardScaler().fit(x_train)
            x_train_scaled = scaler.transform(x_train)
            x_valid_scaled = scaler.transform(x_valid)
            
            def objective(trial):
                params = {
                    'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
                    'gamma': trial.suggest_float('gamma', 1e-5, 1e1, log=True),
                    # 'max_iter': trial.suggest_int('max_iter', 1000, 10000),
                }
                model = SVC(
                    **params,
                    kernel='rbf',
                    tol=1e-4,
                    max_iter=10000,
                    probability=True,
                    cache_size=2000,
                )
                model.fit(x_train_scaled, y_train)
                return cuml.metrics.log_loss(y_valid, model.predict_proba(x_valid_scaled))
            
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(n_startup_trials=20, multivariate=True, seed=self.args.seed),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
            )
            study.optimize(objective, n_trials=100, n_jobs=4, gc_after_trial=True)
            self.args.logger.info(f"Best Params: {study.best_params}")
            self.args.logger.info(f"Best Loss: {study.best_value}")
        elif config['model'] == 'RandomForest':
            def objective(trial):
                params = {
                    # step 1
                    # 'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                    # 'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.2, 0.5]),
                    
                    # step 2
                    'max_depth': trial.suggest_int('max_depth', 15, 50),
                    'min_samples_split': trial.suggest_int('min_samples_split', 20, 200),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
                }
                    
                model = RandomForestClassifier(
                    **params,
                    # step 1
                    # max_depth=30,
                    # min_samples_split=50,
                    # min_samples_leaf=30,

                    # step 2
                    n_estimators=282,
                    max_features='log2',
                    
                    n_bins=256,
                    n_streams=1,
                    device=self.args.device, 
                    random_state=self.args.seed,
                    verbose=0
                )
                model.fit(x_train, y_train)
                return cuml.metrics.log_loss(y_valid, model.predict_proba(x_valid))
            
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(n_startup_trials=20, multivariate=True, seed=self.args.seed),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
            )
            study.optimize(objective, n_trials=100, n_jobs=8)
            self.args.logger.info(f"Best Params: {study.best_params}")
            self.args.logger.info(f"Best Loss: {study.best_value}")
        
        # optuna.visualization.plot_param_importances(study)
        exit()
    
def get_current_time():
    current_time = str(datetime.fromtimestamp(int(time.time())))
    d = current_time.split(' ')[0]
    t = current_time.split(' ')[1]
    d = d.split('-')[0] + d.split('-')[1] + d.split('-')[2]
    t = t.split(':')[0] + t.split(':')[1] + t.split(':')[2]
    return d+ '_' +t

def set_seed(args):
    # set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
def set_logging_config(logdir):
    """_summary_
    set logging configuration

    Args:
        logdir (str): directory put logs
    """
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, 'out.log')),
                                  logging.StreamHandler(os.sys.stdout)])

def set_logger(args, config):
    set_logging_config(args.log_dir)
    logger = logging.getLogger(' ')
    
    # Load the configuration params of the experiment
    logger.info('Launching experiment from: {}'.format(args.config_file))
    logger.info('Generated logs will be saved to: {}'.format(args.log_dir))
    logger.info('Generated checkpoints will be saved to: {}'.format(args.checkpoint_dir))
    logger.info('')
    logger.info('------------------------------command line arguments------------------------------')
    logger.info(args)
    logger.info('')
    logger.info('--------------------------------------configs-------------------------------------')
    logger.info(config)
    return logger

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, default='', help='the files of config')

    parser.add_argument('--log_dir', type=str, default='./results/logs_ml', help='path that log will be saved')
    parser.add_argument('--checkpoint_dir', type=str, default='./results/checkpoints_ml', help='the path of checkpoint')
    
    parser.add_argument('--gpu_ids', type=list, default='0')
    
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers (-1 is max.)')
    
    args = parser.parse_args()
    return args

def init_configs():
    args = parse()

    current_time = get_current_time()
    args.log_dir = os.path.join(args.log_dir, current_time)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, current_time)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    args.device = 'cuda:' + args.gpu_ids[0]
    args.gpu_ids = [int(id) for id in args.gpu_ids]
    
    args.logger = set_logger(args, config)
    
    set_seed(args)

    return args, config

def main():
    args, config = init_configs()
    
    if config['kfold']:
        for fold in range(config['n_folds']):
            args.logger.info('')
            args.logger.info("======================================== Starting fold {}: ========================================".format(fold))
            
            trainer = Trainer(args, config)
            trainer.run(fold)
    else:
        trainer = Trainer(args, config)
        trainer.run()


if __name__ == '__main__':
    main()
