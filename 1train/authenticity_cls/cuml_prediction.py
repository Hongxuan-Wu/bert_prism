import os
import os.path as osp
import numpy as np
import pandas as pd
import random
import argparse
import logging
import time
import joblib
import sklearn
import pdb

# https://docs.rapids.ai/install/
# conda create -n rapids -c rapidsai -c conda-forge -c nvidia cudf=25.04 cuml=25.04 python=3.12 'cuda-version>=12.0,<=12.8'
# pip install scikit-learn xgboost
# pip install optuna

from datetime import datetime
from collections import OrderedDict

from sklearn.model_selection import StratifiedKFold
from cuml.preprocessing import StandardScaler
from xgboost import XGBClassifier
from cuml.svm import SVC
from cuml.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


config = OrderedDict()

# config['mode'] = 'train_valid'
config['mode'] = 'test'
# config['mode'] = 'predict'

config['save_model'] = True

config['save_roc_auc'] = True
config['save_y'] = True
config['save_scatter'] = True
config['save_metrics'] = True

config['root'] = '/data/whx/'

# ------------------------------------------ Data ------------------------------------------
config['root_project'] = config['root'] + 'projects/PRISM/1train/authenticity_cls/data/'
config['root_genes_tokens'] = config['root'] + 'prism/datasets/authenticity/'

config['predict_filepath'] = config['root'] + 'predict/ecos/20240912_043045_ckpt72newtoken_steps2000_ecos_gen100000/gen_seqs.csv'

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
    config['save_model'] = False
    
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

# ---------------------------------------- Checkpoint ----------------------------------------
if config['mode'] == 'test':
    if config['model'] == 'XGBoost':
        model = 'xgboost'
    elif config['model'] == 'RandomForest':
        model = 'rf'
    elif config['model'] == 'LogisticRegression':
        model = 'lr'
    elif config['model'] == 'SVM':
        model = 'svm'
    
    if config['kfold']:
        checkpoint_root = config['root'] + 'projects/PRISM/1train/authenticity_cls/results/ckpt10fold_ml/'
        config['checkpoint_dir'] = checkpoint_root + model + '_' + config['dataset'] + '_10fold'
    else:
        checkpoint_root = config['root'] + 'projects/PRISM/1train/authenticity_cls/results/ckptNto1_ml/'
        config['checkpoint_dir'] = checkpoint_root + model + '_' + config['train_dataset']

elif config['mode'] == 'predict':
    checkpoint_root = '/hy-tmp/projects/DNABERT_Promotor/1train/promoter_real_fake_predict/results/checkpoints_ml/'
    config['checkpoint_dir'] = checkpoint_root + '20241106_133213_xgboost_ESM'

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
    df_values.fillna(0, inplace=True)
    
    return df_values

def load_dataframe(args, config):
    # Energy Values Conversion
    # read dataframe
    if config['mode'] == 'train_valid' or config['mode'] == 'test':
        if config['kfold']:
            if 'genes_' in config['dataset']:
                prokaryotes_type = config['dataset'][6:]
                
                df = pd.read_csv(os.path.join(config['root_genes_tokens'], prokaryotes_type + '/genes_' + prokaryotes_type + '_catATG.csv'), delimiter=',')
            
            length = len(df['sequence'])
            if 'genes_' in config['dataset'] and config['datasize'] < length:
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
    elif config['mode'] == 'predict':
        df_predict = pd.read_csv(config['predict_filepath'], delimiter=',', header=None)
        df_predict = convert_to_values(df_predict[0], convert_type=config['convert_type'])
        return df_predict
    
class Trainer(object):
    def __init__(self, args, config):
        # self.writer = args.writer
        self.logger = args.logger
        
        # load data
        if config['mode'] == 'train_valid' or config['mode'] == 'test':
            if config['kfold']:
                self.df_all = load_dataframe(args, config)
            else:
                self.train_dataset, self.val_dataset = load_dataframe(args, config)
        elif config['mode'] == 'predict':
            self.df_predict = load_dataframe(args, config)
        else:
            exit()
        
        if config['model'] == 'XGBoost':
            if not config['kfold']:
                if config['train_dataset'] == 'genes_corynebacterium':
                    max_depth=8
                    min_child_weight=6
                    learning_rate=0.14965558423218428
                    n_estimators=864
                    subsample=0.8309651203442192
                    colsample_bytree=0.7685832839727329
                    reg_lambda=0.7856686235337745
                    gamma=2.6165832510824076
                elif config['train_dataset'] == 'genes_synechocystis':
                    max_depth=4
                    min_child_weight=3
                    learning_rate=0.24450949283798615
                    n_estimators=486
                    subsample=0.7474755766980469
                    colsample_bytree=0.6238363423635654
                    reg_lambda=1.7277350956626605
                    gamma=2.1500587865519494
                elif config['train_dataset'] == 'genes_bacillus':
                    max_depth=8
                    min_child_weight=2
                    learning_rate=0.17122879386201342
                    n_estimators=789
                    subsample=0.7982297388827739
                    colsample_bytree=0.5264411104428975
                    reg_lambda=1.1287575494977629
                    gamma=4.903295839844507
                else:
                    max_depth=8
                    min_child_weight=5
                    learning_rate=0.2701302383944569
                    n_estimators=246
                    subsample=0.8329053843413553
                    colsample_bytree=0.5464527072763781
                    reg_lambda=0.9346460635289635
                    gamma=0.6224226486202789
                    
            self.model = XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                booster='gbtree',
                
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_lambda=reg_lambda,
                gamma=gamma,
                
                device=args.device, 
                sampling_method='gradient_based',
                n_jobs=args.num_workers, 
                random_state=args.seed, 
            )
        elif config['model'] == 'SVM':
            if not config['kfold']:
                if config['train_dataset'] == 'genes_corynebacterium':
                    C=0.021729466473301484
                    gamma=9.996074726362673
                elif config['train_dataset'] == 'genes_synechocystis':
                    C=0.0270337944370068
                    gamma=9.993323423798186
                elif config['train_dataset'] == 'genes_bacillus':
                    C=0.22620455508981366
                    gamma=9.967226669147006
                else:
                    C=0.004510912297937215,
                    gamma=1.0014216708840215e-05,

            self.model = SVC(
                C=C,
                gamma=gamma,
                
                kernel='rbf', 
                tol=1e-4,
                random_state=args.seed,
                max_iter=10000,
                # verbose=True,
                probability=True,
                cache_size=2000,
            )
        elif config['model'] == 'RandomForest':
            if not config['kfold']:
                if config['train_dataset'] == 'genes_corynebacterium':
                    n_estimators=800
                    max_features='log2'
                    max_depth=41
                    min_samples_split=24
                    min_samples_leaf=25
                elif config['train_dataset'] == 'genes_synechocystis':
                    n_estimators=618
                    max_features='log2'
                    max_depth=22
                    min_samples_split=168
                    min_samples_leaf=40
                elif config['train_dataset'] == 'genes_bacillus':
                    n_estimators=628
                    max_features='log2'
                    max_depth=36
                    min_samples_split=162
                    min_samples_leaf=22
                else:
                    n_estimators=282
                    max_features='log2'
                    max_depth=39
                    min_samples_split=77
                    min_samples_leaf=26
                    
            self.model = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_features=max_features,
                max_depth=max_depth, 
                min_samples_split=min_samples_split, 
                min_samples_leaf=min_samples_leaf,

                n_bins=256,
                n_streams=1,
                device=args.device, 
                random_state=args.seed, 
                verbose=0
            )
        elif config['model'] == 'LogisticRegression':
            if not config['kfold']:
                if config['train_dataset'] == 'genes_corynebacterium':
                    C=1.0074409399598726e-05
                    max_iter=713
                    tol=0.00013773698746550158
                elif config['train_dataset'] == 'genes_synechocystis':
                    C=7.206966044690846e-05
                    max_iter=1400
                    tol=0.00041751411401206933
                elif config['train_dataset'] == 'genes_bacillus':
                    C=7.215893079440166e-05
                    max_iter=1367
                    tol=7.693676644729799e-06
                else:
                    C=7.782179639125782e-05
                    max_iter=1348
                    tol=4.463411002052759e-06
            
            self.model = LogisticRegression(
                C=C,
                max_iter=max_iter,
                tol=tol,
                
                solver='lbfgs',
                penalty='l2', 
                random_state=args.seed,
                n_jobs=args.num_workers,
                verbose=0
            )
        
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
        
        start_time = time.time()

        if self.config['model'] in ['SVM'] and self.config['kfold']:
            self.args.logger.info('Scaling...')
            scaler = StandardScaler().fit(x_train)
            x_train_scaled = scaler.transform(x_train)
            x_valid_scaled = scaler.transform(x_valid)
            self.model.fit(x_train_scaled, y_train)
            y_predict_proba = self.model.predict_proba(x_valid_scaled)
            val_metrics = self.compute_metrics(y_valid, y_predict_proba)
        else:            
            self.model.fit(x_train, y_train)
            y_predict_proba = self.model.predict_proba(x_valid)
            val_metrics = self.compute_metrics(y_valid, y_predict_proba)
            
        elapsed = time.time() - start_time
        
        if config['kfold']:
            self.logger.info("Fold[{}] Time {}s || Val - Loss {}  Accuracy {}  AUC {}  MCC {}  F1 {}  Precision {}  Recall {}"
                        .format(
                            fold,
                            round(elapsed,3),
                            round(val_metrics['loss'], 4), 
                            round(val_metrics['accuracy'], 4), 
                            round(val_metrics['auc'], 4), 
                            round(val_metrics['mcc'], 4), 
                            round(val_metrics['f1'], 4), 
                            round(val_metrics['precision'], 4), 
                            round(val_metrics['recall'], 4), 
                        )
            )
        else:
            self.logger.info("Time {}s || Val - Loss {}  Accuracy {}  AUC {}  MCC {}  F1 {}  Precision {}  Recall {}"
                        .format(
                            round(elapsed,3),
                            round(val_metrics['loss'], 4), 
                            round(val_metrics['accuracy'], 4), 
                            round(val_metrics['auc'], 4), 
                            round(val_metrics['mcc'], 4), 
                            round(val_metrics['f1'], 4), 
                            round(val_metrics['precision'], 4), 
                            round(val_metrics['recall'], 4), 
                        )
            )
        
        return val_metrics

    def test(self, fold=0):
        if self.config['kfold']:
            # split data & dataloader
            # df_train = self.df_all[self.df_all.fold != fold].reset_index(drop=True).drop('fold', axis=1)
            df_valid = self.df_all[self.df_all.fold == fold].reset_index(drop=True).drop('fold', axis=1)
            # x_train = df_train.drop('label', axis=1).iloc[:, :400].to_numpy()
            # y_train = df_train['label'].to_numpy()
            x_valid = df_valid.drop('label', axis=1).iloc[:, :400].to_numpy()
            y_valid = df_valid['label'].to_numpy()
            self.model = joblib.load(os.path.join(self.config['checkpoint_dir'], "last"+str(fold)+".pkl"))
        else:
            # x_train = self.train_dataset.drop('label', axis=1).iloc[:, :400].to_numpy()
            # y_train = self.train_dataset['label'].to_numpy()
            x_valid = self.val_dataset.drop('label', axis=1).iloc[:, :400].to_numpy()
            y_valid = self.val_dataset['label'].to_numpy()
            self.model = joblib.load(os.path.join(self.config['checkpoint_dir'], "last.pkl"))
        
        self.logger.info("Loaded model.")
        
        start_time = time.time()
        y_predict_proba = self.model.predict_proba(x_valid)
        val_metrics = self.compute_metrics(y_valid, y_predict_proba)
        elapsed = time.time() - start_time
        
        if config['kfold']:
            self.logger.info("Fold[{}] Time {}s || Val - Loss {}  Accuracy {}  AUC {}  MCC {}  F1 {}  Precision {}  Recall {}"
                        .format(
                            fold,
                            round(elapsed,3),
                            round(val_metrics['loss'], 4), 
                            round(val_metrics['accuracy'], 4), 
                            round(val_metrics['auc'], 4), 
                            round(val_metrics['mcc'], 4), 
                            round(val_metrics['f1'], 4), 
                            round(val_metrics['precision'], 4), 
                            round(val_metrics['recall'], 4), 
                        )
            )
        else:
            self.logger.info("Time {}s || Val - Loss {}  Accuracy {}  AUC {}  MCC {}  F1 {}  Precision {}  Recall {}"
                        .format(
                            round(elapsed,3),
                            round(val_metrics['loss'], 4), 
                            round(val_metrics['accuracy'], 4), 
                            round(val_metrics['auc'], 4), 
                            round(val_metrics['mcc'], 4), 
                            round(val_metrics['f1'], 4), 
                            round(val_metrics['precision'], 4), 
                            round(val_metrics['recall'], 4), 
                        )
            )
        
        return val_metrics
    
    def predict(self):
        x_valid = self.df_predict.iloc[:, :400].to_numpy()
        
        self.model = joblib.load(os.path.join(self.config['checkpoint_dir'], "last.pkl"))
        self.logger.info("Loaded model.")
        
        start_time = time.time()
        y_predict_proba = self.model.predict_proba(x_valid)
        elapsed = time.time() - start_time
        
        df = pd.DataFrame(y_predict_proba[:,-1])
        df.to_csv(osp.join(self.args.log_dir, 'predict.csv'), sep=',', index=False, header=False)
    
        return
    
    def compute_metrics(self, y_true, y_pred):
        """
        Args:
            y_true (ndarray(int64)): 0/1
            y_pred (ndarray(float32)): [0,1]
        """
        # pdb.set_trace()
        bce = sklearn.metrics.log_loss(y_true, y_pred)

        y_pred = y_pred[:,-1]
        
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
            'loss': bce
        }

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
    args.metrics_dir = os.path.join(args.log_dir, 'metrics')
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, current_time)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.metrics_dir):
        os.makedirs(args.metrics_dir)
        
    args.device = 'cuda:' + args.gpu_ids[0]
    args.gpu_ids = [int(id) for id in args.gpu_ids]
    
    args.logger = set_logger(args, config)
    
    set_seed(args)

    return args, config

def main():
    args, config = init_configs()
    
    if config['mode'] == 'predict':
        trainer = Trainer(args, config)
        trainer.predict()
    elif config['mode'] == 'train_valid' or config['mode'] == 'test':
        if config['kfold']:
            results_valid = {'loss': [], 'accuracy': [], 'auc': [], 'mcc': [], 'f1': [], 'precision': [], 'recall': []}
            for fold in range(config['n_folds']):
                args.logger.info('')
                args.logger.info("======================================== Starting fold {}: ========================================".format(fold))
                
                trainer = Trainer(args, config)
                if config['mode'] == 'train_valid':
                    res = trainer.run(fold)
                else:
                    res = trainer.test(fold)
                
                args.logger.info("Val - Loss {}  Accuracy {}  AUC {}  MCC {}  F1 {}  Precision {}  Recall {}"
                                 .format(
                                    res['loss'],
                                    res['accuracy'],
                                    res['auc'],
                                    res['mcc'],
                                    res['f1'],
                                    res['precision'],
                                    res['recall'],
                                )
                )
                
                results_valid['loss'].append(res['loss'])
                results_valid['accuracy'].append(res['accuracy'])
                results_valid['auc'].append(res['auc'])
                results_valid['mcc'].append(res['mcc'])
                results_valid['f1'].append(res['f1'])
                results_valid['precision'].append(res['precision'])
                results_valid['recall'].append(res['recall'])

            args.logger.info("{}Fold-Mean - Val - Loss {}  Accuracy {}  AUC {}  MCC {}  F1 {}  Precision {}  Recall {}"
                            .format(
                                config['n_folds'],
                                np.mean(results_valid['loss']),
                                np.mean(results_valid['accuracy']),
                                np.mean(results_valid['auc']),
                                np.mean(results_valid['mcc']),
                                np.mean(results_valid['f1']),
                                np.mean(results_valid['precision']),
                                np.mean(results_valid['recall']),
                            )
            )
        else:
            trainer = Trainer(args, config)
            
            if config['mode'] == 'train_valid':
                res = trainer.run()
            else:
                res = trainer.test()
                
            args.logger.info("Val - Loss {}  Accuracy {}  AUC {}  MCC {}  F1 {}  Precision {}  Recall {}"
                            .format(
                                res['loss'],
                                res['accuracy'],
                                res['auc'],
                                res['mcc'],
                                res['f1'],
                                res['precision'],
                                res['recall'],
                            )
            )


if __name__ == '__main__':
    main()
