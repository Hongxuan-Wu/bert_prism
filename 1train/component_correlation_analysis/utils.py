import random
import numpy as np
import os
import logging
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import scipy.stats


def kl_divergence_score(y_true, y_pred, num_bins=30, epsilon=1e-10):

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    
    bins = np.linspace(min_val, max_val, num_bins + 1)
    
    true_hist, _ = np.histogram(y_true, bins=bins, density=True)
    pred_hist, _ = np.histogram(y_pred, bins=bins, density=True)
    
    true_hist += epsilon
    pred_hist += epsilon
    
    true_prob = true_hist / np.sum(true_hist)
    pred_prob = pred_hist / np.sum(pred_hist)
    
    kl_divergence = scipy.stats.entropy(true_prob, pred_prob)
    return kl_divergence

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_current_time():
    current_time = str(datetime.fromtimestamp(int(time.time())))
    d = current_time.split(' ')[0]
    t = current_time.split(' ')[1]
    d = d.split('-')[0] + d.split('-')[1] + d.split('-')[2]
    t = t.split(':')[0] + t.split(':')[1] + t.split(':')[2]
    return d+ '_' +t

def to_gpu(args, logger, module):
    # multi-gpu configuration
    [logger.info('GPU: {}  Spec: {}'.format(i, torch.cuda.get_device_name(i))) for i in args.gpu_ids]
    
    if len(args.gpu_ids) > 1:
        logger.info('Construct multi-gpu model ...')
        module = nn.DataParallel(module, device_ids=args.gpu_ids, dim=0)
        logger.info('done!\n')

    return module

def set_seed(args):
    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
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
    logger.info('')
    logger.info('------------------------------command line arguments------------------------------')
    logger.info(args)
    logger.info('')
    logger.info('--------------------------------------configs-------------------------------------')
    logger.info(config)
    return logger
