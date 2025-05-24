import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
import importlib
import pdb

from utils import get_current_time, set_seed, set_logger
from trainer import Trainer
from tensorboardX import SummaryWriter


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, default='1train/authenticity_cls/config.py', help='the files of config')

    parser.add_argument('--display_step', type=int, default=100, help='display training information in how many step')
    parser.add_argument('--print_step', type=bool, default=False, help='print the step information')
    parser.add_argument('--log_dir', type=str, default='./results/logs/authenticity_cls/', help='path that log will be saved')
    
    parser.add_argument('--gpu_ids', type=list, default='0')
    
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    
    args = parser.parse_args()
    return args

def init_configs():
    args = parse()
    
    spec2 = importlib.util.spec_from_file_location("", args.config_file)
    odm = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(odm)
    config = odm.config

    current_time = get_current_time()
    args.log_dir = os.path.join(args.log_dir, current_time)
    args.metrics_dir = os.path.join(args.log_dir, 'metrics')

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.metrics_dir):
        os.makedirs(args.metrics_dir)
    
    args.device = 'cuda:' + args.gpu_ids[0]
    args.gpu_ids = [int(id) for id in args.gpu_ids]
    
    args.writer = SummaryWriter(args.log_dir)
    args.logger = set_logger(args, config)
    
    set_seed(args)

    return args, config

def main():
    args, config = init_configs()
    
    if config['mode'] == 'predict':
        trainer = Trainer(args, config)
        trainer.predict()
    elif config['mode'] == 'test':
        trainer = Trainer(args, config)
        res = trainer.test()
        args.logger.info("Val - Loss {}  Accuracy {}  AUC {}  MCC {}  F1 {}  Precision {}  Recall {}"
                        .format(
                            res[0],
                            res[1]['accuracy'],
                            res[1]['auc'],
                            res[1]['mcc'],
                            res[1]['f1'],
                            res[1]['precision'],
                            res[1]['recall'],
                        )
        )
    else:
        exit("Mode error!")


if __name__ == '__main__':
    main()
