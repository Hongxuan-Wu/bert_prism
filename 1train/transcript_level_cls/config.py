from collections import OrderedDict

config = OrderedDict()

# config['mode'] = 'test'
config['mode'] = 'predict'

config['save_metrics'] = True
config['save_y'] = True
config['save_scatter'] = True

# config['num_class'] = 2
config['num_class'] = 4

if config['num_class'] == 2:
    config['strong_weak'] = False
    
    config['save_roc_auc'] = True
else:
    config['save_roc_auc'] = False

config['root'] = '/data/whx/projects/bert_prism_data/'

################################################################### Model ##############################################################
config['output_type'] = 'pool'  # pool / cls / mean

config['pretrained_path'] = config['root']+ "pretrained/BERT-PRISM-1/"

#################################################################### Data ##############################################################

config['root_project'] = config['root'] + 'transcript_level_pred/'
config['root_data'] = config['root_project']

config['catATG'] = True

################################################################### Training ###############################################################
config['valid_batch_size'] = 2048  # 128

config['val_dataset'] = 'EColi_GeneExpression'

if config['mode'] == 'test':
    config['checkpoint_dir'] = config['root'] + 'transcript_level_cls/m1_genes_escherichia_cls' + str(config['num_class'])
    if config['catATG']:
        config['checkpoint_dir'] += '_catATG'

elif config['mode'] == 'predict':
    config['predict_dir'] = './results/predicts/'
    config['predict_filepath'] = config['predict_dir'] + 'gen_seqs.csv'
    
    config['checkpoint_dir'] = config['root'] + 'transcript_level_cls/m1_genes_escherichia_cls' + str(config['num_class'])
    if config['catATG']:
        config['checkpoint_dir'] += '_catATG'
