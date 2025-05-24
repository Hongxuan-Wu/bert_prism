from collections import OrderedDict

config = OrderedDict()

# config['mode'] = 'test'
config['mode'] = 'predict'

config['save_metrics'] = True
config['save_y'] = True
config['save_scatter'] = True

config['root'] = '/data/whx/projects/bert_prism_data/'

################################################################### Model ##############################################################
config['output_type'] = 'pool'  # pool / cls / mean

config['pretrained_path'] = config['root']+ "pretrained/BERT-PRISM-1/"

#################################################################### Data ##############################################################
config['root_project'] = config['root'] + 'transcript_level_pred/'
config['root_data'] = config['root_project']

################################################################### Training ###############################################################
config['valid_batch_size'] = 2048  # 128

config['val_dataset'] = 'EColi_GeneExpression'
# config['val_dataset'] = 'Bsub_GeneExpression'
# config['val_dataset'] = 'Cglu_GeneExpression'
# config['val_dataset'] = 'Vibr_GeneExpression'
# config['val_dataset'] = 'WCFS1_GeneExpression'

if config['mode'] == 'test':
    config['checkpoint_dir'] = config['root'] + 'transcript_level_pred/m1_genes_escherichia/'

elif config['mode'] == 'predict':
    config['checkpoint_dir'] = config['root'] + 'transcript_level_pred/m1_genes_escherichia_catATG/'    
    
    config['predict_dir'] = './results/predicts/'
    config['predict_filepath'] = config['predict_dir'] + 'gen_seqs.csv'
