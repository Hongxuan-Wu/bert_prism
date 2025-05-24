from collections import OrderedDict

config = OrderedDict()

# config['mode'] = 'test'
config['mode'] = 'predict'

config['save_metrics'] = True
config['save_roc_auc'] = True
config['save_y'] = True
config['save_scatter'] = True

config['root'] = '/data/whx/projects/bert_prism_data/'

################################################################### Model ##############################################################
config['output_type'] = 'pool'  # pool / cls / mean

config['pretrained_path'] = config['root']+ "pretrained/BERT-PRISM-1/"

#################################################################### Data ##############################################################

config['root_project'] = config['root'] + 'authenticity_cls/'

################################################################### Training ###############################################################

config['valid_batch_size'] = 128
config['val_dataset'] = 'MG1655_RegulonDB'

if config['mode'] == 'test':
    config['checkpoint_dir'] = config['root'] + 'authenticity_cls/m1_genes_escherichia/'
    # config['checkpoint_dir'] = config['root'] + 'authenticity_cls/m2_genes_escherichia/'
    # config['checkpoint_dir'] = config['root'] + 'authenticity_cls/dnabert2_genes_escherichia/'
    # config['checkpoint_dir'] = config['root'] + 'authenticity_cls/m1_Escherichia/'
    # config['checkpoint_dir'] = config['root'] + 'authenticity_cls/m1_ESM/'

elif config['mode'] == 'predict':
    config['checkpoint_dir'] = config['root'] + 'authenticity_cls/m1_genes_escherichia/'

    config['predict_dir'] = './results/predicts/'
    config['predict_filepath'] = config['predict_dir'] + 'gen_seqs.csv'
