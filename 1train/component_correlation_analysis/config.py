from collections import OrderedDict

config = OrderedDict()

config['mode'] = 'test'

config['root'] = '/data/whx/projects/prism_data/'

################################################################### Model ##############################################################
config['output_type'] = 'pool'  # pool / cls / mean

config['pretrained_path'] = config['root']+ "pretrained/PRISM-M1/"

config['components'] = True
# config['components'] = False

#################################################################### Data ##############################################################
config['root_project'] = config['root'] + 'component_correlation_analysis/'
config['root_data'] = config['root_project']

################################################################### Training ###############################################################

if config['mode'] == 'test':
    config['valid_batch_size'] = 256  # 2048

    config['checkpoint_dir'] = config['root'] + 'component_correlation_analysis/m1_components_filtered_crossattn/'

    config['dataset'] = 'components'
    
    if config['dataset'] == 'components':
        config['datasize'] = 100000000
        config['n_folds'] = 36  # filtered - 36 /  - 
        config['fold'] = [0]
    elif config['dataset'] == 'prokaryotes596':
        config['datasize'] = 100000000
        config['n_folds'] = 400  # filtered - 400
        config['fold'] = [0]
    
    config['checkpoint_path'] = config['checkpoint_dir'] + 'last.pth'
