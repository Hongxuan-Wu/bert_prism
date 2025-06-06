from collections import OrderedDict

config = OrderedDict()

config['token_length'] = 100

config['mode'] = 'generate'

config['root'] = '/data/whx/projects/bert_prism_data/'

################################################################### Model ##############################################################
config['pretrained_path'] = config['root']+ "pretrained/BERT-PRISM-1/"

if config['mode'] == 'generate':
    config['checkpoint_dir'] = config['root'] + 'generation/m1_genes_escherichia/'

# ------------------------------------------ Diffusion ------------------------------------------
# config['dtype'] = 'fp16'
config['dtype'] = 'bf16'

config['num_steps'] = 2000  # 1000 (500, 1000, 2000, 5000)
config['sample_strategy'] = 'Categorical'
config['word_freq_lambda'] = 0.3
config['timestep'] = 'none'  # 'none', 'token', 'layerwise'
config['hybrid_lambda'] = 1e-2
config['predict_x0'] = True

# config['eval_step_size'] = config['num_steps'] // 50  # eval_num_steps = 50
config['eval_step_size'] = config['num_steps'] // 2  # fastest eval
# config['eval_step_size'] = 1

config['predict_filter_topk'] = 10  # 10
config['predict_filter_topp'] = -1.0  # -1.0
config['predict_num'] = 200  # 10000, 100000
config['predict_batch'] = 100

#################################################################### Data ##############################################################

config['root_project'] = config['root'] + 'generation/'
config['root_data'] = config['root_project']

config['catATG'] = True
# config['catATG'] = False

# config['select_active'] = True
config['select_active'] = False

################################################################## Training ###############################################################
config['n_folds'] = 25
config['fold'] = [0]
config['dataset'] = 'genes_escherichia'

config['predict_dir'] = './results/predicts/'