import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
import scipy.stats
import sklearn.metrics
import torch
import torch.nn as nn
import time
import sklearn
import scipy
import pdb

from torch import autocast
from transformers import AutoTokenizer

from model import ComponentsAnalysisRegression
from dataloader import load_dataframe, make_loader
from utils import AverageMeter, to_gpu, kl_divergence_score
from metrics import cosine_similarity, euclidean_distance, dtw_distance, pearson_correlation, mutual_information


class Trainer(object):
    def __init__(self, args, config):
        self.writer = args.writer
        self.logger = args.logger
        
        # load data
        self.df_all = load_dataframe(args, config)
        
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['pretrained_path'],
            use_fast=True,
            trust_remote_code=True,
        )
        
        # set module
        self.model = ComponentsAnalysisRegression(config).to(args.device)
        self.model = to_gpu(args, self.logger, self.model)
        
        # set loss
        self.criterion = nn.MSELoss(reduction='mean')
        
        self.args = args
        self.config = config

    def valid_step_components(self, dataloader, epoch=0):
        self.model.eval()
        
        losses = AverageMeter()
        
        total_len = len(dataloader)
        predictions = []
        Y = []
        features = {
            'hidden_states':[],
            'hidden_states_blank':[],
            'hidden_states_up':[],
            'hidden_states_core':[],
            'hidden_states_down':[],
            'hidden_states_gene':[],
            'attn_blank_up':[],
            'attn_up_core':[],
            'attn_core_down':[],
            'attn_down_gene':[],
            'output':[]
        }

        # pdb.set_trace()
        res_handle = open(osp.join(self.args.metrics_dir, 'sim_matrix_up90.csv'), 'w')
        res_handle.write('id,intensity_y,intensity_pred,sim_blank_up,sim_blank_core,sim_blank_down,sim_blank_gene,sim_up_core,sim_up_down,sim_up_gene,sim_core_down,sim_core_gene,sim_down_gene\n')
        
        for step, (X, mask, masks_blank, masks_up, masks_core, masks_down, masks_gene, y) in enumerate(dataloader):
            X = X.to(self.args.device)
            mask = mask.to(self.args.device)
            masks_blank = masks_blank.to(self.args.device)
            masks_up = masks_up.to(self.args.device)
            masks_core = masks_core.to(self.args.device)
            masks_down = masks_down.to(self.args.device)
            masks_gene = masks_gene.to(self.args.device)
            y = y.to(self.args.device)
            batch_size = y.size(0)

            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.float16):
                    preds, hidden_states_dict = self.model(X, mask, masks_blank, masks_up, masks_core, masks_down, masks_gene)
                    loss = self.criterion(preds, y)
                    
                    if self.config['mode'] == 'test':
                        hidden_states = hidden_states_dict['hidden_states']
                        hidden_states_blank = hidden_states_dict['hidden_states_blank']
                        hidden_states_up = hidden_states_dict['hidden_states_up']
                        hidden_states_core = hidden_states_dict['hidden_states_core']
                        hidden_states_down = hidden_states_dict['hidden_states_down']
                        hidden_states_gene = hidden_states_dict['hidden_states_gene']
                        
                        mean = 4.813026468891894
                        std = 1.6166693708525242
                        intensity_y = torch.expm1(y * std + mean)
                        intensity_pred = torch.expm1(preds * std + mean)
                        
                        idlist = torch.where((torch.abs(intensity_y - intensity_pred)/intensity_y)<0.1)[0]  # acc > 90%

                        for i in range(len(idlist)):
                            id_i = idlist[i]

                            hidden_states_blank_selected_idlist = torch.where(hidden_states_blank[id_i].max(dim=1)[0]!=0)[0]
                            hidden_states_blank_selected = hidden_states_blank[id_i][hidden_states_blank_selected_idlist[0]:hidden_states_blank_selected_idlist[-1]+1]
                        
                            hidden_states_up_selected_idlist = torch.where(hidden_states_up[id_i].max(dim=1)[0]!=0)[0]
                            hidden_states_up_selected = hidden_states_up[id_i][hidden_states_up_selected_idlist[0]:hidden_states_up_selected_idlist[-1]+1]
                            
                            hidden_states_core_selected_idlist = torch.where(hidden_states_core[id_i].max(dim=1)[0]!=0)[0]
                            hidden_states_core_selected = hidden_states_core[id_i][hidden_states_core_selected_idlist[0]:hidden_states_core_selected_idlist[-1]+1]
                            
                            hidden_states_down_selected_idlist = torch.where(hidden_states_down[id_i].max(dim=1)[0]!=0)[0]
                            hidden_states_down_selected = hidden_states_down[id_i][hidden_states_down_selected_idlist[0]:hidden_states_down_selected_idlist[-1]+1]
                            
                            hidden_states_gene_selected_idlist = torch.where(hidden_states_gene[id_i].max(dim=1)[0]!=0)[0]
                            hidden_states_gene_selected = hidden_states_gene[id_i][hidden_states_gene_selected_idlist[0]:hidden_states_gene_selected_idlist[-1]+1]
                            
                            # res_cosine_similarity = cosine_similarity(hidden_states_up_fix, hidden_states_core_fix)
                            # res_euclidean_distance = euclidean_distance(hidden_states_up_fix, hidden_states_core_fix)
                            # res_dtw_distance = dtw_distance(hidden_states_up_fix, hidden_states_core_fix)
                            # res_pearson_correlation = pearson_correlation(hidden_states_up_selected, hidden_states_core_selected)
                            # res_mutual_information = mutual_information(hidden_states_up_fix, hidden_states_core_fix)

                            # sim_up_core = cosine_similarity(hidden_states_up_selected, hidden_states_core_selected).item()
                            # sim_up_down = cosine_similarity(hidden_states_up_selected, hidden_states_down_selected).item()
                            # sim_up_gene = cosine_similarity(hidden_states_up_selected, hidden_states_gene_selected).item()
                            # sim_core_down = cosine_similarity(hidden_states_core_selected, hidden_states_down_selected).item()
                            # sim_core_gene = cosine_similarity(hidden_states_core_selected, hidden_states_gene_selected).item()
                            # sim_down_gene = cosine_similarity(hidden_states_down_selected, hidden_states_gene_selected).item()
                            # res_handle.write(f'{intensity_y[id_i]},{intensity_pred[id_i]},{sim_up_core},{sim_up_down},{sim_up_gene},{sim_core_down},{sim_core_gene},{sim_down_gene}\n')
                            
                            sim_blank_up = pearson_correlation(hidden_states_blank_selected, hidden_states_up_selected).item()
                            sim_blank_core = pearson_correlation(hidden_states_blank_selected, hidden_states_core_selected).item()
                            sim_blank_down = pearson_correlation(hidden_states_blank_selected, hidden_states_down_selected).item()
                            sim_blank_gene = pearson_correlation(hidden_states_blank_selected, hidden_states_gene_selected).item()
                            sim_up_core = pearson_correlation(hidden_states_up_selected, hidden_states_core_selected).item()
                            sim_up_down = pearson_correlation(hidden_states_up_selected, hidden_states_down_selected).item()
                            sim_up_gene = pearson_correlation(hidden_states_up_selected, hidden_states_gene_selected).item()
                            sim_core_down = pearson_correlation(hidden_states_core_selected, hidden_states_down_selected).item()
                            sim_core_gene = pearson_correlation(hidden_states_core_selected, hidden_states_gene_selected).item()
                            sim_down_gene = pearson_correlation(hidden_states_down_selected, hidden_states_gene_selected).item()
                            res_handle.write(f'{id_i},{intensity_y[id_i]},{intensity_pred[id_i]},{sim_blank_up},{sim_blank_core},{sim_blank_down},{sim_blank_gene},{sim_up_core},{sim_up_down},{sim_up_gene},{sim_core_down},{sim_core_gene},{sim_down_gene}\n')
                            
                            # print(f'intensity_y: {intensity_y[id_i].item()}')
                            # print(f'intensity_pred: {intensity_pred[id_i].item()}')
                            # print(f'\tup\t\tcore\t\tdown\t\tgene'.expandtabs(12))
                            # print(f'up\t1.0\t\t{sim_up_core}\t{sim_up_down}\t{sim_up_gene}'.expandtabs(12))
                            # print(f'core\t{sim_up_core}\t1.0\t\t{sim_core_down}\t{sim_core_gene}'.expandtabs(12))
                            # print(f'down\t{sim_up_down}\t{sim_core_down}\t1.0\t\t{sim_down_gene}'.expandtabs(12))
                            # print(f'gene\t{sim_up_gene}\t{sim_core_gene}\t{sim_down_gene}\t1.0'.expandtabs(12))
                            # print()
            # print(str(step+1)+'/'+str(len(dataloader)), flush=True)
            
            losses.update(loss.item(), batch_size)
            predictions.append(preds)
            Y.append(y)
            features['hidden_states'].append(hidden_states_dict['hidden_states'])
            features['hidden_states_blank'].append(hidden_states_dict['hidden_states_blank'])
            features['hidden_states_up'].append(hidden_states_dict['hidden_states_up'])
            features['hidden_states_core'].append(hidden_states_dict['hidden_states_core'])
            features['hidden_states_down'].append(hidden_states_dict['hidden_states_down'])
            features['hidden_states_gene'].append(hidden_states_dict['hidden_states_gene'])
            features['attn_blank_up'].append(hidden_states_dict['attn_blank_up'])
            features['attn_up_core'].append(hidden_states_dict['attn_up_core'])
            features['attn_core_down'].append(hidden_states_dict['attn_core_down'])
            features['attn_down_gene'].append(hidden_states_dict['attn_down_gene'])
            features['output'].append(hidden_states_dict['output'])
            
            if self.args.print_step:
                if step % self.args.display_step == 0 or step == (total_len - 1):
                    self.logger.info(f"Eval[{step}/{total_len}]  "
                        f"Loss: {losses.val:.5f} ({losses.avg:.5f})  "
                        # f"bce: {weighted_bce_loss:.5f}  "
                        # f"Elapsed: {time_since(start, float(step + 1) / total_len)} "
                        )
        res_handle.close()
        
        predictions = torch.cat(predictions)
        Y = torch.cat(Y)

        y_true = Y.cpu().numpy()
        y_pred = predictions.detach().cpu().numpy()

        features['hidden_states'] = torch.cat(features['hidden_states']).detach().cpu().numpy()
        features['hidden_states_blank'] = torch.cat(features['hidden_states_blank']).detach().cpu().numpy()
        features['hidden_states_up'] = torch.cat(features['hidden_states_up']).detach().cpu().numpy()
        features['hidden_states_core'] = torch.cat(features['hidden_states_core']).detach().cpu().numpy()
        features['hidden_states_down'] = torch.cat(features['hidden_states_down']).detach().cpu().numpy()
        features['hidden_states_gene'] = torch.cat(features['hidden_states_gene']).detach().cpu().numpy()
        features['attn_blank_up'] = torch.cat(features['attn_blank_up']).detach().cpu().numpy()
        features['attn_up_core'] = torch.cat(features['attn_up_core']).detach().cpu().numpy()
        features['attn_core_down'] = torch.cat(features['attn_core_down']).detach().cpu().numpy()
        features['attn_down_gene'] = torch.cat(features['attn_down_gene']).detach().cpu().numpy()
        features['output'] = torch.cat(features['output']).detach().cpu().numpy()
        
        metrics = self.compute_metrics(y_true, y_pred)
        losses_avg = losses.avg

        self.writer.add_scalar('val_loss', losses_avg, epoch)
        self.writer.add_scalar('val_mse', metrics['mse'], epoch)
        self.writer.add_scalar('val_r2', metrics['r2'], epoch)
        self.writer.add_scalar('val_rmse', metrics['rmse'], epoch)
        self.writer.add_scalar('val_mae', metrics['mae'], epoch)
        self.writer.add_scalar('val_pearsonr_corr_coefficient', metrics['pearsonr_corr_coefficient'], epoch)
        self.writer.add_scalar('val_pearsonr_p', metrics['pearsonr_p'], epoch)
        self.writer.add_scalar('val_spearmanr_corr_coefficient', metrics['spearmanr_corr_coefficient'], epoch)
        self.writer.add_scalar('val_spearmanr_p', metrics['spearmanr_p'], epoch)
        self.writer.add_scalar('val_kl_divergence', metrics['kl_divergence'], epoch)
        
        return losses_avg, metrics, features, y_true, y_pred
    
    def test(self, fold=0):
        check_point = torch.load(self.config['checkpoint_path'], map_location=self.args.device)
        self.model.load_state_dict(check_point)
        self.logger.info("Loaded checkpoint model.")
        
        fold_valid = torch.where(self.df_all['fold']==fold)[0]
        
        df_valid = {
            'sequences': torch.index_select(self.df_all['sequences'], dim=0, index=fold_valid),
            'intensity': torch.index_select(self.df_all['intensity'], dim=0, index=fold_valid),
            'masks': torch.index_select(self.df_all['masks'], dim=0, index=fold_valid),
            'masks_blank': torch.index_select(self.df_all['masks_blank'], dim=0, index=fold_valid),
            'masks_up': torch.index_select(self.df_all['masks_up'], dim=0, index=fold_valid),
            'masks_core': torch.index_select(self.df_all['masks_core'], dim=0, index=fold_valid),
            'masks_down': torch.index_select(self.df_all['masks_down'], dim=0, index=fold_valid),
            'masks_gene': torch.index_select(self.df_all['masks_gene'], dim=0, index=fold_valid),
        }
        
        data_loader = make_loader(self.args, self.config, df_valid)
        
        start_time = time.time()
        valid_loss_avg, val_metrics, val_features, val_y_true, val_y_pred = self.valid_step_components(data_loader['val'])
        elapsed = time.time() - start_time
        
        self.logger.info("Time {}s || Val - Loss {}  MSE {}  R2 {}  RMSE {}  MAE {}  Pearsonr_corr {}  Pearsonr_p {}  Spearmanr_corr {}  Spearmanr_p {} KL_divergence {}"
                        .format(
                            round(elapsed,3),
                            valid_loss_avg,
                            val_metrics['mse'],
                            val_metrics['r2'],
                            val_metrics['rmse'],
                            val_metrics['mae'],
                            val_metrics['pearsonr_corr_coefficient'],
                            val_metrics['pearsonr_p'],
                            val_metrics['spearmanr_corr_coefficient'],
                            val_metrics['spearmanr_p'],
                            val_metrics['kl_divergence'],
                        )
        )
    
    def compute_metrics(self, y_true, y_pred):
        r2 = sklearn.metrics.r2_score(y_true, y_pred)
        mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
        rmse = sklearn.metrics.mean_squared_error(y_true, y_pred, squared=False)
        mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
        pearsonr_corr_coefficient, pearsonr_p = scipy.stats.pearsonr(y_true, y_pred)
        spearmanr_corr_coefficient, spearmanr_p = scipy.stats.spearmanr(y_true, y_pred)
        kl_divergence = kl_divergence_score(y_true, y_pred)
        
        return {
            'r2': r2, 
            'pearsonr_corr_coefficient': pearsonr_corr_coefficient,
            'pearsonr_p': pearsonr_p,
            'spearmanr_corr_coefficient': spearmanr_corr_coefficient,
            'spearmanr_p': spearmanr_p,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'kl_divergence': kl_divergence,
        }
