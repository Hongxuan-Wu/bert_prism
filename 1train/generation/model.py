import torch
import torch.nn as nn
import pdb

from transformers import AutoModel, AutoConfig, AutoModelForMaskedLM
from sample import Categorical, WholeWordMasking
import diffusion_word_freq


class PromoterGenerationDiffusion(nn.Module):
    def __init__(self, config, tokenizer, device):
        super(PromoterGenerationDiffusion, self).__init__()
        
        # set diffusion
        if config['sample_strategy'] == 'Categorical':
            self.sample_cls = Categorical()
        elif config['sample_strategy'] == 'wwm':
            self.sample_cls = WholeWordMasking(self.tokenizer)
        else:
            raise ValueError
        
        diffusion_schedule = diffusion_word_freq.create_discrete_diffusion_schedule(
            kind='mutual', 
            num_steps=config['num_steps']
        )
        self.diffusion_instance = diffusion_word_freq.MaskDiffusion(
            dim=tokenizer.vocab_size,
            schedule=diffusion_schedule,
            tokenizer=tokenizer,
            sample_cls=self.sample_cls,
            word_freq_lambda=config['word_freq_lambda'],
            device=device
        )
        
        # load pretrained model
        autocfg = AutoConfig.from_pretrained(config['pretrained_path'], trust_remote_code=True)
        autocfg.overall_timestep = config['num_steps']
        # if config['timestep'] == 'layerwise':
        #     autocfg.auto_map['AutoModelForMaskedLM']='bert_layers_new_timestep.BertForMaskedLM'

        self.model = AutoModelForMaskedLM.from_pretrained(config['pretrained_path'], config=autocfg, trust_remote_code=True)
        
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

    def denoise_fn(self, targets, timestep, attention_mask):
        assert len(targets.size()) == 2  # bsz * seqlen
        bsz = targets.size(0)

        self.cls = torch.full((1, 1), fill_value=self.tokenizer.cls_token_id).to(targets.device)
        self.sep = torch.full((1, 1), fill_value=self.tokenizer.sep_token_id).to(targets.device)
        self.att_ones = torch.ones((1, 1)).to(targets.device)
        self.att_zeros = torch.zeros((1, 1)).to(targets.device)
        
        if self.config['timestep'] == 'none':
            targets = torch.cat((self.cls.repeat(bsz, 1), targets, self.sep.repeat(bsz, 1)), dim=1)
            attention_mask = torch.cat((self.att_ones.repeat(bsz, 1), attention_mask, self.att_zeros.repeat(bsz, 1)), dim=1)
            # return self.model(input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)['logits'][:, 1:-1, :]
            
            # attention_mask[:, 0] = 0            
            # attention_mask[:, -1] = 0
            return self.model(input_ids=targets, attention_mask=attention_mask)['logits'][:, 1:-1, :]
        elif self.config['timestep'] == 'token':
            targets = torch.cat((
                self.cls.repeat(bsz, 1),
                torch.full((bsz, 1), fill_value=timestep.item() + 110),
                targets,
                self.sep.repeat(bsz, 1)
            ), dim=1)
            attention_mask = torch.cat((self.att_ones.repeat(bsz, 2), attention_mask, self.att_zeros.repeat(bsz, 1)), dim=1)
            return self.model(input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)['logits'][:, 2:-1, :]
        elif self.config['timestep'] == 'layerwise':
            # targets = torch.cat((
            #     self.cls.repeat(bsz, 1),
            #     targets,
            #     self.sep.repeat(bsz, 1)
            # ), dim=1)
            # attention_mask = torch.cat((self.att_ones.repeat(bsz, 1), attention_mask, self.att_zeros.repeat(bsz, 1)), dim=1)
            logits = self.model(input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)['logits']
            return logits
            # return self.model(input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)['logits'][:, 1:-1, :]
        else:
            raise NotImplementedError
    
    def generate(self, word_freq):
        shape = torch.Size([self.config['predict_batch'], self.config['token_length'] - 2])
        self.diffusion_instance.word_freq = word_freq
        
        # pdb.set_trace()
        res_dict = diffusion_word_freq.discrete_diffusion_predict_fn(
            shape=shape,
            denoise_fn=self.denoise_fn,
            diffusion=self.diffusion_instance,
            predict_x0=self.config['predict_x0'],
            sample_cls=self.sample_cls,
            step_size=1,  # When the artificial initiator is generated, step_size is always 1
            topk=self.config['predict_filter_topk'],
            topp=self.config['predict_filter_topp'],
            target_mask=torch.ones(shape, device=word_freq.device),
            show_process=False,
            temperature=1,
            device=word_freq.device,
            # word_freq=True
            # context_fn=context_fn
        )
        return res_dict
