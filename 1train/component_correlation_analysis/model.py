import torch
import torch.nn as nn
import pdb
from transformers import AutoModel, AutoConfig


class CrossAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, batch_first=True):
        super(CrossAttention, self).__init__()
        
        self.layer_before = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=batch_first
        )
        
        self.layer_after = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.GELU(),
        )
        
        self.model_init()
        
    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
    
    def forward(self, featureA, featureB):
        hA = self.layer_before(featureA)
        hB = self.layer_before(featureB)
        attn, _ = self.cross_attn(query=hB, key=hA, value=hA)
        out = self.layer_after(attn)
        return out

class Regressor(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super(Regressor, self).__init__()

        channels = [in_channels, 256, 128, 64, out_channels]
        self.dense = nn.Linear(channels[0], channels[-1])
    
        self.model_init()
        
    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        x = self.dense(x)
        return x

class ComponentsAnalysisRegression(nn.Module):
    def __init__(self, config):
        super(ComponentsAnalysisRegression, self).__init__()
        
        autocfg = AutoConfig.from_pretrained(config['pretrained_path'], trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(config['pretrained_path'], config=autocfg, trust_remote_code=True)
        self.backbone.pooler = nn.Identity()
        
        num_heads = 8
        input_dim = 768
        hidden_dim = input_dim*3
        self.cross_attention = nn.ModuleList([
            CrossAttention(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads, batch_first=True) 
            for _ in range(4)
        ])
        
        self.attn_layer = nn.Sequential(
            nn.Linear(input_dim*2, input_dim),
            nn.GELU(),
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = Regressor(config, input_dim, 1)
        
        self.config = config

    def forward(self, X, mask, masks_blank, masks_up, masks_core, masks_down, masks_gene):
        # token (batch_size, sequence_length, hidden_size)
        hidden_states = self.backbone(input_ids=X, attention_mask=mask)[0]
        hidden_states_blank = self.backbone(input_ids=X, attention_mask=masks_blank)[0]
        hidden_states_up = self.backbone(input_ids=X, attention_mask=masks_up)[0]
        hidden_states_core = self.backbone(input_ids=X, attention_mask=masks_core)[0]
        hidden_states_down = self.backbone(input_ids=X, attention_mask=masks_down)[0]
        hidden_states_gene = self.backbone(input_ids=X, attention_mask=masks_gene)[0]

        attn_blank_up = self.cross_attention[0](hidden_states_blank, hidden_states_up)
        attn_up_core = self.cross_attention[1](hidden_states_up, hidden_states_core)
        attn_core_down = self.cross_attention[2](hidden_states_core, hidden_states_down)
        attn_down_gene = self.cross_attention[3](hidden_states_down, hidden_states_gene)
        
        attn_out = attn_blank_up + attn_up_core + attn_core_down + attn_down_gene
        attn_out = torch.cat([hidden_states, attn_out], dim=-1)
        attn_out = self.attn_layer(attn_out)

        if self.config['output_type'] == 'cls':
            output = hidden_states[:,0,:]
        elif self.config['output_type'] == 'mean':
            output = torch.mean(hidden_states, dim=1)
        elif self.config['output_type'] == 'pool':
            output = self.global_avg_pool(hidden_states.transpose(1, 2)).squeeze(-1)  # (batch_size, hidden_size)

        preds = self.regressor(output).flatten()

        hidden_states_dict = {
            'hidden_states': hidden_states,
            'hidden_states_blank':hidden_states_blank,
            'hidden_states_up':hidden_states_up,
            'hidden_states_core':hidden_states_core,
            'hidden_states_down':hidden_states_down,
            'hidden_states_gene':hidden_states_gene,
            'attn_blank_up':attn_blank_up,
            'attn_up_core':attn_up_core,
            'attn_core_down':attn_core_down,
            'attn_down_gene':attn_down_gene,
            'output': output
        }
        return preds, hidden_states_dict
