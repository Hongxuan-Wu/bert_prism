import torch
import torch.nn as nn
import pdb
from transformers import AutoModel, AutoConfig

class Classifier(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super(Classifier, self).__init__()
        
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
    
class AuthenticityCls(nn.Module):
    def __init__(self, config):
        super(AuthenticityCls, self).__init__()
        
        autocfg = AutoConfig.from_pretrained(config['pretrained_path'], trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(config['pretrained_path'], config=autocfg, trust_remote_code=True)
        self.backbone.pooler = nn.Identity()
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = Classifier(config, 768, 1)
        # self.sigmoid = nn.Sigmoid()
        
        self.config = config

    def forward(self, X, mask):
        hidden_states = self.backbone(input_ids=X, attention_mask=mask)[0]  # token (batch_size, sequence_length, hidden_size)
        
        if self.config['output_type'] == 'cls':
            output = hidden_states[:,0,:]
        elif self.config['output_type'] == 'mean':
            output = torch.mean(hidden_states, dim=1)
        elif self.config['output_type'] == 'pool':
            output = self.global_avg_pool(hidden_states.transpose(1, 2)).squeeze(-1)  # (batch_size, hidden_size)
        
        preds = self.classifier(output)
        
        # preds = self.sigmoid(preds).flatten()
        preds = preds.flatten()

        return preds, output
