import torch
import torch.nn as nn
from model.layers import LayerNorm, MultiHeadAttention, FFNN

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super(TransformerBlock, self).__init__()
        self.layernorm1 = LayerNorm(cfg['emb_dim'])
        self.layernorm2 = LayerNorm(cfg['emb_dim'])

        self.mha = MultiHeadAttention(cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"], cfg["drop_rate_attn"] , cfg["n_heads"], device=cfg['device'])
        self.dropout = nn.Dropout(cfg['drop_rate_shortcut'])
        self.ff = FFNN(cfg)
        self.relu = nn.ReLU()  # this should be replace with GeLU

    def forward(self, x):
        x_shortcut = x
        x = self.layernorm1(x)
        x = self.mha(x)
        # x = self.linear_project1(x)
        x = self.dropout(x)
        x = x + x_shortcut
        
        x_shortcut = x
        x = self.layernorm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x  = x_shortcut + x
        return x

