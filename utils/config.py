import torch

def get_config():
    return {
    "vocab_size": 50257,
    "context_length": 2,
    "emb_dim": 1024,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate_attn": 0.1,
    "drop_rate_shortcut": 0.1,
    "drop_rate_emb": 0.1,
    "qkv_bias": False,
    "learning_rate": 0.0004,
    "n_epochs":10,
    "batch_size": 5,
    "device": 'cpu'
    }
