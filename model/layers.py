import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift



class GELU(nn.Module):
    def __init_(self, ):
        super().__init__()
    def forward(self, x):
       return  0.5*x*(1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi))*
                                            (x + 0.044715*torch.pow(x,3))
                                    ))
                                    
                                    
                                    

class FFNN(nn.Module):
    def __init__(self, cfg):
        super(FFNN, self).__init__()
        
        self.layer1 = nn.Linear(cfg['emb_dim'], 4*cfg['emb_dim']).to(cfg["device"])
        self.layer2 = nn.Linear(4*cfg['emb_dim'], cfg['emb_dim']).to(cfg["device"])
        self.gelu = GELU()
        
        self.layers = nn.Sequential(
        self.layer1,
        self.gelu,
        self.layer2,
        )
        
    
    def forward(self,x):
        return self.layers(x)
        

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,context_length,dropout , n_heads , qkv_bias=False, device="cuda"):
        super(MultiHeadAttention, self).__init__()
        assert d_out % n_heads == 0, "Embedding dimenssions must be divisible by number of heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.n_heads = n_heads
        self.W_Q   = nn.Linear(d_in, d_out).to(device)
        self.W_K   = nn.Linear(d_in, d_out).to(device)
        self.W_V   = nn.Linear(d_in, d_out).to(device)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_out, d_out).to(device)
        self.register_buffer('mask', 
                            torch.triu(torch.ones(context_length, context_length), 
                            diagonal = 1)
                            )
        
    def forward(self, x):
        batch_size, seq_len, d_in = x.shape
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        head_dim = Q.shape[1]/self.n_heads

        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim)

        Q = Q.transpose(1,2)
        K  = K.transpose(1,2)
        V  = V.transpose(1,2)

        attn_scores = Q @ K.transpose(2,3)
        attn_scores.masked_fill_(self.mask.bool()[:seq_len, :seq_len], -torch.inf)
        attn_weights = torch.softmax(attn_scores / K.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = (attn_weights @ V).transpose(1,2)

        context_vec = context_vec.reshape(batch_size, seq_len, self.d_out)
        context_vec = self.proj(context_vec)  # optional projection
        
        # context_vec = self.proj(context_vec)
        return context_vec
