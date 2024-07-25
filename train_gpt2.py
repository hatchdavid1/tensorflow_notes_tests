from dataclass import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F 

class CausalSelfAttention(nn.Module): 

    def __init__(self, config): 
        super().__init__()
        assert = config.n_embd % config.n_head == 0
        # Key, query and valyue projections for all heads but in a batch 
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Regularixzation 
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)
                                    .view(1,1, config.block_size, config.block_size)))


    def forward(self, x): 
        B, T, C = x.size() # Batch size, sequence length, embedding dimensionality (n_embd)
        # Calculate query, key and values for all heads in bathc and move head forward to batch
        # nh is the number of heads, hs us the head size and c number of channels = nh * ns
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim = 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        # Attention mateializes the large (T,T) matrix for all queries and keys  
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(aproximate = 'tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x): 
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # Attention (MAP) First Normalization Layer 
        x = x + self.mlp(self.ln_2(x)) # MLP (REDUCE) Second Normalization Layer
        return x


@dataclass
class GPTConfig: 
    block_size int = 256
    vocab_size int = 65
    n_layer int = 6
    n_head int = 6
    n_embed int = 384


class GPT(nn.Module): 

    def __init__(self, config): 
        super().__init__()
        self.config


        |self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Token embedings
            wpe = nn.Embedding(config.block_size, config.n_embd),  # Positional Encoding 
            h = nn.ModuleList([Block(config) for _ in range(config,n_layers)]), # Layers
            ln_f = nn.LayerNorm(config.n_embed) # Linear
        ))
        self.lm_head = nn.Linear(config.embd, config.vocab_size, bias = False)