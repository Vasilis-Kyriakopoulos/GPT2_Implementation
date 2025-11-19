import torch
import torch.nn as nn
from .gpt2_blocks import  GPTBlock_Torch

class GPTModel_Torch(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers, dropout):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            GPTBlock_Torch(embed_dim=embed_dim,num_heads=num_heads,ff_dim =ff_dim, dropout=dropout) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim,)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape

        pos = torch.arange(T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)
       
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits
    
    def generate(self, idx, max_new_tokens=100):
        for _ in range(max_new_tokens):
            logits = self(idx)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
    
    
    