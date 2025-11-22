import torch
import torch.nn as nn
from .gpt2_blocks import  GPTBlock_Torch

class GPTModel_Torch(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(dropout)
        ff_dim = embed_dim* 4
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
    
    def generate(self, idx, max_new_tokens=15,context_length = 256,temperature=0.0,top_k=None,eos_id=None):
        print(idx)
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self(idx)
            logits = logits[:,-1,:]
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
            
            if temperature > 0.0:
                logits = logits / temperature


            # subtract rowwise max before softmax
                logits = logits - logits.max(dim=-1, keepdim=True).values
            
            # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

            if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                break


            idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

        return idx
    
    
    