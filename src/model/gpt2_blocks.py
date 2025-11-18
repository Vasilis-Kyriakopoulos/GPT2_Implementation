import torch
import torch.nn as nn
import math

class MultiHeadAttentionCustom(nn.Module):
    def __init__(self, embed_dim, num_heads=12, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads ==0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.Wq = nn.Linear(embed_dim,embed_dim)
        self.Wk = nn.Linear(embed_dim,embed_dim)
        self.Wv = nn.Linear(embed_dim,embed_dim)
        self.Wo = nn.Linear(embed_dim,embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):

        B,T,Q = x.size()
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        Q = Q.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        K = K.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        V = V.view(B,T,self.num_heads,self.head_dim).transpose(1,2)


        attention = Q @ K.transpose(-2,-1) / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            attention = attention.masked_fill(attn_mask == 0, float("-inf"))


        attention = torch.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        out = attention @ V
        
        out = out.transpose(1,2).contiguous().view(B,T,self.embed_dim)

        out = self.Wo(out)
        return out
    

class GPTBlock_Torch(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionCustom(embed_dim, num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,mask=None):

        x = x + self.attn(self.ln1(x), attn_mask=mask)
        x = x + self.ff(self.ln2(x))
        return x
