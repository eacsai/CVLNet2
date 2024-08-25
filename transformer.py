import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from visualize import *

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

class Multi_Head_CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super(Multi_Head_CrossAttention, self).__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, x, y):
        """
        x: (B, M, C)
        y: (B, M, N, C)
        """
        
        B, M, C = x.shape
        _, M, N, C = y.shape

        # Project with multiple heads
        q = self.to_q(x).reshape(B, M, 1, self.heads, self.dim_head).permute(0, 3, 1, 2, 4) # [B, heads, M, 1, dim_head]
        k = self.to_k(y).reshape(B, M, N, self.heads, self.dim_head).permute(0, 3, 1, 2, 4) # [B, heads, M, N, dim_head]
        v = self.to_v(y).reshape(B, M, N, self.heads, self.dim_head).permute(0, 3, 1, 2, 4) # [B, heads, M, N, dim_head]

        # Dot product attention along cameras
        dot = self.scale * torch.matmul(q, k.transpose(-1, -2)).reshape(B, self.heads, M, N, 1)  
        dot = dot.softmax(dim=-2)

        # Combine values (image level features).
        a = torch.sum(dot * v, dim=-2) # [B, self.heads, M, dim_heads]
        a = a.permute(0, 2, 1, 3).reshape(B, M, self.heads * self.dim_head)
        z = self.proj(a)

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)  # [B, M, C]
    
        return z
    
class CrossAttention(nn.Module):
    def __init__(self, dim, qkv_bias, norm=nn.LayerNorm, max_len=5000):
        super(CrossAttention, self).__init__()

        self.scale = dim ** -0.5

        self.key = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.query = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.pe = PositionalEncoding(dim, max_len)
        # self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, dim, bias=qkv_bias))

        # self.prenorm = norm(dim)
        # self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        # self.postnorm = norm(dim)

    def forward(self, x, y, z, threshold=0.3):
        """
        x: (B, C, A, A)
        y: (B, C, H, W)
        z: (B, 1, H, W)
        """

        _, _, _, A = x.shape
        B, C, H, W = y.shape

        x = self.pe(x.view(B, -1, C)).view(B, C, A, A)
        y = self.pe(y.view(B, -1, C)).view(B, C, H, W)

        # Project with multiple heads
        q = self.query(x).view(B, -1, C) # [B, M, dim]
        k = self.key(y).view(B, -1, C) # [B, N, dim]

        

        v = z.view(B, -1, 1) # [B, N, 1]

        # Dot product attention along cameras
        dot = self.scale * torch.matmul(q, k.transpose(-1, -2)) # [B, M, N]
        # dot = nn.LayerNorm(dot.shape[-1]).to(dot.device)(dot)
        dot = dot.softmax(dim=-1)

        res = torch.matmul(dot, v) # [B, M, 1]
        # Combine values (image level features).
        # new_res = torch.matmul(dot, v) # [B, M, 1]
        # res = torch.zeros_like(new_res)
        # confidence = dot.max(dim=-1, keepdim=True)[0]
        # res = torch.where(confidence > threshold, new_res, res)
        return res

