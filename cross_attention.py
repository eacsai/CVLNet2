from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from VGG import L2_norm


class CrossAttention(nn.Module):
    def __init__(self, input_dim, middle_dim, qkv_bias, norm=nn.LayerNorm):
        super(CrossAttention, self).__init__()

        self.scale = middle_dim ** -0.5
        self.middle_dim = middle_dim

        self.to_q = nn.Sequential(norm(input_dim), nn.Linear(input_dim, middle_dim, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(input_dim), nn.Linear(input_dim, middle_dim, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(input_dim), nn.Linear(input_dim, middle_dim, bias=qkv_bias))

        # self.proj = nn.Linear(heads * dim_head, dim)
        # self.prenorm = norm(dim)
        # self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        # self.postnorm = norm(dim)

    def forward(self, x, y, z):
        """
        x: (B, M, C)
        y: (B, M, N, C)
        z: (B, M, N, 1)
        """
        
        B, M, C = x.shape
        _, _, N, _ = y.shape

        # Project with multiple heads
        q = self.to_q(x).reshape(B, M, 1, self.middle_dim) # [B, M, 1, middle_head]
        k = self.to_k(y).reshape(B, M, N, self.middle_dim) # [B, M, N, middle_head]
        v = z # [B, M, N, 1]

        # Dot product attention along cameras
        dot = self.scale * torch.matmul(q, k.transpose(-1, -2)).reshape(B, M, N, 1)  
        dot = dot.softmax(dim=-2)

        # Combine values (image level features).
        a = torch.sum(dot * v, dim=-2) # [B, self.heads, M, dim_heads]
    
        return a


def generate_xy_for_attn(x, y, u):
    '''
    x.shape = [B, C, S, S]
    y.shape = [B, C, H, W]
    uv.shape = [B, S, S]
    
    return:
    x.shape = [B, S^2, C]
    ys.shape = [B, S^2, 2H, C]
    '''
    B, C, S, _ = x.shape
    x = x.reshape(B, C, S*S).permute(0, 2, 1)
    
    _, C, H, W = y.shape
    
    with torch.no_grad():
        u_left = torch.floor(u)
        u_right = u_left + 1
        
        torch.clamp(u_left, 0, W -1, out=u_left)
        torch.clamp(u_right, 0, W -1, out=u_right)
    
    y = y.reshape(B, C*H, W)
    y_left = torch.gather(y, 2, u_left.long().view(B, 1, S*S).repeat(1, C*H, 1)).view(B, C, H, S*S)
    y_right = torch.gather(y, 2, u_right.long().view(B, 1, S*S).repeat(1, C*H, 1)).view(B, C, H, S*S)
    ys = torch.cat([y_left, y_right], dim=2).permute(0, 3, 2, 1)  # [B, S^2, 2H, C]
    
    return x, ys


def generate_y_for_attn(S, y, u):
    '''

    y.shape = [B, C, H, W]
    uv.shape = [B, S, S]

    return:

    ys.shape = [B, S^2, 2H, C]
    '''
    # B, C, S, _ = x.shape
    # x = x.reshape(B, C, S * S).permute(0, 2, 1)

    B, C, H, W = y.shape

    with torch.no_grad():
        torch.clamp(u, 0, W - 1, out=u)

    y = y.reshape(B, C * H, W)
    y = torch.gather(y, 2, u.long().view(B, 1, S * S).repeat(1, C * H, 1)).view(B, C, H, S * S)
    ys = y.permute(0, 3, 2, 1)  # [B, S^2, H, C]

    return ys
    

class CrossViewAttention(nn.Module):
    
    def __init__(self, blocks, dim, qkv_bias, norm=nn.LayerNorm) -> None:
        super(CrossViewAttention, self).__init__()

        self.blocks = blocks

        self.cross_attention_layers = nn.ModuleList()
        self.self_attention_layers = nn.ModuleList()
        for _ in range(blocks):
            self.cross_attention_layers.append(
                CrossAttention(dim, 32, qkv_bias, norm)
            )

    
    
    def forward(self, sat_x, grd_x, grd_height, u):
        '''
         grd2sat.shape = [B, C, S, S]
         grd_x.shape = [B, C, H, W]
         grd_height.shape = [B, 1, H, W]
         u.shape = [B, S, S]
        '''
        B, C, S, _ = sat_x.shape

        x_attn = sat_x
        y_attn = generate_y_for_attn(S, grd_x, u)
        v_attn = generate_y_for_attn(S, grd_height, u)

        x_attn = x_attn.reshape(B, C, S * S).permute(0, 2, 1)
        x_attn = self.cross_attention_layers[0](x_attn, y_attn, v_attn)
        x_attn = x_attn.permute(0, 2, 1).reshape(B, 1, S, S)

        # for layer in self.cross_attention_layers:
        #     x_attn = layer(x_attn, y_attn)
        
        # x_attn = x_attn.permute(0, 2, 1).reshape(B, C, S, S)
        
        return x_attn
        
            
        