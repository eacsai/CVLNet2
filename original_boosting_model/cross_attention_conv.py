from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck

from VGG import L2_norm
import data_utils

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)

def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, scale=0.2, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """

    return [
        [ 0., scale,          w/2.],
        [scale,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]



class BEVEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        A: int,
        scale: float,
        offset: int,
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()
        # bev coordinates
       # meshgrid the sat pannel
        i = j = torch.arange(0, A).cuda()  # to(self.device)
        ii, jj = torch.meshgrid(i, j)  # i:h,j:w

        # uv is coordinate from top/left, v: south, u:east
        uv = torch.stack([jj, ii], dim=-1).float()  # shape = [satmap_sidelength, satmap_sidelength, 2]

        # sat map from top/left to center coordinate
        u0 = v0 = A // 2
        uv_center = uv - torch.tensor(
            [u0, v0]).cuda()  # .to(self.device) # shape = [satmap_sidelength, satmap_sidelength, 2]

        # affine matrix: scale*R
        meter_per_pixel = data_utils.get_meter_per_pixel()
        meter_per_pixel *= data_utils.get_process_satmap_sidelength() / A
        R = torch.tensor([[0, 1], [1, 0]]).float().cuda()  # to(self.device) # u_center->z, v_center->x
        Aff_sat2real = meter_per_pixel * R  # shape = [2,2]

        # Trans matrix from sat to realword
        XZ = torch.einsum('ij, hwj -> hwi', Aff_sat2real,
                          uv_center).permute(2,0,1)  # shape = [2, satmap_sidelength, satmap_sidelength]                   # 3 h w

        # egocentric frame
        self.register_buffer('grid', XZ, persistent=False)                    # 2 h w

class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super(CrossAttention, self).__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head
        n_embd = heads * dim_head
        self.n_embd = n_embd
        self.to_q = nn.Sequential(
            nn.Conv2d(dim, n_embd, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(n_embd, n_embd, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.to_k = nn.Sequential(
            nn.Conv2d(dim, n_embd, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(n_embd, n_embd, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.to_v = nn.Sequential(
            nn.Conv2d(dim, n_embd, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(n_embd, n_embd, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        # self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        # self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        # self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Sequential(
            nn.Conv2d(n_embd, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
        )

    def forward(self, x, y, u):
        """
        x: (B, C, S, S)
        y: (B, C, H, W)
        """
        
        B, C, S, _ = x.shape
        _, _, H, W = y.shape

        # Project with multiple heads
        q = self.to_q(x).view(B, self.n_embd, S * S).permute(0, 2, 1).unsqueeze(-2) # [B, S^2, 1, C] 
        k = self.to_k(y) # [B, C, H, W]
        v = self.to_v(y) # [B, C, H, W]

        k = generate_y_for_attn(S, k, u) # [B, S^2, H, C]
        v = generate_y_for_attn(S, v, u) # [B, S^2, H, C]
        # Dot product attention along cameras
        dot = self.scale * torch.matmul(q, k.transpose(-1, -2))  # [B, S^2, H, C]
        dot = dot.transpose(-2,-1).softmax(dim=-2)

        # Combine values (image level features).
        a = torch.sum(dot * v, dim=-2) # [B, S^2, C]
        a = a.permute(0, 2, 1).view(B, self.n_embd, S, S)
        z = self.proj(a)
    
        return z
    
class CrossHeightAttention(nn.Module):
    def __init__(self, input_dim, middle_dim, qkv_bias, norm=nn.LayerNorm):
        super(CrossHeightAttention, self).__init__()

        self.scale = middle_dim ** -0.5
        self.middle_dim = middle_dim

        self.to_q = nn.Sequential(
            nn.Conv2d(input_dim, middle_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(middle_dim, middle_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.to_k = nn.Sequential(
            nn.Conv2d(input_dim, middle_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(middle_dim, middle_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        # self.to_q = nn.Sequential(norm(input_dim), nn.Linear(input_dim, middle_dim, bias=qkv_bias))
        # self.to_k = nn.Sequential(norm(input_dim), nn.Linear(input_dim, middle_dim, bias=qkv_bias))

        # self.proj = nn.Linear(heads * dim_head, dim)
        # self.prenorm = norm(dim)
        # self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        # self.postnorm = norm(dim)

    def forward(self, x, y, z, u):
        """
        x: (B, C, S, S)
        y: (B, C, H, W)
        """
        
        B, C, S, _ = x.shape
        _, _, H, W = y.shape

        # Project with multiple heads
        q = self.to_q(x).view(B, self.middle_dim, S * S).permute(0, 2, 1).unsqueeze(-2) # [B, S^2, 1, C]
        k = self.to_k(y) # [B, C, H, W]
        v = z # [B, M, N, 1]

        k = generate_y_for_attn(S, k, u) # [B, S^2, H, C]
        v = generate_y_for_attn(S, v, u) # [B, S^2, H, C]
        # Dot product attention along cameras
        dot = self.scale * torch.matmul(q, k.transpose(-1, -2))  # [B, S^2, H, C]
        dot = dot.transpose(-2,-1).softmax(dim=-2)

        # Combine values (image level features).
        a = torch.sum(dot * v, dim=-2) # [B, S^2, 1]
    
        return a


def generate_y_for_attn(S, y, u):
    '''

    y.shape = [B, C, H, W]
    uv.shape = [B, S, S]

    return:

    ys.shape = [B, S^2, H, C]
    '''
    # B, C, S, _ = x.shape
    # x = x.reshape(B, C, S * S).permute(0, 2, 1)

    B, C, H, W = y.shape

    with torch.no_grad():
        torch.clamp(u, 0, W - 1, out=u)

    y = y.reshape(B, C * H, W)
    y = torch.gather(y, 2, u.long().view(B, 1, S*S).repeat(1, C * H, 1)).view(B, C, H, S*S)
    ys = y.permute(0, 3, 2, 1)  # [B, S^2, H, C]

    return ys
    

class CrossViewAttention(nn.Module):
    
    def __init__(self, blocks, dim, qkv_bias, norm=nn.LayerNorm) -> None:
        super(CrossViewAttention, self).__init__()
        meter_per_pixel = data_utils.get_meter_per_pixel()
        self.blocks = blocks
        self.bev_embed = nn.Conv2d(2, 256, 1)
        self.sat_position = BEVEmbedding(256, 64, meter_per_pixel * (2 ** 3), 0)
        self.k_attention = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1, bias=False)
        )

        self.q_attention = nn.Sequential(
            nn.Conv2d(1, dim, 1, bias=False)
        )
        
        self.cross_attention_layers = nn.ModuleList()
        self.self_attention_layers = nn.ModuleList()
    
        self.cross_attention_layers.append(
            CrossAttention(dim, 4, 16, qkv_bias, norm)
        )
        self.cross_attention_layers.append(
            CrossAttention(dim, 4, 16, qkv_bias, norm)
        )
        self.cross_attention_layers.append(
            CrossHeightAttention(dim, 64, qkv_bias, norm)
        )
    
    def forward(self, sat_embedding, grd_feat, grd_height, u, valid_index):
        '''
         grd2sat.shape = [B, C, S, S]
         grd_x.shape = [B, C, H, W]
         grd_height.shape = [B, 1, H, W]
         u.shape = [B, S, S]
         valid_index = [B, S^2, 1]
        '''

        B, C, _, _ = grd_feat.shape
        _, _, S, _ = sat_embedding.shape
        
        sat_position = self.sat_position.grid[:2]                                                    # 2 H W
        sat_position_embed = self.bev_embed(sat_position[None]).repeat(B, 1, 1, 1)                  # B 256 H W 
        
        query = sat_embedding + L2_norm(sat_position_embed)
        # query = sat_embedding
        # key_feat = self.k_attention(grd_feat)
        # value_feat = self.v_attention(grd_feat)
        # key = generate_y_for_attn(S, key_feat, u) # [B, S^2, H, C]
        key = grd_feat
        # value = generate_y_for_attn(S, value_feat, u)

        value_last = grd_height # [B, S^2, H, 1]
        # query = query.reshape(B, C, S * S).permute(0, 2, 1) # [B, S^2, C]
        
        # output = self.cross_attention_layers[0](query, key, u)
        # output = self.cross_attention_layers[1](output, key, u)
        output = self.cross_attention_layers[2](query, key, value_last, u)
        output = output.permute(0, 2, 1).reshape(B, 1, S, S)

        # for layer in self.cross_attention_layers:
        #     x_attn = layer(x_attn, y_attn)
        
        # x_attn = x_attn.permute(0, 2, 1).reshape(B, C, S, S)
        
        return output
        
            
        