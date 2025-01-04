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

        self.to_q = nn.Sequential(norm(dim_head), nn.Linear(dim_head, dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim_head), nn.Linear(dim_head, dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim_head), nn.Linear(dim_head, dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, x, y):
        """
        x: (B, M, C)
        y: (B, N, C)
        """
        
        B, M, C = x.shape
        _, N, C = y.shape

        # Project with multiple heads
        q = self.to_q(x.reshape(B, M, self.heads, -1)).permute(0,2,1,3) # [B, heads, M, dim_head]
        k = self.to_k(y.reshape(B, N, self.heads, -1)).permute(0,2,1,3) # [B, heads, N, dim_head]
        v = self.to_v(y.reshape(B, N, self.heads, -1)).permute(0,2,1,3) # [B, heads, N, dim_head]

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('bhdn,bhen->bhde', q, k)  # [B, heads, M, N]  
        dot = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum("bhde,bhec->bhdc", dot, v) # [B, self.heads, M, dim_heads]
        a = a.permute(0, 2, 1, 3).reshape(B, M, self.heads * self.dim_head)
        z = self.proj(a)

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)  # [B, M, C]
    
        return z
    
class CrossHeightAttention(nn.Module):
    def __init__(self, input_dim, middle_dim, qkv_bias, norm=nn.LayerNorm):
        super(CrossHeightAttention, self).__init__()

        self.scale = middle_dim ** -0.5
        self.middle_dim = middle_dim

        self.to_q = nn.Sequential(norm(input_dim), nn.Linear(input_dim, middle_dim, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(input_dim), nn.Linear(input_dim, middle_dim, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(input_dim), nn.Linear(input_dim, middle_dim, bias=qkv_bias))

        # self.proj = nn.Linear(heads * dim_head, dim)
        # self.prenorm = norm(dim)
        # self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        # self.postnorm = norm(dim)

    def forward(self, x, y, v):
        """
        x: (B, M, C)
        y: (B, N, C)
        """
        
        B, M, C = x.shape
        _, N, C = y.shape

        # Project with multiple heads
        q = self.to_q(x) # [B, M, dim]
        k = self.to_k(y) # [B, N, dim]

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('bmd,bnd->bmn', q, k)  # [B, M, N] 
        dot = dot.softmax(dim=-1)

        # Combine values (image level features).
        z = torch.einsum("bmn,bnd->bmd", dot, v) # [B, M, 1]
    
        return z


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
    y = torch.gather(y, 2, u.long().view(B, 1, S*S).repeat(1, C * H, 1)).view(B, C, H, S*S)
    ys = y.permute(0, 3, 2, 1)  # [B, S^2, H, C]

    return ys
    

class CrossViewAttention(nn.Module):
    
    def __init__(self, blocks, dim, qkv_bias, norm=nn.LayerNorm) -> None:
        super(CrossViewAttention, self).__init__()
        meter_per_pixel = data_utils.get_meter_per_pixel()
        self.blocks = blocks
        self.bev_embed = nn.Conv2d(2, 256, 1)
        self.sat_embed = nn.Conv2d(1, 256, 1)
        self.sat_position = BEVEmbedding(256, 64, meter_per_pixel * (2 ** 3), 0)
        
        self.cross_attention_layers = nn.ModuleList()
        self.self_attention_layers = nn.ModuleList()
    
        self.cross_attention_layers.append(
            CrossAttention(dim, 4, 64, qkv_bias, norm)
        )
        self.cross_attention_layers.append(
            CrossAttention(dim, 4, 64, qkv_bias, norm)
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
        sat_embedding = self.sat_embed(sat_embedding)
                                                        
        query = sat_embedding + L2_norm(sat_position_embed) #[B 256 S S]
        
        query = query.flatten(2).permute(0, 2, 1)  # [B, S*S, C]
        key = grd_feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        value_last = grd_height.flatten(2).permute(0, 2, 1)  # [B, H*W, 1]

        output = self.cross_attention_layers[0](query, key)
        # output = self.cross_attention_layers[1](output, key)
        output = self.cross_attention_layers[2](output, key, value_last)
        output = output.permute(0, 2, 1).reshape(B, 1, S, S)

        # for layer in self.cross_attention_layers:
        #     x_attn = layer(x_attn, y_attn)
        
        # x_attn = x_attn.permute(0, 2, 1).reshape(B, C, S, S)
        
        return output
        
            
        