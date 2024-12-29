import torch
from jaxtyping import Float
from torch import Tensor, nn
from backbone.backbone_dino import BackboneDino
from einops import rearrange
from models.VGGW import L2_norm

class DinoFeat(nn.Module):
    def __init__(self, n_feature_channels: int = 64):
        super(DinoFeat, self).__init__()
        self.dino_feat = BackboneDino()
        self.dino_feat_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.dino_feat.d_out, n_feature_channels),
        )

    def forward(
        self,
        x: Float[Tensor, "batch 3 H W"],
    ) -> Float[Tensor, "batch n_feature_channels H W"]:
        x = self.dino_feat(x)
        h, w = x.shape[-2:]
        x = rearrange(x, "b c h w -> b h w c").contiguous()
        x = self.dino_feat_projection(x)
        x = rearrange(x, "b h w c -> b c h w").contiguous()
        return L2_norm(x)