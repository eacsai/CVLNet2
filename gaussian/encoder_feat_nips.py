from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Optional, Union

import torch, torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from .build_gaussians import get_world_rays
from .build_gaussians import build_covariance

from backbone.backbone_dino_nips import BackboneDino
from depth_predictor.depth_predictor_monocular import DepthPredictorMonocular
from gaussian.diagonal_gaussian_distribution import DiagonalGaussianDistribution
from gaussian.build_gaussians import sample_image_grid
from gaussian.gaussian_adapter_feat import GaussianAdapter

@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    opacities: Float[Tensor, "batch gaussian"]
    features: Float[Tensor, "batch gaussian dim"]
    confidence: Float[Tensor, "batch gaussian 1"]
    rgbs: Float[Tensor, "batch gaussian 3"]

class GaussianFeatEncoder(nn.Module):
    def __init__(self, gs_dim=11) -> None:
        super(GaussianFeatEncoder, self).__init__()
        self.backbone = BackboneDino()
        self.backbone_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.backbone.d_out, 128),
        )

        self.gpv = 3
        self.to_opacity = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.to_gaussians = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                128,
                gs_dim*self.gpv,
            ),
        )

        self.pos_act = nn.Tanh()
        self.scale_act = nn.Sigmoid()
        self.opacity_act = nn.Sigmoid()
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        
        # High resolution skip only required in case of now downscaling
        self.high_resolution_skip = nn.Sequential(
            nn.Conv2d(3, 128, 7, 1, 3),
            nn.ReLU(),
        )

        self.offset_max = [0.3] * 3
        self.scale_max = [0.3] * 3

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
    ) -> Float[Tensor, " *batch"]:
        exponent = 1.0
        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        img: Float[Tensor, "batch view channels height width"],
        grd_depth: Float[Tensor, "batch height width"],
        grd_feat: Float[Tensor, "batch view channels height width"],
        grd_conf: Float[Tensor, "batch view channels height width"],
        camera_k: Float[Tensor, "batch view 3 3"],
        extrinsics: Float[Tensor, "batch view 4 4"],
    ) -> Gaussians:
        b, v, _, h, w = img.shape
        features = self.backbone(img)
        device = features.device
        h, w = features.shape[-2:]
        features = rearrange(features, "b v c h w -> b v h w c").contiguous()
        features = self.backbone_projection(features)
        features = rearrange(features, "b v h w c -> b v c h w").contiguous()

        if self.high_resolution_skip is not None:
            # Add the high-resolution skip connection.
            skip = rearrange(img, "b v c h w -> (b v) c h w")
            skip = self.high_resolution_skip(skip)
            features = features + rearrange(skip, "(b v) c h w -> b v c h w", b=b, v=v)

        # Sample depths from the resulting features.
        features = rearrange(features, "b v c h w -> b v (h w) c")

        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        origins, directions = get_world_rays(xy_ray, extrinsics[:, :, None, None, :, :], camera_k[:, :, None, None, :, :])
        grd_depth = F.interpolate(grd_depth[:,None,:,:], (h, w))
        grd_depth = rearrange(grd_depth, "b c h w -> b (h w) c").contiguous()
        means = origins + directions * grd_depth[:, None, :, None, :]
        
        gaussians = self.to_gaussians(features)
        gaussians = gaussians.view(b, v, h*w, self.gpv, -1)

        gs_offsets_x = self.pos_act(gaussians[..., :1]) * self.offset_max[0]
        gs_offsets_y = self.pos_act(gaussians[..., 1:2]) * self.offset_max[1]
        gs_offsets_z = self.pos_act(gaussians[..., 2:3]) * self.offset_max[1]

        opacities = self.opacity_act(gaussians[..., 3:4]).squeeze(-1)
        # opacities = torch.ones_like(gaussians[..., 0], device=gaussians.device).float()
        rotations = self.rot_act(gaussians[..., 4:8])
        scale_x = self.scale_act(gaussians[..., 8:9]) * self.scale_max[0]
        scale_y = self.scale_act(gaussians[..., 9:10]) * self.scale_max[1]
        scale_z = self.scale_act(gaussians[..., 10:11]) * self.scale_max[2]
        scales = torch.cat([scale_x, scale_y, scale_z], dim=-1)
        offset_xyz = torch.cat([gs_offsets_x, gs_offsets_y, gs_offsets_z], dim=-1)
        means = means + offset_xyz
        covariances = build_covariance(scales, rotations)

        gs_features = rearrange(grd_feat, "batch view channels height width -> batch view (height width) channels").unsqueeze(-2)
        gs_confidences = rearrange(grd_conf, "batch view channels height width -> batch view (height width) channels").unsqueeze(-2)
        gs_rgbs = rearrange(img, "batch view channels height width -> batch view (height width) channels").unsqueeze(-2)

        gs_features = gs_features.broadcast_to((*opacities.shape, 32))
        gs_confidences = gs_confidences.broadcast_to((*opacities.shape, 1))
        gs_rgbs = gs_rgbs.broadcast_to((*opacities.shape, 3))
        # Dump visualizations if needed.
        # if visualization_dump is not None:
        #     visualization_dump["depth"] = rearrange(
        #         depths, "b (h w) s -> b h w s", h=h, w=w
        #     )
        #     visualization_dump["scales"] = rearrange(
        #         gaussians.scales, "b r spp xyz -> b (r spp) xyz"
        #     )
        #     visualization_dump["rotations"] = rearrange(
        #         gaussians.rotations, "b r spp xyzw -> b (r spp) xyzw"
        #     )

        # Optionally apply a per-pixel opacity.
        # opacity_multiplier = (
        #     rearrange(self.to_opacity(features), "b v r () -> b v r () ()")
        #     if self.cfg.predict_opacity
        #     else 1
        # )

        return Gaussians(
            rearrange(
                means,
                "b v r spp xyz -> b (v r spp) xyz",
            ),
            rearrange(
                covariances,
                "b v r spp i j -> b (v r spp) i j",
            ),
            rearrange(
                opacities,
                "b v r spp -> b (v r spp)",
            ),
            rearrange(
                gs_features,
                "b v r spp c -> b (v r spp) c",
            ),
            rearrange(
                gs_confidences,
                "b v r spp c -> b (v r spp) c",
            ),
            rearrange(
                gs_rgbs,
                "b v r spp c -> b (v r spp) c",
            )
        )

    @property
    def last_layer_weights(self) -> Tensor:
        return self.to_gaussians[-1].weight
