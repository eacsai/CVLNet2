from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Optional, Union

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn


from backbone.backbone_dino import BackboneDino
from depth_predictor.depth_predictor_monocular import DepthPredictorMonocular
from gaussian.diagonal_gaussian_distribution import DiagonalGaussianDistribution
from gaussian.build_gaussians import sample_image_grid
from gaussian.gaussian_adapter import GaussianAdapter

@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    opacities: Float[Tensor, "batch gaussian"]
    color_harmonics: Float[Tensor, "batch gaussian 3 color_d_sh"]
    features: Float[Tensor, "batch gaussian dim"]

class GaussianEncoder(nn.Module):
    def __init__(self) -> None:
        super(GaussianEncoder, self).__init__()
        self.backbone = BackboneDino()
        self.backbone_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.backbone.d_out, 128),
        )
        self.depth_predictor = DepthPredictorMonocular()
        self.gaussian_adapter = GaussianAdapter(n_feature_channels=64)
        
        self.to_opacity = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.to_gaussians = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                128,
                84,
            ),
        )
        self.to_gaussians_feat = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                128,
                148,
            ),
        )
        # High resolution skip only required in case of now downscaling
        self.high_resolution_skip = nn.Sequential(
            nn.Conv2d(3, 128, 7, 1, 3),
            nn.ReLU(),
        )

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
    ) -> Float[Tensor, " *batch"]:
        exponent = 1.0
        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        img: Float[Tensor, "batch channels height width"],
        grd_feat: Union[Float[Tensor, "batch channels height width"] | None],
        camera_k: Float[Tensor, "batch 3 3"],
        extrinsics: Float[Tensor, "batch 4 4"],
        global_step: int,
        near: Float[Tensor, "batch"],
        far: Float[Tensor, "batch"],
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        variational: bool = True,
    ) -> Gaussians:
        b = img.shape[:1]
        features = self.backbone(img)
        device = features.device
        h, w = features.shape[-2:]
        features = rearrange(features, "b c h w -> b h w c").contiguous()
        features = self.backbone_projection(features)
        features = rearrange(features, "b h w c -> b c h w").contiguous()

        if self.high_resolution_skip is not None:
            # Add the high-resolution skip connection.
            skip = self.high_resolution_skip(img)
            features = features + skip

        # Sample depths from the resulting features.
        features = rearrange(features, "b c h w -> b (h w) c")
        depths, densities = self.depth_predictor.forward(
            features,
            near,
            far,
            deterministic,
            1 if deterministic else 3,
        )

        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        if grd_feat is not None:
            gaussians = self.to_gaussians(features).unsqueeze(-2)
        else:
            gaussians = self.to_gaussians_feat(features).unsqueeze(-2)
        
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gpp = 3
        gaussians = self.gaussian_adapter.forward(
            extrinsics,
            camera_k,
            xy_ray,
            depths,
            self.map_pdf_to_opacity(densities) / gpp,
            gaussians[..., 2:],
            grd_feat,
            (h, w),
        )

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
                gaussians.means,
                "b r spp xyz -> b (r spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b r spp i j -> b (r spp) i j",
            ),
            rearrange(
                gaussians.opacities,
                "b r spp -> b (r spp)",
            ),
            rearrange(
                gaussians.color_harmonics,
                "b r spp c d_c_sh -> b (r spp) c d_c_sh",
            ),
            rearrange(
                gaussians.features,
                "b r spp c -> b (r spp) c",
            )
        )

    @property
    def last_layer_weights(self) -> Tensor:
        return self.to_gaussians[-1].weight
