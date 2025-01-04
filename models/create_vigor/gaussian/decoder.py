from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, Union

from jaxtyping import Float
from torch import Tensor, nn

import torch
from einops import rearrange, repeat

from .diagonal_gaussian_distribution import DiagonalGaussianDistribution
from .encoder import Gaussians
# from gaussian.nopo_cuda_splatting import render_cuda
from gaussian.latent_splat import render_cuda
DepthRenderingMode = Literal[
    "depth",
    "log",
    "disparity",
    "relative_disparity",
]

@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch 3 height width"]
    depth: Float[Tensor, "batch height width"]
    feature: Float[Tensor, "batch channels height width"]

class GrdDecoder(nn.Module):
    def __init__(self) -> None:
        super(GrdDecoder, self).__init__()
        background_color = torch.zeros(1, 3).float()
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )
    
    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        color_sh = gaussians.color_harmonics
        render_out = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color[0], "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.color_harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            repeat(gaussians.features, "b g c -> (b v) g c", v=v),
        )
        color = rearrange(render_out.color, "(b v) c h w -> b v c h w", b=b, v=v)
        depth = rearrange(render_out.depth, "(b v) h w -> b v h w", b=b, v=v)
        feature = rearrange(render_out.feature, "(b v) c h w -> b v c h w", b=b, v=v)
        out = DecoderOutput(
            color=color,
            depth=depth,
            feature=feature,
        )
        return out


    def last_layer_weights(self) -> Union[Tensor, None]:
        pass