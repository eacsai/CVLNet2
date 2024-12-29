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
        b, _, _ = extrinsics.shape
        color_sh = gaussians.color_harmonics
        render_out = render_cuda(
            extrinsics,
            intrinsics,
            near,
            far,
            image_shape,
            self.background_color.repeat(b, 1),
            gaussians.means,
            gaussians.covariances,
            color_sh,
            gaussians.opacities,
            gaussians.features,
        )
        out = DecoderOutput(
            color=render_out.color,
            depth=render_out.depth,
            feature=render_out.feature,
        )
        return out


    def last_layer_weights(self) -> Union[Tensor, None]:
        pass