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
from gaussian.pano_splat import render_cuda
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
    confidence: Float[Tensor, "batch 1 height width"]

class GrdDecoder(nn.Module):
    def __init__(self) -> None:
        super(GrdDecoder, self).__init__()
        # background_color = torch.zeros(1, 3).float()
        self.register_buffer(
            "background_color",
            torch.zeros(1, 3).float(),
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
        render_out = render_cuda(
            extrinsics,
            intrinsics,
            near,
            far,
            image_shape,
            repeat(self.background_color[0], "c -> b c", b=b),
            gaussians.means,
            gaussians.covariances,
            gaussians.rgbs,
            gaussians.opacities,
            gaussians.features,
            gaussians.confidence,
        )
        color = render_out.color
        depth = render_out.depth
        feature = render_out.feature
        confidence = render_out.confidence
        out = DecoderOutput(
            color=color,
            depth=depth,
            feature=feature,
            confidence=confidence,
        )
        return out


    def last_layer_weights(self) -> Union[Tensor, None]:
        pass