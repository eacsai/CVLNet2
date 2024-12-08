from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

import torch
from einops import rearrange, repeat

from .diagonal_gaussian_distribution import DiagonalGaussianDistribution
from .encoder import Gaussians
from gaussian.latent_splat import render_cuda, RenderOutput

DepthRenderingMode = Literal[
    "depth",
    "log",
    "disparity",
    "relative_disparity",
]

@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch view 3 height width"] | None
    feature_posterior: DiagonalGaussianDistribution | None
    mask: Float[Tensor, "batch view height width"]
    depth: Float[Tensor, "batch view height width"]

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
        return_colors: bool = True,
        return_features: bool = True
    ) -> DecoderOutput:
        b, _, _ = extrinsics.shape
        color_sh = gaussians.color_harmonics \
            if return_colors and gaussians.color_harmonics is not None else None
        feature_sh = gaussians.feature_harmonics \
            if return_features and gaussians.feature_harmonics is not None else None
        rendered: RenderOutput = render_cuda(
            extrinsics,
            intrinsics,
            near,
            far,
            image_shape,
            self.background_color.repeat(b, 1),
            gaussians.means,
            gaussians.covariances,
            gaussians.opacities,
            color_sh,
            feature_sh
        )
        out = self.render_to_decoder_output(rendered, b)
        return out


    def last_layer_weights(self) -> Tensor | None:
        pass

    def render_to_decoder_output(
        self,
        render_output: RenderOutput,
        b: int,
    ) -> DecoderOutput:
        if render_output.feature is not None:
            features = render_output.feature
            # NOTE background feature = 0 = mean = logvar (of normal distribution)
            mean, logvar = (features, (1-rearrange(render_output.mask.detach(), "b h w -> b () h w", b=b)).log().expand_as(features))
            feature_posterior = DiagonalGaussianDistribution(mean, logvar)
        else:
            feature_posterior = None
        return DecoderOutput(
            color=render_output.color if render_output.color is not None else None,
            feature_posterior=feature_posterior,
            mask=render_output.mask,
            depth=render_output.depth
        )