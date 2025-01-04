import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .conversions import relative_disparity_to_depth
from .distribution_sampler import DistributionSampler


class DepthPredictorMonocular(nn.Module):
    projection: nn.Sequential
    sampler: DistributionSampler
    num_samples: int
    num_surfaces: int

    def __init__(
        self,
        d_in: int = 128,
    ) -> None:
        super().__init__()

        self.sampler = DistributionSampler()
        self.num_samples = 32
        self.num_surfaces = 1

        # This exists for hooks to latch onto.
        self.to_pdf = nn.Softmax(dim=-1)
        self.to_offset = nn.Sigmoid()
        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_in, 2 * self.num_samples * self.num_surfaces),
        )
    def forward(
        self,
        features: Float[Tensor, "batch view ray channel"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        deterministic: bool = False,
        gaussians_per_pixel: int = 3,
    ) -> tuple[
        Float[Tensor, "batch view ray surface sample"],  # depth
        Float[Tensor, "batch view ray surface sample"],  # pdf
    ]:
        s = self.num_samples

        # Convert the features into a depth distribution plus intra-bucket offsets.
        features = self.projection(features)
        pdf_raw, offset_raw = rearrange(
            features, "... (dpt c) -> c ... dpt", c=2
        )
        pdf = self.to_pdf(pdf_raw)
        offset = self.to_offset(offset_raw)

        # Sample from the depth distribution.
        index, pdf_i = self.sampler.sample(pdf, deterministic, gaussians_per_pixel)
        offset = self.sampler.gather(index, offset)

        # Convert the sampled bucket and offset to a depth.
        relative_disparity = (index + offset) / s
        depth = relative_disparity_to_depth(
            relative_disparity,
            rearrange(near, "b v -> b v () ()"),
            rearrange(far, "b v -> b v () ()"),
        )

        opacity = pdf_i

        return depth, opacity
