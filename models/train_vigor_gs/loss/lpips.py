from dataclasses import dataclass
from torch import nn

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from .loss import Loss


@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


def convert_to_buffer(module: nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)

class LossLpips():
    def __init__(self) -> None:
        super(LossLpips, self).__init__()
        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction,
        batch,
        gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        image = batch["target"]["image"]

        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=image.device)

        loss = self.lpips.forward(
            rearrange(prediction.color, "b v c h w -> (b v) c h w"),
            rearrange(image, "b v c h w -> (b v) c h w"),
            normalize=True,
        )
        return self.cfg.weight * loss.mean()
