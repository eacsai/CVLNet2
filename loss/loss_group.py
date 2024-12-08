from typing import Dict, Tuple, Union

from jaxtyping import Float
import torch
from torch import nn, Tensor

from .loss import Loss, LossValue
from .loss_generator import LossGenerator
from .loss_discriminator import LossDiscriminator


from typing import Any, Dict, Union


RecursiveStringDict = Dict[str, Union[Any, 'RecursiveStringDict']]

def flatten_nested_string_dict(
    rsd: RecursiveStringDict, 
    prefix: str | None = None
) -> Dict[str, Any]:
    res = {}
    for key, value in rsd.items():
        flattened_key = f"{prefix}/{key}" if prefix is not None else key
        if isinstance(value, dict):
            res.update(flatten_nested_string_dict(value, prefix=flattened_key))
        else:
            res[flattened_key] = value
    return res

class LossGroup(nn.Module):
    def __init__(
        self,
        name: str,
        nll_losses: list[Union[Loss, 'LossGroup']] = [],
        generator_loss: LossGenerator | None = None,
        discriminator_loss: LossDiscriminator | None = None
    ) -> None:
        super().__init__()
        self.name = name
        self.nll_losses = nn.ModuleList(nll_losses)
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss

    @staticmethod
    def get_adaptive_weight(
        nll_loss: Float[Tensor, ""], 
        g_loss: Float[Tensor, ""], 
        last_layer_weights: Tensor
    ):
        nll_grads = torch.autograd.grad(nll_loss, last_layer_weights, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer_weights, retain_graph=True)[0]

        weight = torch.linalg.norm(nll_grads) / (torch.linalg.norm(g_grads) + 1e-4)
        weight = torch.clamp(weight, 0.0, 1.0).detach()
        return weight

    def forward(
        self,
        prediction,
        gt,
        global_step: int = 0,
    ) -> Tuple[
        Float[Tensor, ""] | int,
        Dict[str, LossValue]
    ]:
        return self.forward_generator(prediction, gt, global_step)

    def forward_generator(
        self,
        prediction,
        gt,
        global_step: int = 0,
        last_layer_weights: Tensor | None = None
    ) -> Tuple[
        Float[Tensor, ""] | int,
        Dict[str, LossValue]
    ]:
        losses: Dict[str, LossValue] = flatten_nested_string_dict(
            {l.name: l(prediction, gt, global_step) for l in self.nll_losses},
            prefix=self.name
        )
        total_loss = sum(l.weighted for l in losses.values())
        if self.is_generator_loss_active(global_step):
            generator_loss = self.generator_loss(prediction, gt, global_step)
            adaptive_weight = self.get_adaptive_weight(total_loss, generator_loss.unweighted, last_layer_weights)
            generator_loss.weighted = adaptive_weight * generator_loss.weighted
            total_loss = total_loss + generator_loss.weighted
            losses[f"{self.name}/{self.generator_loss.name}"] = generator_loss
        return total_loss, losses
    
    def forward_discriminator(
        self,
        prediction,
        gt,
        global_step: int
    ) -> Tuple[
        Float[Tensor, ""] | int,
        Dict[str, LossValue]
    ]:
        losses: Dict[str, LossValue] = flatten_nested_string_dict(
            self.discriminator_loss(prediction, gt, global_step),
            prefix=self.name
        )
        total_loss = sum(l.weighted for l in losses.values())
        return total_loss, losses

    def is_active(self, global_step: int) -> bool:
        return any(l.is_active(global_step) for l in self.nll_losses) or \
            self.is_generator_loss_active(global_step) or \
                self.is_discriminator_loss_active(global_step)

    @property
    def has_generator_loss(self) -> bool:
        return self.generator_loss is not None

    @property
    def has_discriminator_loss(self) -> bool:
        return self.discriminator_loss is not None

    def is_generator_loss_active(self, global_step: int) -> bool:
        return self.has_generator_loss and self.generator_loss.is_active(global_step)
    
    def is_discriminator_loss_active(self, global_step: int) -> bool:
        return self.has_discriminator_loss and self.discriminator_loss.is_active(global_step)
