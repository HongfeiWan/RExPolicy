"""One-step trainer for state-conditioned success-mode selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .common import finish_optimizer_step, module_device_dtype, positive_finite


@dataclass(frozen=True)
class SelectorStepMetrics:
    loss: float
    negative_log_likelihood: float
    gradient_norm: float
    optimizer_step: int


class Stage0SelectorTrainer:
    """Optimize selector NLL against latents from a frozen encoder."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        gradient_clip_norm: float = 1.0,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch module")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch optimizer")
        self.model = model
        self.optimizer = optimizer
        self.gradient_clip_norm = positive_finite(
            gradient_clip_norm,
            name="gradient_clip_norm",
        )
        self.optimizer_step = 0
        module_device_dtype(model)

    def step(
        self,
        current_states: torch.Tensor,
        target_latents: torch.Tensor,
    ) -> SelectorStepMetrics:
        self.model.train()
        device, dtype = module_device_dtype(self.model)
        current_states = current_states.to(device=device, dtype=dtype)
        target_latents = target_latents.detach().to(device=device, dtype=dtype)
        self.optimizer.zero_grad(set_to_none=True)
        distribution = self.model(current_states)
        loss = distribution.negative_log_likelihood(target_latents)
        gradient_norm = finish_optimizer_step(
            loss=loss,
            module=self.model,
            optimizer=self.optimizer,
            gradient_clip_norm=self.gradient_clip_norm,
        )
        self.optimizer_step += 1
        value = float(loss.detach())
        return SelectorStepMetrics(
            loss=value,
            negative_log_likelihood=value,
            gradient_norm=gradient_norm,
            optimizer_step=self.optimizer_step,
        )
