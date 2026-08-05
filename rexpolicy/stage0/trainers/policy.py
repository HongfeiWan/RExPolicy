"""One-step flow-matching trainer for the Stage 0 policy runtime."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .batch import Stage0WindowBatch
from .common import finish_optimizer_step, module_device_dtype, positive_finite


@dataclass(frozen=True)
class PolicyStepMetrics:
    loss: float
    flow_matching: float
    gradient_norm: float
    optimizer_step: int


class Stage0PolicyTrainer:
    """Optimize a ``Stage0FlowPolicy`` or compatible caller-wrapped module.

    The trainer never creates DDP. For distributed training the caller wraps
    the complete policy with ``find_unused_parameters=False`` and its
    ``forward`` must delegate to ``flow_matching_loss``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        gradient_clip_norm: float = 1.0,
        action_feature_mask: torch.Tensor | None = None,
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
        if action_feature_mask is not None:
            if (
                not isinstance(action_feature_mask, torch.Tensor)
                or action_feature_mask.dtype is not torch.bool
                or action_feature_mask.ndim != 1
                or not bool(action_feature_mask.any())
            ):
                raise ValueError("action_feature_mask must be a non-empty bool vector")
            action_feature_mask = action_feature_mask.detach().clone()
        self.action_feature_mask = action_feature_mask
        self.optimizer_step = 0
        module_device_dtype(model)

    def step(
        self,
        batch: Stage0WindowBatch,
        success_latents: torch.Tensor | None,
        *,
        generator: torch.Generator | None = None,
    ) -> PolicyStepMetrics:
        if not isinstance(batch, Stage0WindowBatch):
            raise TypeError("batch must be a Stage0WindowBatch")
        self.model.train()
        device, dtype = module_device_dtype(self.model)
        batch = batch.to(device, dtype=dtype)
        if success_latents is not None:
            success_latents = success_latents.detach().to(
                device=device,
                dtype=dtype,
            )
        self.optimizer.zero_grad(set_to_none=True)
        kwargs = {
            "success_latent": success_latents,
            "action_mask": batch.action_mask,
            "action_feature_mask": None
            if self.action_feature_mask is None
            else self.action_feature_mask.to(device=device),
            "generator": generator,
        }
        if self.model.__class__.__name__ == "DistributedDataParallel":
            output = self.model(batch.current_states, batch.action_chunks, **kwargs)
        elif type(self.model).forward is nn.Module.forward:
            if not hasattr(self.model, "flow_matching_loss"):
                raise TypeError("policy must provide forward or flow_matching_loss")
            output = self.model.flow_matching_loss(
                batch.current_states,
                batch.action_chunks,
                **kwargs,
            )
        else:
            output = self.model(batch.current_states, batch.action_chunks, **kwargs)
        loss = output.loss
        gradient_norm = finish_optimizer_step(
            loss=loss,
            module=self.model,
            optimizer=self.optimizer,
            gradient_clip_norm=self.gradient_clip_norm,
        )
        self.optimizer_step += 1
        value = float(loss.detach())
        return PolicyStepMetrics(
            loss=value,
            flow_matching=value,
            gradient_norm=gradient_norm,
            optimizer_step=self.optimizer_step,
        )
