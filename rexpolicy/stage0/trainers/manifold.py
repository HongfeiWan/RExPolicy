"""One-step trainer for the Stage 0 future-success manifold."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn

from .batch import Stage0WindowBatch
from .common import finish_optimizer_step, module_device_dtype, positive_finite
from .losses import manifold_losses


def _weight(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


@dataclass(frozen=True)
class Stage0ManifoldTrainerConfig:
    reconstruction_probability: float = 0.15
    reconstruction_weight: float = 1.0
    contrastive_weight: float = 1.0
    variance_weight: float = 1.0
    covariance_weight: float = 1.0
    temporal_radius: int = 2
    contrastive_temperature: float = 0.1
    variance_target_std: float = 1.0
    variance_epsilon: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    gather_distributed: bool = False
    mode_alignment_weight: float = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.reconstruction_probability, bool)
            or not isinstance(self.reconstruction_probability, (int, float))
            or not 0.0 < float(self.reconstruction_probability) <= 1.0
        ):
            raise ValueError("reconstruction_probability must lie in (0, 1]")
        for name in (
            "reconstruction_weight",
            "contrastive_weight",
            "mode_alignment_weight",
            "variance_weight",
            "covariance_weight",
        ):
            _weight(getattr(self, name), name)
        if not any(
            getattr(self, name) > 0.0
            for name in (
                "reconstruction_weight",
                "contrastive_weight",
                "variance_weight",
                "covariance_weight",
            )
        ):
            raise ValueError("at least one manifold loss weight must be positive")
        if type(self.temporal_radius) is not int or self.temporal_radius < 1:
            raise ValueError("temporal_radius must be a positive integer")
        positive_finite(
            self.contrastive_temperature,
            name="contrastive_temperature",
        )
        positive_finite(self.variance_target_std, name="variance_target_std")
        positive_finite(self.variance_epsilon, name="variance_epsilon")
        positive_finite(self.gradient_clip_norm, name="gradient_clip_norm")
        if not isinstance(self.gather_distributed, bool):
            raise ValueError("gather_distributed must be boolean")


@dataclass(frozen=True)
class ManifoldStepMetrics:
    loss: float
    reconstruction: float
    contrastive: float
    variance: float
    covariance: float
    gradient_norm: float
    optimizer_step: int
    mode_alignment: float = 0.0


class Stage0ManifoldTrainer:
    """Train an encoder or caller-wrapped DDP encoder one batch at a time."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        config: Stage0ManifoldTrainerConfig | None = None,
        distributed_group: Any = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch module")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch optimizer")
        self.model = model
        self.optimizer = optimizer
        self.config = config or Stage0ManifoldTrainerConfig()
        self.distributed_group = distributed_group
        self.optimizer_step = 0
        module_device_dtype(model)

    def step(
        self,
        batch: Stage0WindowBatch,
        *,
        mode_ids: Sequence[Any] | None = None,
        generator: torch.Generator | None = None,
    ) -> ManifoldStepMetrics:
        if not isinstance(batch, Stage0WindowBatch):
            raise TypeError("batch must be a Stage0WindowBatch")
        self.model.train()
        device, dtype = module_device_dtype(self.model)
        batch = batch.to(device, dtype=dtype)
        padding_mask = batch.future_padding_mask
        encoder = getattr(self.model, "module", self.model)
        if not hasattr(encoder, "sample_reconstruction_mask"):
            raise TypeError("manifold model must provide sample_reconstruction_mask")
        reconstruction_mask = encoder.sample_reconstruction_mask(
            padding_mask,
            probability=self.config.reconstruction_probability,
            generator=generator,
        )
        self.optimizer.zero_grad(set_to_none=True)
        encoding = self.model(
            batch.future_states,
            batch.future_actions,
            padding_mask=padding_mask,
            reconstruction_mask=reconstruction_mask,
        )
        losses = manifold_losses(
            encoding,
            batch.future_states,
            batch.future_actions,
            batch.trajectory_ids,
            batch.reset_group_ids,
            batch.starts,
            mode_ids=mode_ids,
            reconstruction_weight=self.config.reconstruction_weight,
            contrastive_weight=self.config.contrastive_weight,
            mode_alignment_weight=self.config.mode_alignment_weight,
            variance_weight=self.config.variance_weight,
            covariance_weight=self.config.covariance_weight,
            temporal_radius=self.config.temporal_radius,
            contrastive_temperature=self.config.contrastive_temperature,
            variance_target_std=self.config.variance_target_std,
            variance_epsilon=self.config.variance_epsilon,
            gather_distributed=self.config.gather_distributed,
            group=self.distributed_group,
        )
        gradient_norm = finish_optimizer_step(
            loss=losses.total,
            module=self.model,
            optimizer=self.optimizer,
            gradient_clip_norm=self.config.gradient_clip_norm,
        )
        self.optimizer_step += 1
        return ManifoldStepMetrics(
            loss=float(losses.total.detach()),
            reconstruction=float(losses.reconstruction.detach()),
            contrastive=float(losses.contrastive.detach()),
            mode_alignment=float(losses.mode_alignment.detach()),
            variance=float(losses.variance.detach()),
            covariance=float(losses.covariance.detach()),
            gradient_norm=gradient_norm,
            optimizer_step=self.optimizer_step,
        )
