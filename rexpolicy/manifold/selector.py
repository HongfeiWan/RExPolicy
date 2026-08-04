"""State-conditioned Gaussian selection over successful future modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .config import SuccessManifoldConfig


@dataclass(frozen=True)
class DiagonalGaussian:
    mean: torch.Tensor
    log_std: torch.Tensor

    @property
    def std(self) -> torch.Tensor:
        return self.log_std.exp()

    def rsample(
        self,
        sample_shape: Sequence[int] = (),
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        shape = tuple(sample_shape)
        if any(type(item) is not int or item < 0 for item in shape):
            raise ValueError("sample_shape must contain non-negative integers")
        noise = torch.randn(
            (*shape, *self.mean.shape),
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        mean = self.mean.reshape((1,) * len(shape) + tuple(self.mean.shape))
        std = self.std.reshape((1,) * len(shape) + tuple(self.std.shape))
        return mean + noise * std

    def sample(
        self,
        sample_shape: Sequence[int] = (),
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.rsample(sample_shape, generator=generator).detach()

    @property
    def mode(self) -> torch.Tensor:
        return self.mean


class SuccessModeSelector(nn.Module):
    """Predict a diagonal Gaussian over success latents from current state."""

    def __init__(self, config: SuccessManifoldConfig) -> None:
        super().__init__()
        if config.state_dim <= 0:
            raise ValueError("SuccessModeSelector requires a positive state_dim")
        self.state_dim = config.state_dim
        self.latent_dim = config.latent_dim
        self.minimum_log_std = float(config.selector_min_log_std)
        self.maximum_log_std = float(config.selector_max_log_std)

        layers: list[nn.Module] = []
        input_dim = config.state_dim
        for _ in range(config.selector_layers):
            layers.extend(
                (
                    nn.Linear(input_dim, config.selector_hidden_dim),
                    nn.LayerNorm(config.selector_hidden_dim),
                    nn.GELU(),
                )
            )
            input_dim = config.selector_hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(input_dim, config.latent_dim)
        self.log_std_head = nn.Linear(input_dim, config.latent_dim)
        nn.init.constant_(self.log_std_head.bias, -1.0)

    def distribution(self, current_state: torch.Tensor) -> DiagonalGaussian:
        if (
            not isinstance(current_state, torch.Tensor)
            or current_state.ndim < 2
            or current_state.shape[-1] != self.state_dim
            or not torch.is_floating_point(current_state)
        ):
            raise ValueError(
                f"current_state must be floating [..., {self.state_dim}]"
            )
        features = self.backbone(current_state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(
            min=self.minimum_log_std,
            max=self.maximum_log_std,
        )
        return DiagonalGaussian(mean=mean, log_std=log_std)

    def forward(self, current_state: torch.Tensor) -> DiagonalGaussian:
        return self.distribution(current_state)

    def predict(self, current_state: torch.Tensor) -> torch.Tensor:
        """Return the deterministic maximum-density latent."""
        return self.distribution(current_state).mode

    def sample(
        self,
        current_state: torch.Tensor,
        *,
        sample_shape: Sequence[int] = (),
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        distribution = self.distribution(current_state)
        if deterministic:
            if sample_shape:
                shape = tuple(sample_shape)
                if any(type(item) is not int or item < 0 for item in shape):
                    raise ValueError(
                        "sample_shape must contain non-negative integers"
                    )
                return distribution.mean.expand(*shape, *distribution.mean.shape)
            return distribution.mean
        return distribution.sample(sample_shape, generator=generator)
