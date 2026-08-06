"""State-conditioned selection over reachable Stage 0 success latents."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from ._validation import (
    require_float_tensor,
    require_positive_int,
)


def _require_temperature(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("temperature must be finite and positive")
    return float(value)


def _sample_shape(value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("sample_shape must be a sequence of integers")
    result = tuple(value)
    if any(type(item) is not int or item < 0 for item in result):
        raise ValueError("sample_shape must contain non-negative integers")
    return result


@dataclass(frozen=True)
class DiagonalGaussian:
    """Minimal differentiable diagonal Gaussian with explicit RNG support."""

    mean: torch.Tensor
    log_std: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.mean, torch.Tensor) or self.mean.ndim != 2:
            raise ValueError("mean must have shape [batch, latent_dim]")
        if not isinstance(self.log_std, torch.Tensor) or self.log_std.ndim != 2:
            raise ValueError("log_std must have shape [batch, latent_dim]")
        if self.mean.shape[0] <= 0 or self.mean.shape[1] <= 0:
            raise ValueError("distribution axes must be non-empty")
        require_float_tensor(
            self.mean,
            name="mean",
            ndim=2,
            last_dim=int(self.mean.shape[-1]),
        )
        require_float_tensor(
            self.log_std,
            name="log_std",
            ndim=2,
            last_dim=int(self.log_std.shape[-1]),
        )
        if self.mean.shape != self.log_std.shape:
            raise ValueError("mean and log_std shapes must match")
        if (
            self.mean.device != self.log_std.device
            or self.mean.dtype != self.log_std.dtype
        ):
            raise ValueError("mean and log_std must share device and dtype")

    @property
    def std(self) -> torch.Tensor:
        return self.log_std.exp()

    @property
    def mode(self) -> torch.Tensor:
        return self.mean

    def rsample(
        self,
        sample_shape: Sequence[int] = (),
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        shape = _sample_shape(sample_shape)
        temperature = _require_temperature(temperature)
        noise = torch.randn(
            (*shape, *self.mean.shape),
            dtype=self.mean.dtype,
            device=self.mean.device,
            generator=generator,
        )
        prefix = (1,) * len(shape)
        mean = self.mean.reshape((*prefix, *self.mean.shape))
        std = self.std.reshape((*prefix, *self.std.shape))
        return mean + noise * std * math.sqrt(temperature)

    def sample(
        self,
        sample_shape: Sequence[int] = (),
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.rsample(
            sample_shape,
            temperature=temperature,
            generator=generator,
        ).detach()

    def negative_log_likelihood(self, target: torch.Tensor) -> torch.Tensor:
        if not isinstance(target, torch.Tensor) or target.ndim != 2:
            raise ValueError("target must have shape [batch, latent_dim]")
        require_float_tensor(
            target,
            name="target",
            ndim=2,
            last_dim=int(self.mean.shape[-1]),
        )
        if (
            target.shape != self.mean.shape
            or target.device != self.mean.device
            or target.dtype != self.mean.dtype
        ):
            raise ValueError("target must align with the distribution")
        inverse_variance = torch.exp(-2.0 * self.log_std)
        elementwise = (
            0.5 * (target - self.mean).square() * inverse_variance
            + self.log_std
            + 0.5 * math.log(2.0 * math.pi)
        )
        return elementwise.sum(dim=-1).mean()


class SuccessModeSelector(nn.Module):
    """Predict ``p(z_success | current_state)`` as a diagonal Gaussian."""

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        *,
        hidden_dim: int = 128,
        layers: int = 2,
        minimum_log_std: float = -5.0,
        maximum_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        self.state_dim = require_positive_int(state_dim, name="state_dim")
        self.latent_dim = require_positive_int(latent_dim, name="latent_dim")
        hidden_dim = require_positive_int(hidden_dim, name="hidden_dim")
        layers = require_positive_int(layers, name="layers")
        if (
            isinstance(minimum_log_std, bool)
            or isinstance(maximum_log_std, bool)
            or not isinstance(minimum_log_std, (int, float))
            or not isinstance(maximum_log_std, (int, float))
            or not math.isfinite(float(minimum_log_std))
            or not math.isfinite(float(maximum_log_std))
            or float(minimum_log_std) >= float(maximum_log_std)
        ):
            raise ValueError(
                "minimum_log_std and maximum_log_std must be finite and ordered"
            )
        self.minimum_log_std = float(minimum_log_std)
        self.maximum_log_std = float(maximum_log_std)

        modules: list[nn.Module] = []
        input_dim = self.state_dim
        for _ in range(layers):
            modules.extend(
                (
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
            )
            input_dim = hidden_dim
        self.backbone = nn.Sequential(*modules)
        self.mean_head = nn.Linear(input_dim, self.latent_dim)
        self.log_std_head = nn.Linear(input_dim, self.latent_dim)
        nn.init.constant_(self.log_std_head.bias, -1.0)

    def forward(self, current_state: torch.Tensor) -> DiagonalGaussian:
        require_float_tensor(
            current_state,
            name="current_state",
            ndim=2,
            last_dim=self.state_dim,
        )
        if current_state.shape[0] <= 0:
            raise ValueError("current_state batch must be non-empty")
        features = self.backbone(current_state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(
            min=self.minimum_log_std,
            max=self.maximum_log_std,
        )
        return DiagonalGaussian(mean=mean, log_std=log_std)

    def predict(self, current_state: torch.Tensor) -> torch.Tensor:
        return self(current_state).mode

    def sample(
        self,
        current_state: torch.Tensor,
        *,
        sample_shape: Sequence[int] = (),
        deterministic: bool = False,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        distribution = self(current_state)
        temperature = _require_temperature(temperature)
        shape = _sample_shape(sample_shape)
        if deterministic:
            if not shape:
                return distribution.mean
            return distribution.mean.expand(*shape, *distribution.mean.shape)
        return distribution.sample(
            shape,
            temperature=temperature,
            generator=generator,
        )
