"""State-conditioned multimodal density over reachable success latents."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from ._validation import require_float_tensor, require_positive_int


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
class DiagonalGaussianMixture:
    """A batched diagonal Gaussian mixture with explicit RNG sampling.

    ``mode`` is the mean of the highest-weight component. Computing the exact
    continuous-density mode of a Gaussian mixture would require an iterative
    optimizer and would make deterministic policy selection unnecessarily
    expensive.
    """

    logits: torch.Tensor
    component_means: torch.Tensor
    component_log_stds: torch.Tensor

    def __post_init__(self) -> None:
        require_float_tensor(self.logits, name="logits", ndim=2)
        require_float_tensor(
            self.component_means,
            name="component_means",
            ndim=3,
        )
        require_float_tensor(
            self.component_log_stds,
            name="component_log_stds",
            ndim=3,
        )
        if self.logits.shape[0] <= 0 or self.logits.shape[1] <= 0:
            raise ValueError("distribution batch and component axes must be non-empty")
        if self.component_means.shape[2] <= 0:
            raise ValueError("distribution latent axis must be non-empty")
        if self.component_means.shape != self.component_log_stds.shape:
            raise ValueError("component means and log standard deviations must align")
        if self.component_means.shape[:2] != self.logits.shape:
            raise ValueError("component parameters must align with logits")
        tensors = (self.logits, self.component_means, self.component_log_stds)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("mixture parameters must share a device")
        if len({tensor.dtype for tensor in tensors}) != 1:
            raise ValueError("mixture parameters must share a dtype")

    @property
    def component_stds(self) -> torch.Tensor:
        return self.component_log_stds.exp()

    @property
    def log_weights(self) -> torch.Tensor:
        return torch.log_softmax(self.logits, dim=-1)

    @property
    def weights(self) -> torch.Tensor:
        return self.log_weights.exp()

    @property
    def mean(self) -> torch.Tensor:
        """Return the mixture expectation for diagnostics."""

        return torch.sum(self.weights.unsqueeze(-1) * self.component_means, dim=1)

    @property
    def mode(self) -> torch.Tensor:
        component = self.logits.argmax(dim=-1)
        index = component[:, None, None].expand(-1, 1, self.component_means.shape[-1])
        return self.component_means.gather(dim=1, index=index).squeeze(1)

    def negative_log_likelihood(self, target: torch.Tensor) -> torch.Tensor:
        if not isinstance(target, torch.Tensor) or target.ndim != 2:
            raise ValueError("target must have shape [batch, latent_dim]")
        require_float_tensor(
            target,
            name="target",
            ndim=2,
            last_dim=int(self.component_means.shape[-1]),
        )
        if (
            target.shape[0] != self.logits.shape[0]
            or target.device != self.logits.device
            or target.dtype != self.logits.dtype
        ):
            raise ValueError("target must align with the distribution")

        residual = target[:, None, :] - self.component_means
        component_log_probability = -(
            0.5
            * residual.square()
            * torch.exp(-2.0 * self.component_log_stds)
            + self.component_log_stds
            + 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)
        log_probability = torch.logsumexp(
            self.log_weights + component_log_probability,
            dim=-1,
        )
        return -log_probability.mean()

    def _component_indices(
        self,
        shape: tuple[int, ...],
        *,
        temperature: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        sample_count = math.prod(shape) if shape else 1
        if sample_count == 0:
            return torch.empty(
                (0, self.logits.shape[0]),
                dtype=torch.long,
                device=self.logits.device,
            )
        weights = torch.softmax(self.logits / temperature, dim=-1)
        return torch.multinomial(
            weights,
            num_samples=sample_count,
            replacement=True,
            generator=generator,
        ).transpose(0, 1)

    def rsample(
        self,
        sample_shape: Sequence[int] = (),
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample a component, then reparameterize its diagonal Gaussian.

        Gradients flow through the selected component's mean and scale. The
        categorical draw itself is intentionally discrete; selector logits are
        trained through exact mixture likelihood rather than sampled actions.
        """

        shape = _sample_shape(sample_shape)
        temperature = _require_temperature(temperature)
        sample_count = math.prod(shape) if shape else 1
        if sample_count == 0:
            return self.component_means.new_empty(
                (*shape, self.logits.shape[0], self.component_means.shape[-1])
            )

        components = self._component_indices(
            shape,
            temperature=temperature,
            generator=generator,
        )
        batch_size, _, latent_dim = self.component_means.shape
        index = components[:, :, None, None].expand(-1, -1, 1, latent_dim)
        means = self.component_means[None, :, :, :].expand(
            sample_count,
            -1,
            -1,
            -1,
        )
        log_stds = self.component_log_stds[None, :, :, :].expand_as(means)
        selected_means = means.gather(dim=2, index=index).squeeze(2)
        selected_log_stds = log_stds.gather(dim=2, index=index).squeeze(2)
        noise = torch.randn(
            selected_means.shape,
            dtype=selected_means.dtype,
            device=selected_means.device,
            generator=generator,
        )
        samples = selected_means + (
            noise * selected_log_stds.exp() * math.sqrt(temperature)
        )
        return samples.reshape((*shape, batch_size, latent_dim))

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


class MixtureSuccessModeSelector(nn.Module):
    """Predict ``p(z_success | current_state)`` as a Gaussian mixture."""

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        *,
        components: int = 4,
        hidden_dim: int = 128,
        layers: int = 2,
        minimum_log_std: float = -5.0,
        maximum_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        self.state_dim = require_positive_int(state_dim, name="state_dim")
        self.latent_dim = require_positive_int(latent_dim, name="latent_dim")
        self.components = require_positive_int(components, name="components")
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
        self.logits_head = nn.Linear(input_dim, self.components)
        self.mean_head = nn.Linear(
            input_dim,
            self.components * self.latent_dim,
        )
        self.log_std_head = nn.Linear(
            input_dim,
            self.components * self.latent_dim,
        )
        nn.init.zeros_(self.logits_head.bias)
        nn.init.constant_(self.log_std_head.bias, -1.0)

    def forward(self, current_state: torch.Tensor) -> DiagonalGaussianMixture:
        require_float_tensor(
            current_state,
            name="current_state",
            ndim=2,
            last_dim=self.state_dim,
        )
        if current_state.shape[0] <= 0:
            raise ValueError("current_state batch must be non-empty")
        features = self.backbone(current_state)
        batch_size = current_state.shape[0]
        raw_logits = self.logits_head(features)
        logits = raw_logits - raw_logits.amax(dim=-1, keepdim=True)
        means = self.mean_head(features).reshape(
            batch_size,
            self.components,
            self.latent_dim,
        )
        log_stds = self.log_std_head(features).reshape_as(means).clamp(
            min=self.minimum_log_std,
            max=self.maximum_log_std,
        )
        return DiagonalGaussianMixture(
            logits=logits,
            component_means=means,
            component_log_stds=log_stds,
        )

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
                return distribution.mode
            return distribution.mode.expand(*shape, *distribution.mode.shape)
        return distribution.sample(
            shape,
            temperature=temperature,
            generator=generator,
        )


__all__ = ["DiagonalGaussianMixture", "MixtureSuccessModeSelector"]
