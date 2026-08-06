"""Lightweight PyTorch-only conditional flow transformer for Stage 0."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from ._validation import (
    require_bool_mask,
    require_float_tensor,
    require_positive_int,
    require_probability,
)
from .conditioning import ConditionTokens


@dataclass(frozen=True)
class FlowMatchingOutput:
    """One rectified-flow training batch and its scalar objective."""

    loss: torch.Tensor
    predicted_velocity: torch.Tensor
    target_velocity: torch.Tensor
    noisy_actions: torch.Tensor
    noise: torch.Tensor
    timesteps: torch.Tensor
    action_mask: torch.Tensor
    action_feature_mask: torch.Tensor


class FourierTimeEmbedding(nn.Module):
    """Fixed Fourier features followed by a trainable time MLP."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        model_dim = require_positive_int(model_dim, name="model_dim")
        half = max(model_dim // 2, 1)
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half))
        self.model_dim = model_dim
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.network = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.SiLU(),
            nn.Linear(model_dim * 2, model_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        require_float_tensor(timesteps, name="timesteps", ndim=1)
        phases = timesteps.unsqueeze(-1) * self.frequencies.to(
            device=timesteps.device,
            dtype=timesteps.dtype,
        )
        features = torch.cat((phases.sin(), phases.cos()), dim=-1)
        if features.shape[-1] < self.model_dim:
            features = torch.nn.functional.pad(
                features,
                (0, self.model_dim - int(features.shape[-1])),
            )
        elif features.shape[-1] > self.model_dim:
            features = features[..., : self.model_dim]
        return self.network(features)


class FlowDiT(nn.Module):
    """Generate fixed-horizon action chunks from generic condition tokens.

    The model implements rectified flow with
    ``x_t = (1 - t) * x_0 + t * x_1`` and target velocity ``x_1 - x_0``.
    """

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        model_dim: int,
        *,
        transformer_layers: int = 3,
        attention_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.action_dim = require_positive_int(action_dim, name="action_dim")
        self.horizon = require_positive_int(horizon, name="horizon")
        self.model_dim = require_positive_int(model_dim, name="model_dim")
        transformer_layers = require_positive_int(
            transformer_layers,
            name="transformer_layers",
        )
        attention_heads = require_positive_int(
            attention_heads,
            name="attention_heads",
        )
        feedforward_dim = require_positive_int(
            feedforward_dim,
            name="feedforward_dim",
        )
        dropout = require_probability(dropout, name="dropout")
        if dropout == 1.0:
            raise ValueError("dropout must be smaller than one")
        if self.model_dim % attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        self.dropout = dropout

        self.action_projector = nn.Linear(self.action_dim, self.model_dim)
        self.action_position = nn.Parameter(
            torch.empty(1, self.horizon, self.model_dim)
        )
        self.action_token_type = nn.Parameter(torch.empty(1, 1, self.model_dim))
        self.condition_token_type = nn.Parameter(torch.empty(1, 1, self.model_dim))
        self.time_embedding = FourierTimeEmbedding(self.model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(self.model_dim),
        )
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.action_dim),
        )
        nn.init.normal_(self.action_position, std=0.02)
        nn.init.normal_(self.action_token_type, std=0.02)
        nn.init.normal_(self.condition_token_type, std=0.02)

    def _validate_actions(
        self,
        actions: torch.Tensor,
        *,
        name: str,
    ) -> tuple[int, torch.Tensor]:
        require_float_tensor(
            actions,
            name=name,
            ndim=3,
            last_dim=self.action_dim,
        )
        if actions.shape[1] != self.horizon:
            raise ValueError(f"{name} horizon must be {self.horizon}")
        if actions.shape[0] <= 0:
            raise ValueError(f"{name} batch must be non-empty")
        return int(actions.shape[0]), actions

    def _action_mask(
        self,
        action_mask: torch.Tensor | None,
        *,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        if action_mask is None:
            action_mask = torch.ones(
                (batch, self.horizon),
                dtype=torch.bool,
                device=device,
            )
        action_mask = require_bool_mask(
            action_mask,
            name="action_mask",
            shape=(batch, self.horizon),
            device=device,
        )
        if bool((~action_mask).all(dim=1).any()):
            raise ValueError("every action row must contain a valid step")
        return action_mask

    def _action_feature_mask(
        self,
        action_feature_mask: torch.Tensor | None,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if action_feature_mask is None:
            return torch.ones(self.action_dim, dtype=torch.bool, device=device)
        return require_bool_mask(
            action_feature_mask,
            name="action_feature_mask",
            shape=(self.action_dim,),
            device=device,
        )

    def _timesteps(
        self,
        timesteps: torch.Tensor | float,
        *,
        batch: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(timesteps, bool):
            raise ValueError("timesteps must be floating values in [0, 1]")
        if isinstance(timesteps, (int, float)):
            value = float(timesteps)
            if not math.isfinite(value):
                raise ValueError("timesteps must contain only finite values")
            result = torch.full(
                (batch,),
                value,
                dtype=reference.dtype,
                device=reference.device,
            )
        else:
            if not isinstance(timesteps, torch.Tensor):
                raise ValueError("timesteps must be a float or tensor")
            if timesteps.ndim == 2 and tuple(timesteps.shape) == (batch, 1):
                timesteps = timesteps[:, 0]
            require_float_tensor(timesteps, name="timesteps", ndim=1)
            if tuple(timesteps.shape) != (batch,):
                raise ValueError("timesteps must have shape [batch] or [batch, 1]")
            if (
                timesteps.device != reference.device
                or timesteps.dtype != reference.dtype
            ):
                raise ValueError("timesteps must share action device and dtype")
            result = timesteps
        if bool(((result < 0.0) | (result > 1.0)).any()):
            raise ValueError("timesteps must lie in [0, 1]")
        return result

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor | float,
        conditions: ConditionTokens,
        *,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict action-space velocity at a batch of flow times."""
        batch, noisy_actions = self._validate_actions(
            noisy_actions,
            name="noisy_actions",
        )
        if not isinstance(conditions, ConditionTokens):
            raise TypeError("conditions must be ConditionTokens")
        if (
            conditions.batch_size != batch
            or conditions.model_dim != self.model_dim
            or conditions.tokens.device != noisy_actions.device
            or conditions.tokens.dtype != noisy_actions.dtype
        ):
            raise ValueError(
                "conditions must share action batch, model dimension, device, and dtype"
            )
        action_mask = self._action_mask(
            action_mask,
            batch=batch,
            device=noisy_actions.device,
        )
        timesteps = self._timesteps(
            timesteps,
            batch=batch,
            reference=noisy_actions,
        )

        time_token = self.time_embedding(timesteps).unsqueeze(1)
        action_tokens = self.action_projector(noisy_actions)
        action_tokens = (
            action_tokens
            + self.action_position.to(dtype=action_tokens.dtype)
            + self.action_token_type.to(dtype=action_tokens.dtype)
            + time_token
        )
        condition_tokens = conditions.tokens + self.condition_token_type.to(
            dtype=conditions.tokens.dtype
        )
        tokens = torch.cat((condition_tokens, action_tokens), dim=1)
        valid_mask = torch.cat(
            (conditions.attention_mask, action_mask),
            dim=1,
        )
        encoded = self.transformer(
            tokens,
            src_key_padding_mask=~valid_mask,
        )
        action_encoded = encoded[:, conditions.token_count :]
        velocity = self.velocity_head(action_encoded)
        return velocity.masked_fill(~action_mask.unsqueeze(-1), 0.0)

    def flow_matching_loss(
        self,
        actions: torch.Tensor,
        conditions: ConditionTokens,
        *,
        action_mask: torch.Tensor | None = None,
        action_feature_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> FlowMatchingOutput:
        """Construct a reproducible rectified-flow batch and compute MSE."""
        batch, actions = self._validate_actions(actions, name="actions")
        action_mask = self._action_mask(
            action_mask,
            batch=batch,
            device=actions.device,
        )
        action_feature_mask = self._action_feature_mask(
            action_feature_mask,
            device=actions.device,
        )
        if not bool(action_feature_mask.any()):
            raise ValueError("action_feature_mask must retain at least one feature")
        if noise is None:
            noise = torch.randn(
                actions.shape,
                dtype=actions.dtype,
                device=actions.device,
                generator=generator,
            )
        else:
            self._validate_actions(noise, name="noise")
            if (
                noise.shape != actions.shape
                or noise.device != actions.device
                or noise.dtype != actions.dtype
            ):
                raise ValueError("noise must share action shape, device, and dtype")
        if timesteps is None:
            timesteps = torch.rand(
                (batch,),
                dtype=actions.dtype,
                device=actions.device,
                generator=generator,
            )
        timesteps = self._timesteps(
            timesteps,
            batch=batch,
            reference=actions,
        )
        feature_mask = action_feature_mask.reshape(1, 1, self.action_dim)
        actions = actions.masked_fill(~feature_mask, 0.0)
        noise = noise.masked_fill(~feature_mask, 0.0)
        interpolation = timesteps[:, None, None]
        noisy_actions = (1.0 - interpolation) * noise + interpolation * actions
        target_velocity = actions - noise
        noisy_actions = noisy_actions.masked_fill(
            ~action_mask.unsqueeze(-1),
            0.0,
        )
        target_velocity = target_velocity.masked_fill(
            ~action_mask.unsqueeze(-1),
            0.0,
        )
        predicted_velocity = self(
            noisy_actions,
            timesteps,
            conditions,
            action_mask=action_mask,
        )
        squared_error = (predicted_velocity - target_velocity).square()
        loss_mask = action_mask.unsqueeze(-1) & feature_mask
        loss = squared_error.masked_select(loss_mask).mean()
        return FlowMatchingOutput(
            loss=loss,
            predicted_velocity=predicted_velocity,
            target_velocity=target_velocity,
            noisy_actions=noisy_actions,
            noise=noise,
            timesteps=timesteps,
            action_mask=action_mask,
            action_feature_mask=action_feature_mask,
        )

    @torch.no_grad()
    def sample(
        self,
        conditions: ConditionTokens,
        *,
        steps: int = 16,
        action_mask: torch.Tensor | None = None,
        action_feature_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate the learned velocity field from noise to actions."""
        steps = require_positive_int(steps, name="steps")
        if not isinstance(conditions, ConditionTokens):
            raise TypeError("conditions must be ConditionTokens")
        batch = conditions.batch_size
        action_mask = self._action_mask(
            action_mask,
            batch=batch,
            device=conditions.tokens.device,
        )
        action_feature_mask = self._action_feature_mask(
            action_feature_mask,
            device=conditions.tokens.device,
        )
        if not bool(action_feature_mask.any()):
            raise ValueError("action_feature_mask must retain at least one feature")
        shape = (batch, self.horizon, self.action_dim)
        if initial_noise is None:
            actions = torch.randn(
                shape,
                dtype=conditions.tokens.dtype,
                device=conditions.tokens.device,
                generator=generator,
            )
        else:
            self._validate_actions(initial_noise, name="initial_noise")
            if (
                initial_noise.shape[0] != batch
                or initial_noise.device != conditions.tokens.device
                or initial_noise.dtype != conditions.tokens.dtype
            ):
                raise ValueError(
                    "initial_noise must share condition batch, device, and dtype"
                )
            actions = initial_noise.clone()
        value_mask = action_mask.unsqueeze(-1) & action_feature_mask.reshape(
            1,
            1,
            self.action_dim,
        )
        actions = actions.masked_fill(~value_mask, 0.0)
        step_size = 1.0 / float(steps)
        for index in range(steps):
            time = float(index) * step_size
            velocity = self(
                actions,
                time,
                conditions,
                action_mask=action_mask,
            )
            actions = actions + step_size * velocity
            actions = actions.masked_fill(~value_mask, 0.0)
        return actions


LightweightFlowDiT = FlowDiT
