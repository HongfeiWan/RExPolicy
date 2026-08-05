"""Independent state/success-conditioned policy runtime for Stage 0."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rexpolicy.stage0.models import (
    FlowDiT,
    FlowMatchingOutput,
    StateLatentConditionEncoder,
)


@dataclass(frozen=True)
class Stage0PolicySample:
    """Action samples from the three scientific conditioning controls."""

    no_z: torch.Tensor
    oracle_z: torch.Tensor
    selector_z: torch.Tensor


class Stage0FlowPolicy(nn.Module):
    """Own all trainable projections and the lightweight conditional Flow-DiT."""

    def __init__(
        self,
        *,
        state_dim: int,
        latent_dim: int,
        action_dim: int,
        action_horizon: int,
        model_dim: int = 128,
        transformer_layers: int = 3,
        attention_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.condition_encoder = StateLatentConditionEncoder(
            state_dim,
            latent_dim,
            model_dim,
        )
        self.flow_dit = FlowDiT(
            action_dim,
            action_horizon,
            model_dim,
            transformer_layers=transformer_layers,
            attention_heads=attention_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

    def forward(
        self,
        current_state: torch.Tensor,
        target_actions: torch.Tensor,
        *,
        success_latent: torch.Tensor | None,
        action_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> FlowMatchingOutput:
        """DDP-visible training forward for conditional flow matching."""
        return self.flow_matching_loss(
            current_state,
            target_actions,
            success_latent=success_latent,
            action_mask=action_mask,
            generator=generator,
            noise=noise,
            timesteps=timesteps,
        )

    def flow_matching_loss(
        self,
        current_state: torch.Tensor,
        target_actions: torch.Tensor,
        *,
        success_latent: torch.Tensor | None,
        action_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> FlowMatchingOutput:
        conditions = self.condition_encoder(current_state, success_latent)
        return self.flow_dit.flow_matching_loss(
            target_actions,
            conditions,
            action_mask=action_mask,
            generator=generator,
            noise=noise,
            timesteps=timesteps,
        )

    @torch.no_grad()
    def sample(
        self,
        current_state: torch.Tensor,
        *,
        success_latent: torch.Tensor | None,
        steps: int = 16,
        action_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        conditions = self.condition_encoder(current_state, success_latent)
        return self.flow_dit.sample(
            conditions,
            steps=steps,
            action_mask=action_mask,
            generator=generator,
            initial_noise=initial_noise,
        )

    @torch.no_grad()
    def sample_controls(
        self,
        current_state: torch.Tensor,
        *,
        oracle_latent: torch.Tensor,
        selector_latent: torch.Tensor,
        steps: int = 16,
        action_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> Stage0PolicySample:
        if initial_noise is None:
            initial_noise = torch.randn(
                (
                    current_state.shape[0],
                    self.action_horizon,
                    self.action_dim,
                ),
                dtype=current_state.dtype,
                device=current_state.device,
                generator=generator,
            )
        return Stage0PolicySample(
            no_z=self.sample(
                current_state,
                success_latent=None,
                steps=steps,
                action_mask=action_mask,
                initial_noise=initial_noise,
            ),
            oracle_z=self.sample(
                current_state,
                success_latent=oracle_latent,
                steps=steps,
                action_mask=action_mask,
                initial_noise=initial_noise,
            ),
            selector_z=self.sample(
                current_state,
                success_latent=selector_latent,
                steps=steps,
                action_mask=action_mask,
                initial_noise=initial_noise,
            ),
        )


__all__ = ["Stage0FlowPolicy", "Stage0PolicySample"]
