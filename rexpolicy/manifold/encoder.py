"""Future-trajectory Transformer encoder and latent-only decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import SuccessManifoldConfig


@dataclass(frozen=True)
class FutureTrajectoryReconstruction:
    states: torch.Tensor
    actions: torch.Tensor
    visual_features: torch.Tensor | None


@dataclass(frozen=True)
class FutureTrajectoryOutput:
    latent: torch.Tensor
    condition_token: torch.Tensor
    projection: torch.Tensor
    encoded_steps: torch.Tensor
    reconstruction: FutureTrajectoryReconstruction
    padding_mask: torch.Tensor
    reconstruction_mask: torch.Tensor


def _require_sequence(
    value: torch.Tensor,
    *,
    name: str,
    feature_dim: int,
) -> tuple[int, int]:
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, steps, features]")
    if value.shape[-1] != feature_dim:
        raise ValueError(f"{name} feature dimension must be {feature_dim}")
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must be floating point")
    return int(value.shape[0]), int(value.shape[1])


class FutureTrajectoryDecoder(nn.Module):
    """Decode a compressed future latent at every requested future step."""

    def __init__(self, config: SuccessManifoldConfig) -> None:
        super().__init__()
        if config.state_dim <= 0:
            raise ValueError("FutureTrajectoryDecoder requires a positive state_dim")
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.visual_dim = config.visual_dim
        self.max_future_steps = config.max_future_steps
        output_dim = self.state_dim + self.action_dim + self.visual_dim
        self.latent_projection = nn.Linear(config.latent_dim, config.model_dim)
        self.position_embedding = nn.Parameter(
            torch.empty(1, config.max_future_steps, config.model_dim)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.feedforward_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, output_dim),
        )
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        latent: torch.Tensor,
        *,
        steps: int,
        padding_mask: torch.Tensor | None = None,
    ) -> FutureTrajectoryReconstruction:
        if not isinstance(latent, torch.Tensor) or latent.ndim != 2:
            raise ValueError("latent must have shape [batch, latent_dim]")
        if type(steps) is not int or not 1 <= steps <= self.max_future_steps:
            raise ValueError(
                f"steps must be between 1 and {self.max_future_steps}, inclusive"
            )
        batch = latent.shape[0]
        if padding_mask is None:
            padding_mask = torch.zeros(
                (batch, steps),
                dtype=torch.bool,
                device=latent.device,
            )
        if (
            not isinstance(padding_mask, torch.Tensor)
            or padding_mask.dtype != torch.bool
            or tuple(padding_mask.shape) != (batch, steps)
            or padding_mask.device != latent.device
        ):
            raise ValueError("padding_mask must be a bool [batch, steps] tensor")
        tokens = self.latent_projection(latent).unsqueeze(1)
        tokens = tokens + self.position_embedding[:, :steps].to(dtype=tokens.dtype)
        decoded = self.output(tokens)
        decoded = decoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        state_end = self.state_dim
        action_end = state_end + self.action_dim
        visual = decoded[..., action_end:] if self.visual_dim else None
        return FutureTrajectoryReconstruction(
            states=decoded[..., :state_end],
            actions=decoded[..., state_end:action_end],
            visual_features=visual,
        )


class FutureTrajectoryEncoder(nn.Module):
    """Compress a padded successful future into a reusable success latent."""

    def __init__(self, config: SuccessManifoldConfig) -> None:
        super().__init__()
        if config.state_dim <= 0:
            raise ValueError("FutureTrajectoryEncoder requires a positive state_dim")
        self.config = config
        self.state_embedding = nn.Linear(config.state_dim, config.model_dim)
        self.action_embedding = nn.Linear(config.action_dim, config.model_dim)
        self.visual_embedding = (
            nn.Linear(config.visual_dim, config.model_dim)
            if config.visual_dim
            else None
        )
        self.input_norm = nn.LayerNorm(config.model_dim)
        self.summary_token = nn.Parameter(torch.empty(1, 1, config.model_dim))
        self.mask_token = nn.Parameter(torch.empty(1, 1, config.model_dim))
        self.position_embedding = nn.Parameter(
            torch.empty(1, config.max_future_steps + 1, config.model_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
            norm=nn.LayerNorm(config.model_dim),
        )
        self.latent_head = nn.Linear(config.model_dim, config.latent_dim)
        self.condition_head = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.condition_dim),
        )
        self.projection_head = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.projection_dim),
            nn.GELU(),
            nn.Linear(config.projection_dim, config.projection_dim),
        )
        self.decoder = FutureTrajectoryDecoder(config)
        nn.init.normal_(self.summary_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def sample_reconstruction_mask(
        self,
        padding_mask: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample valid masked steps and keep at least one when probability > 0."""
        if not isinstance(padding_mask, torch.Tensor) or padding_mask.ndim != 2:
            raise ValueError("padding_mask must have shape [batch, steps]")
        if padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be boolean")
        probability = float(self.config.reconstruction_mask_probability)
        if probability == 0.0:
            return torch.zeros_like(padding_mask)
        random_values = torch.rand(
            padding_mask.shape,
            generator=generator,
            device=padding_mask.device,
        )
        result = (random_values < probability) & ~padding_mask
        for row in range(result.shape[0]):
            if not bool(result[row].any()):
                valid = torch.nonzero(~padding_mask[row], as_tuple=False)
                if valid.numel():
                    result[row, int(valid[0, 0])] = True
        return result

    def forward(
        self,
        future_states: torch.Tensor,
        future_actions: torch.Tensor,
        *,
        visual_features: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        reconstruction_mask: torch.Tensor | None = None,
    ) -> FutureTrajectoryOutput:
        batch, steps = _require_sequence(
            future_states,
            name="future_states",
            feature_dim=self.config.state_dim,
        )
        action_shape = _require_sequence(
            future_actions,
            name="future_actions",
            feature_dim=self.config.action_dim,
        )
        if action_shape != (batch, steps):
            raise ValueError("future_states and future_actions shapes do not align")
        if not 1 <= steps <= self.config.max_future_steps:
            raise ValueError(
                "Future sequence length must be between 1 and "
                f"{self.config.max_future_steps}"
            )
        if future_actions.device != future_states.device:
            raise ValueError("Future modalities must be on the same device")
        if future_actions.dtype != future_states.dtype:
            raise ValueError("Future modalities must use the same dtype")

        if self.config.visual_dim:
            if visual_features is not None:
                visual_shape = _require_sequence(
                    visual_features,
                    name="visual_features",
                    feature_dim=self.config.visual_dim,
                )
                if visual_shape != (batch, steps):
                    raise ValueError("visual_features shape does not align")
                if (
                    visual_features.device != future_states.device
                    or visual_features.dtype != future_states.dtype
                ):
                    raise ValueError("Future modalities must share device and dtype")
        elif visual_features is not None:
            raise ValueError("visual_features were provided while visual_dim is zero")

        if padding_mask is None:
            padding_mask = torch.zeros(
                (batch, steps),
                dtype=torch.bool,
                device=future_states.device,
            )
        if (
            not isinstance(padding_mask, torch.Tensor)
            or padding_mask.dtype != torch.bool
            or tuple(padding_mask.shape) != (batch, steps)
            or padding_mask.device != future_states.device
        ):
            raise ValueError("padding_mask must be a bool [batch, steps] tensor")
        if bool(padding_mask.all(dim=1).any()):
            raise ValueError("Every future trajectory must contain a valid step")

        if reconstruction_mask is None:
            reconstruction_mask = torch.zeros_like(padding_mask)
        if (
            not isinstance(reconstruction_mask, torch.Tensor)
            or reconstruction_mask.dtype != torch.bool
            or tuple(reconstruction_mask.shape) != (batch, steps)
            or reconstruction_mask.device != future_states.device
        ):
            raise ValueError(
                "reconstruction_mask must be a bool [batch, steps] tensor"
            )
        reconstruction_mask = reconstruction_mask & ~padding_mask

        tokens = self.state_embedding(future_states)
        tokens = tokens + self.action_embedding(future_actions)
        if self.visual_embedding is not None and visual_features is not None:
            tokens = tokens + self.visual_embedding(visual_features)
        tokens = self.input_norm(tokens)
        mask_token = self.mask_token.to(dtype=tokens.dtype).expand(batch, steps, -1)
        tokens = torch.where(reconstruction_mask.unsqueeze(-1), mask_token, tokens)
        summary = self.summary_token.to(dtype=tokens.dtype).expand(batch, -1, -1)
        tokens = torch.cat((summary, tokens), dim=1)
        tokens = tokens + self.position_embedding[:, : steps + 1].to(
            dtype=tokens.dtype
        )
        transformer_padding = torch.cat(
            (
                torch.zeros((batch, 1), dtype=torch.bool, device=tokens.device),
                padding_mask,
            ),
            dim=1,
        )
        encoded = self.transformer(
            tokens,
            src_key_padding_mask=transformer_padding,
        )
        latent = self.latent_head(encoded[:, 0])
        condition_token = self.condition_head(latent).unsqueeze(1)
        projection = self.projection_head(latent)
        reconstruction = self.decoder(
            latent,
            steps=steps,
            padding_mask=padding_mask,
        )
        return FutureTrajectoryOutput(
            latent=latent,
            condition_token=condition_token,
            projection=projection,
            encoded_steps=encoded[:, 1:].masked_fill(
                padding_mask.unsqueeze(-1),
                0.0,
            ),
            reconstruction=reconstruction,
            padding_mask=padding_mask,
            reconstruction_mask=reconstruction_mask,
        )
