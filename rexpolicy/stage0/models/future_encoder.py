"""State/action future encoder for the isolated Stage 0 manifold.

The encoder consumes transition-aligned pairs ``(a[t], s[t + 1])``.  It has
no dependency on GR00T, image encoders, language models, or policy wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ._validation import (
    require_bool_mask,
    require_float_tensor,
    require_positive_int,
    require_probability,
)


@dataclass(frozen=True)
class FutureReconstruction:
    """Decoded post-action states and actions."""

    states: torch.Tensor
    actions: torch.Tensor


@dataclass(frozen=True)
class FutureTrajectoryEncoding:
    """Outputs needed by representation objectives and downstream models."""

    latent: torch.Tensor
    projection: torch.Tensor
    encoded_steps: torch.Tensor
    reconstruction: FutureReconstruction
    padding_mask: torch.Tensor
    reconstruction_mask: torch.Tensor


class FutureTrajectoryEncoder(nn.Module):
    """Compress successful state/action transitions into a success latent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        latent_dim: int = 32,
        projection_dim: int = 32,
        model_dim: int = 128,
        max_future_steps: int = 32,
        transformer_layers: int = 2,
        attention_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = require_positive_int(state_dim, name="state_dim")
        self.action_dim = require_positive_int(action_dim, name="action_dim")
        self.latent_dim = require_positive_int(latent_dim, name="latent_dim")
        self.projection_dim = require_positive_int(
            projection_dim,
            name="projection_dim",
        )
        self.model_dim = require_positive_int(model_dim, name="model_dim")
        self.max_future_steps = require_positive_int(
            max_future_steps,
            name="max_future_steps",
        )
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

        self.state_projector = nn.Linear(self.state_dim, self.model_dim)
        self.action_projector = nn.Linear(self.action_dim, self.model_dim)
        self.transition_norm = nn.LayerNorm(self.model_dim)
        self.summary_token = nn.Parameter(torch.empty(1, 1, self.model_dim))
        self.mask_token = nn.Parameter(torch.empty(1, 1, self.model_dim))
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.max_future_steps + 1, self.model_dim)
        )
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
        self.latent_head = nn.Linear(self.model_dim, self.latent_dim)
        self.projection_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.projection_dim),
            nn.GELU(),
            nn.Linear(self.projection_dim, self.projection_dim),
        )
        self.decoder_latent = nn.Linear(self.latent_dim, self.model_dim)
        self.decoder = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, self.state_dim + self.action_dim),
        )
        nn.init.normal_(self.summary_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def sample_reconstruction_mask(
        self,
        padding_mask: torch.Tensor,
        *,
        probability: float = 0.15,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample valid masked steps reproducibly.

        ``padding_mask`` follows the PyTorch convention: ``True`` means that a
        step is padding.  When probability is positive, each non-empty row has
        at least one reconstruction target.
        """
        probability = require_probability(probability, name="probability")
        if not isinstance(padding_mask, torch.Tensor) or padding_mask.ndim != 2:
            raise ValueError("padding_mask must have shape [batch, steps]")
        padding_mask = require_bool_mask(
            padding_mask,
            name="padding_mask",
            shape=(int(padding_mask.shape[0]), int(padding_mask.shape[1])),
            device=padding_mask.device,
        )
        if bool(padding_mask.all(dim=1).any()):
            raise ValueError("every trajectory must contain a valid step")
        if probability == 0.0:
            return torch.zeros_like(padding_mask)
        random_values = torch.rand(
            padding_mask.shape,
            generator=generator,
            device=padding_mask.device,
        )
        sampled = (random_values < probability) & ~padding_mask
        missing = ~sampled.any(dim=1)
        if bool(missing.any()):
            first_valid = (~padding_mask).to(torch.int64).argmax(dim=1)
            rows = torch.nonzero(missing, as_tuple=False).flatten()
            sampled[rows, first_valid[rows]] = True
        return sampled

    def forward(
        self,
        future_states: torch.Tensor,
        future_actions: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        reconstruction_mask: torch.Tensor | None = None,
    ) -> FutureTrajectoryEncoding:
        """Encode paired post-action states and actions.

        Args:
            future_states: Post-action states with shape ``[B, K, state_dim]``.
            future_actions: Executed actions with shape ``[B, K, action_dim]``.
            padding_mask: Boolean ``[B, K]`` where ``True`` is padding.
            reconstruction_mask: Boolean ``[B, K]`` valid steps replaced by a
                learned mask token before encoding.
        """
        require_float_tensor(
            future_states,
            name="future_states",
            ndim=3,
            last_dim=self.state_dim,
        )
        require_float_tensor(
            future_actions,
            name="future_actions",
            ndim=3,
            last_dim=self.action_dim,
        )
        batch, steps = int(future_states.shape[0]), int(future_states.shape[1])
        if batch <= 0:
            raise ValueError("future trajectory batch must be non-empty")
        if tuple(future_actions.shape[:2]) != (batch, steps):
            raise ValueError("future state and action batch/step shapes must align")
        if not 1 <= steps <= self.max_future_steps:
            raise ValueError(
                f"future step count must be between 1 and {self.max_future_steps}"
            )
        if (
            future_actions.device != future_states.device
            or future_actions.dtype != future_states.dtype
        ):
            raise ValueError("future states and actions must share device and dtype")

        shape = (batch, steps)
        if padding_mask is None:
            padding_mask = torch.zeros(
                shape,
                dtype=torch.bool,
                device=future_states.device,
            )
        padding_mask = require_bool_mask(
            padding_mask,
            name="padding_mask",
            shape=shape,
            device=future_states.device,
        )
        if bool(padding_mask.all(dim=1).any()):
            raise ValueError("every trajectory must contain a valid step")
        if reconstruction_mask is None:
            reconstruction_mask = torch.zeros_like(padding_mask)
        reconstruction_mask = require_bool_mask(
            reconstruction_mask,
            name="reconstruction_mask",
            shape=shape,
            device=future_states.device,
        )
        if bool((reconstruction_mask & padding_mask).any()):
            raise ValueError("reconstruction_mask cannot select padded steps")

        transition_tokens = self.state_projector(future_states)
        transition_tokens = transition_tokens + self.action_projector(future_actions)
        transition_tokens = self.transition_norm(transition_tokens)
        mask_token = self.mask_token.to(dtype=transition_tokens.dtype)
        transition_tokens = torch.where(
            reconstruction_mask.unsqueeze(-1),
            mask_token,
            transition_tokens,
        )
        summary = self.summary_token.to(dtype=transition_tokens.dtype).expand(
            batch,
            -1,
            -1,
        )
        tokens = torch.cat((summary, transition_tokens), dim=1)
        positions = self.position_embedding[:, : steps + 1].to(dtype=tokens.dtype)
        tokens = tokens + positions
        transformer_padding = torch.cat(
            (
                torch.zeros(
                    (batch, 1),
                    dtype=torch.bool,
                    device=future_states.device,
                ),
                padding_mask,
            ),
            dim=1,
        )
        encoded = self.transformer(
            tokens,
            src_key_padding_mask=transformer_padding,
        )
        latent = self.latent_head(encoded[:, 0])
        projection = self.projection_head(latent)
        encoded_steps = encoded[:, 1:].masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )

        decoded_tokens = self.decoder_latent(latent).unsqueeze(1)
        decoded_tokens = decoded_tokens + self.position_embedding[:, 1 : steps + 1].to(
            dtype=decoded_tokens.dtype
        )
        decoded = self.decoder(decoded_tokens)
        decoded = decoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        reconstruction = FutureReconstruction(
            states=decoded[..., : self.state_dim],
            actions=decoded[..., self.state_dim :],
        )
        return FutureTrajectoryEncoding(
            latent=latent,
            projection=projection,
            encoded_steps=encoded_steps,
            reconstruction=reconstruction,
            padding_mask=padding_mask,
            reconstruction_mask=reconstruction_mask,
        )


def masked_reconstruction_loss(
    encoding: FutureTrajectoryEncoding,
    future_states: torch.Tensor,
    future_actions: torch.Tensor,
) -> torch.Tensor:
    """Mean squared reconstruction error on selected, non-padded elements."""
    if not isinstance(encoding, FutureTrajectoryEncoding):
        raise TypeError("encoding must be a FutureTrajectoryEncoding")
    if encoding.reconstruction.states.shape != future_states.shape:
        raise ValueError("future_states shape does not match reconstruction")
    if encoding.reconstruction.actions.shape != future_actions.shape:
        raise ValueError("future_actions shape does not match reconstruction")
    require_float_tensor(
        future_states,
        name="future_states",
        ndim=3,
        last_dim=int(future_states.shape[-1]),
    )
    require_float_tensor(
        future_actions,
        name="future_actions",
        ndim=3,
        last_dim=int(future_actions.shape[-1]),
    )
    selected = encoding.reconstruction_mask & ~encoding.padding_mask
    predictions = torch.cat(
        (
            encoding.reconstruction.states,
            encoding.reconstruction.actions,
        ),
        dim=-1,
    )
    targets = torch.cat((future_states, future_actions), dim=-1)
    if not bool(selected.any()):
        return predictions.sum() * 0.0
    squared_error = (predictions - targets).square()
    return squared_error.masked_select(selected.unsqueeze(-1)).mean()
