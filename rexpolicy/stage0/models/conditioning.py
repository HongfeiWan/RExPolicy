"""Generic state/success-token conditioning contract for Stage 0 policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ._validation import (
    require_bool_mask,
    require_float_tensor,
    require_positive_int,
)


@dataclass(frozen=True)
class ConditionTokens:
    """Condition sequence consumed by the lightweight policy.

    ``attention_mask`` uses the model-facing convention: ``True`` means a
    token is valid and may be attended to.
    """

    tokens: torch.Tensor
    attention_mask: torch.Tensor

    def __post_init__(self) -> None:
        require_float_tensor(self.tokens, name="tokens", ndim=3)
        batch, token_count, model_dim = self.tokens.shape
        if batch <= 0 or token_count <= 0 or model_dim <= 0:
            raise ValueError(
                "tokens must have non-empty batch, token, and feature axes"
            )
        require_bool_mask(
            self.attention_mask,
            name="attention_mask",
            shape=(int(batch), int(token_count)),
            device=self.tokens.device,
        )
        if bool((~self.attention_mask).all(dim=1).any()):
            raise ValueError("every condition row must contain a valid token")

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def model_dim(self) -> int:
        return int(self.tokens.shape[2])

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ConditionTokens:
        tokens = self.tokens.to(device=device, dtype=dtype)
        return ConditionTokens(
            tokens=tokens,
            attention_mask=self.attention_mask.to(device=tokens.device),
        )


class StateLatentConditionEncoder(nn.Module):
    """Project one current-state token and an optional success-latent token."""

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        model_dim: int,
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = require_positive_int(state_dim, name="state_dim")
        self.latent_dim = require_positive_int(latent_dim, name="latent_dim")
        self.model_dim = require_positive_int(model_dim, name="model_dim")
        if hidden_dim is None:
            hidden_dim = self.model_dim
        hidden_dim = require_positive_int(hidden_dim, name="hidden_dim")

        self.state_projector = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
        )
        self.latent_projector = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
        )
        self.state_token_type = nn.Parameter(torch.empty(1, 1, self.model_dim))
        self.latent_token_type = nn.Parameter(torch.empty(1, 1, self.model_dim))
        nn.init.normal_(self.state_token_type, std=0.02)
        nn.init.normal_(self.latent_token_type, std=0.02)

    def forward(
        self,
        current_state: torch.Tensor,
        success_latent: torch.Tensor | None = None,
    ) -> ConditionTokens:
        """Build state-only (no-z) or state-plus-success condition tokens."""
        require_float_tensor(
            current_state,
            name="current_state",
            ndim=2,
            last_dim=self.state_dim,
        )
        state_token = self.state_projector(current_state).unsqueeze(1)
        state_token = state_token + self.state_token_type.to(dtype=state_token.dtype)
        tokens = [state_token]
        if success_latent is not None:
            require_float_tensor(
                success_latent,
                name="success_latent",
                ndim=2,
                last_dim=self.latent_dim,
            )
            if (
                success_latent.shape[0] != current_state.shape[0]
                or success_latent.device != current_state.device
                or success_latent.dtype != current_state.dtype
            ):
                raise ValueError(
                    "success_latent must share state batch, device, and dtype"
                )
            latent_token = self.latent_projector(success_latent).unsqueeze(1)
            latent_token = latent_token + self.latent_token_type.to(
                dtype=latent_token.dtype
            )
            tokens.append(latent_token)
        result = torch.cat(tokens, dim=1)
        return ConditionTokens(
            tokens=result,
            attention_mask=torch.ones(
                result.shape[:2],
                dtype=torch.bool,
                device=result.device,
            ),
        )
