"""Self-supervised objectives for success-future representation learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as functional

from .config import SuccessManifoldConfig
from .distributed import all_gather_tensor, all_gather_values
from .encoder import FutureTrajectoryOutput, FutureTrajectoryReconstruction


@dataclass(frozen=True)
class DiversityLosses:
    variance: torch.Tensor
    covariance: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.variance + self.covariance


@dataclass(frozen=True)
class SelfSupervisedLosses:
    reconstruction: torch.Tensor
    info_nce: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor
    total: torch.Tensor


def _labels(value: Sequence[Any] | torch.Tensor, expected: int) -> list[Any]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError("trajectory_ids tensor must be one-dimensional")
        result = value.detach().cpu().tolist()
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError("trajectory_ids must be a sequence")
        result = list(value)
    if len(result) != expected:
        raise ValueError("trajectory_ids length must match the embedding batch")
    return result


def masked_reconstruction_loss(
    reconstruction: FutureTrajectoryReconstruction,
    future_states: torch.Tensor,
    future_actions: torch.Tensor,
    *,
    reconstruction_mask: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    visual_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared error over feature elements at masked, non-padded steps."""
    if future_states.ndim != 3 or future_actions.ndim != 3:
        raise ValueError("Reconstruction targets must have three dimensions")
    if future_states.shape[:2] != future_actions.shape[:2]:
        raise ValueError("Reconstruction target batch and step shapes must align")
    expected_mask_shape = tuple(future_states.shape[:2])
    if (
        not isinstance(reconstruction_mask, torch.Tensor)
        or reconstruction_mask.dtype != torch.bool
        or tuple(reconstruction_mask.shape) != expected_mask_shape
    ):
        raise ValueError("reconstruction_mask must be bool [batch, steps]")
    if padding_mask is None:
        padding_mask = torch.zeros_like(reconstruction_mask)
    if (
        not isinstance(padding_mask, torch.Tensor)
        or padding_mask.dtype != torch.bool
        or tuple(padding_mask.shape) != expected_mask_shape
    ):
        raise ValueError("padding_mask must be bool [batch, steps]")
    predictions = [reconstruction.states, reconstruction.actions]
    targets = [future_states, future_actions]
    if visual_features is not None:
        if reconstruction.visual_features is None:
            raise ValueError("Decoder has no visual reconstruction head")
        predictions.append(reconstruction.visual_features)
        targets.append(visual_features)
    prediction = torch.cat(predictions, dim=-1)
    target = torch.cat(targets, dim=-1)
    if prediction.shape != target.shape:
        raise ValueError("Reconstruction predictions and targets do not align")
    selected = reconstruction_mask & ~padding_mask
    if not bool(selected.any()):
        return prediction.sum() * 0.0
    squared = (prediction - target).square()
    expanded = selected.unsqueeze(-1).expand_as(squared)
    return squared.masked_select(expanded).mean()


def multi_positive_info_nce_loss(
    embeddings: torch.Tensor,
    trajectory_ids: Sequence[Any] | torch.Tensor,
    *,
    temperature: float,
    gather_distributed: bool = False,
    group: Any = None,
) -> torch.Tensor:
    """InfoNCE where every other window from one trajectory is a positive."""
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [batch, features]")
    if embeddings.shape[1] == 0 or not torch.is_floating_point(embeddings):
        raise ValueError("embeddings must have a floating feature dimension")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or float(temperature) <= 0.0
    ):
        raise ValueError("temperature must be positive")
    labels = _labels(trajectory_ids, embeddings.shape[0])
    if gather_distributed:
        embeddings = all_gather_tensor(embeddings, group=group)
        labels = all_gather_values(labels, group=group)
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0

    normalized = functional.normalize(embeddings, dim=-1)
    logits = normalized @ normalized.transpose(0, 1)
    logits = logits / float(temperature)
    count = logits.shape[0]
    diagonal = torch.eye(count, dtype=torch.bool, device=logits.device)
    positive = torch.tensor(
        [
            left == right
            for left in labels
            for right in labels
        ],
        dtype=torch.bool,
        device=logits.device,
    ).reshape(count, count)
    positive = positive & ~diagonal
    valid = positive.any(dim=1)
    if not bool(valid.any()):
        return embeddings.sum() * 0.0
    minimum = torch.finfo(logits.dtype).min
    denominator = torch.logsumexp(logits.masked_fill(diagonal, minimum), dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive, minimum), dim=1)
    return (denominator[valid] - numerator[valid]).mean()


def variance_covariance_diversity_loss(
    latents: torch.Tensor,
    *,
    target_std: float = 1.0,
    epsilon: float = 1e-4,
    gather_distributed: bool = False,
    group: Any = None,
) -> DiversityLosses:
    """VICReg-style variance floor and off-diagonal covariance penalty."""
    if not isinstance(latents, torch.Tensor) or latents.ndim != 2:
        raise ValueError("latents must have shape [batch, features]")
    if latents.shape[1] == 0 or not torch.is_floating_point(latents):
        raise ValueError("latents must have a floating feature dimension")
    if float(target_std) <= 0.0 or float(epsilon) <= 0.0:
        raise ValueError("target_std and epsilon must be positive")
    if gather_distributed:
        latents = all_gather_tensor(latents, group=group)
    centered = latents - latents.mean(dim=0, keepdim=True)
    variance = centered.square().mean(dim=0)
    standard_deviation = torch.sqrt(variance + float(epsilon))
    variance_loss = functional.relu(float(target_std) - standard_deviation).mean()
    if latents.shape[0] < 2:
        covariance_loss = latents.sum() * 0.0
    else:
        covariance = centered.transpose(0, 1) @ centered
        covariance = covariance / (latents.shape[0] - 1)
        diagonal = torch.diagonal(covariance)
        off_diagonal_square_sum = covariance.square().sum() - diagonal.square().sum()
        covariance_loss = off_diagonal_square_sum / latents.shape[1]
    return DiversityLosses(
        variance=variance_loss,
        covariance=covariance_loss,
    )


def success_manifold_losses(
    output: FutureTrajectoryOutput,
    future_states: torch.Tensor,
    future_actions: torch.Tensor,
    trajectory_ids: Sequence[Any] | torch.Tensor,
    *,
    config: SuccessManifoldConfig,
    visual_features: torch.Tensor | None = None,
    gather_distributed: bool = False,
    group: Any = None,
) -> SelfSupervisedLosses:
    """Compute the weighted Phase-2 training objective from one forward pass."""
    reconstruction = masked_reconstruction_loss(
        output.reconstruction,
        future_states,
        future_actions,
        reconstruction_mask=output.reconstruction_mask,
        padding_mask=output.padding_mask,
        visual_features=visual_features,
    )
    info_nce = multi_positive_info_nce_loss(
        output.projection,
        trajectory_ids,
        temperature=config.info_nce_temperature,
        gather_distributed=gather_distributed,
        group=group,
    )
    diversity = variance_covariance_diversity_loss(
        output.latent,
        target_std=config.variance_target_std,
        gather_distributed=gather_distributed,
        group=group,
    )
    total = (
        float(config.reconstruction_weight) * reconstruction
        + float(config.info_nce_weight) * info_nce
        + float(config.variance_weight) * diversity.variance
        + float(config.covariance_weight) * diversity.covariance
    )
    return SelfSupervisedLosses(
        reconstruction=reconstruction,
        info_nce=info_nce,
        variance=diversity.variance,
        covariance=diversity.covariance,
        total=total,
    )
