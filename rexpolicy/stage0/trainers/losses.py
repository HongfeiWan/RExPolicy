"""Stage 0 manifold objectives without VLA or image dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as functional

from rexpolicy.stage0.models.future_encoder import (
    FutureTrajectoryEncoding,
    masked_reconstruction_loss,
)


@dataclass(frozen=True)
class DiversityLosses:
    variance: torch.Tensor
    covariance: torch.Tensor


@dataclass(frozen=True)
class ManifoldLosses:
    reconstruction: torch.Tensor
    contrastive: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor
    total: torch.Tensor


def _metadata(
    value: Sequence[Any],
    *,
    expected: int,
    name: str,
) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(value)
    if len(result) != expected:
        raise ValueError(f"{name} length must match the embedding batch")
    return result


def _gather_for_contrastive(
    embeddings: torch.Tensor,
    trajectory_ids: tuple[Any, ...],
    reset_group_ids: tuple[Any, ...],
    starts: tuple[int, ...],
    *,
    gather_distributed: bool,
    group: Any,
) -> tuple[torch.Tensor, tuple[Any, ...], tuple[Any, ...], tuple[int, ...]]:
    if not gather_distributed:
        return embeddings, trajectory_ids, reset_group_ids, starts
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "distributed contrastive gather requires an initialized group"
        )
    from torch.distributed.nn.functional import all_gather

    gathered_embeddings = torch.cat(all_gather(embeddings, group=group), dim=0)
    world_size = torch.distributed.get_world_size(group=group)
    metadata: list[Any] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(
        metadata,
        (trajectory_ids, reset_group_ids, starts),
        group=group,
    )
    trajectories = tuple(value for item in metadata for value in item[0])
    reset_groups = tuple(value for item in metadata for value in item[1])
    offsets = tuple(int(value) for item in metadata for value in item[2])
    if len(trajectories) != gathered_embeddings.shape[0]:
        raise RuntimeError("distributed contrastive batch sizes do not align")
    return gathered_embeddings, trajectories, reset_groups, offsets


def temporal_group_contrastive_loss(
    embeddings: torch.Tensor,
    trajectory_ids: Sequence[Any],
    reset_group_ids: Sequence[Any],
    starts: Sequence[int],
    *,
    temporal_radius: int,
    temperature: float,
    gather_distributed: bool = False,
    group: Any = None,
) -> torch.Tensor:
    """Multi-positive InfoNCE with explicit temporal and group supervision.

    A positive is a nearby window from the same trajectory. Windows from two
    trajectories in the same reset-state group are excluded from the
    denominator instead of being treated as automatic different-mode
    negatives. Every anchor must have both a compatible positive and an
    admissible negative, otherwise the batch fails rather than returning a
    silently degenerate zero objective.
    """
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 2
        or embeddings.shape[0] < 1
        or embeddings.shape[1] < 1
        or not torch.is_floating_point(embeddings)
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError("embeddings must be finite floating [batch, features]")
    if type(temporal_radius) is not int or temporal_radius < 1:
        raise ValueError("temporal_radius must be a positive integer")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ValueError("temperature must be finite and positive")
    count = int(embeddings.shape[0])
    trajectories = _metadata(
        trajectory_ids,
        expected=count,
        name="trajectory_ids",
    )
    reset_groups = _metadata(
        reset_group_ids,
        expected=count,
        name="reset_group_ids",
    )
    offsets_raw = _metadata(starts, expected=count, name="starts")
    if any(type(value) is not int or value < 0 for value in offsets_raw):
        raise ValueError("starts must contain non-negative integers")
    offsets = tuple(int(value) for value in offsets_raw)
    embeddings, trajectories, reset_groups, offsets = _gather_for_contrastive(
        embeddings,
        trajectories,
        reset_groups,
        offsets,
        gather_distributed=gather_distributed,
        group=group,
    )
    count = int(embeddings.shape[0])
    diagonal = torch.eye(count, dtype=torch.bool, device=embeddings.device)
    same_trajectory = torch.tensor(
        [left == right for left in trajectories for right in trajectories],
        dtype=torch.bool,
        device=embeddings.device,
    ).reshape(count, count)
    same_reset_group = torch.tensor(
        [left == right for left in reset_groups for right in reset_groups],
        dtype=torch.bool,
        device=embeddings.device,
    ).reshape(count, count)
    offset_tensor = torch.tensor(offsets, device=embeddings.device)
    nearby = (offset_tensor[:, None] - offset_tensor[None, :]).abs() <= temporal_radius
    positive = same_trajectory & nearby & ~diagonal
    ambiguous_cross_trajectory = same_reset_group & ~same_trajectory
    candidates = ~diagonal & ~ambiguous_cross_trajectory
    negative = candidates & ~positive
    if not bool(positive.any(dim=1).all()):
        raise ValueError(
            "every contrastive anchor needs a same-trajectory temporal positive"
        )
    if not bool(negative.any(dim=1).all()):
        raise ValueError("every contrastive anchor needs an admissible negative")

    normalized = functional.normalize(embeddings, dim=-1)
    logits = normalized @ normalized.transpose(0, 1)
    logits = logits / float(temperature)
    minimum = torch.finfo(logits.dtype).min
    denominator = torch.logsumexp(logits.masked_fill(~candidates, minimum), dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive, minimum), dim=1)
    return (denominator - numerator).mean()


def variance_covariance_losses(
    latents: torch.Tensor,
    *,
    target_std: float = 1.0,
    epsilon: float = 1.0e-4,
) -> DiversityLosses:
    """VICReg-style variance floor and off-diagonal covariance penalty."""
    if (
        not isinstance(latents, torch.Tensor)
        or latents.ndim != 2
        or latents.shape[0] < 1
        or latents.shape[1] < 1
        or not torch.is_floating_point(latents)
        or not bool(torch.isfinite(latents).all())
    ):
        raise ValueError("latents must be finite floating [batch, features]")
    if not math.isfinite(float(target_std)) or float(target_std) <= 0.0:
        raise ValueError("target_std must be finite and positive")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    centered = latents - latents.mean(dim=0, keepdim=True)
    standard_deviation = torch.sqrt(centered.square().mean(dim=0) + float(epsilon))
    variance = functional.relu(float(target_std) - standard_deviation).mean()
    if latents.shape[0] < 2:
        covariance = latents.sum() * 0.0
    else:
        matrix = centered.transpose(0, 1) @ centered
        matrix = matrix / (latents.shape[0] - 1)
        diagonal = torch.diagonal(matrix)
        covariance = (matrix.square().sum() - diagonal.square().sum()) / latents.shape[
            1
        ]
    return DiversityLosses(variance=variance, covariance=covariance)


def manifold_losses(
    encoding: FutureTrajectoryEncoding,
    future_states: torch.Tensor,
    future_actions: torch.Tensor,
    trajectory_ids: Sequence[Any],
    reset_group_ids: Sequence[Any],
    starts: Sequence[int],
    *,
    reconstruction_weight: float,
    contrastive_weight: float,
    variance_weight: float,
    covariance_weight: float,
    temporal_radius: int,
    contrastive_temperature: float,
    variance_target_std: float,
    variance_epsilon: float,
    gather_distributed: bool,
    group: Any = None,
) -> ManifoldLosses:
    reconstruction = masked_reconstruction_loss(
        encoding,
        future_states,
        future_actions,
    )
    contrastive = temporal_group_contrastive_loss(
        encoding.projection,
        trajectory_ids,
        reset_group_ids,
        starts,
        temporal_radius=temporal_radius,
        temperature=contrastive_temperature,
        gather_distributed=gather_distributed,
        group=group,
    )
    diversity = variance_covariance_losses(
        encoding.latent,
        target_std=variance_target_std,
        epsilon=variance_epsilon,
    )
    total = (
        reconstruction_weight * reconstruction
        + contrastive_weight * contrastive
        + variance_weight * diversity.variance
        + covariance_weight * diversity.covariance
    )
    return ManifoldLosses(
        reconstruction=reconstruction,
        contrastive=contrastive,
        variance=diversity.variance,
        covariance=diversity.covariance,
        total=total,
    )
