"""Original-space latent statistics and diagnostic-only deterministic PCA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


def _finite_non_negative(value: float, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _fraction(value: float, *, name: str) -> float:
    result = _finite_non_negative(value, name=name)
    if result > 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return result


def _validated_latents(latents: torch.Tensor) -> torch.Tensor:
    if not isinstance(latents, torch.Tensor) or latents.ndim != 2:
        raise ValueError("latents must be a rank-2 tensor [samples, latent_dim]")
    if latents.shape[0] < 2:
        raise ValueError("latents must contain at least two samples")
    if latents.shape[1] < 1:
        raise ValueError("latents must have a non-empty latent dimension")
    if not torch.is_floating_point(latents):
        raise ValueError("latents must be floating point")
    if not bool(torch.isfinite(latents).all()):
        raise ValueError("latents must contain only finite values")
    return latents.detach().to(dtype=torch.float64)


@dataclass(frozen=True)
class CollapseGate:
    """Explicit thresholds for rejecting a collapsed held-out latent set."""

    minimum_dimension_variance: float = 1e-4
    minimum_active_dimension_fraction: float = 0.25
    minimum_effective_rank: float = 2.0
    maximum_mean_absolute_pairwise_cosine: float = 0.99

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_dimension_variance",
            _finite_non_negative(
                self.minimum_dimension_variance,
                name="minimum_dimension_variance",
            ),
        )
        object.__setattr__(
            self,
            "minimum_active_dimension_fraction",
            _fraction(
                self.minimum_active_dimension_fraction,
                name="minimum_active_dimension_fraction",
            ),
        )
        object.__setattr__(
            self,
            "minimum_effective_rank",
            _finite_non_negative(
                self.minimum_effective_rank,
                name="minimum_effective_rank",
            ),
        )
        object.__setattr__(
            self,
            "maximum_mean_absolute_pairwise_cosine",
            _fraction(
                self.maximum_mean_absolute_pairwise_cosine,
                name="maximum_mean_absolute_pairwise_cosine",
            ),
        )


@dataclass(frozen=True)
class LatentStatistics:
    """Held-out statistics computed in the authoritative latent space."""

    sample_count: int
    latent_dim: int
    per_dimension_variance: torch.Tensor
    covariance: torch.Tensor
    covariance_offdiag_mean_absolute: float
    covariance_offdiag_max_absolute: float
    effective_rank: float
    pairwise_cosine_mean: float
    pairwise_cosine_std: float
    pairwise_cosine_minimum: float
    pairwise_cosine_maximum: float
    pairwise_cosine_mean_absolute: float
    active_dimension_count: int
    active_dimension_fraction: float
    collapse_gate_passed: bool
    collapse_reasons: tuple[str, ...]

    @property
    def collapsed(self) -> bool:
        return not self.collapse_gate_passed

    def to_record(self) -> dict[str, Any]:
        return {
            "active_dimension_count": self.active_dimension_count,
            "active_dimension_fraction": self.active_dimension_fraction,
            "collapse_gate_passed": self.collapse_gate_passed,
            "collapse_reasons": list(self.collapse_reasons),
            "covariance": self.covariance.cpu().tolist(),
            "covariance_offdiag_max_absolute": (self.covariance_offdiag_max_absolute),
            "covariance_offdiag_mean_absolute": (self.covariance_offdiag_mean_absolute),
            "effective_rank": self.effective_rank,
            "latent_dim": self.latent_dim,
            "pairwise_cosine_maximum": self.pairwise_cosine_maximum,
            "pairwise_cosine_mean": self.pairwise_cosine_mean,
            "pairwise_cosine_mean_absolute": (self.pairwise_cosine_mean_absolute),
            "pairwise_cosine_minimum": self.pairwise_cosine_minimum,
            "pairwise_cosine_std": self.pairwise_cosine_std,
            "per_dimension_variance": (self.per_dimension_variance.cpu().tolist()),
            "sample_count": self.sample_count,
            "space": "original_latent",
        }


def analyze_latents(
    latents: torch.Tensor,
    *,
    gate: CollapseGate | None = None,
) -> LatentStatistics:
    """Measure held-out diversity without projecting or clustering latents."""

    values = _validated_latents(latents)
    gate = CollapseGate() if gate is None else gate
    if not isinstance(gate, CollapseGate):
        raise TypeError("gate must be a CollapseGate")

    sample_count, latent_dim = (int(axis) for axis in values.shape)
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.mT @ centered / sample_count
    per_dimension_variance = covariance.diagonal().clone()

    if latent_dim > 1:
        offdiag_mask = ~torch.eye(
            latent_dim,
            dtype=torch.bool,
            device=values.device,
        )
        absolute_offdiag = covariance.masked_select(offdiag_mask).abs()
        offdiag_mean = float(absolute_offdiag.mean())
        offdiag_max = float(absolute_offdiag.max())
    else:
        offdiag_mean = 0.0
        offdiag_max = 0.0

    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total_eigenvalue = eigenvalues.sum()
    tolerance = torch.finfo(values.dtype).eps * max(sample_count, latent_dim)
    if float(total_eigenvalue) <= tolerance:
        effective_rank = 0.0
    else:
        probabilities = eigenvalues / total_eigenvalue
        positive = probabilities > tolerance
        entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
        effective_rank = float(entropy.exp())

    normalized = torch.nn.functional.normalize(values, dim=1, eps=tolerance)
    cosine_matrix = normalized @ normalized.mT
    upper = torch.triu_indices(
        sample_count,
        sample_count,
        offset=1,
        device=values.device,
    )
    pairwise_cosine = cosine_matrix[upper[0], upper[1]]
    cosine_mean = float(pairwise_cosine.mean())
    cosine_std = float(pairwise_cosine.std(unbiased=False))
    cosine_minimum = float(pairwise_cosine.min())
    cosine_maximum = float(pairwise_cosine.max())
    cosine_mean_absolute = float(pairwise_cosine.abs().mean())

    active_dimension_count = int(
        (per_dimension_variance > gate.minimum_dimension_variance).sum()
    )
    active_dimension_fraction = active_dimension_count / latent_dim
    reasons: list[str] = []
    if active_dimension_fraction < gate.minimum_active_dimension_fraction:
        reasons.append("active_dimension_fraction_below_minimum")
    if effective_rank < gate.minimum_effective_rank:
        reasons.append("effective_rank_below_minimum")
    if cosine_mean_absolute > gate.maximum_mean_absolute_pairwise_cosine:
        reasons.append("mean_absolute_pairwise_cosine_above_maximum")

    return LatentStatistics(
        sample_count=sample_count,
        latent_dim=latent_dim,
        per_dimension_variance=per_dimension_variance,
        covariance=covariance,
        covariance_offdiag_mean_absolute=offdiag_mean,
        covariance_offdiag_max_absolute=offdiag_max,
        effective_rank=effective_rank,
        pairwise_cosine_mean=cosine_mean,
        pairwise_cosine_std=cosine_std,
        pairwise_cosine_minimum=cosine_minimum,
        pairwise_cosine_maximum=cosine_maximum,
        pairwise_cosine_mean_absolute=cosine_mean_absolute,
        active_dimension_count=active_dimension_count,
        active_dimension_fraction=active_dimension_fraction,
        collapse_gate_passed=not reasons,
        collapse_reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class DeterministicPCA:
    """A sign-canonicalized projection that is never training authority."""

    coordinates: torch.Tensor
    components: torch.Tensor
    mean: torch.Tensor
    explained_variance: torch.Tensor
    explained_variance_ratio: torch.Tensor
    authority: str = "diagnostic_only"
    source_space: str = "original_latent"
    valid_for_clustering: bool = False
    valid_for_training: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "components": self.components.cpu().tolist(),
            "coordinates": self.coordinates.cpu().tolist(),
            "explained_variance": self.explained_variance.cpu().tolist(),
            "explained_variance_ratio": (self.explained_variance_ratio.cpu().tolist()),
            "mean": self.mean.cpu().tolist(),
            "source_space": self.source_space,
            "valid_for_clustering": self.valid_for_clustering,
            "valid_for_training": self.valid_for_training,
        }


def deterministic_pca(
    latents: torch.Tensor,
    *,
    components: int = 2,
) -> DeterministicPCA:
    """Project for display only, with deterministic ordering and component signs."""

    values = _validated_latents(latents)
    if type(components) is not int or components <= 0:
        raise ValueError("components must be a positive integer")
    maximum_components = min(int(values.shape[0]) - 1, int(values.shape[1]))
    if components > maximum_components:
        raise ValueError("components cannot exceed min(samples - 1, latent_dim)")

    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.mT @ centered / (int(values.shape[0]) - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    ordered_values = eigenvalues[order].clamp_min(0.0)
    selected_components = eigenvectors[:, order[:components]].mT.contiguous()

    # Eigenvectors have an arbitrary sign.  Make the largest-absolute loading
    # positive so repeated reports and row permutations use the same orientation.
    pivots = selected_components.abs().argmax(dim=1)
    pivot_values = selected_components[
        torch.arange(components, device=values.device),
        pivots,
    ]
    signs = torch.where(pivot_values < 0.0, -1.0, 1.0)
    selected_components = selected_components * signs.unsqueeze(1)
    coordinates = centered @ selected_components.mT

    total_variance = ordered_values.sum()
    if float(total_variance) == 0.0:
        explained_ratio = torch.zeros_like(ordered_values[:components])
    else:
        explained_ratio = ordered_values[:components] / total_variance

    return DeterministicPCA(
        coordinates=coordinates,
        components=selected_components,
        mean=mean,
        explained_variance=ordered_values[:components],
        explained_variance_ratio=explained_ratio,
    )
