"""Optional success-manifold components for RExPolicy v2.

The package root exports the pure-Python configuration and latent memory.  The
neural components are loaded on first access so importing tasking modules never
requires PyTorch.
"""

from __future__ import annotations

from typing import Any

from .config import (
    SUCCESS_MANIFOLD_CONFIG_SCHEMA_VERSION,
    SuccessManifoldConfig,
)
from .memory import (
    LATENT_MEMORY_SCHEMA_VERSION,
    LatentAdmission,
    LatentMemory,
    LatentMemoryEntry,
    LatentNeighbor,
)


_NEURAL_EXPORTS = {
    "DiagonalGaussian": ("selector", "DiagonalGaussian"),
    "DiversityLosses": ("losses", "DiversityLosses"),
    "FutureTrajectoryDecoder": ("encoder", "FutureTrajectoryDecoder"),
    "FutureTrajectoryEncoder": ("encoder", "FutureTrajectoryEncoder"),
    "FutureTrajectoryOutput": ("encoder", "FutureTrajectoryOutput"),
    "FutureTrajectoryReconstruction": (
        "encoder",
        "FutureTrajectoryReconstruction",
    ),
    "SelfSupervisedLosses": ("losses", "SelfSupervisedLosses"),
    "SelectorLosses": ("losses", "SelectorLosses"),
    "ManifoldUpdateMetrics": ("trainer", "ManifoldUpdateMetrics"),
    "SuccessManifoldDdpTrainer": (
        "trainer",
        "SuccessManifoldDdpTrainer",
    ),
    "SuccessModeSelector": ("selector", "SuccessModeSelector"),
    "all_gather_tensor": ("distributed", "all_gather_tensor"),
    "masked_reconstruction_loss": ("losses", "masked_reconstruction_loss"),
    "multi_positive_info_nce_loss": (
        "losses",
        "multi_positive_info_nce_loss",
    ),
    "success_manifold_losses": ("losses", "success_manifold_losses"),
    "success_selector_losses": ("losses", "success_selector_losses"),
    "variance_covariance_diversity_loss": (
        "losses",
        "variance_covariance_diversity_loss",
    ),
}


def __getattr__(name: str) -> Any:
    target = _NEURAL_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    from importlib import import_module

    module = import_module(f"{__name__}.{target[0]}")
    value = getattr(module, target[1])
    globals()[name] = value
    return value


__all__ = [
    "LATENT_MEMORY_SCHEMA_VERSION",
    "SUCCESS_MANIFOLD_CONFIG_SCHEMA_VERSION",
    "DiagonalGaussian",
    "DiversityLosses",
    "FutureTrajectoryDecoder",
    "FutureTrajectoryEncoder",
    "FutureTrajectoryOutput",
    "FutureTrajectoryReconstruction",
    "LatentAdmission",
    "LatentMemory",
    "LatentMemoryEntry",
    "LatentNeighbor",
    "ManifoldUpdateMetrics",
    "SelfSupervisedLosses",
    "SelectorLosses",
    "SuccessManifoldConfig",
    "SuccessManifoldDdpTrainer",
    "SuccessModeSelector",
    "all_gather_tensor",
    "masked_reconstruction_loss",
    "multi_positive_info_nce_loss",
    "success_manifold_losses",
    "success_selector_losses",
    "variance_covariance_diversity_loss",
]
