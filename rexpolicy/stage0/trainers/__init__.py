"""Training primitives for the isolated state-only Stage 0 path."""

from .batch import Stage0WindowBatch, collate_success_windows
from .losses import (
    DiversityLosses,
    ManifoldLosses,
    mode_invariant_alignment_loss,
    temporal_group_contrastive_loss,
    variance_covariance_losses,
)
from .manifold import (
    ManifoldStepMetrics,
    Stage0ManifoldTrainer,
    Stage0ManifoldTrainerConfig,
)
from .policy import PolicyStepMetrics, Stage0PolicyTrainer
from .selector import SelectorStepMetrics, Stage0SelectorTrainer


__all__ = [
    "DiversityLosses",
    "ManifoldLosses",
    "ManifoldStepMetrics",
    "PolicyStepMetrics",
    "SelectorStepMetrics",
    "Stage0ManifoldTrainer",
    "Stage0ManifoldTrainerConfig",
    "Stage0PolicyTrainer",
    "Stage0SelectorTrainer",
    "Stage0WindowBatch",
    "collate_success_windows",
    "mode_invariant_alignment_loss",
    "temporal_group_contrastive_loss",
    "variance_covariance_losses",
]
