"""Hash-bound training view that removes legacy transfer-phase leakage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA
from rexpolicy.stage0.types import canonical_fingerprint


GRASP_LIFT_STATE_VIEW_SCHEMA_VERSION = 1
GRASP_LIFT_STATE_VIEW_ID = "rexpolicy/stage0-grasp-lift-state-view/v1"
GRASP_LIFT_FIXED_PHASE_ONE_HOT = (1.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class GraspLiftStateView:
    """Semantic identity of the Grasp-Lift model-facing 79D view."""

    schema_version: int = GRASP_LIFT_STATE_VIEW_SCHEMA_VERSION
    view_id: str = GRASP_LIFT_STATE_VIEW_ID

    def __post_init__(self) -> None:
        if self.schema_version != GRASP_LIFT_STATE_VIEW_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {GRASP_LIFT_STATE_VIEW_SCHEMA_VERSION}"
            )
        if self.view_id != GRASP_LIFT_STATE_VIEW_ID:
            raise ValueError(f"view_id must be {GRASP_LIFT_STATE_VIEW_ID!r}")

    def to_record(self) -> dict[str, Any]:
        phase_slice = DEFAULT_STAGE0_STATE_SCHEMA.slice("task_phase_one_hot")
        return {
            "fixed_phase_one_hot": list(GRASP_LIFT_FIXED_PHASE_ONE_HOT),
            "operation": "replace legacy transfer task phase before all model use",
            "phase_slice": [phase_slice.start, phase_slice.stop],
            "raw_trajectory_policy": "retain original 79D state unchanged",
            "schema_version": self.schema_version,
            "state_schema_sha256": DEFAULT_STAGE0_STATE_SCHEMA.sha256,
            "view_id": self.view_id,
        }

    @property
    def sha256(self) -> str:
        return canonical_fingerprint(self.to_record())

    def apply(self, state: torch.Tensor, *, clone: bool = True) -> torch.Tensor:
        """Return the model-facing state with a constant non-privileged phase."""
        if not isinstance(state, torch.Tensor):
            raise TypeError("state must be a Torch tensor")
        if state.ndim < 2 or state.shape[-1] != DEFAULT_STAGE0_STATE_SCHEMA.dimension:
            raise ValueError(
                "state must have at least two dimensions and last dimension "
                f"{DEFAULT_STAGE0_STATE_SCHEMA.dimension}"
            )
        if not state.is_floating_point() or not torch.isfinite(state).all():
            raise ValueError("state must be finite floating point")
        viewed = state.clone() if clone else state
        viewed[..., DEFAULT_STAGE0_STATE_SCHEMA.slice("task_phase_one_hot")] = (
            viewed.new_tensor(GRASP_LIFT_FIXED_PHASE_ONE_HOT)
        )
        return viewed


DEFAULT_GRASP_LIFT_STATE_VIEW = GraspLiftStateView()


def apply_grasp_lift_state_view(
    state: torch.Tensor, *, clone: bool = True
) -> torch.Tensor:
    """Apply :data:`DEFAULT_GRASP_LIFT_STATE_VIEW`."""
    return DEFAULT_GRASP_LIFT_STATE_VIEW.apply(state, clone=clone)


__all__ = [
    "DEFAULT_GRASP_LIFT_STATE_VIEW",
    "GRASP_LIFT_FIXED_PHASE_ONE_HOT",
    "GRASP_LIFT_STATE_VIEW_ID",
    "GRASP_LIFT_STATE_VIEW_SCHEMA_VERSION",
    "GraspLiftStateView",
    "apply_grasp_lift_state_view",
]
