"""Hash-bound physical and model action contracts for Stage 0 Grasp-Lift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA
from rexpolicy.stage0.types import canonical_fingerprint


GRASP_LIFT_ACTION_SCHEMA_VERSION = 1
GRASP_LIFT_ACTION_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-action/v1"
GRASP_LIFT_MODEL_ACTION_REPRESENTATION_ID = (
    "rexpolicy/hybrid-delta-eef-xyz_absolute-hand-q/v1"
)
GRASP_LIFT_ACTION_DIM = 19
GRASP_LIFT_HAND_JOINT_NAMES = (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
    "index_mcp_roll",
    "ring_mcp_roll",
    "pinky_mcp_roll",
    "thumb_cmc_roll",
)
GRASP_LIFT_EFFECTIVE_ACTION_MASK = (
    (True, True, True) + (False,) * 6 + (True,) * 10
)
GRASP_LIFT_DEFAULT_CLOSE_HAND_TARGET = (
    0.267592,
    1.055395,
    1.088560,
    1.088560,
    1.088560,
    1.088560,
    0.021810,
    0.021810,
    0.062802,
    0.204102,
)


@dataclass(frozen=True)
class GraspLiftActionSchema:
    """Exact semantic identity of stored and learned Grasp-Lift actions."""

    schema_version: int = GRASP_LIFT_ACTION_SCHEMA_VERSION
    schema_id: str = GRASP_LIFT_ACTION_SCHEMA_ID

    def __post_init__(self) -> None:
        if self.schema_version != GRASP_LIFT_ACTION_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {GRASP_LIFT_ACTION_SCHEMA_VERSION}"
            )
        if self.schema_id != GRASP_LIFT_ACTION_SCHEMA_ID:
            raise ValueError(f"schema_id must be {GRASP_LIFT_ACTION_SCHEMA_ID!r}")

    def to_record(self) -> dict[str, Any]:
        return {
            "action_dimension": GRASP_LIFT_ACTION_DIM,
            "default_close_hand_target": list(
                GRASP_LIFT_DEFAULT_CLOSE_HAND_TARGET
            ),
            "effective_action_mask": list(GRASP_LIFT_EFFECTIVE_ACTION_MASK),
            "hand_joint_names": list(GRASP_LIFT_HAND_JOINT_NAMES),
            "model_representation": {
                "eef_xyz_0_3": "current-state-relative delta in right_robot_base",
                "hand_q_9_19": "absolute active-joint target in policy order",
                "representation_id": GRASP_LIFT_MODEL_ACTION_REPRESENTATION_ID,
                "rotation_6d_3_9": "zero and masked",
            },
            "physical_representation": {
                "eef_xyz_0_3": "absolute target in right_robot_base",
                "hand_q_9_19": "projected absolute active-joint target",
                "rotation_6d_3_9": "reset hold target",
                "stored_value": "environment-projected-and-executed action",
            },
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "state_schema_sha256": DEFAULT_STAGE0_STATE_SCHEMA.sha256,
        }

    @property
    def sha256(self) -> str:
        return canonical_fingerprint(self.to_record())


DEFAULT_GRASP_LIFT_ACTION_SCHEMA = GraspLiftActionSchema()


def _validate_state_and_actions(
    state: torch.Tensor,
    action: torch.Tensor,
    *,
    action_name: str,
) -> None:
    if not isinstance(state, torch.Tensor) or not isinstance(action, torch.Tensor):
        raise TypeError("state and action must be Torch tensors")
    if state.ndim != 2:
        raise ValueError(
            "state must have shape "
            f"[B, {DEFAULT_STAGE0_STATE_SCHEMA.dimension}]"
        )
    expected_state = (state.shape[0], DEFAULT_STAGE0_STATE_SCHEMA.dimension)
    if tuple(state.shape) != expected_state:
        raise ValueError(
            "state must have shape "
            f"[B, {DEFAULT_STAGE0_STATE_SCHEMA.dimension}]"
        )
    if tuple(action.shape) != (state.shape[0], GRASP_LIFT_ACTION_DIM):
        raise ValueError(
            f"{action_name} must have shape ({state.shape[0]}, "
            f"{GRASP_LIFT_ACTION_DIM})"
        )
    if state.device != action.device or state.dtype != action.dtype:
        raise ValueError("state and action must share device and dtype")
    if not state.is_floating_point() or not action.is_floating_point():
        raise ValueError("state and action must be floating point")
    if not (torch.isfinite(state).all() and torch.isfinite(action).all()):
        raise ValueError("state and action must be finite")


def grasp_lift_default_close_hand_target_like(state: torch.Tensor) -> torch.Tensor:
    """Materialize the hash-bound close target for every state row."""
    if not isinstance(state, torch.Tensor):
        raise TypeError("state must be a Torch tensor")
    if state.ndim != 2 or state.shape[1] != DEFAULT_STAGE0_STATE_SCHEMA.dimension:
        raise ValueError(
            "state must have shape "
            f"[B, {DEFAULT_STAGE0_STATE_SCHEMA.dimension}]"
        )
    if not state.is_floating_point() or not torch.isfinite(state).all():
        raise ValueError("state must be finite floating point")
    return state.new_tensor(GRASP_LIFT_DEFAULT_CLOSE_HAND_TARGET)[None, :].expand(
        state.shape[0], -1
    ).clone()


def grasp_lift_physical_to_model_action(
    current_state: torch.Tensor,
    executed_action: torch.Tensor,
    reference_action: torch.Tensor,
) -> torch.Tensor:
    """Encode an executed physical action without changing absolute hand q."""
    _validate_state_and_actions(
        current_state, executed_action, action_name="executed_action"
    )
    _validate_state_and_actions(
        current_state, reference_action, action_name="reference_action"
    )
    if not torch.equal(executed_action[:, 3:9], reference_action[:, 3:9]):
        raise ValueError("executed action changed the reset rotation hold")
    model_action = executed_action.clone()
    model_action[:, :3].sub_(
        current_state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position")]
    )
    model_action[:, 3:9].zero_()
    return model_action


def grasp_lift_model_to_physical_action(
    current_state: torch.Tensor,
    model_action: torch.Tensor,
    reference_action: torch.Tensor,
) -> torch.Tensor:
    """Decode a hybrid model output before environment-side projection."""
    _validate_state_and_actions(
        current_state, model_action, action_name="model_action"
    )
    _validate_state_and_actions(
        current_state, reference_action, action_name="reference_action"
    )
    if not torch.equal(
        model_action[:, 3:9], torch.zeros_like(model_action[:, 3:9])
    ):
        raise ValueError("model rotation channels must be exactly zero")
    physical = reference_action.clone()
    physical[:, :3] = (
        current_state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position")]
        + model_action[:, :3]
    )
    physical[:, 9:19] = model_action[:, 9:19]
    return physical


__all__ = [
    "DEFAULT_GRASP_LIFT_ACTION_SCHEMA",
    "GRASP_LIFT_ACTION_DIM",
    "GRASP_LIFT_ACTION_SCHEMA_ID",
    "GRASP_LIFT_ACTION_SCHEMA_VERSION",
    "GRASP_LIFT_DEFAULT_CLOSE_HAND_TARGET",
    "GRASP_LIFT_EFFECTIVE_ACTION_MASK",
    "GRASP_LIFT_HAND_JOINT_NAMES",
    "GRASP_LIFT_MODEL_ACTION_REPRESENTATION_ID",
    "GraspLiftActionSchema",
    "grasp_lift_default_close_hand_target_like",
    "grasp_lift_model_to_physical_action",
    "grasp_lift_physical_to_model_action",
]
