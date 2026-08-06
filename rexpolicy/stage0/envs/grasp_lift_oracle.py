"""Reward-free grasp-and-lift oracle for the Stage 0 state contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from rexpolicy.stage0.envs.state_schema import (
    DEFAULT_STAGE0_STATE_SCHEMA,
    Stage0StateSchema,
)


GRASP_LIFT_ORACLE_ID = "rexpolicy/stage0-grasp-lift-oracle/v1"


def _rotation_6d_rows(value: torch.Tensor) -> torch.Tensor:
    """Reconstruct orthonormal row-major rotation matrices from 6D rows."""

    if value.ndim != 2 or value.shape[1] != 6 or not torch.isfinite(value).all():
        raise ValueError("object_rotation_6d must be finite [B, 6]")
    row0 = value[:, :3]
    row1 = value[:, 3:]
    epsilon = torch.finfo(value.dtype).eps
    norm0 = torch.linalg.vector_norm(row0, dim=-1, keepdim=True)
    if torch.any(norm0 <= epsilon):
        raise ValueError("object_rotation_6d contains a degenerate first row")
    row0 = row0 / norm0
    row1 = row1 - torch.sum(row0 * row1, dim=-1, keepdim=True) * row0
    norm1 = torch.linalg.vector_norm(row1, dim=-1, keepdim=True)
    if torch.any(norm1 <= epsilon):
        raise ValueError("object_rotation_6d contains collinear rows")
    row1 = row1 / norm1
    row2 = torch.linalg.cross(row0, row1, dim=-1)
    return torch.stack((row0, row1, row2), dim=1)


def _bool_vector(value: Any, *, name: str, count: int, device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Torch tensor")
    if value.dtype != torch.bool or tuple(value.shape) != (count,):
        raise ValueError(f"{name} must be bool [{count}]")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}")
    return value


def _integer_vector(
    value: Any, *, name: str, count: int, device: torch.device
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Torch tensor")
    if value.dtype not in (torch.int32, torch.int64) or tuple(value.shape) != (count,):
        raise ValueError(f"{name} must be an integer [{count}]")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}")
    return value


@dataclass(frozen=True)
class GraspLiftTransientEvents:
    """Safety/contact facts observed across one complete control interval."""

    forbidden_hand_contact: torch.Tensor
    opposed_grasp_max_consecutive_physics_frames: torch.Tensor
    had_hand_contact: torch.Tensor
    collision_buffer_overflow: torch.Tensor

    def validate(self, *, count: int, device: torch.device) -> None:
        _bool_vector(
            self.forbidden_hand_contact,
            name="forbidden_hand_contact",
            count=count,
            device=device,
        )
        _integer_vector(
            self.opposed_grasp_max_consecutive_physics_frames,
            name="opposed_grasp_max_consecutive_physics_frames",
            count=count,
            device=device,
        )
        _bool_vector(
            self.had_hand_contact,
            name="had_hand_contact",
            count=count,
            device=device,
        )
        _bool_vector(
            self.collision_buffer_overflow,
            name="collision_buffer_overflow",
            count=count,
            device=device,
        )


def grasp_lift_transient_events_from_info(
    info: Mapping[str, Any], *, count: int, device: torch.device
) -> GraspLiftTransientEvents:
    """Extract GPU-resident oracle events from a Newton step info mapping."""

    if not isinstance(info, Mapping):
        raise TypeError("info must be a mapping")
    diagnostics = info.get("control_step_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise KeyError("info is missing control_step_diagnostics")

    rigid_overflow = diagnostics.get("rigid_contact_overflow_frame_count")
    triangle_overflow = diagnostics.get("triangle_pair_overflow_frame_count")
    if not isinstance(rigid_overflow, torch.Tensor) or not isinstance(
        triangle_overflow, torch.Tensor
    ):
        raise TypeError("collision overflow diagnostics must be Torch tensors")
    if rigid_overflow.device != device or triangle_overflow.device != device:
        raise ValueError("collision overflow diagnostics must share the oracle device")
    overflow = torch.logical_or(rigid_overflow > 0, triangle_overflow > 0)
    overflow = overflow.reshape(-1).any().expand(count)

    events = GraspLiftTransientEvents(
        forbidden_hand_contact=info[
            "forbidden_hand_contact_any_frame_this_control_step"
        ],
        opposed_grasp_max_consecutive_physics_frames=info[
            "opposed_grasp_max_consecutive_physics_frames_this_control_step"
        ],
        had_hand_contact=info["had_hand_contact_this_control_step"],
        collision_buffer_overflow=overflow,
    )
    events.validate(count=count, device=device)
    return events


@dataclass(frozen=True)
class GraspLiftOracleResult:
    """Per-world authoritative outcome and interpretable physical evidence."""

    success: torch.Tensor
    failure: torch.Tensor
    grasp_confirmed: torch.Tensor
    current_opposed_grasp: torch.Tensor
    lift_height: torch.Tensor
    lateral_displacement: torch.Tensor
    bottle_tilt: torch.Tensor
    success_hold_steps: torch.Tensor
    drop_gap_steps: torch.Tensor
    forbidden_contact_violation: torch.Tensor
    lateral_displacement_violation: torch.Tensor
    bottle_tilt_violation: torch.Tensor
    dropped_grasp_violation: torch.Tensor
    collision_buffer_overflow_violation: torch.Tensor


class GraspLiftSuccessOracle:
    """Latch deterministic grasp-lift outcomes independently of reward shaping."""

    def __init__(
        self,
        num_envs: int,
        *,
        lift_height_m: float = 0.030,
        success_hold_control_steps: int = 2,
        lateral_displacement_limit_m: float = 0.030,
        bottle_tilt_limit_rad: float = math.radians(30.0),
        grasp_confirm_physics_frames: int = 6,
        drop_confirm_control_steps: int = 2,
        schema: Stage0StateSchema = DEFAULT_STAGE0_STATE_SCHEMA,
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if min(
            lift_height_m,
            lateral_displacement_limit_m,
            bottle_tilt_limit_rad,
        ) <= 0.0:
            raise ValueError("grasp-lift physical thresholds must be positive")
        if min(
            success_hold_control_steps,
            grasp_confirm_physics_frames,
            drop_confirm_control_steps,
        ) < 1:
            raise ValueError("grasp-lift confirmation counts must be positive")
        self.num_envs = int(num_envs)
        self.lift_height_m = float(lift_height_m)
        self.success_hold_control_steps = int(success_hold_control_steps)
        self.lateral_displacement_limit_m = float(lateral_displacement_limit_m)
        self.bottle_tilt_limit_rad = float(bottle_tilt_limit_rad)
        self.grasp_confirm_physics_frames = int(grasp_confirm_physics_frames)
        self.drop_confirm_control_steps = int(drop_confirm_control_steps)
        self.schema = schema
        self._initial_object_position: torch.Tensor | None = None
        self._initial_object_z_axis: torch.Tensor | None = None
        self._lift_axis: torch.Tensor | None = None
        self._success_hold: torch.Tensor | None = None
        self._drop_gap: torch.Tensor | None = None
        self._grasp_confirmed: torch.Tensor | None = None
        self._success: torch.Tensor | None = None
        self._failure: torch.Tensor | None = None
        self._forbidden_violation: torch.Tensor | None = None
        self._lateral_violation: torch.Tensor | None = None
        self._tilt_violation: torch.Tensor | None = None
        self._drop_violation: torch.Tensor | None = None
        self._overflow_violation: torch.Tensor | None = None

    def reset(
        self, state: torch.Tensor, world_mask: torch.Tensor | None = None
    ) -> None:
        self._validate_state(state)
        object_position = state[:, self.schema.slice("object_position")]
        rotation = _rotation_6d_rows(
            state[:, self.schema.slice("object_rotation_6d")]
        )
        object_z_axis = rotation[:, :, 2]
        object_to_goal = state[:, self.schema.slice("object_to_goal")]
        lift_axis_norm = torch.linalg.vector_norm(
            object_to_goal, dim=-1, keepdim=True
        )
        if torch.any(lift_axis_norm <= torch.finfo(state.dtype).eps):
            raise ValueError("object_to_goal must define a non-zero lift axis")
        lift_axis = object_to_goal / lift_axis_norm
        if self._initial_object_position is None:
            if world_mask is not None:
                raise RuntimeError("the first oracle reset must initialize every world")
            self._initial_object_position = object_position.clone()
            self._initial_object_z_axis = object_z_axis.clone()
            self._lift_axis = lift_axis.clone()
            self._success_hold = torch.zeros(
                self.num_envs, dtype=torch.int64, device=state.device
            )
            self._drop_gap = torch.zeros_like(self._success_hold)
            self._grasp_confirmed = torch.zeros(
                self.num_envs, dtype=torch.bool, device=state.device
            )
            self._success = torch.zeros_like(self._grasp_confirmed)
            self._failure = torch.zeros_like(self._grasp_confirmed)
            self._forbidden_violation = torch.zeros_like(self._grasp_confirmed)
            self._lateral_violation = torch.zeros_like(self._grasp_confirmed)
            self._tilt_violation = torch.zeros_like(self._grasp_confirmed)
            self._drop_violation = torch.zeros_like(self._grasp_confirmed)
            self._overflow_violation = torch.zeros_like(self._grasp_confirmed)
            return

        self._require_runtime(device=state.device)
        if world_mask is None:
            world_mask = torch.ones(
                self.num_envs, dtype=torch.bool, device=state.device
            )
        else:
            _bool_vector(
                world_mask,
                name="world_mask",
                count=self.num_envs,
                device=state.device,
            )
        self._initial_object_position[world_mask] = object_position[world_mask]
        self._initial_object_z_axis[world_mask] = object_z_axis[world_mask]
        self._lift_axis[world_mask] = lift_axis[world_mask]
        self._success_hold[world_mask] = 0
        self._drop_gap[world_mask] = 0
        self._grasp_confirmed[world_mask] = False
        self._success[world_mask] = False
        self._failure[world_mask] = False
        self._forbidden_violation[world_mask] = False
        self._lateral_violation[world_mask] = False
        self._tilt_violation[world_mask] = False
        self._drop_violation[world_mask] = False
        self._overflow_violation[world_mask] = False

    def evaluate(
        self, state: torch.Tensor, events: GraspLiftTransientEvents
    ) -> GraspLiftOracleResult:
        self._validate_state(state)
        self._require_runtime(device=state.device)
        events.validate(count=self.num_envs, device=state.device)

        object_position = state[:, self.schema.slice("object_position")]
        rotation = _rotation_6d_rows(
            state[:, self.schema.slice("object_rotation_6d")]
        )
        current_z_axis = rotation[:, :, 2]
        current_opposed = (
            state[:, self.schema.slice("is_grasped")].squeeze(-1) > 0.5
        )

        displacement = object_position - self._initial_object_position
        signed_lift = torch.sum(
            displacement * self._lift_axis, dim=-1
        )
        lift = torch.clamp_min(signed_lift, 0.0)
        lateral = torch.linalg.vector_norm(
            displacement - signed_lift[:, None] * self._lift_axis,
            dim=-1,
        )
        tilt_cosine = torch.sum(
            current_z_axis * self._initial_object_z_axis, dim=-1
        )
        tilt = torch.acos(torch.clamp(tilt_cosine, -1.0, 1.0))

        active = ~(self._success | self._failure)
        confirmed_this_step = active & (
            events.opposed_grasp_max_consecutive_physics_frames
            >= self.grasp_confirm_physics_frames
        )
        self._grasp_confirmed.logical_or_(confirmed_this_step)
        missing_support = active & self._grasp_confirmed & ~events.had_hand_contact
        next_drop_gap = torch.where(
            missing_support,
            self._drop_gap + 1,
            torch.zeros_like(self._drop_gap),
        )
        self._drop_gap.copy_(
            torch.where(
                active,
                next_drop_gap,
                self._drop_gap,
            )
        )

        forbidden = events.forbidden_hand_contact
        lateral_violation = lateral > self.lateral_displacement_limit_m
        tilt_violation = tilt > self.bottle_tilt_limit_rad
        dropped = self._drop_gap >= self.drop_confirm_control_steps
        overflow = events.collision_buffer_overflow
        self._forbidden_violation.logical_or_(active & forbidden)
        self._lateral_violation.logical_or_(active & lateral_violation)
        self._tilt_violation.logical_or_(active & tilt_violation)
        self._drop_violation.logical_or_(active & dropped)
        self._overflow_violation.logical_or_(active & overflow)
        new_failure = active & (
            forbidden | lateral_violation | tilt_violation | dropped | overflow
        )
        self._failure.logical_or_(new_failure)

        eligible = (
            active
            & self._grasp_confirmed
            & current_opposed
            & (lift >= self.lift_height_m)
            & ~new_failure
        )
        next_success_hold = torch.where(
            eligible,
            self._success_hold + 1,
            torch.zeros_like(self._success_hold),
        )
        self._success_hold.copy_(
            torch.where(
                active,
                next_success_hold,
                self._success_hold,
            )
        )
        self._success.logical_or_(
            (self._success_hold >= self.success_hold_control_steps)
            & ~self._failure
        )

        return GraspLiftOracleResult(
            success=self._success.clone(),
            failure=self._failure.clone(),
            grasp_confirmed=self._grasp_confirmed.clone(),
            current_opposed_grasp=current_opposed,
            lift_height=lift,
            lateral_displacement=lateral,
            bottle_tilt=tilt,
            success_hold_steps=self._success_hold.clone(),
            drop_gap_steps=self._drop_gap.clone(),
            forbidden_contact_violation=self._forbidden_violation.clone(),
            lateral_displacement_violation=self._lateral_violation.clone(),
            bottle_tilt_violation=self._tilt_violation.clone(),
            dropped_grasp_violation=self._drop_violation.clone(),
            collision_buffer_overflow_violation=self._overflow_violation.clone(),
        )

    def _validate_state(self, state: torch.Tensor) -> None:
        if not isinstance(state, torch.Tensor):
            raise TypeError("state must be a Torch tensor")
        if tuple(state.shape) != (self.num_envs, self.schema.dimension):
            raise ValueError(
                f"state must have shape ({self.num_envs}, {self.schema.dimension})"
            )
        if not torch.is_floating_point(state) or not torch.isfinite(state).all():
            raise ValueError("state must be finite floating point")

    def _require_runtime(self, *, device: torch.device) -> None:
        values = (
            self._initial_object_position,
            self._initial_object_z_axis,
            self._lift_axis,
            self._success_hold,
            self._drop_gap,
            self._grasp_confirmed,
            self._success,
            self._failure,
            self._forbidden_violation,
            self._lateral_violation,
            self._tilt_violation,
            self._drop_violation,
            self._overflow_violation,
        )
        if any(value is None for value in values):
            raise RuntimeError("GraspLiftSuccessOracle.reset() must be called first")
        if any(value.device != device for value in values if value is not None):
            raise ValueError("oracle state and evaluation state must share a device")


__all__ = [
    "GRASP_LIFT_ORACLE_ID",
    "GraspLiftOracleResult",
    "GraspLiftSuccessOracle",
    "GraspLiftTransientEvents",
    "grasp_lift_transient_events_from_info",
]
