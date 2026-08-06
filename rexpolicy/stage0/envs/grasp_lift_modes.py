"""Reward-free scripted authoring modes for the Stage 0 Grasp-Lift task.

The controller is deliberately simulator-independent.  It proposes the
existing 19D absolute EEF/hand action, then consumes the action that the
environment actually projected and executed together with the independent
oracle evidence.  A grasp target is frozen only after that same executed
target survives a second complete control interval.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch

from rexpolicy.stage0.envs.grasp_lift_action import (
    DEFAULT_GRASP_LIFT_ACTION_SCHEMA,
    GRASP_LIFT_ACTION_SCHEMA_ID,
    grasp_lift_default_close_hand_target_like,
)
from rexpolicy.stage0.envs.grasp_lift_oracle import (
    GraspLiftOracleResult,
    GraspLiftTransientEvents,
)
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA
from rexpolicy.stage0.types import canonical_fingerprint


GRASP_LIFT_AUTHORING_CONTROLLER_ID = (
    "rexpolicy/stage0-grasp-lift-authoring-controller/v1"
)
GRASP_LIFT_ACTION_CONTRACT_ID = GRASP_LIFT_ACTION_SCHEMA_ID
_MODE_ID = re.compile(r"^stage0/grasp_lift/[a-z][a-z0-9_]+/v1$")


def _positive_float(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be a finite positive number")


def _unit_interval(value: float, name: str, *, include_zero: bool = False) -> None:
    lower_ok = float(value) >= 0.0 if include_zero else float(value) > 0.0
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not lower_ok
        or float(value) > 1.0
    ):
        interval = "[0, 1]" if include_zero else "(0, 1]"
        raise ValueError(f"{name} must be finite in {interval}")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _check_state(state: torch.Tensor, *, name: str) -> None:
    expected = DEFAULT_STAGE0_STATE_SCHEMA.dimension
    if not isinstance(state, torch.Tensor):
        raise TypeError(f"{name} must be a Torch tensor")
    if tuple(state.shape[1:]) != (expected,) or state.ndim != 2:
        raise ValueError(
            f"{name} must have shape [B, {expected}], got {tuple(state.shape)}"
        )
    if not state.is_floating_point() or not torch.isfinite(state).all():
        raise ValueError(f"{name} must be finite floating point")


def _check_batched_tensor(
    value: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Torch tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} must use dtype {dtype}")


@dataclass(frozen=True)
class GraspLiftAuthoringModeSpec:
    """One immutable, calibrated scripted Grasp-Lift mode."""

    mode_id: str = "stage0/grasp_lift/thumb_index_side/v1"
    pregrasp_offset_base_m: tuple[float, float, float] = (-0.03, 0.19, 0.07)
    close_fraction: float = 0.65
    approach_control_steps: int = 20
    close_schedule_control_steps: int = 16
    minimum_tentative_close_step: int = 4
    maximum_close_control_steps: int = 24
    thumb_lead_gain: float = 1.7
    non_thumb_delay_fraction: float = 0.20
    stable_grasp_control_steps: int = 2
    full_opposed_physics_frames: int = 6
    confirm_hold_control_steps: int = 2
    lift_step_m: float = 0.010
    lift_target_m: float = 0.080
    maximum_lift_control_steps: int = 16
    maximum_success_hold_control_steps: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.mode_id, str) or _MODE_ID.fullmatch(self.mode_id) is None:
            raise ValueError(
                "mode_id must match 'stage0/grasp_lift/<lower_snake_case>/v1'"
            )
        if (
            not isinstance(self.pregrasp_offset_base_m, tuple)
            or len(self.pregrasp_offset_base_m) != 3
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in self.pregrasp_offset_base_m
            )
        ):
            raise ValueError("pregrasp_offset_base_m must contain three finite values")
        _unit_interval(self.close_fraction, "close_fraction")
        _positive_float(self.thumb_lead_gain, "thumb_lead_gain")
        _unit_interval(
            self.non_thumb_delay_fraction,
            "non_thumb_delay_fraction",
            include_zero=True,
        )
        if float(self.non_thumb_delay_fraction) >= 1.0:
            raise ValueError("non_thumb_delay_fraction must be smaller than 1")
        _positive_float(self.lift_step_m, "lift_step_m")
        _positive_float(self.lift_target_m, "lift_target_m")
        for name in (
            "approach_control_steps",
            "close_schedule_control_steps",
            "minimum_tentative_close_step",
            "maximum_close_control_steps",
            "stable_grasp_control_steps",
            "full_opposed_physics_frames",
            "confirm_hold_control_steps",
            "maximum_lift_control_steps",
            "maximum_success_hold_control_steps",
        ):
            _positive_int(getattr(self, name), name)
        if self.minimum_tentative_close_step > self.close_schedule_control_steps:
            raise ValueError(
                "minimum_tentative_close_step cannot exceed close_schedule_control_steps"
            )
        if self.maximum_close_control_steps < self.close_schedule_control_steps:
            raise ValueError(
                "maximum_close_control_steps cannot be smaller than close schedule"
            )
        if self.stable_grasp_control_steps < 2:
            raise ValueError(
                "stable_grasp_control_steps must verify the same target twice"
            )
        if (
            self.lift_step_m * self.maximum_lift_control_steps
            < self.lift_target_m
        ):
            raise ValueError("maximum lift horizon cannot reach lift_target_m")

    def to_record(self) -> dict[str, Any]:
        return {
            "action_contract_id": GRASP_LIFT_ACTION_CONTRACT_ID,
            "action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
            "approach_control_steps": self.approach_control_steps,
            "close_fraction": float(self.close_fraction),
            "close_schedule_control_steps": self.close_schedule_control_steps,
            "confirm_hold_control_steps": self.confirm_hold_control_steps,
            "controller_id": GRASP_LIFT_AUTHORING_CONTROLLER_ID,
            "full_opposed_physics_frames": self.full_opposed_physics_frames,
            "lift_step_m": float(self.lift_step_m),
            "lift_target_m": float(self.lift_target_m),
            "maximum_close_control_steps": self.maximum_close_control_steps,
            "maximum_lift_control_steps": self.maximum_lift_control_steps,
            "maximum_success_hold_control_steps": (
                self.maximum_success_hold_control_steps
            ),
            "minimum_tentative_close_step": self.minimum_tentative_close_step,
            "mode_id": self.mode_id,
            "non_thumb_delay_fraction": float(self.non_thumb_delay_fraction),
            "pregrasp_offset_base_m": [
                float(value) for value in self.pregrasp_offset_base_m
            ],
            "stable_grasp_control_steps": self.stable_grasp_control_steps,
            "state_schema_sha256": DEFAULT_STAGE0_STATE_SCHEMA.sha256,
            "thumb_lead_gain": float(self.thumb_lead_gain),
        }

    @property
    def sha256(self) -> str:
        return canonical_fingerprint(self.to_record())


DEFAULT_GRASP_LIFT_AUTHORING_MODE = GraspLiftAuthoringModeSpec()


class GraspLiftAuthoringPhase(IntEnum):
    """Controller-local phases; these are not the legacy task phase one-hot."""

    APPROACH = 0
    CLOSE = 1
    CONFIRM_HOLD = 2
    LIFT = 3
    SUCCESS_HOLD = 4
    SUCCEEDED = 5
    ORACLE_FAILED = 6
    REJECTED = 7
    EXHAUSTED = 8


_TERMINAL_PHASE_VALUES = (
    int(GraspLiftAuthoringPhase.SUCCEEDED),
    int(GraspLiftAuthoringPhase.ORACLE_FAILED),
    int(GraspLiftAuthoringPhase.REJECTED),
    int(GraspLiftAuthoringPhase.EXHAUSTED),
)


@dataclass(frozen=True)
class GraspLiftAuthoringSnapshot:
    """GPU-resident per-world state for transition-aligned evidence."""

    phase: torch.Tensor
    approach_control_steps: torch.Tensor
    close_control_steps: torch.Tensor
    close_elapsed_control_steps: torch.Tensor
    stable_grasp_control_steps: torch.Tensor
    confirm_hold_control_steps: torch.Tensor
    lift_control_steps: torch.Tensor
    success_hold_control_steps: torch.Tensor
    has_tentative_target: torch.Tensor
    has_frozen_target: torch.Tensor


class GraspLiftAuthoringController:
    """Propose and audit deterministic Grasp-Lift teacher actions."""

    def __init__(
        self,
        spec: GraspLiftAuthoringModeSpec = DEFAULT_GRASP_LIFT_AUTHORING_MODE,
    ) -> None:
        if not isinstance(spec, GraspLiftAuthoringModeSpec):
            raise TypeError("spec must be GraspLiftAuthoringModeSpec")
        self.spec = spec
        self._batch_size: int | None = None
        self._reference_action: torch.Tensor | None = None
        self._pregrasp_target: torch.Tensor | None = None
        self._lift_axis: torch.Tensor | None = None
        self._final_close_hand: torch.Tensor | None = None
        self._phase: torch.Tensor | None = None
        self._approach_steps: torch.Tensor | None = None
        self._close_steps: torch.Tensor | None = None
        self._close_elapsed_steps: torch.Tensor | None = None
        self._stable_steps: torch.Tensor | None = None
        self._confirm_steps: torch.Tensor | None = None
        self._lift_steps: torch.Tensor | None = None
        self._success_hold_steps: torch.Tensor | None = None
        self._tentative_mask: torch.Tensor | None = None
        self._frozen_mask: torch.Tensor | None = None
        self._tentative_eef: torch.Tensor | None = None
        self._tentative_hand: torch.Tensor | None = None
        self._frozen_eef: torch.Tensor | None = None
        self._frozen_hand: torch.Tensor | None = None
        self._lift_start_eef: torch.Tensor | None = None
        self._pending_proposal: torch.Tensor | None = None
        self._pending_active: torch.Tensor | None = None
        self._pending_phase: torch.Tensor | None = None
        self._pending_tentative: torch.Tensor | None = None

    def reset(
        self,
        initial_state: torch.Tensor,
        reference_action: torch.Tensor,
    ) -> None:
        """Capture reset geometry and the open/rotation action reference."""
        _check_state(initial_state, name="initial_state")
        batch_size = initial_state.shape[0]
        _check_batched_tensor(
            reference_action,
            name="reference_action",
            shape=(batch_size, 19),
            device=initial_state.device,
            dtype=initial_state.dtype,
        )
        if not torch.isfinite(reference_action).all():
            raise ValueError("reference action must be finite")
        close_hand_target = grasp_lift_default_close_hand_target_like(initial_state)
        schema = DEFAULT_STAGE0_STATE_SCHEMA
        object_position = initial_state[:, schema.slice("object_position")]
        object_to_goal = initial_state[:, schema.slice("object_to_goal")]
        norm = torch.linalg.vector_norm(object_to_goal, dim=-1, keepdim=True)
        if torch.any(norm <= torch.finfo(initial_state.dtype).eps):
            raise ValueError("object_to_goal must define a non-zero reset lift axis")

        self._batch_size = batch_size
        self._reference_action = reference_action.detach().clone()
        self._pregrasp_target = object_position.detach().clone()
        self._pregrasp_target.add_(
            initial_state.new_tensor(self.spec.pregrasp_offset_base_m)
        )
        self._lift_axis = (object_to_goal / norm).detach().clone()
        open_hand = reference_action[:, 9:19]
        self._final_close_hand = (
            open_hand
            + float(self.spec.close_fraction) * (close_hand_target - open_hand)
        ).detach().clone()
        device = initial_state.device
        self._phase = torch.full(
            (batch_size,),
            int(GraspLiftAuthoringPhase.APPROACH),
            dtype=torch.int64,
            device=device,
        )
        self._approach_steps = torch.zeros_like(self._phase)
        self._close_steps = torch.zeros_like(self._phase)
        self._close_elapsed_steps = torch.zeros_like(self._phase)
        self._stable_steps = torch.zeros_like(self._phase)
        self._confirm_steps = torch.zeros_like(self._phase)
        self._lift_steps = torch.zeros_like(self._phase)
        self._success_hold_steps = torch.zeros_like(self._phase)
        self._tentative_mask = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )
        self._frozen_mask = torch.zeros_like(self._tentative_mask)
        self._tentative_eef = torch.zeros(
            (batch_size, 3), dtype=initial_state.dtype, device=device
        )
        self._tentative_hand = torch.zeros(
            (batch_size, 10), dtype=initial_state.dtype, device=device
        )
        self._frozen_eef = torch.zeros_like(self._tentative_eef)
        self._frozen_hand = torch.zeros_like(self._tentative_hand)
        self._lift_start_eef = torch.zeros_like(self._tentative_eef)
        self._clear_pending()

    @property
    def phase(self) -> torch.Tensor:
        self._require_reset()
        return self._phase.clone()

    @property
    def terminal_mask(self) -> torch.Tensor:
        self._require_reset()
        result = torch.zeros_like(self._phase, dtype=torch.bool)
        for value in _TERMINAL_PHASE_VALUES:
            result.logical_or_(self._phase == value)
        return result

    def snapshot(self) -> GraspLiftAuthoringSnapshot:
        self._require_reset()
        return GraspLiftAuthoringSnapshot(
            phase=self._phase.clone(),
            approach_control_steps=self._approach_steps.clone(),
            close_control_steps=self._close_steps.clone(),
            close_elapsed_control_steps=self._close_elapsed_steps.clone(),
            stable_grasp_control_steps=self._stable_steps.clone(),
            confirm_hold_control_steps=self._confirm_steps.clone(),
            lift_control_steps=self._lift_steps.clone(),
            success_hold_control_steps=self._success_hold_steps.clone(),
            has_tentative_target=self._tentative_mask.clone(),
            has_frozen_target=self._frozen_mask.clone(),
        )

    def propose(
        self,
        current_state: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Propose one action; :meth:`observe` must consume it exactly once."""
        self._validate_runtime_state(current_state)
        if self._pending_proposal is not None:
            raise RuntimeError("observe() must consume the pending proposal first")
        batch_size = self._batch_size
        if active_mask is None:
            active_mask = ~self.terminal_mask
        else:
            _check_batched_tensor(
                active_mask,
                name="active_mask",
                shape=(batch_size,),
                device=current_state.device,
                dtype=torch.bool,
            )
            active_mask = active_mask & ~self.terminal_mask

        schema = DEFAULT_STAGE0_STATE_SCHEMA
        eef = current_state[:, schema.slice("eef_position")]
        hand = current_state[:, schema.slice("hand_joint_position")]
        proposal = self._reference_action.clone()
        proposal[:, :3].copy_(eef)
        proposal[:, 9:19].copy_(hand)

        approach = active_mask & (
            self._phase == int(GraspLiftAuthoringPhase.APPROACH)
        )
        proposal[approach, :3] = self._pregrasp_target[approach]
        proposal[approach, 9:19] = self._reference_action[approach, 9:19]

        close = active_mask & (self._phase == int(GraspLiftAuthoringPhase.CLOSE))
        scheduled = close & ~self._tentative_mask
        next_step = torch.clamp(
            self._close_steps + 1,
            max=self.spec.close_schedule_control_steps,
        ).to(dtype=current_state.dtype)
        progress = next_step / float(self.spec.close_schedule_control_steps)
        non_thumb = torch.clamp(
            (progress - float(self.spec.non_thumb_delay_fraction))
            / (1.0 - float(self.spec.non_thumb_delay_fraction)),
            min=0.0,
            max=1.0,
        )
        thumb = torch.clamp(
            progress * float(self.spec.thumb_lead_gain), min=0.0, max=1.0
        )
        hand_progress = non_thumb[:, None].expand(-1, 10).clone()
        hand_progress[:, 0] = thumb
        hand_progress[:, 1] = thumb
        hand_progress[:, 9] = thumb
        open_hand = self._reference_action[:, 9:19]
        scheduled_hand = open_hand + hand_progress * (
            self._final_close_hand - open_hand
        )
        proposal[scheduled, :3] = self._pregrasp_target[scheduled]
        proposal[scheduled, 9:19] = scheduled_hand[scheduled]
        tentative = close & self._tentative_mask
        proposal[tentative, :3] = self._tentative_eef[tentative]
        proposal[tentative, 9:19] = self._tentative_hand[tentative]

        confirm = active_mask & (
            self._phase == int(GraspLiftAuthoringPhase.CONFIRM_HOLD)
        )
        proposal[confirm, :3] = self._frozen_eef[confirm]
        proposal[confirm, 9:19] = self._frozen_hand[confirm]

        lift = active_mask & (
            (self._phase == int(GraspLiftAuthoringPhase.LIFT))
            | (self._phase == int(GraspLiftAuthoringPhase.SUCCESS_HOLD))
        )
        lift_distance = torch.clamp(
            (self._lift_steps + 1).to(dtype=current_state.dtype)
            * float(self.spec.lift_step_m),
            max=float(self.spec.lift_target_m),
        )
        desired_lift = self._lift_start_eef + lift_distance[:, None] * self._lift_axis
        proposal[lift, :3] = desired_lift[lift]
        proposal[lift, 9:19] = self._frozen_hand[lift]

        self._pending_proposal = proposal.detach().clone()
        self._pending_active = active_mask.detach().clone()
        self._pending_phase = self._phase.detach().clone()
        self._pending_tentative = self._tentative_mask.detach().clone()
        return proposal

    def observe(
        self,
        *,
        executed_action: torch.Tensor,
        next_state: torch.Tensor,
        oracle_result: GraspLiftOracleResult,
        events: GraspLiftTransientEvents,
        is_obj_static: torch.Tensor,
        is_robot_static: torch.Tensor,
    ) -> GraspLiftAuthoringSnapshot:
        """Consume post-step evidence and advance the authoring state machine."""
        self._require_reset()
        if self._pending_proposal is None:
            raise RuntimeError("propose() must create an action before observe()")
        self._validate_runtime_state(next_state)
        batch_size = self._batch_size
        _check_batched_tensor(
            executed_action,
            name="executed_action",
            shape=(batch_size, 19),
            device=next_state.device,
            dtype=next_state.dtype,
        )
        if not torch.isfinite(executed_action).all():
            raise ValueError("executed_action must be finite")
        _check_batched_tensor(
            is_obj_static,
            name="is_obj_static",
            shape=(batch_size,),
            device=next_state.device,
            dtype=torch.bool,
        )
        _check_batched_tensor(
            is_robot_static,
            name="is_robot_static",
            shape=(batch_size,),
            device=next_state.device,
            dtype=torch.bool,
        )
        events.validate(count=batch_size, device=next_state.device)
        self._validate_oracle_result(
            oracle_result,
            device=next_state.device,
            floating_dtype=next_state.dtype,
        )
        if not torch.equal(
            executed_action[:, 3:9], self._pending_proposal[:, 3:9]
        ):
            raise ValueError("executed action changed the held rotation contract")

        active = self._pending_active
        phase_before = self._pending_phase
        used_tentative = self._pending_tentative
        succeeded = active & oracle_result.success
        failed = active & oracle_result.failure
        self._phase[succeeded] = int(GraspLiftAuthoringPhase.SUCCEEDED)
        self._phase[failed] = int(GraspLiftAuthoringPhase.ORACLE_FAILED)
        continuing = active & ~succeeded & ~failed

        full_opposed = (
            continuing
            & oracle_result.current_opposed_grasp
            & (
                events.opposed_grasp_max_consecutive_physics_frames
                >= self.spec.full_opposed_physics_frames
            )
            & ~events.forbidden_hand_contact
            & ~events.collision_buffer_overflow
        )
        schema = DEFAULT_STAGE0_STATE_SCHEMA
        next_eef = next_state[:, schema.slice("eef_position")]

        approach = continuing & (
            phase_before == int(GraspLiftAuthoringPhase.APPROACH)
        )
        self._approach_steps[approach] += 1
        approach_done = approach & (
            self._approach_steps >= self.spec.approach_control_steps
        )
        self._phase[approach_done] = int(GraspLiftAuthoringPhase.CLOSE)

        close = continuing & (phase_before == int(GraspLiftAuthoringPhase.CLOSE))
        self._close_elapsed_steps[close] += 1
        scheduled = close & ~used_tentative
        self._close_steps[scheduled] += 1
        new_tentative = (
            scheduled
            & full_opposed
            & (
                self._close_steps >= self.spec.minimum_tentative_close_step
            )
        )
        self._tentative_mask[new_tentative] = True
        self._tentative_eef[new_tentative] = next_eef[new_tentative]
        self._tentative_hand[new_tentative] = executed_action[
            new_tentative, 9:19
        ]
        self._stable_steps[new_tentative] = 1

        verifying = close & used_tentative
        lost_tentative = verifying & ~full_opposed
        self._tentative_mask[lost_tentative] = False
        self._stable_steps[lost_tentative] = 0
        same_tentative_target = (
            torch.all(executed_action[:, :3] == self._tentative_eef, dim=-1)
            & torch.all(
                executed_action[:, 9:19] == self._tentative_hand, dim=-1
            )
        )
        drifted_tentative = verifying & full_opposed & ~same_tentative_target
        self._tentative_eef[drifted_tentative] = next_eef[drifted_tentative]
        self._tentative_hand[drifted_tentative] = executed_action[
            drifted_tentative, 9:19
        ]
        self._stable_steps[drifted_tentative] = 1
        kept_tentative = verifying & full_opposed & same_tentative_target
        self._stable_steps[kept_tentative] += 1
        verified = (
            kept_tentative
            & (
                self._stable_steps >= self.spec.stable_grasp_control_steps
            )
            & is_obj_static
            & is_robot_static
        )
        self._frozen_mask[verified] = True
        self._frozen_eef[verified] = self._tentative_eef[verified]
        self._frozen_hand[verified] = self._tentative_hand[verified]
        self._phase[verified] = int(GraspLiftAuthoringPhase.CONFIRM_HOLD)
        self._confirm_steps[verified] = 0

        close_exhausted = (
            close
            & ~verified
            & (
                self._close_elapsed_steps
                >= self.spec.maximum_close_control_steps
            )
        )
        self._phase[close_exhausted] = int(GraspLiftAuthoringPhase.EXHAUSTED)

        confirm = continuing & (
            phase_before == int(GraspLiftAuthoringPhase.CONFIRM_HOLD)
        )
        lost_frozen = confirm & ~full_opposed
        can_retry = lost_frozen & (
            self._close_elapsed_steps < self.spec.maximum_close_control_steps
        )
        cannot_retry = lost_frozen & ~can_retry
        self._phase[can_retry] = int(GraspLiftAuthoringPhase.CLOSE)
        self._phase[cannot_retry] = int(GraspLiftAuthoringPhase.REJECTED)
        self._tentative_mask[lost_frozen] = False
        self._frozen_mask[lost_frozen] = False
        self._stable_steps[lost_frozen] = 0
        held = confirm & full_opposed
        self._confirm_steps[held] += 1
        ready_to_lift = held & (
            self._confirm_steps >= self.spec.confirm_hold_control_steps
        )
        self._phase[ready_to_lift] = int(GraspLiftAuthoringPhase.LIFT)
        self._lift_start_eef[ready_to_lift] = next_eef[ready_to_lift]
        self._lift_steps[ready_to_lift] = 0
        self._success_hold_steps[ready_to_lift] = 0

        lift = continuing & (phase_before == int(GraspLiftAuthoringPhase.LIFT))
        unstable_lift = lift & ~full_opposed
        self._phase[unstable_lift] = int(GraspLiftAuthoringPhase.REJECTED)
        stable_lift = lift & full_opposed
        self._lift_steps[stable_lift] += 1
        lift_motion_done = stable_lift & (
            self._lift_steps >= self.spec.maximum_lift_control_steps
        )
        self._phase[lift_motion_done] = int(
            GraspLiftAuthoringPhase.SUCCESS_HOLD
        )

        success_hold = continuing & (
            phase_before == int(GraspLiftAuthoringPhase.SUCCESS_HOLD)
        )
        unstable_success_hold = success_hold & ~full_opposed
        self._phase[unstable_success_hold] = int(
            GraspLiftAuthoringPhase.REJECTED
        )
        stable_success_hold = success_hold & full_opposed
        self._success_hold_steps[stable_success_hold] += 1
        success_hold_exhausted = stable_success_hold & (
            self._success_hold_steps
            >= self.spec.maximum_success_hold_control_steps
        )
        self._phase[success_hold_exhausted] = int(
            GraspLiftAuthoringPhase.EXHAUSTED
        )

        self._clear_pending()
        return self.snapshot()

    def _validate_runtime_state(self, state: torch.Tensor) -> None:
        self._require_reset()
        _check_state(state, name="state")
        if (
            state.shape[0] != self._batch_size
            or state.device != self._reference_action.device
            or state.dtype != self._reference_action.dtype
        ):
            raise ValueError("state must match the reset batch, device, and dtype")

    def _validate_oracle_result(
        self,
        result: GraspLiftOracleResult,
        *,
        device: torch.device,
        floating_dtype: torch.dtype,
    ) -> None:
        if not isinstance(result, GraspLiftOracleResult):
            raise TypeError("oracle_result must be GraspLiftOracleResult")
        if bool(torch.logical_and(result.success, result.failure).any()):
            raise ValueError("oracle success and failure must be mutually exclusive")
        for name, value in result.__dict__.items():
            expected_dtype = (
                torch.int64
                if name in {"success_hold_steps", "drop_gap_steps"}
                else torch.bool
                if name
                in {
                    "success",
                    "failure",
                    "grasp_confirmed",
                    "current_opposed_grasp",
                    "forbidden_contact_violation",
                    "lateral_displacement_violation",
                    "bottle_tilt_violation",
                    "dropped_grasp_violation",
                    "collision_buffer_overflow_violation",
                }
                else floating_dtype
            )
            _check_batched_tensor(
                value,
                name=f"oracle_result.{name}",
                shape=(self._batch_size,),
                device=device,
                dtype=expected_dtype,
            )
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"oracle_result.{name} must be finite")

    def _require_reset(self) -> None:
        if self._reference_action is None:
            raise RuntimeError("GraspLiftAuthoringController.reset() must be called first")

    def _clear_pending(self) -> None:
        self._pending_proposal = None
        self._pending_active = None
        self._pending_phase = None
        self._pending_tentative = None


__all__ = [
    "DEFAULT_GRASP_LIFT_AUTHORING_MODE",
    "GRASP_LIFT_ACTION_CONTRACT_ID",
    "GRASP_LIFT_AUTHORING_CONTROLLER_ID",
    "GraspLiftAuthoringController",
    "GraspLiftAuthoringModeSpec",
    "GraspLiftAuthoringPhase",
    "GraspLiftAuthoringSnapshot",
]
