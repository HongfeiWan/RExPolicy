"""Fixed, paired learned-policy rollouts for Stage 0 Grasp-Lift.

The validation runner owns Newton construction, checkpoint IO, and the
one-shot validation claim.  This module keeps the actual policy-control and
oracle boundary small enough to test on CPU without weakening the formal
CUDA/Newton gate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from rexpolicy.stage0.envs.grasp_lift_action import (
    DEFAULT_GRASP_LIFT_ACTION_SCHEMA,
    GRASP_LIFT_ACTION_DIM,
    GRASP_LIFT_EFFECTIVE_ACTION_MASK,
    grasp_lift_model_to_physical_action,
)
from rexpolicy.stage0.envs.grasp_lift_oracle import (
    GRASP_LIFT_ORACLE_ID,
    GraspLiftOracleResult,
    GraspLiftSuccessOracle,
    GraspLiftTransientEvents,
    grasp_lift_transient_events_from_info,
)
from rexpolicy.stage0.envs.grasp_lift_state_view import (
    DEFAULT_GRASP_LIFT_STATE_VIEW,
    apply_grasp_lift_state_view,
)
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA
from rexpolicy.stage0.normalization import Stage0Normalization
from rexpolicy.stage0.types import Stage0Outcome, canonical_fingerprint


GRASP_LIFT_POLICY_NOISE_NAMESPACE = (
    "rexpolicy/stage0-grasp-lift-policy-noise/20260807-v1"
)
GRASP_LIFT_ROLLOUT_BUDGET_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-rollout-budget/v1"
)
GRASP_LIFT_ROLLOUT_RESULT_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-rollout-result/v1"
)
GRASP_LIFT_FORMAL_NOISE_VARIANTS = 4
GRASP_LIFT_FORMAL_FLOW_SAMPLE_STEPS = 16
GRASP_LIFT_ROLLOUT_PROTOCOL_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-rollout-protocol/v1"
)


def _positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _policy_noise_seed(
    reset_group_id: str,
    reset_seed: int,
    noise_variant: int,
) -> int:
    digest = canonical_fingerprint(
        {
            "namespace": GRASP_LIFT_POLICY_NOISE_NAMESPACE,
            "noise_variant": noise_variant,
            "reset_group_id": reset_group_id,
            "reset_seed": reset_seed,
        }
    )
    return int(digest[:16], 16) % ((1 << 63) - 1)


@dataclass(frozen=True)
class GraspLiftRolloutBudget:
    """Reset-major 4-noise grid paired across every model checkpoint."""

    reset_group_ids: tuple[str, ...]
    reset_seeds: tuple[int, ...]
    noise_variants: tuple[int, ...]
    noise_seeds: tuple[int, ...]
    max_steps_per_episode: int

    def __post_init__(self) -> None:
        values = (
            self.reset_group_ids,
            self.reset_seeds,
            self.noise_variants,
            self.noise_seeds,
        )
        if any(not isinstance(value, tuple) for value in values):
            raise TypeError("rollout budget vectors must be tuples")
        if not self.reset_group_ids or len({len(value) for value in values}) != 1:
            raise ValueError("rollout budget vectors must be non-empty and aligned")
        _positive_int(
            self.max_steps_per_episode,
            name="max_steps_per_episode",
        )
        cells: set[tuple[str, int, int]] = set()
        groups: dict[tuple[str, int], set[int]] = {}
        for index, (group_id, reset_seed, variant, noise_seed) in enumerate(
            zip(*values, strict=True)
        ):
            if not isinstance(group_id, str) or not group_id.strip():
                raise ValueError(
                    f"reset_group_ids[{index}] must be non-empty text"
                )
            _non_negative_int(reset_seed, name=f"reset_seeds[{index}]")
            _non_negative_int(variant, name=f"noise_variants[{index}]")
            _non_negative_int(noise_seed, name=f"noise_seeds[{index}]")
            if variant >= GRASP_LIFT_FORMAL_NOISE_VARIANTS:
                raise ValueError("noise variant is outside the formal 4-noise grid")
            if noise_seed != _policy_noise_seed(group_id, reset_seed, variant):
                raise ValueError("policy noise seed is not deterministically derived")
            cell = (group_id, reset_seed, variant)
            if cell in cells:
                raise ValueError("rollout budget contains a duplicate cell")
            cells.add(cell)
            groups.setdefault((group_id, reset_seed), set()).add(variant)
        expected_variants = set(range(GRASP_LIFT_FORMAL_NOISE_VARIANTS))
        if any(variants != expected_variants for variants in groups.values()):
            raise ValueError("every reset must contain all four noise variants")
        group_ids = tuple(group_id for group_id, _ in groups)
        reset_seeds = tuple(reset_seed for _, reset_seed in groups)
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("reset group IDs must be unique")
        if len(set(reset_seeds)) != len(reset_seeds):
            raise ValueError("reset seeds must be unique")
        expected_order = tuple(
            (group_id, reset_seed, variant)
            for group_id, reset_seed in groups
            for variant in range(GRASP_LIFT_FORMAL_NOISE_VARIANTS)
        )
        if tuple(zip(self.reset_group_ids, self.reset_seeds, self.noise_variants)) != (
            expected_order
        ):
            raise ValueError("rollout cells must use reset-major, variant-minor order")

    @classmethod
    def from_reset_groups(
        cls,
        reset_groups: Sequence[tuple[str, int]],
        *,
        max_steps_per_episode: int = 72,
    ) -> GraspLiftRolloutBudget:
        if isinstance(reset_groups, (str, bytes)) or not isinstance(
            reset_groups, Sequence
        ):
            raise TypeError("reset_groups must be a sequence")
        groups = tuple(reset_groups)
        if not groups:
            raise ValueError("reset_groups must be non-empty")
        if len(set(groups)) != len(groups):
            raise ValueError("reset_groups must be unique")
        rows = tuple(
            (group_id, reset_seed, variant)
            for group_id, reset_seed in groups
            for variant in range(GRASP_LIFT_FORMAL_NOISE_VARIANTS)
        )
        return cls(
            reset_group_ids=tuple(row[0] for row in rows),
            reset_seeds=tuple(row[1] for row in rows),
            noise_variants=tuple(row[2] for row in rows),
            noise_seeds=tuple(_policy_noise_seed(*row) for row in rows),
            max_steps_per_episode=max_steps_per_episode,
        )

    @property
    def episode_count(self) -> int:
        return len(self.reset_group_ids)

    @property
    def reset_count(self) -> int:
        return self.episode_count // GRASP_LIFT_FORMAL_NOISE_VARIANTS

    @property
    def allocated_world_steps(self) -> int:
        return self.episode_count * self.max_steps_per_episode

    def to_record(self) -> dict[str, Any]:
        return {
            "allocated_world_steps": self.allocated_world_steps,
            "cells": [
                {
                    "noise_seed": noise_seed,
                    "noise_variant": variant,
                    "reset_group_id": group_id,
                    "reset_seed": reset_seed,
                }
                for group_id, reset_seed, variant, noise_seed in zip(
                    self.reset_group_ids,
                    self.reset_seeds,
                    self.noise_variants,
                    self.noise_seeds,
                    strict=True,
                )
            ],
            "max_steps_per_episode": self.max_steps_per_episode,
            "noise_namespace": GRASP_LIFT_POLICY_NOISE_NAMESPACE,
            "noise_variants_per_reset": GRASP_LIFT_FORMAL_NOISE_VARIANTS,
            "reset_count": self.reset_count,
            "schema_id": GRASP_LIFT_ROLLOUT_BUDGET_SCHEMA_ID,
        }

    @property
    def sha256(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class GraspLiftRolloutProtocol:
    """Every semantic choice that can change one rollout result."""

    action_horizon: int
    flow_sample_steps: int
    normalization_sha256: str
    oracle_thresholds: Mapping[str, Any]

    def __post_init__(self) -> None:
        _positive_int(self.action_horizon, name="action_horizon")
        if self.flow_sample_steps != GRASP_LIFT_FORMAL_FLOW_SAMPLE_STEPS:
            raise ValueError("formal Grasp-Lift flow sample steps changed")
        if (
            not isinstance(self.normalization_sha256, str)
            or len(self.normalization_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.normalization_sha256
            )
        ):
            raise ValueError("normalization_sha256 must be a SHA-256 digest")
        expected_keys = {
            "bottle_tilt_limit_rad",
            "drop_confirm_control_steps",
            "grasp_confirm_physics_frames",
            "lateral_displacement_limit_m",
            "lift_height_m",
            "oracle_id",
            "state_schema_sha256",
            "success_hold_control_steps",
        }
        if not isinstance(self.oracle_thresholds, Mapping) or set(
            self.oracle_thresholds
        ) != expected_keys:
            raise ValueError("oracle threshold record changed")
        record = dict(self.oracle_thresholds)
        if (
            record["oracle_id"] != GRASP_LIFT_ORACLE_ID
            or record["state_schema_sha256"]
            != DEFAULT_STAGE0_STATE_SCHEMA.sha256
        ):
            raise ValueError("oracle identity or state schema changed")
        for name in (
            "bottle_tilt_limit_rad",
            "lateral_displacement_limit_m",
            "lift_height_m",
        ):
            value = record[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"oracle {name} must be finite and positive")
        for name in (
            "drop_confirm_control_steps",
            "grasp_confirm_physics_frames",
            "success_hold_control_steps",
        ):
            _positive_int(record[name], name=f"oracle {name}")
        object.__setattr__(self, "oracle_thresholds", MappingProxyType(record))

    def to_record(self) -> dict[str, Any]:
        return {
            "action_horizon": self.action_horizon,
            "action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
            "control": "first_action_receding_horizon/v1",
            "flow_sample_steps": self.flow_sample_steps,
            "noise_namespace": GRASP_LIFT_POLICY_NOISE_NAMESPACE,
            "normalization_sha256": self.normalization_sha256,
            "oracle": dict(self.oracle_thresholds),
            "policy_conditioning": "success_latent_none/v1",
            "schema_id": GRASP_LIFT_ROLLOUT_PROTOCOL_SCHEMA_ID,
            "state_schema_sha256": DEFAULT_STAGE0_STATE_SCHEMA.sha256,
            "state_view_sha256": DEFAULT_GRASP_LIFT_STATE_VIEW.sha256,
        }

    @property
    def sha256(self) -> str:
        return canonical_fingerprint(self.to_record())


def grasp_lift_rollout_protocol(
    normalization: Stage0Normalization,
    oracle: GraspLiftSuccessOracle,
    *,
    action_horizon: int,
    flow_sample_steps: int = GRASP_LIFT_FORMAL_FLOW_SAMPLE_STEPS,
) -> GraspLiftRolloutProtocol:
    """Bind and verify the exact formal oracle and policy-control semantics."""

    if not isinstance(normalization, Stage0Normalization):
        raise TypeError("normalization must be Stage0Normalization")
    if not isinstance(oracle, GraspLiftSuccessOracle):
        raise TypeError("oracle must be GraspLiftSuccessOracle")
    thresholds = {
        "bottle_tilt_limit_rad": oracle.bottle_tilt_limit_rad,
        "drop_confirm_control_steps": oracle.drop_confirm_control_steps,
        "grasp_confirm_physics_frames": oracle.grasp_confirm_physics_frames,
        "lateral_displacement_limit_m": oracle.lateral_displacement_limit_m,
        "lift_height_m": oracle.lift_height_m,
        "oracle_id": GRASP_LIFT_ORACLE_ID,
        "state_schema_sha256": oracle.schema.sha256,
        "success_hold_control_steps": oracle.success_hold_control_steps,
    }
    from rexpolicy.stage0.grasp_lift_pilot import grasp_lift_pilot_oracle_record

    if thresholds != grasp_lift_pilot_oracle_record():
        raise ValueError("formal Grasp-Lift oracle thresholds changed")
    return GraspLiftRolloutProtocol(
        action_horizon=action_horizon,
        flow_sample_steps=flow_sample_steps,
        normalization_sha256=normalization.sha256,
        oracle_thresholds=thresholds,
    )


def _cpu_vector(
    value: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    count: int,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype is not dtype
        or tuple(value.shape) != (count,)
    ):
        raise ValueError(f"{name} must be CPU {dtype} [{count}]")
    return value.detach().contiguous()


@dataclass(frozen=True)
class GraspLiftRolloutResult:
    """Per-cell oracle outcomes and zero-tolerance safety/integrity witnesses."""

    budget: GraspLiftRolloutBudget
    protocol: GraspLiftRolloutProtocol
    success: torch.Tensor
    failure: torch.Tensor
    timeout: torch.Tensor
    completed_steps: torch.Tensor
    forbidden_contact_violation: torch.Tensor
    lateral_displacement_violation: torch.Tensor
    bottle_tilt_violation: torch.Tensor
    dropped_grasp_violation: torch.Tensor
    collision_buffer_overflow_violation: torch.Tensor
    oracle_input_disagreement: torch.Tensor
    execution_integrity_violation: torch.Tensor
    final_lift_height_m: torch.Tensor
    final_lateral_displacement_m: torch.Tensor
    final_bottle_tilt_rad: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.budget, GraspLiftRolloutBudget):
            raise TypeError("budget must be GraspLiftRolloutBudget")
        if not isinstance(self.protocol, GraspLiftRolloutProtocol):
            raise TypeError("protocol must be GraspLiftRolloutProtocol")
        count = self.budget.episode_count
        bool_names = (
            "success",
            "failure",
            "timeout",
            "forbidden_contact_violation",
            "lateral_displacement_violation",
            "bottle_tilt_violation",
            "dropped_grasp_violation",
            "collision_buffer_overflow_violation",
            "oracle_input_disagreement",
            "execution_integrity_violation",
        )
        for name in bool_names:
            object.__setattr__(
                self,
                name,
                _cpu_vector(
                    getattr(self, name),
                    name=name,
                    dtype=torch.bool,
                    count=count,
                ),
            )
        object.__setattr__(
            self,
            "completed_steps",
            _cpu_vector(
                self.completed_steps,
                name="completed_steps",
                dtype=torch.int64,
                count=count,
            ),
        )
        for name in (
            "final_lift_height_m",
            "final_lateral_displacement_m",
            "final_bottle_tilt_rad",
        ):
            value = _cpu_vector(
                getattr(self, name),
                name=name,
                dtype=torch.float32,
                count=count,
            )
            if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        terminals = (
            self.success.to(torch.int8)
            + self.failure.to(torch.int8)
            + self.timeout.to(torch.int8)
        )
        if not torch.equal(terminals, torch.ones_like(terminals)):
            raise ValueError("success/failure/timeout must partition every cell")
        if bool((self.completed_steps < 1).any()) or bool(
            (self.completed_steps > self.budget.max_steps_per_episode).any()
        ):
            raise ValueError("completed_steps is outside the rollout budget")
        safety = self.safety_violation
        if bool((self.success & (safety | self.any_integrity_fault)).any()):
            raise ValueError("a successful cell cannot carry a safety/integrity fault")

    @property
    def safety_violation(self) -> torch.Tensor:
        return (
            self.forbidden_contact_violation
            | self.lateral_displacement_violation
            | self.bottle_tilt_violation
            | self.dropped_grasp_violation
            | self.collision_buffer_overflow_violation
        )

    @property
    def any_integrity_fault(self) -> torch.Tensor:
        return self.oracle_input_disagreement | self.execution_integrity_violation

    @property
    def success_count(self) -> int:
        return int(self.success.sum())

    @property
    def success_rate(self) -> float:
        return self.success_count / self.budget.episode_count

    @property
    def outcomes(self) -> tuple[Stage0Outcome, ...]:
        values: list[Stage0Outcome] = []
        for index in range(self.budget.episode_count):
            if bool(self.success[index]):
                values.append(Stage0Outcome.SUCCESS)
            elif bool(self.failure[index]):
                values.append(Stage0Outcome.FAILURE)
            else:
                values.append(Stage0Outcome.TIMEOUT)
        return tuple(values)

    def to_record(self) -> dict[str, Any]:
        cells = []
        for index in range(self.budget.episode_count):
            cells.append(
                {
                    "bottle_tilt_violation": bool(
                        self.bottle_tilt_violation[index]
                    ),
                    "collision_buffer_overflow_violation": bool(
                        self.collision_buffer_overflow_violation[index]
                    ),
                    "completed_steps": int(self.completed_steps[index]),
                    "dropped_grasp_violation": bool(
                        self.dropped_grasp_violation[index]
                    ),
                    "execution_integrity_violation": bool(
                        self.execution_integrity_violation[index]
                    ),
                    "failure": bool(self.failure[index]),
                    "final_bottle_tilt_rad": float(
                        self.final_bottle_tilt_rad[index]
                    ),
                    "final_lateral_displacement_m": float(
                        self.final_lateral_displacement_m[index]
                    ),
                    "final_lift_height_m": float(self.final_lift_height_m[index]),
                    "forbidden_contact_violation": bool(
                        self.forbidden_contact_violation[index]
                    ),
                    "lateral_displacement_violation": bool(
                        self.lateral_displacement_violation[index]
                    ),
                    "noise_variant": self.budget.noise_variants[index],
                    "oracle_input_disagreement": bool(
                        self.oracle_input_disagreement[index]
                    ),
                    "outcome": self.outcomes[index].value,
                    "reset_group_id": self.budget.reset_group_ids[index],
                    "reset_seed": self.budget.reset_seeds[index],
                    "success": bool(self.success[index]),
                    "timeout": bool(self.timeout[index]),
                }
            )
        return {
            "budget_sha256": self.budget.sha256,
            "cells": cells,
            "executed_world_steps": int(self.completed_steps.sum()),
            "any_integrity_fault_count": int(self.any_integrity_fault.sum()),
            "execution_integrity_violation_count": int(
                self.execution_integrity_violation.sum()
            ),
            "oracle_input_disagreement_count": int(
                self.oracle_input_disagreement.sum()
            ),
            "protocol": self.protocol.to_record(),
            "protocol_sha256": self.protocol.sha256,
            "safety_violation_count": int(self.safety_violation.sum()),
            "schema_id": GRASP_LIFT_ROLLOUT_RESULT_SCHEMA_ID,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
        }

    @property
    def sha256(self) -> str:
        return canonical_fingerprint(self.to_record())


def _module_device_dtype(
    module: nn.Module,
    reference: torch.Tensor,
) -> tuple[torch.device, torch.dtype]:
    tensors = tuple(module.parameters()) + tuple(module.buffers())
    if not tensors:
        return reference.device, reference.dtype
    device = tensors[0].device
    dtype = next(
        (value.dtype for value in tensors if torch.is_floating_point(value)),
        reference.dtype,
    )
    if any(value.device != device for value in tensors):
        raise ValueError("policy parameters and buffers must share one device")
    return device, dtype


def _policy_contract(policy: nn.Module) -> tuple[int, int, int]:
    values = []
    for name in ("state_dim", "action_dim", "action_horizon"):
        value = getattr(policy, name, None)
        if type(value) is not int or value < 1:
            raise TypeError(f"policy must expose a positive integer {name}")
        values.append(value)
    if not callable(getattr(policy, "sample", None)):
        raise TypeError("policy must provide sample()")
    return tuple(values)  # type: ignore[return-value]


def _validate_state(state: Any, *, count: int) -> torch.Tensor:
    if (
        not isinstance(state, torch.Tensor)
        or tuple(state.shape)
        != (count, DEFAULT_STAGE0_STATE_SCHEMA.dimension)
        or not state.is_floating_point()
        or not bool(torch.isfinite(state).all())
    ):
        raise ValueError(
            "environment state must be finite floating "
            f"[{count}, {DEFAULT_STAGE0_STATE_SCHEMA.dimension}]"
        )
    return state


def _candidate_method(env: Any, name: str) -> Any:
    for candidate in (env, getattr(env, "env", None)):
        method = getattr(candidate, name, None)
        if callable(method):
            return method
    raise TypeError(f"environment must expose {name}()")


def _step_noise_seed(base_seed: int, control_step: int) -> int:
    digest = canonical_fingerprint(
        {
            "base_seed": base_seed,
            "control_step": control_step,
            "namespace": GRASP_LIFT_POLICY_NOISE_NAMESPACE,
        }
    )
    return int(digest[:16], 16) % ((1 << 63) - 1)


def _initial_noise(
    budget: GraspLiftRolloutBudget,
    *,
    control_step: int,
    horizon: int,
    action_dim: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    rows = []
    for seed in budget.noise_seeds:
        generator = torch.Generator(device=reference.device)
        generator.manual_seed(_step_noise_seed(seed, control_step))
        rows.append(
            torch.randn(
                (horizon, action_dim),
                dtype=reference.dtype,
                device=reference.device,
                generator=generator,
            )
        )
    return torch.stack(rows)


def _info_bool_vector(
    info: Mapping[str, Any],
    name: str,
    *,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    value = info.get(name)
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype is not torch.bool
        or value.device != device
        or tuple(value.shape) != (count,)
    ):
        raise ValueError(f"transition info {name!r} must be device bool [{count}]")
    return value


def _oracle_input_disagreement(
    next_state: torch.Tensor,
    info: Mapping[str, Any],
) -> torch.Tensor:
    count = next_state.shape[0]
    state_grasped = (
        next_state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("is_grasped")].squeeze(-1)
        > 0.5
    )
    state_contact = (
        next_state[
            :, DEFAULT_STAGE0_STATE_SCHEMA.slice("has_hand_contact")
        ].squeeze(-1)
        > 0.5
    )
    info_grasped = _info_bool_vector(
        info,
        "is_grasped",
        count=count,
        device=next_state.device,
    )
    info_contact = _info_bool_vector(
        info,
        "has_hand_contact",
        count=count,
        device=next_state.device,
    )
    return (state_grasped != info_grasped) | (state_contact != info_contact)


def _last_oracle_result(
    oracle: GraspLiftSuccessOracle,
    state: torch.Tensor,
) -> GraspLiftOracleResult:
    zero_bool = torch.zeros(
        state.shape[0], dtype=torch.bool, device=state.device
    )
    zero_int = torch.zeros(
        state.shape[0], dtype=torch.int64, device=state.device
    )
    return oracle.evaluate(
        state,
        GraspLiftTransientEvents(
            forbidden_hand_contact=zero_bool,
            opposed_grasp_max_consecutive_physics_frames=zero_int,
            had_hand_contact=(
                state[
                    :, DEFAULT_STAGE0_STATE_SCHEMA.slice("has_hand_contact")
                ].squeeze(-1)
                > 0.5
            ),
            collision_buffer_overflow=zero_bool.clone(),
        ),
    )


def rollout_grasp_lift_no_z_policy(
    env: Any,
    policy: nn.Module,
    normalization: Stage0Normalization,
    budget: GraspLiftRolloutBudget,
    *,
    oracle: GraspLiftSuccessOracle | None = None,
    flow_sample_steps: int = GRASP_LIFT_FORMAL_FLOW_SAMPLE_STEPS,
) -> GraspLiftRolloutResult:
    """Run a no-z policy with first-action receding-horizon control.

    XYZ remains a current-state-relative model delta, hand targets remain
    absolute, rotation is restored from the reset hold action, and Newton's
    projection is applied exactly once before the action is executed.
    """

    if not isinstance(policy, nn.Module):
        raise TypeError("policy must be a Torch module")
    if not isinstance(normalization, Stage0Normalization):
        raise TypeError("normalization must be Stage0Normalization")
    if not isinstance(budget, GraspLiftRolloutBudget):
        raise TypeError("budget must be GraspLiftRolloutBudget")
    flow_sample_steps = _positive_int(flow_sample_steps, name="flow_sample_steps")
    if flow_sample_steps != GRASP_LIFT_FORMAL_FLOW_SAMPLE_STEPS:
        raise ValueError("formal Grasp-Lift flow sample steps changed")
    count = budget.episode_count
    if getattr(env, "num_envs", None) != count:
        raise ValueError("environment world count must equal rollout cell count")
    state_dim, action_dim, action_horizon = _policy_contract(policy)
    if state_dim != normalization.state_dim or action_dim != normalization.action_dim:
        raise ValueError("policy and normalization dimensions differ")
    if state_dim != DEFAULT_STAGE0_STATE_SCHEMA.dimension:
        raise ValueError("Grasp-Lift rollout requires the canonical 79D state")
    if action_dim != GRASP_LIFT_ACTION_DIM:
        raise ValueError("Grasp-Lift rollout requires the 19D action contract")
    expected_mask = torch.tensor(GRASP_LIFT_EFFECTIVE_ACTION_MASK, dtype=torch.bool)
    if not torch.equal(normalization.effective_action_mask, expected_mask):
        raise ValueError("Grasp-Lift rollout requires the fixed 13D effective mask")

    reset_result = env.reset(seed=list(budget.reset_seeds))
    if not isinstance(reset_result, tuple) or len(reset_result) != 2:
        raise TypeError("environment reset must return (state, info)")
    state = _validate_state(reset_result[0], count=count)
    device, dtype = _module_device_dtype(policy, state)
    if state.device != device or state.dtype != dtype:
        raise ValueError("policy and environment state must share device and dtype")
    hold_action = _candidate_method(env, "hold_action_torch")().detach().clone()
    if (
        tuple(hold_action.shape) != (count, action_dim)
        or hold_action.device != device
        or hold_action.dtype != dtype
        or not bool(torch.isfinite(hold_action).all())
    ):
        raise ValueError(
            "reset hold action changed shape, device, dtype, or finiteness"
        )
    project = _candidate_method(env, "project_effective_action_torch")
    executed_action = _candidate_method(env, "effective_action_torch")

    if oracle is None:
        oracle = GraspLiftSuccessOracle(count)
    if not isinstance(oracle, GraspLiftSuccessOracle) or oracle.num_envs != count:
        raise ValueError("oracle must cover every rollout cell")
    oracle.reset(state)
    protocol = grasp_lift_rollout_protocol(
        normalization,
        oracle,
        action_horizon=action_horizon,
        flow_sample_steps=flow_sample_steps,
    )

    success = torch.zeros(count, dtype=torch.bool, device=device)
    failure = torch.zeros_like(success)
    timeout = torch.zeros_like(success)
    oracle_disagreement = torch.zeros_like(success)
    integrity = torch.zeros_like(success)
    completed_steps = torch.zeros(count, dtype=torch.int64, device=device)
    terminal_lift = torch.zeros(count, dtype=dtype, device=device)
    terminal_lateral = torch.zeros_like(terminal_lift)
    terminal_tilt = torch.zeros_like(terminal_lift)
    terminal_forbidden = torch.zeros_like(success)
    terminal_lateral_violation = torch.zeros_like(success)
    terminal_tilt_violation = torch.zeros_like(success)
    terminal_dropped = torch.zeros_like(success)
    terminal_overflow = torch.zeros_like(success)
    active = torch.ones_like(success)
    latest: GraspLiftOracleResult | None = None
    policy_training = policy.training
    policy.eval()
    try:
        with torch.inference_mode():
            feature_mask = normalization.effective_action_mask.to(device=device)
            for control_step in range(budget.max_steps_per_episode):
                if not bool(active.any()):
                    break
                active_before = active.clone()
                model_state = apply_grasp_lift_state_view(state)
                normalized_state = normalization.normalize_states(model_state)
                noise = _initial_noise(
                    budget,
                    control_step=control_step,
                    horizon=action_horizon,
                    action_dim=action_dim,
                    reference=normalized_state,
                )
                normalized_chunk = policy.sample(
                    normalized_state,
                    success_latent=None,
                    steps=flow_sample_steps,
                    action_feature_mask=feature_mask,
                    initial_noise=noise,
                )
                expected_shape = (count, action_horizon, action_dim)
                if (
                    not isinstance(normalized_chunk, torch.Tensor)
                    or tuple(normalized_chunk.shape) != expected_shape
                    or normalized_chunk.device != device
                    or normalized_chunk.dtype != dtype
                    or not bool(torch.isfinite(normalized_chunk).all())
                ):
                    raise ValueError(
                        "policy sample must be finite and preserve the formal "
                        "batch/horizon/action contract"
                    )
                model_action = normalization.denormalize_actions(
                    normalized_chunk[:, 0, :]
                )
                model_action[~active_before, :3] = 0.0
                model_action[~active_before, 9:19] = state[
                    ~active_before,
                    DEFAULT_STAGE0_STATE_SCHEMA.slice("hand_joint_position"),
                ]
                proposal = grasp_lift_model_to_physical_action(
                    state,
                    model_action,
                    hold_action,
                )
                projected = project(proposal)
                if (
                    not isinstance(projected, torch.Tensor)
                    or projected.shape != proposal.shape
                    or projected.device != device
                    or projected.dtype != dtype
                    or not bool(torch.isfinite(projected).all())
                ):
                    raise ValueError("projected action changed its tensor contract")
                transition = env.step(projected)
                next_state = _validate_state(
                    getattr(transition, "next_state", None),
                    count=count,
                )
                if next_state.device != device or next_state.dtype != dtype:
                    raise ValueError("next state changed device or dtype")
                info = getattr(transition, "info", None)
                if not isinstance(info, Mapping):
                    raise TypeError("transition info must be a mapping")
                reward = getattr(transition, "reward", None)
                reported_state = getattr(transition, "state", None)
                reported_action = getattr(transition, "action", None)
                terminated = getattr(transition, "terminated", None)
                truncated = getattr(transition, "truncated", None)
                for name, value in (
                    ("terminated", terminated),
                    ("truncated", truncated),
                ):
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.dtype is not torch.bool
                        or value.device != device
                        or tuple(value.shape) != (count,)
                    ):
                        raise ValueError(
                            f"transition {name} must be device bool [{count}]"
                        )
                if (
                    not isinstance(reward, torch.Tensor)
                    or reward.device != device
                    or tuple(reward.shape) != (count,)
                    or not reward.is_floating_point()
                    or not bool(torch.isfinite(reward).all())
                ):
                    raise ValueError(
                        "transition reward must be finite floating device "
                        f"[{count}]"
                    )
                integrity |= (reward != 0) & active_before
                for name, reported, expected in (
                    ("state", reported_state, state),
                    ("action", reported_action, projected),
                ):
                    if (
                        not isinstance(reported, torch.Tensor)
                        or reported.shape != expected.shape
                        or reported.device != device
                        or reported.dtype != dtype
                    ):
                        raise ValueError(
                            f"transition {name} changed its tensor contract"
                        )
                    mismatch = (reported != expected).reshape(count, -1).any(dim=1)
                    integrity |= mismatch & active_before
                if bool(terminated.any()):
                    integrity |= terminated & active_before
                if control_step + 1 < budget.max_steps_per_episode:
                    integrity |= truncated & active_before
                actual = executed_action().detach()
                if (
                    actual.shape != projected.shape
                    or actual.device != device
                    or actual.dtype != dtype
                ):
                    raise ValueError("executed action tensor contract changed")
                mismatched = (actual != projected).reshape(count, -1).any(dim=1)
                integrity |= mismatched & active_before
                events = grasp_lift_transient_events_from_info(
                    info,
                    count=count,
                    device=device,
                )
                latest = oracle.evaluate(next_state, events)
                terminal_lift = torch.where(
                    active_before,
                    latest.lift_height,
                    terminal_lift,
                )
                terminal_lateral = torch.where(
                    active_before,
                    latest.lateral_displacement,
                    terminal_lateral,
                )
                terminal_tilt = torch.where(
                    active_before,
                    latest.bottle_tilt,
                    terminal_tilt,
                )
                terminal_forbidden = torch.where(
                    active_before,
                    latest.forbidden_contact_violation,
                    terminal_forbidden,
                )
                terminal_lateral_violation = torch.where(
                    active_before,
                    latest.lateral_displacement_violation,
                    terminal_lateral_violation,
                )
                terminal_tilt_violation = torch.where(
                    active_before,
                    latest.bottle_tilt_violation,
                    terminal_tilt_violation,
                )
                terminal_dropped = torch.where(
                    active_before,
                    latest.dropped_grasp_violation,
                    terminal_dropped,
                )
                terminal_overflow = torch.where(
                    active_before,
                    latest.collision_buffer_overflow_violation,
                    terminal_overflow,
                )
                oracle_disagreement |= (
                    _oracle_input_disagreement(next_state, info) & active_before
                )
                completed_steps.add_(active_before.to(torch.int64))
                step_failure = (
                    latest.failure
                    | latest.collision_buffer_overflow_violation
                    | oracle_disagreement
                    | integrity
                ) & active_before
                step_success = latest.success & ~step_failure & active_before
                success |= step_success
                failure |= step_failure
                active &= ~(step_success | step_failure)
                state = next_state
            timeout |= active
    finally:
        policy.train(policy_training)

    if latest is None:
        latest = _last_oracle_result(oracle, state)
    return GraspLiftRolloutResult(
        budget=budget,
        protocol=protocol,
        success=success.detach().cpu(),
        failure=failure.detach().cpu(),
        timeout=timeout.detach().cpu(),
        completed_steps=completed_steps.detach().cpu(),
        forbidden_contact_violation=terminal_forbidden.detach().cpu(),
        lateral_displacement_violation=(
            terminal_lateral_violation.detach().cpu()
        ),
        bottle_tilt_violation=terminal_tilt_violation.detach().cpu(),
        dropped_grasp_violation=terminal_dropped.detach().cpu(),
        collision_buffer_overflow_violation=terminal_overflow.detach().cpu(),
        oracle_input_disagreement=oracle_disagreement.detach().cpu(),
        execution_integrity_violation=integrity.detach().cpu(),
        final_lift_height_m=terminal_lift.detach().float().cpu(),
        final_lateral_displacement_m=terminal_lateral.detach().float().cpu(),
        final_bottle_tilt_rad=terminal_tilt.detach().float().cpu(),
    )


__all__ = [
    "GRASP_LIFT_FORMAL_FLOW_SAMPLE_STEPS",
    "GRASP_LIFT_FORMAL_NOISE_VARIANTS",
    "GRASP_LIFT_POLICY_NOISE_NAMESPACE",
    "GRASP_LIFT_ROLLOUT_BUDGET_SCHEMA_ID",
    "GRASP_LIFT_ROLLOUT_PROTOCOL_SCHEMA_ID",
    "GRASP_LIFT_ROLLOUT_RESULT_SCHEMA_ID",
    "GraspLiftRolloutBudget",
    "GraspLiftRolloutProtocol",
    "GraspLiftRolloutResult",
    "grasp_lift_rollout_protocol",
    "rollout_grasp_lift_no_z_policy",
]
