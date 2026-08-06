"""State-derived success and failure oracles for Stage 0."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rexpolicy.stage0.envs.state_schema import (
    DEFAULT_STAGE0_STATE_SCHEMA,
    Stage0StateSchema,
)


@dataclass(frozen=True)
class ReachOracleResult:
    success: torch.Tensor
    failure: torch.Tensor
    distance: torch.Tensor
    object_displacement: torch.Tensor
    contact_violation: torch.Tensor
    displacement_violation: torch.Tensor


class ReachSuccessOracle:
    """Binary Reach oracle independent of reward shaping and environment flags."""

    def __init__(
        self,
        num_envs: int,
        *,
        success_threshold_m: float = 0.04,
        success_hold_steps: int = 2,
        displacement_limit_m: float = 0.01,
        schema: Stage0StateSchema = DEFAULT_STAGE0_STATE_SCHEMA,
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if success_threshold_m <= 0.0 or displacement_limit_m <= 0.0:
            raise ValueError("Reach oracle thresholds must be positive")
        if success_hold_steps < 1:
            raise ValueError("success_hold_steps must be positive")
        self.num_envs = num_envs
        self.success_threshold_m = float(success_threshold_m)
        self.success_hold_steps = int(success_hold_steps)
        self.displacement_limit_m = float(displacement_limit_m)
        self.schema = schema
        self._initial_object_position: torch.Tensor | None = None
        self._hold_count: torch.Tensor | None = None

    def reset(
        self, state: torch.Tensor, world_mask: torch.Tensor | None = None
    ) -> None:
        self._validate_state(state)
        object_position = state[:, self.schema.slice("object_position")]
        if self._initial_object_position is None:
            self._initial_object_position = object_position.clone()
            self._hold_count = torch.zeros(
                self.num_envs, dtype=torch.int64, device=state.device
            )
            return
        if world_mask is None:
            self._initial_object_position.copy_(object_position)
            self._hold_count.zero_()
            return
        if world_mask.shape != (self.num_envs,) or world_mask.dtype != torch.bool:
            raise ValueError(f"world_mask must be bool with shape ({self.num_envs},)")
        self._initial_object_position[world_mask] = object_position[world_mask]
        self._hold_count[world_mask] = 0

    def evaluate(self, state: torch.Tensor) -> ReachOracleResult:
        self._validate_state(state)
        if self._initial_object_position is None or self._hold_count is None:
            raise RuntimeError(
                "ReachSuccessOracle.reset() must be called before evaluate()"
            )
        eef_position = state[:, self.schema.slice("eef_position")]
        goal_position = state[:, self.schema.slice("goal_position")]
        object_position = state[:, self.schema.slice("object_position")]
        contact = state[:, self.schema.slice("has_hand_contact")].squeeze(-1) > 0.5
        distance = torch.linalg.vector_norm(goal_position - eef_position, dim=-1)
        displacement = torch.linalg.vector_norm(
            object_position - self._initial_object_position, dim=-1
        )
        displacement_violation = displacement > self.displacement_limit_m
        failure = contact | displacement_violation
        within_goal = (distance <= self.success_threshold_m) & ~failure
        self._hold_count.copy_(
            torch.where(
                within_goal, self._hold_count + 1, torch.zeros_like(self._hold_count)
            )
        )
        success = (self._hold_count >= self.success_hold_steps) & ~failure
        return ReachOracleResult(
            success=success,
            failure=failure,
            distance=distance,
            object_displacement=displacement,
            contact_violation=contact,
            displacement_violation=displacement_violation,
        )

    def _validate_state(self, state: torch.Tensor) -> None:
        if state.shape != (self.num_envs, self.schema.dimension):
            raise ValueError(
                f"state must have shape ({self.num_envs}, {self.schema.dimension}), got {tuple(state.shape)}"
            )
        if not torch.isfinite(state).all():
            raise ValueError("state contains non-finite values")


__all__ = ["ReachOracleResult", "ReachSuccessOracle"]
