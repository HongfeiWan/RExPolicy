"""GPU-resident state-only adapter for the Newton environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from rexpolicy.stage0.envs.state_schema import (
    DEFAULT_STAGE0_STATE_SCHEMA,
    Stage0StateSchema,
)


def _check_last_dim(tensor: torch.Tensor, width: int, name: str) -> None:
    if tensor.ndim != 2 or tensor.shape[1] != width:
        raise ValueError(
            f"{name} must have shape [B, {width}], got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")


def _normalize_quaternion_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    _check_last_dim(quaternion, 4, "quaternion")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if torch.any(norm <= torch.finfo(quaternion.dtype).eps):
        raise ValueError("quaternion contains a zero-norm row")
    normalized = quaternion / norm
    # A canonical sign makes any persisted quaternion diagnostics stable.  The
    # packed rotation-6D representation itself is already sign invariant.
    sign = torch.where(normalized[:, 3:4] < 0.0, -1.0, 1.0)
    return normalized * sign


def _quaternion_conjugate_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((-quaternion[:, :3], quaternion[:, 3:4]), dim=-1)


def _quaternion_multiply_xyzw(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lx, ly, lz, lw = left.unbind(dim=-1)
    rx, ry, rz, rw = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


def _quaternion_rotate_xyzw(
    quaternion: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    _check_last_dim(vector, 3, "vector")
    q_vector = quaternion[:, :3]
    twice_cross = 2.0 * torch.linalg.cross(q_vector, vector, dim=-1)
    return (
        vector
        + quaternion[:, 3:4] * twice_cross
        + torch.linalg.cross(q_vector, twice_cross, dim=-1)
    )


def _quaternion_to_rotation_6d_rows(quaternion: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quaternion.unbind(dim=-1)
    row0 = torch.stack(
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        dim=-1,
    )
    row1 = torch.stack(
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        dim=-1,
    )
    return torch.cat((row0, row1), dim=-1)


def canonicalize_stage0_components(
    components: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert raw Newton world-frame components into right-base-frame fields."""
    required = {
        "agent_qpos": 17,
        "agent_qvel": 17,
        "eef_9d_base": 9,
        "object_pose_world": 7,
        "object_twist_world": 6,
        "base_pose_world": 7,
        "base_twist_world": 6,
        "goal_position_world": 3,
        "finger_contact_counts": 5,
    }
    for name, width in required.items():
        if name not in components:
            raise KeyError(f"Missing Stage 0 state component {name!r}")
        _check_last_dim(components[name], width, name)

    qpos = components["agent_qpos"]
    batch_size = qpos.shape[0]
    device = qpos.device
    dtype = qpos.dtype
    for name in required:
        value = components[name]
        if value.shape[0] != batch_size or value.device != device:
            raise ValueError(f"{name} must share the Stage 0 batch and device")

    base_pose = components["base_pose_world"]
    object_pose = components["object_pose_world"]
    base_position = base_pose[:, :3]
    object_position_world = object_pose[:, :3]
    base_rotation_inverse = _quaternion_conjugate_xyzw(
        _normalize_quaternion_xyzw(base_pose[:, 3:])
    )
    object_rotation_world = _normalize_quaternion_xyzw(object_pose[:, 3:])

    object_position = _quaternion_rotate_xyzw(
        base_rotation_inverse,
        object_position_world - base_position,
    )
    object_rotation = _normalize_quaternion_xyzw(
        _quaternion_multiply_xyzw(base_rotation_inverse, object_rotation_world)
    )
    goal_position = _quaternion_rotate_xyzw(
        base_rotation_inverse,
        components["goal_position_world"] - base_position,
    )

    # Newton's existing task evaluator treats spatial_top as linear and
    # spatial_bottom as angular; keep that established contract here.
    object_linear_world = components["object_twist_world"][:, :3]
    object_angular_world = components["object_twist_world"][:, 3:]
    base_linear_world = components["base_twist_world"][:, :3]
    base_angular_world = components["base_twist_world"][:, 3:]
    relative_world_position = object_position_world - base_position
    object_linear_velocity = _quaternion_rotate_xyzw(
        base_rotation_inverse,
        object_linear_world
        - base_linear_world
        - torch.linalg.cross(base_angular_world, relative_world_position, dim=-1),
    )
    object_angular_velocity = _quaternion_rotate_xyzw(
        base_rotation_inverse,
        object_angular_world - base_angular_world,
    )

    eef_position = components["eef_9d_base"][:, :3]
    task_phase = components["task_phase"].to(device=device, dtype=torch.long)
    if task_phase.shape != (batch_size,):
        raise ValueError(
            f"task_phase must have shape [{batch_size}], got {tuple(task_phase.shape)}"
        )
    if torch.any((task_phase < 0) | (task_phase > 4)):
        raise ValueError("task_phase must be in [0, 4]")

    return {
        "arm_joint_position": qpos[:, :7],
        "arm_joint_velocity": components["agent_qvel"][:, :7],
        "hand_joint_position": qpos[:, 7:],
        "hand_joint_velocity": components["agent_qvel"][:, 7:],
        "eef_position": eef_position,
        "eef_rotation_6d": components["eef_9d_base"][:, 3:],
        "object_position": object_position,
        "object_rotation_6d": _quaternion_to_rotation_6d_rows(object_rotation),
        "object_linear_velocity": object_linear_velocity,
        "object_angular_velocity": object_angular_velocity,
        "goal_position": goal_position,
        "eef_to_object": object_position - eef_position,
        "object_to_goal": goal_position - object_position,
        "finger_contact_counts": components["finger_contact_counts"].to(dtype=dtype),
        "has_hand_contact": components["has_hand_contact"]
        .to(device=device, dtype=dtype)
        .reshape(batch_size, 1),
        "is_grasped": components["is_grasped"]
        .to(device=device, dtype=dtype)
        .reshape(batch_size, 1),
        "task_phase_one_hot": torch.nn.functional.one_hot(task_phase, num_classes=5).to(
            dtype=dtype
        ),
    }


def pack_stage0_state(
    fields: Mapping[str, torch.Tensor],
    schema: Stage0StateSchema = DEFAULT_STAGE0_STATE_SCHEMA,
) -> torch.Tensor:
    """Pack named canonical fields using the hash-bound schema order."""
    columns: list[torch.Tensor] = []
    batch_size: int | None = None
    for field in schema.fields:
        if field.name not in fields:
            raise KeyError(f"Missing canonical Stage 0 field {field.name!r}")
        value = fields[field.name]
        _check_last_dim(value, field.width, field.name)
        if batch_size is None:
            batch_size = value.shape[0]
        elif value.shape[0] != batch_size:
            raise ValueError(
                "All canonical Stage 0 fields must share one batch dimension"
            )
        columns.append(value)
    packed = torch.cat(columns, dim=-1).contiguous()
    if packed.shape[1] != schema.dimension:
        raise RuntimeError(
            f"Packed Stage 0 dimension {packed.shape[1]} != schema {schema.dimension}"
        )
    return packed


@dataclass(frozen=True)
class Stage0Transition:
    state: torch.Tensor
    action: torch.Tensor
    next_state: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: Mapping[str, Any]


class Stage0StateOnlyEnv:
    """No-image, no-auto-reset adapter around one-control-step Newton calls."""

    def __init__(
        self, env: Any, *, schema: Stage0StateSchema = DEFAULT_STAGE0_STATE_SCHEMA
    ):
        if not hasattr(env, "stage0_state_components_torch"):
            raise TypeError("env must expose stage0_state_components_torch()")
        if getattr(env, "action_size", None) != 19:
            raise ValueError("Stage 0 requires the 19-D absolute EEF action contract")
        self.env = env
        self.schema = schema
        self.num_envs = int(env.num_envs)
        self._state: torch.Tensor | None = None

    def observe(self) -> torch.Tensor:
        components = self.env.stage0_state_components_torch()
        state = pack_stage0_state(
            canonicalize_stage0_components(components), self.schema
        )
        if state.shape != (self.num_envs, self.schema.dimension):
            raise RuntimeError(f"Unexpected Stage 0 state shape {tuple(state.shape)}")
        self._state = state
        return state

    def reset(
        self,
        world_mask: torch.Tensor | None = None,
        *,
        seed: int | list[int] | None = None,
    ) -> tuple[torch.Tensor, Mapping[str, Any]]:
        _, info = self.env.reset_torch(world_mask, seed=seed)
        return self.observe(), info

    def step(self, action: torch.Tensor) -> Stage0Transition:
        if action.shape != (self.num_envs, 19):
            raise ValueError(
                f"action must have shape ({self.num_envs}, 19), got {tuple(action.shape)}"
            )
        if not torch.isfinite(action).all():
            raise ValueError("action contains non-finite values")
        state = self._state if self._state is not None else self.observe()
        _, reward, terminated, truncated, info = self.env.step_torch(action)
        return Stage0Transition(
            state=state,
            action=action,
            next_state=self.observe(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def reach_heuristic_action(
        self, *, max_translation_step_m: float = 0.025
    ) -> torch.Tensor:
        """Produce a deterministic open-hand reach action for smoke collection."""
        if max_translation_step_m <= 0.0:
            raise ValueError("max_translation_step_m must be positive")
        state = self._state if self._state is not None else self.observe()
        action = self.env.hold_action_torch().clone()
        eef = state[:, self.schema.slice("eef_position")]
        goal = state[:, self.schema.slice("goal_position")]
        delta = goal - eef
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        scale = torch.clamp(
            max_translation_step_m / torch.clamp_min(distance, 1.0e-8), max=1.0
        )
        action[:, :3] = eef + delta * scale
        return action


__all__ = [
    "Stage0StateOnlyEnv",
    "Stage0Transition",
    "canonicalize_stage0_components",
    "pack_stage0_state",
]
