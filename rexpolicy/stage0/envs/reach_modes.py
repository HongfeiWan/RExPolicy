"""Deterministic multi-mode waypoint policies for the Stage 0 Reach task.

The policies in this module are deliberately geometry-only.  They consume the
canonical 79D right-base-frame state and produce the existing 19D absolute EEF
action contract without importing Newton or GR00T.  Rotation and open-hand
targets are captured once at reset and then held exactly constant.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA
from rexpolicy.stage0.types import canonical_fingerprint


REACH_MODE_PLANNER_ID = "rexpolicy/stage0-reach-modes/v1"
_MODE_ID = re.compile(r"^stage0/reach/[a-z][a-z0-9_]+/v1$")


def _finite_non_negative(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")


def _positive(value: float, name: str) -> None:
    _finite_non_negative(value, name)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _check_state(state: torch.Tensor, *, name: str = "state") -> None:
    expected = DEFAULT_STAGE0_STATE_SCHEMA.dimension
    if state.ndim != 2 or state.shape[1] != expected:
        raise ValueError(
            f"{name} must have shape [B, {expected}], got {tuple(state.shape)}"
        )
    if not state.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    if not torch.isfinite(state).all():
        raise ValueError(f"{name} contains non-finite values")


def _normalize_with_fallback(
    vector: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    eps = torch.finfo(vector.dtype).eps
    normalized = vector / torch.clamp_min(norm, eps)
    return torch.where((norm > eps).expand_as(vector), normalized, fallback)


@dataclass(frozen=True)
class ReachModeSpec:
    """One hash-bound homotopy of the reset EEF-to-goal segment."""

    mode_id: str
    lateral_offset_m: float
    vertical_offset_m: float
    outward_offset_m: float
    waypoint_progress: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)

    def __post_init__(self) -> None:
        if not isinstance(self.mode_id, str) or not _MODE_ID.fullmatch(self.mode_id):
            raise ValueError("mode_id must match 'stage0/reach/<lower_snake_case>/v1'")
        for name in ("lateral_offset_m", "vertical_offset_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be finite")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        _finite_non_negative(self.outward_offset_m, "outward_offset_m")
        if not isinstance(self.waypoint_progress, tuple) or not self.waypoint_progress:
            raise ValueError("waypoint_progress must be a non-empty tuple")
        previous = 0.0
        for progress in self.waypoint_progress:
            if isinstance(progress, bool) or not isinstance(progress, (int, float)):
                raise ValueError("waypoint progress values must be finite numbers")
            progress = float(progress)
            if not math.isfinite(progress) or progress <= previous or progress > 1.0:
                raise ValueError(
                    "waypoint progress must be strictly increasing in (0, 1]"
                )
            previous = progress
        if float(self.waypoint_progress[-1]) != 1.0:
            raise ValueError("the final waypoint progress must be exactly 1.0")

    @property
    def short_name(self) -> str:
        return self.mode_id.split("/")[2]

    def to_record(self) -> dict[str, Any]:
        return {
            "planner_id": REACH_MODE_PLANNER_ID,
            "state_schema_sha256": DEFAULT_STAGE0_STATE_SCHEMA.sha256,
            "mode_id": self.mode_id,
            "lateral_offset_m": float(self.lateral_offset_m),
            "vertical_offset_m": float(self.vertical_offset_m),
            "outward_offset_m": float(self.outward_offset_m),
            "waypoint_progress": [float(value) for value in self.waypoint_progress],
        }

    @property
    def sha256(self) -> str:
        return canonical_fingerprint(self.to_record())


DEFAULT_REACH_MODE_SPECS = (
    ReachModeSpec(
        mode_id="stage0/reach/direct/v1",
        lateral_offset_m=0.0,
        vertical_offset_m=0.0,
        outward_offset_m=0.0,
    ),
    ReachModeSpec(
        mode_id="stage0/reach/left_arc/v1",
        lateral_offset_m=0.075,
        vertical_offset_m=0.020,
        outward_offset_m=0.020,
    ),
    ReachModeSpec(
        mode_id="stage0/reach/right_arc/v1",
        lateral_offset_m=-0.075,
        vertical_offset_m=0.020,
        outward_offset_m=0.020,
    ),
    ReachModeSpec(
        mode_id="stage0/reach/over_arc/v1",
        lateral_offset_m=0.0,
        vertical_offset_m=0.095,
        outward_offset_m=0.020,
    ),
)


@dataclass(frozen=True)
class ReachModeLibrary:
    """Ordered mode registry with deterministic per-reset enumeration."""

    specs: tuple[ReachModeSpec, ...] = DEFAULT_REACH_MODE_SPECS

    def __post_init__(self) -> None:
        if not isinstance(self.specs, tuple) or len(self.specs) < 3:
            raise ValueError("Reach mode library requires at least three modes")
        if not all(isinstance(spec, ReachModeSpec) for spec in self.specs):
            raise ValueError("specs must contain ReachModeSpec values")
        ids = [spec.mode_id for spec in self.specs]
        if len(ids) != len(set(ids)):
            raise ValueError("Reach mode ids must be unique")

        progress = self.specs[0].waypoint_progress
        if any(spec.waypoint_progress != progress for spec in self.specs[1:]):
            raise ValueError("all Reach modes must share one waypoint schedule")

    @property
    def mode_ids(self) -> tuple[str, ...]:
        return tuple(spec.mode_id for spec in self.specs)

    @property
    def sha256(self) -> str:
        return canonical_fingerprint(
            {
                "planner_id": REACH_MODE_PLANNER_ID,
                "mode_sha256": [spec.sha256 for spec in self.specs],
            }
        )

    def spec(self, mode_id: str) -> ReachModeSpec:
        for spec in self.specs:
            if spec.mode_id == mode_id:
                return spec
        raise KeyError(f"Unknown Reach mode id {mode_id!r}")

    def mode_hash(self, mode_id: str) -> str:
        return self.spec(mode_id).sha256

    def ordered_mode_ids(self, reset_seed: int) -> tuple[str, ...]:
        """Return every mode once, in a stable seed-dependent rotation."""
        if isinstance(reset_seed, bool) or not isinstance(reset_seed, int):
            raise ValueError("reset_seed must be an integer")
        digest = canonical_fingerprint(
            {
                "library_sha256": self.sha256,
                "reset_seed": reset_seed,
            }
        )
        offset = int(digest[:16], 16) % len(self.specs)
        ids = self.mode_ids
        return ids[offset:] + ids[:offset]

    def assign(self, reset_seeds: Sequence[int], mode_slot: int) -> tuple[str, ...]:
        """Assign one of every reset's modes for a repeated collection slot."""
        if isinstance(mode_slot, bool) or not isinstance(mode_slot, int):
            raise ValueError("mode_slot must be an integer")
        if mode_slot < 0 or mode_slot >= len(self.specs):
            raise ValueError(f"mode_slot must be in [0, {len(self.specs)})")
        return tuple(self.ordered_mode_ids(seed)[mode_slot] for seed in reset_seeds)


DEFAULT_REACH_MODE_LIBRARY = ReachModeLibrary()


def minimum_polyline_object_clearance(
    start: torch.Tensor,
    waypoints: torch.Tensor,
    object_position: torch.Tensor,
) -> torch.Tensor:
    """Return the exact point-to-segment clearance for each batched polyline."""
    if start.ndim != 2 or start.shape[1] != 3:
        raise ValueError("start must have shape [B, 3]")
    if (
        waypoints.ndim != 3
        or waypoints.shape[0] != start.shape[0]
        or waypoints.shape[2] != 3
    ):
        raise ValueError("waypoints must have shape [B, W, 3]")
    if waypoints.shape[1] < 1:
        raise ValueError("waypoints must contain at least one point")
    if object_position.shape != start.shape:
        raise ValueError("object_position must share start shape")
    if waypoints.device != start.device or object_position.device != start.device:
        raise ValueError("polyline tensors must share one device")
    if waypoints.dtype != start.dtype or object_position.dtype != start.dtype:
        raise ValueError("polyline tensors must share one dtype")
    if not (
        torch.isfinite(start).all()
        and torch.isfinite(waypoints).all()
        and torch.isfinite(object_position).all()
    ):
        raise ValueError("polyline tensors must be finite")

    segment_start = torch.cat((start[:, None, :], waypoints[:, :-1, :]), dim=1)
    segment_end = waypoints
    segment = segment_end - segment_start
    relative = object_position[:, None, :] - segment_start
    denominator = torch.sum(segment * segment, dim=-1)
    eps = torch.finfo(start.dtype).eps
    projection = torch.sum(relative * segment, dim=-1) / torch.clamp_min(
        denominator, eps
    )
    projection = torch.clamp(projection, 0.0, 1.0)
    closest = segment_start + projection[:, :, None] * segment
    distances = torch.linalg.vector_norm(closest - object_position[:, None, :], dim=-1)
    return torch.amin(distances, dim=1)


@dataclass(frozen=True)
class ReachModePlan:
    """Materialized fixed-reset waypoint plans and their provenance."""

    waypoints: torch.Tensor
    mode_ids: tuple[str, ...]
    mode_hashes: tuple[str, ...]
    minimum_object_clearance: torch.Tensor


def build_reach_mode_plan(
    initial_state: torch.Tensor,
    mode_ids: Sequence[str],
    *,
    library: ReachModeLibrary = DEFAULT_REACH_MODE_LIBRARY,
    minimum_object_clearance_m: float = 0.050,
) -> ReachModePlan:
    """Build deterministic, base-frame waypoint plans for one reset batch."""
    _check_state(initial_state, name="initial_state")
    _finite_non_negative(minimum_object_clearance_m, "minimum_object_clearance_m")
    if isinstance(mode_ids, (str, bytes)) or len(mode_ids) != initial_state.shape[0]:
        raise ValueError("mode_ids must contain one mode id per state row")
    specs = tuple(library.spec(mode_id) for mode_id in mode_ids)

    schema = DEFAULT_STAGE0_STATE_SCHEMA
    start = initial_state[:, schema.slice("eef_position")]
    goal = initial_state[:, schema.slice("goal_position")]
    object_position = initial_state[:, schema.slice("object_position")]

    batch_size = initial_state.shape[0]
    radial_fallback = torch.zeros_like(goal)
    radial_fallback[:, 0] = 1.0
    radial = _normalize_with_fallback(goal - object_position, radial_fallback)

    base_z = torch.zeros_like(radial)
    base_z[:, 2] = 1.0
    lateral_raw = torch.linalg.cross(base_z, radial, dim=-1)
    base_y = torch.zeros_like(radial)
    base_y[:, 1] = 1.0
    lateral_fallback = _normalize_with_fallback(
        torch.linalg.cross(base_y, radial, dim=-1), radial_fallback
    )
    lateral = _normalize_with_fallback(lateral_raw, lateral_fallback)
    vertical = _normalize_with_fallback(
        torch.linalg.cross(radial, lateral, dim=-1), base_z
    )

    progress = torch.tensor(
        specs[0].waypoint_progress,
        device=initial_state.device,
        dtype=initial_state.dtype,
    )
    linear = start[:, None, :] + progress[None, :, None] * (goal - start)[:, None, :]
    bump = torch.sin(progress * math.pi)[None, :, None]
    lateral_offset = initial_state.new_tensor(
        [spec.lateral_offset_m for spec in specs]
    ).reshape(batch_size, 1, 1)
    vertical_offset = initial_state.new_tensor(
        [spec.vertical_offset_m for spec in specs]
    ).reshape(batch_size, 1, 1)
    outward_offset = initial_state.new_tensor(
        [spec.outward_offset_m for spec in specs]
    ).reshape(batch_size, 1, 1)
    deviation = (
        lateral_offset * lateral[:, None, :]
        + vertical_offset * vertical[:, None, :]
        + outward_offset * radial[:, None, :]
    )
    waypoints = (linear + bump * deviation).contiguous()
    # Make the terminal contract exact even if sin(pi) incurs round-off.
    waypoints[:, -1, :].copy_(goal)

    clearance = minimum_polyline_object_clearance(start, waypoints, object_position)
    unsafe = clearance < float(minimum_object_clearance_m)
    if bool(unsafe.any()):
        rows = torch.nonzero(unsafe, as_tuple=False).flatten().tolist()
        details = ", ".join(
            f"row={row} mode={specs[row].mode_id} "
            f"clearance={float(clearance[row]):.6f}m"
            for row in rows
        )
        raise ValueError(
            "Reach mode violates the configured object clearance "
            f"({float(minimum_object_clearance_m):.6f}m): {details}"
        )
    return ReachModePlan(
        waypoints=waypoints,
        mode_ids=tuple(spec.mode_id for spec in specs),
        mode_hashes=tuple(spec.sha256 for spec in specs),
        minimum_object_clearance=clearance,
    )


class ReachModeController:
    """Advance fixed-reset Reach waypoints while pinning open-hand targets."""

    def __init__(
        self,
        *,
        max_translation_step_m: float = 0.030,
        waypoint_tolerance_m: float = 0.030,
        minimum_object_clearance_m: float = 0.050,
        library: ReachModeLibrary = DEFAULT_REACH_MODE_LIBRARY,
    ) -> None:
        _positive(max_translation_step_m, "max_translation_step_m")
        _positive(waypoint_tolerance_m, "waypoint_tolerance_m")
        _finite_non_negative(minimum_object_clearance_m, "minimum_object_clearance_m")
        if not isinstance(library, ReachModeLibrary):
            raise TypeError("library must be ReachModeLibrary")
        self.max_translation_step_m = float(max_translation_step_m)
        self.waypoint_tolerance_m = float(waypoint_tolerance_m)
        self.minimum_object_clearance_m = float(minimum_object_clearance_m)
        self.library = library
        self._plan: ReachModePlan | None = None
        self._reference_action: torch.Tensor | None = None
        self._waypoint_indices: torch.Tensor | None = None

    @property
    def plan(self) -> ReachModePlan:
        if self._plan is None:
            raise RuntimeError("ReachModeController.reset() must be called first")
        return self._plan

    @property
    def mode_ids(self) -> tuple[str, ...]:
        return self.plan.mode_ids

    @property
    def mode_hashes(self) -> tuple[str, ...]:
        return self.plan.mode_hashes

    @property
    def waypoint_indices(self) -> torch.Tensor:
        if self._waypoint_indices is None:
            raise RuntimeError("ReachModeController.reset() must be called first")
        return self._waypoint_indices

    def reset(
        self,
        initial_state: torch.Tensor,
        open_hand_reference_action: torch.Tensor,
        mode_ids: Sequence[str],
    ) -> ReachModePlan:
        """Capture reset rotation/open-hand values and build immutable paths."""
        _check_state(initial_state, name="initial_state")
        expected_action_shape = (initial_state.shape[0], 19)
        if open_hand_reference_action.shape != expected_action_shape:
            raise ValueError(
                "open_hand_reference_action must have shape "
                f"{expected_action_shape}, got {tuple(open_hand_reference_action.shape)}"
            )
        if (
            open_hand_reference_action.device != initial_state.device
            or open_hand_reference_action.dtype != initial_state.dtype
        ):
            raise ValueError("reference action must share state dtype and device")
        if not torch.isfinite(open_hand_reference_action).all():
            raise ValueError("open_hand_reference_action contains non-finite values")
        plan = build_reach_mode_plan(
            initial_state,
            mode_ids,
            library=self.library,
            minimum_object_clearance_m=self.minimum_object_clearance_m,
        )
        self._plan = plan
        self._reference_action = open_hand_reference_action.detach().clone()
        self._waypoint_indices = torch.zeros(
            initial_state.shape[0], dtype=torch.long, device=initial_state.device
        )
        return plan

    def action(
        self,
        current_state: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one bounded absolute EEF command for every active world."""
        _check_state(current_state, name="current_state")
        if self._reference_action is None or self._waypoint_indices is None:
            raise RuntimeError("ReachModeController.reset() must be called first")
        if (
            current_state.shape[0] != self._reference_action.shape[0]
            or current_state.device != self._reference_action.device
            or current_state.dtype != self._reference_action.dtype
        ):
            raise ValueError(
                "current_state must match the reset batch, device, and dtype"
            )
        if active_mask is None:
            active_mask = torch.ones(
                current_state.shape[0], dtype=torch.bool, device=current_state.device
            )
        elif active_mask.shape != (current_state.shape[0],):
            raise ValueError("active_mask must have shape [B]")
        elif (
            active_mask.device != current_state.device
            or active_mask.dtype != torch.bool
        ):
            raise ValueError("active_mask must be a boolean tensor on the state device")

        eef = current_state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position")]
        last_index = self.plan.waypoints.shape[1] - 1
        batch_indices = torch.arange(
            current_state.shape[0], device=current_state.device
        )
        # Skip any waypoint already reached.  A bounded loop avoids a CPU sync
        # and handles duplicated/very short geometric segments deterministically.
        for _ in range(self.plan.waypoints.shape[1]):
            target = self.plan.waypoints[batch_indices, self._waypoint_indices]
            reached = (
                torch.linalg.vector_norm(target - eef, dim=-1)
                <= self.waypoint_tolerance_m
            ) & active_mask
            advance = reached & (self._waypoint_indices < last_index)
            self._waypoint_indices.add_(advance.to(dtype=torch.long))

        target = self.plan.waypoints[batch_indices, self._waypoint_indices]
        delta = target - eef
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        scale = torch.clamp(
            self.max_translation_step_m / torch.clamp_min(distance, 1.0e-8),
            max=1.0,
        )
        commanded_position = eef + delta * scale
        action = self._reference_action.clone()
        action[:, :3].copy_(torch.where(active_mask[:, None], commanded_position, eef))
        return action

    def next_action(
        self,
        current_state: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Spelling alias convenient for collection loops."""
        return self.action(current_state, active_mask)


__all__ = [
    "DEFAULT_REACH_MODE_LIBRARY",
    "DEFAULT_REACH_MODE_SPECS",
    "REACH_MODE_PLANNER_ID",
    "ReachModeController",
    "ReachModeLibrary",
    "ReachModePlan",
    "ReachModeSpec",
    "build_reach_mode_plan",
    "minimum_polyline_object_clearance",
]
