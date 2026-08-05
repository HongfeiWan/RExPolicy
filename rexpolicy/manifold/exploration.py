"""Shadow-only diversity scoring and generation-stable exploration schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping

from rexpolicy.tasking.canonical import canonical_fingerprint

from .occupancy import OccupancyQuery


EXPLORATION_POLICY_SCHEMA_VERSION = 1


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name, minimum=0.0)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _label(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise ValueError(f"{name} must be a non-empty bounded string")
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class ExplorationPolicy:
    """A manifest-bound policy; all fields are default-off control state."""

    schema_version: int = EXPLORATION_POLICY_SCHEMA_VERSION
    enabled: bool = False
    selector_temperature_start: float = 1.0
    selector_temperature_end: float = 1.0
    novelty_coefficient_start: float = 0.0
    novelty_coefficient_end: float = 0.0
    anneal_generations: int = 1
    novelty_scale: float = 1.0
    novelty_cap: float = 4.0
    active_candidates: int = 4

    def __post_init__(self) -> None:
        if self.schema_version != EXPLORATION_POLICY_SCHEMA_VERSION:
            raise ValueError(
                "ExplorationPolicy schema_version must be "
                f"{EXPLORATION_POLICY_SCHEMA_VERSION}"
            )
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        _positive(
            self.selector_temperature_start,
            "selector_temperature_start",
        )
        _positive(
            self.selector_temperature_end,
            "selector_temperature_end",
        )
        _finite(
            self.novelty_coefficient_start,
            "novelty_coefficient_start",
            minimum=0.0,
        )
        _finite(
            self.novelty_coefficient_end,
            "novelty_coefficient_end",
            minimum=0.0,
        )
        _positive_int(self.anneal_generations, "anneal_generations")
        _positive(self.novelty_scale, "novelty_scale")
        _positive(self.novelty_cap, "novelty_cap")
        _positive_int(self.active_candidates, "active_candidates")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExplorationPolicy:
        if not isinstance(value, Mapping):
            raise ValueError("Exploration policy must be an object")
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(value).difference(known))
        if unknown:
            raise ValueError(
                "Unknown exploration policy fields: " + ", ".join(unknown)
            )
        return cls(**dict(value))

    def to_record(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())

    def at_generation(self, generation: int) -> ExplorationSchedulePoint:
        _non_negative_int(generation, "generation")
        progress = min(
            max(generation - 1, 0) / self.anneal_generations,
            1.0,
        )

        def interpolate(start: float, end: float) -> float:
            return float(start) + progress * (float(end) - float(start))

        return ExplorationSchedulePoint(
            generation=generation,
            progress=progress,
            selector_temperature=interpolate(
                self.selector_temperature_start,
                self.selector_temperature_end,
            ),
            novelty_coefficient=interpolate(
                self.novelty_coefficient_start,
                self.novelty_coefficient_end,
            ),
            policy_sha256=self.fingerprint,
        )


@dataclass(frozen=True)
class ExplorationSchedulePoint:
    generation: int
    progress: float
    selector_temperature: float
    novelty_coefficient: float
    policy_sha256: str

    def to_record(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "progress": self.progress,
            "selector_temperature": self.selector_temperature,
            "novelty_coefficient": self.novelty_coefficient,
            "policy_sha256": self.policy_sha256,
        }


@dataclass(frozen=True)
class ExplorationScore:
    sample_id: str
    latent_sha256: str
    latent_source: str
    task_success_score: float
    nearest_mode_id: str | None
    novelty_distance: float | None
    normalized_novelty: float
    novelty_coefficient: float
    shadow_score: float
    bootstrap: bool
    is_novel: bool
    successful_latent_count: int
    discovered_modes: int
    authority: str = "shadow_only"

    def to_record(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "sample_id": self.sample_id,
            "latent_sha256": self.latent_sha256,
            "latent_source": self.latent_source,
            "task_success_score": self.task_success_score,
            "nearest_mode_id": self.nearest_mode_id,
            "novelty_distance": self.novelty_distance,
            "normalized_novelty": self.normalized_novelty,
            "novelty_coefficient": self.novelty_coefficient,
            "shadow_score": self.shadow_score,
            "bootstrap": self.bootstrap,
            "is_novel": self.is_novel,
            "successful_latent_count": self.successful_latent_count,
            "discovered_modes": self.discovered_modes,
        }


def score_shadow_exploration(
    *,
    sample_id: str,
    latent_sha256: str,
    latent_source: str,
    task_success_score: float,
    occupancy: OccupancyQuery,
    schedule: ExplorationSchedulePoint,
    policy: ExplorationPolicy,
) -> ExplorationScore:
    """Return a diagnostic score without granting it selection authority."""
    if not policy.enabled:
        raise ValueError("Shadow exploration scoring requires an enabled policy")
    if schedule.policy_sha256 != policy.fingerprint:
        raise ValueError("Exploration schedule does not match its policy")
    _label(sample_id, "sample_id")
    _sha256(latent_sha256, "latent_sha256")
    _label(latent_source, "latent_source")
    task_success_score = _finite(task_success_score, "task_success_score")
    if not isinstance(occupancy, OccupancyQuery):
        raise TypeError("occupancy must be an OccupancyQuery")
    if occupancy.nearest_distance is None:
        bootstrap = True
        normalized_novelty = float(policy.novelty_cap)
    else:
        bootstrap = False
        distance = _finite(
            occupancy.nearest_distance,
            "occupancy.nearest_distance",
            minimum=0.0,
        )
        normalized_novelty = min(
            distance / float(policy.novelty_scale),
            float(policy.novelty_cap),
        )
    shadow_score = (
        task_success_score
        + schedule.novelty_coefficient * normalized_novelty
    )
    return ExplorationScore(
        sample_id=sample_id,
        latent_sha256=latent_sha256,
        latent_source=latent_source,
        task_success_score=task_success_score,
        nearest_mode_id=occupancy.nearest_mode_id,
        novelty_distance=occupancy.nearest_distance,
        normalized_novelty=normalized_novelty,
        novelty_coefficient=schedule.novelty_coefficient,
        shadow_score=shadow_score,
        bootstrap=bootstrap,
        is_novel=occupancy.is_novel,
        successful_latent_count=occupancy.successful_latent_count,
        discovered_modes=occupancy.discovered_modes,
    )


__all__ = [
    "EXPLORATION_POLICY_SCHEMA_VERSION",
    "ExplorationPolicy",
    "ExplorationSchedulePoint",
    "ExplorationScore",
    "score_shadow_exploration",
]
