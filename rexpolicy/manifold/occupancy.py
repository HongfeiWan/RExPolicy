"""Deterministic shadow occupancy over simulator-verified success latents.

The index is control-plane state.  It never changes reward, replay admission,
candidate ranking, or policy gradients.  Records are merged in one canonical
order so every DDP rank can reproduce the same clusters and fingerprint.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

from rexpolicy.tasking.canonical import canonical_fingerprint


SUCCESS_OCCUPANCY_CONFIG_SCHEMA_VERSION = 1
SUCCESS_OCCUPANCY_STATE_SCHEMA_VERSION = 1


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


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


def _vector(value: Any, *, latent_dim: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a numeric sequence")
    if len(value) != latent_dim:
        raise ValueError(f"{name} must contain exactly {latent_dim} values")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name}[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(number)
    return tuple(result)


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(left, right)))


@dataclass(frozen=True)
class SuccessOccupancyConfig:
    """Strict occupancy policy whose fingerprint is bound to checkpoints."""

    schema_version: int = SUCCESS_OCCUPANCY_CONFIG_SCHEMA_VERSION
    enabled: bool = False
    latent_dim: int = 64
    distance_metric: str = "raw_l2"
    cluster_radius: float = 0.25
    density_bandwidth: float = 0.25
    coverage_target_modes: int = 32
    max_records: int = 100_000

    def __post_init__(self) -> None:
        if self.schema_version != SUCCESS_OCCUPANCY_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                "SuccessOccupancyConfig schema_version must be "
                f"{SUCCESS_OCCUPANCY_CONFIG_SCHEMA_VERSION}"
            )
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        _positive_int(self.latent_dim, "latent_dim")
        _positive_int(self.coverage_target_modes, "coverage_target_modes")
        _positive_int(self.max_records, "max_records")
        if self.distance_metric != "raw_l2":
            raise ValueError("distance_metric must be raw_l2")
        _positive_float(self.cluster_radius, "cluster_radius")
        _positive_float(self.density_bandwidth, "density_bandwidth")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SuccessOccupancyConfig:
        if not isinstance(value, Mapping):
            raise ValueError("Success occupancy config must be an object")
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(value).difference(known))
        if unknown:
            raise ValueError(
                "Unknown success occupancy config fields: "
                + ", ".join(unknown)
            )
        return cls(**dict(value))

    def to_record(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class OccupancySuccessRecord:
    sample_id: str
    latent: tuple[float, ...]
    generation: int
    rank: int
    episode_id: str
    source_sha256: str

    @classmethod
    def from_record(
        cls,
        value: Mapping[str, Any],
        *,
        latent_dim: int,
    ) -> OccupancySuccessRecord:
        if not isinstance(value, Mapping) or set(value) != {
            "sample_id",
            "latent",
            "generation",
            "rank",
            "episode_id",
            "source_sha256",
        }:
            raise ValueError("Occupancy success record inventory is invalid")
        return cls(
            sample_id=_label(value["sample_id"], "sample_id"),
            latent=_vector(
                value["latent"], latent_dim=latent_dim, name="latent"
            ),
            generation=_non_negative_int(value["generation"], "generation"),
            rank=_non_negative_int(value["rank"], "rank"),
            episode_id=_label(value["episode_id"], "episode_id"),
            source_sha256=_sha256(value["source_sha256"], "source_sha256"),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "latent": list(self.latent),
            "generation": self.generation,
            "rank": self.rank,
            "episode_id": self.episode_id,
            "source_sha256": self.source_sha256,
        }

    @property
    def sort_key(self) -> tuple[int, int, str, str]:
        return (self.generation, self.rank, self.episode_id, self.sample_id)


@dataclass(frozen=True)
class OccupancyQuery:
    nearest_mode_id: str | None
    nearest_distance: float | None
    log_density: float | None
    is_novel: bool
    successful_latent_count: int
    discovered_modes: int


@dataclass(frozen=True)
class OccupancyUpdate:
    sample_id: str
    accepted: bool
    reason: str
    mode_id: str
    is_new_mode: bool
    nearest_distance: float | None
    log_density_before: float | None
    successful_latent_count: int
    discovered_modes: int


@dataclass(frozen=True)
class OccupancyMetrics:
    generation: int | None
    successful_latent_count: int
    discovered_modes: int
    active_modes: int
    novel_successes: int
    novel_success_rate: float
    cluster_assignment_entropy_nats: float
    effective_mode_count: float
    target_mode_coverage: float


@dataclass
class _Mode:
    mode_id: str
    centroid: tuple[float, ...]
    count: int
    first_generation: int
    last_generation: int


class SuccessOccupancyIndex:
    """Append-only raw-L2 mode inventory and Gaussian-kernel density view."""

    def __init__(self, config: SuccessOccupancyConfig) -> None:
        if not config.enabled:
            raise ValueError("SuccessOccupancyIndex requires enabled config")
        self.config = config
        self._records: list[OccupancySuccessRecord] = []
        self._records_by_id: dict[str, OccupancySuccessRecord] = {}
        self._modes: list[_Mode] = []
        self._assignments: dict[str, tuple[str, bool]] = {}

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[OccupancySuccessRecord, ...]:
        return tuple(self._records)

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.state_dict())

    def _nearest_mode(
        self, latent: Sequence[float]
    ) -> tuple[_Mode | None, float | None]:
        if not self._modes:
            return None, None
        distance, _mode_id, mode = min(
            (
                (_distance(latent, item.centroid), item.mode_id, item)
                for item in self._modes
            ),
            key=lambda candidate: (candidate[0], candidate[1]),
        )
        return mode, distance

    def _log_density(self, latent: Sequence[float]) -> float | None:
        if not self._records:
            return None
        scale = 2.0 * float(self.config.density_bandwidth) ** 2
        terms = [
            -(_distance(latent, item.latent) ** 2) / scale
            for item in self._records
        ]
        maximum = max(terms)
        return maximum + math.log(
            math.fsum(math.exp(item - maximum) for item in terms)
        ) - math.log(len(terms))

    def query(self, latent: Sequence[float]) -> OccupancyQuery:
        encoded = _vector(
            latent, latent_dim=self.config.latent_dim, name="latent"
        )
        mode, distance = self._nearest_mode(encoded)
        return OccupancyQuery(
            nearest_mode_id=None if mode is None else mode.mode_id,
            nearest_distance=distance,
            log_density=self._log_density(encoded),
            is_novel=(
                distance is None
                or distance >= float(self.config.cluster_radius)
            ),
            successful_latent_count=len(self._records),
            discovered_modes=len(self._modes),
        )

    @staticmethod
    def _new_mode_id(sample_id: str) -> str:
        return "mode-" + canonical_fingerprint({"sample_id": sample_id})[:16]

    def observe_success(
        self,
        record: OccupancySuccessRecord | Mapping[str, Any],
    ) -> OccupancyUpdate:
        if not isinstance(record, OccupancySuccessRecord):
            record = OccupancySuccessRecord.from_record(
                record, latent_dim=self.config.latent_dim
            )
        else:
            record = OccupancySuccessRecord.from_record(
                record.to_record(), latent_dim=self.config.latent_dim
            )
        previous = self._records_by_id.get(record.sample_id)
        if previous is not None:
            if previous != record:
                raise ValueError(
                    f"Conflicting occupancy record for sample_id={record.sample_id}"
                )
            mode_id, is_new_mode = self._assignments[record.sample_id]
            query = self.query(record.latent)
            return OccupancyUpdate(
                sample_id=record.sample_id,
                accepted=False,
                reason="duplicate_sample_id",
                mode_id=mode_id,
                is_new_mode=is_new_mode,
                nearest_distance=query.nearest_distance,
                log_density_before=query.log_density,
                successful_latent_count=len(self._records),
                discovered_modes=len(self._modes),
            )
        if len(self._records) >= self.config.max_records:
            raise ValueError("Success occupancy max_records would be exceeded")
        if self._records and record.sort_key <= self._records[-1].sort_key:
            raise ValueError(
                "New occupancy records must append in canonical order"
            )
        query = self.query(record.latent)
        nearest_mode, nearest_distance = self._nearest_mode(record.latent)
        is_new_mode = query.is_novel
        if is_new_mode:
            mode = _Mode(
                mode_id=self._new_mode_id(record.sample_id),
                centroid=record.latent,
                count=1,
                first_generation=record.generation,
                last_generation=record.generation,
            )
            self._modes.append(mode)
            self._modes.sort(key=lambda item: item.mode_id)
        else:
            if nearest_mode is None:
                raise RuntimeError("Non-novel occupancy record has no nearest mode")
            mode = nearest_mode
            new_count = mode.count + 1
            mode.centroid = tuple(
                (old * mode.count + value) / new_count
                for old, value in zip(mode.centroid, record.latent)
            )
            mode.count = new_count
            mode.last_generation = record.generation
        self._records.append(record)
        self._records_by_id[record.sample_id] = record
        self._assignments[record.sample_id] = (mode.mode_id, is_new_mode)
        return OccupancyUpdate(
            sample_id=record.sample_id,
            accepted=True,
            reason="new_mode" if is_new_mode else "assigned_existing_mode",
            mode_id=mode.mode_id,
            is_new_mode=is_new_mode,
            nearest_distance=nearest_distance,
            log_density_before=query.log_density,
            successful_latent_count=len(self._records),
            discovered_modes=len(self._modes),
        )

    def merge(
        self,
        records: Sequence[OccupancySuccessRecord | Mapping[str, Any]],
    ) -> tuple[OccupancyUpdate, ...]:
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise ValueError("records must be a sequence")
        normalized = [
            item
            if isinstance(item, OccupancySuccessRecord)
            else OccupancySuccessRecord.from_record(
                item, latent_dim=self.config.latent_dim
            )
            for item in records
        ]
        normalized.sort(key=lambda item: item.sort_key)
        return tuple(self.observe_success(item) for item in normalized)

    def snapshot(self, generation: int | None = None) -> OccupancyMetrics:
        if generation is not None:
            _non_negative_int(generation, "generation")
            selected_ids = {
                item.sample_id
                for item in self._records
                if item.generation == generation
            }
        else:
            selected_ids = {item.sample_id for item in self._records}
        counts: dict[str, int] = {}
        novel_successes = 0
        for sample_id in selected_ids:
            mode_id, is_new_mode = self._assignments[sample_id]
            counts[mode_id] = counts.get(mode_id, 0) + 1
            novel_successes += int(is_new_mode)
        successful_latent_count = len(selected_ids)
        if successful_latent_count:
            probabilities = [
                count / successful_latent_count for count in counts.values()
            ]
            entropy = -math.fsum(
                probability * math.log(probability)
                for probability in probabilities
            )
        else:
            entropy = 0.0
        return OccupancyMetrics(
            generation=generation,
            successful_latent_count=successful_latent_count,
            discovered_modes=len(self._modes),
            active_modes=len(counts),
            novel_successes=novel_successes,
            novel_success_rate=(
                novel_successes / successful_latent_count
                if successful_latent_count
                else 0.0
            ),
            cluster_assignment_entropy_nats=entropy,
            effective_mode_count=(
                math.exp(entropy) if successful_latent_count else 0.0
            ),
            target_mode_coverage=min(
                len(self._modes) / self.config.coverage_target_modes,
                1.0,
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SUCCESS_OCCUPANCY_STATE_SCHEMA_VERSION,
            "config_sha256": self.config.fingerprint,
            "records": [item.to_record() for item in self._records],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or set(state) != {
            "schema_version",
            "config_sha256",
            "records",
        }:
            raise ValueError("Success occupancy state inventory is invalid")
        if state["schema_version"] != SUCCESS_OCCUPANCY_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported success occupancy state schema")
        if state["config_sha256"] != self.config.fingerprint:
            raise ValueError("Success occupancy config changed on resume")
        raw_records = state["records"]
        if not isinstance(raw_records, list):
            raise ValueError("Success occupancy state records must be an array")
        restored = SuccessOccupancyIndex(self.config)
        restored.merge(copy.deepcopy(raw_records))
        self._records = restored._records
        self._records_by_id = restored._records_by_id
        self._modes = restored._modes
        self._assignments = restored._assignments

    @classmethod
    def from_state_dict(
        cls,
        config: SuccessOccupancyConfig,
        state: Mapping[str, Any],
    ) -> SuccessOccupancyIndex:
        index = cls(config)
        index.load_state_dict(state)
        return index


__all__ = [
    "SUCCESS_OCCUPANCY_CONFIG_SCHEMA_VERSION",
    "SUCCESS_OCCUPANCY_STATE_SCHEMA_VERSION",
    "OccupancyMetrics",
    "OccupancyQuery",
    "OccupancySuccessRecord",
    "OccupancyUpdate",
    "SuccessOccupancyConfig",
    "SuccessOccupancyIndex",
]
