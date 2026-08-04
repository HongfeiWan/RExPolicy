"""Rebuildable quality-diversity indexing and balanced success replay.

The Success Archive remains the immutable source of truth.  This module builds
one content-addressed view over exact success-reference fingerprints and raw
behavior metrics.  Cell elites are metadata only: lower-quality successes stay
in every cell's member list and remain reachable by the replay planner.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from rexpolicy.tasking.canonical import (
    canonical_fingerprint,
    canonical_json,
    strict_json_loads,
)

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BALANCE_AXES = (
    "behavior_cell",
    "initial_state_group",
    "reward_profile",
)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _keys(record: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = sorted(expected.difference(record))
    unknown = sorted(set(record).difference(expected))
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{path} has unknown fields: {', '.join(unknown)}")


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} must be a versioned identifier")
    return value


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256")
    return value


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


@dataclass(frozen=True)
class BehaviorDimension:
    dimension_id: str
    minimum: float
    maximum: float
    bins: int

    @classmethod
    def from_record(cls, value: Any, path: str) -> BehaviorDimension:
        record = _mapping(value, path)
        _keys(record, {"dimension_id", "minimum", "maximum", "bins"}, path)
        dimension = cls(
            dimension_id=_identifier(
                record["dimension_id"], f"{path}.dimension_id"
            ),
            minimum=_finite(record["minimum"], f"{path}.minimum"),
            maximum=_finite(record["maximum"], f"{path}.maximum"),
            bins=record["bins"],
        )
        dimension.validate()
        return dimension

    def validate(self) -> None:
        _identifier(self.dimension_id, "dimension_id")
        minimum = _finite(self.minimum, "minimum")
        maximum = _finite(self.maximum, "maximum")
        if minimum >= maximum:
            raise ValueError("Behavior dimension minimum must be below maximum")
        if type(self.bins) is not int or not 1 <= self.bins <= 1024:
            raise ValueError("Behavior dimension bins must be between 1 and 1024")

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "dimension_id": self.dimension_id,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "bins": self.bins,
        }

    def bucket(self, value: float) -> int:
        number = _finite(value, self.dimension_id)
        if number < self.minimum or number > self.maximum:
            raise ValueError(
                f"Behavior metric {self.dimension_id!r} is outside its policy range"
            )
        if number == self.maximum:
            return self.bins - 1
        width = (self.maximum - self.minimum) / self.bins
        return min(self.bins - 1, int((number - self.minimum) / width))


@dataclass(frozen=True)
class QualityDiversityPolicy:
    schema_version: int
    policy_id: str
    task_id: str
    task_oracle_sha256: str
    dimensions: tuple[BehaviorDimension, ...]
    quality_metric_id: str
    maximize_quality: bool
    balance_axes: tuple[str, ...] = _BALANCE_AXES

    @classmethod
    def from_record(cls, value: Any) -> QualityDiversityPolicy:
        record = _mapping(value, "$quality_diversity_policy")
        _keys(
            record,
            {
                "schema_version",
                "policy_id",
                "task_id",
                "task_oracle_sha256",
                "dimensions",
                "quality_metric_id",
                "maximize_quality",
                "balance_axes",
            },
            "$quality_diversity_policy",
        )
        raw_dimensions = record["dimensions"]
        raw_axes = record["balance_axes"]
        if not isinstance(raw_dimensions, list) or not raw_dimensions:
            raise ValueError("Quality-diversity dimensions must be non-empty")
        if not isinstance(raw_axes, list):
            raise ValueError("Quality-diversity balance_axes must be an array")
        policy = cls(
            schema_version=record["schema_version"],
            policy_id=_identifier(record["policy_id"], "$policy.policy_id"),
            task_id=_identifier(record["task_id"], "$policy.task_id"),
            task_oracle_sha256=_sha256(
                record["task_oracle_sha256"], "$policy.task_oracle_sha256"
            ),
            dimensions=tuple(
                BehaviorDimension.from_record(item, f"$policy.dimensions[{index}]")
                for index, item in enumerate(raw_dimensions)
            ),
            quality_metric_id=_identifier(
                record["quality_metric_id"], "$policy.quality_metric_id"
            ),
            maximize_quality=record["maximize_quality"],
            balance_axes=tuple(raw_axes),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("QualityDiversityPolicy schema_version must be 1")
        _identifier(self.policy_id, "policy_id")
        _identifier(self.task_id, "task_id")
        _sha256(self.task_oracle_sha256, "task_oracle_sha256")
        if type(self.maximize_quality) is not bool:
            raise ValueError("maximize_quality must be a boolean")
        dimensions = tuple(self.dimensions)
        if not dimensions:
            raise ValueError("Quality-diversity dimensions must be non-empty")
        for dimension in dimensions:
            dimension.validate()
        ids = tuple(item.dimension_id for item in dimensions)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("Behavior dimensions must be sorted and unique")
        _identifier(self.quality_metric_id, "quality_metric_id")
        if tuple(self.balance_axes) != _BALANCE_AXES:
            raise ValueError("Quality-diversity v1 balance axes are fixed")

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "task_id": self.task_id,
            "task_oracle_sha256": self.task_oracle_sha256,
            "dimensions": [item.to_record() for item in self.dimensions],
            "quality_metric_id": self.quality_metric_id,
            "maximize_quality": self.maximize_quality,
            "balance_axes": list(self.balance_axes),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class SuccessBehaviorDescriptor:
    schema_version: int
    sample_id: str
    success_reference_sha256: str
    event_ledger_sha256: str
    compilation_policy_sha256: str
    task_id: str
    task_oracle_sha256: str
    reward_profile_id: str
    reward_profile_sha256: str
    initial_state_group_id: str
    initial_state_group_sha256: str
    metrics: tuple[tuple[str, float], ...]

    @classmethod
    def from_record(cls, value: Any) -> SuccessBehaviorDescriptor:
        path = "$success_behavior_descriptor"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "sample_id",
                "success_reference_sha256",
                "event_ledger_sha256",
                "compilation_policy_sha256",
                "task_id",
                "task_oracle_sha256",
                "reward_profile_id",
                "reward_profile_sha256",
                "initial_state_group_id",
                "initial_state_group_sha256",
                "metrics",
            },
            path,
        )
        raw_metrics = _mapping(record["metrics"], f"{path}.metrics")
        descriptor = cls(
            schema_version=record["schema_version"],
            sample_id=_identifier(record["sample_id"], f"{path}.sample_id"),
            success_reference_sha256=_sha256(
                record["success_reference_sha256"],
                f"{path}.success_reference_sha256",
            ),
            event_ledger_sha256=_sha256(
                record["event_ledger_sha256"],
                f"{path}.event_ledger_sha256",
            ),
            compilation_policy_sha256=_sha256(
                record["compilation_policy_sha256"],
                f"{path}.compilation_policy_sha256",
            ),
            task_id=_identifier(record["task_id"], f"{path}.task_id"),
            task_oracle_sha256=_sha256(
                record["task_oracle_sha256"], f"{path}.task_oracle_sha256"
            ),
            reward_profile_id=_identifier(
                record["reward_profile_id"], f"{path}.reward_profile_id"
            ),
            reward_profile_sha256=_sha256(
                record["reward_profile_sha256"],
                f"{path}.reward_profile_sha256",
            ),
            initial_state_group_id=_identifier(
                record["initial_state_group_id"],
                f"{path}.initial_state_group_id",
            ),
            initial_state_group_sha256=_sha256(
                record["initial_state_group_sha256"],
                f"{path}.initial_state_group_sha256",
            ),
            metrics=tuple(
                sorted(
                    (
                        _identifier(name, f"{path}.metrics key"),
                        _finite(number, f"{path}.metrics.{name}"),
                    )
                    for name, number in raw_metrics.items()
                )
            ),
        )
        descriptor.validate()
        return descriptor

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("SuccessBehaviorDescriptor schema_version must be 1")
        for name, value in (
            ("sample_id", self.sample_id),
            ("task_id", self.task_id),
            ("reward_profile_id", self.reward_profile_id),
            ("initial_state_group_id", self.initial_state_group_id),
        ):
            _identifier(value, name)
        _sha256(self.success_reference_sha256, "success_reference_sha256")
        for name, value in (
            ("event_ledger_sha256", self.event_ledger_sha256),
            ("compilation_policy_sha256", self.compilation_policy_sha256),
            ("task_oracle_sha256", self.task_oracle_sha256),
            ("reward_profile_sha256", self.reward_profile_sha256),
            ("initial_state_group_sha256", self.initial_state_group_sha256),
        ):
            _sha256(value, name)
        metrics = tuple(self.metrics)
        names = tuple(name for name, _ in metrics)
        if not metrics or names != tuple(sorted(set(names))):
            raise ValueError("Behavior metrics must be non-empty, sorted, and unique")
        for name, value in metrics:
            _identifier(name, "metric name")
            _finite(value, name)

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "success_reference_sha256": self.success_reference_sha256,
            "event_ledger_sha256": self.event_ledger_sha256,
            "compilation_policy_sha256": self.compilation_policy_sha256,
            "task_id": self.task_id,
            "task_oracle_sha256": self.task_oracle_sha256,
            "reward_profile_id": self.reward_profile_id,
            "reward_profile_sha256": self.reward_profile_sha256,
            "initial_state_group_id": self.initial_state_group_id,
            "initial_state_group_sha256": self.initial_state_group_sha256,
            "metrics": dict(self.metrics),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class QualityDiversityCell:
    coordinates: tuple[int, ...]
    member_descriptor_sha256s: tuple[str, ...]
    elite_descriptor_sha256: str

    @property
    def cell_id(self) -> str:
        return ":".join(str(value) for value in self.coordinates)

    def to_record(self) -> dict[str, Any]:
        return {
            "coordinates": list(self.coordinates),
            "cell_id": self.cell_id,
            "member_descriptor_sha256s": list(self.member_descriptor_sha256s),
            "elite_descriptor_sha256": self.elite_descriptor_sha256,
        }


def _canonical_descriptors(
    descriptors: Iterable[SuccessBehaviorDescriptor],
) -> tuple[SuccessBehaviorDescriptor, ...]:
    canonical = []
    for index, item in enumerate(descriptors):
        if not isinstance(item, SuccessBehaviorDescriptor):
            raise ValueError(f"descriptors[{index}] must be a behavior descriptor")
        canonical.append(SuccessBehaviorDescriptor.from_record(item.to_record()))
    canonical.sort(key=lambda item: item.sample_id)
    sample_ids = tuple(item.sample_id for item in canonical)
    reference_hashes = tuple(item.success_reference_sha256 for item in canonical)
    if not canonical:
        raise ValueError("Quality-diversity index requires at least one success")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Quality-diversity sample IDs must be unique")
    if len(set(reference_hashes)) != len(reference_hashes):
        raise ValueError("Success references cannot appear more than once")
    return tuple(canonical)


def _descriptor_metrics(
    descriptor: SuccessBehaviorDescriptor,
    policy: QualityDiversityPolicy,
) -> dict[str, float]:
    if descriptor.task_id != policy.task_id:
        raise ValueError("Behavior descriptor task does not match QD policy")
    if descriptor.task_oracle_sha256 != policy.task_oracle_sha256:
        raise ValueError("Behavior descriptor task oracle does not match QD policy")
    metrics = dict(descriptor.metrics)
    required = {
        *(item.dimension_id for item in policy.dimensions),
        policy.quality_metric_id,
    }
    if set(metrics) != required:
        raise ValueError("Behavior metrics do not exactly match the QD policy")
    return metrics


def _build_cells(
    *,
    policy: QualityDiversityPolicy,
    descriptors: tuple[SuccessBehaviorDescriptor, ...],
) -> tuple[QualityDiversityCell, ...]:
    by_cell: dict[tuple[int, ...], list[SuccessBehaviorDescriptor]] = {}
    for descriptor in descriptors:
        metrics = _descriptor_metrics(descriptor, policy)
        coordinates = tuple(
            dimension.bucket(metrics[dimension.dimension_id])
            for dimension in policy.dimensions
        )
        by_cell.setdefault(coordinates, []).append(descriptor)
    cells = []
    for coordinates, members in sorted(by_cell.items()):
        ordered = tuple(sorted(members, key=lambda item: item.fingerprint))
        if policy.maximize_quality:
            elite = min(
                ordered,
                key=lambda item: (
                    -dict(item.metrics)[policy.quality_metric_id],
                    item.fingerprint,
                ),
            )
        else:
            elite = min(
                ordered,
                key=lambda item: (
                    dict(item.metrics)[policy.quality_metric_id],
                    item.fingerprint,
                ),
            )
        cells.append(
            QualityDiversityCell(
                coordinates=coordinates,
                member_descriptor_sha256s=tuple(
                    item.fingerprint for item in ordered
                ),
                elite_descriptor_sha256=elite.fingerprint,
            )
        )
    return tuple(cells)


@dataclass(frozen=True, init=False)
class QualityDiversityIndex:
    schema_version: int
    artifact_type: str
    policy: QualityDiversityPolicy
    policy_sha256: str
    event_ledger_set_sha256: str
    source_success_set_sha256: str
    descriptors: tuple[SuccessBehaviorDescriptor, ...]
    cells: tuple[QualityDiversityCell, ...]

    def __init__(
        self,
        *,
        policy: QualityDiversityPolicy,
        descriptors: Iterable[SuccessBehaviorDescriptor],
    ) -> None:
        canonical_policy = QualityDiversityPolicy.from_record(policy.to_record())
        canonical_descriptors = _canonical_descriptors(descriptors)
        cells = _build_cells(
            policy=canonical_policy,
            descriptors=canonical_descriptors,
        )
        object.__setattr__(self, "schema_version", 1)
        object.__setattr__(self, "artifact_type", "rexpolicy_qd_index")
        object.__setattr__(self, "policy", canonical_policy)
        object.__setattr__(self, "policy_sha256", canonical_policy.fingerprint)
        object.__setattr__(
            self,
            "event_ledger_set_sha256",
            canonical_fingerprint(
                sorted(
                    {
                        item.event_ledger_sha256
                        for item in canonical_descriptors
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "source_success_set_sha256",
            canonical_fingerprint(
                [item.to_record() for item in canonical_descriptors]
            ),
        )
        object.__setattr__(self, "descriptors", canonical_descriptors)
        object.__setattr__(self, "cells", cells)

    @classmethod
    def from_record(cls, value: Any) -> QualityDiversityIndex:
        path = "$quality_diversity_index"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "artifact_type",
                "policy",
                "policy_sha256",
                "event_ledger_set_sha256",
                "source_success_set_sha256",
                "descriptors",
                "cells",
            },
            path,
        )
        if record["schema_version"] != 1:
            raise ValueError("QualityDiversityIndex schema_version must be 1")
        if record["artifact_type"] != "rexpolicy_qd_index":
            raise ValueError("Unsupported quality-diversity artifact type")
        raw_descriptors = record["descriptors"]
        if not isinstance(raw_descriptors, list):
            raise ValueError("Quality-diversity descriptors must be an array")
        index = cls(
            policy=QualityDiversityPolicy.from_record(record["policy"]),
            descriptors=tuple(
                SuccessBehaviorDescriptor.from_record(item)
                for item in raw_descriptors
            ),
        )
        if index.to_record() != record:
            raise ValueError("Quality-diversity index is not exactly rebuildable")
        return index

    @classmethod
    def from_json(cls, text: str) -> QualityDiversityIndex:
        return cls.from_record(strict_json_loads(text))

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "policy": self.policy.to_record(),
            "policy_sha256": self.policy_sha256,
            "event_ledger_set_sha256": self.event_ledger_set_sha256,
            "source_success_set_sha256": self.source_success_set_sha256,
            "descriptors": [item.to_record() for item in self.descriptors],
            "cells": [item.to_record() for item in self.cells],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())

    @property
    def event_ledger_sha256(self) -> str:
        """Expose the exact ledger-set binding expected by derived-view storage."""
        return self.event_ledger_set_sha256

    @property
    def coverage(self) -> float:
        total = math.prod(item.bins for item in self.policy.dimensions)
        return len(self.cells) / total


@dataclass(frozen=True)
class BalancedReplayState:
    schema_version: int
    quality_diversity_index_sha256: str
    stratum_cursor: int
    member_cursors: tuple[tuple[str, int], ...]

    @classmethod
    def initial(cls, index: QualityDiversityIndex) -> BalancedReplayState:
        return cls(
            schema_version=1,
            quality_diversity_index_sha256=index.fingerprint,
            stratum_cursor=0,
            member_cursors=(),
        )

    @classmethod
    def from_record(
        cls,
        value: Any,
        *,
        index: QualityDiversityIndex,
    ) -> BalancedReplayState:
        path = "$balanced_replay_state"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "quality_diversity_index_sha256",
                "stratum_cursor",
                "member_cursors",
            },
            path,
        )
        raw_cursors = _mapping(record["member_cursors"], f"{path}.member_cursors")
        state = cls(
            schema_version=record["schema_version"],
            quality_diversity_index_sha256=_sha256(
                record["quality_diversity_index_sha256"],
                f"{path}.quality_diversity_index_sha256",
            ),
            stratum_cursor=record["stratum_cursor"],
            member_cursors=tuple(sorted(raw_cursors.items())),
        )
        state.validate(index)
        return state

    def validate(self, index: QualityDiversityIndex) -> None:
        if self.schema_version != 1:
            raise ValueError("BalancedReplayState schema_version must be 1")
        if self.quality_diversity_index_sha256 != index.fingerprint:
            raise ValueError("Balanced replay state belongs to another QD index")
        if type(self.stratum_cursor) is not int or self.stratum_cursor < 0:
            raise ValueError("Balanced replay stratum cursor must be non-negative")
        keys = tuple(key for key, _ in self.member_cursors)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("Balanced replay member cursors must be sorted and unique")
        if any(type(value) is not int or value < 0 for _, value in self.member_cursors):
            raise ValueError("Balanced replay member cursors must be non-negative")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "quality_diversity_index_sha256": (
                self.quality_diversity_index_sha256
            ),
            "stratum_cursor": self.stratum_cursor,
            "member_cursors": dict(self.member_cursors),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())


@dataclass(frozen=True)
class BalancedReplayPlan:
    descriptor_sha256s: tuple[str, ...]
    sample_ids: tuple[str, ...]
    next_state: BalancedReplayState


def _balanced_replay_strata(
    index: QualityDiversityIndex,
) -> dict[str, list[SuccessBehaviorDescriptor]]:
    """Return the exact deterministic strata addressed by replay cursors."""
    cell_by_descriptor = {
        descriptor_sha256: cell.cell_id
        for cell in index.cells
        for descriptor_sha256 in cell.member_descriptor_sha256s
    }
    strata: dict[str, list[SuccessBehaviorDescriptor]] = {}
    for descriptor in index.descriptors:
        key = "|".join(
            (
                cell_by_descriptor[descriptor.fingerprint],
                descriptor.initial_state_group_id,
                descriptor.initial_state_group_sha256,
                descriptor.reward_profile_id,
                descriptor.reward_profile_sha256,
            )
        )
        strata.setdefault(key, []).append(descriptor)
    for key in strata:
        strata[key].sort(key=lambda item: item.fingerprint)
    return strata


def rebase_balanced_replay_state(
    *,
    previous_index: QualityDiversityIndex,
    replacement_index: QualityDiversityIndex,
    state: BalancedReplayState,
) -> BalancedReplayState:
    """Carry compatible cursors when an append-only archive rebuilds its QD view.

    Index fingerprints necessarily change when successes are added or
    quarantined.  Reusing a state without an explicit rebase fails closed;
    this operation validates both indexes, drops cursors for strata that no
    longer exist, and preserves progress only for exact stratum identities.
    """
    previous = QualityDiversityIndex.from_record(previous_index.to_record())
    replacement = QualityDiversityIndex.from_record(
        replacement_index.to_record()
    )
    state.validate(previous)
    previous_strata = _balanced_replay_strata(previous)
    replacement_strata = _balanced_replay_strata(replacement)
    unknown = sorted(set(dict(state.member_cursors)).difference(previous_strata))
    if unknown:
        raise ValueError("Balanced replay state contains unknown previous strata")
    carried = tuple(
        (key, cursor)
        for key, cursor in state.member_cursors
        if key in replacement_strata
    )
    rebased = BalancedReplayState(
        schema_version=1,
        quality_diversity_index_sha256=replacement.fingerprint,
        stratum_cursor=state.stratum_cursor,
        member_cursors=carried,
    )
    rebased.validate(replacement)
    return rebased


def plan_balanced_replay(
    *,
    index: QualityDiversityIndex,
    count: int,
    state: BalancedReplayState,
) -> BalancedReplayPlan:
    """Round-robin across strata without repeating a success in one plan."""
    if type(count) is not int or count < 0:
        raise ValueError("Balanced replay count must be non-negative")
    canonical_index = QualityDiversityIndex.from_record(index.to_record())
    canonical_state = BalancedReplayState.from_record(
        state.to_record(),
        index=canonical_index,
    )
    descriptor_by_sha = {
        item.fingerprint: item for item in canonical_index.descriptors
    }
    strata = _balanced_replay_strata(canonical_index)
    ordered_strata = tuple(sorted(strata))
    cursors = dict(canonical_state.member_cursors)
    unknown = sorted(set(cursors).difference(ordered_strata))
    if unknown:
        raise ValueError("Balanced replay state contains unknown strata")
    target_count = min(count, len(canonical_index.descriptors))
    chosen: list[SuccessBehaviorDescriptor] = []
    chosen_sha256s: set[str] = set()
    attempted_strata = 0
    while len(chosen) < target_count:
        stratum = ordered_strata[
            (canonical_state.stratum_cursor + attempted_strata)
            % len(ordered_strata)
        ]
        attempted_strata += 1
        members = strata[stratum]
        cursor = cursors.get(stratum, 0)
        descriptor = None
        cursor_advance = 0
        for member_offset in range(len(members)):
            candidate = members[(cursor + member_offset) % len(members)]
            if candidate.fingerprint not in chosen_sha256s:
                descriptor = candidate
                cursor_advance = member_offset + 1
                break
        if descriptor is None:
            continue
        if descriptor_by_sha[descriptor.fingerprint] != descriptor:
            raise AssertionError("Internal quality-diversity descriptor drift")
        chosen.append(descriptor)
        chosen_sha256s.add(descriptor.fingerprint)
        cursors[stratum] = cursor + cursor_advance
    next_state = BalancedReplayState(
        schema_version=1,
        quality_diversity_index_sha256=canonical_index.fingerprint,
        stratum_cursor=canonical_state.stratum_cursor + attempted_strata,
        member_cursors=tuple(sorted(cursors.items())),
    )
    next_state.validate(canonical_index)
    return BalancedReplayPlan(
        descriptor_sha256s=tuple(item.fingerprint for item in chosen),
        sample_ids=tuple(item.sample_id for item in chosen),
        next_state=next_state,
    )


def build_quality_diversity_index(
    *,
    policy: QualityDiversityPolicy,
    descriptors: Iterable[SuccessBehaviorDescriptor],
) -> QualityDiversityIndex:
    return QualityDiversityIndex(policy=policy, descriptors=descriptors)
