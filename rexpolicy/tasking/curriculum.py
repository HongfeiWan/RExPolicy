"""Sealed-evidence curriculum scheduling for instruction/state task units.

The scheduler deliberately has no reward input.  A curriculum unit is the
combination of one instruction and one committed initial-state recipe, and all
performance estimates come from a K=1 sealed evaluation report for that exact
unit and policy checkpoint.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import canonical_fingerprint, canonical_json, strict_json_loads

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_INSTRUCTION_CHARS = 512

INSUFFICIENT_EVIDENCE = "insufficient_evidence"
SAFETY_EXCLUDED = "safety_excluded"
PREREQUISITE_BLOCKED = "prerequisite_blocked"
MASTERED = "mastered"
HARD = "hard"
FRONTIER = "frontier"


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _keys(record: dict[str, Any], expected: set[str], path: str) -> None:
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


def _non_negative_int(value: Any, path: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _positive_int(value: Any, path: str) -> int:
    result = _non_negative_int(value, path)
    if result == 0:
        raise ValueError(f"{path} must be a positive integer")
    return result


def _probability(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{path} must be a finite probability")
    return result


def _positive_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{path} must be positive and finite")
    return result


def _instruction(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be non-empty text")
    if value != value.strip() or len(value) > _MAX_INSTRUCTION_CHARS:
        raise ValueError(f"{path} must be trimmed and at most 512 characters")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{path} must use NFC Unicode normalization")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError(f"{path} cannot contain control characters")
    return value


@dataclass(frozen=True)
class CurriculumTaskUnit:
    """One schedulable instruction plus one exact initial-state recipe."""

    schema_version: int
    unit_id: str
    task_contract_id: str
    task_oracle_sha256: str
    instruction: str
    initial_state_id: str
    initial_state_sha256: str
    prerequisite_unit_ids: tuple[str, ...]

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported curriculum task-unit schema")
        _identifier(self.unit_id, "unit_id")
        _identifier(self.task_contract_id, "task_contract_id")
        _identifier(self.initial_state_id, "initial_state_id")
        _sha256(self.task_oracle_sha256, "task_oracle_sha256")
        _sha256(self.initial_state_sha256, "initial_state_sha256")
        _instruction(self.instruction, "instruction")
        prerequisites = tuple(self.prerequisite_unit_ids)
        if prerequisites != tuple(sorted(set(prerequisites))):
            raise ValueError("prerequisite_unit_ids must be unique and sorted")
        for prerequisite in prerequisites:
            _identifier(prerequisite, "prerequisite_unit_ids[]")
        if self.unit_id in prerequisites:
            raise ValueError("A curriculum unit cannot require itself")

    @classmethod
    def from_record(cls, value: Any) -> CurriculumTaskUnit:
        record = _mapping(value, "CurriculumTaskUnit")
        _keys(
            record,
            {
                "schema_version",
                "unit_id",
                "task_contract_id",
                "task_oracle_sha256",
                "instruction",
                "initial_state_id",
                "initial_state_sha256",
                "prerequisite_unit_ids",
            },
            "CurriculumTaskUnit",
        )
        raw_prerequisites = record["prerequisite_unit_ids"]
        if not isinstance(raw_prerequisites, list):
            raise ValueError("prerequisite_unit_ids must be an array")
        unit = cls(
            schema_version=record["schema_version"],
            unit_id=record["unit_id"],
            task_contract_id=record["task_contract_id"],
            task_oracle_sha256=record["task_oracle_sha256"],
            instruction=record["instruction"],
            initial_state_id=record["initial_state_id"],
            initial_state_sha256=record["initial_state_sha256"],
            prerequisite_unit_ids=tuple(raw_prerequisites),
        )
        unit.validate()
        return unit

    @classmethod
    def from_json(cls, text: str) -> CurriculumTaskUnit:
        return cls.from_record(strict_json_loads(text))

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "unit_id": self.unit_id,
            "task_contract_id": self.task_contract_id,
            "task_oracle_sha256": self.task_oracle_sha256,
            "instruction": self.instruction,
            "initial_state_id": self.initial_state_id,
            "initial_state_sha256": self.initial_state_sha256,
            "prerequisite_unit_ids": list(self.prerequisite_unit_ids),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class SealedEvaluationEvidence:
    """Reward-free K=1 outcome counts from one committed audit suite."""

    schema_version: int
    evidence_id: str
    unit_id: str
    unit_sha256: str
    candidate_policy_sha256: str
    audit_suite_sha256: str
    evaluation_report_sha256: str
    candidate_count: int
    episode_count: int
    success_count: int
    safety_failure_count: int
    invalid_action_count: int

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported sealed curriculum-evidence schema")
        _identifier(self.evidence_id, "evidence_id")
        _identifier(self.unit_id, "unit_id")
        for name, value in (
            ("unit_sha256", self.unit_sha256),
            ("candidate_policy_sha256", self.candidate_policy_sha256),
            ("audit_suite_sha256", self.audit_suite_sha256),
            ("evaluation_report_sha256", self.evaluation_report_sha256),
        ):
            _sha256(value, name)
        if self.candidate_count != 1:
            raise ValueError("Curriculum evidence must use K=1 evaluation")
        episodes = _positive_int(self.episode_count, "episode_count")
        for name, value in (
            ("success_count", self.success_count),
            ("safety_failure_count", self.safety_failure_count),
            ("invalid_action_count", self.invalid_action_count),
        ):
            count = _non_negative_int(value, name)
            if count > episodes:
                raise ValueError(f"{name} cannot exceed episode_count")

    @classmethod
    def from_record(cls, value: Any) -> SealedEvaluationEvidence:
        record = _mapping(value, "SealedEvaluationEvidence")
        expected = {
            "schema_version",
            "evidence_id",
            "unit_id",
            "unit_sha256",
            "candidate_policy_sha256",
            "audit_suite_sha256",
            "evaluation_report_sha256",
            "candidate_count",
            "episode_count",
            "success_count",
            "safety_failure_count",
            "invalid_action_count",
        }
        _keys(record, expected, "SealedEvaluationEvidence")
        evidence = cls(**record)
        evidence.validate()
        return evidence

    @classmethod
    def from_json(cls, text: str) -> SealedEvaluationEvidence:
        return cls.from_record(strict_json_loads(text))

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            "unit_id": self.unit_id,
            "unit_sha256": self.unit_sha256,
            "candidate_policy_sha256": self.candidate_policy_sha256,
            "audit_suite_sha256": self.audit_suite_sha256,
            "evaluation_report_sha256": self.evaluation_report_sha256,
            "candidate_count": self.candidate_count,
            "episode_count": self.episode_count,
            "success_count": self.success_count,
            "safety_failure_count": self.safety_failure_count,
            "invalid_action_count": self.invalid_action_count,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class CurriculumPolicy:
    """Trusted evidence, frontier, safety, and replay policy."""

    schema_version: int
    policy_id: str
    target_success_probability: float
    frontier_bandwidth: float
    confidence_z: float
    min_evaluation_episodes: int
    hard_success_upper_bound: float
    mastered_success_lower_bound: float
    max_safety_failures: int
    max_invalid_actions: int
    min_mastered_replay_slots: int
    min_hard_exposure_slots: int

    @classmethod
    def production_v1(cls) -> CurriculumPolicy:
        return cls(
            schema_version=1,
            policy_id="sealed_frontier/v1",
            target_success_probability=0.5,
            frontier_bandwidth=0.2,
            confidence_z=1.959963984540054,
            min_evaluation_episodes=64,
            hard_success_upper_bound=0.2,
            mastered_success_lower_bound=0.8,
            max_safety_failures=0,
            max_invalid_actions=0,
            min_mastered_replay_slots=1,
            min_hard_exposure_slots=1,
        )

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported curriculum policy schema")
        _identifier(self.policy_id, "policy_id")
        target = _probability(
            self.target_success_probability,
            "target_success_probability",
        )
        if target != 0.5:
            raise ValueError("Curriculum v1 frontier target must be 0.5")
        bandwidth = _positive_float(self.frontier_bandwidth, "frontier_bandwidth")
        if bandwidth > 1.0:
            raise ValueError("frontier_bandwidth must not exceed 1")
        _positive_float(self.confidence_z, "confidence_z")
        _positive_int(self.min_evaluation_episodes, "min_evaluation_episodes")
        hard = _probability(
            self.hard_success_upper_bound,
            "hard_success_upper_bound",
        )
        mastered = _probability(
            self.mastered_success_lower_bound,
            "mastered_success_lower_bound",
        )
        if not hard < target < mastered:
            raise ValueError(
                "hard, target, and mastered probabilities must be strictly ordered"
            )
        _non_negative_int(self.max_safety_failures, "max_safety_failures")
        _non_negative_int(self.max_invalid_actions, "max_invalid_actions")
        _non_negative_int(
            self.min_mastered_replay_slots,
            "min_mastered_replay_slots",
        )
        _non_negative_int(
            self.min_hard_exposure_slots,
            "min_hard_exposure_slots",
        )

    @classmethod
    def from_record(cls, value: Any) -> CurriculumPolicy:
        record = _mapping(value, "CurriculumPolicy")
        expected = {
            "schema_version",
            "policy_id",
            "target_success_probability",
            "frontier_bandwidth",
            "confidence_z",
            "min_evaluation_episodes",
            "hard_success_upper_bound",
            "mastered_success_lower_bound",
            "max_safety_failures",
            "max_invalid_actions",
            "min_mastered_replay_slots",
            "min_hard_exposure_slots",
        }
        _keys(record, expected, "CurriculumPolicy")
        policy = cls(**record)
        policy.validate()
        return policy

    @classmethod
    def from_json(cls, text: str) -> CurriculumPolicy:
        return cls.from_record(strict_json_loads(text))

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "target_success_probability": self.target_success_probability,
            "frontier_bandwidth": self.frontier_bandwidth,
            "confidence_z": self.confidence_z,
            "min_evaluation_episodes": self.min_evaluation_episodes,
            "hard_success_upper_bound": self.hard_success_upper_bound,
            "mastered_success_lower_bound": self.mastered_success_lower_bound,
            "max_safety_failures": self.max_safety_failures,
            "max_invalid_actions": self.max_invalid_actions,
            "min_mastered_replay_slots": self.min_mastered_replay_slots,
            "min_hard_exposure_slots": self.min_hard_exposure_slots,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a two-sided Wilson interval for a Bernoulli success rate."""
    if type(trials) is not int or type(successes) is not int:
        raise ValueError("Wilson successes and trials must be integers")
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("Invalid successes/trials for Wilson interval")
    z_value = _positive_float(z, "z")
    proportion = successes / trials
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / trials
    centre = proportion + z_squared / (2.0 * trials)
    margin = z_value * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z_squared / (4.0 * trials * trials)
    )
    return (
        max(0.0, (centre - margin) / denominator),
        min(1.0, (centre + margin) / denominator),
    )


def frontier_weight(
    success_probability: float,
    *,
    target_success_probability: float = 0.5,
    bandwidth: float = 0.2,
) -> float:
    """Return symmetric Gaussian mass, maximized at the learning frontier.

    Unlike a monotonic difficulty score, this gives equal mass to points the
    same distance above and below the target, and its unique maximum is the
    target itself.
    """
    probability = _probability(success_probability, "success_probability")
    target = _probability(
        target_success_probability,
        "target_success_probability",
    )
    width = _positive_float(bandwidth, "bandwidth")
    distance = (probability - target) / width
    return math.exp(-0.5 * distance * distance)


@dataclass(frozen=True)
class CurriculumAssessment:
    """One immutable unit estimate and its scheduling gate result."""

    unit_id: str
    unit_sha256: str
    evidence_sha256: str | None
    episode_count: int
    success_count: int
    safety_failure_count: int
    invalid_action_count: int
    success_probability: float | None
    confidence_lower_bound: float | None
    confidence_upper_bound: float | None
    status: str
    missing_prerequisite_ids: tuple[str, ...]
    frontier_mass: float
    confidence_mass: float
    selection_weight: float

    @property
    def eligible(self) -> bool:
        return self.status in {MASTERED, HARD, FRONTIER}

    def to_record(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "unit_sha256": self.unit_sha256,
            "evidence_sha256": self.evidence_sha256,
            "episode_count": self.episode_count,
            "success_count": self.success_count,
            "safety_failure_count": self.safety_failure_count,
            "invalid_action_count": self.invalid_action_count,
            "success_probability": self.success_probability,
            "confidence_lower_bound": self.confidence_lower_bound,
            "confidence_upper_bound": self.confidence_upper_bound,
            "status": self.status,
            "missing_prerequisite_ids": list(self.missing_prerequisite_ids),
            "frontier_mass": self.frontier_mass,
            "confidence_mass": self.confidence_mass,
            "selection_weight": self.selection_weight,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


def _validate_task_graph(units: tuple[CurriculumTaskUnit, ...]) -> None:
    by_id = {unit.unit_id: unit for unit in units}
    if len(by_id) != len(units):
        raise ValueError("Curriculum unit IDs must be unique")
    identities = {
        (unit.instruction, unit.initial_state_id, unit.initial_state_sha256)
        for unit in units
    }
    if len(identities) != len(units):
        raise ValueError("Instruction/initial-state curriculum units must be unique")
    for unit in units:
        unknown = sorted(set(unit.prerequisite_unit_ids).difference(by_id))
        if unknown:
            raise ValueError(
                f"Unit {unit.unit_id!r} has unknown prerequisites: "
                + ", ".join(unknown)
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(unit_id: str) -> None:
        if unit_id in visiting:
            raise ValueError("Curriculum prerequisite graph contains a cycle")
        if unit_id in visited:
            return
        visiting.add(unit_id)
        for prerequisite in by_id[unit_id].prerequisite_unit_ids:
            visit(prerequisite)
        visiting.remove(unit_id)
        visited.add(unit_id)

    for unit_id in sorted(by_id):
        visit(unit_id)


def assess_curriculum(
    *,
    units: Iterable[CurriculumTaskUnit],
    evidence: Iterable[SealedEvaluationEvidence],
    candidate_policy_sha256: str,
    policy: CurriculumPolicy,
) -> tuple[CurriculumAssessment, ...]:
    """Classify exact task units from sealed outcomes and trusted gates."""
    _sha256(candidate_policy_sha256, "candidate_policy_sha256")
    policy.validate()
    ordered_units = tuple(sorted(units, key=lambda item: item.unit_id))
    if not ordered_units:
        raise ValueError("Curriculum requires at least one task unit")
    for unit in ordered_units:
        unit.validate()
    _validate_task_graph(ordered_units)
    by_id = {unit.unit_id: unit for unit in ordered_units}

    by_unit: dict[str, SealedEvaluationEvidence] = {}
    evidence_ids: set[str] = set()
    report_hashes: set[str] = set()
    for item in evidence:
        item.validate()
        if item.evidence_id in evidence_ids:
            raise ValueError("Curriculum evidence IDs must be unique")
        if item.evaluation_report_sha256 in report_hashes:
            raise ValueError("Sealed evaluation reports cannot be reused")
        evidence_ids.add(item.evidence_id)
        report_hashes.add(item.evaluation_report_sha256)
        unit = by_id.get(item.unit_id)
        if unit is None:
            raise ValueError(f"Evidence references unknown unit {item.unit_id!r}")
        if item.unit_sha256 != unit.fingerprint:
            raise ValueError("Curriculum evidence unit fingerprint mismatch")
        if item.candidate_policy_sha256 != candidate_policy_sha256:
            raise ValueError("Curriculum evidence candidate policy mismatch")
        if item.unit_id in by_unit:
            raise ValueError("Each curriculum unit must have one sealed report")
        by_unit[item.unit_id] = item

    base_status: dict[str, str] = {}
    estimates: dict[str, tuple[Any, ...]] = {}
    for unit in ordered_units:
        item = by_unit.get(unit.unit_id)
        if item is None:
            estimates[unit.unit_id] = (None, 0, 0, 0, 0, None, None, None, 0.0, 0.0)
            base_status[unit.unit_id] = INSUFFICIENT_EVIDENCE
            continue
        probability = item.success_count / item.episode_count
        lower, upper = wilson_interval(
            item.success_count,
            item.episode_count,
            z=policy.confidence_z,
        )
        mass = frontier_weight(
            probability,
            target_success_probability=policy.target_success_probability,
            bandwidth=policy.frontier_bandwidth,
        )
        confidence_mass = max(0.0, 1.0 - (upper - lower))
        selection_weight = max(1e-12, mass * confidence_mass)
        estimates[unit.unit_id] = (
            item,
            item.episode_count,
            item.success_count,
            item.safety_failure_count,
            item.invalid_action_count,
            probability,
            lower,
            upper,
            mass,
            confidence_mass,
            selection_weight,
        )
        if (
            item.safety_failure_count > policy.max_safety_failures
            or item.invalid_action_count > policy.max_invalid_actions
        ):
            base_status[unit.unit_id] = SAFETY_EXCLUDED
        elif item.episode_count < policy.min_evaluation_episodes:
            base_status[unit.unit_id] = INSUFFICIENT_EVIDENCE
        elif lower >= policy.mastered_success_lower_bound:
            base_status[unit.unit_id] = MASTERED
        elif upper <= policy.hard_success_upper_bound:
            base_status[unit.unit_id] = HARD
        else:
            base_status[unit.unit_id] = FRONTIER

    qualified_mastered: set[str] = set()
    changed = True
    while changed:
        changed = False
        for unit in ordered_units:
            if (
                base_status[unit.unit_id] == MASTERED
                and unit.unit_id not in qualified_mastered
                and set(unit.prerequisite_unit_ids).issubset(qualified_mastered)
            ):
                qualified_mastered.add(unit.unit_id)
                changed = True

    assessments = []
    for unit in ordered_units:
        values = estimates[unit.unit_id]
        if values[0] is None:
            (
                item,
                episode_count,
                success_count,
                safety_count,
                invalid_count,
                probability,
                lower,
                upper,
                mass,
                confidence_mass,
            ) = values
            selection_weight = 0.0
        else:
            (
                item,
                episode_count,
                success_count,
                safety_count,
                invalid_count,
                probability,
                lower,
                upper,
                mass,
                confidence_mass,
                selection_weight,
            ) = values
        status = base_status[unit.unit_id]
        missing = tuple(
            prerequisite
            for prerequisite in unit.prerequisite_unit_ids
            if prerequisite not in qualified_mastered
        )
        if status not in {SAFETY_EXCLUDED, INSUFFICIENT_EVIDENCE} and missing:
            status = PREREQUISITE_BLOCKED
            selection_weight = 0.0
        if status in {SAFETY_EXCLUDED, INSUFFICIENT_EVIDENCE}:
            selection_weight = 0.0
        assessments.append(
            CurriculumAssessment(
                unit_id=unit.unit_id,
                unit_sha256=unit.fingerprint,
                evidence_sha256=None if item is None else item.fingerprint,
                episode_count=episode_count,
                success_count=success_count,
                safety_failure_count=safety_count,
                invalid_action_count=invalid_count,
                success_probability=probability,
                confidence_lower_bound=lower,
                confidence_upper_bound=upper,
                status=status,
                missing_prerequisite_ids=missing,
                frontier_mass=mass,
                confidence_mass=confidence_mass,
                selection_weight=selection_weight,
            )
        )
    return tuple(assessments)


def rank_curriculum(
    assessments: Iterable[CurriculumAssessment],
) -> tuple[CurriculumAssessment, ...]:
    """Return a deterministic frontier-first ranking of eligible units."""
    return tuple(
        sorted(
            (item for item in assessments if item.eligible),
            key=lambda item: (-item.selection_weight, item.unit_sha256),
        )
    )


@dataclass(frozen=True)
class CurriculumScheduleSlot:
    slot: int
    lane: str
    unit_id: str
    unit_sha256: str
    assessment_sha256: str

    def to_record(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "lane": self.lane,
            "unit_id": self.unit_id,
            "unit_sha256": self.unit_sha256,
            "assessment_sha256": self.assessment_sha256,
        }


@dataclass(frozen=True)
class CurriculumSchedule:
    schema_version: int
    candidate_policy_sha256: str
    curriculum_policy_sha256: str
    task_set_sha256: str
    evidence_set_sha256: str
    selection_seed: int | None
    total_slots: int
    slots: tuple[CurriculumScheduleSlot, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_policy_sha256": self.candidate_policy_sha256,
            "curriculum_policy_sha256": self.curriculum_policy_sha256,
            "task_set_sha256": self.task_set_sha256,
            "evidence_set_sha256": self.evidence_set_sha256,
            "selection_seed": self.selection_seed,
            "total_slots": self.total_slots,
            "slots": [slot.to_record() for slot in self.slots],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


def _seeded_uniform(
    *,
    seed: int,
    slot: int,
    lane: str,
    pool: tuple[CurriculumAssessment, ...],
) -> float:
    payload = canonical_json(
        {
            "domain": "rexpolicy.curriculum.selection/v1",
            "seed": seed,
            "slot": slot,
            "lane": lane,
            "pool": [item.fingerprint for item in pool],
        }
    ).encode("ascii")
    numerator = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return numerator / (1 << 64)


def _choose(
    pool: tuple[CurriculumAssessment, ...],
    *,
    seed: int | None,
    slot: int,
    lane: str,
) -> CurriculumAssessment:
    ranked = rank_curriculum(pool)
    if not ranked:
        raise ValueError(f"Curriculum lane {lane!r} has no eligible units")
    if seed is None:
        return ranked[slot % len(ranked)]
    draw = _seeded_uniform(seed=seed, slot=slot, lane=lane, pool=ranked)
    total = sum(item.selection_weight for item in ranked)
    threshold = draw * total
    cumulative = 0.0
    for item in ranked:
        cumulative += item.selection_weight
        if threshold < cumulative:
            return item
    return ranked[-1]


def build_curriculum_schedule(
    *,
    units: Iterable[CurriculumTaskUnit],
    evidence: Iterable[SealedEvaluationEvidence],
    candidate_policy_sha256: str,
    policy: CurriculumPolicy,
    total_slots: int,
    seed: int | None = None,
) -> CurriculumSchedule:
    """Build a replay-safe deterministic schedule with explicit lane floors."""
    count = _positive_int(total_slots, "total_slots")
    if seed is not None:
        _non_negative_int(seed, "seed")
    unit_tuple = tuple(units)
    evidence_tuple = tuple(evidence)
    assessments = assess_curriculum(
        units=unit_tuple,
        evidence=evidence_tuple,
        candidate_policy_sha256=candidate_policy_sha256,
        policy=policy,
    )
    mastered = tuple(item for item in assessments if item.status == MASTERED)
    hard = tuple(item for item in assessments if item.status == HARD)
    eligible = tuple(item for item in assessments if item.eligible)
    if policy.min_mastered_replay_slots and not mastered:
        raise ValueError("Mastered replay minimum cannot be satisfied")
    if policy.min_hard_exposure_slots and not hard:
        raise ValueError("Hard exposure minimum cannot be satisfied")
    reserved = (
        policy.min_mastered_replay_slots
        + policy.min_hard_exposure_slots
    )
    if count < reserved:
        raise ValueError("total_slots is smaller than the required lane minimums")
    if not eligible:
        raise ValueError("Curriculum has no eligible units")

    lanes = (
        ["mastered_replay"] * policy.min_mastered_replay_slots
        + ["hard_exposure"] * policy.min_hard_exposure_slots
        + ["frontier_mix"] * (count - reserved)
    )
    pools = {
        "mastered_replay": mastered,
        "hard_exposure": hard,
        "frontier_mix": eligible,
    }
    slots = []
    lane_offsets = {lane: 0 for lane in pools}
    for index, lane in enumerate(lanes):
        selected = _choose(
            pools[lane],
            seed=seed,
            slot=lane_offsets[lane],
            lane=lane,
        )
        lane_offsets[lane] += 1
        slots.append(
            CurriculumScheduleSlot(
                slot=index,
                lane=lane,
                unit_id=selected.unit_id,
                unit_sha256=selected.unit_sha256,
                assessment_sha256=selected.fingerprint,
            )
        )
    ordered_units = sorted(unit_tuple, key=lambda item: item.unit_id)
    ordered_evidence = sorted(evidence_tuple, key=lambda item: item.evidence_id)
    return CurriculumSchedule(
        schema_version=1,
        candidate_policy_sha256=candidate_policy_sha256,
        curriculum_policy_sha256=policy.fingerprint,
        task_set_sha256=canonical_fingerprint(
            [unit.to_record() for unit in ordered_units]
        ),
        evidence_set_sha256=canonical_fingerprint(
            [item.to_record() for item in ordered_evidence]
        ),
        selection_seed=seed,
        total_slots=count,
        slots=tuple(slots),
    )
