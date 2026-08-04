"""Declassified curriculum briefs for automatic TaskSpec authoring."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .authoring import AuthoringIntent
from .canonical import canonical_fingerprint
from .contract import CompiledTaskContract
from .curriculum import (
    FRONTIER,
    HARD,
    PREREQUISITE_BLOCKED,
    CurriculumSnapshot,
)
from .model import TaskSpecV2

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PUBLIC_PATH = re.compile(r"^\$(?:\.[a-z][a-z0-9_]*)*$")
_PUBLIC_GAP_STATUSES = frozenset(("frontier", "hard", "prerequisite_blocked"))
_MAX_OBJECTIVE_CHARS = 2_048
_MAX_CONSTRAINT_CHARS = 512
_MAX_CONSTRAINTS = 64


class CandidateBriefViolation(ValueError):
    """Candidate content exceeds the already-authorized public brief."""


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} must be a versioned identifier")
    return value


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256")
    return value


def _public_text(value: Any, path: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be non-empty public text")
    if value != value.strip() or len(value) > maximum:
        raise ValueError(f"{path} must be trimmed and at most {maximum} characters")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{path} must use NFC Unicode normalization")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError(f"{path} cannot contain control characters")
    return value


@dataclass(frozen=True)
class PublicCurriculumGap:
    """Declassified curriculum signal with no counts, rates, or audit data."""

    unit_id: str
    assessment_fingerprint: str
    status: str
    gap_code: str

    @classmethod
    def from_record(cls, value: Any, path: str) -> PublicCurriculumGap:
        if not isinstance(value, dict) or set(value) != {
            "unit_id",
            "assessment_fingerprint",
            "status",
            "gap_code",
        }:
            raise ValueError(f"{path} must be one strict public gap object")
        status = _identifier(value["status"], f"{path}.status")
        if status not in _PUBLIC_GAP_STATUSES:
            raise ValueError(f"{path}.status is not a public authoring gap state")
        return cls(
            unit_id=_identifier(value["unit_id"], f"{path}.unit_id"),
            assessment_fingerprint=_sha256(
                value["assessment_fingerprint"],
                f"{path}.assessment_fingerprint",
            ),
            status=status,
            gap_code=_identifier(value["gap_code"], f"{path}.gap_code"),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "assessment_fingerprint": self.assessment_fingerprint,
            "status": self.status,
            "gap_code": self.gap_code,
        }


@dataclass(frozen=True)
class PublicAuthoringBrief:
    """Hash-bound objective containing only declassified curriculum context."""

    schema_version: int
    brief_id: str
    task_id: str
    family_id: str
    objective: str
    source_curriculum_snapshot_fingerprint: str
    gap_targets: tuple[PublicCurriculumGap, ...]
    parent_task_spec_fingerprint: str | None
    parent_task_oracle_fingerprint: str | None
    allowed_change_paths: tuple[str, ...]
    required_curriculum_prerequisite_ids: tuple[str, ...]
    public_constraints: tuple[str, ...]

    @classmethod
    def from_record(cls, value: Any) -> PublicAuthoringBrief:
        if not isinstance(value, dict):
            raise ValueError("$authoring_brief must be an object")
        expected = {
            "schema_version",
            "brief_id",
            "task_id",
            "family_id",
            "objective",
            "source_curriculum_snapshot_fingerprint",
            "gap_targets",
            "parent_task_spec_fingerprint",
            "parent_task_oracle_fingerprint",
            "allowed_change_paths",
            "required_curriculum_prerequisite_ids",
            "public_constraints",
        }
        missing = sorted(expected.difference(value))
        unknown = sorted(set(value).difference(expected))
        if missing:
            raise ValueError(
                "$authoring_brief is missing fields: " + ", ".join(missing)
            )
        if unknown:
            raise ValueError(
                "$authoring_brief has unknown fields: " + ", ".join(unknown)
            )
        if value["schema_version"] != 1:
            raise ValueError("PublicAuthoringBrief schema_version must be 1")
        raw_constraints = value["public_constraints"]
        if (
            not isinstance(raw_constraints, list)
            or not raw_constraints
            or len(raw_constraints) > _MAX_CONSTRAINTS
        ):
            raise ValueError(
                "$authoring_brief.public_constraints must be a bounded non-empty array"
            )
        constraints = tuple(
            _public_text(
                item,
                f"$authoring_brief.public_constraints[{index}]",
                maximum=_MAX_CONSTRAINT_CHARS,
            )
            for index, item in enumerate(raw_constraints)
        )
        if constraints != tuple(sorted(set(constraints))):
            raise ValueError("Public authoring constraints must be sorted and unique")
        raw_gaps = value["gap_targets"]
        if not isinstance(raw_gaps, list) or not raw_gaps:
            raise ValueError("$authoring_brief.gap_targets must be non-empty")
        gaps = tuple(
            PublicCurriculumGap.from_record(
                item,
                f"$authoring_brief.gap_targets[{index}]",
            )
            for index, item in enumerate(raw_gaps)
        )
        gap_keys = tuple((gap.unit_id, gap.assessment_fingerprint) for gap in gaps)
        if gap_keys != tuple(sorted(set(gap_keys))):
            raise ValueError("Public curriculum gaps must be sorted and unique")
        raw_paths = value["allowed_change_paths"]
        if not isinstance(raw_paths, list) or not raw_paths:
            raise ValueError("$authoring_brief.allowed_change_paths must be non-empty")
        paths = tuple(raw_paths)
        if (
            any(
                not isinstance(path, str) or _PUBLIC_PATH.fullmatch(path) is None
                for path in paths
            )
            or paths != tuple(sorted(set(paths)))
        ):
            raise ValueError(
                "Allowed change paths must be normalized, sorted, and unique"
            )
        raw_prerequisites = value["required_curriculum_prerequisite_ids"]
        if not isinstance(raw_prerequisites, list):
            raise ValueError(
                "$authoring_brief.required_curriculum_prerequisite_ids must be an array"
            )
        prerequisites = tuple(
            _identifier(
                item,
                f"$authoring_brief.required_curriculum_prerequisite_ids[{index}]",
            )
            for index, item in enumerate(raw_prerequisites)
        )
        if prerequisites != tuple(sorted(set(prerequisites))):
            raise ValueError(
                "Required curriculum prerequisites must be sorted and unique"
            )
        raw_parent_spec = value["parent_task_spec_fingerprint"]
        raw_parent_oracle = value["parent_task_oracle_fingerprint"]
        if (raw_parent_spec is None) != (raw_parent_oracle is None):
            raise ValueError("Parent task spec and oracle fingerprints must be paired")
        parent_spec = (
            None
            if raw_parent_spec is None
            else _sha256(
                raw_parent_spec,
                "$authoring_brief.parent_task_spec_fingerprint",
            )
        )
        parent_oracle = (
            None
            if raw_parent_oracle is None
            else _sha256(
                raw_parent_oracle,
                "$authoring_brief.parent_task_oracle_fingerprint",
            )
        )
        if parent_spec is None and paths != ("$",):
            raise ValueError("Parentless authoring briefs must authorize the root path")
        return cls(
            schema_version=1,
            brief_id=_identifier(value["brief_id"], "$authoring_brief.brief_id"),
            task_id=_identifier(value["task_id"], "$authoring_brief.task_id"),
            family_id=_identifier(value["family_id"], "$authoring_brief.family_id"),
            objective=_public_text(
                value["objective"],
                "$authoring_brief.objective",
                maximum=_MAX_OBJECTIVE_CHARS,
            ),
            source_curriculum_snapshot_fingerprint=_sha256(
                value["source_curriculum_snapshot_fingerprint"],
                "$authoring_brief.source_curriculum_snapshot_fingerprint",
            ),
            gap_targets=gaps,
            parent_task_spec_fingerprint=parent_spec,
            parent_task_oracle_fingerprint=parent_oracle,
            allowed_change_paths=paths,
            required_curriculum_prerequisite_ids=prerequisites,
            public_constraints=constraints,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "brief_id": self.brief_id,
            "task_id": self.task_id,
            "family_id": self.family_id,
            "objective": self.objective,
            "source_curriculum_snapshot_fingerprint": (
                self.source_curriculum_snapshot_fingerprint
            ),
            "gap_targets": [gap.to_record() for gap in self.gap_targets],
            "parent_task_spec_fingerprint": self.parent_task_spec_fingerprint,
            "parent_task_oracle_fingerprint": self.parent_task_oracle_fingerprint,
            "allowed_change_paths": list(self.allowed_change_paths),
            "required_curriculum_prerequisite_ids": list(
                self.required_curriculum_prerequisite_ids
            ),
            "public_constraints": list(self.public_constraints),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())

    def validate_against(self, intent: AuthoringIntent) -> None:
        canonical = PublicAuthoringBrief.from_record(self.to_record())
        if intent.schema_version != 2:
            raise ValueError(
                "Automatic authoring requires a brief-bound AuthoringIntent v2"
            )
        if canonical.task_id != intent.task_id:
            raise ValueError("Public authoring brief task identity mismatch")
        if canonical.family_id != intent.family_id:
            raise ValueError("Public authoring brief family identity mismatch")
        if canonical.brief_id != intent.authoring_brief_id:
            raise ValueError("Public authoring brief identity binding drifted")
        if canonical.fingerprint != intent.authoring_brief_fingerprint:
            raise ValueError("Public authoring brief fingerprint binding drifted")


def _changed_paths(left: Any, right: Any, path: str = "$") -> tuple[str, ...]:
    if type(left) is not type(right):
        return (path,)
    if isinstance(left, dict):
        if set(left) != set(right):
            return (path,)
        changed = []
        for key in sorted(left):
            changed.extend(_changed_paths(left[key], right[key], f"{path}.{key}"))
        return tuple(changed)
    if isinstance(left, list):
        return () if left == right else (path,)
    return () if left == right else (path,)


def _path_is_allowed(path: str, allowed_paths: tuple[str, ...]) -> bool:
    return any(
        allowed == "$" or path == allowed or path.startswith(f"{allowed}.")
        for allowed in allowed_paths
    )


def validate_candidate_against_brief(
    candidate: TaskSpecV2,
    *,
    brief: PublicAuthoringBrief,
    parent_task: TaskSpecV2 | None,
    parent_contract: CompiledTaskContract | None,
) -> None:
    """Enforce machine-checkable brief constraints without interpreting prose."""
    canonical_brief = PublicAuthoringBrief.from_record(brief.to_record())
    canonical_candidate = TaskSpecV2.from_record(candidate.to_record())
    if tuple(canonical_candidate.curriculum_prerequisite_ids) != (
        canonical_brief.required_curriculum_prerequisite_ids
    ):
        raise CandidateBriefViolation("Candidate prerequisites exceed the brief")
    expects_parent = canonical_brief.parent_task_spec_fingerprint is not None
    if expects_parent != (parent_task is not None and parent_contract is not None):
        raise ValueError("Trusted authoring parent artifacts are incomplete")
    if not expects_parent:
        return
    assert parent_task is not None and parent_contract is not None
    canonical_parent = TaskSpecV2.from_record(parent_task.to_record())
    if (
        canonical_parent.fingerprint
        != canonical_brief.parent_task_spec_fingerprint
        or parent_contract.task_spec_fingerprint != canonical_parent.fingerprint
        or parent_contract.oracle_fingerprint
        != canonical_brief.parent_task_oracle_fingerprint
    ):
        raise ValueError("Trusted authoring parent binding drifted")
    changed = _changed_paths(
        canonical_parent.to_record(),
        canonical_candidate.to_record(),
    )
    if any(
        not _path_is_allowed(path, canonical_brief.allowed_change_paths)
        for path in changed
    ):
        raise CandidateBriefViolation("Candidate changed a field outside the brief")


def derive_public_authoring_brief(
    *,
    brief_id: str,
    task_id: str,
    family_id: str,
    objective: str,
    curriculum_snapshot: CurriculumSnapshot,
    gap_codes_by_unit: Mapping[str, str],
    parent_task: TaskSpecV2 | None,
    parent_contract: CompiledTaskContract | None,
    allowed_change_paths: Iterable[str],
    required_curriculum_prerequisite_ids: Iterable[str],
    public_constraints: Iterable[str],
) -> PublicAuthoringBrief:
    """Declassify curriculum gaps without copying quantitative evidence."""
    if not isinstance(curriculum_snapshot, CurriculumSnapshot):
        raise ValueError("curriculum_snapshot must be one exact CurriculumSnapshot")
    snapshot = CurriculumSnapshot.from_record(curriculum_snapshot.to_record())
    allowed_statuses = {HARD, FRONTIER, PREREQUISITE_BLOCKED}
    selected = tuple(
        sorted(
            (
                item
                for item in snapshot.assessments
                if item.status in allowed_statuses
            ),
            key=lambda item: (item.unit_id, item.fingerprint),
        )
    )
    if not selected:
        raise ValueError("No authorable public curriculum gap is available")
    unit_ids = tuple(item.unit_id for item in selected)
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("Public authoring gaps must reference unique units")
    if set(gap_codes_by_unit) != set(unit_ids):
        raise ValueError("Gap codes must exactly cover the public curriculum gaps")
    if (parent_task is None) != (parent_contract is None):
        raise ValueError("Parent TaskSpec and contract must be paired")
    if parent_task is None:
        parent_task_fingerprint = None
        parent_oracle_fingerprint = None
    else:
        assert parent_contract is not None
        if parent_contract.task_spec_fingerprint != parent_task.fingerprint:
            raise ValueError("Parent task contract binding drifted")
        parent_task_fingerprint = parent_task.fingerprint
        parent_oracle_fingerprint = parent_contract.oracle_fingerprint
    record = {
        "schema_version": 1,
        "brief_id": brief_id,
        "task_id": task_id,
        "family_id": family_id,
        "objective": objective,
        "source_curriculum_snapshot_fingerprint": (
            snapshot.fingerprint
        ),
        "gap_targets": [
            {
                "unit_id": item.unit_id,
                "assessment_fingerprint": item.fingerprint,
                "status": item.status,
                "gap_code": gap_codes_by_unit[item.unit_id],
            }
            for item in selected
        ],
        "parent_task_spec_fingerprint": parent_task_fingerprint,
        "parent_task_oracle_fingerprint": parent_oracle_fingerprint,
        "allowed_change_paths": sorted(set(allowed_change_paths)),
        "required_curriculum_prerequisite_ids": sorted(
            set(required_curriculum_prerequisite_ids)
        ),
        "public_constraints": sorted(set(public_constraints)),
    }
    return PublicAuthoringBrief.from_record(record)
