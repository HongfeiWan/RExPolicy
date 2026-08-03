"""Sealed K=1 audit-suite and capability-claim contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value with one stable, whitespace-free encoding."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _require_identifier(name: str, value: str) -> None:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"Invalid {name} {value!r}")


def _require_sha256(name: str, value: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"Invalid {name} SHA-256")


@dataclass(frozen=True)
class CapabilityClaimPolicy:
    """Trusted minimum evidence policy; task proposals cannot weaken it."""

    schema_version: int
    claim_spec_id: str
    candidate_count: int
    min_independent_training_seeds: int
    min_episodes_per_seed: int
    min_success_lower_bound: float
    max_new_safety_failures: int
    max_invalid_actions: int

    @classmethod
    def production_v1(cls) -> CapabilityClaimPolicy:
        """Return the formal self-improvement evidence floor."""
        return cls(
            schema_version=1,
            claim_spec_id="paired_k1_mastery/v1",
            candidate_count=1,
            min_independent_training_seeds=3,
            min_episodes_per_seed=256,
            min_success_lower_bound=0.80,
            max_new_safety_failures=0,
            max_invalid_actions=0,
        )

    def validate(self) -> None:
        """Reject policies that could turn search or weak evidence into mastery."""
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported capability claim policy {self.schema_version}"
            )
        _require_identifier("claim_spec_id", self.claim_spec_id)
        if self.candidate_count != 1:
            raise ValueError("Capability claims must use K=1 evaluation")
        if self.min_independent_training_seeds < 2:
            raise ValueError("Capability claims require independent training seeds")
        if self.min_episodes_per_seed < 1:
            raise ValueError("Capability claims require episodes per seed")
        if (
            not math.isfinite(self.min_success_lower_bound)
            or not 0.0 <= self.min_success_lower_bound <= 1.0
        ):
            raise ValueError("Success lower-bound threshold must be in [0, 1]")
        if min(
            self.max_new_safety_failures,
            self.max_invalid_actions,
        ) < 0:
            raise ValueError("Failure allowances cannot be negative")

    def to_record(self) -> dict[str, Any]:
        """Return the complete trusted policy record."""
        self.validate()
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        """Bind reports to the exact trusted policy."""
        return canonical_sha256(self.to_record())


@dataclass(frozen=True)
class SealedAuditSuite:
    """Commitments to held-out instructions and paired K=1 recipes."""

    schema_version: int
    audit_suite_id: str
    task_id: str
    task_spec_fingerprint: str
    claim_spec_id: str
    audit_instruction_count: int
    audit_instruction_sha256: str
    episodes_per_training_seed: int
    episode_recipe_sha256: str
    independent_training_seeds: tuple[int, ...]
    candidate_count: int

    def validate(self) -> None:
        """Reject unsealed, overlapping, or search-based audit suites."""
        if self.schema_version != 1:
            raise ValueError(f"Unsupported audit suite schema {self.schema_version}")
        for name, value in (
            ("audit_suite_id", self.audit_suite_id),
            ("task_id", self.task_id),
            ("claim_spec_id", self.claim_spec_id),
        ):
            _require_identifier(name, value)
        for name, value in (
            ("task_spec_fingerprint", self.task_spec_fingerprint),
            ("audit_instruction", self.audit_instruction_sha256),
            ("episode_recipe", self.episode_recipe_sha256),
        ):
            _require_sha256(name, value)
        if self.candidate_count != 1:
            raise ValueError("Audit suites must execute exactly one candidate")
        if min(
            self.audit_instruction_count,
            self.episodes_per_training_seed,
        ) < 1:
            raise ValueError("Audit suite counts must be positive")
        seeds = tuple(self.independent_training_seeds)
        if not seeds or seeds != tuple(sorted(set(seeds))):
            raise ValueError("Training seeds must be non-empty, unique, and sorted")
        if any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("Training seeds must be non-negative integers")

    def validate_against(self, policy: CapabilityClaimPolicy) -> None:
        """Require the sealed suite to meet the trusted claim policy."""
        self.validate()
        policy.validate()
        if self.claim_spec_id != policy.claim_spec_id:
            raise ValueError("Audit suite claim spec does not match policy")
        if self.candidate_count != policy.candidate_count:
            raise ValueError("Audit suite candidate count does not match policy")
        if (
            len(self.independent_training_seeds)
            < policy.min_independent_training_seeds
        ):
            raise ValueError("Audit suite has too few independent training seeds")
        if self.episodes_per_training_seed < policy.min_episodes_per_seed:
            raise ValueError("Audit suite has too few episodes per training seed")

    def to_record(self) -> dict[str, Any]:
        """Serialize commitments without exposing held-out instructions."""
        self.validate()
        record = asdict(self)
        record["independent_training_seeds"] = list(
            self.independent_training_seeds
        )
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> SealedAuditSuite:
        """Strictly restore a suite commitment."""
        values = dict(record)
        values["independent_training_seeds"] = tuple(
            values["independent_training_seeds"]
        )
        suite = cls(**values)
        suite.validate()
        return suite

    @property
    def fingerprint(self) -> str:
        """Identify the exact sealed suite used by an evaluation report."""
        return canonical_sha256(self.to_record())
