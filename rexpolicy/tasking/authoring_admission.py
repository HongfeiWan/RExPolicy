"""Fail-closed certification and admission for authored task candidates.

Static authoring success is deliberately insufficient for production use.  This
module binds a held-out audit plan before a candidate exists, validates the
eventual sealed suite against that exact candidate, and records certification,
admission, and activation as an immutable hash chain.  Nothing in this module
executes a task or grants authority implicitly.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rexpolicy.flywheel.audit import (
        CapabilityClaimPolicy,
        CapabilityClaimReport,
        SealedAuditSuite,
    )
else:
    CapabilityClaimPolicy = Any
    CapabilityClaimReport = Any
    SealedAuditSuite = Any

from .canonical import canonical_fingerprint, canonical_json, strict_json_loads
from .authoring import (
    DEFAULT_AUTHORING_POLICY,
    AuthoringIntent,
    AuthoringPolicy,
    ProposalAttempt,
)
from .authoring_brief import PublicAuthoringBrief
from .authoring_candidate import StaticCandidateBundle
from .capabilities import CapabilityCatalog
from .model import TaskSpecV2
from .contract import (
    DEFAULT_TASK_COMPILER_POLICY,
    CompiledTaskContract,
    TaskCompilerPolicy,
)
from .process import ProcessSpecV1
from .process_contract import (
    DEFAULT_PROCESS_COMPILER_POLICY,
    ProcessCompilerPolicy,
)
from .property_validation import (
    DEFAULT_PROPERTY_VALIDATION_POLICY,
    PropertyValidationPolicy,
)

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

QUARANTINED_STATIC = "QUARANTINED_STATIC"
ADMISSION_READY = "ADMISSION_READY"
ADMITTED_DORMANT = "ADMITTED_DORMANT"
ACTIVE = "ACTIVE"

_TRANSITIONS: dict[str | None, str] = {
    None: QUARANTINED_STATIC,
    QUARANTINED_STATIC: ADMISSION_READY,
    ADMISSION_READY: ADMITTED_DORMANT,
    ADMITTED_DORMANT: ACTIVE,
}
_STATES = frozenset(_TRANSITIONS.values())


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return dict(value)


def _keys(record: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(record)
    missing = sorted(expected.difference(actual))
    unknown = sorted(actual.difference(expected))
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


def _positive_integer(value: Any, path: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, path: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{path} must be Boolean")
    return value


def _reasons(value: Any, path: str, *, json_record: bool) -> tuple[str, ...]:
    expected_type = list if json_record else tuple
    if not isinstance(value, expected_type):
        label = "an array" if json_record else "a tuple"
        raise ValueError(f"{path} must be {label}")
    output: list[str] = []
    for index, reason in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"{item_path} must be non-empty text")
        if reason != reason.strip() or len(reason) > 512:
            raise ValueError(f"{item_path} must be trimmed and at most 512 characters")
        if unicodedata.normalize("NFC", reason) != reason:
            raise ValueError(f"{item_path} must use NFC Unicode normalization")
        if any(unicodedata.category(char).startswith("C") for char in reason):
            raise ValueError(f"{item_path} cannot contain control characters")
        output.append(reason)
    if len(set(output)) != len(output):
        raise ValueError(f"{path} must not contain duplicates")
    return tuple(output)


def _validate_decision(
    accepted: Any,
    reasons: Any,
    path: str,
) -> tuple[bool, tuple[str, ...]]:
    decision = _boolean(accepted, f"{path}.accepted")
    normalized = _reasons(reasons, f"{path}.reasons", json_record=False)
    if decision == bool(normalized):
        raise ValueError(f"{path} accepted/reasons are inconsistent")
    return decision, normalized


def _artifact_record(value: Any, path: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    serializer = getattr(value, "to_record", None)
    if not callable(serializer):
        raise ValueError(f"{path} must provide a canonical record")
    return _mapping(serializer(), path)


@dataclass(frozen=True)
class StaticCandidateVerificationContext:
    """Trusted inputs required to rerun the complete static authoring gate."""

    attempt: ProposalAttempt
    brief: PublicAuthoringBrief
    parent_task: TaskSpecV2 | None
    parent_contract: CompiledTaskContract | None
    catalog: CapabilityCatalog
    process_specs: Mapping[str, ProcessSpecV1]
    compiler_policy: TaskCompilerPolicy = DEFAULT_TASK_COMPILER_POLICY
    process_policy: ProcessCompilerPolicy = DEFAULT_PROCESS_COMPILER_POLICY
    property_policy: PropertyValidationPolicy = DEFAULT_PROPERTY_VALIDATION_POLICY
    authoring_policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY

    def verify(
        self,
        candidate: Any,
        *,
        intent: AuthoringIntent,
    ) -> StaticCandidateBundle:
        record = _artifact_record(candidate, "$candidate")
        fingerprint = canonical_fingerprint(record)
        advertised = getattr(candidate, "fingerprint", None)
        if advertised is not None and advertised != fingerprint:
            raise ValueError("Candidate fingerprint property does not match its record")
        return StaticCandidateBundle.from_record(
            record,
            expected_fingerprint=fingerprint,
            intent=intent,
            brief=self.brief,
            parent_task=self.parent_task,
            parent_contract=self.parent_contract,
            attempt=self.attempt,
            catalog=self.catalog,
            process_specs=self.process_specs,
            compiler_policy=self.compiler_policy,
            process_policy=self.process_policy,
            property_policy=self.property_policy,
            authoring_policy=self.authoring_policy,
        )


@dataclass(frozen=True)
class CandidateArtifactBinding:
    """Minimal exact binding shared by every certification artifact."""

    schema_version: int
    candidate_fingerprint: str
    task_id: str
    task_spec_fingerprint: str
    task_contract_fingerprint: str
    task_oracle_sha256: str
    process_contract_fingerprint: str

    @classmethod
    def from_candidate(cls, candidate: Any) -> CandidateArtifactBinding:
        supplied = _artifact_record(candidate, "$candidate")
        supplied_fingerprint = canonical_fingerprint(supplied)
        canonical_candidate = StaticCandidateBundle.from_record(
            supplied,
            expected_fingerprint=supplied_fingerprint,
        )
        record = canonical_candidate.to_record()
        required = {
            "candidate_state",
            "admission_status",
            "preauthoring_audit_plan_fingerprint",
            "task_spec",
            "task_spec_fingerprint",
            "task_contract",
            "task_contract_fingerprint",
            "task_oracle",
            "task_oracle_sha256",
            "process_contract",
            "process_contract_fingerprint",
        }
        missing = sorted(required.difference(record))
        if missing:
            raise ValueError(f"$candidate is missing fields: {', '.join(missing)}")
        if record["candidate_state"] != "quarantined_static":
            raise ValueError("Candidate must originate at quarantined_static")
        if record["admission_status"] != "not_admitted":
            raise ValueError("Candidate record must carry no admission authority")

        task_record = _mapping(record["task_spec"], "$candidate.task_spec")
        task = TaskSpecV2.from_record(task_record)
        task_fingerprint = _sha256(
            record["task_spec_fingerprint"],
            "$candidate.task_spec_fingerprint",
        )
        if task.fingerprint != task_fingerprint:
            raise ValueError("Candidate TaskSpec fingerprint mismatch")

        artifact_checks = (
            (
                "task_contract",
                "task_contract_fingerprint",
                "$candidate.task_contract",
            ),
            ("task_oracle", "task_oracle_sha256", "$candidate.task_oracle"),
            (
                "process_contract",
                "process_contract_fingerprint",
                "$candidate.process_contract",
            ),
        )
        artifact_fingerprints: dict[str, str] = {}
        for artifact_key, fingerprint_key, artifact_path in artifact_checks:
            artifact = _mapping(record[artifact_key], artifact_path)
            fingerprint = _sha256(
                record[fingerprint_key],
                f"$candidate.{fingerprint_key}",
            )
            if canonical_fingerprint(artifact) != fingerprint:
                raise ValueError(f"Candidate {artifact_key} fingerprint mismatch")
            artifact_fingerprints[fingerprint_key] = fingerprint

        contract = _mapping(record["task_contract"], "$candidate.task_contract")
        oracle = _mapping(record["task_oracle"], "$candidate.task_oracle")
        process_contract = _mapping(
            record["process_contract"],
            "$candidate.process_contract",
        )
        if (
            contract.get("task_id") != task.task_id
            or contract.get("task_spec_fingerprint") != task_fingerprint
            or oracle.get("task_id") != task.task_id
            or process_contract.get("task_spec_fingerprint") != task_fingerprint
            or process_contract.get("task_contract_fingerprint")
            != artifact_fingerprints["task_contract_fingerprint"]
        ):
            raise ValueError("Candidate artifacts do not bind the exact TaskSpec")

        candidate_fingerprint = canonical_fingerprint(record)
        advertised = getattr(candidate, "fingerprint", None)
        if advertised is not None and advertised != candidate_fingerprint:
            raise ValueError("Candidate fingerprint property does not match its record")
        return cls(
            schema_version=1,
            candidate_fingerprint=candidate_fingerprint,
            task_id=task.task_id,
            task_spec_fingerprint=task_fingerprint,
            task_contract_fingerprint=artifact_fingerprints[
                "task_contract_fingerprint"
            ],
            task_oracle_sha256=artifact_fingerprints["task_oracle_sha256"],
            process_contract_fingerprint=artifact_fingerprints[
                "process_contract_fingerprint"
            ],
        )

    @classmethod
    def from_record(cls, value: Any) -> CandidateArtifactBinding:
        path = "$candidate_binding"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "candidate_fingerprint",
                "task_id",
                "task_spec_fingerprint",
                "task_contract_fingerprint",
                "task_oracle_sha256",
                "process_contract_fingerprint",
            },
            path,
        )
        if type(record["schema_version"]) is not int or record["schema_version"] != 1:
            raise ValueError("CandidateArtifactBinding schema_version must be 1")
        return cls(
            schema_version=1,
            candidate_fingerprint=_sha256(
                record["candidate_fingerprint"],
                f"{path}.candidate_fingerprint",
            ),
            task_id=_identifier(record["task_id"], f"{path}.task_id"),
            task_spec_fingerprint=_sha256(
                record["task_spec_fingerprint"],
                f"{path}.task_spec_fingerprint",
            ),
            task_contract_fingerprint=_sha256(
                record["task_contract_fingerprint"],
                f"{path}.task_contract_fingerprint",
            ),
            task_oracle_sha256=_sha256(
                record["task_oracle_sha256"],
                f"{path}.task_oracle_sha256",
            ),
            process_contract_fingerprint=_sha256(
                record["process_contract_fingerprint"],
                f"{path}.process_contract_fingerprint",
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_fingerprint": self.candidate_fingerprint,
            "task_id": self.task_id,
            "task_spec_fingerprint": self.task_spec_fingerprint,
            "task_contract_fingerprint": self.task_contract_fingerprint,
            "task_oracle_sha256": self.task_oracle_sha256,
            "process_contract_fingerprint": self.process_contract_fingerprint,
        }


@dataclass(frozen=True)
class PreAuthoringAuditPlan:
    """Sealed evaluation commitments fixed before proposer output exists."""

    schema_version: int
    audit_plan_id: str
    task_id: str
    audit_instruction_count: int
    audit_instruction_sha256: str
    episode_recipe_sha256: str
    independent_training_seeds: tuple[int, ...]
    episodes_per_training_seed: int
    candidate_count: int
    claim_policy_id: str
    claim_policy_fingerprint: str

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("PreAuthoringAuditPlan schema_version must be 1")
        _identifier(self.audit_plan_id, "$audit_plan.audit_plan_id")
        _identifier(self.task_id, "$audit_plan.task_id")
        _positive_integer(
            self.audit_instruction_count,
            "$audit_plan.audit_instruction_count",
        )
        _sha256(
            self.audit_instruction_sha256,
            "$audit_plan.audit_instruction_sha256",
        )
        _sha256(self.episode_recipe_sha256, "$audit_plan.episode_recipe_sha256")
        _positive_integer(
            self.episodes_per_training_seed,
            "$audit_plan.episodes_per_training_seed",
        )
        if type(self.candidate_count) is not int or self.candidate_count != 1:
            raise ValueError("Pre-authoring audit plans must bind K=1")
        _identifier(self.claim_policy_id, "$audit_plan.claim_policy_id")
        _sha256(
            self.claim_policy_fingerprint,
            "$audit_plan.claim_policy_fingerprint",
        )
        seeds = self.independent_training_seeds
        if not seeds or seeds != tuple(sorted(set(seeds))):
            raise ValueError(
                "Audit-plan training seeds must be non-empty, unique, and sorted"
            )
        if any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("Audit-plan training seeds must be non-negative integers")

    def validate_against(self, policy: CapabilityClaimPolicy) -> None:
        self.validate()
        policy.validate()
        if (
            self.claim_policy_id != policy.claim_spec_id
            or self.claim_policy_fingerprint != policy.fingerprint
        ):
            raise ValueError("Audit plan does not bind the exact claim policy")
        if self.candidate_count != policy.candidate_count:
            raise ValueError("Audit-plan candidate count does not match policy")
        if (
            len(self.independent_training_seeds)
            < policy.min_independent_training_seeds
        ):
            raise ValueError("Audit plan has too few independent training seeds")
        if self.episodes_per_training_seed < policy.min_episodes_per_seed:
            raise ValueError("Audit plan has too few episodes per training seed")

    @classmethod
    def from_record(
        cls,
        value: Any,
        *,
        policy: CapabilityClaimPolicy | None = None,
        expected_fingerprint: str | None = None,
    ) -> PreAuthoringAuditPlan:
        path = "$audit_plan"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "audit_plan_id",
                "task_id",
                "audit_instruction_count",
                "audit_instruction_sha256",
                "episode_recipe_sha256",
                "independent_training_seeds",
                "episodes_per_training_seed",
                "candidate_count",
                "claim_policy_id",
                "claim_policy_fingerprint",
            },
            path,
        )
        if expected_fingerprint is not None:
            _sha256(expected_fingerprint, "expected_fingerprint")
            if canonical_fingerprint(record) != expected_fingerprint:
                raise ValueError("Pre-authoring audit-plan fingerprint mismatch")
        seeds = record["independent_training_seeds"]
        if not isinstance(seeds, list):
            raise ValueError("$audit_plan.independent_training_seeds must be an array")
        plan = cls(
            schema_version=record["schema_version"],
            audit_plan_id=_identifier(
                record["audit_plan_id"],
                f"{path}.audit_plan_id",
            ),
            task_id=_identifier(record["task_id"], f"{path}.task_id"),
            audit_instruction_count=_positive_integer(
                record["audit_instruction_count"],
                f"{path}.audit_instruction_count",
            ),
            audit_instruction_sha256=_sha256(
                record["audit_instruction_sha256"],
                f"{path}.audit_instruction_sha256",
            ),
            episode_recipe_sha256=_sha256(
                record["episode_recipe_sha256"],
                f"{path}.episode_recipe_sha256",
            ),
            independent_training_seeds=tuple(seeds),
            episodes_per_training_seed=_positive_integer(
                record["episodes_per_training_seed"],
                f"{path}.episodes_per_training_seed",
            ),
            candidate_count=record["candidate_count"],
            claim_policy_id=_identifier(
                record["claim_policy_id"],
                f"{path}.claim_policy_id",
            ),
            claim_policy_fingerprint=_sha256(
                record["claim_policy_fingerprint"],
                f"{path}.claim_policy_fingerprint",
            ),
        )
        if policy is None:
            plan.validate()
        else:
            plan.validate_against(policy)
        return plan

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "audit_plan_id": self.audit_plan_id,
            "task_id": self.task_id,
            "audit_instruction_count": self.audit_instruction_count,
            "audit_instruction_sha256": self.audit_instruction_sha256,
            "episode_recipe_sha256": self.episode_recipe_sha256,
            "independent_training_seeds": list(self.independent_training_seeds),
            "episodes_per_training_seed": self.episodes_per_training_seed,
            "candidate_count": self.candidate_count,
            "claim_policy_id": self.claim_policy_id,
            "claim_policy_fingerprint": self.claim_policy_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class AuditSuiteCandidateBinding:
    """Proof that one sealed suite is the exact precommitted candidate audit."""

    schema_version: int
    candidate: CandidateArtifactBinding
    audit_plan_fingerprint: str
    audit_suite_fingerprint: str
    claim_policy_id: str
    claim_policy_fingerprint: str

    @classmethod
    def from_record(cls, value: Any) -> AuditSuiteCandidateBinding:
        path = "$audit_suite_candidate_binding"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "candidate",
                "audit_plan_fingerprint",
                "audit_suite_fingerprint",
                "claim_policy_id",
                "claim_policy_fingerprint",
            },
            path,
        )
        if type(record["schema_version"]) is not int or record["schema_version"] != 1:
            raise ValueError("AuditSuiteCandidateBinding schema_version must be 1")
        return cls(
            schema_version=1,
            candidate=CandidateArtifactBinding.from_record(record["candidate"]),
            audit_plan_fingerprint=_sha256(
                record["audit_plan_fingerprint"],
                f"{path}.audit_plan_fingerprint",
            ),
            audit_suite_fingerprint=_sha256(
                record["audit_suite_fingerprint"],
                f"{path}.audit_suite_fingerprint",
            ),
            claim_policy_id=_identifier(
                record["claim_policy_id"],
                f"{path}.claim_policy_id",
            ),
            claim_policy_fingerprint=_sha256(
                record["claim_policy_fingerprint"],
                f"{path}.claim_policy_fingerprint",
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate": self.candidate.to_record(),
            "audit_plan_fingerprint": self.audit_plan_fingerprint,
            "audit_suite_fingerprint": self.audit_suite_fingerprint,
            "claim_policy_id": self.claim_policy_id,
            "claim_policy_fingerprint": self.claim_policy_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


def _canonical_policy_record(policy: CapabilityClaimPolicy) -> dict[str, Any]:
    validator = getattr(policy, "validate", None)
    if not callable(validator):
        raise ValueError("Claim policy must provide validation")
    validator()
    record = _artifact_record(policy, "$claim_policy")
    _keys(
        record,
        {
            "schema_version",
            "claim_spec_id",
            "candidate_count",
            "min_independent_training_seeds",
            "min_episodes_per_seed",
            "min_success_lower_bound",
            "max_new_safety_failures",
            "max_invalid_actions",
        },
        "$claim_policy",
    )
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError("Claim policy schema_version must be 1")
    _identifier(record["claim_spec_id"], "$claim_policy.claim_spec_id")
    if type(record["candidate_count"]) is not int or record["candidate_count"] != 1:
        raise ValueError("Claim policy must bind K=1")
    if _positive_integer(
        record["min_independent_training_seeds"],
        "$claim_policy.min_independent_training_seeds",
    ) < 2:
        raise ValueError("Claim policy requires independent training seeds")
    _positive_integer(
        record["min_episodes_per_seed"],
        "$claim_policy.min_episodes_per_seed",
    )
    success_floor = record["min_success_lower_bound"]
    if (
        isinstance(success_floor, bool)
        or not isinstance(success_floor, (int, float))
        or not math.isfinite(float(success_floor))
        or not 0.0 <= float(success_floor) <= 1.0
    ):
        raise ValueError("Claim policy success floor must be a finite probability")
    for field in ("max_new_safety_failures", "max_invalid_actions"):
        _nonnegative_integer(record[field], f"$claim_policy.{field}")
    fingerprint = canonical_fingerprint(record)
    if getattr(policy, "fingerprint", None) != fingerprint:
        raise ValueError("Claim policy fingerprint property does not match its record")
    return record


def _canonical_suite_record(
    suite: SealedAuditSuite,
    policy: CapabilityClaimPolicy,
) -> dict[str, Any]:
    validator = getattr(suite, "validate_against", None)
    if not callable(validator):
        raise ValueError("Sealed audit suite must provide policy validation")
    validator(policy)
    record = _artifact_record(suite, "$sealed_audit_suite")
    _keys(
        record,
        {
            "schema_version",
            "audit_suite_id",
            "task_id",
            "task_spec_fingerprint",
            "claim_spec_id",
            "audit_instruction_count",
            "audit_instruction_sha256",
            "episodes_per_training_seed",
            "episode_recipe_sha256",
            "independent_training_seeds",
            "candidate_count",
        },
        "$sealed_audit_suite",
    )
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError("Sealed audit suite schema_version must be 1")
    for field in ("audit_suite_id", "task_id", "claim_spec_id"):
        _identifier(record[field], f"$sealed_audit_suite.{field}")
    for field in (
        "task_spec_fingerprint",
        "audit_instruction_sha256",
        "episode_recipe_sha256",
    ):
        _sha256(record[field], f"$sealed_audit_suite.{field}")
    _positive_integer(
        record["audit_instruction_count"],
        "$sealed_audit_suite.audit_instruction_count",
    )
    _positive_integer(
        record["episodes_per_training_seed"],
        "$sealed_audit_suite.episodes_per_training_seed",
    )
    if type(record["candidate_count"]) is not int or record["candidate_count"] != 1:
        raise ValueError("Sealed audit suite must bind K=1")
    seeds = record["independent_training_seeds"]
    if not isinstance(seeds, list):
        raise ValueError(
            "$sealed_audit_suite.independent_training_seeds must be an array"
        )
    if not seeds or seeds != sorted(set(seeds)):
        raise ValueError(
            "Sealed audit-suite seeds must be non-empty, unique, and sorted"
        )
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Sealed audit-suite seeds must be non-negative integers")
    fingerprint = canonical_fingerprint(record)
    if getattr(suite, "fingerprint", None) != fingerprint:
        raise ValueError(
            "Sealed audit-suite fingerprint property does not match its record"
        )
    return record


def _wilson_lower_bound(successes: int, trials: int) -> float:
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("Invalid successes/trials for Wilson interval")
    z = 1.959963984540054
    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    centre = proportion + z_squared / (2.0 * trials)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z_squared / (4.0 * trials * trials)
    )
    return max(0.0, (centre - margin) / denominator)


def _recompute_capability_report_record(
    *,
    suite: SealedAuditSuite,
    policy: CapabilityClaimPolicy,
    episode_results: Sequence[Any],
) -> dict[str, Any]:
    suite_record = _canonical_suite_record(suite, policy)
    policy_record = _canonical_policy_record(policy)
    if isinstance(episode_results, (str, bytes)):
        raise ValueError("Capability episode results must be a sequence")
    records: list[dict[str, Any]] = []
    expected_fields = {
        "training_seed",
        "reset_seed",
        "diffusion_seed",
        "success",
        "safety_failure",
        "baseline_safety_failure",
        "invalid_action",
    }
    for index, episode in enumerate(episode_results):
        path = f"$capability_episode_results[{index}]"
        record = _artifact_record(episode, path)
        _keys(record, expected_fields, path)
        for field in ("training_seed", "reset_seed", "diffusion_seed"):
            _nonnegative_integer(record[field], f"{path}.{field}")
        for field in (
            "success",
            "safety_failure",
            "baseline_safety_failure",
            "invalid_action",
        ):
            _boolean(record[field], f"{path}.{field}")
        records.append(record)
    records.sort(
        key=lambda item: (
            item["training_seed"],
            item["reset_seed"],
            item["diffusion_seed"],
        )
    )
    keys = [
        (item["training_seed"], item["reset_seed"], item["diffusion_seed"])
        for item in records
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Capability claim contains duplicate episode keys")
    expected_seeds = tuple(suite_record["independent_training_seeds"])
    actual_seeds = tuple(sorted({item["training_seed"] for item in records}))
    if actual_seeds != expected_seeds:
        raise ValueError("Capability claim training seeds do not match audit suite")

    by_seed: dict[int, list[dict[str, Any]]] = {
        seed: [] for seed in expected_seeds
    }
    for record in records:
        by_seed[record["training_seed"]].append(record)
    expected_recipes: set[tuple[int, int]] | None = None
    metrics: list[dict[str, Any]] = []
    reasons: list[str] = []
    for seed in expected_seeds:
        episodes = by_seed[seed]
        expected_count = suite_record["episodes_per_training_seed"]
        if len(episodes) != expected_count:
            raise ValueError(f"Training seed {seed} has incomplete capability evidence")
        recipes = {
            (item["reset_seed"], item["diffusion_seed"])
            for item in episodes
        }
        if len(recipes) != len(episodes):
            raise ValueError(f"Training seed {seed} repeats an audit recipe")
        if expected_recipes is None:
            expected_recipes = recipes
        elif recipes != expected_recipes:
            raise ValueError("Capability claim is not paired across training seeds")
        successes = sum(item["success"] for item in episodes)
        lower_bound = _wilson_lower_bound(successes, len(episodes))
        new_safety = sum(
            item["safety_failure"] and not item["baseline_safety_failure"]
            for item in episodes
        )
        invalid_actions = sum(item["invalid_action"] for item in episodes)
        metrics.append(
            {
                "training_seed": seed,
                "episode_count": len(episodes),
                "success_count": successes,
                "success_rate": successes / len(episodes),
                "success_lower_bound": lower_bound,
                "new_safety_failure_count": new_safety,
                "invalid_action_count": invalid_actions,
            }
        )
        success_floor = float(policy_record["min_success_lower_bound"])
        if lower_bound < success_floor:
            reasons.append(
                f"training seed {seed} success lower bound "
                f"{lower_bound:.6f} < {success_floor:.6f}"
            )
    total_new_safety = sum(item["new_safety_failure_count"] for item in metrics)
    total_invalid = sum(item["invalid_action_count"] for item in metrics)
    if total_new_safety > policy_record["max_new_safety_failures"]:
        reasons.append(
            f"new safety failures {total_new_safety} > "
            f"{policy_record['max_new_safety_failures']}"
        )
    if total_invalid > policy_record["max_invalid_actions"]:
        reasons.append(
            f"invalid actions {total_invalid} > "
            f"{policy_record['max_invalid_actions']}"
        )
    return {
        "schema_version": 1,
        "claim_spec_id": policy_record["claim_spec_id"],
        "task_id": suite_record["task_id"],
        "task_spec_fingerprint": suite_record["task_spec_fingerprint"],
        "audit_suite_fingerprint": canonical_fingerprint(suite_record),
        "claim_policy_fingerprint": canonical_fingerprint(policy_record),
        "candidate_count": 1,
        "episode_results_sha256": canonical_fingerprint(records),
        "total_episode_count": len(records),
        "seed_metrics": metrics,
        "accepted": not reasons,
        "reasons": reasons,
    }


def _validate_intent_plan_candidate(
    *,
    intent: AuthoringIntent,
    plan: PreAuthoringAuditPlan,
    candidate_record: Mapping[str, Any],
    task: TaskSpecV2,
) -> AuthoringIntent:
    canonical_intent = AuthoringIntent.from_record(intent.to_record())
    if canonical_intent.schema_version != 2:
        raise ValueError("Admission requires an AuthoringIntent v2")
    if canonical_intent.preauthoring_audit_plan_fingerprint != plan.fingerprint:
        raise ValueError("AuthoringIntent does not bind the pre-authoring audit plan")
    if (
        candidate_record.get("intent_fingerprint") != canonical_intent.fingerprint
        or canonical_intent.task_id != task.task_id
        or canonical_intent.task_version != task.version
        or canonical_intent.audit_commitment != task.instructions.audit
    ):
        raise ValueError("AuthoringIntent does not bind the exact candidate TaskSpec")
    advertised_plan = _sha256(
        candidate_record.get("preauthoring_audit_plan_fingerprint"),
        "$candidate.preauthoring_audit_plan_fingerprint",
    )
    if advertised_plan != plan.fingerprint:
        raise ValueError("Candidate audit-plan binding mismatch")
    return canonical_intent


def validate_audit_suite_for_candidate(
    *,
    suite: SealedAuditSuite,
    candidate: Any,
    plan: PreAuthoringAuditPlan,
    policy: CapabilityClaimPolicy,
    intent: AuthoringIntent,
    verification_context: StaticCandidateVerificationContext,
) -> AuditSuiteCandidateBinding:
    """Cross-check every candidate, audit-plan, suite, and policy commitment."""
    policy_record = _canonical_policy_record(policy)
    suite_record = _canonical_suite_record(suite, policy)
    canonical_plan = PreAuthoringAuditPlan.from_record(
        plan.to_record(),
        policy=policy,
        expected_fingerprint=plan.fingerprint,
    )
    verified_candidate = verification_context.verify(candidate, intent=intent)
    candidate_record = verified_candidate.to_record()
    candidate_binding = CandidateArtifactBinding.from_candidate(verified_candidate)
    task = TaskSpecV2.from_record(
        _mapping(candidate_record["task_spec"], "$candidate.task_spec")
    )
    _validate_intent_plan_candidate(
        intent=intent,
        plan=canonical_plan,
        candidate_record=candidate_record,
        task=task,
    )
    audit = task.instructions.audit
    if (
        canonical_plan.task_id != task.task_id
        or canonical_plan.audit_instruction_count != audit.count
        or canonical_plan.audit_instruction_sha256 != audit.sha256
    ):
        raise ValueError("Pre-authoring audit plan does not match candidate TaskSpec")
    if (
        suite_record["task_id"] != task.task_id
        or suite_record["task_spec_fingerprint"] != task.fingerprint
        or suite_record["audit_instruction_count"] != audit.count
        or suite_record["audit_instruction_sha256"] != audit.sha256
    ):
        raise ValueError("Sealed audit suite does not match candidate TaskSpec")
    if (
        suite_record["claim_spec_id"] != canonical_plan.claim_policy_id
        or suite_record["audit_instruction_count"]
        != canonical_plan.audit_instruction_count
        or suite_record["audit_instruction_sha256"]
        != canonical_plan.audit_instruction_sha256
        or suite_record["episode_recipe_sha256"]
        != canonical_plan.episode_recipe_sha256
        or tuple(suite_record["independent_training_seeds"])
        != canonical_plan.independent_training_seeds
        or suite_record["episodes_per_training_seed"]
        != canonical_plan.episodes_per_training_seed
        or suite_record["candidate_count"] != canonical_plan.candidate_count
        or canonical_plan.claim_policy_id != policy_record["claim_spec_id"]
        or canonical_plan.claim_policy_fingerprint
        != canonical_fingerprint(policy_record)
    ):
        raise ValueError("Sealed audit suite does not match pre-authoring plan")
    return AuditSuiteCandidateBinding(
        schema_version=1,
        candidate=candidate_binding,
        audit_plan_fingerprint=canonical_plan.fingerprint,
        audit_suite_fingerprint=canonical_fingerprint(suite_record),
        claim_policy_id=policy_record["claim_spec_id"],
        claim_policy_fingerprint=canonical_fingerprint(policy_record),
    )


@dataclass(frozen=True)
class RuntimeCertification:
    """Trusted runtime evidence bound to one exact static candidate."""

    schema_version: int
    certification_id: str
    candidate: CandidateArtifactBinding
    runtime_policy_id: str
    runtime_policy_fingerprint: str
    runtime_evidence_sha256: str
    accepted: bool
    reasons: tuple[str, ...]

    @classmethod
    def issue(
        cls,
        *,
        certification_id: str,
        candidate: Any,
        runtime_policy_id: str,
        runtime_policy_fingerprint: str,
        runtime_evidence_sha256: str,
        accepted: bool,
        reasons: Sequence[str],
        intent: AuthoringIntent,
        verification_context: StaticCandidateVerificationContext,
    ) -> RuntimeCertification:
        if isinstance(reasons, str):
            raise ValueError("Runtime certification reasons must be a sequence")
        verified_candidate = verification_context.verify(candidate, intent=intent)
        record = {
            "schema_version": 1,
            "certification_id": certification_id,
            "candidate": CandidateArtifactBinding.from_candidate(
                verified_candidate
            ).to_record(),
            "runtime_policy_id": runtime_policy_id,
            "runtime_policy_fingerprint": runtime_policy_fingerprint,
            "runtime_evidence_sha256": runtime_evidence_sha256,
            "accepted": accepted,
            "reasons": list(reasons),
        }
        return cls.from_record(
            record,
            candidate=verified_candidate,
            intent=intent,
            verification_context=verification_context,
        )

    @classmethod
    def from_record(
        cls,
        value: Any,
        *,
        candidate: Any | None = None,
        intent: AuthoringIntent | None = None,
        verification_context: StaticCandidateVerificationContext | None = None,
        expected_fingerprint: str | None = None,
    ) -> RuntimeCertification:
        path = "$runtime_certification"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "certification_id",
                "candidate",
                "runtime_policy_id",
                "runtime_policy_fingerprint",
                "runtime_evidence_sha256",
                "accepted",
                "reasons",
            },
            path,
        )
        if expected_fingerprint is not None:
            _sha256(expected_fingerprint, "expected_fingerprint")
            if canonical_fingerprint(record) != expected_fingerprint:
                raise ValueError("Runtime certification fingerprint mismatch")
        if type(record["schema_version"]) is not int or record["schema_version"] != 1:
            raise ValueError("RuntimeCertification schema_version must be 1")
        if not isinstance(record["reasons"], list):
            raise ValueError("$runtime_certification.reasons must be an array")
        reasons = _reasons(record["reasons"], f"{path}.reasons", json_record=True)
        accepted = _boolean(record["accepted"], f"{path}.accepted")
        if accepted == bool(reasons):
            raise ValueError("Runtime certification accepted/reasons are inconsistent")
        certification = cls(
            schema_version=1,
            certification_id=_identifier(
                record["certification_id"],
                f"{path}.certification_id",
            ),
            candidate=CandidateArtifactBinding.from_record(record["candidate"]),
            runtime_policy_id=_identifier(
                record["runtime_policy_id"],
                f"{path}.runtime_policy_id",
            ),
            runtime_policy_fingerprint=_sha256(
                record["runtime_policy_fingerprint"],
                f"{path}.runtime_policy_fingerprint",
            ),
            runtime_evidence_sha256=_sha256(
                record["runtime_evidence_sha256"],
                f"{path}.runtime_evidence_sha256",
            ),
            accepted=accepted,
            reasons=reasons,
        )
        if candidate is not None:
            certification.validate_for_candidate(
                candidate,
                intent=intent,
                verification_context=verification_context,
            )
        return certification

    def validate_for_candidate(
        self,
        candidate: Any,
        *,
        intent: AuthoringIntent | None = None,
        verification_context: StaticCandidateVerificationContext | None = None,
        runtime_policy_id: str | None = None,
        runtime_policy_fingerprint: str | None = None,
    ) -> None:
        if verification_context is not None:
            if intent is None:
                raise ValueError(
                    "AuthoringIntent is required for semantic candidate verification"
                )
            candidate = verification_context.verify(candidate, intent=intent)
        elif intent is not None:
            raise ValueError(
                "Static verification context is required with AuthoringIntent"
            )
        if self.candidate != CandidateArtifactBinding.from_candidate(candidate):
            raise ValueError("Runtime certification candidate binding mismatch")
        if runtime_policy_id is not None:
            if self.runtime_policy_id != _identifier(
                runtime_policy_id,
                "runtime_policy_id",
            ):
                raise ValueError("Runtime certification policy ID mismatch")
        if runtime_policy_fingerprint is not None:
            if self.runtime_policy_fingerprint != _sha256(
                runtime_policy_fingerprint,
                "runtime_policy_fingerprint",
            ):
                raise ValueError("Runtime certification policy fingerprint mismatch")

    def to_record(self) -> dict[str, Any]:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("RuntimeCertification schema_version must be 1")
        _identifier(self.certification_id, "$runtime_certification.certification_id")
        _identifier(self.runtime_policy_id, "$runtime_certification.runtime_policy_id")
        _sha256(
            self.runtime_policy_fingerprint,
            "$runtime_certification.runtime_policy_fingerprint",
        )
        _sha256(
            self.runtime_evidence_sha256,
            "$runtime_certification.runtime_evidence_sha256",
        )
        _validate_decision(
            self.accepted,
            self.reasons,
            "$runtime_certification",
        )
        return {
            "schema_version": self.schema_version,
            "certification_id": self.certification_id,
            "candidate": self.candidate.to_record(),
            "runtime_policy_id": self.runtime_policy_id,
            "runtime_policy_fingerprint": self.runtime_policy_fingerprint,
            "runtime_evidence_sha256": self.runtime_evidence_sha256,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class SealedAuditCertification:
    """Formal K=1 audit outcome bound to one candidate and precommitment."""

    schema_version: int
    certification_id: str
    candidate: CandidateArtifactBinding
    audit_plan_fingerprint: str
    audit_suite_fingerprint: str
    claim_policy_id: str
    claim_policy_fingerprint: str
    capability_claim_report_fingerprint: str
    accepted: bool
    reasons: tuple[str, ...]

    @classmethod
    def issue(
        cls,
        *,
        certification_id: str,
        candidate: Any,
        plan: PreAuthoringAuditPlan,
        suite: SealedAuditSuite,
        policy: CapabilityClaimPolicy,
        report: CapabilityClaimReport,
        intent: AuthoringIntent,
        verification_context: StaticCandidateVerificationContext,
        episode_results: Sequence[Any],
    ) -> SealedAuditCertification:
        binding = validate_audit_suite_for_candidate(
            suite=suite,
            candidate=candidate,
            plan=plan,
            policy=policy,
            intent=intent,
            verification_context=verification_context,
        )
        report_record = report.to_record()
        report_fingerprint = canonical_fingerprint(report_record)
        if getattr(report, "fingerprint", None) != report_fingerprint:
            raise ValueError("Capability claim report fingerprint mismatch")
        recomputed_report = _recompute_capability_report_record(
            suite=suite,
            policy=policy,
            episode_results=episode_results,
        )
        if report_record != recomputed_report:
            raise ValueError("Capability claim report binding mismatch")
        return cls.from_record(
            {
                "schema_version": 1,
                "certification_id": certification_id,
                "candidate": binding.candidate.to_record(),
                "audit_plan_fingerprint": binding.audit_plan_fingerprint,
                "audit_suite_fingerprint": binding.audit_suite_fingerprint,
                "claim_policy_id": binding.claim_policy_id,
                "claim_policy_fingerprint": binding.claim_policy_fingerprint,
                "capability_claim_report_fingerprint": report_fingerprint,
                "accepted": report_record["accepted"],
                "reasons": list(report_record["reasons"]),
            },
            candidate=candidate,
            plan=plan,
            suite=suite,
            policy=policy,
            report=report,
            intent=intent,
            verification_context=verification_context,
            episode_results=episode_results,
        )

    @classmethod
    def from_record(
        cls,
        value: Any,
        *,
        candidate: Any | None = None,
        plan: PreAuthoringAuditPlan | None = None,
        suite: SealedAuditSuite | None = None,
        policy: CapabilityClaimPolicy | None = None,
        report: CapabilityClaimReport | None = None,
        intent: AuthoringIntent | None = None,
        verification_context: StaticCandidateVerificationContext | None = None,
        episode_results: Sequence[Any] | None = None,
        expected_fingerprint: str | None = None,
    ) -> SealedAuditCertification:
        path = "$sealed_audit_certification"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "certification_id",
                "candidate",
                "audit_plan_fingerprint",
                "audit_suite_fingerprint",
                "claim_policy_id",
                "claim_policy_fingerprint",
                "capability_claim_report_fingerprint",
                "accepted",
                "reasons",
            },
            path,
        )
        if expected_fingerprint is not None:
            _sha256(expected_fingerprint, "expected_fingerprint")
            if canonical_fingerprint(record) != expected_fingerprint:
                raise ValueError("Sealed audit certification fingerprint mismatch")
        if type(record["schema_version"]) is not int or record["schema_version"] != 1:
            raise ValueError("SealedAuditCertification schema_version must be 1")
        if not isinstance(record["reasons"], list):
            raise ValueError("$sealed_audit_certification.reasons must be an array")
        reasons = _reasons(record["reasons"], f"{path}.reasons", json_record=True)
        accepted = _boolean(record["accepted"], f"{path}.accepted")
        if accepted == bool(reasons):
            raise ValueError(
                "Sealed audit certification accepted/reasons are inconsistent"
            )
        certification = cls(
            schema_version=1,
            certification_id=_identifier(
                record["certification_id"],
                f"{path}.certification_id",
            ),
            candidate=CandidateArtifactBinding.from_record(record["candidate"]),
            audit_plan_fingerprint=_sha256(
                record["audit_plan_fingerprint"],
                f"{path}.audit_plan_fingerprint",
            ),
            audit_suite_fingerprint=_sha256(
                record["audit_suite_fingerprint"],
                f"{path}.audit_suite_fingerprint",
            ),
            claim_policy_id=_identifier(
                record["claim_policy_id"],
                f"{path}.claim_policy_id",
            ),
            claim_policy_fingerprint=_sha256(
                record["claim_policy_fingerprint"],
                f"{path}.claim_policy_fingerprint",
            ),
            capability_claim_report_fingerprint=_sha256(
                record["capability_claim_report_fingerprint"],
                f"{path}.capability_claim_report_fingerprint",
            ),
            accepted=accepted,
            reasons=reasons,
        )
        certification.validate_bindings(
            candidate=candidate,
            plan=plan,
            suite=suite,
            policy=policy,
            report=report,
            intent=intent,
            verification_context=verification_context,
            episode_results=episode_results,
        )
        return certification

    def validate_bindings(
        self,
        *,
        candidate: Any | None = None,
        plan: PreAuthoringAuditPlan | None = None,
        suite: SealedAuditSuite | None = None,
        policy: CapabilityClaimPolicy | None = None,
        report: CapabilityClaimReport | None = None,
        intent: AuthoringIntent | None = None,
        verification_context: StaticCandidateVerificationContext | None = None,
        episode_results: Sequence[Any] | None = None,
    ) -> None:
        if candidate is not None:
            if verification_context is not None:
                if intent is None:
                    raise ValueError(
                        "AuthoringIntent is required for semantic candidate "
                        "verification"
                    )
                candidate = verification_context.verify(candidate, intent=intent)
            expected = CandidateArtifactBinding.from_candidate(candidate)
            if self.candidate != expected:
                raise ValueError(
                    "Sealed audit certification candidate binding mismatch"
                )
        if plan is not None:
            plan.validate()
            if (
                self.audit_plan_fingerprint != plan.fingerprint
                or self.claim_policy_id != plan.claim_policy_id
                or self.claim_policy_fingerprint != plan.claim_policy_fingerprint
            ):
                raise ValueError("Sealed audit certification plan binding mismatch")
        if intent is not None:
            if candidate is None or plan is None:
                raise ValueError(
                    "Candidate and audit plan are required to validate AuthoringIntent"
                )
            candidate_record = _artifact_record(candidate, "$candidate")
            task = TaskSpecV2.from_record(
                _mapping(candidate_record["task_spec"], "$candidate.task_spec")
            )
            _validate_intent_plan_candidate(
                intent=intent,
                plan=plan,
                candidate_record=candidate_record,
                task=task,
            )
        if suite is not None and self.audit_suite_fingerprint != suite.fingerprint:
            raise ValueError("Sealed audit certification suite binding mismatch")
        if policy is not None:
            policy.validate()
            if (
                self.claim_policy_id != policy.claim_spec_id
                or self.claim_policy_fingerprint != policy.fingerprint
            ):
                raise ValueError("Sealed audit certification policy binding mismatch")
        if report is not None:
            if suite is None or policy is None or episode_results is None:
                raise ValueError(
                    "Suite, policy, and episodes are required to validate audit report"
                )
            report_record = report.to_record()
            report_fingerprint = canonical_fingerprint(report_record)
            recomputed_report = _recompute_capability_report_record(
                suite=suite,
                policy=policy,
                episode_results=episode_results,
            )
            if (
                getattr(report, "fingerprint", None) != report_fingerprint
                or report_record != recomputed_report
                or self.capability_claim_report_fingerprint != report_fingerprint
                or self.accepted != report_record["accepted"]
                or self.reasons != tuple(report_record["reasons"])
                or self.candidate.task_id != report_record["task_id"]
                or self.candidate.task_spec_fingerprint
                != report_record["task_spec_fingerprint"]
                or self.audit_suite_fingerprint
                != report_record["audit_suite_fingerprint"]
                or self.claim_policy_fingerprint
                != report_record["claim_policy_fingerprint"]
            ):
                raise ValueError("Sealed audit certification report binding mismatch")

    def to_record(self) -> dict[str, Any]:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("SealedAuditCertification schema_version must be 1")
        _identifier(
            self.certification_id,
            "$sealed_audit_certification.certification_id",
        )
        _sha256(
            self.audit_plan_fingerprint,
            "$sealed_audit_certification.audit_plan_fingerprint",
        )
        _sha256(
            self.audit_suite_fingerprint,
            "$sealed_audit_certification.audit_suite_fingerprint",
        )
        _identifier(
            self.claim_policy_id,
            "$sealed_audit_certification.claim_policy_id",
        )
        _sha256(
            self.claim_policy_fingerprint,
            "$sealed_audit_certification.claim_policy_fingerprint",
        )
        _sha256(
            self.capability_claim_report_fingerprint,
            "$sealed_audit_certification.capability_claim_report_fingerprint",
        )
        _validate_decision(
            self.accepted,
            self.reasons,
            "$sealed_audit_certification",
        )
        return {
            "schema_version": self.schema_version,
            "certification_id": self.certification_id,
            "candidate": self.candidate.to_record(),
            "audit_plan_fingerprint": self.audit_plan_fingerprint,
            "audit_suite_fingerprint": self.audit_suite_fingerprint,
            "claim_policy_id": self.claim_policy_id,
            "claim_policy_fingerprint": self.claim_policy_fingerprint,
            "capability_claim_report_fingerprint": (
                self.capability_claim_report_fingerprint
            ),
            "accepted": self.accepted,
            "reasons": list(self.reasons),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class LifecycleAuthority:
    """Explicit, hash-bound authority for one admission or activation act."""

    schema_version: int
    authority_id: str
    action: str
    candidate_fingerprint: str
    predecessor_event_sha256: str
    decision_sha256: str

    @classmethod
    def issue(
        cls,
        *,
        authority_id: str,
        action: str,
        candidate: Any,
        predecessor_event: LifecycleEvent,
        decision_sha256: str,
    ) -> LifecycleAuthority:
        return cls.from_record(
            {
                "schema_version": 1,
                "authority_id": authority_id,
                "action": action,
                "candidate_fingerprint": (
                    CandidateArtifactBinding.from_candidate(
                        candidate
                    ).candidate_fingerprint
                ),
                "predecessor_event_sha256": predecessor_event.fingerprint,
                "decision_sha256": decision_sha256,
            }
        )

    @classmethod
    def from_record(cls, value: Any) -> LifecycleAuthority:
        path = "$lifecycle_authority"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "authority_id",
                "action",
                "candidate_fingerprint",
                "predecessor_event_sha256",
                "decision_sha256",
            },
            path,
        )
        if type(record["schema_version"]) is not int or record["schema_version"] != 1:
            raise ValueError("LifecycleAuthority schema_version must be 1")
        action = _identifier(record["action"], f"{path}.action")
        if action not in {"admit", "activate"}:
            raise ValueError("Lifecycle authority action must be admit or activate")
        return cls(
            schema_version=1,
            authority_id=_identifier(record["authority_id"], f"{path}.authority_id"),
            action=action,
            candidate_fingerprint=_sha256(
                record["candidate_fingerprint"],
                f"{path}.candidate_fingerprint",
            ),
            predecessor_event_sha256=_sha256(
                record["predecessor_event_sha256"],
                f"{path}.predecessor_event_sha256",
            ),
            decision_sha256=_sha256(
                record["decision_sha256"],
                f"{path}.decision_sha256",
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "authority_id": _identifier(
                self.authority_id,
                "$lifecycle_authority.authority_id",
            ),
            "action": _identifier(
                self.action,
                "$lifecycle_authority.action",
            ),
            "candidate_fingerprint": _sha256(
                self.candidate_fingerprint,
                "$lifecycle_authority.candidate_fingerprint",
            ),
            "predecessor_event_sha256": _sha256(
                self.predecessor_event_sha256,
                "$lifecycle_authority.predecessor_event_sha256",
            ),
            "decision_sha256": _sha256(
                self.decision_sha256,
                "$lifecycle_authority.decision_sha256",
            ),
        }


@dataclass(frozen=True)
class GenerationBoundaryBinding:
    """Exact generation boundary at which a dormant task may become active."""

    schema_version: int
    boundary_id: str
    candidate_fingerprint: str
    admitted_event_sha256: str
    generation_manifest_sha256: str

    @classmethod
    def issue(
        cls,
        *,
        boundary_id: str,
        candidate: Any,
        admitted_event: LifecycleEvent,
        generation_manifest_sha256: str,
    ) -> GenerationBoundaryBinding:
        if admitted_event.state != ADMITTED_DORMANT:
            raise ValueError("Generation boundary must bind an admitted dormant event")
        return cls.from_record(
            {
                "schema_version": 1,
                "boundary_id": boundary_id,
                "candidate_fingerprint": (
                    CandidateArtifactBinding.from_candidate(
                        candidate
                    ).candidate_fingerprint
                ),
                "admitted_event_sha256": admitted_event.fingerprint,
                "generation_manifest_sha256": generation_manifest_sha256,
            }
        )

    @classmethod
    def from_record(cls, value: Any) -> GenerationBoundaryBinding:
        path = "$generation_boundary"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "boundary_id",
                "candidate_fingerprint",
                "admitted_event_sha256",
                "generation_manifest_sha256",
            },
            path,
        )
        if type(record["schema_version"]) is not int or record["schema_version"] != 1:
            raise ValueError("GenerationBoundaryBinding schema_version must be 1")
        return cls(
            schema_version=1,
            boundary_id=_identifier(record["boundary_id"], f"{path}.boundary_id"),
            candidate_fingerprint=_sha256(
                record["candidate_fingerprint"],
                f"{path}.candidate_fingerprint",
            ),
            admitted_event_sha256=_sha256(
                record["admitted_event_sha256"],
                f"{path}.admitted_event_sha256",
            ),
            generation_manifest_sha256=_sha256(
                record["generation_manifest_sha256"],
                f"{path}.generation_manifest_sha256",
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "boundary_id": _identifier(
                self.boundary_id,
                "$generation_boundary.boundary_id",
            ),
            "candidate_fingerprint": _sha256(
                self.candidate_fingerprint,
                "$generation_boundary.candidate_fingerprint",
            ),
            "admitted_event_sha256": _sha256(
                self.admitted_event_sha256,
                "$generation_boundary.admitted_event_sha256",
            ),
            "generation_manifest_sha256": _sha256(
                self.generation_manifest_sha256,
                "$generation_boundary.generation_manifest_sha256",
            ),
        }


@dataclass(frozen=True)
class LifecycleEvent:
    """One append-only candidate lifecycle transition linked to its predecessor."""

    schema_version: int
    candidate_fingerprint: str
    audit_plan_fingerprint: str
    sequence: int
    previous_event_sha256: str | None
    previous_state: str | None
    state: str
    runtime_certification_fingerprint: str | None
    sealed_audit_certification_fingerprint: str | None
    authority: LifecycleAuthority | None
    generation_boundary: GenerationBoundaryBinding | None

    @classmethod
    def from_record(
        cls,
        value: Any,
        *,
        candidate: Any,
        plan: PreAuthoringAuditPlan,
        intent: AuthoringIntent,
        verification_context: StaticCandidateVerificationContext,
        history: Sequence[LifecycleEvent] = (),
        runtime_certification: RuntimeCertification | None = None,
        sealed_audit_certification: SealedAuditCertification | None = None,
        runtime_policy_id: str | None = None,
        runtime_policy_fingerprint: str | None = None,
        suite: SealedAuditSuite | None = None,
        policy: CapabilityClaimPolicy | None = None,
        report: CapabilityClaimReport | None = None,
        episode_results: Sequence[Any] | None = None,
        expected_fingerprint: str | None = None,
    ) -> LifecycleEvent:
        verified_history = validate_lifecycle_chain(
            history,
            candidate=candidate,
            plan=plan,
            intent=intent,
            verification_context=verification_context,
            runtime_certification=runtime_certification,
            sealed_audit_certification=sealed_audit_certification,
            runtime_policy_id=runtime_policy_id,
            runtime_policy_fingerprint=runtime_policy_fingerprint,
            suite=suite,
            policy=policy,
            report=report,
            episode_results=episode_results,
            allow_empty=True,
        )
        return cls._from_record_step(
            value,
            candidate=candidate,
            plan=plan,
            intent=intent,
            verification_context=verification_context,
            previous_event=(verified_history[-1] if verified_history else None),
            runtime_certification=runtime_certification,
            sealed_audit_certification=sealed_audit_certification,
            runtime_policy_id=runtime_policy_id,
            runtime_policy_fingerprint=runtime_policy_fingerprint,
            suite=suite,
            policy=policy,
            report=report,
            episode_results=episode_results,
            expected_fingerprint=expected_fingerprint,
        )

    @classmethod
    def _from_record_step(
        cls,
        value: Any,
        *,
        candidate: Any,
        plan: PreAuthoringAuditPlan,
        intent: AuthoringIntent,
        verification_context: StaticCandidateVerificationContext,
        previous_event: LifecycleEvent | None = None,
        runtime_certification: RuntimeCertification | None = None,
        sealed_audit_certification: SealedAuditCertification | None = None,
        runtime_policy_id: str | None = None,
        runtime_policy_fingerprint: str | None = None,
        suite: SealedAuditSuite | None = None,
        policy: CapabilityClaimPolicy | None = None,
        report: CapabilityClaimReport | None = None,
        episode_results: Sequence[Any] | None = None,
        expected_fingerprint: str | None = None,
    ) -> LifecycleEvent:
        path = "$lifecycle_event"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "candidate_fingerprint",
                "audit_plan_fingerprint",
                "sequence",
                "previous_event_sha256",
                "previous_state",
                "state",
                "runtime_certification_fingerprint",
                "sealed_audit_certification_fingerprint",
                "authority",
                "generation_boundary",
            },
            path,
        )
        if expected_fingerprint is not None:
            _sha256(expected_fingerprint, "expected_fingerprint")
            if canonical_fingerprint(record) != expected_fingerprint:
                raise ValueError("Lifecycle event fingerprint mismatch")
        if type(record["schema_version"]) is not int or record["schema_version"] != 1:
            raise ValueError("LifecycleEvent schema_version must be 1")

        verified_candidate = verification_context.verify(candidate, intent=intent)
        binding = CandidateArtifactBinding.from_candidate(verified_candidate)
        canonical_plan = PreAuthoringAuditPlan.from_record(
            plan.to_record(),
            expected_fingerprint=plan.fingerprint,
        )
        candidate_record = verified_candidate.to_record()
        task = TaskSpecV2.from_record(
            _mapping(candidate_record["task_spec"], "$candidate.task_spec")
        )
        _validate_intent_plan_candidate(
            intent=intent,
            plan=canonical_plan,
            candidate_record=candidate_record,
            task=task,
        )
        candidate_fingerprint = _sha256(
            record["candidate_fingerprint"],
            f"{path}.candidate_fingerprint",
        )
        audit_plan_fingerprint = _sha256(
            record["audit_plan_fingerprint"],
            f"{path}.audit_plan_fingerprint",
        )
        if candidate_fingerprint != binding.candidate_fingerprint:
            raise ValueError("Lifecycle event candidate binding mismatch")
        if audit_plan_fingerprint != canonical_plan.fingerprint:
            raise ValueError("Lifecycle event audit-plan binding mismatch")

        sequence = _nonnegative_integer(record["sequence"], f"{path}.sequence")
        state = record["state"]
        previous_state = record["previous_state"]
        if state not in _STATES:
            raise ValueError("Lifecycle event state is invalid")
        if previous_state is not None and previous_state not in _STATES:
            raise ValueError("Lifecycle event previous_state is invalid")

        previous_hash_value = record["previous_event_sha256"]
        if previous_event is None:
            if (
                sequence != 0
                or previous_state is not None
                or previous_hash_value is not None
                or state != QUARANTINED_STATIC
            ):
                raise ValueError("Lifecycle must begin at QUARANTINED_STATIC")
        else:
            if (
                sequence != previous_event.sequence + 1
                or previous_state != previous_event.state
                or previous_hash_value != previous_event.fingerprint
                or candidate_fingerprint != previous_event.candidate_fingerprint
                or audit_plan_fingerprint != previous_event.audit_plan_fingerprint
            ):
                raise ValueError("Lifecycle hash-chain predecessor mismatch")
            expected_state = _TRANSITIONS.get(previous_event.state)
            if state != expected_state:
                raise ValueError("Lifecycle transition skips or reverses state")

        runtime_fingerprint = record["runtime_certification_fingerprint"]
        audit_fingerprint = record["sealed_audit_certification_fingerprint"]
        if state == QUARANTINED_STATIC:
            if runtime_fingerprint is not None or audit_fingerprint is not None:
                raise ValueError("Quarantine event cannot carry certifications")
            runtime_value = None
            audit_value = None
        else:
            if (
                runtime_certification is None
                or sealed_audit_certification is None
                or runtime_policy_id is None
                or runtime_policy_fingerprint is None
                or suite is None
                or policy is None
                or report is None
                or episode_results is None
            ):
                raise ValueError(
                    "Post-quarantine lifecycle events require complete evidence"
                )
            runtime_certification.validate_for_candidate(
                verified_candidate,
                intent=intent,
                verification_context=verification_context,
                runtime_policy_id=runtime_policy_id,
                runtime_policy_fingerprint=runtime_policy_fingerprint,
            )
            sealed_audit_certification.validate_bindings(
                candidate=verified_candidate,
                plan=canonical_plan,
                suite=suite,
                policy=policy,
                report=report,
                intent=intent,
                verification_context=verification_context,
                episode_results=episode_results,
            )
            if (
                not runtime_certification.accepted
                or not sealed_audit_certification.accepted
            ):
                raise ValueError(
                    "Lifecycle readiness requires two accepted certificates"
                )
            runtime_value = _sha256(
                runtime_fingerprint,
                f"{path}.runtime_certification_fingerprint",
            )
            audit_value = _sha256(
                audit_fingerprint,
                f"{path}.sealed_audit_certification_fingerprint",
            )
            if (
                runtime_value != runtime_certification.fingerprint
                or audit_value != sealed_audit_certification.fingerprint
            ):
                raise ValueError("Lifecycle certification fingerprint mismatch")
            if (
                previous_event is not None
                and previous_event.state != QUARANTINED_STATIC
            ):
                if (
                    runtime_value
                    != previous_event.runtime_certification_fingerprint
                    or audit_value
                    != previous_event.sealed_audit_certification_fingerprint
                ):
                    raise ValueError("Lifecycle cannot replace readiness certificates")

        authority_record = record["authority"]
        if state in {ADMITTED_DORMANT, ACTIVE}:
            if authority_record is None:
                raise ValueError("Admission and activation require explicit authority")
            authority = LifecycleAuthority.from_record(authority_record)
            expected_action = "admit" if state == ADMITTED_DORMANT else "activate"
            if previous_event is None or (
                authority.action != expected_action
                or authority.candidate_fingerprint != candidate_fingerprint
                or authority.predecessor_event_sha256 != previous_event.fingerprint
            ):
                raise ValueError("Lifecycle authority binding mismatch")
        else:
            if authority_record is not None:
                raise ValueError("Quarantine and readiness cannot carry authority")
            authority = None

        boundary_record = record["generation_boundary"]
        if state == ACTIVE:
            if boundary_record is None:
                raise ValueError("Activation requires a generation-boundary binding")
            boundary = GenerationBoundaryBinding.from_record(boundary_record)
            if previous_event is None or (
                boundary.candidate_fingerprint != candidate_fingerprint
                or boundary.admitted_event_sha256 != previous_event.fingerprint
            ):
                raise ValueError("Generation-boundary candidate binding mismatch")
        else:
            if boundary_record is not None:
                raise ValueError("Only ACTIVE may carry a generation boundary")
            boundary = None

        return cls(
            schema_version=1,
            candidate_fingerprint=candidate_fingerprint,
            audit_plan_fingerprint=audit_plan_fingerprint,
            sequence=sequence,
            previous_event_sha256=previous_hash_value,
            previous_state=previous_state,
            state=state,
            runtime_certification_fingerprint=runtime_value,
            sealed_audit_certification_fingerprint=audit_value,
            authority=authority,
            generation_boundary=boundary,
        )

    @classmethod
    def from_json(cls, text: str, **kwargs: Any) -> LifecycleEvent:
        return cls.from_record(strict_json_loads(text), **kwargs)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_fingerprint": self.candidate_fingerprint,
            "audit_plan_fingerprint": self.audit_plan_fingerprint,
            "sequence": self.sequence,
            "previous_event_sha256": self.previous_event_sha256,
            "previous_state": self.previous_state,
            "state": self.state,
            "runtime_certification_fingerprint": (
                self.runtime_certification_fingerprint
            ),
            "sealed_audit_certification_fingerprint": (
                self.sealed_audit_certification_fingerprint
            ),
            "authority": None if self.authority is None else self.authority.to_record(),
            "generation_boundary": (
                None
                if self.generation_boundary is None
                else self.generation_boundary.to_record()
            ),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


def transition_candidate_lifecycle(
    *,
    candidate: Any,
    plan: PreAuthoringAuditPlan,
    intent: AuthoringIntent,
    verification_context: StaticCandidateVerificationContext,
    target_state: str,
    history: Sequence[LifecycleEvent] = (),
    runtime_certification: RuntimeCertification | None = None,
    sealed_audit_certification: SealedAuditCertification | None = None,
    runtime_policy_id: str | None = None,
    runtime_policy_fingerprint: str | None = None,
    suite: SealedAuditSuite | None = None,
    policy: CapabilityClaimPolicy | None = None,
    report: CapabilityClaimReport | None = None,
    episode_results: Sequence[Any] | None = None,
    authority: LifecycleAuthority | None = None,
    generation_boundary: GenerationBoundaryBinding | None = None,
) -> LifecycleEvent:
    """Append exactly one authorized event; no transition is implicit."""
    verified_history = validate_lifecycle_chain(
        history,
        candidate=candidate,
        plan=plan,
        intent=intent,
        verification_context=verification_context,
        runtime_certification=runtime_certification,
        sealed_audit_certification=sealed_audit_certification,
        runtime_policy_id=runtime_policy_id,
        runtime_policy_fingerprint=runtime_policy_fingerprint,
        suite=suite,
        policy=policy,
        report=report,
        episode_results=episode_results,
        allow_empty=True,
    )
    previous_event = verified_history[-1] if verified_history else None
    previous_state = None if previous_event is None else previous_event.state
    if _TRANSITIONS.get(previous_state) != target_state:
        raise ValueError("Lifecycle transition skips, reverses, or follows ACTIVE")
    if target_state in {ADMITTED_DORMANT, ACTIVE} and authority is None:
        raise ValueError("Admission and activation require explicit authority")
    if target_state == ACTIVE and generation_boundary is None:
        raise ValueError("Activation requires a generation-boundary binding")
    if target_state != ACTIVE and generation_boundary is not None:
        raise ValueError("Generation boundary is only valid for activation")
    if target_state not in {ADMITTED_DORMANT, ACTIVE} and authority is not None:
        raise ValueError("Authority is only valid for admission or activation")

    verified_candidate = verification_context.verify(candidate, intent=intent)
    candidate_binding = CandidateArtifactBinding.from_candidate(verified_candidate)
    record = {
        "schema_version": 1,
        "candidate_fingerprint": candidate_binding.candidate_fingerprint,
        "audit_plan_fingerprint": plan.fingerprint,
        "sequence": 0 if previous_event is None else previous_event.sequence + 1,
        "previous_event_sha256": (
            None if previous_event is None else previous_event.fingerprint
        ),
        "previous_state": previous_state,
        "state": target_state,
        "runtime_certification_fingerprint": (
            None
            if runtime_certification is None
            else runtime_certification.fingerprint
        ),
        "sealed_audit_certification_fingerprint": (
            None
            if sealed_audit_certification is None
            else sealed_audit_certification.fingerprint
        ),
        "authority": None if authority is None else authority.to_record(),
        "generation_boundary": (
            None
            if generation_boundary is None
            else generation_boundary.to_record()
        ),
    }
    return LifecycleEvent._from_record_step(
        record,
        candidate=verified_candidate,
        plan=plan,
        intent=intent,
        verification_context=verification_context,
        previous_event=previous_event,
        runtime_certification=runtime_certification,
        sealed_audit_certification=sealed_audit_certification,
        runtime_policy_id=runtime_policy_id,
        runtime_policy_fingerprint=runtime_policy_fingerprint,
        suite=suite,
        policy=policy,
        report=report,
        episode_results=episode_results,
    )


def validate_lifecycle_chain(
    events: Sequence[LifecycleEvent],
    *,
    candidate: Any,
    plan: PreAuthoringAuditPlan,
    intent: AuthoringIntent,
    verification_context: StaticCandidateVerificationContext,
    runtime_certification: RuntimeCertification | None = None,
    sealed_audit_certification: SealedAuditCertification | None = None,
    runtime_policy_id: str | None = None,
    runtime_policy_fingerprint: str | None = None,
    suite: SealedAuditSuite | None = None,
    policy: CapabilityClaimPolicy | None = None,
    report: CapabilityClaimReport | None = None,
    episode_results: Sequence[Any] | None = None,
    allow_empty: bool = False,
) -> tuple[LifecycleEvent, ...]:
    """Strictly replay an append-only lifecycle chain from canonical records."""
    if isinstance(events, (str, bytes)):
        raise ValueError("Lifecycle chain must be an event sequence")
    if not events and not allow_empty:
        raise ValueError("Lifecycle chain must be a non-empty sequence")
    restored: list[LifecycleEvent] = []
    previous: LifecycleEvent | None = None
    for event in events:
        if not isinstance(event, LifecycleEvent):
            raise ValueError("Lifecycle chain entries must be LifecycleEvent values")
        current = LifecycleEvent._from_record_step(
            event.to_record(),
            candidate=candidate,
            plan=plan,
            intent=intent,
            verification_context=verification_context,
            previous_event=previous,
            runtime_certification=(
                runtime_certification if event.state != QUARANTINED_STATIC else None
            ),
            sealed_audit_certification=(
                sealed_audit_certification
                if event.state != QUARANTINED_STATIC
                else None
            ),
            runtime_policy_id=(
                runtime_policy_id if event.state != QUARANTINED_STATIC else None
            ),
            runtime_policy_fingerprint=(
                runtime_policy_fingerprint
                if event.state != QUARANTINED_STATIC
                else None
            ),
            suite=suite if event.state != QUARANTINED_STATIC else None,
            policy=policy if event.state != QUARANTINED_STATIC else None,
            report=report if event.state != QUARANTINED_STATIC else None,
            episode_results=(
                episode_results if event.state != QUARANTINED_STATIC else None
            ),
            expected_fingerprint=event.fingerprint,
        )
        restored.append(current)
        previous = current
    return tuple(restored)
