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


@dataclass(frozen=True)
class ClaimEpisodeResult:
    """One candidate K=1 result paired with its last-good safety result."""

    training_seed: int
    reset_seed: int
    diffusion_seed: int
    success: bool
    safety_failure: bool
    baseline_safety_failure: bool
    invalid_action: bool

    def validate(self) -> None:
        """Reject ambiguous episode keys and non-Boolean outcomes."""
        if any(
            type(value) is not int or value < 0
            for value in (
                self.training_seed,
                self.reset_seed,
                self.diffusion_seed,
            )
        ):
            raise ValueError("Claim episode seeds must be non-negative integers")
        if any(
            type(value) is not bool
            for value in (
                self.success,
                self.safety_failure,
                self.baseline_safety_failure,
                self.invalid_action,
            )
        ):
            raise ValueError("Claim episode outcomes must be booleans")

    @property
    def recipe(self) -> tuple[int, int]:
        """Paired reset/diffusion key shared across training seeds."""
        return self.reset_seed, self.diffusion_seed

    def to_record(self) -> dict[str, Any]:
        """Return one canonical episode-result record."""
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class SeedClaimMetrics:
    """Formal success and safety evidence for one independent policy seed."""

    training_seed: int
    episode_count: int
    success_count: int
    success_rate: float
    success_lower_bound: float
    new_safety_failure_count: int
    invalid_action_count: int


@dataclass(frozen=True)
class CapabilityClaimReport:
    """A hash-bound formal claim derived from a complete sealed audit suite."""

    schema_version: int
    claim_spec_id: str
    task_id: str
    task_spec_fingerprint: str
    audit_suite_fingerprint: str
    claim_policy_fingerprint: str
    candidate_count: int
    episode_results_sha256: str
    total_episode_count: int
    seed_metrics: tuple[SeedClaimMetrics, ...]
    accepted: bool
    reasons: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        """Return the evidence summary used by curriculum scheduling."""
        if self.schema_version != 1:
            raise ValueError("Unsupported capability claim report schema")
        for name, value in (
            ("task_spec", self.task_spec_fingerprint),
            ("audit_suite", self.audit_suite_fingerprint),
            ("claim_policy", self.claim_policy_fingerprint),
            ("episode_results", self.episode_results_sha256),
        ):
            _require_sha256(name, value)
        if self.candidate_count != 1:
            raise ValueError("Capability report must describe K=1 evaluation")
        if self.accepted == bool(self.reasons):
            raise ValueError("Capability report acceptance/reasons are inconsistent")
        record = asdict(self)
        record["seed_metrics"] = [
            asdict(metrics) for metrics in self.seed_metrics
        ]
        record["reasons"] = list(self.reasons)
        return record

    @property
    def fingerprint(self) -> str:
        """Hash the complete formal claim report."""
        return canonical_sha256(self.to_record())


def wilson_lower_bound(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> float:
    """Return the two-sided 95% Wilson success lower bound by default."""
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("Invalid successes/trials for Wilson interval")
    if not math.isfinite(z) or z <= 0.0:
        raise ValueError("Wilson z must be positive and finite")
    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    centre = proportion + z_squared / (2.0 * trials)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z_squared / (4.0 * trials * trials)
    )
    return max(0.0, (centre - margin) / denominator)


def evaluate_capability_claim(
    *,
    suite: SealedAuditSuite,
    policy: CapabilityClaimPolicy,
    episodes: list[ClaimEpisodeResult],
) -> CapabilityClaimReport:
    """Evaluate complete paired K=1 evidence without using shaping reward."""
    suite.validate_against(policy)
    expected_seeds = tuple(suite.independent_training_seeds)
    ordered = sorted(
        episodes,
        key=lambda episode: (
            episode.training_seed,
            episode.reset_seed,
            episode.diffusion_seed,
        ),
    )
    for episode in ordered:
        episode.validate()
    keys = [
        (episode.training_seed, episode.reset_seed, episode.diffusion_seed)
        for episode in ordered
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Capability claim contains duplicate episode keys")
    actual_seeds = tuple(sorted({episode.training_seed for episode in ordered}))
    if actual_seeds != expected_seeds:
        raise ValueError("Capability claim training seeds do not match audit suite")

    by_seed: dict[int, list[ClaimEpisodeResult]] = {
        seed: [] for seed in expected_seeds
    }
    for episode in ordered:
        by_seed[episode.training_seed].append(episode)
    expected_recipes: set[tuple[int, int]] | None = None
    seed_metrics = []
    reasons = []
    for seed in expected_seeds:
        seed_episodes = by_seed[seed]
        if len(seed_episodes) != suite.episodes_per_training_seed:
            raise ValueError(
                f"Training seed {seed} has incomplete capability evidence"
            )
        recipes = {episode.recipe for episode in seed_episodes}
        if len(recipes) != len(seed_episodes):
            raise ValueError(f"Training seed {seed} repeats an audit recipe")
        if expected_recipes is None:
            expected_recipes = recipes
        elif recipes != expected_recipes:
            raise ValueError("Capability claim is not paired across training seeds")
        success_count = sum(episode.success for episode in seed_episodes)
        lower_bound = wilson_lower_bound(success_count, len(seed_episodes))
        new_safety = sum(
            episode.safety_failure and not episode.baseline_safety_failure
            for episode in seed_episodes
        )
        invalid = sum(episode.invalid_action for episode in seed_episodes)
        seed_metrics.append(
            SeedClaimMetrics(
                training_seed=seed,
                episode_count=len(seed_episodes),
                success_count=success_count,
                success_rate=success_count / len(seed_episodes),
                success_lower_bound=lower_bound,
                new_safety_failure_count=new_safety,
                invalid_action_count=invalid,
            )
        )
        if lower_bound < policy.min_success_lower_bound:
            reasons.append(
                f"training seed {seed} success lower bound "
                f"{lower_bound:.6f} < {policy.min_success_lower_bound:.6f}"
            )
    total_new_safety = sum(
        metrics.new_safety_failure_count for metrics in seed_metrics
    )
    total_invalid = sum(metrics.invalid_action_count for metrics in seed_metrics)
    if total_new_safety > policy.max_new_safety_failures:
        reasons.append(
            f"new safety failures {total_new_safety} > "
            f"{policy.max_new_safety_failures}"
        )
    if total_invalid > policy.max_invalid_actions:
        reasons.append(
            f"invalid actions {total_invalid} > {policy.max_invalid_actions}"
        )
    episode_records = [episode.to_record() for episode in ordered]
    return CapabilityClaimReport(
        schema_version=1,
        claim_spec_id=policy.claim_spec_id,
        task_id=suite.task_id,
        task_spec_fingerprint=suite.task_spec_fingerprint,
        audit_suite_fingerprint=suite.fingerprint,
        claim_policy_fingerprint=policy.fingerprint,
        candidate_count=1,
        episode_results_sha256=canonical_sha256(episode_records),
        total_episode_count=len(ordered),
        seed_metrics=tuple(seed_metrics),
        accepted=not reasons,
        reasons=tuple(reasons),
    )
