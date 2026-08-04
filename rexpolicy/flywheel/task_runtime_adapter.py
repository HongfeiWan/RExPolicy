"""Explicit migration boundary from legacy flywheel tasks to TaskSpec v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rexpolicy.tasking.canonical import canonical_fingerprint
from rexpolicy.tasking.repository import (
    TaskArtifacts,
    load_production_reach_artifacts,
)

from .task_spec import FlywheelTaskSpec, REACH_GREEN_CAP_V1

_TRUSTED_LEGACY_MIGRATIONS = {
    ("reach_green_cap/v1", "reach_progress/v1"): (
        "reach_green_cap/v2",
        "reach_progress/v2",
    ),
}


@dataclass(frozen=True)
class RuntimeTaskBinding:
    """Hash-bound compatibility mapping; matching IDs alone never migrate."""

    legacy_task_id: str
    legacy_task_fingerprint: str
    legacy_reward_profile_id: str
    task_contract_id: str
    task_contract_fingerprint: str
    task_spec_fingerprint: str
    reward_profile_id: str
    reward_profile_fingerprint: str
    adapter_id: str
    adapter_fingerprint: str
    event_schema_id: str
    event_schema_fingerprint: str
    process_spec_id: str

    def to_record(self) -> dict[str, Any]:
        return {
            "legacy_task_id": self.legacy_task_id,
            "legacy_task_fingerprint": self.legacy_task_fingerprint,
            "legacy_reward_profile_id": self.legacy_reward_profile_id,
            "task_contract_id": self.task_contract_id,
            "task_contract_fingerprint": self.task_contract_fingerprint,
            "task_spec_fingerprint": self.task_spec_fingerprint,
            "reward_profile_id": self.reward_profile_id,
            "reward_profile_fingerprint": self.reward_profile_fingerprint,
            "adapter_id": self.adapter_id,
            "adapter_fingerprint": self.adapter_fingerprint,
            "event_schema_id": self.event_schema_id,
            "event_schema_fingerprint": self.event_schema_fingerprint,
            "process_spec_id": self.process_spec_id,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


def bind_legacy_task(
    legacy: FlywheelTaskSpec,
    artifacts: TaskArtifacts,
) -> RuntimeTaskBinding:
    """Require one exact trusted migration and all runtime-facing parity."""
    migration_key = (legacy.task_id, legacy.reward_profile_id)
    try:
        expected_task_id, expected_reward_id = _TRUSTED_LEGACY_MIGRATIONS[
            migration_key
        ]
    except KeyError as error:
        raise ValueError("Legacy task has no trusted TaskSpec v2 migration") from error
    task = artifacts.task
    contract = artifacts.contract
    if task.task_id != expected_task_id:
        raise ValueError("Canonical task does not match the trusted migration")
    if legacy.instruction not in task.instructions.train:
        raise ValueError("Legacy instruction is absent from the canonical train split")
    if task.environment.task_mode != legacy.environment_task_mode:
        raise ValueError("Legacy/canonical environment task mode mismatch")
    expected_projection = {
        key: tuple(mask) for key, mask in legacy.action_dimension_masks.items()
    }
    if dict(task.action_projection) != expected_projection:
        raise ValueError("Legacy/canonical action projection mismatch")
    profiles = {
        item.reward_profile_id: item for item in contract.reward_profiles
    }
    try:
        profile = profiles[expected_reward_id]
    except KeyError as error:
        raise ValueError("Canonical reward profile migration is unavailable") from error
    return RuntimeTaskBinding(
        legacy_task_id=legacy.task_id,
        legacy_task_fingerprint=legacy.fingerprint,
        legacy_reward_profile_id=legacy.reward_profile_id,
        task_contract_id=contract.task_contract_id,
        task_contract_fingerprint=contract.fingerprint,
        task_spec_fingerprint=task.fingerprint,
        reward_profile_id=profile.reward_profile_id,
        reward_profile_fingerprint=profile.fingerprint,
        adapter_id=contract.adapter_id,
        adapter_fingerprint=contract.adapter_fingerprint,
        event_schema_id=contract.event_schema_id,
        event_schema_fingerprint=contract.event_schema_fingerprint,
        process_spec_id=contract.process_spec_id,
    )


def load_production_reach_runtime_binding() -> tuple[
    TaskArtifacts,
    RuntimeTaskBinding,
]:
    """Load the only currently certified legacy-to-v2 runtime bridge."""
    artifacts = load_production_reach_artifacts()
    return artifacts, bind_legacy_task(REACH_GREEN_CAP_V1, artifacts)
