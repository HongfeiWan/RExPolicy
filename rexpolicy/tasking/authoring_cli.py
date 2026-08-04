"""Strict production bundle loading and CLI execution for task authoring."""

from __future__ import annotations

import argparse
import hashlib
import stat
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from .authoring import DEFAULT_AUTHORING_POLICY, AuthoringIntent
from .authoring_admission import PreAuthoringAuditPlan
from .authoring_brief import PublicAuthoringBrief
from .authoring_orchestrator import run_automatic_authoring
from .authoring_provider import (
    Ed25519Verifier,
    ProviderAttestationPolicy,
)
from .authoring_runner import (
    DEFAULT_BUBBLEWRAP_SANDBOX_PROFILE,
    DEFAULT_PROPOSER_EXECUTION_POLICY,
    BubblewrapProposerCommand,
    ProposerCommand,
)
from .canonical import canonical_fingerprint, canonical_json, strict_json_loads
from .capabilities import CapabilityCatalog
from .contract import (
    DEFAULT_TASK_COMPILER_POLICY,
    CompiledTaskContract,
    compile_task_contract,
)
from .curriculum import CurriculumSnapshot
from .model import TaskSpecV2
from .process import ProcessSpecV1
from .process_contract import DEFAULT_PROCESS_COMPILER_POLICY
from .property_validation import DEFAULT_PROPERTY_VALIDATION_POLICY

_SHA256_CHARS = frozenset("0123456789abcdef")
_MAX_BUNDLE_BYTES = 8_000_000
_BUNDLE_FIELDS = {
    "schema_version",
    "bundle_id",
    "curriculum_snapshot",
    "expected_curriculum_snapshot_fingerprint",
    "public_brief",
    "expected_public_brief_fingerprint",
    "authoring_intent",
    "expected_authoring_intent_fingerprint",
    "preauthoring_audit_plan",
    "expected_preauthoring_audit_plan_fingerprint",
    "capability_catalog",
    "expected_capability_catalog_fingerprint",
    "process_specs",
    "expected_process_registry_fingerprint",
    "parent_task",
    "expected_parent_task_fingerprint",
    "expected_parent_contract_fingerprint",
    "expected_parent_oracle_fingerprint",
    "proposer",
    "provider_attestation_policy",
    "expected_provider_attestation_policy_fingerprint",
    "expected_policy_fingerprints",
}


def _sha256(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{path} must be a lowercase SHA-256")
    return value


def _expected(actual: str, expected: Any, path: str) -> None:
    if actual != _sha256(expected, path):
        raise ValueError(f"{path} drifted")


def _strict_object(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{path} does not match its strict schema")
    return value


def _process_registry_fingerprint(specs: tuple[ProcessSpecV1, ...]) -> str:
    return canonical_fingerprint(
        {
            "process_specs": [
                {
                    "process_spec_id": spec.process_spec_id,
                    "process_spec_fingerprint": spec.fingerprint,
                }
                for spec in specs
            ]
        }
    )


def _validate_policy_fingerprints(value: Any) -> dict[str, str]:
    expected = {
        "task_compiler_policy",
        "property_validation_policy",
        "process_compiler_policy",
        "authoring_policy",
        "proposer_execution_policy",
    }
    record = _strict_object(value, expected, "$bundle.expected_policy_fingerprints")
    actual = {
        "task_compiler_policy": DEFAULT_TASK_COMPILER_POLICY.fingerprint,
        "property_validation_policy": DEFAULT_PROPERTY_VALIDATION_POLICY.fingerprint,
        "process_compiler_policy": DEFAULT_PROCESS_COMPILER_POLICY.fingerprint,
        "authoring_policy": DEFAULT_AUTHORING_POLICY.fingerprint,
        "proposer_execution_policy": DEFAULT_PROPOSER_EXECUTION_POLICY.fingerprint,
    }
    for name, fingerprint in actual.items():
        _expected(
            fingerprint,
            record[name],
            f"$bundle.expected_policy_fingerprints.{name}",
        )
    return dict(record)


def _production_command(value: Any) -> BubblewrapProposerCommand:
    fields = {
        "command_id",
        "argv",
        "expected_proposer_command_fingerprint",
        "bubblewrap_path",
        "network_policy",
        "expected_sandbox_profile_fingerprint",
        "expected_bubblewrap_command_fingerprint",
    }
    record = _strict_object(value, fields, "$bundle.proposer")
    argv = record["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) for item in argv)
    ):
        raise ValueError("$bundle.proposer.argv must be a non-empty string array")
    command = ProposerCommand.bind(
        command_id=record["command_id"],
        argv=tuple(argv),
    )
    _expected(
        command.fingerprint,
        record["expected_proposer_command_fingerprint"],
        "$bundle.proposer.expected_proposer_command_fingerprint",
    )
    network_policy = record["network_policy"]
    if network_policy not in {"deny", "inherit"}:
        raise ValueError("Production proposer network_policy must be deny or inherit")
    profile = replace(
        DEFAULT_BUBBLEWRAP_SANDBOX_PROFILE,
        network_policy=network_policy,
    ).canonical()
    _expected(
        profile.fingerprint,
        record["expected_sandbox_profile_fingerprint"],
        "$bundle.proposer.expected_sandbox_profile_fingerprint",
    )
    bubblewrap = BubblewrapProposerCommand.bind(
        proposer_command=command,
        sandbox_profile=profile,
        bubblewrap_path=record["bubblewrap_path"],
    )
    _expected(
        bubblewrap.fingerprint,
        record["expected_bubblewrap_command_fingerprint"],
        "$bundle.proposer.expected_bubblewrap_command_fingerprint",
    )
    return bubblewrap


@dataclass(frozen=True)
class ProductionAuthoringJobBundle:
    bundle_id: str
    curriculum_snapshot: CurriculumSnapshot
    public_brief: PublicAuthoringBrief
    intent: AuthoringIntent
    preauthoring_audit_plan: PreAuthoringAuditPlan
    catalog: CapabilityCatalog
    process_specs: tuple[ProcessSpecV1, ...]
    parent_task: TaskSpecV2 | None
    parent_contract: CompiledTaskContract | None
    command: BubblewrapProposerCommand
    provider_attestation_policy: ProviderAttestationPolicy
    record: dict[str, Any]

    @classmethod
    def from_record(cls, value: Any) -> ProductionAuthoringJobBundle:
        record = _strict_object(value, _BUNDLE_FIELDS, "$bundle")
        if record["schema_version"] != 1:
            raise ValueError("Production authoring bundle schema_version must be 1")
        bundle_id = record["bundle_id"]
        if (
            not isinstance(bundle_id, str)
            or not bundle_id.startswith("rexpolicy/")
            or any(character.isspace() for character in bundle_id)
        ):
            raise ValueError("Production authoring bundle ID is invalid")

        snapshot = CurriculumSnapshot.from_record(record["curriculum_snapshot"])
        _expected(
            snapshot.fingerprint,
            record["expected_curriculum_snapshot_fingerprint"],
            "$bundle.expected_curriculum_snapshot_fingerprint",
        )
        brief = PublicAuthoringBrief.from_record(record["public_brief"])
        _expected(
            brief.fingerprint,
            record["expected_public_brief_fingerprint"],
            "$bundle.expected_public_brief_fingerprint",
        )
        if brief.source_curriculum_snapshot_fingerprint != snapshot.fingerprint:
            raise ValueError("Production brief does not bind the exact snapshot")
        intent = AuthoringIntent.from_record(record["authoring_intent"])
        _expected(
            intent.fingerprint,
            record["expected_authoring_intent_fingerprint"],
            "$bundle.expected_authoring_intent_fingerprint",
        )
        brief.validate_against(intent)

        audit_plan = PreAuthoringAuditPlan.from_record(
            record["preauthoring_audit_plan"],
            expected_fingerprint=record[
                "expected_preauthoring_audit_plan_fingerprint"
            ],
        )
        if intent.preauthoring_audit_plan_fingerprint != audit_plan.fingerprint:
            raise ValueError("Authoring intent does not bind the exact audit plan")
        if (
            audit_plan.task_id != intent.task_id
            or audit_plan.audit_instruction_count != intent.audit_commitment.count
            or audit_plan.audit_instruction_sha256 != intent.audit_commitment.sha256
        ):
            raise ValueError("Pre-authoring audit plan does not match the intent")

        catalog = CapabilityCatalog.from_record(record["capability_catalog"])
        _expected(
            catalog.fingerprint,
            record["expected_capability_catalog_fingerprint"],
            "$bundle.expected_capability_catalog_fingerprint",
        )
        if intent.capability_catalog_fingerprint != catalog.fingerprint:
            raise ValueError("Authoring intent capability catalog drifted")

        raw_specs = record["process_specs"]
        if not isinstance(raw_specs, list) or not raw_specs:
            raise ValueError("$bundle.process_specs must be a non-empty array")
        specs = tuple(ProcessSpecV1.from_record(item) for item in raw_specs)
        spec_ids = tuple(spec.process_spec_id for spec in specs)
        if spec_ids != tuple(sorted(set(spec_ids))):
            raise ValueError("Production ProcessSpecs must be sorted and unique")
        _expected(
            _process_registry_fingerprint(specs),
            record["expected_process_registry_fingerprint"],
            "$bundle.expected_process_registry_fingerprint",
        )
        if tuple(intent.allowed_process_spec_ids) != spec_ids or dict(
            intent.process_spec_fingerprints
        ) != {spec.process_spec_id: spec.fingerprint for spec in specs}:
            raise ValueError("Authoring intent ProcessSpec registry drifted")

        parent_record = record["parent_task"]
        expected_parent_fields = (
            record["expected_parent_task_fingerprint"],
            record["expected_parent_contract_fingerprint"],
            record["expected_parent_oracle_fingerprint"],
        )
        if parent_record is None:
            if any(value is not None for value in expected_parent_fields):
                raise ValueError("Parentless bundle cannot advertise parent fingerprints")
            parent_task = None
            parent_contract = None
        else:
            if any(value is None for value in expected_parent_fields):
                raise ValueError("Production parent fingerprints are incomplete")
            parent_task = TaskSpecV2.from_record(parent_record)
            _expected(
                parent_task.fingerprint,
                expected_parent_fields[0],
                "$bundle.expected_parent_task_fingerprint",
            )
            parent_contract = compile_task_contract(
                parent_task,
                catalog=catalog,
                policy=DEFAULT_TASK_COMPILER_POLICY,
            )
            _expected(
                parent_contract.fingerprint,
                expected_parent_fields[1],
                "$bundle.expected_parent_contract_fingerprint",
            )
            _expected(
                parent_contract.oracle_fingerprint,
                expected_parent_fields[2],
                "$bundle.expected_parent_oracle_fingerprint",
            )
        _validate_policy_fingerprints(record["expected_policy_fingerprints"])

        provider_policy = ProviderAttestationPolicy.from_ed25519_record(
            record["provider_attestation_policy"]
        )
        if not isinstance(provider_policy.signature_verifier, Ed25519Verifier):
            raise ValueError("Production provider verification must use Ed25519")
        _expected(
            provider_policy.fingerprint,
            record["expected_provider_attestation_policy_fingerprint"],
            "$bundle.expected_provider_attestation_policy_fingerprint",
        )
        command = _production_command(record["proposer"])
        canonical_record = strict_json_loads(canonical_json(record))
        return cls(
            bundle_id=bundle_id,
            curriculum_snapshot=snapshot,
            public_brief=brief,
            intent=intent,
            preauthoring_audit_plan=audit_plan,
            catalog=catalog,
            process_specs=specs,
            parent_task=parent_task,
            parent_contract=parent_contract,
            command=command,
            provider_attestation_policy=provider_policy,
            record=canonical_record,
        )

    def to_record(self) -> dict[str, Any]:
        return strict_json_loads(canonical_json(self.record))

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


def load_production_authoring_bundle(
    path: Path,
    *,
    expected_fingerprint: str,
) -> ProductionAuthoringJobBundle:
    resolved = Path(path).expanduser().resolve(strict=True)
    mode = resolved.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError("Production authoring bundle must be a regular file")
    payload = resolved.read_bytes()
    if len(payload) > _MAX_BUNDLE_BYTES:
        raise ValueError("Production authoring bundle exceeds the size limit")
    try:
        record = strict_json_loads(payload.decode("ascii"), max_bytes=_MAX_BUNDLE_BYTES)
    except (TypeError, UnicodeError, ValueError) as error:
        raise ValueError("Production authoring bundle is not strict JSON") from error
    if not isinstance(record, dict) or payload != canonical_json(record).encode("ascii"):
        raise ValueError("Production authoring bundle must use canonical JSON")
    bundle = ProductionAuthoringJobBundle.from_record(record)
    _expected(bundle.fingerprint, expected_fingerprint, "$expected_bundle_fingerprint")
    if hashlib.sha256(payload).hexdigest() != bundle.fingerprint:
        raise ValueError("Production authoring bundle byte fingerprint drifted")
    return bundle


def run_production_authoring_bundle(
    bundle: ProductionAuthoringJobBundle,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    if not isinstance(bundle.command, BubblewrapProposerCommand):
        raise ValueError("Production authoring requires BubblewrapProposerCommand")
    if not isinstance(
        bundle.provider_attestation_policy.signature_verifier,
        Ed25519Verifier,
    ):
        raise ValueError("Production authoring requires Ed25519 provider attestation")
    process_specs = {spec.process_spec_id: spec for spec in bundle.process_specs}
    run = run_automatic_authoring(
        intent=bundle.intent,
        brief=bundle.public_brief,
        curriculum_snapshot=bundle.curriculum_snapshot,
        catalog=bundle.catalog,
        process_specs=process_specs,
        parent_task=bundle.parent_task,
        parent_contract=bundle.parent_contract,
        command=bundle.command,
        model_id=bundle.provider_attestation_policy.model_id,
        quarantine_dir=Path(run_dir),
        provider_attestation_policy=bundle.provider_attestation_policy,
    )
    result = {
        "schema_version": 1,
        "artifact_type": "rexpolicy_production_authoring_result/v1",
        "bundle_id": bundle.bundle_id,
        "bundle_fingerprint": bundle.fingerprint,
        "provider_attestation_policy_fingerprint": (
            bundle.provider_attestation_policy.fingerprint
        ),
        "authoring_job_result": run.result.to_record(),
        "authoring_job_result_fingerprint": run.result.fingerprint,
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume one pinned production TaskSpec authoring job."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bundle = load_production_authoring_bundle(
        args.bundle,
        expected_fingerprint=args.expected_bundle_sha256,
    )
    result = run_production_authoring_bundle(bundle, run_dir=args.run_dir)
    sys.stdout.write(canonical_json(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
