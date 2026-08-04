"""Tests for the strict production authoring bundle and CLI boundary."""

from __future__ import annotations

import importlib.util
import io
import json
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock

from rexpolicy.tasking import authoring_cli
from rexpolicy.tasking.authoring import DEFAULT_AUTHORING_POLICY, AuthoringIntent
from rexpolicy.tasking.authoring_admission import PreAuthoringAuditPlan
from rexpolicy.tasking.authoring_cli import (
    ProductionAuthoringJobBundle,
    load_production_authoring_bundle,
    run_production_authoring_bundle,
)
from rexpolicy.tasking.authoring_provider import (
    Ed25519Verifier,
    ProviderAttestationPolicy,
)
from rexpolicy.tasking.authoring_runner import (
    DEFAULT_BUBBLEWRAP_SANDBOX_PROFILE,
    DEFAULT_PROPOSER_EXECUTION_POLICY,
    BubblewrapProposerCommand,
    ProposerCommand,
)
from rexpolicy.tasking.canonical import canonical_fingerprint, canonical_json
from rexpolicy.tasking.contract import DEFAULT_TASK_COMPILER_POLICY
from rexpolicy.tasking.process_contract import DEFAULT_PROCESS_COMPILER_POLICY
from rexpolicy.tasking.property_validation import DEFAULT_PROPERTY_VALIDATION_POLICY
from tools.test_tasking_authoring_job import DurableAuthoringFixture

_REAL_E2E_AVAILABLE = (
    sys.platform.startswith("linux")
    and Path("/usr/bin/bwrap").is_file()
    and Path("/usr/bin/openssl").is_file()
    and importlib.util.find_spec("cryptography") is not None
)


class ProductionBundleFixture(DurableAuthoringFixture):
    def setUp(self) -> None:
        super().setUp()
        audit = self.candidate["instructions"]["audit"]
        self.audit_plan = PreAuthoringAuditPlan.from_record(
            {
                "schema_version": 1,
                "audit_plan_id": "reach_green_cap/v3/audit_plan/v1",
                "task_id": self.candidate["task_id"],
                "audit_instruction_count": audit["count"],
                "audit_instruction_sha256": audit["sha256"],
                "episode_recipe_sha256": "e" * 64,
                "independent_training_seeds": [11, 22, 33],
                "episodes_per_training_seed": 256,
                "candidate_count": 1,
                "claim_policy_id": "paired_k1_mastery/v1",
                "claim_policy_fingerprint": "f" * 64,
            },
        )
        intent_record = self.intent.to_record()
        intent_record["preauthoring_audit_plan_fingerprint"] = (
            self.audit_plan.fingerprint
        )
        self.intent = AuthoringIntent.from_record(intent_record)

    def _fake_bwrap(self, directory: Path) -> Path:
        path = directory / "bwrap"
        path.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def _provider_policy(self, public_key: bytes | None = None):
        verifier = Ed25519Verifier(
            verifier_id="rexpolicy/local_ed25519_provider/v1",
            public_key=bytes(range(32)) if public_key is None else public_key,
        )
        return ProviderAttestationPolicy(
            schema_version=1,
            policy_id="rexpolicy/provider_attestation/v1",
            provider_id="local/fake_provider/v1",
            model_id="local/fake_model/v1",
            model_version="local/fake_model_build/v1",
            signature_verifier=verifier,
        )

    def _bundle_record(
        self,
        directory: Path,
        *,
        proposer_script: Path | None = None,
        proposer_arguments: tuple[str, ...] = (),
        public_key: bytes | None = None,
        bubblewrap_path: Path | None = None,
    ) -> dict:
        script = proposer_script or directory / "provider.py"
        if proposer_script is None:
            script.write_text("print('{}')\n", encoding="utf-8")
        proposer = ProposerCommand.bind(
            command_id="rexpolicy/production_provider_wrapper/v1",
            argv=("/usr/bin/python3", str(script), *proposer_arguments),
        )
        bwrap_path = bubblewrap_path or self._fake_bwrap(directory)
        sandbox = BubblewrapProposerCommand.bind(
            proposer_command=proposer,
            sandbox_profile=DEFAULT_BUBBLEWRAP_SANDBOX_PROFILE,
            bubblewrap_path=str(bwrap_path),
        )
        provider_policy = self._provider_policy(public_key)
        specs = tuple(self.process_specs[key] for key in sorted(self.process_specs))
        process_registry_fingerprint = canonical_fingerprint(
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
        return {
            "schema_version": 1,
            "bundle_id": "rexpolicy/reach_v3_authoring_job/v1",
            "curriculum_snapshot": self.curriculum_snapshot.to_record(),
            "expected_curriculum_snapshot_fingerprint": (
                self.curriculum_snapshot.fingerprint
            ),
            "public_brief": self.brief.to_record(),
            "expected_public_brief_fingerprint": self.brief.fingerprint,
            "authoring_intent": self.intent.to_record(),
            "expected_authoring_intent_fingerprint": self.intent.fingerprint,
            "preauthoring_audit_plan": self.audit_plan.to_record(),
            "expected_preauthoring_audit_plan_fingerprint": (
                self.audit_plan.fingerprint
            ),
            "capability_catalog": self.catalog.to_record(),
            "expected_capability_catalog_fingerprint": self.catalog.fingerprint,
            "process_specs": [spec.to_record() for spec in specs],
            "expected_process_registry_fingerprint": process_registry_fingerprint,
            "parent_task": self.parent.task.to_record(),
            "expected_parent_task_fingerprint": self.parent.task.fingerprint,
            "expected_parent_contract_fingerprint": self.parent.contract.fingerprint,
            "expected_parent_oracle_fingerprint": self.parent.contract.oracle_fingerprint,
            "proposer": {
                "command_id": proposer.command_id,
                "argv": list(proposer.argv),
                "expected_proposer_command_fingerprint": proposer.fingerprint,
                "bubblewrap_path": str(bwrap_path),
                "network_policy": "deny",
                "expected_sandbox_profile_fingerprint": (
                    DEFAULT_BUBBLEWRAP_SANDBOX_PROFILE.fingerprint
                ),
                "expected_bubblewrap_command_fingerprint": sandbox.fingerprint,
            },
            "provider_attestation_policy": provider_policy.to_record(),
            "expected_provider_attestation_policy_fingerprint": (
                provider_policy.fingerprint
            ),
            "expected_policy_fingerprints": {
                "task_compiler_policy": DEFAULT_TASK_COMPILER_POLICY.fingerprint,
                "property_validation_policy": (
                    DEFAULT_PROPERTY_VALIDATION_POLICY.fingerprint
                ),
                "process_compiler_policy": (
                    DEFAULT_PROCESS_COMPILER_POLICY.fingerprint
                ),
                "authoring_policy": DEFAULT_AUTHORING_POLICY.fingerprint,
                "proposer_execution_policy": (
                    DEFAULT_PROPOSER_EXECUTION_POLICY.fingerprint
                ),
            },
        }

    def _write_bundle(self, directory: Path, record: dict) -> tuple[Path, str]:
        path = directory / "job-bundle.json"
        payload = canonical_json(record)
        path.write_text(payload, encoding="ascii")
        return path, canonical_fingerprint(record)


class TestProductionAuthoringBundle(ProductionBundleFixture):
    def test_strict_bundle_round_trip_rebuilds_only_bubblewrap_command(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            record = self._bundle_record(directory)
            path, fingerprint = self._write_bundle(directory, record)
            bundle = load_production_authoring_bundle(
                path,
                expected_fingerprint=fingerprint,
            )
            self.assertIsInstance(bundle.command, BubblewrapProposerCommand)
            self.assertEqual(bundle.fingerprint, fingerprint)
            self.assertEqual(bundle.intent, self.intent)
            self.assertEqual(bundle.preauthoring_audit_plan, self.audit_plan)

    def test_bundle_byte_pin_and_every_embedded_pin_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            record = self._bundle_record(directory)
            path, fingerprint = self._write_bundle(directory, record)
            with self.assertRaisesRegex(ValueError, "expected_bundle_fingerprint"):
                load_production_authoring_bundle(
                    path,
                    expected_fingerprint="0" * 64,
                )

            changed = json.loads(canonical_json(record))
            changed["proposer"]["expected_proposer_command_fingerprint"] = "0" * 64
            changed_path, changed_fingerprint = self._write_bundle(directory, changed)
            with self.assertRaisesRegex(ValueError, "proposer_command_fingerprint"):
                load_production_authoring_bundle(
                    changed_path,
                    expected_fingerprint=changed_fingerprint,
                )

            changed = json.loads(canonical_json(record))
            changed["expected_provider_attestation_policy_fingerprint"] = "0" * 64
            changed_path, changed_fingerprint = self._write_bundle(directory, changed)
            with self.assertRaisesRegex(ValueError, "provider_attestation"):
                load_production_authoring_bundle(
                    changed_path,
                    expected_fingerprint=changed_fingerprint,
                )

    def test_noncanonical_bundle_and_plain_command_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            record = self._bundle_record(directory)
            path = directory / "pretty.json"
            path.write_text(json.dumps(record, indent=2), encoding="ascii")
            with self.assertRaisesRegex(ValueError, "canonical JSON"):
                load_production_authoring_bundle(
                    path,
                    expected_fingerprint=canonical_fingerprint(record),
                )

            bundle = ProductionAuthoringJobBundle.from_record(record)
            plain = bundle.command.proposer_command
            with self.assertRaisesRegex(ValueError, "Bubblewrap"):
                run_production_authoring_bundle(
                    replace(bundle, command=plain),
                    run_dir=directory / "run",
                )

    def test_cli_prints_one_exact_canonical_result_line(self) -> None:
        bundle = mock.sentinel.bundle
        result = {
            "schema_version": 1,
            "artifact_type": "rexpolicy_production_authoring_result/v1",
            "authoring_job_result": {"final_state": "quarantined_static"},
        }
        output = io.StringIO()
        with (
            mock.patch.object(
                authoring_cli,
                "load_production_authoring_bundle",
                return_value=bundle,
            ) as load,
            mock.patch.object(
                authoring_cli,
                "run_production_authoring_bundle",
                return_value=result,
            ) as run,
            redirect_stdout(output),
        ):
            exit_code = authoring_cli.main(
                [
                    "--bundle",
                    "/not/read/while/mocked.json",
                    "--expected-bundle-sha256",
                    "a" * 64,
                    "--run-dir",
                    "/also/not/created",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue(), canonical_json(result) + "\n")
        load.assert_called_once_with(
            Path("/not/read/while/mocked.json"),
            expected_fingerprint="a" * 64,
        )
        run.assert_called_once_with(bundle, run_dir=Path("/also/not/created"))


@unittest.skipUnless(
    _REAL_E2E_AVAILABLE,
    "Signed production CLI e2e requires Linux bwrap, openssl, and cryptography",
)
class TestSignedProductionAuthoringE2E(ProductionBundleFixture):
    def test_fake_ed25519_provider_runs_once_and_never_receives_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            private_key = directory / "provider_private.pem"
            public_der = directory / "provider_public.der"
            subprocess.run(
                [
                    "/usr/bin/openssl",
                    "genpkey",
                    "-algorithm",
                    "ED25519",
                    "-out",
                    str(private_key),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [
                    "/usr/bin/openssl",
                    "pkey",
                    "-in",
                    str(private_key),
                    "-pubout",
                    "-outform",
                    "DER",
                    "-out",
                    str(public_der),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            der = public_der.read_bytes()
            self.assertEqual(der[:12].hex(), "302a300506032b6570032100")
            public_key = der[12:]
            self.assertEqual(len(public_key), 32)

            script = directory / "ed25519_provider.py"
            script.write_text(
                "import base64, hashlib, json, sys\n"
                "from cryptography.hazmat.primitives.serialization "
                "import load_pem_private_key\n"
                "raw = sys.stdin.buffer.read()\n"
                "request = json.loads(raw)\n"
                "assert 'curriculum_snapshot' not in request\n"
                "assert 'evidence' not in request\n"
                "challenge = request['provider_attestation_challenge']\n"
                "policy = challenge['provider_attestation_policy']\n"
                f"task = json.loads({canonical_json(self.candidate)!r})\n"
                "encode = lambda value: json.dumps(value, sort_keys=True, "
                "separators=(',', ':'), ensure_ascii=True, allow_nan=False)\n"
                "receipt = {'schema_version': 1, "
                "'provider_id': policy['provider_id'], "
                "'model_id': policy['model_id'], "
                "'model_version': policy['model_version'], "
                "'request_id': 'req-ed25519-local-0001', "
                "'request_sha256': hashlib.sha256(raw).hexdigest(), "
                "'response_sha256': hashlib.sha256(encode(task).encode('ascii')).hexdigest(), "
                "'wrapper_command_fingerprint': challenge['wrapper_command_fingerprint'], "
                "'session_fingerprint': challenge['session_fingerprint'], "
                "'verifier_id': policy['signature_verifier']['verifier_id'], "
                "'signature_algorithm': policy['signature_verifier']['algorithm_id']}\n"
                "message = encode({'domain': 'rexpolicy.provider_receipt_signature/v1', "
                "'receipt': receipt}).encode('ascii')\n"
                "with open(sys.argv[1], 'rb') as key_file:\n"
                "    private_key = load_pem_private_key(key_file.read(), password=None)\n"
                "signature = private_key.sign(message)\n"
                "envelope = {'schema_version': 1, 'task_spec': task, "
                "'provider_receipt': {**receipt, 'signature': "
                "base64.b64encode(signature).decode('ascii')}}\n"
                "sys.stdout.write(encode(envelope))\n",
                encoding="utf-8",
            )
            record = self._bundle_record(
                directory,
                proposer_script=script,
                proposer_arguments=(str(private_key),),
                public_key=public_key,
                bubblewrap_path=Path("/usr/bin/bwrap"),
            )
            bundle = ProductionAuthoringJobBundle.from_record(record)
            first = run_production_authoring_bundle(
                bundle,
                run_dir=directory / "run",
            )
            second = run_production_authoring_bundle(
                bundle,
                run_dir=directory / "run",
            )
            self.assertEqual(first, second)
            self.assertEqual(
                first["authoring_job_result"]["final_state"],
                "quarantined_static",
            )
            request = json.loads(
                (
                    directory / "run" / "authoring-job" / "attempts"
                    / "attempt-001" / "request.json"
                ).read_text(encoding="ascii")
            )
            self.assertNotIn("curriculum_snapshot", request)
            self.assertNotIn("evidence", canonical_json(request))


if __name__ == "__main__":
    unittest.main()
