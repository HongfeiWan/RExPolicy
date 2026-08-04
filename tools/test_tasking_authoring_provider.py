"""Tests for strict provider/model response attestation."""

from __future__ import annotations

import base64
import json
import unittest

from rexpolicy.tasking.authoring_provider import (
    Ed25519Verifier,
    HMACSHA256Verifier,
    ProviderAttestationError,
    ProviderAttestationPolicy,
    build_signed_envelope_for_test,
    verify_provider_envelope,
)
from rexpolicy.tasking.canonical import canonical_json


class TestProviderAttestation(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = HMACSHA256Verifier(
            verifier_id="rexpolicy/local_provider_verifier/v1",
            secret_key=b"local-test-provider-key-32-bytes!!",
        )
        self.policy = ProviderAttestationPolicy(
            schema_version=1,
            policy_id="rexpolicy/provider_attestation/v1",
            provider_id="local/fake_provider/v1",
            model_id="local/fake_model/v1",
            model_version="local/fake_model_build/v1",
            signature_verifier=self.verifier,
        )
        self.task = {"schema_version": 2, "task_id": "reach/v3"}
        self.request_sha256 = "1" * 64
        self.command_sha256 = "2" * 64
        self.session_sha256 = "3" * 64

    def _envelope(self) -> dict:
        return build_signed_envelope_for_test(
            task_spec=self.task,
            policy=self.policy,
            request_id="req-local-0001",
            request_sha256=self.request_sha256,
            wrapper_command_fingerprint=self.command_sha256,
            session_fingerprint=self.session_sha256,
        )

    def _verify(self, envelope):
        return verify_provider_envelope(
            canonical_json(envelope).encode("ascii"),
            policy=self.policy,
            request_sha256=self.request_sha256,
            wrapper_command_fingerprint=self.command_sha256,
            session_fingerprint=self.session_sha256,
        )

    def test_signed_envelope_binds_task_provider_model_and_runtime(self) -> None:
        verified = self._verify(self._envelope())
        self.assertEqual(verified.task_spec_record, self.task)
        self.assertEqual(verified.receipt.provider_id, self.policy.provider_id)
        self.assertEqual(verified.receipt.model_id, self.policy.model_id)
        self.assertEqual(verified.receipt.model_version, self.policy.model_version)
        self.assertEqual(verified.receipt.request_sha256, self.request_sha256)
        self.assertEqual(
            verified.receipt.wrapper_command_fingerprint,
            self.command_sha256,
        )
        self.assertEqual(verified.receipt.session_fingerprint, self.session_sha256)

    def test_missing_envelope_and_noncanonical_json_fail_closed(self) -> None:
        with self.assertRaisesRegex(ProviderAttestationError, "fields"):
            self._verify(self.task)
        pretty = json.dumps(self._envelope(), indent=2).encode("utf-8")
        with self.assertRaisesRegex(ProviderAttestationError, "not canonical"):
            verify_provider_envelope(
                pretty,
                policy=self.policy,
                request_sha256=self.request_sha256,
                wrapper_command_fingerprint=self.command_sha256,
                session_fingerprint=self.session_sha256,
            )

    def test_each_signed_runtime_binding_is_enforced(self) -> None:
        cases = {
            "provider_id": "local/other_provider/v1",
            "model_id": "local/other_model/v1",
            "model_version": "local/other_build/v1",
            "request_sha256": "4" * 64,
            "response_sha256": "5" * 64,
            "wrapper_command_fingerprint": "6" * 64,
            "session_fingerprint": "7" * 64,
            "verifier_id": "rexpolicy/other_verifier/v1",
            "signature_algorithm": "hmac-sha256/v2",
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                envelope = self._envelope()
                envelope["provider_receipt"][field] = value
                with self.assertRaises(ProviderAttestationError):
                    self._verify(envelope)

    def test_task_or_signature_tampering_fails_closed(self) -> None:
        envelope = self._envelope()
        envelope["task_spec"]["task_id"] = "reach/v4"
        with self.assertRaisesRegex(ProviderAttestationError, "response_sha256"):
            self._verify(envelope)

        envelope = self._envelope()
        signature = base64.b64decode(envelope["provider_receipt"]["signature"])
        envelope["provider_receipt"]["signature"] = base64.b64encode(
            bytes([signature[0] ^ 1]) + signature[1:]
        ).decode("ascii")
        with self.assertRaisesRegex(ProviderAttestationError, "signature"):
            self._verify(envelope)

    def test_ed25519_adapter_is_strict_and_importable_without_crypto(self) -> None:
        verifier = Ed25519Verifier(
            verifier_id="rexpolicy/provider_ed25519/v1",
            public_key=bytes(range(32)),
        )
        restored = Ed25519Verifier.from_record(verifier.to_record())
        self.assertEqual(restored, verifier)
        changed = verifier.to_record()
        changed["public_key_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "fingerprint drifted"):
            Ed25519Verifier.from_record(changed)


if __name__ == "__main__":
    unittest.main()
