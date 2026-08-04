"""Strict provider/model attestation for automatic authoring responses.

Production proposers return a canonical envelope containing one raw TaskSpec
object and a detached, provider-facing receipt.  The receipt binds the exact
request, TaskSpec response, wrapper command, authoring session, provider, and
actual model version.  Signature backends are deliberately small protocols so
the core module remains importable with ``python -S``; the Ed25519 adapter loads
``cryptography`` only when verification is requested.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .canonical import canonical_fingerprint, canonical_json, strict_json_loads

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_ENVELOPE_KEYS = {"schema_version", "task_spec", "provider_receipt"}
_RECEIPT_KEYS = {
    "schema_version",
    "provider_id",
    "model_id",
    "model_version",
    "request_id",
    "request_sha256",
    "response_sha256",
    "wrapper_command_fingerprint",
    "session_fingerprint",
    "verifier_id",
    "signature_algorithm",
    "signature",
}


class ProviderAttestationError(ValueError):
    """The proposer response lacks the exact trusted provider proof."""


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ProviderAttestationError(f"{path} must be a versioned identifier")
    return value


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ProviderAttestationError(f"{path} must be a lowercase SHA-256")
    return value


def _strict_keys(record: dict[str, Any], expected: set[str], path: str) -> None:
    if set(record) != expected:
        raise ProviderAttestationError(f"{path} fields do not match the protocol")


def _signature_bytes(value: Any) -> bytes:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ProviderAttestationError("Provider receipt signature is invalid")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ProviderAttestationError(
            "Provider receipt signature is not canonical base64"
        ) from error
    if base64.b64encode(decoded).decode("ascii") != value:
        raise ProviderAttestationError("Provider receipt signature is not canonical")
    return decoded


@runtime_checkable
class DetachedSignatureVerifier(Protocol):
    @property
    def verifier_id(self) -> str: ...

    @property
    def algorithm_id(self) -> str: ...

    @property
    def fingerprint(self) -> str: ...

    def to_record(self) -> dict[str, Any]: ...

    def verify(self, message: bytes, signature: bytes) -> None: ...


@dataclass(frozen=True)
class HMACSHA256Verifier:
    """Pure-stdlib verifier for trusted local wrappers and deterministic tests."""

    verifier_id: str
    secret_key: bytes

    @property
    def algorithm_id(self) -> str:
        return "hmac-sha256/v1"

    def _canonical_key(self) -> bytes:
        _identifier(self.verifier_id, "$verifier.verifier_id")
        if not isinstance(self.secret_key, bytes) or len(self.secret_key) < 32:
            raise ValueError("HMAC verifier key must contain at least 32 bytes")
        return bytes(self.secret_key)

    def to_record(self) -> dict[str, Any]:
        key = self._canonical_key()
        return {
            "schema_version": 1,
            "verifier_id": self.verifier_id,
            "algorithm_id": self.algorithm_id,
            "key_sha256": hashlib.sha256(key).hexdigest(),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())

    def sign_for_test(self, message: bytes) -> bytes:
        return hmac.new(self._canonical_key(), message, hashlib.sha256).digest()

    def verify(self, message: bytes, signature: bytes) -> None:
        expected = self.sign_for_test(message)
        if not hmac.compare_digest(expected, signature):
            raise ProviderAttestationError("Provider receipt signature is invalid")


@dataclass(frozen=True)
class Ed25519Verifier:
    """Pinned Ed25519 public key with a lazy optional-cryptography adapter."""

    verifier_id: str
    public_key: bytes

    @property
    def algorithm_id(self) -> str:
        return "ed25519/v1"

    def _canonical_key(self) -> bytes:
        _identifier(self.verifier_id, "$verifier.verifier_id")
        if not isinstance(self.public_key, bytes) or len(self.public_key) != 32:
            raise ValueError("Ed25519 public key must contain exactly 32 bytes")
        return bytes(self.public_key)

    def to_record(self) -> dict[str, Any]:
        key = self._canonical_key()
        return {
            "schema_version": 1,
            "verifier_id": self.verifier_id,
            "algorithm_id": self.algorithm_id,
            "public_key_base64": base64.b64encode(key).decode("ascii"),
            "public_key_sha256": hashlib.sha256(key).hexdigest(),
        }

    @classmethod
    def from_record(cls, value: Any) -> Ed25519Verifier:
        if not isinstance(value, dict):
            raise ValueError("$verifier must be an object")
        expected = {
            "schema_version",
            "verifier_id",
            "algorithm_id",
            "public_key_base64",
            "public_key_sha256",
        }
        if set(value) != expected or value["schema_version"] != 1:
            raise ValueError("Ed25519 verifier record is not strict schema v1")
        if value["algorithm_id"] != "ed25519/v1":
            raise ValueError("Ed25519 verifier algorithm mismatch")
        try:
            key = base64.b64decode(value["public_key_base64"], validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("Ed25519 public key is not canonical base64") from error
        verifier = cls(verifier_id=value["verifier_id"], public_key=key)
        if verifier.to_record() != value:
            raise ValueError("Ed25519 verifier record or fingerprint drifted")
        return verifier

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())

    def verify(self, message: bytes, signature: bytes) -> None:
        try:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey,
            )
        except ImportError as error:
            raise RuntimeError(
                "Ed25519 verification requires the optional cryptography package"
            ) from error
        try:
            Ed25519PublicKey.from_public_bytes(self._canonical_key()).verify(
                signature,
                message,
            )
        except InvalidSignature as error:
            raise ProviderAttestationError(
                "Provider receipt signature is invalid"
            ) from error


@dataclass(frozen=True)
class ProviderAttestationPolicy:
    schema_version: int
    policy_id: str
    provider_id: str
    model_id: str
    model_version: str
    signature_verifier: DetachedSignatureVerifier

    @classmethod
    def from_ed25519_record(cls, value: Any) -> ProviderAttestationPolicy:
        if not isinstance(value, dict):
            raise ValueError("$provider_policy must be an object")
        expected = {
            "schema_version",
            "policy_id",
            "provider_id",
            "model_id",
            "model_version",
            "signature_verifier",
            "signature_verifier_fingerprint",
        }
        if set(value) != expected or value["schema_version"] != 1:
            raise ValueError("Provider attestation policy is not strict schema v1")
        verifier = Ed25519Verifier.from_record(value["signature_verifier"])
        if value["signature_verifier_fingerprint"] != verifier.fingerprint:
            raise ValueError("Provider signature-verifier fingerprint drifted")
        policy = cls(
            schema_version=1,
            policy_id=value["policy_id"],
            provider_id=value["provider_id"],
            model_id=value["model_id"],
            model_version=value["model_version"],
            signature_verifier=verifier,
        )
        if policy.to_record() != value:
            raise ValueError("Provider attestation policy record drifted")
        return policy

    def to_record(self) -> dict[str, Any]:
        if self.schema_version != 1:
            raise ValueError("Provider attestation policy schema_version must be 1")
        _identifier(self.policy_id, "$provider_policy.policy_id")
        _identifier(self.provider_id, "$provider_policy.provider_id")
        _identifier(self.model_id, "$provider_policy.model_id")
        _identifier(self.model_version, "$provider_policy.model_version")
        verifier = self.signature_verifier
        if not isinstance(verifier, DetachedSignatureVerifier):
            raise TypeError("signature_verifier does not implement the verifier protocol")
        _identifier(verifier.verifier_id, "$provider_policy.verifier_id")
        _identifier(verifier.algorithm_id, "$provider_policy.algorithm_id")
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "signature_verifier": verifier.to_record(),
            "signature_verifier_fingerprint": verifier.fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class ProviderReceipt:
    provider_id: str
    model_id: str
    model_version: str
    request_id: str
    request_sha256: str
    response_sha256: str
    wrapper_command_fingerprint: str
    session_fingerprint: str
    verifier_id: str
    signature_algorithm: str
    signature: str

    def unsigned_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "request_id": self.request_id,
            "request_sha256": self.request_sha256,
            "response_sha256": self.response_sha256,
            "wrapper_command_fingerprint": self.wrapper_command_fingerprint,
            "session_fingerprint": self.session_fingerprint,
            "verifier_id": self.verifier_id,
            "signature_algorithm": self.signature_algorithm,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.unsigned_record(), "signature": self.signature}

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class VerifiedProviderResponse:
    task_spec_record: dict[str, Any]
    task_spec_bytes: bytes
    receipt: ProviderReceipt


def receipt_signature_message(unsigned_receipt: dict[str, Any]) -> bytes:
    return canonical_json(
        {
            "domain": "rexpolicy.provider_receipt_signature/v1",
            "receipt": unsigned_receipt,
        }
    ).encode("ascii")


def build_signed_envelope_for_test(
    *,
    task_spec: dict[str, Any],
    policy: ProviderAttestationPolicy,
    request_id: str,
    request_sha256: str,
    wrapper_command_fingerprint: str,
    session_fingerprint: str,
) -> dict[str, Any]:
    """Build a deterministic local HMAC envelope; never used by production CLI."""
    verifier = policy.signature_verifier
    if not isinstance(verifier, HMACSHA256Verifier):
        raise TypeError("The local test envelope helper requires HMACSHA256Verifier")
    response_sha256 = hashlib.sha256(
        canonical_json(task_spec).encode("ascii")
    ).hexdigest()
    receipt = ProviderReceipt(
        provider_id=policy.provider_id,
        model_id=policy.model_id,
        model_version=policy.model_version,
        request_id=request_id,
        request_sha256=request_sha256,
        response_sha256=response_sha256,
        wrapper_command_fingerprint=wrapper_command_fingerprint,
        session_fingerprint=session_fingerprint,
        verifier_id=verifier.verifier_id,
        signature_algorithm=verifier.algorithm_id,
        signature="",
    )
    signature = verifier.sign_for_test(
        receipt_signature_message(receipt.unsigned_record())
    )
    return {
        "schema_version": 1,
        "task_spec": task_spec,
        "provider_receipt": {
            **receipt.unsigned_record(),
            "signature": base64.b64encode(signature).decode("ascii"),
        },
    }


def verify_provider_envelope(
    raw_response: bytes | str,
    *,
    policy: ProviderAttestationPolicy,
    request_sha256: str,
    wrapper_command_fingerprint: str,
    session_fingerprint: str,
) -> VerifiedProviderResponse:
    """Strictly parse and authenticate one complete provider response envelope."""
    policy_record = policy.to_record()
    if isinstance(raw_response, str):
        payload = raw_response.encode("utf-8")
    elif isinstance(raw_response, bytes):
        payload = bytes(raw_response)
    else:
        raise TypeError("Provider response must be bytes or text")
    try:
        text = payload.decode("utf-8")
        envelope = strict_json_loads(text, max_bytes=len(payload) or 1)
    except (TypeError, UnicodeError, ValueError) as error:
        raise ProviderAttestationError(
            "Provider response is not one strict JSON envelope"
        ) from error
    if not isinstance(envelope, dict):
        raise ProviderAttestationError("Provider response envelope must be an object")
    _strict_keys(envelope, _ENVELOPE_KEYS, "$provider_envelope")
    if envelope["schema_version"] != 1:
        raise ProviderAttestationError("Provider response envelope schema mismatch")
    if payload != canonical_json(envelope).encode("ascii"):
        raise ProviderAttestationError("Provider response envelope is not canonical")
    task_spec = envelope["task_spec"]
    if not isinstance(task_spec, dict):
        raise ProviderAttestationError("Provider TaskSpec payload must be an object")
    task_spec_bytes = canonical_json(task_spec).encode("ascii")
    response_sha256 = hashlib.sha256(task_spec_bytes).hexdigest()
    raw_receipt = envelope["provider_receipt"]
    if not isinstance(raw_receipt, dict):
        raise ProviderAttestationError("Provider receipt must be an object")
    _strict_keys(raw_receipt, _RECEIPT_KEYS, "$provider_receipt")
    if raw_receipt["schema_version"] != 1:
        raise ProviderAttestationError("Provider receipt schema mismatch")
    receipt = ProviderReceipt(
        provider_id=_identifier(raw_receipt["provider_id"], "$receipt.provider_id"),
        model_id=_identifier(raw_receipt["model_id"], "$receipt.model_id"),
        model_version=_identifier(
            raw_receipt["model_version"], "$receipt.model_version"
        ),
        request_id=(
            raw_receipt["request_id"]
            if isinstance(raw_receipt["request_id"], str)
            and _REQUEST_ID.fullmatch(raw_receipt["request_id"])
            else ""
        ),
        request_sha256=_sha256(raw_receipt["request_sha256"], "$receipt.request_sha256"),
        response_sha256=_sha256(
            raw_receipt["response_sha256"], "$receipt.response_sha256"
        ),
        wrapper_command_fingerprint=_sha256(
            raw_receipt["wrapper_command_fingerprint"],
            "$receipt.wrapper_command_fingerprint",
        ),
        session_fingerprint=_sha256(
            raw_receipt["session_fingerprint"], "$receipt.session_fingerprint"
        ),
        verifier_id=_identifier(raw_receipt["verifier_id"], "$receipt.verifier_id"),
        signature_algorithm=_identifier(
            raw_receipt["signature_algorithm"], "$receipt.signature_algorithm"
        ),
        signature=raw_receipt["signature"],
    )
    if not receipt.request_id:
        raise ProviderAttestationError("Provider request ID is invalid")
    expected = {
        "provider_id": policy_record["provider_id"],
        "model_id": policy_record["model_id"],
        "model_version": policy_record["model_version"],
        "request_sha256": _sha256(request_sha256, "$expected.request_sha256"),
        "response_sha256": response_sha256,
        "wrapper_command_fingerprint": _sha256(
            wrapper_command_fingerprint,
            "$expected.wrapper_command_fingerprint",
        ),
        "session_fingerprint": _sha256(
            session_fingerprint,
            "$expected.session_fingerprint",
        ),
        "verifier_id": policy.signature_verifier.verifier_id,
        "signature_algorithm": policy.signature_verifier.algorithm_id,
    }
    actual = receipt.unsigned_record()
    for name, value in expected.items():
        if actual[name] != value:
            raise ProviderAttestationError(
                f"Provider receipt {name} does not match the pinned job"
            )
    signature = _signature_bytes(receipt.signature)
    policy.signature_verifier.verify(
        receipt_signature_message(receipt.unsigned_record()),
        signature,
    )
    return VerifiedProviderResponse(
        task_spec_record=dict(task_spec),
        task_spec_bytes=task_spec_bytes,
        receipt=receipt,
    )
