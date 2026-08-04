from __future__ import annotations

import copy
import hashlib
import multiprocessing
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rexpolicy.tasking import authoring_lifecycle_store as lifecycle_store_module
from rexpolicy.tasking.authoring_lifecycle_store import (
    AdmissionLifecycleReplay,
    AuthoringLifecycleStore,
    DetachedSignature,
    Ed25519DetachedVerifier,
    FileLifecycleAnchor,
    LifecycleCASMismatch,
    LifecycleJournalEntry,
    LifecycleRollbackDetected,
    LifecycleStoreError,
    LifecycleStoreManifest,
)
from rexpolicy.tasking.canonical import canonical_fingerprint, canonical_json
from tools import test_tasking_authoring_admission as admission_fixtures


class _HashVerifier:
    def __init__(self, keys: dict[str, bytes]) -> None:
        self.keys = dict(keys)

    def verify(self, *, key_id: str, payload: bytes, signature: bytes) -> bool:
        secret = self.keys.get(key_id)
        return secret is not None and signature == hashlib.sha256(secret + payload).digest()


class _FailOnceAnchor:
    def __init__(self, delegate: FileLifecycleAnchor) -> None:
        self.delegate = delegate
        self.fail_next = False

    def read(self):
        return self.delegate.read()

    def compare_and_set(self, *, expected, replacement):
        if self.fail_next:
            self.fail_next = False
            raise LifecycleCASMismatch("injected external anchor CAS failure")
        self.delegate.compare_and_set(expected=expected, replacement=replacement)


def _canonical_write(path: Path, record: dict) -> None:
    path.write_text(canonical_json(record), encoding="ascii")


class TestAuthoringLifecycleStore(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = admission_fixtures.AdmissionFixture(methodName="runTest")
        self.fixture.setUp()
        self.events = self.fixture.lifecycle()
        self.temporary = tempfile.TemporaryDirectory()
        base = Path(self.temporary.name)
        self.root = base / "mutable-store"
        self.anchor_path = base / "independent-anchor" / "candidate.json"
        self.keys = {
            "keys/admission/v1": b"admission-secret",
            "keys/admission/v2": b"alternate-admission-secret",
            "keys/generation/v1": b"generation-secret",
        }
        evidence = self.fixture.evidence()
        self.evidence_fingerprints = {
            "runtime_certification": evidence["runtime_certification"].fingerprint,
            "sealed_audit_certification": evidence[
                "sealed_audit_certification"
            ].fingerprint,
            "runtime_policy": evidence["runtime_policy_fingerprint"],
            "sealed_audit_suite": canonical_fingerprint(
                evidence["suite"].to_record()
            ),
            "capability_claim_policy": canonical_fingerprint(
                evidence["policy"].to_record()
            ),
            "capability_claim_report": canonical_fingerprint(
                evidence["report"].to_record()
            ),
            "episode_results": canonical_fingerprint(
                [item.to_record() for item in evidence["episode_results"]]
            ),
        }
        self.manifest = LifecycleStoreManifest.issue(
            store_id="reach_green_cap/v3/admission_lifecycle/v1",
            candidate=self.fixture.candidate,
            audit_plan_fingerprint=self.fixture.plan.fingerprint,
            authoring_intent_fingerprint=self.fixture.intent.fingerprint,
            evidence_fingerprints=self.evidence_fingerprints,
            trusted_signers={
                "lifecycle_authority": (
                    "keys/admission/v1",
                    "keys/admission/v2",
                ),
                "generation_boundary": ("keys/generation/v1",),
            },
        )
        self.replay = AdmissionLifecycleReplay(
            candidate=self.fixture.candidate,
            plan=self.fixture.plan,
            intent=self.fixture.intent,
            verification_context=self.fixture.verification_context,
            **evidence,
        )
        self.store = AuthoringLifecycleStore(
            self.root,
            replay=self.replay,
            signature_verifier=_HashVerifier(self.keys),
            anchor_store=FileLifecycleAnchor(self.anchor_path),
        )
        self.store.initialize(self.manifest)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _signature(self, event, head, purpose, key_id):
        payload = self.store.signing_payload(
            manifest=self.manifest,
            event=event,
            expected_head=head,
            purpose=purpose,
            key_id=key_id,
        )
        return DetachedSignature.encode(
            purpose=purpose,
            key_id=key_id,
            payload=payload,
            signature=hashlib.sha256(self.keys[key_id] + payload).digest(),
        )

    def _append_through(self, last_index: int):
        snapshot = self.store.load(self.manifest)
        for event in self.events[: last_index + 1]:
            head = snapshot.head_fingerprint
            signatures = []
            if event.state == "ADMITTED_DORMANT":
                signatures.append(
                    self._signature(
                        event,
                        head,
                        "lifecycle_authority",
                        "keys/admission/v1",
                    )
                )
            elif event.state == "ACTIVE":
                signatures.extend(
                    (
                        self._signature(
                            event,
                            head,
                            "lifecycle_authority",
                            "keys/admission/v1",
                        ),
                        self._signature(
                            event,
                            head,
                            "generation_boundary",
                            "keys/generation/v1",
                        ),
                    )
                )
            snapshot = self.store.append(
                manifest=self.manifest,
                event=event,
                expected_head=head,
                signatures=signatures,
            )
        return snapshot

    def test_manifest_binds_candidate_intent_plan_and_complete_evidence(self) -> None:
        restored = LifecycleStoreManifest.from_record(self.manifest.to_record())
        self.assertEqual(restored, self.manifest)
        changed = copy.deepcopy(self.manifest.to_record())
        changed["evidence_fingerprints"]["episode_results"] = "f" * 64
        with self.assertRaisesRegex(LifecycleStoreError, "manifest"):
            self.store.load(LifecycleStoreManifest.from_record(changed))

        incomplete = copy.deepcopy(self.manifest.to_record())
        del incomplete["evidence_fingerprints"]["sealed_audit_suite"]
        with self.assertRaisesRegex(LifecycleStoreError, "evidence set"):
            LifecycleStoreManifest.from_record(incomplete)

    def test_every_replay_context_binding_is_cross_checked_before_writing(self) -> None:
        mutations = [
            ("candidate_fingerprint", None),
            ("audit_plan_fingerprint", None),
            ("authoring_intent_fingerprint", None),
            *[("evidence_fingerprints", name) for name in sorted(self.evidence_fingerprints)],
        ]
        for index, (field, nested) in enumerate(mutations):
            with self.subTest(field=field, nested=nested):
                changed = copy.deepcopy(self.manifest.to_record())
                if nested is None:
                    changed[field] = "f" * 64
                else:
                    changed[field][nested] = "f" * 64
                manifest = LifecycleStoreManifest.from_record(changed)
                root = Path(self.temporary.name) / f"wrong-context-{index}"
                store = AuthoringLifecycleStore(
                    root,
                    replay=self.replay,
                    signature_verifier=_HashVerifier(self.keys),
                    anchor_store=FileLifecycleAnchor(
                        Path(self.temporary.name) / f"wrong-anchor-{index}.json"
                    ),
                )
                with self.assertRaisesRegex(
                    LifecycleStoreError,
                    "replay context differs",
                ):
                    store.initialize(manifest)
                self.assertFalse(root.exists())

    def test_signer_roles_must_be_disjoint(self) -> None:
        changed = copy.deepcopy(self.manifest.to_record())
        changed["trusted_signers"]["generation_boundary"] = [
            "keys/admission/v1"
        ]
        with self.assertRaisesRegex(LifecycleStoreError, "disjoint"):
            LifecycleStoreManifest.from_record(changed)

    def test_contiguous_chain_replays_and_reaches_one_active_head(self) -> None:
        snapshot = self._append_through(3)
        self.assertEqual([entry.sequence for entry in snapshot.entries], [0, 1, 2, 3])
        self.assertEqual(snapshot.entries[-1].event.state, "ACTIVE")
        self.assertEqual(self.store.load(self.manifest), snapshot)

    def test_stale_cas_and_same_sequence_second_head_are_rejected(self) -> None:
        snapshot = self._append_through(0)
        stale = None
        with self.assertRaisesRegex(LifecycleCASMismatch, "stale"):
            self.store.append(
                manifest=self.manifest,
                event=self.events[1],
                expected_head=stale,
            )
        snapshot = self.store.append(
            manifest=self.manifest,
            event=self.events[1],
            expected_head=snapshot.head_fingerprint,
        )
        with self.assertRaisesRegex(LifecycleCASMismatch, "stale"):
            self.store.append(
                manifest=self.manifest,
                event=self.events[1],
                expected_head=snapshot.entries[0].fingerprint,
            )

    def test_tamper_and_cross_candidate_replay_fail_closed(self) -> None:
        self._append_through(1)
        event_path = self.root / "journal" / "00000000000000000001.json"
        changed = copy.deepcopy(
            __import__("json").loads(event_path.read_text(encoding="ascii"))
        )
        changed["lifecycle_event"]["candidate_fingerprint"] = "f" * 64
        changed["lifecycle_event_fingerprint"] = canonical_fingerprint(
            changed["lifecycle_event"]
        )
        _canonical_write(event_path, changed)
        with self.assertRaises(LifecycleStoreError):
            self.store.load(self.manifest)

    def test_forged_signature_and_signature_replay_across_head_fail(self) -> None:
        snapshot = self._append_through(1)
        event = self.events[2]
        valid = self._signature(
            event,
            snapshot.head_fingerprint,
            "lifecycle_authority",
            "keys/admission/v1",
        )
        forged_record = valid.to_record()
        forged_record["signature_base64"] = DetachedSignature.encode(
            purpose="lifecycle_authority",
            key_id="keys/admission/v1",
            payload=b"{}",
            signature=b"forged",
        ).signature_base64
        with self.assertRaisesRegex(LifecycleStoreError, "verification failed"):
            self.store.append(
                manifest=self.manifest,
                event=event,
                expected_head=snapshot.head_fingerprint,
                signatures=(DetachedSignature.from_record(forged_record),),
            )

        wrong_head_signature = self._signature(
            event,
            snapshot.entries[0].fingerprint,
            "lifecycle_authority",
            "keys/admission/v1",
        )
        with self.assertRaisesRegex(LifecycleStoreError, "payload binding"):
            self.store.append(
                manifest=self.manifest,
                event=event,
                expected_head=snapshot.head_fingerprint,
                signatures=(wrong_head_signature,),
            )

        changed_key = valid.to_record()
        changed_key["key_id"] = "keys/admission/v2"
        with self.assertRaisesRegex(LifecycleStoreError, "payload binding"):
            self.store.append(
                manifest=self.manifest,
                event=event,
                expected_head=snapshot.head_fingerprint,
                signatures=(DetachedSignature.from_record(changed_key),),
            )

    def test_anchor_cas_failure_requires_explicit_idempotent_recovery(self) -> None:
        snapshot = self._append_through(0)
        failing_anchor = _FailOnceAnchor(FileLifecycleAnchor(self.anchor_path))
        interrupted = AuthoringLifecycleStore(
            self.root,
            replay=self.replay,
            signature_verifier=_HashVerifier(self.keys),
            anchor_store=failing_anchor,
        )
        failing_anchor.fail_next = True
        with self.assertRaisesRegex(LifecycleCASMismatch, "injected"):
            interrupted.append(
                manifest=self.manifest,
                event=self.events[1],
                expected_head=snapshot.head_fingerprint,
            )
        with self.assertRaisesRegex(LifecycleRollbackDetected, "unanchored"):
            self.store.load(self.manifest)

        orphan = self.root / ".00000000000000000001.json.partial-crash"
        orphan.write_bytes(b"fully-written-but-never-published")
        recovered = self.store.recover_pending_tail(manifest=self.manifest)
        self.assertEqual(recovered.entries[-1].event, self.events[1])
        self.assertEqual(
            self.store.recover_pending_tail(manifest=self.manifest),
            recovered,
        )
        self.assertEqual(self.store.load(self.manifest), recovered)

    def test_root_staging_residue_does_not_enter_journal_enumeration(self) -> None:
        snapshot = self._append_through(0)
        orphan = self.root / ".00000000000000000001.json.partial-crash"
        orphan.write_bytes(b"orphaned staging inode")
        self.assertEqual(self.store.load(self.manifest), snapshot)

    def test_file_anchor_inside_lifecycle_root_is_rejected(self) -> None:
        root = Path(self.temporary.name) / "unsafe-co-located-store"
        store = AuthoringLifecycleStore(
            root,
            replay=self.replay,
            signature_verifier=_HashVerifier(self.keys),
            anchor_store=FileLifecycleAnchor(root / "anchor" / "watermark.json"),
        )
        with self.assertRaisesRegex(LifecycleStoreError, "outside"):
            store.initialize(self.manifest)

    def test_recovery_rejects_multiple_pending_tails(self) -> None:
        snapshot = self._append_through(0)
        failing_anchor = _FailOnceAnchor(FileLifecycleAnchor(self.anchor_path))
        interrupted = AuthoringLifecycleStore(
            self.root,
            replay=self.replay,
            signature_verifier=_HashVerifier(self.keys),
            anchor_store=failing_anchor,
        )
        failing_anchor.fail_next = True
        with self.assertRaises(LifecycleCASMismatch):
            interrupted.append(
                manifest=self.manifest,
                event=self.events[1],
                expected_head=snapshot.head_fingerprint,
            )
        first_tail_record = __import__("json").loads(
            (
                self.root / "journal" / "00000000000000000001.json"
            ).read_text(encoding="ascii")
        )
        first_tail_head = canonical_fingerprint(first_tail_record)
        signature = self._signature(
            self.events[2],
            first_tail_head,
            "lifecycle_authority",
            "keys/admission/v1",
        )
        second_tail = LifecycleJournalEntry(
            schema_version=1,
            manifest_fingerprint=self.manifest.fingerprint,
            sequence=2,
            previous_entry_sha256=first_tail_head,
            event=self.events[2],
            signatures=(signature,),
        )
        _canonical_write(
            self.root / "journal" / "00000000000000000002.json",
            second_tail.to_record(),
        )
        with self.assertRaisesRegex(
            LifecycleRollbackDetected,
            "multiple|forked",
        ):
            self.store.recover_pending_tail(manifest=self.manifest)

    def test_interrupted_immutable_write_never_publishes_partial_final(self) -> None:
        target_dir = Path(self.temporary.name) / "partial-write"
        target_dir.mkdir()
        target = target_dir / "record.json"
        original_write = os.write
        calls = 0

        def interrupted_write(descriptor, payload):
            nonlocal calls
            calls += 1
            if calls == 1:
                return original_write(descriptor, bytes(payload[:4]))
            raise OSError("injected write interruption")

        with mock.patch.object(
            lifecycle_store_module.os,
            "write",
            side_effect=interrupted_write,
        ):
            with self.assertRaisesRegex(OSError, "interruption"):
                lifecycle_store_module._write_new(
                    target,
                    {"payload": "x" * 100},
                )
        self.assertFalse(target.exists())
        self.assertEqual(list(target_dir.iterdir()), [])

    def test_truncation_and_fork_are_detected_by_independent_anchor(self) -> None:
        snapshot = self._append_through(2)
        fork_root = Path(self.temporary.name) / "fork-copy"
        shutil.copytree(self.root, fork_root)
        (self.root / "journal" / "00000000000000000002.json").unlink()
        _canonical_write(
            self.root / "head.json",
            {
                "schema_version": 1,
                "manifest_fingerprint": self.manifest.fingerprint,
                "sequence": 1,
                "head_fingerprint": snapshot.entries[1].fingerprint,
            },
        )
        with self.assertRaisesRegex(LifecycleRollbackDetected, "truncated|rolled back"):
            self.store.load(self.manifest)

        fork_event = fork_root / "journal" / "00000000000000000002.json"
        fork_record = __import__("json").loads(fork_event.read_text(encoding="ascii"))
        payload = self.store.signing_payload(
            manifest=self.manifest,
            event=self.events[2],
            expected_head=snapshot.entries[1].fingerprint,
            purpose="lifecycle_authority",
            key_id="keys/admission/v2",
        )
        fork_record["signatures"][0] = DetachedSignature.encode(
            purpose="lifecycle_authority",
            key_id="keys/admission/v2",
            payload=payload,
            signature=hashlib.sha256(
                self.keys["keys/admission/v2"] + payload
            ).digest(),
        ).to_record()
        _canonical_write(fork_event, fork_record)
        fork_head = __import__("json").loads(
            (fork_root / "head.json").read_text(encoding="ascii")
        )
        fork_head["head_fingerprint"] = canonical_fingerprint(fork_record)
        _canonical_write(fork_root / "head.json", fork_head)
        fork_store = AuthoringLifecycleStore(
            fork_root,
            replay=self.replay,
            signature_verifier=_HashVerifier(self.keys),
            anchor_store=FileLifecycleAnchor(self.anchor_path),
        )
        with self.assertRaisesRegex(LifecycleRollbackDetected, "fork"):
            fork_store.load(self.manifest)

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX process locks")
    def test_cross_process_writers_cannot_both_advance_one_head(self) -> None:
        snapshot = self._append_through(0)
        expected = snapshot.head_fingerprint
        context = multiprocessing.get_context("fork")
        start = context.Event()
        queue = context.Queue()

        def worker():
            start.wait()
            try:
                self.store.append(
                    manifest=self.manifest,
                    event=self.events[1],
                    expected_head=expected,
                )
                queue.put("ok")
            except LifecycleCASMismatch:
                queue.put("stale")

        processes = [context.Process(target=worker) for _ in range(2)]
        for process in processes:
            process.start()
        start.set()
        outcomes = [queue.get(timeout=10) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)
        self.assertCountEqual(outcomes, ["ok", "stale"])

    def test_real_ed25519_adapter_when_optional_dependency_is_available(self) -> None:
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
        except ImportError:
            self.skipTest("optional cryptography dependency is unavailable")
        private = Ed25519PrivateKey.generate()
        public = private.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        payload = b"detached lifecycle authorization"
        verifier = Ed25519DetachedVerifier({"keys/real/v1": public})
        self.assertTrue(
            verifier.verify(
                key_id="keys/real/v1",
                payload=payload,
                signature=private.sign(payload),
            )
        )
        self.assertFalse(
            verifier.verify(
                key_id="keys/real/v1",
                payload=payload + b"!",
                signature=private.sign(payload),
            )
        )


if __name__ == "__main__":
    unittest.main()
