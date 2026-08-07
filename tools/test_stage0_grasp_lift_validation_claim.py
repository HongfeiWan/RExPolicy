"""Tests for the one-shot Grasp-Lift validation claim boundary."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import call, patch

from rexpolicy.stage0.data.grasp_lift_validation import (
    GRASP_LIFT_LOCKED_TEST_ABSENT,
    GRASP_LIFT_VALIDATION_CLAIM_SCHEMA_ID,
    GRASP_LIFT_VALIDATION_PURPOSE,
)
from rexpolicy.stage0.data.grasp_lift_validation_claim import (
    GRASP_LIFT_VALIDATION_CLAIM_REGISTRY,
    GraspLiftValidationClaimReceipt,
    persist_grasp_lift_validation_claim,
    verify_persisted_grasp_lift_validation_claim,
)
from rexpolicy.stage0.types import canonical_fingerprint


_MODULE = "rexpolicy.stage0.data.grasp_lift_validation_claim"


def _claim(*, claim_id: str = "1" * 64) -> dict[str, Any]:
    record: dict[str, Any] = {
        "candidate_inventory_sha256": "2" * 64,
        "claim_id": claim_id,
        "cohort_identity_sha256": "3" * 64,
        "evaluator_implementation_sha256": "4" * 64,
        "locked_test": dict(GRASP_LIFT_LOCKED_TEST_ABSENT),
        "preflight_sha256": "5" * 64,
        "purpose": GRASP_LIFT_VALIDATION_PURPOSE,
        "schema_id": GRASP_LIFT_VALIDATION_CLAIM_SCHEMA_ID,
        "selection_protocol_sha256": "6" * 64,
    }
    record["self_sha256"] = canonical_fingerprint(record)
    return record


def _payload(claim: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            claim,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


class GraspLiftValidationClaimPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "worktree"
        self.common = Path(self.temporary.name) / "common.git"
        self.root.mkdir()
        self.common.mkdir()
        self.claim = _claim()
        self.verifier_calls: list[dict[str, Any]] = []

    def verifier(self, claim: Any) -> None:
        self.verifier_calls.append(dict(claim))
        if claim != self.claim:
            raise ValueError("claim does not match the fixed preflight")

    def _common_patch(self) -> Any:
        return patch(
            f"{_MODULE}._git_common_directory",
            return_value=(self.root.resolve(), self.common.resolve()),
        )

    def _persist(self) -> GraspLiftValidationClaimReceipt:
        with self._common_patch():
            return persist_grasp_lift_validation_claim(
                self.root,
                self.claim,
                verifier=self.verifier,
            )

    def test_uses_absolute_git_common_dir_and_returns_verified_receipt(self) -> None:
        responses = [
            SimpleNamespace(stdout=f"{self.root.resolve()}\n"),
            SimpleNamespace(stdout=f"{self.common.resolve()}\n"),
        ]
        with patch(f"{_MODULE}.subprocess.run", side_effect=responses) as run:
            receipt = persist_grasp_lift_validation_claim(
                self.root,
                self.claim,
                verifier=self.verifier,
            )
        self.assertEqual(
            run.call_args_list,
            [
                call(
                    [
                        "git",
                        "rev-parse",
                        "--path-format=absolute",
                        "--show-toplevel",
                    ],
                    cwd=self.root.resolve(),
                    check=True,
                    capture_output=True,
                    text=True,
                ),
                call(
                    [
                        "git",
                        "rev-parse",
                        "--path-format=absolute",
                        "--git-common-dir",
                    ],
                    cwd=self.root.resolve(),
                    check=True,
                    capture_output=True,
                    text=True,
                ),
            ],
        )
        expected_path = (
            self.common
            / GRASP_LIFT_VALIDATION_CLAIM_REGISTRY
            / f"{self.claim['claim_id']}.json"
        ).resolve()
        self.assertEqual(receipt.claim_path, expected_path)
        self.assertEqual(receipt.claim_self_sha256, self.claim["self_sha256"])
        self.assertEqual(
            receipt.claim_file_sha256,
            hashlib.sha256(_payload(self.claim)).hexdigest(),
        )
        receipt_record = receipt.to_record()
        unsigned_receipt = dict(receipt_record)
        receipt_self_sha256 = unsigned_receipt.pop("self_sha256")
        self.assertEqual(
            canonical_fingerprint(unsigned_receipt),
            receipt_self_sha256,
        )
        self.assertEqual(
            GraspLiftValidationClaimReceipt.from_record(receipt_record),
            receipt,
        )
        changed_receipt_record = dict(receipt_record)
        changed_receipt_record["claim_file_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "self_sha256"):
            GraspLiftValidationClaimReceipt.from_record(changed_receipt_record)
        self.assertEqual(expected_path.read_bytes(), _payload(self.claim))
        with self._common_patch():
            observed = verify_persisted_grasp_lift_validation_claim(
                self.root,
                self.claim,
                verifier=self.verifier,
                receipt=receipt,
            )
        self.assertEqual(observed, receipt)

    def test_invalid_record_and_verifier_failure_touch_no_filesystem(self) -> None:
        invalid = dict(self.claim)
        invalid["self_sha256"] = "0" * 64
        with (
            patch(f"{_MODULE}._git_common_directory") as locate,
            self.assertRaisesRegex(ValueError, "self_sha256"),
        ):
            persist_grasp_lift_validation_claim(
                self.root,
                invalid,
                verifier=self.verifier,
            )
        locate.assert_not_called()

        def reject(_: Any) -> None:
            raise ValueError("preflight binding failed")

        with (
            patch(f"{_MODULE}._git_common_directory") as locate,
            self.assertRaisesRegex(ValueError, "preflight binding failed"),
        ):
            persist_grasp_lift_validation_claim(
                self.root,
                self.claim,
                verifier=reject,
            )
        locate.assert_not_called()
        self.assertEqual(tuple(self.common.iterdir()), ())

    def test_claim_id_cannot_select_a_path(self) -> None:
        invalid = _claim(claim_id="../" + "7" * 61)
        with (
            patch(f"{_MODULE}._git_common_directory") as locate,
            self.assertRaisesRegex(ValueError, "claim_id"),
        ):
            persist_grasp_lift_validation_claim(
                self.root,
                invalid,
                verifier=lambda _: None,
            )
        locate.assert_not_called()

    def test_existing_claim_always_fails_without_read_or_overwrite(self) -> None:
        receipt = self._persist()
        sentinel = b"already-consumed-even-if-corrupt\n"
        receipt.claim_path.write_bytes(sentinel)
        with (
            self._common_patch(),
            patch(f"{_MODULE}._read_all") as read_all,
            self.assertRaisesRegex(FileExistsError, "already exists"),
        ):
            persist_grasp_lift_validation_claim(
                self.root,
                self.claim,
                verifier=self.verifier,
            )
        read_all.assert_not_called()
        self.assertEqual(receipt.claim_path.read_bytes(), sentinel)

    def test_concurrent_claim_has_exactly_one_winner(self) -> None:
        workers = 8
        barrier = threading.Barrier(workers)
        receipts: list[GraspLiftValidationClaimReceipt] = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        def claim_once() -> None:
            barrier.wait()
            try:
                receipt = persist_grasp_lift_validation_claim(
                    self.root,
                    self.claim,
                    verifier=lambda _: None,
                )
            except BaseException as error:
                with lock:
                    errors.append(error)
            else:
                with lock:
                    receipts.append(receipt)

        with self._common_patch():
            threads = [threading.Thread(target=claim_once) for _ in range(workers)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(len(errors), workers - 1)
        self.assertTrue(all(isinstance(error, FileExistsError) for error in errors))
        self.assertEqual(receipts[0].claim_path.read_bytes(), _payload(self.claim))

    def test_partial_write_is_retained_and_blocks_retry(self) -> None:
        partial = _payload(self.claim)[:23]

        def fail_after_prefix(descriptor: int, _: bytes) -> None:
            os.write(descriptor, partial)
            raise OSError("injected short write")

        with (
            self._common_patch(),
            patch(f"{_MODULE}._write_all", side_effect=fail_after_prefix),
            self.assertRaisesRegex(OSError, "injected short write"),
        ):
            persist_grasp_lift_validation_claim(
                self.root,
                self.claim,
                verifier=self.verifier,
            )
        claim_path = (
            self.common
            / GRASP_LIFT_VALIDATION_CLAIM_REGISTRY
            / f"{self.claim['claim_id']}.json"
        )
        self.assertEqual(claim_path.read_bytes(), partial)
        with (
            self._common_patch(),
            self.assertRaises(FileExistsError),
        ):
            persist_grasp_lift_validation_claim(
                self.root,
                self.claim,
                verifier=self.verifier,
            )
        self.assertEqual(claim_path.read_bytes(), partial)

    def test_verify_requires_exact_persisted_regular_file_and_receipt(self) -> None:
        with (
            self._common_patch(),
            self.assertRaises(FileNotFoundError),
        ):
            verify_persisted_grasp_lift_validation_claim(
                self.root,
                self.claim,
                verifier=self.verifier,
            )

        receipt = self._persist()
        receipt.claim_path.write_bytes(_payload({**self.claim, "purpose": "changed"}))
        with (
            self._common_patch(),
            self.assertRaisesRegex(ValueError, "non-canonical or changed"),
        ):
            verify_persisted_grasp_lift_validation_claim(
                self.root,
                self.claim,
                verifier=self.verifier,
                receipt=receipt,
            )

        receipt.claim_path.unlink()
        receipt.claim_path.symlink_to(self.root)
        with (
            self._common_patch(),
            self.assertRaises(OSError),
        ):
            verify_persisted_grasp_lift_validation_claim(
                self.root,
                self.claim,
                verifier=self.verifier,
            )

    def test_changed_receipt_is_rejected_before_opening_claim(self) -> None:
        receipt = self._persist()
        changed = GraspLiftValidationClaimReceipt(
            repository_root=receipt.repository_root,
            git_common_directory=receipt.git_common_directory,
            claim_path=receipt.claim_path,
            claim_id=receipt.claim_id,
            claim_self_sha256="f" * 64,
            claim_file_sha256=receipt.claim_file_sha256,
        )
        with (
            self._common_patch(),
            patch(f"{_MODULE}._read_all") as read_all,
            self.assertRaisesRegex(ValueError, "receipt changed"),
        ):
            verify_persisted_grasp_lift_validation_claim(
                self.root,
                self.claim,
                verifier=self.verifier,
                receipt=changed,
            )
        read_all.assert_not_called()


if __name__ == "__main__":
    unittest.main()
