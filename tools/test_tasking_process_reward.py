"""Tests for fail-closed process-reward model shadow artifacts."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace

from rexpolicy.tasking.canonical import canonical_fingerprint
from rexpolicy.tasking.event_ledger import EpisodeEventLedger
from rexpolicy.tasking.process_label_view import materialize_process_label_view
from rexpolicy.tasking.process_reward import (
    CONFIDENCE_SEMANTICS,
    ORACLE_AUTHORITY,
    PROCESS_LABEL_SCHEMA_FINGERPRINT,
    PROCESS_LABEL_SCHEMA_ID,
    REWARD_ACTIVATION,
    SHADOW_MODE,
    TASK_SPEC_SCHEMA_FINGERPRINT,
    TASK_SPEC_SCHEMA_ID,
    ProcessRewardModelManifest,
    ProcessRewardShadowView,
    materialize_process_reward_shadow,
)
from rexpolicy.tasking.repository import (
    load_production_reach_artifacts,
    load_production_reach_process_artifacts,
)
from tools.test_tasking_event_ledger import ledger_record


def _artifacts():
    tasks = load_production_reach_artifacts()
    process = load_production_reach_process_artifacts(tasks)
    record = ledger_record()
    record["bindings"]["task_contract_id"] = tasks.contract.task_contract_id
    record["bindings"]["task_contract_sha256"] = tasks.contract.fingerprint
    record["bindings"]["task_oracle_sha256"] = (
        tasks.contract.oracle_fingerprint
    )
    record["events"][0]["signals"]["reach.distance_m"] = 0.12
    ledger = EpisodeEventLedger.from_record(
        record,
        event_schema=tasks.capabilities.event_schemas[0],
    )
    labels = materialize_process_label_view(
        ledger,
        task_contract=tasks.contract,
        process_contract=process.contract,
        catalog=tasks.capabilities,
    )
    return tasks, process, ledger, labels


def _manifest_record(tasks, process, labels):
    provenance = {
        "provenance_id": "reach_process_labels/v1",
        "method_id": "rexpolicy/deterministic_process_labels/v1",
        "annotation_policy_sha256": "9" * 64,
        "source_labeler_policy_id": labels.labeler_policy_id,
        "source_labeler_policy_sha256": labels.labeler_policy_sha256,
    }
    provenance_sha256 = canonical_fingerprint(provenance)
    train_sha256 = "1" * 64
    audit_sha256 = "2" * 64
    return {
        "artifact_type": "rexpolicy_process_reward_model_manifest",
        "schema_version": 1,
        "model_id": "reach_process_prm/v1",
        "model_family_id": "reach_process_prm",
        "task": {
            "task_schema_id": TASK_SPEC_SCHEMA_ID,
            "task_schema_sha256": TASK_SPEC_SCHEMA_FINGERPRINT,
            "task_id": tasks.task.task_id,
            "task_spec_sha256": tasks.task.fingerprint,
            "task_contract_id": tasks.contract.task_contract_id,
            "task_contract_sha256": tasks.contract.fingerprint,
            "task_oracle_sha256": tasks.contract.oracle_fingerprint,
        },
        "process_labels": {
            "process_label_schema_id": PROCESS_LABEL_SCHEMA_ID,
            "process_label_schema_sha256": PROCESS_LABEL_SCHEMA_FINGERPRINT,
            "process_contract_id": process.contract.process_contract_id,
            "process_contract_sha256": process.contract.fingerprint,
            "process_spec_id": process.process_spec.process_spec_id,
            "process_spec_sha256": process.process_spec.fingerprint,
            "labeler_policy_id": labels.labeler_policy_id,
            "labeler_policy_sha256": labels.labeler_policy_sha256,
        },
        "label_provenance": provenance,
        "datasets": {
            "train": {
                "split": "train",
                "dataset_id": "reach_process_train/v1",
                "example_count": 64,
                "dataset_sha256": train_sha256,
                "ordered_process_label_views_sha256": "3" * 64,
                "label_provenance_sha256": provenance_sha256,
            },
            "audit": {
                "split": "audit",
                "dataset_id": "reach_process_audit/v1",
                "example_count": 16,
                "dataset_sha256": audit_sha256,
                "ordered_process_label_views_sha256": "4" * 64,
                "label_provenance_sha256": provenance_sha256,
            },
        },
        "output_contract": {
            "score_min": -1.0,
            "score_max": 1.0,
            "confidence_semantics": CONFIDENCE_SEMANTICS,
        },
        "calibration": {
            "method_id": "rexpolicy/isotonic_calibration/v1",
            "audit_dataset_sha256": audit_sha256,
            "report_sha256": "5" * 64,
            "sample_count": 16,
            "expected_calibration_error": 0.04,
            "maximum_expected_calibration_error": 0.05,
        },
        "usage_policy": {
            "mode": SHADOW_MODE,
            "oracle_authority": ORACLE_AUTHORITY,
            "reward_activation": REWARD_ACTIVATION,
        },
    }


def _outputs(labels):
    return [
        {
            "source_sequence": label.source_sequence,
            "process_score": float(index) / 10.0,
            "calibrated_confidence": 0.8,
        }
        for index, label in enumerate(labels.transition_labels)
    ]


class TestProcessRewardManifest(unittest.TestCase):
    def test_manifest_commits_task_labels_datasets_and_provenance(self) -> None:
        tasks, process, _, labels = _artifacts()
        record = _manifest_record(tasks, process, labels)
        manifest = ProcessRewardModelManifest.from_record(record)
        restored = ProcessRewardModelManifest.from_json(manifest.to_json())

        self.assertEqual(restored, manifest)
        self.assertEqual(restored.fingerprint, manifest.fingerprint)
        self.assertEqual(manifest.task.task_spec_sha256, tasks.task.fingerprint)
        self.assertEqual(
            manifest.task.task_oracle_sha256,
            tasks.contract.oracle_fingerprint,
        )
        self.assertEqual(
            manifest.process_labels.process_contract_sha256,
            process.contract.fingerprint,
        )
        self.assertEqual(
            manifest.train_dataset.label_provenance_sha256,
            manifest.label_provenance.fingerprint,
        )
        self.assertNotEqual(
            manifest.train_dataset.dataset_sha256,
            manifest.audit_dataset.dataset_sha256,
        )

    def test_manifest_rejects_schema_provenance_and_split_drift(self) -> None:
        tasks, process, _, labels = _artifacts()
        base = _manifest_record(tasks, process, labels)

        cases = []
        changed = json.loads(json.dumps(base))
        changed["task"]["task_schema_sha256"] = "0" * 64
        cases.append((changed, "task_schema_sha256"))
        changed = json.loads(json.dumps(base))
        changed["datasets"]["train"]["label_provenance_sha256"] = "0" * 64
        cases.append((changed, "provenance"))
        changed = json.loads(json.dumps(base))
        changed["datasets"]["audit"]["dataset_sha256"] = "1" * 64
        cases.append((changed, "disjoint"))
        changed = json.loads(json.dumps(base))
        changed["datasets"]["audit"]["split"] = "train"
        cases.append((changed, "split"))

        for changed, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    ProcessRewardModelManifest.from_record(changed)

    def test_manifest_requires_a_bound_passing_calibration_audit(self) -> None:
        tasks, process, _, labels = _artifacts()
        base = _manifest_record(tasks, process, labels)

        wrong_dataset = json.loads(json.dumps(base))
        wrong_dataset["calibration"]["audit_dataset_sha256"] = "6" * 64
        with self.assertRaisesRegex(ValueError, "audit dataset fingerprint"):
            ProcessRewardModelManifest.from_record(wrong_dataset)

        partial = json.loads(json.dumps(base))
        partial["calibration"]["sample_count"] = 15
        with self.assertRaisesRegex(ValueError, "cover the audit dataset"):
            ProcessRewardModelManifest.from_record(partial)

        uncalibrated = json.loads(json.dumps(base))
        uncalibrated["calibration"]["expected_calibration_error"] = 0.11
        uncalibrated["calibration"]["maximum_expected_calibration_error"] = 0.11
        with self.assertRaisesRegex(ValueError, "between 0.0 and 0.1"):
            ProcessRewardModelManifest.from_record(uncalibrated)

    def test_direct_dataclass_drift_is_revalidated_at_use(self) -> None:
        tasks, process, _, labels = _artifacts()
        manifest = ProcessRewardModelManifest.from_record(
            _manifest_record(tasks, process, labels)
        )
        bypassed = replace(manifest, reward_activation="enabled")

        with self.assertRaisesRegex(ValueError, "shadow-only"):
            materialize_process_reward_shadow(
                labels,
                manifest=bypassed,
                checkpoint_sha256="a" * 64,
                outputs=_outputs(labels),
            )


class TestProcessRewardShadowView(unittest.TestCase):
    def _view(self):
        tasks, process, ledger, labels = _artifacts()
        manifest = ProcessRewardModelManifest.from_record(
            _manifest_record(tasks, process, labels)
        )
        view = materialize_process_reward_shadow(
            labels,
            manifest=manifest,
            checkpoint_sha256="a" * 64,
            outputs=_outputs(labels),
        )
        return tasks, process, ledger, labels, manifest, view

    def test_predictions_bind_label_view_manifest_and_checkpoint(self) -> None:
        _, _, _, labels, manifest, view = self._view()

        self.assertEqual(view.process_label_view_sha256, labels.fingerprint)
        self.assertEqual(
            view.event_ledger_sha256,
            labels.event_ledger_sha256,
        )
        self.assertEqual(view.model_manifest_sha256, manifest.fingerprint)
        self.assertEqual(view.checkpoint_sha256, "a" * 64)
        self.assertEqual(len(view.predictions), len(labels.transition_labels))
        for prediction, label in zip(view.predictions, labels.transition_labels):
            self.assertEqual(prediction.source_sequence, label.source_sequence)
            self.assertEqual(
                prediction.source_label_sha256,
                canonical_fingerprint(label.to_record()),
            )
        restored = ProcessRewardShadowView.from_json(
            view.to_json(), process_labels=labels, manifest=manifest
        )
        self.assertEqual(restored, view)

    def test_prediction_bounds_and_coverage_fail_closed(self) -> None:
        tasks, process, _, labels = _artifacts()
        manifest = ProcessRewardModelManifest.from_record(
            _manifest_record(tasks, process, labels)
        )
        cases = []
        changed = _outputs(labels)
        changed[0]["process_score"] = 1.01
        cases.append((changed, "process_score"))
        changed = _outputs(labels)
        changed[0]["calibrated_confidence"] = -0.01
        cases.append((changed, "calibrated_confidence"))
        changed = _outputs(labels)[:-1]
        cases.append((changed, "cover every"))
        changed = _outputs(labels)
        changed[0]["source_sequence"] = 999
        cases.append((changed, "source_sequence"))

        for changed, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    materialize_process_reward_shadow(
                        labels,
                        manifest=manifest,
                        checkpoint_sha256="a" * 64,
                        outputs=changed,
                    )

    def test_sidecar_tampering_cannot_change_bindings_or_add_an_oracle(self) -> None:
        _, _, _, labels, manifest, view = self._view()
        base = view.to_record()
        cases = []
        changed = json.loads(json.dumps(base))
        changed["process_label_view_sha256"] = "0" * 64
        cases.append((changed, "process_label_view_sha256"))
        changed = json.loads(json.dumps(base))
        changed["model_manifest_sha256"] = "0" * 64
        cases.append((changed, "model_manifest_sha256"))
        changed = json.loads(json.dumps(base))
        changed["predictions"][0]["source_label_sha256"] = "0" * 64
        cases.append((changed, "source_label_sha256"))
        changed = json.loads(json.dumps(base))
        changed["predictions"][0]["success"] = True
        cases.append((changed, "unknown fields"))
        changed = json.loads(json.dumps(base))
        changed["oracle_authority"] = "success"
        cases.append((changed, "oracle_authority"))

        for changed, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    ProcessRewardShadowView.from_record(
                        changed,
                        process_labels=labels,
                        manifest=manifest,
                    )

    def test_task_or_process_drift_cannot_reuse_labels(self) -> None:
        tasks, process, _, labels = _artifacts()
        record = _manifest_record(tasks, process, labels)
        record["task"]["task_contract_sha256"] = "0" * 64
        manifest = ProcessRewardModelManifest.from_record(record)

        with self.assertRaisesRegex(ValueError, "Task binding mismatch"):
            materialize_process_reward_shadow(
                labels,
                manifest=manifest,
                checkpoint_sha256="a" * 64,
                outputs=_outputs(labels),
            )

    def test_shadow_scoring_mutates_neither_ledger_nor_process_labels(self) -> None:
        _, _, ledger, labels, manifest, _ = self._view()
        ledger_before = ledger.fingerprint
        labels_before = labels.fingerprint
        view = materialize_process_reward_shadow(
            labels,
            manifest=manifest,
            checkpoint_sha256="b" * 64,
            outputs=_outputs(labels),
        )

        self.assertEqual(ledger.fingerprint, ledger_before)
        self.assertEqual(labels.fingerprint, labels_before)
        self.assertEqual(view.mode, "shadow_only")
        self.assertEqual(view.oracle_authority, "none")
        self.assertEqual(view.reward_activation, "forbidden")
        self.assertFalse(hasattr(view, "success"))
        self.assertFalse(hasattr(view, "failure"))
        self.assertFalse(hasattr(view, "safety"))
        prediction_keys = set(view.predictions[0].to_record())
        self.assertNotIn("reward", prediction_keys)
        self.assertNotIn("success", prediction_keys)
        self.assertNotIn("failure", prediction_keys)
        self.assertNotIn("safety", prediction_keys)

    def test_manifest_and_sidecar_cannot_enable_reward_application(self) -> None:
        tasks, process, _, labels = _artifacts()
        record = _manifest_record(tasks, process, labels)
        record["usage_policy"]["reward_activation"] = "enabled"
        with self.assertRaisesRegex(ValueError, "shadow-only"):
            ProcessRewardModelManifest.from_record(record)

        _, _, _, labels, manifest, view = self._view()
        record = view.to_record()
        record["reward_activation"] = "enabled"
        with self.assertRaisesRegex(ValueError, "reward_activation"):
            ProcessRewardShadowView.from_record(
                record,
                process_labels=labels,
                manifest=manifest,
            )


if __name__ == "__main__":
    unittest.main()
