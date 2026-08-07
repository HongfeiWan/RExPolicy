"""CPU tests for the model-only Grasp-Lift validation checkpoint loader."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rexpolicy.stage0.checkpoint import (
    Stage0CheckpointHashes,
    Stage0CheckpointIntegrityError,
    Stage0CheckpointManager,
    Stage0CheckpointMismatchError,
    file_sha256,
)
from rexpolicy.stage0.evaluation.grasp_lift_checkpoint_loader import (
    GRASP_LIFT_EVALUATION_CHECKPOINT_SCHEMA_ID,
    load_grasp_lift_no_z_evaluation_checkpoint,
)
from rexpolicy.stage0.grasp_lift_no_z import (
    build_grasp_lift_no_z_policy,
    load_grasp_lift_no_z_config,
)
from rexpolicy.stage0.types import canonical_fingerprint


_MODULE = "rexpolicy.stage0.evaluation.grasp_lift_checkpoint_loader"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )


class TestGraspLiftCheckpointLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = (Path(self.temporary.name).resolve() / "run").resolve()
        self.root.mkdir()
        self.config = load_grasp_lift_no_z_config(
            Path(__file__).resolve().parents[1]
            / "configs/stage0/grasp_lift_no_z_bc.json"
        )
        torch.manual_seed(123)
        self.source_policy, _ = build_grasp_lift_no_z_policy(
            self.config,
            device="cpu",
        )
        optimizer = torch.optim.AdamW(
            (
                parameter
                for parameter in self.source_policy.parameters()
                if parameter.requires_grad
            ),
            lr=self.config.learning_rate,
        )
        manager = Stage0CheckpointManager(self.root)
        self.checkpoint = manager.save(
            sequence=500,
            phase="no_z_policy",
            steps={"global": 500, "no_z_policy": 500},
            hashes=Stage0CheckpointHashes(
                config_sha256=_digest("config"),
                state_schema_sha256=_digest("state-schema"),
                dataset_sha256=_digest("dataset"),
                split_sha256=_digest("split"),
            ),
            models={"no_z_policy": self.source_policy},
            optimizers={"no_z_policy": optimizer},
        )
        self.manifest = manager.verify(self.checkpoint)
        self.payload_path = self.checkpoint / "models/no_z_policy.pt"
        self.payload_sha256 = file_sha256(self.payload_path)
        self.schema_sha256 = self.manifest["models"]["no_z_policy"]["schema_sha256"]
        self.preflight_checkpoint = self._checkpoint_commitment()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _load(self, **changes: object) -> tuple[torch.nn.Module, dict[str, object]]:
        arguments: dict[str, object] = {
            "step": 500,
            "config": self.config,
            "device": "cpu",
            "preflight_checkpoint": self.preflight_checkpoint,
        }
        arguments.update(changes)
        return load_grasp_lift_no_z_evaluation_checkpoint(
            self.root,
            **arguments,
        )

    def _checkpoint_commitment(self, **changes: object) -> dict[str, object]:
        record: dict[str, object] = {
            "checkpoint_name": self.checkpoint.name,
            "checksums_file_sha256": file_sha256(self.checkpoint / "checksums.json"),
            "complete_file_sha256": file_sha256(self.checkpoint / "COMPLETE"),
            "manifest_file_sha256": file_sha256(self.checkpoint / "manifest.json"),
            "model_payload_sha256": file_sha256(self.payload_path),
            "model_schema_sha256": self.schema_sha256,
            "sequence": 500,
        }
        record.update(changes)
        record["checkpoint_sha256"] = canonical_fingerprint(
            {
                "checksums_file_sha256": record["checksums_file_sha256"],
                "complete_file_sha256": record["complete_file_sha256"],
                "manifest_file_sha256": record["manifest_file_sha256"],
                "model_payload_sha256": record["model_payload_sha256"],
                "sequence": record["sequence"],
            }
        )
        return record

    def _recommit_model_payload(self, payload: object) -> dict[str, object]:
        torch.save(payload, self.payload_path)
        new_payload_sha256 = file_sha256(self.payload_path)
        checksums_path = self.checkpoint / "checksums.json"
        checksums = json.loads(checksums_path.read_text(encoding="ascii"))
        checksums["files"]["models/no_z_policy.pt"] = new_payload_sha256
        _write_json(checksums_path, checksums)
        complete_path = self.checkpoint / "COMPLETE"
        complete = json.loads(complete_path.read_text(encoding="ascii"))
        complete["checksums_sha256"] = file_sha256(checksums_path)
        _write_json(complete_path, complete)
        Stage0CheckpointManager(self.root).verify(self.checkpoint)
        return self._checkpoint_commitment()

    def test_round_trip_is_eval_frozen_audited_and_model_only(self) -> None:
        original = {
            name: value.detach().clone()
            for name, value in self.source_policy.state_dict().items()
        }
        with patch(f"{_MODULE}.torch.load", wraps=torch.load) as mocked_load:
            policy, record = self._load()

        self.assertFalse(policy.training)
        self.assertEqual(mocked_load.call_count, 1)
        loaded_path = Path(mocked_load.call_args.args[0])
        self.assertEqual(loaded_path, self.payload_path)
        self.assertTrue(mocked_load.call_args.kwargs["weights_only"])
        for name, value in policy.state_dict().items():
            self.assertTrue(torch.equal(value, original[name]), name)
        frozen = [
            name
            for name, parameter in policy.named_parameters()
            if not parameter.requires_grad
        ]
        self.assertEqual(frozen, record["freeze_record"]["frozen_parameter_names"])
        self.assertEqual(
            record["schema_id"],
            GRASP_LIFT_EVALUATION_CHECKPOINT_SCHEMA_ID,
        )
        self.assertEqual(record["model"]["payload_sha256"], self.payload_sha256)
        self.assertEqual(
            record["read_scope"],
            {
                "model_payload_loaded": True,
                "optimizer_payload_loaded": False,
                "rank_payload_loaded": False,
                "validation_data_loaded": False,
            },
        )
        unsealed = dict(record)
        observed_self_sha256 = unsealed.pop("self_sha256")
        self.assertEqual(observed_self_sha256, canonical_fingerprint(unsealed))

    def test_wrong_step_and_preflight_hashes_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            Stage0CheckpointMismatchError,
            "requested step",
        ):
            self._load(step=1_000)
        with self.assertRaisesRegex(
            Stage0CheckpointMismatchError,
            "model_payload_sha256.*preflight commitment",
        ):
            self._load(
                preflight_checkpoint=self._checkpoint_commitment(
                    model_payload_sha256="0" * 64
                )
            )
        with self.assertRaisesRegex(
            Stage0CheckpointMismatchError,
            "schema.*preflight commitment",
        ):
            self._load(
                preflight_checkpoint=self._checkpoint_commitment(
                    model_schema_sha256="1" * 64
                )
            )
        with self.assertRaisesRegex(
            Stage0CheckpointMismatchError,
            "manifest_file_sha256.*preflight commitment",
        ):
            self._load(
                preflight_checkpoint=self._checkpoint_commitment(
                    manifest_file_sha256="2" * 64
                )
            )

    def test_unsafe_torch_fallback_is_never_attempted(self) -> None:
        with patch(f"{_MODULE}.torch.load", side_effect=TypeError) as mocked_load:
            with self.assertRaisesRegex(
                Stage0CheckpointIntegrityError,
                "weights_only=True",
            ):
                self._load()
        self.assertEqual(mocked_load.call_count, 1)
        self.assertTrue(mocked_load.call_args.kwargs["weights_only"])

    def test_all_checkpoint_commitments_are_rechecked_after_safe_load(self) -> None:
        from rexpolicy.stage0.evaluation import grasp_lift_checkpoint_loader

        safe_load = grasp_lift_checkpoint_loader._torch_load_model_payload

        def load_then_drift(path: Path) -> object:
            payload = safe_load(path)
            with (self.checkpoint / "COMPLETE").open("ab") as stream:
                stream.write(b"drift")
            return payload

        with patch(
            f"{_MODULE}._torch_load_model_payload",
            side_effect=load_then_drift,
        ):
            with self.assertRaisesRegex(
                Stage0CheckpointIntegrityError,
                "complete_file_sha256 changed while loading",
            ):
                self._load()

    def test_checkpoint_tampering_is_rejected_before_deserialization(self) -> None:
        with self.payload_path.open("ab") as stream:
            stream.write(b"tamper")
        with patch(f"{_MODULE}.torch.load", wraps=torch.load) as mocked_load:
            with self.assertRaisesRegex(
                Stage0CheckpointMismatchError,
                "model_payload_sha256.*preflight commitment",
            ):
                self._load()
        mocked_load.assert_not_called()

    def test_exact_payload_fields_and_symlinked_run_are_rejected(self) -> None:
        payload = torch.load(
            self.payload_path,
            map_location="cpu",
            weights_only=True,
        )
        payload["unexpected"] = "field"
        changed_commitment = self._recommit_model_payload(payload)
        with self.assertRaisesRegex(
            Stage0CheckpointIntegrityError,
            "payload fields changed",
        ):
            self._load(preflight_checkpoint=changed_commitment)

        link = self.root.parent / "run-link"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(Stage0CheckpointIntegrityError, "symlink"):
            load_grasp_lift_no_z_evaluation_checkpoint(
                link,
                step=500,
                config=self.config,
                device="cpu",
                preflight_checkpoint=changed_commitment,
            )


if __name__ == "__main__":
    unittest.main()
