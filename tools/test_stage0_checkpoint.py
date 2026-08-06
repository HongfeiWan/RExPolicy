"""Unit tests for independent Stage 0 checkpoint and DDP foundations."""

from __future__ import annotations

import hashlib
import random
import tempfile
import unittest

import numpy as np
import torch

from rexpolicy.stage0.checkpoint import (
    Stage0CheckpointHashes,
    Stage0CheckpointIntegrityError,
    Stage0CheckpointManager,
    Stage0CheckpointMismatchError,
)
from rexpolicy.stage0.distributed import (
    Stage0DistributedError,
    Stage0ProcessEnvironment,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class TestStage0Checkpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.hashes = Stage0CheckpointHashes(
            config_sha256=_digest("config"),
            state_schema_sha256=_digest("schema"),
            dataset_sha256=_digest("dataset"),
            split_sha256=_digest("split"),
        )

    @staticmethod
    def _components() -> tuple[dict[str, torch.nn.Module], dict[str, object]]:
        models = {
            "future_encoder": torch.nn.Linear(3, 4),
            "selector": torch.nn.Linear(4, 2),
        }
        optimizers = {
            name: torch.optim.AdamW(model.parameters(), lr=1.0e-3)
            for name, model in models.items()
        }
        for name, model in models.items():
            loss = model(torch.ones(2, model.in_features)).square().mean()
            loss.backward()
            optimizers[name].step()
            optimizers[name].zero_grad(set_to_none=True)
        return models, optimizers

    def test_atomic_multi_component_round_trip_and_rng(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = Stage0CheckpointManager(temporary)
            models, optimizers = self._components()
            expected = {
                group: {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                for group, model in models.items()
            }
            random.seed(11)
            np.random.seed(12)
            torch.manual_seed(13)
            checkpoint = manager.save(
                sequence=7,
                phase="future_encoder",
                steps={"global": 19, "encoder": 17},
                hashes=self.hashes,
                models=models,
                optimizers=optimizers,
                rank_state={"sampler_cursor": 5},
                extra={"normalization": "v1"},
            )
            expected_rng = (
                random.random(),
                float(np.random.random()),
                float(torch.rand(())),
            )
            for model in models.values():
                for parameter in model.parameters():
                    parameter.data.add_(10.0)
            random.seed(1)
            np.random.seed(1)
            torch.manual_seed(1)
            resumed = manager.load(
                hashes=self.hashes,
                models=models,
                optimizers=optimizers,
            )
            self.assertEqual(resumed.phase, "future_encoder")
            self.assertEqual(resumed.steps, {"encoder": 17, "global": 19})
            self.assertEqual(resumed.rank_state["sampler_cursor"], 5)
            self.assertEqual(resumed.extra["normalization"], "v1")
            for group, model in models.items():
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, expected[group][key])
            actual_rng = (
                random.random(),
                float(np.random.random()),
                float(torch.rand(())),
            )
            self.assertEqual(actual_rng, expected_rng)
            self.assertTrue((checkpoint / "COMPLETE").is_file())
            self.assertTrue((checkpoint / "manifest.json").is_file())
            self.assertTrue((checkpoint / "checksums.json").is_file())
            self.assertFalse(any(manager.checkpoints_dir.glob(".partial-*")))

    def test_hash_mismatch_and_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = Stage0CheckpointManager(temporary)
            models, optimizers = self._components()
            checkpoint = manager.save(
                sequence=1,
                phase="selector",
                steps={"global": 1},
                hashes=self.hashes,
                models=models,
                optimizers=optimizers,
            )
            changed = Stage0CheckpointHashes(
                **{**self.hashes.to_record(), "dataset_sha256": _digest("changed")}
            )
            with self.assertRaisesRegex(
                Stage0CheckpointMismatchError, "dataset_sha256"
            ):
                manager.load(
                    hashes=changed,
                    models=models,
                    optimizers=optimizers,
                    checkpoint=checkpoint,
                )
            with (checkpoint / "models/future_encoder.pt").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(
                Stage0CheckpointIntegrityError, "checksum mismatch"
            ):
                manager.verify(checkpoint)

    def test_one_model_component_can_be_selected_without_restoring_rng(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = Stage0CheckpointManager(temporary)
            models, optimizers = self._components()
            expected_encoder = {
                key: value.detach().clone()
                for key, value in models["future_encoder"].state_dict().items()
            }
            manager.save(
                sequence=3,
                phase="conditional_policy",
                steps={"conditional_policy": 2, "global": 3},
                hashes=self.hashes,
                models=models,
                optimizers=optimizers,
            )
            for parameter in models["future_encoder"].parameters():
                parameter.data.add_(10.0)
            for parameter in models["selector"].parameters():
                parameter.data.add_(20.0)
            changed_selector = {
                key: value.detach().clone()
                for key, value in models["selector"].state_dict().items()
            }
            models["future_encoder"].train()
            torch.manual_seed(91)
            expected_random = float(torch.rand(()))
            torch.manual_seed(91)

            manifest = manager.load_model_component(
                hashes=self.hashes,
                name="future_encoder",
                model=models["future_encoder"],
            )

            self.assertEqual(manifest["steps"]["conditional_policy"], 2)
            self.assertTrue(models["future_encoder"].training)
            self.assertEqual(float(torch.rand(())), expected_random)
            for key, value in models["future_encoder"].state_dict().items():
                torch.testing.assert_close(value, expected_encoder[key])
            for key, value in models["selector"].state_dict().items():
                torch.testing.assert_close(value, changed_selector[key])

            with self.assertRaisesRegex(
                Stage0CheckpointMismatchError,
                "absent",
            ):
                manager.load_model_component(
                    hashes=self.hashes,
                    name="no_z_policy",
                    model=torch.nn.Linear(3, 4),
                )

    def test_partial_directory_is_never_a_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = Stage0CheckpointManager(temporary)
            partial = (
                manager.checkpoints_dir / ".partial-checkpoint-000000000001-deadbeef"
            )
            partial.mkdir(parents=True)
            (partial / "COMPLETE").write_text("{}", encoding="ascii")
            self.assertIsNone(manager.latest_checkpoint())
            with self.assertRaises(Stage0CheckpointIntegrityError):
                manager.verify(partial)


class TestStage0DistributedEnvironment(unittest.TestCase):
    def test_single_process_default_and_valid_torchrun(self) -> None:
        self.assertEqual(
            Stage0ProcessEnvironment.parse({}),
            Stage0ProcessEnvironment(rank=0, local_rank=0, world_size=1),
        )
        self.assertEqual(
            Stage0ProcessEnvironment.parse(
                {"RANK": "2", "LOCAL_RANK": "1", "WORLD_SIZE": "4"}
            ),
            Stage0ProcessEnvironment(rank=2, local_rank=1, world_size=4),
        )

    def test_partial_or_invalid_environment_is_rejected(self) -> None:
        with self.assertRaisesRegex(Stage0DistributedError, "set together"):
            Stage0ProcessEnvironment.parse({"RANK": "0"})
        with self.assertRaisesRegex(Stage0DistributedError, "invalid"):
            Stage0ProcessEnvironment.parse(
                {"RANK": "2", "LOCAL_RANK": "0", "WORLD_SIZE": "2"}
            )


if __name__ == "__main__":
    unittest.main()
