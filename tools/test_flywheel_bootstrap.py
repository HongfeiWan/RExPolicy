"""Focused tests for the minimal distributed flywheel scaffolding."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from rexpolicy.flywheel.distributed import DistributedContext
from rexpolicy.flywheel.experience import (
    ChunkCandidate,
    EpisodeExperience,
    TrainingSample,
    collate_training_samples,
    select_advantage_chunks,
)
from tools.run_flywheel_ddp import (
    _directory_descriptors,
    _early_gpu_preflight,
    _git_source_descriptor,
    _normalized_action_diversity,
    _validate_run_directory_mode,
    create_parser,
)


def _sample() -> TrainingSample:
    return TrainingSample(
        backbone_features=torch.ones(2, 3),
        backbone_attention_mask=torch.ones(2, dtype=torch.bool),
        image_mask=torch.ones(2, dtype=torch.bool),
        state=torch.zeros(1, 5),
        embodiment_id=10,
        action=torch.zeros(3, 7),
        action_mask=torch.ones(3, 7),
        valid_steps=3,
    )


def _candidate(world: int, score: float, *, success: bool = False) -> ChunkCandidate:
    return ChunkCandidate(
        world=world,
        score=score,
        success=success,
        terminated=success,
        truncated=False,
        valid_steps=1,
        rewards=[score],
        action={"eef_9d": np.zeros((1, 9), dtype=np.float32)},
        sample=_sample(),
    )


def _episode() -> EpisodeExperience:
    return EpisodeExperience(
        generation=1,
        sampling_policy_generation=0,
        rank=0,
        episode=0,
        reset_seed=100,
        instruction="reach",
        task_id="reach_green_cap/v1",
        reward_profile_id="reach_progress/v1",
        score=1.5,
        success=False,
        initial_state={"eef_9d": np.zeros(9, dtype=np.float32)},
        decisions=[
            {
                "decision_index": 0,
                "advantage_baseline": 1.0,
                "candidates": [_candidate(0, 1.5).archive_record()],
            }
        ],
    )


class TestFlywheelExperience(unittest.TestCase):
    def test_selects_positive_same_state_advantages(self) -> None:
        candidates = [
            _candidate(0, 0.0),
            _candidate(1, 1.0),
            _candidate(2, 2.0),
            _candidate(3, 3.0),
        ]
        selection = select_advantage_chunks(
            candidates,
            fraction=0.5,
            temperature=1.0,
        )
        self.assertEqual(selection.baseline, 1.5)
        self.assertEqual([candidate.world for candidate in selection.selected], [3, 2])
        self.assertEqual(selection.continuation.world, 3)
        self.assertTrue(candidates[3].chosen_for_continuation)
        self.assertTrue(candidates[2].selected_for_training)
        self.assertFalse(candidates[1].selected_for_training)
        self.assertAlmostEqual(
            sum(candidate.sample_weight for candidate in selection.selected),
            2.0,
        )

    def test_all_successes_train_and_best_success_continues(self) -> None:
        candidates = [
            _candidate(0, 10.0),
            _candidate(1, 1.0, success=True),
            _candidate(2, 2.0, success=True),
        ]
        selection = select_advantage_chunks(
            candidates,
            fraction=1.0,
            temperature=1.0,
        )
        self.assertEqual(selection.mode, "all_success")
        self.assertEqual(selection.continuation.world, 2)
        self.assertEqual(
            [candidate.world for candidate in selection.selected],
            [2, 1],
        )
        self.assertFalse(candidates[0].selected_for_training)
        self.assertTrue(
            all(candidate.sample_weight > 0.0 for candidate in selection.selected)
        )

    def test_normalized_action_diversity_uses_valid_training_dimensions(self) -> None:
        candidates = [_candidate(0, 0.0), _candidate(1, 1.0)]
        candidates[1].sample.action.fill_(1.0)
        mean_rms, max_rms = _normalized_action_diversity(candidates)
        self.assertAlmostEqual(mean_rms, 1.0)
        self.assertAlmostEqual(max_rms, 1.0)

    def test_archive_omits_transient_features(self) -> None:
        episode = _episode()
        record = episode.archive_record()
        self.assertEqual(record["schema_version"], 4)
        self.assertEqual(record["data_generation"], 1)
        self.assertEqual(record["sampling_policy_generation"], 0)
        self.assertEqual(record["policy_version"], "generation-000000")
        self.assertNotIn("samples", record)
        self.assertNotIn("images", record)
        self.assertEqual(record["reset_recipe"]["seed"], 100)
        self.assertIn("decisions", record)

    def test_collate_pads_frozen_vlm_sequences(self) -> None:
        samples = []
        for sequence_length in (2, 4):
            samples.append(
                TrainingSample(
                    backbone_features=torch.ones(sequence_length, 3),
                    backbone_attention_mask=torch.ones(
                        sequence_length, dtype=torch.bool
                    ),
                    image_mask=torch.tensor([True] * sequence_length),
                    state=torch.zeros(1, 5),
                    embodiment_id=10,
                    action=torch.zeros(3, 7),
                    action_mask=torch.ones(3, 7),
                    valid_steps=3,
                )
            )
        backbone, action = collate_training_samples(
            samples,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(tuple(backbone.backbone_features.shape), (2, 4, 3))
        self.assertEqual(tuple(backbone.backbone_attention_mask.shape), (2, 4))
        self.assertFalse(bool(backbone.backbone_attention_mask[0, 2]))
        self.assertEqual(tuple(action.state.shape), (2, 1, 5))
        self.assertEqual(tuple(action.action.shape), (2, 3, 7))


class TestDistributedContext(unittest.TestCase):
    def test_initialize_registers_local_cuda_device_with_nccl(self) -> None:
        environment = {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2"}
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.device_count", return_value=2),
            mock.patch("torch.cuda.set_device") as set_device,
            mock.patch("torch.distributed.is_initialized", return_value=False),
            mock.patch("torch.distributed.init_process_group") as initialize,
        ):
            context = DistributedContext.initialize(timeout_minutes=7)

        set_device.assert_called_once_with(1)
        self.assertEqual(context.device, torch.device("cuda", 1))
        self.assertEqual(initialize.call_args.kwargs["backend"], "nccl")
        self.assertEqual(
            initialize.call_args.kwargs["device_id"], torch.device("cuda", 1)
        )

    def test_barrier_uses_rank_local_cuda_device(self) -> None:
        context = DistributedContext(
            rank=1,
            local_rank=1,
            world_size=2,
            device=torch.device("cuda", 1),
            initialized=True,
        )
        with mock.patch("torch.distributed.barrier") as barrier:
            context.barrier()

        barrier.assert_called_once_with(device_ids=[1])


class TestRunnerConfiguration(unittest.TestCase):
    def test_artifact_directory_hashes_all_processor_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "statistics.json").write_text("stats", encoding="utf-8")
            (root / "tokenizer").mkdir()
            (root / "tokenizer" / "tokenizer.json").write_text(
                "tokens",
                encoding="utf-8",
            )
            descriptors = _directory_descriptors(root)
        self.assertEqual(
            {item["relative_path"] for item in descriptors},
            {"statistics.json", "tokenizer/tokenizer.json"},
        )
        self.assertTrue(all(len(item["sha256"]) == 64 for item in descriptors))

    def test_non_git_source_descriptor_ignores_generated_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "gr00t").mkdir()
            (root / "gr00t/model.py").write_text("VALUE = 1\n", encoding="utf-8")
            cache = root / "gr00t/__pycache__"
            cache.mkdir()
            bytecode = cache / "model.cpython-311.pyc"
            bytecode.write_bytes(b"generated")

            descriptor = _git_source_descriptor(root)

            self.assertFalse(descriptor["git_available"])
            self.assertIsNone(descriptor["commit"])
            self.assertEqual(
                [item["relative_path"] for item in descriptor["files"]],
                ["gr00t/model.py"],
            )
            first_hash = descriptor["source_tree_sha256"]
            bytecode.write_bytes(b"changed")
            self.assertEqual(
                _git_source_descriptor(root)["source_tree_sha256"],
                first_hash,
            )

    def test_git_source_descriptor_scopes_dirty_state_to_runtime_tree(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            runtime.mkdir()
            source = runtime / "module.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            log = root / "volatile.log"
            log.write_text("initial\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=RExPolicy Test",
                    "-c",
                    "user.email=rexpolicy@example.invalid",
                    "commit",
                    "-qm",
                    "initial",
                ],
                check=True,
            )
            cache = runtime / "__pycache__"
            cache.mkdir()
            (cache / "module.cpython-311.pyc").write_bytes(b"generated")

            clean = _git_source_descriptor(runtime)
            log.write_text("changed\n", encoding="utf-8")
            unrelated_dirty = _git_source_descriptor(runtime)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            runtime_dirty = _git_source_descriptor(runtime)

            self.assertEqual(clean["source_scope"], "runtime")
            self.assertTrue(clean["clean"])
            self.assertEqual(clean["untracked"], [])
            self.assertEqual(
                unrelated_dirty["tracked_diff_sha256"],
                clean["tracked_diff_sha256"],
            )
            self.assertTrue(unrelated_dirty["clean"])
            self.assertFalse(runtime_dirty["clean"])

    def test_artifact_environment_variables_feed_parser_defaults(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "ISAAC_GROOT_ROOT": "/runtime/Isaac-GR00T",
                "GROOT_POLICY_CHECKPOINT": "/weights/policy",
                "GROOT_VLM_MODEL": "/weights/vlm",
            },
            clear=False,
        ):
            args = create_parser().parse_args([])
        self.assertEqual(args.isaac_groot_root, Path("/runtime/Isaac-GR00T"))
        self.assertEqual(args.policy_checkpoint, Path("/weights/policy"))
        self.assertEqual(args.vlm_model, Path("/weights/vlm"))

    def test_early_preflight_checks_physical_gpu_before_distributed_init(
        self,
    ) -> None:
        args = create_parser().parse_args(["--validate-only"])
        with (
            mock.patch.dict(
                os.environ,
                {"CUDA_VISIBLE_DEVICES": "2,4", "LOCAL_RANK": "1"},
                clear=False,
            ),
            mock.patch(
                "tools.run_flywheel_ddp.preflight_selected_gpus"
            ) as preflight,
        ):
            _early_gpu_preflight(args)
        preflight.assert_called_once_with(
            ["4"],
            minimum_free_mib=args.minimum_free_gpu_mib,
            allowed_pids=[os.getpid()],
        )

    def test_new_run_rejects_existing_flywheel_artifacts(self) -> None:
        context = DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
            initialized=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "flywheel artifacts"):
                _validate_run_directory_mode(
                    run_dir=run_dir,
                    context=context,
                    resume=False,
                    resume_from=None,
                )


if __name__ == "__main__":
    unittest.main()
