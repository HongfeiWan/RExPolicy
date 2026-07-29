"""Focused tests for the minimal distributed flywheel scaffolding."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
from rexpolicy.flywheel.groot_policy import (
    CachedCondition,
    GrootFlowDitPolicy,
    PolicyBatch,
    RawPolicyObservation,
)
from tools.run_flywheel_ddp import (
    _directory_descriptors,
    _early_gpu_preflight,
    _git_source_descriptor,
    _normalized_action_diversity,
    _raw_observation,
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


class TestSharedConditionSampling(unittest.TestCase):
    def test_same_state_encodes_reference_once_and_expands_for_dit(self) -> None:
        policy = object.__new__(GrootFlowDitPolicy)
        condition = CachedCondition(
            backbone_features=torch.ones(2, 3),
            backbone_attention_mask=torch.ones(2, dtype=torch.bool),
            image_mask=None,
            state=torch.zeros(1, 5),
            embodiment_id=10,
        )
        observation = RawPolicyObservation(
            images={"camera": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
            state={"eef": np.zeros((1, 3), dtype=np.float32)},
            instruction="reach",
        )
        candidate_count = 8
        expected = PolicyBatch(
            decoded_action={"eef_9d": np.zeros((8, 1, 9), dtype=np.float32)},
            conditions=[condition] * candidate_count,
        )
        prediction = {"action_pred": torch.zeros(candidate_count, 1, 9)}

        with (
            mock.patch.object(
                policy,
                "encode_conditions",
                return_value=[condition],
            ) as encode,
            mock.patch.object(
                policy,
                "_sample_same_state_prediction",
                return_value=prediction,
            ) as sample_prediction,
            mock.patch.object(
                policy,
                "_decode_prediction",
                return_value=expected,
            ) as decode,
        ):
            actual = policy.sample_same_state(
                observation,
                candidate_count=candidate_count,
            )

        self.assertIs(actual, expected)
        encode.assert_called_once_with([observation])
        sample_prediction.assert_called_once_with(
            condition=condition,
            candidate_count=candidate_count,
        )
        decode.assert_called_once()
        expanded = decode.call_args.kwargs["conditions"]
        self.assertEqual(len(expanded), candidate_count)
        self.assertTrue(all(item is condition for item in expanded))
        decoded_observations = decode.call_args.kwargs["observations"]
        self.assertEqual(len(decoded_observations), candidate_count)
        self.assertTrue(all(item is observation for item in decoded_observations))

    def test_same_state_rejects_non_positive_candidate_count(self) -> None:
        policy = object.__new__(GrootFlowDitPolicy)
        observation = RawPolicyObservation(images={}, state={}, instruction="reach")
        with self.assertRaisesRegex(ValueError, "candidate_count"):
            policy.sample_same_state(observation, candidate_count=0)

    def test_same_state_keeps_dit_noise_independent_and_replayable(self) -> None:
        from transformers.feature_extraction_utils import BatchFeature

        class FakeActionHead:
            def __init__(self) -> None:
                self.encode_batch_sizes = []
                self.sample_batch_sizes = []
                self.expanded_feature_strides = []
                self.mask_batch_sizes = []
                self.image_masks = []

            def _encode_features(self, backbone_output, action_input):
                self.encode_batch_sizes.append(
                    int(backbone_output["backbone_features"].shape[0])
                )
                return SimpleNamespace(
                    backbone_features=backbone_output["backbone_features"] + 1,
                    state_features=action_input["state"] + 1,
                )

            def get_action_with_features(
                self,
                *,
                backbone_features,
                state_features,
                embodiment_id,
                backbone_output,
                action_input,
            ):
                del state_features, embodiment_id, action_input
                batch_size = int(backbone_features.shape[0])
                self.sample_batch_sizes.append(batch_size)
                self.expanded_feature_strides.append(tuple(backbone_features.stride()))
                self.mask_batch_sizes.append(
                    (
                        int(backbone_output["backbone_attention_mask"].shape[0]),
                        int(backbone_output["image_mask"].shape[0]),
                    )
                )
                self.image_masks.append(backbone_output["image_mask"].cpu().tolist())
                return {
                    "action_pred": torch.randn(
                        batch_size,
                        2,
                        3,
                        dtype=torch.float32,
                    )
                }

        class FakeProcessor:
            def __init__(self) -> None:
                self.decoded_states = None

            def decode_action(self, normalized, embodiment_tag, states):
                del embodiment_tag
                self.decoded_states = states
                return {"eef_9d": normalized}

        policy = object.__new__(GrootFlowDitPolicy)
        policy.torch = torch
        policy.device = torch.device("cpu")
        policy.compute_dtype = torch.float32
        policy.model = SimpleNamespace(action_head=FakeActionHead())
        policy.processor = FakeProcessor()
        policy.state_keys = ("eef",)
        policy.embodiment_tag = "test"
        condition = CachedCondition(
            backbone_features=torch.ones(2, 3),
            backbone_attention_mask=torch.ones(2, dtype=torch.bool),
            image_mask=None,
            state=torch.zeros(1, 3),
            embodiment_id=10,
        )
        policy.encode_conditions = mock.Mock(return_value=[condition])
        policy._collate_conditions = mock.Mock(
            side_effect=lambda conditions: (
                BatchFeature(
                    data={
                        "backbone_features": torch.ones(len(conditions), 2, 3),
                        "backbone_attention_mask": torch.ones(
                            len(conditions), 2, dtype=torch.bool
                        ),
                        "image_mask": torch.tensor([[True, False]] * len(conditions)),
                    }
                ),
                BatchFeature(
                    data={
                        "state": torch.zeros(len(conditions), 1, 3),
                        "embodiment_id": torch.full(
                            (len(conditions),), 10, dtype=torch.long
                        ),
                    }
                ),
            )
        )
        observation = RawPolicyObservation(
            images={},
            state={"eef": np.ones((1, 3), dtype=np.float32)},
            instruction="reach",
        )

        torch.manual_seed(17)
        first = policy.sample_same_state(observation, candidate_count=8)
        torch.manual_seed(17)
        second = policy.sample_same_state(observation, candidate_count=8)

        self.assertEqual(first.decoded_action["eef_9d"].shape, (8, 2, 3))
        np.testing.assert_array_equal(
            first.decoded_action["eef_9d"],
            second.decoded_action["eef_9d"],
        )
        self.assertFalse(
            np.all(
                first.decoded_action["eef_9d"][0] == first.decoded_action["eef_9d"][1]
            )
        )
        self.assertEqual(policy.processor.decoded_states["eef"].shape, (8, 1, 3))
        self.assertTrue(np.all(policy.processor.decoded_states["eef"] == 1.0))
        self.assertEqual(policy.model.action_head.encode_batch_sizes, [1, 1])
        self.assertEqual(policy.model.action_head.sample_batch_sizes, [8, 8])
        self.assertEqual(
            policy.model.action_head.expanded_feature_strides,
            [(6, 3, 1), (6, 3, 1)],
        )
        self.assertEqual(
            policy.model.action_head.mask_batch_sizes,
            [(8, 8), (8, 8)],
        )
        expected_image_mask = [[True, False]] * 8
        self.assertEqual(
            policy.model.action_head.image_masks,
            [expected_image_mask, expected_image_mask],
        )
        for call in policy._collate_conditions.call_args_list:
            self.assertEqual(len(call.args[0]), 1)
            self.assertIs(call.args[0][0], condition)

    def test_same_state_requires_pinned_action_head_feature_api(self) -> None:
        policy = object.__new__(GrootFlowDitPolicy)
        policy.model = SimpleNamespace(action_head=SimpleNamespace())
        with self.assertRaisesRegex(RuntimeError, "_encode_features"):
            policy._validate_same_state_condition_api()

    def test_shared_condition_does_not_alias_sample_actions_or_identity(self) -> None:
        class IdentityStateActionProcessor:
            @staticmethod
            def apply(*, state, action, embodiment_tag):
                del embodiment_tag
                return state, action

        policy = object.__new__(GrootFlowDitPolicy)
        policy.torch = torch
        policy.processor_action_horizon = 2
        policy.processor = SimpleNamespace(
            state_action_processor=IdentityStateActionProcessor()
        )
        policy.embodiment_tag = SimpleNamespace(value="test")
        policy.model = SimpleNamespace(
            config=SimpleNamespace(max_action_dim=26, action_horizon=2)
        )
        policy.action_keys = (
            "eef_9d",
            "hand_joint_target",
            "arm_joint_target",
        )
        condition = CachedCondition(
            backbone_features=torch.ones(2, 3),
            backbone_attention_mask=torch.ones(2, dtype=torch.bool),
            image_mask=None,
            state=torch.zeros(1, 3),
            embodiment_id=10,
        )
        executed_action = {
            "eef_9d": np.zeros((1, 9), dtype=np.float32),
            "hand_joint_target": np.zeros((1, 10), dtype=np.float32),
            "arm_joint_target": np.zeros((1, 7), dtype=np.float32),
        }
        samples = []
        for world in range(2):
            samples.append(
                policy.make_training_sample(
                    condition=condition,
                    raw_state={"eef": np.zeros((1, 3), dtype=np.float32)},
                    executed_action=executed_action,
                    sample_metadata={
                        "generation": 1,
                        "rank": 0,
                        "episode": 0,
                        "decision": 0,
                        "world": world,
                        "task_id": "reach_green_cap/v1",
                        "reward_profile_id": "reach_progress/v1",
                        "source": "current",
                    },
                )
            )

        self.assertEqual(
            samples[0].backbone_features.data_ptr(),
            samples[1].backbone_features.data_ptr(),
        )
        self.assertNotEqual(samples[0].action.data_ptr(), samples[1].action.data_ptr())
        self.assertNotEqual(
            samples[0].action_mask.data_ptr(),
            samples[1].action_mask.data_ptr(),
        )
        self.assertNotEqual(samples[0].sample_id, samples[1].sample_id)
        samples[0].action[0, 0] = 1.0
        self.assertEqual(float(samples[1].action[0, 0]), 0.0)

    def test_raw_observation_copies_only_selected_world(self) -> None:
        observation = {
            "extra": {"eef_9d": torch.arange(27).reshape(3, 9)},
            "agent": {
                "hand_joint_pos": torch.arange(30).reshape(3, 10),
                "arm_joint_pos": torch.arange(21).reshape(3, 7),
            },
            "sensor_data": {
                "ego_view": {
                    "rgb": torch.arange(36, dtype=torch.uint8).reshape(3, 2, 2, 3)
                },
                "wrist_view": {
                    "rgb": torch.arange(36, dtype=torch.uint8).reshape(3, 2, 2, 3)
                },
            },
        }

        raw = _raw_observation(observation, instruction="reach", world=1)

        self.assertEqual(raw.instruction, "reach")
        self.assertEqual(raw.images["ego_view"].shape, (1, 2, 2, 3))
        self.assertEqual(raw.state["eef_9d"].shape, (1, 9))
        np.testing.assert_array_equal(
            raw.state["eef_9d"][0],
            np.arange(9, 18, dtype=np.float32),
        )
        with self.assertRaises(IndexError):
            _raw_observation(observation, instruction="reach", world=3)


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
            mock.patch("tools.run_flywheel_ddp.preflight_selected_gpus") as preflight,
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
