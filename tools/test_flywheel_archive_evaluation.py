"""Tests for metadata-only success replay and the K=1 promotion gate."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from rexpolicy.flywheel.evaluation import (
    TRAIN_RESET_SEED_DOMAIN,
    HeldOutEpisodeResult,
    HeldOutMetrics,
    domain_seed,
    evaluate_non_regression,
    fixed_evaluation_suite,
)
from rexpolicy.flywheel.checkpoint import file_sha256
from rexpolicy.flywheel.experience import (
    DIRECT_SUCCESS_ROLE,
    SUCCESS_PATH_ROLE,
    TrainingSample,
)
from rexpolicy.flywheel.success_archive import (
    SuccessArchive,
    SuccessReference,
    load_episode_record,
)
from rexpolicy.flywheel.replay import validate_replay_fingerprint


def _sample(sample_world: int, *roles: str) -> TrainingSample:
    sample = TrainingSample(
        backbone_features=torch.ones(2, 3),
        backbone_attention_mask=torch.ones(2, dtype=torch.bool),
        image_mask=None,
        state=torch.zeros(1, 4),
        embodiment_id=10,
        action=torch.zeros(2, 5),
        action_mask=torch.ones(2, 5),
        valid_steps=1,
        sample_weight=1.5,
    )
    sample.set_provenance(
        generation=3,
        rank=0,
        episode=1,
        decision=2,
        world=sample_world,
        task_id="reach_green_cap/v1",
        reward_profile_id="reach_progress/v1",
    )
    for role in roles:
        sample.add_success_role(role)
    return sample


def _reference(
    sample_world: int,
    *,
    episode_sha256: str = "a" * 64,
) -> SuccessReference:
    sample = _sample(sample_world, DIRECT_SUCCESS_ROLE)
    return SuccessReference.from_sample(
        sample,
        episode_path="archive/attempts/a/generation-000003/rank-00000.episodes.jsonl",
        episode_sha256=episode_sha256,
        episode_record_index=0,
        simulator_fingerprint="sim-v1",
    )


class TestSuccessArchive(unittest.TestCase):
    def _populated_archive(
        self,
        run_dir: Path,
        *,
        worlds: int = 3,
    ) -> SuccessArchive:
        episode_path = (
            run_dir
            / "archive/attempts/a/generation-000003/rank-00000.episodes.jsonl"
        )
        episode_path.parent.mkdir(parents=True)
        episode_path.write_text(
            json.dumps({"generation": 3, "rank": 0, "episode": 1}) + "\n",
            encoding="utf-8",
        )
        archive = SuccessArchive(run_dir=run_dir, rank=0, attempt_id="a")
        relative = archive.write_generation(
            generation=3,
            references=[
                _reference(world, episode_sha256=file_sha256(episode_path))
                for world in range(worlds)
            ],
        )
        archive.commit_shard(relative or "")
        return archive

    def test_round_robin_is_unique_and_restartable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            episode_path = (
                run_dir
                / "archive/attempts/a/generation-000003/rank-00000.episodes.jsonl"
            )
            episode_path.parent.mkdir(parents=True)
            episode_payload = (
                json.dumps({"generation": 3, "rank": 0, "episode": 1}) + "\n"
            )
            episode_path.write_text(episode_payload, encoding="utf-8")
            archive = SuccessArchive(
                run_dir=run_dir,
                rank=0,
                attempt_id="a",
            )
            relative = archive.write_generation(
                generation=3,
                references=[
                    _reference(
                        world,
                        episode_sha256=file_sha256(episode_path),
                    )
                    for world in range(3)
                ],
            )
            self.assertIsNotNone(relative)
            archive.commit_shard(relative or "")

            first = archive.plan(
                replay_per_rank=2,
                max_generation_exclusive=4,
            )
            second = archive.plan(
                replay_per_rank=2,
                max_generation_exclusive=4,
            )
            self.assertEqual(len({item.sample_id for item in first}), 2)
            self.assertNotEqual(
                [item.sample_id for item in first],
                [item.sample_id for item in second],
            )

            restored = SuccessArchive(
                run_dir=run_dir,
                rank=0,
                attempt_id="b",
                state=archive.state_dict(),
            )
            self.assertEqual(restored.size, 3)
            self.assertEqual(restored.cursor, archive.cursor)
            record = load_episode_record(run_dir, first[0])
            self.assertEqual(record["episode"], 1)

            episode_path.write_text(
                episode_path.read_text(encoding="utf-8") + " \n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Episode archive checksum"):
                load_episode_record(run_dir, first[0])
            with self.assertRaisesRegex(ValueError, "Episode archive checksum"):
                SuccessArchive(
                    run_dir=run_dir,
                    rank=0,
                    attempt_id="c",
                    state=archive.state_dict(),
                )
            episode_path.write_text(episode_payload, encoding="utf-8")

            shard_path = run_dir / archive.shards[0]
            shard_path.write_text(
                shard_path.read_text(encoding="utf-8").splitlines()[0] + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                SuccessArchive(
                    run_dir=run_dir,
                    rank=0,
                    attempt_id="d",
                    state=archive.state_dict(),
                )

    def test_reference_contains_no_transient_or_bottle_state(self) -> None:
        record = _reference(0).to_record()
        forbidden = {
            "images",
            "video",
            "backbone_features",
            "vlm_features",
            "bottle_pose",
            "bottle_trajectory",
        }
        self.assertTrue(forbidden.isdisjoint(record))

    def test_success_roles_are_deduplicated(self) -> None:
        sample = _sample(0)
        sample.add_success_role(SUCCESS_PATH_ROLE)
        sample.add_success_role(SUCCESS_PATH_ROLE)
        self.assertEqual(sample.success_roles, (SUCCESS_PATH_ROLE,))

    def test_quarantine_separates_retention_from_replay_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            archive = self._populated_archive(run_dir)
            quarantined = _reference(
                1,
                episode_sha256=file_sha256(
                    run_dir
                    / "archive/attempts/a/generation-000003/"
                    "rank-00000.episodes.jsonl"
                ),
            )
            shard_path = run_dir / archive.shards[0]
            original_shard = shard_path.read_bytes()
            archive.quarantine(
                quarantined,
                reason_code="candidate_replay_mismatch",
                detail="reward drifted by 0.2",
                detected_generation=4,
            )

            planned = archive.plan(
                replay_per_rank=3,
                max_generation_exclusive=5,
            )
            self.assertEqual(archive.size, 3)
            self.assertEqual(archive.eligible_size, 2)
            self.assertEqual(archive.quarantined_size, 1)
            self.assertEqual(len(planned), 2)
            self.assertNotIn(
                quarantined.sample_id,
                {item.sample_id for item in planned},
            )
            self.assertEqual(shard_path.read_bytes(), original_shard)

    def test_quarantine_is_strict_restartable_and_v2_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            archive = self._populated_archive(run_dir)
            reference = archive.plan(
                replay_per_rank=1,
                max_generation_exclusive=4,
            )[0]
            self.assertTrue(
                archive.quarantine(
                    reference,
                    reason_code="candidate_replay_mismatch",
                    detail="outcome changed",
                    detected_generation=4,
                )
            )
            state = archive.state_dict()
            restored = SuccessArchive(
                run_dir=run_dir,
                rank=0,
                attempt_id="b",
                state=state,
            )
            self.assertEqual(restored.quarantined, archive.quarantined)
            self.assertEqual(restored.quarantined_size, 1)

            v2_state = dict(state)
            v2_state["schema_version"] = 2
            v2_state.pop("quarantined")
            migrated = SuccessArchive(
                run_dir=run_dir,
                rank=0,
                attempt_id="c",
                state=v2_state,
            )
            self.assertEqual(migrated.quarantined_size, 0)
            self.assertEqual(migrated.state_dict()["schema_version"], 3)

    def test_quarantine_rejects_unknown_and_conflicting_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            archive = self._populated_archive(run_dir)
            with self.assertRaisesRegex(KeyError, "Unknown success"):
                archive.quarantine(
                    _reference(99),
                    reason_code="candidate_replay_mismatch",
                    detail="test",
                    detected_generation=4,
                )
            reference = archive.plan(
                replay_per_rank=1,
                max_generation_exclusive=4,
            )[0]
            self.assertTrue(
                archive.quarantine(
                    reference,
                    reason_code="candidate_replay_mismatch",
                    detail="reward changed",
                    detected_generation=4,
                )
            )
            self.assertFalse(
                archive.quarantine(
                    reference,
                    reason_code="candidate_replay_mismatch",
                    detail="reward changed",
                    detected_generation=4,
                )
            )
            with self.assertRaisesRegex(ValueError, "Conflicting quarantine"):
                archive.quarantine(
                    reference,
                    reason_code="candidate_replay_mismatch",
                    detail="different outcome",
                    detected_generation=4,
                )


def _metrics(
    generation: int,
    *,
    successes: int,
    safety_failures: int = 0,
    invalid: int = 0,
    distance: float = 0.1,
    episode_count: int = 32,
) -> HeldOutMetrics:
    episodes = []
    for index in range(episode_count):
        episodes.append(
            HeldOutEpisodeResult(
                reset_seed=index,
                diffusion_seed=100 + index,
                success=index < successes,
                safety_failure=index < safety_failures,
                invalid_action=index < invalid,
                final_distance=distance,
                episode_return=0.0,
                object_displacement=0.0,
                control_steps=8,
            )
        )
    return HeldOutMetrics.from_episodes(
        generation=generation,
        episodes=episodes,
    )


class TestHeldOutGate(unittest.TestCase):
    def test_suite_is_fixed_8_by_4(self) -> None:
        suite = fixed_evaluation_suite(base_seed=7)
        self.assertEqual(len(suite), 32)
        self.assertEqual(len(set(suite)), 32)

    def test_held_out_reset_namespace_never_enters_twenty_generation_soak(self) -> None:
        base_seed = 20260722
        held_out_resets = {
            reset_seed
            for reset_seed, _ in fixed_evaluation_suite(base_seed=base_seed)
        }
        training_resets = {
            domain_seed(
                base_seed,
                TRAIN_RESET_SEED_DOMAIN,
                generation,
                rank,
                episode,
            )
            for generation in range(1, 21)
            for rank in range(2)
            for episode in range(2)
        }
        self.assertTrue(held_out_resets.isdisjoint(training_resets))

    def test_accepts_neutral_and_one_success_drop(self) -> None:
        baseline = _metrics(0, successes=4)
        candidate = _metrics(5, successes=3)
        self.assertTrue(
            evaluate_non_regression(
                last_good=baseline,
                candidate=candidate,
            ).accepted
        )

    def test_success_drop_threshold_is_a_rate_for_any_suite_size(self) -> None:
        baseline = _metrics(0, successes=10, episode_count=64)
        tolerated = _metrics(5, successes=8, episode_count=64)
        rejected = _metrics(5, successes=7, episode_count=64)
        self.assertTrue(
            evaluate_non_regression(
                last_good=baseline,
                candidate=tolerated,
            ).accepted
        )
        self.assertFalse(
            evaluate_non_regression(
                last_good=baseline,
                candidate=rejected,
            ).accepted
        )

    def test_rejects_safety_invalid_and_distance_regression(self) -> None:
        baseline = _metrics(0, successes=0, distance=0.10)
        for candidate in (
            _metrics(5, successes=0, safety_failures=1, distance=0.10),
            _metrics(5, successes=0, invalid=1, distance=0.10),
            _metrics(5, successes=0, distance=0.106),
        ):
            self.assertFalse(
                evaluate_non_regression(
                    last_good=baseline,
                    candidate=candidate,
                ).accepted
            )

    def test_rejects_a_new_paired_safety_failure_even_if_count_is_flat(self) -> None:
        baseline = _metrics(0, successes=0, safety_failures=1)
        candidate_episodes = list(
            _metrics(5, successes=0, safety_failures=0).episodes
        )
        candidate_episodes[1] = HeldOutEpisodeResult(
            **{
                **candidate_episodes[1].__dict__,
                "safety_failure": True,
            }
        )
        candidate = HeldOutMetrics.from_episodes(
            generation=5,
            episodes=candidate_episodes,
        )
        decision = evaluate_non_regression(
            last_good=baseline,
            candidate=candidate,
        )
        self.assertFalse(decision.accepted)
        self.assertTrue(any("new safety" in reason for reason in decision.reasons))


class TestReplayFingerprint(unittest.TestCase):
    def test_hashes_world_zero_without_archiving_values(self) -> None:
        tree = {
            "state": torch.tensor(
                [[1.0, 2.0], [1.0 + 1.0e-7, 2.0]],
                dtype=torch.float32,
            ),
            "image": torch.tensor(
                [[[1, 2]], [[1, 2]]],
                dtype=torch.uint8,
            ),
        }
        fingerprint = validate_replay_fingerprint(
            tree,
            world_count=2,
            float_tolerance=1.0e-5,
        )
        self.assertEqual(len(fingerprint.digest), 64)
        self.assertEqual(fingerprint.field_count, 2)
        self.assertLess(fingerprint.max_float_spread, 1.0e-5)

    def test_rejects_float_and_exact_state_divergence(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_replay_fingerprint(
                {"state": torch.tensor([[0.0], [1.0e-3]])},
                world_count=2,
                float_tolerance=1.0e-5,
            )
        with self.assertRaises(RuntimeError):
            validate_replay_fingerprint(
                {"image": torch.tensor([[1], [2]], dtype=torch.uint8)},
                world_count=2,
                float_tolerance=1.0e-5,
            )


if __name__ == "__main__":
    unittest.main()
