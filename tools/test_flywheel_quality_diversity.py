"""Tests for rebuildable QD indexing and balanced Success Archive replay."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import unittest
import tempfile
from dataclasses import replace
from pathlib import Path

from rexpolicy.flywheel.checkpoint import file_sha256
from rexpolicy.flywheel.derived_views import write_derived_view
from rexpolicy.flywheel.experience import DIRECT_SUCCESS_ROLE
from rexpolicy.flywheel.quality_diversity import (
    BalancedReplayState,
    BehaviorDimension,
    QualityDiversityIndex,
    QualityDiversityPolicy,
    SuccessBehaviorDescriptor,
    build_quality_diversity_index,
    plan_balanced_replay,
    rebase_balanced_replay_state,
)
from rexpolicy.flywheel.success_archive import SuccessArchive, SuccessReference
from rexpolicy.tasking.canonical import canonical_fingerprint


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _policy() -> QualityDiversityPolicy:
    return QualityDiversityPolicy.from_record(
        {
            "schema_version": 1,
            "policy_id": "reach_qd/v1",
            "task_id": "reach_green_cap/v3",
            "task_oracle_sha256": _hash("reach-oracle/v3"),
            "dimensions": [
                {
                    "dimension_id": "reach.approach_angle_rad",
                    "minimum": -math.pi,
                    "maximum": math.pi,
                    "bins": 4,
                },
                {
                    "dimension_id": "reach.path_length_m",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "bins": 2,
                },
            ],
            "quality_metric_id": "reach.clearance_m",
            "maximize_quality": True,
            "balance_axes": [
                "behavior_cell",
                "initial_state_group",
                "reward_profile",
            ],
        }
    )


def _descriptor(
    name: str,
    *,
    angle: float,
    path_length: float,
    clearance: float,
    reward: str = "reach_progress/v1",
    state: str = "reach_reset/centre/v1",
) -> SuccessBehaviorDescriptor:
    return SuccessBehaviorDescriptor.from_record(
        {
            "schema_version": 1,
            "sample_id": f"sample/{name}",
            "success_reference_sha256": _hash(f"reference:{name}"),
            "event_ledger_sha256": _hash(f"event-ledger:{name}"),
            "compilation_policy_sha256": _hash("reach-qd-compiler/v1"),
            "task_id": "reach_green_cap/v3",
            "task_oracle_sha256": _hash("reach-oracle/v3"),
            "reward_profile_id": reward,
            "reward_profile_sha256": _hash(f"reward:{reward}"),
            "initial_state_group_id": state,
            "initial_state_group_sha256": _hash(f"initial-state:{state}"),
            "metrics": {
                "reach.approach_angle_rad": angle,
                "reach.path_length_m": path_length,
                "reach.clearance_m": clearance,
            },
        }
    )


def _append_quality_generation(
    *,
    archive: SuccessArchive,
    run_dir: Path,
    generation: int,
    definitions: tuple[tuple[str, str], ...],
    task_id: str = "reach_green_cap/v3",
) -> dict[str, SuccessReference]:
    episode_path = Path("episodes") / f"generation-{generation:06d}.jsonl"
    absolute_episode_path = run_dir / episode_path
    absolute_episode_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_episode_path.write_text(
        "".join(
            json.dumps(
                {
                    "generation": generation,
                    "rank": 0,
                    "episode": episode,
                }
            )
            + "\n"
            for episode in range(len(definitions))
        ),
        encoding="utf-8",
    )
    episode_sha256 = file_sha256(absolute_episode_path)
    references = {
        name: SuccessReference(
            schema_version=2,
            sample_id=f"sample/{name}",
            source_generation=generation,
            source_rank=0,
            source_episode=episode,
            decision=0,
            world=episode,
            success_roles=(DIRECT_SUCCESS_ROLE,),
            episode_path=episode_path.as_posix(),
            episode_sha256=episode_sha256,
            episode_record_index=episode,
            archived_weight=1.0,
            task_id=task_id,
            reward_profile_id=reward_profile_id,
            simulator_fingerprint="simulator/v1",
        )
        for episode, (name, reward_profile_id) in enumerate(definitions)
    }
    relative = archive.write_generation(
        generation=generation,
        references=list(references.values()),
    )
    archive.commit_shard(relative or "")
    return references


def _quality_archive(
    run_dir: Path,
    generations: dict[int, tuple[tuple[str, str], ...]],
    *,
    task_id: str = "reach_green_cap/v3",
) -> tuple[SuccessArchive, dict[str, SuccessReference]]:
    archive = SuccessArchive(run_dir=run_dir, rank=0, attempt_id="quality")
    references = {}
    for generation, definitions in sorted(generations.items()):
        references.update(
            _append_quality_generation(
                archive=archive,
                run_dir=run_dir,
                generation=generation,
                definitions=definitions,
                task_id=task_id,
            )
        )
    return archive, references


def _bound_descriptor(
    reference: SuccessReference,
    *,
    angle: float = 0.0,
    path_length: float = 0.2,
    clearance: float = 0.02,
    reward_profile_id: str | None = None,
    task_id: str | None = None,
    initial_state_group_id: str = "reach_reset/centre/v1",
    success_reference_sha256: str | None = None,
) -> SuccessBehaviorDescriptor:
    reward = reward_profile_id or reference.reward_profile_id
    task = task_id or reference.task_id
    return SuccessBehaviorDescriptor.from_record(
        {
            "schema_version": 1,
            "sample_id": reference.sample_id,
            "success_reference_sha256": (
                success_reference_sha256
                or canonical_fingerprint(reference.to_record())
            ),
            "event_ledger_sha256": _hash(
                f"event-ledger:{reference.sample_id}"
            ),
            "compilation_policy_sha256": _hash("reach-qd-compiler/v1"),
            "task_id": task,
            "task_oracle_sha256": _hash("reach-oracle/v3"),
            "reward_profile_id": reward,
            "reward_profile_sha256": _hash(f"reward:{reward}"),
            "initial_state_group_id": initial_state_group_id,
            "initial_state_group_sha256": _hash(
                f"initial-state:{initial_state_group_id}"
            ),
            "metrics": {
                "reach.approach_angle_rad": angle,
                "reach.path_length_m": path_length,
                "reach.clearance_m": clearance,
            },
        }
    )


class TestQualityDiversityIndex(unittest.TestCase):
    def test_index_is_deterministic_and_retains_non_elites(self) -> None:
        low = _descriptor(
            "low",
            angle=-2.0,
            path_length=0.2,
            clearance=0.01,
        )
        high = _descriptor(
            "high",
            angle=-2.1,
            path_length=0.25,
            clearance=0.04,
        )
        other = _descriptor(
            "other",
            angle=2.0,
            path_length=0.8,
            clearance=0.02,
        )
        first = build_quality_diversity_index(
            policy=_policy(), descriptors=(low, high, other)
        )
        second = build_quality_diversity_index(
            policy=_policy(), descriptors=(other, high, low)
        )

        self.assertEqual(first, second)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(len(first.cells), 2)
        same_cell = next(
            cell for cell in first.cells if low.fingerprint in cell.member_descriptor_sha256s
        )
        self.assertEqual(
            set(same_cell.member_descriptor_sha256s),
            {low.fingerprint, high.fingerprint},
        )
        self.assertEqual(same_cell.elite_descriptor_sha256, high.fingerprint)
        self.assertGreater(first.coverage, 0.0)
        self.assertEqual(len(first.event_ledger_set_sha256), 64)

        with tempfile.TemporaryDirectory() as raw_directory:
            first_write = write_derived_view(
                run_dir=Path(raw_directory),
                view=first,
            )
            second_write = write_derived_view(
                run_dir=Path(raw_directory),
                view=first,
            )
            self.assertTrue(first_write.created)
            self.assertFalse(second_write.created)
            self.assertEqual(
                first_write.event_ledger_sha256,
                first.event_ledger_set_sha256,
            )

    def test_round_trip_recomputes_cells_and_rejects_tampering(self) -> None:
        index = build_quality_diversity_index(
            policy=_policy(),
            descriptors=(
                _descriptor("a", angle=-2.0, path_length=0.2, clearance=0.02),
                _descriptor("b", angle=2.0, path_length=0.8, clearance=0.03),
            ),
        )
        restored = QualityDiversityIndex.from_json(index.to_json())
        self.assertEqual(restored, index)

        changed = copy.deepcopy(index.to_record())
        changed["cells"][0]["elite_descriptor_sha256"] = _hash("forged")
        with self.assertRaisesRegex(ValueError, "exactly rebuildable"):
            QualityDiversityIndex.from_record(changed)

        changed = copy.deepcopy(index.to_record())
        changed["descriptors"][0]["metrics"]["reach.clearance_m"] = 0.5
        with self.assertRaisesRegex(ValueError, "exactly rebuildable"):
            QualityDiversityIndex.from_record(changed)

    def test_policy_and_metric_boundaries_fail_closed(self) -> None:
        record = _policy().to_record()
        record["dimensions"][1]["dimension_id"] = record["dimensions"][0][
            "dimension_id"
        ]
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            QualityDiversityPolicy.from_record(record)

        outside = _descriptor(
            "outside", angle=4.0, path_length=0.2, clearance=0.02
        )
        with self.assertRaisesRegex(ValueError, "outside its policy range"):
            build_quality_diversity_index(
                policy=_policy(), descriptors=(outside,)
            )

        missing = _descriptor(
            "missing", angle=0.0, path_length=0.2, clearance=0.02
        ).to_record()
        del missing["metrics"]["reach.clearance_m"]
        descriptor = SuccessBehaviorDescriptor.from_record(missing)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            build_quality_diversity_index(
                policy=_policy(), descriptors=(descriptor,)
            )


class TestBalancedReplay(unittest.TestCase):
    def test_balances_cell_reward_and_initial_state_without_deletion(self) -> None:
        crowded = tuple(
            _descriptor(
                f"crowded-{index}",
                angle=-2.0,
                path_length=0.2,
                clearance=0.01 + index / 1000.0,
            )
            for index in range(5)
        )
        rare_profile = _descriptor(
            "rare-profile",
            angle=-2.0,
            path_length=0.2,
            clearance=0.02,
            reward="reach_smooth/v1",
        )
        rare_state = _descriptor(
            "rare-state",
            angle=2.0,
            path_length=0.8,
            clearance=0.03,
            state="reach_reset/left/v1",
        )
        index = build_quality_diversity_index(
            policy=_policy(),
            descriptors=(*crowded, rare_profile, rare_state),
        )
        state = BalancedReplayState.initial(index)
        first = plan_balanced_replay(index=index, count=3, state=state)

        self.assertEqual(len(first.sample_ids), 3)
        self.assertIn(rare_profile.sample_id, first.sample_ids)
        self.assertIn(rare_state.sample_id, first.sample_ids)

        observed = set(first.sample_ids)
        current = first.next_state
        for _ in range(8):
            plan = plan_balanced_replay(index=index, count=3, state=current)
            observed.update(plan.sample_ids)
            current = plan.next_state
        self.assertEqual(observed, {item.sample_id for item in index.descriptors})

    def test_state_is_bound_to_exact_index_and_strata(self) -> None:
        descriptor = _descriptor(
            "one", angle=0.0, path_length=0.2, clearance=0.02
        )
        index = build_quality_diversity_index(
            policy=_policy(), descriptors=(descriptor,)
        )
        other = build_quality_diversity_index(
            policy=_policy(),
            descriptors=(
                descriptor,
                _descriptor("two", angle=2.0, path_length=0.8, clearance=0.03),
            ),
        )
        state = BalancedReplayState.initial(index)
        restored = BalancedReplayState.from_record(
            state.to_record(),
            index=index,
        )
        self.assertEqual(restored, state)
        with self.assertRaisesRegex(ValueError, "another QD index"):
            plan_balanced_replay(index=other, count=1, state=state)

        forged = BalancedReplayState(
            schema_version=1,
            quality_diversity_index_sha256=index.fingerprint,
            stratum_cursor=0,
            member_cursors=(("unknown|stratum|v1", 1),),
        )
        with self.assertRaisesRegex(ValueError, "unknown strata"):
            plan_balanced_replay(index=index, count=1, state=forged)

    def test_state_rebase_preserves_only_exact_surviving_strata(self) -> None:
        first = _descriptor(
            "first", angle=-2.0, path_length=0.2, clearance=0.02
        )
        removed = _descriptor(
            "removed",
            angle=2.0,
            path_length=0.8,
            clearance=0.03,
            reward="reach_smooth/v1",
        )
        previous = build_quality_diversity_index(
            policy=_policy(), descriptors=(first, removed)
        )
        advanced = plan_balanced_replay(
            index=previous,
            count=2,
            state=BalancedReplayState.initial(previous),
        ).next_state
        added = _descriptor(
            "added",
            angle=2.1,
            path_length=0.75,
            clearance=0.04,
            state="reach_reset/right/v1",
        )
        replacement = build_quality_diversity_index(
            policy=_policy(), descriptors=(first, added)
        )

        rebased = rebase_balanced_replay_state(
            previous_index=previous,
            replacement_index=replacement,
            state=advanced,
        )

        self.assertEqual(
            rebased.quality_diversity_index_sha256,
            replacement.fingerprint,
        )
        self.assertEqual(rebased.stratum_cursor, advanced.stratum_cursor)
        self.assertEqual(len(rebased.member_cursors), 1)
        self.assertEqual(
            plan_balanced_replay(
                index=replacement,
                count=2,
                state=rebased,
            ).sample_ids,
            (first.sample_id, added.sample_id),
        )


class TestSuccessArchiveQualityBalanced(unittest.TestCase):
    def test_rejects_tampered_reference_task_and_reward_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            archive, references = _quality_archive(
                Path(raw_directory),
                {3: (("hash", "reach_progress/v1"),)},
            )
            tampered = _bound_descriptor(
                references["hash"],
                success_reference_sha256=_hash("tampered-reference"),
            )
            index = build_quality_diversity_index(
                policy=_policy(), descriptors=(tampered,)
            )
            with self.assertRaisesRegex(ValueError, "reference hash mismatch"):
                archive.plan_quality_balanced(
                    replay_per_rank=1,
                    max_generation_exclusive=4,
                    index=index,
                    state=BalancedReplayState.initial(index),
                )

        with tempfile.TemporaryDirectory() as raw_directory:
            archive, references = _quality_archive(
                Path(raw_directory),
                {3: (("task", "reach_progress/v1"),)},
                task_id="reach_another_cap/v1",
            )
            mismatched_task = _bound_descriptor(
                references["task"], task_id=_policy().task_id
            )
            index = build_quality_diversity_index(
                policy=_policy(), descriptors=(mismatched_task,)
            )
            with self.assertRaisesRegex(ValueError, "task binding mismatch"):
                archive.plan_quality_balanced(
                    replay_per_rank=1,
                    max_generation_exclusive=4,
                    index=index,
                    state=BalancedReplayState.initial(index),
                )

        with tempfile.TemporaryDirectory() as raw_directory:
            archive, references = _quality_archive(
                Path(raw_directory),
                {3: (("reward", "reach_progress/v1"),)},
            )
            mismatched_reward = _bound_descriptor(
                references["reward"],
                reward_profile_id="reach_smooth/v1",
            )
            index = build_quality_diversity_index(
                policy=_policy(), descriptors=(mismatched_reward,)
            )
            with self.assertRaisesRegex(
                ValueError, "reward-profile binding mismatch"
            ):
                archive.plan_quality_balanced(
                    replay_per_rank=1,
                    max_generation_exclusive=4,
                    index=index,
                    state=BalancedReplayState.initial(index),
                )

    def test_requires_exact_eligible_set(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            archive, references = _quality_archive(
                Path(raw_directory),
                {
                    3: (
                        ("a", "reach_progress/v1"),
                        ("b", "reach_progress/v1"),
                    ),
                    5: (("future", "reach_progress/v1"),),
                },
            )
            descriptors = {
                name: _bound_descriptor(reference)
                for name, reference in references.items()
            }

            missing = build_quality_diversity_index(
                policy=_policy(), descriptors=(descriptors["a"],)
            )
            with self.assertRaisesRegex(ValueError, "exactly match"):
                archive.plan_quality_balanced(
                    replay_per_rank=1,
                    max_generation_exclusive=5,
                    index=missing,
                    state=BalancedReplayState.initial(missing),
                )

            extra_reference = replace(
                references["b"], sample_id="sample/extra"
            )
            extra = build_quality_diversity_index(
                policy=_policy(),
                descriptors=(
                    descriptors["a"],
                    descriptors["b"],
                    _bound_descriptor(extra_reference),
                ),
            )
            with self.assertRaisesRegex(ValueError, "extra=sample/extra"):
                archive.plan_quality_balanced(
                    replay_per_rank=1,
                    max_generation_exclusive=5,
                    index=extra,
                    state=BalancedReplayState.initial(extra),
                )

            includes_future = build_quality_diversity_index(
                policy=_policy(), descriptors=tuple(descriptors.values())
            )
            with self.assertRaisesRegex(ValueError, "extra=sample/future"):
                archive.plan_quality_balanced(
                    replay_per_rank=1,
                    max_generation_exclusive=5,
                    index=includes_future,
                    state=BalancedReplayState.initial(includes_future),
                )

            archive.quarantine(
                references["b"],
                reason_code="candidate_replay_mismatch",
                detail="test quarantine",
                detected_generation=4,
            )
            includes_quarantined = build_quality_diversity_index(
                policy=_policy(),
                descriptors=(descriptors["a"], descriptors["b"]),
            )
            with self.assertRaisesRegex(ValueError, "extra=sample/b"):
                archive.plan_quality_balanced(
                    replay_per_rank=1,
                    max_generation_exclusive=5,
                    index=includes_quarantined,
                    state=BalancedReplayState.initial(includes_quarantined),
                )
            exact = build_quality_diversity_index(
                policy=_policy(), descriptors=(descriptors["a"],)
            )
            planned, _ = archive.plan_quality_balanced(
                replay_per_rank=10,
                max_generation_exclusive=5,
                index=exact,
                state=BalancedReplayState.initial(exact),
            )
            self.assertEqual([item.sample_id for item in planned], ["sample/a"])

    def test_plan_is_unique_clamped_and_balanced_across_calls(self) -> None:
        definitions = tuple(
            (f"crowded-{index}", "reach_progress/v1")
            for index in range(5)
        ) + (
            ("rare-profile", "reach_smooth/v1"),
            ("rare-state", "reach_progress/v1"),
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            archive, references = _quality_archive(
                Path(raw_directory), {3: definitions}
            )
            descriptors = []
            for name, reference in references.items():
                descriptors.append(
                    _bound_descriptor(
                        reference,
                        angle=2.0 if name == "rare-state" else -2.0,
                        path_length=0.8 if name == "rare-state" else 0.2,
                        clearance=0.02,
                        initial_state_group_id=(
                            "reach_reset/left/v1"
                            if name == "rare-state"
                            else "reach_reset/centre/v1"
                        ),
                    )
                )
            index = build_quality_diversity_index(
                policy=_policy(), descriptors=descriptors
            )
            planned, _ = archive.plan_quality_balanced(
                replay_per_rank=100,
                max_generation_exclusive=4,
                index=index,
                state=BalancedReplayState.initial(index),
            )
            planned_ids = [item.sample_id for item in planned]
            self.assertEqual(len(planned_ids), len(references))
            self.assertEqual(len(set(planned_ids)), len(planned_ids))

            observed = set()
            state = BalancedReplayState.initial(index)
            for _ in range(8):
                batch, state = archive.plan_quality_balanced(
                    replay_per_rank=3,
                    max_generation_exclusive=4,
                    index=index,
                    state=state,
                )
                batch_ids = [item.sample_id for item in batch]
                self.assertEqual(len(batch_ids), len(set(batch_ids)))
                observed.update(batch_ids)
            self.assertEqual(
                observed,
                {item.sample_id for item in references.values()},
            )

            first, _ = archive.plan_quality_balanced(
                replay_per_rank=3,
                max_generation_exclusive=4,
                index=index,
                state=BalancedReplayState.initial(index),
            )
            first_ids = {item.sample_id for item in first}
            self.assertIn("sample/rare-profile", first_ids)
            self.assertIn("sample/rare-state", first_ids)

    def test_resume_is_external_and_index_growth_requires_rebase(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            run_dir = Path(raw_directory)
            archive, references = _quality_archive(
                run_dir,
                {
                    3: (
                        ("a", "reach_progress/v1"),
                        ("b", "reach_smooth/v1"),
                    )
                },
            )
            previous = build_quality_diversity_index(
                policy=_policy(),
                descriptors=tuple(
                    _bound_descriptor(reference)
                    for reference in references.values()
                ),
            )
            _, advanced = archive.plan_quality_balanced(
                replay_per_rank=1,
                max_generation_exclusive=4,
                index=previous,
                state=BalancedReplayState.initial(previous),
            )
            checkpoint = archive.state_dict()
            restored = SuccessArchive(
                run_dir=run_dir,
                rank=0,
                attempt_id="restored",
                state=checkpoint,
            )
            restored_state = BalancedReplayState.from_record(
                advanced.to_record(), index=previous
            )
            expected, _ = archive.plan_quality_balanced(
                replay_per_rank=1,
                max_generation_exclusive=4,
                index=previous,
                state=advanced,
            )
            actual, _ = restored.plan_quality_balanced(
                replay_per_rank=1,
                max_generation_exclusive=4,
                index=previous,
                state=restored_state,
            )
            self.assertEqual(actual, expected)

            legacy_control = SuccessArchive(
                run_dir=run_dir,
                rank=0,
                attempt_id="legacy-control",
                state=checkpoint,
            )
            self.assertEqual(restored.cursor, legacy_control.cursor)
            self.assertEqual(
                restored.plan(
                    replay_per_rank=2, max_generation_exclusive=4
                ),
                legacy_control.plan(
                    replay_per_rank=2, max_generation_exclusive=4
                ),
            )

            references.update(
                _append_quality_generation(
                    archive=archive,
                    run_dir=run_dir,
                    generation=4,
                    definitions=(("added", "reach_progress/v1"),),
                )
            )
            replacement = build_quality_diversity_index(
                policy=_policy(),
                descriptors=tuple(
                    _bound_descriptor(reference)
                    for reference in references.values()
                ),
            )
            with self.assertRaisesRegex(ValueError, "another QD index"):
                archive.plan_quality_balanced(
                    replay_per_rank=1,
                    max_generation_exclusive=5,
                    index=replacement,
                    state=advanced,
                )
            rebased = rebase_balanced_replay_state(
                previous_index=previous,
                replacement_index=replacement,
                state=advanced,
            )
            planned, _ = archive.plan_quality_balanced(
                replay_per_rank=3,
                max_generation_exclusive=5,
                index=replacement,
                state=rebased,
            )
            self.assertEqual(len(planned), 3)
            self.assertEqual(len({item.sample_id for item in planned}), 3)


if __name__ == "__main__":
    unittest.main()
