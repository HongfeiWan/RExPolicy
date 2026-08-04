"""Tests for rebuildable QD indexing and balanced Success Archive replay."""

from __future__ import annotations

import copy
import hashlib
import math
import unittest
import tempfile
from pathlib import Path

from rexpolicy.flywheel.derived_views import write_derived_view
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


if __name__ == "__main__":
    unittest.main()
