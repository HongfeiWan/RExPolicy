"""Tests for deterministic Event Ledger construction at collection time."""

from __future__ import annotations

import unittest

from rexpolicy.flywheel.event_ledger import (
    EpisodeEventLedgerBuilder,
    production_ledger_bindings,
)
from rexpolicy.flywheel.experience import TrainingSample, format_sample_id
from rexpolicy.tasking.dynamics_contract import (
    DYNAMICS_CONTRACT_V1_SHA256,
)
from rexpolicy.tasking.event_ledger import (
    CollectionProvenance,
    ResetRecipe,
)
from rexpolicy.tasking.repository import load_production_reach_artifacts


def _signals(distance: float, *, contact=False, displacement=False):
    return {
        "reach.distance_m": distance,
        "reach.contact_violation": contact,
        "reach.displacement_violation": displacement,
    }


def _builder():
    artifacts = load_production_reach_artifacts()
    schema = artifacts.capabilities.event_schemas[0]
    bindings = production_ledger_bindings(
        task_contract_id=artifacts.contract.task_contract_id,
        task_contract_sha256=artifacts.contract.fingerprint,
        task_oracle_sha256=artifacts.contract.oracle_fingerprint,
        observation_contract_sha256="c" * 64,
        event_schema=schema,
    )
    return EpisodeEventLedgerBuilder(
        bindings=bindings,
        collection=CollectionProvenance.from_record(
            {
                "data_generation": 1,
                "sampling_policy_generation": 0,
                "rank": 0,
                "episode": 0,
                "instruction_variant_id": "train/reach_safe/v1",
            }
        ),
        reset_recipe=ResetRecipe.from_record(
            {
                "environment_adapter_id": artifacts.contract.adapter_id,
                "seed": 123,
            }
        ),
        event_schema=schema,
    ), bindings


class TestFlywheelEventLedgerBuilder(unittest.TestCase):
    def test_parallel_transitions_are_flattened_deterministically(self) -> None:
        builder, bindings = _builder()
        builder.open_decision(
            decision_index=0,
            root_control_step=0,
            dynamics_digest_sha256="d" * 64,
            current_metrics=_signals(0.10),
        )
        sample_1 = builder.append_transition(
            decision_index=0,
            world=1,
            branch_step=0,
            effective_action_19d=[0.1] * 19,
            post_dynamics_digest_sha256="f" * 64,
            current_metrics=_signals(0.09),
            terminated=False,
            truncated=False,
        )
        sample_0 = builder.append_transition(
            decision_index=0,
            world=0,
            branch_step=0,
            effective_action_19d=[0.0] * 19,
            post_dynamics_digest_sha256="e" * 64,
            current_metrics=_signals(0.08),
            terminated=False,
            truncated=False,
        )
        builder.append_transition(
            decision_index=0,
            world=1,
            branch_step=1,
            effective_action_19d=[0.2] * 19,
            post_dynamics_digest_sha256="1" * 64,
            current_metrics=_signals(0.07),
            terminated=False,
            truncated=False,
        )
        builder.close_decision(
            decision_index=0,
            root_continuation_sample_id=sample_0,
        )
        builder.open_decision(
            decision_index=1,
            root_control_step=1,
            dynamics_digest_sha256="e" * 64,
            current_metrics=_signals(0.08),
        )
        terminal = builder.append_transition(
            decision_index=1,
            world=0,
            branch_step=0,
            effective_action_19d=[0.3] * 19,
            post_dynamics_digest_sha256="2" * 64,
            current_metrics=_signals(0.03),
            terminated=True,
            truncated=False,
        )
        builder.close_decision(
            decision_index=1,
            root_continuation_sample_id=terminal,
        )
        ledger = builder.finish()

        transitions = [
            event for event in ledger.events if event.kind == "transition"
        ]
        self.assertEqual(
            tuple((event.decision_index, event.world, event.branch_step) for event in transitions),
            ((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0)),
        )
        self.assertEqual(sample_0, format_sample_id(1, 0, 0, 0, 0))
        self.assertEqual(sample_1, format_sample_id(1, 0, 0, 0, 1))
        self.assertEqual(
            bindings.dynamics_contract_sha256,
            DYNAMICS_CONTRACT_V1_SHA256,
        )
        ledger.assert_reward_agnostic()

    def test_sample_identity_matches_training_sample_provenance(self) -> None:
        sample = TrainingSample(
            backbone_features=None,
            backbone_attention_mask=None,
            image_mask=None,
            state=None,
            embodiment_id=0,
            action=None,
            action_mask=None,
            valid_steps=1,
        )
        sample.set_provenance(
            generation=1,
            rank=2,
            episode=3,
            decision=4,
            world=5,
            task_id="reach_green_cap/v1",
            reward_profile_id="reach_progress/v1",
        )
        self.assertEqual(sample.sample_id, format_sample_id(1, 2, 3, 4, 5))

    def test_builder_lifecycle_and_terminal_boundaries_fail_closed(self) -> None:
        builder, _ = _builder()
        with self.assertRaisesRegex(ValueError, "open decision"):
            builder.append_transition(
                decision_index=0,
                world=0,
                branch_step=0,
                effective_action_19d=[0.0] * 19,
                post_dynamics_digest_sha256="e" * 64,
                current_metrics=_signals(0.08),
                terminated=False,
                truncated=False,
            )
        builder.open_decision(
            decision_index=0,
            root_control_step=0,
            dynamics_digest_sha256="d" * 64,
            current_metrics=_signals(0.10),
        )
        sample_id = builder.append_transition(
            decision_index=0,
            world=0,
            branch_step=0,
            effective_action_19d=[0.0] * 19,
            post_dynamics_digest_sha256="e" * 64,
            current_metrics=_signals(0.03),
            terminated=True,
            truncated=False,
        )
        with self.assertRaisesRegex(ValueError, "after termination"):
            builder.append_transition(
                decision_index=0,
                world=0,
                branch_step=1,
                effective_action_19d=[0.0] * 19,
                post_dynamics_digest_sha256="f" * 64,
                current_metrics=_signals(0.02),
                terminated=False,
                truncated=False,
            )
        builder.close_decision(
            decision_index=0,
            root_continuation_sample_id=sample_id,
        )
        with self.assertRaisesRegex(ValueError, "terminal path"):
            builder.open_decision(
                decision_index=1,
                root_control_step=1,
                dynamics_digest_sha256="e" * 64,
                current_metrics=_signals(0.03),
            )
        builder.finish()
        with self.assertRaisesRegex(ValueError, "already finished"):
            builder.finish()

    def test_reward_or_unknown_signals_cannot_enter_the_builder(self) -> None:
        builder, _ = _builder()
        signals = _signals(0.10)
        signals["reward"] = 1.0
        with self.assertRaisesRegex(ValueError, "fields do not match"):
            builder.open_decision(
                decision_index=0,
                root_control_step=0,
                dynamics_digest_sha256="d" * 64,
                current_metrics=signals,
            )


if __name__ == "__main__":
    unittest.main()
