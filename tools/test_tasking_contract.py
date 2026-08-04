"""Tests for immutable policy-bound compiled task contracts."""

from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace

from rexpolicy.tasking.capabilities import CapabilityCatalog
from rexpolicy.tasking.contract import (
    DEFAULT_TASK_COMPILER_POLICY,
    compile_task_contract,
    validate_compiled_task_contract,
)
from rexpolicy.tasking.model import TaskSpecV2
from tools.test_tasking_capabilities import capability_record
from tools.test_tasking_model import task_record


def compiled_contract():
    return compile_task_contract(
        TaskSpecV2.from_record(task_record()),
        catalog=CapabilityCatalog.from_record(capability_record()),
    )


class TestCompiledTaskContract(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = CapabilityCatalog.from_record(capability_record())
        self.task = TaskSpecV2.from_record(task_record())

    def test_reach_contract_binds_all_trusted_dependencies(self) -> None:
        contract = compile_task_contract(self.task, catalog=self.catalog)

        self.assertEqual(
            contract.task_contract_id,
            "rexpolicy_taskspec_v2/reach_green_cap/v2",
        )
        self.assertEqual(contract.task_spec_fingerprint, self.task.fingerprint)
        self.assertEqual(
            contract.capability_catalog_fingerprint,
            self.catalog.fingerprint,
        )
        self.assertEqual(
            contract.compiler_policy_fingerprint,
            DEFAULT_TASK_COMPILER_POLICY.fingerprint,
        )
        self.assertEqual(contract.goal_program.result_unit, "bool")
        self.assertEqual(
            contract.required_metrics,
            (
                "reach.contact_violation",
                "reach.displacement_violation",
                "reach.distance_m",
            ),
        )
        self.assertEqual(len(contract.fingerprint), 64)
        self.assertEqual(len(contract.oracle_fingerprint), 64)
        self.assertNotIn("reward_profiles", contract.oracle_record())
        json.dumps(contract.to_record(), allow_nan=False)
        validate_compiled_task_contract(
            contract,
            catalog=self.catalog,
        )

    def test_goal_hold_must_fit_the_trusted_horizon(self) -> None:
        record = task_record()
        record["goal"]["hold_steps"] = 9
        with self.assertRaisesRegex(ValueError, "episode horizon"):
            compile_task_contract(
                TaskSpecV2.from_record(record),
                catalog=self.catalog,
            )

    def test_legacy_task_and_reward_identities_are_reserved(self) -> None:
        task = task_record()
        task["task_id"] = "reach_green_cap/v1"
        task["version"] = 1
        with self.assertRaisesRegex(ValueError, "legacy semantics"):
            compile_task_contract(
                TaskSpecV2.from_record(task),
                catalog=self.catalog,
            )

        reward = task_record()
        reward["reward_profiles"][0]["reward_profile_id"] = (
            "reach_progress/v1"
        )
        with self.assertRaisesRegex(ValueError, "legacy semantics"):
            compile_task_contract(
                TaskSpecV2.from_record(reward),
                catalog=self.catalog,
            )

    def test_terminal_kind_cannot_escalate_metric_permissions(self) -> None:
        task = task_record()
        task["terminal_rules"][0]["kind"] = "failure"
        with self.assertRaisesRegex(ValueError, "failure expression cannot use"):
            compile_task_contract(
                TaskSpecV2.from_record(task),
                catalog=self.catalog,
            )

    def test_reward_bounds_and_weights_are_policy_checked(self) -> None:
        ordering = task_record()
        ordering["reward_profiles"][0]["terminal_values"]["failure"] = 0.0
        with self.assertRaisesRegex(ValueError, "bound the shaping interval"):
            compile_task_contract(
                TaskSpecV2.from_record(ordering),
                catalog=self.catalog,
            )

        weight = task_record()
        weight["reward_profiles"][0]["terms"][0]["weight"] = 0.0
        with self.assertRaisesRegex(ValueError, "weight"):
            compile_task_contract(
                TaskSpecV2.from_record(weight),
                catalog=self.catalog,
            )

    def test_contract_is_detached_from_source_objects(self) -> None:
        task_record_value = task_record()
        capability_record_value = capability_record()
        task = TaskSpecV2.from_record(task_record_value)
        catalog = CapabilityCatalog.from_record(capability_record_value)
        contract = compile_task_contract(task, catalog=catalog)
        fingerprint = contract.fingerprint

        task_record_value["goal"]["hold_steps"] = 10
        capability_record_value["adapters"][0][
            "max_episode_control_steps"
        ] = 1
        self.assertEqual(contract.fingerprint, fingerprint)
        with self.assertRaises(TypeError):
            contract.environment_parameters[0][1][0] = 0.0

    def test_tampered_program_and_dependency_drift_fail_closed(self) -> None:
        contract = compiled_contract()
        tampered_goal = replace(contract.goal_program, purpose="safety")
        tampered = replace(contract, goal_program=tampered_goal)
        with self.assertRaisesRegex(ValueError, "program fingerprint mismatch"):
            validate_compiled_task_contract(
                tampered,
                catalog=self.catalog,
            )

        drift = capability_record()
        drift["event_schemas"][0]["metrics"][0]["maximum"] = 3.0
        with self.assertRaisesRegex(ValueError, "catalog fingerprint mismatch"):
            validate_compiled_task_contract(
                contract,
                catalog=CapabilityCatalog.from_record(drift),
            )

        changed_policy = replace(
            DEFAULT_TASK_COMPILER_POLICY,
            max_goal_hold_steps=128,
        )
        with self.assertRaisesRegex(ValueError, "policy fingerprint mismatch"):
            validate_compiled_task_contract(
                contract,
                catalog=self.catalog,
                policy=changed_policy,
            )

    def test_semantic_change_changes_contract_fingerprint(self) -> None:
        changed = copy.deepcopy(task_record())
        changed["goal"]["hold_steps"] = 3
        contracts = (
            compile_task_contract(self.task, catalog=self.catalog),
            compile_task_contract(
                TaskSpecV2.from_record(changed),
                catalog=self.catalog,
            ),
        )
        self.assertNotEqual(contracts[0].fingerprint, contracts[1].fingerprint)
        self.assertNotEqual(
            contracts[0].oracle_fingerprint,
            contracts[1].oracle_fingerprint,
        )

    def test_reward_only_change_preserves_task_oracle_identity(self) -> None:
        changed = copy.deepcopy(task_record())
        changed["reward_profiles"][0]["reward_profile_id"] = (
            "reach_progress_alt/v2"
        )
        changed["reward_profiles"][0]["terms"][0]["weight"] = 0.5
        changed["reward_profiles"][0]["shaping_bounds"] = [-0.5, 0.5]
        original = compile_task_contract(self.task, catalog=self.catalog)
        alternate = compile_task_contract(
            TaskSpecV2.from_record(changed),
            catalog=self.catalog,
        )

        self.assertNotEqual(original.fingerprint, alternate.fingerprint)
        self.assertNotEqual(
            original.task_spec_fingerprint,
            alternate.task_spec_fingerprint,
        )
        self.assertEqual(
            original.oracle_fingerprint,
            alternate.oracle_fingerprint,
        )


if __name__ == "__main__":
    unittest.main()
