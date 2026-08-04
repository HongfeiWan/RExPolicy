"""Tests for policy-bound ProcessSpec compilation."""

from __future__ import annotations

import unittest
from dataclasses import replace

from rexpolicy.tasking.capabilities import CapabilityCatalog
from rexpolicy.tasking.contract import compile_task_contract
from rexpolicy.tasking.model import TaskSpecV2
from rexpolicy.tasking.process import ProcessSpecV1
from rexpolicy.tasking.process_contract import (
    DEFAULT_PROCESS_COMPILER_POLICY,
    ProcessCompilerPolicy,
    compile_process_contract,
    validate_compiled_process_contract,
)
from tools.test_tasking_capabilities import capability_record
from tools.test_tasking_model import task_record
from tools.test_tasking_process import process_record


def _compiled(process: dict | None = None):
    task = TaskSpecV2.from_record(task_record())
    catalog = CapabilityCatalog.from_record(capability_record())
    task_contract = compile_task_contract(task, catalog=catalog)
    spec = ProcessSpecV1.from_record(process or process_record())
    process_contract = compile_process_contract(
        spec,
        task=task,
        task_contract=task_contract,
        catalog=catalog,
    )
    return task, catalog, task_contract, spec, process_contract


class TestProcessContract(unittest.TestCase):
    def test_compile_binds_every_trusted_dependency(self) -> None:
        task, catalog, task_contract, spec, contract = _compiled()

        self.assertEqual(contract.process_spec_fingerprint, spec.fingerprint)
        self.assertEqual(contract.task_spec_fingerprint, task.fingerprint)
        self.assertEqual(contract.task_contract_fingerprint, task_contract.fingerprint)
        self.assertEqual(contract.capability_catalog_fingerprint, catalog.fingerprint)
        self.assertEqual(contract.default_stage_id, "far")
        self.assertEqual(
            [stage.stage_id for stage in contract.stages],
            ["unsafe", "goal_zone", "approach"],
        )
        self.assertEqual(contract.required_metrics, (
            "reach.contact_violation",
            "reach.displacement_violation",
            "reach.distance_m",
        ))
        self.assertEqual(len(contract.fingerprint), 64)

    def test_process_and_task_identity_must_match(self) -> None:
        task = TaskSpecV2.from_record(task_record())
        catalog = CapabilityCatalog.from_record(capability_record())
        task_contract = compile_task_contract(task, catalog=catalog)
        changed = process_record()
        changed["task_id"] = "other_task/v2"
        with self.assertRaisesRegex(ValueError, "task identity mismatch"):
            compile_process_contract(
                ProcessSpecV1.from_record(changed),
                task=task,
                task_contract=task_contract,
                catalog=catalog,
            )

    def test_stage_predicates_cannot_read_previous_state(self) -> None:
        changed = process_record()
        changed["stages"][2]["predicate"] = {
            "op": "lt",
            "args": [
                {
                    "op": "metric",
                    "name": "reach.distance_m",
                    "at": "current",
                },
                {
                    "op": "metric",
                    "name": "reach.distance_m",
                    "at": "previous",
                },
            ],
        }
        task = TaskSpecV2.from_record(task_record())
        catalog = CapabilityCatalog.from_record(capability_record())
        task_contract = compile_task_contract(task, catalog=catalog)

        with self.assertRaisesRegex(ValueError, "current-state metrics"):
            compile_process_contract(
                ProcessSpecV1.from_record(changed),
                task=task,
                task_contract=task_contract,
                catalog=catalog,
            )

    def test_stage_and_segment_limits_are_policy_owned(self) -> None:
        task = TaskSpecV2.from_record(task_record())
        catalog = CapabilityCatalog.from_record(capability_record())
        task_contract = compile_task_contract(task, catalog=catalog)
        spec = ProcessSpecV1.from_record(process_record())
        policy = ProcessCompilerPolicy(
            policy_id="rexpolicy/process_compiler/test",
            expression_limits=DEFAULT_PROCESS_COMPILER_POLICY.expression_limits,
            max_stages=2,
            max_segment_steps=3,
            require_current_state_predicates=True,
        )
        with self.assertRaisesRegex(ValueError, "stage limit"):
            compile_process_contract(
                spec,
                task=task,
                task_contract=task_contract,
                catalog=catalog,
                process_policy=policy,
            )

        segment_record = process_record()
        segment_record["segmentation"]["max_segment_steps"] = 9
        with self.assertRaisesRegex(ValueError, "trusted horizon"):
            compile_process_contract(
                ProcessSpecV1.from_record(segment_record),
                task=task,
                task_contract=task_contract,
                catalog=catalog,
            )

    def test_tampered_compiled_program_fails_validation(self) -> None:
        _, catalog, task_contract, _, contract = _compiled()
        stage = contract.stages[0]
        tampered_stage = replace(stage, program_fingerprint="0" * 64)
        tampered = replace(
            contract,
            stages=(tampered_stage, *contract.stages[1:]),
        )

        with self.assertRaisesRegex(ValueError, "program fingerprint mismatch"):
            validate_compiled_process_contract(
                tampered,
                task_contract=task_contract,
                catalog=catalog,
            )


if __name__ == "__main__":
    unittest.main()
