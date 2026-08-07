"""CPU tests for the train-only Grasp-Lift evaluator parity smoke."""

from __future__ import annotations

import ast
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import tools.smoke_stage0_grasp_lift_no_z_evaluator as smoke


def _member(index: int, seed: int) -> Any:
    return SimpleNamespace(
        reset_seed=seed,
        trajectory=SimpleNamespace(
            provenance=SimpleNamespace(
                reset_group_id=f"train-group-{index:02d}",
                reset_seed=seed,
            )
        ),
    )


def _artifact() -> Any:
    remaining = tuple(range(8000, 8018))
    members = tuple(
        _member(index, seed)
        for index, seed in enumerate(smoke._TRAIN_SMOKE_SEEDS + remaining)
    )
    corpus = SimpleNamespace(
        train=members,
        training_membership_sha256="1" * 64,
    )
    return SimpleNamespace(
        artifact_sha256="2" * 64,
        data=SimpleNamespace(corpus=corpus),
        train_content_sha256="3" * 64,
    )


class _Budget:
    def __init__(self, groups: tuple[tuple[str, int], ...]):
        self.groups = groups
        self.episode_count = 24
        self.reset_count = 6
        self.reset_seeds = tuple(seed for _, seed in groups for _ in range(4))

    def to_record(self) -> dict[str, Any]:
        return {
            "allocated_world_steps": 24 * 72,
            "cells": [
                {
                    "noise_seed": index,
                    "noise_variant": variant,
                    "reset_group_id": group,
                    "reset_seed": seed,
                }
                for index, (group, seed, variant) in enumerate(
                    (group, seed, variant)
                    for group, seed in self.groups
                    for variant in range(4)
                )
            ],
            "max_steps_per_episode": 72,
            "noise_variants_per_reset": 4,
            "reset_count": 6,
            "schema_id": "test/train-smoke-budget/v1",
        }

    @property
    def sha256(self) -> str:
        return smoke._fingerprint(self.to_record())


class _Rollout:
    def __init__(
        self,
        groups: tuple[tuple[str, int], ...],
        *,
        execution_index: int | None = None,
        disagreement_index: int | None = None,
        overflow_index: int | None = None,
        ordinary_safety: bool = True,
    ):
        self.groups = groups
        self.execution_index = execution_index
        self.disagreement_index = disagreement_index
        self.overflow_index = overflow_index
        self.ordinary_safety = ordinary_safety

    def to_record(self) -> dict[str, Any]:
        cells = []
        for index, (group, seed, variant) in enumerate(
            (group, seed, variant)
            for group, seed in self.groups
            for variant in range(4)
        ):
            cells.append(
                {
                    "collision_buffer_overflow_violation": (
                        index == self.overflow_index
                    ),
                    "completed_steps": 1,
                    "execution_integrity_violation": (index == self.execution_index),
                    "noise_variant": variant,
                    "oracle_input_disagreement": (index == self.disagreement_index),
                    "ordinary_safety_violation": self.ordinary_safety,
                    "reset_group_id": group,
                    "reset_seed": seed,
                    "success": False,
                }
            )
        return {
            "budget_sha256": _Budget(self.groups).sha256,
            "cells": cells,
            "execution_integrity_violation_count": sum(
                cell["execution_integrity_violation"] for cell in cells
            ),
            "oracle_input_disagreement_count": sum(
                cell["oracle_input_disagreement"] for cell in cells
            ),
            "safety_violation_count": len(cells) if self.ordinary_safety else 0,
            "schema_id": "test/train-smoke-rollout/v1",
            "success_count": 0,
        }

    @property
    def sha256(self) -> str:
        return smoke._fingerprint(self.to_record())


def _arguments(temporary: Path) -> smoke.SmokeArguments:
    root = temporary.resolve()
    return smoke.SmokeArguments(
        repository_root=root,
        config=root / "config.json",
        pilot_directory=root / "pilot",
        training_artifact=root / "artifact",
        output=root / "smoke.json",
        device="cuda:0",
    )


class _Backend:
    def __init__(
        self,
        *,
        rollout: _Rollout | None = None,
        rollout_error: BaseException | None = None,
    ):
        self.artifact = _artifact()
        self.groups = smoke._train_reset_groups(self.artifact)
        self.budget = _Budget(self.groups)
        self.rollout_result = rollout or _Rollout(self.groups)
        self.rollout_error = rollout_error
        self.calls: list[str] = []

    def prepare(self, arguments: Any) -> Any:
        self.calls.append("prepare")
        return SimpleNamespace(
            artifact=self.artifact,
            config=SimpleNamespace(sha256="4" * 64),
            evaluator_sha256="7" * 64,
            implementation_sha256="5" * 64,
            normalization=SimpleNamespace(sha256="6" * 64),
            runtime_record={"schema_id": "test/runtime/v1"},
        )

    def build_budget(self, prepared: Any) -> tuple[_Budget, Any]:
        self.calls.append("build_budget")
        return self.budget, self.groups

    def build_environment(self, *arguments: Any) -> Any:
        self.calls.append("build_environment")
        return SimpleNamespace(
            environment=object(),
            oracle=object(),
            raw_environment=object(),
            record={
                "rollout_budget_sha256": self.budget.sha256,
                "schema_id": "test/environment/v1",
            },
        )

    def build_policy(self, *arguments: Any) -> tuple[Any, dict[str, Any]]:
        self.calls.append("build_policy")
        return object(), smoke._sealed({"schema_id": "test/policy/v1"})

    def rollout(self, *arguments: Any) -> _Rollout:
        self.calls.append("rollout")
        if self.rollout_error is not None:
            raise self.rollout_error
        return self.rollout_result

    def close_environment(self, environment: Any) -> None:
        self.calls.append("close_environment")


class GraspLiftEvaluatorSmokeTests(unittest.TestCase):
    def test_first_six_train_resets_form_exact_reset_major_four_noise_grid(
        self,
    ) -> None:
        artifact = _artifact()
        groups = smoke._train_reset_groups(artifact)
        self.assertEqual(
            tuple(seed for _, seed in groups),
            smoke._TRAIN_SMOKE_SEEDS,
        )
        budget = _Budget(groups)
        cells = budget.to_record()["cells"]
        self.assertEqual(len(cells), 24)
        self.assertEqual(
            tuple((cell["reset_seed"], cell["noise_variant"]) for cell in cells),
            tuple(
                (seed, variant)
                for seed in smoke._TRAIN_SMOKE_SEEDS
                for variant in range(4)
            ),
        )

        changed = _artifact()
        changed.data.corpus.train = (
            _member(0, 9999),
            *changed.data.corpus.train[1:],
        )
        with self.assertRaisesRegex(ValueError, "first six formal train seeds"):
            smoke._train_reset_groups(changed)

    def test_gate_ignores_success_and_ordinary_safety_but_not_integrity(self) -> None:
        groups = smoke._train_reset_groups(_artifact())
        budget = _Budget(groups)
        ordinary_failure = _Rollout(groups, ordinary_safety=True)
        gate = smoke._gate_rollout_record(
            ordinary_failure.to_record(),
            budget_record=budget.to_record(),
        )
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["success_is_not_a_gate"])
        self.assertTrue(gate["ordinary_safety_is_not_a_gate"])

        for rollout in (
            _Rollout(groups, execution_index=3),
            _Rollout(groups, disagreement_index=7),
            _Rollout(groups, overflow_index=11),
        ):
            self.assertFalse(
                smoke._gate_rollout_record(
                    rollout.to_record(),
                    budget_record=budget.to_record(),
                )["passed"]
            )

        changed_group = ordinary_failure.to_record()
        changed_group["cells"][0]["reset_group_id"] = "different-group"
        with self.assertRaisesRegex(ValueError, "fixed train-reset"):
            smoke._gate_rollout_record(
                changed_group,
                budget_record=budget.to_record(),
            )

        changed_budget = ordinary_failure.to_record()
        changed_budget["budget_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "exact budget"):
            smoke._gate_rollout_record(
                changed_budget,
                budget_record=budget.to_record(),
            )

    def test_workflow_is_train_only_write_once_and_self_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            arguments = _arguments(Path(temporary))
            backend = _Backend()
            result = smoke._run_smoke(arguments, backend)

            self.assertTrue(result["gate"]["passed"])
            self.assertTrue(result["train_only"])
            self.assertEqual(result["data_scope"]["held_out_shards_opened"], 0)
            self.assertEqual(result["data_scope"]["train_reset_count"], 6)
            self.assertEqual(
                tuple(item["reset_seed"] for item in result["reset_groups"]),
                smoke._TRAIN_SMOKE_SEEDS,
            )
            smoke._verify_sealed(result, "test smoke output")
            self.assertEqual(smoke._strict_json(arguments.output), result)
            self.assertEqual(backend.calls.count("rollout"), 1)
            self.assertEqual(backend.calls.count("close_environment"), 1)
            with self.assertRaises(FileExistsError):
                smoke._run_smoke(arguments, backend)
            self.assertEqual(backend.calls.count("rollout"), 1)

    def test_exception_writes_failure_and_same_output_cannot_be_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            arguments = _arguments(Path(temporary))
            backend = _Backend(rollout_error=RuntimeError("injected rollout"))
            with self.assertRaisesRegex(RuntimeError, "injected rollout"):
                smoke._run_smoke(arguments, backend)
            failure = smoke._strict_json(arguments.output)
            self.assertFalse(failure["complete"])
            self.assertEqual(failure["state"], "error")
            self.assertTrue(failure["train_only"])
            smoke._verify_sealed(failure, "test smoke failure")
            calls = tuple(backend.calls)
            with self.assertRaises(FileExistsError):
                smoke._run_smoke(arguments, backend)
            self.assertEqual(tuple(backend.calls), calls)

    def test_gate_failure_is_sealed_and_cannot_be_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            arguments = _arguments(Path(temporary))
            groups = smoke._train_reset_groups(_artifact())
            backend = _Backend(rollout=_Rollout(groups, execution_index=0))
            result = smoke._run_smoke(arguments, backend)
            self.assertFalse(result["gate"]["passed"])
            self.assertEqual(result["state"], "gate_failed")
            smoke._verify_sealed(result, "test failed smoke gate")
            calls = tuple(backend.calls)
            with self.assertRaises(FileExistsError):
                smoke._run_smoke(arguments, backend)
            self.assertEqual(tuple(backend.calls), calls)

    def test_module_has_no_held_out_data_or_claim_import(self) -> None:
        tree = ast.parse(inspect.getsource(smoke))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertFalse(
            any(
                name.endswith(("grasp_lift_validation", "grasp_lift_validation_claim"))
                for name in imported
            )
        )
        source = inspect.getsource(smoke.DefaultSmokeBackend)
        self.assertNotIn("FORMAL_VALIDATION_SEEDS", source)
        self.assertNotIn("load_claimed_grasp_lift_validation", source)
        self.assertNotIn("persist_grasp_lift_validation_claim", source)


if __name__ == "__main__":
    unittest.main()
