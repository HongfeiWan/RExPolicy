"""CPU-only tests for the one-shot Grasp-Lift selection runner core."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping

from tools.run_stage0_grasp_lift_no_z_selection import (
    CandidateEvaluation,
    EnvironmentHandle,
    PreparedSelection,
    SelectionArguments,
    SelectionOutput,
    _CANDIDATE_STEPS,
    _CHECKPOINT_STEPS,
    _TRAINING_SEEDS,
    _accepted_simulator_snapshot,
    _accepted_pilot_simulator_contract,
    _canonical_fingerprint,
    _candidate_commitments,
    _run_selection,
    _sealed,
    _strict_json,
)


def _checkpoint(step: int, *, suffix: str) -> dict[str, Any]:
    digest = (suffix * 64)[:64]
    return {
        "checkpoint_name": f"checkpoint-{step:012d}",
        "checkpoint_sha256": digest,
        "checksums_file_sha256": digest,
        "complete_file_sha256": digest,
        "manifest_file_sha256": digest,
        "model_payload_sha256": digest,
        "model_schema_sha256": digest,
        "sequence": step,
    }


def _preflight(run_directories: Mapping[int, Path]) -> dict[str, Any]:
    runs = []
    for index, seed in enumerate(_TRAINING_SEEDS, start=1):
        runs.append(
            {
                "checkpoints": [
                    _checkpoint(step, suffix=str(index)) for step in _CHECKPOINT_STEPS
                ],
                "implementation_sha256": str(index) * 64,
                "run_contract_file_sha256": str(index) * 64,
                "run_contract_sha256": str(index) * 64,
                "run_directory": str(run_directories[seed]),
                "status_file_sha256": str(index) * 64,
                "training_seed": seed,
                "world_size": 1,
            }
        )
    return _sealed(
        {
            "candidate_inventory_sha256": "a" * 64,
            "runs": runs,
            "schema_id": "test/preflight/v1",
        }
    )


def _arguments(temporary: Path) -> SelectionArguments:
    root = temporary.resolve()
    return SelectionArguments(
        repository_root=root,
        config=root / "config.json",
        pilot_directory=root / "pilot",
        training_artifact=root / "artifact",
        output_directory=root / "selection",
        device="cuda:0",
        run_directories={seed: root / f"run-{seed}" for seed in _TRAINING_SEEDS},
    )


@dataclass(frozen=True)
class _Receipt:
    claim_id: str

    def to_record(self) -> dict[str, Any]:
        return _sealed(
            {
                "claim_id": self.claim_id,
                "schema_id": "test/receipt/v1",
            }
        )


@dataclass(frozen=True)
class _Evidence:
    inventory_key: tuple[int, int]


class _Selection:
    gate_passed = True
    selected_checkpoint_step = 500

    def to_record(self) -> dict[str, Any]:
        return _sealed(
            {
                "gate_passed": self.gate_passed,
                "schema_id": "test/selection/v1",
                "selected_checkpoint_step": self.selected_checkpoint_step,
            }
        )


class _Backend:
    def __init__(self, arguments: SelectionArguments, *, fail_at: str | None = None):
        self.arguments = arguments
        self.fail_at = fail_at
        self.calls: list[str] = []
        self.preflight_record = _preflight(arguments.run_directories)
        self.receipt = _Receipt("b" * 64)
        self.validation = SimpleNamespace(
            data_sha256="c" * 64,
            normalization=object(),
            normalized_batch=object(),
            record=_sealed({"schema_id": "test/validation/v1"}),
        )
        reset_seeds = tuple(
            seed for seed in (7009, 7014, 7019, 7025, 7026, 7031) for _ in range(4)
        )
        self.budget = SimpleNamespace(
            episode_count=24,
            reset_count=6,
            reset_seeds=reset_seeds,
        )

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"injected {name} failure")

    def prepare(self, arguments: SelectionArguments) -> PreparedSelection:
        self._call("prepare")
        return PreparedSelection(
            artifact=object(),
            config=object(),
            normalization=object(),
            implementation_record={"schema_id": "test/implementation/v1"},
            implementation_sha256="d" * 64,
            evaluator_record=_sealed({"schema_id": "test/evaluator/v1"}),
            evaluator_sha256="e" * 64,
            selection_protocol_sha256="f" * 64,
            runtime_record={"schema_id": "test/runtime/v1"},
            newton_runtime=object(),
            environment_config_record={},
            task_metadata_record={},
            oracle_record={},
            accepted_simulator_record={},
            accepted_simulator_sha256="1" * 64,
        )

    def preflight(self, arguments: Any, prepared: Any) -> dict[str, Any]:
        self._call("preflight")
        return self.preflight_record

    def verify_preflight(self, preflight: Any, prepared: Any) -> None:
        self._call("verify_preflight")

    def build_claim(self, preflight: Any, prepared: Any) -> dict[str, Any]:
        self._call("build_claim")
        return _sealed(
            {
                "claim_id": self.receipt.claim_id,
                "schema_id": "test/claim/v1",
            }
        )

    def persist_claim(self, *arguments: Any) -> _Receipt:
        self._call("persist_claim")
        return self.receipt

    def verify_claim_receipt(self, *arguments: Any) -> _Receipt:
        self._call("verify_claim_receipt")
        return self.receipt

    def load_validation(self, *arguments: Any) -> Any:
        self._call("load_validation")
        return self.validation

    def build_budget(self, validation: Any) -> Any:
        self._call("build_budget")
        return self.budget

    def build_environment(self, *arguments: Any) -> EnvironmentHandle:
        self._call("build_environment")
        return EnvironmentHandle(
            raw_environment=object(),
            environment=object(),
            oracle=object(),
            record=_sealed({"schema_id": "test/environment/v1"}),
        )

    def move_validation_batch(self, validation: Any, device: str) -> object:
        self._call("move_validation_batch")
        return object()

    def evaluate_candidate(self, candidate: Any, **arguments: Any) -> Any:
        self._call(f"evaluate:{candidate.checkpoint_step}:{candidate.training_seed}")
        return CandidateEvaluation(
            evidence=_Evidence(candidate.inventory_key),
            record=_sealed(
                {
                    "checkpoint_step": candidate.checkpoint_step,
                    "schema_id": "test/candidate/v1",
                    "training_seed": candidate.training_seed,
                }
            ),
        )

    def select(self, evidence: Any) -> _Selection:
        self._call("select")
        return _Selection()

    def verify_selection(self, record: Any) -> None:
        self._call("verify_selection")

    def build_audit(self, **arguments: Any) -> dict[str, Any]:
        self._call("build_audit")
        return _sealed({"schema_id": "test/audit/v1"})

    def close_environment(self, environment: Any) -> None:
        self._call("close_environment")


class GraspLiftSelectionRunnerTests(unittest.TestCase):
    def test_task_metadata_sequence_types_share_one_json_commitment(self) -> None:
        self.assertEqual(
            _canonical_fingerprint({"effective_action_mask": (True, False)}),
            _canonical_fingerprint({"effective_action_mask": [True, False]}),
        )

    def test_read_only_simulator_record_has_a_canonical_json_snapshot(self) -> None:
        simulator = MappingProxyType(
            {
                "environment_config": {"device": "cuda:0", "num_envs": 24},
                "schema_id": "rexpolicy/stage0-grasp-lift-newton-runtime/v1",
            }
        )
        prepared = SimpleNamespace(
            accepted_simulator_record=simulator,
            accepted_simulator_sha256=_canonical_fingerprint(dict(simulator)),
        )
        snapshot = _accepted_simulator_snapshot(prepared)
        self.assertEqual(snapshot, simulator)
        self.assertIsInstance(snapshot, dict)

        prepared.accepted_simulator_sha256 = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "changed in memory"):
            _accepted_simulator_snapshot(prepared)

    def test_evaluator_is_bound_to_accepted_pilot_simulator_mechanics(self) -> None:
        accepted_environment = {
            "device": "cuda:0",
            "num_envs": 8,
            "render_images": False,
            "reward_mode": "none",
        }
        task = {"effective_action_mask": [True, False]}
        oracle = {"lift_height_m": 0.03}
        simulator = {
            "action_schema_sha256": "1" * 64,
            "authoring_mode_sha256": "2" * 64,
            "environment_config": accepted_environment,
            "implementation_sha256": "3" * 64,
            "oracle": oracle,
            "oracle_id": "test/oracle/v1",
            "schema_id": "rexpolicy/stage0-grasp-lift-newton-runtime/v1",
            "task_metadata": task,
        }
        manifest = _sealed(
            {
                "locked_test": {
                    "artifact": None,
                    "consumed": False,
                    "policy": "not_created",
                },
                "simulator_record": simulator,
                "simulator_sha256": _canonical_fingerprint(simulator),
            }
        )
        evaluation_environment = {
            **accepted_environment,
            "device": "cuda:1",
            "num_envs": 24,
        }
        observed, observed_sha256 = _accepted_pilot_simulator_contract(
            manifest,
            expected_manifest_sha256=manifest["self_sha256"],
            evaluation_environment=evaluation_environment,
            evaluation_task_metadata={"effective_action_mask": (True, False)},
            evaluation_oracle=oracle,
            action_schema_sha256="1" * 64,
            authoring_mode_sha256="2" * 64,
            oracle_id="test/oracle/v1",
        )
        self.assertEqual(observed, simulator)
        self.assertEqual(observed_sha256, manifest["simulator_sha256"])

        changed = {**evaluation_environment, "reward_mode": "dense"}
        with self.assertRaisesRegex(ValueError, "mechanics differ"):
            _accepted_pilot_simulator_contract(
                manifest,
                expected_manifest_sha256=manifest["self_sha256"],
                evaluation_environment=changed,
                evaluation_task_metadata=task,
                evaluation_oracle=oracle,
                action_schema_sha256="1" * 64,
                authoring_mode_sha256="2" * 64,
                oracle_id="test/oracle/v1",
            )

    def test_candidate_inventory_is_exact_step_major_seed_minor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            arguments = _arguments(Path(temporary))
            preflight = _preflight(arguments.run_directories)
            candidates = _candidate_commitments(
                preflight,
                arguments.run_directories,
            )
            self.assertEqual(len(candidates), 60)
            self.assertEqual(
                tuple(item.inventory_key for item in candidates),
                tuple(
                    (step, seed)
                    for step in _CANDIDATE_STEPS
                    for seed in _TRAINING_SEEDS
                ),
            )
            self.assertEqual(candidates[0].checkpoint_step, 500)
            self.assertEqual(candidates[-1].checkpoint_step, 10_000)

            missing = dict(preflight)
            missing_runs = [dict(item) for item in preflight["runs"]]
            missing_runs[1]["checkpoints"] = missing_runs[1]["checkpoints"][:-1]
            missing["runs"] = missing_runs
            with self.assertRaisesRegex(ValueError, "inventory size"):
                _candidate_commitments(missing, arguments.run_directories)

            reordered = dict(preflight)
            reordered_runs = [dict(item) for item in preflight["runs"]]
            checkpoints = list(reordered_runs[0]["checkpoints"])
            checkpoints[1], checkpoints[2] = checkpoints[2], checkpoints[1]
            reordered_runs[0]["checkpoints"] = checkpoints
            reordered["runs"] = reordered_runs
            with self.assertRaisesRegex(ValueError, "order or identity"):
                _candidate_commitments(reordered, arguments.run_directories)

    def test_claim_is_durable_before_validation_newton_and_sixty_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            arguments = _arguments(Path(temporary))
            backend = _Backend(arguments)
            status = _run_selection(arguments, backend)

            self.assertTrue(status["complete"])
            self.assertEqual(status["evaluated_candidates"], 60)
            self.assertLess(
                backend.calls.index("persist_claim"),
                backend.calls.index("verify_claim_receipt"),
            )
            self.assertLess(
                backend.calls.index("verify_claim_receipt"),
                backend.calls.index("load_validation"),
            )
            self.assertLess(
                backend.calls.index("load_validation"),
                backend.calls.index("build_environment"),
            )
            evaluations = [
                item for item in backend.calls if item.startswith("evaluate:")
            ]
            self.assertEqual(len(evaluations), 60)
            self.assertEqual(evaluations[0], "evaluate:500:31001")
            self.assertEqual(evaluations[-1], "evaluate:10000:31003")
            self.assertEqual(backend.calls.count("move_validation_batch"), 1)
            self.assertEqual(backend.calls[-1], "close_environment")
            self.assertEqual(
                {item.name for item in arguments.output_directory.iterdir()},
                {
                    "audit.json",
                    "claim.json",
                    "preflight.json",
                    "selection.json",
                    "status.json",
                },
            )

    def test_post_claim_failure_is_burned_and_preclaim_failure_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            arguments = _arguments(Path(temporary))
            backend = _Backend(arguments, fail_at="load_validation")
            with self.assertRaisesRegex(RuntimeError, "load_validation"):
                _run_selection(arguments, backend)
            status = _strict_json(arguments.output_directory / "status.json")
            self.assertEqual(status["state"], "burned")
            self.assertTrue(status["validation_claimed"])
            self.assertIn("forbidden", status["retry_policy"])
            self.assertNotIn("build_environment", backend.calls)

        with tempfile.TemporaryDirectory() as temporary:
            arguments = _arguments(Path(temporary))
            backend = _Backend(arguments, fail_at="preflight")
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                _run_selection(arguments, backend)
            status = _strict_json(arguments.output_directory / "status.json")
            self.assertEqual(status["state"], "error")
            self.assertFalse(status["validation_claimed"])

    def test_write_once_records_cannot_be_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = SelectionOutput(Path(temporary).resolve() / "selection")
            output.create()
            original = _sealed({"schema_id": "test/selection/v1"})
            output.write_once("selection.json", original)
            with self.assertRaises(FileExistsError):
                output.write_once(
                    "selection.json",
                    _sealed({"schema_id": "test/replacement/v1"}),
                )
            self.assertEqual(
                _strict_json(output.root / "selection.json"),
                original,
            )


if __name__ == "__main__":
    unittest.main()
