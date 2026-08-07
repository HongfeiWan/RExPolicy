#!/usr/bin/env python3
"""Run a train-only exact-Newton parity smoke before validation is claimed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.run_stage0_grasp_lift_no_z_selection import (
    DefaultSelectionBackend,
    SelectionArguments,
    _configure_torch,
    _restrict_preimport_runtime,
    _sealed,
    _strict_json,
    _verify_sealed,
    _write_json_exclusive,
)

_TRAIN_SMOKE_SEEDS = (7000, 7002, 7003, 7004, 7006, 7007)
_NOISE_VARIANTS = (0, 1, 2, 3)
_EPISODE_COUNT = len(_TRAIN_SMOKE_SEEDS) * len(_NOISE_VARIANTS)
_MAX_EPISODE_STEPS = 72
_POLICY_INITIALIZATION_SEED = 20_260_807
_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-no-z-evaluator-smoke/v1"
_LOCKED_TEST_ABSENT = {
    "artifact": None,
    "consumed": False,
    "policy": "not_created",
}


@dataclass(frozen=True)
class SmokeArguments:
    repository_root: Path
    config: Path
    pilot_directory: Path
    training_artifact: Path
    output: Path
    device: str

    def __post_init__(self) -> None:
        for name in (
            "repository_root",
            "config",
            "pilot_directory",
            "training_artifact",
            "output",
        ):
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError(f"{name} must be an absolute pathlib.Path")
        if self.output.suffix != ".json":
            raise ValueError("output must be one explicit .json path")
        if (
            not isinstance(self.device, str)
            or not self.device.startswith("cuda:")
            or not self.device.removeprefix("cuda:").isdigit()
        ):
            raise ValueError("device must be an explicit cuda:N device")


def _selection_arguments(arguments: SmokeArguments) -> SelectionArguments:
    dummy = {
        seed: arguments.repository_root / f".evaluator-smoke-unused-run-{seed}"
        for seed in (31_001, 31_002, 31_003)
    }
    return SelectionArguments(
        repository_root=arguments.repository_root,
        config=arguments.config,
        pilot_directory=arguments.pilot_directory,
        validation_cohort_directory=(
            arguments.repository_root / ".evaluator-smoke-unused-validation-cohort"
        ),
        training_artifact=arguments.training_artifact,
        output_directory=arguments.output.parent / ".unused-selection-output",
        device=arguments.device,
        run_directories=dummy,
    )


def _train_reset_groups(artifact: Any) -> tuple[tuple[str, int], ...]:
    corpus = getattr(getattr(artifact, "data", None), "corpus", None)
    train = getattr(corpus, "train", None)
    if not isinstance(train, tuple) or len(train) != 24:
        raise ValueError("smoke requires the accepted 24-member train corpus")
    selected = train[: len(_TRAIN_SMOKE_SEEDS)]
    observed_seeds = tuple(member.reset_seed for member in selected)
    if observed_seeds != _TRAIN_SMOKE_SEEDS:
        raise ValueError("the fixed first six formal train seeds changed")
    groups: list[tuple[str, int]] = []
    for member, seed in zip(selected, _TRAIN_SMOKE_SEEDS, strict=True):
        provenance = member.trajectory.provenance
        if (
            provenance.reset_seed != seed
            or not isinstance(provenance.reset_group_id, str)
            or not provenance.reset_group_id
        ):
            raise ValueError("train smoke reset provenance changed")
        groups.append((provenance.reset_group_id, seed))
    result = tuple(groups)
    if len({group for group, _ in result}) != len(result):
        raise ValueError("train smoke reset groups must be unique")
    return result


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def _policy_record(policy: Any, freeze_record: Mapping[str, Any]) -> dict[str, Any]:
    state = policy.state_dict()
    tensors = [
        {
            "dtype": str(value.dtype),
            "name": name,
            "payload_sha256": _tensor_sha256(value),
            "shape": list(value.shape),
        }
        for name, value in state.items()
    ]
    if not tensors:
        raise ValueError("smoke policy state cannot be empty")
    return _sealed(
        {
            "freeze_record": dict(freeze_record),
            "initialization_seed": _POLICY_INITIALIZATION_SEED,
            "schema_id": "rexpolicy/stage0-grasp-lift-no-z-smoke-policy/v1",
            "state_tensor_count": len(tensors),
            "state_tensors_sha256": _fingerprint(tensors),
        }
    )


def _fingerprint(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError("smoke fingerprint input is not canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _gate_rollout_record(
    rollout_record: Mapping[str, Any],
    *,
    budget_record: Mapping[str, Any],
) -> dict[str, Any]:
    cells = rollout_record.get("cells")
    budget_cells = budget_record.get("cells")
    if not isinstance(cells, list) or len(cells) != _EPISODE_COUNT:
        raise ValueError("smoke rollout must contain exactly 24 cells")
    if not isinstance(budget_cells, list) or len(budget_cells) != _EPISODE_COUNT:
        raise ValueError("smoke budget must contain exactly 24 cells")
    if (
        budget_record.get("max_steps_per_episode") != _MAX_EPISODE_STEPS
        or budget_record.get("allocated_world_steps")
        != _EPISODE_COUNT * _MAX_EPISODE_STEPS
        or budget_record.get("reset_count") != len(_TRAIN_SMOKE_SEEDS)
        or budget_record.get("noise_variants_per_reset") != len(_NOISE_VARIANTS)
    ):
        raise ValueError("smoke budget dimensions changed")
    expected = tuple(
        (seed, variant) for seed in _TRAIN_SMOKE_SEEDS for variant in _NOISE_VARIANTS
    )
    observed_budget = tuple(
        (
            cell.get("reset_group_id"),
            cell.get("reset_seed"),
            cell.get("noise_variant"),
        )
        for cell in budget_cells
        if isinstance(cell, Mapping)
    )
    observed_rollout = tuple(
        (
            cell.get("reset_group_id"),
            cell.get("reset_seed"),
            cell.get("noise_variant"),
        )
        for cell in cells
        if isinstance(cell, Mapping)
    )
    if (
        tuple((seed, variant) for _, seed, variant in observed_budget) != expected
        or observed_rollout != observed_budget
    ):
        raise ValueError("smoke cells are not the fixed train-reset 6 by 4 grid")
    if any(not isinstance(group, str) or not group for group, _, _ in observed_budget):
        raise ValueError("smoke budget reset groups changed")
    if rollout_record.get("budget_sha256") != _fingerprint(budget_record):
        raise ValueError("smoke rollout is not bound to the exact budget")
    if any(
        type(cell.get("completed_steps")) is not int
        or not 1 <= cell["completed_steps"] <= _MAX_EPISODE_STEPS
        for cell in cells
    ):
        raise ValueError("smoke rollout completed steps are outside the budget")
    execution_count = rollout_record.get("execution_integrity_violation_count")
    disagreement_count = rollout_record.get("oracle_input_disagreement_count")
    cell_execution_count = sum(
        cell.get("execution_integrity_violation") is True for cell in cells
    )
    cell_disagreement_count = sum(
        cell.get("oracle_input_disagreement") is True for cell in cells
    )
    overflow_count = sum(
        cell.get("collision_buffer_overflow_violation") is True for cell in cells
    )
    if (
        type(execution_count) is not int
        or execution_count < 0
        or type(disagreement_count) is not int
        or disagreement_count < 0
    ):
        raise ValueError("smoke rollout integrity counters changed")
    if (
        execution_count != cell_execution_count
        or disagreement_count != cell_disagreement_count
    ):
        raise ValueError("smoke rollout integrity counters disagree with cells")
    checks = {
        "collision_buffer_overflow_count": overflow_count,
        "episode_count": len(cells),
        "execution_integrity_violation_count": execution_count,
        "oracle_input_disagreement_count": disagreement_count,
        "reward_check": "every rollout transition folded into execution integrity",
    }
    return {
        "checks": checks,
        "ordinary_safety_is_not_a_gate": True,
        "passed": (
            len(cells) == _EPISODE_COUNT
            and execution_count == 0
            and disagreement_count == 0
            and overflow_count == 0
        ),
        "success_is_not_a_gate": True,
    }


class DefaultSmokeBackend:
    def __init__(
        self,
        *,
        torch: Any,
        affinity: tuple[int, ...],
        cuda_runtime: Mapping[str, Any],
    ):
        self.torch = torch
        self.selection_backend = DefaultSelectionBackend(
            torch=torch,
            affinity=affinity,
            cuda_runtime=cuda_runtime,
        )

    def prepare(self, arguments: SmokeArguments) -> Any:
        from rexpolicy.stage0.data.grasp_lift_training import (
            GRASP_LIFT_FORMAL_TRAIN_SEEDS,
        )

        if tuple(GRASP_LIFT_FORMAL_TRAIN_SEEDS[:6]) != _TRAIN_SMOKE_SEEDS:
            raise RuntimeError("formal train smoke seed prefix changed")
        return self.selection_backend.prepare(
            _selection_arguments(arguments),
            require_validation_cohort=False,
        )

    def build_budget(self, prepared: Any) -> tuple[Any, tuple[tuple[str, int], ...]]:
        from rexpolicy.stage0.evaluation.grasp_lift_rollout import (
            GraspLiftRolloutBudget,
        )

        groups = _train_reset_groups(prepared.artifact)
        budget = GraspLiftRolloutBudget.from_reset_groups(
            groups,
            max_steps_per_episode=_MAX_EPISODE_STEPS,
        )
        if budget.episode_count != _EPISODE_COUNT or budget.reset_count != 6:
            raise RuntimeError("train smoke budget is not six resets by four noises")
        return budget, groups

    def build_environment(
        self,
        arguments: SmokeArguments,
        prepared: Any,
        budget: Any,
    ) -> Any:
        return self.selection_backend.build_environment(
            _selection_arguments(arguments),
            prepared,
            budget,
        )

    def build_policy(self, arguments: SmokeArguments, prepared: Any) -> tuple[Any, Any]:
        from rexpolicy.stage0.grasp_lift_no_z import build_grasp_lift_no_z_policy

        self.torch.manual_seed(_POLICY_INITIALIZATION_SEED)
        self.torch.cuda.manual_seed(_POLICY_INITIALIZATION_SEED)
        policy, freeze_record = build_grasp_lift_no_z_policy(
            prepared.config,
            device=arguments.device,
        )
        policy.eval()
        return policy, _policy_record(policy, freeze_record)

    def rollout(
        self,
        prepared: Any,
        environment: Any,
        policy: Any,
        budget: Any,
    ) -> Any:
        from rexpolicy.stage0.evaluation.grasp_lift_rollout import (
            rollout_grasp_lift_no_z_policy,
        )

        return rollout_grasp_lift_no_z_policy(
            environment.environment,
            policy,
            prepared.normalization,
            budget,
            oracle=environment.oracle,
        )

    def close_environment(self, environment: Any) -> None:
        self.selection_backend.close_environment(environment)


def _failure_record(error: BaseException) -> dict[str, Any]:
    return _sealed(
        {
            "complete": False,
            "error": str(error)[:1000],
            "error_type": type(error).__name__,
            "locked_test": dict(_LOCKED_TEST_ABSENT),
            "retry_policy": "same output path is write-once and cannot be reused",
            "schema_id": _SCHEMA_ID,
            "state": "error",
            "train_only": True,
        }
    )


def _run_smoke(arguments: SmokeArguments, backend: Any) -> dict[str, Any]:
    if arguments.output.is_symlink() or arguments.output.exists():
        raise FileExistsError(f"smoke output already exists: {arguments.output}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.output.parent.is_symlink() or not arguments.output.parent.is_dir():
        raise ValueError("smoke output parent must be a real directory")
    environment = None
    try:
        prepared = backend.prepare(arguments)
        budget, reset_groups = backend.build_budget(prepared)
        budget_record = budget.to_record()
        environment = backend.build_environment(arguments, prepared, budget)
        environment_record = dict(environment.record)
        policy, policy_record = backend.build_policy(arguments, prepared)
        rollout = backend.rollout(prepared, environment, policy, budget)
        rollout_record = rollout.to_record()
        gate = _gate_rollout_record(
            rollout_record,
            budget_record=budget_record,
        )
        backend.close_environment(environment)
        environment = None
        result = _sealed(
            {
                "complete": True,
                "config_sha256": prepared.config.sha256,
                "data_scope": {
                    "claim_created": False,
                    "held_out_shards_opened": 0,
                    "source": "accepted_pilot_train_only",
                    "train_reset_count": len(reset_groups),
                    "train_content_sha256": prepared.artifact.train_content_sha256,
                    "training_artifact_sha256": prepared.artifact.artifact_sha256,
                    "training_membership_sha256": (
                        prepared.artifact.data.corpus.training_membership_sha256
                    ),
                },
                "environment": environment_record,
                "evaluator_sha256": prepared.evaluator_sha256,
                "gate": gate,
                "implementation_sha256": prepared.implementation_sha256,
                "locked_test": dict(_LOCKED_TEST_ABSENT),
                "normalization_sha256": prepared.normalization.sha256,
                "policy": dict(policy_record),
                "reset_groups": [
                    {"reset_group_id": group, "reset_seed": seed}
                    for group, seed in reset_groups
                ],
                "retry_policy": "same output path is write-once and cannot be reused",
                "rollout": rollout_record,
                "rollout_sha256": rollout.sha256,
                "runtime": dict(prepared.runtime_record),
                "schema_id": _SCHEMA_ID,
                "state": "passed" if gate["passed"] else "gate_failed",
                "train_only": True,
                "train_content_sha256": prepared.artifact.train_content_sha256,
                "training_artifact_sha256": prepared.artifact.artifact_sha256,
            }
        )
        if environment_record.get("rollout_budget_sha256") != budget.sha256:
            raise RuntimeError("smoke environment and rollout budget differ")
        _write_json_exclusive(arguments.output, result)
        persisted = _strict_json(arguments.output)
        if persisted != result:
            raise RuntimeError("write-once smoke output changed on disk")
        _verify_sealed(persisted, "evaluator smoke output")
        return persisted
    except BaseException as error:
        if environment is not None:
            try:
                backend.close_environment(environment)
            except Exception as close_error:  # noqa: BLE001 - preserve root failure
                print(
                    "failed to close evaluator smoke environment: "
                    f"{type(close_error).__name__}: {close_error}",
                    file=sys.stderr,
                )
        if not arguments.output.exists() and not arguments.output.is_symlink():
            _write_json_exclusive(arguments.output, _failure_record(error))
        raise


def _absolute(value: str) -> Path:
    return Path(os.path.abspath(Path(value).expanduser()))


def _parse_args() -> SmokeArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/stage0/grasp_lift_no_z_bc.json",
    )
    parser.add_argument("--pilot-directory", required=True)
    parser.add_argument("--training-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parsed = parser.parse_args()
    return SmokeArguments(
        repository_root=Path(__file__).resolve().parents[1],
        config=_absolute(parsed.config),
        pilot_directory=_absolute(parsed.pilot_directory),
        training_artifact=_absolute(parsed.training_artifact),
        output=_absolute(parsed.output),
        device=parsed.device,
    )


def main() -> None:
    arguments = _parse_args()
    affinity = _restrict_preimport_runtime()

    import torch

    cuda_runtime = _configure_torch(torch, arguments.device)
    backend = DefaultSmokeBackend(
        torch=torch,
        affinity=affinity,
        cuda_runtime=cuda_runtime,
    )
    result = _run_smoke(arguments, backend)
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    if result["gate"]["passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
