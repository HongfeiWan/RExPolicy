"""CPU-only tests for the Grasp-Lift no-z runner's sealed bookkeeping."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rexpolicy.stage0.long_run import atomic_write_json
from tools.run_stage0_grasp_lift_no_z import (
    _begin_heartbeat_attempt,
    _checkpoint_extra,
    _distributed_require_identical_contract,
    _distributed_setup_directory,
    _rank_state,
    _read_json,
)


class _Context:
    def __init__(
        self,
        *,
        is_main: bool = True,
        process_seed: int = 909,
        broadcast_result: str | None = None,
        rank: int = 0,
        world_size: int = 1,
        peer_sha256s: tuple[str, ...] | None = None,
    ) -> None:
        self.is_main = is_main
        self.process_seed = process_seed
        self.broadcast_result = broadcast_result
        self.rank = rank
        self.world_size = world_size
        self.peer_sha256s = peer_sha256s
        self.broadcast_inputs: list[str | None] = []
        self.barriers = 0

    def broadcast_object(
        self,
        value: str | None,
        *,
        source: int = 0,
    ) -> str | None:
        self.broadcast_inputs.append(value)
        if self.peer_sha256s is not None:
            return self.peer_sha256s[source]
        if self.is_main:
            return value
        return self.broadcast_result

    def barrier(self) -> None:
        self.barriers += 1


class _Cursor:
    def __init__(self, optimizer_step: int) -> None:
        self.optimizer_step = optimizer_step

    def to_record(self) -> dict[str, int | str]:
        total_samples = self.optimizer_step * 2048
        return {
            "epoch": total_samples // 1030,
            "offset": total_samples % 1030,
            "optimizer_step": self.optimizer_step,
            "schema_id": "rexpolicy/stage0-grasp-lift-no-z-sampler-cursor/v1",
            "total_samples": total_samples,
        }


class _Sampler:
    def cursor(self, optimizer_step: int) -> _Cursor:
        return _Cursor(optimizer_step)


def _contract() -> dict[str, Any]:
    return {
        "schema_id": "rexpolicy/test-no-z-run-contract/v1",
        "self_sha256": "a" * 64,
        "training_seed": 31001,
        "validation_policy": "not opened during fixed-budget training",
    }


class GraspLiftNoZRunnerTest(unittest.TestCase):
    def test_heartbeat_resume_is_an_explicit_new_attempt(self) -> None:
        from rexpolicy.stage0.long_run import AtomicJsonlLog

        with tempfile.TemporaryDirectory() as temporary:
            log = AtomicJsonlLog(Path(temporary) / "heartbeat.jsonl")
            self.assertEqual(
                _begin_heartbeat_attempt(
                    log,
                    resume=False,
                    global_step=0,
                    training_seed=31001,
                    world_size=1,
                ),
                0,
            )
            log.append(
                {
                    "attempt_index": 0,
                    "event": "training",
                    "global_step": 700,
                    "schema_id": (
                        "rexpolicy/stage0-grasp-lift-no-z-heartbeat/v2"
                    ),
                }
            )
            self.assertEqual(
                _begin_heartbeat_attempt(
                    log,
                    resume=True,
                    global_step=500,
                    training_seed=31001,
                    world_size=1,
                ),
                1,
            )
            records = log.records()
            self.assertEqual(records[-1]["event"], "resume")
            self.assertEqual(records[-1]["attempt_index"], 1)
            self.assertEqual(records[-1]["global_step"], 500)
            self.assertEqual(records[-1]["prior_observed_global_step"], 700)
            self.assertEqual(records[-1]["resume_from_committed_step"], 500)

    def test_distributed_contract_requires_exact_consensus(self) -> None:
        contract = _contract()
        accepted = _Context(
            world_size=2,
            peer_sha256s=("a" * 64, "a" * 64),
        )
        _distributed_require_identical_contract(contract, accepted)
        self.assertEqual(accepted.barriers, 1)

        rejected = _Context(
            world_size=2,
            peer_sha256s=("a" * 64, "b" * 64),
        )
        with self.assertRaisesRegex(RuntimeError, "differ between ranks"):
            _distributed_require_identical_contract(contract, rejected)
        self.assertEqual(rejected.barriers, 0)

        malformed = dict(contract)
        malformed["self_sha256"] = "A" * 64
        with self.assertRaisesRegex(ValueError, "not sealed"):
            _distributed_require_identical_contract(malformed, _Context())

    def test_fresh_setup_writes_exact_contract_and_initialized_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "run"
            context = _Context()
            contract = _contract()

            _distributed_setup_directory(
                output,
                resume=False,
                contract=contract,
                context=context,
                atomic_write_json=atomic_write_json,
            )

            self.assertEqual(_read_json(output / "run-contract.json"), contract)
            self.assertEqual(
                _read_json(output / "status.json"),
                {
                    "complete": False,
                    "global_step": 0,
                    "schema_id": ("rexpolicy/stage0-grasp-lift-no-z-status/v1"),
                    "state": "initialized",
                },
            )
            self.assertEqual(context.broadcast_inputs, [None])
            self.assertEqual(context.barriers, 1)

    def test_fresh_setup_never_reuses_or_overwrites_an_existing_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            output.mkdir()
            sentinel = output / "owner-data.txt"
            sentinel.write_text("preserve", encoding="utf-8")

            with self.assertRaisesRegex(
                RuntimeError,
                "no-z output setup failed: FileExistsError",
            ):
                _distributed_setup_directory(
                    output,
                    resume=False,
                    contract=_contract(),
                    context=_Context(),
                    atomic_write_json=atomic_write_json,
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
            self.assertFalse((output / "run-contract.json").exists())
            self.assertFalse((output / "status.json").exists())

    def test_resume_requires_the_exact_existing_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            contract = _contract()
            _distributed_setup_directory(
                output,
                resume=False,
                contract=contract,
                context=_Context(),
                atomic_write_json=atomic_write_json,
            )
            initialized = (output / "status.json").read_bytes()

            resumed = _Context()
            _distributed_setup_directory(
                output,
                resume=True,
                contract=dict(contract),
                context=resumed,
                atomic_write_json=atomic_write_json,
            )
            self.assertEqual((output / "status.json").read_bytes(), initialized)
            self.assertEqual(resumed.broadcast_inputs, [None])
            self.assertEqual(resumed.barriers, 1)

            changed = dict(contract)
            changed["training_seed"] = 31002
            failed = _Context()
            with self.assertRaisesRegex(
                RuntimeError,
                "no-z output setup failed: ValueError: resume run contract changed",
            ):
                _distributed_setup_directory(
                    output,
                    resume=True,
                    contract=changed,
                    context=failed,
                    atomic_write_json=atomic_write_json,
                )
            self.assertEqual(failed.barriers, 0)
            self.assertEqual(_read_json(output / "run-contract.json"), contract)
            self.assertEqual((output / "status.json").read_bytes(), initialized)

    def test_missing_resume_and_non_main_broadcast_error_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "missing"
            main = _Context()
            with self.assertRaisesRegex(
                RuntimeError,
                "resume output is missing or a symlink",
            ):
                _distributed_setup_directory(
                    output,
                    resume=True,
                    contract=_contract(),
                    context=main,
                    atomic_write_json=atomic_write_json,
                )
            self.assertEqual(main.barriers, 0)

            worker = _Context(
                is_main=False,
                broadcast_result="ValueError: rank-zero rejected output",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "rank-zero rejected output",
            ):
                _distributed_setup_directory(
                    output,
                    resume=False,
                    contract=_contract(),
                    context=worker,
                    atomic_write_json=atomic_write_json,
                )
            self.assertFalse(output.exists())
            self.assertEqual(worker.broadcast_inputs, [None])
            self.assertEqual(worker.barriers, 0)

    def test_output_and_run_contract_symlinks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_output = root / "real-run"
            real_output.mkdir()
            output_link = root / "linked-run"
            output_link.symlink_to(real_output, target_is_directory=True)

            for resume in (False, True):
                with (
                    self.subTest(resume=resume),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "symlink|already exists",
                    ),
                ):
                    _distributed_setup_directory(
                        output_link,
                        resume=resume,
                        contract=_contract(),
                        context=_Context(),
                        atomic_write_json=atomic_write_json,
                    )

            contract_target = root / "contract-target.json"
            contract_target.write_text(json.dumps(_contract()), encoding="utf-8")
            (real_output / "run-contract.json").symlink_to(contract_target)
            with self.assertRaisesRegex(
                RuntimeError,
                "required run record is missing or a symlink",
            ):
                _distributed_setup_directory(
                    real_output,
                    resume=True,
                    contract=_contract(),
                    context=_Context(),
                    atomic_write_json=atomic_write_json,
                )

    def test_read_json_rejects_duplicate_keys_and_non_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            path.write_text('{"seed": 31001, "seed": 31002}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key: seed"):
                _read_json(path)
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(TypeError, "must contain an object"):
                _read_json(path)

    def test_rank_state_records_the_exact_process_seed_and_sampler_cursor(self) -> None:
        context = _Context(process_seed=982451653)
        sampler = _Sampler()

        self.assertEqual(
            _rank_state(
                context,
                sampler,
                7,
                cpu_affinity=tuple(range(16)),
            ),
            {
                "cpu_affinity": list(range(16)),
                "process_seed": 982451653,
                "sampler": {
                    "epoch": 13,
                    "offset": 946,
                    "optimizer_step": 7,
                    "schema_id": ("rexpolicy/stage0-grasp-lift-no-z-sampler-cursor/v1"),
                    "total_samples": 14336,
                },
                "schema_id": "rexpolicy/stage0-grasp-lift-no-z-rank/v1",
            },
        )

    def test_checkpoint_extra_is_training_only_and_binds_all_hashes(self) -> None:
        artifact = SimpleNamespace(
            audit_dataset_sha256="b" * 64,
            train_content_sha256="c" * 64,
        )
        self.assertEqual(
            _checkpoint_extra(
                artifact=artifact,
                contract={"self_sha256": "d" * 64},
                implementation_sha256="e" * 64,
                normalization_sha256="f" * 64,
            ),
            {
                "audit_dataset_sha256": "b" * 64,
                "implementation_sha256": "e" * 64,
                "locked_test": "not_created",
                "normalization_sha256": "f" * 64,
                "run_contract_sha256": "d" * 64,
                "schema_id": ("rexpolicy/stage0-grasp-lift-no-z-checkpoint-extra/v1"),
                "train_content_sha256": "c" * 64,
                "training_only": True,
                "validation_consumed": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
