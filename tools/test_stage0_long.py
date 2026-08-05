"""CPU-only tests for Stage 0 long-run contracts and recovery primitives."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rexpolicy.stage0.long_run import (
    AtomicJsonlLog,
    LongRunCursor,
    LongRunPhase,
    LongRunTrainingConfig,
    Stage0LongRunConfig,
    WallClockBudget,
    atomic_write_json,
)
from tools.run_stage0_long import (
    _MANIFOLD_TRAINER_CONFIG,
    _phase_training_windows,
    _sample_contrastive_windows,
)


def _record() -> dict[str, object]:
    return {
        "corpus": {
            "minimum_success_modes_per_reset": 2,
            "mode_ids": [
                "stage0/reach/direct/v1",
                "stage0/reach/left_arc/v1",
            ],
            "reset_groups": 4,
        },
        "runtime": {
            "cpu_threads": 2,
            "eval_window_count": 4,
            "flow_sample_steps": 2,
            "max_wall_seconds": 60.0,
            "seed": 17,
        },
        "schema_version": 1,
        "stage0": {
            "action_dim": 19,
            "action_horizon": 2,
            "attention_heads": 2,
            "dropout": 0.0,
            "enabled": True,
            "feedforward_dim": 16,
            "future_horizon": 4,
            "latent_dim": 4,
            "max_episode_steps": 8,
            "model_dim": 8,
            "num_envs": 2,
            "render_images": False,
            "schema_version": 1,
            "seed": 17,
            "state_dim": 79,
            "state_schema_id": "stage0/newton-state/v1",
            "system_path": "stage0_state",
            "task": "reach",
            "transformer_layers": 1,
            "use_groot": False,
            "window_stride": 1,
        },
        "training": {
            "batch_size": 8,
            "checkpoint_every_steps": 2,
            "conditional_policy_steps": 2,
            "eval_every_steps": 2,
            "future_encoder_steps": 2,
            "learning_rate": 0.001,
            "log_every_steps": 1,
            "no_z_policy_steps": 2,
            "selector_steps": 1,
            "weight_decay": 0.0,
        },
    }


class Stage0LongRunConfigTest(unittest.TestCase):
    def test_config_is_exact_fingerprinted_and_equal_budget(self) -> None:
        first = Stage0LongRunConfig.from_mapping(_record())
        second = Stage0LongRunConfig.from_mapping(
            json.loads(json.dumps(_record(), sort_keys=True))
        )
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(len(first.fingerprint), 64)
        self.assertEqual(first.corpus.reset_groups, 4)
        self.assertEqual(
            first.training.conditional_policy_steps,
            first.training.no_z_policy_steps,
        )

        unequal = _record()
        unequal["training"]["no_z_policy_steps"] = 3  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "equal-budget"):
            Stage0LongRunConfig.from_mapping(unequal)

        unknown = _record()
        unknown["surprise"] = True
        with self.assertRaisesRegex(ValueError, "unexpected"):
            Stage0LongRunConfig.from_mapping(unknown)

    def test_reset_groups_must_tile_the_fixed_vector_environment(self) -> None:
        record = _record()
        record["corpus"]["reset_groups"] = 5  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "divisible"):
            Stage0LongRunConfig.from_mapping(record)

    def test_manifold_batch_is_composed_of_complete_octets(self) -> None:
        record = _record()
        record["training"]["batch_size"] = 4  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "multiple of eight"):
            Stage0LongRunConfig.from_mapping(record)

    def test_validate_only_cli_does_not_need_torch_or_newton(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repository)
        for config in (
            "configs/stage0/reach_tiny.json",
            "configs/stage0/reach_pilot.json",
            "configs/stage0/reach_long.json",
        ):
            with self.subTest(config=config):
                result = subprocess.run(
                    [
                        sys.executable,
                        "tools/run_stage0_long.py",
                        "--validate-config-only",
                        "--config",
                        config,
                    ],
                    cwd=repository,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                payload = json.loads(result.stdout)
                self.assertEqual(len(payload["config_sha256"]), 64)
        audit = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, runpy, sys; "
                    "sys.argv=['tools/run_stage0_long.py', "
                    "'--validate-config-only', '--config', "
                    "'configs/stage0/reach_tiny.json']; "
                    "runpy.run_path(sys.argv[0], run_name='__main__'); "
                    "print(json.dumps({'torch': 'torch' in sys.modules, "
                    "'newton': 'newton' in sys.modules}))"
                ),
            ],
            cwd=repository,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        imported = json.loads(audit.stdout.splitlines()[-1])
        self.assertEqual(imported, {"newton": False, "torch": False})


class Stage0LongRunCursorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.training = LongRunTrainingConfig.from_mapping(_record()["training"])

    def test_cursor_recovers_exact_phase_and_rejects_inconsistent_global(self) -> None:
        cursor = LongRunCursor()
        expected = (
            LongRunPhase.FUTURE_ENCODER,
            LongRunPhase.FUTURE_ENCODER,
            LongRunPhase.SELECTOR,
            LongRunPhase.CONDITIONAL_POLICY,
            LongRunPhase.CONDITIONAL_POLICY,
            LongRunPhase.NO_Z_POLICY,
            LongRunPhase.NO_Z_POLICY,
        )
        for phase in expected:
            self.assertIs(cursor.next_phase(self.training), phase)
            cursor = cursor.advance(self.training, phase)
            recovered = LongRunCursor.from_steps(cursor.to_steps())
            self.assertEqual(recovered, cursor)
        self.assertIs(cursor.next_phase(self.training), LongRunPhase.COMPLETE)
        self.assertEqual(cursor.global_step, 7)

        corrupt = cursor.to_steps()
        corrupt["global"] += 1
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            LongRunCursor.from_steps(corrupt)

    def test_cursor_cannot_skip_or_exceed_a_phase(self) -> None:
        cursor = LongRunCursor()
        with self.assertRaisesRegex(ValueError, "next phase"):
            cursor.advance(self.training, LongRunPhase.SELECTOR)
        exceeded = LongRunCursor(future_encoder=3)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            exceeded.next_phase(self.training)


class Stage0LongRunSamplingTest(unittest.TestCase):
    def test_checkpointed_manifold_config_matches_runtime_config(self) -> None:
        from dataclasses import asdict

        from rexpolicy.stage0.trainers import Stage0ManifoldTrainerConfig

        runtime_config = Stage0ManifoldTrainerConfig(**_MANIFOLD_TRAINER_CONFIG)
        self.assertEqual(asdict(runtime_config), _MANIFOLD_TRAINER_CONFIG)

    def test_manifold_octet_pairs_modes_across_resets(self) -> None:
        import torch

        modes = ("direct", "left_arc")
        windows = []
        trajectory_modes = {}
        for reset_group_id in ("reset-0", "reset-1"):
            for mode_id in modes:
                trajectory_id = f"{reset_group_id}/{mode_id}"
                trajectory_modes[trajectory_id] = {"mode_id": mode_id}
                for start in (0, 1):
                    windows.append(
                        SimpleNamespace(
                            reset_group_id=reset_group_id,
                            start=start,
                            trajectory_id=trajectory_id,
                        )
                    )
        sampled = _sample_contrastive_windows(
            windows,
            trajectory_modes,
            batch_size=8,
            generator=torch.Generator().manual_seed(7),
        )
        self.assertEqual(len(sampled), 8)
        for offset in (0, 4):
            rows = sampled[offset : offset + 4]
            self.assertEqual({row.start for row in rows}, {0, 1})
            self.assertEqual(len({row.reset_group_id for row in rows}), 2)
            self.assertEqual(
                len({trajectory_modes[row.trajectory_id]["mode_id"] for row in rows}),
                1,
            )
        self.assertEqual(
            {trajectory_modes[row.trajectory_id]["mode_id"] for row in sampled},
            set(modes),
        )

    def test_selector_only_samples_episode_starts(self) -> None:
        windows = tuple(SimpleNamespace(start=start) for start in (0, 1, 0, 2))
        selected = _phase_training_windows(windows, LongRunPhase.SELECTOR)
        self.assertEqual(tuple(window.start for window in selected), (0, 0))
        self.assertIs(
            _phase_training_windows(windows, LongRunPhase.CONDITIONAL_POLICY),
            windows,
        )

    def test_sparse_mode_groups_only_use_compatible_reset_pairs(self) -> None:
        import torch

        group_modes = {
            "reset-0": ("direct", "left"),
            "reset-1": ("left", "right"),
            "reset-2": ("direct", "left"),
        }
        windows = []
        trajectory_modes = {}
        for reset_group_id, modes in group_modes.items():
            for mode_id in modes:
                trajectory_id = f"{reset_group_id}/{mode_id}"
                trajectory_modes[trajectory_id] = {"mode_id": mode_id}
                windows.extend(
                    SimpleNamespace(
                        reset_group_id=reset_group_id,
                        start=start,
                        trajectory_id=trajectory_id,
                    )
                    for start in (0, 1)
                )
        sampled = _sample_contrastive_windows(
            windows,
            trajectory_modes,
            batch_size=16,
            generator=torch.Generator().manual_seed(11),
        )
        self.assertEqual(
            {window.reset_group_id for window in sampled},
            {"reset-0", "reset-2"},
        )


class Stage0LongRunDurabilityTest(unittest.TestCase):
    def test_wall_budget_uses_monotonic_injected_clock(self) -> None:
        now = [100.0]
        budget = WallClockBudget(5.0, clock=lambda: now[0])
        self.assertFalse(budget.expired)
        self.assertEqual(budget.remaining_seconds, 5.0)
        now[0] = 104.5
        self.assertAlmostEqual(budget.elapsed_seconds, 4.5)
        self.assertFalse(budget.expired)
        now[0] = 105.0
        self.assertTrue(budget.expired)
        self.assertEqual(budget.remaining_seconds, 0.0)

    def test_jsonl_append_and_status_replace_leave_no_partial_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            heartbeat = AtomicJsonlLog(root / "heartbeat.jsonl")
            heartbeat.append({"event": "started", "step": 0})
            heartbeat.append({"event": "training", "step": 1})
            self.assertEqual(
                heartbeat.records(),
                (
                    {"event": "started", "step": 0},
                    {"event": "training", "step": 1},
                ),
            )
            status = root / "status.json"
            atomic_write_json(status, {"step": 1})
            atomic_write_json(status, {"step": 2})
            self.assertEqual(json.loads(status.read_text()), {"step": 2})
            self.assertFalse(any(root.glob(".*.tmp-*")))

    def test_torn_log_fails_closed_instead_of_guessing_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "heartbeat.jsonl"
            path.write_bytes(b'{"event":"started"}\n{"event":')
            heartbeat = AtomicJsonlLog(path)
            with self.assertRaisesRegex(RuntimeError, "torn"):
                heartbeat.records()
            with self.assertRaisesRegex(RuntimeError, "torn"):
                heartbeat.append({"event": "training"})


if __name__ == "__main__":
    unittest.main()
