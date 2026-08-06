"""Tests for the dependency-free Stage 0 config and seed contracts."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from rexpolicy.stage0 import (
    MAX_STAGE0_SEED,
    STAGE0_ACTION_DIM,
    STAGE0_STATE_DIM,
    Stage0Config,
    Stage0ExperimentConfig,
    Stage0SeedSequence,
    Stage0SystemPath,
    Stage0Task,
    derive_stage0_seed,
)


class Stage0ConfigTest(unittest.TestCase):
    def test_default_is_disabled_state_only_and_round_trips(self) -> None:
        config = Stage0ExperimentConfig()
        self.assertFalse(config.enabled)
        self.assertIs(config.system_path, Stage0SystemPath.STAGE0_STATE)
        self.assertIs(config.task, Stage0Task.REACH)
        self.assertEqual(config.state_dim, STAGE0_STATE_DIM)
        self.assertEqual(config.action_dim, STAGE0_ACTION_DIM)
        self.assertIs(Stage0Config, Stage0ExperimentConfig)
        self.assertEqual(
            Stage0ExperimentConfig.from_mapping(config.to_record()),
            config,
        )
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            config.require_enabled_stage0()
        Stage0ExperimentConfig(enabled=True).require_enabled_stage0()

    def test_config_is_strict_and_forbids_cross_path_dependencies(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown Stage 0"):
            Stage0ExperimentConfig.from_mapping({"enabledd": True})
        with self.assertRaisesRegex(ValueError, "object"):
            Stage0ExperimentConfig.from_mapping([])  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "system_path"):
            Stage0ExperimentConfig(system_path="state")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "stage0_state"):
            Stage0ExperimentConfig(
                enabled=True,
                system_path=Stage0SystemPath.FULL_GROOT,
            )
        with self.assertRaisesRegex(ValueError, "image"):
            Stage0ExperimentConfig(render_images=True)
        with self.assertRaisesRegex(ValueError, "GR00T"):
            Stage0ExperimentConfig(use_groot=True)
        with self.assertRaisesRegex(ValueError, "seed"):
            Stage0ExperimentConfig(seed=True)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "state_dim"):
            Stage0ExperimentConfig(state_dim=78)
        with self.assertRaisesRegex(ValueError, "action_dim"):
            Stage0ExperimentConfig(action_dim=18)
        with self.assertRaisesRegex(ValueError, "divisible"):
            Stage0ExperimentConfig(model_dim=130, attention_heads=8)

    def test_action_horizon_cannot_exceed_future_horizon(self) -> None:
        Stage0ExperimentConfig(action_horizon=8, future_horizon=8)
        with self.assertRaisesRegex(ValueError, "action_horizon"):
            Stage0ExperimentConfig(action_horizon=9, future_horizon=8)

    def test_fingerprint_is_deterministic_and_semantic(self) -> None:
        config = Stage0ExperimentConfig(enabled=True, num_envs=64, seed=17)
        restored = Stage0ExperimentConfig.from_mapping(
            dict(reversed(list(config.to_record().items())))
        )
        self.assertEqual(restored.fingerprint, config.fingerprint)
        self.assertEqual(len(config.fingerprint), 64)
        self.assertNotEqual(
            Stage0ExperimentConfig(enabled=True, num_envs=65, seed=17).fingerprint,
            config.fingerprint,
        )

    def test_seed_hierarchy_is_stateless_and_partitioned(self) -> None:
        seeds = Stage0SeedSequence(123, namespace="reach-rollout")
        self.assertEqual(seeds.world(2, 7), seeds.world(2, 7))
        self.assertNotEqual(seeds.world(2, 7), seeds.world(2, 8))
        self.assertNotEqual(seeds.world(2, 7), seeds.episode(2, 7, 0))
        self.assertGreaterEqual(seeds.world(2, 7), 0)
        self.assertLessEqual(seeds.world(2, 7), MAX_STAGE0_SEED)
        self.assertEqual(
            derive_stage0_seed(123, "rank", 2, "world", 7, namespace="reach-rollout"),
            seeds.world(2, 7),
        )
        self.assertEqual(
            Stage0SeedSequence(123).fingerprint, Stage0SeedSequence(123).fingerprint
        )
        with self.assertRaisesRegex(ValueError, "base_seed"):
            Stage0SeedSequence(True)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "coordinate"):
            seeds.derive(-1)

    def test_package_root_does_not_import_heavy_dependencies(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        command = (
            "import sys; "
            f"sys.path.insert(0, {str(repository)!r}); "
            "import rexpolicy.stage0; "
            "forbidden={'torch','gr00t','transformers','diffusers'}; "
            "loaded={name.split('.')[0] for name in sys.modules}; "
            "assert not forbidden.intersection(loaded), "
            "sorted(forbidden.intersection(loaded))"
        )
        subprocess.run(
            [sys.executable, "-S", "-c", command],
            check=True,
            cwd=repository,
        )


if __name__ == "__main__":
    unittest.main()
