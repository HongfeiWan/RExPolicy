"""Contract tests for the cohort-only Grasp-Lift validation authoring path."""

from __future__ import annotations

import ast
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from rexpolicy.stage0.data.grasp_lift_validation_cohort import (
    GRASP_LIFT_VALIDATION_AUTHORING_CLAIM_REGISTRY,
    GRASP_LIFT_VALIDATION_AUTHORING_COMMIT_SCHEMA_ID,
    GRASP_LIFT_VALIDATION_AUTHORING_MANIFEST_SCHEMA_ID,
    GRASP_LIFT_VALIDATION_AUTHORING_STATUS_SCHEMA_ID,
    GRASP_LIFT_VALIDATION_LOCKED_TEST,
    GRASP_LIFT_VALIDATION_RESET_SEEDS,
    derive_grasp_lift_validation_seeds,
    grasp_lift_validation_authoring_claim_record,
    grasp_lift_validation_cohort_slot_id,
    grasp_lift_validation_environment_config_record,
    grasp_lift_validation_preregistration_record,
    load_grasp_lift_validation_authoring_config,
    load_grasp_lift_validation_authoring_manifest,
    load_grasp_lift_validation_authoring_metadata,
    persist_grasp_lift_validation_authoring_claim,
    validation_authoring_file_sha256,
    verify_persisted_grasp_lift_validation_authoring_claim,
)
from rexpolicy.stage0.grasp_lift_pilot import (
    grasp_lift_pilot_oracle_record,
    grasp_lift_pilot_task_metadata_record,
    seal_grasp_lift_pilot_record,
)
from rexpolicy.stage0.types import canonical_fingerprint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage0/grasp_lift_validation_cohort_v2.json"
RUNNER = ROOT / "tools/run_stage0_grasp_lift_validation_authoring.py"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class GraspLiftValidationAuthoringConfigTests(unittest.TestCase):
    def test_namespace_derivation_and_config_are_exact(self) -> None:
        config = load_grasp_lift_validation_authoring_config(CONFIG)
        self.assertEqual(
            derive_grasp_lift_validation_seeds(), GRASP_LIFT_VALIDATION_RESET_SEEDS
        )
        self.assertEqual(config.seeds, GRASP_LIFT_VALIDATION_RESET_SEEDS)
        self.assertEqual(
            config.sha256,
            "9530b58ea9921982ce0542885b35769dd48319af9b3e960454ffb91d7e7aadac",
        )
        self.assertEqual(
            grasp_lift_validation_cohort_slot_id(config),
            "e61cc978674aa6597e06dc8d551f7c172148a1a4e6ff549c38aec5e842b67821",
        )
        environment = grasp_lift_validation_environment_config_record(config)
        self.assertEqual(environment["num_envs"], 6)
        self.assertEqual(environment["reward_mode"], "none")
        self.assertFalse(environment["render_images"])
        self.assertFalse(environment["camera_textures"])
        self.assertFalse(environment["load_scene_visuals"])

    def test_mutated_seed_or_acceptance_is_rejected(self) -> None:
        record = json.loads(CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            record["cohort"]["seeds"][0] += 1
            _write_json(path, record)
            with self.assertRaisesRegex(ValueError, "preregistered derivation"):
                load_grasp_lift_validation_authoring_config(path)
            record = json.loads(CONFIG.read_text(encoding="utf-8"))
            record["acceptance"]["require_all_success"] = False
            _write_json(path, record)
            with self.assertRaisesRegex(ValueError, "mandatory"):
                load_grasp_lift_validation_authoring_config(path)

    def test_runner_has_no_learned_model_or_checkpoint_import(self) -> None:
        tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
        forbidden = (
            "checkpoint",
            "grasp_lift_no_z",
            "model",
            "optimizer",
            "selection",
            "training_artifact",
        )
        for module in modules:
            self.assertFalse(
                any(fragment in module for fragment in forbidden),
                f"learned-data import leaked into authoring runner: {module}",
            )
        source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("torch.load", source)
        self.assertNotIn("load_state_dict", source)


class GraspLiftValidationAuthoringMetadataTests(unittest.TestCase):
    def _write_metadata_only_artifact(self, root: Path) -> dict[str, object]:
        from rexpolicy.stage0.envs.grasp_lift_action import (
            DEFAULT_GRASP_LIFT_ACTION_SCHEMA,
        )
        from rexpolicy.stage0.envs.grasp_lift_modes import (
            DEFAULT_GRASP_LIFT_AUTHORING_MODE,
        )
        from rexpolicy.stage0.envs.grasp_lift_state_view import (
            DEFAULT_GRASP_LIFT_STATE_VIEW,
        )

        config = load_grasp_lift_validation_authoring_config(CONFIG)
        _write_json(root / "config.json", config.to_record())
        environment = grasp_lift_validation_environment_config_record(config)
        oracle = grasp_lift_pilot_oracle_record()
        task = grasp_lift_pilot_task_metadata_record()
        implementation = {"fixture": "metadata-only"}
        implementation_sha256 = canonical_fingerprint(implementation)
        simulator = {
            "action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
            "authoring_mode_sha256": DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
            "environment_config": environment,
            "implementation_sha256": implementation_sha256,
            "oracle": oracle,
            "oracle_id": oracle["oracle_id"],
            "schema_id": "rexpolicy/stage0-grasp-lift-newton-runtime/v1",
            "task_metadata": task,
        }
        members = []
        for index, seed in enumerate(config.seeds):
            reset_group_id = f"grasp-validation-v2-g{index:06d}-s{seed:010d}"
            trajectory_id = (
                f"grasp-validation-v2-g{index:06d}-thumb-index-side-close7-"
                f"h{DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256[:12]}-s{seed:010d}"
            )
            members.append(
                {
                    "evidence_descriptor": (
                        f"shards/{trajectory_id}.grasp-lift-evidence.json"
                    ),
                    "evidence_sha256": f"{index + 1:064x}",
                    "final_authoring_phase": 7,
                    "outcome": "success",
                    "reset_group_id": reset_group_id,
                    "reset_seed": seed,
                    "role": "formal_validation",
                    "safety_violation": False,
                    "split": "validation",
                    "terminal_transition_from_phase": 7,
                    "trajectory_descriptor": f"shards/{trajectory_id}.stage0.json",
                    "trajectory_id": trajectory_id,
                    "trajectory_sha256": f"{index + 7:064x}",
                }
            )
        preregistration = grasp_lift_validation_preregistration_record(config)
        counts = {"failure": 0, "success": 6, "timeout": 0}
        manifest = seal_grasp_lift_pilot_record(
            {
                "acceptance": {
                    "collision_buffer_overflow_count": 0,
                    "nonzero_reward_transition_count": 0,
                    "outcome_counts": counts,
                    "passed": True,
                    "policy": dict(config.acceptance),
                    "safety_violation_count": 0,
                },
                "action_schema": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.to_record(),
                "action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
                "authoring_mode": DEFAULT_GRASP_LIFT_AUTHORING_MODE.to_record(),
                "authoring_mode_sha256": DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
                "cohort_slot_id": grasp_lift_validation_cohort_slot_id(config),
                "composite_corpus_sha256": "a" * 64,
                "config_sha256": config.sha256,
                "cpu_affinity": list(range(16)),
                "elapsed_seconds": 1.0,
                "environment_config": environment,
                "environment_config_sha256": canonical_fingerprint(environment),
                "implementation": implementation,
                "implementation_sha256": implementation_sha256,
                "locked_test": dict(GRASP_LIFT_VALIDATION_LOCKED_TEST),
                "members": members,
                "oracle": oracle,
                "oracle_sha256": canonical_fingerprint(oracle),
                "preregistration": preregistration,
                "preregistration_sha256": canonical_fingerprint(preregistration),
                "purpose": config.purpose,
                "reset_seeds": list(config.seeds),
                "schema_id": GRASP_LIFT_VALIDATION_AUTHORING_MANIFEST_SCHEMA_ID,
                "seed_derivation": config.seed_derivation,
                "seed_namespace": config.seed_namespace,
                "simulator_record": simulator,
                "simulator_sha256": canonical_fingerprint(simulator),
                "state_view": DEFAULT_GRASP_LIFT_STATE_VIEW.to_record(),
                "state_view_sha256": DEFAULT_GRASP_LIFT_STATE_VIEW.sha256,
                "task_metadata": task,
                "task_metadata_sha256": canonical_fingerprint(task),
                "visuals": {
                    "camera_textures": False,
                    "scene_asset": None,
                    "scene_visuals": False,
                },
                "world_steps": 1,
            }
        )
        _write_json(root / "manifest.json", manifest)
        status = seal_grasp_lift_pilot_record(
            {
                "accepted": True,
                "cohort_slot_id": manifest["cohort_slot_id"],
                "collision_buffer_overflow_count": 0,
                "complete": True,
                "manifest_sha256": manifest["self_sha256"],
                "nonzero_reward_transition_count": 0,
                "outcome_counts": counts,
                "safety_violation_count": 0,
                "schema_id": GRASP_LIFT_VALIDATION_AUTHORING_STATUS_SCHEMA_ID,
            }
        )
        _write_json(root / "status.json", status)
        commit = seal_grasp_lift_pilot_record(
            {
                "config_file": "config.json",
                "config_file_sha256": validation_authoring_file_sha256(
                    root / "config.json"
                ),
                "config_sha256": config.sha256,
                "manifest_file": "manifest.json",
                "manifest_file_sha256": validation_authoring_file_sha256(
                    root / "manifest.json"
                ),
                "manifest_sha256": manifest["self_sha256"],
                "schema_id": GRASP_LIFT_VALIDATION_AUTHORING_COMMIT_SCHEMA_ID,
                "status_file": "status.json",
                "status_file_sha256": validation_authoring_file_sha256(
                    root / "status.json"
                ),
                "status_sha256": status["self_sha256"],
            }
        )
        _write_json(root / "commit.json", commit)
        return manifest

    def test_metadata_loader_never_requires_or_opens_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self._write_metadata_only_artifact(root)
            metadata = load_grasp_lift_validation_authoring_metadata(root)
            self.assertEqual(metadata["manifest"], expected)
            self.assertFalse((root / "shards").exists())
            with self.assertRaisesRegex(ValueError, "file set changed"):
                load_grasp_lift_validation_authoring_manifest(root)

    def test_persisted_pre_cuda_claim_is_required_in_the_same_git_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "repository"
            repository.mkdir()
            subprocess.run(
                ["git", "init", "--quiet"],
                cwd=repository,
                check=True,
            )
            cohort = repository / "cohort"
            cohort.mkdir()
            manifest = self._write_metadata_only_artifact(cohort)
            config = load_grasp_lift_validation_authoring_config(cohort / "config.json")
            claim = grasp_lift_validation_authoring_claim_record(
                config,
                implementation_sha256=manifest["implementation_sha256"],
            )
            persisted = persist_grasp_lift_validation_authoring_claim(
                repository,
                config,
                implementation_sha256=manifest["implementation_sha256"],
            )
            self.assertEqual(persisted, claim)
            registry = (
                repository / ".git" / GRASP_LIFT_VALIDATION_AUTHORING_CLAIM_REGISTRY
            )
            claim_path = (
                registry / f"{grasp_lift_validation_cohort_slot_id(config)}.json"
            )
            with self.assertRaisesRegex(RuntimeError, "already authored"):
                persist_grasp_lift_validation_authoring_claim(
                    repository,
                    config,
                    implementation_sha256=manifest["implementation_sha256"],
                )

            observed_sha256 = verify_persisted_grasp_lift_validation_authoring_claim(
                repository,
                cohort,
            )
            self.assertEqual(
                observed_sha256,
                validation_authoring_file_sha256(claim_path),
            )

            claim["implementation_sha256"] = "0" * 64
            _write_json(claim_path, claim)
            with self.assertRaisesRegex(ValueError, "non-canonical or changed"):
                verify_persisted_grasp_lift_validation_authoring_claim(
                    repository,
                    cohort,
                )

    def test_authoring_claim_registry_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            repository.mkdir()
            subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
            cohort = repository / "cohort"
            cohort.mkdir()
            manifest = self._write_metadata_only_artifact(cohort)
            config = load_grasp_lift_validation_authoring_config(cohort / "config.json")
            external = root / "redirected-registry"
            external.mkdir()
            registry = (
                repository / ".git" / GRASP_LIFT_VALIDATION_AUTHORING_CLAIM_REGISTRY
            )
            registry.symlink_to(external, target_is_directory=True)

            with self.assertRaisesRegex(RuntimeError, "missing or unsafe"):
                persist_grasp_lift_validation_authoring_claim(
                    repository,
                    config,
                    implementation_sha256=manifest["implementation_sha256"],
                )

            claim = grasp_lift_validation_authoring_claim_record(
                config,
                implementation_sha256=manifest["implementation_sha256"],
            )
            claim_path = external / (
                f"{grasp_lift_validation_cohort_slot_id(config)}.json"
            )
            _write_json(claim_path, claim)
            with self.assertRaisesRegex(RuntimeError, "missing or unsafe"):
                verify_persisted_grasp_lift_validation_authoring_claim(
                    repository,
                    cohort,
                )


if __name__ == "__main__":
    unittest.main()
