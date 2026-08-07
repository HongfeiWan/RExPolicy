from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rexpolicy.stage0.data.grasp_lift_training import GRASP_LIFT_VALIDATION_ROLE
from rexpolicy.stage0.data.grasp_lift_validation_claim import (
    GRASP_LIFT_VALIDATION_CLAIM_REGISTRY,
    GraspLiftValidationClaimReceipt,
)
from rexpolicy.stage0.data.grasp_lift_validation_cohort import (
    GRASP_LIFT_VALIDATION_AUTHORING_COMMIT_SCHEMA_ID,
    GRASP_LIFT_VALIDATION_AUTHORING_CONFIG_SCHEMA_ID,
    GRASP_LIFT_VALIDATION_AUTHORING_MANIFEST_SCHEMA_ID,
    GRASP_LIFT_VALIDATION_AUTHORING_STATUS_SCHEMA_ID,
    GRASP_LIFT_VALIDATION_COHORT_SLOT_SCHEMA_ID,
    grasp_lift_validation_environment_config_record,
    grasp_lift_validation_preregistration_record,
    load_grasp_lift_validation_authoring_config,
)
from rexpolicy.stage0.data.grasp_lift_revalidation import (
    GRASP_LIFT_REVALIDATION_PURPOSE,
    GRASP_LIFT_REVALIDATION_RESET_SEEDS,
    GRASP_LIFT_REVALIDATION_SEED_NAMESPACE,
    build_grasp_lift_revalidation_claim_record,
    grasp_lift_revalidation_claim_id,
    load_claimed_grasp_lift_revalidation,
    load_grasp_lift_revalidation_metadata,
    verify_grasp_lift_revalidation_claim_record,
)
from rexpolicy.stage0.envs.grasp_lift_action import DEFAULT_GRASP_LIFT_ACTION_SCHEMA
from rexpolicy.stage0.envs.grasp_lift_modes import DEFAULT_GRASP_LIFT_AUTHORING_MODE
from rexpolicy.stage0.envs.grasp_lift_oracle import GRASP_LIFT_ORACLE_ID
from rexpolicy.stage0.envs.grasp_lift_state_view import DEFAULT_GRASP_LIFT_STATE_VIEW
from rexpolicy.stage0.grasp_lift_pilot import (
    grasp_lift_pilot_oracle_record,
    grasp_lift_pilot_task_metadata_record,
)
from rexpolicy.stage0.grasp_lift_pilot import seal_grasp_lift_pilot_record
from rexpolicy.stage0.types import canonical_fingerprint


_DERIVATION = (
    "int.from_bytes(sha256(f'{namespace}/{index}').digest()[:4], 'big') "
    "& 0x7fffffff; canonical_numeric_sort"
)
_ABSENT = {"artifact": None, "consumed": False, "policy": "not_created"}
_HASHES = {
    "candidate": "1" * 64,
    "preflight": "2" * 64,
    "protocol": "3" * 64,
    "evaluator": "4" * 64,
}


def _payload(record: object) -> bytes:
    return (
        json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")


def _write(path: Path, record: object) -> None:
    path.write_bytes(_payload(record))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic(name: str) -> tuple[dict[str, str], str]:
    record = {"kind": name}
    return record, canonical_fingerprint(record)


def _slot_id() -> str:
    return canonical_fingerprint(
        {
            "purpose": GRASP_LIFT_REVALIDATION_PURPOSE,
            "reset_seeds": list(GRASP_LIFT_REVALIDATION_RESET_SEEDS),
            "schema_id": GRASP_LIFT_VALIDATION_COHORT_SLOT_SCHEMA_ID,
            "seed_namespace": GRASP_LIFT_REVALIDATION_SEED_NAMESPACE,
        }
    )


def _authoring_fixture(root: Path) -> None:
    config = {
        "acceptance": {
            "require_all_success": True,
            "require_zero_collision_buffer_overflows": True,
            "require_zero_nonzero_rewards": True,
            "require_zero_oracle_failures": True,
            "require_zero_safety_violations": True,
            "require_zero_timeouts": True,
        },
        "cohort": {
            "purpose": GRASP_LIFT_REVALIDATION_PURPOSE,
            "seed_derivation": _DERIVATION,
            "seed_namespace": GRASP_LIFT_REVALIDATION_SEED_NAMESPACE,
            "seeds": list(GRASP_LIFT_REVALIDATION_RESET_SEEDS),
        },
        "runtime": {
            "cpu_threads": 16,
            "device": "cuda:0",
            "max_episode_steps": 72,
            "num_envs": 6,
            "rigid_contacts_per_env": 2048,
            "triangle_pairs_per_env": 200_000,
        },
        "schema_version": 1,
    }
    config_sha256 = canonical_fingerprint(
        {"schema_id": GRASP_LIFT_VALIDATION_AUTHORING_CONFIG_SCHEMA_ID, **config}
    )
    _write(root / "config.json", config)
    config_object = load_grasp_lift_validation_authoring_config(root / "config.json")

    members = []
    for seed in GRASP_LIFT_REVALIDATION_RESET_SEEDS:
        index = len(members)
        trajectory_id = (
            f"grasp-validation-v2-g{index:06d}-thumb-index-side-close7-"
            f"h{DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256[:12]}-s{seed:010d}"
        )
        members.append(
            {
                "evidence_descriptor": (
                    f"shards/{trajectory_id}.grasp-lift-evidence.json"
                ),
                "evidence_sha256": hashlib.sha256(
                    f"evidence:{seed}".encode("ascii")
                ).hexdigest(),
                "final_authoring_phase": 8,
                "outcome": "success",
                "reset_group_id": (f"grasp-validation-v2-g{index:06d}-s{seed:010d}"),
                "reset_seed": seed,
                "role": GRASP_LIFT_VALIDATION_ROLE,
                "safety_violation": False,
                "split": "validation",
                "terminal_transition_from_phase": 7,
                "trajectory_descriptor": f"shards/{trajectory_id}.stage0.json",
                "trajectory_id": trajectory_id,
                "trajectory_sha256": hashlib.sha256(
                    f"trajectory:{seed}".encode("ascii")
                ).hexdigest(),
            }
        )
    composite = canonical_fingerprint(
        {
            "members": sorted(
                (
                    {
                        "evidence_sha256": item["evidence_sha256"],
                        "trajectory_id": item["trajectory_id"],
                        "trajectory_sha256": item["trajectory_sha256"],
                    }
                    for item in members
                ),
                key=lambda item: item["trajectory_id"],
            ),
            "schema_id": "rexpolicy/stage0-grasp-lift-trajectory-evidence-corpus/v1",
        }
    )
    environment = grasp_lift_validation_environment_config_record(config_object)
    task = grasp_lift_pilot_task_metadata_record()
    oracle = grasp_lift_pilot_oracle_record()
    preregistration = grasp_lift_validation_preregistration_record(config_object)
    implementation, implementation_sha256 = _semantic("implementation")
    simulator = {
        "action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
        "authoring_mode_sha256": DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
        "environment_config": environment,
        "implementation_sha256": implementation_sha256,
        "oracle": oracle,
        "oracle_id": GRASP_LIFT_ORACLE_ID,
        "schema_id": "rexpolicy/stage0-grasp-lift-newton-runtime/v1",
        "task_metadata": task,
    }
    manifest = seal_grasp_lift_pilot_record(
        {
            "acceptance": {
                "collision_buffer_overflow_count": 0,
                "nonzero_reward_transition_count": 0,
                "outcome_counts": {"failure": 0, "success": 6, "timeout": 0},
                "passed": True,
                "policy": dict(config["acceptance"]),
                "safety_violation_count": 0,
            },
            "action_schema": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.to_record(),
            "action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
            "authoring_mode": DEFAULT_GRASP_LIFT_AUTHORING_MODE.to_record(),
            "authoring_mode_sha256": DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
            "cohort_slot_id": _slot_id(),
            "composite_corpus_sha256": composite,
            "config_sha256": config_sha256,
            "cpu_affinity": list(range(16)),
            "elapsed_seconds": 1.0,
            "environment_config": environment,
            "environment_config_sha256": canonical_fingerprint(environment),
            "implementation": implementation,
            "implementation_sha256": implementation_sha256,
            "locked_test": dict(_ABSENT),
            "members": members,
            "oracle": oracle,
            "oracle_sha256": canonical_fingerprint(oracle),
            "preregistration": preregistration,
            "preregistration_sha256": canonical_fingerprint(preregistration),
            "purpose": GRASP_LIFT_REVALIDATION_PURPOSE,
            "reset_seeds": list(GRASP_LIFT_REVALIDATION_RESET_SEEDS),
            "schema_id": GRASP_LIFT_VALIDATION_AUTHORING_MANIFEST_SCHEMA_ID,
            "seed_derivation": _DERIVATION,
            "seed_namespace": GRASP_LIFT_REVALIDATION_SEED_NAMESPACE,
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
            "world_steps": 432,
        }
    )
    _write(root / "manifest.json", manifest)
    status = seal_grasp_lift_pilot_record(
        {
            "accepted": True,
            "cohort_slot_id": _slot_id(),
            "collision_buffer_overflow_count": 0,
            "complete": True,
            "manifest_sha256": manifest["self_sha256"],
            "nonzero_reward_transition_count": 0,
            "outcome_counts": {"failure": 0, "success": 6, "timeout": 0},
            "safety_violation_count": 0,
            "schema_id": GRASP_LIFT_VALIDATION_AUTHORING_STATUS_SCHEMA_ID,
        }
    )
    _write(root / "status.json", status)
    commit = seal_grasp_lift_pilot_record(
        {
            "config_file": "config.json",
            "config_file_sha256": _file_sha256(root / "config.json"),
            "config_sha256": config_sha256,
            "manifest_file": "manifest.json",
            "manifest_file_sha256": _file_sha256(root / "manifest.json"),
            "manifest_sha256": manifest["self_sha256"],
            "schema_id": GRASP_LIFT_VALIDATION_AUTHORING_COMMIT_SCHEMA_ID,
            "status_file": "status.json",
            "status_file_sha256": _file_sha256(root / "status.json"),
            "status_sha256": status["self_sha256"],
        }
    )
    _write(root / "commit.json", commit)


class GraspLiftRevalidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        _authoring_fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _claim(self):
        metadata = load_grasp_lift_revalidation_metadata(self.root)
        claim = build_grasp_lift_revalidation_claim_record(
            metadata,
            candidate_inventory_sha256=_HASHES["candidate"],
            preflight_sha256=_HASHES["preflight"],
            selection_protocol_sha256=_HASHES["protocol"],
            evaluator_implementation_sha256=_HASHES["evaluator"],
        )
        return metadata, claim

    def test_seed_derivation_is_fixed_fresh_and_canonical(self) -> None:
        derived = sorted(
            int.from_bytes(
                hashlib.sha256(
                    f"{GRASP_LIFT_REVALIDATION_SEED_NAMESPACE}/{index}".encode()
                ).digest()[:4],
                "big",
            )
            & 0x7FFF_FFFF
            for index in range(6)
        )
        self.assertEqual(tuple(derived), GRASP_LIFT_REVALIDATION_RESET_SEEDS)
        self.assertTrue(
            set(GRASP_LIFT_REVALIDATION_RESET_SEEDS).isdisjoint(
                {7009, 7014, 7019, 7025, 7026, 7031}
            )
        )

    def test_metadata_preclaim_does_not_require_or_open_shards(self) -> None:
        with (
            mock.patch(
                "rexpolicy.stage0.data.grasp_lift_revalidation.load_trajectory_shard",
                side_effect=AssertionError("preclaim opened trajectory tensors"),
            ),
            mock.patch(
                "rexpolicy.stage0.data.grasp_lift_revalidation."
                "load_grasp_lift_evidence_sidecar",
                side_effect=AssertionError("preclaim opened evidence tensors"),
            ),
        ):
            metadata = load_grasp_lift_revalidation_metadata(self.root)
        self.assertFalse((self.root / "shards").exists())
        self.assertEqual(
            tuple(item.reset_seed for item in metadata.commitments),
            GRASP_LIFT_REVALIDATION_RESET_SEEDS,
        )
        self.assertEqual(
            set(metadata.cohort_identity),
            {
                "excluded_commitments",
                "formal_validation_commitments",
                "locked_test",
                "purpose",
                "role_registry_sha256",
                "schema_id",
                "self_sha256",
                "source_composite_corpus_sha256",
                "source_manifest_sha256",
            },
        )

    def test_claim_uses_existing_envelope_and_binds_new_cohort(self) -> None:
        metadata, claim = self._claim()
        self.assertEqual(claim["claim_id"], grasp_lift_revalidation_claim_id(metadata))
        self.assertEqual(
            claim["cohort_identity_sha256"], metadata.cohort_identity_sha256
        )
        verify_grasp_lift_revalidation_claim_record(
            claim,
            metadata=metadata,
            candidate_inventory_sha256=_HASHES["candidate"],
            preflight_sha256=_HASHES["preflight"],
            selection_protocol_sha256=_HASHES["protocol"],
            evaluator_implementation_sha256=_HASHES["evaluator"],
        )
        with self.assertRaisesRegex(ValueError, "binding changed"):
            verify_grasp_lift_revalidation_claim_record(
                claim,
                metadata=metadata,
                candidate_inventory_sha256="9" * 64,
                preflight_sha256=_HASHES["preflight"],
                selection_protocol_sha256=_HASHES["protocol"],
                evaluator_implementation_sha256=_HASHES["evaluator"],
            )

    def test_seed_mutation_fails_before_any_shard_open(self) -> None:
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["reset_seeds"][0] = 7009
        unsigned = dict(manifest)
        unsigned.pop("self_sha256")
        manifest["self_sha256"] = canonical_fingerprint(unsigned)
        _write(manifest_path, manifest)
        with self.assertRaises(ValueError):
            load_grasp_lift_revalidation_metadata(self.root)

    def test_postclaim_loader_verifies_receipt_before_six_shard_reads(self) -> None:
        metadata, claim = self._claim()
        common = self.root / "git-common"
        registry = common / GRASP_LIFT_VALIDATION_CLAIM_REGISTRY
        registry.mkdir(parents=True)
        receipt = GraspLiftValidationClaimReceipt(
            repository_root=self.root,
            git_common_directory=common,
            claim_path=registry / f"{claim['claim_id']}.json",
            claim_id=claim["claim_id"],
            claim_self_sha256=claim["self_sha256"],
            claim_file_sha256="5" * 64,
        )
        events: list[str] = []
        fake_normalization = SimpleNamespace(sha256="6" * 64)
        fake_artifact = SimpleNamespace(
            artifact_sha256="7" * 64,
            train_content_sha256="8" * 64,
            normalization=fake_normalization,
        )
        group_by_trajectory_file = {
            Path(member["trajectory_descriptor"]).name: member["reset_group_id"]
            for member in metadata.manifest["members"]
        }

        def verify_persisted(*args, **kwargs):
            events.append("claim_verified")
            return receipt

        def strict_path(root, relative, *, trajectory_id, evidence):
            return root / relative

        def load_trajectory(path):
            events.append(f"trajectory:{path.name}")
            return SimpleNamespace(
                provenance=SimpleNamespace(
                    reset_group_id=group_by_trajectory_file[path.name]
                )
            )

        def load_evidence(path, *, trajectory):
            events.append(f"evidence:{path.name}")
            return SimpleNamespace()

        def member(**kwargs):
            return SimpleNamespace(
                commitment=kwargs["commitment"],
                model_trajectory=SimpleNamespace(
                    provenance=SimpleNamespace(
                        reset_group_id=kwargs["trajectory"].reset_group_id
                        if hasattr(kwargs["trajectory"], "reset_group_id")
                        else kwargs["trajectory"].provenance.reset_group_id
                    )
                ),
            )

        fake_result = object()
        fake_batch = SimpleNamespace()
        patches = (
            mock.patch(
                "rexpolicy.stage0.data.grasp_lift_revalidation."
                "_require_training_artifact",
                return_value=fake_artifact,
            ),
            mock.patch(
                "rexpolicy.stage0.data.grasp_lift_validation_claim."
                "verify_persisted_grasp_lift_validation_claim",
                side_effect=verify_persisted,
            ),
            mock.patch(
                "rexpolicy.stage0.data.grasp_lift_revalidation."
                "_strict_validation_shard_path",
                side_effect=strict_path,
            ),
            mock.patch(
                "rexpolicy.stage0.data.grasp_lift_revalidation.load_trajectory_shard",
                side_effect=load_trajectory,
            ),
            mock.patch(
                "rexpolicy.stage0.data.grasp_lift_revalidation."
                "load_grasp_lift_evidence_sidecar",
                side_effect=load_evidence,
            ),
            mock.patch(
                "rexpolicy.stage0.data.grasp_lift_revalidation."
                "materialize_grasp_lift_model_trajectory",
                return_value=SimpleNamespace(),
            ),
            mock.patch(
                "rexpolicy.stage0.data.grasp_lift_revalidation."
                "GraspLiftValidationMember",
                side_effect=member,
            ),
            mock.patch(
                "rexpolicy.stage0.data.grasp_lift_revalidation."
                "trajectory_corpus_sha256",
                return_value="a" * 64,
            ),
            mock.patch(
                "rexpolicy.stage0.data.grasp_lift_revalidation.extract_success_windows",
                return_value=(object(),),
            ),
            mock.patch(
                "rexpolicy.stage0.data.grasp_lift_revalidation.collate_success_windows",
                return_value=object(),
            ),
            mock.patch(
                "rexpolicy.stage0.data.grasp_lift_revalidation."
                "normalize_grasp_lift_no_z_batch_once",
                return_value=fake_batch,
            ),
            mock.patch(
                "rexpolicy.stage0.data.grasp_lift_revalidation._window_content_sha256",
                return_value="b" * 64,
            ),
            mock.patch(
                "rexpolicy.stage0.data.grasp_lift_revalidation._batch_content_sha256",
                return_value="c" * 64,
            ),
            mock.patch(
                "rexpolicy.stage0.data.grasp_lift_revalidation.trajectory_sha256",
                return_value="d" * 64,
            ),
            mock.patch(
                "rexpolicy.stage0.data.grasp_lift_revalidation."
                "GraspLiftClaimedRevalidationData",
                return_value=fake_result,
            ),
            mock.patch(
                "rexpolicy.stage0.data.grasp_lift_revalidation."
                "grasp_lift_composite_corpus_sha256",
                return_value=metadata.manifest["composite_corpus_sha256"],
            ),
        )
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
            patches[10],
            patches[11],
            patches[12],
            patches[13],
            patches[14],
            patches[15],
        ):
            result = load_claimed_grasp_lift_revalidation(
                self.root,
                repository_root=self.root,
                training_artifact=fake_artifact,
                claim=claim,
                claim_receipt=receipt,
                candidate_inventory_sha256=_HASHES["candidate"],
                preflight_sha256=_HASHES["preflight"],
                selection_protocol_sha256=_HASHES["protocol"],
                evaluator_implementation_sha256=_HASHES["evaluator"],
            )
        self.assertIs(result, fake_result)
        self.assertEqual(events[0], "claim_verified")
        self.assertEqual(sum(value.startswith("trajectory:") for value in events), 6)
        self.assertEqual(sum(value.startswith("evidence:") for value in events), 6)


if __name__ == "__main__":
    unittest.main()
