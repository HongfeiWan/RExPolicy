"""Tests for immutable, transition-aligned Grasp-Lift evidence sidecars."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from rexpolicy.stage0.data.grasp_lift_evidence import (
    GRASP_LIFT_EVIDENCE_TENSOR_SPECS,
    GraspLiftEvidence,
    GraspLiftEvidenceIntegrityError,
    GraspLiftEvidenceMetadata,
    grasp_lift_composite_corpus_sha256,
    grasp_lift_evidence_sha256,
    load_grasp_lift_evidence_sidecar,
    validate_grasp_lift_evidence_trajectory,
    write_grasp_lift_evidence_sidecar,
)
from rexpolicy.stage0.data.trajectory import (
    Stage0Trajectory,
    Stage0TrajectoryProvenance,
)
from rexpolicy.stage0.envs.grasp_lift_action import (
    DEFAULT_GRASP_LIFT_ACTION_SCHEMA,
)
from rexpolicy.stage0.envs.grasp_lift_modes import (
    DEFAULT_GRASP_LIFT_AUTHORING_MODE,
)
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA
from rexpolicy.stage0.types import Stage0Outcome


def _trajectory(
    trajectory_id: str = "grasp-000",
    *,
    world: int = 2,
    reset_seed: int = 42,
    transition_count: int = 4,
) -> Stage0Trajectory:
    states = torch.arange(
        (transition_count + 1) * DEFAULT_STAGE0_STATE_SCHEMA.dimension,
        dtype=torch.float32,
    ).reshape(transition_count + 1, DEFAULT_STAGE0_STATE_SCHEMA.dimension)
    states[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("object_to_goal")] = torch.tensor(
        (0.0, 0.0, 1.0), dtype=torch.float32
    )
    actions = torch.arange(
        transition_count * 19, dtype=torch.float32
    ).reshape(transition_count, 19) / 100.0
    provenance = Stage0TrajectoryProvenance(
        trajectory_id=trajectory_id,
        reset_group_id=f"reset-{reset_seed}",
        task_id="grasp_lift_green_bottle/v1",
        oracle_id="grasp-lift-oracle/v1",
        simulator_sha256="1" * 64,
        experiment_sha256="2" * 64,
        state_schema_sha256=DEFAULT_STAGE0_STATE_SCHEMA.sha256,
        action_schema_sha256=DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
        generation=1,
        rank=0,
        world=world,
        reset_seed=reset_seed,
    )
    return Stage0Trajectory(
        provenance=provenance,
        states=states,
        actions=actions,
        outcome=Stage0Outcome.TIMEOUT,
        terminated=False,
        truncated=True,
    )


def _tensor_payload(trajectory: Stage0Trajectory) -> dict[str, torch.Tensor]:
    count = trajectory.transition_count
    tensors: dict[str, torch.Tensor] = {}
    dtype_by_name = {
        "float32": torch.float32,
        "bool": torch.bool,
        "int64": torch.int64,
    }
    for name, spec in GRASP_LIFT_EVIDENCE_TENSOR_SPECS.items():
        tensors[name] = torch.zeros(
            spec.shape(count), dtype=dtype_by_name[spec.dtype]
        )
    tensors["proposal_action"] = trajectory.actions.clone()
    tensors["executed_action"] = trajectory.actions.clone()
    tensors["measured_eef_position"] = trajectory.states[
        1:, DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position")
    ].clone()
    tensors["measured_hand_joint_position"] = trajectory.states[
        1:, DEFAULT_STAGE0_STATE_SCHEMA.slice("hand_joint_position")
    ].clone()
    tensors["triangle_pair_buffer_available"].fill_(True)
    return tensors


def _evidence(trajectory: Stage0Trajectory) -> GraspLiftEvidence:
    metadata = GraspLiftEvidenceMetadata.from_trajectory(
        trajectory,
        mode_id=DEFAULT_GRASP_LIFT_AUTHORING_MODE.mode_id,
        mode_sha256=DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
        reset_lift_axis=(0.0, 0.0, 1.0),
    )
    return GraspLiftEvidence(metadata=metadata, tensors=_tensor_payload(trajectory))


class GraspLiftEvidenceTests(unittest.TestCase):
    def test_round_trip_is_semantic_and_immutable(self) -> None:
        trajectory = _trajectory()
        source = _evidence(trajectory)
        with tempfile.TemporaryDirectory() as temporary:
            descriptor = write_grasp_lift_evidence_sidecar(
                temporary, source, trajectory=trajectory
            )
            restored = load_grasp_lift_evidence_sidecar(
                descriptor, trajectory=trajectory
            )
            self.assertEqual(restored.metadata, source.metadata)
            self.assertEqual(
                grasp_lift_evidence_sha256(restored),
                grasp_lift_evidence_sha256(source),
            )
            for name in source.tensors:
                self.assertTrue(torch.equal(restored.tensors[name], source.tensors[name]))

            record = json.loads(descriptor.read_text(encoding="utf-8"))
            payload = descriptor.parent / record["payload_file"]
            descriptor_bytes = descriptor.read_bytes()
            payload_bytes = payload.read_bytes()
            with self.assertRaises(FileExistsError):
                write_grasp_lift_evidence_sidecar(temporary, source)
            self.assertEqual(descriptor.read_bytes(), descriptor_bytes)
            self.assertEqual(payload.read_bytes(), payload_bytes)

    def test_cpu_dtype_shape_and_exact_keys_are_strict(self) -> None:
        trajectory = _trajectory()
        metadata = _evidence(trajectory).metadata
        tensors = _tensor_payload(trajectory)
        tensors.pop("proposal_action")
        with self.assertRaisesRegex(ValueError, "missing"):
            GraspLiftEvidence(metadata, tensors)

        tensors = _tensor_payload(trajectory)
        tensors["proposal_action"] = tensors["proposal_action"].double()
        with self.assertRaisesRegex(ValueError, "float32"):
            GraspLiftEvidence(metadata, tensors)

        tensors = _tensor_payload(trajectory)
        tensors["authoring_phase_before"] = tensors[
            "authoring_phase_before"
        ].to(torch.int32)
        with self.assertRaisesRegex(ValueError, "int64"):
            GraspLiftEvidence(metadata, tensors)

        tensors = _tensor_payload(trajectory)
        tensors["oracle_lift_height"][0] = torch.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            GraspLiftEvidence(metadata, tensors)

        tensors = _tensor_payload(trajectory)
        tensors["finger_contact_counts"][0, 0] = -1
        with self.assertRaisesRegex(ValueError, "negative"):
            GraspLiftEvidence(metadata, tensors)

        if torch.cuda.is_available():
            tensors = _tensor_payload(trajectory)
            tensors["proposal_action"] = tensors["proposal_action"].cuda()
            with self.assertRaisesRegex(ValueError, "CPU"):
                GraspLiftEvidence(metadata, tensors)

    def test_executed_action_and_post_state_measurements_are_bound(self) -> None:
        trajectory = _trajectory()
        source = _evidence(trajectory)
        validate_grasp_lift_evidence_trajectory(source, trajectory)

        tensors = dict(source.tensors)
        tensors["executed_action"] = tensors["executed_action"].clone()
        tensors["executed_action"][1, 2] += 1.0
        changed = GraspLiftEvidence(source.metadata, tensors)
        with self.assertRaisesRegex(ValueError, "trajectory.actions"):
            validate_grasp_lift_evidence_trajectory(changed, trajectory)

        tensors = dict(source.tensors)
        tensors["measured_eef_position"] = tensors[
            "measured_eef_position"
        ].clone()
        tensors["measured_eef_position"][0, 0] += 1.0
        changed = GraspLiftEvidence(source.metadata, tensors)
        with self.assertRaisesRegex(ValueError, r"state\[t \+ 1\]"):
            validate_grasp_lift_evidence_trajectory(changed, trajectory)

        wrong_axis = replace(source.metadata, reset_lift_axis=(0.0, 1.0, 0.0))
        with self.assertRaisesRegex(ValueError, "reset_lift_axis"):
            validate_grasp_lift_evidence_trajectory(
                GraspLiftEvidence(wrong_axis, source.tensors), trajectory
            )

    def test_latches_and_collision_diagnostics_fail_closed(self) -> None:
        trajectory = _trajectory()
        source = _evidence(trajectory)
        tensors = dict(source.tensors)
        tensors["oracle_failure"] = torch.tensor(
            (False, True, False, False), dtype=torch.bool
        )
        with self.assertRaisesRegex(ValueError, "latched"):
            GraspLiftEvidence(source.metadata, tensors)

        tensors = dict(source.tensors)
        tensors["rigid_contact_overflow_frame_count"] = torch.tensor(
            (0, 0, 1, 0), dtype=torch.int64
        )
        with self.assertRaisesRegex(ValueError, "collision diagnostics"):
            GraspLiftEvidence(source.metadata, tensors)

        # The transient flag covers every physics frame while the count is a
        # current/last-frame diagnostic, so they are intentionally independent.
        tensors = dict(source.tensors)
        tensors["forbidden_hand_contact"] = torch.tensor(
            (True, False, False, False), dtype=torch.bool
        )
        tensors["forbidden_hand_contact_count"] = torch.tensor(
            (0, 1, 0, 0), dtype=torch.int64
        )
        GraspLiftEvidence(source.metadata, tensors)

    def test_real_implementation_hash_is_independent_of_simulator_hash(self) -> None:
        trajectory = _trajectory()
        metadata = GraspLiftEvidenceMetadata.from_trajectory(
            trajectory,
            mode_id=DEFAULT_GRASP_LIFT_AUTHORING_MODE.mode_id,
            mode_sha256=DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
            reset_lift_axis=(0.0, 0.0, 1.0),
            implementation_sha256="9" * 64,
        )
        self.assertNotEqual(
            metadata.implementation_sha256,
            trajectory.provenance.simulator_sha256,
        )
        validate_grasp_lift_evidence_trajectory(
            GraspLiftEvidence(metadata, _tensor_payload(trajectory)), trajectory
        )

    def test_payload_and_descriptor_tampering_are_detected(self) -> None:
        trajectory = _trajectory()
        with tempfile.TemporaryDirectory() as temporary:
            descriptor = write_grasp_lift_evidence_sidecar(
                temporary, _evidence(trajectory), trajectory=trajectory
            )
            record = json.loads(descriptor.read_text(encoding="utf-8"))
            payload = descriptor.parent / record["payload_file"]
            changed = bytearray(payload.read_bytes())
            changed[len(changed) // 2] ^= 1
            payload.write_bytes(changed)
            with self.assertRaisesRegex(
                GraspLiftEvidenceIntegrityError, "SHA-256 mismatch"
            ):
                load_grasp_lift_evidence_sidecar(descriptor)

        with tempfile.TemporaryDirectory() as temporary:
            descriptor = write_grasp_lift_evidence_sidecar(
                temporary, _evidence(trajectory)
            )
            record = json.loads(descriptor.read_text(encoding="utf-8"))
            record["payload_file"] = "../outside.pt"
            descriptor.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(
                GraspLiftEvidenceIntegrityError, "local basename"
            ):
                load_grasp_lift_evidence_sidecar(descriptor)

    def test_composite_corpus_hash_is_sorted_and_binds_both_hashes(self) -> None:
        first_trajectory = _trajectory("grasp-a", world=0, reset_seed=10)
        second_trajectory = _trajectory("grasp-b", world=1, reset_seed=11)
        first = _evidence(first_trajectory)
        second = _evidence(second_trajectory)
        forward = grasp_lift_composite_corpus_sha256(
            ((first_trajectory, first), (second_trajectory, second))
        )
        reverse = grasp_lift_composite_corpus_sha256(
            ((second_trajectory, second), (first_trajectory, first))
        )
        self.assertEqual(forward, reverse)

        tensors = dict(second.tensors)
        tensors["proposal_action"] = tensors["proposal_action"].clone()
        tensors["proposal_action"][0, 0] += 0.125
        changed_second = GraspLiftEvidence(second.metadata, tensors)
        changed = grasp_lift_composite_corpus_sha256(
            ((first_trajectory, first), (second_trajectory, changed_second))
        )
        self.assertNotEqual(forward, changed)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            grasp_lift_composite_corpus_sha256(
                ((first_trajectory, first), (first_trajectory, first))
            )


if __name__ == "__main__":
    unittest.main()
