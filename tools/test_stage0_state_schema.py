"""Tests for the canonical 79D Stage 0 Newton state schema."""

from __future__ import annotations

import unittest
from dataclasses import replace

from rexpolicy.stage0 import (
    DEFAULT_STAGE0_STATE_SCHEMA,
    FRAME_INDEPENDENT,
    JOINT_SPACE_FRAME,
    RIGHT_ROBOT_BASE_FRAME,
    STAGE0_STATE_DIM,
    STAGE0_STATE_DTYPE,
    STAGE0_STATE_SCHEMA_ID,
    Stage0StateField,
    Stage0StateSchema,
)


EXPECTED_FIELDS = (
    ("arm_joint_position", 7),
    ("arm_joint_velocity", 7),
    ("hand_joint_position", 10),
    ("hand_joint_velocity", 10),
    ("eef_position", 3),
    ("eef_rotation_6d", 6),
    ("object_position", 3),
    ("object_rotation_6d", 6),
    ("object_linear_velocity", 3),
    ("object_angular_velocity", 3),
    ("goal_position", 3),
    ("eef_to_object", 3),
    ("object_to_goal", 3),
    ("finger_contact_counts", 5),
    ("has_hand_contact", 1),
    ("is_grasped", 1),
    ("task_phase_one_hot", 5),
)


class Stage0StateSchemaTest(unittest.TestCase):
    def test_default_schema_has_exact_order_dimension_and_slices(self) -> None:
        schema = DEFAULT_STAGE0_STATE_SCHEMA
        self.assertEqual(schema.schema_id, STAGE0_STATE_SCHEMA_ID)
        self.assertEqual(schema.dtype, STAGE0_STATE_DTYPE)
        self.assertEqual(schema.dimension, STAGE0_STATE_DIM)
        self.assertEqual(
            tuple((field.name, field.width) for field in schema.fields),
            EXPECTED_FIELDS,
        )

        expected_offset = 0
        covered = []
        for name, width in EXPECTED_FIELDS:
            field_slice = schema.slice(name)
            self.assertEqual(
                field_slice, slice(expected_offset, expected_offset + width)
            )
            covered.extend(range(field_slice.start, field_slice.stop))
            expected_offset += width
        self.assertEqual(covered, list(range(STAGE0_STATE_DIM)))
        with self.assertRaises(KeyError):
            schema.slice("world_position")
        with self.assertRaises(TypeError):
            schema.offsets["eef_position"] = 0  # type: ignore[index]

    def test_units_and_coordinate_frames_are_explicit(self) -> None:
        schema = DEFAULT_STAGE0_STATE_SCHEMA
        for name in (
            "arm_joint_position",
            "arm_joint_velocity",
            "hand_joint_position",
            "hand_joint_velocity",
        ):
            self.assertEqual(schema.field(name).frame, JOINT_SPACE_FRAME)
        for name in (
            "eef_position",
            "eef_rotation_6d",
            "object_position",
            "object_rotation_6d",
            "object_linear_velocity",
            "object_angular_velocity",
            "goal_position",
            "eef_to_object",
            "object_to_goal",
        ):
            self.assertEqual(schema.field(name).frame, RIGHT_ROBOT_BASE_FRAME)
        for name in (
            "finger_contact_counts",
            "has_hand_contact",
            "is_grasped",
            "task_phase_one_hot",
        ):
            self.assertEqual(schema.field(name).frame, FRAME_INDEPENDENT)

        self.assertEqual(schema.field("eef_position").unit, "metre")
        self.assertEqual(
            schema.field("object_linear_velocity").unit, "metre_per_second"
        )
        self.assertEqual(
            schema.field("object_angular_velocity").unit, "radian_per_second"
        )
        self.assertEqual(schema.field("finger_contact_counts").unit, "count")
        self.assertEqual(schema.field("has_hand_contact").unit, "boolean")
        self.assertEqual(schema.field("task_phase_one_hot").unit, "one_hot")

    def test_schema_round_trip_and_fingerprint_are_deterministic(self) -> None:
        schema = DEFAULT_STAGE0_STATE_SCHEMA
        restored = Stage0StateSchema.from_mapping(schema.to_record())
        self.assertEqual(restored, schema)
        self.assertEqual(restored.fingerprint, schema.fingerprint)
        self.assertEqual(schema.sha256, schema.fingerprint)
        self.assertEqual(len(schema.fingerprint), 64)

        changed_record = schema.to_record()
        changed_record["schema_id"] = "stage0/newton-state/diagnostic"
        changed = Stage0StateSchema.from_mapping(changed_record)
        self.assertNotEqual(changed.fingerprint, schema.fingerprint)

    def test_schema_records_reject_unknown_or_inconsistent_fields(self) -> None:
        record = DEFAULT_STAGE0_STATE_SCHEMA.to_record()
        record["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "Unknown state schema"):
            Stage0StateSchema.from_mapping(record)

        record = DEFAULT_STAGE0_STATE_SCHEMA.to_record()
        record["fields"][1]["offset"] = 99
        with self.assertRaisesRegex(ValueError, "contiguous"):
            Stage0StateSchema.from_mapping(record)

        record = DEFAULT_STAGE0_STATE_SCHEMA.to_record()
        record["fields"][0]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "Unknown state field"):
            Stage0StateSchema.from_mapping(record)

        first = DEFAULT_STAGE0_STATE_SCHEMA.fields[0]
        duplicate = replace(DEFAULT_STAGE0_STATE_SCHEMA.fields[1], name=first.name)
        with self.assertRaisesRegex(ValueError, "unique"):
            Stage0StateSchema(fields=(first, duplicate))

    def test_world_frame_and_invalid_metadata_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame"):
            Stage0StateField(
                "object_position",
                3,
                "object",
                "metre",
                "world",
            )
        with self.assertRaisesRegex(ValueError, "lower_snake_case"):
            Stage0StateField(
                "ObjectPosition",
                3,
                "object",
                "metre",
                RIGHT_ROBOT_BASE_FRAME,
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            Stage0StateField(
                "object_position",
                0,
                "object",
                "metre",
                RIGHT_ROBOT_BASE_FRAME,
            )


if __name__ == "__main__":
    unittest.main()
