"""Tests for name-safe Linker L10 hand target ordering."""

from __future__ import annotations

import unittest

from rexpolicy.models import NamedJointValues
from rexpolicy.retargeting.hand_config import ordered_joint_positions


class OrderedJointPositionsTest(unittest.TestCase):
    def test_reorders_by_name_instead_of_source_position(self) -> None:
        pose = NamedJointValues(
            joint_names=("roll", "pitch", "yaw"),
            joint_positions=(1.0, 2.0, 3.0),
        )

        result = ordered_joint_positions(pose, ("pitch", "yaw", "roll"))

        self.assertEqual(result, (2.0, 3.0, 1.0))

    def test_missing_and_duplicate_names_fail_closed(self) -> None:
        pose = NamedJointValues(
            joint_names=("pitch", "yaw"),
            joint_positions=(2.0, 3.0),
        )
        with self.assertRaises(KeyError):
            ordered_joint_positions(pose, ("pitch", "roll"))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            ordered_joint_positions(pose, ("pitch", "pitch"))
        duplicate_pose = NamedJointValues(
            joint_names=("pitch", "pitch"),
            joint_positions=(1.0, 2.0),
        )
        with self.assertRaisesRegex(ValueError, "duplicate joint names"):
            ordered_joint_positions(duplicate_pose, ("pitch",))


if __name__ == "__main__":
    unittest.main()
