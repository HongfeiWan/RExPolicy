"""Tests for the dependency-free strict canonical JSON boundary."""

from __future__ import annotations

import unittest

from rexpolicy.tasking.canonical import (
    canonical_fingerprint,
    strict_json_loads,
)


class TestCanonicalTaskJson(unittest.TestCase):
    def test_duplicate_and_non_finite_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            strict_json_loads('{"version":1,"version":2}')
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            strict_json_loads('{"value":NaN}')

    def test_fingerprint_ignores_object_key_order(self) -> None:
        self.assertEqual(
            canonical_fingerprint({"b": 2, "a": 1}),
            canonical_fingerprint({"a": 1, "b": 2}),
        )

    def test_size_limit_is_enforced_before_parsing(self) -> None:
        with self.assertRaisesRegex(ValueError, "size limit"):
            strict_json_loads('{"long":"value"}', max_bytes=4)


if __name__ == "__main__":
    unittest.main()
