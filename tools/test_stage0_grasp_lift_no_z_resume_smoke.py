"""CPU tests for the Grasp-Lift no-z bitwise resume smoke helpers."""

from __future__ import annotations

import unittest

from tools.smoke_stage0_grasp_lift_no_z_resume import (
    _assert_bitwise_equal,
    _validate_smoke_schedule,
)


try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - minimal orchestration host
    torch = None


class GraspLiftNoZResumeSmokeTest(unittest.TestCase):
    def test_schedule_is_bounded_and_strictly_split(self) -> None:
        self.assertEqual(_validate_smoke_schedule(4, 2), (4, 2))
        self.assertEqual(_validate_smoke_schedule(2, 1), (2, 1))

        for total, checkpoint in (
            (True, 1),
            (1, 1),
            (33, 2),
            (4, False),
            (4, 0),
            (4, 4),
            (4, 5),
            (4.0, 2),
            (4, 2.0),
        ):
            with self.subTest(total=total, checkpoint=checkpoint):
                with self.assertRaises(ValueError):
                    _validate_smoke_schedule(total, checkpoint)

    def test_recursive_comparator_accepts_exact_plain_structure(self) -> None:
        left = {
            "groups": [{"betas": (0.9, 0.999), "fused": True}],
            "state": {0: {"step": 2, "name": "weight"}},
            "optional": None,
        }
        right = {
            "optional": None,
            "state": {0: {"name": "weight", "step": 2}},
            "groups": [{"fused": True, "betas": (0.9, 0.999)}],
        }
        self.assertEqual(_assert_bitwise_equal(left, right), 0)

    def test_recursive_comparator_reports_precise_structure_mismatches(self) -> None:
        with self.assertRaisesRegex(AssertionError, r"root\['state'\]\[0\]"):
            _assert_bitwise_equal(
                {"state": {0: {"step": 2}}},
                {"state": {0: {"step": 3}}},
            )
        with self.assertRaisesRegex(AssertionError, "mapping keys differ"):
            _assert_bitwise_equal({"a": 1}, {"b": 1})
        with self.assertRaisesRegex(AssertionError, "types differ"):
            _assert_bitwise_equal([1, 2], (1, 2))

    def test_recursive_comparator_treats_python_float_bytes_strictly(self) -> None:
        self.assertEqual(_assert_bitwise_equal(0.0, 0.0), 0)
        with self.assertRaisesRegex(AssertionError, "float bytes differ"):
            _assert_bitwise_equal(0.0, -0.0)

    @unittest.skipIf(torch is None, "Torch is unavailable on this CPU test host")
    def test_tensor_comparison_is_byte_exact_not_merely_value_equal(self) -> None:
        assert torch is not None
        left = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float32)
        right = left.clone()
        self.assertEqual(
            _assert_bitwise_equal({"model": left}, {"model": right}),
            1,
        )

        right[0] = -0.0
        self.assertTrue(torch.equal(left, right))
        with self.assertRaisesRegex(AssertionError, "tensor bytes differ"):
            _assert_bitwise_equal({"model": left}, {"model": right})

    @unittest.skipIf(torch is None, "Torch is unavailable on this CPU test host")
    def test_tensor_comparison_rejects_dtype_and_shape_changes(self) -> None:
        assert torch is not None
        with self.assertRaisesRegex(AssertionError, "tensor dtype differs"):
            _assert_bitwise_equal(
                torch.ones(2, dtype=torch.float32),
                torch.ones(2, dtype=torch.float64),
            )
        with self.assertRaisesRegex(AssertionError, "tensor shape differs"):
            _assert_bitwise_equal(
                torch.ones(2, dtype=torch.float32),
                torch.ones((1, 2), dtype=torch.float32),
            )


if __name__ == "__main__":
    unittest.main()
