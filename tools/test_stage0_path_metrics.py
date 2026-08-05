"""CPU tests for Stage 0 same-reset rollout path adherence."""

from __future__ import annotations

import json
import unittest

import torch

from rexpolicy.stage0.evaluation.path_metrics import (
    UNSCORED_PREDICTION,
    analyze_path_adherence,
)


MODES = ("direct", "left", "over", "right")


def _path(reset_offset: float, mode_index: int, *, points: int = 9) -> torch.Tensor:
    progress = torch.linspace(0.0, 1.0, points, dtype=torch.float64)
    lateral = (mode_index - 1.5) * 0.15 * torch.sin(torch.pi * progress)
    vertical = 0.08 * mode_index * torch.sin(torch.pi * progress)
    return torch.stack(
        (
            reset_offset + progress,
            reset_offset * 0.1 + lateral,
            0.5 + vertical,
        ),
        dim=1,
    )


def _references() -> dict[str, dict[str, torch.Tensor]]:
    return {
        reset_id: {
            mode: _path(reset_offset, index)
            for index, mode in enumerate(MODES)
        }
        for reset_id, reset_offset in (("reset-a", 0.0), ("reset-b", 7.0))
    }


class Stage0PathMetricsTest(unittest.TestCase):
    def test_variable_speed_paths_match_same_reset_references(self) -> None:
        references = _references()
        learned: list[torch.Tensor] = []
        resets: list[str] = []
        expected: list[str] = []
        for reset_id in ("reset-a", "reset-b"):
            for mode_id in MODES:
                reference = references[reset_id][mode_id]
                # Duplicates and nonuniform segment subdivisions alter the speed
                # profile but not the geometric polyline.
                learned.append(
                    torch.cat(
                        (
                            reference[:2],
                            reference[1:2],
                            reference[2:6],
                            reference[5:6],
                            reference[6:],
                        )
                    ).float()
                )
                resets.append(reset_id)
                expected.append(mode_id)

        metrics = analyze_path_adherence(
            learned,
            successful=torch.ones(len(learned), dtype=torch.bool),
            reset_group_ids=resets,
            expected_mode_ids=expected,
            reference_paths=references,
            mode_ids=MODES,
            resample_points=101,
        )

        self.assertEqual(metrics.predicted_mode_ids, tuple(expected))
        self.assertEqual(metrics.successful_count, 8)
        self.assertEqual(metrics.scored_count, 8)
        self.assertAlmostEqual(metrics.macro_f1, 1.0)
        self.assertAlmostEqual(metrics.adherence_rate, 1.0)
        self.assertLess(metrics.mean_expected_reference_rms_m or 1.0, 1e-6)
        self.assertEqual(metrics.statuses, ("scored",) * 8)
        json.dumps(metrics.to_record(), allow_nan=False, sort_keys=True)

    def test_matching_never_uses_a_reference_from_another_reset(self) -> None:
        def point(x_coordinate: float) -> torch.Tensor:
            return torch.tensor([[x_coordinate, 0.0, 0.0]], dtype=torch.float64)

        references = {
            "reset-a": {
                mode: point(10.0 * index) for index, mode in enumerate(MODES)
            },
            "reset-b": {
                "direct": point(100.0),
                "left": point(110.0),
                "over": point(120.0),
                "right": point(0.1),
            },
        }
        # reset-b/right is an exact global match.  For a reset-a episode it is
        # ineligible, so reset-a/direct must be selected at a 0.1 m distance.
        learned = references["reset-b"]["right"]
        metrics = analyze_path_adherence(
            (learned,),
            successful=(True,),
            reset_group_ids=("reset-a",),
            expected_mode_ids=("direct",),
            reference_paths=references,
            mode_ids=MODES,
        )

        self.assertEqual(metrics.predicted_mode_ids, ("direct",))
        self.assertAlmostEqual(metrics.nearest_reference_rms_m[0] or 0.0, 0.1)

    def test_unsuccessful_and_missing_paths_are_explicit_false_negatives(self) -> None:
        references = _references()
        learned = (
            references["reset-a"]["direct"],
            references["reset-a"]["left"],
            None,
            references["reset-a"]["right"],
        )
        metrics = analyze_path_adherence(
            learned,
            successful=(True, False, True, True),
            reset_group_ids=("reset-a",) * 4,
            expected_mode_ids=MODES,
            reference_paths=references,
            mode_ids=MODES,
        )

        self.assertEqual(
            metrics.predicted_mode_ids,
            ("direct", None, None, "right"),
        )
        self.assertEqual(
            metrics.statuses,
            ("scored", "unsuccessful", "missing_success_path", "scored"),
        )
        self.assertEqual(metrics.successful_count, 3)
        self.assertEqual(metrics.scored_count, 2)
        self.assertEqual(metrics.missing_success_path_count, 1)
        self.assertAlmostEqual(metrics.macro_f1, 0.5)
        self.assertAlmostEqual(metrics.conditional_macro_f1, 0.5)
        self.assertAlmostEqual(metrics.adherence_rate, 0.5)
        self.assertAlmostEqual(metrics.conditional_adherence_rate, 1.0)
        per_mode = {item.mode_id: item for item in metrics.per_mode}
        self.assertEqual(per_mode["left"].unscored, 1)
        self.assertEqual(per_mode["over"].missing_success_path, 1)
        record = metrics.to_record()
        self.assertEqual(
            record["confusion"]["column_labels"],
            [*MODES, UNSCORED_PREDICTION],
        )
        self.assertEqual(record["confusion"]["matrix"][1][-1], 1)
        self.assertEqual(
            record["per_mode"]["left"]["predicted_counts"][
                UNSCORED_PREDICTION
            ],
            1,
        )
        json.dumps(record, allow_nan=False, sort_keys=True)

    def test_supplied_targets_score_a_permuted_latent_causally(self) -> None:
        references = _references()
        permuted_targets = ("left", "over", "right", "direct")
        learned = tuple(references["reset-a"][mode] for mode in permuted_targets)
        causal = analyze_path_adherence(
            learned,
            successful=(True,) * 4,
            reset_group_ids=("reset-a",) * 4,
            expected_mode_ids=permuted_targets,
            reference_paths=references,
            mode_ids=MODES,
        )
        stale_order = analyze_path_adherence(
            learned,
            successful=(True,) * 4,
            reset_group_ids=("reset-a",) * 4,
            expected_mode_ids=MODES,
            reference_paths=references,
            mode_ids=MODES,
        )

        self.assertAlmostEqual(causal.macro_f1, 1.0)
        self.assertAlmostEqual(causal.adherence_rate, 1.0)
        self.assertAlmostEqual(stale_order.macro_f1, 0.0)
        self.assertAlmostEqual(stale_order.adherence_rate, 0.0)

    def test_constant_one_point_paths_are_finite_and_deterministic(self) -> None:
        references = {
            "reset": {
                mode: torch.tensor(
                    [[float(index), 0.0, 0.0]], dtype=torch.float32
                )
                for index, mode in enumerate(MODES)
            }
        }
        kwargs = {
            "successful": (True,),
            "reset_group_ids": ("reset",),
            "expected_mode_ids": ("over",),
            "reference_paths": references,
            "mode_ids": MODES,
        }
        first = analyze_path_adherence(
            (torch.tensor([[2.0, 0.0, 0.0]]),),
            **kwargs,
        )
        second = analyze_path_adherence(
            (torch.tensor([[2.0, 0.0, 0.0]]),),
            **kwargs,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.predicted_mode_ids, ("over",))
        self.assertEqual(first.nearest_reference_rms_m, (0.0,))

    def test_inputs_are_strictly_validated(self) -> None:
        references = _references()
        valid = references["reset-a"]["direct"]
        common = {
            "successful": (True,),
            "reset_group_ids": ("reset-a",),
            "expected_mode_ids": ("direct",),
            "reference_paths": references,
            "mode_ids": MODES,
        }
        with self.assertRaisesRegex(ValueError, "finite"):
            invalid = valid.clone()
            invalid[0, 0] = float("nan")
            analyze_path_adherence((invalid,), **common)
        with self.assertRaisesRegex(ValueError, "finite"):
            invalid_references = _references()
            invalid_references["reset-a"]["left"][0, 0] = float("inf")
            analyze_path_adherence(
                (valid,),
                **{**common, "reference_paths": invalid_references},
            )
        with self.assertRaisesRegex(ValueError, "one value per episode"):
            analyze_path_adherence(
                (valid,),
                **{**common, "expected_mode_ids": ("direct", "left")},
            )
        with self.assertRaisesRegex(ValueError, "unconfigured"):
            analyze_path_adherence(
                (valid,),
                **{**common, "expected_mode_ids": ("unknown",)},
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            analyze_path_adherence(
                (valid,),
                **{**common, "mode_ids": ("direct", "direct")},
            )
        with self.assertRaisesRegex(ValueError, "missing mode"):
            incomplete = _references()
            del incomplete["reset-a"]["over"]
            analyze_path_adherence(
                (valid,),
                **{**common, "reference_paths": incomplete},
            )
        with self.assertRaisesRegex(ValueError, "only booleans"):
            analyze_path_adherence(
                (valid,),
                **{**common, "successful": (1,)},
            )
        with self.assertRaisesRegex(ValueError, "at least two"):
            analyze_path_adherence(
                (valid,),
                **{**common, "resample_points": 1},
            )


if __name__ == "__main__":
    unittest.main()
