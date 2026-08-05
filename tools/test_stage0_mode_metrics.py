"""CPU tests for held-out Stage 0 latent mode diagnostics."""

from __future__ import annotations

import json
import math
import unittest

import torch

from rexpolicy.stage0.evaluation.mode_metrics import analyze_latent_modes


def _mode_separated_fixture() -> tuple[
    torch.Tensor,
    tuple[str, ...],
    tuple[str, ...],
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(73)
    mode_names = ("direct", "left", "right", "over")
    centers = torch.eye(4, dtype=torch.float64) * 12.0
    rows: list[torch.Tensor] = []
    modes: list[str] = []
    resets: list[str] = []
    reset_coordinates: list[torch.Tensor] = []
    for reset_index in range(10):
        coordinate = torch.tensor(
            [reset_index / 9.0, (reset_index % 3) / 2.0],
            dtype=torch.float64,
        )
        reset_effect = (
            torch.tensor(
                [coordinate[0], coordinate[1], -coordinate[0], -coordinate[1]],
                dtype=torch.float64,
            )
            * 0.08
        )
        for mode_index, mode_name in enumerate(mode_names):
            for _ in range(2):
                rows.append(
                    centers[mode_index]
                    + reset_effect
                    + 0.01 * torch.randn(4, generator=generator, dtype=torch.float64)
                )
                modes.append(mode_name)
                resets.append(f"reset-{reset_index:02d}")
                reset_coordinates.append(coordinate)
    return (
        torch.stack(rows),
        tuple(modes),
        tuple(resets),
        torch.stack(reset_coordinates),
    )


class Stage0LatentModeMetricsTest(unittest.TestCase):
    def test_four_clear_modes_score_high_on_held_out_resets(self) -> None:
        latents, modes, resets, reset_xy = _mode_separated_fixture()
        metrics = analyze_latent_modes(
            latents,
            modes,
            resets,
            reset_xy=reset_xy,
            ridge_alpha=1e-4,
        )

        self.assertEqual(metrics.sample_count, 80)
        self.assertEqual(metrics.mode_ids, ("direct", "left", "over", "right"))
        self.assertEqual(metrics.reset_group_count, 10)
        self.assertGreater(metrics.nearest_centroid_macro_f1, 0.99)
        self.assertGreater(metrics.nearest_centroid_nmi, 0.99)
        self.assertGreater(metrics.euclidean_silhouette, 0.98)
        self.assertGreater(metrics.mode_to_reset_distance_ratio, 50.0)
        self.assertEqual(tuple(metrics.nearest_centroid_predictions), modes)

        record = metrics.to_record()
        self.assertEqual(record["nearest_centroid"]["protocol"], "leave_one_reset_out")
        self.assertEqual(record["space"], "original_latent")
        self.assertEqual(record["authority"], "diagnostic_only")
        json.dumps(record, allow_nan=False, sort_keys=True)

    def test_reset_coded_latents_have_a_high_held_out_ridge_probe(self) -> None:
        modes = ("direct", "left", "over", "right")
        rows: list[torch.Tensor] = []
        mode_ids: list[str] = []
        reset_ids: list[str] = []
        coordinates: list[torch.Tensor] = []
        for reset_index in range(12):
            angle = 2.0 * math.pi * reset_index / 12.0
            xy = torch.tensor(
                [math.cos(angle), math.sin(angle)],
                dtype=torch.float64,
            )
            for mode_index, mode_name in enumerate(modes):
                weak_mode = torch.zeros(4, dtype=torch.float64)
                weak_mode[mode_index] = 0.01
                rows.append(torch.cat((8.0 * xy, weak_mode)))
                mode_ids.append(mode_name)
                reset_ids.append(f"reset-{reset_index:02d}")
                coordinates.append(xy)

        metrics = analyze_latent_modes(
            torch.stack(rows),
            tuple(mode_ids),
            tuple(reset_ids),
            reset_xy=torch.stack(coordinates),
            ridge_alpha=1e-6,
        )
        self.assertIsNotNone(metrics.reset_xy_linear_probe_r2)
        self.assertGreater(metrics.reset_xy_linear_probe_r2 or 0.0, 0.999)
        self.assertIsNotNone(metrics.reset_xy_linear_probe_r2_by_axis)
        for score in metrics.reset_xy_linear_probe_r2_by_axis or ():
            self.assertIsNotNone(score)
            self.assertGreater(score or 0.0, 0.999)

    def test_results_are_deterministic_and_optional_probe_is_json_null(self) -> None:
        latents, modes, resets, _ = _mode_separated_fixture()
        first = analyze_latent_modes(latents, modes, resets)
        second = analyze_latent_modes(latents, modes, resets)
        self.assertEqual(first, second)
        self.assertIsNone(first.reset_xy_linear_probe_r2)
        self.assertIsNone(first.to_record()["reset_xy_linear_probe"])

    def test_inputs_and_factorial_design_are_strict(self) -> None:
        latents, modes, resets, reset_xy = _mode_separated_fixture()
        with self.assertRaisesRegex(ValueError, "one value per latent"):
            analyze_latent_modes(latents, modes[:-1], resets)
        with self.assertRaisesRegex(ValueError, "at least two unique modes"):
            analyze_latent_modes(latents, ("one",) * len(modes), resets)

        isolated_modes = list(modes)
        isolated_modes[0:2] = ("isolated", "isolated")
        with self.assertRaisesRegex(ValueError, "at least two reset groups"):
            analyze_latent_modes(latents, tuple(isolated_modes), resets)

        non_finite = latents.clone()
        non_finite[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            analyze_latent_modes(non_finite, modes, resets)

        with self.assertRaisesRegex(ValueError, "shape"):
            analyze_latent_modes(
                latents,
                modes,
                resets,
                reset_xy=reset_xy[:, :1],
            )
        changing_xy = reset_xy.clone()
        changing_xy[1, 0] += 0.25
        with self.assertRaisesRegex(ValueError, "invariant"):
            analyze_latent_modes(
                latents,
                modes,
                resets,
                reset_xy=changing_xy,
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            analyze_latent_modes(latents, modes, resets, ridge_alpha=0.0)


if __name__ == "__main__":
    unittest.main()
