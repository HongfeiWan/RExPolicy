"""CPU tests for Stage 0 latent diagnostics and policy controls."""

from __future__ import annotations

import unittest

import torch

from rexpolicy.stage0.evaluation import (
    CollapseGate,
    ConditioningMode,
    FixedEvaluationBudget,
    PolicyEvaluationRun,
    analyze_latents,
    compare_conditioning_runs,
    deterministic_pca,
)
from rexpolicy.stage0.types import Stage0Outcome


class Stage0LatentAnalysisTest(unittest.TestCase):
    def test_original_space_statistics_pass_a_spread_latent_set(self) -> None:
        latents = torch.randn(
            256,
            6,
            generator=torch.Generator().manual_seed(17),
        )
        statistics = analyze_latents(
            latents,
            gate=CollapseGate(
                minimum_dimension_variance=0.5,
                minimum_active_dimension_fraction=1.0,
                minimum_effective_rank=4.0,
                maximum_mean_absolute_pairwise_cosine=0.6,
            ),
        )
        self.assertEqual(statistics.sample_count, 256)
        self.assertEqual(statistics.latent_dim, 6)
        self.assertEqual(tuple(statistics.per_dimension_variance.shape), (6,))
        self.assertEqual(tuple(statistics.covariance.shape), (6, 6))
        torch.testing.assert_close(
            statistics.per_dimension_variance,
            statistics.covariance.diagonal(),
        )
        self.assertGreater(statistics.effective_rank, 4.0)
        self.assertTrue(statistics.collapse_gate_passed)
        self.assertFalse(statistics.collapsed)
        self.assertEqual(statistics.to_record()["space"], "original_latent")

    def test_collapse_gate_catches_constant_and_rank_one_latents(self) -> None:
        constant = analyze_latents(torch.ones(12, 4))
        self.assertTrue(constant.collapsed)
        self.assertEqual(constant.effective_rank, 0.0)
        self.assertIn(
            "active_dimension_fraction_below_minimum",
            constant.collapse_reasons,
        )

        scalar = torch.linspace(-1.0, 1.0, 32).unsqueeze(1)
        rank_one = torch.cat((scalar, 2.0 * scalar, -scalar), dim=1)
        statistics = analyze_latents(rank_one)
        self.assertLess(statistics.effective_rank, 1.01)
        self.assertIn(
            "effective_rank_below_minimum",
            statistics.collapse_reasons,
        )
        self.assertIn(
            "mean_absolute_pairwise_cosine_above_maximum",
            statistics.collapse_reasons,
        )

    def test_latent_inputs_and_gate_are_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank-2"):
            analyze_latents(torch.zeros(2, 2, 2))
        with self.assertRaisesRegex(ValueError, "at least two"):
            analyze_latents(torch.zeros(1, 3))
        with self.assertRaisesRegex(ValueError, "floating"):
            analyze_latents(torch.zeros(2, 3, dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "finite"):
            values = torch.zeros(2, 3)
            values[0, 0] = float("nan")
            analyze_latents(values)
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            CollapseGate(minimum_active_dimension_fraction=1.1)

    def test_pca_is_deterministic_and_diagnostic_only(self) -> None:
        generator = torch.Generator().manual_seed(23)
        source = torch.randn(80, 3, generator=generator, dtype=torch.float64)
        source[:, 0] *= 4.0
        source[:, 1] *= 2.0
        first = deterministic_pca(source)
        second = deterministic_pca(source)
        torch.testing.assert_close(first.coordinates, second.coordinates)
        torch.testing.assert_close(first.components, second.components)

        permutation = torch.randperm(
            source.shape[0],
            generator=torch.Generator().manual_seed(29),
        )
        permuted = deterministic_pca(source[permutation])
        inverse = torch.argsort(permutation)
        torch.testing.assert_close(
            first.coordinates,
            permuted.coordinates[inverse],
            rtol=1e-10,
            atol=1e-10,
        )
        torch.testing.assert_close(first.components, permuted.components)
        self.assertEqual(first.authority, "diagnostic_only")
        self.assertEqual(first.source_space, "original_latent")
        self.assertFalse(first.valid_for_clustering)
        self.assertFalse(first.valid_for_training)


class Stage0ConditioningComparisonTest(unittest.TestCase):
    def _budget(self) -> FixedEvaluationBudget:
        return FixedEvaluationBudget(
            reset_group_ids=("reset-a", "reset-b", "reset-c", "reset-d"),
            rollout_seeds=(11, 12, 13, 14),
            max_steps_per_episode=100,
        )

    def _run(
        self,
        mode: ConditioningMode,
        outcomes: tuple[Stage0Outcome, ...],
        *,
        budget: FixedEvaluationBudget | None = None,
    ) -> PolicyEvaluationRun:
        return PolicyEvaluationRun(
            mode=mode,
            budget=self._budget() if budget is None else budget,
            outcomes=outcomes,
            completed_steps=(100, 80, 70, 100),
        )

    def test_three_modes_are_compared_under_one_budget(self) -> None:
        comparison = compare_conditioning_runs(
            (
                self._run(
                    ConditioningMode.NO_Z,
                    (
                        Stage0Outcome.FAILURE,
                        Stage0Outcome.SUCCESS,
                        Stage0Outcome.TIMEOUT,
                        Stage0Outcome.FAILURE,
                    ),
                ),
                self._run(
                    ConditioningMode.ORACLE_Z,
                    (
                        Stage0Outcome.SUCCESS,
                        Stage0Outcome.SUCCESS,
                        Stage0Outcome.SUCCESS,
                        Stage0Outcome.FAILURE,
                    ),
                ),
                self._run(
                    ConditioningMode.SELECTOR_Z,
                    (
                        Stage0Outcome.SUCCESS,
                        Stage0Outcome.SUCCESS,
                        Stage0Outcome.TIMEOUT,
                        Stage0Outcome.FAILURE,
                    ),
                ),
            )
        )
        self.assertEqual(comparison.budget.allocated_world_steps, 400)
        no_z = comparison.summary("no-z")
        oracle_z = comparison.summary("oracle-z")
        selector_z = comparison.summary("selector-z")
        self.assertEqual(no_z.success_count, 1)
        self.assertEqual(oracle_z.success_count, 3)
        self.assertEqual(selector_z.success_count, 2)
        self.assertAlmostEqual(oracle_z.success_rate_delta_from_no_z, 0.5)
        self.assertAlmostEqual(selector_z.success_rate_delta_from_no_z, 0.25)
        self.assertEqual(
            comparison.to_record()["comparison"],
            "no-z_vs_oracle-z_vs_selector-z",
        )

    def test_comparison_rejects_duplicate_or_mismatched_controls(self) -> None:
        outcomes = (Stage0Outcome.SUCCESS,) * 4
        no_z = self._run(ConditioningMode.NO_Z, outcomes)
        oracle_z = self._run(ConditioningMode.ORACLE_Z, outcomes)
        with self.assertRaisesRegex(ValueError, "unique"):
            compare_conditioning_runs((no_z, no_z, oracle_z))

        other_budget = FixedEvaluationBudget(
            reset_group_ids=("reset-a", "reset-b", "reset-c", "reset-d"),
            rollout_seeds=(11, 12, 13, 99),
            max_steps_per_episode=100,
        )
        selector_z = self._run(
            ConditioningMode.SELECTOR_Z,
            outcomes,
            budget=other_budget,
        )
        with self.assertRaisesRegex(ValueError, "identical"):
            compare_conditioning_runs((no_z, oracle_z, selector_z))

    def test_budget_and_run_shapes_are_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "align"):
            FixedEvaluationBudget(("a", "b"), (1,), 10)
        with self.assertRaisesRegex(ValueError, "one value"):
            PolicyEvaluationRun(
                mode=ConditioningMode.NO_Z,
                budget=self._budget(),
                outcomes=(Stage0Outcome.SUCCESS,),
                completed_steps=(1,),
            )
        with self.assertRaisesRegex(ValueError, r"\[1, 100\]"):
            PolicyEvaluationRun(
                mode=ConditioningMode.NO_Z,
                budget=self._budget(),
                outcomes=(Stage0Outcome.SUCCESS,) * 4,
                completed_steps=(100, 101, 50, 50),
            )


if __name__ == "__main__":
    unittest.main()
