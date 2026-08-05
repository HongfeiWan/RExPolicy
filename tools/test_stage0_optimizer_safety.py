"""CPU tests for grouped Stage 0 optimizer safety checks."""

from __future__ import annotations

import unittest

import torch

from rexpolicy.stage0.trainers.common import (
    _non_finite_names,
    finish_optimizer_step,
    materialize_finite_scalars,
)


class _PoisonOptimizer(torch.optim.SGD):
    def step(self, closure=None):  # type: ignore[no-untyped-def,override]
        result = super().step(closure)
        with torch.no_grad():
            self.param_groups[0]["params"][0].fill_(float("nan"))
        return result


class Stage0OptimizerSafetyTest(unittest.TestCase):
    def test_grouped_finite_check_reports_every_bad_tensor(self) -> None:
        bad = _non_finite_names(
            (
                ("good", torch.tensor([1.0, 2.0])),
                ("nan", torch.tensor([float("nan")])),
                ("inf", torch.tensor([float("inf")])),
            )
        )
        self.assertEqual(bad, ("nan", "inf"))

    def test_scalar_materialization_is_finite_and_json_safe(self) -> None:
        values = materialize_finite_scalars(
            loss=torch.tensor(1.25),
            gradient_norm=torch.tensor(0.5),
        )
        self.assertEqual(values, {"loss": 1.25, "gradient_norm": 0.5})
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            materialize_finite_scalars(loss=torch.tensor(float("nan")))

    def test_non_finite_loss_is_rejected_before_optimizer_step(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD((parameter,), lr=0.1)
        loss = parameter.sum() * torch.tensor(float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "training loss"):
            finish_optimizer_step(
                loss=loss,
                module=torch.nn.ParameterList((parameter,)),
                optimizer=optimizer,
                gradient_clip_norm=1.0,
            )
        torch.testing.assert_close(parameter.detach(), torch.tensor([1.0]))

    def test_optimizer_parameter_corruption_is_rejected(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        module = torch.nn.ParameterList((parameter,))
        optimizer = _PoisonOptimizer(module.parameters(), lr=0.1)
        with self.assertRaisesRegex(FloatingPointError, "non-finite parameters"):
            finish_optimizer_step(
                loss=parameter.square().sum(),
                module=module,
                optimizer=optimizer,
                gradient_clip_norm=1.0,
            )


if __name__ == "__main__":
    unittest.main()
