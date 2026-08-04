"""Focused tests for the optional RExPolicy v2 success manifold core."""

from __future__ import annotations

import multiprocessing
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from rexpolicy.manifold.config import SuccessManifoldConfig
from rexpolicy.manifold.distributed import all_gather_tensor
from rexpolicy.manifold.encoder import FutureTrajectoryEncoder
from rexpolicy.manifold.losses import (
    multi_positive_info_nce_loss,
    success_manifold_losses,
    variance_covariance_diversity_loss,
)
from rexpolicy.manifold.memory import LatentMemory
from rexpolicy.manifold.selector import SuccessModeSelector


def _all_gather_worker(
    rank: int,
    init_file: str,
    result_queue: multiprocessing.Queue,
) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    local = torch.full(
        (rank + 1, 2),
        float(rank + 1),
        requires_grad=True,
    )
    gathered = all_gather_tensor(local)
    gathered.sum().backward()
    result_queue.put(
        (
            rank,
            gathered.detach().tolist(),
            local.grad.detach().tolist(),
        )
    )
    dist.destroy_process_group()


def _config() -> SuccessManifoldConfig:
    return SuccessManifoldConfig(
        enabled=True,
        state_dim=4,
        action_dim=3,
        visual_dim=2,
        model_dim=16,
        latent_dim=6,
        condition_dim=12,
        projection_dim=8,
        max_future_steps=5,
        transformer_layers=2,
        attention_heads=4,
        feedforward_dim=32,
        dropout=0.0,
        reconstruction_mask_probability=0.5,
        selector_hidden_dim=12,
        selector_layers=2,
        memory_capacity=3,
        novelty_threshold=0.5,
    )


class SuccessManifoldConfigTest(unittest.TestCase):
    def test_default_is_disabled_and_strict(self) -> None:
        config = SuccessManifoldConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.state_dim, 0)
        self.assertEqual(
            SuccessManifoldConfig.from_mapping(config.to_record()),
            config,
        )
        with self.assertRaisesRegex(ValueError, "Unknown"):
            SuccessManifoldConfig.from_mapping({"enabledd": True})
        with self.assertRaisesRegex(ValueError, "state_dim"):
            SuccessManifoldConfig(enabled=True)
        with self.assertRaisesRegex(ValueError, "divisible"):
            SuccessManifoldConfig(model_dim=10, attention_heads=4)

    def test_package_root_does_not_import_torch(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        command = (
            "import sys; "
            f"sys.path.insert(0, {str(repository)!r}); "
            "import rexpolicy.manifold; "
            "assert 'torch' not in sys.modules"
        )
        subprocess.run(
            [sys.executable, "-S", "-c", command],
            check=True,
            cwd=repository,
        )


class FutureTrajectoryEncoderTest(unittest.TestCase):
    def _batch(self) -> tuple[torch.Tensor, ...]:
        torch.manual_seed(11)
        states = torch.randn(4, 4, 4)
        actions = torch.randn(4, 4, 3)
        visuals = torch.randn(4, 4, 2)
        padding = torch.tensor(
            [
                [False, False, False, False],
                [False, False, False, True],
                [False, False, True, True],
                [False, False, False, False],
            ]
        )
        masked = torch.tensor(
            [
                [True, False, False, False],
                [False, True, False, False],
                [True, False, False, False],
                [False, False, True, False],
            ]
        )
        return states, actions, visuals, padding, masked

    def test_encoder_decoder_losses_and_gradients(self) -> None:
        config = _config()
        model = FutureTrajectoryEncoder(config)
        states, actions, visuals, padding, masked = self._batch()
        output = model(
            states,
            actions,
            visual_features=visuals,
            padding_mask=padding,
            reconstruction_mask=masked,
        )
        self.assertEqual(tuple(output.latent.shape), (4, 6))
        self.assertEqual(tuple(output.condition_token.shape), (4, 1, 12))
        self.assertEqual(tuple(output.projection.shape), (4, 8))
        self.assertEqual(tuple(output.encoded_steps.shape), (4, 4, 16))
        self.assertEqual(tuple(output.reconstruction.states.shape), (4, 4, 4))
        self.assertEqual(tuple(output.reconstruction.actions.shape), (4, 4, 3))
        self.assertEqual(
            tuple(output.reconstruction.visual_features.shape),
            (4, 4, 2),
        )
        self.assertTrue(
            torch.equal(
                output.encoded_steps.masked_select(padding.unsqueeze(-1)),
                torch.zeros(padding.sum() * config.model_dim),
            )
        )

        losses = success_manifold_losses(
            output,
            states,
            actions,
            ["trajectory-a", "trajectory-a", "trajectory-b", "trajectory-b"],
            config=config,
            visual_features=visuals,
        )
        for value in (
            losses.reconstruction,
            losses.info_nce,
            losses.variance,
            losses.covariance,
            losses.total,
        ):
            self.assertTrue(torch.isfinite(value))
        losses.total.backward()
        self.assertIsNotNone(model.state_embedding.weight.grad)
        self.assertIsNotNone(model.decoder.output[-1].weight.grad)

    def test_padding_validation_mask_sampling_and_state_dict(self) -> None:
        config = _config()
        model = FutureTrajectoryEncoder(config).eval()
        states, actions, visuals, padding, masked = self._batch()
        generator_a = torch.Generator().manual_seed(9)
        generator_b = torch.Generator().manual_seed(9)
        self.assertTrue(
            torch.equal(
                model.sample_reconstruction_mask(padding, generator=generator_a),
                model.sample_reconstruction_mask(padding, generator=generator_b),
            )
        )
        with torch.no_grad():
            expected = model(
                states,
                actions,
                visual_features=visuals,
                padding_mask=padding,
                reconstruction_mask=masked,
            )
            restored = FutureTrajectoryEncoder(config).eval()
            restored.load_state_dict(model.state_dict())
            actual = restored(
                states,
                actions,
                visual_features=visuals,
                padding_mask=padding,
                reconstruction_mask=masked,
            )
        torch.testing.assert_close(actual.latent, expected.latent)
        torch.testing.assert_close(actual.condition_token, expected.condition_token)
        invalid_padding = padding.clone()
        invalid_padding[0] = True
        with self.assertRaisesRegex(ValueError, "valid step"):
            model(
                states,
                actions,
                visual_features=visuals,
                padding_mask=invalid_padding,
            )


class ManifoldLossTest(unittest.TestCase):
    def test_multi_positive_info_nce_rewards_matching_trajectories(self) -> None:
        embeddings = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
            requires_grad=True,
        )
        matching = multi_positive_info_nce_loss(
            embeddings,
            ["a", "a", "b", "b"],
            temperature=0.2,
        )
        mismatched = multi_positive_info_nce_loss(
            embeddings,
            ["a", "b", "a", "b"],
            temperature=0.2,
        )
        self.assertLess(float(matching), float(mismatched))
        matching.backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all())

    def test_diversity_and_non_distributed_gather(self) -> None:
        tensor = torch.randn(3, 4, requires_grad=True)
        self.assertIs(all_gather_tensor(tensor), tensor)
        collapsed = variance_covariance_diversity_loss(torch.zeros(4, 3))
        varied = variance_covariance_diversity_loss(
            torch.tensor(
                [
                    [-2.0, -2.0, 2.0],
                    [-2.0, 2.0, -2.0],
                    [2.0, -2.0, -2.0],
                    [2.0, 2.0, 2.0],
                ]
            )
        )
        self.assertGreater(float(collapsed.variance), float(varied.variance))
        self.assertAlmostEqual(float(varied.covariance), 0.0, places=6)

    def test_uneven_two_rank_all_gather_preserves_gradients(self) -> None:
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        with tempfile.TemporaryDirectory() as directory:
            init_file = str(Path(directory) / "manifold-gloo-init")
            processes = [
                context.Process(
                    target=_all_gather_worker,
                    args=(rank, init_file, queue),
                )
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            results = [queue.get(timeout=30) for _ in processes]
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)
        results.sort()
        expected = [[1.0, 1.0], [2.0, 2.0], [2.0, 2.0]]
        self.assertEqual(results[0][1], expected)
        self.assertEqual(results[1][1], expected)
        self.assertEqual(results[0][2], [[2.0, 2.0]])
        self.assertEqual(results[1][2], [[2.0, 2.0], [2.0, 2.0]])


class SuccessModeSelectorTest(unittest.TestCase):
    def test_gaussian_prediction_sampling_and_checkpoint(self) -> None:
        config = _config()
        selector = SuccessModeSelector(config)
        states = torch.randn(3, config.state_dim)
        distribution = selector(states)
        self.assertEqual(tuple(distribution.mean.shape), (3, config.latent_dim))
        self.assertEqual(tuple(distribution.log_std.shape), (3, config.latent_dim))
        torch.testing.assert_close(selector.predict(states), distribution.mean)
        generator_a = torch.Generator().manual_seed(17)
        generator_b = torch.Generator().manual_seed(17)
        first = selector.sample(states, generator=generator_a)
        second = selector.sample(states, generator=generator_b)
        torch.testing.assert_close(first, second)
        self.assertFalse(first.requires_grad)
        self.assertEqual(
            tuple(selector.sample(states, sample_shape=(2,)).shape),
            (2, 3, config.latent_dim),
        )

        restored = SuccessModeSelector(config)
        restored.load_state_dict(selector.state_dict())
        torch.testing.assert_close(restored.predict(states), selector.predict(states))


class LatentMemoryTest(unittest.TestCase):
    def test_novelty_sampling_rebase_and_checkpoint(self) -> None:
        memory = LatentMemory(
            latent_dim=2,
            capacity=3,
            novelty_threshold=0.5,
            encoder_fingerprint="encoder/v1",
        )
        self.assertTrue(memory.add("a", [0.0, 0.0], metadata={"index": 0}).accepted)
        rejected = memory.add("near-a", [0.1, 0.1])
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, "below_novelty_threshold")
        self.assertTrue(memory.add("b", [1.0, 0.0]).accepted)
        self.assertTrue(memory.add("c", [0.0, 1.0]).accepted)
        admission = memory.add("d", [2.0, 2.0])
        self.assertEqual(admission.evicted_sample_id, "a")
        self.assertEqual(memory.nearest([0.9, 0.0])[0].entry.sample_id, "b")
        self.assertEqual([item.sample_id for item in memory.sample(2)], ["b", "c"])

        before_failed_rebase = memory.state_dict()
        with self.assertRaisesRegex(ValueError, "do not match"):
            memory.rebase({"b": [2.0, 0.0]}, encoder_fingerprint="encoder/v2")
        self.assertEqual(memory.state_dict(), before_failed_rebase)
        memory.rebase(
            {"b": [2.0, 0.0], "c": [0.0, 2.0], "d": [3.0, 3.0]},
            encoder_fingerprint="encoder/v2",
        )
        self.assertEqual(memory.rebase_state["rebase_revision"], 1)
        self.assertEqual(memory.rebase_state["encoder_fingerprint"], "encoder/v2")
        state = memory.state_dict()
        restored = LatentMemory.from_state_dict(state)
        self.assertEqual(restored.state_dict(), state)
        self.assertEqual(
            [item.sample_id for item in restored.sample(2)],
            [item.sample_id for item in memory.sample(2)],
        )


if __name__ == "__main__":
    unittest.main()
