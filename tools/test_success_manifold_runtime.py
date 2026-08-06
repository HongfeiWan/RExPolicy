"""Tests for checksum-pinned manifold rollout conditioning and resume."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class TestSuccessManifoldRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch

    def _components(self):
        from rexpolicy.manifold.config import SuccessManifoldConfig
        from rexpolicy.manifold.encoder import FutureTrajectoryEncoder
        from rexpolicy.manifold.selector import SuccessModeSelector

        config = SuccessManifoldConfig(
            enabled=True,
            state_dim=3,
            action_dim=2,
            model_dim=8,
            latent_dim=2,
            condition_dim=4,
            projection_dim=4,
            max_future_steps=4,
            transformer_layers=1,
            attention_heads=2,
            feedforward_dim=16,
            dropout=0.0,
            selector_hidden_dim=8,
            selector_layers=1,
            memory_capacity=4,
            novelty_threshold=0.0,
        )
        return config, FutureTrajectoryEncoder(config), SuccessModeSelector(config)

    def _policy(self):
        from rexpolicy.flywheel.groot_policy import GrootFlowDitPolicy

        policy = object.__new__(GrootFlowDitPolicy)
        policy.torch = self.torch
        policy.success_conditioning_enabled = True
        policy.model = SimpleNamespace(
            config=SimpleNamespace(backbone_embedding_dim=4)
        )
        return policy

    def _conditions(self):
        from rexpolicy.flywheel.groot_policy import CachedCondition

        torch = self.torch
        return [
            CachedCondition(
                backbone_features=torch.zeros(2, 4),
                backbone_attention_mask=torch.ones(2, dtype=torch.bool),
                image_mask=torch.tensor([True, False]),
                state=torch.tensor([[1.0, 2.0, 3.0]]),
                embodiment_id=1,
            )
            for _ in range(2)
        ]

    def test_bundle_condition_memory_and_resume(self) -> None:
        from rexpolicy.manifold.runtime import (
            SuccessManifoldRuntime,
            save_success_manifold_bundle,
        )

        config, encoder, selector = self._components()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifold.pt"
            digest = save_success_manifold_bundle(
                path,
                config=config,
                encoder=encoder,
                selector=selector,
            )
            runtime = SuccessManifoldRuntime.load(
                path,
                expected_sha256=digest,
                device="cpu",
            )
            generated = runtime.condition(
                policy=self._policy(),
                conditions=self._conditions(),
                deterministic_selector=True,
            )
            self.assertEqual(tuple(generated.latents.shape), (2, 2))
            self.assertEqual(tuple(generated.condition_tokens.shape), (2, 4))
            self.assertTrue(
                all(item.success_latent_token is not None for item in generated.conditions)
            )

            admissions = runtime.record_successes(
                sample_ids=["success-a"],
                latents=generated.latents[:1],
                metadata=[{"generation": 1}],
            )
            self.assertTrue(admissions[0].accepted)
            recalled = runtime.condition(
                policy=self._policy(),
                conditions=self._conditions(),
                memory_candidates=1,
                selector_temperature=4.0,
            )
            self.assertEqual(recalled.sources, ("latent_memory", "selector"))
            self.torch.testing.assert_close(
                recalled.latents[0],
                generated.latents[0],
            )

            restored = SuccessManifoldRuntime.load(
                path,
                expected_sha256=digest,
                device="cpu",
            )
            restored.load_state_dict(runtime.state_dict())
            self.assertEqual(restored.memory.state_dict(), runtime.memory.state_dict())

    def test_selector_temperature_is_threaded_without_changing_t1(self) -> None:
        from rexpolicy.manifold.runtime import (
            SuccessManifoldRuntime,
            save_success_manifold_bundle,
        )

        config, encoder, selector = self._components()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifold.pt"
            digest = save_success_manifold_bundle(
                path,
                config=config,
                encoder=encoder,
                selector=selector,
            )
            runtime = SuccessManifoldRuntime.load(
                path,
                expected_sha256=digest,
                device="cpu",
            )
            conditions = self._conditions()
            states = self.torch.stack([item.state for item in conditions]).flatten(1)
            with self.torch.inference_mode():
                mean = runtime.selector.predict(states).cpu()
            baseline = runtime.condition(
                policy=self._policy(),
                conditions=conditions,
                selector_temperature=1.0,
                generator=self.torch.Generator().manual_seed(41),
            )
            hot = runtime.condition(
                policy=self._policy(),
                conditions=conditions,
                selector_temperature=4.0,
                generator=self.torch.Generator().manual_seed(41),
            )
            self.torch.testing.assert_close(
                hot.latents - mean,
                2.0 * (baseline.latents - mean),
            )
            with self.assertRaisesRegex(ValueError, "temperature"):
                runtime.condition(
                    policy=self._policy(),
                    conditions=conditions,
                    selector_temperature=0.0,
                )

    def test_active_sampling_prefers_occupancy_novel_candidates(self) -> None:
        from rexpolicy.manifold.exploration import (
            ExplorationPolicy,
            normalize_occupancy_novelty,
        )
        from rexpolicy.manifold.occupancy import (
            SuccessOccupancyConfig,
            SuccessOccupancyIndex,
        )
        from rexpolicy.manifold.runtime import (
            SuccessManifoldRuntime,
            save_success_manifold_bundle,
        )

        config, encoder, selector = self._components()
        occupancy = SuccessOccupancyIndex(
            SuccessOccupancyConfig(
                enabled=True,
                latent_dim=config.latent_dim,
                cluster_radius=0.25,
                density_bandwidth=0.25,
            )
        )
        occupancy.observe_success(
            {
                "sample_id": "known-mode",
                "latent": [0.0, 0.0],
                "generation": 0,
                "rank": 0,
                "episode_id": "baseline",
                "source_sha256": "a" * 64,
            }
        )
        exploration = ExplorationPolicy(
            enabled=True,
            novelty_coefficient_start=10.0,
            novelty_coefficient_end=10.0,
            active_candidates=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifold.pt"
            digest = save_success_manifold_bundle(
                path,
                config=config,
                encoder=encoder,
                selector=selector,
            )
            runtime = SuccessManifoldRuntime.load(
                path,
                expected_sha256=digest,
                device="cpu",
            )
            conditions = self._conditions()
            states = self.torch.stack([item.state for item in conditions]).flatten(1)
            with self.torch.inference_mode():
                distribution = runtime.selector.distribution(states)
                candidates = distribution.sample(
                    (exploration.active_candidates,),
                    generator=self.torch.Generator().manual_seed(53),
                )
                standardized = (
                    candidates - distribution.mean.unsqueeze(0)
                ) / distribution.std.unsqueeze(0)
                log_probability = -0.5 * standardized.square().sum(dim=-1)
            expected_indices = []
            for batch_index in range(len(conditions)):
                scores = []
                for candidate_index in range(exploration.active_candidates):
                    query = occupancy.query(
                        candidates[candidate_index, batch_index].tolist()
                    )
                    novelty, _bootstrap = normalize_occupancy_novelty(
                        query,
                        policy=exploration,
                    )
                    scores.append(
                        float(log_probability[candidate_index, batch_index])
                        / config.latent_dim
                        + 10.0 * novelty
                    )
                expected_indices.append(
                    max(range(len(scores)), key=lambda index: (scores[index], -index))
                )
            selected = runtime.condition(
                policy=self._policy(),
                conditions=conditions,
                exploration_policy=exploration,
                occupancy=occupancy,
                generation=3,
                generator=self.torch.Generator().manual_seed(53),
            )
            self.assertEqual(tuple(selected.latents.shape), (2, config.latent_dim))
            self.assertEqual(len(selected.selection_metadata), 2)
            for batch_index, metadata in enumerate(selected.selection_metadata):
                self.assertEqual(
                    metadata["authority"],
                    "active_manifold_sampling",
                )
                self.assertGreaterEqual(metadata["candidate_index"], 0)
                self.assertLess(metadata["candidate_index"], 4)
                self.assertEqual(metadata["candidate_count"], 4)
                self.assertEqual(
                    metadata["candidate_index"],
                    expected_indices[batch_index],
                )
                self.torch.testing.assert_close(
                    selected.latents[batch_index],
                    candidates[expected_indices[batch_index], batch_index],
                )

            with self.assertRaisesRegex(ValueError, "requires occupancy"):
                runtime.condition(
                    policy=self._policy(),
                    conditions=self._conditions(),
                    exploration_policy=exploration,
                    generation=3,
                )

    def test_bundle_checksum_and_condition_width_fail_closed(self) -> None:
        from rexpolicy.manifold.runtime import (
            SuccessManifoldRuntime,
            save_success_manifold_bundle,
        )

        config, encoder, selector = self._components()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifold.pt"
            digest = save_success_manifold_bundle(
                path,
                config=config,
                encoder=encoder,
                selector=selector,
            )
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                SuccessManifoldRuntime.load(
                    path,
                    expected_sha256="0" * 64,
                    device="cpu",
                )
            runtime = SuccessManifoldRuntime.load(
                path,
                expected_sha256=digest,
                device="cpu",
            )
            policy = self._policy()
            policy.model.config.backbone_embedding_dim = 5
            with self.assertRaisesRegex(ValueError, "backbone width"):
                runtime.condition(policy=policy, conditions=self._conditions())

    def test_runner_feature_flag_is_paired_and_bounded(self) -> None:
        from rexpolicy.flywheel.task_spec import get_task_spec
        from tools.run_flywheel_ddp import _validate_args, create_parser

        parser = create_parser()
        missing_hash = parser.parse_args(
            ["--validate-only", "--success-manifold-bundle", "/tmp/model.pt"]
        )
        with self.assertRaisesRegex(ValueError, "required together"):
            _validate_args(missing_hash, get_task_spec(missing_hash.task_id))

        enabled = parser.parse_args(
            [
                "--validate-only",
                "--success-manifold-bundle",
                "/tmp/model.pt",
                "--expected-success-manifold-bundle-sha256",
                "a" * 64,
                "--manifold-memory-candidates",
                "1",
            ]
        )
        _validate_args(enabled, get_task_spec(enabled.task_id))
        enabled.manifold_memory_candidates = enabled.candidates_per_state + 1
        with self.assertRaisesRegex(ValueError, "between zero"):
            _validate_args(enabled, get_task_spec(enabled.task_id))


if __name__ == "__main__":
    unittest.main()
