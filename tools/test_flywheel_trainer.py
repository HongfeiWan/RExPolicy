"""Unit tests for the flywheel's FP32 and globally weighted update core."""

from __future__ import annotations

import multiprocessing
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

from rexpolicy.flywheel.distributed import DistributedContext
from rexpolicy.flywheel.experience import DIRECT_SUCCESS_ROLE, TrainingSample
from rexpolicy.flywheel.trainer import DitDdpTrainer, build_update_schedule


class _ToyActionHead(torch.nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.model = torch.nn.Linear(1, 1, bias=False, dtype=dtype)
        with torch.no_grad():
            self.model.weight.zero_()

    def forward(self, backbone_output: Any, action_input: Any) -> dict[str, Any]:
        del backbone_output
        prediction = self.model(action_input["x"])
        loss = (prediction - action_input["target"]).square().reshape(-1, 1, 1)
        return {
            "action_loss": loss,
            "weighted_action_mask": torch.ones_like(loss),
        }


def _toy_sample(
    target: float,
    *,
    weight: float = 1.0,
    sample_id: str | None = None,
    success: bool = False,
    source: str = "current",
) -> TrainingSample:
    sample = TrainingSample(
        backbone_features=torch.zeros(1, 1),
        backbone_attention_mask=torch.ones(1, dtype=torch.bool),
        image_mask=None,
        state=torch.ones(1, 1),
        embodiment_id=0,
        action=torch.tensor([[target]], dtype=torch.float32),
        action_mask=torch.ones(1, 1),
        valid_steps=1,
        sample_weight=weight,
    )
    sample.sample_id = sample_id or f"target-{target}-weight-{weight}"
    if success:
        sample.add_success_role(DIRECT_SUCCESS_ROLE)
    sample.source = source
    return sample


def _toy_collate(
    samples: list[TrainingSample],
    *,
    device: Any,
    dtype: Any,
) -> tuple[None, dict[str, torch.Tensor]]:
    return None, {
        "x": torch.stack([sample.state.reshape(1) for sample in samples])
        .to(device=device, dtype=dtype),
        "target": torch.stack([sample.action.reshape(1) for sample in samples])
        .to(device=device, dtype=dtype),
    }


def _context(*, rank: int = 0, world_size: int = 1, initialized: bool = False):
    return DistributedContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        device=torch.device("cpu"),
        initialized=initialized,
    )


def _ddp_weighted_worker(
    rank: int,
    init_file: str,
    result_queue: multiprocessing.Queue,
) -> None:
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    context = _context(rank=rank, world_size=2, initialized=True)
    model = _ToyActionHead()
    trainer = DitDdpTrainer(
        action_head=model,
        context=context,
        learning_rate=0.01,
        weight_decay=0.0,
        gradient_clip_norm=0.0,
        parameter_probe_size=1,
        collate_fn=_toy_collate,
    )
    local_samples = (
        [_toy_sample(1.0, sample_id="rank0")]
        if rank == 0
        else [
            _toy_sample(3.0, weight=2.0, sample_id="rank1-a"),
            _toy_sample(5.0, sample_id="rank1-b"),
        ]
    )
    metrics = trainer.update(
        local_samples,
        optimizer_steps=1,
        batch_size=1,
        gradient_accumulation=2,
        seed=17,
        dtype=torch.bfloat16,
    )
    result_queue.put(
        (
            rank,
            float(model.model.weight.detach().item()),
            metrics.mean_loss,
            metrics.global_weight,
            metrics.global_dummy_slots,
            metrics.current_coverage,
        )
    )
    context.close()


def _global_reference_samples() -> list[TrainingSample]:
    return [
        _toy_sample(-2.0, weight=0.5, sample_id="negative"),
        _toy_sample(1.0, weight=3.0, sample_id="near"),
        _toy_sample(4.0, weight=1.25, sample_id="far"),
    ]


def _ddp_global_reference_worker(
    rank: int,
    init_file: str,
    result_queue: multiprocessing.Queue,
) -> None:
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    context = _context(rank=rank, world_size=2, initialized=True)
    model = _ToyActionHead()
    with torch.no_grad():
        model.model.weight.fill_(0.25)
    trainer = DitDdpTrainer(
        action_head=model,
        context=context,
        learning_rate=0.03,
        weight_decay=0.07,
        gradient_clip_norm=0.0,
        parameter_probe_size=1,
        collate_fn=_toy_collate,
    )
    gradients: list[float] = []

    def capture_gradient(
        optimizer: torch.optim.Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del optimizer, args, kwargs
        gradient = model.model.weight.grad
        assert gradient is not None
        gradients.append(float(gradient.detach().item()))

    trainer.optimizer.register_step_pre_hook(capture_gradient)
    local_samples = [] if rank == 0 else _global_reference_samples()
    dummy_sample = _toy_sample(
        0.0,
        sample_id=f"rank-{rank}-dummy",
        source="dummy",
    )
    metrics = trainer.update(
        local_samples,
        optimizer_steps=0,
        batch_size=1,
        gradient_accumulation=2,
        seed=29,
        dtype=torch.float32,
        dummy_sample=dummy_sample,
    )
    result_queue.put(
        (
            rank,
            float(model.model.weight.detach().item()),
            gradients,
            metrics.mean_loss,
            metrics.global_weight,
            metrics.global_dummy_slots,
            metrics.optimizer_steps,
            metrics.current_coverage,
        )
    )
    context.close()


def _single_process_global_weighted_reference() -> tuple[
    float,
    list[float],
    float,
    float,
]:
    model = _ToyActionHead()
    with torch.no_grad():
        model.model.weight.fill_(0.25)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.03,
        weight_decay=0.07,
    )
    dummy_samples = [
        _toy_sample(0.0, sample_id=f"rank-{rank}-dummy", source="dummy")
        for rank in range(2)
    ]
    rank_samples = [[], _global_reference_samples()]
    schedules = [
        build_update_schedule(
            samples,
            requested_optimizer_steps=0,
            batch_size=1,
            gradient_accumulation=2,
            seed=29,
            rank=rank,
            synchronized_item_count=3,
            dummy_sample=dummy_samples[rank],
        )
        for rank, samples in enumerate(rank_samples)
    ]

    gradients: list[float] = []
    weighted_loss_total = 0.0
    weight_total = 0.0
    for step_index in range(schedules[0].optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        rank_steps = [schedule.steps[step_index] for schedule in schedules]
        window_weight = sum(
            item.weight
            for rank_step in rank_steps
            for microbatch in rank_step
            for item in microbatch
            if not item.is_dummy
        )
        objective = torch.zeros((), dtype=torch.float32)
        for rank_step in rank_steps:
            for scheduled_batch in rank_step:
                batch = [item.sample for item in scheduled_batch]
                backbone_output, action_input = _toy_collate(
                    batch,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                output = model(backbone_output, action_input)
                action_loss = output["action_loss"].float()
                action_mask = output["weighted_action_mask"].float()
                per_sample_loss = action_loss.sum(dim=(1, 2)) / (
                    action_mask.sum(dim=(1, 2)) + 1.0e-6
                )
                sample_weight = torch.tensor(
                    [
                        0.0 if item.is_dummy else item.weight
                        for item in scheduled_batch
                    ],
                    dtype=torch.float32,
                )
                numerator = (per_sample_loss * sample_weight).sum()
                objective = objective + numerator / window_weight
                weighted_loss_total += float(numerator.detach().item())
        objective.backward()
        gradient = model.model.weight.grad
        assert gradient is not None
        gradients.append(float(gradient.detach().item()))
        optimizer.step()
        weight_total += window_weight

    return (
        float(model.model.weight.detach().item()),
        gradients,
        weighted_loss_total / weight_total,
        weight_total,
    )


class TestUpdateSchedule(unittest.TestCase):
    def test_success_current_history_order_and_coverage(self) -> None:
        success = _toy_sample(1.0, sample_id="success", success=True)
        current = _toy_sample(2.0, sample_id="current")
        history = _toy_sample(3.0, sample_id="history", source="history")
        schedule = build_update_schedule(
            [current, success],
            historical_samples=[history],
            requested_optimizer_steps=0,
            batch_size=1,
            gradient_accumulation=1,
            seed=1,
            rank=0,
            synchronized_item_count=3,
        )
        categories = [
            item.category
            for step in schedule.steps
            for microbatch in step
            for item in microbatch
        ]
        self.assertEqual(categories, ["success", "current", "history"])
        self.assertEqual(schedule.optimizer_steps, 3)
        self.assertEqual(schedule.current_coverage, 1.0)
        self.assertEqual(schedule.local_current_exposed, 2)

    def test_short_rank_uses_dummy_in_coverage_epoch(self) -> None:
        only = _toy_sample(1.0, sample_id="only")
        schedule = build_update_schedule(
            [only],
            requested_optimizer_steps=0,
            batch_size=1,
            gradient_accumulation=2,
            seed=1,
            rank=0,
            synchronized_item_count=3,
        )
        flat = [
            item
            for step in schedule.steps
            for microbatch in step
            for item in microbatch
        ]
        self.assertEqual(schedule.optimizer_steps, 2)
        self.assertEqual(sum(not item.is_dummy for item in flat), 1)
        self.assertEqual(sum(item.is_dummy for item in flat), 3)
        self.assertEqual(sum(item.weight for item in flat), 1.0)

    def test_extra_requested_steps_sample_only_after_coverage(self) -> None:
        samples = [
            _toy_sample(1.0, sample_id="one"),
            _toy_sample(2.0, sample_id="two"),
        ]
        schedule = build_update_schedule(
            samples,
            requested_optimizer_steps=3,
            batch_size=1,
            gradient_accumulation=1,
            seed=4,
            rank=0,
            synchronized_item_count=2,
        )
        flat = [
            item
            for step in schedule.steps
            for microbatch in step
            for item in microbatch
        ]
        self.assertFalse(any(item.is_dummy for item in flat))
        self.assertTrue(flat[-1].category.endswith("_replacement"))
        self.assertEqual({_id(item.sample) for item in flat[:2]}, {"one", "two"})


def _id(sample: TrainingSample) -> str:
    return str(sample.sample_id)


class TestFp32Trainer(unittest.TestCase):
    def test_rejects_non_fp32_trainable_parameters(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "non-FP32"):
            DitDdpTrainer(
                action_head=_ToyActionHead(dtype=torch.bfloat16),
                context=_context(),
                learning_rate=1.0e-5,
                weight_decay=0.0,
                gradient_clip_norm=1.0,
                collate_fn=_toy_collate,
            )

    def test_bf16_autocast_updates_fp32_and_offloads_adam(self) -> None:
        model = _ToyActionHead()
        trainer = DitDdpTrainer(
            action_head=model,
            context=_context(),
            learning_rate=1.0e-5,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            optimizer_state_offload=True,
            parameter_probe_size=1,
            collate_fn=_toy_collate,
        )
        metrics = trainer.update(
            [
                _toy_sample(1.0, weight=1.0, sample_id="one"),
                _toy_sample(3.0, weight=3.0, sample_id="three"),
            ],
            optimizer_steps=1,
            batch_size=1,
            gradient_accumulation=2,
            seed=3,
            dtype=torch.bfloat16,
        )
        self.assertEqual(model.model.weight.dtype, torch.float32)
        self.assertNotEqual(float(model.model.weight.detach().item()), 0.0)
        self.assertGreater(metrics.update_max_abs, 0.0)
        self.assertGreater(metrics.gradient_norm, 0.0)
        self.assertEqual(metrics.current_coverage, 1.0)
        self.assertAlmostEqual(metrics.mean_loss, 7.0, places=4)
        self.assertGreaterEqual(metrics.optimizer_restore_seconds, 0.0)
        self.assertGreaterEqual(metrics.optimizer_offload_seconds, 0.0)
        self.assertEqual(metrics.optimizer_state_device, "cpu")
        for state in trainer.optimizer.state.values():
            self.assertEqual(state["exp_avg"].dtype, torch.float32)
            self.assertEqual(state["exp_avg"].device.type, "cpu")
            self.assertEqual(state["exp_avg_sq"].device.type, "cpu")

        checkpoint = trainer.state_dict()
        restored = DitDdpTrainer(
            action_head=_ToyActionHead(),
            context=_context(),
            learning_rate=1.0e-5,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            optimizer_state_offload=True,
            parameter_probe_size=1,
            collate_fn=_toy_collate,
        )
        restored.load_state_dict(checkpoint)
        self.assertEqual(restored.global_optimizer_step, 1)
        for state in restored.optimizer.state.values():
            self.assertEqual(state["exp_avg"].device.type, "cpu")
            self.assertEqual(state["exp_avg"].dtype, torch.float32)
        restored.update(
            [_toy_sample(2.0, sample_id="resume")],
            optimizer_steps=1,
            batch_size=1,
            gradient_accumulation=1,
            seed=5,
            dtype=torch.bfloat16,
        )
        self.assertEqual(restored.global_optimizer_step, 2)
        for state in restored.optimizer.state.values():
            self.assertEqual(state["exp_avg"].device.type, "cpu")

    def test_zero_learning_rate_trips_effective_update_gate(self) -> None:
        trainer = DitDdpTrainer(
            action_head=_ToyActionHead(),
            context=_context(),
            learning_rate=0.0,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            parameter_probe_size=1,
            collate_fn=_toy_collate,
        )
        with self.assertRaisesRegex(RuntimeError, "no effective FP32"):
            trainer.update(
                [_toy_sample(1.0)],
                optimizer_steps=1,
                batch_size=1,
                gradient_accumulation=1,
                seed=1,
                dtype=torch.bfloat16,
            )

    def test_two_rank_weighted_objective_handles_rank_imbalance(self) -> None:
        multiprocessing_context = multiprocessing.get_context("spawn")
        result_queue = multiprocessing_context.Queue()
        with tempfile.TemporaryDirectory() as directory:
            init_file = str(Path(directory) / "gloo-init")
            processes = [
                multiprocessing_context.Process(
                    target=_ddp_weighted_worker,
                    args=(rank, init_file, result_queue),
                )
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            results = [result_queue.get(timeout=30) for _ in processes]
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)

        results.sort()
        self.assertAlmostEqual(results[0][1], results[1][1], places=7)
        self.assertAlmostEqual(results[0][2], 11.0, places=4)
        self.assertAlmostEqual(results[1][2], 11.0, places=4)
        self.assertEqual(results[0][3], 4.0)
        self.assertEqual(results[0][4], 1)
        self.assertEqual(results[0][5], 1.0)

    def test_two_rank_gradients_match_single_process_global_objective(self) -> None:
        reference = _single_process_global_weighted_reference()
        multiprocessing_context = multiprocessing.get_context("spawn")
        result_queue = multiprocessing_context.Queue()
        with tempfile.TemporaryDirectory() as directory:
            init_file = str(Path(directory) / "gloo-reference-init")
            processes = [
                multiprocessing_context.Process(
                    target=_ddp_global_reference_worker,
                    args=(rank, init_file, result_queue),
                )
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            results = [result_queue.get(timeout=30) for _ in processes]
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)

        results.sort()
        reference_weight, reference_gradients, reference_loss, reference_total_weight = (
            reference
        )
        self.assertEqual(len(reference_gradients), 2)
        self.assertNotAlmostEqual(
            abs(reference_gradients[0]),
            abs(reference_gradients[1]),
            places=3,
        )
        for result in results:
            self.assertAlmostEqual(result[1], reference_weight, places=7)
            self.assertEqual(len(result[2]), len(reference_gradients))
            for actual, expected in zip(result[2], reference_gradients):
                self.assertAlmostEqual(actual, expected, places=6)
            self.assertAlmostEqual(result[3], reference_loss, places=6)
            self.assertAlmostEqual(result[4], reference_total_weight, places=7)
            self.assertEqual(result[5], 5)
            self.assertEqual(result[6], 2)
            self.assertEqual(result[7], 1.0)


if __name__ == "__main__":
    unittest.main()
