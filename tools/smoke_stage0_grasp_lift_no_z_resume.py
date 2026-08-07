#!/usr/bin/env python3
"""Bitwise CUDA checkpoint/resume smoke for Grasp-Lift no-z training.

The smoke deliberately uses the immutable formal training data and optimizer
contract, but runs only four optimizer steps.  It never materializes formal
validation or locked-test tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_CPU_THREADS = 16
_INTEROP_THREADS = 4
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_TF32_OVERRIDE = "0"
_TOTAL_STEPS = 4
_CHECKPOINT_STEP = 2
_MAX_SMOKE_STEPS = 32


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/stage0/grasp_lift_no_z_bc.json",
    )
    parser.add_argument("--pilot-directory", required=True)
    parser.add_argument("--training-artifact", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--training-seed", required=True, type=int)
    return parser.parse_args()


def _validate_smoke_schedule(
    total_steps: Any,
    checkpoint_step: Any,
) -> tuple[int, int]:
    """Return a bounded strict split schedule suitable only for a smoke."""

    if (
        isinstance(total_steps, bool)
        or not isinstance(total_steps, int)
        or total_steps < 2
        or total_steps > _MAX_SMOKE_STEPS
    ):
        raise ValueError(
            f"total_steps must be an integer in [2, {_MAX_SMOKE_STEPS}]"
        )
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or not 1 <= checkpoint_step < total_steps
    ):
        raise ValueError("checkpoint_step must be an integer in [1, total_steps)")
    return total_steps, checkpoint_step


def _configure_preimport_runtime() -> tuple[int, ...]:
    """Set the formal runner's environment before importing Torch."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
    os.environ["NVIDIA_TF32_OVERRIDE"] = _TF32_OVERRIDE
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = _TF32_OVERRIDE
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(_CPU_THREADS)
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("resume smoke requires Linux CPU affinity")
    available = tuple(sorted(os.sched_getaffinity(0)))
    if len(available) < _CPU_THREADS:
        raise RuntimeError(
            f"resume smoke requires {_CPU_THREADS} CPUs; found {len(available)}"
        )
    selected = available[:_CPU_THREADS]
    os.sched_setaffinity(0, selected)
    return selected


def _looks_like_torch_tensor(value: Any) -> bool:
    kind = type(value)
    return kind.__module__.startswith("torch") and all(
        hasattr(value, name) for name in ("detach", "dtype", "device", "shape")
    )


def _assert_bitwise_equal(left: Any, right: Any, *, path: str = "root") -> int:
    """Recursively require identical structure and tensor bytes.

    The return value is the number of tensor leaves compared.  Torch is lazily
    imported only when a tensor leaf is encountered, keeping structure-only
    unit tests independent of a CUDA runtime.
    """

    left_tensor = _looks_like_torch_tensor(left)
    right_tensor = _looks_like_torch_tensor(right)
    if left_tensor or right_tensor:
        if not (left_tensor and right_tensor):
            raise AssertionError(f"{path}: tensor/non-tensor mismatch")
        import torch

        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise AssertionError(f"{path}: unsupported Torch tensor-like value")
        for name in ("dtype", "device", "layout"):
            if getattr(left, name) != getattr(right, name):
                raise AssertionError(
                    f"{path}: tensor {name} differs: "
                    f"{getattr(left, name)} != {getattr(right, name)}"
                )
        if tuple(left.shape) != tuple(right.shape):
            raise AssertionError(
                f"{path}: tensor shape differs: "
                f"{tuple(left.shape)} != {tuple(right.shape)}"
            )
        if left.layout is not torch.strided:
            raise AssertionError(f"{path}: only dense strided tensors are supported")
        left_bytes = left.detach().contiguous().reshape(-1).view(torch.uint8)
        right_bytes = right.detach().contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(left_bytes, right_bytes):
            raise AssertionError(f"{path}: tensor bytes differ")
        return 1

    if type(left) is not type(right):
        raise AssertionError(
            f"{path}: types differ: {type(left).__name__} != {type(right).__name__}"
        )
    if isinstance(left, Mapping):
        if set(left) != set(right):
            missing = tuple(key for key in left if key not in right)
            unexpected = tuple(key for key in right if key not in left)
            raise AssertionError(
                f"{path}: mapping keys differ: missing={missing}, "
                f"unexpected={unexpected}"
            )
        return sum(
            _assert_bitwise_equal(
                left[key],
                right[key],
                path=f"{path}[{key!r}]",
            )
            for key in left
        )
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(
                f"{path}: sequence lengths differ: {len(left)} != {len(right)}"
            )
        return sum(
            _assert_bitwise_equal(item, right[index], path=f"{path}[{index}]")
            for index, item in enumerate(left)
        )
    if isinstance(left, float):
        if struct.pack("!d", left) != struct.pack("!d", right):
            raise AssertionError(f"{path}: float bytes differ: {left!r} != {right!r}")
        return 0
    if isinstance(left, (str, bytes, int, bool, type(None))):
        if left != right:
            raise AssertionError(f"{path}: values differ: {left!r} != {right!r}")
        return 0
    try:
        equal = left == right
    except Exception as error:  # noqa: BLE001 - report unsupported leaves precisely
        raise AssertionError(f"{path}: cannot compare {type(left).__name__}") from error
    if type(equal) is not bool or not equal:
        raise AssertionError(f"{path}: values differ: {left!r} != {right!r}")
    return 0


def _sealed_record(
    record: Mapping[str, Any],
    canonical_fingerprint: Any,
) -> dict[str, Any]:
    result = dict(record)
    if "self_sha256" in result:
        raise ValueError("record is already sealed")
    result["self_sha256"] = canonical_fingerprint(result)
    return result


def _rank_state(
    *,
    process_seed: int,
    sampler: Any,
    step: int,
    cpu_affinity: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "cpu_affinity": list(cpu_affinity),
        "process_seed": process_seed,
        "sampler": sampler.cursor(step).to_record(),
        "schema_id": "rexpolicy/stage0-grasp-lift-no-z-resume-smoke-rank/v1",
    }


def _require_formal_runtime(config: Any, torch: Any) -> None:
    expected = {
        "allow_tf32": False,
        "cpu_threads": _CPU_THREADS,
        "cublas_workspace_config": _CUBLAS_WORKSPACE_CONFIG,
        "cuda_fused_adamw": True,
        "deterministic_algorithms": True,
        "dtype": "float32",
        "global_batch_size": 2048,
        "interop_threads": _INTEROP_THREADS,
        "nvidia_tf32_override": _TF32_OVERRIDE,
        "require_cuda": True,
        "torch_allow_tf32_cublas_override": _TF32_OVERRIDE,
        "torch_compile": False,
    }
    changed = tuple(
        name for name, expected_value in expected.items()
        if getattr(config, name) != expected_value
    )
    if changed:
        raise RuntimeError(f"formal no-z runtime contract changed: {changed}")
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Torch deterministic algorithms are not enabled")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 must remain disabled")
    for name, value in (
        ("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE_CONFIG),
        ("NVIDIA_TF32_OVERRIDE", _TF32_OVERRIDE),
        ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", _TF32_OVERRIDE),
    ):
        if os.environ.get(name) != value:
            raise RuntimeError(f"{name} changed after Torch import")


def _build_stack(
    *,
    torch: Any,
    config: Any,
    normalization: Any,
    training_seed: int,
    device: Any,
    derive_stage0_seed: Any,
    build_policy: Any,
    trainer_type: Any,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    model_seed = derive_stage0_seed(
        training_seed,
        "model",
        namespace="stage0-grasp-lift-no-z",
    )
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed(model_seed)
    policy, freeze_record = build_policy(config, device=device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=config.cuda_fused_adamw,
    )
    if optimizer.defaults.get("fused") is not True:
        raise RuntimeError("resume smoke did not construct fused AdamW")
    trainer = trainer_type(
        policy,
        optimizer,
        gradient_clip_norm=config.gradient_clip_norm,
        action_feature_mask=normalization.effective_action_mask,
    )
    return policy, optimizer, trainer, freeze_record


def _run_steps(
    *,
    torch: Any,
    trainer: Any,
    sampler: Any,
    full_batch: Any,
    training_seed: int,
    device: Any,
    start: int,
    stop: int,
    derive_stage0_seed: Any,
    select_batch: Any,
) -> None:
    if trainer.optimizer_step != start:
        raise RuntimeError("trainer step does not match smoke branch start")
    for step in range(start, stop):
        indices = sampler.rank_indices(step, rank=0, world_size=1)
        batch = select_batch(full_batch, indices)
        flow_seed = derive_stage0_seed(
            training_seed,
            "flow",
            "rank",
            0,
            "step",
            step,
            namespace="stage0-grasp-lift-no-z",
        )
        generator = torch.Generator(device=device).manual_seed(flow_seed)
        metrics = trainer.step(
            batch,
            None,
            generator=generator,
            materialize_metrics=False,
        )
        if metrics is not None or trainer.optimizer_step != step + 1:
            raise RuntimeError("smoke trainer step bookkeeping changed")


def main() -> None:
    args = _parse_args()
    total_steps, checkpoint_step = _validate_smoke_schedule(
        _TOTAL_STEPS,
        _CHECKPOINT_STEP,
    )
    affinity = _configure_preimport_runtime()

    import torch

    torch.set_num_threads(_CPU_THREADS)
    torch.set_num_interop_threads(_INTEROP_THREADS)

    from rexpolicy.stage0.checkpoint import (
        Stage0CheckpointHashes,
        Stage0CheckpointManager,
    )
    from rexpolicy.stage0.data.grasp_lift_training import (
        GRASP_LIFT_MODEL_ACTION_VIEW_SHA256,
    )
    from rexpolicy.stage0.data.grasp_lift_training_artifact import (
        load_grasp_lift_training_artifact,
    )
    from rexpolicy.stage0.distributed import Stage0DistributedContext
    from rexpolicy.stage0.envs.grasp_lift_state_view import (
        DEFAULT_GRASP_LIFT_STATE_VIEW,
    )
    from rexpolicy.stage0.grasp_lift_no_z import (
        GraspLiftNoZSampler,
        build_grasp_lift_no_z_policy,
        load_grasp_lift_no_z_config,
        normalize_grasp_lift_no_z_batch_once,
        select_grasp_lift_no_z_batch,
        validate_grasp_lift_no_z_training_data,
    )
    from rexpolicy.stage0.implementation_fingerprint import (
        capture_stage0_implementation_fingerprint,
    )
    from rexpolicy.stage0.long_run import atomic_write_json
    from rexpolicy.stage0.seed import derive_stage0_seed
    from rexpolicy.stage0.trainers import Stage0PolicyTrainer
    from rexpolicy.stage0.trainers.batch import collate_success_windows
    from rexpolicy.stage0.types import canonical_fingerprint

    config = load_grasp_lift_no_z_config(args.config)
    config.require_training_seed(args.training_seed)
    torch.use_deterministic_algorithms(config.deterministic_algorithms)
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    _require_formal_runtime(config, torch)

    context = Stage0DistributedContext.from_environment(
        args.training_seed,
        initialize=False,
        require_cuda=config.require_cuda,
    )
    try:
        if context.world_size != 1 or context.rank != 0 or context.local_rank != 0:
            raise RuntimeError("resume smoke is a single-rank CUDA check")
        if context.device.type != "cuda":
            raise RuntimeError("resume smoke requires CUDA")

        repository_root = Path(__file__).resolve().parents[1]
        implementation = capture_stage0_implementation_fingerprint(repository_root)
        artifact = load_grasp_lift_training_artifact(
            args.training_artifact,
            pilot_directory=args.pilot_directory,
        )
        validate_grasp_lift_no_z_training_data(artifact.data, config)
        if artifact.data.window_splits.validation or artifact.data.window_splits.test:
            raise RuntimeError("resume smoke materialized held-out windows")
        normalization = artifact.normalization
        source_corpus_sha256 = artifact.data.window_splits.corpus_sha256
        full_batch = normalize_grasp_lift_no_z_batch_once(
            collate_success_windows(artifact.data.window_splits.train),
            normalization,
            source_corpus_sha256=source_corpus_sha256,
        ).to(context.device, dtype=torch.float32)
        sampler = GraspLiftNoZSampler(
            training_seed=args.training_seed,
            dataset_size=len(artifact.data.window_splits.train),
            global_batch_size=config.global_batch_size,
        )

        output = Path(os.path.abspath(Path(args.output_directory).expanduser()))
        if output.is_symlink() or output.exists():
            raise FileExistsError(f"resume smoke output already exists: {output}")
        output.mkdir(parents=True)

        protocol_sha256 = canonical_fingerprint(
            {
                "action_model_view_sha256": GRASP_LIFT_MODEL_ACTION_VIEW_SHA256,
                "normalization_sha256": normalization.sha256,
                "sampler": sampler.to_record(),
                "schema_id": (
                    "rexpolicy/stage0-grasp-lift-no-z-resume-smoke-protocol/v1"
                ),
                "training_membership_sha256": (
                    artifact.data.corpus.training_membership_sha256
                ),
            }
        )
        contract = _sealed_record(
            {
                "artifact_sha256": artifact.artifact_sha256,
                "checkpoint_step": checkpoint_step,
                "config_sha256": config.sha256,
                "implementation": implementation.to_record(),
                "implementation_sha256": implementation.sha256,
                "locked_test": "not_created",
                "protocol_sha256": protocol_sha256,
                "schema_id": "rexpolicy/stage0-grasp-lift-no-z-resume-smoke/v1",
                "total_steps": total_steps,
                "train_content_sha256": artifact.train_content_sha256,
                "training_seed": args.training_seed,
                "validation_consumed": False,
                "world_size": 1,
            },
            canonical_fingerprint,
        )
        atomic_write_json(output / "smoke-contract.json", contract)
        hashes = Stage0CheckpointHashes(
            config_sha256=contract["self_sha256"],
            state_schema_sha256=DEFAULT_GRASP_LIFT_STATE_VIEW.sha256,
            dataset_sha256=artifact.train_content_sha256,
            split_sha256=protocol_sha256,
        )
        extra = {
            "implementation_sha256": implementation.sha256,
            "locked_test": "not_created",
            "schema_id": "rexpolicy/stage0-grasp-lift-no-z-resume-smoke-extra/v1",
            "smoke_contract_sha256": contract["self_sha256"],
            "train_content_sha256": artifact.train_content_sha256,
            "training_only": True,
            "validation_consumed": False,
        }

        build_arguments = {
            "torch": torch,
            "config": config,
            "normalization": normalization,
            "training_seed": args.training_seed,
            "device": context.device,
            "derive_stage0_seed": derive_stage0_seed,
            "build_policy": build_grasp_lift_no_z_policy,
            "trainer_type": Stage0PolicyTrainer,
        }
        continuous_policy, continuous_optimizer, continuous_trainer, freeze_a = (
            _build_stack(**build_arguments)
        )
        split_policy, split_optimizer, split_trainer, freeze_b = _build_stack(
            **build_arguments
        )
        _assert_bitwise_equal(freeze_a, freeze_b, path="initial.freeze_record")
        initial_model_tensors = _assert_bitwise_equal(
            continuous_policy.state_dict(),
            split_policy.state_dict(),
            path="initial.model",
        )
        if initial_model_tensors < 1:
            raise RuntimeError("initial model comparison contained no tensors")

        run_arguments = {
            "torch": torch,
            "sampler": sampler,
            "full_batch": full_batch,
            "training_seed": args.training_seed,
            "device": context.device,
            "derive_stage0_seed": derive_stage0_seed,
            "select_batch": select_grasp_lift_no_z_batch,
        }
        _run_steps(
            trainer=continuous_trainer,
            start=0,
            stop=total_steps,
            **run_arguments,
        )
        _run_steps(
            trainer=split_trainer,
            start=0,
            stop=checkpoint_step,
            **run_arguments,
        )

        manager = Stage0CheckpointManager(output, rank=0, world_size=1)
        checkpoint = manager.save(
            sequence=checkpoint_step,
            phase="no_z_policy",
            steps={"global": checkpoint_step, "no_z_policy": checkpoint_step},
            hashes=hashes,
            models={"no_z_policy": split_policy},
            optimizers={"no_z_policy": split_optimizer},
            rank_state=_rank_state(
                process_seed=context.process_seed,
                sampler=sampler,
                step=checkpoint_step,
                cpu_affinity=affinity,
            ),
            extra=extra,
            cuda_device=context.device,
        )

        resumed_policy, resumed_optimizer, resumed_trainer, freeze_c = _build_stack(
            **build_arguments
        )
        if resumed_policy is split_policy or resumed_optimizer is split_optimizer:
            raise RuntimeError("resume smoke did not construct fresh components")
        resumed = manager.load(
            hashes=hashes,
            models={"no_z_policy": resumed_policy},
            optimizers={"no_z_policy": resumed_optimizer},
            checkpoint=checkpoint,
            restore_rng=True,
            cuda_device=context.device,
        )
        expected_rank_state = _rank_state(
            process_seed=context.process_seed,
            sampler=sampler,
            step=checkpoint_step,
            cpu_affinity=affinity,
        )
        if (
            resumed.sequence != checkpoint_step
            or resumed.phase != "no_z_policy"
            or resumed.steps
            != {"global": checkpoint_step, "no_z_policy": checkpoint_step}
            or resumed.rank_state != expected_rank_state
            or resumed.extra != extra
        ):
            raise RuntimeError("strict checkpoint metadata restore changed")
        resumed_trainer.optimizer_step = resumed.sequence
        _assert_bitwise_equal(freeze_b, freeze_c, path="resume.freeze_record")
        checkpoint_model_tensors = _assert_bitwise_equal(
            split_policy.state_dict(),
            resumed_policy.state_dict(),
            path="resume.model",
        )
        checkpoint_optimizer_tensors = _assert_bitwise_equal(
            split_optimizer.state_dict(),
            resumed_optimizer.state_dict(),
            path="resume.optimizer",
        )
        if checkpoint_model_tensors < 1 or checkpoint_optimizer_tensors < 1:
            raise RuntimeError("checkpoint comparison contained no tensor state")

        _run_steps(
            trainer=resumed_trainer,
            start=checkpoint_step,
            stop=total_steps,
            **run_arguments,
        )
        final_model_tensors = _assert_bitwise_equal(
            continuous_policy.state_dict(),
            resumed_policy.state_dict(),
            path="final.model",
        )
        final_optimizer_tensors = _assert_bitwise_equal(
            continuous_optimizer.state_dict(),
            resumed_optimizer.state_dict(),
            path="final.optimizer",
        )
        if continuous_trainer.optimizer_step != resumed_trainer.optimizer_step:
            raise AssertionError("final trainer optimizer steps differ")
        if continuous_trainer.optimizer_step != total_steps:
            raise AssertionError("final trainer optimizer step changed")
        torch.cuda.synchronize(context.device)

        result = {
            "checkpoint": checkpoint.name,
            "checkpoint_step": checkpoint_step,
            "complete": True,
            "comparison": {
                "checkpoint_model_tensors": checkpoint_model_tensors,
                "checkpoint_optimizer_tensors": checkpoint_optimizer_tensors,
                "final_model_tensors": final_model_tensors,
                "final_optimizer_tensors": final_optimizer_tensors,
                "initial_model_tensors": initial_model_tensors,
                "method": "dense tensor bytes plus exact recursive structure",
            },
            "fused_adamw": True,
            "passed": True,
            "schema_id": "rexpolicy/stage0-grasp-lift-no-z-resume-smoke-result/v1",
            "smoke_contract_sha256": contract["self_sha256"],
            "tf32": False,
            "total_steps": total_steps,
            "train_content_sha256": artifact.train_content_sha256,
            "trainer_optimizer_step": resumed_trainer.optimizer_step,
            "training_seed": args.training_seed,
            "validation_consumed": False,
        }
        atomic_write_json(output / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        context.close()


if __name__ == "__main__":
    main()
