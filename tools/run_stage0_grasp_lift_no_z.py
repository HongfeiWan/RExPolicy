#!/usr/bin/env python3
"""Run one fixed-budget, train-only Grasp-Lift no-z policy seed."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_CPU_THREADS = 16
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_TF32_OVERRIDE = "0"


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
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _restrict_cpu_threads() -> tuple[int, ...]:
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
        raise RuntimeError("formal no-z training requires Linux CPU affinity")
    available = tuple(sorted(os.sched_getaffinity(0)))
    if len(available) < _CPU_THREADS:
        raise RuntimeError(
            f"no-z training requires {_CPU_THREADS} CPUs; found {len(available)}"
        )
    selected = available[:_CPU_THREADS]
    os.sched_setaffinity(0, selected)
    return selected


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required run record is missing or a symlink: {path}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {constant}")
        ),
    )
    if not isinstance(value, dict):
        raise TypeError("run record must contain an object")
    return value


def _distributed_setup_directory(
    output: Path,
    *,
    resume: bool,
    contract: Mapping[str, Any],
    context: Any,
    atomic_write_json: Any,
) -> None:
    error: str | None = None
    if context.is_main:
        try:
            if resume:
                if output.is_symlink() or not output.is_dir():
                    raise ValueError("resume output is missing or a symlink")
                if _read_json(output / "run-contract.json") != dict(contract):
                    raise ValueError("resume run contract changed")
            else:
                if output.is_symlink() or output.exists():
                    raise FileExistsError(f"output already exists: {output}")
                output.mkdir(parents=True)
                atomic_write_json(output / "run-contract.json", contract)
                atomic_write_json(
                    output / "status.json",
                    {
                        "complete": False,
                        "global_step": 0,
                        "schema_id": "rexpolicy/stage0-grasp-lift-no-z-status/v1",
                        "state": "initialized",
                    },
                )
        except Exception as exception:  # noqa: BLE001 - broadcast setup failures
            error = f"{type(exception).__name__}: {exception}"
    error = context.broadcast_object(error)
    if error is not None:
        raise RuntimeError("no-z output setup failed: " + error)
    context.barrier()


def _distributed_require_identical_contract(
    contract: Mapping[str, Any],
    context: Any,
) -> None:
    local_sha256 = contract.get("self_sha256")
    if (
        not isinstance(local_sha256, str)
        or len(local_sha256) != 64
        or any(character not in "0123456789abcdef" for character in local_sha256)
    ):
        raise ValueError("run contract is not sealed")
    rank_sha256s = tuple(
        context.broadcast_object(
            local_sha256 if context.rank == source else None,
            source=source,
        )
        for source in range(context.world_size)
    )
    if any(value != rank_sha256s[0] for value in rank_sha256s):
        raise RuntimeError("distributed run contracts differ between ranks")
    context.barrier()


def _rank_state(
    context: Any,
    sampler: Any,
    step: int,
    *,
    cpu_affinity: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "cpu_affinity": list(cpu_affinity),
        "process_seed": context.process_seed,
        "sampler": sampler.cursor(step).to_record(),
        "schema_id": "rexpolicy/stage0-grasp-lift-no-z-rank/v1",
    }


def _checkpoint_extra(
    *,
    artifact: Any,
    contract: Mapping[str, Any],
    implementation_sha256: str,
    normalization_sha256: str,
) -> dict[str, Any]:
    return {
        "audit_dataset_sha256": artifact.audit_dataset_sha256,
        "implementation_sha256": implementation_sha256,
        "locked_test": "not_created",
        "normalization_sha256": normalization_sha256,
        "run_contract_sha256": contract["self_sha256"],
        "schema_id": "rexpolicy/stage0-grasp-lift-no-z-checkpoint-extra/v1",
        "train_content_sha256": artifact.train_content_sha256,
        "training_only": True,
        "validation_consumed": False,
    }


def _cuda_runtime_record(torch: Any, device: Any, *, world_size: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    nccl_version: Any = None
    if world_size > 1:
        if not torch.distributed.is_nccl_available():
            raise RuntimeError("multi-rank no-z training requires NCCL")
        raw_nccl_version = torch.cuda.nccl.version()
        nccl_version = (
            list(raw_nccl_version)
            if isinstance(raw_nccl_version, tuple)
            else raw_nccl_version
        )
    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "compute_capability": [properties.major, properties.minor],
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "multi_processor_count": properties.multi_processor_count,
        "nccl_version": nccl_version,
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "schema_id": "rexpolicy/stage0-grasp-lift-no-z-cuda-runtime/v1",
        "torch_allow_tf32_cublas_override": os.environ.get(
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"
        ),
        "torch_compiled_cuda_version": torch.version.cuda,
        "visible_cuda_devices": torch.cuda.device_count(),
    }


def _distributed_metric_means(
    loss: float,
    gradient_norm: float,
    *,
    context: Any,
    torch: Any,
) -> tuple[float, float]:
    if context.world_size == 1:
        return loss, gradient_norm
    values = torch.tensor(
        [loss, gradient_norm],
        dtype=torch.float64,
        device=context.device,
    )
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    values.div_(context.world_size)
    return float(values[0].item()), float(values[1].item())


def _begin_heartbeat_attempt(
    log: Any,
    *,
    resume: bool,
    global_step: int,
    training_seed: int,
    world_size: int,
) -> int:
    records = log.records()
    if not resume and records:
        raise RuntimeError("fresh no-z run already has heartbeat records")
    attempt_indices: list[int] = []
    observed_steps: list[int] = []
    for record in records:
        if record.get("schema_id") != "rexpolicy/stage0-grasp-lift-no-z-heartbeat/v2":
            raise RuntimeError("existing no-z heartbeat schema changed")
        attempt_index = record.get("attempt_index")
        observed_step = record.get("global_step")
        if (
            type(attempt_index) is not int
            or attempt_index < 0
            or type(observed_step) is not int
            or observed_step < 0
        ):
            raise RuntimeError("existing no-z heartbeat cursor is invalid")
        attempt_indices.append(attempt_index)
        observed_steps.append(observed_step)
    attempt_index = (max(attempt_indices) + 1) if resume and records else int(resume)
    log.append(
        {
            "attempt_index": attempt_index,
            "event": "resume" if resume else "start",
            "global_step": global_step,
            "prior_observed_global_step": (
                max(observed_steps) if observed_steps else None
            ),
            "resume_from_committed_step": global_step if resume else None,
            "schema_id": "rexpolicy/stage0-grasp-lift-no-z-heartbeat/v2",
            "training_seed": training_seed,
            "world_size": world_size,
        }
    )
    return attempt_index


def main() -> None:
    args = _parse_args()
    affinity = _restrict_cpu_threads()

    import torch

    torch.set_num_threads(_CPU_THREADS)
    torch.set_num_interop_threads(4)

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
    from rexpolicy.stage0.grasp_lift_pilot import seal_grasp_lift_pilot_record
    from rexpolicy.stage0.implementation_fingerprint import (
        capture_stage0_implementation_fingerprint,
    )
    from rexpolicy.stage0.long_run import AtomicJsonlLog, atomic_write_json
    from rexpolicy.stage0.seed import derive_stage0_seed
    from rexpolicy.stage0.trainers import Stage0PolicyTrainer
    from rexpolicy.stage0.trainers.batch import collate_success_windows
    from rexpolicy.stage0.types import canonical_fingerprint

    config = load_grasp_lift_no_z_config(args.config)
    if (
        config.cpu_threads != _CPU_THREADS
        or config.interop_threads != 4
        or config.cublas_workspace_config != _CUBLAS_WORKSPACE_CONFIG
        or config.nvidia_tf32_override != _TF32_OVERRIDE
        or config.torch_allow_tf32_cublas_override != _TF32_OVERRIDE
    ):
        raise RuntimeError("runtime thread contract changed after Torch import")
    config.require_training_seed(args.training_seed)
    torch.use_deterministic_algorithms(config.deterministic_algorithms)
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.benchmark = config.cudnn_benchmark

    context = Stage0DistributedContext.from_environment(
        args.training_seed,
        initialize=True,
        require_cuda=config.require_cuda,
    )
    output: Path | None = None
    output_owned = False
    step = 0
    committed_step: int | None = None
    try:
        if context.device.type != "cuda":
            raise RuntimeError("formal Grasp-Lift no-z training requires CUDA")
        if config.global_batch_size % context.world_size:
            raise RuntimeError("global batch is not divisible by world size")
        cuda_runtime = _cuda_runtime_record(
            torch,
            context.device,
            world_size=context.world_size,
        )

        repository_root = Path(__file__).resolve().parents[1]
        implementation = capture_stage0_implementation_fingerprint(repository_root)
        artifact = load_grasp_lift_training_artifact(
            args.training_artifact,
            pilot_directory=args.pilot_directory,
        )
        validate_grasp_lift_no_z_training_data(artifact.data, config)
        normalization = artifact.normalization
        source_corpus_sha256 = artifact.data.window_splits.corpus_sha256
        full_batch = normalize_grasp_lift_no_z_batch_once(
            collate_success_windows(artifact.data.window_splits.train),
            normalization,
            source_corpus_sha256=source_corpus_sha256,
        ).to(context.device, dtype=torch.float32)

        model_seed = derive_stage0_seed(
            args.training_seed,
            "model",
            namespace="stage0-grasp-lift-no-z",
        )
        torch.manual_seed(model_seed)
        torch.cuda.manual_seed(model_seed)
        policy, freeze_record = build_grasp_lift_no_z_policy(
            config,
            device=context.device,
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in policy.parameters() if parameter.requires_grad),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=config.cuda_fused_adamw,
        )
        model: Any = policy
        if context.world_size > 1:
            from torch.nn.parallel import DistributedDataParallel

            model = DistributedDataParallel(
                policy,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
        trainer = Stage0PolicyTrainer(
            model,
            optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
            action_feature_mask=normalization.effective_action_mask,
        )
        sampler = GraspLiftNoZSampler(
            training_seed=args.training_seed,
            dataset_size=len(artifact.data.window_splits.train),
            global_batch_size=config.global_batch_size,
        )

        run_config_sha256 = config.run_sha256(
            args.training_seed,
            world_size=context.world_size,
        )
        protocol_sha256 = canonical_fingerprint(
            {
                "action_feature_mask": normalization.effective_action_mask.tolist(),
                "action_model_view_sha256": GRASP_LIFT_MODEL_ACTION_VIEW_SHA256,
                "freeze_record": freeze_record,
                "implementation_sha256": implementation.sha256,
                "normalization_sha256": normalization.sha256,
                "run_config_sha256": run_config_sha256,
                "sampler": sampler.to_record(),
                "schema_id": "rexpolicy/stage0-grasp-lift-no-z-protocol/v1",
                "training_membership_sha256": (
                    artifact.data.corpus.training_membership_sha256
                ),
            }
        )
        experiment_sha256 = canonical_fingerprint(
            {
                "implementation_sha256": implementation.sha256,
                "protocol_sha256": protocol_sha256,
                "run_config_sha256": run_config_sha256,
                "schema_id": "rexpolicy/stage0-grasp-lift-no-z-experiment/v1",
                "train_content_sha256": artifact.train_content_sha256,
            }
        )
        hashes = Stage0CheckpointHashes(
            config_sha256=experiment_sha256,
            state_schema_sha256=DEFAULT_GRASP_LIFT_STATE_VIEW.sha256,
            dataset_sha256=artifact.train_content_sha256,
            split_sha256=protocol_sha256,
        )
        contract = seal_grasp_lift_pilot_record(
            {
                "artifact_sha256": artifact.artifact_sha256,
                "audit_dataset_sha256": artifact.audit_dataset_sha256,
                "checkpoint_hashes": hashes.to_record(),
                "config": config.to_record(),
                "config_sha256": config.sha256,
                "cuda_runtime": cuda_runtime,
                "freeze_record": freeze_record,
                "implementation": implementation.to_record(),
                "implementation_sha256": implementation.sha256,
                "locked_test": "not_created",
                "normalization_sha256": normalization.sha256,
                "protocol_sha256": protocol_sha256,
                "run_config_sha256": run_config_sha256,
                "schema_id": "rexpolicy/stage0-grasp-lift-no-z-run-contract/v1",
                "train_content_sha256": artifact.train_content_sha256,
                "training_seed": args.training_seed,
                "validation_policy": "not opened during fixed-budget training",
                "world_size": context.world_size,
            }
        )
        _distributed_require_identical_contract(contract, context)
        output = Path(os.path.abspath(Path(args.output_directory).expanduser()))
        _distributed_setup_directory(
            output,
            resume=args.resume,
            contract=contract,
            context=context,
            atomic_write_json=atomic_write_json,
        )
        output_owned = True
        manager = Stage0CheckpointManager(
            output,
            rank=context.rank,
            world_size=context.world_size,
            barrier=context.barrier,
            broadcast_rank0=lambda value: context.broadcast_object(value),
        )
        extra = _checkpoint_extra(
            artifact=artifact,
            contract=contract,
            implementation_sha256=implementation.sha256,
            normalization_sha256=normalization.sha256,
        )

        if args.resume:
            resumed = manager.load(
                hashes=hashes,
                models={"no_z_policy": model},
                optimizers={"no_z_policy": optimizer},
                restore_rng=True,
                cuda_device=context.device,
            )
            step = resumed.sequence
            if step > config.total_steps or step % config.checkpoint_every_steps:
                raise RuntimeError("checkpoint sequence is outside the fixed schedule")
            expected_phase = "complete" if step == config.total_steps else "no_z_policy"
            if resumed.phase != expected_phase:
                raise RuntimeError("checkpoint phase changed")
            if resumed.steps != {"global": step, "no_z_policy": step}:
                raise RuntimeError("checkpoint step record changed")
            if resumed.extra != extra:
                raise RuntimeError("checkpoint experiment record changed")
            if resumed.rank_state != _rank_state(
                context,
                sampler,
                step,
                cpu_affinity=affinity,
            ):
                raise RuntimeError("checkpoint rank state changed")
            trainer.optimizer_step = step
            committed_step = step
        else:
            manager.save(
                sequence=0,
                phase="no_z_policy",
                steps={"global": 0, "no_z_policy": 0},
                hashes=hashes,
                models={"no_z_policy": model},
                optimizers={"no_z_policy": optimizer},
                rank_state=_rank_state(
                    context,
                    sampler,
                    0,
                    cpu_affinity=affinity,
                ),
                extra=extra,
                cuda_device=context.device,
            )
            committed_step = 0

        log = AtomicJsonlLog(output / "heartbeat.jsonl")
        attempt_index = _begin_heartbeat_attempt(
            log,
            resume=args.resume,
            global_step=step,
            training_seed=args.training_seed,
            world_size=context.world_size,
        )
        started = time.perf_counter()
        last_log_time = started
        last_log_step = step
        while step < config.total_steps:
            indices = sampler.rank_indices(
                step,
                rank=context.rank,
                world_size=context.world_size,
            )
            batch = select_grasp_lift_no_z_batch(full_batch, indices)
            flow_seed = derive_stage0_seed(
                args.training_seed,
                "flow",
                "rank",
                context.rank,
                "step",
                step,
                namespace="stage0-grasp-lift-no-z",
            )
            generator = torch.Generator(device=context.device).manual_seed(flow_seed)
            next_step = step + 1
            should_log = next_step % config.log_every_steps == 0
            metrics = trainer.step(
                batch,
                None,
                generator=generator,
                materialize_metrics=should_log,
            )
            step = next_step

            if should_log:
                if metrics is None:
                    raise RuntimeError("scheduled no-z metrics were not materialized")
                mean_loss, mean_gradient_norm = _distributed_metric_means(
                    metrics.loss,
                    metrics.gradient_norm,
                    context=context,
                    torch=torch,
                )
                torch.cuda.synchronize(context.device)
                now = time.perf_counter()
                rate = (
                    (step - last_log_step)
                    * config.global_batch_size
                    / max(now - last_log_time, 1.0e-9)
                )
                phase_counts: dict[str, int] = {}
                for index in indices.tolist():
                    window = artifact.data.window_splits.train[index]
                    phase = str(
                        artifact.data.window_phase_by_key[
                            (window.trajectory_id, window.start)
                        ]
                    )
                    phase_counts[phase] = phase_counts.get(phase, 0) + 1
                record = {
                    "attempt_elapsed_seconds": now - started,
                    "attempt_index": attempt_index,
                    "event": "training",
                    "global_step": step,
                    "gradient_norm": mean_gradient_norm,
                    "loss": mean_loss,
                    "metrics_reduction": "mean_across_ranks",
                    "phase_counts_rank": dict(sorted(phase_counts.items())),
                    "rank": context.rank,
                    "samples_per_second_global_nominal": rate,
                    "schema_id": "rexpolicy/stage0-grasp-lift-no-z-heartbeat/v2",
                    "training_seed": args.training_seed,
                    "world_size": context.world_size,
                }
                if context.is_main:
                    log.append(record)
                    atomic_write_json(
                        output / "status.json",
                        {
                            "complete": False,
                            "attempt_index": attempt_index,
                            "global_step": step,
                            "last_loss": mean_loss,
                            "schema_id": ("rexpolicy/stage0-grasp-lift-no-z-status/v1"),
                            "state": "training",
                        },
                    )
                last_log_time = now
                last_log_step = step

            if step % config.checkpoint_every_steps == 0:
                manager.save(
                    sequence=step,
                    phase=("complete" if step == config.total_steps else "no_z_policy"),
                    steps={"global": step, "no_z_policy": step},
                    hashes=hashes,
                    models={"no_z_policy": model},
                    optimizers={"no_z_policy": optimizer},
                    rank_state=_rank_state(
                        context,
                        sampler,
                        step,
                        cpu_affinity=affinity,
                    ),
                    extra=extra,
                    cuda_device=context.device,
                )
                committed_step = step

        if context.is_main:
            final = {
                "artifact_sha256": artifact.artifact_sha256,
                "attempt_index": attempt_index,
                "complete": True,
                "final_checkpoint": manager.checkpoint_name(step),
                "global_step": step,
                "schema_id": "rexpolicy/stage0-grasp-lift-no-z-status/v1",
                "state": "complete",
                "train_content_sha256": artifact.train_content_sha256,
                "training_seed": args.training_seed,
                "validation_consumed": False,
            }
            atomic_write_json(output / "status.json", final)
            print(json.dumps(final, indent=2, sort_keys=True))
        context.barrier()
    except BaseException as error:
        if (
            context.is_main
            and output_owned
            and output is not None
            and output.is_dir()
            and not output.is_symlink()
        ):
            try:
                atomic_write_json(
                    output / "status.json",
                    {
                        "complete": False,
                        "error": str(error)[:1000],
                        "error_type": type(error).__name__,
                        "global_step": step,
                        "last_committed_checkpoint": (
                            None
                            if committed_step is None
                            else f"checkpoint-{committed_step:012d}"
                        ),
                        "last_committed_step": committed_step,
                        "schema_id": "rexpolicy/stage0-grasp-lift-no-z-status/v1",
                        "state": "error",
                        "training_seed": args.training_seed,
                        "validation_consumed": False,
                    },
                )
            except Exception as status_error:  # noqa: BLE001 - preserve root error
                print(
                    "failed to persist no-z error status: "
                    f"{type(status_error).__name__}: {status_error}",
                    file=sys.stderr,
                )
        raise
    finally:
        context.close()


if __name__ == "__main__":
    main()
