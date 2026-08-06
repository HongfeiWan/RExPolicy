"""Minimal torch.distributed environment and seeding helpers for Stage 0."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Mapping

import numpy as np

from .seed import derive_stage0_seed


class Stage0DistributedError(RuntimeError):
    """The process environment cannot form a valid Stage 0 worker."""


def _environment_int(environment: Mapping[str, str], name: str) -> int:
    try:
        value = int(environment[name])
    except (KeyError, TypeError, ValueError) as error:
        raise Stage0DistributedError(f"{name} must be an integer") from error
    return value


@dataclass(frozen=True)
class Stage0ProcessEnvironment:
    rank: int
    local_rank: int
    world_size: int

    @classmethod
    def parse(
        cls, environment: Mapping[str, str] | None = None
    ) -> Stage0ProcessEnvironment:
        source = os.environ if environment is None else environment
        names = {"RANK", "LOCAL_RANK", "WORLD_SIZE"}
        present = names.intersection(source)
        if not present:
            return cls(rank=0, local_rank=0, world_size=1)
        if present != names:
            raise Stage0DistributedError(
                "RANK, LOCAL_RANK, and WORLD_SIZE must be set together"
            )
        rank = _environment_int(source, "RANK")
        local_rank = _environment_int(source, "LOCAL_RANK")
        world_size = _environment_int(source, "WORLD_SIZE")
        if world_size < 1 or rank < 0 or rank >= world_size or local_rank < 0:
            raise Stage0DistributedError(
                f"invalid rank environment: rank={rank}, local_rank={local_rank}, "
                f"world_size={world_size}"
            )
        return cls(rank=rank, local_rank=local_rank, world_size=world_size)


def seed_stage0_process(base_seed: int, rank: int, *, device: Any) -> int:
    """Seed Python, NumPy, Torch CPU, and this rank's CUDA device."""
    import torch

    seed = derive_stage0_seed(base_seed, "rank", rank, namespace="stage0-process")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed(seed)
    return seed


@dataclass
class Stage0DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: Any
    process_seed: int
    initialized: bool
    owns_process_group: bool = False

    @classmethod
    def from_environment(
        cls,
        base_seed: int,
        *,
        initialize: bool = False,
        require_cuda: bool = False,
        backend: str | None = None,
        timeout_seconds: int = 1800,
    ) -> Stage0DistributedContext:
        """Read torchrun identity, select a device, seed, and optionally init."""
        import torch
        import torch.distributed as dist

        if type(timeout_seconds) is not int or timeout_seconds < 1:
            raise ValueError("timeout_seconds must be a positive integer")
        process = Stage0ProcessEnvironment.parse()
        cuda = torch.cuda.is_available()
        if require_cuda and not cuda:
            raise Stage0DistributedError("Stage 0 was configured to require CUDA")
        if cuda:
            count = torch.cuda.device_count()
            if process.local_rank >= count:
                raise Stage0DistributedError(
                    f"LOCAL_RANK={process.local_rank} is outside {count} visible GPUs"
                )
            torch.cuda.set_device(process.local_rank)
            device = torch.device("cuda", process.local_rank)
        else:
            device = torch.device("cpu")

        selected_backend = backend or ("nccl" if cuda else "gloo")
        if selected_backend == "nccl" and not cuda:
            raise Stage0DistributedError("NCCL requires CUDA")
        owns = False
        if initialize and process.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend=selected_backend,
                init_method="env://",
                rank=process.rank,
                world_size=process.world_size,
                timeout=timedelta(seconds=timeout_seconds),
                **({"device_id": device} if selected_backend == "nccl" else {}),
            )
            owns = True
        initialized = dist.is_available() and dist.is_initialized()
        if initialized and (
            dist.get_rank() != process.rank
            or dist.get_world_size() != process.world_size
        ):
            raise Stage0DistributedError(
                "initialized process group disagrees with environment"
            )
        process_seed = seed_stage0_process(base_seed, process.rank, device=device)
        return cls(
            rank=process.rank,
            local_rank=process.local_rank,
            world_size=process.world_size,
            device=device,
            process_seed=process_seed,
            initialized=initialized,
            owns_process_group=owns,
        )

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def _require_collectives(self) -> None:
        if self.world_size > 1 and not self.initialized:
            raise Stage0DistributedError(
                "distributed collectives require initialize=True or an existing group"
            )

    def barrier(self) -> None:
        self._require_collectives()
        if not self.initialized:
            return
        import torch.distributed as dist

        if self.device.type == "cuda":
            dist.barrier(device_ids=[self.local_rank])
        else:
            dist.barrier()

    def broadcast_object(self, value: Any, *, source: int = 0) -> Any:
        self._require_collectives()
        if not self.initialized:
            return value
        import torch.distributed as dist

        payload = [value if self.rank == source else None]
        dist.broadcast_object_list(payload, src=source)
        return payload[0]

    def close(self) -> None:
        if not self.owns_process_group or not self.initialized:
            return
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
        self.initialized = False
        self.owns_process_group = False


__all__ = [
    "Stage0DistributedContext",
    "Stage0DistributedError",
    "Stage0ProcessEnvironment",
    "seed_stage0_process",
]
