"""Small torch.distributed lifecycle used by the flywheel runner."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any


@dataclass
class DistributedContext:
    """Process identity and collectives for one GPU worker."""

    rank: int
    local_rank: int
    world_size: int
    device: Any
    initialized: bool

    @classmethod
    def initialize(cls, *, timeout_minutes: int = 30) -> DistributedContext:
        """Initialize NCCL when launched by ``torchrun``."""
        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available():
            raise RuntimeError("The flywheel requires CUDA")

        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        visible_devices = torch.cuda.device_count()
        if local_rank < 0 or local_rank >= visible_devices:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is outside the {visible_devices} visible CUDA devices"
            )

        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        initialized = False
        if world_size > 1:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=timedelta(minutes=timeout_minutes),
                    device_id=device,
                )
            initialized = True
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
            initialized=initialized,
        )

    @property
    def is_main(self) -> bool:
        """Whether this worker owns shared logging and checkpoints."""
        return self.rank == 0

    def barrier(self) -> None:
        """Wait for all workers when distributed mode is active."""
        if not self.initialized:
            return
        import torch.distributed as dist

        dist.barrier(device_ids=[self.local_rank])

    def mean(self, value: float) -> float:
        """Average a scalar across ranks."""
        if not self.initialized:
            return float(value)
        import torch
        import torch.distributed as dist

        tensor = torch.tensor(float(value), dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item() / self.world_size)

    def sum_int(self, value: int) -> int:
        """Sum an integer across ranks."""
        if not self.initialized:
            return int(value)
        import torch
        import torch.distributed as dist

        tensor = torch.tensor(int(value), dtype=torch.int64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return int(tensor.item())

    def max_int(self, value: int) -> int:
        """Return the maximum integer across ranks."""
        if not self.initialized:
            return int(value)
        import torch
        import torch.distributed as dist

        tensor = torch.tensor(int(value), dtype=torch.int64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return int(tensor.item())

    def sum_float(self, value: float) -> float:
        """Sum a floating-point scalar across ranks."""
        if not self.initialized:
            return float(value)
        import torch
        import torch.distributed as dist

        tensor = torch.tensor(float(value), dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item())

    def all_gather_objects(self, value: Any) -> list[Any]:
        """Gather a small Python metadata object from every rank."""
        if not self.initialized:
            return [value]
        import torch.distributed as dist

        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, value)
        return gathered

    def broadcast_object(self, value: Any, *, source: int = 0) -> Any:
        """Broadcast one small Python metadata object from ``source``."""
        if not self.initialized:
            return value
        import torch.distributed as dist

        payload = [value if self.rank == source else None]
        dist.broadcast_object_list(payload, src=source, device=self.device)
        return payload[0]

    def parameter_sync_error(self, module: Any, *, sample_size: int = 4096) -> float:
        """Return the rank-to-rank checksum spread for a trainable parameter."""
        import torch

        parameter = next(
            (item for item in module.parameters() if item.requires_grad), None
        )
        if parameter is None:
            raise RuntimeError(
                "No trainable parameter is available for the DDP sync check"
            )
        checksum = (
            parameter.detach().reshape(-1)[:sample_size].float().sum().to(self.device)
        )
        if not self.initialized:
            return 0.0

        import torch.distributed as dist

        minimum = checksum.clone()
        maximum = checksum.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        return float(torch.abs(maximum - minimum).item())

    def close(self) -> None:
        """Destroy the process group after every worker has finished."""
        if not self.initialized:
            return
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
        self.initialized = False
