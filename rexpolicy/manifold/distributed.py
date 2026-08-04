"""Autograd-safe tensor gathering for manifold self-supervision."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


def distributed_is_active(group: Any = None) -> bool:
    return (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size(group=group) > 1
    )


class _AllGather(torch.autograd.Function):
    """Equal-shape all-gather whose backward routes each rank's slice."""

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, group: Any) -> tuple[torch.Tensor, ...]:
        ctx.group = group
        ctx.rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)
        outputs = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(outputs, tensor.contiguous(), group=group)
        return tuple(outputs)

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        # This is a reduce-scatter over the output-index dimension.  Reducing
        # only each rank's own slice would mix gradients for different source
        # ranks when local batch sizes (and therefore padding masks) differ.
        reduced = torch.stack(tuple(item.contiguous() for item in grad_outputs))
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=ctx.group)
        return reduced[ctx.rank], None


def all_gather_tensor(
    tensor: torch.Tensor,
    *,
    group: Any = None,
) -> torch.Tensor:
    """Gather a possibly uneven leading dimension while preserving gradients.

    In single-process or uninitialized-distributed execution this returns the
    input tensor itself.  Non-leading dimensions must agree across ranks.
    """
    if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
        raise ValueError("all_gather_tensor expects a tensor with a batch dimension")
    if not distributed_is_active(group):
        return tensor

    world_size = dist.get_world_size(group=group)
    local_shape = tuple(tensor.shape[1:])
    gathered_shapes: list[tuple[int, ...] | None] = [None] * world_size
    dist.all_gather_object(gathered_shapes, local_shape, group=group)
    if any(shape != local_shape for shape in gathered_shapes):
        raise ValueError("Non-leading tensor dimensions differ across ranks")

    size = torch.tensor([tensor.shape[0]], dtype=torch.int64, device=tensor.device)
    gathered_sizes = [torch.zeros_like(size) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, size, group=group)
    sizes = [int(item.item()) for item in gathered_sizes]
    maximum = max(sizes)
    if maximum == 0:
        return tensor
    if tensor.shape[0] < maximum:
        padding = tensor.new_zeros((maximum - tensor.shape[0], *tensor.shape[1:]))
        padded = torch.cat((tensor, padding), dim=0)
    else:
        padded = tensor
    gathered = _AllGather.apply(padded, group)
    return torch.cat(
        [chunk[:item_size] for chunk, item_size in zip(gathered, sizes)],
        dim=0,
    )


def all_gather_values(values: list[Any], *, group: Any = None) -> list[Any]:
    """Gather per-example labels in the same rank order as tensors."""
    if not distributed_is_active(group):
        return list(values)
    world_size = dist.get_world_size(group=group)
    gathered: list[list[Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, list(values), group=group)
    return [value for rank_values in gathered for value in (rank_values or [])]
