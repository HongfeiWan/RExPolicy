#!/usr/bin/env python3
"""Exercise the exact NCCL and CUDA DDP path before loading GR00T."""

from __future__ import annotations

import argparse
import json
import os
import time

from rexpolicy.flywheel.distributed import DistributedContext
from rexpolicy.flywheel.operations import preflight_selected_gpus


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large-mib", type=int, default=256)
    parser.add_argument("--large-rounds", type=int, default=5)
    parser.add_argument("--barrier-rounds", type=int, default=5)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if min(args.large_mib, args.large_rounds, args.barrier_rounds) < 1:
        raise ValueError("All NCCL preflight dimensions must be positive")

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if local_rank >= len(tokens):
            raise RuntimeError("LOCAL_RANK exceeds CUDA_VISIBLE_DEVICES")
        physical_gpu = tokens[local_rank]
    else:
        physical_gpu = local_rank
    preflight_selected_gpus(
        [physical_gpu],
        minimum_free_mib=2048,
        allowed_pids=[os.getpid()],
    )

    context = DistributedContext.initialize(timeout_minutes=1)
    try:
        if context.world_size < 2:
            raise RuntimeError("NCCL preflight requires at least two ranks")
        if torch.cuda.current_device() != context.local_rank:
            raise RuntimeError("Rank is mapped to the wrong CUDA device")
        worker_pids = [
            int(pid) for pid in context.all_gather_objects(os.getpid())
        ]
        preflight_error = None
        try:
            preflight_selected_gpus(
                [physical_gpu],
                minimum_free_mib=2048,
                allowed_pids=worker_pids,
            )
        except Exception as error:
            preflight_error = f"{type(error).__name__}: {error}"
        errors = [
            error
            for error in context.all_gather_objects(preflight_error)
            if error is not None
        ]
        if errors:
            raise RuntimeError("GPU safety preflight failed: " + "; ".join(errors))

        for _ in range(args.barrier_rounds):
            context.barrier()

        small = torch.tensor(
            float(context.rank + 1),
            device=context.device,
            dtype=torch.float32,
        )
        dist.all_reduce(small)
        expected = context.world_size * (context.world_size + 1) / 2
        if float(small.item()) != expected:
            raise RuntimeError(
                f"Incorrect NCCL all-reduce: {small.item()} != {expected}"
            )

        elements = args.large_mib * 1024 * 1024 // 4
        large = torch.full(
            (elements,),
            float(context.rank + 1),
            device=context.device,
            dtype=torch.float32,
        )
        torch.cuda.synchronize(context.device)
        started = time.perf_counter()
        for _ in range(args.large_rounds):
            dist.all_reduce(large)
            large.div_(context.world_size)
        torch.cuda.synchronize(context.device)
        elapsed = time.perf_counter() - started
        if not torch.isfinite(large).all():
            raise FloatingPointError("Large NCCL all-reduce produced non-finite data")

        torch.manual_seed(1234)
        model = torch.nn.Linear(32, 16, bias=False, device=context.device)
        ddp = DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
        )
        optimizer = torch.optim.AdamW(ddp.parameters(), lr=1.0e-3)
        before = model.weight.detach().clone()
        input_tensor = torch.full(
            (8, 32),
            float(context.rank + 1),
            device=context.device,
        )
        loss = ddp(input_tensor).square().mean()
        loss.backward()
        optimizer.step()
        update = float((model.weight.detach() - before).abs().max().item())
        if not update > 0.0:
            raise RuntimeError("Tiny DDP optimizer produced a zero update")

        weights = model.weight.detach().reshape(-1)
        minimum = weights.clone()
        maximum = weights.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        sync_error = float((maximum - minimum).abs().max().item())
        if sync_error != 0.0:
            raise RuntimeError(f"Tiny DDP parameters diverged by {sync_error:.3e}")

        context.barrier()
        if context.is_main:
            gib_per_second = (
                args.large_mib
                * args.large_rounds
                * context.world_size
                / 1024.0
                / elapsed
            )
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "world_size": context.world_size,
                        "large_mib": args.large_mib,
                        "large_rounds": args.large_rounds,
                        "elapsed_seconds": elapsed,
                        "aggregate_gib_per_second": gib_per_second,
                        "tiny_ddp_update_max_abs": update,
                        "parameter_sync_max_abs": sync_error,
                        "nccl_environment": {
                            key: value
                            for key, value in os.environ.items()
                            if key.startswith("NCCL_")
                        },
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        context.close()


if __name__ == "__main__":
    main()
