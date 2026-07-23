#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${REXPOLICY_NPROC:-}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a rexpolicy_visible_devices <<<"${CUDA_VISIBLE_DEVICES}"
        REXPOLICY_NPROC="${#rexpolicy_visible_devices[@]}"
    else
        REXPOLICY_NPROC="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    fi
fi
if [[ "${REXPOLICY_NPROC}" -lt 1 ]]; then
    echo "No visible NVIDIA GPU was found" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"

exec python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${REXPOLICY_NPROC}" \
    -m tools.run_flywheel_ddp \
    "$@"
