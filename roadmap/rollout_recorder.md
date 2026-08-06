# Deterministic read-only rollout recorder

`tools.run_rollout_recorder` produces visible policy evidence without entering
the training data plane. It loads the same Newton Reach environment and
`GrootFlowDitPolicy` used by the flywheel, but it does not construct DDP, an
optimizer, a trainer, `RunOperations`, an Event Ledger, or a Success Archive.
After an optional DiT overlay is loaded, every model parameter is frozen and
the rollout runs under `torch.inference_mode()`.

The only files it creates are three MP4 views per recorded world, an
`INCOMPLETE` marker while recording is in progress, and `summary.json` after a
successful close. An existing output directory is never overwritten.

## Node3 CUDA 1 command

`CUDA_VISIBLE_DEVICES=1` exposes physical GPU 1 as logical `cuda:0` inside the
process. Keep `--device cuda:0`; using `--device cuda:1` with that environment
would incorrectly request a second visible device.

Record the immutable base policy at `H=2`, `K=1`:

```bash
cd /home/user/project/RExPolicy

CUDA_VISIBLE_DEVICES=1 \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 \
python -m tools.run_rollout_recorder \
  --device cuda:0 \
  --isaac-groot-root /home/user/project/Isaac-GR00T \
  --policy-checkpoint /home/user/project/Isaac-GR00T/checkpoints/finetune/checkpoint-400000 \
  --vlm-model /home/user/project/Isaac-GR00T/checkpoints/nvidia/Cosmos-Reason2-2B \
  --output-dir /home/user/runs/rexpolicy/rollouts/base-r20260722-d20260723-h2 \
  --reset-seed 20260722 \
  --diffusion-seed 20260723 \
  --num-envs 1 \
  --episode-control-steps 8 \
  --execution-horizon 2 \
  --camera-textures \
  --scene-visuals \
  --scene-glb /home/user/project/RExPolicy/scene/scene.glb \
  --no-capture-graph \
  --no-hydroelastic
```

Record a completed training generation with the exact same reset and diffusion
seeds:

```bash
cd /home/user/project/RExPolicy

CUDA_VISIBLE_DEVICES=1 \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 \
python -m tools.run_rollout_recorder \
  --device cuda:0 \
  --isaac-groot-root /home/user/project/Isaac-GR00T \
  --policy-checkpoint /home/user/project/Isaac-GR00T/checkpoints/finetune/checkpoint-400000 \
  --vlm-model /home/user/project/Isaac-GR00T/checkpoints/nvidia/Cosmos-Reason2-2B \
  --dit-overlay /home/user/runs/rexpolicy/node3-train/checkpoints/generation-000100 \
  --output-dir /home/user/runs/rexpolicy/rollouts/generation-000100-r20260722-d20260723-h2 \
  --reset-seed 20260722 \
  --diffusion-seed 20260723 \
  --num-envs 1 \
  --episode-control-steps 8 \
  --execution-horizon 2 \
  --camera-textures \
  --scene-visuals \
  --scene-glb /home/user/project/RExPolicy/scene/scene.glb \
  --no-capture-graph \
  --no-hydroelastic
```

A directory named `generation-NNNNNN` is accepted only after the existing
`CheckpointManager.verify()` validates its `COMPLETE` marker and every
recorded checksum. The recorder may inspect a verified rejected candidate—the
summary preserves its `accepted` value—but it never promotes or resumes that
checkpoint. A standalone overlay directory containing `model.safetensors` or
`diffusion_pytorch_model.safetensors` is also supported and is SHA-256 bound in
the summary.

## Comparison protocol

For a useful before/after comparison, vary exactly one of these inputs at a
time:

- omit versus set `--dit-overlay`;
- use the same `--reset-seed`, `--diffusion-seed`, task, physics, cameras,
  episode length, and execution horizon;
- use one of the supported execution horizons: `H=1`, `2`, `4`, or `8`;
- keep `K=1` for a clean policy rollout. `K=2` through `8` are available for
  inspecting independent diffusion samples, with separate videos per world.

The diffusion seed is applied after model/environment initialization and
immediately before reset, matching the held-out inference contract. The
summary records every control-step distance and latched success, safety,
termination, and truncation status. Videos contain the reset frame followed by
one frame per completed 10 Hz control step; they are not 60 Hz physics videos.

Output for `K=1`:

```text
rollout-directory/
  ego.mp4
  wrist.mp4
  side_by_side.mp4
  summary.json
```

For `K>1`, the three videos live under `world-000/`, `world-001/`, and so on.
Each video overlays control step, reach distance, success, safety failure,
termination, and truncation. `summary.json` additionally declares
`training_updates: 0` and `archive_writes: 0`.

OpenCV with an MP4 encoder is required in the Isaac-GR00T/Newton environment.
When `--scene-visuals` is enabled, the recorder validates the GLB header and
declared byte length before constructing Newton, then records the asset path,
size, and SHA-256 in `summary.json`. Missing or placeholder assets fail closed
instead of silently producing a gray background. Use `--no-scene-visuals`
only for an intentionally geometry-only comparison.

The CPU-only orchestration and side-effect boundary are covered by:

```bash
python -m unittest tools.test_rollout_recorder -v
```
