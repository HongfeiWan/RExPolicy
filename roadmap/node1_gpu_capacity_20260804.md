# node1 GPU capacity and camera/pipeline roadmap (2026-08-04)

## Decision summary

The measured node is one NVIDIA RTX PRO 5000 72GB Blackwell GPU
(`73,415 MiB` reported total), with one flywheel rank.

- The current dual-camera Newton runner accepts **6,990 worlds** at the
  `640x480` wrist resolution. `6,991` is the first invalid world count because
  the flattened wrist-pixel dimension exceeds signed int32. The 6,990 result
  is a one-step boundary probe, not a stability recommendation.
- The useful raw-Newton throughput knee is **2,048--4,096 worlds**. At 4,096,
  the runner reached about `2,130 world-control-steps/s` with `32,089 MiB`
  peak GPU use. Moving to 6,990 gained only about 4.7% throughput while peak
  use rose to `55,387 MiB`.
- The complete single-rank flywheel capacity slice completed through **K=384,
  H=8**. This setting performs one decision and exercises model load, VLM/DiT,
  camera/physics collection, archive, coverage-driven training, and offload.
  It is **not the default H=2 multi-decision workload**.
- Use **K=256 as the production target/cap**, after the staged H=2 validation
  below. Its measured H=8 peak was `42,874 MiB`, leaving materially more
  operating margin than K=384 (`59,245 MiB`). K=512 was **not run**; a rough
  linear extrapolation from K=128/256/384 is about `76 GiB`, beyond this GPU.
- After canonical continuation witnesses were introduced, the default
  **K=2, H=2** run completed four decisions without the previous Event Ledger
  root-digest mismatch.

These are capacity measurements, not policy-quality or learning baselines.
No success was produced by the one-generation capacity runs.

## Measurement contract

`K` is `--candidates-per-state`; on this single-rank node it is also the
Newton world count. `H` is `--execution-horizon`. Raw Newton throughput is:

```text
world-control-steps/s = num_envs * completed batched env.step calls/s
```

One control step contains six 60 Hz simulation frames at a 10 Hz controller,
with 16 solver substeps per frame: 96 physics substeps per world-control-step.
All raw probes used both RGB cameras, `state_dict+rgb`, no scene visuals, no
hydroelastic contacts, no CUDA graph, and 16 substeps per simulation frame.

The flywheel measurements used task `reach_green_cap/v1`, one episode and one
generation, eight episode control steps, elite fraction 0.5, batch size 1,
gradient accumulation 2, optimizer-state offload, VLM offload during update,
and no evaluation or checkpoint save. Although requested training steps were
zero, current-generation coverage still trained all selected samples; the
reported effective optimizer steps are therefore nonzero.

## Raw Newton world x step curve

The timed section excludes environment construction and reset. GPU peaks were
sampled externally and should be treated as approximate for short probes.

| Worlds | Timed steps | Elapsed (s) | Batched steps/s | World-control-steps/s | Peak GPU (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 40 | 15.148 | 2.64 | 2.64 | 915 |
| 2 | 30 | 11.220 | 2.67 | 5.35 | 945 |
| 4 | 30 | 11.362 | 2.64 | 10.56 | 945 |
| 8 | 30 | 11.306 | 2.65 | 21.2 | 945 |
| 16 | 30 | 11.718 | 2.56 | 40.96 | 977 |
| 32 | 30 | 11.781 | 2.55 | 81.6 | 1,105 |
| 64 | 30 | 12.399 | 2.42 | 154.9 | 1,361 |
| 128 | 20 | 8.891 | 2.25 | 288.0 | 1,841 |
| 256 | 20 | 10.084 | 1.98 | 506.9 | 2,801 |
| 512 | 12 | 7.289 | 1.65 | 844.8 | 4,753 |
| 1,024 | 8 | 6.362 | 1.26 | 1,290.2 | 8,659 |
| 2,048 | 5 | 5.817 | 0.86 | 1,761.3 | 16,469 |
| 4,096 | 3 | 5.745 | 0.52 | 2,129.9 | 32,089 |
| 6,990 | 1 | 3.134 | 0.32 | 2,230.4 | 55,387 |

At 4,096 worlds this is about `204,470 physics world-substeps/s`; the one-step
6,990 probe is about `214,118 physics world-substeps/s`. The 2,048--4,096
range is the recommended range for a longer raw simulator soak.

### Exact camera-array boundary

The wrist output has `640 * 480 = 307,200` pixels per world. A Warp array
dimension must fit signed int32:

```text
floor((2^31 - 1) / 307,200) = 6,990
6,990 * 307,200 = 2,147,328,000       # valid
6,991 * 307,200 = 2,147,635,200       # exceeds 2,147,483,647
```

Thus 6,990 succeeded and 6,991 is the first invalid configuration. An earlier
8,192 launch reached the same underlying failure with a flattened dimension
of `2,516,582,400`. Current code fails fast before allocating an oversized
camera launch (commit `346a69d`). This is a representation limit, not a GPU
memory OOM.

The raw probes wrote timing to stdout only. Temporary monitor logs were
removed, so there is no formal raw-run directory; the table above is the
retained summary. Do not treat the single-step 6,990 point as a soak result.

### Camera microbenchmarks and canonicalization copy

On the same GPU, synchronized `_render_cameras()` measurements at 256 worlds
gave the following means after warmup:

| Wrist resolution | Renderer order | Dual-camera mean |
| ---: | --- | ---: |
| 640x480 | pixel priority | 103.0 ms |
| 640x480 | view priority | 90.4 ms |
| 320x240 | pixel priority | 41.65 ms |
| 320x240 | view priority | 41.42 ms |

At the current pixel-priority default, 320x240 reduced render time by 59.6%
and was 2.47x faster. This is only a performance result: changing resolution
changes the visual/render contract and requires the held-out non-inferiority
gate below. Renderer order also needs scale-specific autotuning: at 32 worlds,
pixel priority was faster (`9.75 ms` versus `14.69 ms`), whereas view priority
was about 12% faster at 256 worlds and 640x480. Disabling textures improved
render-only time by just 2.5--3.4%, so it is not a useful first trade.

Packed uint32 and unpacked RGB buffers for the current two cameras consume
`2,553,600 bytes/world`, or about 16.62 GiB at 6,990 worlds. Before commit
`81a6690`, canonicalization cloned an already expanded K-world view. A
1024-world wrist-RGB copy microbenchmark measured `900.0 MiB` extra allocation
and `3.375 ms` median with the old order, versus `0.879 MiB` and `1.691 ms`
after cloning one source row before expansion; outputs were bitwise exact.

## Complete flywheel H=8 capacity slice

All rows completed generation 1 with exit status 0. `CUDA reserve` is the
framework high-water mark; `GPU peak` is the one-second external monitor and
includes other process allocations. Wall time includes model/runtime startup.

| K | Decisions | Collect (s) | Update (s) | Selected | Effective optimizer steps | CUDA reserve (GiB) | GPU peak (GiB) | Peak GPU util | Wall (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 3.49 | 1.99 | 1 | 1 | 18.23 | 19.99 | 44% | 38.29 |
| 4 | 1 | 3.45 | 1.84 | 2 | 1 | 18.23 | 19.16 | 47% | 38.92 |
| 8 | 1 | 3.60 | 2.03 | 4 | 2 | 20.08 | 21.00 | 36% | 38.83 |
| 16 | 1 | 3.86 | 2.31 | 8 | 4 | 20.23 | 21.19 | 88% | 37.98 |
| 32 | 1 | 4.37 | 2.88 | 16 | 8 | 20.23 | 21.31 | 90% | 39.03 |
| 64 | 1 | 5.57 | 4.05 | 32 | 16 | 20.23 | 21.53 | 100% | 41.35 |
| 128 | 1 | 7.89 | 6.38 | 64 | 32 | 23.53 | 25.31 | 100% | 46.80 |
| 256 | 1 | 12.91 | 12.81 | 128 | 64 | 38.78 | 41.87 | 100% | 59.16 |
| 384 | 1 | 17.81 | 15.63 | 192 | 96 | 54.04 | 57.86 | 100% | 68.19 |

The H=8 results establish a one-decision memory envelope. They do not include
the default H=2 reset-and-replay cadence, and K=384 has only a one-generation
completion, not a leak or thermal soak. K=256 is the production target because
it preserves roughly 30 GiB of externally observed headroom for allocator
variation, longer runs, evaluation, checkpointing, and monitoring.

## Default H=2 correctness result

The final combined run `k2-h2-camera-fixed` used K=2, H=2 and eight episode
control steps on commit `346a69d`. It completed:

- four decisions and eight candidate chunks;
- `8.31 s` collection and `1.98 s` update;
- four selected chunks and two effective optimizer steps;
- `20.08 GiB` CUDA reserved and `21.00 GiB` externally observed peak;
- `41.36 s` wall time with exit status 0;
- exact digest and signal equality at all three decision boundaries.

Commit `5ab2b7b` fixed the earlier digest mismatch without weakening the
fail-closed ledger: the reset-and-replayed canonical state is promoted as the
selected branch's continuation witness and cached as the exact next root.
Commit `81a6690` removed redundant branch-canonicalization copies, and commit
`346a69d` added the camera int32 fail-fast. The final combined H=2 run above
and the full 476-test suite passed after all three changes.

Only H=2 K=2 has completed after the correctness fix. Before calling K=256 a
validated default production configuration, run H=2 at K=16, 64, 128, and 256,
then a 10-generation K=256 soak. Stop escalation if any gate below fails.

## GPU and camera optimization order

### Low-risk: remove synchronization and transfer waste

1. **Keep conditions on GPU.** `_raw_observations` currently materializes
   state and both cameras on CPU; `encode_conditions` then moves frozen VLM
   outputs to CPU, and `_collate_conditions` copies them back to CUDA. Keep
   backbone features, masks, and state device-resident through DiT sampling.
2. **Keep actions on GPU until the boundary.** The sampled action currently
   travels CUDA -> NumPy/CPU -> CUDA before `env.step`. Add a tensor-native
   decode/validation path and copy only the compact ledger payload to host.
3. **Pack audit scalars into one asynchronous transfer.** Replace repeated
   `.cpu().tolist()` and full dynamics-tree NumPy conversions with one compact
   GPU digest/metric buffer and a pinned, nonblocking D2H transfer per control
   step. Preserve the existing canonical digest definition.
4. **A/B offload flags one at a time.** At K=256, optimizer and VLM offload
   took about `1.65 s` and `0.80 s`. Test keeping only one component resident;
   do not assume both fit. Promote only if K=256 remains below `60 GiB` peak
   and median generation time improves by at least 5% over three warm runs.

Gate: fixed-seed H=2 K=2 must produce the same selected sample IDs, actions
(within the declared numeric tolerance), Event Ledger chain, archive facts,
and terminal signals. H=8 K=64 and K=256 must improve median collection time
by at least 10% with no more than 2 GiB extra peak memory and no update-time
regression above 5%.

### Medium-risk: reduce camera and batching cost

1. **A/B the wrist camera at 320x240 or 342x256.** The current wrist render is
   640x480, while the policy processor resizes/crops around 256 pixels. A
   320x240 render has one quarter of the wrist pixels and raises the wrist-only
   int32 ceiling to about 27,962 worlds, although another array may become the
   next limit. Keep the 320x180 ego camera unchanged in the first experiment;
   its short edge is currently upsampled rather than discarded.
2. **Avoid packed-RGBA -> duplicate RGB output when possible.** Let policy
   preprocessing consume the renderer's packed device buffer, or fuse unpack,
   layout conversion, normalization, and resize into one kernel.
3. **Microbatch DiT candidates independently of VLM encoding.** Select the
   candidate microbatch from measured free memory, keeping K=256 as the logical
   decision width. This provides a bounded-memory fallback without changing
   candidate seeds or selection semantics.

Gate for resolution changes: update the render contract intentionally, pass a
paired held-out suite using identical reset recipes and diffusion seeds, and
then run a power-sized evaluation. The lower 95% confidence bound for success
non-inferiority must be no worse than -2 percentage points, with zero new
process/safety violations. Include a hand/cap occlusion subset and compare
stage transitions, not just aggregate reward. Processor features and decoded
actions should also be checked for drift before policy-quality promotion.

### Structural: exploit same-state branching and replay semantics

1. **Encode and render the canonical root once.** All K worlds start a decision
   from the same canonical state, but the current path preprocesses K duplicate
   image/state/instruction records through the frozen VLM. Render world 0 once,
   encode one condition, and expand/share immutable embeddings across K while
   retaining independent DiT noise and candidate latent tokens.
2. **Add physics-only canonical replay.** For an eight-step H=2 episode with
   four decisions, branch execution is eight batched control steps and replay
   adds `0 + 2 + 4 + 6 = 12`. Replay currently renders cameras at intermediate
   actions even though only the final canonical root image is consumed. Step
   physics without rendering, then render once at the final root.
3. **Fuse/overlap the camera pipeline.** `_render_cameras` refits the BVH,
   launches separate ego/wrist transform kernels, updates two tiled cameras
   sequentially, and launches separate unpack kernels. After proving sensor
   stream safety, fuse transforms/unpack and overlap independent camera passes.
4. **Tile only if workloads truly need more than 6,990 worlds.** Chunked camera
   rendering can avoid the per-array int32 dimension, but it is unnecessary for
   the K=256 production target and should follow the higher-return work above.
5. **Apply CUDA graphs last.** Capture stable physics or DiT-denoise segments
   only after CPU synchronizations and dynamic host branches are removed.

Gate: repeated-condition versus shared-condition features must be bitwise equal
or within a documented tolerance, candidate seeds and diversity must remain
unchanged, and no expanded tensor may be mutated in place. Physics-only replay
must reproduce canonical dynamics digests, signals, terminations, and final
root images for H=2 K=2 before scaling. Use an Nsight trace to demonstrate
fewer launches/synchronizations; require at least 15% median H=2 collection
improvement at K=64 before merging structural camera-stream work.

## Release ladder

Every optimization should pass this sequence before the next K increase:

1. Unit/contract tests, including camera bounds, Event Ledger continuation,
   replay handoff, render contract, and deterministic task evaluation.
2. Fixed-seed H=2 K=2 parity run; reject any unexplained digest, selection,
   reward/signal, termination, or archive difference.
3. Three warm H=8 K=64 and K=256 runs; report median and range for collection,
   update, CUDA allocated/reserved, external peak, utilization, power, and host
   RSS. A one-second GPU sample alone is not sufficient.
4. H=2 staged capacity runs at K=16, 64, 128, and 256.
5. Ten generations at H=2 K=256 with peak GPU use at or below `60 GiB`, no
   Xid/OOM/NaN/Inf, and no growth above 1 GiB between early and late comparable
   generations.
6. Paired held-out quality gates before any camera-resolution or precision
   change becomes the default.

## Reproduction and artifacts

The retained flywheel parent directory on node1 is:

```text
/home/user/runs/rexpolicy/node1-capacity-20260804
```

Its successful subdirectories are `k2-h8`, `k4-h8`, `k8-h8-r2`, `k16-h8`,
`k32-h8`, `k64-h8`, `k128-h8`, `k256-h8`, `k384-h8`, and `k2-h2-fixed`.
The final post-camera-optimization correctness run is `k2-h2-camera-fixed`.
Each contains `run_manifest.json`, `metrics.jsonl`, `gpu_metrics.csv`,
`gpu_processes.csv`, and the rank exit record; console timing is stored beside
the directory where present. Use `k8-h8-r2`, not the aborted pre-launch
`k8-h8` attempt.

Run raw Newton from the configured Isaac-GR00T virtual environment:

```bash
cd /home/user/project/RExPolicy
python -m tools.run_newton_groot_rl_env \
  --device cuda:0 \
  --num-envs "$WORLDS" \
  --steps "$TIMED_STEPS" \
  --obs-mode state_dict+rgb \
  --substeps-per-frame 16 \
  --no-scene-visuals \
  --no-hydroelastic \
  --no-capture-graph
```

The H=8 flywheel command shape was:

```bash
cd /home/user/project/RExPolicy
CUDA_VISIBLE_DEVICES=0 REXPOLICY_NPROC=1 \
  tools/launch_flywheel_ddp.sh \
  --isaac-groot-root /home/user/project/Isaac-GR00T \
  --policy-checkpoint /home/user/project/Isaac-GR00T/checkpoints/finetune/checkpoint-400000 \
  --vlm-model /home/user/project/Isaac-GR00T/checkpoints/nvidia/Cosmos-Reason2-2B \
  --run-dir "/home/user/runs/rexpolicy/node1-capacity-20260804/k${K}-h8" \
  --task-id reach_green_cap/v1 \
  --max-generations 1 \
  --candidates-per-state "$K" \
  --episodes-per-generation 1 \
  --episode-control-steps 8 \
  --execution-horizon 8 \
  --chunk-elite-fraction 0.5 \
  --train-steps-per-generation 0 \
  --train-batch-size 1 \
  --gradient-accumulation 2 \
  --success-replay-per-rank 0 \
  --no-save \
  --eval-every 0 \
  --no-eval-at-start \
  --no-eval-at-end \
  --gpu-monitor-seconds 1 \
  --camera-textures \
  --no-scene-visuals \
  --no-capture-graph \
  --no-hydroelastic \
  --bottle-settle-frames 60 \
  --substeps-per-frame 16
```

For the final default correctness run, set the directory to
`k2-h2-camera-fixed`, K to 2, and `--execution-horizon 2`. H=8 measurements
used RExPolicy commit `2e41e00`;
the first repaired H=2 result used `5ab2b7b`, and the final combined H=2 run
used `346a69d`. Re-run the release ladder on the final production commit rather
than comparing results across unrecorded source changes.
