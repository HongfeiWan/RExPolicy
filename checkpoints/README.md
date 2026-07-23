# Checkpoint layout

## Immutable base weights

The initial GR00T policy and Cosmos VLM are stored locally under:

```text
checkpoints/groot/checkpoint-200000/
checkpoints/nvidia/Cosmos-Reason2-2B/
```

They are ignored by Git and must be copied or downloaded separately on a new
machine. The runtime also accepts `GROOT_POLICY_CHECKPOINT` and
`GROOT_VLM_MODEL`. A flywheel run references these immutable artifacts by
resolved path and hash; it never duplicates the frozen Cosmos VLM.

## Resumable flywheel checkpoints

Each run owns a separate directory selected with `--run-dir`. Completed
generation checkpoints live under:

```text
<run-dir>/checkpoints/generation-NNNNNN/
```

A complete checkpoint contains:

- FP32 Flow-DiT weights;
- FP32 AdamW state stored on CPU;
- completed and next generation plus global optimizer step;
- per-rank Python, NumPy, Torch CPU, and Torch CUDA RNG state;
- each rank's committed Success Archive shard list and replay cursor;
- task, action-schema, configuration, code, asset, base-weight, and evaluation
  fingerprints;
- file checksums and a `COMPLETE` commit marker.

Checkpoint creation first writes a same-filesystem `.partial-*` directory,
flushes it, atomically renames it, and atomically updates `latest.json`.
`--resume-latest` ignores partial directories. `--resume-from` verifies
checksums and semantic compatibility before restoring any training state.
World size, task/action schema, base weights, and semantic training settings
must match; extending `--max-generations` and changing non-semantic logging
intervals are allowed.

Generation archives and checkpoints serve different purposes. Episode and
Success Archive JSONL shards retain replay provenance without images or video;
checkpoints retain exactly the model, optimizer, RNG, and cursors needed to
continue a run.
