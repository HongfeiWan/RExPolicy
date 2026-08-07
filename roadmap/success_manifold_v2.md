# RExPolicy v2: Success Manifold Guided Flow-DiT

Canonical roadmap position: **Phase 2**.

Status: **implementation components SM-1 through SM-5 are present behind
default-off contracts; Phase 2 is not runtime-validated or active**.

The phase order and activation status are governed by
[README.md](README.md). `RExPolicy v2` is the architecture name for this phase,
not a parallel delivery route.

When Phase 2 is enabled after acceptance, RExPolicy v2 adds an explicit
distribution over simulator-verified successful futures to the Phase 1 learner.
It does not replace the canonical success predicate or create another reward
oracle. Newton and the sealed task contract still decide whether an executed
trajectory succeeds; the latent only tells Flow-DiT which reachable kind of
successful continuation to attempt.

The v1 path remains the compatibility baseline. No v2 component runs unless a
caller either enables graph publication or supplies a checksum-pinned Success
Manifold deployment bundle.

The implementation labels used below are:

- **SM-1:** successful-future graph and window materialization;
- **SM-2:** future encoder and state-to-mode selector training;
- **SM-3:** diagnostic latent projection;
- **SM-4:** Flow-DiT success-token conditioning;
- **SM-5:** simulator-verified bounded online latent memory.

These are Phase 2 component labels, not global roadmap phases.

## SM-1 - data and trust boundary

The immutable Episode Archive, Event Ledger, and committed SuccessReferences
remain the source of truth. `SuccessExperienceGraph` is a derived,
metadata-only view over those records. It contains content fingerprints,
transition locators, path membership, and fixed-offset window policy, but never
persists images, rendered observations, raw state tensors, frozen VLM
features, actions, rewards, or scores.

Graph compilation fails closed unless all of the following agree:

- episode, ledger, and SuccessReference identities and fingerprints;
- selected continuation prefixes and terminal success paths;
- task, reward, simulator, policy, generation, rank, and episode provenance;
- requested horizon, action horizon, and stride;
- content-addressed derived-view descriptor and archive pair binding.

Materialization reconstructs current state witnesses, the executed 19D action
window, and future witnesses from the exact bound ledger. Fixed transition
offsets are used in this MVP because the current ledger does not yet contain a
sealed gripper/EEF-velocity milestone schema. A heuristic milestone fallback
would make the representation contract ambiguous and is therefore rejected.

## SM-2 - learned representation

`FutureTrajectoryEncoder` is a Transformer encoder over current state,
executed action, future state, and optional visual features. It emits one
normalized success latent and owns a decoder for self-supervised masked
reconstruction. Training combines:

- masked reconstruction of state/action/visual tokens;
- multi-positive InfoNCE, with positives defined by caller-supplied successful
  trajectory groups;
- per-dimension variance floors to prevent collapse;
- off-diagonal covariance regularization to discourage redundant dimensions.

Distributed contrastive gathering supports unequal rank-local batch sizes and
preserves local gradients. The configuration is strict, versioned, rejects
unknown keys, and is disabled by default.

`SuccessModeSelector` maps the flattened current GR00T state to a diagonal
Gaussian over success latents. Its supervised target is the detached latent of
a verified future window. Selector training combines Gaussian negative
log-likelihood and cosine alignment. It therefore learns reachability from the
present state instead of inventing a new reward model.

`SuccessManifoldDdpTrainer` exposes two explicit stages:

1. `train_encoder_batch` optimizes the future encoder/decoder objectives.
2. `train_selector_batch` freezes the target encoder representation and trains
   state-to-mode prediction.

Optimizer state, global steps, and stage counters are resumable. Export writes
only the encoder and selector inference weights plus their complete config to
an atomic deployment bundle and returns its SHA-256.

## SM-3 - diagnostic projection

`tools/project_success_manifold.py` validates a content-addressed latent
dataset and produces a two-dimensional projection plus SVG. PCA is
deterministic; UMAP and t-SNE are optional lazy dependencies. Projection
records are marked `diagnostic_only` and bind both dataset and projection
policy fingerprints. They cannot be consumed as training labels, rewards, or
activation evidence.

## SM-4 - Flow-DiT conditioning

When enabled, the frozen encoder condition head projects one success latent to
the GR00T backbone width, and `GrootFlowDitPolicy` appends that token after
valid backbone tokens. The new token participates in attention but is not
marked as an image token. Flow-DiT remains the only online-trainable policy
component; the GR00T VLM, future encoder, and selector are frozen during
flywheel collection and policy updates.

Historical v1 samples may omit the token. The collator accepts mixed v1/v2
batches and applies success conditioning only where present. With conditioning
disabled, providing a token is a contract error rather than a silent no-op.

## SM-5 - online expansion

At each same-state branch point, the frozen selector samples one mode per
candidate world. An optional bounded number of worlds may instead reuse modes
from `LatentMemory`. The memory admits a latent only after Newton verifies the
corresponding rollout as successful, applies a novelty threshold, stores no
observation tensors, and is checkpointed per rank. Held-out K=1 evaluation
uses the selector but does not mutate memory.

The deployment bundle path and its lowercase SHA-256 are required together:

```bash
tools/launch_flywheel_ddp.sh \
  --success-manifold-bundle /path/to/success-manifold.pt \
  --expected-success-manifold-bundle-sha256 <sha256> \
  --manifold-memory-candidates 1 \
  --success-experience-graph \
  --success-future-horizon 8 \
  --success-action-horizon 2 \
  --success-window-stride 1 \
  --run-dir /path/to/new-run
```

`--manifold-memory-candidates` must be between zero and the number of
same-state candidates. Graph horizons and stride must be positive, and the
action horizon cannot exceed the future horizon. Omitting all v2 flags gives
the original v1 manifest, metrics, checkpoint, sampling, and training path.

## Phase 2 activation order

The safe activation order is:

1. publish and audit graphs without changing policy inputs;
2. reconstruct a real successful-window corpus and train the encoder offline;
3. inspect held-out reconstruction, retrieval, collapse, and projection
   diagnostics;
4. train the selector and measure held-out latent likelihood/alignment;
5. export and checksum-pin a bundle, then run conditioning in shadow mode;
6. run single-GPU smoke, resume equivalence, two-GPU node3 smoke, and the
   staged 10→20 generation non-regression soak;
7. activate bounded latent memory only after selector-only shadow metrics are
   stable.

Checkpoint or bundle incompatibility, non-finite values, dimensional mismatch,
archive drift, or fingerprint drift fails closed. No automatic fallback to an
unconditioned or differently conditioned policy is allowed inside an enabled
run.

## Deferred Phase 2 work

The following are not part of the implemented SM-1 through SM-5 code boundary
and must not be inferred from the presence of the modules above:

- diversity/progress reward shaping derived from latent occupancy;
- a successor-representation or successor-feature prediction head;
- milestone-based future windows before a sealed ledger milestone schema;
- automatic promotion of a trained bundle into active flywheel runs;
- corpus-scale Newton replay, GPU training, and node3 acceptance evidence.

Diversity/progress shaping should first run as a logged shadow diagnostic. It
may enter selection only after proving that simulator-defined task success and
safety do not regress. Successor representation is optional research work and
should be added only if it improves held-out mode reachability or data
efficiency beyond this simpler latent contract.
