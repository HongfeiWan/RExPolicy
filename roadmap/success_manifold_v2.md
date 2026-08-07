# RExPolicy v2: Success Manifold Guided Flow-DiT

Canonical roadmap position: **Phase 2**.

The independent, state-only [Stage 0 validation
harness](stage0_success_manifold_validation.md) is preserved as isolated
research infrastructure. Its results do not activate Phase 2 or replace the
canonical phase gates.

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
- **SM-R:** planned progress-aligned, frequency-selective success-memory
  correction; no implementation or runtime evidence exists yet.

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

## SM-R research candidate — retrieve in time, correct in frequency

[Retrieve in Time, Correct in
Frequency](https://arxiv.org/html/2608.04527v1) demonstrates a complementary,
training-free test-time mechanism: align the current execution history to
progress within complete successful trajectories, then apply a bounded
low-frequency action correction to a frozen policy proposal. RExPolicy should
test that mechanism as **SM-R**, not relabel it as implemented SM-6 and not use
it to replace the learned Success Manifold.

The paper's matched 2,000-episode conditions improved aggregate LIBERO success
from 86.4% to 88.4% and LIBERO-Long from 61.6% to 68.6%. Its history-free
Frame-NN and Time-Domain ablations reached only 63.6% and 50.0% on Long, while
the full method still regressed on three of ten Long tasks. This is evidence
for testing both ideas together, not evidence that they transfer automatically
to Newton, GR00T, the 19D action schema, or this flywheel.

SM-R has three strict boundaries:

1. **Retrieval proposes; it does not judge.** Newton, the sealed TaskSpec, and
   the existing safety contract remain the sole authorities for executed
   success and failure. Similarity, alignment cost, and spectral residuals are
   never rewards, labels, or oracle inputs.
2. **Memory is train-derived and frozen for evaluation.** A bank generation
   contains only trajectories that were actually executed and independently
   verified successful during training or an explicitly designated
   memory-collection split. Validation, exposed or burned cohorts, selection
   data, and locked tests never enter the bank or tune it.
3. **Correction is bounded and abstaining.** An unavailable, invalid, or
   insufficiently confident match executes the exact frozen base proposal and
   logs an abstention. Configuration, fingerprint, cache, or integrity mismatch
   still fails closed; it must not masquerade as an ordinary no-match.

### Data and retrieval contract

The Episode Archive, Event Ledger, and committed `SuccessReference` remain the
source of truth. `SuccessExperienceGraph` may locate candidate trajectories,
but it remains metadata-only. Any descriptor, PCA, alignment, DCT, or action
cache is a reproducible derived artifact bound to the source identities,
schemas, normalization, policy checkpoint, task, simulator, and SHA-256. It has
no independent authority.

For each action chunk, SM-R would retain a causal descriptor history and use a
progressive, monotonic alignment frontier over every eligible successful
trajectory. The alignment may advance only within a pre-registered maximum
progress jump and pays a pre-registered jump penalty. Retrieval returns both a
trajectory identity and aligned progress; it is not a single-frame nearest
neighbor. A history-free frame-nearest-neighbor variant is a mandatory
ablation, not an acceptable substitute for the full method.

Online retrieval may read only the same causal observation or state fields
available to the frozen policy. Reset and trajectory identities, reward,
outcome, future state, oracle fields, authoring metadata, and hand-authored task
phase labels are forbidden retrieval inputs. Success is used once to admit a
completed training trajectory to the positive bank; it is not exposed to the
retriever during execution.

The first Stage 0 analogue uses normalized state descriptors. The full Phase 3
version must derive frozen visual descriptors from the policy observation
contract and bind camera, crop, scene, and encoder fingerprints. The state-only
result cannot validate image retrieval, `scene.glb` grounding, or camera
robustness.

### Spectral correction and execution contract

The insertion point is deliberately downstream of policy generation and
upstream of existing physical safeguards:

```text
frozen Flow-DiT proposal
  -> causal progress-aligned successful-memory lookup
  -> clipped low-frequency motion residual in action DCT space
  -> inverse DCT
  -> action denormalization / model-to-physical conversion
  -> existing Newton projection and safety checks
  -> same-state candidate execution
  -> authoritative TaskSpec outcome
```

The DCT residual masks out DC, high frequencies, event-like channels, and all
channels not explicitly admitted as continuous motion. Each admitted
coefficient is clipped before a bounded scale is applied. The descriptor
version, normalization/PCA, bank manifest, maximum progress jump, jump penalty,
cutoff frequency, residual scale, coefficient clip, confidence/abstention
rule, and channel mask must all be pre-registered and checksum-bound before a
fresh validation cohort is opened.

For the first normalized Stage 0 action chunk `[B, H=8, D=19]`, the conservative
proposal admits only XYZ translation channels `0:3` and frequencies strictly
above DC and below the registered cutoff. Rotation channels and hand joints
`9:19` remain untouched. Finger correction is a separate research hypothesis,
because preserving the learned close/open event decision is safer than
borrowing a grasp command from a merely similar trajectory.

During flywheel collection the corrected chunk is one explicit same-state K
candidate; it does not replace all candidates and cannot bypass projection or
safety. Only its actually executed, post-projection actions may become training
targets. A corrected trajectory may enter the **next** bank generation only
after Newton verifies success and the normal provenance, novelty, coverage,
and generation-boundary checks pass. Failed and timed-out trajectories remain
in corpus accounting but never become positive memory.

### Matched evaluation and activation ladder

SM-R requires a fixed-bank, fixed-checkpoint comparison using identical reset
groups, noise seeds, rollout budgets, and safety/oracle contracts:

1. frozen policy with no retrieval or correction;
2. history-free frame-nearest-neighbor retrieval with the same bounded spectral
   correction;
3. progress alignment with a time-domain or full-band residual ablation;
4. full progress alignment plus bounded low-frequency motion correction.

Report task/reset-level success, safety, integrity and oracle disagreement,
rescues and regressions relative to the frozen baseline, abstention rate,
alignment coverage, CPU latency, and memory size. Reset groups, rather than
individual rollout cells, are the scientific units. Paired rescue/regression
transitions are descriptive unless the simulator restart is proven byte-exact.

The activation ladder is:

1. freeze and audit one content-addressed train-success bank;
2. run progress alignment and frame-nearest-neighbor retrieval in shadow mode;
3. run spectral correction in shadow mode and verify the motion-only, no-DC,
   clipped residual contract;
4. freeze the bank and every hyperparameter, then open one fresh independent
   matched-evaluation cohort;
5. require zero safety, integrity, and oracle-disagreement events, a
   pre-registered net-success improvement and regression cap, evidence that the
   full method beats its ablations, and pre-registered CPU latency/memory
   budgets;
6. only then admit SM-R as a bounded candidate-world option at a clean
   generation boundary; evaluate sealed K=1 non-regression separately and
   never mutate memory during evaluation.

Threshold values belong in the future implementation manifest, not in this
planning document. A provider or language model may propose an offline
experiment, but it may not choose bank members, advance an alignment frontier,
change correction parameters, or decide online admission.

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
- SM-R progress-aligned retrieval, spectral correction, fixed-bank ablations,
  and activation evidence.

Diversity/progress shaping should first run as a logged shadow diagnostic. It
may enter selection only after proving that simulator-defined task success and
safety do not regress. Successor representation is optional research work and
should be added only if it improves held-out mode reachability or data
efficiency beyond this simpler latent contract.
