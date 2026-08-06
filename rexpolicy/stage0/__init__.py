"""State-only Success Manifold validation components.

The package root is deliberately dependency-free. Neural and Newton modules
are imported only by their dedicated entry points, never as an import side
effect of reading configuration or persisted metadata.
"""

from .config import (
    STAGE0_ACTION_DIM,
    STAGE0_CONFIG_SCHEMA_VERSION,
    STAGE0_STATE_DIM,
    STAGE0_STATE_SCHEMA_ID,
    Stage0Config,
    Stage0ExperimentConfig,
)
from .envs import (
    DEFAULT_STAGE0_STATE_FIELDS,
    DEFAULT_STAGE0_STATE_SCHEMA,
    FRAME_INDEPENDENT,
    JOINT_SPACE_FRAME,
    RIGHT_ROBOT_BASE_FRAME,
    STAGE0_STATE_DTYPE,
    STAGE0_STATE_SCHEMA_VERSION,
    Stage0StateField,
    Stage0StateSchema,
)
from .seed import (
    MAX_STAGE0_SEED,
    STAGE0_SEED_SCHEMA_VERSION,
    Stage0SeedSequence,
    derive_stage0_seed,
)
from .types import Stage0Outcome, Stage0SystemPath, Stage0Task


__all__ = [
    "DEFAULT_STAGE0_STATE_FIELDS",
    "DEFAULT_STAGE0_STATE_SCHEMA",
    "FRAME_INDEPENDENT",
    "JOINT_SPACE_FRAME",
    "MAX_STAGE0_SEED",
    "RIGHT_ROBOT_BASE_FRAME",
    "STAGE0_ACTION_DIM",
    "STAGE0_CONFIG_SCHEMA_VERSION",
    "STAGE0_SEED_SCHEMA_VERSION",
    "STAGE0_STATE_DIM",
    "STAGE0_STATE_DTYPE",
    "STAGE0_STATE_SCHEMA_ID",
    "STAGE0_STATE_SCHEMA_VERSION",
    "Stage0Config",
    "Stage0ExperimentConfig",
    "Stage0Outcome",
    "Stage0SeedSequence",
    "Stage0StateField",
    "Stage0StateSchema",
    "Stage0SystemPath",
    "Stage0Task",
    "derive_stage0_seed",
]
