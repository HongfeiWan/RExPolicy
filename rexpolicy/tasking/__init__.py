"""Declarative, auditable task contracts and authoring boundaries."""

from .model import TaskSpecV2, decode_task_spec_json

__all__ = ["TaskSpecV2", "decode_task_spec_json"]
