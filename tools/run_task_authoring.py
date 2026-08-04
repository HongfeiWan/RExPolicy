#!/usr/bin/env python3
"""Production entry point for a pinned, resumable TaskSpec authoring job."""

from rexpolicy.tasking.authoring_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
