"""Command-line entry point for canonical alloy CMC workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from gcmc.workflows import AlloyCMCWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run canonical alloy CMC from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the alloy CMC YAML config.",
    )
    args = parser.parse_args()

    workflow = AlloyCMCWorkflow.from_yaml(args.config)
    workflow.run()


if __name__ == "__main__":
    main()
