"""Command-line entry point for alloy parallel-tempering CMC workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from gcmc.workflows import AlloyReplicaExchangeWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run alloy parallel-tempering CMC from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the alloy PT YAML config.",
    )
    args = parser.parse_args()

    workflow = AlloyReplicaExchangeWorkflow.from_yaml(args.config)
    workflow.run()


if __name__ == "__main__":
    main()
