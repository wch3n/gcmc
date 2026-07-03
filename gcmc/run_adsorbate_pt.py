"""Command-line entry point for adsorbate parallel-tempering CMC workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from gcmc.workflows import AdsorbateReplicaExchangeWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an adsorbate parallel-tempering CMC workflow from YAML."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the adsorbate PT YAML config.",
    )
    args = parser.parse_args()

    workflow = AdsorbateReplicaExchangeWorkflow.from_yaml(args.config)
    workflow.run()


if __name__ == "__main__":
    main()
