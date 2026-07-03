"""Command-line entry point for adsorbate GCMC workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from gcmc.workflows import AdsorbateGCMCWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run adsorbate GCMC from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the adsorbate GCMC YAML config.",
    )
    args = parser.parse_args()

    workflow = AdsorbateGCMCWorkflow.from_yaml(args.config)
    workflow.run()


if __name__ == "__main__":
    main()
