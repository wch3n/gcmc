"""Command-line entry point for canonical adsorbate CMC workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from gcmc.workflows import AdsorbateCMCWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a canonical adsorbate CMC workflow from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the canonical adsorbate CMC YAML config.",
    )
    args = parser.parse_args()

    workflow = AdsorbateCMCWorkflow.from_yaml(args.config)
    workflow.run()


if __name__ == "__main__":
    main()
