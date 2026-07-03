"""Command-line entry point for adsorbate GCMC scan workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from gcmc.workflows import AdsorbateGCMCScanWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run adsorbate GCMC scan or mu-exchange from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the adsorbate GCMC scan YAML config.",
    )
    args = parser.parse_args()

    workflow = AdsorbateGCMCScanWorkflow.from_yaml(args.config)
    workflow.run()


if __name__ == "__main__":
    main()
