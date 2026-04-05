#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from gcmc.workflows import AlloyCMCWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run canonical alloy MC from a YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "alloy_cmc.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    workflow = AlloyCMCWorkflow.from_yaml(args.config)
    workflow.run()


if __name__ == "__main__":
    main()
