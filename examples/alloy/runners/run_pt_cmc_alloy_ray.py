#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from gcmc.workflows import AlloyReplicaExchangeWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Ray-backed alloy replica exchange from a YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "alloy_pt_ray.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    workflow = AlloyReplicaExchangeWorkflow.from_yaml(args.config)
    workflow.run()


if __name__ == "__main__":
    main()
