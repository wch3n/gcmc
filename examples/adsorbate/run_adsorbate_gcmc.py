from __future__ import annotations

import argparse
from pathlib import Path

from gcmc.workflows import AdsorbateGCMCWorkflow


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = THIS_DIR / "adsorbate_gcmc.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the single-run adsorbate GCMC YAML config.",
    )
    args = parser.parse_args()

    workflow = AdsorbateGCMCWorkflow.from_yaml(args.config)
    workflow.run()


if __name__ == "__main__":
    main()
