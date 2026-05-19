"""Command-line entry points for reaction workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from .workflow import ReactionPostProcessingWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reaction post-processing from a YAML config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Reaction post-processing YAML config.",
    )
    args = parser.parse_args()

    workflow = ReactionPostProcessingWorkflow.from_yaml(args.config)
    result = workflow.run()
    print(f"Analyzed parent samples: {result['n_parent_samples']}")
    for label, path in result["output_paths"].items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
