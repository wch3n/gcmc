"""Shared output file naming for reaction post-processing."""

from __future__ import annotations

from pathlib import Path


_OUTPUT_NAMES = {
    "summary_csv": "summary.csv",
    "representatives_csv": "representatives.csv",
    "representatives_traj": "representatives.traj",
    "site_manifest_csv": "site_manifest.csv",
    "candidate_manifest_csv": "candidate_manifest.csv",
    "local_cmc_manifest_csv": "local_cmc_manifest.csv",
    "state_relaxation_manifest_csv": "state_relaxation_manifest.csv",
    "state_relaxation_log": "state_relaxation.log",
    "vibration_summary_csv": "vibration_summary.csv",
    "vibration_log": "vibration.log",
    "oer_routes_csv": "oer_routes.csv",
    "oer_states_csv": "oer_states.csv",
    "oer_ensemble_csv": "oer_ensemble.csv",
}


def output_path(config: object, key: str) -> Path:
    """Return a top-level output path for a workflow artifact."""

    output_dir = Path(str(getattr(config, "output_dir")))
    return output_dir / _OUTPUT_NAMES[key]
