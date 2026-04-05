#!/usr/bin/env python3
"""Script example for generating metal-centered adsorption-site candidates."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gcmc.analysis import MXeneAdsorptionSiteAnalyzer


THIS_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = THIS_DIR.parent

CONFIG = SimpleNamespace(
    # Replace these with the trajectory files you want to analyze.
    traj_files=[
        Path("replica_300K.traj"),
        Path("replica_600K.traj"),
    ],
    elements=["Ti", "Zr"],
    termination_elements=["O"],
    axis="z",
    n_alloy_layers=2,
    alloy_layer_gap=None,
    bridge_cutoff=None,
    bridge_cutoff_scale=1.15,
    env_shell_size=3,
    support_xy_tol=2.5,
    vertical_offset=1.5,
    min_termination_dist=0.75,
    start=0,
    stop=None,
    step=1,
    align_translation=True,
    out_dir=EXAMPLE_DIR / "outputs" / "adsorption_site_analysis",
    out_prefix="mxene_adsorption_sites",
)


def _write_combined_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not CONFIG.traj_files:
        raise ValueError("Set CONFIG.traj_files to one or more trajectory files.")

    analyzer = MXeneAdsorptionSiteAnalyzer(
        alloy_elements=CONFIG.elements,
        termination_elements=CONFIG.termination_elements,
        layer_axis=CONFIG.axis,
        n_alloy_layers=CONFIG.n_alloy_layers,
        alloy_layer_gap=CONFIG.alloy_layer_gap,
        align_translation=CONFIG.align_translation,
        bridge_cutoff=CONFIG.bridge_cutoff,
        bridge_cutoff_scale=CONFIG.bridge_cutoff_scale,
        env_shell_size=CONFIG.env_shell_size,
        support_xy_tol=CONFIG.support_xy_tol,
        vertical_offset=CONFIG.vertical_offset,
        min_termination_dist=CONFIG.min_termination_dist,
    )

    out_dir = Path(CONFIG.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_rows = []
    for traj in CONFIG.traj_files:
        traj_path = Path(traj)
        result = analyzer.analyze_trajectory(
            traj_path=traj_path,
            start=CONFIG.start,
            stop=CONFIG.stop,
            step=CONFIG.step,
        )
        per_traj_dir = out_dir / traj_path.stem
        per_traj_dir.mkdir(parents=True, exist_ok=True)
        per_traj_prefix = per_traj_dir / CONFIG.out_prefix
        analyzer.export_csv(result, per_traj_prefix)
        combined_rows.extend(result.get("candidate_site_summary", []))

        alloy_layers = np.round(np.array(result["alloy_layer_centers_ref_A"], dtype=float), 4)
        print(f"{traj_path}:")
        print(f"  inferred_alloy_layers({CONFIG.axis}) = {alloy_layers}")
        print(f"  wrote CSVs under: {per_traj_dir}")

    if combined_rows:
        combined_path = out_dir / f"{CONFIG.out_prefix}_combined_summary.csv"
        _write_combined_csv(combined_rows, combined_path)
        print(f"  wrote combined site summary: {combined_path}")


if __name__ == "__main__":
    main()
