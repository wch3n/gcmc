#!/usr/bin/env python3
"""Script example for surface-site motif analysis on MXene alloy trajectories."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gcmc.analysis import MXeneSurfaceMotifAnalyzer


THIS_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = THIS_DIR.parent

CONFIG = SimpleNamespace(
    # Replace these with the trajectory files you want to analyze.
    traj_files=[
        Path("replica_300K.traj"),
        Path("replica_600K.traj"),
    ],
    elements=["Ti", "Zr"],
    site_elements=["O"],
    axis="z",
    n_site_layers=None,
    site_layer_gap=None,
    shell1_size=3,
    shell2_size=3,
    start=0,
    stop=None,
    step=1,
    align_translation=True,
    include_inner_sites=False,
    out_dir=EXAMPLE_DIR / "outputs" / "surface_motif_analysis",
    out_prefix="mxene_surface_motifs",
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

    analyzer = MXeneSurfaceMotifAnalyzer(
        alloy_elements=CONFIG.elements,
        site_elements=CONFIG.site_elements,
        layer_axis=CONFIG.axis,
        n_site_layers=CONFIG.n_site_layers,
        site_layer_gap=CONFIG.site_layer_gap,
        align_translation=CONFIG.align_translation,
        shell1_size=CONFIG.shell1_size,
        shell2_size=CONFIG.shell2_size,
        surface_only=not CONFIG.include_inner_sites,
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
        combined_rows.extend(result.get("surface_motifs_summary", []))

        site_layers = np.round(np.array(result["site_layer_centers_ref_A"], dtype=float), 4)
        print(f"{traj_path}:")
        print(f"  inferred_site_layers({CONFIG.axis}) = {site_layers}")
        print(f"  wrote CSVs under: {per_traj_dir}")

    if combined_rows:
        combined_path = out_dir / f"{CONFIG.out_prefix}_combined_summary.csv"
        _write_combined_csv(combined_rows, combined_path)
        print(f"  wrote combined motif summary: {combined_path}")


if __name__ == "__main__":
    main()
