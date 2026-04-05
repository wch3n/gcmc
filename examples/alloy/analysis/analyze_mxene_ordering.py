#!/usr/bin/env python3
"""Script example for MXene ordering analysis (layer composition + WC-SRO + coordination)."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gcmc.analysis import MXeneOrderingAnalyzer


THIS_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = THIS_DIR.parent

CONFIG = SimpleNamespace(
    # Replace these with the trajectory files you want to analyze.
    traj_files=[
        Path("replica_300K.traj"),
        Path("replica_600K.traj"),
    ],
    elements=["Ti", "Zr"],
    axis="z",
    n_layers=2,
    layer_gap=None,
    wc_cutoff=3.3,
    start=0,
    stop=None,
    step=1,
    align_translation=True,
    out_dir=EXAMPLE_DIR / "outputs" / "ordering_analysis",
    out_prefix="mxene_ordering",
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

    analyzer = MXeneOrderingAnalyzer(
        alloy_elements=CONFIG.elements,
        layer_axis=CONFIG.axis,
        n_layers=CONFIG.n_layers,
        layer_gap=CONFIG.layer_gap,
        align_translation=CONFIG.align_translation,
        wc_cutoff=CONFIG.wc_cutoff,
    )

    out_dir = Path(CONFIG.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    phase_rows = []
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
        phase_rows.append(result["phase_indicators"])

        layer_centers = np.round(np.array(result["layer_centers_ref_A"], dtype=float), 4)
        print(f"{traj_path}:")
        print(f"  inferred_alloy_layers({CONFIG.axis}) = {layer_centers}")
        print(f"  wrote CSVs under: {per_traj_dir}")

    if phase_rows:
        combined_path = out_dir / f"{CONFIG.out_prefix}_phase_summary.csv"
        _write_combined_csv(phase_rows, combined_path)
        print(f"  wrote combined phase summary: {combined_path}")


if __name__ == "__main__":
    main()
