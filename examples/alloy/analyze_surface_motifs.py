#!/usr/bin/env python3
"""CLI wrapper for surface-site motif analysis on MXene alloy trajectories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from gcmc.analysis import MXeneSurfaceMotifAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identify simple surface-site motifs for MXene alloy trajectories."
    )
    parser.add_argument(
        "--traj",
        nargs="+",
        required=True,
        help="Trajectory file(s), e.g. replica_300K.traj or replica_*K.traj",
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        default=["Ti", "Zr"],
        help="Alloy elements used to build the local shells (default: Ti Zr).",
    )
    parser.add_argument(
        "--site-elements",
        nargs="+",
        default=["O"],
        help="Surface-site elements to analyze (default: O).",
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="z",
        help="Surface normal axis (default: z).",
    )
    parser.add_argument(
        "--n-site-layers",
        type=int,
        default=None,
        help="Expected number of site-element layers. If omitted, infer from coordinate gaps.",
    )
    parser.add_argument(
        "--site-layer-gap",
        type=float,
        default=None,
        help="Site-layer split gap (A) when n-site-layers is not set.",
    )
    parser.add_argument(
        "--shell1-size",
        type=int,
        default=3,
        help="Number of nearest alloy atoms in shell 1 (default: 3).",
    )
    parser.add_argument(
        "--shell2-size",
        type=int,
        default=3,
        help="Number of nearest alloy atoms in shell 2 (default: 3).",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--stop", type=int, default=None, help="Stop frame index (exclusive).")
    parser.add_argument("--step", type=int, default=1, help="Frame stride.")
    parser.add_argument(
        "--no-align-translation",
        action="store_true",
        help="Disable rigid per-frame translation alignment for site layers.",
    )
    parser.add_argument(
        "--include-inner-sites",
        action="store_true",
        help="Include non-surface site layers instead of restricting to the outermost layers.",
    )
    parser.add_argument(
        "--out-dir",
        default="surface_motif_analysis",
        help="Output directory for motif analysis files (default: surface_motif_analysis).",
    )
    parser.add_argument(
        "--out-prefix",
        default="mxene_surface_motifs",
        help="Output file prefix inside each trajectory subfolder (default: mxene_surface_motifs).",
    )
    args = parser.parse_args()

    analyzer = MXeneSurfaceMotifAnalyzer(
        alloy_elements=args.elements,
        site_elements=args.site_elements,
        layer_axis=args.axis,
        n_site_layers=args.n_site_layers,
        site_layer_gap=args.site_layer_gap,
        align_translation=not args.no_align_translation,
        shell1_size=args.shell1_size,
        shell2_size=args.shell2_size,
        surface_only=not args.include_inner_sites,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_rows = []
    for traj in args.traj:
        traj_path = Path(traj)
        result = analyzer.analyze_trajectory(
            traj_path=traj_path,
            start=args.start,
            stop=args.stop,
            step=args.step,
        )
        per_traj_dir = out_dir / traj_path.stem
        per_traj_dir.mkdir(parents=True, exist_ok=True)
        per_traj_prefix = per_traj_dir / args.out_prefix
        analyzer.export_csv(result, per_traj_prefix)
        combined_rows.extend(result.get("surface_motifs_summary", []))

        site_layers = np.round(np.array(result["site_layer_centers_ref_A"], dtype=float), 4)
        print(f"{traj_path}:")
        print(f"  inferred_site_layers({args.axis}) = {site_layers}")
        print(f"  wrote CSVs under: {per_traj_dir}")

    if combined_rows:
        all_fields = []
        field_set = set()
        for row in combined_rows:
            for key in row.keys():
                if key not in field_set:
                    field_set.add(key)
                    all_fields.append(key)

        combined_path = out_dir / f"{args.out_prefix}_combined_summary.csv"
        with combined_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(combined_rows)
        print(f"  wrote combined motif summary: {combined_path}")


if __name__ == "__main__":
    main()
