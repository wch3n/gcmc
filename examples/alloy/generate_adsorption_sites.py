#!/usr/bin/env python3
"""Generate metal-centered adsorption-site candidates for MXene alloy trajectories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from gcmc.analysis import MXeneAdsorptionSiteAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate atop/bridge/hollow adsorption-site candidates on MXene alloy surfaces."
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
        help="Alloy elements forming the metal-site lattice (default: Ti Zr).",
    )
    parser.add_argument(
        "--termination-elements",
        nargs="+",
        default=["O"],
        help="Termination elements used for steric checks (default: O).",
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="z",
        help="Surface normal axis (default: z).",
    )
    parser.add_argument(
        "--n-alloy-layers",
        type=int,
        default=None,
        help="Expected number of alloy layers. If omitted, infer from coordinate gaps.",
    )
    parser.add_argument(
        "--alloy-layer-gap",
        type=float,
        default=None,
        help="Alloy-layer split gap (A) when n-alloy-layers is not set.",
    )
    parser.add_argument(
        "--bridge-cutoff",
        type=float,
        default=None,
        help="Explicit in-plane bridge cutoff in A. If omitted, infer from nearest-neighbor spacing.",
    )
    parser.add_argument(
        "--bridge-cutoff-scale",
        type=float,
        default=1.15,
        help="Scale factor for the inferred bridge cutoff (default: 1.15).",
    )
    parser.add_argument(
        "--env-shell-size",
        type=int,
        default=3,
        help="Number of nearby alloy atoms used for the environment key (default: 3).",
    )
    parser.add_argument(
        "--support-xy-tol",
        type=float,
        default=2.5,
        help="Lateral support radius in A for suggested z placement (default: 2.5).",
    )
    parser.add_argument(
        "--vertical-offset",
        type=float,
        default=1.5,
        help="Suggested vertical offset in A from the local support envelope (default: 1.5).",
    )
    parser.add_argument(
        "--min-termination-dist",
        type=float,
        default=0.75,
        help="Flag candidates closer than this to termination atoms at the suggested height (default: 0.75 A).",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--stop", type=int, default=None, help="Stop frame index (exclusive).")
    parser.add_argument("--step", type=int, default=1, help="Frame stride.")
    parser.add_argument(
        "--no-align-translation",
        action="store_true",
        help="Disable rigid per-frame translation alignment for alloy layers.",
    )
    parser.add_argument(
        "--out-dir",
        default="adsorption_site_analysis",
        help="Output directory for site analysis files (default: adsorption_site_analysis).",
    )
    parser.add_argument(
        "--out-prefix",
        default="mxene_adsorption_sites",
        help="Output file prefix inside each trajectory subfolder (default: mxene_adsorption_sites).",
    )
    args = parser.parse_args()

    analyzer = MXeneAdsorptionSiteAnalyzer(
        alloy_elements=args.elements,
        termination_elements=args.termination_elements,
        layer_axis=args.axis,
        n_alloy_layers=args.n_alloy_layers,
        alloy_layer_gap=args.alloy_layer_gap,
        align_translation=not args.no_align_translation,
        bridge_cutoff=args.bridge_cutoff,
        bridge_cutoff_scale=args.bridge_cutoff_scale,
        env_shell_size=args.env_shell_size,
        support_xy_tol=args.support_xy_tol,
        vertical_offset=args.vertical_offset,
        min_termination_dist=args.min_termination_dist,
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
        combined_rows.extend(result.get("candidate_site_summary", []))

        alloy_layers = np.round(np.array(result["alloy_layer_centers_ref_A"], dtype=float), 4)
        print(f"{traj_path}:")
        print(f"  inferred_alloy_layers({args.axis}) = {alloy_layers}")
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
        print(f"  wrote combined site summary: {combined_path}")


if __name__ == "__main__":
    main()
