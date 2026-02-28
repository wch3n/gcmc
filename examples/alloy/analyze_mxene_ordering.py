#!/usr/bin/env python3
"""CLI wrapper for MXene ordering analysis (layer composition + WC-SRO + coordination)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from gcmc.analysis import MXeneOrderingAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="General ordering analysis for MXene alloy trajectories."
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
        help="Alloy elements to analyze (default: Ti Zr).",
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="z",
        help="Layer decomposition axis (default: z).",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=None,
        help="Expected number of alloy layers. If omitted, infer from coordinate gaps.",
    )
    parser.add_argument(
        "--layer-gap",
        type=float,
        default=None,
        help="Layer split gap (A) when n-layers is not set.",
    )
    parser.add_argument(
        "--wc-cutoff",
        type=float,
        default=3.3,
        help="Neighbor cutoff (A) for WC-SRO and coordination (default: 3.3).",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--stop", type=int, default=None, help="Stop frame index (exclusive).")
    parser.add_argument("--step", type=int, default=1, help="Frame stride.")
    parser.add_argument(
        "--no-align-translation",
        action="store_true",
        help="Disable rigid per-frame translation alignment for layer centers.",
    )
    parser.add_argument(
        "--out-dir",
        default="ordering_analysis",
        help="Output directory for analysis files (default: ordering_analysis).",
    )
    parser.add_argument(
        "--out-prefix",
        default="mxene_ordering",
        help=(
            "Output file prefix inside each trajectory subfolder "
            "(default: mxene_ordering)."
        ),
    )
    args = parser.parse_args()

    analyzer = MXeneOrderingAnalyzer(
        alloy_elements=args.elements,
        layer_axis=args.axis,
        n_layers=args.n_layers,
        layer_gap=args.layer_gap,
        align_translation=not args.no_align_translation,
        wc_cutoff=args.wc_cutoff,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    phase_rows = []
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
        phase_rows.append(result["phase_indicators"])

        layer_centers = np.round(np.array(result["layer_centers_ref_A"], dtype=float), 4)
        print(f"{traj_path}:")
        print(f"  inferred_alloy_layers({args.axis}) = {layer_centers}")
        print(f"  wrote CSVs under: {per_traj_dir}")

    # Combined compact phase summary across all trajectories.
    if phase_rows:
        all_fields = []
        field_set = set()
        for r in phase_rows:
            for k in r.keys():
                if k not in field_set:
                    field_set.add(k)
                    all_fields.append(k)

        combined_path = out_dir / f"{args.out_prefix}_phase_summary.csv"
        with combined_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(phase_rows)
        print(f"  wrote combined phase summary: {combined_path}")


if __name__ == "__main__":
    main()
