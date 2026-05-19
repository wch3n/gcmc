#!/usr/bin/env python3
"""Analyze local adsorption motifs for tagged adsorbate trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

from gcmc.analysis import LocalAdsorptionMotifAnalyzer


def _support_label(row: dict[str, object]) -> str:
    support_indices = str(row.get("support_indices", "")).strip()
    if not support_indices:
        return "support=?"
    return f"support={support_indices}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traj",
        type=Path,
        nargs="+",
        required=True,
        help="One or more tagged adsorbate trajectories.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional reference slab without adsorbates. Default: strip tagged adsorbates from the first analyzed frame.",
    )
    parser.add_argument("--site-elements", nargs="+", default=["Ti", "Zr"])
    parser.add_argument("--substrate-elements", nargs="+", default=["Ti", "Zr", "C"])
    parser.add_argument("--functional-elements", nargs="+", default=["O"])
    parser.add_argument("--site-types", nargs="+", default=["atop", "fcc", "hcp"])
    parser.add_argument("--surface-side", choices=["top", "bottom"], default="top")
    parser.add_argument("--anchor-element", default="O")
    parser.add_argument("--shell1-size", type=int, default=6)
    parser.add_argument("--shell2-size", type=int, default=6)
    parser.add_argument("--functional-cutoff", type=float, default=3.0)
    parser.add_argument("--surface-layer-tol", type=float, default=0.5)
    parser.add_argument("--site-match-tol", type=float, default=0.6)
    parser.add_argument("--support-xy-tol", type=float, default=1.2)
    parser.add_argument("--termination-site-xy-tol", type=float, default=2.2)
    parser.add_argument("--vertical-offset", type=float, default=1.5)
    parser.add_argument("--min-termination-dist", type=float, default=0.8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-prefix", default="local_motifs")
    parser.add_argument(
        "--write-representatives",
        action="store_true",
        help="Also write representative full-slab frames for the top motifs/sites.",
    )
    parser.add_argument(
        "--representative-group-by",
        choices=["motif", "site"],
        default="motif",
    )
    parser.add_argument("--representative-top-k", type=int, default=5)
    parser.add_argument("--representative-n-per-group", type=int, default=1)
    args = parser.parse_args()

    analyzer = LocalAdsorptionMotifAnalyzer(
        site_elements=args.site_elements,
        substrate_elements=args.substrate_elements,
        functional_elements=args.functional_elements,
        site_types=args.site_types,
        surface_side=args.surface_side,
        layer_tol=args.surface_layer_tol,
        xy_tol=args.site_match_tol,
        support_xy_tol=args.support_xy_tol,
        termination_site_xy_tol=args.termination_site_xy_tol,
        vertical_offset=args.vertical_offset,
        min_termination_dist=args.min_termination_dist,
        anchor_element=args.anchor_element,
        shell1_size=args.shell1_size,
        shell2_size=args.shell2_size,
        functional_cutoff=args.functional_cutoff,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for traj_path in args.traj:
        result = analyzer.analyze_trajectory(
            traj_path,
            reference=args.reference,
            start=args.start,
            stop=args.stop,
            step=args.step,
        )
        out_prefix = out_dir / traj_path.stem / args.out_prefix
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        analyzer.export_csv(result, out_prefix)
        if args.write_representatives:
            analyzer.export_representatives(
                result,
                out_prefix,
                group_by=args.representative_group_by,
                top_k=args.representative_top_k,
                n_per_group=args.representative_n_per_group,
            )

        summary_rows = result.get("local_motif_summary", [])
        frame_rows = result.get("local_motif_frames", [])
        support_by_site = {}
        for frame_row in frame_rows:
            site_local_index = int(frame_row["site_local_index"])
            support_by_site.setdefault(site_local_index, _support_label(frame_row))

        print(f"{traj_path}:")
        for row in summary_rows[:5]:
            support_label = support_by_site.get(int(row["site_local_index"]), "support=?")
            print(
                f"  {support_label:<20} | {row['site_type']:<4} | "
                f"{row['support_key']:<12} | pop={row['population']:.3f} | "
                f"shell1={row['shell1_key']} | shell2={row['shell2_key']}"
            )
        if args.write_representatives:
            print(
                f"  wrote representatives: {out_prefix}_representatives.traj"
            )
        print(f"  wrote CSVs under: {out_prefix.parent}")


if __name__ == "__main__":
    main()
