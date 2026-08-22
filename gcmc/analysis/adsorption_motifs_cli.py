"""Command-line adsorption-motif analysis for tagged adsorbate trajectories."""

from __future__ import annotations

import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from .local_motifs import LocalAdsorptionMotifAnalyzer
from .ordering import _write_csv


_SAMPLED_REPLICA_RE = re.compile(r"replica_[0-9]+(?:\.[0-9]+)?K\.traj$")


def _discover_trajectories(values: Sequence[str], *, recursive: bool) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        expanded = Path(value).expanduser()
        if any(token in value for token in "*?["):
            paths.extend(Path(path) for path in glob.glob(str(expanded), recursive=recursive))
        elif expanded.is_dir():
            iterator = expanded.rglob("replica_*K.traj") if recursive else expanded.glob("replica_*K.traj")
            paths.extend(iterator)
        else:
            paths.append(expanded)

    unique = {
        path.resolve()
        for path in paths
        if path.is_file() and _SAMPLED_REPLICA_RE.fullmatch(path.name)
    }
    if not unique:
        raise FileNotFoundError("No sampled replica_*K.traj files were found.")
    return sorted(unique)


def _reference_for(traj: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    candidate = traj.parent / "adsorbate_pt_initial.traj"
    return candidate if candidate.is_file() else None


def _aggregate_rows(
    rows: Iterable[dict[str, object]],
    group_fields: Sequence[str],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    totals: dict[object, int] = defaultdict(int)
    for row in rows:
        temperature = row["temperature_K"]
        key = tuple(row[field] for field in group_fields)
        groups[key].append(row)
        totals[temperature] += 1

    summary: list[dict[str, object]] = []
    for key, group in groups.items():
        output = dict(zip(group_fields, key))
        temperature = output["temperature_K"]
        output.update(
            {
                "samples": len(group),
                "population": len(group) / totals[temperature],
                "n_trajectories": len({str(row["traj"]) for row in group}),
                "n_frames": len(
                    {(str(row["traj"]), int(row["frame"])) for row in group}
                ),
            }
        )
        summary.append(output)
    return sorted(
        summary,
        key=lambda row: (
            float(row["temperature_K"]),
            -int(row["samples"]),
            *(str(row[field]) for field in group_fields[1:]),
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify tagged adsorbates as atop (1-fold), bridge (2-fold), "
            "hollow/fcc/hcp (3-fold), off-site, or detached."
        )
    )
    parser.add_argument(
        "--traj",
        nargs="+",
        required=True,
        help="Trajectory files, globs, or run directories.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively discover sampled replica trajectories in directories.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help=(
            "Common clean/reference slab. By default, use the adjacent "
            "adsorbate_pt_initial.traj or strip tagged adsorbates from frame 0."
        ),
    )
    parser.add_argument("--site-elements", nargs="+", default=["Ti", "Zr"])
    parser.add_argument("--substrate-elements", nargs="+", default=["Ti", "Zr", "C"])
    parser.add_argument("--functional-elements", nargs="+", default=["O"])
    parser.add_argument(
        "--site-types",
        nargs="+",
        default=["atop", "bridge", "fcc", "hcp"],
    )
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
    parser.add_argument(
        "--max-site-distance",
        type=float,
        default=0.75,
        help="Maximum anchor-to-registry lateral distance before labeling off_site.",
    )
    parser.add_argument(
        "--max-anchor-support-distance",
        type=float,
        default=3.0,
        help="Maximum anchor-support distance before labeling detached.",
    )
    parser.add_argument(
        "--include-blocked-sites",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include sites marked blocked by terminations when classifying observed frames.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("adsorption_motifs"))
    parser.add_argument(
        "--write-representatives",
        action="store_true",
        help="Write representative full-slab frames for the most populated groups.",
    )
    parser.add_argument(
        "--representative-group-by",
        choices=["pattern", "motif", "site"],
        default="pattern",
    )
    parser.add_argument("--representative-top-k", type=int, default=6)
    parser.add_argument("--representative-n-per-group", type=int, default=1)
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write motif-family and detailed-pattern distribution plots.",
    )
    parser.add_argument(
        "--plot-target-temperature",
        type=float,
        default=None,
        help="Temperature for the detailed-pattern panel. Default: lowest available.",
    )
    parser.add_argument("--plot-top-patterns", type=int, default=8)
    parser.add_argument(
        "--plot-support-elements",
        nargs=2,
        default=["Ti", "Zr"],
        metavar=("ELEMENT_A", "ELEMENT_B"),
        help="Color support compositions from ELEMENT_A to ELEMENT_B.",
    )
    parser.add_argument(
        "--plot-color-by-support",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Color aggregate motif curves by their mean support composition.",
    )
    parser.add_argument(
        "--plot-formats",
        nargs="+",
        choices=["pdf", "png", "svg"],
        default=["pdf", "png"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    trajectories = _discover_trajectories(args.traj, recursive=args.recursive)
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
        max_site_distance_A=args.max_site_distance,
        max_anchor_support_distance_A=args.max_anchor_support_distance,
        include_blocked_sites=args.include_blocked_sites,
    )

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    all_frames: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []

    for traj_index, traj in enumerate(trajectories):
        reference = _reference_for(traj, args.reference)
        result = analyzer.analyze_trajectory(
            traj,
            reference=reference,
            start=args.start,
            stop=args.stop,
            step=args.step,
        )
        output_name = f"traj_{traj_index:03d}_{traj.stem}"
        out_prefix = out_dir / output_name / "adsorption_motifs"
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

        frame_rows = list(result.get("local_motif_frames", []))
        all_frames.extend(frame_rows)
        manifest.append(
            {
                "trajectory_index": traj_index,
                "traj": str(traj),
                "reference": "" if reference is None else str(reference),
                "temperature_K": result["temperature_K"],
                "samples": len(frame_rows),
                "output_dir": str(out_prefix.parent),
            }
        )
        families = result.get("adsorption_family_summary", [])
        family_text = ", ".join(
            f"{row['adsorption_motif']}={float(row['population']):.1%}"
            for row in families
        )
        print(f"[{traj_index + 1}/{len(trajectories)}] {traj}: {family_text}")

    _write_csv(out_dir / "trajectories.csv", manifest, list(manifest[0].keys()))
    if all_frames:
        _write_csv(out_dir / "all_per_frame.csv", all_frames, list(all_frames[0].keys()))
        patterns = _aggregate_rows(
            all_frames,
            ("temperature_K", "adsorption_motif", "site_type", "support_key"),
        )
        families = _aggregate_rows(
            all_frames,
            ("temperature_K", "adsorption_motif"),
        )
        _write_csv(out_dir / "all_patterns.csv", patterns, list(patterns[0].keys()))
        _write_csv(out_dir / "all_families.csv", families, list(families[0].keys()))
        if args.plot:
            from .adsorption_motif_plots import plot_adsorption_motif_distributions

            written_plots = plot_adsorption_motif_distributions(
                all_frames,
                out_dir / "motif_distribution",
                target_temperature_K=args.plot_target_temperature,
                top_patterns=args.plot_top_patterns,
                support_color_elements=args.plot_support_elements,
                color_temperature_by_support=args.plot_color_by_support,
                formats=args.plot_formats,
            )
            print("Wrote plots: " + ", ".join(str(path) for path in written_plots))
    print(f"Wrote adsorption-motif analysis to {out_dir}")


if __name__ == "__main__":
    main()
