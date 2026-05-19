#!/usr/bin/env python3
"""Overlay adsorption-site markers on a slab structure for visual inspection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from ase.io import read, write

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gcmc.analysis import overlay_site_markers, summarize_site_registry
from gcmc.constants import ADSORBATE_TAG_OFFSET
from gcmc.utils import build_surface_site_registry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", required=True, help="Input structure/trajectory file.")
    parser.add_argument("--frame", default="-1", help="Frame index to read (ASE syntax, default: -1).")
    parser.add_argument("--out", required=True, help="Output file for the slab + site markers.")
    parser.add_argument("--site-elements", nargs="+", required=True, help="Elements used to define candidate sites.")
    parser.add_argument(
        "--substrate-elements",
        nargs="+",
        required=True,
        help="Elements defining the substrate lattice used for surface-layer detection.",
    )
    parser.add_argument(
        "--termination-elements",
        nargs="*",
        default=["O"],
        help="Termination elements used for termination blocking and MXene metal-site matching.",
    )
    parser.add_argument(
        "--site-types",
        nargs="+",
        default=["atop", "fcc", "hcp"],
        help="Site families to generate.",
    )
    parser.add_argument("--surface-side", choices=("top", "bottom"), default="top")
    parser.add_argument("--surface-layer-tol", type=float, default=0.5)
    parser.add_argument("--site-match-tol", type=float, default=0.6)
    parser.add_argument("--support-xy-tol", type=float, default=1.2)
    parser.add_argument("--termination-site-xy-tol", type=float, default=2.2)
    parser.add_argument("--vertical-offset", type=float, default=1.5)
    parser.add_argument("--min-termination-dist", type=float, default=0.8)
    parser.add_argument(
        "--include-blocked",
        action="store_true",
        help="Include sites flagged as blocked by nearby terminations.",
    )
    parser.add_argument(
        "--strip-adsorbates",
        action="store_true",
        help="Drop tagged adsorbate atoms before building the site registry.",
    )
    parser.add_argument(
        "--z-field",
        choices=("suggested_z_A", "anchor_z_A"),
        default="suggested_z_A",
        help="Which registry z-coordinate to use for the marker atoms.",
    )
    return parser.parse_args()


def _strip_tagged_adsorbates(atoms):
    tags = np.asarray(atoms.get_tags(), dtype=int)
    if tags.size == 0 or not np.any(tags >= ADSORBATE_TAG_OFFSET):
        return atoms
    keep = tags < ADSORBATE_TAG_OFFSET
    return atoms[keep]


def main() -> None:
    args = _parse_args()

    atoms = read(args.traj, index=args.frame)
    if args.strip_adsorbates:
        atoms = _strip_tagged_adsorbates(atoms)

    registry = build_surface_site_registry(
        atoms,
        site_elements=tuple(args.site_elements),
        substrate_elements=tuple(args.substrate_elements),
        termination_elements=tuple(args.termination_elements),
        surface_side=args.surface_side,
        site_types=tuple(args.site_types),
        layer_tol=float(args.surface_layer_tol),
        xy_tol=float(args.site_match_tol),
        support_xy_tol=float(args.support_xy_tol),
        termination_site_xy_tol=float(args.termination_site_xy_tol),
        vertical_offset=float(args.vertical_offset),
        min_termination_dist=float(args.min_termination_dist),
    )

    counts_all = summarize_site_registry(registry, include_blocked=True)
    counts_visible = summarize_site_registry(registry, include_blocked=args.include_blocked)
    overlay = overlay_site_markers(
        atoms,
        registry,
        include_blocked=args.include_blocked,
        z_field=args.z_field,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(out_path, overlay)

    total_sites = int(sum(counts_all.values()))
    visible_sites = int(sum(counts_visible.values()))
    blocked_sites = total_sites - int(sum(summarize_site_registry(registry, include_blocked=False).values()))
    print(f"input: {args.traj}[{args.frame}]")
    print(f"output: {out_path}")
    print(f"registry_total: {total_sites}")
    print(f"visible_sites: {visible_sites}")
    print(f"blocked_sites: {blocked_sites}")
    print(f"counts_all: {counts_all}")
    if args.include_blocked:
        print(f"counts_visible: {counts_all}")
    else:
        print(f"counts_visible: {counts_visible}")
    print("marker elements: atop=He bridge=Ne fcc=Ar hcp=Kr")


if __name__ == "__main__":
    main()
