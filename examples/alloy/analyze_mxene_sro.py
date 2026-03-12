#!/usr/bin/env python3
"""Script example for reference-lattice multicomponent MXene SRO analysis."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase.io import read

from gcmc.analysis import MXeneSROAnalyzer


THIS_DIR = Path(__file__).resolve().parent
CONFIG = SimpleNamespace(
    # Use exactly one of `reference` or `primitive`.
    reference=None,
    primitive=THIS_DIR / "POSCAR.Ti2CO2",
    supercell_matrix=np.array([[8, 0, 0], [5, 10, 0], [0, 0, 1]], dtype=int),
    scale_factor=None,
    site_element="Ti",
    composition={"Ti": 1.0 / 3.0, "Zr": 1.0 / 3.0, "Mo": 1.0 / 3.0},
    seed=67,
    # Replace these with the trajectory files you want to analyze.
    traj_files=[
        Path("replica_300K.traj"),
        Path("replica_600K.traj"),
    ],
    elements=["Ti", "Zr", "Mo"],
    axis="z",
    n_layers=2,
    layer_gap=None,
    n_shells=1,
    shell_tol=0.15,
    fft_qmax=1,
    analysis_mode="reference",  # or "adaptive" for strongly distorted/high-T trajectories
    start=0,
    stop=None,
    step=1,
    align_translation=True,
    canonicalize_layer_flip=False,
    canonical_species=None,
    out_dir=Path("sro_analysis"),
    out_prefix="mxene_sro",
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


def _build_analyzer() -> MXeneSROAnalyzer:
    if bool(CONFIG.reference) == bool(CONFIG.primitive):
        raise ValueError("Set exactly one of CONFIG.reference or CONFIG.primitive.")

    common_kwargs = dict(
        alloy_elements=CONFIG.elements,
        layer_axis=CONFIG.axis,
        n_layers=CONFIG.n_layers,
        layer_gap=CONFIG.layer_gap,
        align_translation=CONFIG.align_translation,
        n_shells=CONFIG.n_shells,
        shell_tol=CONFIG.shell_tol,
        fft_qmax=CONFIG.fft_qmax,
        analysis_mode=CONFIG.analysis_mode,
        canonicalize_layer_flip=CONFIG.canonicalize_layer_flip,
        canonical_species=CONFIG.canonical_species,
    )

    if CONFIG.reference is not None:
        reference_atoms = read(CONFIG.reference)
        reference_mask = None
        if CONFIG.site_element:
            reference_symbols = np.array(reference_atoms.get_chemical_symbols(), dtype=object)
            reference_mask = reference_symbols == CONFIG.site_element
            if not np.any(reference_mask):
                raise ValueError(
                    f"No atoms of type {CONFIG.site_element!r} found in the reference structure."
                )
        return MXeneSROAnalyzer(
            reference_atoms=reference_atoms,
            reference_mask=reference_mask,
            **common_kwargs,
        )

    return MXeneSROAnalyzer.from_primitive(
        primitive_atoms=read(CONFIG.primitive),
        sc_matrix=np.asarray(CONFIG.supercell_matrix, dtype=int),
        scale_factor=CONFIG.scale_factor,
        site_element=CONFIG.site_element,
        composition=CONFIG.composition,
        seed=CONFIG.seed,
        **common_kwargs,
    )


def main() -> None:
    if not CONFIG.traj_files:
        raise ValueError("Set CONFIG.traj_files to one or more trajectory files.")

    analyzer = _build_analyzer()
    out_dir = Path(CONFIG.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    phase_rows = []
    shell_table = np.array(
        [
            (int(shell["shell_id"]), float(shell["shell_distance_A"]))
            for shell in analyzer.reference_shells
        ],
        dtype=float,
    )
    q_table = np.array(
        [
            (int(q["h"]), int(q["k"]), float(q["q_abs_invA"]))
            for q in analyzer.reference_q_grid
        ],
        dtype=float,
    )

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

        layer_centers = np.round(np.array(result["reference_layer_centers_A"], dtype=float), 4)
        print(f"{traj_path}:")
        print(f"  reference_alloy_layers({CONFIG.axis}) = {layer_centers}")
        print(
            "  reference_shells =",
            np.array2string(shell_table, precision=4) if shell_table.size else "[]",
        )
        if q_table.size:
            print("  reciprocal_q_grid =", np.array2string(q_table, precision=4))
        print(f"  wrote CSVs under: {per_traj_dir}")

    if phase_rows:
        combined_path = out_dir / f"{CONFIG.out_prefix}_phase_summary.csv"
        _write_combined_csv(phase_rows, combined_path)
        print(f"  wrote combined phase summary: {combined_path}")


if __name__ == "__main__":
    main()
