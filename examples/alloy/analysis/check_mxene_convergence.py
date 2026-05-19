#!/usr/bin/env python3
"""Check live MXene alloy MC convergence from trajectory and thermo outputs.

This script intentionally analyzes the live ``replica_*K.traj`` and matching
``replica_*K.dat`` files, not checkpoint pickle files.  It is meant to answer:
"are the sampled observables stationary enough that the surface has likely
equilibrated?"
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import tempfile
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml
from ase.io import read

from gcmc.analysis import MXeneOrderingAnalyzer, MXeneSurfaceMotifAnalyzer

_DEFAULT_CORE_ELEMENTS = ("C", "B", "N")


def _expand_paths(values: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = sorted(glob.glob(value))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(value))
    return list(dict.fromkeys(path.resolve() for path in paths))


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping.")
    return raw


def _section(config: dict[str, object], name: str) -> dict[str, object]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def _infer_alloy_elements(config: dict[str, object]) -> list[str]:
    system = _section(config, "system")
    composition = system.get("composition")
    if isinstance(composition, dict) and composition:
        elements = [str(key) for key in composition.keys()]
        if len(elements) >= 2:
            return elements

    mc = _section(config, "mc")
    swap_elements = mc.get("swap_elements")
    if isinstance(swap_elements, (list, tuple)) and swap_elements:
        elements = [str(value) for value in swap_elements]
        if len(elements) >= 2:
            return elements

    flat_composition = config.get("composition")
    if isinstance(flat_composition, dict) and flat_composition:
        elements = [str(key) for key in flat_composition.keys()]
        if len(elements) >= 2:
            return elements
    flat_swap = config.get("swap_elements")
    if isinstance(flat_swap, (list, tuple)) and flat_swap:
        elements = [str(value) for value in flat_swap]
        if len(elements) >= 2:
            return elements
    return []


def _infer_site_elements(
    traj_paths: Sequence[Path],
    *,
    alloy_elements: Sequence[str],
    core_elements: Sequence[str],
) -> list[str]:
    if not traj_paths:
        return []
    atoms = read(str(traj_paths[0]), index=0)
    symbols = list(dict.fromkeys(atoms.get_chemical_symbols()))
    excluded = set(alloy_elements) | set(core_elements)
    inferred = [symbol for symbol in symbols if symbol not in excluded]
    return inferred


def _parse_dat(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            numeric: list[float] = []
            for part in parts:
                try:
                    numeric.append(float(part))
                except ValueError:
                    pass
            if len(numeric) < 2:
                continue
            rows.append({"sweep": numeric[0], "energy_eV": numeric[1]})
    return rows


def _blocks(n_items: int, n_blocks: int) -> list[tuple[int, int]]:
    if n_items <= 0:
        return []
    n_blocks = max(1, min(int(n_blocks), n_items))
    edges = np.linspace(0, n_items, n_blocks + 1, dtype=int)
    return [
        (int(edges[idx]), int(edges[idx + 1]))
        for idx in range(n_blocks)
        if int(edges[idx + 1]) > int(edges[idx])
    ]


def _safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if arr.size else float("nan")


def _block_energy_rows(
    traj: Path,
    dat_rows: Sequence[dict[str, float]],
    *,
    n_blocks: int,
    n_atoms: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for block_id, (start, stop) in enumerate(_blocks(len(dat_rows), n_blocks), start=1):
        rows = dat_rows[start:stop]
        energies = np.asarray([row["energy_eV"] for row in rows], dtype=float)
        sweeps = np.asarray([row["sweep"] for row in rows], dtype=float)
        mean = float(np.mean(energies))
        std = float(np.std(energies))
        sem = float(std / math.sqrt(energies.size)) if energies.size else float("nan")
        output.append(
            {
                "traj": str(traj),
                "block": block_id,
                "n_samples": int(energies.size),
                "sweep_start": float(sweeps[0]),
                "sweep_stop": float(sweeps[-1]),
                "energy_mean_eV": mean,
                "energy_std_eV": std,
                "energy_sem_eV": sem,
                "energy_mean_eV_per_atom": mean / n_atoms if n_atoms else "",
                "energy_std_eV_per_atom": std / n_atoms if n_atoms else "",
            }
        )
    return output


def _frame_to_block_map(frames: Sequence[int], n_blocks: int) -> dict[int, int]:
    ordered = sorted(dict.fromkeys(int(frame) for frame in frames))
    mapping: dict[int, int] = {}
    for block_id, (start, stop) in enumerate(_blocks(len(ordered), n_blocks), start=1):
        for frame in ordered[start:stop]:
            mapping[frame] = block_id
    return mapping


def _block_layer_composition_rows(
    rows: Sequence[dict[str, object]],
    *,
    elements: Sequence[str],
    n_blocks: int,
) -> list[dict[str, object]]:
    frames = [int(row["frame"]) for row in rows]
    frame_blocks = _frame_to_block_map(frames, n_blocks)
    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        block = frame_blocks.get(int(row["frame"]))
        if block is None:
            continue
        grouped[(str(row["traj"]), int(row["layer_id"]), block)].append(dict(row))

    output: list[dict[str, object]] = []
    for (traj, layer_id, block), group in sorted(grouped.items()):
        out: dict[str, object] = {
            "traj": traj,
            "block": block,
            "layer_id": layer_id,
            "n_frames": len({int(row["frame"]) for row in group}),
        }
        for element in elements:
            out[f"frac_{element}_mean"] = _safe_mean(
                float(row[f"frac_{element}"]) for row in group
            )
            out[f"frac_{element}_std"] = _safe_std(
                float(row[f"frac_{element}"]) for row in group
            )
        output.append(out)
    return output


def _block_wc_rows(
    rows: Sequence[dict[str, object]],
    *,
    n_blocks: int,
) -> list[dict[str, object]]:
    frames = [int(row["frame"]) for row in rows]
    frame_blocks = _frame_to_block_map(frames, n_blocks)
    grouped: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        block = frame_blocks.get(int(row["frame"]))
        if block is None:
            continue
        key = (
            str(row["traj"]),
            str(row["central_species"]),
            str(row["neighbor_species"]),
            block,
        )
        grouped[key].append(dict(row))

    output: list[dict[str, object]] = []
    for (traj, central, neighbor, block), group in sorted(grouped.items()):
        values = [float(row["alpha_wc"]) for row in group]
        output.append(
            {
                "traj": traj,
                "block": block,
                "central_species": central,
                "neighbor_species": neighbor,
                "n_frames": len({int(row["frame"]) for row in group}),
                "alpha_wc_mean": _safe_mean(values),
                "alpha_wc_std": _safe_std(values),
            }
        )
    return output


def _block_motif_rows(
    rows: Sequence[dict[str, object]],
    *,
    n_blocks: int,
    top_n: int,
) -> list[dict[str, object]]:
    if not rows:
        return []
    total_by_motif: dict[str, float] = defaultdict(float)
    for row in rows:
        total_by_motif[str(row["motif_key"])] += float(row["site_fraction_frame"])
    top_motifs = {
        motif
        for motif, _ in sorted(
            total_by_motif.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]
    }
    frames = sorted({int(row["frame"]) for row in rows})
    frame_blocks = _frame_to_block_map(frames, n_blocks)
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    seen_frame_motifs: set[tuple[str, int, str]] = set()

    for row in rows:
        motif = str(row["motif_key"])
        if motif not in top_motifs:
            continue
        traj = str(row["traj"])
        frame = int(row["frame"])
        block = frame_blocks.get(frame)
        if block is None:
            continue
        grouped[(traj, motif, block)].append(float(row["site_fraction_frame"]))
        seen_frame_motifs.add((traj, frame, motif))

    # Missing top motifs in a frame have zero population.
    trajs = sorted({str(row["traj"]) for row in rows})
    for traj in trajs:
        for frame in frames:
            block = frame_blocks.get(frame)
            if block is None:
                continue
            for motif in top_motifs:
                if (traj, frame, motif) not in seen_frame_motifs:
                    grouped[(traj, motif, block)].append(0.0)

    output: list[dict[str, object]] = []
    for (traj, motif, block), values in sorted(grouped.items()):
        output.append(
            {
                "traj": traj,
                "block": block,
                "motif_key": motif,
                "n_frames": len(values),
                "site_fraction_mean": _safe_mean(values),
                "site_fraction_std": _safe_std(values),
            }
        )
    return output


def _stationarity_rows(
    rows: Sequence[dict[str, object]],
    *,
    metric_columns: Sequence[str],
    group_columns: Sequence[str],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(col, "") for col in group_columns)].append(dict(row))
    for key, group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(row["block"]))
        if len(group) < 2:
            continue
        base = {col: value for col, value in zip(group_columns, key)}
        for metric in metric_columns:
            values = np.asarray([float(row.get(metric, np.nan)) for row in group], dtype=float)
            if not np.any(np.isfinite(values)):
                continue
            first = float(values[0])
            last = float(values[-1])
            finite = values[np.isfinite(values)]
            out = dict(base)
            out.update(
                {
                    "metric": metric,
                    "n_blocks": len(group),
                    "first_block": first,
                    "last_block": last,
                    "last_minus_first": last - first,
                    "block_range": float(np.max(finite) - np.min(finite)),
                    "block_std": float(np.std(finite)),
                }
            )
            output.append(out)
    return output


def _write_report(
    path: Path,
    *,
    traj_paths: Sequence[Path],
    energy_stationarity: Sequence[dict[str, object]],
    composition_stationarity: Sequence[dict[str, object]],
    wc_stationarity: Sequence[dict[str, object]],
    motif_stationarity: Sequence[dict[str, object]],
) -> None:
    def largest(rows: Sequence[dict[str, object]], n: int = 5) -> list[dict[str, object]]:
        return sorted(
            rows,
            key=lambda row: abs(float(row.get("last_minus_first", 0.0))),
            reverse=True,
        )[:n]

    lines = [
        "# MXene Live-Trajectory Convergence Check",
        "",
        "Analyzed trajectories:",
        *[f"- {path}" for path in traj_paths],
        "",
        "Interpretation:",
        "- energy fluctuations should remain finite; look for stable block means;",
        "- layer composition, WC-SRO, and motif populations should not drift block-to-block;",
        "- convergence is strongest when independent replicas/seeds show consistent block statistics.",
        "",
        "Largest last-block minus first-block changes:",
        "",
        "Energy:",
    ]
    for row in largest(energy_stationarity):
        lines.append(
            f"- {row.get('traj')}: {row.get('metric')} "
            f"delta={float(row['last_minus_first']):.6g}"
        )
    lines.append("")
    lines.append("Layer Composition:")
    for row in largest(composition_stationarity):
        lines.append(
            f"- layer={row.get('layer_id')} {row.get('metric')} "
            f"delta={float(row['last_minus_first']):.6g}"
        )
    lines.append("")
    lines.append("Global WC-SRO:")
    for row in largest(wc_stationarity):
        lines.append(
            f"- {row.get('central_species')}-{row.get('neighbor_species')} "
            f"delta={float(row['last_minus_first']):.6g}"
        )
    lines.append("")
    lines.append("Surface Motifs:")
    for row in largest(motif_stationarity):
        lines.append(
            f"- {row.get('motif_key')} delta={float(row['last_minus_first']):.6g}"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _short_label(value: object, *, max_len: int = 60) -> str:
    label = str(value)
    if "/" in label:
        label = Path(label).stem
    return textwrap.shorten(label, width=max_len, placeholder="...")


def _load_pyplot():
    mpl_config = os.environ.get("MPLCONFIGDIR")
    if not mpl_config or not os.access(Path(mpl_config).expanduser(), os.W_OK):
        cache_dir = Path(tempfile.gettempdir()) / f"gcmc-matplotlib-{os.getuid()}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional matplotlib
        return None, exc
    return plt, None


def _group_rows(
    rows: Sequence[dict[str, object]],
    columns: Sequence[str],
) -> dict[tuple[object, ...], list[dict[str, object]]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(column, "") for column in columns)].append(dict(row))
    return grouped


def _plot_block_lines(
    plt,
    path: Path,
    rows_by_label: Sequence[tuple[str, Sequence[dict[str, object]], str]],
    *,
    ylabel: str,
    title: str,
) -> bool:
    plotted = False
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for label, rows, column in rows_by_label:
        points: list[tuple[float, float]] = []
        for row in rows:
            x = _finite_float(row.get("block"))
            y = _finite_float(row.get(column))
            if x is not None and y is not None:
                points.append((x, y))
        if not points:
            continue
        points.sort()
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=4.0, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("time block")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small", loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _write_plots(
    out_dir: Path,
    *,
    energy_rows: Sequence[dict[str, object]],
    layer_rows: Sequence[dict[str, object]],
    wc_rows: Sequence[dict[str, object]],
    motif_rows: Sequence[dict[str, object]],
    alloy_elements: Sequence[str],
    plot_format: str,
) -> list[Path]:
    plt, error = _load_pyplot()
    plots_dir = out_dir / "plots"
    if error is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)
        (plots_dir / "plotting_error.txt").write_text(
            "Plot generation was skipped because matplotlib could not be imported:\n"
            f"{error}\n"
        )
        return []

    written: list[Path] = []

    energy_groups = _group_rows(energy_rows, ["traj"])
    path = plots_dir / f"energy_blocks.{plot_format}"
    rows_by_label = [
        (_short_label(traj), rows, "energy_mean_eV_per_atom")
        for (traj,), rows in sorted(energy_groups.items())
    ]
    if _plot_block_lines(
        plt,
        path,
        rows_by_label,
        ylabel="mean energy / atom (eV)",
        title="Block-averaged MC energy",
    ):
        written.append(path)

    comp_entries: list[tuple[str, Sequence[dict[str, object]], str]] = []
    layer_groups = _group_rows(layer_rows, ["traj", "layer_id"])
    for (traj, layer_id), rows in sorted(layer_groups.items()):
        for element in alloy_elements:
            column = f"frac_{element}_mean"
            if any(_finite_float(row.get(column)) is not None for row in rows):
                comp_entries.append(
                    (
                        f"{_short_label(traj, max_len=24)} L{layer_id} {element}",
                        rows,
                        column,
                    )
                )
    path = plots_dir / f"layer_composition_blocks.{plot_format}"
    if _plot_block_lines(
        plt,
        path,
        comp_entries,
        ylabel="layer fraction",
        title="Block-averaged alloy layer composition",
    ):
        written.append(path)

    wc_groups = _group_rows(wc_rows, ["traj", "central_species", "neighbor_species"])
    wc_entries = [
        (
            f"{_short_label(traj, max_len=18)} {central}-{neighbor}",
            rows,
            "alpha_wc_mean",
        )
        for (traj, central, neighbor), rows in sorted(wc_groups.items())
    ]
    path = plots_dir / f"wc_sro_blocks.{plot_format}"
    if _plot_block_lines(
        plt,
        path,
        wc_entries,
        ylabel="Warren-Cowley alpha",
        title="Block-averaged global WC-SRO",
    ):
        written.append(path)

    motif_groups = _group_rows(motif_rows, ["traj", "motif_key"])
    motif_entries = [
        (
            f"{_short_label(traj, max_len=18)} {_short_label(motif, max_len=34)}",
            rows,
            "site_fraction_mean",
        )
        for (traj, motif), rows in sorted(motif_groups.items())
    ]
    path = plots_dir / f"surface_motif_blocks.{plot_format}"
    if _plot_block_lines(
        plt,
        path,
        motif_entries,
        ylabel="site fraction",
        title="Block-averaged top surface motifs",
    ):
        written.append(path)

    if written:
        (plots_dir / "README.md").write_text(
            "\n".join(
                [
                    "# Convergence Plots",
                    "",
                    "These quick-look figures are generated from the block CSV files.",
                    "",
                    *[f"- `{path.name}`" for path in written],
                    "",
                ]
            )
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check MXene alloy MC convergence from live replica trajectories.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional workflow config.yaml for element inference.")
    parser.add_argument("--traj", nargs="+", required=True, help="Trajectory path(s) or glob(s).")
    parser.add_argument("--dat", nargs="*", default=None, help="Optional matching thermo .dat files.")
    parser.add_argument("--elements", nargs="+", default=None, help="Alloy elements. Inferred from config composition/swap_elements when omitted.")
    parser.add_argument("--site-elements", nargs="+", default=None, help="Surface site/termination elements. Inferred from first trajectory when omitted.")
    parser.add_argument("--core-elements", nargs="+", default=list(_DEFAULT_CORE_ELEMENTS), help="Non-surface MXene core elements excluded during site-element inference.")
    parser.add_argument("--axis", default="z", choices=["x", "y", "z"], help="Layer axis.")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of alloy layers.")
    parser.add_argument("--wc-cutoff", type=float, default=4.0, help="WC-SRO neighbor cutoff in A.")
    parser.add_argument("--start", type=int, default=0, help="First trajectory frame.")
    parser.add_argument("--stop", type=int, default=None, help="Stop trajectory frame.")
    parser.add_argument("--step", type=int, default=1, help="Trajectory frame stride.")
    parser.add_argument("--n-blocks", type=int, default=5, help="Number of time blocks.")
    parser.add_argument("--top-motifs", type=int, default=10, help="Number of motifs to track.")
    parser.add_argument("--out-dir", type=Path, default=Path("convergence_check"), help="Output directory.")
    parser.add_argument("--no-plots", action="store_true", help="Skip quick-look plot generation.")
    parser.add_argument("--plot-format", default="png", choices=["png", "pdf", "svg"], help="Quick-look plot format.")
    args = parser.parse_args()

    config = _load_config(args.config)
    traj_paths = _expand_paths(args.traj)
    if not traj_paths:
        raise ValueError("No trajectory files matched --traj.")
    dat_paths = _expand_paths(args.dat) if args.dat else []
    dat_by_stem = {path.stem: path for path in dat_paths}
    alloy_elements = args.elements or _infer_alloy_elements(config)
    if not alloy_elements:
        raise ValueError(
            "Could not infer alloy elements. Pass --elements or provide --config "
            "with system.composition or mc.swap_elements."
        )
    site_elements = args.site_elements or _infer_site_elements(
        traj_paths,
        alloy_elements=alloy_elements,
        core_elements=args.core_elements,
    )
    if not site_elements:
        raise ValueError(
            "Could not infer site elements. Pass --site-elements explicitly."
        )

    ordering = MXeneOrderingAnalyzer(
        alloy_elements=alloy_elements,
        layer_axis=args.axis,
        n_layers=args.n_layers,
        wc_cutoff=args.wc_cutoff,
    )
    motifs = MXeneSurfaceMotifAnalyzer(
        alloy_elements=alloy_elements,
        site_elements=site_elements,
        layer_axis=args.axis,
        surface_only=True,
    )

    out_dir = Path(args.out_dir)
    energy_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    wc_rows: list[dict[str, object]] = []
    motif_rows: list[dict[str, object]] = []

    for traj in traj_paths:
        atoms0 = read(str(traj), index=0)
        dat_path = dat_by_stem.get(traj.stem, traj.with_suffix(".dat"))
        energy_rows.extend(
            _block_energy_rows(
                traj,
                _parse_dat(dat_path),
                n_blocks=args.n_blocks,
                n_atoms=len(atoms0),
            )
        )

        ordering_result = ordering.analyze_trajectory(
            traj,
            start=args.start,
            stop=args.stop,
            step=args.step,
        )
        layer_rows.extend(
            _block_layer_composition_rows(
                ordering_result.get("layer_composition_per_frame", []),
                elements=alloy_elements,
                n_blocks=args.n_blocks,
            )
        )
        wc_rows.extend(
            _block_wc_rows(
                ordering_result.get("wc_global_per_frame", []),
                n_blocks=args.n_blocks,
            )
        )

        motif_result = motifs.analyze_trajectory(
            traj,
            start=args.start,
            stop=args.stop,
            step=args.step,
        )
        motif_rows.extend(
            _block_motif_rows(
                motif_result.get("surface_motifs_per_frame", []),
                n_blocks=args.n_blocks,
                top_n=args.top_motifs,
            )
        )

    _write_csv(out_dir / "block_energy.csv", energy_rows)
    _write_csv(out_dir / "block_layer_composition.csv", layer_rows)
    _write_csv(out_dir / "block_wc_sro.csv", wc_rows)
    _write_csv(out_dir / "block_surface_motifs.csv", motif_rows)

    energy_stat = _stationarity_rows(
        energy_rows,
        metric_columns=["energy_mean_eV_per_atom"],
        group_columns=["traj"],
    )
    comp_stat = _stationarity_rows(
        layer_rows,
        metric_columns=[f"frac_{element}_mean" for element in alloy_elements],
        group_columns=["traj", "layer_id"],
    )
    wc_stat = _stationarity_rows(
        wc_rows,
        metric_columns=["alpha_wc_mean"],
        group_columns=["traj", "central_species", "neighbor_species"],
    )
    motif_stat = _stationarity_rows(
        motif_rows,
        metric_columns=["site_fraction_mean"],
        group_columns=["traj", "motif_key"],
    )
    _write_csv(out_dir / "stationarity_energy.csv", energy_stat)
    _write_csv(out_dir / "stationarity_layer_composition.csv", comp_stat)
    _write_csv(out_dir / "stationarity_wc_sro.csv", wc_stat)
    _write_csv(out_dir / "stationarity_surface_motifs.csv", motif_stat)
    _write_report(
        out_dir / "convergence_report.md",
        traj_paths=traj_paths,
        energy_stationarity=energy_stat,
        composition_stationarity=comp_stat,
        wc_stationarity=wc_stat,
        motif_stationarity=motif_stat,
    )
    plot_paths: list[Path] = []
    if not args.no_plots:
        plot_paths = _write_plots(
            out_dir,
            energy_rows=energy_rows,
            layer_rows=layer_rows,
            wc_rows=wc_rows,
            motif_rows=motif_rows,
            alloy_elements=alloy_elements,
            plot_format=args.plot_format,
        )

    print(f"Wrote convergence diagnostics to: {out_dir}")
    print(f"  trajectories: {len(traj_paths)}")
    print(f"  alloy elements: {' '.join(alloy_elements)}")
    print(f"  site elements: {' '.join(site_elements)}")
    print(f"  blocks: {args.n_blocks}")
    if args.no_plots:
        print("  plots: skipped")
    elif plot_paths:
        print(f"  plots: {len(plot_paths)} files under {out_dir / 'plots'}")
    else:
        print(f"  plots: none written; see {out_dir / 'plots' / 'plotting_error.txt'} if present")
    print("  inspect convergence_report.md first, then plots/ and the block_*.csv files")


if __name__ == "__main__":
    main()
