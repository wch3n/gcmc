#!/usr/bin/env python3
"""Generic equilibration diagnostics for fixed-sublattice alloy MC/PT runs.

The script is intentionally lighter than the MXene-specific convergence
checker.  It works directly from one or more ASE trajectories, optional
``replica_*.dat`` energy traces, and optional PT swap-stat CSV files.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml
from ase import Atoms
from ase.io import iread, read
from ase.neighborlist import neighbor_list


_DEFAULT_EXCLUDED_ELEMENTS = ("H", "B", "C", "N", "O", "F", "Cl", "Br", "I")
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _expand_paths(values: Sequence[str] | None) -> list[Path]:
    if not values:
        return []
    paths: list[Path] = []
    for value in values:
        matches = sorted(glob.glob(value))
        paths.extend(Path(match) for match in matches) if matches else paths.append(Path(value))
    return list(dict.fromkeys(path.resolve() for path in paths))


def _load_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _section(config: dict[str, object], name: str) -> dict[str, object]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def _infer_elements_from_config(config: dict[str, object]) -> list[str]:
    system = _section(config, "system")
    composition = system.get("composition")
    if isinstance(composition, dict) and len(composition) >= 2:
        return [str(key) for key in composition]

    mc = _section(config, "mc")
    swap_elements = mc.get("swap_elements")
    if isinstance(swap_elements, (list, tuple)) and len(swap_elements) >= 2:
        return [str(value) for value in swap_elements]

    composition = config.get("composition")
    if isinstance(composition, dict) and len(composition) >= 2:
        return [str(key) for key in composition]

    swap_elements = config.get("swap_elements")
    if isinstance(swap_elements, (list, tuple)) and len(swap_elements) >= 2:
        return [str(value) for value in swap_elements]
    return []


def _infer_elements_from_atoms(
    atoms: Atoms,
    *,
    excluded_elements: Sequence[str],
    max_elements: int,
) -> list[str]:
    excluded = set(excluded_elements)
    counts = Counter(atoms.get_chemical_symbols())
    candidates = [(symbol, count) for symbol, count in counts.items() if symbol not in excluded]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return [symbol for symbol, _ in candidates[:max_elements]]


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


def _parse_energy_dat(path: Path | None) -> list[dict[str, float]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            values: list[float] = []
            for part in parts:
                try:
                    values.append(float(part))
                except ValueError:
                    continue
            if len(values) >= 2:
                rows.append({"cycle": values[0], "energy_eV": values[1]})
    return rows


def _slice_energy_rows(
    rows: Sequence[dict[str, float]],
    *,
    start: int,
    stop: int | None,
    step: int,
    energy_start_sample: int | None,
    energy_stop_sample: int | None,
    energy_step: int | None,
    energy_start_cycle: float | None,
    energy_stop_cycle: float | None,
) -> list[dict[str, float]]:
    """Apply sample-index and optional cycle filters to energy rows.

    By default the sample-index slicing mirrors trajectory ``start:stop:step``.
    This keeps block energy and block structure diagnostics aligned for the
    usual case where ``replica_*.traj`` and ``replica_*.dat`` are written at
    the same interval.  The explicit energy options handle runs where they are
    not aligned.
    """

    sample_start = start if energy_start_sample is None else energy_start_sample
    sample_stop = stop if energy_stop_sample is None else energy_stop_sample
    sample_step = step if energy_step is None else energy_step
    selected = list(rows[sample_start:sample_stop:sample_step])
    if energy_start_cycle is not None:
        selected = [row for row in selected if row["cycle"] >= energy_start_cycle]
    if energy_stop_cycle is not None:
        selected = [row for row in selected if row["cycle"] < energy_stop_cycle]
    return selected


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


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), std


def _temperature_from_name(path: Path) -> float:
    match = re.search(r"replica_([0-9]+(?:\.[0-9]+)?)K", path.name)
    return float(match.group(1)) if match else float("nan")


def _matching_dat(traj: Path, dat_paths: Sequence[Path]) -> Path | None:
    by_stem = {path.stem: path for path in dat_paths}
    if traj.stem in by_stem:
        return by_stem[traj.stem]
    candidate = traj.with_suffix(".dat")
    return candidate if candidate.exists() else None


def _iter_selected_frames(traj: Path, start: int, stop: int | None, step: int):
    index = f"{start}:{'' if stop is None else stop}:{step}"
    for local_idx, atoms in enumerate(iread(str(traj), index=index)):
        yield start + local_idx * step, atoms


def _reference_edges(
    atoms: Atoms,
    alloy_indices: np.ndarray,
    alloy_elements: Sequence[str],
    cutoff: float,
) -> list[tuple[int, int]]:
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    alloy_set = set(alloy_elements)
    i_vals, j_vals = neighbor_list("ij", atoms, cutoff)
    edges: list[tuple[int, int]] = []
    selected = set(int(idx) for idx in alloy_indices)
    for i, j in zip(i_vals, j_vals):
        if i >= j:
            continue
        if i in selected and j in selected and symbols[i] in alloy_set and symbols[j] in alloy_set:
            edges.append((int(i), int(j)))
    return edges


def _assign_reference_layers(
    atoms: Atoms,
    alloy_indices: np.ndarray,
    *,
    axis: str,
    n_layers: int,
) -> np.ndarray:
    if n_layers <= 0:
        return np.zeros(len(alloy_indices), dtype=int)
    axis_idx = _AXIS_INDEX[axis]
    coords = np.asarray(atoms.positions[alloy_indices, axis_idx], dtype=float)
    order = np.argsort(coords)
    layers = np.empty(len(alloy_indices), dtype=int)
    for layer_id, group in enumerate(np.array_split(order, n_layers)):
        layers[group] = layer_id
    return layers


def _frame_metrics(
    symbols: np.ndarray,
    *,
    ref_symbols: np.ndarray,
    alloy_indices: np.ndarray,
    alloy_elements: Sequence[str],
    edges: Sequence[tuple[int, int]],
    ref_layers: np.ndarray | None,
) -> dict[str, object]:
    row: dict[str, object] = {}
    alloy_symbols = symbols[alloy_indices]
    ref_alloy_symbols = ref_symbols[alloy_indices]
    row["changed_vs_initial"] = float(np.mean(alloy_symbols != ref_alloy_symbols))

    for element in alloy_elements:
        row[f"frac_{element}"] = float(np.mean(alloy_symbols == element))

    if edges:
        hetero = sum(1 for i, j in edges if symbols[i] != symbols[j])
        row["hetero_nn_frac"] = float(hetero / len(edges))
        for center in alloy_elements:
            center_links = 0
            counts = {neighbor: 0 for neighbor in alloy_elements}
            for i, j in edges:
                if symbols[i] == center:
                    center_links += 1
                    if symbols[j] in counts:
                        counts[str(symbols[j])] += 1
                if symbols[j] == center:
                    center_links += 1
                    if symbols[i] in counts:
                        counts[str(symbols[i])] += 1
            for neighbor in alloy_elements:
                global_fraction = float(np.mean(alloy_symbols == neighbor))
                probability = counts[neighbor] / center_links if center_links else float("nan")
                alpha = 1.0 - probability / global_fraction if global_fraction > 0 else float("nan")
                row[f"wc_alpha_{center}_{neighbor}"] = float(alpha)

    if ref_layers is not None:
        for layer_id in sorted(set(int(value) for value in ref_layers)):
            mask = ref_layers == layer_id
            layer_symbols = alloy_symbols[mask]
            for element in alloy_elements:
                row[f"layer{layer_id}_frac_{element}"] = float(np.mean(layer_symbols == element))
    return row


def _analyze_trajectory(
    traj: Path,
    *,
    dat_path: Path | None,
    alloy_elements: Sequence[str],
    start: int,
    stop: int | None,
    step: int,
    energy_start_sample: int | None,
    energy_stop_sample: int | None,
    energy_step: int | None,
    energy_start_cycle: float | None,
    energy_stop_cycle: float | None,
    n_blocks: int,
    nn_cutoff: float,
    axis: str,
    n_layers: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    first = read(str(traj), index=0)
    ref_symbols = np.asarray(first.get_chemical_symbols(), dtype=object)
    alloy_set = set(alloy_elements)
    alloy_indices = np.asarray([i for i, s in enumerate(ref_symbols) if s in alloy_set], dtype=int)
    if alloy_indices.size == 0:
        raise ValueError(f"No alloy elements {tuple(alloy_elements)} found in {traj}")
    edges = _reference_edges(first, alloy_indices, alloy_elements, nn_cutoff)
    ref_layers = (
        _assign_reference_layers(first, alloy_indices, axis=axis, n_layers=n_layers)
        if n_layers > 0
        else None
    )

    frame_rows: list[dict[str, object]] = []
    labels_seen: set[tuple[str, ...]] = set()
    previous_label: tuple[str, ...] | None = None
    transitions = 0
    for frame, atoms in _iter_selected_frames(traj, start=start, stop=stop, step=step):
        symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
        label = tuple(str(value) for value in symbols[alloy_indices])
        labels_seen.add(label)
        if previous_label is not None and label != previous_label:
            transitions += 1
        previous_label = label
        row = {
            "traj": str(traj),
            "temperature_K": _temperature_from_name(traj),
            "frame": int(frame),
        }
        row.update(
            _frame_metrics(
                symbols,
                ref_symbols=ref_symbols,
                alloy_indices=alloy_indices,
                alloy_elements=alloy_elements,
                edges=edges,
                ref_layers=ref_layers,
            )
        )
        frame_rows.append(row)

    block_rows: list[dict[str, object]] = []
    for block_id, (start_idx, stop_idx) in enumerate(_blocks(len(frame_rows), n_blocks), start=1):
        group = frame_rows[start_idx:stop_idx]
        out: dict[str, object] = {
            "block_type": "order",
            "traj": str(traj),
            "temperature_K": _temperature_from_name(traj),
            "block": block_id,
            "frame_start": group[0]["frame"],
            "frame_stop": group[-1]["frame"],
            "n_frames": len(group),
        }
        metric_keys = [key for key in group[0] if key not in {"traj", "temperature_K", "frame"}]
        for key in metric_keys:
            mean, std = _mean_std(float(row[key]) for row in group)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
        block_rows.append(out)

    raw_energy_rows = _parse_energy_dat(dat_path)
    energy_rows = _slice_energy_rows(
        raw_energy_rows,
        start=start,
        stop=stop,
        step=step,
        energy_start_sample=energy_start_sample,
        energy_stop_sample=energy_stop_sample,
        energy_step=energy_step,
        energy_start_cycle=energy_start_cycle,
        energy_stop_cycle=energy_stop_cycle,
    )
    energy_block_rows: list[dict[str, object]] = []
    for block_id, (start_idx, stop_idx) in enumerate(_blocks(len(energy_rows), n_blocks), start=1):
        group = energy_rows[start_idx:stop_idx]
        energies = [row["energy_eV"] for row in group]
        mean, std = _mean_std(energies)
        energy_block_rows.append(
            {
                "block_type": "energy",
                "traj": str(traj),
                "dat": str(dat_path) if dat_path else "",
                "temperature_K": _temperature_from_name(traj),
                "block": block_id,
                "cycle_start": group[0]["cycle"],
                "cycle_stop": group[-1]["cycle"],
                "n_samples": len(group),
                "energy_mean_eV": mean,
                "energy_std_eV": std,
                "energy_sem_eV": std / math.sqrt(len(group)) if group else float("nan"),
                "energy_mean_eV_per_atom": mean / len(first),
            }
        )

    summary = {
        "traj": str(traj),
        "temperature_K": _temperature_from_name(traj),
        "n_atoms": len(first),
        "formula": first.get_chemical_formula(),
        "n_alloy_sites": int(alloy_indices.size),
        "n_frames": len(frame_rows),
        "unique_alloy_configurations": len(labels_seen),
        "saved_frame_transitions": transitions,
        "nn_cutoff_A": float(nn_cutoff),
        "nn_edges": len(edges),
        "dat": str(dat_path) if dat_path else "",
        "n_energy_samples_raw": len(raw_energy_rows),
        "n_energy_samples": len(energy_rows),
    }
    return frame_rows, block_rows + energy_block_rows, summary


def _parse_pt_stats(path: Path | None) -> list[dict[str, object]]:
    if path is None or not path.exists():
        return []
    grouped: dict[tuple[float, float], list[bool]] = defaultdict(list)
    cycles: list[int] = []
    with path.open() as handle:
        for raw in csv.reader(handle):
            if len(raw) < 6:
                continue
            try:
                cycle = int(float(raw[0]))
                t1 = float(raw[1])
                t2 = float(raw[2])
            except ValueError:
                continue
            accepted = str(raw[5]).strip().lower() in {"true", "1", "yes", "accepted"}
            key = tuple(sorted((round(t1, 6), round(t2, 6))))
            grouped[key].append(accepted)
            cycles.append(cycle)
    rows: list[dict[str, object]] = []
    for (t_low, t_high), accepted in sorted(grouped.items()):
        total = len(accepted)
        count = sum(int(value) for value in accepted)
        rows.append(
            {
                "stats_file": str(path),
                "temperature_low_K": t_low,
                "temperature_high_K": t_high,
                "attempts": total,
                "accepted": count,
                "acceptance_rate": count / total if total else float("nan"),
                "cycle_min": min(cycles) if cycles else "",
                "cycle_max": max(cycles) if cycles else "",
            }
        )
    return rows


def _stationarity_rows(block_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in block_rows:
        key = (str(row["traj"]), str(row.get("temperature_K", "")))
        grouped[key].append(dict(row))

    output: list[dict[str, object]] = []
    for (traj, temperature), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["block"]))
        metric_keys = sorted(
            {
                key
                for row in rows
                for key in row
                if key.endswith("_mean")
                or key in {"energy_mean_eV", "energy_mean_eV_per_atom"}
            }
        )
        for metric in sorted(set(metric_keys)):
            values = []
            for row in rows:
                if metric not in row:
                    continue
                try:
                    values.append(float(row[metric]))
                except (TypeError, ValueError):
                    continue
            values = [value for value in values if np.isfinite(value)]
            if len(values) < 2:
                continue
            first = values[0]
            last = values[-1]
            output.append(
                {
                    "traj": traj,
                    "temperature_K": temperature,
                    "metric": metric,
                    "n_blocks": len(values),
                    "first_block": first,
                    "last_block": last,
                    "last_minus_first": last - first,
                    "block_range": max(values) - min(values),
                    "block_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
            )
    return output


def _write_report(
    path: Path,
    *,
    summaries: Sequence[dict[str, object]],
    stationarity: Sequence[dict[str, object]],
    pt_rows: Sequence[dict[str, object]],
) -> None:
    def top_drift(prefix: str, n: int = 8) -> list[dict[str, object]]:
        rows = [row for row in stationarity if str(row["metric"]).startswith(prefix)]
        return sorted(rows, key=lambda row: abs(float(row["last_minus_first"])), reverse=True)[:n]

    lines = [
        "# Alloy MC Equilibration Diagnostics",
        "",
        "This report is a block-stationarity check.  Equilibration is stronger when",
        "block means stop drifting and independent runs give consistent distributions.",
        "",
        "## Trajectories",
    ]
    for row in summaries:
        lines.append(
            "- {traj}: T={temperature_K:g} K, frames={n_frames}, alloy_sites={n_alloy_sites}, "
            "unique_configs={unique_alloy_configurations}, transitions={saved_frame_transitions}, "
            "energy_samples={n_energy_samples}/{n_energy_samples_raw}".format(**row)
        )

    lines.extend(["", "## Largest Drifts", "", "Energy:"])
    for row in top_drift("energy_"):
        lines.append(
            f"- {Path(str(row['traj'])).name} {row['metric']}: "
            f"{float(row['last_minus_first']):.6g}"
        )
    lines.append("")
    lines.append("Configuration/order metrics:")
    order_rows = [
        row
        for row in stationarity
        if not str(row["metric"]).startswith("energy_")
    ]
    for row in sorted(order_rows, key=lambda item: abs(float(item["last_minus_first"])), reverse=True)[:12]:
        lines.append(
            f"- {Path(str(row['traj'])).name} {row['metric']}: "
            f"{float(row['last_minus_first']):.6g}"
        )

    if pt_rows:
        rates = np.asarray([float(row["acceptance_rate"]) for row in pt_rows], dtype=float)
        lines.extend(
            [
                "",
                "## PT Swap Acceptance",
                f"- pairs: {len(pt_rows)}",
                f"- min/mean/max: {np.min(rates):.3f} / {np.mean(rates):.3f} / {np.max(rates):.3f}",
                "",
                "Lowest-acceptance pairs:",
            ]
        )
        for row in sorted(pt_rows, key=lambda item: float(item["acceptance_rate"]))[:8]:
            lines.append(
                f"- {float(row['temperature_low_K']):g}-{float(row['temperature_high_K']):g} K: "
                f"{float(row['acceptance_rate']):.3f} ({row['accepted']}/{row['attempts']})"
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
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        return None, exc
    return plt, None


def _short_label(path: object) -> str:
    text = str(path)
    return Path(text).stem if "/" in text else text


def _group_by_traj(rows: Sequence[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("traj", ""))].append(dict(row))
    return grouped


def _plot_metric_lines(
    plt,
    path: Path,
    series: Sequence[tuple[str, Sequence[dict[str, object]], str]],
    *,
    ylabel: str,
) -> bool:
    plotted = False
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for label, rows, column in series:
        points: list[tuple[float, float]] = []
        for row in rows:
            block = _finite_float(row.get("block"))
            value = _finite_float(row.get(column))
            if block is not None and value is not None:
                points.append((block, value))
        if not points:
            continue
        points.sort()
        x, y = zip(*points)
        ax.plot(x, y, marker="o", linewidth=1.7, markersize=4.0, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("time block")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small", loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _write_plots(
    out_dir: Path,
    *,
    block_rows: Sequence[dict[str, object]],
    alloy_elements: Sequence[str],
    plot_format: str,
) -> list[Path]:
    plt, error = _load_pyplot()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if error is not None:
        (plots_dir / "plotting_error.txt").write_text(
            "Plotting was skipped because matplotlib could not be imported:\n"
            f"{error}\n"
        )
        return []

    written: list[Path] = []
    energy_rows = [row for row in block_rows if row.get("block_type") == "energy"]
    order_rows = [row for row in block_rows if row.get("block_type") == "order"]

    energy_series = [
        (_short_label(traj), rows, "energy_mean_eV")
        for traj, rows in sorted(_group_by_traj(energy_rows).items())
    ]
    path = plots_dir / f"energy_drift.{plot_format}"
    if _plot_metric_lines(plt, path, energy_series, ylabel="mean energy (eV)"):
        written.append(path)

    hetero_series = [
        (_short_label(traj), rows, "hetero_nn_frac_mean")
        for traj, rows in sorted(_group_by_traj(order_rows).items())
    ]
    path = plots_dir / f"hetero_nn_fraction.{plot_format}"
    if _plot_metric_lines(plt, path, hetero_series, ylabel="hetero NN fraction"):
        written.append(path)

    sro_series: list[tuple[str, Sequence[dict[str, object]], str]] = []
    for traj, rows in sorted(_group_by_traj(order_rows).items()):
        for center in alloy_elements:
            for neighbor in alloy_elements:
                if center == neighbor:
                    continue
                column = f"wc_alpha_{center}_{neighbor}_mean"
                if any(column in row for row in rows):
                    sro_series.append((f"{_short_label(traj)} {center}-{neighbor}", rows, column))
    path = plots_dir / f"wc_sro_unlike_pairs.{plot_format}"
    if _plot_metric_lines(plt, path, sro_series, ylabel="Warren-Cowley alpha"):
        written.append(path)

    layer_series: list[tuple[str, Sequence[dict[str, object]], str]] = []
    for traj, rows in sorted(_group_by_traj(order_rows).items()):
        columns = sorted(
            {
                key
                for row in rows
                for key in row
                if key.startswith("layer") and key.endswith("_mean")
            }
        )
        for column in columns:
            label = column.removesuffix("_mean").replace("_frac_", " ")
            layer_series.append((f"{_short_label(traj)} {label}", rows, column))
    path = plots_dir / f"layer_composition.{plot_format}"
    if _plot_metric_lines(plt, path, layer_series, ylabel="layer fraction"):
        written.append(path)

    (plots_dir / "README.md").write_text(
        "\n".join(
            [
                "# Alloy Equilibration Plots",
                "",
                "Plots are generated from block-averaged diagnostics.",
                "",
                *[f"- `{path.name}`" for path in written],
                "",
            ]
        )
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Block-stationarity diagnostics for fixed-sublattice alloy MC/PT trajectories.",
    )
    parser.add_argument("--traj", nargs="+", required=True, help="Trajectory path(s) or globs.")
    parser.add_argument("--dat", nargs="*", default=None, help="Optional matching replica_*.dat file(s).")
    parser.add_argument("--pt-stats", type=Path, default=None, help="Optional PT replica_stats.csv.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config for element inference.")
    parser.add_argument("--elements", nargs="+", default=None, help="Alloy elements. Inferred from config when omitted.")
    parser.add_argument("--excluded-elements", nargs="+", default=list(_DEFAULT_EXCLUDED_ELEMENTS), help="Elements excluded from fallback alloy-element inference.")
    parser.add_argument("--max-inferred-elements", type=int, default=3, help="Maximum elements to infer from the first trajectory if --elements and --config are omitted.")
    parser.add_argument("--start", type=int, default=0, help="First trajectory frame to analyze.")
    parser.add_argument("--stop", type=int, default=None, help="Stop trajectory frame.")
    parser.add_argument("--step", type=int, default=1, help="Trajectory frame stride.")
    parser.add_argument("--energy-start-sample", type=int, default=None, help="First .dat sample to analyze. Defaults to --start.")
    parser.add_argument("--energy-stop-sample", type=int, default=None, help="Stop .dat sample. Defaults to --stop.")
    parser.add_argument("--energy-step", type=int, default=None, help=".dat sample stride. Defaults to --step.")
    parser.add_argument("--energy-start-cycle", type=float, default=None, help="Discard .dat rows before this MC cycle after sample slicing.")
    parser.add_argument("--energy-stop-cycle", type=float, default=None, help="Discard .dat rows at or after this MC cycle after sample slicing.")
    parser.add_argument("--n-blocks", type=int, default=5, help="Number of stationarity blocks.")
    parser.add_argument("--nn-cutoff", type=float, default=4.0, help="Alloy nearest-neighbor cutoff in Angstrom.")
    parser.add_argument("--axis", choices=sorted(_AXIS_INDEX), default="z", help="Layer axis for optional layer composition.")
    parser.add_argument("--n-layers", type=int, default=0, help="Number of alloy layers to track; 0 disables layer metrics.")
    parser.add_argument("--out-dir", type=Path, default=Path("alloy_equilibration"), help="Output directory.")
    parser.add_argument("--no-plots", action="store_true", help="Skip quick-look plot generation.")
    parser.add_argument("--plot-format", default="png", choices=["png", "pdf", "svg"], help="Plot file format.")
    args = parser.parse_args()

    traj_paths = _expand_paths(args.traj)
    if not traj_paths:
        raise ValueError("No trajectories matched --traj.")
    dat_paths = _expand_paths(args.dat)
    config = _load_config(args.config)

    alloy_elements = list(args.elements or _infer_elements_from_config(config))
    if not alloy_elements:
        alloy_elements = _infer_elements_from_atoms(
            read(str(traj_paths[0]), index=0),
            excluded_elements=args.excluded_elements,
            max_elements=args.max_inferred_elements,
        )
    if len(alloy_elements) < 2:
        raise ValueError("Could not infer at least two alloy elements. Pass --elements explicitly.")

    out_dir = args.out_dir
    frame_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for traj in traj_paths:
        dat_path = _matching_dat(traj, dat_paths)
        frames, blocks, summary = _analyze_trajectory(
            traj,
            dat_path=dat_path,
            alloy_elements=alloy_elements,
            start=args.start,
            stop=args.stop,
            step=args.step,
            energy_start_sample=args.energy_start_sample,
            energy_stop_sample=args.energy_stop_sample,
            energy_step=args.energy_step,
            energy_start_cycle=args.energy_start_cycle,
            energy_stop_cycle=args.energy_stop_cycle,
            n_blocks=args.n_blocks,
            nn_cutoff=args.nn_cutoff,
            axis=args.axis,
            n_layers=args.n_layers,
        )
        frame_rows.extend(frames)
        block_rows.extend(blocks)
        summaries.append(summary)

    pt_rows = _parse_pt_stats(args.pt_stats)
    stationarity = _stationarity_rows(block_rows)

    _write_csv(out_dir / "frame_metrics.csv", frame_rows)
    _write_csv(out_dir / "block_metrics.csv", block_rows)
    _write_csv(out_dir / "stationarity.csv", stationarity)
    _write_csv(out_dir / "pt_swap_acceptance.csv", pt_rows)
    _write_csv(out_dir / "summary.csv", summaries)
    _write_report(
        out_dir / "equilibration_report.md",
        summaries=summaries,
        stationarity=stationarity,
        pt_rows=pt_rows,
    )
    plot_paths: list[Path] = []
    if not args.no_plots:
        plot_paths = _write_plots(
            out_dir,
            block_rows=block_rows,
            alloy_elements=alloy_elements,
            plot_format=args.plot_format,
        )

    print(f"Wrote alloy equilibration diagnostics to: {out_dir}")
    print(f"  trajectories: {len(traj_paths)}")
    print(f"  alloy elements: {' '.join(alloy_elements)}")
    print(f"  blocks: {args.n_blocks}")
    if args.no_plots:
        print("  plots: skipped")
    elif plot_paths:
        print(f"  plots: {len(plot_paths)} files under {out_dir / 'plots'}")
    else:
        print(f"  plots: none written; see plots/plotting_error.txt if present")
    print("  inspect equilibration_report.md, block_metrics.csv, and stationarity.csv")


if __name__ == "__main__":
    main()
