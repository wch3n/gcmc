"""Reference-connectivity SRO and layer-LRO analysis for alloy MXenes."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml
from ase import Atoms
from ase.io import iread, read
from ase.io.trajectory import Trajectory


_AXIS = {"x": 0, "y": 1, "z": 2}
_TEMPERATURE_RE = re.compile(r"replica_([0-9]+(?:\.[0-9]+)?)K")


def _load_config(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Alloy config must contain a YAML mapping.")
    return raw


def _section(config: dict[str, object], name: str) -> dict[str, object]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def _infer_elements(config: dict[str, object]) -> list[str]:
    composition = _section(config, "system").get("composition")
    if not isinstance(composition, dict):
        composition = config.get("composition")
    if isinstance(composition, dict) and len(composition) >= 2:
        return [str(element) for element in composition]

    swap_elements = _section(config, "mc").get("swap_elements")
    if not isinstance(swap_elements, (list, tuple)):
        swap_elements = config.get("swap_elements")
    if isinstance(swap_elements, (list, tuple)) and len(swap_elements) >= 2:
        return [str(element) for element in swap_elements]
    return []


def _temperature(path: Path) -> float:
    match = _TEMPERATURE_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot infer temperature from trajectory name: {path.name}")
    return float(match.group(1))


def _discover_trajectories(
    run_dir: Path,
    config: dict[str, object],
    trajectory_values: Sequence[str] | None,
) -> list[Path]:
    if trajectory_values:
        paths: list[Path] = []
        for value in trajectory_values:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = run_dir / candidate
            if any(char in str(candidate) for char in "*?["):
                paths.extend(candidate.parent.glob(candidate.name))
            else:
                paths.append(candidate)
    else:
        output_dir = _section(config, "output").get("output_dir", "results")
        results_dir = Path(str(output_dir)).expanduser()
        if not results_dir.is_absolute():
            results_dir = run_dir / results_dir
        paths = list(results_dir.glob("replica_*K.traj"))

    paths = list(dict.fromkeys(path.resolve() for path in paths if path.is_file()))
    if not paths:
        raise FileNotFoundError("No replica_*K.traj files were found.")
    return sorted(paths, key=_temperature)


def _safe_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _block_ranges(n_items: int, n_blocks: int) -> list[tuple[int, int]]:
    if n_items < 1:
        return []
    edges = np.linspace(0, n_items, min(n_items, max(1, n_blocks)) + 1, dtype=int)
    return [
        (int(start), int(stop))
        for start, stop in zip(edges[:-1], edges[1:])
        if stop > start
    ]


def _block_sem(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return float("nan")
    return float(np.std(array, ddof=1) / math.sqrt(array.size))


def _finite_random_unlike_alpha(n_sites: int) -> float:
    if n_sites < 2:
        raise ValueError("At least two alloy sites are required for SRO analysis.")
    return -1.0 / float(n_sites - 1)


def _finite_random_layer_polarization(
    layers: np.ndarray,
    n_target: int,
) -> float:
    layer_ids = np.unique(layers)
    if len(layer_ids) < 2:
        raise ValueError("At least two alloy layers are required for polarization.")

    n_sites = len(layers)
    n_bottom = int(np.count_nonzero(layers == layer_ids[0]))
    n_top = int(np.count_nonzero(layers == layer_ids[-1]))
    n_other = n_sites - n_bottom - n_top
    if not 0 <= n_target <= n_sites:
        raise ValueError("Target-element count is incompatible with the alloy sites.")

    total_assignments = math.comb(n_sites, n_target)
    weighted_imbalance = 0
    top_min = max(0, n_target - n_sites + n_top)
    top_max = min(n_top, n_target)
    for n_target_top in range(top_min, top_max + 1):
        remaining = n_target - n_target_top
        bottom_min = max(0, remaining - n_other)
        bottom_max = min(n_bottom, remaining)
        for n_target_bottom in range(bottom_min, bottom_max + 1):
            n_target_other = remaining - n_target_bottom
            assignments = (
                math.comb(n_top, n_target_top)
                * math.comb(n_bottom, n_target_bottom)
                * math.comb(n_other, n_target_other)
            )
            imbalance_numerator = abs(
                n_target_top * n_bottom - n_target_bottom * n_top
            )
            weighted_imbalance += imbalance_numerator * assignments

    return float(weighted_imbalance / (n_top * n_bottom * total_assignments))


def _assign_reference_layers(
    positions: np.ndarray,
    *,
    axis: int,
    n_layers: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_layers < 1 or len(positions) < n_layers:
        raise ValueError("The reference does not contain enough alloy sites for --n-layers.")
    coordinates = np.asarray(positions[:, axis], dtype=float)
    order = np.argsort(coordinates)
    gaps = np.diff(coordinates[order])
    split_indices = (
        np.sort(np.argsort(gaps)[-(n_layers - 1) :]) if n_layers > 1 else np.array([], dtype=int)
    )
    layers = np.empty(len(coordinates), dtype=int)
    centers: list[float] = []
    start = 0
    for layer_id, split in enumerate([*split_indices, len(order) - 1]):
        stop = int(split) + 1
        indices = order[start:stop]
        layers[indices] = layer_id
        centers.append(float(np.mean(coordinates[indices])))
        start = stop
    return layers, np.asarray(centers, dtype=float)


def _infer_relation_coordination(
    distances: np.ndarray,
    layers: np.ndarray,
    *,
    same_layer: bool,
    max_rank: int = 12,
) -> tuple[int, np.ndarray]:
    sorted_rows: list[np.ndarray] = []
    for central in range(len(distances)):
        relation_mask = layers == layers[central]
        if not same_layer:
            relation_mask = ~relation_mask
        relation_mask[central] = False
        values = np.sort(distances[central, relation_mask])
        if values.size:
            sorted_rows.append(values)
    if not sorted_rows:
        relation = "intralayer" if same_layer else "interlayer"
        raise ValueError(f"No {relation} neighbors were found in the reference.")

    n_ranks = min(max_rank, min(len(row) for row in sorted_rows))
    rank_means = np.mean([row[:n_ranks] for row in sorted_rows], axis=0)
    if n_ranks == 1:
        return 1, rank_means
    coordination = int(np.argmax(np.diff(rank_means)) + 1)
    return coordination, rank_means


def _relation_edges(
    distances: np.ndarray,
    layers: np.ndarray,
    *,
    same_layer: bool,
    coordination: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    central_indices: list[int] = []
    neighbor_indices: list[int] = []
    selected_distances: list[float] = []
    excluded_distances: list[float] = []
    for central in range(len(distances)):
        relation_mask = layers == layers[central]
        if not same_layer:
            relation_mask = ~relation_mask
        relation_mask[central] = False
        candidates = np.where(relation_mask)[0]
        order = candidates[np.argsort(distances[central, candidates])]
        if len(order) < coordination:
            raise ValueError("Requested shell coordination exceeds available reference sites.")
        selected = order[:coordination]
        central_indices.extend([central] * coordination)
        neighbor_indices.extend(selected.tolist())
        selected_distances.extend(distances[central, selected].tolist())
        if len(order) > coordination:
            excluded_distances.append(float(distances[central, order[coordination]]))
    next_distance = _safe_mean(excluded_distances) if excluded_distances else float("nan")
    return (
        np.asarray(central_indices, dtype=int),
        np.asarray(neighbor_indices, dtype=int),
        _safe_mean(selected_distances),
        next_distance,
    )


def _build_reference_connectivity(
    reference: Atoms,
    *,
    elements: Sequence[str],
    lro_element: str,
    axis: str,
    n_layers: int,
    intralayer_coordination: int | None,
    interlayer_coordination: int | None,
) -> dict[str, object]:
    symbols = np.asarray(reference.get_chemical_symbols(), dtype=object)
    site_indices = np.where(np.isin(symbols, elements))[0]
    if len(site_indices) == 0:
        raise ValueError(f"No reference sites matched alloy elements {list(elements)}.")
    positions = reference.positions[site_indices]
    site_symbols = symbols[site_indices]
    element_counts = {
        element: int(np.count_nonzero(site_symbols == element)) for element in elements
    }
    layers, layer_centers = _assign_reference_layers(
        positions, axis=_AXIS[axis], n_layers=n_layers
    )
    distances = np.asarray(reference[site_indices].get_all_distances(mic=True), dtype=float)
    np.fill_diagonal(distances, np.inf)

    inferred_intra, intra_ranks = _infer_relation_coordination(
        distances, layers, same_layer=True
    )
    inferred_inter, inter_ranks = _infer_relation_coordination(
        distances, layers, same_layer=False
    )
    intra_coord = int(intralayer_coordination or inferred_intra)
    inter_coord = int(interlayer_coordination or inferred_inter)
    if intra_coord < 1 or inter_coord < 1:
        raise ValueError("Reference coordinations must be positive.")

    intra_i, intra_j, intra_distance, intra_next = _relation_edges(
        distances,
        layers,
        same_layer=True,
        coordination=intra_coord,
    )
    inter_i, inter_j, inter_distance, inter_next = _relation_edges(
        distances,
        layers,
        same_layer=False,
        coordination=inter_coord,
    )
    return {
        "n_atoms": len(reference),
        "n_alloy_sites": len(site_indices),
        "elements": tuple(elements),
        "element_counts": element_counts,
        "lro_element": lro_element,
        "site_indices": site_indices,
        "layers": layers,
        "layer_centers_A": layer_centers,
        "intralayer_i": intra_i,
        "intralayer_j": intra_j,
        "interlayer_i": inter_i,
        "interlayer_j": inter_j,
        "intralayer_coordination": intra_coord,
        "interlayer_coordination": inter_coord,
        "intralayer_distance_A": intra_distance,
        "interlayer_distance_A": inter_distance,
        "intralayer_next_distance_A": intra_next,
        "interlayer_next_distance_A": inter_next,
        "intralayer_rank_means_A": intra_ranks,
        "interlayer_rank_means_A": inter_ranks,
        "alpha_finite_random_baseline": _finite_random_unlike_alpha(
            len(site_indices)
        ),
        "layer_polarization_finite_random_baseline": (
            _finite_random_layer_polarization(
                layers,
                element_counts[lro_element],
            )
        ),
    }


def _relation_alpha(
    site_symbols: np.ndarray,
    central_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    pair: tuple[str, str],
) -> float:
    element_a, element_b = pair
    concentrations = {
        element: float(np.mean(site_symbols == element)) for element in pair
    }
    central_symbols = site_symbols[central_indices]
    neighbor_symbols = site_symbols[neighbor_indices]
    directed: list[float] = []
    for central_element, neighbor_element in ((element_a, element_b), (element_b, element_a)):
        mask = central_symbols == central_element
        concentration = concentrations[neighbor_element]
        if np.any(mask) and concentration > 0.0:
            probability = float(np.mean(neighbor_symbols[mask] == neighbor_element))
            directed.append(1.0 - probability / concentration)
    return _safe_mean(directed)


def _frame_metrics(
    path: Path,
    connectivity: dict[str, object],
    *,
    pair: tuple[str, str],
    lro_element: str,
    start: int,
    stop: int,
    step: int,
) -> list[dict[str, float]]:
    index = f"{start}:{stop}:{step}"
    site_indices = np.asarray(connectivity["site_indices"], dtype=int)
    layers = np.asarray(connectivity["layers"], dtype=int)
    intra_i = np.asarray(connectivity["intralayer_i"], dtype=int)
    intra_j = np.asarray(connectivity["intralayer_j"], dtype=int)
    inter_i = np.asarray(connectivity["interlayer_i"], dtype=int)
    inter_j = np.asarray(connectivity["interlayer_j"], dtype=int)
    unique_layers = sorted(int(layer) for layer in np.unique(layers))

    rows: list[dict[str, float]] = []
    for local_frame, atoms in enumerate(iread(str(path), index=index)):
        if len(atoms) != int(connectivity["n_atoms"]):
            raise ValueError(f"Atom count changed relative to the reference in {path}.")
        symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
        site_symbols = symbols[site_indices]
        if not np.all(np.isin(site_symbols, connectivity["elements"])):
            raise ValueError(
                f"The selected reference sites in {path} contain species outside "
                f"{connectivity['elements']}."
            )
        element_counts = {
            element: int(np.count_nonzero(site_symbols == element))
            for element in connectivity["elements"]
        }
        if element_counts != connectivity["element_counts"]:
            raise ValueError(
                f"Alloy composition changed relative to the reference in {path}."
            )
        layer_fractions = {
            layer: float(np.mean(site_symbols[layers == layer] == lro_element))
            for layer in unique_layers
        }
        frame = start + local_frame * step
        rows.append(
            {
                "frame": float(frame),
                "alpha_intralayer_1nn": _relation_alpha(
                    site_symbols, intra_i, intra_j, pair
                ),
                "alpha_interlayer_nearest": _relation_alpha(
                    site_symbols, inter_i, inter_j, pair
                ),
                "layer_polarization": abs(
                    layer_fractions[unique_layers[-1]] - layer_fractions[unique_layers[0]]
                ),
            }
        )
    return rows


def _summarize_trajectory(
    path: Path,
    connectivity: dict[str, object],
    *,
    pair: tuple[str, str],
    lro_element: str,
    start: int | None,
    start_fraction: float,
    stop: int | None,
    step: int,
    n_blocks: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    trajectory = Trajectory(path)
    try:
        n_total = len(trajectory)
    finally:
        trajectory.close()
    first = int(start) if start is not None else int(math.floor(n_total * start_fraction))
    first = max(0, min(first, n_total - 1))
    last = n_total if stop is None else max(first + 1, min(int(stop), n_total))
    metrics = _frame_metrics(
        path,
        connectivity,
        pair=pair,
        lro_element=lro_element,
        start=first,
        stop=last,
        step=step,
    )
    if not metrics:
        raise ValueError(f"No finite ordering metrics were obtained from {path}.")

    block_rows: list[dict[str, object]] = []
    metric_names = (
        "alpha_intralayer_1nn",
        "alpha_interlayer_nearest",
        "layer_polarization",
    )
    for block, (block_start, block_stop) in enumerate(
        _block_ranges(len(metrics), n_blocks), start=1
    ):
        group = metrics[block_start:block_stop]
        row: dict[str, object] = {
            "traj": str(path),
            "temperature_K": _temperature(path),
            "block": block,
            "n_frames": len(group),
            "frame_start": int(group[0]["frame"]),
            "frame_stop": int(group[-1]["frame"]),
        }
        for metric in metric_names:
            row[f"{metric}_mean"] = _safe_mean([entry[metric] for entry in group])
        block_rows.append(row)

    summary: dict[str, object] = {
        "traj": str(path),
        "temperature_K": _temperature(path),
        "n_total_frames": n_total,
        "analysis_start": first,
        "analysis_stop": last,
        "analysis_step": step,
        "n_analyzed_frames": len(metrics),
        "n_blocks": len(block_rows),
        "n_alloy_sites": int(connectivity["n_alloy_sites"]),
        "alpha_finite_random_baseline": float(
            connectivity["alpha_finite_random_baseline"]
        ),
        "layer_polarization_finite_random_baseline": float(
            connectivity["layer_polarization_finite_random_baseline"]
        ),
        "intralayer_coordination": int(connectivity["intralayer_coordination"]),
        "interlayer_coordination": int(connectivity["interlayer_coordination"]),
        "intralayer_distance_A": float(connectivity["intralayer_distance_A"]),
        "interlayer_distance_A": float(connectivity["interlayer_distance_A"]),
    }
    for metric in metric_names:
        block_values = [float(row[f"{metric}_mean"]) for row in block_rows]
        summary[f"{metric}_mean"] = _safe_mean(block_values)
        summary[f"{metric}_sem"] = _block_sem(block_values)
    return summary, block_rows


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    rows: Sequence[dict[str, object]],
    *,
    pair: tuple[str, str],
    lro_element: str,
    output_prefix: Path,
) -> list[Path]:
    mpl_config = os.environ.get("MPLCONFIGDIR")
    if not mpl_config or not os.access(Path(mpl_config).expanduser(), os.W_OK):
        cache = Path(tempfile.gettempdir()) / f"gcmc-matplotlib-{os.getuid()}"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    width = 16.0 / 2.54
    plt.rcParams.update(
        {
            "axes.linewidth": 0.6,
            "axes.labelsize": 9,
            "font.size": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "DejaVu Sans"],
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "lines.markersize": 3.5,
        }
    )

    temperatures = np.asarray([float(row["temperature_K"]) for row in rows])
    random_baselines = np.asarray(
        [float(row["alpha_finite_random_baseline"]) for row in rows]
    )
    if not np.allclose(random_baselines, random_baselines[0]):
        raise ValueError("All trajectories in one figure must use the same SRO baseline.")
    random_baseline = float(random_baselines[0])
    polarization_random_baselines = np.asarray(
        [
            float(row["layer_polarization_finite_random_baseline"])
            for row in rows
        ]
    )
    if not np.allclose(
        polarization_random_baselines,
        polarization_random_baselines[0],
    ):
        raise ValueError(
            "All trajectories in one figure must use the same polarization baseline."
        )
    polarization_random_baseline = float(polarization_random_baselines[0])
    fig, (ax_sro, ax_lro) = plt.subplots(1, 2, figsize=(width, width * 0.4))
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.96, wspace=0.32)

    for metric, label, color, marker in (
        ("alpha_intralayer_1nn", "Intralayer 1NN", "#32688E", "o"),
        ("alpha_interlayer_nearest", "Nearest interlayer", "#3C8C6E", "^"),
    ):
        values = np.asarray([float(row[f"{metric}_mean"]) for row in rows])
        errors = np.asarray([float(row[f"{metric}_sem"]) for row in rows])
        ax_sro.errorbar(
            temperatures,
            values,
            yerr=errors,
            color=color,
            marker=marker,
            linewidth=1.2,
            capsize=2.0,
            label=label,
        )

    ax_sro.axhline(
        random_baseline,
        color="0.35",
        linestyle="--",
        linewidth=0.7,
        zorder=0,
        label="Finite-cell random",
    )

    polarization = np.asarray(
        [float(row["layer_polarization_mean"]) for row in rows]
    )
    polarization_sem = np.asarray(
        [float(row["layer_polarization_sem"]) for row in rows]
    )
    ax_lro.errorbar(
        temperatures,
        polarization,
        yerr=polarization_sem,
        color="#B55432",
        marker="s",
        linewidth=1.2,
        capsize=2.0,
    )

    for axis in (ax_sro, ax_lro):
        axis.set_xlabel("MC temperature (K)")
    ax_lro.axhline(0.0, color="0.35", linewidth=0.6, zorder=0)
    ax_lro.axhline(
        polarization_random_baseline,
        color="0.35",
        linestyle="--",
        linewidth=0.7,
        zorder=0,
        label="Finite-cell random",
    )
    ax_sro.set_ylabel(rf"Warren-Cowley $\alpha_{{{pair[0]}-{pair[1]}}}$")
    ax_sro.legend(frameon=False, loc="best")
    ax_lro.set_ylabel(rf"Layer polarization $|\Delta x_{{{lro_element}}}|$")
    ax_lro.legend(frameon=False, loc="best")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_prefix.with_suffix(".pdf"), output_prefix.with_suffix(".png")]
    for path in outputs:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gcmc-analyze-alloy-ordering",
        description=(
            "Plot fixed-reference intralayer/interlayer Warren-Cowley SRO and "
            "MXene metal-layer polarization with block standard errors."
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--traj",
        nargs="+",
        default=None,
        help="Optional trajectories or globs; defaults to output_dir/replica_*K.traj.",
    )
    parser.add_argument("--elements", nargs="+", default=None)
    parser.add_argument("--pair", nargs=2, default=None, metavar=("A", "B"))
    parser.add_argument("--lro-element", default=None)
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--intralayer-coordination", type=int, default=None)
    parser.add_argument("--interlayer-coordination", type=int, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--start-fraction", type=float, default=0.5)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--n-blocks", type=int, default=5)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("ordering/sro_lro_vs_temperature"),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def run(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_dir = args.run_dir.expanduser().resolve()
    config_path = args.config or run_dir / "config.yaml"
    if not config_path.is_absolute():
        config_path = run_dir / config_path
    config = _load_config(config_path)

    elements = list(args.elements or _infer_elements(config))
    if len(elements) < 2:
        raise ValueError("Pass --elements or provide at least two config composition elements.")
    pair = tuple(args.pair or elements[:2])
    if any(element not in elements for element in pair):
        raise ValueError("Both --pair elements must be present in --elements/config composition.")
    if pair[0] == pair[1]:
        raise ValueError("--pair must contain two distinct elements.")
    lro_element = str(args.lro_element or pair[0])
    if lro_element not in elements:
        raise ValueError("--lro-element must be present in --elements/config composition.")
    if not (0.0 <= args.start_fraction < 1.0):
        raise ValueError("--start-fraction must be in [0, 1).")
    if args.step < 1 or args.n_blocks < 1:
        raise ValueError("--step and --n-blocks must be positive.")

    trajectories = _discover_trajectories(run_dir, config, args.traj)
    reference_path = args.reference
    if reference_path is None:
        reference_path = trajectories[0]
    elif not reference_path.is_absolute():
        reference_path = run_dir / reference_path
    reference = read(str(reference_path), index=args.reference_frame)
    connectivity = _build_reference_connectivity(
        reference,
        elements=elements,
        lro_element=lro_element,
        axis=args.axis,
        n_layers=args.n_layers,
        intralayer_coordination=args.intralayer_coordination,
        interlayer_coordination=args.interlayer_coordination,
    )
    print(f"Reference: {reference_path} frame {args.reference_frame}")
    print(
        "Intralayer 1NN: "
        f"z={connectivity['intralayer_coordination']} "
        f"mean_distance={connectivity['intralayer_distance_A']:.4f} A "
        f"next_shell={connectivity['intralayer_next_distance_A']:.4f} A"
    )
    print(
        "Nearest interlayer: "
        f"z={connectivity['interlayer_coordination']} "
        f"mean_distance={connectivity['interlayer_distance_A']:.4f} A "
        f"next_shell={connectivity['interlayer_next_distance_A']:.4f} A"
    )
    print(
        "Finite-cell random SRO: "
        f"alpha={connectivity['alpha_finite_random_baseline']:+.6f} "
        f"(N_alloy={connectivity['n_alloy_sites']})"
    )
    print(
        "Finite-cell random layer polarization: "
        f"{connectivity['layer_polarization_finite_random_baseline']:.6f} "
        f"(element={lro_element})"
    )

    summary_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    for path in trajectories:
        summary, blocks = _summarize_trajectory(
            path,
            connectivity,
            pair=(str(pair[0]), str(pair[1])),
            lro_element=lro_element,
            start=args.start,
            start_fraction=args.start_fraction,
            stop=args.stop,
            step=args.step,
            n_blocks=args.n_blocks,
        )
        summary_rows.append(summary)
        block_rows.extend(blocks)
        print(
            f"{summary['temperature_K']:7.1f} K | frames={summary['n_analyzed_frames']:4d} "
            f"| alpha_intra={summary['alpha_intralayer_1nn_mean']:+.4f} "
            f"| alpha_inter={summary['alpha_interlayer_nearest_mean']:+.4f} "
            f"| layer_polarization={summary['layer_polarization_mean']:.4f}"
        )

    output_prefix = args.output_prefix.expanduser()
    if not output_prefix.is_absolute():
        output_prefix = run_dir / output_prefix
    summary_path = Path(f"{output_prefix}_summary.csv")
    blocks_path = Path(f"{output_prefix}_blocks.csv")
    _write_csv(summary_path, summary_rows)
    _write_csv(blocks_path, block_rows)
    print(f"Summary CSV: {summary_path}")
    print(f"Block CSV: {blocks_path}")
    if not args.no_plots:
        for path in _plot(
            summary_rows,
            pair=(str(pair[0]), str(pair[1])),
            lro_element=lro_element,
            output_prefix=output_prefix,
        ):
            print(f"Figure: {path}")
    return summary_rows, block_rows


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
