"""Ordering analysis utilities for 2D MXene alloy trajectories."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from ase.io import iread, read
from ase.neighborlist import neighbor_list


AXIS_MAP = {"x": 0, "y": 1, "z": 2}


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _grouped_mean_std(
    rows: List[Dict[str, object]],
    group_keys: List[str],
    value_keys: List[str],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)

    out_rows: List[Dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda kv: kv[0]):
        out: Dict[str, object] = {k: v for k, v in zip(group_keys, key)}
        out["n_frames"] = len(group)
        for vk in value_keys:
            vals = np.array([float(g[vk]) for g in group], dtype=float)
            out[f"{vk}_mean"] = float(np.nanmean(vals))
            out[f"{vk}_std"] = float(np.nanstd(vals))
        out_rows.append(out)
    return out_rows


class MXeneOrderingAnalyzer:
    """General ordering analysis for layered 2D alloy trajectories."""

    def __init__(
        self,
        alloy_elements: Iterable[str] = ("Ti", "Zr"),
        layer_axis: str = "z",
        n_layers: int | None = None,
        layer_gap: float | None = None,
        align_translation: bool = True,
        wc_cutoff: float = 3.3,
    ) -> None:
        if layer_axis not in AXIS_MAP:
            raise ValueError(f"layer_axis must be one of {tuple(AXIS_MAP)}.")
        if wc_cutoff <= 0:
            raise ValueError("wc_cutoff must be > 0.")
        self.alloy_elements = tuple(alloy_elements)
        if len(self.alloy_elements) < 2:
            raise ValueError("Need at least two alloy elements for ordering analysis.")
        self.layer_axis = layer_axis
        self.axis_idx = AXIS_MAP[layer_axis]
        self.n_layers = n_layers
        self.layer_gap = layer_gap
        self.align_translation = bool(align_translation)
        self.wc_cutoff = float(wc_cutoff)

    @staticmethod
    def _parse_temperature_from_traj_name(traj: str) -> float:
        m = re.search(r"replica_([0-9]+(?:\\.[0-9]+)?)K", Path(traj).name)
        if m is None:
            return float("nan")
        return float(m.group(1))

    def _infer_layer_centers(self, coord_1d: np.ndarray) -> np.ndarray:
        if coord_1d.size == 0:
            raise ValueError("No selected alloy atoms found for layer inference.")

        coord_sorted = np.sort(coord_1d)
        diffs = np.diff(coord_sorted)

        if self.n_layers is not None:
            if self.n_layers < 1:
                raise ValueError("n_layers must be >= 1.")
            if self.n_layers == 1:
                split_idx = np.array([], dtype=int)
            else:
                if diffs.size < (self.n_layers - 1):
                    raise ValueError(
                        f"Cannot infer {self.n_layers} layers from {coord_1d.size} selected atoms."
                    )
                split_idx = np.argsort(diffs)[-(self.n_layers - 1) :]
                split_idx = np.sort(split_idx)
        else:
            if self.layer_gap is None:
                # Keep robust behavior for layered systems with many equal coordinates.
                auto_gap = max(0.15, 3.0 * float(np.median(diffs)))
            else:
                auto_gap = float(self.layer_gap)
            split_idx = np.where(diffs > auto_gap)[0]

        groups: List[np.ndarray] = []
        start = 0
        for idx in split_idx:
            groups.append(coord_sorted[start : idx + 1])
            start = idx + 1
        groups.append(coord_sorted[start:])
        centers = np.array([float(np.mean(g)) for g in groups], dtype=float)
        return np.sort(centers)

    @staticmethod
    def _assign_layers(coord_1d: np.ndarray, centers: np.ndarray) -> np.ndarray:
        dist = np.abs(coord_1d[:, None] - centers[None, :])
        return np.argmin(dist, axis=1)

    @staticmethod
    def _iter_frames(traj_path: Path, start: int, stop: int | None, step: int):
        index = f"{start}:{'' if stop is None else stop}:{step}"
        for local_idx, atoms in enumerate(iread(str(traj_path), index=index)):
            frame_idx = start + local_idx * step
            yield frame_idx, atoms

    def _build_neighbor_data(
        self,
        atoms,
        alloy_mask: np.ndarray,
        alloy_symbols: np.ndarray,
        alloy_layers: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        i_idx, j_idx = neighbor_list("ij", atoms, self.wc_cutoff)
        if i_idx.size == 0:
            return {
                "central_species": np.array([], dtype=object),
                "neighbor_species": np.array([], dtype=object),
                "central_layer": np.array([], dtype=int),
                "neighbor_layer": np.array([], dtype=int),
            }

        keep = alloy_mask[i_idx] & alloy_mask[j_idx]
        i_idx = i_idx[keep]
        j_idx = j_idx[keep]
        if i_idx.size == 0:
            return {
                "central_species": np.array([], dtype=object),
                "neighbor_species": np.array([], dtype=object),
                "central_layer": np.array([], dtype=int),
                "neighbor_layer": np.array([], dtype=int),
            }

        # Map full-atom index -> alloy-local index.
        global_to_alloy = -np.ones(len(atoms), dtype=int)
        alloy_global_idx = np.where(alloy_mask)[0]
        global_to_alloy[alloy_global_idx] = np.arange(len(alloy_global_idx))
        i_alloy = global_to_alloy[i_idx]
        j_alloy = global_to_alloy[j_idx]

        # Directed edges for conditional probabilities p(j|i).
        central_alloy = np.concatenate([i_alloy, j_alloy])
        neighbor_alloy = np.concatenate([j_alloy, i_alloy])

        return {
            "central_species": alloy_symbols[central_alloy],
            "neighbor_species": alloy_symbols[neighbor_alloy],
            "central_layer": alloy_layers[central_alloy],
            "neighbor_layer": alloy_layers[neighbor_alloy],
        }

    def analyze_trajectory(
        self,
        traj_path: str | Path,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> Dict[str, object]:
        traj_path = Path(traj_path)
        ref_atoms = read(str(traj_path), index=start)

        ref_symbols = np.array(ref_atoms.get_chemical_symbols(), dtype=object)
        ref_mask = np.isin(ref_symbols, self.alloy_elements)
        if not np.any(ref_mask):
            raise ValueError(
                f"No atoms matched alloy_elements={self.alloy_elements} in {traj_path}."
            )
        ref_coord = ref_atoms.positions[ref_mask, self.axis_idx]
        ref_center = float(np.mean(ref_coord))
        layer_centers_ref = self._infer_layer_centers(ref_coord)

        rows_layer_comp: List[Dict[str, object]] = []
        rows_wc_global: List[Dict[str, object]] = []
        rows_wc_layer: List[Dict[str, object]] = []
        rows_coord_global: List[Dict[str, object]] = []
        rows_coord_layer: List[Dict[str, object]] = []
        rows_connectivity: List[Dict[str, object]] = []

        for frame_idx, atoms in self._iter_frames(traj_path, start=start, stop=stop, step=step):
            symbols = np.array(atoms.get_chemical_symbols(), dtype=object)
            alloy_mask = np.isin(symbols, self.alloy_elements)
            alloy_symbols = symbols[alloy_mask]
            alloy_coord = atoms.positions[alloy_mask, self.axis_idx]
            if alloy_coord.size == 0:
                continue

            shift = float(np.mean(alloy_coord) - ref_center) if self.align_translation else 0.0
            layer_centers = layer_centers_ref + shift
            alloy_layers = self._assign_layers(alloy_coord, layer_centers)

            # Layer-resolved composition.
            for lid in range(len(layer_centers_ref)):
                in_layer = alloy_layers == lid
                n_layer = int(np.sum(in_layer))
                row = {
                    "traj": str(traj_path),
                    "frame": frame_idx,
                    "layer_id": lid,
                    "layer_pos_ref_A": float(layer_centers_ref[lid]),
                    "layer_pos_A": float(layer_centers[lid]),
                    "n_alloy_layer": n_layer,
                }
                for el in self.alloy_elements:
                    n_el = int(np.sum(alloy_symbols[in_layer] == el))
                    row[f"count_{el}"] = n_el
                    row[f"frac_{el}"] = float(n_el / n_layer) if n_layer > 0 else float("nan")
                rows_layer_comp.append(row)

            # WC-SRO + coordination.
            nb = self._build_neighbor_data(atoms, alloy_mask, alloy_symbols, alloy_layers)
            cs = nb["central_species"]
            ns = nb["neighbor_species"]
            cl = nb["central_layer"]
            nl = nb["neighbor_layer"]

            conc = {
                el: float(np.mean(alloy_symbols == el))
                for el in self.alloy_elements
            }

            # Connectivity metrics.
            if cs.size > 0:
                same_layer = float(np.mean(cl == nl))
            else:
                same_layer = float("nan")
            rows_connectivity.append(
                {
                    "traj": str(traj_path),
                    "frame": frame_idx,
                    "same_layer_neighbor_frac": same_layer,
                    "inter_layer_neighbor_frac": float(1.0 - same_layer)
                    if not np.isnan(same_layer)
                    else float("nan"),
                }
            )

            for el_i in self.alloy_elements:
                n_atoms_i = int(np.sum(alloy_symbols == el_i))
                mask_i = cs == el_i
                total_neighbors_i = int(np.sum(mask_i))
                mean_coord = (
                    float(total_neighbors_i / n_atoms_i) if n_atoms_i > 0 else float("nan")
                )
                rows_coord_global.append(
                    {
                        "traj": str(traj_path),
                        "frame": frame_idx,
                        "central_species": el_i,
                        "n_atoms_central": n_atoms_i,
                        "mean_coordination": mean_coord,
                    }
                )
                for el_j in self.alloy_elements:
                    if total_neighbors_i > 0 and conc[el_j] > 0:
                        p_ij = float(np.sum(mask_i & (ns == el_j)) / total_neighbors_i)
                        alpha_ij = float(1.0 - p_ij / conc[el_j])
                    else:
                        alpha_ij = float("nan")
                    rows_wc_global.append(
                        {
                            "traj": str(traj_path),
                            "frame": frame_idx,
                            "central_species": el_i,
                            "neighbor_species": el_j,
                            "alpha_wc": alpha_ij,
                        }
                    )

            for lid in range(len(layer_centers_ref)):
                layer_mask = cl == lid
                for el_i in self.alloy_elements:
                    n_atoms_i_layer = int(np.sum((alloy_layers == lid) & (alloy_symbols == el_i)))
                    mask_i_layer = layer_mask & (cs == el_i)
                    total_neighbors_i_layer = int(np.sum(mask_i_layer))
                    mean_coord_layer = (
                        float(total_neighbors_i_layer / n_atoms_i_layer)
                        if n_atoms_i_layer > 0
                        else float("nan")
                    )
                    rows_coord_layer.append(
                        {
                            "traj": str(traj_path),
                            "frame": frame_idx,
                            "layer_id": lid,
                            "central_species": el_i,
                            "n_atoms_central_layer": n_atoms_i_layer,
                            "mean_coordination": mean_coord_layer,
                        }
                    )
                    for el_j in self.alloy_elements:
                        if total_neighbors_i_layer > 0 and conc[el_j] > 0:
                            p_ij_l = float(
                                np.sum(mask_i_layer & (ns == el_j)) / total_neighbors_i_layer
                            )
                            alpha_ij_l = float(1.0 - p_ij_l / conc[el_j])
                        else:
                            alpha_ij_l = float("nan")
                        rows_wc_layer.append(
                            {
                                "traj": str(traj_path),
                                "frame": frame_idx,
                                "layer_id": lid,
                                "central_species": el_i,
                                "neighbor_species": el_j,
                                "alpha_wc": alpha_ij_l,
                            }
                        )

        result = {
            "traj": str(traj_path),
            "layer_centers_ref_A": layer_centers_ref,
            "layer_composition_per_frame": rows_layer_comp,
            "wc_global_per_frame": rows_wc_global,
            "wc_layer_per_frame": rows_wc_layer,
            "coord_global_per_frame": rows_coord_global,
            "coord_layer_per_frame": rows_coord_layer,
            "connectivity_per_frame": rows_connectivity,
            "layer_composition_summary": _grouped_mean_std(
                rows_layer_comp,
                group_keys=["traj", "layer_id"],
                value_keys=[f"frac_{el}" for el in self.alloy_elements],
            )
            if rows_layer_comp
            else [],
            "wc_global_summary": _grouped_mean_std(
                rows_wc_global,
                group_keys=["traj", "central_species", "neighbor_species"],
                value_keys=["alpha_wc"],
            )
            if rows_wc_global
            else [],
            "wc_layer_summary": _grouped_mean_std(
                rows_wc_layer,
                group_keys=["traj", "layer_id", "central_species", "neighbor_species"],
                value_keys=["alpha_wc"],
            )
            if rows_wc_layer
            else [],
            "coord_global_summary": _grouped_mean_std(
                rows_coord_global,
                group_keys=["traj", "central_species"],
                value_keys=["mean_coordination"],
            )
            if rows_coord_global
            else [],
            "coord_layer_summary": _grouped_mean_std(
                rows_coord_layer,
                group_keys=["traj", "layer_id", "central_species"],
                value_keys=["mean_coordination"],
            )
            if rows_coord_layer
            else [],
            "connectivity_summary": _grouped_mean_std(
                rows_connectivity,
                group_keys=["traj"],
                value_keys=["same_layer_neighbor_frac", "inter_layer_neighbor_frac"],
            )
            if rows_connectivity
            else [],
        }
        result["phase_indicators"] = self.compute_phase_indicators(result)
        return result

    def compute_phase_indicators(self, result: Dict[str, object]) -> Dict[str, object]:
        traj = str(result.get("traj", ""))
        row: Dict[str, object] = {
            "traj": traj,
            "temperature_K": self._parse_temperature_from_traj_name(traj),
            "n_layers": int(len(result.get("layer_centers_ref_A", []))),
        }

        layer_rows = result.get("layer_composition_per_frame", [])
        frame_map: Dict[int, List[Dict[str, object]]] = {}
        for r in layer_rows:
            frame_map.setdefault(int(r["frame"]), []).append(r)
        frame_ids = sorted(frame_map)
        row["n_frames"] = len(frame_ids)

        # Global composition over frames, plus layer polarization indicators.
        for el in self.alloy_elements:
            global_frac = []
            top_minus_bottom = []
            layer_spread = []
            for fid in frame_ids:
                rows_f = frame_map[fid]
                n_tot = float(sum(float(x["n_alloy_layer"]) for x in rows_f))
                c_tot = float(sum(float(x[f"count_{el}"]) for x in rows_f))
                global_frac.append(c_tot / n_tot if n_tot > 0 else float("nan"))

                layer_fracs = {
                    int(x["layer_id"]): float(x[f"frac_{el}"])
                    for x in rows_f
                }
                if layer_fracs:
                    lids = sorted(layer_fracs)
                    top_minus_bottom.append(layer_fracs[lids[-1]] - layer_fracs[lids[0]])
                    layer_spread.append(max(layer_fracs.values()) - min(layer_fracs.values()))

            row[f"global_frac_{el}_mean"] = (
                float(np.nanmean(global_frac)) if global_frac else float("nan")
            )
            row[f"global_frac_{el}_std"] = (
                float(np.nanstd(global_frac)) if global_frac else float("nan")
            )
            row[f"layer_top_minus_bottom_frac_{el}_mean"] = (
                float(np.nanmean(top_minus_bottom)) if top_minus_bottom else float("nan")
            )
            row[f"layer_top_minus_bottom_frac_{el}_std"] = (
                float(np.nanstd(top_minus_bottom)) if top_minus_bottom else float("nan")
            )
            row[f"layer_frac_spread_{el}_mean"] = (
                float(np.nanmean(layer_spread)) if layer_spread else float("nan")
            )
            row[f"layer_frac_spread_{el}_std"] = (
                float(np.nanstd(layer_spread)) if layer_spread else float("nan")
            )

        # Layer summary composition details.
        for r in result.get("layer_composition_summary", []):
            lid = int(r["layer_id"])
            for el in self.alloy_elements:
                row[f"layer{lid}_frac_{el}_mean"] = float(r.get(f"frac_{el}_mean", np.nan))
                row[f"layer{lid}_frac_{el}_std"] = float(r.get(f"frac_{el}_std", np.nan))

        # WC global summary terms.
        wc_g = result.get("wc_global_summary", [])
        for r in wc_g:
            i = str(r["central_species"])
            j = str(r["neighbor_species"])
            row[f"wc_global_{i}_{j}_mean"] = float(r.get("alpha_wc_mean", np.nan))
            row[f"wc_global_{i}_{j}_std"] = float(r.get("alpha_wc_std", np.nan))

        # WC layer contrast (spread and top-bottom delta).
        wc_l = result.get("wc_layer_summary", [])
        if wc_l:
            by_pair: Dict[Tuple[str, str], Dict[int, float]] = {}
            for r in wc_l:
                key = (str(r["central_species"]), str(r["neighbor_species"]))
                by_pair.setdefault(key, {})[int(r["layer_id"])] = float(
                    r.get("alpha_wc_mean", np.nan)
                )
            for (i, j), lid_map in by_pair.items():
                vals = [v for _, v in sorted(lid_map.items())]
                if vals:
                    row[f"wc_layer_spread_{i}_{j}"] = float(np.nanmax(vals) - np.nanmin(vals))
                    lids = sorted(lid_map)
                    row[f"wc_layer_top_minus_bottom_{i}_{j}"] = float(
                        lid_map[lids[-1]] - lid_map[lids[0]]
                    )

        # Coordination summary.
        for r in result.get("coord_global_summary", []):
            i = str(r["central_species"])
            row[f"coord_global_{i}_mean"] = float(r.get("mean_coordination_mean", np.nan))
            row[f"coord_global_{i}_std"] = float(r.get("mean_coordination_std", np.nan))

        for r in result.get("coord_layer_summary", []):
            lid = int(r["layer_id"])
            i = str(r["central_species"])
            row[f"coord_layer{lid}_{i}_mean"] = float(r.get("mean_coordination_mean", np.nan))
            row[f"coord_layer{lid}_{i}_std"] = float(r.get("mean_coordination_std", np.nan))

        # Connectivity summary.
        conn = result.get("connectivity_summary", [])
        if conn:
            c0 = conn[0]
            row["same_layer_neighbor_frac_mean"] = float(
                c0.get("same_layer_neighbor_frac_mean", np.nan)
            )
            row["same_layer_neighbor_frac_std"] = float(
                c0.get("same_layer_neighbor_frac_std", np.nan)
            )
            row["inter_layer_neighbor_frac_mean"] = float(
                c0.get("inter_layer_neighbor_frac_mean", np.nan)
            )
            row["inter_layer_neighbor_frac_std"] = float(
                c0.get("inter_layer_neighbor_frac_std", np.nan)
            )

        return row

    def export_csv(self, result: Dict[str, object], out_prefix: str | Path) -> None:
        out_prefix = Path(out_prefix)

        per_frame_specs = [
            (
                "layer_composition_per_frame",
                [
                    "traj",
                    "frame",
                    "layer_id",
                    "layer_pos_ref_A",
                    "layer_pos_A",
                    "n_alloy_layer",
                ]
                + [f"count_{el}" for el in self.alloy_elements]
                + [f"frac_{el}" for el in self.alloy_elements],
            ),
            (
                "wc_global_per_frame",
                ["traj", "frame", "central_species", "neighbor_species", "alpha_wc"],
            ),
            (
                "wc_layer_per_frame",
                [
                    "traj",
                    "frame",
                    "layer_id",
                    "central_species",
                    "neighbor_species",
                    "alpha_wc",
                ],
            ),
            (
                "coord_global_per_frame",
                ["traj", "frame", "central_species", "n_atoms_central", "mean_coordination"],
            ),
            (
                "coord_layer_per_frame",
                [
                    "traj",
                    "frame",
                    "layer_id",
                    "central_species",
                    "n_atoms_central_layer",
                    "mean_coordination",
                ],
            ),
            (
                "connectivity_per_frame",
                ["traj", "frame", "same_layer_neighbor_frac", "inter_layer_neighbor_frac"],
            ),
        ]

        summary_specs = [
            ("layer_composition_summary", None),
            ("wc_global_summary", None),
            ("wc_layer_summary", None),
            ("coord_global_summary", None),
            ("coord_layer_summary", None),
            ("connectivity_summary", None),
        ]

        for key, fields in per_frame_specs:
            rows = result.get(key, [])
            if rows:
                if fields is None:
                    fields = list(rows[0].keys())
                _write_csv(Path(f"{out_prefix}_{key}.csv"), rows, fields)

        for key, fields in summary_specs:
            rows = result.get(key, [])
            if rows:
                if fields is None:
                    fields = list(rows[0].keys())
                _write_csv(Path(f"{out_prefix}_{key}.csv"), rows, fields)

        phase_row = result.get("phase_indicators")
        if isinstance(phase_row, dict) and phase_row:
            _write_csv(
                Path(f"{out_prefix}_phase_summary.csv"),
                [phase_row],
                list(phase_row.keys()),
            )
