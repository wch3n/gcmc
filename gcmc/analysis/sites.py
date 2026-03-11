"""Adsorption-site candidate generation for layered MXene alloy trajectories."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from ase.geometry import find_mic
from ase.io import iread, read

from gcmc.utils import get_hollow_xy

from .ordering import AXIS_MAP, _write_csv


class MXeneAdsorptionSiteAnalyzer:
    """Generate metal-centered adsorption-site candidates for MXene surfaces."""

    def __init__(
        self,
        alloy_elements: Iterable[str] = ("Ti", "Zr"),
        termination_elements: Iterable[str] = ("O",),
        layer_axis: str = "z",
        n_alloy_layers: int | None = None,
        alloy_layer_gap: float | None = None,
        align_translation: bool = True,
        bridge_cutoff: float | None = None,
        bridge_cutoff_scale: float = 1.15,
        env_shell_size: int = 3,
        support_xy_tol: float = 2.5,
        vertical_offset: float = 1.5,
        min_termination_dist: float = 0.75,
    ) -> None:
        if layer_axis not in AXIS_MAP:
            raise ValueError(f"layer_axis must be one of {tuple(AXIS_MAP)}.")
        self.alloy_elements = tuple(alloy_elements)
        self.termination_elements = tuple(termination_elements)
        if not self.alloy_elements:
            raise ValueError("Need at least one alloy element.")
        self.axis_idx = AXIS_MAP[layer_axis]
        self.layer_axis = layer_axis
        self.n_alloy_layers = n_alloy_layers
        self.alloy_layer_gap = alloy_layer_gap
        self.align_translation = bool(align_translation)
        self.bridge_cutoff = bridge_cutoff
        self.bridge_cutoff_scale = float(bridge_cutoff_scale)
        self.env_shell_size = int(env_shell_size)
        self.support_xy_tol = float(support_xy_tol)
        self.vertical_offset = float(vertical_offset)
        self.min_termination_dist = float(min_termination_dist)
        if self.env_shell_size < 0:
            raise ValueError("env_shell_size must be >= 0.")
        if self.support_xy_tol <= 0.0:
            raise ValueError("support_xy_tol must be > 0.")
        if self.vertical_offset <= 0.0:
            raise ValueError("vertical_offset must be > 0.")
        if self.min_termination_dist < 0.0:
            raise ValueError("min_termination_dist must be >= 0.")

    @staticmethod
    def _parse_temperature_from_traj_name(traj: str) -> float:
        match = re.search(r"replica_([0-9]+(?:\.[0-9]+)?)K", Path(traj).name)
        if match is None:
            return float("nan")
        return float(match.group(1))

    @staticmethod
    def _iter_frames(traj_path: Path, start: int, stop: int | None, step: int):
        index = f"{start}:{'' if stop is None else stop}:{step}"
        for local_idx, atoms in enumerate(iread(str(traj_path), index=index)):
            frame_idx = start + local_idx * step
            yield frame_idx, atoms

    def _infer_layer_centers(self, coord_1d: np.ndarray) -> np.ndarray:
        if coord_1d.size == 0:
            raise ValueError("No atoms available for layer inference.")

        coord_sorted = np.sort(coord_1d)
        diffs = np.diff(coord_sorted)

        if self.n_alloy_layers is not None:
            if self.n_alloy_layers < 1:
                raise ValueError("n_alloy_layers must be >= 1.")
            if self.n_alloy_layers == 1:
                split_idx = np.array([], dtype=int)
            else:
                if diffs.size < (self.n_alloy_layers - 1):
                    raise ValueError(
                        f"Cannot infer {self.n_alloy_layers} alloy layers from {coord_1d.size} atoms."
                    )
                split_idx = np.argsort(diffs)[-(self.n_alloy_layers - 1) :]
                split_idx = np.sort(split_idx)
        else:
            if self.alloy_layer_gap is None:
                auto_gap = max(0.15, 3.0 * float(np.median(diffs))) if diffs.size else 0.15
            else:
                auto_gap = float(self.alloy_layer_gap)
            split_idx = np.where(diffs > auto_gap)[0]

        groups: List[np.ndarray] = []
        start = 0
        for idx in split_idx:
            groups.append(coord_sorted[start : idx + 1])
            start = idx + 1
        groups.append(coord_sorted[start:])
        centers = np.array([float(np.mean(group)) for group in groups], dtype=float)
        return np.sort(centers)

    @staticmethod
    def _assign_layers(coord_1d: np.ndarray, centers: np.ndarray) -> np.ndarray:
        dist = np.abs(coord_1d[:, None] - centers[None, :])
        return np.argmin(dist, axis=1)

    @staticmethod
    def _surface_layer_map(centers: np.ndarray) -> Dict[int, str]:
        if centers.size <= 1:
            return {0: "surface"}
        order = np.argsort(centers)
        return {int(order[0]): "bottom", int(order[-1]): "top"}

    @staticmethod
    def _format_counts(symbols: np.ndarray, allowed_symbols: Tuple[str, ...]) -> Tuple[str, Dict[str, int]]:
        counts = {el: int(np.sum(symbols == el)) for el in allowed_symbols}
        key = "+".join(f"{el}{counts[el]}" for el in allowed_symbols if counts[el] > 0)
        return (key if key else "none"), counts

    @staticmethod
    def _safe_mean(values: np.ndarray) -> float:
        if values.size == 0:
            return float("nan")
        return float(np.mean(values))

    @staticmethod
    def _safe_std(values: np.ndarray) -> float:
        if values.size == 0:
            return float("nan")
        return float(np.std(values))

    @staticmethod
    def _cell_2d(cell: np.ndarray) -> np.ndarray:
        return np.column_stack((cell[0, :2], cell[1, :2]))

    def _wrap_xy(self, xy: np.ndarray, cell: np.ndarray) -> np.ndarray:
        cell_2d = self._cell_2d(cell)
        frac = np.linalg.solve(cell_2d, np.asarray(xy, dtype=float))
        frac -= np.floor(frac)
        return np.dot(cell_2d, frac)

    def _xy_distances(
        self,
        xy: np.ndarray,
        points_xy: np.ndarray,
        cell: np.ndarray,
        pbc: np.ndarray,
    ) -> np.ndarray:
        if points_xy.size == 0:
            return np.array([], dtype=float)
        displacements = np.zeros((len(points_xy), 3), dtype=float)
        displacements[:, :2] = points_xy - xy[None, :]
        displacements_mic, _ = find_mic(displacements, cell, pbc)
        return np.linalg.norm(displacements_mic[:, :2], axis=1)

    def _midpoint_xy(self, pos_i: np.ndarray, pos_j: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        disp = np.zeros((1, 3), dtype=float)
        disp[0] = pos_j - pos_i
        disp_mic, _ = find_mic(disp, cell, pbc)
        xy = pos_i[:2] + 0.5 * disp_mic[0, :2]
        return self._wrap_xy(xy, cell)

    def _auto_bridge_cutoff(self, positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> float:
        if positions.shape[0] < 2:
            return float("nan")
        nearest = []
        for i in range(positions.shape[0]):
            disp = np.zeros((positions.shape[0] - 1, 3), dtype=float)
            others = np.delete(positions, i, axis=0)
            disp[:] = others - positions[i]
            disp_mic, _ = find_mic(disp, cell, pbc)
            nearest.append(float(np.min(np.linalg.norm(disp_mic[:, :2], axis=1))))
        return float(np.median(nearest)) * self.bridge_cutoff_scale

    def _support_anchor(
        self,
        atoms,
        xy: np.ndarray,
        surface_side: str,
        slab_midpoint: float,
    ) -> Tuple[float, float, int]:
        positions = atoms.positions
        side_mask = positions[:, self.axis_idx] >= slab_midpoint if surface_side == "top" else positions[:, self.axis_idx] <= slab_midpoint
        side_positions = positions[side_mask]
        side_indices = np.where(side_mask)[0]
        distances_xy = self._xy_distances(
            xy,
            side_positions[:, :2],
            atoms.cell.array,
            atoms.pbc,
        )
        support_mask = distances_xy <= self.support_xy_tol
        if not np.any(support_mask):
            return float("nan"), float("nan"), 0

        support_positions = side_positions[support_mask]
        anchor_z = (
            float(np.max(support_positions[:, self.axis_idx]))
            if surface_side == "top"
            else float(np.min(support_positions[:, self.axis_idx]))
        )
        suggested_z = (
            anchor_z + self.vertical_offset
            if surface_side == "top"
            else anchor_z - self.vertical_offset
        )
        return anchor_z, suggested_z, int(np.sum(support_mask))

    def _termination_distance(
        self,
        xy: np.ndarray,
        suggested_z: float,
        surface_side: str,
        slab_midpoint: float,
        termination_positions: np.ndarray,
        cell: np.ndarray,
        pbc: np.ndarray,
    ) -> float:
        if termination_positions.size == 0 or not np.isfinite(suggested_z):
            return float("nan")

        side_mask = (
            termination_positions[:, self.axis_idx] >= slab_midpoint
            if surface_side == "top"
            else termination_positions[:, self.axis_idx] <= slab_midpoint
        )
        side_termination = termination_positions[side_mask]
        if side_termination.size == 0:
            return float("nan")

        disp = np.zeros((len(side_termination), 3), dtype=float)
        disp[:, :2] = side_termination[:, :2] - xy[None, :]
        disp[:, self.axis_idx] = side_termination[:, self.axis_idx] - suggested_z
        disp_mic, _ = find_mic(disp, cell, pbc)
        return float(np.min(np.linalg.norm(disp_mic, axis=1)))

    def _unique_sites(self, site_rows: List[Dict[str, object]], cell: np.ndarray) -> List[Dict[str, object]]:
        unique: List[Dict[str, object]] = []
        seen: Dict[Tuple[str, str], List[np.ndarray]] = defaultdict(list)
        tol = 1e-4
        for row in site_rows:
            key = (str(row["surface_side"]), str(row["site_type"]))
            xy = self._wrap_xy(np.array([float(row["x_A"]), float(row["y_A"])]), cell)
            if any(np.linalg.norm(xy - other) < tol for other in seen[key]):
                continue
            row["x_A"] = float(xy[0])
            row["y_A"] = float(xy[1])
            seen[key].append(xy)
            unique.append(row)
        return unique

    def _support_and_env_keys(
        self,
        atoms,
        xy: np.ndarray,
        alloy_indices: np.ndarray,
        support_indices: np.ndarray,
    ) -> Tuple[str, Dict[str, int], str, Dict[str, int], float]:
        symbols = np.array(atoms.get_chemical_symbols(), dtype=object)
        support_symbols = symbols[support_indices]
        support_key, support_counts = self._format_counts(support_symbols, self.alloy_elements)

        env_key = "none"
        env_counts = {el: 0 for el in self.alloy_elements}
        env_mean_dist = float("nan")
        if self.env_shell_size > 0:
            remaining = np.array([idx for idx in alloy_indices if idx not in set(support_indices.tolist())], dtype=int)
            if remaining.size > 0:
                distances = np.asarray(
                    self._xy_distances(xy, atoms.positions[remaining, :2], atoms.cell.array, atoms.pbc),
                    dtype=float,
                )
                order = np.argsort(distances)
                env_indices = remaining[order[: min(self.env_shell_size, order.size)]]
                env_symbols = symbols[env_indices]
                env_key, env_counts = self._format_counts(env_symbols, self.alloy_elements)
                env_mean_dist = self._safe_mean(distances[order[: min(self.env_shell_size, order.size)]])

        return support_key, support_counts, env_key, env_counts, env_mean_dist

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
        ref_alloy_mask = np.isin(ref_symbols, self.alloy_elements)
        if not np.any(ref_alloy_mask):
            raise ValueError(
                f"No atoms matched alloy_elements={self.alloy_elements} in {traj_path}."
            )

        ref_alloy_coord = ref_atoms.positions[ref_alloy_mask, self.axis_idx]
        ref_alloy_center = float(np.mean(ref_alloy_coord))
        alloy_layer_centers_ref = self._infer_layer_centers(ref_alloy_coord)
        surface_layer_map = self._surface_layer_map(alloy_layer_centers_ref)

        temperature = self._parse_temperature_from_traj_name(str(traj_path))
        rows_candidates: List[Dict[str, object]] = []
        frame_totals: Dict[Tuple[int, str, str], int] = defaultdict(int)

        for frame_idx, atoms in self._iter_frames(traj_path, start=start, stop=stop, step=step):
            symbols = np.array(atoms.get_chemical_symbols(), dtype=object)
            alloy_indices = np.where(np.isin(symbols, self.alloy_elements))[0]
            if alloy_indices.size == 0:
                continue

            alloy_coord = atoms.positions[alloy_indices, self.axis_idx]
            shift = (
                float(np.mean(alloy_coord) - ref_alloy_center)
                if self.align_translation
                else 0.0
            )
            alloy_layer_centers = alloy_layer_centers_ref + shift
            alloy_layers = self._assign_layers(alloy_coord, alloy_layer_centers)
            slab_midpoint = float(np.mean(atoms.positions[:, self.axis_idx]))
            termination_positions = atoms.positions[
                np.isin(symbols, self.termination_elements)
            ]

            per_frame_candidates: List[Dict[str, object]] = []
            for layer_id, surface_side in sorted(surface_layer_map.items()):
                surface_mask_local = alloy_layers == layer_id
                surface_indices = alloy_indices[surface_mask_local]
                if surface_indices.size == 0:
                    continue

                surface_positions = atoms.positions[surface_indices]
                bridge_cutoff = (
                    float(self.bridge_cutoff)
                    if self.bridge_cutoff is not None
                    else self._auto_bridge_cutoff(surface_positions, atoms.cell.array, atoms.pbc)
                )

                # Atop sites.
                for atom_index in surface_indices:
                    pos = atoms.positions[int(atom_index)]
                    per_frame_candidates.append(
                        {
                            "traj": str(traj_path),
                            "temperature_K": temperature,
                            "frame": frame_idx,
                            "surface_side": surface_side,
                            "site_type": "atop",
                            "x_A": float(pos[0]),
                            "y_A": float(pos[1]),
                            "support_indices": str(int(atom_index)),
                            "support_n": 1,
                            "surface_layer_id": int(layer_id),
                            "surface_layer_pos_ref_A": float(alloy_layer_centers_ref[layer_id]),
                            "surface_layer_pos_A": float(alloy_layer_centers[layer_id]),
                        }
                    )

                # Bridge sites.
                if surface_indices.size >= 2 and np.isfinite(bridge_cutoff):
                    for i_local, idx_i in enumerate(surface_indices[:-1]):
                        for idx_j in surface_indices[i_local + 1 :]:
                            vector = np.asarray(
                                atoms.get_distances(int(idx_i), int(idx_j), mic=True, vector=True),
                                dtype=float,
                            )
                            dist_xy = float(np.linalg.norm(vector[:2]))
                            if dist_xy <= bridge_cutoff:
                                xy = self._midpoint_xy(
                                    atoms.positions[int(idx_i)],
                                    atoms.positions[int(idx_j)],
                                    atoms.cell.array,
                                    atoms.pbc,
                                )
                                per_frame_candidates.append(
                                    {
                                        "traj": str(traj_path),
                                        "temperature_K": temperature,
                                        "frame": frame_idx,
                                        "surface_side": surface_side,
                                        "site_type": "bridge",
                                        "x_A": float(xy[0]),
                                        "y_A": float(xy[1]),
                                        "support_indices": f"{int(idx_i)};{int(idx_j)}",
                                        "support_n": 2,
                                        "surface_layer_id": int(layer_id),
                                        "surface_layer_pos_ref_A": float(alloy_layer_centers_ref[layer_id]),
                                        "surface_layer_pos_A": float(alloy_layer_centers[layer_id]),
                                    }
                                )

                # Hollow sites.
                if surface_indices.size >= 3:
                    hollow_xy = get_hollow_xy(surface_positions[:, :2], atoms.cell.array)
                    for xy in hollow_xy:
                        distances_xy = self._xy_distances(
                            xy,
                            surface_positions[:, :2],
                            atoms.cell.array,
                            atoms.pbc,
                        )
                        order = np.argsort(distances_xy)
                        support = surface_indices[order[:3]]
                        per_frame_candidates.append(
                            {
                                "traj": str(traj_path),
                                "temperature_K": temperature,
                                "frame": frame_idx,
                                "surface_side": surface_side,
                                "site_type": "hollow",
                                "x_A": float(xy[0]),
                                "y_A": float(xy[1]),
                                "support_indices": ";".join(str(int(idx)) for idx in support),
                                "support_n": 3,
                                "surface_layer_id": int(layer_id),
                                "surface_layer_pos_ref_A": float(alloy_layer_centers_ref[layer_id]),
                                "surface_layer_pos_A": float(alloy_layer_centers[layer_id]),
                            }
                        )

            per_frame_candidates = self._unique_sites(per_frame_candidates, atoms.cell.array)

            for site_id, row in enumerate(per_frame_candidates):
                support_indices = np.array(
                    [int(token) for token in str(row["support_indices"]).split(";") if token],
                    dtype=int,
                )
                xy = np.array([float(row["x_A"]), float(row["y_A"])], dtype=float)
                support_key, support_counts, env_key, env_counts, env_mean_dist = self._support_and_env_keys(
                    atoms,
                    xy,
                    alloy_indices,
                    support_indices,
                )
                anchor_z, suggested_z, n_support_atoms = self._support_anchor(
                    atoms,
                    xy,
                    str(row["surface_side"]),
                    slab_midpoint,
                )
                termination_dist = self._termination_distance(
                    xy,
                    suggested_z,
                    str(row["surface_side"]),
                    slab_midpoint,
                    termination_positions,
                    atoms.cell.array,
                    atoms.pbc,
                )
                motif_key = (
                    f"{row['surface_side']}:{row['site_type']}"
                    f"|support={support_key}|env={env_key}"
                )
                candidate_row = dict(row)
                candidate_row.update(
                    {
                        "site_id": int(site_id),
                        "support_key": support_key,
                        "env_key": env_key,
                        "motif_key": motif_key,
                        "env_mean_xy_dist_A": env_mean_dist,
                        "anchor_z_A": anchor_z,
                        "suggested_z_A": suggested_z,
                        "support_atom_count_local": n_support_atoms,
                        "nearest_termination_dist_A": termination_dist,
                        "blocked_by_termination": (
                            bool(np.isfinite(termination_dist) and termination_dist < self.min_termination_dist)
                        ),
                    }
                )
                for el in self.alloy_elements:
                    candidate_row[f"support_count_{el}"] = support_counts[el]
                    candidate_row[f"env_count_{el}"] = env_counts[el]
                rows_candidates.append(candidate_row)
                frame_totals[(frame_idx, str(row["surface_side"]), str(row["site_type"]))] += 1

        summary_groups: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
        for row in rows_candidates:
            summary_groups[
                (str(row["surface_side"]), str(row["site_type"]), str(row["motif_key"]))
            ].append(row)

        summary_rows: List[Dict[str, object]] = []
        total_frames = len({int(row["frame"]) for row in rows_candidates})
        total_sites = len(rows_candidates)
        for (surface_side, site_type, motif_key), group in sorted(summary_groups.items()):
            frames_present = sorted({int(row["frame"]) for row in group})
            summary: Dict[str, object] = {
                "traj": str(traj_path),
                "temperature_K": temperature,
                "surface_side": surface_side,
                "site_type": site_type,
                "motif_key": motif_key,
                "support_key": str(group[0]["support_key"]),
                "env_key": str(group[0]["env_key"]),
                "n_occurrences": len(group),
                "n_frames_present": len(frames_present),
                "n_frames_total": total_frames,
                "frame_presence_frac": (
                    float(len(frames_present) / total_frames) if total_frames > 0 else float("nan")
                ),
                "site_fraction": (
                    float(len(group) / total_sites) if total_sites > 0 else float("nan")
                ),
                "blocked_fraction": float(np.mean([bool(row["blocked_by_termination"]) for row in group])),
            }
            for el in self.alloy_elements:
                summary[f"support_count_{el}"] = int(group[0][f"support_count_{el}"])
                summary[f"env_count_{el}"] = int(group[0][f"env_count_{el}"])
            for field in (
                "env_mean_xy_dist_A",
                "anchor_z_A",
                "suggested_z_A",
                "nearest_termination_dist_A",
            ):
                values = np.array([float(row[field]) for row in group], dtype=float)
                finite = values[np.isfinite(values)]
                summary[f"{field}_mean"] = self._safe_mean(finite)
                summary[f"{field}_std"] = self._safe_std(finite)
            summary_rows.append(summary)

        return {
            "traj": str(traj_path),
            "temperature_K": temperature,
            "alloy_layer_centers_ref_A": alloy_layer_centers_ref,
            "candidate_sites": rows_candidates,
            "candidate_site_summary": summary_rows,
        }

    def export_csv(self, result: Dict[str, object], out_prefix: str | Path) -> None:
        out_prefix = Path(out_prefix)

        candidate_rows = result.get("candidate_sites", [])
        if candidate_rows:
            _write_csv(
                Path(f"{out_prefix}_candidate_sites.csv"),
                candidate_rows,
                list(candidate_rows[0].keys()),
            )

        summary_rows = result.get("candidate_site_summary", [])
        if summary_rows:
            _write_csv(
                Path(f"{out_prefix}_candidate_site_summary.csv"),
                summary_rows,
                list(summary_rows[0].keys()),
            )
