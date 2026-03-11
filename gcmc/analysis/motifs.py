"""Surface-site motif analysis for layered MXene alloy trajectories."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from ase.io import iread, read

from .ordering import AXIS_MAP, _write_csv


class MXeneSurfaceMotifAnalyzer:
    """Identify simple site-centered chemical motifs on MXene surfaces."""

    def __init__(
        self,
        alloy_elements: Iterable[str] = ("Ti", "Zr"),
        site_elements: Iterable[str] = ("O",),
        layer_axis: str = "z",
        n_site_layers: int | None = None,
        site_layer_gap: float | None = None,
        align_translation: bool = True,
        shell1_size: int = 3,
        shell2_size: int = 3,
        surface_only: bool = True,
    ) -> None:
        if layer_axis not in AXIS_MAP:
            raise ValueError(f"layer_axis must be one of {tuple(AXIS_MAP)}.")
        self.alloy_elements = tuple(alloy_elements)
        self.site_elements = tuple(site_elements)
        if not self.alloy_elements:
            raise ValueError("Need at least one alloy element.")
        if not self.site_elements:
            raise ValueError("Need at least one site element.")
        self.axis_idx = AXIS_MAP[layer_axis]
        self.layer_axis = layer_axis
        self.n_site_layers = n_site_layers
        self.site_layer_gap = site_layer_gap
        self.align_translation = bool(align_translation)
        self.shell1_size = int(shell1_size)
        self.shell2_size = int(shell2_size)
        self.surface_only = bool(surface_only)
        if self.shell1_size < 1:
            raise ValueError("shell1_size must be >= 1.")
        if self.shell2_size < 0:
            raise ValueError("shell2_size must be >= 0.")

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

        if self.n_site_layers is not None:
            if self.n_site_layers < 1:
                raise ValueError("n_site_layers must be >= 1.")
            if self.n_site_layers == 1:
                split_idx = np.array([], dtype=int)
            else:
                if diffs.size < (self.n_site_layers - 1):
                    raise ValueError(
                        f"Cannot infer {self.n_site_layers} site layers from {coord_1d.size} atoms."
                    )
                split_idx = np.argsort(diffs)[-(self.n_site_layers - 1) :]
                split_idx = np.sort(split_idx)
        else:
            if self.site_layer_gap is None:
                auto_gap = max(0.15, 3.0 * float(np.median(diffs))) if diffs.size else 0.15
            else:
                auto_gap = float(self.site_layer_gap)
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

    def _surface_layer_map(self, centers: np.ndarray) -> Dict[int, str]:
        if centers.size <= 1:
            return {0: "surface"}
        order = np.argsort(centers)
        layer_map = {int(order[0]): "bottom", int(order[-1]): "top"}
        return layer_map

    def _motif_geometry(
        self,
        atoms,
        site_index: int,
        shell_indices: np.ndarray,
    ) -> Dict[str, float]:
        if shell_indices.size == 0:
            return {
                "mean_dist_A": float("nan"),
                "mean_abs_dz_A": float("nan"),
                "mean_inplane_dist_A": float("nan"),
            }

        distances = np.asarray(
            atoms.get_distances(site_index, shell_indices, mic=True),
            dtype=float,
        )
        vectors = np.asarray(
            atoms.get_distances(site_index, shell_indices, mic=True, vector=True),
            dtype=float,
        )
        dz = np.abs(vectors[:, self.axis_idx])
        inplane_axes = [idx for idx in range(3) if idx != self.axis_idx]
        inplane = np.sqrt(np.sum(vectors[:, inplane_axes] ** 2, axis=1))
        return {
            "mean_dist_A": self._safe_mean(distances),
            "mean_abs_dz_A": self._safe_mean(dz),
            "mean_inplane_dist_A": self._safe_mean(inplane),
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
        ref_alloy_mask = np.isin(ref_symbols, self.alloy_elements)
        ref_site_mask = np.isin(ref_symbols, self.site_elements)
        if not np.any(ref_alloy_mask):
            raise ValueError(
                f"No atoms matched alloy_elements={self.alloy_elements} in {traj_path}."
            )
        if not np.any(ref_site_mask):
            raise ValueError(
                f"No atoms matched site_elements={self.site_elements} in {traj_path}."
            )

        ref_alloy_coord = ref_atoms.positions[ref_alloy_mask, self.axis_idx]
        ref_alloy_center = float(np.mean(ref_alloy_coord))
        ref_site_coord = ref_atoms.positions[ref_site_mask, self.axis_idx]
        site_layer_centers_ref = self._infer_layer_centers(ref_site_coord)
        surface_layer_map = self._surface_layer_map(site_layer_centers_ref)

        rows_per_site: List[Dict[str, object]] = []
        frame_totals: Dict[int, int] = {}
        temperature = self._parse_temperature_from_traj_name(str(traj_path))

        for frame_idx, atoms in self._iter_frames(traj_path, start=start, stop=stop, step=step):
            symbols = np.array(atoms.get_chemical_symbols(), dtype=object)
            alloy_global_idx = np.where(np.isin(symbols, self.alloy_elements))[0]
            site_global_idx = np.where(np.isin(symbols, self.site_elements))[0]
            if alloy_global_idx.size == 0 or site_global_idx.size == 0:
                continue

            alloy_coord = atoms.positions[alloy_global_idx, self.axis_idx]
            shift = (
                float(np.mean(alloy_coord) - ref_alloy_center)
                if self.align_translation
                else 0.0
            )
            site_layer_centers = site_layer_centers_ref + shift
            site_layers = self._assign_layers(
                atoms.positions[site_global_idx, self.axis_idx],
                site_layer_centers,
            )

            surface_site_count = 0
            for local_site_idx, atom_index in enumerate(site_global_idx):
                site_layer_id = int(site_layers[local_site_idx])
                surface_side = surface_layer_map.get(site_layer_id, "inner")
                if self.surface_only and surface_side == "inner":
                    continue

                distances = np.asarray(
                    atoms.get_distances(int(atom_index), alloy_global_idx, mic=True),
                    dtype=float,
                )
                if distances.size == 0:
                    continue
                order = np.argsort(distances)

                shell1_end = min(self.shell1_size, order.size)
                shell2_end = min(shell1_end + self.shell2_size, order.size)
                shell1_idx = alloy_global_idx[order[:shell1_end]]
                shell2_idx = alloy_global_idx[order[shell1_end:shell2_end]]

                shell1_symbols = symbols[shell1_idx]
                shell2_symbols = symbols[shell2_idx]
                shell1_key, shell1_counts = self._format_counts(
                    shell1_symbols,
                    self.alloy_elements,
                )
                shell2_key, shell2_counts = self._format_counts(
                    shell2_symbols,
                    self.alloy_elements,
                )
                motif_key = (
                    f"{surface_side}:{symbols[atom_index]}"
                    f"|shell1={shell1_key}|shell2={shell2_key}"
                )

                shell1_geom = self._motif_geometry(atoms, int(atom_index), shell1_idx)
                shell2_geom = self._motif_geometry(atoms, int(atom_index), shell2_idx)

                row: Dict[str, object] = {
                    "traj": str(traj_path),
                    "temperature_K": temperature,
                    "frame": frame_idx,
                    "site_index": int(atom_index),
                    "site_element": str(symbols[atom_index]),
                    "surface_side": surface_side,
                    "site_layer_id": site_layer_id,
                    "site_layer_pos_ref_A": float(site_layer_centers_ref[site_layer_id]),
                    "site_layer_pos_A": float(site_layer_centers[site_layer_id]),
                    "motif_key": motif_key,
                    "shell1_key": shell1_key,
                    "shell2_key": shell2_key,
                    "shell1_n": int(shell1_idx.size),
                    "shell2_n": int(shell2_idx.size),
                    "shell1_mean_dist_A": shell1_geom["mean_dist_A"],
                    "shell1_mean_abs_dz_A": shell1_geom["mean_abs_dz_A"],
                    "shell1_mean_inplane_dist_A": shell1_geom["mean_inplane_dist_A"],
                    "shell2_mean_dist_A": shell2_geom["mean_dist_A"],
                    "shell2_mean_abs_dz_A": shell2_geom["mean_abs_dz_A"],
                    "shell2_mean_inplane_dist_A": shell2_geom["mean_inplane_dist_A"],
                }
                for el in self.alloy_elements:
                    row[f"shell1_count_{el}"] = shell1_counts[el]
                    row[f"shell2_count_{el}"] = shell2_counts[el]

                rows_per_site.append(row)
                surface_site_count += 1

            if surface_site_count > 0:
                frame_totals[frame_idx] = surface_site_count

        rows_per_frame: List[Dict[str, object]] = []
        frame_motif_counts: Dict[Tuple[int, str, str, str], int] = defaultdict(int)
        for row in rows_per_site:
            key = (
                int(row["frame"]),
                str(row["site_element"]),
                str(row["surface_side"]),
                str(row["motif_key"]),
            )
            frame_motif_counts[key] += 1

        for (frame_idx, site_element, surface_side, motif_key), count in sorted(frame_motif_counts.items()):
            n_surface_sites = frame_totals.get(frame_idx, 0)
            rows_per_frame.append(
                {
                    "traj": str(traj_path),
                    "temperature_K": temperature,
                    "frame": frame_idx,
                    "site_element": site_element,
                    "surface_side": surface_side,
                    "motif_key": motif_key,
                    "n_sites": count,
                    "n_surface_sites_frame": n_surface_sites,
                    "site_fraction_frame": (
                        float(count / n_surface_sites) if n_surface_sites > 0 else float("nan")
                    ),
                }
            )

        total_frames = len(frame_totals)
        total_sites = len(rows_per_site)
        grouped_rows: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
        for row in rows_per_site:
            grouped_rows[
                (
                    str(row["site_element"]),
                    str(row["surface_side"]),
                    str(row["motif_key"]),
                )
            ].append(row)

        summary_rows: List[Dict[str, object]] = []
        for (site_element, surface_side, motif_key), group in sorted(grouped_rows.items()):
            frames_present = sorted({int(row["frame"]) for row in group})
            summary: Dict[str, object] = {
                "traj": str(traj_path),
                "temperature_K": temperature,
                "site_element": site_element,
                "surface_side": surface_side,
                "motif_key": motif_key,
                "shell1_key": str(group[0]["shell1_key"]),
                "shell2_key": str(group[0]["shell2_key"]),
                "n_occurrences": len(group),
                "n_frames_present": len(frames_present),
                "n_frames_total": total_frames,
                "frame_presence_frac": (
                    float(len(frames_present) / total_frames) if total_frames > 0 else float("nan")
                ),
                "site_fraction": (
                    float(len(group) / total_sites) if total_sites > 0 else float("nan")
                ),
                "mean_sites_per_frame": (
                    float(len(group) / total_frames) if total_frames > 0 else float("nan")
                ),
                "shell1_n": int(group[0]["shell1_n"]),
                "shell2_n": int(group[0]["shell2_n"]),
            }
            for el in self.alloy_elements:
                summary[f"shell1_count_{el}"] = int(group[0][f"shell1_count_{el}"])
                summary[f"shell2_count_{el}"] = int(group[0][f"shell2_count_{el}"])

            for field in (
                "shell1_mean_dist_A",
                "shell1_mean_abs_dz_A",
                "shell1_mean_inplane_dist_A",
                "shell2_mean_dist_A",
                "shell2_mean_abs_dz_A",
                "shell2_mean_inplane_dist_A",
            ):
                values = np.array([float(row[field]) for row in group], dtype=float)
                finite = values[np.isfinite(values)]
                summary[f"{field}_mean"] = self._safe_mean(finite)
                summary[f"{field}_std"] = self._safe_std(finite)

            summary_rows.append(summary)

        return {
            "traj": str(traj_path),
            "temperature_K": temperature,
            "site_layer_centers_ref_A": site_layer_centers_ref,
            "surface_motifs_per_site": rows_per_site,
            "surface_motifs_per_frame": rows_per_frame,
            "surface_motifs_summary": summary_rows,
        }

    def export_csv(self, result: Dict[str, object], out_prefix: str | Path) -> None:
        out_prefix = Path(out_prefix)

        per_site_rows = result.get("surface_motifs_per_site", [])
        if per_site_rows:
            fieldnames = list(per_site_rows[0].keys())
            _write_csv(Path(f"{out_prefix}_per_site.csv"), per_site_rows, fieldnames)

        per_frame_rows = result.get("surface_motifs_per_frame", [])
        if per_frame_rows:
            fieldnames = list(per_frame_rows[0].keys())
            _write_csv(Path(f"{out_prefix}_per_frame.csv"), per_frame_rows, fieldnames)

        summary_rows = result.get("surface_motifs_summary", [])
        if summary_rows:
            fieldnames = list(summary_rows[0].keys())
            _write_csv(Path(f"{out_prefix}_summary.csv"), summary_rows, fieldnames)
