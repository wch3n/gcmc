"""SRO analysis for relaxed multicomponent MXene trajectories."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.geometry import get_distances
from ase.io import iread
from scipy.optimize import linear_sum_assignment

from ..utils import initialize_alloy_sublattice
from .ordering import AXIS_MAP, _grouped_mean_std, _write_csv


class MXeneSROAnalyzer:
    """WC-SRO analysis for MXene alloy sublattices.

    Modes:
    - ``reference``: map each frame onto a fixed reference lattice.
    - ``adaptive``: avoid site mapping and use instantaneous layers plus
      rank-based neighbor shells derived from the reference coordination.
    """

    def __init__(
        self,
        reference_atoms: Atoms,
        alloy_elements: Iterable[str] = ("Ti", "Zr"),
        reference_mask: np.ndarray | None = None,
        layer_axis: str = "z",
        n_layers: int | None = None,
        layer_gap: float | None = None,
        align_translation: bool = True,
        n_shells: int = 1,
        shell_tol: float = 0.15,
        fft_qmax: int = 0,
        analysis_mode: str = "reference",
        canonicalize_layer_flip: bool = False,
        canonical_species: str | None = None,
    ) -> None:
        if layer_axis not in AXIS_MAP:
            raise ValueError(f"layer_axis must be one of {tuple(AXIS_MAP)}.")
        self.reference_atoms = reference_atoms.copy()
        self.alloy_elements = tuple(dict.fromkeys(alloy_elements))
        if len(self.alloy_elements) < 2:
            raise ValueError("Need at least two alloy elements for SRO analysis.")
        self.reference_mask_input = None if reference_mask is None else np.asarray(reference_mask, dtype=bool)
        self.axis_idx = AXIS_MAP[layer_axis]
        self.layer_axis = layer_axis
        self.n_layers = n_layers
        self.layer_gap = layer_gap
        self.align_translation = bool(align_translation)
        self.n_shells = int(n_shells)
        self.shell_tol = float(shell_tol)
        self.fft_qmax = int(fft_qmax)
        self.analysis_mode = str(analysis_mode).lower()
        self.canonicalize_layer_flip = bool(canonicalize_layer_flip)
        self.canonical_species = canonical_species
        if self.n_shells < 1:
            raise ValueError("n_shells must be >= 1.")
        if self.shell_tol <= 0.0:
            raise ValueError("shell_tol must be > 0.")
        if self.fft_qmax < 0:
            raise ValueError("fft_qmax must be >= 0.")
        if self.analysis_mode not in {"reference", "adaptive"}:
            raise ValueError("analysis_mode must be 'reference' or 'adaptive'.")
        if self.canonical_species is not None and self.canonical_species not in self.alloy_elements:
            raise ValueError("canonical_species must be one of alloy_elements.")
        self._prepare_reference()

    @classmethod
    def from_primitive(
        cls,
        primitive_atoms: Atoms,
        sc_matrix: Sequence[Sequence[int]] | Sequence[int] | np.ndarray,
        alloy_elements: Iterable[str] = ("Ti", "Zr"),
        scale_factor: Sequence[float] | float | None = None,
        site_element: str | None = None,
        composition: Dict[str, float] | None = None,
        seed: int = 67,
        **kwargs,
    ) -> "MXeneSROAnalyzer":
        reference_atoms = primitive_atoms.copy()
        if scale_factor is not None:
            scale = np.asarray(scale_factor, dtype=float).reshape(-1)
            if scale.size == 1:
                new_cell = np.asarray(reference_atoms.cell, dtype=float) * float(scale[0])
            elif scale.size == 3:
                new_cell = np.asarray(reference_atoms.cell, dtype=float).copy()
                for idx, factor in enumerate(scale):
                    new_cell[idx, :] *= float(factor)
            else:
                raise ValueError("scale_factor must contain either 1 or 3 values.")
            reference_atoms.set_cell(new_cell, scale_atoms=True)

        sc_matrix_arr = cls._as_supercell_matrix(sc_matrix)
        reference_atoms = make_supercell(reference_atoms, sc_matrix_arr)

        reference_mask = None
        if site_element is not None:
            site_symbols = np.array(reference_atoms.get_chemical_symbols(), dtype=object)
            reference_mask = site_symbols == site_element
            if not np.any(reference_mask):
                raise ValueError(
                    f"No atoms of type {site_element!r} found in the reconstructed supercell."
                )
            if composition is not None:
                reference_atoms = initialize_alloy_sublattice(
                    atoms=reference_atoms,
                    site_element=site_element,
                    composition=dict(composition),
                    seed=seed,
                )
        elif composition is not None:
            raise ValueError("composition requires site_element when reconstructing the reference.")

        return cls(
            reference_atoms=reference_atoms,
            alloy_elements=alloy_elements,
            reference_mask=reference_mask,
            **kwargs,
        )

    @staticmethod
    def _as_supercell_matrix(
        sc_matrix: Sequence[Sequence[int]] | Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        arr = np.asarray(sc_matrix, dtype=int)
        if arr.shape == (3, 3):
            return arr
        if arr.ndim == 1 and arr.size == 3:
            return np.diag(arr)
        if arr.ndim == 1 and arr.size == 9:
            return arr.reshape(3, 3)
        raise ValueError("sc_matrix must be a 3x3 matrix or 3/9 integers in row-major order.")

    @staticmethod
    def _parse_temperature_from_traj_name(traj: str) -> float:
        m = re.search(r"replica_([0-9]+(?:\.[0-9]+)?)K", Path(traj).name)
        if m is None:
            return float("nan")
        return float(m.group(1))

    @staticmethod
    def _iter_frames(traj_path: Path, start: int, stop: int | None, step: int):
        index = f"{start}:{'' if stop is None else stop}:{step}"
        for local_idx, atoms in enumerate(iread(str(traj_path), index=index)):
            yield start + local_idx * step, atoms

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
                split_idx = np.sort(np.argsort(diffs)[-(self.n_layers - 1) :])
        else:
            auto_gap = (
                float(self.layer_gap)
                if self.layer_gap is not None
                else (max(0.15, 3.0 * float(np.median(diffs))) if diffs.size else 0.15)
            )
            split_idx = np.where(diffs > auto_gap)[0]

        groups: List[np.ndarray] = []
        start = 0
        for idx in split_idx:
            groups.append(coord_sorted[start : idx + 1])
            start = idx + 1
        groups.append(coord_sorted[start:])
        return np.sort(np.array([float(np.mean(g)) for g in groups], dtype=float))

    @staticmethod
    def _assign_layers(coord_1d: np.ndarray, centers: np.ndarray) -> np.ndarray:
        return np.argmin(np.abs(coord_1d[:, None] - centers[None, :]), axis=1)

    @staticmethod
    def _cluster_shell_distances(distances: np.ndarray, shell_tol: float) -> np.ndarray:
        positive = np.sort(distances[distances > 1e-8])
        if positive.size == 0:
            return np.array([], dtype=float)
        clusters: List[List[float]] = [[float(positive[0])]]
        for value in positive[1:]:
            if abs(float(value) - float(np.mean(clusters[-1]))) <= shell_tol:
                clusters[-1].append(float(value))
            else:
                clusters.append([float(value)])
        return np.array([float(np.mean(c)) for c in clusters], dtype=float)

    def _prepare_reference(self) -> None:
        ref_symbols = np.array(self.reference_atoms.get_chemical_symbols(), dtype=object)
        if self.reference_mask_input is not None:
            if len(self.reference_mask_input) != len(ref_symbols):
                raise ValueError("reference_mask length must match reference_atoms.")
            ref_mask = self.reference_mask_input.copy()
        else:
            ref_mask = np.isin(ref_symbols, self.alloy_elements)
        if not np.any(ref_mask):
            raise ValueError(
                f"No atoms matched alloy_elements={self.alloy_elements} in the reference structure."
            )

        self.reference_mask = ref_mask
        self.reference_site_indices = np.where(ref_mask)[0]
        self.reference_positions = self.reference_atoms.positions[ref_mask].copy()
        self.reference_layer_centers = self._infer_layer_centers(
            self.reference_positions[:, self.axis_idx]
        )
        self.reference_layers = self._assign_layers(
            self.reference_positions[:, self.axis_idx],
            self.reference_layer_centers,
        )

        ref_alloy_atoms = self.reference_atoms[ref_mask]
        dist_matrix = np.asarray(ref_alloy_atoms.get_all_distances(mic=True), dtype=float)
        upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        shell_centers = self._cluster_shell_distances(upper, self.shell_tol)[: self.n_shells]
        if shell_centers.size == 0:
            raise ValueError("No non-zero shell distances found in the reference structure.")

        self.reference_shells: List[Dict[str, object]] = []
        for shell_id, distance in enumerate(shell_centers, start=1):
            mask = (np.abs(dist_matrix - distance) <= self.shell_tol) & (
                ~np.eye(len(dist_matrix), dtype=bool)
            )
            central_idx, neighbor_idx = np.where(mask)
            coordination = int(round(float(np.mean(np.sum(mask, axis=1)))))
            self.reference_shells.append(
                {
                    "shell_id": shell_id,
                    "shell_distance_A": float(distance),
                    "coordination_number": coordination,
                    "central_idx": central_idx.astype(int),
                    "neighbor_idx": neighbor_idx.astype(int),
                }
            )

        self._prepare_reciprocal_grid()

    def _prepare_reciprocal_grid(self) -> None:
        self.inplane_axes = tuple(idx for idx in range(3) if idx != self.axis_idx)
        self.reference_q_grid: List[Dict[str, object]] = []
        self.reference_q_vectors = np.zeros((0, 3), dtype=float)
        self.reference_phase_matrix = np.zeros((len(self.reference_positions), 0), dtype=complex)
        if self.fft_qmax == 0:
            return

        cell = np.asarray(self.reference_atoms.cell, dtype=float)
        reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
        b1 = reciprocal[self.inplane_axes[0]]
        b2 = reciprocal[self.inplane_axes[1]]

        q_rows: List[Dict[str, object]] = []
        q_vectors: List[np.ndarray] = []
        for h in range(-self.fft_qmax, self.fft_qmax + 1):
            for k in range(-self.fft_qmax, self.fft_qmax + 1):
                if h == 0 and k == 0:
                    continue
                q_vec = h * b1 + k * b2
                q_rows.append(
                    {
                        "h": int(h),
                        "k": int(k),
                        "qx_invA": float(q_vec[0]),
                        "qy_invA": float(q_vec[1]),
                        "qz_invA": float(q_vec[2]),
                        "q_abs_invA": float(np.linalg.norm(q_vec)),
                    }
                )
                q_vectors.append(np.asarray(q_vec, dtype=float))

        order = sorted(
            range(len(q_rows)),
            key=lambda idx: (
                q_rows[idx]["q_abs_invA"],
                abs(q_rows[idx]["h"]),
                abs(q_rows[idx]["k"]),
                q_rows[idx]["h"],
                q_rows[idx]["k"],
            ),
        )
        self.reference_q_grid = [q_rows[idx] for idx in order]
        self.reference_q_vectors = np.array([q_vectors[idx] for idx in order], dtype=float)
        self.reference_phase_matrix = np.exp(
            -1j * (self.reference_positions @ self.reference_q_vectors.T)
        )

    @staticmethod
    def _group_rows_by_frame(
        rows: List[Dict[str, object]],
    ) -> Dict[int, List[Dict[str, object]]]:
        frame_map: Dict[int, List[Dict[str, object]]] = {}
        for row in rows:
            frame_map.setdefault(int(row["frame"]), []).append(row)
        return frame_map

    def _determine_layer_flip_map(
        self,
        layer_rows: List[Dict[str, object]],
    ) -> Tuple[Dict[int, bool], str | None]:
        if not self.canonicalize_layer_flip or not layer_rows:
            return {}, None
        n_layers = len({int(row["layer_id"]) for row in layer_rows})
        if n_layers != 2:
            raise ValueError(
                "canonicalize_layer_flip currently requires exactly 2 alloy layers."
            )

        frame_map = self._group_rows_by_frame(layer_rows)
        if self.canonical_species is not None:
            anchor = self.canonical_species
        else:
            scores: Dict[str, float] = {}
            for el in self.alloy_elements:
                deltas = []
                for rows_f in frame_map.values():
                    layer_fracs = {
                        int(x["layer_id"]): float(x.get(f"frac_{el}", np.nan))
                        for x in rows_f
                    }
                    if len(layer_fracs) >= 2:
                        lids = sorted(layer_fracs)
                        deltas.append(abs(layer_fracs[lids[-1]] - layer_fracs[lids[0]]))
                scores[el] = float(np.nanmean(deltas)) if deltas else float("-inf")
            anchor = max(scores, key=scores.get)

        flip_by_frame: Dict[int, bool] = {}
        for frame_id, rows_f in frame_map.items():
            layer_fracs = {
                int(x["layer_id"]): float(x.get(f"frac_{anchor}", np.nan))
                for x in rows_f
            }
            lids = sorted(layer_fracs)
            delta = layer_fracs[lids[-1]] - layer_fracs[lids[0]]
            flip_by_frame[int(frame_id)] = bool(np.isfinite(delta) and delta < 0.0)
        return flip_by_frame, anchor

    @staticmethod
    def _canonicalize_layer_rows(
        rows: List[Dict[str, object]],
        flip_by_frame: Dict[int, bool],
    ) -> List[Dict[str, object]]:
        if not rows or not flip_by_frame:
            return rows

        out_rows: List[Dict[str, object]] = []
        for row in rows:
            out_row = dict(row)
            frame_id = int(out_row["frame"])
            if flip_by_frame.get(frame_id, False):
                out_row["layer_id"] = 1 - int(out_row["layer_id"])
            out_rows.append(out_row)
        return out_rows

    def _select_frame_alloy_atoms(
        self,
        atoms: Atoms,
    ) -> Tuple[np.ndarray, np.ndarray]:
        symbols = np.array(atoms.get_chemical_symbols(), dtype=object)
        alloy_mask = np.isin(symbols, self.alloy_elements)
        if not np.any(alloy_mask):
            raise ValueError(
                f"No atoms matched alloy_elements={self.alloy_elements} in frame."
            )
        if int(np.sum(alloy_mask)) != len(self.reference_positions):
            raise ValueError(
                "Frame alloy-site count does not match the reference alloy-site count. "
                "Adaptive SRO analysis still assumes a fixed substitutional alloy sublattice."
            )
        return symbols[alloy_mask], atoms.positions[alloy_mask].copy()

    def _build_adaptive_shells(
        self, atoms: Atoms
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]]]:
        frame_symbols, frame_positions = self._select_frame_alloy_atoms(atoms)
        layer_centers = self._infer_layer_centers(frame_positions[:, self.axis_idx])
        frame_layers = self._assign_layers(frame_positions[:, self.axis_idx], layer_centers)

        frame_alloy = atoms[np.isin(atoms.get_chemical_symbols(), self.alloy_elements)]
        dist_matrix = np.asarray(frame_alloy.get_all_distances(mic=True), dtype=float)
        order = np.argsort(dist_matrix, axis=1)
        sorted_dist = np.take_along_axis(dist_matrix, order, axis=1)

        shells: List[Dict[str, object]] = []
        rank_start = 1
        max_neighbors = len(frame_positions) - 1
        for shell in self.reference_shells:
            coord = int(shell.get("coordination_number", 0))
            if coord <= 0:
                continue
            rank_stop = rank_start + coord
            if rank_stop > max_neighbors + 1:
                raise ValueError(
                    "Not enough alloy atoms in frame to construct adaptive neighbor shells."
                )
            central_idx = np.repeat(np.arange(len(frame_positions), dtype=int), coord)
            neighbor_idx = order[:, rank_start:rank_stop].reshape(-1).astype(int)
            shell_dist = sorted_dist[:, rank_start:rank_stop]
            shells.append(
                {
                    "shell_id": int(shell["shell_id"]),
                    "shell_distance_A": float(np.mean(shell_dist)),
                    "coordination_number": coord,
                    "central_idx": central_idx,
                    "neighbor_idx": neighbor_idx,
                }
            )
            rank_start = rank_stop
        return frame_symbols, frame_positions, frame_layers, layer_centers, shells

    def _compute_structure_factor_rows_positions(
        self,
        traj_path: Path,
        frame_idx: int,
        positions: np.ndarray,
        species: np.ndarray,
        layers: np.ndarray,
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        if not self.reference_q_grid:
            return [], []

        phase_matrix = np.exp(-1j * (positions @ self.reference_q_vectors.T))
        occ = np.vstack([(species == el).astype(float) for el in self.alloy_elements])
        conc = np.mean(occ, axis=1)
        amp = (occ - conc[:, None]) @ phase_matrix
        n_sites = int(len(species))

        rows_global: List[Dict[str, object]] = []
        rows_layer: List[Dict[str, object]] = []
        for q_idx, q_meta in enumerate(self.reference_q_grid):
            for i_idx, el_i in enumerate(self.alloy_elements):
                for j_idx, el_j in enumerate(self.alloy_elements):
                    sq_val = amp[i_idx, q_idx] * np.conjugate(amp[j_idx, q_idx]) / n_sites
                    rows_global.append(
                        {
                            "traj": str(traj_path),
                            "frame": frame_idx,
                            **q_meta,
                            "species_i": el_i,
                            "species_j": el_j,
                            "sq_real": float(np.real(sq_val)),
                            "sq_imag": float(np.imag(sq_val)),
                            "sq_abs": float(np.abs(sq_val)),
                        }
                    )

        unique_layers = sorted(int(x) for x in np.unique(layers))
        for lid in unique_layers:
            layer_mask = layers == lid
            n_layer = int(np.sum(layer_mask))
            if n_layer == 0:
                continue
            layer_phase = phase_matrix[layer_mask]
            layer_species = species[layer_mask]
            layer_occ = np.vstack([(layer_species == el).astype(float) for el in self.alloy_elements])
            layer_conc = np.mean(layer_occ, axis=1)
            layer_amp = (layer_occ - layer_conc[:, None]) @ layer_phase
            for q_idx, q_meta in enumerate(self.reference_q_grid):
                for i_idx, el_i in enumerate(self.alloy_elements):
                    for j_idx, el_j in enumerate(self.alloy_elements):
                        sq_val = layer_amp[i_idx, q_idx] * np.conjugate(layer_amp[j_idx, q_idx]) / n_layer
                        rows_layer.append(
                            {
                                "traj": str(traj_path),
                                "frame": frame_idx,
                                "layer_id": lid,
                                **q_meta,
                                "species_i": el_i,
                                "species_j": el_j,
                                "sq_real": float(np.real(sq_val)),
                                "sq_imag": float(np.imag(sq_val)),
                                "sq_abs": float(np.abs(sq_val)),
                            }
                        )
        return rows_global, rows_layer

    def _map_frame_to_reference(
        self, atoms: Atoms
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        if len(atoms) != len(self.reference_atoms):
            raise ValueError(
                "Frame atom count does not match the reference atom count. "
                "Reference-lattice SRO analysis assumes consistent atom ordering."
            )

        symbols = np.array(atoms.get_chemical_symbols(), dtype=object)
        frame_symbols = symbols[self.reference_site_indices]
        frame_positions = atoms.positions[self.reference_site_indices].copy()

        shift_axis = 0.0
        if self.align_translation:
            shift_axis = float(
                np.mean(frame_positions[:, self.axis_idx])
                - np.mean(self.reference_positions[:, self.axis_idx])
            )
            frame_positions[:, self.axis_idx] -= shift_axis

        distance_matrix = get_distances(
            frame_positions,
            self.reference_positions,
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc(),
        )[1]
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        mapped_symbols = np.empty(len(self.reference_positions), dtype=object)
        mapped_symbols[col_ind] = frame_symbols[row_ind]
        mapped_distances = np.asarray(distance_matrix[row_ind, col_ind], dtype=float)
        return mapped_symbols, mapped_distances, shift_axis

    def analyze_trajectory(
        self,
        traj_path: str | Path,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> Dict[str, object]:
        traj_path = Path(traj_path)
        rows_layer_comp: List[Dict[str, object]] = []
        rows_wc_global: List[Dict[str, object]] = []
        rows_wc_layer: List[Dict[str, object]] = []
        rows_coord_global: List[Dict[str, object]] = []
        rows_coord_layer: List[Dict[str, object]] = []
        rows_mapping: List[Dict[str, object]] = []
        rows_sq_global: List[Dict[str, object]] = []
        rows_sq_layer: List[Dict[str, object]] = []

        for frame_idx, atoms in self._iter_frames(traj_path, start=start, stop=stop, step=step):
            if self.analysis_mode == "reference":
                frame_symbols, mapped_distances, shift_axis = self._map_frame_to_reference(atoms)
                frame_positions = self.reference_positions
                frame_layers = self.reference_layers
                frame_layer_centers = self.reference_layer_centers
                shells = self.reference_shells
                rows_mapping.append(
                    {
                        "traj": str(traj_path),
                        "frame": frame_idx,
                        "assignment_rms_A": float(np.sqrt(np.mean(mapped_distances**2))),
                        "assignment_max_A": float(np.max(mapped_distances)),
                        "shift_axis_A": shift_axis,
                    }
                )
            else:
                (
                    frame_symbols,
                    frame_positions,
                    frame_layers,
                    frame_layer_centers,
                    shells,
                ) = self._build_adaptive_shells(atoms)

            conc = {el: float(np.mean(frame_symbols == el)) for el in self.alloy_elements}

            for lid in sorted(int(x) for x in np.unique(frame_layers)):
                in_layer = frame_layers == lid
                n_layer = int(np.sum(in_layer))
                row = {
                    "traj": str(traj_path),
                    "frame": frame_idx,
                    "layer_id": lid,
                    "layer_pos_ref_A": float(frame_layer_centers[lid]),
                    "n_alloy_layer": n_layer,
                }
                for el in self.alloy_elements:
                    n_el = int(np.sum(frame_symbols[in_layer] == el))
                    row[f"count_{el}"] = n_el
                    row[f"frac_{el}"] = float(n_el / n_layer) if n_layer > 0 else float("nan")
                rows_layer_comp.append(row)

            for shell in shells:
                shell_id = int(shell["shell_id"])
                shell_distance = float(shell["shell_distance_A"])
                central_idx = np.asarray(shell["central_idx"], dtype=int)
                neighbor_idx = np.asarray(shell["neighbor_idx"], dtype=int)
                central_symbols = frame_symbols[central_idx]
                neighbor_symbols = frame_symbols[neighbor_idx]
                central_layers = frame_layers[central_idx]

                for el_i in self.alloy_elements:
                    n_atoms_i = int(np.sum(frame_symbols == el_i))
                    mask_i = central_symbols == el_i
                    total_neighbors_i = int(np.sum(mask_i))
                    rows_coord_global.append(
                        {
                            "traj": str(traj_path),
                            "frame": frame_idx,
                            "shell_id": shell_id,
                            "shell_distance_A": shell_distance,
                            "central_species": el_i,
                            "n_atoms_central": n_atoms_i,
                            "mean_coordination": (
                                float(total_neighbors_i / n_atoms_i)
                                if n_atoms_i > 0
                                else float("nan")
                            ),
                        }
                    )
                    for el_j in self.alloy_elements:
                        if total_neighbors_i > 0 and conc[el_j] > 0.0:
                            p_ij = float(
                                np.sum(mask_i & (neighbor_symbols == el_j)) / total_neighbors_i
                            )
                            alpha_ij = float(1.0 - p_ij / conc[el_j])
                        else:
                            alpha_ij = float("nan")
                        rows_wc_global.append(
                            {
                                "traj": str(traj_path),
                                "frame": frame_idx,
                                "shell_id": shell_id,
                                "shell_distance_A": shell_distance,
                                "central_species": el_i,
                                "neighbor_species": el_j,
                                "alpha_wc": alpha_ij,
                            }
                        )

                for lid in range(len(frame_layer_centers)):
                    layer_mask = central_layers == lid
                    for el_i in self.alloy_elements:
                        n_atoms_i_layer = int(
                            np.sum((frame_layers == lid) & (frame_symbols == el_i))
                        )
                        mask_i_layer = layer_mask & (central_symbols == el_i)
                        total_neighbors_i_layer = int(np.sum(mask_i_layer))
                        rows_coord_layer.append(
                            {
                                "traj": str(traj_path),
                                "frame": frame_idx,
                                "shell_id": shell_id,
                                "shell_distance_A": shell_distance,
                                "layer_id": lid,
                                "central_species": el_i,
                                "n_atoms_central_layer": n_atoms_i_layer,
                                "mean_coordination": (
                                    float(total_neighbors_i_layer / n_atoms_i_layer)
                                    if n_atoms_i_layer > 0
                                    else float("nan")
                                ),
                            }
                        )
                        for el_j in self.alloy_elements:
                            if total_neighbors_i_layer > 0 and conc[el_j] > 0.0:
                                p_ij_l = float(
                                    np.sum(mask_i_layer & (neighbor_symbols == el_j))
                                    / total_neighbors_i_layer
                                )
                                alpha_ij_l = float(1.0 - p_ij_l / conc[el_j])
                            else:
                                alpha_ij_l = float("nan")
                            rows_wc_layer.append(
                                {
                                    "traj": str(traj_path),
                                    "frame": frame_idx,
                                    "shell_id": shell_id,
                                    "shell_distance_A": shell_distance,
                                    "layer_id": lid,
                                    "central_species": el_i,
                                    "neighbor_species": el_j,
                                    "alpha_wc": alpha_ij_l,
                                }
                            )

            sq_global, sq_layer = self._compute_structure_factor_rows_positions(
                traj_path=traj_path,
                frame_idx=frame_idx,
                positions=frame_positions,
                species=frame_symbols,
                layers=frame_layers,
            )
            rows_sq_global.extend(sq_global)
            rows_sq_layer.extend(sq_layer)

        result = {
            "traj": str(traj_path),
            "analysis_mode": self.analysis_mode,
            "reference_layer_centers_A": self.reference_layer_centers,
            "reference_site_indices": self.reference_site_indices.astype(int).tolist(),
            "reference_shells": [
                {
                    "shell_id": int(shell["shell_id"]),
                    "shell_distance_A": float(shell["shell_distance_A"]),
                    "coordination_number": int(shell.get("coordination_number", 0)),
                }
                for shell in self.reference_shells
            ],
            "reference_q_grid": [dict(q) for q in self.reference_q_grid],
            "layer_composition_per_frame": rows_layer_comp,
            "wc_global_per_frame": rows_wc_global,
            "wc_layer_per_frame": rows_wc_layer,
            "coord_global_per_frame": rows_coord_global,
            "coord_layer_per_frame": rows_coord_layer,
            "mapping_per_frame": rows_mapping,
            "sq_global_per_frame": rows_sq_global,
            "sq_layer_per_frame": rows_sq_layer,
            "layer_composition_summary": _grouped_mean_std(
                rows_layer_comp,
                group_keys=["traj", "layer_id"],
                value_keys=[f"frac_{el}" for el in self.alloy_elements],
            )
            if rows_layer_comp
            else [],
            "wc_global_summary": _grouped_mean_std(
                rows_wc_global,
                group_keys=[
                    "traj",
                    "shell_id",
                    "shell_distance_A",
                    "central_species",
                    "neighbor_species",
                ],
                value_keys=["alpha_wc"],
            )
            if rows_wc_global
            else [],
            "wc_layer_summary": _grouped_mean_std(
                rows_wc_layer,
                group_keys=[
                    "traj",
                    "shell_id",
                    "shell_distance_A",
                    "layer_id",
                    "central_species",
                    "neighbor_species",
                ],
                value_keys=["alpha_wc"],
            )
            if rows_wc_layer
            else [],
            "coord_global_summary": _grouped_mean_std(
                rows_coord_global,
                group_keys=["traj", "shell_id", "shell_distance_A", "central_species"],
                value_keys=["mean_coordination"],
            )
            if rows_coord_global
            else [],
            "coord_layer_summary": _grouped_mean_std(
                rows_coord_layer,
                group_keys=[
                    "traj",
                    "shell_id",
                    "shell_distance_A",
                    "layer_id",
                    "central_species",
                ],
                value_keys=["mean_coordination"],
            )
            if rows_coord_layer
            else [],
            "mapping_summary": _grouped_mean_std(
                rows_mapping,
                group_keys=["traj"],
                value_keys=["assignment_rms_A", "assignment_max_A", "shift_axis_A"],
            )
            if rows_mapping
            else [],
            "sq_global_summary": _grouped_mean_std(
                rows_sq_global,
                group_keys=[
                    "traj",
                    "h",
                    "k",
                    "qx_invA",
                    "qy_invA",
                    "qz_invA",
                    "q_abs_invA",
                    "species_i",
                    "species_j",
                ],
                value_keys=["sq_real", "sq_imag", "sq_abs"],
            )
            if rows_sq_global
            else [],
            "sq_layer_summary": _grouped_mean_std(
                rows_sq_layer,
                group_keys=[
                    "traj",
                    "layer_id",
                    "h",
                    "k",
                    "qx_invA",
                    "qy_invA",
                    "qz_invA",
                    "q_abs_invA",
                    "species_i",
                    "species_j",
                ],
                value_keys=["sq_real", "sq_imag", "sq_abs"],
            )
            if rows_sq_layer
            else [],
        }
        flip_by_frame, flip_anchor_species = self._determine_layer_flip_map(rows_layer_comp)
        canonical_layer_rows: List[Dict[str, object]] = []
        canonical_wc_layer_rows: List[Dict[str, object]] = []
        canonical_coord_layer_rows: List[Dict[str, object]] = []
        canonical_sq_layer_rows: List[Dict[str, object]] = []
        if self.canonicalize_layer_flip:
            canonical_layer_rows = self._canonicalize_layer_rows(rows_layer_comp, flip_by_frame)
            canonical_wc_layer_rows = self._canonicalize_layer_rows(rows_wc_layer, flip_by_frame)
            canonical_coord_layer_rows = self._canonicalize_layer_rows(rows_coord_layer, flip_by_frame)
            canonical_sq_layer_rows = self._canonicalize_layer_rows(rows_sq_layer, flip_by_frame)
        result["canonicalize_layer_flip"] = self.canonicalize_layer_flip
        result["canonical_anchor_species"] = flip_anchor_species
        result["n_flipped_frames"] = int(sum(bool(v) for v in flip_by_frame.values()))
        result["flip_fraction"] = (
            float(np.mean([1.0 if flip else 0.0 for flip in flip_by_frame.values()]))
            if flip_by_frame
            else 0.0
        )
        result["layer_composition_canonical_summary"] = (
            _grouped_mean_std(
                canonical_layer_rows,
                group_keys=["traj", "layer_id"],
                value_keys=[f"frac_{el}" for el in self.alloy_elements],
            )
            if canonical_layer_rows
            else []
        )
        result["wc_layer_canonical_summary"] = (
            _grouped_mean_std(
                canonical_wc_layer_rows,
                group_keys=[
                    "traj",
                    "shell_id",
                    "shell_distance_A",
                    "layer_id",
                    "central_species",
                    "neighbor_species",
                ],
                value_keys=["alpha_wc"],
            )
            if canonical_wc_layer_rows
            else []
        )
        result["coord_layer_canonical_summary"] = (
            _grouped_mean_std(
                canonical_coord_layer_rows,
                group_keys=[
                    "traj",
                    "shell_id",
                    "shell_distance_A",
                    "layer_id",
                    "central_species",
                ],
                value_keys=["mean_coordination"],
            )
            if canonical_coord_layer_rows
            else []
        )
        result["sq_layer_canonical_summary"] = (
            _grouped_mean_std(
                canonical_sq_layer_rows,
                group_keys=[
                    "traj",
                    "layer_id",
                    "h",
                    "k",
                    "qx_invA",
                    "qy_invA",
                    "qz_invA",
                    "q_abs_invA",
                    "species_i",
                    "species_j",
                ],
                value_keys=["sq_real", "sq_imag", "sq_abs"],
            )
            if canonical_sq_layer_rows
            else []
        )
        result["phase_indicators"] = self.compute_phase_indicators(result)
        return result

    def compute_phase_indicators(self, result: Dict[str, object]) -> Dict[str, object]:
        traj = str(result.get("traj", ""))
        row: Dict[str, object] = {
            "traj": traj,
            "temperature_K": self._parse_temperature_from_traj_name(traj),
            "analysis_mode": str(result.get("analysis_mode", self.analysis_mode)),
            "n_layers": int(len(result.get("reference_layer_centers_A", []))),
            "n_shells": int(len(result.get("reference_shells", []))),
            "n_q_points": int(len(result.get("reference_q_grid", []))),
        }

        layer_rows = result.get("layer_composition_per_frame", [])
        frame_map = self._group_rows_by_frame(layer_rows)
        frame_ids = sorted(frame_map)
        row["n_frames"] = len(frame_ids)
        if layer_rows:
            row["n_layers"] = len({int(r["layer_id"]) for r in layer_rows})
        row["canonicalize_layer_flip"] = bool(result.get("canonicalize_layer_flip", False))
        row["n_flipped_frames"] = int(result.get("n_flipped_frames", 0))
        row["flip_fraction"] = float(result.get("flip_fraction", 0.0))
        if result.get("canonical_anchor_species") is not None:
            row["canonical_anchor_species"] = str(result["canonical_anchor_species"])

        def _add_layer_metrics(
            rows_per_frame: List[Dict[str, object]],
            summary_rows: List[Dict[str, object]],
            prefix: str = "",
            include_global: bool = False,
        ) -> None:
            local_frame_map = self._group_rows_by_frame(rows_per_frame)
            local_frame_ids = sorted(local_frame_map)
            for el in self.alloy_elements:
                global_frac = []
                top_minus_bottom = []
                abs_top_minus_bottom = []
                layer_spread = []
                for fid in local_frame_ids:
                    rows_f = local_frame_map[fid]
                    n_tot = float(sum(float(x["n_alloy_layer"]) for x in rows_f))
                    c_tot = float(sum(float(x[f"count_{el}"]) for x in rows_f))
                    global_frac.append(c_tot / n_tot if n_tot > 0 else float("nan"))
                    layer_fracs = {int(x["layer_id"]): float(x[f"frac_{el}"]) for x in rows_f}
                    if layer_fracs:
                        lids = sorted(layer_fracs)
                        delta = layer_fracs[lids[-1]] - layer_fracs[lids[0]]
                        top_minus_bottom.append(delta)
                        abs_top_minus_bottom.append(abs(delta))
                        layer_spread.append(max(layer_fracs.values()) - min(layer_fracs.values()))

                if include_global:
                    row[f"{prefix}global_frac_{el}_mean"] = (
                        float(np.nanmean(global_frac)) if global_frac else float("nan")
                    )
                    row[f"{prefix}global_frac_{el}_std"] = (
                        float(np.nanstd(global_frac)) if global_frac else float("nan")
                    )
                row[f"{prefix}layer_top_minus_bottom_frac_{el}_mean"] = (
                    float(np.nanmean(top_minus_bottom)) if top_minus_bottom else float("nan")
                )
                row[f"{prefix}layer_top_minus_bottom_frac_{el}_std"] = (
                    float(np.nanstd(top_minus_bottom)) if top_minus_bottom else float("nan")
                )
                row[f"{prefix}layer_abs_top_minus_bottom_frac_{el}_mean"] = (
                    float(np.nanmean(abs_top_minus_bottom))
                    if abs_top_minus_bottom
                    else float("nan")
                )
                row[f"{prefix}layer_abs_top_minus_bottom_frac_{el}_std"] = (
                    float(np.nanstd(abs_top_minus_bottom))
                    if abs_top_minus_bottom
                    else float("nan")
                )
                row[f"{prefix}layer_frac_spread_{el}_mean"] = (
                    float(np.nanmean(layer_spread)) if layer_spread else float("nan")
                )
                row[f"{prefix}layer_frac_spread_{el}_std"] = (
                    float(np.nanstd(layer_spread)) if layer_spread else float("nan")
                )

            for r in summary_rows:
                lid = int(r["layer_id"])
                for el in self.alloy_elements:
                    row[f"{prefix}layer{lid}_frac_{el}_mean"] = float(
                        r.get(f"frac_{el}_mean", np.nan)
                    )
                    row[f"{prefix}layer{lid}_frac_{el}_std"] = float(
                        r.get(f"frac_{el}_std", np.nan)
                    )

        _add_layer_metrics(
            rows_per_frame=layer_rows,
            summary_rows=result.get("layer_composition_summary", []),
            prefix="",
            include_global=True,
        )
        if result.get("layer_composition_canonical_summary"):
            flip_by_frame, _ = self._determine_layer_flip_map(layer_rows)
            canonical_layer_rows = self._canonicalize_layer_rows(layer_rows, flip_by_frame)
            _add_layer_metrics(
                rows_per_frame=canonical_layer_rows,
                summary_rows=result.get("layer_composition_canonical_summary", []),
                prefix="canonical_",
                include_global=False,
            )

        for r in result.get("wc_global_summary", []):
            shell_id = int(r["shell_id"])
            i = str(r["central_species"])
            j = str(r["neighbor_species"])
            row[f"wc_shell{shell_id}_global_{i}_{j}_mean"] = float(
                r.get("alpha_wc_mean", np.nan)
            )
            row[f"wc_shell{shell_id}_global_{i}_{j}_std"] = float(
                r.get("alpha_wc_std", np.nan)
            )

        def _add_layer_wc_metrics(
            summary_rows: List[Dict[str, object]],
            prefix: str = "",
        ) -> None:
            by_pair: Dict[Tuple[int, str, str], Dict[int, float]] = {}
            for r in summary_rows:
                key = (int(r["shell_id"]), str(r["central_species"]), str(r["neighbor_species"]))
                by_pair.setdefault(key, {})[int(r["layer_id"])] = float(
                    r.get("alpha_wc_mean", np.nan)
                )
            for (shell_id, i, j), lid_map in by_pair.items():
                vals = [v for _, v in sorted(lid_map.items())]
                if vals:
                    row[f"{prefix}wc_shell{shell_id}_layer_spread_{i}_{j}"] = float(
                        np.nanmax(vals) - np.nanmin(vals)
                    )
                    lids = sorted(lid_map)
                    row[f"{prefix}wc_shell{shell_id}_layer_top_minus_bottom_{i}_{j}"] = float(
                        lid_map[lids[-1]] - lid_map[lids[0]]
                    )
        if result.get("wc_layer_summary"):
            _add_layer_wc_metrics(result["wc_layer_summary"])
        if result.get("wc_layer_canonical_summary"):
            _add_layer_wc_metrics(
                result["wc_layer_canonical_summary"],
                prefix="canonical_",
            )

        for r in result.get("coord_global_summary", []):
            shell_id = int(r["shell_id"])
            i = str(r["central_species"])
            row[f"coord_shell{shell_id}_global_{i}_mean"] = float(
                r.get("mean_coordination_mean", np.nan)
            )
            row[f"coord_shell{shell_id}_global_{i}_std"] = float(
                r.get("mean_coordination_std", np.nan)
            )

        mapping = result.get("mapping_summary", [])
        if mapping:
            m0 = mapping[0]
            row["assignment_rms_A_mean"] = float(m0.get("assignment_rms_A_mean", np.nan))
            row["assignment_rms_A_std"] = float(m0.get("assignment_rms_A_std", np.nan))
            row["assignment_max_A_mean"] = float(m0.get("assignment_max_A_mean", np.nan))
            row["assignment_max_A_std"] = float(m0.get("assignment_max_A_std", np.nan))

        peak_rows: Dict[str, Dict[str, object]] = {}
        for r in result.get("sq_global_summary", []):
            if str(r["species_i"]) != str(r["species_j"]):
                continue
            species = str(r["species_i"])
            value = float(r.get("sq_abs_mean", np.nan))
            if np.isnan(value):
                continue
            if species not in peak_rows or value > float(peak_rows[species]["sq_abs_mean"]):
                peak_rows[species] = r
        for species, peak in peak_rows.items():
            row[f"sq_peak_global_{species}_h"] = int(peak["h"])
            row[f"sq_peak_global_{species}_k"] = int(peak["k"])
            row[f"sq_peak_global_{species}_q_abs_invA"] = float(peak["q_abs_invA"])
            row[f"sq_peak_global_{species}_sq_abs_mean"] = float(peak["sq_abs_mean"])
            row[f"sq_peak_global_{species}_sq_abs_std"] = float(peak["sq_abs_std"])
        return row

    def export_csv(self, result: Dict[str, object], out_prefix: str | Path) -> None:
        out_prefix = Path(out_prefix)
        per_frame_specs = [
            (
                "layer_composition_per_frame",
                ["traj", "frame", "layer_id", "layer_pos_ref_A", "n_alloy_layer"]
                + [f"count_{el}" for el in self.alloy_elements]
                + [f"frac_{el}" for el in self.alloy_elements],
            ),
            (
                "wc_global_per_frame",
                [
                    "traj",
                    "frame",
                    "shell_id",
                    "shell_distance_A",
                    "central_species",
                    "neighbor_species",
                    "alpha_wc",
                ],
            ),
            (
                "wc_layer_per_frame",
                [
                    "traj",
                    "frame",
                    "shell_id",
                    "shell_distance_A",
                    "layer_id",
                    "central_species",
                    "neighbor_species",
                    "alpha_wc",
                ],
            ),
            (
                "coord_global_per_frame",
                [
                    "traj",
                    "frame",
                    "shell_id",
                    "shell_distance_A",
                    "central_species",
                    "n_atoms_central",
                    "mean_coordination",
                ],
            ),
            (
                "coord_layer_per_frame",
                [
                    "traj",
                    "frame",
                    "shell_id",
                    "shell_distance_A",
                    "layer_id",
                    "central_species",
                    "n_atoms_central_layer",
                    "mean_coordination",
                ],
            ),
            (
                "mapping_per_frame",
                ["traj", "frame", "assignment_rms_A", "assignment_max_A", "shift_axis_A"],
            ),
            (
                "sq_global_per_frame",
                [
                    "traj",
                    "frame",
                    "h",
                    "k",
                    "qx_invA",
                    "qy_invA",
                    "qz_invA",
                    "q_abs_invA",
                    "species_i",
                    "species_j",
                    "sq_real",
                    "sq_imag",
                    "sq_abs",
                ],
            ),
            (
                "sq_layer_per_frame",
                [
                    "traj",
                    "frame",
                    "layer_id",
                    "h",
                    "k",
                    "qx_invA",
                    "qy_invA",
                    "qz_invA",
                    "q_abs_invA",
                    "species_i",
                    "species_j",
                    "sq_real",
                    "sq_imag",
                    "sq_abs",
                ],
            ),
        ]
        summary_keys = [
            "layer_composition_summary",
            "layer_composition_canonical_summary",
            "wc_global_summary",
            "wc_layer_summary",
            "wc_layer_canonical_summary",
            "coord_global_summary",
            "coord_layer_summary",
            "coord_layer_canonical_summary",
            "mapping_summary",
            "sq_global_summary",
            "sq_layer_summary",
            "sq_layer_canonical_summary",
        ]

        for key, fields in per_frame_specs:
            rows = result.get(key, [])
            if rows:
                _write_csv(Path(f"{out_prefix}_{key}.csv"), rows, fields)
        for key in summary_keys:
            rows = result.get(key, [])
            if rows:
                _write_csv(Path(f"{out_prefix}_{key}.csv"), rows, list(rows[0].keys()))
        if result.get("phase_indicators"):
            _write_csv(
                Path(f"{out_prefix}_phase_summary.csv"),
                [result["phase_indicators"]],
                list(result["phase_indicators"].keys()),
            )
