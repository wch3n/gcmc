"""Local adsorption-motif analysis for tagged adsorbate trajectories."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import iread, read, write

from gcmc.constants import ADSORBATE_TAG_OFFSET
from gcmc.utils import build_surface_site_registry

from .ordering import _write_csv


class LocalAdsorptionMotifAnalyzer:
    """Assign tagged adsorbates to trial sites and summarize local motifs.

    The motif fingerprint is built from the reference slab site registry plus the
    local support-centered metal environment around each registry site. Per-frame
    rows then attach the tagged adsorbate anchor to the nearest eligible site.
    """

    def __init__(
        self,
        *,
        site_elements: Iterable[str] = ("Ti", "Zr"),
        substrate_elements: Iterable[str] = ("Ti", "Zr", "C"),
        functional_elements: Iterable[str] = ("O",),
        site_types: Iterable[str] = ("atop", "bridge", "fcc", "hcp"),
        surface_side: str = "top",
        layer_tol: float = 0.5,
        xy_tol: float = 0.6,
        support_xy_tol: float = 1.2,
        termination_site_xy_tol: float | None = None,
        vertical_offset: float = 1.5,
        min_termination_dist: float = 0.8,
        anchor_element: str | None = "O",
        shell1_size: int = 6,
        shell2_size: int = 6,
        functional_cutoff: float = 3.0,
        max_site_distance_A: float | None = None,
        max_anchor_support_distance_A: float | None = None,
        include_blocked_sites: bool = False,
    ) -> None:
        self.site_elements = tuple(site_elements)
        self.substrate_elements = tuple(substrate_elements)
        self.functional_elements = tuple(functional_elements)
        self.site_types = tuple(site_types)
        self.surface_side = str(surface_side)
        self.layer_tol = float(layer_tol)
        self.xy_tol = float(xy_tol)
        self.support_xy_tol = float(support_xy_tol)
        self.termination_site_xy_tol = (
            None if termination_site_xy_tol is None else float(termination_site_xy_tol)
        )
        self.vertical_offset = float(vertical_offset)
        self.min_termination_dist = float(min_termination_dist)
        self.anchor_element = None if anchor_element is None else str(anchor_element)
        self.shell1_size = int(shell1_size)
        self.shell2_size = int(shell2_size)
        self.functional_cutoff = float(functional_cutoff)
        self.max_site_distance_A = (
            None if max_site_distance_A is None else float(max_site_distance_A)
        )
        self.max_anchor_support_distance_A = (
            None
            if max_anchor_support_distance_A is None
            else float(max_anchor_support_distance_A)
        )
        self.include_blocked_sites = bool(include_blocked_sites)

        if self.surface_side not in {"top", "bottom"}:
            raise ValueError("surface_side must be 'top' or 'bottom'.")
        if not self.site_elements:
            raise ValueError("Need at least one site element.")
        if self.shell1_size < 0 or self.shell2_size < 0:
            raise ValueError("shell sizes must be >= 0.")
        if self.functional_cutoff < 0.0:
            raise ValueError("functional_cutoff must be >= 0.")
        if self.max_site_distance_A is not None and self.max_site_distance_A <= 0.0:
            raise ValueError("max_site_distance_A must be > 0 when provided.")
        if (
            self.max_anchor_support_distance_A is not None
            and self.max_anchor_support_distance_A <= 0.0
        ):
            raise ValueError(
                "max_anchor_support_distance_A must be > 0 when provided."
            )

    @staticmethod
    def _parse_temperature_from_traj_name(path: str | Path) -> float:
        match = re.search(r"replica_([0-9]+(?:\.[0-9]+)?)K", Path(path).name)
        if match is None:
            return float("nan")
        return float(match.group(1))

    @staticmethod
    def _iter_frames(traj_path: Path, start: int, stop: int | None, step: int):
        index = f"{start}:{'' if stop is None else stop}:{step}"
        for local_idx, atoms in enumerate(iread(str(traj_path), index=index)):
            frame_idx = start + local_idx * step
            yield frame_idx, atoms

    @staticmethod
    def _strip_tagged_adsorbates(atoms: Atoms) -> Atoms:
        tags = np.asarray(atoms.get_tags(), dtype=int)
        if tags.size == 0 or not np.any(tags >= ADSORBATE_TAG_OFFSET):
            return atoms.copy()
        return atoms[tags < ADSORBATE_TAG_OFFSET].copy()

    @staticmethod
    def _safe_mean(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return float("nan")
        return float(np.mean(arr))

    @staticmethod
    def _safe_std(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return float("nan")
        return float(np.std(arr))

    @staticmethod
    def _format_counts(symbols: np.ndarray, allowed: Tuple[str, ...]) -> Tuple[str, Dict[str, int]]:
        counts = {el: int(np.sum(symbols == el)) for el in allowed}
        key = "+".join(f"{el}{counts[el]}" for el in allowed if counts[el] > 0)
        return (key if key else "none"), counts

    @staticmethod
    def _xy_distances(
        xy: np.ndarray,
        points_xy: np.ndarray,
        cell: np.ndarray,
        pbc: np.ndarray,
    ) -> np.ndarray:
        if points_xy.size == 0:
            return np.array([], dtype=float)
        disp = np.zeros((len(points_xy), 3), dtype=float)
        disp[:, :2] = points_xy - xy[None, :]
        disp_mic, _ = find_mic(disp, cell, pbc)
        return np.linalg.norm(disp_mic[:, :2], axis=1)

    @staticmethod
    def _point_distances(
        point: np.ndarray,
        positions: np.ndarray,
        cell: np.ndarray,
        pbc: np.ndarray,
    ) -> np.ndarray:
        if positions.size == 0:
            return np.array([], dtype=float)
        disp = np.asarray(positions, dtype=float) - np.asarray(point, dtype=float)[None, :]
        disp_mic, _ = find_mic(disp, cell, pbc)
        return np.linalg.norm(disp_mic, axis=1)

    def _pick_anchor_index(self, atoms: Atoms, group_indices: np.ndarray) -> int:
        symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
        candidate_indices = np.asarray(group_indices, dtype=int)
        if self.anchor_element is not None:
            matches = candidate_indices[symbols[candidate_indices] == self.anchor_element]
            if matches.size > 0:
                candidate_indices = matches
        z = atoms.positions[candidate_indices, 2]
        if self.surface_side == "top":
            return int(candidate_indices[np.argmin(z)])
        return int(candidate_indices[np.argmax(z)])

    @staticmethod
    def _coordination_family(site_type: str, support_size: int) -> tuple[str, str]:
        """Return a coarse motif and an explicit coordination label."""
        site_type = str(site_type).lower()
        support_size = int(support_size)
        if support_size == 1 or site_type == "atop":
            return "atop", "atop_1fold"
        if support_size == 2 or site_type == "bridge":
            return "bridge", "bridge_2fold"
        if support_size == 3 or site_type in {"fcc", "hcp", "hollow"}:
            return "hollow", "hollow_3fold"
        return "multifold", f"multifold_{support_size}fold"

    def _build_reference_site_descriptors(self, reference_atoms: Atoms) -> list[dict[str, object]]:
        symbols = np.asarray(reference_atoms.get_chemical_symbols(), dtype=object)
        metal_indices = np.where(np.isin(symbols, self.site_elements))[0]
        functional_indices = np.where(np.isin(symbols, self.functional_elements))[0]
        registry = build_surface_site_registry(
            reference_atoms,
            site_elements=self.site_elements,
            substrate_elements=self.substrate_elements,
            site_types=self.site_types,
            surface_side=self.surface_side,
            layer_tol=self.layer_tol,
            xy_tol=self.xy_tol,
            support_xy_tol=self.support_xy_tol,
            termination_site_xy_tol=self.termination_site_xy_tol,
            vertical_offset=self.vertical_offset,
            termination_elements=(
                self.functional_elements if functional_indices.size > 0 else ()
            ),
            min_termination_dist=self.min_termination_dist,
        )
        descriptors: list[dict[str, object]] = []
        cell = reference_atoms.cell.array
        pbc = reference_atoms.pbc
        for local_index, site in enumerate(registry):
            if (
                bool(site.get("blocked_by_termination", False))
                and not self.include_blocked_sites
            ):
                continue

            support_indices = np.asarray(site.get("support_indices", ()), dtype=int)
            support_symbols = symbols[support_indices]
            support_key, support_counts = self._format_counts(
                support_symbols,
                self.site_elements,
            )

            point_anchor = np.array(
                [
                    float(site["xy"][0]),
                    float(site["xy"][1]),
                    float(site["anchor_z_A"]),
                ],
                dtype=float,
            )
            point_suggested = np.array(
                [
                    float(site["xy"][0]),
                    float(site["xy"][1]),
                    float(site["suggested_z_A"]),
                ],
                dtype=float,
            )

            support_set = {int(idx) for idx in support_indices.tolist()}
            shell_candidates = np.array(
                [idx for idx in metal_indices if int(idx) not in support_set],
                dtype=int,
            )
            shell_dists = self._point_distances(
                point_anchor,
                reference_atoms.positions[shell_candidates],
                cell,
                pbc,
            )
            shell_order = np.argsort(shell_dists)
            shell1_end = min(self.shell1_size, shell_order.size)
            shell2_end = min(shell1_end + self.shell2_size, shell_order.size)
            shell1_indices = shell_candidates[shell_order[:shell1_end]]
            shell2_indices = shell_candidates[shell_order[shell1_end:shell2_end]]
            shell1_symbols = symbols[shell1_indices]
            shell2_symbols = symbols[shell2_indices]
            shell1_key, shell1_counts = self._format_counts(shell1_symbols, self.site_elements)
            shell2_key, shell2_counts = self._format_counts(shell2_symbols, self.site_elements)

            functional_count = 0
            if functional_indices.size > 0 and self.functional_cutoff > 0.0:
                functional_dists = self._point_distances(
                    point_suggested,
                    reference_atoms.positions[functional_indices],
                    cell,
                    pbc,
                )
                functional_count = int(np.sum(functional_dists <= self.functional_cutoff))

            descriptor: dict[str, object] = {
                "site_local_index": int(len(descriptors)),
                "registry_index": int(local_index),
                "site_type": str(site["site_type"]),
                "surface_side": str(site["surface_side"]),
                "support_size": int(len(support_indices)),
                "support_key": support_key,
                "shell1_key": shell1_key,
                "shell2_key": shell2_key,
                "functional_count": functional_count,
                "site_x_A": float(site["xy"][0]),
                "site_y_A": float(site["xy"][1]),
                "anchor_z_A": float(site["anchor_z_A"]),
                "suggested_z_A": float(site["suggested_z_A"]),
                "nearest_termination_dist_A": float(
                    site.get("nearest_termination_dist_A", np.nan)
                ),
                "blocked_by_termination": bool(
                    site.get("blocked_by_termination", False)
                ),
                "motif_key": (
                    f"{site['site_type']}|support={support_key}"
                    f"|shell1={shell1_key}|shell2={shell2_key}|func={functional_count}"
                ),
                "support_indices": " ".join(str(int(i)) for i in support_indices),
                "_site_xy": np.asarray(site["xy"], dtype=float),
                "_support_indices": support_indices,
            }
            for el, count in support_counts.items():
                descriptor[f"support_count_{el}"] = int(count)
            for el, count in shell1_counts.items():
                descriptor[f"shell1_count_{el}"] = int(count)
            for el, count in shell2_counts.items():
                descriptor[f"shell2_count_{el}"] = int(count)
            descriptors.append(descriptor)
        return descriptors

    def analyze_trajectory(
        self,
        traj_path: str | Path,
        *,
        reference: str | Path | Atoms | None = None,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> Dict[str, object]:
        traj_path = Path(traj_path)
        if reference is None:
            reference_atoms = self._strip_tagged_adsorbates(read(str(traj_path), index=start))
        elif isinstance(reference, Atoms):
            reference_atoms = self._strip_tagged_adsorbates(reference)
        else:
            reference_atoms = self._strip_tagged_adsorbates(read(str(reference), index=-1))

        descriptors = self._build_reference_site_descriptors(reference_atoms)
        if not descriptors:
            raise ValueError("No eligible adsorption sites found in the reference slab.")

        site_xy = np.asarray([row["_site_xy"] for row in descriptors], dtype=float)
        surface_indices = np.unique(
            np.concatenate(
                [np.asarray(row["_support_indices"], dtype=int) for row in descriptors]
            )
        )
        frame_rows: list[dict[str, object]] = []
        temperature = self._parse_temperature_from_traj_name(traj_path)

        for frame_idx, atoms in self._iter_frames(traj_path, start=start, stop=stop, step=step):
            tags = np.asarray(atoms.get_tags(), dtype=int)
            group_tags = sorted(int(tag) for tag in np.unique(tags) if tag >= ADSORBATE_TAG_OFFSET)
            if not group_tags:
                continue

            for group_tag in group_tags:
                group_indices = np.where(tags == group_tag)[0]
                if group_indices.size == 0:
                    continue
                anchor_index = self._pick_anchor_index(atoms, group_indices)
                anchor_pos = atoms.positions[anchor_index]
                d_xy = self._xy_distances(anchor_pos[:2], site_xy, atoms.cell.array, atoms.pbc)
                if d_xy.size == 0:
                    continue
                site_idx = int(np.argmin(d_xy))
                descriptor = descriptors[site_idx]
                support_indices = np.asarray(descriptor["_support_indices"], dtype=int)
                support_dists = np.asarray(
                    atoms.get_distances(anchor_index, support_indices, mic=True),
                    dtype=float,
                )
                site_distance = float(d_xy[site_idx])
                support_min_distance = float(np.min(support_dists))
                adsorption_motif, coordination_label = self._coordination_family(
                    str(descriptor["site_type"]),
                    int(descriptor["support_size"]),
                )
                coordination_n = int(descriptor["support_size"])
                assignment_status = "assigned"
                if (
                    self.max_site_distance_A is not None
                    and site_distance > self.max_site_distance_A
                ):
                    adsorption_motif = "off_site"
                    coordination_label = "off_site_0fold"
                    coordination_n = 0
                    assignment_status = "off_site"
                elif (
                    self.max_anchor_support_distance_A is not None
                    and support_min_distance > self.max_anchor_support_distance_A
                ):
                    adsorption_motif = "detached"
                    coordination_label = "detached_0fold"
                    coordination_n = 0
                    assignment_status = "detached"

                surface_shift = float(
                    np.median(
                        atoms.positions[surface_indices, 2]
                        - reference_atoms.positions[surface_indices, 2]
                    )
                )
                support_dz = (
                    atoms.positions[support_indices, 2]
                    - reference_atoms.positions[support_indices, 2]
                    - surface_shift
                )
                outward_sign = 1.0 if self.surface_side == "top" else -1.0
                support_outward = outward_sign * support_dz
                support_z = (
                    float(np.max(atoms.positions[support_indices, 2]))
                    if self.surface_side == "top"
                    else float(np.min(atoms.positions[support_indices, 2]))
                )
                anchor_height = outward_sign * (float(anchor_pos[2]) - support_z)
                row = {
                    key: value
                    for key, value in descriptor.items()
                    if not str(key).startswith("_")
                }
                row.update(
                    {
                        "traj": str(traj_path),
                        "temperature_K": temperature,
                        "frame": int(frame_idx),
                        "group_tag": int(group_tag),
                        "group_size": int(group_indices.size),
                        "anchor_index": int(anchor_index),
                        "anchor_element": str(atoms[anchor_index].symbol),
                        "anchor_x_A": float(anchor_pos[0]),
                        "anchor_y_A": float(anchor_pos[1]),
                        "anchor_z_A_inst": float(anchor_pos[2]),
                        "adsorption_motif": adsorption_motif,
                        "coordination_label": coordination_label,
                        "coordination_n": coordination_n,
                        "assignment_status": assignment_status,
                        "anchor_site_xy_dist_A": site_distance,
                        "anchor_support_min_dist_A": support_min_distance,
                        "anchor_support_mean_dist_A": float(np.mean(support_dists)),
                        "anchor_z_offset_A": float(anchor_pos[2] - float(descriptor["anchor_z_A"])),
                        "anchor_height_above_support_A": anchor_height,
                        "support_outward_displacement_A_mean": float(
                            np.mean(support_outward)
                        ),
                        "support_outward_displacement_A_max": float(
                            np.max(support_outward)
                        ),
                    }
                )
                frame_rows.append(row)

        motif_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        site_groups: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in frame_rows:
            motif_groups[str(row["motif_key"])].append(row)
            site_groups[int(row["site_local_index"])].append(row)

        motif_summary: list[dict[str, object]] = []
        total_samples = len(frame_rows)
        for motif_key, rows in sorted(
            motif_groups.items(), key=lambda item: (-len(item[1]), item[0])
        ):
            first = rows[0]
            summary = {
                "motif_key": motif_key,
                "site_local_index": int(first["site_local_index"]),
                "registry_index": int(first["registry_index"]),
                "site_type": str(first["site_type"]),
                "support_key": str(first["support_key"]),
                "shell1_key": str(first["shell1_key"]),
                "shell2_key": str(first["shell2_key"]),
                "functional_count": int(first["functional_count"]),
                "samples": int(len(rows)),
                "population": float(len(rows) / total_samples) if total_samples else float("nan"),
                "n_frames": int(len({int(row["frame"]) for row in rows})),
                "first_frame": int(min(int(row["frame"]) for row in rows)),
                "anchor_site_xy_dist_A_mean": self._safe_mean(
                    [float(row["anchor_site_xy_dist_A"]) for row in rows]
                ),
                "anchor_site_xy_dist_A_std": self._safe_std(
                    [float(row["anchor_site_xy_dist_A"]) for row in rows]
                ),
                "anchor_support_min_dist_A_mean": self._safe_mean(
                    [float(row["anchor_support_min_dist_A"]) for row in rows]
                ),
                "anchor_z_offset_A_mean": self._safe_mean(
                    [float(row["anchor_z_offset_A"]) for row in rows]
                ),
            }
            motif_summary.append(summary)

        site_summary: list[dict[str, object]] = []
        for site_local_index, rows in sorted(site_groups.items()):
            first = rows[0]
            site_summary.append(
                {
                    "site_local_index": int(site_local_index),
                    "registry_index": int(first["registry_index"]),
                    "site_type": str(first["site_type"]),
                    "support_key": str(first["support_key"]),
                    "samples": int(len(rows)),
                    "population": float(len(rows) / total_samples) if total_samples else float("nan"),
                    "n_motifs": int(len({str(row["motif_key"]) for row in rows})),
                    "anchor_site_xy_dist_A_mean": self._safe_mean(
                        [float(row["anchor_site_xy_dist_A"]) for row in rows]
                    ),
                }
            )

        pattern_groups: dict[
            tuple[str, str, str], list[dict[str, object]]
        ] = defaultdict(list)
        family_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in frame_rows:
            pattern_groups[
                (
                    str(row["adsorption_motif"]),
                    str(row["site_type"]),
                    str(row["support_key"]),
                )
            ].append(row)
            family_groups[str(row["adsorption_motif"])].append(row)

        pattern_summary: list[dict[str, object]] = []
        for (family, site_type, support_key), rows in sorted(
            pattern_groups.items(), key=lambda item: (-len(item[1]), item[0])
        ):
            pattern_summary.append(
                {
                    "adsorption_motif": family,
                    "coordination_label": str(rows[0]["coordination_label"]),
                    "coordination_n": int(rows[0]["coordination_n"]),
                    "site_type": site_type,
                    "support_key": support_key,
                    "samples": int(len(rows)),
                    "population": (
                        float(len(rows) / total_samples)
                        if total_samples
                        else float("nan")
                    ),
                    "n_sites": int(
                        len({int(row["site_local_index"]) for row in rows})
                    ),
                    "n_frames": int(len({int(row["frame"]) for row in rows})),
                    "anchor_site_xy_dist_A_mean": self._safe_mean(
                        [float(row["anchor_site_xy_dist_A"]) for row in rows]
                    ),
                    "anchor_support_min_dist_A_mean": self._safe_mean(
                        [float(row["anchor_support_min_dist_A"]) for row in rows]
                    ),
                    "anchor_height_above_support_A_mean": self._safe_mean(
                        [float(row["anchor_height_above_support_A"]) for row in rows]
                    ),
                    "support_outward_displacement_A_max_mean": self._safe_mean(
                        [
                            float(row["support_outward_displacement_A_max"])
                            for row in rows
                        ]
                    ),
                }
            )

        family_summary: list[dict[str, object]] = []
        for family, rows in sorted(
            family_groups.items(), key=lambda item: (-len(item[1]), item[0])
        ):
            family_summary.append(
                {
                    "adsorption_motif": family,
                    "samples": int(len(rows)),
                    "population": (
                        float(len(rows) / total_samples)
                        if total_samples
                        else float("nan")
                    ),
                    "n_sites": int(
                        len({int(row["site_local_index"]) for row in rows})
                    ),
                    "n_support_compositions": int(
                        len({str(row["support_key"]) for row in rows})
                    ),
                }
            )

        transition_counts: dict[tuple[str, str], int] = defaultdict(int)
        rows_by_group: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in frame_rows:
            rows_by_group[int(row["group_tag"])].append(row)
        for rows in rows_by_group.values():
            ordered = sorted(rows, key=lambda row: int(row["frame"]))
            for previous, current in zip(ordered[:-1], ordered[1:]):
                transition_counts[
                    (
                        str(previous["adsorption_motif"]),
                        str(current["adsorption_motif"]),
                    )
                ] += 1
        transitions_from: dict[str, int] = defaultdict(int)
        for (source, _), count in transition_counts.items():
            transitions_from[source] += count
        transition_summary = [
            {
                "from_motif": source,
                "to_motif": target,
                "count": int(count),
                "probability_from": float(count / transitions_from[source]),
                "changed": bool(source != target),
            }
            for (source, target), count in sorted(
                transition_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]

        return {
            "traj": str(traj_path),
            "temperature_K": temperature,
            "reference_site_descriptors": [
                {key: value for key, value in row.items() if not str(key).startswith("_")}
                for row in descriptors
            ],
            "local_motif_frames": frame_rows,
            "local_motif_summary": motif_summary,
            "local_site_summary": site_summary,
            "adsorption_pattern_summary": pattern_summary,
            "adsorption_family_summary": family_summary,
            "adsorption_motif_transitions": transition_summary,
        }

    def export_csv(self, result: Dict[str, object], out_prefix: str | Path) -> None:
        out_prefix = Path(out_prefix)

        frame_rows = result.get("local_motif_frames", [])
        if frame_rows:
            _write_csv(
                Path(f"{out_prefix}_per_frame.csv"),
                frame_rows,
                list(frame_rows[0].keys()),
            )

        motif_rows = result.get("local_motif_summary", [])
        if motif_rows:
            _write_csv(
                Path(f"{out_prefix}_summary.csv"),
                motif_rows,
                list(motif_rows[0].keys()),
            )

        site_rows = result.get("local_site_summary", [])
        if site_rows:
            _write_csv(
                Path(f"{out_prefix}_site_summary.csv"),
                site_rows,
                list(site_rows[0].keys()),
            )

        pattern_rows = result.get("adsorption_pattern_summary", [])
        if pattern_rows:
            _write_csv(
                Path(f"{out_prefix}_patterns.csv"),
                pattern_rows,
                list(pattern_rows[0].keys()),
            )

        family_rows = result.get("adsorption_family_summary", [])
        if family_rows:
            _write_csv(
                Path(f"{out_prefix}_families.csv"),
                family_rows,
                list(family_rows[0].keys()),
            )

        transition_rows = result.get("adsorption_motif_transitions", [])
        if transition_rows:
            _write_csv(
                Path(f"{out_prefix}_transitions.csv"),
                transition_rows,
                list(transition_rows[0].keys()),
            )

    @staticmethod
    def _representative_score_rows(
        rows: Sequence[dict[str, object]],
    ) -> list[tuple[float, dict[str, object]]]:
        if not rows:
            return []
        fields = (
            "anchor_site_xy_dist_A",
            "anchor_support_min_dist_A",
            "anchor_z_offset_A",
        )
        arrays = {
            field: np.asarray([float(row[field]) for row in rows], dtype=float)
            for field in fields
        }
        means = {field: float(np.mean(values)) for field, values in arrays.items()}
        stds = {
            field: float(np.std(values)) if float(np.std(values)) > 1.0e-12 else 1.0
            for field, values in arrays.items()
        }
        scored: list[tuple[float, dict[str, object]]] = []
        for row in rows:
            score = 0.0
            for field in fields:
                score += ((float(row[field]) - means[field]) / stds[field]) ** 2
            scored.append((float(score), row))
        scored.sort(key=lambda item: (item[0], int(item[1]["frame"])))
        return scored

    def export_representatives(
        self,
        result: Dict[str, object],
        out_prefix: str | Path,
        *,
        group_by: str = "motif",
        top_k: int = 5,
        n_per_group: int = 1,
    ) -> None:
        out_prefix = Path(out_prefix)
        traj_path = Path(str(result["traj"]))
        frame_rows = list(result.get("local_motif_frames", []))
        if not frame_rows:
            return

        if group_by not in {"motif", "site", "pattern"}:
            raise ValueError("group_by must be 'motif', 'site', or 'pattern'.")
        if top_k < 1 or n_per_group < 1:
            raise ValueError("top_k and n_per_group must be >= 1.")

        if group_by == "motif":
            key_field = "motif_key"
            summary_rows = list(result.get("local_motif_summary", []))
        elif group_by == "site":
            key_field = "site_local_index"
            summary_rows = list(result.get("local_site_summary", []))
        else:
            key_field = "adsorption_motif"
            summary_rows = list(result.get("adsorption_family_summary", []))
        selected_keys = [row[key_field] for row in summary_rows[:top_k]]
        grouped_rows: dict[object, list[dict[str, object]]] = defaultdict(list)
        for row in frame_rows:
            grouped_rows[row[key_field]].append(row)

        selected_rows: list[dict[str, object]] = []
        for key in selected_keys:
            scored = self._representative_score_rows(grouped_rows.get(key, ()))
            if not scored:
                continue
            used_frames: set[int] = set()
            chosen = 0
            for score, row in scored:
                frame = int(row["frame"])
                if frame in used_frames:
                    continue
                picked = dict(row)
                picked["representative_group_by"] = group_by
                picked["representative_score"] = float(score)
                picked["representative_rank_within_group"] = int(chosen)
                selected_rows.append(picked)
                used_frames.add(frame)
                chosen += 1
                if chosen >= n_per_group:
                    break

        if not selected_rows:
            return

        frames = read(str(traj_path), index=":")
        selected_rows.sort(
            key=lambda row: (
                str(row[key_field]),
                int(row["representative_rank_within_group"]),
                int(row["frame"]),
            )
        )
        selected_atoms: list[Atoms] = []
        for row in selected_rows:
            atoms = frames[int(row["frame"])].copy()
            atoms.info["local_motif_key"] = str(row["motif_key"])
            atoms.info["adsorption_motif"] = str(row["adsorption_motif"])
            atoms.info["coordination_n"] = int(row["coordination_n"])
            atoms.info["local_site_index"] = int(row["site_local_index"])
            atoms.info["local_registry_index"] = int(row["registry_index"])
            atoms.info["representative_group_by"] = str(group_by)
            atoms.info["representative_rank_within_group"] = int(
                row["representative_rank_within_group"]
            )
            atoms.info["source_frame"] = int(row["frame"])
            selected_atoms.append(atoms)

        write(Path(f"{out_prefix}_representatives.traj"), selected_atoms)
        _write_csv(
            Path(f"{out_prefix}_representatives.csv"),
            selected_rows,
            list(selected_rows[0].keys()),
        )
