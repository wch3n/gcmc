"""Generate candidate structures for site-conditioned reaction states."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import LBFGS

from gcmc.workflows import build_adsorbate_gcmc_calculator
from gcmc.utils import build_surface_site_registry

from .output_files import output_path
from .parent_sites import safe_float, write_csv


_DEFAULT_CANDIDATE_CONFIG = {
    "enabled": True,
    "oh_bond_cutoff_A": 1.35,
    "nearby_site_radius_A": 3.0,
    "max_nearby_sites": 8,
    "include_original_o_site": True,
    "include_nearby_o_sites": True,
    "ooh_orientations": 8,
    "oo_bond_A": 1.45,
    "terminal_oh_bond_A": 0.98,
    "ooh_tilt_z": 0.45,
    "child_slab_mode": "parent_conditioned",
    "child_slab_relax_fmax": 0.05,
    "child_slab_relax_steps": 300,
    "child_slab_relax_log_file": None,
    "calculator": None,
    "model": None,
    "model_file": None,
    "device": None,
    "use_kokkos": None,
}

_ORIGIN_INDEX_ARRAY = "reaction_origin_index"
_IS_ADSORBATE_ARRAY = "reaction_is_adsorbate"


def default_candidate_generation_config() -> dict[str, object]:
    return dict(_DEFAULT_CANDIDATE_CONFIG)


def state_species(state_dir: str) -> str:
    """Return normalized species name from a state directory such as 02_O."""

    suffix = str(state_dir).split("_", 1)[-1]
    return suffix.upper()


class ReactionCandidateGenerator:
    """Create clean/OH/O/OOH candidate structures for selected parent sites."""

    def __init__(self, config: SimpleNamespace):
        self.config = config
        raw = getattr(config, "candidate_generation", {}) or {}
        merged = default_candidate_generation_config()
        if isinstance(raw, dict):
            merged.update(raw)
        self.candidate_config = merged

    def enabled(self) -> bool:
        return bool(self.candidate_config.get("enabled", True))

    def generate(self, site_manifest_path: str | Path) -> dict[str, str]:
        if not self.enabled():
            return {}

        site_manifest_path = Path(site_manifest_path)
        rows = self._read_csv(site_manifest_path)
        if not rows:
            return {}

        candidate_manifest_path = output_path(self.config, "candidate_manifest_csv")
        manifest_rows: list[dict[str, object]] = []

        for site_row in rows:
            representatives_traj = Path(str(site_row["site_representatives_traj"]))
            representative_rows = self._read_csv(site_row["site_representatives_csv"])
            representatives = self._read_atoms_list(representatives_traj)
            reaction_state_dirs = self._state_dirs_for_row(site_row)
            by_species: dict[str, list[tuple[Atoms, dict[str, object]]]] = {}

            for state_dir in reaction_state_dirs:
                species = state_species(state_dir)
                state_path = Path(str(site_row["reactions_dir"])) / state_dir
                state_path.mkdir(parents=True, exist_ok=True)

                if species == "CLEAN":
                    candidates = self._generate_clean_candidates(
                        representatives,
                        representative_rows,
                        site_row,
                    )
                elif species == "OH":
                    candidates = self._generate_oh_candidates(
                        representatives,
                        representative_rows,
                        site_row,
                    )
                elif species == "O":
                    candidates = self._generate_o_candidates(
                        representatives,
                        representative_rows,
                        site_row,
                    )
                elif species == "OOH":
                    o_candidates = by_species.get("O")
                    if o_candidates is None:
                        o_candidates = self._generate_o_candidates(
                            representatives,
                            representative_rows,
                            site_row,
                        )
                    candidates = self._generate_ooh_candidates(
                        o_candidates,
                        site_row,
                    )
                else:
                    candidates = []

                by_species[species] = candidates
                candidates_traj = state_path / "candidates.traj"
                candidates_csv = state_path / "candidates.csv"
                self._write_candidates(candidates, candidates_traj, candidates_csv)
                manifest_rows.append(
                    {
                        "site_id": site_row.get("site_id", ""),
                        "site_population_rank": site_row.get(
                            "site_population_rank",
                            "",
                        ),
                        "population_total": site_row.get("population_total", ""),
                        "state_dir": state_dir,
                        "species": species,
                        "n_candidates": len(candidates),
                        "candidates_traj": str(candidates_traj),
                        "candidates_csv": str(candidates_csv),
                    }
                )

        write_csv(candidate_manifest_path, manifest_rows)
        return {"candidate_manifest_csv": str(candidate_manifest_path)}

    @staticmethod
    def _read_csv(path: str | Path) -> list[dict[str, str]]:
        with Path(path).open(newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    @staticmethod
    def _read_atoms_list(path: Path) -> list[Atoms]:
        atoms = read(str(path), index=":")
        if isinstance(atoms, Atoms):
            return [atoms]
        return list(atoms)

    def _state_dirs_for_row(self, row: dict[str, object]) -> tuple[str, ...]:
        value = str(row.get("reaction_state_dirs", "")).strip()
        if value:
            return tuple(value.split())
        configured = getattr(self.config, "reaction_state_dirs", ())
        if isinstance(configured, str):
            return (configured,)
        return tuple(str(item) for item in configured) or (
            "00_clean",
            "01_OH",
            "02_O",
            "03_OOH",
        )

    def _generate_clean_candidates(
        self,
        representatives: Sequence[Atoms],
        representative_rows: Sequence[dict[str, object]],
        site_row: dict[str, object],
    ) -> list[tuple[Atoms, dict[str, object]]]:
        candidates: list[tuple[Atoms, dict[str, object]]] = []
        for rep_index, atoms in enumerate(representatives):
            row = self._representative_row(representative_rows, rep_index)
            anchor_idx = self._anchor_index(atoms, row)
            adsorbate_indices = self._adsorbate_indices(atoms, anchor_idx)
            atoms = atoms.copy()
            self._ensure_provenance_arrays(atoms, adsorbate_indices)
            candidate, _ = self._delete_indices(atoms, adsorbate_indices, anchor_idx)
            metadata = self._base_metadata(
                site_row,
                row,
                state="00_clean",
                species="CLEAN",
                candidate_id=f"rep{rep_index:03d}_clean",
                candidate_kind="adsorbate_removed",
                anchor_index=-1,
            )
            metadata["source_anchor_index"] = int(anchor_idx)
            metadata["removed_adsorbate_indices"] = " ".join(
                str(int(idx)) for idx in adsorbate_indices
            )
            self._annotate_candidate(candidate, metadata)
            candidates.append((candidate, metadata))
        return candidates

    def _generate_oh_candidates(
        self,
        representatives: Sequence[Atoms],
        representative_rows: Sequence[dict[str, object]],
        site_row: dict[str, object],
    ) -> list[tuple[Atoms, dict[str, object]]]:
        candidates: list[tuple[Atoms, dict[str, object]]] = []
        for rep_index, atoms in enumerate(representatives):
            row = self._representative_row(representative_rows, rep_index)
            candidate = atoms.copy()
            anchor_idx = self._anchor_index(candidate, row)
            adsorbate_indices = self._adsorbate_indices(candidate, anchor_idx)
            self._ensure_provenance_arrays(candidate, adsorbate_indices)
            metadata = self._base_metadata(
                site_row,
                row,
                state="01_OH",
                species="OH",
                candidate_id=f"rep{rep_index:03d}_oh_parent",
                candidate_kind="parent",
                anchor_index=anchor_idx,
            )
            self._annotate_candidate(candidate, metadata)
            candidates.append((candidate, metadata))
        return candidates

    def _generate_o_candidates(
        self,
        representatives: Sequence[Atoms],
        representative_rows: Sequence[dict[str, object]],
        site_row: dict[str, object],
    ) -> list[tuple[Atoms, dict[str, object]]]:
        candidates: list[tuple[Atoms, dict[str, object]]] = []
        for rep_index, atoms in enumerate(representatives):
            row = self._representative_row(representative_rows, rep_index)
            atoms = atoms.copy()
            anchor_idx = self._anchor_index(atoms, row)
            h_idx = self._bound_h_index(atoms, anchor_idx)
            if h_idx is None:
                continue
            adsorbate_indices = self._adsorbate_indices(atoms, anchor_idx)
            self._ensure_provenance_arrays(atoms, adsorbate_indices)

            child_slab_mode = self._child_slab_mode()
            if child_slab_mode != "parent_conditioned":
                slab, _ = self._delete_indices(atoms, adsorbate_indices, anchor_idx)
                if child_slab_mode == "relaxed_parent_stripped":
                    slab = self._relax_child_slab(slab, site_row, rep_index)

                if bool(self.candidate_config.get("include_original_o_site", True)):
                    direct = self._generate_direct_o_on_slab(
                        slab,
                        atoms,
                        row,
                        site_row,
                        rep_index,
                        anchor_idx,
                    )
                    if direct is not None:
                        candidates.append(direct)

                if bool(self.candidate_config.get("include_nearby_o_sites", True)):
                    candidates.extend(
                        self._generate_shifted_o_on_slab(
                            slab,
                            atoms,
                            row,
                            site_row,
                            rep_index,
                            anchor_idx,
                        )
                    )
                continue

            if bool(self.candidate_config.get("include_original_o_site", True)):
                o_atoms, o_anchor = self._delete_indices(atoms, [h_idx], anchor_idx)
                metadata = self._base_metadata(
                    site_row,
                    row,
                    state="02_O",
                    species="O",
                    candidate_id=f"rep{rep_index:03d}_o_direct",
                    candidate_kind="direct_deprotonation",
                    anchor_index=o_anchor,
                )
                self._annotate_candidate(o_atoms, metadata)
                candidates.append((o_atoms, metadata))

            if bool(self.candidate_config.get("include_nearby_o_sites", True)):
                candidates.extend(
                    self._generate_shifted_o_candidates(
                        atoms,
                        row,
                        site_row,
                        rep_index,
                        anchor_idx,
                    )
                )
        return candidates

    def _child_slab_mode(self) -> str:
        mode = str(
            self.candidate_config.get("child_slab_mode", "parent_conditioned")
        ).lower()
        aliases = {
            "parent": "parent_conditioned",
            "oh_parent": "parent_conditioned",
            "stripped": "parent_stripped",
            "clean": "parent_stripped",
            "relaxed_clean": "relaxed_parent_stripped",
            "relaxed": "relaxed_parent_stripped",
        }
        mode = aliases.get(mode, mode)
        allowed = {
            "parent_conditioned",
            "parent_stripped",
            "relaxed_parent_stripped",
        }
        if mode not in allowed:
            raise ValueError(
                "candidate_generation.child_slab_mode must be one of "
                "'parent_conditioned', 'parent_stripped', or "
                "'relaxed_parent_stripped'."
            )
        return mode

    def _generate_direct_o_on_slab(
        self,
        slab: Atoms,
        source: Atoms,
        representative_row: dict[str, object],
        site_row: dict[str, object],
        rep_index: int,
        anchor_idx: int,
    ) -> tuple[Atoms, dict[str, object]] | None:
        registry = self._surface_site_registry(slab)
        site = self._matching_parent_site(registry, site_row)
        if site is None:
            return None
        candidate = slab.copy()
        position = np.array(
            [
                float(np.asarray(site["xy"])[0]),
                float(np.asarray(site["xy"])[1]),
                float(site["suggested_z_A"]),
            ],
            dtype=float,
        )
        anchor_new = self._append_adsorbate_atoms(
            candidate,
            Atoms("O", positions=[position]),
            source,
        )[0]
        metadata = self._base_metadata(
            site_row,
            representative_row,
            state="02_O",
            species="O",
            candidate_id=f"rep{rep_index:03d}_o_direct",
            candidate_kind="direct_parent_stripped",
            anchor_index=anchor_new,
        )
        metadata.update(
            {
                "source_anchor_index": int(anchor_idx),
                "candidate_site_type": site.get("site_type", ""),
                "candidate_support_indices": " ".join(
                    str(int(idx)) for idx in site.get("support_indices", [])
                ),
                "candidate_site_xy_distance_A": 0.0,
                "candidate_site_suggested_z_A": float(site["suggested_z_A"]),
                "child_slab_mode": self._child_slab_mode(),
            }
        )
        self._annotate_candidate(candidate, metadata)
        return candidate, metadata

    def _matching_parent_site(
        self,
        registry: Sequence[dict[str, object]],
        site_row: dict[str, object],
    ) -> dict[str, object] | None:
        parent_support = self._support_tuple(site_row.get("support_indices_sorted", ""))
        parent_site_type = str(site_row.get("site_type", ""))
        for site in registry:
            if bool(site.get("blocked_by_termination", False)):
                continue
            support = tuple(sorted(int(idx) for idx in site.get("support_indices", [])))
            site_type = str(site.get("site_type", ""))
            if site_type == parent_site_type and support == parent_support:
                return dict(site)
        return None

    def _generate_shifted_o_candidates(
        self,
        atoms: Atoms,
        representative_row: dict[str, object],
        site_row: dict[str, object],
        rep_index: int,
        anchor_idx: int,
    ) -> list[tuple[Atoms, dict[str, object]]]:
        adsorbate_indices = self._adsorbate_indices(atoms, anchor_idx)
        slab, _ = self._delete_indices(atoms, adsorbate_indices, anchor_idx)
        return self._generate_shifted_o_on_slab(
            slab,
            atoms,
            representative_row,
            site_row,
            rep_index,
            anchor_idx,
        )

    def _generate_shifted_o_on_slab(
        self,
        slab: Atoms,
        source: Atoms,
        representative_row: dict[str, object],
        site_row: dict[str, object],
        rep_index: int,
        anchor_idx: int,
    ) -> list[tuple[Atoms, dict[str, object]]]:
        registry = self._surface_site_registry(slab)
        anchor_xy = source.positions[anchor_idx, :2]
        parent_support = self._support_tuple(site_row.get("support_indices_sorted", ""))
        parent_site_type = str(site_row.get("site_type", ""))

        nearby: list[tuple[float, dict[str, object]]] = []
        radius = float(self.candidate_config.get("nearby_site_radius_A", 3.0))
        for site in registry:
            if bool(site.get("blocked_by_termination", False)):
                continue
            support = tuple(sorted(int(idx) for idx in site.get("support_indices", [])))
            site_type = str(site.get("site_type", ""))
            if site_type == parent_site_type and support == parent_support:
                continue
            distance = self._xy_distance(slab, anchor_xy, np.asarray(site["xy"], dtype=float))
            if distance <= radius:
                nearby.append((distance, site))

        nearby.sort(key=lambda item: (item[0], str(item[1].get("site_type", ""))))
        max_nearby = int(self.candidate_config.get("max_nearby_sites", 8))
        candidates: list[tuple[Atoms, dict[str, object]]] = []
        for nearby_index, (distance, site) in enumerate(nearby[:max_nearby]):
            candidate = slab.copy()
            position = np.array(
                [
                    float(np.asarray(site["xy"])[0]),
                    float(np.asarray(site["xy"])[1]),
                    float(site["suggested_z_A"]),
                ],
                dtype=float,
            )
            anchor_new = self._append_adsorbate_atoms(
                candidate,
                Atoms("O", positions=[position]),
                source,
            )[0]
            support = " ".join(str(int(idx)) for idx in site.get("support_indices", []))
            metadata = self._base_metadata(
                site_row,
                representative_row,
                state="02_O",
                species="O",
                candidate_id=f"rep{rep_index:03d}_o_nearby{nearby_index:02d}",
                candidate_kind="nearby_site_shift",
                anchor_index=anchor_new,
            )
            metadata.update(
                {
                    "candidate_site_type": site.get("site_type", ""),
                    "candidate_support_indices": support,
                    "candidate_site_xy_distance_A": float(distance),
                    "candidate_site_suggested_z_A": float(site["suggested_z_A"]),
                }
            )
            if self._child_slab_mode() != "parent_conditioned":
                metadata["child_slab_mode"] = self._child_slab_mode()
            self._annotate_candidate(candidate, metadata)
            candidates.append((candidate, metadata))
        return candidates

    def _relax_child_slab(
        self,
        slab: Atoms,
        site_row: dict[str, object],
        rep_index: int,
    ) -> Atoms:
        atoms = slab.copy()
        atoms.calc = self._build_child_slab_calculator()
        log_file = self.candidate_config.get("child_slab_relax_log_file")
        logfile = None
        if log_file not in (None, "", False):
            path = Path(str(log_file))
            if not path.is_absolute():
                path = Path(str(site_row.get("reactions_dir", "."))) / path
            path.parent.mkdir(parents=True, exist_ok=True)
            logfile = str(path.with_name(f"{path.stem}_rep{rep_index:03d}{path.suffix}"))
        dyn = LBFGS(atoms, logfile=logfile)
        dyn.run(
            fmax=float(self.candidate_config.get("child_slab_relax_fmax", 0.05)),
            steps=int(self.candidate_config.get("child_slab_relax_steps", 300)),
        )
        atoms.calc = None
        return atoms

    def _build_child_slab_calculator(self):
        values = vars(self.config).copy()
        for key in ("calculator", "model", "model_file", "device", "use_kokkos"):
            value = self.candidate_config.get(key)
            if value not in (None, ""):
                values[key] = value
        cfg = SimpleNamespace(**values)
        return build_adsorbate_gcmc_calculator(
            cfg,
            {"device": values.get("device") or getattr(self.config, "device", None)},
        )

    def _generate_ooh_candidates(
        self,
        o_candidates: Sequence[tuple[Atoms, dict[str, object]]],
        site_row: dict[str, object],
    ) -> list[tuple[Atoms, dict[str, object]]]:
        candidates: list[tuple[Atoms, dict[str, object]]] = []
        n_orientations = int(self.candidate_config.get("ooh_orientations", 8))
        oo_bond = float(self.candidate_config.get("oo_bond_A", 1.45))
        oh_bond = float(self.candidate_config.get("terminal_oh_bond_A", 0.98))
        tilt_z = float(self.candidate_config.get("ooh_tilt_z", 0.45))
        z_sign = 1.0 if str(self.config.surface_side).lower() == "top" else -1.0
        for o_atoms, o_meta in o_candidates:
            anchor_idx = int(o_meta["anchor_index"])
            anchor_position = o_atoms.positions[anchor_idx]
            for orientation in range(max(n_orientations, 1)):
                angle = 2.0 * math.pi * orientation / max(n_orientations, 1)
                oo_direction = np.array(
                    [math.cos(angle), math.sin(angle), z_sign * tilt_z],
                    dtype=float,
                )
                oo_direction /= np.linalg.norm(oo_direction)
                distal_o = anchor_position + oo_bond * oo_direction
                oh_direction = np.array(
                    [0.25 * math.cos(angle), 0.25 * math.sin(angle), z_sign],
                    dtype=float,
                )
                oh_direction /= np.linalg.norm(oh_direction)
                terminal_h = distal_o + oh_bond * oh_direction

                candidate = o_atoms.copy()
                new_indices = self._append_adsorbate_atoms(
                    candidate,
                    Atoms("OH", positions=[distal_o, terminal_h]),
                    o_atoms,
                )
                metadata = dict(o_meta)
                metadata.update(
                    {
                        "state": "03_OOH",
                        "species": "OOH",
                        "candidate_id": (
                            f"{o_meta['candidate_id']}_ooh_ori{orientation:02d}"
                        ),
                        "candidate_kind": "ooh_orientation",
                        "parent_o_candidate_id": o_meta["candidate_id"],
                        "orientation_index": orientation,
                        "oo_bond_A": oo_bond,
                        "terminal_oh_bond_A": oh_bond,
                    }
                )
                self._annotate_candidate(candidate, metadata)
                candidates.append((candidate, metadata))
        return candidates

    def _surface_site_registry(self, slab: Atoms) -> list[dict[str, object]]:
        return build_surface_site_registry(
            slab,
            site_elements=self.config.site_elements,
            substrate_elements=self.config.substrate_elements,
            surface_side=self.config.surface_side,
            site_types=self.config.site_types,
            layer_tol=float(self.config.surface_layer_tol),
            xy_tol=float(self.config.site_match_tol),
            support_xy_tol=float(self.config.support_xy_tol),
            termination_site_xy_tol=self.config.termination_site_xy_tol,
            vertical_offset=float(self.config.vertical_offset),
            termination_elements=self.config.functional_elements,
            min_termination_dist=float(self.config.termination_clearance),
        )

    @staticmethod
    def _representative_row(
        rows: Sequence[dict[str, object]],
        index: int,
    ) -> dict[str, object]:
        if not rows:
            return {}
        return dict(rows[min(index, len(rows) - 1)])

    @staticmethod
    def _support_tuple(value: object) -> tuple[int, ...]:
        text = str(value).strip()
        if not text:
            return ()
        return tuple(sorted(int(part) for part in text.split()))

    @staticmethod
    def _xy_distance(atoms: Atoms, xy_a: np.ndarray, xy_b: np.ndarray) -> float:
        vec = np.array([xy_b[0] - xy_a[0], xy_b[1] - xy_a[1], 0.0], dtype=float)
        if any(atoms.pbc[:2]):
            cell = atoms.cell.array
            try:
                frac = np.linalg.solve(cell.T, vec)
                for axis in (0, 1):
                    if atoms.pbc[axis]:
                        frac[axis] -= np.round(frac[axis])
                vec = frac @ cell
            except np.linalg.LinAlgError:
                pass
        return float(np.linalg.norm(vec[:2]))

    @staticmethod
    def _anchor_index(atoms: Atoms, row: dict[str, object]) -> int:
        idx = int(safe_float(row.get("representative_anchor_index"), -1))
        if 0 <= idx < len(atoms) and atoms[idx].symbol == "O":
            return idx
        tags = atoms.get_tags()
        tagged = np.flatnonzero(tags != 0)
        oxygen_tagged = [int(i) for i in tagged if atoms[int(i)].symbol == "O"]
        if oxygen_tagged:
            return oxygen_tagged[0]
        oxygen = [idx for idx, atom in enumerate(atoms) if atom.symbol == "O"]
        if not oxygen:
            raise ValueError("Cannot identify O anchor for reaction candidate.")
        return oxygen[-1]

    def _bound_h_index(self, atoms: Atoms, anchor_idx: int) -> int | None:
        cutoff = float(self.candidate_config.get("oh_bond_cutoff_A", 1.35))
        tags = atoms.get_tags()
        candidate_indices: Iterable[int]
        if tags[anchor_idx] != 0:
            candidate_indices = np.flatnonzero(tags == tags[anchor_idx])
        else:
            candidate_indices = range(len(atoms))

        best: tuple[float, int] | None = None
        for idx in candidate_indices:
            idx = int(idx)
            if idx == anchor_idx or atoms[idx].symbol != "H":
                continue
            distance = float(atoms.get_distance(anchor_idx, idx, mic=True))
            if distance <= cutoff and (best is None or distance < best[0]):
                best = (distance, idx)
        return best[1] if best is not None else None

    @staticmethod
    def _adsorbate_indices(atoms: Atoms, anchor_idx: int) -> list[int]:
        tags = atoms.get_tags()
        if tags[anchor_idx] != 0:
            return [int(idx) for idx in np.flatnonzero(tags == tags[anchor_idx])]
        indices = [anchor_idx]
        for idx, atom in enumerate(atoms):
            if atom.symbol == "H" and atoms.get_distance(anchor_idx, idx, mic=True) < 1.35:
                indices.append(idx)
        return sorted(set(indices))

    @staticmethod
    def _delete_indices(
        atoms: Atoms,
        delete_indices: Sequence[int],
        anchor_idx: int,
    ) -> tuple[Atoms, int]:
        delete_set = set(int(idx) for idx in delete_indices)
        candidate = atoms.copy()
        for idx in sorted(delete_set, reverse=True):
            del candidate[idx]
        if anchor_idx in delete_set:
            return candidate, len(candidate) - 1
        shift = sum(1 for idx in delete_set if idx < anchor_idx)
        return candidate, int(anchor_idx - shift)

    @staticmethod
    def _set_adsorbate_tag(candidate: Atoms, indices: Sequence[int], source: Atoms) -> None:
        tags = candidate.get_tags()
        source_tags = source.get_tags()
        tag = int(np.max(source_tags)) if source_tags.size else 0
        if tag == 0:
            tag = 1_000_000
        tags[list(indices)] = tag
        candidate.set_tags(tags)

    @classmethod
    def _append_adsorbate_atoms(
        cls,
        candidate: Atoms,
        adsorbate: Atoms,
        source: Atoms,
    ) -> list[int]:
        start = len(candidate)
        cls._ensure_provenance_arrays(candidate)
        added = adsorbate.copy()
        origin = -np.arange(1, len(added) + 1, dtype=int)
        added.new_array(_ORIGIN_INDEX_ARRAY, origin)
        added.new_array(_IS_ADSORBATE_ARRAY, np.ones(len(added), dtype=int))
        candidate += added
        new_indices = list(range(start, len(candidate)))
        cls._set_adsorbate_tag(candidate, new_indices, source)
        return new_indices

    @staticmethod
    def _ensure_provenance_arrays(
        atoms: Atoms,
        adsorbate_indices: Sequence[int] | None = None,
    ) -> None:
        if _ORIGIN_INDEX_ARRAY not in atoms.arrays:
            atoms.new_array(_ORIGIN_INDEX_ARRAY, np.arange(len(atoms), dtype=int))

        if adsorbate_indices is None:
            tags = atoms.get_tags()
            if tags.size:
                adsorbate_indices = np.flatnonzero(tags != 0)
            else:
                adsorbate_indices = ()
        mask = np.zeros(len(atoms), dtype=int)
        adsorbate_list = [int(idx) for idx in adsorbate_indices]
        if adsorbate_list:
            mask[adsorbate_list] = 1
        if _IS_ADSORBATE_ARRAY in atoms.arrays:
            atoms.set_array(_IS_ADSORBATE_ARRAY, mask)
        else:
            atoms.new_array(_IS_ADSORBATE_ARRAY, mask)

    @staticmethod
    def _base_metadata(
        site_row: dict[str, object],
        representative_row: dict[str, object],
        *,
        state: str,
        species: str,
        candidate_id: str,
        candidate_kind: str,
        anchor_index: int,
    ) -> dict[str, object]:
        return {
            "site_id": site_row.get("site_id", ""),
            "site_population_rank": site_row.get("site_population_rank", ""),
            "state": state,
            "species": species,
            "candidate_id": candidate_id,
            "candidate_kind": candidate_kind,
            "anchor_index": int(anchor_index),
            "source_representative_frame": representative_row.get(
                "representative_frame",
                "",
            ),
            "source_representative_traj": representative_row.get(
                "representative_traj",
                "",
            ),
            "source_representative_rank": representative_row.get(
                "representative_rank_within_group",
                "",
            ),
        }

    @staticmethod
    def _annotate_candidate(atoms: Atoms, metadata: dict[str, object]) -> None:
        if _ORIGIN_INDEX_ARRAY in atoms.arrays:
            metadata.setdefault(
                "atom_origin_indices",
                " ".join(str(int(value)) for value in atoms.arrays[_ORIGIN_INDEX_ARRAY]),
            )
        if _IS_ADSORBATE_ARRAY in atoms.arrays:
            metadata.setdefault(
                "atom_is_adsorbate",
                " ".join(str(int(value)) for value in atoms.arrays[_IS_ADSORBATE_ARRAY]),
            )
        for key, value in metadata.items():
            atoms.info[f"reaction_{key}"] = value

    @staticmethod
    def _write_candidates(
        candidates: Sequence[tuple[Atoms, dict[str, object]]],
        traj_path: Path,
        csv_path: Path,
    ) -> None:
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        if candidates:
            write(str(traj_path), [atoms for atoms, _ in candidates])
        rows = [metadata for _, metadata in candidates]
        if not rows:
            return
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
