"""Relax generated reaction-state candidates with an ASE calculator."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.optimize import LBFGS

from gcmc.workflows import build_adsorbate_gcmc_calculator

from .output_files import output_path


_DEFAULT_STATE_RELAXATION_CONFIG = {
    "enabled": False,
    "fmax": 0.05,
    "steps": 300,
    "log_file": None,
    "progress_log": None,
    "progress_stdout": True,
    "max_candidates_per_state": None,
    "states": [],
    "skip_existing": False,
    "calculator": None,
    "model": None,
    "model_file": None,
    "device": None,
    "use_kokkos": None,
    "enforce_adsorbate_integrity": True,
    "adsorbate_bond_detection_factor": 1.2,
    "adsorbate_bond_stretch_factor": 1.35,
    "adsorbate_bond_abs_tol_A": 0.35,
}


def default_state_relaxation_config() -> dict[str, object]:
    return dict(_DEFAULT_STATE_RELAXATION_CONFIG)


class ReactionStateRelaxer:
    """Relax candidate structures listed in a reaction candidate manifest."""

    def __init__(self, config: SimpleNamespace):
        self.config = config
        raw = getattr(config, "state_relaxation", {}) or {}
        merged = default_state_relaxation_config()
        if isinstance(raw, dict):
            merged.update(raw)
        self.relax_config = merged

    def enabled(self) -> bool:
        return bool(self.relax_config.get("enabled", False))

    def relax(self, candidate_manifest_path: str | Path) -> dict[str, str]:
        if not self.enabled():
            return {}

        candidate_manifest_path = Path(candidate_manifest_path)
        if not candidate_manifest_path.exists():
            return {}

        selected_states = self._selected_states()
        manifest_rows = [
            row
            for row in self._read_csv(candidate_manifest_path)
            if self._row_selected(row, selected_states)
        ]
        if not manifest_rows:
            return {}

        output_rows: list[dict[str, object]] = []
        work_items: list[
            tuple[dict[str, object], Path, Path, Path, Path]
        ] = []

        for manifest_row in manifest_rows:
            state_dir = str(manifest_row.get("state_dir", ""))
            candidates_traj = Path(str(manifest_row.get("candidates_traj", "")))
            candidates_csv = Path(str(manifest_row.get("candidates_csv", "")))
            if not candidates_traj.exists():
                self._progress(
                    "state-relaxation skip missing candidates "
                    f"site={manifest_row.get('site_id', '')} state={state_dir} "
                    f"path={candidates_traj}"
                )
                continue

            state_path = candidates_traj.parent
            relaxed_traj = state_path / "relaxed.traj"
            energies_csv = state_path / "energies.csv"
            if bool(self.relax_config.get("skip_existing", False)) and energies_csv.exists():
                self._progress(
                    "state-relaxation skip existing "
                    f"site={manifest_row.get('site_id', '')} state={state_dir} "
                    f"energies_csv={energies_csv}"
                )
                output_rows.append(
                    {
                        "site_id": manifest_row.get("site_id", ""),
                        "state_dir": state_dir,
                        "species": manifest_row.get("species", ""),
                        "n_candidates": manifest_row.get("n_candidates", ""),
                        "n_relaxed": "existing",
                        "n_failed": "",
                        "relaxed_traj": str(relaxed_traj),
                        "energies_csv": str(energies_csv),
                    }
                )
                continue

            work_items.append(
                (
                    manifest_row,
                    candidates_traj,
                    candidates_csv,
                    relaxed_traj,
                    energies_csv,
                )
            )

        if not work_items:
            if output_rows:
                state_relaxation_manifest = self._manifest_path()
                self._write_csv(state_relaxation_manifest, output_rows)
                return {
                    "state_relaxation_manifest_csv": str(state_relaxation_manifest),
                }
            return {}

        self._progress(
            "state-relaxation start "
            f"manifest={candidate_manifest_path} "
            f"work_blocks={len(work_items)} skipped_existing={len(output_rows)}"
        )
        self._progress("state-relaxation build calculator")
        try:
            calculator = self._build_calculator()
        except Exception as exc:
            self._progress(
                "state-relaxation calculator error "
                f"{type(exc).__name__}: {exc}"
            )
            raise
        self._progress("state-relaxation calculator ready")

        for state_index, (
            manifest_row,
            candidates_traj,
            candidates_csv,
            relaxed_traj,
            energies_csv,
        ) in enumerate(work_items, start=1):
            candidate_rows = self._read_csv(candidates_csv) if candidates_csv.exists() else []
            candidates = self._read_atoms_list(candidates_traj)
            max_candidates = self.relax_config.get("max_candidates_per_state")
            if max_candidates is not None:
                candidates = candidates[: int(max_candidates)]
                candidate_rows = candidate_rows[: int(max_candidates)]

            self._progress(
                f"state-relaxation block {state_index}/{len(work_items)} start "
                f"site={manifest_row.get('site_id', '')} "
                f"state={manifest_row.get('state_dir', '')} "
                f"species={manifest_row.get('species', '')} "
                f"candidates={len(candidates)} "
                f"relaxed_traj={relaxed_traj} energies_csv={energies_csv}"
            )
            relaxed_atoms, energy_rows = self._relax_candidates(
                candidates,
                candidate_rows,
                manifest_row,
                calculator,
                relaxed_traj,
                energies_csv,
                state_index,
                len(work_items),
            )
            n_failed = sum(1 for row in energy_rows if str(row.get("error", "")))
            output_rows.append(
                {
                    "site_id": manifest_row.get("site_id", ""),
                    "state_dir": manifest_row.get("state_dir", ""),
                    "species": manifest_row.get("species", ""),
                    "n_candidates": manifest_row.get("n_candidates", ""),
                    "n_relaxed": len(relaxed_atoms),
                    "n_failed": n_failed,
                    "relaxed_traj": str(relaxed_traj),
                    "energies_csv": str(energies_csv),
                }
            )
            self._progress(
                f"state-relaxation block {state_index}/{len(work_items)} done "
                f"site={manifest_row.get('site_id', '')} "
                f"state={manifest_row.get('state_dir', '')} "
                f"relaxed={len(relaxed_atoms)} failed={n_failed}"
            )

        state_relaxation_manifest = self._manifest_path()
        self._write_csv(state_relaxation_manifest, output_rows)
        self._progress(
            "state-relaxation complete "
            f"manifest={state_relaxation_manifest} blocks={len(output_rows)}"
        )
        return {"state_relaxation_manifest_csv": str(state_relaxation_manifest)}

    def _manifest_path(self) -> Path:
        return output_path(self.config, "state_relaxation_manifest_csv")

    def _build_calculator(self):
        values = vars(self.config).copy()
        for key in ("calculator", "model", "model_file", "device", "use_kokkos"):
            value = self.relax_config.get(key)
            if value not in (None, ""):
                values[key] = value
        cfg = SimpleNamespace(**values)
        return build_adsorbate_gcmc_calculator(
            cfg,
            {"device": values.get("device") or getattr(self.config, "device", None)},
        )

    def _selected_states(self) -> set[str]:
        states = self.relax_config.get("states") or []
        if isinstance(states, str):
            states = [states]
        return {str(state) for state in states}

    @staticmethod
    def _row_selected(
        row: dict[str, object],
        selected_states: set[str],
    ) -> bool:
        if not selected_states:
            return True
        return (
            str(row.get("state_dir", "")) in selected_states
            or str(row.get("species", "")) in selected_states
        )

    def _relax_candidates(
        self,
        candidates: Sequence[Atoms],
        candidate_rows: Sequence[dict[str, object]],
        manifest_row: dict[str, object],
        calculator,
        relaxed_traj: Path,
        energies_csv: Path,
        state_index: int,
        state_count: int,
    ) -> tuple[list[Atoms], list[dict[str, object]]]:
        relaxed_atoms: list[Atoms] = []
        rows: list[dict[str, object]] = []
        fmax_target = float(self.relax_config.get("fmax", 0.05))
        max_steps = int(self.relax_config.get("steps", 300))
        log_file = self.relax_config.get("log_file")

        relaxed_traj.parent.mkdir(parents=True, exist_ok=True)
        traj = Trajectory(str(relaxed_traj), "w")
        try:
            for index, candidate in enumerate(candidates):
                atoms = candidate.copy()
                source_row = (
                    dict(candidate_rows[index]) if index < len(candidate_rows) else {}
                )
                candidate_id = str(source_row.get("candidate_id", index))
                self._progress(
                    f"state-relaxation block {state_index}/{state_count} "
                    f"candidate {index + 1}/{len(candidates)} start "
                    f"site={manifest_row.get('site_id', '')} "
                    f"state={manifest_row.get('state_dir', '')} "
                    f"id={candidate_id}"
                )
                atoms.calc = calculator
                converged = False
                energy = float("nan")
                final_fmax = float("nan")
                nsteps = 0
                error = ""
                try:
                    dyn = LBFGS(
                        atoms,
                        logfile=self._logfile_path(log_file, manifest_row, index),
                    )
                    converged = bool(dyn.run(fmax=fmax_target, steps=max_steps))
                    nsteps = int(getattr(dyn, "nsteps", 0))
                    energy = float(atoms.get_potential_energy())
                    forces = np.asarray(atoms.get_forces(), dtype=float)
                    final_fmax = float(np.sqrt((forces * forces).sum(axis=1).max()))
                except Exception as exc:  # noqa: BLE001 - record failed candidates for auditability.
                    error = f"{type(exc).__name__}: {exc}"

                intact, integrity_error, bond_distances = self._adsorbate_integrity(
                    candidate,
                    atoms,
                    source_row,
                    manifest_row,
                )
                if not intact:
                    converged = False
                    error = self._append_error(error, integrity_error)

                atoms.info["reaction_relax_converged"] = bool(converged)
                atoms.info["reaction_relaxed_energy_eV"] = energy
                atoms.info["reaction_relaxed_fmax_eV_A"] = final_fmax
                atoms.info["reaction_relax_error"] = error
                atoms.info["reaction_adsorbate_intact"] = bool(intact)
                atoms.info["reaction_adsorbate_bond_distances_A"] = bond_distances
                atoms.calc = None
                relaxed_atoms.append(atoms)

                row = self._energy_row(
                    manifest_row,
                    source_row,
                    index,
                    converged,
                    energy,
                    final_fmax,
                    nsteps,
                    error,
                    intact,
                    integrity_error,
                    bond_distances,
                )
                rows.append(row)
                traj.write(atoms)
                self._write_csv(energies_csv, rows)

                if error:
                    status = "error"
                elif converged:
                    status = "converged"
                else:
                    status = "not_converged"
                self._progress(
                    f"state-relaxation block {state_index}/{state_count} "
                    f"candidate {index + 1}/{len(candidates)} done "
                    f"site={manifest_row.get('site_id', '')} "
                    f"state={manifest_row.get('state_dir', '')} "
                    f"id={candidate_id} status={status} "
                    f"energy_eV={energy:.8f} fmax_eV_A={final_fmax:.6f} "
                    f"nsteps={nsteps}"
                )
        finally:
            traj.close()
        return relaxed_atoms, rows

    def _adsorbate_integrity(
        self,
        initial: Atoms,
        relaxed: Atoms,
        source_row: dict[str, object],
        manifest_row: dict[str, object],
    ) -> tuple[bool, str, str]:
        if not bool(self.relax_config.get("enforce_adsorbate_integrity", True)):
            return True, "", ""

        species = str(manifest_row.get("species", "")).upper()
        if species in {"", "CLEAN", "O"}:
            return True, "", ""

        indices = self._adsorbate_indices(initial, source_row)
        if len(indices) < 2:
            return True, "", ""

        bonds = self._adsorbate_template_bonds(initial, indices)
        if not bonds:
            return True, "", ""

        distances: list[str] = []
        for idx_i, idx_j, max_dist in bonds:
            if idx_i >= len(relaxed) or idx_j >= len(relaxed):
                message = (
                    "AdsorbateIntegrityError: relaxed structure is missing "
                    f"adsorbate bond atoms {idx_i}-{idx_j}."
                )
                return False, message, " ".join(distances)
            dist = float(relaxed.get_distance(idx_i, idx_j, mic=True))
            distances.append(f"{idx_i}-{idx_j}:{dist:.6f}")
            if dist > max_dist:
                message = (
                    "AdsorbateIntegrityError: molecular adsorbate dissociated "
                    f"for species={species}; bond {idx_i}-{idx_j} "
                    f"distance={dist:.3f} A exceeds limit={max_dist:.3f} A."
                )
                return False, message, " ".join(distances)
        return True, "", " ".join(distances)

    def _adsorbate_template_bonds(
        self,
        atoms: Atoms,
        indices: Sequence[int],
    ) -> list[tuple[int, int, float]]:
        detection_factor = float(
            self.relax_config.get("adsorbate_bond_detection_factor", 1.2)
        )
        stretch_factor = float(
            self.relax_config.get("adsorbate_bond_stretch_factor", 1.35)
        )
        abs_tol = float(self.relax_config.get("adsorbate_bond_abs_tol_A", 0.35))
        bonds: list[tuple[int, int, float]] = []
        for pos_i, idx_i in enumerate(indices):
            symbol_i = atoms[int(idx_i)].symbol
            radius_i = float(covalent_radii[atomic_numbers[symbol_i]])
            for idx_j in indices[pos_i + 1 :]:
                symbol_j = atoms[int(idx_j)].symbol
                radius_j = float(covalent_radii[atomic_numbers[symbol_j]])
                dist = float(atoms.get_distance(int(idx_i), int(idx_j), mic=True))
                if dist <= detection_factor * (radius_i + radius_j):
                    max_dist = max(dist * stretch_factor, dist + abs_tol)
                    bonds.append((int(idx_i), int(idx_j), max_dist))
        return bonds

    @staticmethod
    def _adsorbate_indices(
        atoms: Atoms,
        source_row: dict[str, object],
    ) -> list[int]:
        if "reaction_is_adsorbate" in atoms.arrays:
            mask = np.asarray(atoms.arrays["reaction_is_adsorbate"], dtype=bool)
            return [int(idx) for idx, value in enumerate(mask) if value]

        text = str(source_row.get("atom_is_adsorbate", "")).strip()
        if text:
            values = [int(part) for part in text.split()]
            if len(values) == len(atoms):
                return [idx for idx, value in enumerate(values) if value]

        tags = atoms.get_tags()
        if tags.size:
            return [int(idx) for idx, value in enumerate(tags) if value != 0]
        return []

    @staticmethod
    def _append_error(existing: str, extra: str) -> str:
        if not extra:
            return existing
        if not existing:
            return extra
        return f"{existing}; {extra}"

    def _progress(self, message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        line = f"{timestamp} {message}"
        if bool(self.relax_config.get("progress_stdout", True)):
            print(line, flush=True)
        progress_path = self._progress_log_path()
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _progress_log_path(self) -> Path | None:
        value = self.relax_config.get("progress_log")
        if value in ("", False):
            return None
        if isinstance(value, str) and value.lower() in {"false", "none", "off"}:
            return None
        output_dir = Path(self.config.output_dir)
        if value is None:
            return output_path(self.config, "state_relaxation_log")
        path = Path(str(value))
        if path.parent == Path("."):
            path = output_dir / path
        return path

    def _logfile_path(
        self,
        log_file: object,
        manifest_row: dict[str, object],
        index: int,
    ) -> str | None:
        if log_file in (None, "", False):
            return None
        log_path = Path(str(log_file))
        if log_path.name == str(log_file):
            state_dir = Path(str(manifest_row.get("candidates_traj", ""))).parent
            stem = log_path.stem
            suffix = log_path.suffix or ".log"
            log_path = state_dir / f"{stem}_{index:04d}{suffix}"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return str(log_path)

    @staticmethod
    def _energy_row(
        manifest_row: dict[str, object],
        source_row: dict[str, object],
        index: int,
        converged: bool,
        energy: float,
        final_fmax: float,
        nsteps: int,
        error: str,
        adsorbate_intact: bool = True,
        adsorbate_integrity_error: str = "",
        adsorbate_bond_distances_A: str = "",
    ) -> dict[str, object]:
        return {
            "candidate_index": index,
            "candidate_id": source_row.get("candidate_id", ""),
            "candidate_kind": source_row.get("candidate_kind", ""),
            "parent_o_candidate_id": source_row.get("parent_o_candidate_id", ""),
            "anchor_index": source_row.get("anchor_index", ""),
            "converged": bool(converged),
            "energy_eV": energy,
            "fmax_eV_A": final_fmax,
            "nsteps": nsteps,
            "adsorbate_intact": bool(adsorbate_intact),
            "adsorbate_integrity_error": adsorbate_integrity_error,
            "adsorbate_bond_distances_A": adsorbate_bond_distances_A,
            "error": error,
        }

    @staticmethod
    def _read_atoms_list(path: Path) -> list[Atoms]:
        atoms = read(str(path), index=":")
        if isinstance(atoms, Atoms):
            return [atoms]
        return list(atoms)

    @staticmethod
    def _read_csv(path: str | Path) -> list[dict[str, str]]:
        with Path(path).open(newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    @staticmethod
    def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        if not fields:
            return
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
