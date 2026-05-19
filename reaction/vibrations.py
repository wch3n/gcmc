"""Local harmonic vibration workflow for all relaxed reaction-state candidates."""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
from ase import Atoms
from ase.io import read
from ase.thermochemistry import HarmonicThermo
from ase.vibrations import Vibrations

from gcmc.workflows import build_adsorbate_gcmc_calculator

from .output_files import output_path

_ORIGIN_INDEX_ARRAY = "reaction_origin_index"
_IS_ADSORBATE_ARRAY = "reaction_is_adsorbate"
_RESULT_MODE = "all_relaxed_v1"

_DEFAULT_VIBRATION_CONFIG = {
    "enabled": False,
    "temperature_K": 303.0,
    "slab_cutoff_A": 3.5,
    "delta_A": 0.01,
    "nfree": 2,
    "ignore_imag_modes": False,
    "states": ["00_clean", "01_OH", "02_O", "03_OOH"],
    "skip_existing": True,
    "progress_stdout": True,
    "progress_log": None,
    "calculator": None,
    "model": None,
    "model_file": None,
    "device": None,
    "use_kokkos": None,
}

_STATE_ALIASES = {
    "clean": "00_clean",
    "00_clean": "00_clean",
    "*": "00_clean",
    "oh": "01_OH",
    "01_oh": "01_OH",
    "o": "02_O",
    "02_o": "02_O",
    "ooh": "03_OOH",
    "03_ooh": "03_OOH",
}

_CLEAN_STATE_DIR = "00_clean"


def default_vibration_config() -> dict[str, object]:
    return dict(_DEFAULT_VIBRATION_CONFIG)


class ReactionStateVibrationWorkflow:
    """Run local harmonic vibrations for every relaxed reaction-state candidate.

    The slab mask is derived from each relaxed clean candidate and then reused
    exactly for the adsorbate candidates that originate from the same parent
    representative, matched through atom-origin provenance written during
    candidate generation.
    """

    def __init__(self, config: SimpleNamespace):
        self.config = config
        raw = getattr(config, "vibrations", {}) or {}
        merged = default_vibration_config()
        if isinstance(raw, dict):
            merged.update(raw)
        self.vibration_config = merged

    def enabled(self) -> bool:
        return bool(self.vibration_config.get("enabled", False))

    def run(self, state_relaxation_manifest_path: str | Path | None) -> dict[str, str]:
        if not self.enabled():
            return {}
        if state_relaxation_manifest_path in (None, ""):
            return {}

        state_relaxation_manifest_path = Path(state_relaxation_manifest_path)
        if not state_relaxation_manifest_path.exists():
            return {}

        manifest_rows = self._read_csv(state_relaxation_manifest_path)
        if not manifest_rows:
            return {}

        selected_state_dirs = self._selected_state_dirs()
        blocks = [
            row
            for row in manifest_rows
            if self._row_selected(row, selected_state_dirs)
        ]
        if not blocks:
            return {}

        self._progress(
            "vibrations start "
            f"manifest={state_relaxation_manifest_path} blocks={len(blocks)}"
        )
        calculator = self._build_calculator()
        clean_masks = self._build_clean_masks(blocks)
        summary_rows: list[dict[str, object]] = []

        for block in sorted(
            blocks,
            key=lambda row: (
                str(row.get("site_id", "")),
                str(row.get("state_dir", "")),
                str(row.get("species", "")),
            ),
        ):
            summary_rows.extend(self._run_block(block, clean_masks, calculator))

        summary_path = output_path(self.config, "vibration_summary_csv")
        self._write_csv(summary_path, summary_rows)
        self._progress(
            "vibrations complete "
            f"summary={summary_path} candidates={len(summary_rows)}"
        )
        return {"vibration_summary_csv": str(summary_path)}

    def _selected_state_dirs(self) -> set[str]:
        values = self.vibration_config.get("states") or []
        if isinstance(values, str):
            values = [values]
        selected: set[str] = set()
        for value in values:
            mapped = _STATE_ALIASES.get(str(value).strip().lower())
            if mapped is not None:
                selected.add(mapped)
        return selected or {"00_clean", "01_OH", "02_O", "03_OOH"}

    @staticmethod
    def _row_selected(row: dict[str, object], selected_state_dirs: set[str]) -> bool:
        return str(row.get("state_dir", "")) in selected_state_dirs

    def _build_calculator(self):
        values = vars(self.config).copy()
        for key in ("calculator", "model", "model_file", "device", "use_kokkos"):
            value = self.vibration_config.get(key)
            if value not in (None, ""):
                values[key] = value
        cfg = SimpleNamespace(**values)
        return build_adsorbate_gcmc_calculator(
            cfg,
            {"device": values.get("device") or getattr(self.config, "device", None)},
        )

    def _build_clean_masks(
        self,
        manifest_rows: Sequence[dict[str, object]],
    ) -> dict[str, dict[tuple[str, str, str, str], dict[str, object]]]:
        clean_masks: dict[str, dict[tuple[str, str, str, str], dict[str, object]]] = {}
        for block in manifest_rows:
            if str(block.get("state_dir", "")) != _CLEAN_STATE_DIR:
                continue
            site_id = str(block.get("site_id", ""))
            if not site_id:
                continue
            for record in self._iter_block_candidates(block):
                atoms = record.get("atoms")
                metadata = record.get("candidate_metadata", {})
                if not isinstance(atoms, Atoms):
                    continue
                try:
                    slab_origin_indices = self._clean_local_slab_origins(site_id, atoms)
                except Exception as exc:  # noqa: BLE001 - record block-level use later.
                    self._progress(
                        "vibrations clean-mask skip "
                        f"site={site_id} candidate={metadata.get('candidate_id', '')} "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    continue
                parent_key = self._parent_key(metadata)
                entry = {
                    "candidate_id": metadata.get("candidate_id", ""),
                    "source_representative_traj": metadata.get(
                        "source_representative_traj",
                        "",
                    ),
                    "source_representative_frame": metadata.get(
                        "source_representative_frame",
                        "",
                    ),
                    "source_representative_rank": metadata.get(
                        "source_representative_rank",
                        "",
                    ),
                    "slab_origin_indices": slab_origin_indices,
                    "energy_eV": self._as_float(record["energy_row"].get("energy_eV")),
                }
                site_masks = clean_masks.setdefault(site_id, {})
                previous = site_masks.get(parent_key)
                previous_energy = self._as_float(previous.get("energy_eV")) if previous else float("nan")
                current_energy = self._as_float(entry.get("energy_eV"))
                replace = previous is None
                if not replace and math_isfinite(current_energy) and not math_isfinite(previous_energy):
                    replace = True
                if not replace and math_isfinite(current_energy) and math_isfinite(previous_energy):
                    replace = current_energy < previous_energy
                if replace:
                    site_masks[parent_key] = entry
        return clean_masks

    def _run_block(
        self,
        block: dict[str, object],
        clean_masks: dict[str, dict[tuple[str, str, str, str], dict[str, object]]],
        calculator,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        site_id = str(block.get("site_id", ""))
        state_dir = str(block.get("state_dir", ""))
        for record in self._iter_block_candidates(block):
            summary = self._run_candidate(record, clean_masks, calculator)
            rows.append(summary)
        if not rows:
            self._progress(
                "vibrations skip empty block "
                f"site={site_id} state={state_dir}"
            )
        return rows

    def _run_candidate(
        self,
        record: dict[str, object],
        clean_masks: dict[str, dict[tuple[str, str, str, str], dict[str, object]]],
        calculator,
    ) -> dict[str, object]:
        site_id = str(record.get("site_id", ""))
        state_dir = str(record.get("state_dir", ""))
        metadata = dict(record.get("candidate_metadata", {}))
        energy_row = dict(record.get("energy_row", {}))
        atoms = record.get("atoms")
        relaxed_traj = Path(str(record.get("relaxed_traj", "")))
        vib_dir = relaxed_traj.parent / "vibrations" / self._sanitize_token(
            str(metadata.get("candidate_id", "candidate"))
        )
        result_csv = vib_dir / "result.csv"
        if bool(self.vibration_config.get("skip_existing", True)) and result_csv.exists():
            existing = self._read_csv(result_csv)
            if existing and existing[0].get("vibration_mode") == _RESULT_MODE:
                return self._summary_row(existing[0])

        output = self._base_output_row(record, vib_dir)
        output["vibration_mode"] = _RESULT_MODE
        output["relax_converged"] = energy_row.get("converged", "")
        output["potential_energy_eV"] = energy_row.get("energy_eV", "")

        try:
            if not isinstance(atoms, Atoms):
                raise ValueError("Missing relaxed Atoms object.")
            if not self._row_bool_or_true(energy_row.get("adsorbate_intact", "")):
                detail = str(energy_row.get("adsorbate_integrity_error", "")).strip()
                raise ValueError(
                    "Relaxed molecular adsorbate failed integrity check"
                    + (f": {detail}" if detail else ".")
                )
            if not math_isfinite(self._as_float(energy_row.get("energy_eV"))):
                raise ValueError("Missing finite relaxed energy.")
            clean_mask = self._resolve_clean_mask(site_id, metadata, clean_masks)
            output["mask_source_candidate_id"] = clean_mask.get("candidate_id", "")
            output["mask_source_representative_traj"] = clean_mask.get(
                "source_representative_traj",
                "",
            )
            output["mask_source_representative_frame"] = clean_mask.get(
                "source_representative_frame",
                "",
            )
            output["mask_source_representative_rank"] = clean_mask.get(
                "source_representative_rank",
                "",
            )
            slab_origin_indices = tuple(int(value) for value in clean_mask["slab_origin_indices"])
            output["slab_origin_indices"] = " ".join(str(value) for value in slab_origin_indices)
            vib_indices, slab_indices, adsorbate_indices = self._state_vibration_indices(
                atoms,
                state_dir,
                slab_origin_indices,
            )
            output["n_slab_mask_atoms"] = len(slab_indices)
            output["n_adsorbate_atoms"] = len(adsorbate_indices)
            output["n_vibrated_atoms"] = len(vib_indices)

            vib_dir.mkdir(parents=True, exist_ok=True)
            work_atoms = atoms.copy()
            work_atoms.calc = calculator
            vib = Vibrations(
                work_atoms,
                indices=vib_indices,
                name=str(vib_dir / "vib"),
                delta=float(self.vibration_config.get("delta_A", 0.01)),
                nfree=int(self.vibration_config.get("nfree", 2)),
            )
            self._progress(
                "vibrations run "
                f"site={site_id} state={state_dir} candidate={metadata.get('candidate_id', '')} "
                f"nvib={len(vib_indices)}"
            )
            vib.run()
            vib_energies = vib.get_energies()
            temperature = float(self.vibration_config.get("temperature_K", 303.0))
            potential = float(energy_row.get("energy_eV"))
            thermo = HarmonicThermo(
                vib_energies=vib_energies,
                potentialenergy=potential,
                ignore_imag_modes=bool(self.vibration_config.get("ignore_imag_modes", False)),
            )
            zpe = float(thermo.get_ZPE_correction())
            internal = float(thermo.get_internal_energy(temperature, verbose=False))
            entropy = float(thermo.get_entropy(temperature, verbose=False))
            free_energy = float(thermo.get_helmholtz_energy(temperature, verbose=False))
            output.update(
                {
                    "ready": True,
                    "temperature_K": temperature,
                    "slab_cutoff_A": float(self.vibration_config.get("slab_cutoff_A", 3.5)),
                    "delta_A": float(self.vibration_config.get("delta_A", 0.01)),
                    "nfree": int(self.vibration_config.get("nfree", 2)),
                    "zpe_eV": zpe,
                    "thermal_internal_eV": internal - potential - zpe,
                    "entropy_eV_K": entropy,
                    "minus_TS_eV": -temperature * entropy,
                    "harmonic_free_energy_eV": free_energy,
                    "harmonic_correction_eV": free_energy - potential,
                    "n_imag_modes": int(getattr(thermo, "n_imag", 0)),
                    "vibrated_indices": " ".join(str(idx) for idx in vib_indices),
                    "adsorbate_indices": " ".join(str(idx) for idx in adsorbate_indices),
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep failed candidates visible.
            output.update(
                {
                    "ready": False,
                    "temperature_K": self.vibration_config.get("temperature_K", 303.0),
                    "slab_cutoff_A": float(self.vibration_config.get("slab_cutoff_A", 3.5)),
                    "delta_A": float(self.vibration_config.get("delta_A", 0.01)),
                    "nfree": int(self.vibration_config.get("nfree", 2)),
                    "zpe_eV": "",
                    "thermal_internal_eV": "",
                    "entropy_eV_K": "",
                    "minus_TS_eV": "",
                    "harmonic_free_energy_eV": "",
                    "harmonic_correction_eV": "",
                    "n_imag_modes": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        self._write_csv(result_csv, [output])
        return self._summary_row(output)

    def _resolve_clean_mask(
        self,
        site_id: str,
        candidate_metadata: dict[str, object],
        clean_masks: dict[str, dict[tuple[str, str, str, str], dict[str, object]]],
    ) -> dict[str, object]:
        site_masks = clean_masks.get(site_id, {})
        if not site_masks:
            raise ValueError(f"No clean vibration mask available for site {site_id}.")
        parent_key = self._parent_key(candidate_metadata)
        exact = site_masks.get(parent_key)
        if exact is not None:
            return exact
        if len(site_masks) == 1:
            return next(iter(site_masks.values()))
        raise ValueError(
            "Could not match adsorbate candidate to a unique clean vibration mask "
            f"for site {site_id} and parent key {parent_key}."
        )

    @staticmethod
    def _parent_key(candidate_metadata: dict[str, object]) -> tuple[str, str, str, str]:
        traj = str(candidate_metadata.get("source_representative_traj", "")).strip()
        frame = str(candidate_metadata.get("source_representative_frame", "")).strip()
        rank = str(candidate_metadata.get("source_representative_rank", "")).strip()
        if traj or frame or rank:
            return (traj, frame, rank, "")
        candidate_id = str(candidate_metadata.get("candidate_id", "")).strip()
        match = re.match(r"^(rep\d+)", candidate_id)
        token = match.group(1) if match else candidate_id
        return ("", "", "", token)

    @staticmethod
    def _base_output_row(record: dict[str, object], vib_dir: Path) -> dict[str, object]:
        metadata = dict(record.get("candidate_metadata", {}))
        energy_row = dict(record.get("energy_row", {}))
        return {
            "site_id": record.get("site_id", ""),
            "state_dir": record.get("state_dir", ""),
            "species": record.get("species", ""),
            "candidate_id": metadata.get("candidate_id", ""),
            "candidate_index": energy_row.get("candidate_index", ""),
            "candidate_kind": metadata.get("candidate_kind", ""),
            "source_representative_traj": metadata.get("source_representative_traj", ""),
            "source_representative_frame": metadata.get("source_representative_frame", ""),
            "source_representative_rank": metadata.get("source_representative_rank", ""),
            "relaxed_traj": record.get("relaxed_traj", ""),
            "vibration_dir": str(vib_dir),
            "mask_source_candidate_id": "",
            "mask_source_representative_traj": "",
            "mask_source_representative_frame": "",
            "mask_source_representative_rank": "",
            "slab_origin_indices": "",
            "n_slab_mask_atoms": "",
            "n_adsorbate_atoms": "",
            "n_vibrated_atoms": "",
            "vibrated_indices": "",
            "adsorbate_indices": "",
        }

    @staticmethod
    def _summary_row(result_row: dict[str, object]) -> dict[str, object]:
        columns = (
            "site_id",
            "state_dir",
            "species",
            "candidate_id",
            "candidate_index",
            "candidate_kind",
            "source_representative_traj",
            "source_representative_frame",
            "source_representative_rank",
            "mask_source_candidate_id",
            "mask_source_representative_traj",
            "mask_source_representative_frame",
            "mask_source_representative_rank",
            "relaxed_traj",
            "vibration_dir",
            "slab_origin_indices",
            "n_slab_mask_atoms",
            "n_adsorbate_atoms",
            "n_vibrated_atoms",
            "relax_converged",
            "ready",
            "potential_energy_eV",
            "zpe_eV",
            "thermal_internal_eV",
            "entropy_eV_K",
            "minus_TS_eV",
            "harmonic_free_energy_eV",
            "harmonic_correction_eV",
            "n_imag_modes",
            "error",
        )
        return {column: result_row.get(column, "") for column in columns}

    def _iter_block_candidates(
        self,
        block: dict[str, object],
    ) -> Iterable[dict[str, object]]:
        site_id = str(block.get("site_id", ""))
        state_dir = str(block.get("state_dir", ""))
        species = str(block.get("species", ""))
        energies_csv = Path(str(block.get("energies_csv", "")))
        relaxed_traj = Path(str(block.get("relaxed_traj", "")))
        candidates_csv = energies_csv.parent / "candidates.csv"
        if not energies_csv.exists() or not relaxed_traj.exists() or not candidates_csv.exists():
            return []
        energy_rows = self._read_csv(energies_csv)
        candidate_rows = self._read_csv(candidates_csv)
        candidate_by_id = {
            str(row.get("candidate_id", "")): dict(row)
            for row in candidate_rows
            if row.get("candidate_id", "")
        }
        atoms_list = self._read_atoms_list(relaxed_traj)
        records: list[dict[str, object]] = []
        for fallback_index, energy_row in enumerate(energy_rows):
            candidate_index = self._candidate_index(energy_row, fallback_index)
            candidate_id = str(energy_row.get("candidate_id", ""))
            metadata = candidate_by_id.get(candidate_id, {})
            atoms = atoms_list[candidate_index].copy() if 0 <= candidate_index < len(atoms_list) else None
            if isinstance(atoms, Atoms):
                self._attach_candidate_mapping(atoms, metadata)
            records.append(
                {
                    "site_id": site_id,
                    "state_dir": state_dir,
                    "species": species,
                    "energy_row": dict(energy_row),
                    "candidate_metadata": metadata,
                    "atoms": atoms,
                    "relaxed_traj": str(relaxed_traj),
                }
            )
        return records

    @staticmethod
    def _candidate_index(energy_row: dict[str, object], fallback_index: int) -> int:
        try:
            return int(energy_row.get("candidate_index", fallback_index))
        except (TypeError, ValueError):
            return fallback_index

    def _clean_local_slab_origins(
        self,
        site_id: str,
        clean_atoms: Atoms,
    ) -> tuple[int, ...]:
        support_origin_indices = self._support_origin_indices(site_id)
        atom_origins = self._origin_indices(clean_atoms)
        adsorbate_mask = self._adsorbate_mask(clean_atoms)
        support_indices = [
            int(idx)
            for idx, origin in enumerate(atom_origins)
            if not adsorbate_mask[idx] and int(origin) in support_origin_indices
        ]
        if len(support_indices) != len(support_origin_indices):
            if max(support_origin_indices, default=-1) < len(clean_atoms):
                support_indices = list(support_origin_indices)
            else:
                raise ValueError(
                    f"Could not map support indices {support_origin_indices} onto clean state."
                )

        center = np.mean(clean_atoms.positions[support_indices], axis=0)
        distances = self._distances_to_point(clean_atoms, center)
        cutoff = float(self.vibration_config.get("slab_cutoff_A", 3.5))
        selected = [
            int(atom_origins[idx])
            for idx, (distance, is_adsorbate) in enumerate(zip(distances, adsorbate_mask))
            if not is_adsorbate and (idx in support_indices or distance <= cutoff)
        ]
        unique = tuple(sorted({int(origin) for origin in selected}))
        if not unique:
            raise ValueError("Clean-state slab mask is empty.")
        return unique

    def _state_vibration_indices(
        self,
        atoms: Atoms,
        state_dir: str,
        slab_origin_indices: Sequence[int],
    ) -> tuple[list[int], list[int], list[int]]:
        atom_origins = self._origin_indices(atoms)
        adsorbate_mask = self._adsorbate_mask(atoms)
        slab_set = {int(value) for value in slab_origin_indices}
        slab_indices = [
            int(idx)
            for idx, origin in enumerate(atom_origins)
            if not adsorbate_mask[idx] and int(origin) in slab_set
        ]
        if len(slab_indices) != len(slab_set):
            present = {int(atom_origins[idx]) for idx in slab_indices}
            missing = sorted(slab_set.difference(present))
            raise ValueError(
                f"State {state_dir} is missing clean-mask slab atoms with origin indices {missing}."
            )
        adsorbate_indices: list[int] = []
        if state_dir != _CLEAN_STATE_DIR:
            adsorbate_indices = [
                int(idx)
                for idx, is_adsorbate in enumerate(adsorbate_mask)
                if is_adsorbate
            ]
            if not adsorbate_indices:
                raise ValueError(
                    f"State {state_dir} has no adsorbate atoms in the vibration mask."
                )
        vib_indices = sorted(set(slab_indices + adsorbate_indices))
        if not vib_indices:
            raise ValueError(f"State {state_dir} produced an empty vibration mask.")
        return vib_indices, slab_indices, adsorbate_indices

    @staticmethod
    def _support_origin_indices(site_id: str) -> tuple[int, ...]:
        support = str(site_id).split(":", 1)[-1].strip()
        if not support:
            return ()
        return tuple(sorted(int(part) for part in support.split("-")))

    @staticmethod
    def _origin_indices(atoms: Atoms) -> np.ndarray:
        if _ORIGIN_INDEX_ARRAY in atoms.arrays:
            return np.asarray(atoms.arrays[_ORIGIN_INDEX_ARRAY], dtype=int)
        return np.arange(len(atoms), dtype=int)

    @staticmethod
    def _adsorbate_mask(atoms: Atoms) -> np.ndarray:
        if _IS_ADSORBATE_ARRAY in atoms.arrays:
            return np.asarray(atoms.arrays[_IS_ADSORBATE_ARRAY], dtype=bool)
        tags = atoms.get_tags()
        if tags.size:
            return np.asarray(tags != 0, dtype=bool)
        return np.zeros(len(atoms), dtype=bool)

    def _distances_to_point(self, atoms: Atoms, point: np.ndarray) -> np.ndarray:
        delta = np.asarray(atoms.positions, dtype=float) - np.asarray(point, dtype=float)
        if any(bool(flag) for flag in atoms.pbc):
            cell = np.asarray(atoms.cell.array, dtype=float)
            try:
                frac = np.linalg.solve(cell.T, delta.T).T
                for axis, periodic in enumerate(atoms.pbc):
                    if periodic:
                        frac[:, axis] -= np.round(frac[:, axis])
                delta = frac @ cell
            except np.linalg.LinAlgError:
                pass
        return np.linalg.norm(delta, axis=1)

    @staticmethod
    def _sanitize_token(value: str) -> str:
        token = value.strip().replace("/", "_").replace(":", "_")
        return token or "candidate"

    @staticmethod
    def _read_atoms_list(path: Path) -> list[Atoms]:
        atoms = read(str(path), index=":")
        if isinstance(atoms, Atoms):
            return [atoms]
        return list(atoms)

    @staticmethod
    def _attach_candidate_mapping(
        atoms: Atoms,
        candidate_metadata: dict[str, object],
    ) -> None:
        if _ORIGIN_INDEX_ARRAY not in atoms.arrays:
            values = ReactionStateVibrationWorkflow._parse_int_sequence(
                candidate_metadata.get("atom_origin_indices", ""),
            )
            if len(values) == len(atoms):
                atoms.new_array(_ORIGIN_INDEX_ARRAY, np.asarray(values, dtype=int))
        if _IS_ADSORBATE_ARRAY not in atoms.arrays:
            values = ReactionStateVibrationWorkflow._parse_int_sequence(
                candidate_metadata.get("atom_is_adsorbate", ""),
            )
            if len(values) == len(atoms):
                atoms.new_array(_IS_ADSORBATE_ARRAY, np.asarray(values, dtype=int))

    @staticmethod
    def _parse_int_sequence(value: object) -> list[int]:
        text = str(value).strip()
        if not text:
            return []
        return [int(part) for part in text.split()]

    def _progress(self, message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        line = f"{timestamp} {message}"
        if bool(self.vibration_config.get("progress_stdout", True)):
            print(line, flush=True)
        progress_path = self._progress_log_path()
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _progress_log_path(self) -> Path | None:
        value = self.vibration_config.get("progress_log")
        if value in (None, "", False):
            return output_path(self.config, "vibration_log")
        if isinstance(value, str) and value.lower() in {"false", "none", "off"}:
            return None
        output_dir = Path(self.config.output_dir)
        path = Path(str(value))
        if path.parent == Path("."):
            path = output_dir / path
        return path

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
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _as_float(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    @staticmethod
    def _row_bool_or_true(value: object) -> bool:
        if value in (None, ""):
            return True
        return str(value).strip().lower() in {"1", "true", "yes", "y"}


def math_isfinite(value: float) -> bool:
    return bool(np.isfinite(value))
